// Hy4-flash HIP forward (contract: hy4_hip_forward.hpp). Binding only: every stage calls a shared entry of
// hy4_hip_plan.hpp; the model data here are tensor names, the layer schedule and the Traits / presets.
// Compile with -ffp-contract=off -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt (declared-arithmetic ops).
// The HIP runtime comes first: the shared weight sources call std::memcpy in device code, and <cstring>'s
// using-declaration sees the device overload only when the runtime was included before it.
#include <hip/hip_runtime.h>
#include "hy4/hip/hy4_hip_forward.hpp"
#include "hy4/hip/hy4_hip_plan.hpp"
#include "hy4/hip/hy4_hip_rows.hpp"

#include "attn_core/cache_rows.hpp"
#include "attn_core/group_pool.hpp"
#include "attn_core/group_score_scan.hpp"
#include "attn_core/group_select_kernel.hpp"
#include "attn_core/sparse_flash_mla.hpp"
#include "core/f16_dense_q8a8.hpp"
#include "core/hc_core.hpp"
#include "core/hip_range_weight_loader.hpp"
#include "engram/hip/inject_kernel.hpp"
#include "engram/hip/row_kernel.hpp"
#include "moe_core/route_wave_kernel.hpp"
#include "ops/glu/glu_device.hpp"
#include "ops/norm/norm_presets.hpp"
#include "ops/rope/rope_device.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/engram/row_reader.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hc/hc_contract.hpp"
#include "strix/hy4/presets.hpp"
#include "strix/hip_embedding_launcher.hpp"
#include "strix/hip_check.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_ggml_weight_formats.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "../../hip_q8_rows_sites.hpp"
#include "strix/env_flags.hpp"
#include "strix/hip_moe_a8_down.hpp"
#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#include "strix/q51_known_rows.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/moe/route_policy.hpp"
#include "strix/moe_a8_down_contract.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

namespace strix::hy4::hipbind {
namespace {

namespace hcc = strix::core::hc;
namespace hcs = strix::hc;
namespace glu = strix::ops::glu;
namespace mla = strix::attn_core::sparse::mla;
namespace a8 = strix::moe_q4k_a8;

void check(hipError_t e, const char* what) { strix::hip_check_prefixed("hy4 hip forward: ", e, what); }

constexpr hcs::Spec kSub = preset::hc_spec(hcs::Site::sublayer);
constexpr hcs::Spec kHead = preset::hc_spec(hcs::Site::head);

// RMSNorm of this model: the BF16 module's double rounding (ops/norm OutputForm::bf16_double), F32 weights, the declared
// square root: the shared preset.
using NormTraits = strix::ops::norm::DeclaredF32Traits<strix::ops::norm::OutputForm::bf16_double>;

// Sparse MLA flash policies (strix/hy4/presets.hpp dsa / swa): 32 heads per group.
struct DsaSpec {
    static constexpr unsigned kHeads = 32U, kLat = 512U, kRope = 64U, kGroups = 1U;
    static constexpr float kScale = 0.0625F;  // (192 + 64)^-1/2, exact
    static constexpr strix::attn::mla::Select kSelect = strix::attn::mla::Select::List;
    static constexpr bool kSink = true;
};
struct SwaSpec {
    static constexpr unsigned kHeads = 32U, kLat = 512U, kRope = 64U, kGroups = 2U;  // 64 heads = 2 groups of 32
    static constexpr float kScale = 0.0625F;
    static constexpr strix::attn::mla::Select kSelect = strix::attn::mla::Select::Window;
    static constexpr bool kSink = true;
};
using DsaPolicy = mla::LatentPolicy<DsaSpec>;
using SwaPolicy = mla::LatentPolicy<SwaSpec>;
static_assert(DsaSpec::kHeads * DsaSpec::kGroups == preset::dsa().g.heads);
static_assert(SwaSpec::kHeads * SwaSpec::kGroups == preset::swa().g.heads);

// ---- indexer views (data over the shared pool / score scan / select cores) -------------------------------------
namespace gpl = strix::attn::group_pool;  // gi = strix::attn::group_indexer (hy4_hip_plan.hpp)
constexpr unsigned kIdxDim = preset::indexer.head_dim, kIdxHeads = preset::indexer.heads, kIdxRatio = preset::indexer.ratio;

// Pooling weights of one indexed layer (Pooling::GatedSoftmaxApe): the pack's Q8_0 value / gate [128][128] (out, in)
// decoded exactly per element by the shared GGUF element decode; ape [ratio][128] and norm [128] F32.
struct Q8Ape {
    const std::uint8_t* wv;
    const std::uint8_t* wg;
    const float* ape_w;
    const float* norm_w;
    float eps_v;
    __device__ static float q8(const std::uint8_t* w, unsigned o, unsigned i) {
        return strix::hip_ggml::block_element<strix::hip_ggml::type_q8_0>(
            w + (static_cast<std::size_t>(o) * (kIdxDim / 32U) + i / 32U) * 34U, i % 32U);
    }
    __device__ float v(unsigned o, unsigned i) const { return q8(wv, o, i); }
    __device__ float g(unsigned o, unsigned i) const { return q8(wg, o, i); }
    __device__ float ape(unsigned j, unsigned o) const { return ape_w[static_cast<std::size_t>(j) * kIdxDim + o]; }
    __device__ float norm(unsigned o) const { return norm_w[o]; }
    __device__ float eps() const { return eps_v; }
};

// Block b pools group *first + b from the token-key ring; out [b][128] = the bf16-valued pooled key before RoPE.
struct PoolView {
    static constexpr unsigned kRatio = kIdxRatio, kDim = kIdxDim;
    const std::uint16_t* ring;   // token keys bf16 [ring_rows][128], slot = position % ring_rows
    unsigned ring_rows;
    const std::uint32_t* first;  // device: first group of the launch
    Q8Ape w;
    float* out;
    __device__ bool live(unsigned) const { return true; }
    __device__ unsigned group(unsigned b) const { return *first + b; }
    __device__ float key(unsigned g, unsigned j, unsigned i) const {
        return gpl::bf16_value(ring[static_cast<std::size_t>((g * kRatio + j) % ring_rows) * kDim + i]);
    }
    __device__ const Q8Ape& weights() const { return w; }
    __device__ float exp(float x) const { return strix::ds41::declared_exp(x); }
    __device__ void store(unsigned g, unsigned lane, const float* o) const {
        out[static_cast<std::size_t>(g - *first) * kDim + lane] = o[lane];
    }
};

// Score scan rows: the layer's group-key cache (bf16, rotated) and each row's query slot (visible groups = t / 4).
struct ScanRows {
    const std::uint16_t* keys;
    const std::uint32_t* tpos;
};
struct ScanView {
    using Rows = ScanRows;
    using Key = std::uint16_t;
    static constexpr unsigned kHeads = kIdxHeads, kDim = kIdxDim;
    __device__ static const Key* keys(const Rows& d, unsigned) { return d.keys; }
    __device__ static unsigned pools(const Rows& d, unsigned r) { return gi::visible_groups(preset::indexer, d.tpos[r]); }
    __device__ static float weight(float raw) { return raw * __builtin_bit_cast(float, kIndexHeadScaleBits); }  // InHeadWeight
    __device__ static float scaled(float dot) { return dot; }
    __device__ static std::uint32_t key_of(float score) { return gi::score_key(score); }
};
using SelectView = strix::attn_core::group_select::PositionView<preset::indexer, kIndexListStride>;

// Row bytes of a GGUF tensor type from the shared format table (strix/gguf_reader.hpp tensor_type_geometry).
std::uint64_t row_bytes(std::uint32_t type, std::uint64_t width) {
    const auto g = strix::gguf::tensor_type_geometry(type);
    if (!g || g->block_elements == 0U) throw std::invalid_argument("hy4 hip forward: row bytes of GGUF type " + std::to_string(type));
    return width / g->block_elements * g->block_bytes;
}

struct DevMem {
    void* p = nullptr;
    DevMem() = default;
    explicit DevMem(std::size_t bytes) { check(hipMalloc(&p, bytes == 0 ? 4 : bytes), "hipMalloc"); }
    DevMem(DevMem&& o) noexcept : p(o.p) { o.p = nullptr; }
    DevMem& operator=(DevMem&& o) noexcept { std::swap(p, o.p); return *this; }
    DevMem(const DevMem&) = delete;
    DevMem& operator=(const DevMem&) = delete;
    ~DevMem() { if (p) (void)hipFree(p); }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

struct Tensor {
    const std::uint8_t* p = nullptr;
    std::uint32_t type = 0;
    std::vector<std::uint64_t> dims;
    std::uint64_t bytes = 0;
    [[nodiscard]] const float* f() const { return reinterpret_cast<const float*>(p); }
};

std::string blk(std::uint32_t l, const char* name) { return "blk." + std::to_string(l) + "." + name; }

}  // namespace

struct Forward::Impl {
    Config c;
    std::uint32_t first = 0, last = 0, context = 0, pos = 0;
    std::vector<std::uint32_t> tokens;
    hipStream_t s = nullptr;
    LoadReport report;
    StepTimes times;
    bool timing = false;

    std::vector<DevMem> mem;  // every allocation (weights, caches, workspace)
    std::map<std::string, Tensor> w;
    std::map<std::uint32_t, float*> engram_w1;  // [copy scale 4 x hidden | pre gamma hidden] per Engram layer
    bool store_epilogue = true;  // STRIX_HY4_Q8_STORE_EPILOGUE
    bool pf_store_epilogue = true;  // STRIX_HY4_PF_STORE_EPILOGUE: prompt windows' bf16 epilogues in the producer's store (EXACT)
    // Pinned host staging of the decode step (host-to-device copies never wait for the stream; each slot is written once
    // per step and the step ends with a stream sync, so no slot is rewritten while its copy is pending).
    struct Staging {
        std::array<std::uint32_t, 4> pos;         // {position, rope slot 4g, closing group g, 0}
        std::uint32_t list_n[64];                 // dense-path list length, by layer
        std::uint32_t status[8 * 64];             // A8 probe / down status words (16-byte slots: [layer][probe, down][4])
    };
    Staging* stage = nullptr;
    std::uint8_t* engram_stage = nullptr;  // pinned: Engram n-gram rows, one slot of heads x engram_dim per Engram ordinal
    std::uint32_t* dstatus_all = nullptr;  // device status words [layers][2][4] (A8 views are 16-byte aligned), zeroed once per step
    std::unique_ptr<strix::engram::RowReader> engram_reader;

    // caches (bf16 rows) per layer; rope tables
    std::vector<std::uint16_t*> latent;
    std::vector<strix::attn::cache_rows::Rows> latent_rows;
    float *cos_idx = nullptr, *sin_idx = nullptr, *cos_swa = nullptr, *sin_swa = nullptr;
    // indexer (indexed layers, hy4_hip_plan layer_caches): token-key ring, group-key cache by group; shared scratch
    std::vector<std::uint16_t*> tkeys, gkeys;
    std::vector<strix::attn::cache_rows::Rows> tk_rows;
    float *ik = nullptr, *pooled = nullptr, *iq = nullptr, *hwraw = nullptr;
    std::uint32_t *skeys = nullptr, *hist0 = nullptr, *ilist = nullptr;
    std::uint32_t group_cap = 0;            // group-key rows per layer (context / 4, rounded up)
    std::array<std::uint32_t, 4> hpos{};    // uploaded to dpos: {position, rope slot 4g, closing group g, 0}

    // workspace
    float *x0 = nullptr, *x1 = nullptr, *u = nullptr, *xn = nullptr, *f = nullptr, *partial = nullptr;
    hcc::MixRow<kSub>* mix = nullptr;
    hcc::MixRow<kHead>* mixh = nullptr;
    float *gate = nullptr, *qa = nullptr, *qv = nullptr, *ckv = nullptr, *qfull = nullptr, *olat = nullptr, *o = nullptr;
    std::uint16_t* qt = nullptr;
    float *g = nullptr, *up = nullptr, *act = nullptr;
    float *rlog = nullptr, *rw = nullptr, *z = nullptr, *pout = nullptr, *rawdots = nullptr, *acc = nullptr, *routed = nullptr,
          *shared = nullptr;
    std::uint32_t* ids = nullptr;
    std::int8_t *pcodes = nullptr, *dcodes = nullptr;
    float *pdeltas = nullptr, *psums = nullptr, *ddeltas = nullptr, *dsums = nullptr;
    std::uint32_t *flag = nullptr, *dpos = nullptr, *iota = nullptr, *list_n = nullptr;
    std::uint8_t* erows = nullptr;
    float *e = nullptr, *kvout = nullptr, *logits_d = nullptr;
    std::vector<float> logits, empty;
    std::uint8_t* greedy_partials = nullptr;  // shared device greedy (strix/hip_generation_output.hpp) scratch, one row
    std::uint64_t greedy_bytes = 0;
    strix::HipGreedyRowsReply* greedy_reply = nullptr;
    std::uint32_t greedy_token = 0;

    // batched prompt (prompt_rows > 0): [rows][...] workspaces of one window; attn_k_b re-laid 256 wide per layer
    std::uint32_t rmax = 0;
    float* seam = nullptr;  // [rmax][4][hidden] X rows of a layer-pipeline window (stage of a layer split only)
    struct PromptRows {
        float *x0 = nullptr, *x1 = nullptr, *u = nullptr, *xn = nullptr, *f = nullptr, *partial = nullptr;
        hcc::MixRow<kSub>* mix = nullptr;
        std::int8_t* codes = nullptr;
        float* deltas = nullptr;
        std::int32_t* sums = nullptr;
        float *gate = nullptr, *qa = nullptr, *qv = nullptr, *ckv = nullptr, *ckn = nullptr, *qfull = nullptr, *olat = nullptr, *o = nullptr;
        std::uint16_t *qt = nullptr, *zb = nullptr;
        float *g = nullptr, *up = nullptr, *act = nullptr, *rlog = nullptr, *rw = nullptr, *z = nullptr, *acc = nullptr, *routed = nullptr,
              *shared = nullptr;
        std::uint32_t *ids = nullptr, *status = nullptr;
        std::int8_t *mcodes = nullptr, *dcodes = nullptr;
        float *mdeltas = nullptr, *ddeltas = nullptr, *hout = nullptr, *clear = nullptr, *rawdots = nullptr;
        std::int32_t *msums = nullptr, *dsums = nullptr;
        std::uint8_t* csr = nullptr;
        float *ik = nullptr, *pooled = nullptr, *iq = nullptr, *hwraw = nullptr;
        std::uint32_t *skeys = nullptr, *hist0 = nullptr, *ilist = nullptr, *list_n = nullptr, *iota4 = nullptr;
        std::uint8_t *erows = nullptr, *gate_off = nullptr;
        float *e = nullptr, *kvout = nullptr;
    } pr;
    std::vector<const std::uint8_t*> kb_pad;
    std::vector<std::uint8_t> engram_host;
    std::vector<std::uint32_t> status_host;

    template <class T> T* alloc(std::size_t count) {
        mem.emplace_back(count * sizeof(T));
        return mem.back().as<T>();
    }

    const Tensor& t(const std::string& name) const {
        const auto it = w.find(name);
        if (it == w.end()) throw std::runtime_error("hy4 hip forward: tensor not loaded: " + name);
        return it->second;
    }

    [[nodiscard]] bool wanted(const std::string& name) const {
        if (name == "token_embd.weight") return first == 0U;
        if (name.rfind("output", 0) == 0) return last == c.layers;
        if (name.rfind("blk.", 0) != 0) return false;
        const auto layer = static_cast<std::uint32_t>(std::stoul(name.substr(4, name.find('.', 4) - 4)));
        return layer >= first && layer < last;
    }

    void load(const std::vector<std::filesystem::path>& parts) {
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto l : c.engram_layers)
            if (l >= first && l < last) engram_w1[l] = alloc<float>(5U * std::size_t{c.hidden});
        std::vector<std::unique_ptr<strix::core::HipRangeWeightLoader>> loaders;
        for (const auto& part : parts) {
            const auto info = gguf::read_file(part);
            auto loader = std::make_unique<strix::core::HipRangeWeightLoader>(part.string());
            for (const auto& ti : info.tensors) {
                if (ti.name == "per_layer_token_embd.weight") {  // the Engram table stays in host pages: read by row
                    engram_reader = std::make_unique<strix::engram::RowReader>(
                        part, std::vector<strix::engram::Table>{{ti.absolute_offset, ti.dimensions.at(1)}},
                        strix::engram::ReaderOptions{.row_bytes = static_cast<std::uint32_t>(ti.dimensions.at(0))});
                    continue;
                }
                if (!wanted(ti.name)) continue;
                std::uint8_t* dst = nullptr;
                const auto dot = ti.name.find(".engram_");
                if (dot != std::string::npos && (ti.name.ends_with("engram_hc_scale.weight") || ti.name.ends_with("engram_pre_gamma.weight"))) {
                    const auto layer = static_cast<std::uint32_t>(std::stoul(ti.name.substr(4, dot - 4)));
                    float* w1 = engram_w1.at(layer);
                    dst = reinterpret_cast<std::uint8_t*>(ti.name.ends_with("hc_scale.weight") ? w1 : w1 + 4U * std::size_t{c.hidden});
                } else {
                    dst = alloc<std::uint8_t>(ti.byte_size);
                }
                loader->load(ti.absolute_offset, ti.byte_size, dst);
                w[ti.name] = Tensor{dst, ti.ggml_type, ti.dimensions, ti.byte_size};
                report.bytes += ti.byte_size;
                ++report.tensors;
            }
            loaders.push_back(std::move(loader));
        }
        for (auto& l : loaders) {
            l->ready();
            report.queue_line += strix::core::weight_load_line(l->stats(), l->peak_pinned_bytes(), l->options()) + "\n";
        }
        report.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        bool engram_here = false;
        for (const auto l : c.engram_layers) engram_here = engram_here || (l >= first && l < last);
        if (engram_here && !engram_reader) throw std::runtime_error("hy4 hip forward: the pack has no Engram table");
    }

    void allocate() {
        const std::size_t h = c.hidden, hmax = 64U * c.v_head;
        x0 = alloc<float>(4U * h); x1 = alloc<float>(4U * h);
        u = alloc<float>(h); xn = alloc<float>(h); f = alloc<float>(h);
        partial = alloc<float>(std::max(hcc::mix_partial_floats<kSub>(1, c.hidden), hcc::mix_partial_floats<kHead>(1, c.hidden)));
        mix = alloc<hcc::MixRow<kSub>>(1); mixh = alloc<hcc::MixRow<kHead>>(1);
        gate = alloc<float>(hmax); qa = alloc<float>(c.q_lora); qv = alloc<float>(hmax);
        ckv = alloc<float>(c.kv_lora + c.qk_rope);
        qfull = alloc<float>(64U * std::size_t{c.kv_lora + c.qk_rope}); qt = alloc<std::uint16_t>(64U * std::size_t{c.kv_lora + c.qk_rope});
        olat = alloc<float>(64U * std::size_t{c.kv_lora}); o = alloc<float>(hmax);
        g = alloc<float>(c.dense_ff); up = alloc<float>(c.dense_ff); act = alloc<float>(c.dense_ff);
        rlog = alloc<float>(c.experts); rw = alloc<float>(c.top_k); ids = alloc<std::uint32_t>(c.top_k);
        z = alloc<float>(c.latent);
        pcodes = alloc<std::int8_t>(c.latent); pdeltas = alloc<float>(c.latent / 32U); psums = alloc<float>(c.latent / 32U);
        pout = alloc<float>(std::size_t{c.top_k} * 2U * c.expert_ff);
        dcodes = alloc<std::int8_t>(std::size_t{c.top_k} * c.expert_ff);
        ddeltas = alloc<float>(std::size_t{c.top_k} * c.expert_ff / 32U); dsums = alloc<float>(std::size_t{c.top_k} * c.expert_ff / 32U);
        rawdots = alloc<float>(std::size_t{c.top_k} * c.latent);
        acc = alloc<float>(c.latent); routed = alloc<float>(h); shared = alloc<float>(h);
        flag = alloc<std::uint32_t>(1); dpos = alloc<std::uint32_t>(4); list_n = alloc<std::uint32_t>(1);
        iota = alloc<std::uint32_t>(context + 1U);  // + 1: the batched prompt's causal list lengths t + 1 read iota + p0 + 1
        erows = alloc<std::uint8_t>(std::size_t{c.engram_heads()} * c.engram_dim);
        e = alloc<float>(std::size_t{c.engram_heads()} * c.engram_dim); kvout = alloc<float>(5U * h);
        if (last == c.layers) {
            logits_d = alloc<float>(c.vocab);
            logits.resize(c.vocab);
            greedy_bytes = strix::hip_greedy_rows_scratch_bytes(1U, c.vocab);
            greedy_partials = alloc<std::uint8_t>(greedy_bytes);
            void* reply = nullptr;  // host-mapped coherent: the reply is read after the stream drains, no copy
            check(hipHostMalloc(&reply, sizeof(strix::HipGreedyRowsReply), hipHostMallocMapped | hipHostMallocCoherent), "greedy reply");
            greedy_reply = static_cast<strix::HipGreedyRowsReply*>(reply);
        }
        if (c.layers > 64U) throw std::runtime_error("hy4 hip forward: more than 64 layers (decode staging)");
        void* st = nullptr;
        check(hipHostMalloc(&st, sizeof(Staging), hipHostMallocDefault), "decode staging");
        stage = static_cast<Staging*>(st);
        const std::size_t engram_slot = std::size_t{c.engram_heads()} * c.engram_dim;
        void* es = nullptr;
        check(hipHostMalloc(&es, std::max<std::size_t>(1U, c.engram_layers.size()) * engram_slot, hipHostMallocDefault), "engram staging");
        engram_stage = static_cast<std::uint8_t*>(es);
        dstatus_all = alloc<std::uint32_t>(8U * c.layers);
        store_epilogue = strix::envcfg::flag("STRIX_HY4_Q8_STORE_EPILOGUE", true);
        pf_store_epilogue = strix::envcfg::flag("STRIX_HY4_PF_STORE_EPILOGUE", true);
        const std::uint32_t launch_rows = rows::cache_rows(rmax, vr.max);  // rows a launch / verify pass may write (ring sizes)
        std::vector<std::uint32_t> seq(context + 1U);
        std::iota(seq.begin(), seq.end(), 0U);
        check(hipMemcpy(iota, seq.data(), seq.size() * 4U, hipMemcpyHostToDevice), "iota upload");
        check(hipMemset(flag, 0, 4), "flag clear");
        // caches: the plan's layouts (sliding ring / indexed by position), one position per launch
        latent.assign(c.layers, nullptr);
        latent_rows.assign(c.layers, {});
        for (std::uint32_t l = first; l < last; ++l) {
            const auto k = layer_caches(c, l, context, launch_rows);
            latent_rows[l] = k.latent;
            latent[l] = alloc<std::uint16_t>(k.latent_rows * k.latent.width);
        }
        // indexer caches (token ring + group keys) and the shared scratch of the select stages
        tkeys.assign(c.layers, nullptr);
        gkeys.assign(c.layers, nullptr);
        tk_rows.assign(c.layers, {});
        group_cap = (context + kIdxRatio - 1U) / kIdxRatio;
        bool any_indexed = false;
        for (std::uint32_t l = first; l < last; ++l) {
            if (c.indexed[l] == 0U) continue;
            if (c.index_dim != kIdxDim || c.index_heads != kIdxHeads || c.index_ratio != kIdxRatio ||
                c.index_groups != preset::indexer.group_budget)
                throw std::runtime_error("hy4 hip forward: indexer shape outside the model preset");
            const auto k = layer_caches(c, l, context, launch_rows);
            tk_rows[l] = k.token_keys;
            tkeys[l] = alloc<std::uint16_t>(k.token_rows * k.token_keys.width);
            gkeys[l] = alloc<std::uint16_t>(std::size_t{group_cap} * kIdxDim);
            any_indexed = true;
        }
        if (any_indexed) {
            ik = alloc<float>(kIdxDim); pooled = alloc<float>(kIdxDim);
            iq = alloc<float>(std::size_t{kIdxHeads} * kIdxDim); hwraw = alloc<float>(kIdxHeads);
            skeys = alloc<std::uint32_t>(group_cap); hist0 = alloc<std::uint32_t>(256);
            ilist = alloc<std::uint32_t>(kIndexListStride);
        }
        // RoPE tables [context][32] (BF16-valued cos / sin, the oracle's rounding)
        const auto table = [&](float theta, float*& cs, float*& sn) {
            const std::uint32_t pairs = c.qk_rope / 2U;
            std::vector<float> hc(std::size_t{context} * pairs), hs(hc.size());
            strix::ops::rope::build_native_table(theta, pairs, 0U, context, strix::ops::rope::Round::bf16, hc.data(), hs.data());
            cs = alloc<float>(hc.size()); sn = alloc<float>(hs.size());
            check(hipMemcpy(cs, hc.data(), hc.size() * 4U, hipMemcpyHostToDevice), "rope cos upload");
            check(hipMemcpy(sn, hs.data(), hs.size() * 4U, hipMemcpyHostToDevice), "rope sin upload");
        };
        table(c.rope_theta_indexed, cos_idx, sin_idx);
        table(c.rope_theta_sliding, cos_swa, sin_swa);
    }

    // Batched prompt workspaces ([rmax][...]) and the 256-wide attn_k_b images (hy4_hip_plan kKbPadInput).
    void allocate_prompt() {
        const std::size_t R = rmax, h = c.hidden, H = 64U, row = c.kv_lora + c.qk_rope, K = c.top_k, I = c.expert_ff, L = c.latent;
        const std::size_t eh = std::size_t{c.engram_heads()} * c.engram_dim;
        const std::size_t wide = std::max({H * c.qk_head(), H * c.kv_lora, std::size_t{c.dense_ff}, eh, 4U * h});
        pr.x0 = alloc<float>(R * 4U * h); pr.x1 = alloc<float>(R * 4U * h);
        if (first != 0U || last != c.layers) seam = alloc<float>(R * 4U * h);  // a stage of a layer split
        pr.u = alloc<float>(R * h); pr.xn = alloc<float>(R * h); pr.f = alloc<float>(R * h);
        pr.partial = alloc<float>(std::max(hcc::mix_partial_floats<kSub>(rmax, c.hidden), hcc::mix_partial_floats<kHead>(1, c.hidden)));
        pr.mix = alloc<hcc::MixRow<kSub>>(R);
        pr.codes = alloc<std::int8_t>(R * wide); pr.deltas = alloc<float>(R * wide / 32U); pr.sums = alloc<std::int32_t>(R * wide / 32U);
        pr.gate = alloc<float>(R * H * c.v_head); pr.qa = alloc<float>(R * c.q_lora); pr.qv = alloc<float>(R * H * c.qk_head());
        pr.ckv = alloc<float>(R * row); pr.ckn = alloc<float>(R * c.kv_lora);
        pr.qfull = alloc<float>(R * H * row); pr.qt = alloc<std::uint16_t>(R * H * row);
        pr.olat = alloc<float>(R * H * c.kv_lora); pr.o = alloc<float>(R * H * c.v_head);
        const std::size_t ff = std::max<std::size_t>(c.dense_ff, c.shared_ff);
        pr.g = alloc<float>(R * ff); pr.up = alloc<float>(R * ff); pr.act = alloc<float>(R * ff);
        pr.rlog = alloc<float>(R * c.experts); pr.ids = alloc<std::uint32_t>(R * K); pr.rw = alloc<float>(R * K);
        pr.z = alloc<float>(R * L); pr.zb = alloc<std::uint16_t>(R * L);
        pr.mcodes = alloc<std::int8_t>(R * L); pr.mdeltas = alloc<float>(R * L / 32U); pr.msums = alloc<std::int32_t>(R * L / 32U);
        pr.status = alloc<std::uint32_t>(R * 64U);  // 256-byte row stride (the grouped core's per-row status word)
        pr.hout = alloc<float>(R * K * 2U * I); pr.clear = alloc<float>(R * L);
        pr.dcodes = alloc<std::int8_t>(R * K * I); pr.ddeltas = alloc<float>(R * K * I / 32U); pr.dsums = alloc<std::int32_t>(R * K * I / 32U);
        pr.rawdots = alloc<float>(R * K * L);
        pr.csr = alloc<std::uint8_t>(strix::q51_known_rows::scratch_layout(rmax, c.experts, c.top_k, true).bytes);
        pr.acc = alloc<float>(R * L); pr.routed = alloc<float>(R * h); pr.shared = alloc<float>(R * h);
        pr.ik = alloc<float>(R * kIdxDim); pr.pooled = alloc<float>(R / kIdxRatio * kIdxDim);
        pr.iq = alloc<float>(R * kIdxHeads * kIdxDim); pr.hwraw = alloc<float>(R * kIdxHeads);
        pr.skeys = alloc<std::uint32_t>(R * group_cap); pr.hist0 = alloc<std::uint32_t>(R * 256U);
        pr.ilist = alloc<std::uint32_t>(R * kIndexListStride); pr.list_n = alloc<std::uint32_t>(R);
        pr.erows = alloc<std::uint8_t>(R * eh); pr.gate_off = alloc<std::uint8_t>(R);
        pr.e = alloc<float>(R * eh); pr.kvout = alloc<float>(R * 5U * h);
        std::vector<std::uint32_t> slot4(group_cap + 1U);
        for (std::size_t g4 = 0; g4 < slot4.size(); ++g4) slot4[g4] = static_cast<std::uint32_t>(g4 * kIdxRatio);
        pr.iota4 = alloc<std::uint32_t>(slot4.size());
        check(hipMemcpy(pr.iota4, slot4.data(), slot4.size() * 4U, hipMemcpyHostToDevice), "group slot upload");
        // attn_k_b [heads][kv_lora] rows of 192 Q8_0 -> rows of 256: the two added blocks are zero (scale 0, codes 0)
        kb_pad.assign(c.layers, nullptr);
        const std::size_t src = row_bytes(8U, c.qk_nope), dst = row_bytes(8U, kKbPadInput);
        for (std::uint32_t l = first; l < last; ++l) {
            const auto& kb = t(blk(l, "attn_k_b.weight"));
            const std::size_t rows = std::size_t{c.heads[l]} * c.kv_lora;
            if (kb.type != 8U || kb.bytes != rows * src) throw std::runtime_error("hy4 hip forward: attn_k_b is not Q8_0 [heads][kv_lora][192]");
            auto* p = alloc<std::uint8_t>(rows * dst);
            check(hipMemset(p, 0, rows * dst), "k_b image clear");
            check(hipMemcpy2D(p, dst, kb.p, src, src, rows, hipMemcpyDeviceToDevice), "k_b image");
            kb_pad[l] = p;
        }
        strix::moe_q4k_a8::known_rows::reserve_prefill_pools(((rmax * c.top_k + 9U) / 10U + 7U) / 8U * 8U);  // CSR pool: rows x 10
    }

    // ---- stages (each one shared entry) ----------------------------------------------------------------------
    void norm(const float* x, const float* weight, unsigned n, float eps, float* y) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(strix::ops::norm::rmsnorm_kernel<NormTraits, 256>), dim3(1), dim3(256), 0, s, x, weight, 0,
                           n, eps, y);
        check(hipGetLastError(), "rmsnorm launch");
    }
    // y = W x (+ bias), stored as the BF16 Linear output (round) or FP32 (router / head).
    void linear(const std::uint8_t* weights, std::uint32_t type, const float* x, std::uint32_t width, std::uint32_t outputs, float* y,
                bool round, const float* bias = nullptr) {
        if (type == 8U && width % 32U == 0U) {  // Q8_0: shared q8_rows direct form
            strix::q8_rows::LinearF32 site{};
            site.input[0] = x;
            site.segments = strix::q8_rows::segments({{weights, outputs}});
            site.width = width;
            site.outputs = outputs;
            site.out = y;
            site.bias = bias;
            site.round = round && store_epilogue;
            strix::q8_rows::launch(site, false, s);
            if (round && !store_epilogue) glu::launch_epilogue<glu::Round::bf16>(y, bias, outputs, outputs, y, s);
            check(hipGetLastError(), "q8 rows launch");
            return;
        }
        HipDenseRowsView v;
        v.x = x;
        v.input = HipActivationStorage::fp32;
        v.weights = weights;
        v.ggml_type = type;
        v.width = width;
        v.outputs = outputs;
        v.out = y;
        launch_hip_dense_rows(v, s);
        if (round) glu::launch_epilogue<glu::Round::bf16>(y, bias, outputs, outputs, y, s);
        check(hipGetLastError(), "linear epilogue launch");
    }
    void linear(const std::string& name, const float* x, std::uint32_t width, std::uint32_t outputs, float* y, bool round = true,
                const float* bias = nullptr) {
        const auto& tw = t(name);
        if (tw.dims.at(0) != width || tw.bytes != row_bytes(tw.type, width) * outputs)
            throw std::runtime_error("hy4 hip forward: shape of " + name);
        linear(tw.p, tw.type, x, width, outputs, y, round, bias);
    }

    void sync_time(double& slot, std::chrono::steady_clock::time_point& mark) {
        if (!timing) return;
        check(hipStreamSynchronize(s), "timing sync");
        const auto now = std::chrono::steady_clock::now();
        slot += std::chrono::duration<double, std::milli>(now - mark).count();
        mark = now;
    }

    void engram(std::uint32_t l, std::uint32_t ordinal, float* x) {
        const auto q = pos;
        if (tokens[q] == c.pad_token) return;  // masked position: the Engram output is zero
        std::vector<std::uint32_t> window(c.ngram);
        for (std::uint32_t k = 0; k < c.ngram; ++k) window[k] = q >= k ? tokens[q - k] : c.pad_token;
        const auto rows = strix::hy4::engram_rows(c, ordinal, window);
        std::vector<strix::engram::RowRequest> req;
        for (const auto r : rows) req.push_back({0U, r});
        // pinned slot of this Engram ordinal: the copy never waits for the stream, so the host read overlaps the GPU's work
        const std::size_t slot = std::size_t{c.engram_heads()} * c.engram_dim, bytes = rows.size() * c.engram_dim;
        if (bytes > slot) throw std::runtime_error("hy4 hip forward: Engram rows exceed the staging slot");
        const std::span<std::uint8_t> host(engram_stage + std::size_t{ordinal} * slot, bytes);
        engram_reader->read(req, host);
        check(hipMemcpyAsync(erows, host.data(), host.size(), hipMemcpyHostToDevice, s), "engram rows upload");
        strix::engram::RowPrepView pv;
        pv.form = strix::engram::RowPrepForm::head_rms_bf16;
        pv.fmt = strix::engram::RowFormat{96U, strix::engram::RowScale::none};
        pv.rows = erows;
        pv.row_count = rows.size();
        pv.out = e;
        pv.norm_weight = t(blk(l, "engram_lookup_norm.weight")).f();
        pv.heads = c.engram_heads();
        pv.eps = c.engram_norm_eps;
        strix::engram::hip::enqueue_row_prep<strix::engram::RowPrepForm::head_rms_bf16, strix::engram::RowScale::none, 96U>(pv, s);
        const auto width = c.engram_heads() * c.engram_dim;
        linear(blk(l, "engram_kv.weight"), e, width, 5U * c.hidden, kvout, true, t(blk(l, "engram_kv_b.weight")).f());
        strix::engram::InjectView iv;
        iv.form = strix::engram::InjectForm::normalized_bf16;
        iv.stream = x;
        iv.kv = kvout;
        iv.w0 = t(blk(l, "engram_gate.weight")).f();
        iv.w1 = engram_w1.at(l);
        iv.rows = 1;
        iv.hidden = c.hidden;
        iv.copies = 4;
        iv.eps = c.engram_norm_eps;
        strix::engram::hip::enqueue_inject<strix::engram::InjectForm::normalized_bf16>(iv, s);
        check(hipGetLastError(), "engram launch");
    }

    const Tensor& tensor_of(std::uint32_t l, const char* name, std::uint32_t type, std::uint64_t bytes) const {
        const auto& x = t(blk(l, name));
        if (x.type != type || x.bytes != bytes) throw std::runtime_error("hy4 hip forward: type / shape of " + blk(l, name));
        return x;
    }

    // Indexed layer, every position (also in the dense range): token key -> ring; at a group's 4th slot pool the group
    // (shared pool core), rotate the trailing 64 at its first slot (shared rope), store it in the group-key cache.
    void index_keys(std::uint32_t l) {
        namespace dcr = strix::attn_core::cache_rows;
        namespace rp = strix::ops::rope;
        linear(blk(l, "indexer.attn_k.weight"), xn, c.hidden, kIdxDim, ik);
        norm(ik, t(blk(l, "indexer.k_norm.weight")).f(), kIdxDim, c.rms_eps, ik);
        check(dcr::store_rows({ik, kIdxDim, dpos, 0U, 1U, tk_rows[l], tkeys[l], flag}, s), "token key store");
        if ((pos + 1U) % kIdxRatio != 0U) return;
        constexpr std::uint64_t q8_square = std::uint64_t{kIdxDim} * (kIdxDim / 32U) * 34U;
        const auto& wv = tensor_of(l, "indexer_compressor_value.weight", strix::hip_ggml::type_q8_0, q8_square);
        const auto& wg = tensor_of(l, "indexer_compressor_gate.weight", strix::hip_ggml::type_q8_0, q8_square);
        const auto& ape = tensor_of(l, "indexer_compressor_ape.weight", 0U, std::uint64_t{kIdxRatio} * kIdxDim * 4U);
        const auto& nw = tensor_of(l, "indexer_compressor_norm.weight", 0U, std::uint64_t{kIdxDim} * 4U);
        const PoolView v{tkeys[l], tk_rows[l].ring, dpos + 2, Q8Ape{wv.p, wg.p, ape.f(), nw.f(), c.rms_eps}, pooled};
        hipLaunchKernelGGL(HIP_KERNEL_NAME(strix::attn_core::group_pool::pool_groups_kernel<PoolView>), dim3(1), dim3(kIdxDim), 0, s, v);
        check(hipGetLastError(), "group pool launch");
        rp::launch<rp::Round::bf16>({pooled, 1U, 1U, kIdxDim, kIdxDim, kIdxDim / 2U, kIdxDim / 4U, cos_idx, sin_idx, dpos + 1, false}, s);
        check(dcr::store_rows({pooled, kIdxDim, dpos + 2, 0U, 1U, {kIdxDim, 0U}, gkeys[l], flag}, s), "group key store");
    }

    // Indexed layer past the dense range: query (q_resid = the normalised q_a), head weights, EXACT score scan of the
    // closed groups, G6 select -> ilist / list_n (511 groups + the own group's slots up to pos, ascending).
    void index_select(std::uint32_t l) {
        namespace rp = strix::ops::rope;
        namespace gs = strix::attn_core::group_scan;
        linear(blk(l, "indexer.attn_q_b.weight"), qa, c.q_lora, kIdxHeads * kIdxDim, iq);
        rp::launch<rp::Round::bf16>({iq, 1U, kIdxHeads, kIdxHeads * kIdxDim, kIdxDim, kIdxDim / 2U, kIdxDim / 4U, cos_idx, sin_idx, dpos, false}, s);
        linear(blk(l, "indexer.proj.weight"), xn, c.hidden, kIdxHeads, hwraw, false);
        check(hipMemsetAsync(hist0, 0, 256U * sizeof(std::uint32_t), s), "score histogram clear");
        const unsigned P = gi::visible_groups(preset::indexer, pos);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gs::score_kernel<ScanView>), dim3((P + gs::kThreads - 1U) / gs::kThreads, 1U), dim3(gs::kThreads),
                           0, s, ScanRows{gkeys[l], dpos}, static_cast<const float*>(iq), static_cast<const float*>(hwraw), skeys,
                           group_cap, hist0);
        check(hipGetLastError(), "group score launch");
        strix::attn_core::group_select::enqueue_select<SelectView>(strix::attn_core::group_select::PositionRows{dpos}, 1U, skeys, group_cap,
                                                                  hist0, ilist, list_n, s);
        check(hipGetLastError(), "group select launch");
    }

    void attention(std::uint32_t l) {  // xn -> f
        const std::uint32_t H = c.heads[l], qk = c.qk_head(), kl = c.kv_lora, rope = c.qk_rope, vd = c.v_head;
        const bool idx = c.indexed[l] != 0U;
        if (idx ? H != 32U : H != 64U) throw std::runtime_error("hy4 hip forward: head count outside the flash presets");
        const std::uint32_t row = kl + rope;
        linear(blk(l, "attn_gate.weight"), xn, c.hidden, H * vd, gate);
        linear(blk(l, "attn_q_a.weight"), xn, c.hidden, c.q_lora, qa);
        norm(qa, t(blk(l, "attn_q_a_norm.weight")).f(), c.q_lora, c.rms_eps, qa);
        linear(blk(l, "attn_q_b.weight"), qa, c.q_lora, H * qk, qv);
        const float* cs = idx ? cos_idx : cos_swa;
        const float* sn = idx ? sin_idx : sin_swa;
        strix::ops::rope::launch<strix::ops::rope::Round::bf16>({qv, 1U, H, H * qk, qk, c.qk_nope, rope / 2U, cs, sn, dpos, false}, s);
        linear(blk(l, "attn_kv_a_mqa.weight"), xn, c.hidden, row, ckv);
        norm(ckv, t(blk(l, "attn_kv_a_norm.weight")).f(), kl, c.rms_eps, ckv);
        strix::ops::rope::launch<strix::ops::rope::Round::bf16>({ckv, 1U, 1U, row, row, kl, rope / 2U, cs, sn, dpos, false}, s);
        namespace dcr = strix::attn_core::cache_rows;
        check(dcr::store_rows({ckv, row, dpos, 0U, 1U, latent_rows[l], latent[l], flag}, s), "latent row store");
        if (idx) index_keys(l);
        // absorbed query: q_lat[h] = K_b[h] q_nope[h] (rows of qk_nope), then the rope tail, as the bf16 operand image
        const auto& kb = t(blk(l, "attn_k_b.weight"));
        const auto kb_head = row_bytes(kb.type, c.qk_nope) * kl;
        if (kb.type == 8U) strix::q8_rows::launch_heads_f32(kb.p, H, c.qk_nope, kl, qv, qk, qfull, row, s);  // Q8_0: one launch
        else
            for (std::uint32_t h = 0; h < H; ++h)
                linear(kb.p + h * kb_head, kb.type, qv + std::size_t{h} * qk, c.qk_nope, kl, qfull + std::size_t{h} * row, false);
        check(hipMemcpy2DAsync(qfull + kl, row * 4U, qv + c.qk_nope, qk * 4U, rope * 4U, H, hipMemcpyDeviceToDevice, s), "q rope copy");
        check(dcr::store_rows({qfull, row, nullptr, 0U, H, {row, 0U}, qt, flag}, s), "query operand image");
        const float* sinks = t(blk(l, "attn_sinks.weight")).f();
        const std::uint32_t keys = pos + 1U;
        if (idx && gi::dense(preset::indexer, pos)) {  // every closed group selected: the causal prefix
            stage->list_n[l] = keys;  // pinned slot per layer (async copy, no wait)
            check(hipMemcpyAsync(list_n, &stage->list_n[l], 4, hipMemcpyHostToDevice, s), "list length");
            DsaPolicy::Args a{latent[l], iota, list_n, context, 0U, 0U, 0U, qt, sinks, olat};
            strix::attn_core::sparse::launch_flash_rows<DsaPolicy, 32U>(a, DsaSpec::kGroups, s, keys);
        } else if (idx) {
            index_select(l);
            DsaPolicy::Args a{latent[l], ilist, list_n, kIndexListStride, 0U, 0U, 0U, qt, sinks, olat};
            strix::attn_core::sparse::launch_flash_rows<DsaPolicy, 32U>(a, DsaSpec::kGroups, s, index_list_tokens(pos));
        } else {
            const auto ring = latent_rows[l].ring;
            SwaPolicy::Args a{latent[l], nullptr, nullptr, 0U, pos, c.window, ring - 1U, qt, sinks, olat};
            strix::attn_core::sparse::launch_flash_rows<SwaPolicy, 32U>(a, SwaSpec::kGroups, s, std::min(keys, c.window));
        }
        check(hipGetLastError(), "flash launch");
        const auto& vb = t(blk(l, "attn_v_b.weight"));
        const auto vb_head = row_bytes(vb.type, kl) * vd;
        if (vb.type == 8U) strix::q8_rows::launch_heads_f32(vb.p, H, kl, vd, olat, kl, o, vd, s);  // Q8_0: one launch
        else
            for (std::uint32_t h = 0; h < H; ++h)
                linear(vb.p + h * vb_head, vb.type, olat + std::size_t{h} * kl, kl, vd, o + std::size_t{h} * vd, false);
        glu::launch_epilogue<glu::Round::bf16>(o, nullptr, H * vd, std::size_t{H} * vd, o, s);
        glu::launch_sigmoid_gate<glu::DeclaredExp, glu::Round::bf16>(o, gate, std::size_t{H} * vd, o, s);
        linear(blk(l, "attn_output.weight"), o, H * vd, c.hidden, f);
        norm(f, t(blk(l, "post_attention_norm.weight")).f(), c.hidden, c.rms_eps, f);
        check(hipGetLastError(), "attention launch");
    }

    void ffn(std::uint32_t l) {  // xn -> f
        const std::uint32_t h = c.hidden;
        if (c.dense[l]) {
            linear(blk(l, "ffn_gate.weight"), xn, h, c.dense_ff, g);
            linear(blk(l, "ffn_up.weight"), xn, h, c.dense_ff, up);
            glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(g, up, c.dense_ff, glu::no_clamp, act, s);
            linear(blk(l, "ffn_down.weight"), act, c.dense_ff, h, f);
        } else {
            linear(blk(l, "ffn_gate_inp.weight"), xn, h, c.experts, rlog, false);
            strix::moe_core::enqueue_route_wave<preset::route, glu::DeclaredExp>(rlog, t(blk(l, "exp_probs_b.bias")).f(), ids, rw, 1U, s);
            linear(blk(l, "ffn_routed_down.weight"), xn, h, c.latent, z);
            const auto& wg = t(blk(l, "ffn_gate_exps.weight"));
            const auto& wu = t(blk(l, "ffn_up_exps.weight"));
            const auto& wd = t(blk(l, "ffn_down_exps.weight"));
            // this layer's two status words (zeroed once per step, read once at the step end: no per-layer host sync)
            std::uint32_t* const ps = dstatus_all + 8U * std::size_t{l};
            std::uint32_t* const ds = ps + 4;
            const auto dev =[](const void* p, std::uint64_t bytes) { return a8::DeviceBytes{reinterpret_cast<std::uintptr_t>(p), bytes}; };
            const std::uint64_t K = c.top_k, I = c.expert_ff, L = c.latent;
            a8::Request rq;
            rq.enabled = true;
            rq.approved_numerical_identity = a8::numerical_identity_for_gate_type(wg.type);
            rq.input = a8::InputKind::fp32;
            rq.shape = {c.latent, c.expert_ff, c.experts, c.top_k, 1U};
            rq.gate_type = wg.type;
            rq.up_type = wu.type;
            a8::Bindings b{dev(z, L * 4U), dev(wg.p, wg.bytes), dev(wu.p, wu.bytes), dev(ids, K * 4U), dev(pcodes, L),
                           dev(pdeltas, L / 32U * 4U), dev(psums, L / 32U * 4U), dev(ps, 4U), dev(pout, K * 2U * I * 4U)};
            a8::enqueue_probe(rq, b, s);
            a8::DownRequest dq;
            dq.enabled = true;
            dq.approved_numerical_identity = a8::down_numerical_identity(wd.type);
            dq.shape = {c.latent, c.expert_ff, c.experts, c.top_k};
            dq.weight_type = wd.type;
            dq.swiglu_limit = c.swiglu_limit;
            a8::DownBindings db{dev(pout, K * 2U * I * 4U), dev(wd.p, wd.bytes), dev(ids, K * 4U), dev(dcodes, K * I),
                                dev(ddeltas, K * I / 32U * 4U), dev(dsums, K * I / 32U * 4U), dev(rawdots, K * L * 4U),
                                dev(ds, 4U), {0U, 0U}};
            a8::enqueue_down_activation_quantize(dq, db, s);
            a8::enqueue_down_rawdots(dq, db, s);
            glu::launch_epilogue<glu::Round::bf16>(rawdots, nullptr, L, K * L, rawdots, s);  // each expert's down Linear output
            strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, 16U, 1536U>(rw, rawdots, nullptr, acc, 1U, s);
            linear(blk(l, "ffn_routed_up.weight"), acc, c.latent, h, routed);
            linear(blk(l, "ffn_gate_shexp.weight"), xn, h, c.shared_ff, g);
            linear(blk(l, "ffn_up_shexp.weight"), xn, h, c.shared_ff, up);
            glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(g, up, c.shared_ff, glu::no_clamp, act, s);
            linear(blk(l, "ffn_down_shexp.weight"), act, c.shared_ff, h, shared);
            strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, 1U, 3072U>(nullptr, routed, shared, f, 1U, s);
        }
        norm(f, t(blk(l, "post_ffw_norm.weight")).f(), h, c.rms_eps, f);
        check(hipGetLastError(), "ffn launch");
    }

    // one hyper-connection site: mixes, collapse, norm -> xn; then the sublayer; then expand (x0 <- expand)
    template <class Sub>
    void site(std::uint32_t l, const char* prefix, const char* norm_name, Sub&& sub) {
        const std::string p(prefix);
        check(hcc::enqueue_mixes<kSub>(x0, t(blk(l, (p + "_fn.weight").c_str())).f(), t(blk(l, (p + "_base.weight").c_str())).f(),
                                       t(blk(l, (p + "_scale.weight").c_str())).f(), 1U, c.hidden, 1U, c.hc_eps, c.rms_eps, mix, partial, s),
              "hc mixes");
        check(hcc::enqueue_collapse<kSub>(x0, mix, 1U, c.hidden, u, s), "hc collapse");
        norm(u, t(blk(l, norm_name)).f(), c.hidden, c.rms_eps, xn);
        sub();
        check(hcc::enqueue_expand<kSub>(hcc::SubSum{f, nullptr, nullptr}, x0, mix, 1U, c.hidden, x1, s), "hc expand");
        std::swap(x0, x1);
    }

    // ---- batched prompt: the same stages over a window of R rows (row r = position p0 + r) -----------------------
    void norm_rows(const float* x, const float* weight, unsigned n, unsigned R, float* y) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(strix::ops::norm::rmsnorm_kernel<NormTraits, 256>), dim3(R), dim3(256), 0, s, x, weight, 0, n,
                           c.rms_eps, y);
        check(hipGetLastError(), "rmsnorm rows launch");
    }
    // One A8 rows GEMM (shared core::f16_dense_tall::q8a8) over activation rows x [R][x_stride]; `batch` > 1 runs one product
    // per head (grid z) at x + z * batch_in, weights + z * batch_weight_bytes, y + z * batch_out.
    void q8a8_rows(const std::uint8_t* weights, std::uint32_t width, std::uint32_t outputs, std::uint32_t R, const float* x,
                   std::uint32_t x_stride, float* y, std::uint32_t y_stride, std::uint32_t batch = 1U, std::uint32_t batch_in = 0U,
                   std::uint32_t batch_out = 0U, bool round = false, const float* bias = nullptr) {
        namespace q8 = strix::core::f16_dense_tall::q8a8;
        strix::ds41::enqueue_quantize_a8(x, x_stride, R, pr.codes, pr.deltas, pr.sums, s);
        q8::View v{};
        v.weight = weights;
        v.codes = pr.codes;
        v.deltas = pr.deltas;
        v.out = y;
        v.row_bytes = static_cast<std::uint32_t>(row_bytes(8U, width));
        v.in = width;
        v.count = outputs;
        v.rows = R;
        v.x_stride = x_stride;
        v.out_stride = y_stride;
        v.batch = batch;
        v.batch_in = batch_in;
        v.batch_out = batch_out;
        v.batch_weight_bytes = std::uint64_t{outputs} * v.row_bytes;
        v.round_bf16 = round ? 1U : 0U;  // the Linear's bf16 epilogue (+ bias) in the GEMM's store: y = R(sum + bias)
        v.bias = round ? bias : nullptr;
        if (!q8::supported(v)) throw std::runtime_error("hy4 hip forward: A8 rows GEMM refuses width " + std::to_string(width));
        q8::enqueue(v, s);
        check(hipGetLastError(), "q8a8 rows launch");
    }
    // y[r] = W x[r] (+ bias): Q8_0 on the A8 rows GEMM, other types on the dense rows launcher; BF16 output when round.
    void linear_rows(const std::string& name, const float* x, std::uint32_t width, std::uint32_t outputs, std::uint32_t R, float* y,
                     bool round = true, const float* bias = nullptr) {
        const auto& tw = t(name);
        if (tw.dims.at(0) != width || tw.bytes != row_bytes(tw.type, width) * outputs)
            throw std::runtime_error("hy4 hip forward: shape of " + name);
        if (tw.type == 8U) {
            // pf_store_epilogue: the epilogue below runs in the GEMM's store instead (same function, same FP32 sum: same bits)
            const bool fused = round && pf_store_epilogue;
            q8a8_rows(tw.p, width, outputs, R, x, width, y, outputs, 1U, 0U, 0U, fused, bias);
            if (fused) {
                check(hipGetLastError(), "linear rows epilogue launch");
                return;
            }
        } else {
            HipDenseRowsView v;
            v.x = x;
            v.input = HipActivationStorage::fp32;
            v.weights = tw.p;
            v.ggml_type = tw.type;
            v.width = width;
            v.outputs = outputs;
            v.rows = R;
            v.out = y;
            launch_hip_dense_rows(v, s);
        }
        if (round) glu::launch_epilogue<glu::Round::bf16>(y, bias, outputs, std::size_t{R} * outputs, y, s);
        check(hipGetLastError(), "linear rows epilogue launch");
    }

    void engram_rows_stage(std::uint32_t l, std::uint32_t ordinal, std::uint32_t p0, std::uint32_t R) {
        const std::uint32_t heads = c.engram_heads();
        std::vector<strix::engram::RowRequest> req;
        req.reserve(std::size_t{R} * heads);
        std::vector<std::uint8_t> off(R, 0U);
        bool any_off = false;
        std::vector<std::uint32_t> window(c.ngram);
        for (std::uint32_t r = 0; r < R; ++r) {
            const auto q = p0 + r;
            if (tokens[q] == c.pad_token) { off[r] = 1U; any_off = true; }  // masked position: no Engram output
            for (std::uint32_t k = 0; k < c.ngram; ++k) window[k] = q >= k ? tokens[q - k] : c.pad_token;
            const auto rows = strix::hy4::engram_rows(c, ordinal, window);
            if (rows.size() != heads) throw std::runtime_error("hy4 hip forward: Engram rows per position");
            for (const auto x : rows) req.push_back({0U, x});
        }
        engram_host.resize(req.size() * c.engram_dim);
        engram_reader->read(req, engram_host);
        check(hipMemcpyAsync(pr.erows, engram_host.data(), engram_host.size(), hipMemcpyHostToDevice, s), "engram rows upload");
        strix::engram::RowPrepView pv;
        pv.form = strix::engram::RowPrepForm::head_rms_bf16;
        pv.fmt = strix::engram::RowFormat{96U, strix::engram::RowScale::none};
        pv.rows = pr.erows;
        pv.row_count = req.size();
        pv.out = pr.e;
        pv.norm_weight = t(blk(l, "engram_lookup_norm.weight")).f();
        pv.heads = heads;
        pv.eps = c.engram_norm_eps;
        strix::engram::hip::enqueue_row_prep<strix::engram::RowPrepForm::head_rms_bf16, strix::engram::RowScale::none, 96U>(pv, s);
        const auto width = heads * c.engram_dim;
        linear_rows(blk(l, "engram_kv.weight"), pr.e, width, 5U * c.hidden, R, pr.kvout, true, t(blk(l, "engram_kv_b.weight")).f());
        if (any_off) check(hipMemcpyAsync(pr.gate_off, off.data(), R, hipMemcpyHostToDevice, s), "engram gate-off upload");
        strix::engram::InjectView iv;
        iv.form = strix::engram::InjectForm::normalized_bf16;
        iv.stream = pr.x0;
        iv.kv = pr.kvout;
        iv.w0 = t(blk(l, "engram_gate.weight")).f();
        iv.w1 = engram_w1.at(l);
        iv.gate_off = any_off ? pr.gate_off : nullptr;
        iv.rows = R;
        iv.hidden = c.hidden;
        iv.copies = 4;
        iv.eps = c.engram_norm_eps;
        strix::engram::hip::enqueue_inject<strix::engram::InjectForm::normalized_bf16>(iv, s);
        if (any_off) check(hipStreamSynchronize(s), "engram gate-off sync");  // `off` is a local host buffer
        check(hipGetLastError(), "engram rows launch");
    }

    // Token keys of the window -> ring; the R / 4 groups the window closes (p0 % 8 == 0) pooled, rotated, stored.
    void index_keys_rows(std::uint32_t l, std::uint32_t p0, std::uint32_t R) {
        namespace dcr = strix::attn_core::cache_rows;
        namespace rp = strix::ops::rope;
        linear_rows(blk(l, "indexer.attn_k.weight"), pr.xn, c.hidden, kIdxDim, R, pr.ik);
        norm_rows(pr.ik, t(blk(l, "indexer.k_norm.weight")).f(), kIdxDim, R, pr.ik);
        check(dcr::store_rows({pr.ik, kIdxDim, iota + p0, 0U, R, tk_rows[l], tkeys[l], flag}, s), "token key rows store");
        const std::uint32_t g0 = p0 / kIdxRatio, ng = R / kIdxRatio;
        constexpr std::uint64_t q8_square = std::uint64_t{kIdxDim} * (kIdxDim / 32U) * 34U;
        const auto& wv = tensor_of(l, "indexer_compressor_value.weight", strix::hip_ggml::type_q8_0, q8_square);
        const auto& wg = tensor_of(l, "indexer_compressor_gate.weight", strix::hip_ggml::type_q8_0, q8_square);
        const auto& ape = tensor_of(l, "indexer_compressor_ape.weight", 0U, std::uint64_t{kIdxRatio} * kIdxDim * 4U);
        const auto& nw = tensor_of(l, "indexer_compressor_norm.weight", 0U, std::uint64_t{kIdxDim} * 4U);
        const PoolView v{tkeys[l], tk_rows[l].ring, iota + g0, Q8Ape{wv.p, wg.p, ape.f(), nw.f(), c.rms_eps}, pr.pooled};
        hipLaunchKernelGGL(HIP_KERNEL_NAME(strix::attn_core::group_pool::pool_groups_kernel<PoolView>), dim3(ng), dim3(kIdxDim), 0, s, v);
        check(hipGetLastError(), "group pool rows launch");
        rp::launch<rp::Round::bf16>({pr.pooled, ng, 1U, kIdxDim, kIdxDim, kIdxDim / 2U, kIdxDim / 4U, cos_idx, sin_idx, pr.iota4 + g0, false}, s);
        check(dcr::store_rows({pr.pooled, kIdxDim, iota + g0, 0U, ng, {kIdxDim, 0U}, gkeys[l], flag}, s), "group key rows store");
    }

    // Rows [a, a + n) of the window (positions p .. p + n - 1, all past the dense range): query, head weights, score scan
    // of each row's visible groups, G6 select -> pr.ilist [n][stride] / pr.list_n [n].
    void index_select_rows(std::uint32_t l, std::uint32_t p, std::uint32_t n, std::uint32_t a) {
        namespace rp = strix::ops::rope;
        namespace gs = strix::attn_core::group_scan;
        linear_rows(blk(l, "indexer.attn_q_b.weight"), pr.qa + std::size_t{a} * c.q_lora, c.q_lora, kIdxHeads * kIdxDim, n, pr.iq);
        rp::launch<rp::Round::bf16>({pr.iq, n, kIdxHeads, kIdxHeads * kIdxDim, kIdxDim, kIdxDim / 2U, kIdxDim / 4U, cos_idx, sin_idx, iota + p, false},
                                    s);
        linear_rows(blk(l, "indexer.proj.weight"), pr.xn + std::size_t{a} * c.hidden, c.hidden, kIdxHeads, n, pr.hwraw, false);
        check(hipMemsetAsync(pr.hist0, 0, std::size_t{n} * 256U * sizeof(std::uint32_t), s), "score histogram rows clear");
        const unsigned P = gi::visible_groups(preset::indexer, p + n - 1U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gs::score_kernel<ScanView>), dim3((P + gs::kThreads - 1U) / gs::kThreads, n), dim3(gs::kThreads), 0, s,
                           ScanRows{gkeys[l], iota + p}, static_cast<const float*>(pr.iq), static_cast<const float*>(pr.hwraw), pr.skeys, group_cap,
                           pr.hist0);
        check(hipGetLastError(), "group score rows launch");
        strix::attn_core::group_select::enqueue_select<SelectView>(strix::attn_core::group_select::PositionRows{iota + p}, n, pr.skeys, group_cap,
                                                                  pr.hist0, pr.ilist, pr.list_n, s);
        check(hipGetLastError(), "group select rows launch");
    }

    void attention_rows(std::uint32_t l, std::uint32_t p0, std::uint32_t R) {  // pr.xn -> pr.f
        namespace dcr = strix::attn_core::cache_rows;
        namespace rp = strix::ops::rope;
        const std::uint32_t H = c.heads[l], qk = c.qk_head(), kl = c.kv_lora, rope = c.qk_rope, vd = c.v_head, row = kl + rope;
        const bool idx = c.indexed[l] != 0U;
        if (idx ? H != 32U : H != 64U) throw std::runtime_error("hy4 hip forward: head count outside the flash presets");
        if (qk != kKbPadInput) throw std::runtime_error("hy4 hip forward: q head row is not the 256-wide K_b image input");
        const std::uint32_t* pp = iota + p0;
        linear_rows(blk(l, "attn_gate.weight"), pr.xn, c.hidden, H * vd, R, pr.gate);
        linear_rows(blk(l, "attn_q_a.weight"), pr.xn, c.hidden, c.q_lora, R, pr.qa);
        norm_rows(pr.qa, t(blk(l, "attn_q_a_norm.weight")).f(), c.q_lora, R, pr.qa);
        linear_rows(blk(l, "attn_q_b.weight"), pr.qa, c.q_lora, H * qk, R, pr.qv);
        const float* cs = idx ? cos_idx : cos_swa;
        const float* sn = idx ? sin_idx : sin_swa;
        rp::launch<rp::Round::bf16>({pr.qv, R, H, H * qk, qk, c.qk_nope, rope / 2U, cs, sn, pp, false}, s);
        linear_rows(blk(l, "attn_kv_a_mqa.weight"), pr.xn, c.hidden, row, R, pr.ckv);
        // the latent 512 of each 576-wide row normalised (the norm reads rows of n): gather, norm, scatter back
        check(hipMemcpy2DAsync(pr.ckn, kl * 4U, pr.ckv, row * 4U, kl * 4U, R, hipMemcpyDeviceToDevice, s), "latent gather");
        norm_rows(pr.ckn, t(blk(l, "attn_kv_a_norm.weight")).f(), kl, R, pr.ckn);
        check(hipMemcpy2DAsync(pr.ckv, row * 4U, pr.ckn, kl * 4U, kl * 4U, R, hipMemcpyDeviceToDevice, s), "latent scatter");
        rp::launch<rp::Round::bf16>({pr.ckv, R, 1U, row, row, kl, rope / 2U, cs, sn, pp, false}, s);
        check(dcr::store_rows({pr.ckv, row, pp, 0U, R, latent_rows[l], latent[l], flag}, s), "latent rows store");
        if (idx) index_keys_rows(l, p0, R);
        // absorbed query per head: K_b image (256 wide; the rope tail meets zero weights) x the q head row, batched over heads
        q8a8_rows(kb_pad[l], kKbPadInput, kl, R, pr.qv, H * qk, pr.qfull, H * row, H, qk, row);
        check(hipMemcpy2DAsync(pr.qfull + kl, row * 4U, pr.qv + c.qk_nope, qk * 4U, rope * 4U, std::size_t{R} * H, hipMemcpyDeviceToDevice, s),
              "q rope rows copy");
        check(dcr::store_rows({pr.qfull, row, nullptr, 0U, R * H, {row, 0U}, pr.qt, flag}, s), "query operand rows image");
        const float* sinks = t(blk(l, "attn_sinks.weight")).f();
        if (idx) {
            const std::uint32_t nd = index_dense_rows(p0, R);
            if (nd) {  // causal prefix: list = iota (stride 0), length t + 1 = iota[p0 + 1 + r]
                DsaPolicy::Args a{latent[l], iota, iota + p0 + 1U, 0U, 0U, 0U, 0U, pr.qt, sinks, pr.olat};
                strix::attn_core::sparse::launch_flash_rows<DsaPolicy, 32U>(a, nd * DsaSpec::kGroups, s, p0 + nd);
            }
            if (nd < R) {
                index_select_rows(l, p0 + nd, R - nd, nd);
                DsaPolicy::Args a{latent[l], pr.ilist, pr.list_n, kIndexListStride, 0U, 0U, 0U, pr.qt + std::size_t{nd} * H * row, sinks,
                                  pr.olat + std::size_t{nd} * H * kl};
                strix::attn_core::sparse::launch_flash_rows<DsaPolicy, 32U>(a, (R - nd) * DsaSpec::kGroups, s, kIndexListStride);
            }
        } else {
            SwaPolicy::Args a{latent[l], nullptr, nullptr, 0U, p0, c.window, latent_rows[l].ring - 1U, pr.qt, sinks, pr.olat};
            strix::attn_core::sparse::launch_flash_rows<SwaPolicy, 32U>(a, R * SwaSpec::kGroups, s, std::min(p0 + R, c.window));
        }
        check(hipGetLastError(), "flash rows launch");
        const auto& vb = t(blk(l, "attn_v_b.weight"));
        if (vb.type != 8U || vb.bytes != row_bytes(8U, kl) * vd * H) throw std::runtime_error("hy4 hip forward: attn_v_b is not Q8_0");
        q8a8_rows(vb.p, kl, vd, R, pr.olat, H * kl, pr.o, H * vd, H, kl, vd);
        const std::size_t n = std::size_t{R} * H * vd;
        glu::launch_epilogue<glu::Round::bf16>(pr.o, nullptr, n, n, pr.o, s);
        glu::launch_sigmoid_gate<glu::DeclaredExp, glu::Round::bf16>(pr.o, pr.gate, n, pr.o, s);
        linear_rows(blk(l, "attn_output.weight"), pr.o, H * vd, c.hidden, R, pr.f);
        norm_rows(pr.f, t(blk(l, "post_attention_norm.weight")).f(), c.hidden, R, pr.f);
        check(hipGetLastError(), "attention rows launch");
    }

    // Routed experts of the window on the shared grouped FP16 prefill core (known_rows::enqueue_prefill form 3, K-quant
    // envelope): raw slot dots pr.rawdots [R][top_k][latent].
    void experts_rows(std::uint32_t l, std::uint32_t R) {
        namespace kr = strix::moe_q4k_a8::known_rows;
        const auto& wg = t(blk(l, "ffn_gate_exps.weight"));
        const auto& wu = t(blk(l, "ffn_up_exps.weight"));
        const auto& wd = t(blk(l, "ffn_down_exps.weight"));
        const std::uint64_t K = c.top_k, I = c.expert_ff, L = c.latent;
        a8::Request rq;
        rq.enabled = true;
        rq.input = a8::InputKind::bf16;
        rq.shape = {c.latent, c.expert_ff, c.experts, c.top_k, R};
        rq.gate_type = wg.type;
        rq.up_type = wu.type;
        rq.swiglu_limit = c.swiglu_limit;
        kr::PrefillRows v;
        v.input = {pr.zb, L * 2U};
        v.routes = {pr.ids, K * 4U};
        v.route_weights = {pr.rw, K * 4U};
        v.codes = {pr.mcodes, L};
        v.deltas = {pr.mdeltas, L / 32U * 4U};
        v.sums = {pr.msums, L / 32U * 4U};
        v.status = {pr.status, 256U};
        v.output = {pr.hout, K * 2U * I * 4U};
        v.clear = {pr.clear, L * 4U};
        v.down_codes = {pr.dcodes, K * I};
        v.down_deltas = {pr.ddeltas, K * I / 32U * 4U};
        v.down_sums = {pr.dsums, K * I / 32U * 4U};
        v.rawdots = {pr.rawdots, K * L * 4U};
        v.rawdots_bf16 = pf_store_epilogue;  // each expert's Linear output rounded in the down kernel's store (ffn_rows skips its pass)
        const auto lay = strix::q51_known_rows::scratch_layout(R, c.experts, c.top_k, true);
        const auto ptr = [&](std::uint64_t at) { return reinterpret_cast<unsigned*>(pr.csr + at); };
        v.csr = {ptr(lay.header), ptr(lay.offsets), ptr(lay.touched), ptr(lay.rows), ptr(lay.slots), ptr(lay.batch_experts), ptr(lay.batch_begins)};
        kr::enqueue_prefill(rq, v, R, c.latent, wg.p, wu.p, wd.type, wd.p, s, 3);
        check(hipGetLastError(), "grouped expert prefill launch");
        status_host.assign(std::size_t{R} * 64U, 0U);
        check(hipMemcpyAsync(status_host.data(), pr.status, status_host.size() * 4U, hipMemcpyDeviceToHost, s), "expert rows status read");
        check(hipStreamSynchronize(s), "expert rows status sync");
        for (std::uint32_t r = 0; r < R; ++r)
            if (status_host[std::size_t{r} * 64U])
                throw std::runtime_error("hy4 hip forward: layer " + std::to_string(l) + " expert prefill status " +
                                         std::to_string(status_host[std::size_t{r} * 64U]) + " at window row " + std::to_string(r));
    }

    void ffn_rows(std::uint32_t l, std::uint32_t R) {  // pr.xn -> pr.f
        namespace dcr = strix::attn_core::cache_rows;
        const std::uint32_t h = c.hidden;
        if (c.dense[l]) {
            linear_rows(blk(l, "ffn_gate.weight"), pr.xn, h, c.dense_ff, R, pr.g);
            linear_rows(blk(l, "ffn_up.weight"), pr.xn, h, c.dense_ff, R, pr.up);
            glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(pr.g, pr.up, std::size_t{R} * c.dense_ff, glu::no_clamp, pr.act, s);
            linear_rows(blk(l, "ffn_down.weight"), pr.act, c.dense_ff, h, R, pr.f);
        } else {
            const std::uint64_t K = c.top_k, L = c.latent;
            linear_rows(blk(l, "ffn_gate_inp.weight"), pr.xn, h, c.experts, R, pr.rlog, false);
            strix::moe_core::enqueue_route_wave<preset::route, glu::DeclaredExp>(pr.rlog, t(blk(l, "exp_probs_b.bias")).f(), pr.ids, pr.rw, R, s);
            linear_rows(blk(l, "ffn_routed_down.weight"), pr.xn, h, c.latent, R, pr.z);
            check(dcr::store_rows({pr.z, L, nullptr, 0U, R, {c.latent, 0U}, pr.zb, flag}, s), "expert input rows (bf16)");
            experts_rows(l, R);
            if (!pf_store_epilogue)  // else the down kernel stored the rounded dots (rawdots_bf16)
                glu::launch_epilogue<glu::Round::bf16>(pr.rawdots, nullptr, L, R * K * L, pr.rawdots, s);  // each expert's Linear output
            strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, 16U, 1536U>(pr.rw, pr.rawdots, nullptr, pr.acc, R, s);
            linear_rows(blk(l, "ffn_routed_up.weight"), pr.acc, c.latent, h, R, pr.routed);
            linear_rows(blk(l, "ffn_gate_shexp.weight"), pr.xn, h, c.shared_ff, R, pr.g);
            linear_rows(blk(l, "ffn_up_shexp.weight"), pr.xn, h, c.shared_ff, R, pr.up);
            glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(pr.g, pr.up, std::size_t{R} * c.shared_ff, glu::no_clamp, pr.act, s);
            linear_rows(blk(l, "ffn_down_shexp.weight"), pr.act, c.shared_ff, h, R, pr.shared);
            strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, 1U, 3072U>(nullptr, pr.routed, pr.shared, pr.f, R, s);
        }
        norm_rows(pr.f, t(blk(l, "post_ffw_norm.weight")).f(), h, R, pr.f);
        check(hipGetLastError(), "ffn rows launch");
    }

    template <class Sub>
    void site_rows(std::uint32_t l, const char* prefix, const char* norm_name, std::uint32_t R, Sub&& sub) {
        const std::string p(prefix);
        check(hcc::enqueue_mixes<kSub>(pr.x0, t(blk(l, (p + "_fn.weight").c_str())).f(), t(blk(l, (p + "_base.weight").c_str())).f(),
                                       t(blk(l, (p + "_scale.weight").c_str())).f(), R, c.hidden, 1U, c.hc_eps, c.rms_eps, pr.mix, pr.partial, s),
              "hc mixes rows");
        check(hcc::enqueue_collapse<kSub>(pr.x0, pr.mix, R, c.hidden, pr.u, s), "hc collapse rows");
        norm_rows(pr.u, t(blk(l, norm_name)).f(), c.hidden, R, pr.xn);
        sub();
        check(hcc::enqueue_expand<kSub>(hcc::SubSum{pr.f, nullptr, nullptr}, pr.x0, pr.mix, R, c.hidden, pr.x1, s), "hc expand rows");
        std::swap(pr.x0, pr.x1);
    }

    // One window [p0, p0 + R) whose tokens are already in `tokens`; head = the last row's logits.
    void prefill_window(std::uint32_t p0, std::uint32_t R, bool head) {
        auto mark = std::chrono::steady_clock::now();
        const std::size_t n = 4U * std::size_t{c.hidden};
        if (first == 0U) {
            const auto& te = t("token_embd.weight");
            std::vector<std::uint32_t> four(4U * std::size_t{R});
            for (std::size_t i = 0; i < four.size(); ++i) four[i] = tokens[p0 + i / 4U];
            launch_hip_embedding_rows({te.p, te.type, c.hidden, c.vocab}, HipActivationStorage::fp32, four, pr.x0, c.hidden, s);
            glu::launch_epilogue<glu::Round::bf16>(pr.x0, nullptr, n * R, n * R, pr.x0, s);
        } else {  // X rows of the previous stage (layer pipeline seam)
            check(hipMemcpyAsync(pr.x0, seam, n * R * 4U, hipMemcpyDeviceToDevice, s), "seam rows in");
        }
        sync_time(times.embed, mark);
        for (std::uint32_t l = first; l < last; ++l) {
            const auto found = std::find(c.engram_layers.begin(), c.engram_layers.end(), l);
            if (found != c.engram_layers.end()) {
                engram_rows_stage(l, static_cast<std::uint32_t>(found - c.engram_layers.begin()), p0, R);
                sync_time(times.engram, mark);
            }
            site_rows(l, "hc_attn", "attn_norm.weight", R, [&] { attention_rows(l, p0, R); });
            sync_time(times.attention, mark);
            site_rows(l, "hc_ffn", "ffn_norm.weight", R, [&] { ffn_rows(l, R); });
            sync_time(times.ffn, mark);
        }
        std::uint32_t bad = 0;
        if (last != c.layers)  // X rows for the next stage (layer pipeline seam)
            check(hipMemcpyAsync(seam, pr.x0, n * R * 4U, hipMemcpyDeviceToDevice, s), "seam rows out");
        if (head) {
            const float* xl = pr.x0 + std::size_t{R - 1U} * n;
            check(hcc::enqueue_mixes<kHead>(xl, t("output_hc_fn.weight").f(), t("output_hc_base.weight").f(), t("output_hc_scale.weight").f(), 1U,
                                            c.hidden, 1U, c.hc_eps, c.rms_eps, mixh, partial, s),
                  "head mixes");
            check(hcc::enqueue_collapse<kHead>(xl, mixh, 1U, c.hidden, u, s), "head collapse");
            norm(u, t("output_norm.weight").f(), c.hidden, c.rms_eps, xn);
            linear("output.weight", xn, c.hidden, c.vocab, logits_d, false);
            check(hipMemcpyAsync(logits.data(), logits_d, logits.size() * 4U, hipMemcpyDeviceToHost, s), "logits read");
        }
        check(hipMemcpyAsync(&bad, flag, 4, hipMemcpyDeviceToHost, s), "flag read");
        check(hipStreamSynchronize(s), "window sync");
        sync_time(times.head, mark);
        if (bad) throw std::runtime_error("hy4 hip forward: non-finite value in a cache / operand row (prompt window)");
    }

    const std::vector<float>& prefill(const std::vector<std::uint32_t>& prompt) {
        if (first != 0U || last != c.layers) throw std::runtime_error("hy4 hip forward: the batched prompt runs on a whole-model rank");
        if (prompt.empty()) throw std::invalid_argument("hy4 hip forward: empty prompt");
        if (pos + prompt.size() > context) throw std::runtime_error("hy4 hip forward: context full");
        const auto plan = prompt_plan(pos, static_cast<std::uint32_t>(prompt.size()), rmax);
        std::vector<float> state;
        const std::vector<float>* out = &empty;
        std::size_t i = 0;
        for (std::uint32_t k = 0; k < plan.lead; ++k) out = &step(prompt[i++], state);
        const auto start = std::chrono::steady_clock::now();
        StepTimes sum{};
        for (std::size_t wi = 0; wi < plan.windows.size(); ++wi) {
            const auto win = plan.windows[wi];
            if (win.p0 != pos) throw std::logic_error("hy4 hip forward: prompt plan position");
            for (std::uint32_t r = 0; r < win.rows; ++r) {
                if (prompt[i] >= c.vocab) throw std::invalid_argument("hy4 hip forward: token outside the vocabulary");
                tokens.push_back(prompt[i++]);
            }
            times = {};
            const bool head = wi + 1U == plan.windows.size() && plan.tail == 0U;
            prefill_window(win.p0, win.rows, head);
            pos += win.rows;
            sum.embed += times.embed; sum.engram += times.engram; sum.attention += times.attention; sum.ffn += times.ffn; sum.head += times.head;
            if (head) out = &logits;
        }
        sum.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (std::uint32_t k = 0; k < plan.tail; ++k) out = &step(prompt[i++], state);
        times = sum;  // the windows' stage sums (the lead / tail steps are not included)
        return *out;
    }

    // One window of the shared layer pipeline on this stage (hy4_hip_plan stage_window_plan): the rows form over the
    // admissible head (input / output rows through `seam` on a partial range), single steps for the tail (< 16 rows,
    // their X rows through the same seam rows by host copy).
    const std::vector<float>& stage_window(std::span<const std::uint32_t> win, std::uint32_t start, bool head) {
        if (rmax == 0U) throw std::runtime_error("hy4 hip forward: the layer pipeline needs prompt rows");
        if (start != pos) throw std::logic_error("hy4 hip forward: stage window position");
        if (head && last != c.layers) throw std::logic_error("hy4 hip forward: head window on a stage without the head");
        if (pos + win.size() > context) throw std::runtime_error("hy4 hip forward: context full");
        const auto plan = stage_window_plan(start, static_cast<std::uint32_t>(win.size()), rmax);
        const std::size_t n = 4U * std::size_t{c.hidden};
        const std::vector<float>* out = &empty;
        std::size_t i = 0;
        for (const auto& pw : plan.windows) {  // at most one (rows <= rmax)
            for (std::uint32_t r = 0; r < pw.rows; ++r) {
                if (win[i] >= c.vocab) throw std::invalid_argument("hy4 hip forward: token outside the vocabulary");
                tokens.push_back(win[i++]);
            }
            times = {};
            const bool last_rows = head && plan.tail == 0U;
            prefill_window(pw.p0, pw.rows, last_rows);
            pos += pw.rows;
            if (last_rows) out = &logits;
        }
        std::vector<float> state(n);
        for (std::uint32_t k = 0; k < plan.tail; ++k, ++i) {
            if (first != 0U) {
                check(hipMemcpyAsync(state.data(), seam + i * n, n * 4U, hipMemcpyDeviceToHost, s), "seam row in");
                check(hipStreamSynchronize(s), "seam row sync");
            }
            out = &step(win[i], state);
            if (last != c.layers) {
                check(hipMemcpyAsync(seam + i * n, state.data(), n * 4U, hipMemcpyHostToDevice, s), "seam row out");
                check(hipStreamSynchronize(s), "seam row sync");
            }
        }
        return head ? *out : empty;
    }

    const std::vector<float>& step(std::uint32_t token, std::vector<float>& state, bool greedy = false) {
        if (pos >= context) throw std::runtime_error("hy4 hip forward: context full");
        if (token >= c.vocab) throw std::invalid_argument("hy4 hip forward: token outside the vocabulary");
        tokens.push_back(token);
        times = {};
        auto mark = std::chrono::steady_clock::now();
        const auto start = mark;
        hpos = {pos, pos / kIdxRatio * kIdxRatio, pos / kIdxRatio, 0U};
        stage->pos = hpos;  // pinned: the copy does not wait for the stream
        check(hipMemcpyAsync(dpos, stage->pos.data(), sizeof hpos, hipMemcpyHostToDevice, s), "position upload");
        check(hipMemsetAsync(dstatus_all, 0, 8U * c.layers * 4U, s), "expert status clear");
        const std::size_t n = 4U * std::size_t{c.hidden};
        if (first == 0U) {
            const auto& te = t("token_embd.weight");
            const std::array<std::uint32_t, 4> four{token, token, token, token};
            launch_hip_embedding_rows({te.p, te.type, c.hidden, c.vocab}, HipActivationStorage::fp32, four, x0, c.hidden, s);
            glu::launch_epilogue<glu::Round::bf16>(x0, nullptr, n, n, x0, s);  // the BF16 embedding module's output
        } else {
            if (state.size() != n) throw std::invalid_argument("hy4 hip forward: state size");
            check(hipMemcpyAsync(x0, state.data(), n * 4U, hipMemcpyHostToDevice, s), "state upload");
        }
        sync_time(times.embed, mark);
        for (std::uint32_t l = first; l < last; ++l) {
            const auto found = std::find(c.engram_layers.begin(), c.engram_layers.end(), l);
            if (found != c.engram_layers.end()) {
                engram(l, static_cast<std::uint32_t>(found - c.engram_layers.begin()), x0);
                sync_time(times.engram, mark);
            }
            site(l, "hc_attn", "attn_norm.weight", [&] { attention(l); });
            sync_time(times.attention, mark);
            site(l, "hc_ffn", "ffn_norm.weight", [&] { ffn(l); });
            sync_time(times.ffn, mark);
        }
        ++pos;
        std::uint32_t bad = 0;
        if (last == c.layers) {
            check(hcc::enqueue_mixes<kHead>(x0, t("output_hc_fn.weight").f(), t("output_hc_base.weight").f(), t("output_hc_scale.weight").f(), 1U,
                                            c.hidden, 1U, c.hc_eps, c.rms_eps, mixh, partial, s),
                  "head mixes");
            check(hcc::enqueue_collapse<kHead>(x0, mixh, 1U, c.hidden, u, s), "head collapse");
            norm(u, t("output_norm.weight").f(), c.hidden, c.rms_eps, xn);
            linear("output.weight", xn, c.hidden, c.vocab, logits_d, false);
            if (greedy) {  // the shared device greedy row: first max, lowest token on ties (no logits copy)
                const strix::ExecutionBufferAddress lg{{}, reinterpret_cast<std::uintptr_t>(logits_d), std::uint64_t{c.vocab} * 4U, 0U};
                const strix::ExecutionBufferAddress pa{{}, reinterpret_cast<std::uintptr_t>(greedy_partials), greedy_bytes, 0U};
                strix::enqueue_hip_greedy_rows(std::span<const strix::ExecutionBufferAddress>(&lg, 1U), c.vocab, pa, greedy_reply, s);
            } else {
                check(hipMemcpyAsync(logits.data(), logits_d, logits.size() * 4U, hipMemcpyDeviceToHost, s), "logits read");
            }
        } else {
            state.resize(n);
            check(hipMemcpyAsync(state.data(), x0, n * 4U, hipMemcpyDeviceToHost, s), "state read");
        }
        check(hipMemcpyAsync(&bad, flag, 4, hipMemcpyDeviceToHost, s), "flag read");
        check(hipMemcpyAsync(stage->status, dstatus_all, 8U * c.layers * 4U, hipMemcpyDeviceToHost, s), "expert status read");
        check(hipStreamSynchronize(s), "step sync");
        sync_time(times.head, mark);
        times.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (std::uint32_t l = first; l < last; ++l)  // the first layer whose A8 probe / down refused (as the per-layer check did)
            if (stage->status[8U * l] != 0U || stage->status[8U * l + 4U] != 0U)
                throw std::runtime_error("hy4 hip forward: layer " + std::to_string(l) + " expert status probe " +
                                         std::to_string(stage->status[8U * l]) + " down " + std::to_string(stage->status[8U * l + 4U]));
        if (bad) throw std::runtime_error("hy4 hip forward: non-finite value in a cache / operand row");
        if (greedy && last == c.layers) {
            if (greedy_reply->status != 0U) throw std::runtime_error("hy4 hip forward: non-finite logit (device greedy status)");
            greedy_token = greedy_reply->token;
        }
        return last == c.layers ? logits : empty;
    }

#include "hy4/hip/hy4_hip_rows.inc"  // verify rows (hy4_rows): forward_rows / commit_rows over the step's helpers
};

Forward::Forward(const ForwardOptions& o) : impl_(std::make_unique<Impl>()) {
    if (o.parts.empty()) throw std::invalid_argument("hy4 hip forward: no pack parts");
    const auto full = Config::from_gguf(o.parts.front());
    impl_->c = o.depth == 0U ? full : full.truncated(o.depth);
    impl_->first = o.first;
    impl_->last = o.last == 0U ? impl_->c.layers : o.last;
    impl_->context = o.context;
    impl_->rmax = o.prompt_rows;
    if (impl_->first >= impl_->last || impl_->last > impl_->c.layers || o.context == 0U)
        throw std::invalid_argument("hy4 hip forward: layer range / context");
    if (o.prompt_rows != 0U && !prompt_window_ok(o.prompt_rows))
        throw std::invalid_argument("hy4 hip forward: prompt rows must be a multiple of 8, at least 16");
    if (o.verify_rows > rows::kMaxVerifyRows) throw std::invalid_argument("hy4 hip forward: verify rows outside 0..8");
    impl_->vr.max = o.verify_rows;
    check(hipStreamCreateWithFlags(&impl_->s, hipStreamNonBlocking), "stream");
    impl_->load(o.parts);
    impl_->allocate();
    if (impl_->rmax) impl_->allocate_prompt();
    if (impl_->vr.max) impl_->allocate_verify();
}
void Forward::forward_rows(std::span<const std::uint32_t> tokens, bool logits) { impl_->forward_rows(tokens, logits); }
std::span<const std::uint32_t> Forward::rows_greedy() const noexcept { return impl_->vr.greedy; }
const std::vector<float>& Forward::rows_logits() const noexcept { return impl_->vr.logits_host; }
void Forward::commit_rows(std::uint32_t k) { impl_->commit_rows(k); }
void Forward::set_rows_expert_union(bool on) noexcept { impl_->vr.expert_union = on; }
const std::vector<float>& Forward::prefill(const std::vector<std::uint32_t>& prompt) { return impl_->prefill(prompt); }
Forward::~Forward() {
    if (impl_ && impl_->s) {
        (void)hipStreamSynchronize(impl_->s);
        (void)hipStreamDestroy(impl_->s);
    }
    if (impl_ && impl_->greedy_reply) (void)hipHostFree(impl_->greedy_reply);
    if (impl_ && impl_->stage) (void)hipHostFree(impl_->stage);
    if (impl_ && impl_->engram_stage) (void)hipHostFree(impl_->engram_stage);
}
const Config& Forward::config() const noexcept { return impl_->c; }
const LoadReport& Forward::load_report() const noexcept { return impl_->report; }
std::uint32_t Forward::position() const noexcept { return impl_->pos; }
void Forward::rewind() {
    if (impl_->pos == 0U || impl_->tokens.empty()) throw std::logic_error("hy4 hip forward: rewind before any step");
    --impl_->pos;
    impl_->tokens.pop_back();
}
bool Forward::has_head() const noexcept { return impl_->last == impl_->c.layers; }
bool Forward::has_embedding() const noexcept { return impl_->first == 0U; }
const std::vector<float>& Forward::step(std::uint32_t token, std::vector<float>& state) { return impl_->step(token, state); }
std::uint32_t Forward::step_greedy(std::uint32_t token) {
    if (!has_head()) throw std::logic_error("hy4 hip forward: step_greedy needs the head");
    std::vector<float> state;
    (void)impl_->step(token, state, true);
    return impl_->greedy_token;
}
std::uint32_t Forward::step_greedy(std::uint32_t token, std::vector<float>& state) {
    if (!has_head()) throw std::logic_error("hy4 hip forward: step_greedy needs the head");
    (void)impl_->step(token, state, true);
    return impl_->greedy_token;
}
const std::vector<float>& Forward::prefill_window(std::span<const std::uint32_t> tokens, std::uint32_t start, bool head) {
    return impl_->stage_window(tokens, start, head);
}
void* Forward::seam_rows() const noexcept { return impl_->seam; }
std::size_t Forward::seam_row_bytes() const noexcept { return 4U * std::size_t{impl_->c.hidden} * sizeof(float); }
void* Forward::stream_handle() const noexcept { return impl_->s; }
void Forward::sync() { check(hipStreamSynchronize(impl_->s), "forward sync"); }
void Forward::reset() noexcept {
    impl_->pos = 0;
    impl_->tokens.clear();
    impl_->times = {};
}
void Forward::set_timing(bool on) noexcept { impl_->timing = on; }
const StepTimes& Forward::times() const noexcept { return impl_->times; }

}  // namespace strix::hy4::hipbind
