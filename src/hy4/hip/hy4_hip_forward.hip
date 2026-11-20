// Hy4-flash HIP forward (contract: hy4_hip_forward.hpp). Binding only: every stage calls a shared entry of
// hy4_hip_plan.hpp; the model data here are tensor names, the layer schedule and the Traits / presets.
// Compile with -ffp-contract=off -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt (declared-arithmetic ops).
// The HIP runtime comes first: the shared weight sources call std::memcpy in device code, and <cstring>'s
// using-declaration sees the device overload only when the runtime was included before it.
#include <hip/hip_runtime.h>
#include "hy4/hip/hy4_hip_forward.hpp"
#include "hy4/hip/hy4_hip_plan.hpp"
#include "hy4/hip/hy4_hip_rows.hpp"
#include "strix/hy4/tp_layout.hpp"

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
#include "strix/a8_row_pitch.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/engram/read_ahead.hpp"
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
namespace htp = strix::hy4::tp;

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
// Tensor-parallel share of an indexed layer (Forward::enable_tp): 16 of the 32 heads per rank, one M tile of the same
// kernel (H = 16 is a declared tile, sparse_flash_wmma.hpp Geometry). A sliding layer's share is one of SwaSpec's two
// groups: SwaPolicy itself over one row (grid 1) with the group's qt / sinks / out offsets.
struct DsaHalfSpec : DsaSpec {
    static constexpr unsigned kHeads = 16U;
};
using DsaPolicy = mla::LatentPolicy<DsaSpec>;
using DsaHalfPolicy = mla::LatentPolicy<DsaHalfSpec>;
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
// The same pooling weights stored BF16 (the next-token block's indexer): value / gate [128][128] (out, in), exact values.
struct Bf16Ape {
    const std::uint16_t* wv;
    const std::uint16_t* wg;
    const float* ape_w;
    const float* norm_w;
    float eps_v;
    __device__ float v(unsigned o, unsigned i) const { return gpl::bf16_value(wv[static_cast<std::size_t>(o) * kIdxDim + i]); }
    __device__ float g(unsigned o, unsigned i) const { return gpl::bf16_value(wg[static_cast<std::size_t>(o) * kIdxDim + i]); }
    __device__ float ape(unsigned j, unsigned o) const { return ape_w[static_cast<std::size_t>(j) * kIdxDim + o]; }
    __device__ float norm(unsigned o) const { return norm_w[o]; }
    __device__ float eps() const { return eps_v; }
};

// Block b pools group *first + b from the token-key ring; out [b][128] = the bf16-valued pooled key before RoPE.
// W: the pooling weights' storage (Q8Ape / Bf16Ape), a trait of the one shared pool kernel.
template <class W>
struct PoolViewOf {
    static constexpr unsigned kRatio = kIdxRatio, kDim = kIdxDim;
    const std::uint16_t* ring;   // token keys bf16 [ring_rows][128], slot = position % ring_rows
    unsigned ring_rows;
    const std::uint32_t* first;  // device: first group of the launch
    W w;
    float* out;
    __device__ bool live(unsigned) const { return true; }
    __device__ unsigned group(unsigned b) const { return *first + b; }
    __device__ float key(unsigned g, unsigned j, unsigned i) const {
        return gpl::bf16_value(ring[static_cast<std::size_t>((g * kRatio + j) % ring_rows) * kDim + i]);
    }
    __device__ const W& weights() const { return w; }
    __device__ float exp(float x) const { return strix::ds41::declared_exp(x); }
    __device__ void store(unsigned g, unsigned lane, const float* o) const {
        out[static_cast<std::size_t>(g - *first) * kDim + lane] = o[lane];
    }
};
using PoolView = PoolViewOf<Q8Ape>;

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

constexpr std::uint32_t kTypeBf16 = 30U;  // GGML_TYPE_BF16 (the next-token block's storage)

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
    bool q_image = false;           // pf_store_epilogue: K_b stores the bf16 query operand image in the GEMM (allocate_prompt)
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

    // exact tensor parallelism (Forward::enable_tp; strix/hy4/tp_layout.hpp): world 1 = the one-node step, unchanged
    struct Tp {
        std::uint32_t rank = 0, world = 1;
        Forward::TpGather gather;
        float* moe = nullptr;  // [2][slot] MoE gather buffer (tp::moe layout)
        std::uint64_t calls = 0;
    } tp;

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
        std::uint32_t* qtail = nullptr;  // q_image: [k] = row of query row k's rope tail in pr.qt read as rope-wide rows
        std::uint8_t *erows = nullptr, *gate_off = nullptr;
        float *e = nullptr, *kvout = nullptr;
    } pr;
    std::vector<const std::uint8_t*> kb_pad;
    // Prompt-window Engram read-ahead (engram_submit): tickets per Engram ordinal of the window starting at the key.
    struct EngramWindow {
        std::uint32_t rows = 0;
        std::vector<std::uint64_t> tickets;
    };
    static constexpr std::uint64_t kNoEngramTicket = ~std::uint64_t{0};
    std::map<std::uint32_t, EngramWindow> engram_windows;
    std::unique_ptr<strix::engram::ReadAhead> engram_ahead;  // after engram_reader: destroyed (joined) before it
    std::vector<std::uint32_t> status_host;

    // next-token (MTP) block, block id c.layers (ForwardOptions::nextn): output rows, its workspace, the prompt sink
    struct Nextn {
        bool on = false;
        std::uint32_t* top = nullptr;              // pinned host [kNextnRows]: the greedy id of a want_top row
        float *pre = nullptr, *post = nullptr;     // device [kNextnRows][hidden]
        float *x = nullptr, *cat = nullptr, *act = nullptr;  // [hidden] residual, [2 hidden] eh_proj input, [top_k][expert_ff]
        bool row_loop = false;                     // one row at a time (the rows form's identity reference)
        float *xr = nullptr, *catr = nullptr;      // rows form: [vr.max][hidden] residual rows, [vr.max][2 hidden] eh_proj input
        std::uint32_t* words = nullptr;            // rows form: pinned [kNextnRows][4] position words, one slot per row of a call
        Staging* stages = nullptr;  // pinned, one per row of a call: rows run with no host sync between them, so each row's
                                    // async uploads (position, dense list length) read their own slot
        std::vector<float> logits;                 // host copy of the last want_top row (nextn_logits)
        bool logits_ready = false;
        float* ph = nullptr;                       // [rmax][hidden] prompt rows: output_norm(hc_head collapse)
        float* pu = nullptr;                       // [rmax][hidden] collapse
        hcc::MixRow<kHead>* mixh = nullptr;        // [rmax]
        float* partial = nullptr;
        std::function<void(const float*, std::uint32_t, std::uint32_t)> sink;
    } nx;

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
        if (nx.on && layer == c.layers) return true;  // the next-token block (whole-model rank only)
        return layer >= first && layer < last;
    }
    // per-block attention / FFN form: the text layers' arrays, else the next-token block's (hy4_host_ref State::heads_of)
    [[nodiscard]] std::uint32_t heads_of(std::uint32_t l) const { return l < c.layers ? c.heads[l] : c.nextn_heads; }
    [[nodiscard]] bool indexed_of(std::uint32_t l) const { return l < c.layers ? c.indexed[l] != 0U : c.nextn_indexed != 0U; }
    [[nodiscard]] bool dense_of(std::uint32_t l) const { return l < c.layers && c.dense[l] != 0U; }

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
        const std::size_t a8w = wide + strix::kA8PitchPadBytes;  // any strix::a8_row_pitch(x_stride <= wide) fits
        pr.codes = alloc<std::int8_t>(R * a8w); pr.deltas = alloc<float>(R * a8w / 32U); pr.sums = alloc<std::int32_t>(R * a8w / 32U);
        pr.gate = alloc<float>(R * H * c.v_head); pr.qa = alloc<float>(R * c.q_lora); pr.qv = alloc<float>(R * H * c.qk_head());
        pr.ckv = alloc<float>(R * row); pr.ckn = alloc<float>(R * c.kv_lora);
        // pf_store_epilogue: K_b stores the query operand image in its GEMM (no FP32 qfull); the rope tail of query row k
        // is row subrow(k) of pr.qt read as rope-wide rows (cache_rows contract), written by the shared row writer.
        namespace acr = strix::attn::cache_rows;
        q_image = pf_store_epilogue && acr::subrow_ok(c.kv_lora + c.qk_rope, c.qk_rope, c.kv_lora);
        if (!q_image) pr.qfull = alloc<float>(R * H * row);
        pr.qt = alloc<std::uint16_t>(R * H * row);
        if (q_image) {
            std::vector<std::uint32_t> tail(R * H);
            for (std::size_t k = 0; k < tail.size(); ++k)
                tail[k] = acr::subrow(static_cast<std::uint32_t>(k), c.kv_lora + c.qk_rope, c.qk_rope, c.kv_lora);
            pr.qtail = alloc<std::uint32_t>(tail.size());
            check(hipMemcpy(pr.qtail, tail.data(), tail.size() * 4U, hipMemcpyHostToDevice), "query tail rows upload");
        }
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

    // linear() (no rounding) over `groups` weight slices of [outputs][width] in ONE launch of the shared grouped dense rows:
    // group g = slice gids[g] (device ids, no host read) or g, input x + g * x_stride, output y + g * y_stride; every output
    // bitwise equal to its own linear() call (tests/hip_dense_rows_tile_tests.hip).
    void linear_groups(const std::uint8_t* weights, std::uint32_t type, std::uint64_t slice_bytes, std::uint32_t slices,
                       const std::uint32_t* gids, std::uint32_t groups, const float* x, std::uint32_t x_stride, std::uint32_t width,
                       std::uint32_t outputs, float* y, std::uint32_t y_stride) {
        HipDenseRowsView v;
        v.x = x;
        v.input = HipActivationStorage::fp32;
        v.weights = weights;
        v.ggml_type = type;
        v.width = width;
        v.outputs = outputs;
        v.out = y;
        v.groups = groups;
        v.group_ids = gids;
        v.group_slices = slices;
        v.group_weight_bytes = slice_bytes;
        v.group_x_stride = x_stride;
        v.group_out_stride = y_stride;
        launch_hip_dense_rows(v, s);
        check(hipGetLastError(), "grouped linear launch");
    }

    // ---- exact tensor parallelism (Forward::enable_tp; strix/hy4/tp_layout.hpp) ----
    // Rows [r.first, r.first + r.count) of the Linear `name` (outputs rows) into y + r.first: the same launch as linear()
    // over fewer rows. Every output row is its own dot product, so the rows a rank computes carry the one-node bits; the
    // whole range is the one-node call itself.
    void linear_rows(const std::string& name, const float* x, std::uint32_t width, std::uint32_t outputs, htp::Range r, float* y,
                     bool round = true) {
        const auto& tw = t(name);
        if (tw.dims.at(0) != width || tw.bytes != row_bytes(tw.type, width) * outputs || r.first + r.count > outputs)
            throw std::runtime_error("hy4 hip forward: shape of " + name);
        linear(tw.p + row_bytes(tw.type, width) * r.first, tw.type, x, width, static_cast<std::uint32_t>(r.count), y + r.first, round);
    }
    [[nodiscard]] htp::Range share(std::size_t n) const { return tp.world == 1U ? htp::Range{0U, n} : htp::half(n, tp.rank); }
    // The symmetric swap of the two halves of `buffer` (slot = one half), stream-ordered after its producers.
    void tp_gather(void* buffer, std::size_t slot_bytes) {
        tp.gather.fn(tp.gather.ctx, s, buffer, slot_bytes, slot_bytes);
        ++tp.calls;
    }
    void allocate_tp() {
        if (tp.moe == nullptr) tp.moe = alloc<float>(2U * htp::moe(0U, c.top_k, c.latent, c.shared_ff).slot);
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
        constexpr std::uint64_t bf16_square = std::uint64_t{kIdxDim} * kIdxDim * 2U;
        const auto& ape = tensor_of(l, "indexer_compressor_ape.weight", 0U, std::uint64_t{kIdxRatio} * kIdxDim * 4U);
        const auto& nw = tensor_of(l, "indexer_compressor_norm.weight", 0U, std::uint64_t{kIdxDim} * 4U);
        if (t(blk(l, "indexer_compressor_value.weight")).type == kTypeBf16) {  // the next-token block stores them BF16
            const auto& wv = tensor_of(l, "indexer_compressor_value.weight", kTypeBf16, bf16_square);
            const auto& wg = tensor_of(l, "indexer_compressor_gate.weight", kTypeBf16, bf16_square);
            const PoolViewOf<Bf16Ape> v{tkeys[l], tk_rows[l].ring, dpos + 2,
                                        Bf16Ape{reinterpret_cast<const std::uint16_t*>(wv.p), reinterpret_cast<const std::uint16_t*>(wg.p),
                                                ape.f(), nw.f(), c.rms_eps},
                                        pooled};
            hipLaunchKernelGGL(HIP_KERNEL_NAME(strix::attn_core::group_pool::pool_groups_kernel<PoolViewOf<Bf16Ape>>), dim3(1), dim3(kIdxDim), 0,
                               s, v);
        } else {
            const auto& wv = tensor_of(l, "indexer_compressor_value.weight", strix::hip_ggml::type_q8_0, q8_square);
            const auto& wg = tensor_of(l, "indexer_compressor_gate.weight", strix::hip_ggml::type_q8_0, q8_square);
            const PoolView v{tkeys[l], tk_rows[l].ring, dpos + 2, Q8Ape{wv.p, wg.p, ape.f(), nw.f(), c.rms_eps}, pooled};
            hipLaunchKernelGGL(HIP_KERNEL_NAME(strix::attn_core::group_pool::pool_groups_kernel<PoolView>), dim3(1), dim3(kIdxDim), 0, s, v);
        }
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

    void attention(std::uint32_t l) {  // xn -> f (with enable_tp: this rank's heads, o and o_proj rows joined)
        const std::uint32_t H = heads_of(l), qk = c.qk_head(), kl = c.kv_lora, rope = c.qk_rope, vd = c.v_head;
        const bool idx = indexed_of(l);
        if (idx ? H != 32U : H != 64U) throw std::runtime_error("hy4 hip forward: head count outside the flash presets");
        const std::uint32_t row = kl + rope;
        const bool par = tp.world == 2U;
        const auto hs = share(H);  // this rank's heads [h0, h0 + Hs)
        const auto h0 = static_cast<std::uint32_t>(hs.first), Hs = static_cast<std::uint32_t>(hs.count);
        linear_rows(blk(l, "attn_gate.weight"), xn, c.hidden, H * vd, {std::size_t{h0} * vd, std::size_t{Hs} * vd}, gate);
        linear(blk(l, "attn_q_a.weight"), xn, c.hidden, c.q_lora, qa);
        norm(qa, t(blk(l, "attn_q_a_norm.weight")).f(), c.q_lora, c.rms_eps, qa);
        linear_rows(blk(l, "attn_q_b.weight"), qa, c.q_lora, H * qk, {std::size_t{h0} * qk, std::size_t{Hs} * qk}, qv);
        const float* cs = idx ? cos_idx : cos_swa;
        const float* sn = idx ? sin_idx : sin_swa;
        strix::ops::rope::launch<strix::ops::rope::Round::bf16>(
            {qv + std::size_t{h0} * qk, 1U, Hs, H * qk, qk, c.qk_nope, rope / 2U, cs, sn, dpos, false}, s);
        linear(blk(l, "attn_kv_a_mqa.weight"), xn, c.hidden, row, ckv);
        norm(ckv, t(blk(l, "attn_kv_a_norm.weight")).f(), kl, c.rms_eps, ckv);
        strix::ops::rope::launch<strix::ops::rope::Round::bf16>({ckv, 1U, 1U, row, row, kl, rope / 2U, cs, sn, dpos, false}, s);
        namespace dcr = strix::attn_core::cache_rows;
        check(dcr::store_rows({ckv, row, dpos, 0U, 1U, latent_rows[l], latent[l], flag}, s), "latent row store");
        if (idx) index_keys(l);
        // absorbed query: q_lat[h] = K_b[h] q_nope[h] (rows of qk_nope), then the rope tail, as the bf16 operand image
        const auto& kb = t(blk(l, "attn_k_b.weight"));
        const auto kb_head = row_bytes(kb.type, c.qk_nope) * kl;
        const std::size_t q0 = std::size_t{h0} * qk, r0 = std::size_t{h0} * row, l0 = std::size_t{h0} * kl, v0 = std::size_t{h0} * vd;
        if (kb.type == 8U)  // Q8_0: one launch
            strix::q8_rows::launch_heads_f32(kb.p + h0 * kb_head, Hs, c.qk_nope, kl, qv + q0, qk, qfull + r0, row, s);
        else  // one grouped launch over this rank's heads
            linear_groups(kb.p + h0 * kb_head, kb.type, kb_head, Hs, nullptr, Hs, qv + q0, qk, c.qk_nope, kl, qfull + r0, row);
        check(hipMemcpy2DAsync(qfull + r0 + kl, row * 4U, qv + q0 + c.qk_nope, qk * 4U, rope * 4U, Hs, hipMemcpyDeviceToDevice, s),
              "q rope copy");
        check(dcr::store_rows({qfull + r0, row, nullptr, h0, Hs, {row, 0U}, qt, flag}, s), "query operand image");
        const float* sinks = t(blk(l, "attn_sinks.weight")).f() + h0;
        const std::uint16_t* qs = qt + r0;  // this rank's heads of the operand image (one group: row 0 of the launch)
        float* const os = olat + l0;
        const std::uint32_t keys = pos + 1U;
        // the flash over this rank's heads: one node = every group; a share = one 16-head tile (indexed) or one of the
        // two 32-head groups (sliding) as row 0 of the same kernels
        const auto dsa = [&](const std::uint32_t* lists, const std::uint32_t* counts, unsigned stride, unsigned max_keys) {
            if (par) {
                DsaHalfPolicy::Args a{latent[l], lists, counts, stride, 0U, 0U, 0U, qs, sinks, os};
                strix::attn_core::sparse::launch_flash_rows<DsaHalfPolicy, 32U>(a, 1U, s, max_keys);
            } else {
                DsaPolicy::Args a{latent[l], lists, counts, stride, 0U, 0U, 0U, qs, sinks, os};
                strix::attn_core::sparse::launch_flash_rows<DsaPolicy, 32U>(a, DsaSpec::kGroups, s, max_keys);
            }
        };
        if (idx && gi::dense(preset::indexer, pos)) {  // every closed group selected: the causal prefix
            stage->list_n[l] = keys;  // pinned slot per layer (async copy, no wait)
            check(hipMemcpyAsync(list_n, &stage->list_n[l], 4, hipMemcpyHostToDevice, s), "list length");
            dsa(iota, list_n, context, keys);
        } else if (idx) {
            index_select(l);
            dsa(ilist, list_n, kIndexListStride, index_list_tokens(pos));
        } else {
            const auto ring = latent_rows[l].ring;
            SwaPolicy::Args a{latent[l], nullptr, nullptr, 0U, pos, c.window, ring - 1U, qs, sinks, os};
            strix::attn_core::sparse::launch_flash_rows<SwaPolicy, 32U>(a, par ? 1U : SwaSpec::kGroups, s, std::min(keys, c.window));
        }
        check(hipGetLastError(), "flash launch");
        const auto& vb = t(blk(l, "attn_v_b.weight"));
        const auto vb_head = row_bytes(vb.type, kl) * vd;
        if (vb.type == 8U) strix::q8_rows::launch_heads_f32(vb.p + h0 * vb_head, Hs, kl, vd, olat + l0, kl, o + v0, vd, s);  // Q8_0: one launch
        else  // one grouped launch over this rank's heads
            linear_groups(vb.p + h0 * vb_head, vb.type, vb_head, Hs, nullptr, Hs, olat + l0, kl, kl, vd, o + v0, vd);
        glu::launch_epilogue<glu::Round::bf16>(o + v0, nullptr, Hs * vd, std::size_t{Hs} * vd, o + v0, s);
        glu::launch_sigmoid_gate<glu::DeclaredExp, glu::Round::bf16>(o + v0, gate + v0, std::size_t{Hs} * vd, o + v0, s);
        if (par) tp_gather(o, std::size_t{Hs} * vd * 4U);  // o = [rank 0 heads | rank 1 heads]: the one-node o
        const auto fr = share(c.hidden);
        linear_rows(blk(l, "attn_output.weight"), o, H * vd, c.hidden, fr, f);
        if (par) tp_gather(f, fr.count * 4U);
        norm(f, t(blk(l, "post_attention_norm.weight")).f(), c.hidden, c.rms_eps, f);
        check(hipGetLastError(), "attention launch");
    }

    // Routed experts of a BF16 block (the next-token block; the oracle's order, hy4_host_ref ffn): the device route's ids
    // stay on the device: the shared grouped dense rows read them (no host read, no sync): g, u = W[ids[k]] z for the
    // top_k slices (one launch each, BF16 outputs), act = swiglu(g, u, swiglu_limit), rawdots[k] = W_d[ids[k]] act[k]
    // (one launch; rounded by the caller's epilogue). 3 launches; every output bitwise equal to its per-expert linear().
    // A routed id outside the experts writes NaN (caught by the non-finite checks downstream).
    // One row's experts: zr = its routed_down output [latent], idr = its device top-k ids, raw = its [K][latent] down outputs
    // (the decode step: z / ids / rawdots; the rows form: row r of vr.z / vr.ids / vr.rawdots).
    void experts_dense(const Tensor& wg, const Tensor& wu, const Tensor& wd, const float* zr, const std::uint32_t* idr, float* raw) {
        const std::uint32_t K = c.top_k, I = c.expert_ff, L = c.latent;
        if (!nx.act) throw std::logic_error("hy4 hip forward: BF16 experts without the next-token workspace");
        if (wu.type != wg.type || wd.type != wg.type) throw std::runtime_error("hy4 hip forward: mixed BF16 expert types");
        const std::uint64_t gate_slice = row_bytes(wg.type, L) * I, down_slice = row_bytes(wd.type, I) * L;
        if (wg.bytes != gate_slice * c.experts || wu.bytes != gate_slice * c.experts || wd.bytes != down_slice * c.experts)
            throw std::runtime_error("hy4 hip forward: BF16 expert shapes");
        float* const gk = pout;                        // [K][I]
        float* const uk = pout + std::size_t{K} * I;   // [K][I]
        linear_groups(wg.p, wg.type, gate_slice, c.experts, idr, K, zr, 0U, L, I, gk, I);
        linear_groups(wu.p, wu.type, gate_slice, c.experts, idr, K, zr, 0U, L, I, uk, I);
        glu::launch_epilogue<glu::Round::bf16>(pout, nullptr, 2U * std::size_t{K} * I, 2U * std::size_t{K} * I, pout, s);  // BF16 Linear outputs
        glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(gk, uk, std::size_t{K} * I, c.swiglu_limit, nx.act, s);
        linear_groups(wd.p, wd.type, down_slice, c.experts, idr, K, nx.act, I, I, L, raw, L);
        check(hipGetLastError(), "BF16 experts launch");
    }

    void ffn(std::uint32_t l) {  // xn -> f (with enable_tp: this rank's rows / slots, joined with the peer's)
        const std::uint32_t h = c.hidden;
        const bool par = tp.world == 2U;
        if (dense_of(l)) {
            const auto fr = share(c.dense_ff), hr = share(h);
            linear_rows(blk(l, "ffn_gate.weight"), xn, h, c.dense_ff, fr, g);
            linear_rows(blk(l, "ffn_up.weight"), xn, h, c.dense_ff, fr, up);
            glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(g + fr.first, up + fr.first, fr.count, glu::no_clamp, act + fr.first, s);
            if (par) tp_gather(act, fr.count * 4U);
            linear_rows(blk(l, "ffn_down.weight"), act, c.dense_ff, h, hr, f);
            if (par) tp_gather(f, hr.count * 4U);
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
            // routed slots of this rank (one node: all top_k into rawdots; a share: top_k / 2 at ids + first slot, written
            // into its tp::moe slot) and its shared-expert rows; one join leaves the 16 slot rows contiguous
            htp::MoeLayout m;
            m.slots = c.top_k;
            if (par) m = htp::moe(tp.rank, c.top_k, c.latent, c.shared_ff);
            float* const raw = par ? tp.moe + m.raw_mine : rawdots;
            const std::uint32_t* const sid = ids + m.slot_first;
            const std::uint64_t K = m.slots, L = c.latent;
            if (wg.type == kTypeBf16) {  // the next-token block: BF16 slices by the shared dense rows (the A8 form has no BF16)
                if (par) throw std::logic_error("hy4 hip forward: BF16 experts (next-token block) under tensor parallelism");
                experts_dense(wg, wu, wd, z, ids, rawdots);
            } else {
                const std::uint64_t I = c.expert_ff;
                a8::Request rq;
                rq.enabled = true;
                rq.approved_numerical_identity = a8::numerical_identity_for_gate_type(wg.type);
                rq.input = a8::InputKind::fp32;
                rq.shape = {c.latent, c.expert_ff, c.experts, static_cast<std::uint32_t>(K), 1U};
                rq.gate_type = wg.type;
                rq.up_type = wu.type;
                a8::Bindings b{dev(z, L * 4U), dev(wg.p, wg.bytes), dev(wu.p, wu.bytes), dev(sid, K * 4U), dev(pcodes, L),
                               dev(pdeltas, L / 32U * 4U), dev(psums, L / 32U * 4U), dev(ps, 4U), dev(pout, K * 2U * I * 4U)};
                a8::enqueue_probe(rq, b, s);
                a8::DownRequest dq;
                dq.enabled = true;
                dq.approved_numerical_identity = a8::down_numerical_identity(wd.type);
                dq.shape = {c.latent, c.expert_ff, c.experts, static_cast<std::uint32_t>(K)};
                dq.weight_type = wd.type;
                dq.swiglu_limit = c.swiglu_limit;
                a8::DownBindings db{dev(pout, K * 2U * I * 4U), dev(wd.p, wd.bytes), dev(sid, K * 4U), dev(dcodes, K * I),
                                    dev(ddeltas, K * I / 32U * 4U), dev(dsums, K * I / 32U * 4U), dev(raw, K * L * 4U),
                                    dev(ds, 4U), {0U, 0U}};
                a8::enqueue_down_activation_quantize(dq, db, s);
                a8::enqueue_down_rawdots(dq, db, s);
            }
            glu::launch_epilogue<glu::Round::bf16>(raw, nullptr, L, K * L, raw, s);  // each expert's down Linear output
            if (par) {  // shared gate/up rows of this rank into its slot, one join, the activation halves into act
                const auto sr = share(c.shared_ff);
                linear_rows(blk(l, "ffn_gate_shexp.weight"), xn, h, c.shared_ff, sr, g);
                linear_rows(blk(l, "ffn_up_shexp.weight"), xn, h, c.shared_ff, sr, up);
                glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(g + sr.first, up + sr.first, sr.count, glu::no_clamp,
                                                                      tp.moe + m.act_mine, s);
                tp_gather(tp.moe, m.slot * 4U);
                check(hipMemcpy2DAsync(act, sr.count * 4U, tp.moe + m.act_first, m.act_pitch * 4U, sr.count * 4U, 2U,
                                       hipMemcpyDeviceToDevice, s),
                      "shared activation join");
            }
            strix::moe_core::enqueue_combine<strix::moe::CombineForm::bf16_steps, 16U, 1536U>(
                rw, par ? tp.moe + m.raw_all : rawdots, nullptr, acc, 1U, s);
            linear(blk(l, "ffn_routed_up.weight"), acc, c.latent, h, routed);
            if (!par) {
                linear(blk(l, "ffn_gate_shexp.weight"), xn, h, c.shared_ff, g);
                linear(blk(l, "ffn_up_shexp.weight"), xn, h, c.shared_ff, up);
                glu::launch_swiglu<glu::DeclaredExp, glu::Round::bf16>(g, up, c.shared_ff, glu::no_clamp, act, s);
            }
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
                   std::uint32_t batch_out = 0U, bool round = false, const float* bias = nullptr, std::uint16_t* image = nullptr) {
        namespace q8 = strix::core::f16_dense_tall::q8a8;
        // A8 rows at the shared operand pitch: a pitch that is a multiple of 8 KiB (K_b 16384, V_b 32768, o_proj 16384)
        // aliases the 128 rows a CTA stage reads (strix/a8_row_pitch.hpp); same values, same bits.
        const std::uint32_t pitch = strix::a8_row_pitch(x_stride);
        strix::ds41::enqueue_quantize_a8(x, x_stride, R, pitch, pr.codes, pr.deltas, pr.sums, s);
        q8::View v{};
        v.weight = weights;
        v.codes = pr.codes;
        v.deltas = pr.deltas;
        v.out = y != nullptr ? y : reinterpret_cast<float*>(image);  // the image form never writes out (supported() wants one)
        v.row_bytes = static_cast<std::uint32_t>(row_bytes(8U, width));
        v.in = width;
        v.count = outputs;
        v.rows = R;
        v.x_stride = pitch;
        v.out_stride = y_stride;
        v.batch = batch;
        v.batch_in = batch_in;
        v.batch_out = batch_out;
        v.batch_weight_bytes = std::uint64_t{outputs} * v.row_bytes;
        v.round_bf16 = round ? 1U : 0U;  // the Linear's bf16 epilogue (+ bias) in the GEMM's store: y = R(sum + bias)
        v.bias = round ? bias : nullptr;
        if (image != nullptr) {  // the bf16 cache-row image of the outputs (y unused): store_rows' bits and flag rule
            if (!q8::supported_image(v, image)) throw std::runtime_error("hy4 hip forward: A8 rows image GEMM refuses width " + std::to_string(width));
            q8::enqueue_image<strix::attn_core::cache_rows::ImageEpilogue>(v, image, flag, s);
            check(hipGetLastError(), "q8a8 rows image launch");
            return;
        }
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

    // Engram row requests of window [p0, p0 + R) for one Engram ordinal; tok(q) is the token at position q.
    template <class Tok> std::vector<strix::engram::RowRequest> engram_window_requests(std::uint32_t ordinal, std::uint32_t p0, std::uint32_t R,
                                                                                    Tok&& tok) const {
        const std::uint32_t heads = c.engram_heads();
        std::vector<strix::engram::RowRequest> req;
        req.reserve(std::size_t{R} * heads);
        std::vector<std::uint32_t> window(c.ngram);
        for (std::uint32_t r = 0; r < R; ++r) {
            const auto q = p0 + r;
            for (std::uint32_t k = 0; k < c.ngram; ++k) window[k] = q >= k ? tok(q - k) : c.pad_token;
            const auto rows = strix::hy4::engram_rows(c, ordinal, window);
            if (rows.size() != heads) throw std::runtime_error("hy4 hip forward: Engram rows per position");
            for (const auto x : rows) req.push_back({0U, x});
        }
        return req;
    }

    // Read-ahead of a prompt window's Engram rows (the ids are known before the layers run): one block per Engram layer
    // of this stage, in layer order, on the shared engram::ReadAhead worker. The bytes are RowReader::read's for the
    // same requests (the incumbent synchronous read at the layer), so no bit changes; the read overlaps the GPU's
    // layers instead of running after the MoE status sync of the layer before. A window already queued is kept.
    template <class Tok> void engram_submit(std::uint32_t p0, std::uint32_t R, Tok&& tok) {
        if (!engram_reader || engram_windows.count(p0) != 0U) return;
        if (!engram_ahead) engram_ahead = std::make_unique<strix::engram::ReadAhead>(*engram_reader);
        EngramWindow queued{R, std::vector<std::uint64_t>(c.engram_layers.size(), kNoEngramTicket)};
        bool here = false;
        for (std::uint32_t ord = 0; ord < c.engram_layers.size(); ++ord)
            if (c.engram_layers[ord] >= first && c.engram_layers[ord] < last) {
                queued.tickets[ord] = engram_ahead->submit(engram_window_requests(ord, p0, R, tok));
                here = true;
            }
        if (here) engram_windows.emplace(p0, std::move(queued));
    }

    // Frees a window's read-ahead buffers (after the window sync: the uploads have consumed them).
    void engram_release(std::uint32_t p0) noexcept {
        const auto it = engram_windows.find(p0);
        if (it == engram_windows.end()) return;
        try {
            for (const auto ticket : it->second.tickets)
                if (ticket != kNoEngramTicket) engram_ahead->release(ticket);
        } catch (...) {  // release only waits on the worker; nothing to recover
        }
        engram_windows.erase(it);
    }
    void engram_release_all() noexcept {
        while (!engram_windows.empty()) engram_release(engram_windows.begin()->first);
    }

    void engram_rows_stage(std::uint32_t l, std::uint32_t ordinal, std::uint32_t p0, std::uint32_t R) {
        const std::uint32_t heads = c.engram_heads();
        std::vector<std::uint8_t> off(R, 0U);
        bool any_off = false;
        for (std::uint32_t r = 0; r < R; ++r)
            if (tokens[p0 + r] == c.pad_token) { off[r] = 1U; any_off = true; }  // masked position: no Engram output
        const auto win = engram_windows.find(p0);
        if (win == engram_windows.end() || win->second.rows != R || win->second.tickets.at(ordinal) == kNoEngramTicket)
            throw std::logic_error("hy4 hip forward: Engram rows of the window were not queued");
        const auto host = engram_ahead->wait(win->second.tickets[ordinal]);
        const std::size_t row_count = std::size_t{R} * heads;
        if (host.size() != row_count * c.engram_dim) throw std::logic_error("hy4 hip forward: Engram read-ahead block size");
        // the block stays allocated until engram_release after the window sync
        check(hipMemcpyAsync(pr.erows, host.data(), host.size(), hipMemcpyHostToDevice, s), "engram rows upload");
        strix::engram::RowPrepView pv;
        pv.form = strix::engram::RowPrepForm::head_rms_bf16;
        pv.fmt = strix::engram::RowFormat{96U, strix::engram::RowScale::none};
        pv.rows = pr.erows;
        pv.row_count = row_count;
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
        if (q_image) {
            // pf_store_epilogue: the GEMM stores encode(sum) into pr.qt's first kl columns with the writer's flag rule
            // (dcr::ImageEpilogue) and the roped tail (already bf16 values in qv) goes through the same writer as rope-wide
            // rows at positions subrow(k): every element of pr.qt once, the bits of the FP32 qfull + store_rows path.
            q8a8_rows(kb_pad[l], kKbPadInput, kl, R, pr.qv, H * qk, nullptr, H * row, H, qk, row, false, nullptr, pr.qt);
            check(dcr::store_rows({pr.qv + c.qk_nope, qk, pr.qtail, 0U, R * H, {rope, 0U}, pr.qt, flag}, s), "query tail rows image");
        } else {
            q8a8_rows(kb_pad[l], kKbPadInput, kl, R, pr.qv, H * qk, pr.qfull, H * row, H, qk, row);
            check(hipMemcpy2DAsync(pr.qfull + kl, row * 4U, pr.qv + c.qk_nope, qk * 4U, rope * 4U, std::size_t{R} * H, hipMemcpyDeviceToDevice, s),
                  "q rope rows copy");
            check(dcr::store_rows({pr.qfull, row, nullptr, 0U, R * H, {row, 0U}, pr.qt, flag}, s), "query operand rows image");
        }
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
        // pf_store_epilogue: each head's product rounds its vd columns in the GEMM's store (batched form, no bias); the H
        // products cover the R x H*vd rows once, so this is the separate pass's function on the same FP32 sums: same bits.
        q8a8_rows(vb.p, kl, vd, R, pr.olat, H * kl, pr.o, H * vd, H, kl, vd, pf_store_epilogue);
        const std::size_t n = std::size_t{R} * H * vd;
        if (!pf_store_epilogue) glu::launch_epilogue<glu::Round::bf16>(pr.o, nullptr, n, n, pr.o, s);
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
        engram_submit(p0, R, [this](std::uint32_t q) { return tokens[q]; });  // no-op when prefill() queued it ahead
        struct EngramRelease {  // after the window sync below (or on a throw): the uploads have consumed the rows
            Impl& m;
            std::uint32_t at;
            ~EngramRelease() { m.engram_release(at); }
        } engram_release_guard{*this, p0};
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
        for (std::uint32_t k = 0; k < plan.lead; ++k) {
            out = &step(prompt[i++], state);
            prompt_hidden_step();
        }
        const auto start = std::chrono::steady_clock::now();
        StepTimes sum{};
        struct EngramReleaseAll {  // a failed window leaves no queued rows behind (the next prompt may reuse a position)
            Impl& m;
            ~EngramReleaseAll() { m.engram_release_all(); }
        } engram_release_guard{*this};
        for (std::size_t wi = 0; wi < plan.windows.size(); ++wi) {
            const auto win = plan.windows[wi];
            if (win.p0 != pos) throw std::logic_error("hy4 hip forward: prompt plan position");
            for (std::uint32_t r = 0; r < win.rows; ++r) {
                if (prompt[i] >= c.vocab) throw std::invalid_argument("hy4 hip forward: token outside the vocabulary");
                tokens.push_back(prompt[i++]);
            }
            if (tokens.size() != std::size_t{win.p0} + win.rows) throw std::logic_error("hy4 hip forward: prompt history position");
            // Engram read-ahead: this window's rows first, then the next window's (its ids are in the prompt), so the
            // next window's reads run while this window computes.
            engram_submit(win.p0, win.rows, [this](std::uint32_t q) { return tokens[q]; });
            if (wi + 1U < plan.windows.size()) {
                const auto next_win = plan.windows[wi + 1U];
                const std::size_t have = tokens.size();
                bool ids_ok = next_win.p0 == win.p0 + win.rows && i + next_win.rows <= prompt.size();
                for (std::uint32_t r = 0; ids_ok && r < next_win.rows; ++r) ids_ok = prompt[i + r] < c.vocab;
                if (ids_ok)  // else the window's own checks throw when it is reached
                    engram_submit(next_win.p0, next_win.rows, [this, &prompt, have, i](std::uint32_t q) {
                        return q < have ? tokens[q] : prompt[i + (q - have)];
                    });
            }
            times = {};
            const bool head = wi + 1U == plan.windows.size() && plan.tail == 0U;
            prefill_window(win.p0, win.rows, head);
            prompt_hidden_rows(win.p0, win.rows);
            pos += win.rows;
            sum.embed += times.embed; sum.engram += times.engram; sum.attention += times.attention; sum.ffn += times.ffn; sum.head += times.head;
            if (head) out = &logits;
        }
        sum.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (std::uint32_t k = 0; k < plan.tail; ++k) {
            out = &step(prompt[i++], state);
            prompt_hidden_step();
        }
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
            const auto vocab_rows = share(c.vocab);  // enable_tp: this rank's vocabulary rows, then the logits joined
            linear_rows("output.weight", xn, c.hidden, c.vocab, vocab_rows, logits_d, false);
            if (tp.world == 2U) tp_gather(logits_d, vocab_rows.count * 4U);
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

    // ---- next-token (MTP) block (Forward::nextn_rows; host oracle HostRef::nextn_step) ----------------------------------
    // Its caches at index c.layers of the per-layer arrays (the plan's layout of an indexed / sliding block), its output
    // rows and workspace. Called after allocate() / allocate_prompt() / allocate_verify() (ring sizes from rmax / vr.max).
    void allocate_nextn() {
        const std::uint32_t L = c.layers;
        const std::size_t h = c.hidden;
        if (c.nextn != 1U) throw std::runtime_error("hy4 hip forward: the pack has no next-token block");
        if (first != 0U || last != c.layers) throw std::logic_error("hy4 hip forward: the next-token block runs on a whole-model rank");
        if (L >= 64U) throw std::runtime_error("hy4 hip forward: next-token block id outside the decode staging");
        Config cx = c;  // the block's form appended to the per-layer arrays: layer_caches reads them by block id
        cx.heads.push_back(c.nextn_heads);
        cx.indexed.push_back(c.nextn_indexed);
        cx.dense.push_back(0U);
        const auto k = layer_caches(cx, L, context, rows::cache_rows(rmax, vr.max));
        latent_rows.push_back(k.latent);
        latent.push_back(alloc<std::uint16_t>(std::size_t{k.latent_rows} * k.latent.width));
        tk_rows.push_back(k.token_keys);
        tkeys.push_back(nullptr);
        gkeys.push_back(nullptr);
        if (indexed_of(L)) {
            if (!ik) throw std::runtime_error("hy4 hip forward: indexed next-token block without an indexed text layer (select scratch)");
            tkeys[L] = alloc<std::uint16_t>(std::size_t{k.token_rows} * k.token_keys.width);
            gkeys[L] = alloc<std::uint16_t>(std::size_t{group_cap} * kIdxDim);
        }
        nx.pre = alloc<float>(kNextnRows * h);
        nx.post = alloc<float>(kNextnRows * h);
        nx.x = alloc<float>(h);
        nx.cat = alloc<float>(2U * h);
        nx.act = alloc<float>(std::size_t{c.top_k} * c.expert_ff);
        void* sp = nullptr;
        check(hipHostMalloc(&sp, kNextnRows * sizeof(Staging), hipHostMallocDefault), "next-token row staging");
        nx.stages = static_cast<Staging*>(sp);
        void* top_host = nullptr;
        check(hipHostMalloc(&top_host, kNextnRows * sizeof(std::uint32_t), hipHostMallocDefault), "next-token ids");
        nx.top = static_cast<std::uint32_t*>(top_host);
        std::fill(nx.top, nx.top + kNextnRows, 0U);
        if (vr.max >= 2U) {  // the rows form runs through the verify workspace (allocate_verify before this)
            nx.xr = alloc<float>(std::size_t{vr.max} * h);
            nx.catr = alloc<float>(std::size_t{vr.max} * 2U * h);
            void* wp = nullptr;
            check(hipHostMalloc(&wp, kNextnRows * 4U * sizeof(std::uint32_t), hipHostMallocDefault), "next-token position words");
            nx.words = static_cast<std::uint32_t*>(wp);
        }
        if (rmax) {
            nx.ph = alloc<float>(std::size_t{rmax} * h);
            nx.pu = alloc<float>(std::size_t{rmax} * h);
            nx.mixh = alloc<hcc::MixRow<kHead>>(rmax);
            nx.partial = alloc<float>(hcc::mix_partial_floats<kHead>(rmax, c.hidden));
        }
    }

    // Rows form of nextn_rows: rows off .. off + rows_n - 1 of the call (rows_n <= vr.max) through the verify rows form at
    // block L (attention_vrows / ffn_vrows: every non-expert weight read once for the chunk; BF16 experts and absorb heads
    // per row by the grouped dense rows). Per-row arithmetic = the per-row loop's (rows tile == single-row dense rows,
    // bitwise; the norms / rope / epilogues / flash / route are the same kernels over more rows).
    // kv_only: rows whose outputs the caller never reads stop after the block's caches (cache_vrows: the latent row and the
    // indexer keys depend only on the row's attn_norm input, so they are bitwise the full form's); their pre / post rows
    // are left as they were and their top is 0.
    void nextn_chunk_rows(const std::uint32_t* toks, const float* hid, std::uint32_t off, std::uint32_t rows_n, std::uint32_t p0,
                          bool kv_only) {
        const std::uint32_t L = c.layers;
        const std::size_t h = c.hidden, nh = std::size_t{rows_n} * h;
        vr.p0 = p0 + off;
        std::uint32_t* const words = nx.words + 4U * std::size_t{off};  // the rows' own pinned slots: no sync inside a call
        for (std::uint32_t r = 0; r < rows_n; ++r) {
            const auto pw = rows::position_words(vr.p0 + r, kIdxRatio);
            std::copy(pw.begin(), pw.end(), words + 4U * r);
        }
        check(hipMemcpyAsync(vr.dpos, words, 16U * std::size_t{rows_n}, hipMemcpyHostToDevice, s), "next-token position words upload");
        const auto& te = t("token_embd.weight");
        launch_hip_embedding_rows({te.p, te.type, c.hidden, c.vocab}, HipActivationStorage::fp32,
                                  std::span<const std::uint32_t>(toks + off, rows_n), nx.xr, c.hidden, s);
        glu::launch_epilogue<glu::Round::bf16>(nx.xr, nullptr, nh, nh, nx.xr, s);  // e: BF16 values
        const float* const en = t(blk(L, "nextn.enorm.weight")).f();
        const float* const hn = t(blk(L, "nextn.hnorm.weight")).f();
        for (std::uint32_t r = 0; r < rows_n; ++r) {
            norm(nx.xr + r * h, en, c.hidden, c.rms_eps, nx.catr + 2U * r * h);
            norm(hid + (off + r) * h, hn, c.hidden, c.rms_eps, nx.catr + 2U * r * h + h);
        }
        linear_vrows(blk(L, "nextn.eh_proj.weight"), nx.catr, 2U * c.hidden, c.hidden, rows_n, nx.xr);
        norm_rows(nx.xr, t(blk(L, "attn_norm.weight")).f(), c.hidden, rows_n, vr.xn);
        if (kv_only) {
            cache_vrows(L, rows_n);
            std::fill(nx.top + off, nx.top + off + rows_n, 0U);
            return;
        }
        attention_vrows(L, rows_n);                                                 // vr.xn -> vr.f (post_attention_norm)
        glu::launch_epilogue<glu::Round::bf16>(nx.xr, vr.f, nh, nh, nx.xr, s);    // x = bf(x + attn)
        norm_rows(nx.xr, t(blk(L, "ffn_norm.weight")).f(), c.hidden, rows_n, vr.xn);
        ffn_vrows(L, rows_n);                                                       // vr.xn -> vr.f (post_ffw_norm)
        glu::launch_epilogue<glu::Round::bf16>(nx.xr, vr.f, nh, nh, nx.xr, s);    // x = bf(x + ffn)
        check(hipMemcpyAsync(nx.pre + off * h, nx.xr, nh * 4U, hipMemcpyDeviceToDevice, s), "next-token hidden rows copy");
        norm_rows(nx.xr, t(blk(L, "nextn.shared_head_norm.weight")).f(), c.hidden, rows_n, nx.post + off * h);
        std::fill(nx.top + off, nx.top + off + rows_n, 0U);
    }

    Forward::NextnOut nextn_rows(const std::uint32_t* toks, const float* hid, std::uint32_t R, std::uint32_t p0, bool want_top,
                                 std::uint32_t outputs) {
        if (!nx.on) throw std::logic_error("hy4 hip forward: nextn_rows needs ForwardOptions::nextn");
        if (R == 0U || R > kNextnRows) throw std::invalid_argument("hy4 hip forward: next-token rows outside 1 .. 32");
        outputs = std::min(outputs, R);
        if (want_top && outputs == 0U) throw std::invalid_argument("hy4 hip forward: want_top reads the last row's output");
        if (std::uint64_t{p0} + R > context) throw std::runtime_error("hy4 hip forward: next-token context full");
        for (std::uint32_t r = 0; r < R; ++r)
            if (toks[r] >= c.vocab) throw std::invalid_argument("hy4 hip forward: token outside the vocabulary");
        const std::uint32_t L = c.layers;
        const std::size_t h = c.hidden;
        const auto& te = t("token_embd.weight");
        const std::uint32_t saved = pos;  // attention / indexer read the query position from pos (the oracle's form)
        Staging* const saved_stage = stage;
        const std::uint32_t saved_p0 = vr.p0;  // the rows form binds the verify workspace (committed: vr.rows == 0)
        const bool rows_form = !nx.row_loop && nx.words != nullptr && vr.rows == 0U && R >= 2U;
        nx.logits_ready = false;
        std::uint32_t bad = 0;
        try {
            if (rows_form) {  // KV-only rows [0, R - outputs), then the output rows: each range in chunks of vr.max
                const std::uint32_t kv_end = R - outputs;
                for (std::uint32_t off = 0, rows_n = 0; (rows_n = rows::nextn_chunk(off, kv_end, vr.max)) != 0U; off += rows_n)
                    nextn_chunk_rows(toks, hid, off, rows_n, p0, true);
                for (std::uint32_t off = kv_end, rows_n = 0; (rows_n = rows::nextn_chunk(off, R, vr.max)) != 0U; off += rows_n)
                    nextn_chunk_rows(toks, hid, off, rows_n, p0, false);
            }
            for (std::uint32_t r = 0; !rows_form && r < R; ++r) {
                const std::uint32_t p = p0 + r;
                pos = p;
                hpos = {p, p / kIdxRatio * kIdxRatio, p / kIdxRatio, 0U};
                // the row's own pinned slot: no host sync between rows, so row r's queued uploads (position here, the
                // dense list length in attention) must not be overwritten by row r + 1; the call ends with a sync
                stage = nx.stages + r;
                stage->pos = hpos;
                check(hipMemcpyAsync(dpos, stage->pos.data(), sizeof hpos, hipMemcpyHostToDevice, s), "next-token position upload");
                const std::array<std::uint32_t, 1> one{toks[r]};
                launch_hip_embedding_rows({te.p, te.type, c.hidden, c.vocab}, HipActivationStorage::fp32, one, nx.x, c.hidden, s);
                glu::launch_epilogue<glu::Round::bf16>(nx.x, nullptr, h, h, nx.x, s);  // e: BF16 values
                norm(nx.x, t(blk(L, "nextn.enorm.weight")).f(), c.hidden, c.rms_eps, nx.cat);
                norm(hid + r * h, t(blk(L, "nextn.hnorm.weight")).f(), c.hidden, c.rms_eps, nx.cat + h);
                linear(blk(L, "nextn.eh_proj.weight"), nx.cat, 2U * c.hidden, c.hidden, nx.x);
                norm(nx.x, t(blk(L, "attn_norm.weight")).f(), c.hidden, c.rms_eps, xn);
                attention(L);                                                       // xn -> f (post_attention_norm)
                glu::launch_epilogue<glu::Round::bf16>(nx.x, f, h, h, nx.x, s);    // x = bf(x + attn)
                norm(nx.x, t(blk(L, "ffn_norm.weight")).f(), c.hidden, c.rms_eps, xn);
                ffn(L);                                                             // xn -> f (post_ffw_norm)
                glu::launch_epilogue<glu::Round::bf16>(nx.x, f, h, h, nx.x, s);    // x = bf(x + ffn)
                check(hipMemcpyAsync(nx.pre + r * h, nx.x, h * 4U, hipMemcpyDeviceToDevice, s), "next-token hidden copy");
                norm(nx.x, t(blk(L, "nextn.shared_head_norm.weight")).f(), c.hidden, c.rms_eps, nx.post + r * h);
                nx.top[r] = 0U;
            }
            if (want_top) {  // the shared head and device greedy row on the last row
                linear("output.weight", nx.post + (R - 1U) * h, c.hidden, c.vocab, logits_d, false);
                const strix::ExecutionBufferAddress lg{{}, reinterpret_cast<std::uintptr_t>(logits_d), std::uint64_t{c.vocab} * 4U, 0U};
                const strix::ExecutionBufferAddress pa{{}, reinterpret_cast<std::uintptr_t>(greedy_partials), greedy_bytes, 0U};
                strix::enqueue_hip_greedy_rows(std::span<const strix::ExecutionBufferAddress>(&lg, 1U), c.vocab, pa, greedy_reply, s);
            }
            check(hipMemcpyAsync(&bad, flag, 4, hipMemcpyDeviceToHost, s), "flag read");
            check(hipStreamSynchronize(s), "next-token sync");
        } catch (...) {
            pos = saved;
            stage = saved_stage;
            vr.p0 = saved_p0;
            throw;
        }
        pos = saved;
        stage = saved_stage;
        vr.p0 = saved_p0;
        if (bad) throw std::runtime_error("hy4 hip forward: non-finite value in a cache / operand row (next-token block)");
        if (want_top) {
            if (greedy_reply->status != 0U) throw std::runtime_error("hy4 hip forward: non-finite logit (next-token greedy status)");
            nx.top[R - 1U] = greedy_reply->token;
            nx.logits_ready = true;
        }
        return {nx.top, nx.pre, nx.post};
    }

    const std::vector<float>& nextn_logits() {
        if (!nx.logits_ready) throw std::logic_error("hy4 hip forward: nextn_logits without a want_top next-token call");
        nx.logits.resize(c.vocab);
        check(hipMemcpyAsync(nx.logits.data(), logits_d, nx.logits.size() * 4U, hipMemcpyDeviceToHost, s), "next-token logits read");
        check(hipStreamSynchronize(s), "next-token logits sync");
        return nx.logits;
    }

    // prefill(): the sink receives every prompt row's output_norm(hc_head collapse) (the MTP teacher input)
    void prompt_hidden_step() {  // after a whole-model step(): xn holds the head's input
        if (nx.sink) nx.sink(xn, pos - 1U, 1U);
    }
    void prompt_hidden_rows(std::uint32_t p0, std::uint32_t R) {  // after prefill_window: pr.x0 holds the window's X rows
        if (!nx.sink) return;
        if (!nx.ph) throw std::logic_error("hy4 hip forward: prompt hidden rows without the next-token workspace");
        check(hcc::enqueue_mixes<kHead>(pr.x0, t("output_hc_fn.weight").f(), t("output_hc_base.weight").f(), t("output_hc_scale.weight").f(), R,
                                        c.hidden, 1U, c.hc_eps, c.rms_eps, nx.mixh, nx.partial, s),
              "prompt head mixes");
        check(hcc::enqueue_collapse<kHead>(pr.x0, nx.mixh, R, c.hidden, nx.pu, s), "prompt head collapse");
        norm_rows(nx.pu, t("output_norm.weight").f(), c.hidden, R, nx.ph);
        nx.sink(nx.ph, p0, R);
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
    impl_->nx.on = o.nextn;  // before load(): wanted() admits the next-token block
    if (o.nextn && (impl_->first != 0U || impl_->last != impl_->c.layers || impl_->c.nextn != 1U))
        throw std::invalid_argument("hy4 hip forward: the next-token block needs a whole-model rank of a pack that carries it");
    check(hipStreamCreateWithFlags(&impl_->s, hipStreamNonBlocking), "stream");
    impl_->load(o.parts);
    impl_->allocate();
    if (impl_->rmax) impl_->allocate_prompt();
    if (impl_->vr.max) impl_->allocate_verify();
    if (impl_->nx.on) impl_->allocate_nextn();
}
void Forward::forward_rows(std::span<const std::uint32_t> tokens, bool logits) { impl_->forward_rows(tokens, logits); }
std::span<const std::uint32_t> Forward::rows_greedy() const noexcept { return impl_->vr.greedy; }
const std::vector<float>& Forward::rows_logits() const noexcept { return impl_->vr.logits_host; }
void Forward::commit_rows(std::uint32_t k) { impl_->commit_rows(k); }
void Forward::set_rows_expert_union(bool on) noexcept { impl_->vr.expert_union = on; }
std::uint64_t Forward::tensor_bytes(const std::string& name) const noexcept {
    const auto it = impl_->w.find(name);
    return it == impl_->w.end() ? 0U : it->second.bytes;
}
void Forward::enable_tp(std::uint32_t rank, TpGather gather) {
    auto& im = *impl_;
    const Config& c = im.c;
    if (rank > 1U || gather.fn == nullptr) throw std::invalid_argument("hy4 hip forward: tensor parallelism needs rank 0|1 and a gather");
    if (im.first != 0U || im.last != c.layers) throw std::logic_error("hy4 hip forward: tensor parallelism needs a whole-model rank");
    // the joins assume the flash presets (32 indexed / 64 sliding heads), the combine's 16 x 1536 slot rows and even
    // split dimensions (tp::half refuses odd ones)
    for (std::uint32_t l = 0; l < c.layers; ++l)
        if (c.heads[l] != (c.indexed[l] != 0U ? 32U : 64U)) throw std::invalid_argument("hy4 hip forward: tp head layout");
    if (c.top_k != 16U || c.latent != 1536U || c.hidden != 3072U) throw std::invalid_argument("hy4 hip forward: tp slot layout");
    (void)htp::half(c.vocab, rank);
    (void)htp::half(c.dense_ff, rank);
    (void)htp::moe(rank, c.top_k, c.latent, c.shared_ff);
    im.allocate_tp();
    im.tp.rank = rank;
    im.tp.world = 2U;
    im.tp.gather = gather;
    im.tp.calls = 0;
}
std::uint32_t Forward::tp_gathers_per_step() const noexcept {
    return impl_->tp.world == 2U ? htp::step_count(htp::dims_of(impl_->c)).gathers : 0U;
}
std::string Forward::tp_receipt() const {
    if (impl_->tp.world != 2U) return "hy4 tp: off (one node)";
    const auto n = htp::step_count(htp::dims_of(impl_->c));
    return "hy4 tp: rank " + std::to_string(impl_->tp.rank) + " of 2 (heads, o_proj / FFN / shared rows, routed slots 8 + 8, vocab halves); " +
           std::to_string(n.gathers) + " gathers per step, " + std::to_string(n.bytes / 1024U) + " KiB sent per rank per step; " +
           std::to_string(impl_->tp.calls) + " gathers so far";
}
const std::vector<float>& Forward::prefill(const std::vector<std::uint32_t>& prompt) { return impl_->prefill(prompt); }
Forward::~Forward() {
    if (impl_ && impl_->s) {
        (void)hipStreamSynchronize(impl_->s);
        (void)hipStreamDestroy(impl_->s);
    }
    if (impl_ && impl_->greedy_reply) (void)hipHostFree(impl_->greedy_reply);
    if (impl_ && impl_->stage) (void)hipHostFree(impl_->stage);
    if (impl_ && impl_->engram_stage) (void)hipHostFree(impl_->engram_stage);
    if (impl_ && impl_->nx.top) (void)hipHostFree(impl_->nx.top);
    if (impl_ && impl_->nx.stages) (void)hipHostFree(impl_->nx.stages);
    if (impl_ && impl_->nx.words) (void)hipHostFree(impl_->nx.words);
}
const float* Forward::rows_hidden() const noexcept { return impl_->vr.xn; }
bool Forward::has_nextn() const noexcept { return impl_->nx.on; }
void Forward::set_nextn_row_loop(bool on) noexcept { impl_->nx.row_loop = on; }
Forward::NextnOut Forward::nextn_rows(const std::uint32_t* tokens, const float* hidden, std::uint32_t rows, std::uint32_t pos0, bool want_top,
                                      std::uint32_t outputs) {
    return impl_->nextn_rows(tokens, hidden, rows, pos0, want_top, outputs);
}
const std::vector<float>& Forward::nextn_logits() { return impl_->nextn_logits(); }
void Forward::set_prompt_hidden(std::function<void(const float*, std::uint32_t, std::uint32_t)> sink) {
    if (sink && impl_->rmax && !impl_->nx.ph) throw std::logic_error("hy4 hip forward: prompt hidden rows need ForwardOptions::nextn");
    impl_->nx.sink = std::move(sink);
}
float* Forward::alloc_f32(std::size_t floats) {
    void* p = nullptr;
    check(hipMalloc(&p, std::max<std::size_t>(1U, floats) * sizeof(float)), "hipMalloc f32");
    return static_cast<float*>(p);
}
void Forward::free_f32(float* p) noexcept {
    if (p) (void)hipFree(p);
}
void Forward::copy_f32_async(float* dst, const float* src, std::size_t floats) {
    check(hipMemcpyAsync(dst, src, floats * sizeof(float), hipMemcpyDeviceToDevice, impl_->s), "f32 device copy");
}
void Forward::read_ids(const std::uint32_t* device, std::uint32_t n, std::uint32_t* host) {
    const std::uint32_t* top = impl_->nx.top;  // the next-token ids are host-resident (written after the call's sync)
    if (top && device >= top && device + n <= top + kNextnRows) {
        std::copy(device, device + n, host);
        return;
    }
    check(hipMemcpyAsync(host, device, std::size_t{n} * sizeof(std::uint32_t), hipMemcpyDeviceToHost, impl_->s), "id read");
    check(hipStreamSynchronize(impl_->s), "id read sync");
}
void Forward::read_f32(const float* device, std::size_t floats, float* host) {
    check(hipMemcpyAsync(host, device, floats * sizeof(float), hipMemcpyDeviceToHost, impl_->s), "f32 read");
    check(hipStreamSynchronize(impl_->s), "f32 read sync");
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
