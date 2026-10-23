#include "strix/deepseek41_batch_forward.hpp"

#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_plan.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/staged_ready.hpp"
#include "strix/ds41_engram.hpp"
#include "strix/ds41_expert_tp.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/ds41_weight_overlay.hpp"
#include "strix/ds41_packed.hpp"
#include "strix/ds41_shexp_tp.hpp"
#include "strix/ds41_trellis.hpp"
#include "strix/ds41_trellis_group.hpp"
#include "strix/hip_ds41_attention.hpp"
#include "strix/hip_ds41_expert_tp.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"

#include "hip_ds41_prepared_load.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <cstring>
#include <exception>
#include <future>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// The step mirrors DeviceForward::step_window op by op (same kernels, same views, same declared orders); what
// differs is where each row's state lives: per request (slot) instead of one sequence. See the header.
namespace strix::deepseek41 {
namespace {

constexpr std::size_t control_bytes = 64U << 10U;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("deepseek41 batch forward: ") + what + ": " + hipGetErrorString(e));
}

template <typename T>
struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    Dev() = default;
    explicit Dev(std::size_t count) : n(count) {
        check(hipMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc");
    }
    ~Dev() {
        if (p != nullptr) (void)hipFree(p);
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(std::exchange(o.p, nullptr)), n(o.n) {}
    Dev& operator=(Dev&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
    [[nodiscard]] std::size_t bytes() const noexcept { return n * sizeof(T); }
    template <typename C>
    void upload(const C& host, hipStream_t s) {
        check(hipMemcpyAsync(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice, s), "upload");
    }
};

// Pinned host memory (asynchronous uploads of per-step inputs, asynchronous copies of per-layer counters).
template <typename T>
struct Pinned {
    T* p = nullptr;
    std::size_t n = 0;
    Pinned() = default;
    explicit Pinned(std::size_t count) : n(count) {
        check(hipHostMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T), hipHostMallocDefault), "hipHostMalloc");
    }
    ~Pinned() {
        if (p != nullptr) (void)hipHostFree(p);
    }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    Pinned(Pinned&& o) noexcept : p(std::exchange(o.p, nullptr)), n(o.n) {}
    Pinned& operator=(Pinned&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
};

class Mapped {
public:
    explicit Mapped(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("deepseek41 batch forward: cannot open " + path.string());
        struct stat st {};
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("deepseek41 batch forward: cannot stat the model");
        size_ = static_cast<std::uint64_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) throw std::runtime_error("deepseek41 batch forward: mmap failed");
        base_ = static_cast<const std::uint8_t*>(p);
    }
    ~Mapped() {
        if (base_ != nullptr) ::munmap(const_cast<std::uint8_t*>(base_), size_);
        if (fd_ >= 0) ::close(fd_);
    }
    Mapped(const Mapped&) = delete;
    Mapped& operator=(const Mapped&) = delete;
    [[nodiscard]] std::span<const std::uint8_t> bytes(std::uint64_t offset, std::uint64_t size) const {
        if (offset > size_ || size > size_ - offset) throw std::out_of_range("deepseek41 batch forward: tensor outside the file");
        return {base_ + offset, static_cast<std::size_t>(size)};
    }

private:
    int fd_ = -1;
    std::uint64_t size_ = 0;
    const std::uint8_t* base_ = nullptr;
};

// Byte ranges of the model file into device memory through a pinned staging buffer, with O_DIRECT (DeviceForward's
// loader: the weights never enter the page cache).
class DirectLoader {
public:
    static constexpr std::uint64_t align = 4096, chunk = 256ULL << 20U;
    explicit DirectLoader(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd_ < 0) fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("deepseek41 batch forward: cannot open " + path.string());
        check(hipHostMalloc(reinterpret_cast<void**>(&buffer_), chunk + 2U * align, hipHostMallocDefault), "staging buffer");
    }
    ~DirectLoader() {
        if (buffer_ != nullptr) (void)hipHostFree(buffer_);
        if (fd_ >= 0) ::close(fd_);
    }
    DirectLoader(const DirectLoader&) = delete;
    DirectLoader& operator=(const DirectLoader&) = delete;
    void load(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* device) {
        std::uint64_t done = 0;
        while (done < bytes) {
            const std::uint64_t at = offset + done, base = at & ~(align - 1U), lead = at - base;
            const std::uint64_t take = std::min(bytes - done, chunk - lead);
            const std::uint64_t want = (lead + take + align - 1U) & ~(align - 1U);
            std::uint64_t got = 0;
            while (got < lead + take) {
                const auto n = ::pread(fd_, buffer_ + got, want - got, static_cast<off_t>(base + got));
                if (n <= 0) throw std::runtime_error("deepseek41 batch forward: short read of the weights");
                got += static_cast<std::uint64_t>(n);
            }
            check(hipMemcpy(device + done, buffer_ + lead, take, hipMemcpyHostToDevice), "weight upload");
            done += take;
        }
    }

private:
    int fd_ = -1;
    std::uint8_t* buffer_ = nullptr;
};

// out = a + b elementwise (one rounding): the canonical y = y0 + y1 of the two K-halves of wo_b.
__global__ void add_halves(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
// Staged readiness (STRIX_DIST_STAGED_READY_BYTES): the producer's release after the o_b product (one block), and the
// add of the two halves behind the consumer acquire, publishing the consumer token at its end (every thread of every
// block, no early return).
__global__ void publish_ready_kernel(dist::StagedReady* flag, std::uint64_t token) { dist::publish_staged_block(flag, token, 1U); }
__global__ void add_halves_ready(const float* a, const float* b, float* __restrict__ out, std::uint32_t n, dist::StagedReady* flag,
                                 std::uint64_t token) {
    dist::staged_acquire(flag, token);
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
    dist::publish_staged_block(flag, token, gridDim.x);
}
// The same add for one column chunk of STRIX_DS41_OB_SPLIT: a and b hold [rows][cols], out the rows of `width` values
// from column col0.
__global__ void add_halves_cols(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::uint32_t rows,
                                std::uint32_t cols, std::uint32_t col0, std::uint32_t width) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[std::size_t{i / cols} * width + col0 + i % cols] = a[i] + b[i];
}
// STRIX_DS41_OB_SPLIT=K: the tensor-parallel o_b half in K column chunks, each one's exchange overlapping the next one's
// product (every output's product and add unchanged, so the same bits). 1 / unset: one product, one exchange.
std::uint32_t ob_split(std::uint32_t hidden) {
    static const std::uint32_t k = [] {
        const char* v = std::getenv("STRIX_DS41_OB_SPLIT");
        return v && *v ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10)) : 1U;
    }();
    if (k <= 1U) return 1U;
    if (k > 8U || hidden % k != 0U || (hidden / k) % 32U != 0U) throw std::invalid_argument("STRIX_DS41_OB_SPLIT: 2, 4, 5 or 8");
    return k;
}
// STRIX_DS41_LATE_HC_LAYERS=all (default since P17.5) | none | <comma-separated layers>: see late_mix in the forward.
bool late_hc_layer(std::uint32_t layer) {
    static const std::vector<bool> on = [] {
        std::vector<bool> r(1024, false);
        const char* v = std::getenv("STRIX_DS41_LATE_HC_LAYERS");
        if (!v || !*v) return std::vector<bool>(1024, true);
        const std::string s(v);
        if (s == "all") return std::vector<bool>(1024, true);
        if (s == "none" || s == "off") return r;
        std::size_t i = 0;
        while (i < s.size()) {
            const std::size_t j = s.find(',', i);
            const auto l = std::stoul(s.substr(i, j == std::string::npos ? std::string::npos : j - i));
            if (l >= r.size()) throw std::invalid_argument("STRIX_DS41_LATE_HC_LAYERS: layer out of range");
            r[l] = true;
            if (j == std::string::npos) break;
            i = j + 1U;
        }
        return r;
    }();
    return layer < on.size() && on[layer];
}

// STRIX_DS41_COMPRESS_RING: the ratio-2 carries of a rolled-back request, every layer in one launch (block b: layer b),
// exactly the copies and memsets of commit: open, the kept last row's (kv, score) and valid 1; else zeros.
struct CarryRestore {
    std::uint32_t* state;  // 2 x kHeadDim words: kv, score
    std::uint32_t* valid;
    const std::uint32_t* kv;
    const std::uint32_t* score;
};
constexpr std::uint32_t max_carry_restores = 64;  // per launch: 2 KiB of kernel arguments
struct CarryRestores {
    CarryRestore layer[max_carry_restores];
};
__global__ void restore_carries(CarryRestores a, std::uint32_t open) {
    const CarryRestore& c = a.layer[blockIdx.x];
    constexpr std::uint32_t D = ds41::attn::kHeadDim;
    for (std::uint32_t i = threadIdx.x; i < 2U * D; i += blockDim.x) c.state[i] = open != 0U ? (i < D ? c.kv[i] : c.score[i - D]) : 0U;
    if (threadIdx.x == 0U) *c.valid = open;
}

// Reads `n` 16-byte words and keeps nothing (the cache lines stay in the MALL for the kernels that follow).
__global__ void touch_kernel(const uint4* __restrict__ p, std::uint64_t n, std::uint32_t* __restrict__ never) {
    std::uint32_t acc = 0;
    for (std::uint64_t i = blockIdx.x * std::uint64_t{blockDim.x} + threadIdx.x; i < n; i += std::uint64_t{gridDim.x} * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9U && n == 0U) never[0] = acc;  // never true: keeps the loads
}

std::uint64_t row_bytes(ds41::WeightFormat f, std::uint32_t width) {
    return std::uint64_t{width} / ds41::block_elements(f) * ds41::block_bytes(f);
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// Bytes this rank keeps resident for `options` (the loader's slicing: its expert range, its vocabulary slice).
std::uint64_t resident_weight_bytes(const TargetModel& model, const Plan& plan, const BatchForwardOptions& options) {
    std::map<std::string, const ManifestTensor*> tensors;
    for (const auto& t : model.manifest.tensors) tensors.emplace(t.name, &t);
    std::set<std::string> names;
    for (const auto& s : plan.steps)
        for (const auto& w : s.weights)
            if (w.find("engram_embd") == std::string::npos) names.insert(w);
    const auto& c = model.config;
    const auto head = options.world == 2U ? ds41::vocab_slice(c.vocab_size, 2U, options.rank) : ds41::VocabSlice{0U, c.vocab_size};
    std::uint64_t total = 0;
    for (const auto& n : names) {
        const std::uint64_t bytes = tensors.at(n)->byte_size;
        if (n.ends_with("_exps.weight") && ds41::expert_tp::decode_enabled(options.world, c.moe_intermediate_size, false))  // every expert's part
            total += ds41::expert_tp::part_bytes(bytes, c.n_routed_experts, n.ends_with("ffn_down_exps.weight"), c.hidden_size);
        else if (n.ends_with("_exps.weight")) total += bytes / c.n_routed_experts * (c.n_routed_experts / options.world);
        else if (n == "output.weight") total += bytes / c.vocab_size * head.count;
        else total += bytes;
    }
    return total;
}

// Device state of one request: per layer the window ring and, for the layers that own compressed KV, its caches and
// the two ratio-2 carries.
std::uint64_t slot_state_bytes(const BatchForwardOptions& options) {
    namespace attn = ds41::attn;
    const auto roles = attn::released_plan();
    std::uint64_t n = 0;
    for (std::uint32_t l = 0; l < options.layers; ++l) {
        n += std::uint64_t{AttentionHost::ring_capacity} * attn::kWindowEntryBytes;
        if (roles[l].owns_kv) {
            const std::uint64_t entries = options.max_positions / roles[l].ratio + 1U;
            if (options.pool_positions == 0U) n += entries * (attn::kKvEntryBytes + attn::kIdxEntryBytes);
            n += 2U * (2U * attn::kHeadDim * 4U + 4U);
        }
    }
    return n;
}

// KV pool: position units of the arenas (the pool's positions plus a slack of 16 per slot: a reservation is rounded up
// to a multiple of 8 plus 8, so every ratio's entry range starts aligned and holds positions / ratio + 1 entries).
std::uint64_t pool_units(const BatchForwardOptions& options) {
    return options.pool_positions == 0U ? 0U : std::uint64_t{options.pool_positions} + 16U * options.slots;
}
std::uint64_t pool_bytes_for(const BatchForwardOptions& options) {
    namespace attn = ds41::attn;
    const auto roles = attn::released_plan();
    std::uint64_t n = 0;
    for (std::uint32_t l = 0; l < options.layers; ++l)
        if (roles[l].owns_kv) n += pool_units(options) / roles[l].ratio * (attn::kKvEntryBytes + attn::kIdxEntryBytes);
    return n;
}

// Activations and scratch of a step of up to max_rows rows (the constructor's allocations, term by term).
std::uint64_t workspace_bytes_for(const TargetModel& model, const BatchForwardOptions& options) {
    namespace attn = ds41::attn;
    const auto& c = model.config;
    const auto roles = attn::released_plan();
    const std::uint64_t W = BatchForward::configured_row_limit(), D = attn::kHeadDim;
    const std::uint64_t H = c.hidden_size, hc = c.hc_mult, I = c.moe_intermediate_size, K = c.num_experts_per_tok, V = c.vocab_size;
    std::uint64_t max_count = 1;
    std::uint32_t source_ratio = 0;
    std::uint64_t n = 0;
    for (std::uint32_t l = 0; l < options.layers; ++l) {
        if (roles[l].owns_kv) max_count = std::max<std::uint64_t>(max_count, options.max_positions / roles[l].ratio + 1U);
        if (roles[l].candidate_source) source_ratio = roles[l].ratio;
        if (roles[l].owns_index) n += W * attn::kTopK * 4U + W * 4U;
        if (roles[l].owns_kv) n += 2U * W * D * 4U;  // compressor inputs
    }
    const std::uint64_t cand_bytes = source_ratio != 0U ? (options.max_positions / source_ratio + attn::kCandBlockSize) / attn::kCandBlockSize
                                                        : (max_count + attn::kCandBlockSize) / attn::kCandBlockSize;
    const std::uint64_t cols = model.engram.rows_per_token();
    const std::uint64_t site_width = std::max<std::uint64_t>({H, c.q_lora_rank, std::uint64_t{c.num_attention_heads} * c.head_dim,
                                                              std::uint64_t{c.o_groups} * c.o_lora_rank});
    std::uint32_t partial_blocks = 0;
    for (std::uint32_t r = 0; r < 2U; ++r)
        partial_blocks = std::max(partial_blocks, ds41::argmax_partial_blocks(ds41::vocab_slice(c.vocab_size, 2U, r).count));
    n += W * cand_bytes + 2U * W * max_count * 4U;                                         // candidates, scores, keys
    n += W * attn::kIdxHeads * attn::kIdxDim * 4U + W * attn::kIdxHeads * 4U;              // iq, iw
    n += W * attn::hip::kIdxPrepBytes + W * attn::hip::kAttnGroupCount * 4U + 4U;          // qprep, tickets (+ router)
    n += W * attn::kHeads * attn::kMaxChunks * (2U + D) * 4U;                             // attention partials
    n += W * 8U + 3U * W * sizeof(attn::RopeRow);                                          // positions, RoPE rows
    n += 2U * W * hc * H * 4U;                                                             // streams
    if (options.capture_main_hidden) n += 3U * W * hc * H * 4U;                            // captured layer inputs
    n += W * H * 4U + W * c.q_lora_rank * 4U + 2U * W * c.num_attention_heads * c.head_dim * 4U + W * c.head_dim * 4U;  // x qr q o kv
    n += W * c.o_groups * c.o_lora_rank * 4U + 2U * W * H * 4U + W * K * H * 4U + 3U * W * H * 4U;  // o_low y shared raw low high h
    n += W * V * 4U;                                                                       // logits
    n += std::uint64_t{model.engram.layer_ids.size()} * W * cols * 264U + W * cols * 256U * 4U + W * H * (hc + 1U) * 4U;  // Engram
    n += W * c.n_routed_experts * 4U + 2U * W * K * 4U + W + 2U * W * 4U;                  // scores route_w ids image token pos
    n += 2U * W * sizeof(ds41::HcMix) + ds41::hc_mix_partial_floats(static_cast<std::uint32_t>(W), static_cast<std::uint32_t>(H)) * 4U;
    n += ds41::hc_site_counter_count(static_cast<std::uint32_t>(W)) * 4U;
    n += W * site_width + 2U * W * site_width / 32U * 4U;                                  // site A8
    n += W * c.num_attention_heads * c.head_dim + 2U * W * c.num_attention_heads * c.head_dim / 32U * 4U;  // attention output A8
    n += W * H + 2U * W * H / 32U * 4U;                                                    // FFN input A8
    n += W * K * I + 2U * W * K * I / 32U * 4U + W * I + 2U * W * I / 32U * 4U;            // expert activations A8
    n += (W * partial_blocks + 5U * W) * sizeof(ds41::ArgmaxCandidate);                     // argmax
    n += 2U * (W * H + W * H / 2U) * 4U;                                                   // half sums (+ split shared down rows)
    n += 2U * (W * I / 2U + W * I / 64U * 4U);                                             // split shared activation halves
    n += 6U * (W * K + 1U) * 4U + 2U * 4U;                                                 // built routes
    if (!ds41::trellis_env_spec().empty()) {
        n += 3U * W * I * 4U;  // trellis gate up act
        if (const char* t = std::getenv("STRIX_DS41_SHEXP_TP"); t != nullptr && *t == '1') n += W * H * 4U;  // shexp-tp rank 1 partial
    }
    return n;
}

// Device bytes of the trellis sites this rank prepares for `options` (STRIX_DS41_TRELLIS / _MIX, _SITES, _SHEXP_TP,
// _HEADOUT, _OA_GROUP): the prepared weights (trellis + suh + svh of every distinct pack tensor or slice), the context's
// x / y staging, the HEADOUT row buffers and the o_a group runner's staging; the library workspaces are queried from the
// library at load (BatchForward::trellis_library_bytes()). The same selection as the constructor (checked there).
struct TrellisBytes {
    std::uint64_t weights = 0, staging = 0, head_rows = 0, oa_staging = 0;
    [[nodiscard]] std::uint64_t total() const noexcept { return weights + staging + head_rows + oa_staging; }
};
TrellisBytes trellis_bytes_for(const TargetModel& model, const Plan& plan, const BatchForwardOptions& options) {
    TrellisBytes out;
    const auto spec = ds41::trellis_env_spec();
    if (spec.empty()) return out;
    ds41::TrellisPack pack(spec);
    const char* sites_env = std::getenv("STRIX_DS41_TRELLIS_SITES");
    const auto classes = ds41::parse_trellis_site_classes(sites_env != nullptr ? sites_env : "");
    const auto& c = model.config;
    const bool distributed = options.world == 2U;
    const bool tp = distributed && options.tp_attention;
    std::set<std::string> seen;
    std::uint32_t max_columns = 128, max_rows = 128;
    const auto add = [&](const std::string& pack_name, const ds41::TrellisSlice& slice = {}) {
        const auto info = pack.info(pack_name);
        if (!info) throw std::runtime_error("deepseek41 batch forward: the trellis pack has no " + pack_name);
        std::uint32_t rows = info->rows, columns = info->columns;
        if (slice.kind == ds41::TrellisSlice::Kind::rows) rows = slice.count;
        if (slice.kind == ds41::TrellisSlice::Kind::columns) columns = slice.count;
        max_columns = std::max(max_columns, columns);
        max_rows = std::max(max_rows, rows);
        if (seen.insert(pack_name + "@" + slice.describe()).second) out.weights += ds41::trellis_site_bytes(*info, slice);
        return *info;
    };
    const auto pack_of = [](const std::string& model_name, int group = -1) { return *ds41::trellis_pack_name(model_name, group); };
    const char* shexp_tp_env = std::getenv("STRIX_DS41_SHEXP_TP");
    const bool shexp_tp = shexp_tp_env != nullptr && *shexp_tp_env == '1';
    std::uint32_t head_slices = 0, head_count = 0;
    for (const auto& st : plan.steps) {
        if (st.op == Op::shared_expert && ds41::trellis_class_enabled(classes, st.weights[0])) {
            if (shexp_tp) {
                for (std::uint32_t r = 0; r < ds41::shexp_tp_ranks; ++r) {
                    if (distributed && r != options.rank) continue;
                    const auto gu = ds41::shexp_tp_gate_up_slice(c.moe_intermediate_size, r), dn = ds41::shexp_tp_down_slice(c.moe_intermediate_size, r);
                    add(pack_of(st.weights[0]), gu);
                    add(pack_of(st.weights[1]), gu);
                    add(pack_of(st.weights[2]), dn);
                }
            } else {
                for (std::size_t k = 0; k < 3U; ++k) add(pack_of(st.weights[k]));
            }
        }
        if (st.op == Op::head && ds41::trellis_class_enabled(classes, "output.weight"))
            for (std::uint32_t r = 0; r < 2U; ++r) {
                if (distributed && r != options.rank) continue;
                const auto sl = ds41::vocab_slice(c.vocab_size, 2U, r);
                add("head", {ds41::TrellisSlice::Kind::rows, sl.first, sl.count});
                ++head_slices;
                head_count = sl.count;
            }
    }
    const std::uint32_t first_rank = tp ? options.rank : 0U, ranks = tp ? 1U : 2U;
    const std::uint32_t q_rows = c.num_attention_heads / 2U * c.head_dim, o_half = c.o_groups * c.o_lora_rank / 2U;
    std::uint32_t oa_groups = 0, oa_columns = 0, oa_rows = 0;
    bool oa_first_layer = true;
    for (const auto& st : plan.steps) {
        if (st.layer < 0 || (st.op != Op::q_b && st.op != Op::o_a && st.op != Op::o_b)) continue;
        if (!ds41::trellis_class_enabled(classes, st.weights[0])) continue;
        std::uint32_t layer_groups = 0;
        for (std::uint32_t r = first_rank; r < first_rank + ranks; ++r) {
            if (st.op == Op::q_b) add(pack_of(st.weights[0]), {ds41::TrellisSlice::Kind::rows, r * q_rows, q_rows});
            else if (st.op == Op::o_b) add(pack_of(st.weights[0]), {ds41::TrellisSlice::Kind::columns, r * o_half, o_half});
            else
                for (std::uint32_t g = r * (c.o_groups / 2U); g < (r + 1U) * (c.o_groups / 2U); ++g) {
                    const auto info = add(pack_of(st.weights[0], static_cast<int>(g)));
                    ++layer_groups;
                    oa_columns = info.columns;
                    oa_rows = info.rows;
                }
        }
        if (st.op == Op::o_a && oa_first_layer) {
            oa_groups = layer_groups;
            oa_first_layer = false;
        }
    }
    if (seen.empty()) return out;
    out.staging = ds41::TrellisContext::staging_bytes_for(max_columns, max_rows);
    if (head_slices != 0U && ds41::trellis_headout_enabled())
        out.head_rows = std::uint64_t{head_slices} * BatchForward::configured_row_limit() * head_count * sizeof(float);
    if (oa_groups != 0U && ds41::trellis_oa_group_mode() != 0)
        out.oa_staging = ds41::TrellisGroupRunner::staging_bytes_for(oa_groups, oa_columns, oa_rows);
    return out;
}

}  // namespace

BatchForward::Footprint BatchForward::footprint(const std::filesystem::path& gguf, const BatchForwardOptions& options) {
    const auto model = validate_target(read_file(gguf));
    const auto plan = build_decode_plan(model, 0, options.layers);
    return {resident_weight_bytes(model, plan, options), slot_state_bytes(options), workspace_bytes_for(model, options),
            pool_bytes_for(options), trellis_bytes_for(model, plan, options).total()};
}

struct BatchForward::Impl {
    BatchForwardOptions options;
    const std::uint32_t row_capacity = BatchForward::configured_row_limit();
    TargetModel model;
    Plan plan;
    std::unique_ptr<Mapped> file;
    std::map<std::string, const ManifestTensor*> tensors;
    std::map<std::string, Dev<std::uint8_t>> weights;
    // trellis sites (STRIX_DS41_TRELLIS=pack dir, STRIX_DS41_TRELLIS_SITES=classes; ds41_trellis.hpp, identity
    // ds41-trellis-k5-v1): as DeviceForward, the same sites, slices and calls (a step of R <= max_rows rows is one window)
    std::unique_ptr<ds41::TrellisPack> trellis_pack;
    std::unique_ptr<ds41::TrellisContext> trellis;
    std::vector<std::string> trellis_classes;
    std::string trellis_announced;          // identity of the prepared sites (TrellisContext::announced_identity)
    std::uint64_t trellis_device = 0;       // weights + staging + HEADOUT rows + o_a group staging (== footprint().trellis)
    std::uint64_t trellis_library = 0;      // the library workspaces (queried at load; not in footprint())
    std::map<std::string, const ds41::TrellisSite*> trellis_sites;
    Dev<float> tr_gate, tr_up, tr_act;  // trellis shared expert [R][I] (F32)
    // STRIX_DS41_SHEXP_TP=1 (ds41-trellis-k5-v1:shexp-tp, as DeviceForward): per layer the rank slices this process runs;
    // tr_p1 = rank 1's partial when one process runs both
    bool shexp_tp = false;
    std::vector<std::vector<ds41::ShexpTpSlices>> trellis_shexp_tp;
    Dev<float> tr_p1;
    // trellis attention projections (as DeviceForward): per layer, this process's linears in rank-slice order
    struct TrellisAttn { std::vector<const ds41::TrellisSite*> q_b, o_a, o_b; };
    std::vector<TrellisAttn> trellis_attn;
    std::unique_ptr<ds41::TrellisGroupRunner> trellis_oa;  // STRIX_DS41_TRELLIS_OA_GROUP=1|2: o_a groups in one staged call
    // trellis LM head (ds41-trellis-k5-v1:head): as DeviceForward, the vocabulary slices this process computes
    std::vector<std::pair<ds41::VocabSlice, const ds41::TrellisSite*>> trellis_head;
    // STRIX_DS41_TRELLIS_HEADOUT (as DeviceForward): a step of R > 1 rows writes the head to head_rows ([slice k][max_rows]
    // [count], in place, R5..7 as 8 rows; not in the workspace bytes, like the trellis weights), R1 writes logits in place
    bool headout = false, head_rows_valid = false;
    [[nodiscard]] std::pair<const float*, std::size_t> logits_rows(std::uint32_t first, std::uint32_t vocab) const {
        if (head_rows_valid)
            for (std::size_t k = 0; k < trellis_head.size(); ++k)
                if (const auto& sl = trellis_head[k].first; sl.first == first)
                    return {head_rows.p + k * row_capacity * sl.count, sl.count};
        return {logits.p + first, vocab};
    }
    [[nodiscard]] const ds41::TrellisSite* trellis_site(const std::string& name) const {
        const auto it = trellis_sites.find(name);
        return it == trellis_sites.end() ? nullptr : it->second;
    }
    hipStream_t stream = nullptr;
    std::array<ds41::attn::LayerAttention, 40> roles{};

    // ---- per request (slot) ----
    struct LayerState {
        Dev<std::uint8_t> ring, kv_cache, idx_cache;  // the caches are owned only without a KV pool
        std::uint8_t *kv = nullptr, *idx = nullptr;   // the slot's caches (its own, or its range of the pool's arenas)
        std::size_t kv_bytes = 0, idx_bytes = 0;
        Dev<float> carry[2];
        Dev<std::uint32_t> carry_valid[2];
        int carry_cur = 0;
    };
    struct Slot {
        std::vector<LayerState> layers;
        std::unique_ptr<ds41::EngramHasher> hasher;
        std::uint32_t position = 0;
        std::uint32_t capacity = 0;                    // positions the slot may hold
        std::uint64_t pool_begin = 0, pool_units = 0;  // KV pool: its range of arena position units
    };
    std::vector<Slot> slots;
    // KV pool: per KV-source layer one arena of pool_units(options) / ratio entries for the KV and one for the index
    // keys; the free ranges of position units (begin -> length), merged on release
    std::vector<Dev<std::uint8_t>> kv_arena, idx_arena;
    std::map<std::uint64_t, std::uint64_t> pool_free;

    // ---- per step (rows) ----
    struct LayerScratch {
        Dev<std::uint32_t> idx, idx_count;  // index sources: R x top-k selection read by later layers of the step
    };
    std::vector<LayerScratch> scratch;
    std::vector<Dev<float>> ckv, cscore;  // per KV-owning layer: compressor inputs of the last step's rows (commit)
    Dev<std::uint8_t> candidates;
    std::size_t cand_bytes = 0, max_count = 1;  // per-row strides of the candidate pool and the index scores
    Dev<float> index_scores, iq, iw, attn_partial;
    Dev<std::uint32_t> select_keys;
    Dev<std::uint8_t> index_qprep;
    Dev<std::uint32_t> attn_tickets;
    Dev<std::uint32_t> router_tickets;  // 1, zero between calls (STRIX_DS41_ROUTER_FAST: the last scores block selects)
    Dev<std::uint64_t> pos64;
    Dev<ds41::attn::RopeRow> rope_rows;  // [window @ pos][compressed @ pos][compressed @ even group start], W rows each
    Dev<float> streams[2];
    Dev<float> capture[3];  // capture_main_hidden: input streams of the last three layers, [R][hc][hidden]
    int cur = 0;
    bool identity_pre = true;
    Dev<float> x, qr, q, kv, o, o_low, y, shared, raw, low, high, h, logits, emb, ekv, scores, route_w;
    Dev<float> head_rows;  // STRIX_DS41_TRELLIS_HEADOUT (see headout)
    Dev<std::uint8_t> erows, image_row;
    Pinned<std::uint8_t> erows_pinned;
    Pinned<std::uint32_t> token_pinned, pos_pinned;
    Pinned<std::uint64_t> pos64_pinned;
    Pinned<ds41::attn::RopeRow> rope_pinned;
    Dev<std::uint32_t> token, pos, ids;
    Dev<ds41::HcMix> mix_attn, mix_ffn;
    Dev<float> mix_partial;
    Dev<std::uint32_t> hc_counters;  // enqueue_hc_site's per-row counters, back at zero after every launch
    Dev<std::int8_t> site_codes;
    Dev<std::int8_t> o_codes;  // the attention output's A8 from its combine (STRIX_DS41_ATTN_A8=1)
    Dev<float> o_deltas;
    Dev<std::int32_t> o_sums;
    Dev<float> site_deltas;
    Dev<std::int32_t> site_sums;
    Dev<std::int8_t> codes, act_codes, sh_act_codes;
    Dev<float> deltas, act_deltas, sh_act_deltas;
    Dev<std::int32_t> sums, act_sums, sh_act_sums;
    Dev<ds41::ArgmaxCandidate> partial, cand0, cand1, best, cand_own, cand_other;
    Pinned<ds41::ArgmaxCandidate> best_pinned;
    struct BuiltRoutes {
        Dev<std::uint32_t> experts, begins, rows, slots, order, low_counts, counts;
    } built;
    Pinned<std::uint32_t> counts_pinned;  // [layer][2]: the rank's union of experts and the most rows of one expert
    struct SharedRoutes {
        Dev<std::uint32_t> experts, begins, rows, slots;
        std::uint32_t count = 0, max_rows = 0;
    };
    std::map<std::uint32_t, SharedRoutes> shared_routes;  // per row count: the shared expert as expert 0
    std::uint32_t expert_first = 0, expert_end = 0;
    // STRIX_DS41_EXPERT_TP=1 (two ranks): every routed expert split by its intermediate rows over the ranks
    // (ds41-expert-tp2-i1152-v1, ds41_expert_tp.hpp); expert_first/end then cover all experts.
    bool expert_tp = false;
    ds41::VocabSlice head_slice{};
    std::unique_ptr<ds41::EngramRowReader> engram_reader;
    std::vector<std::uint8_t> engram_bytes;
    // STRIX_DS41_ENGRAM_ASYNC=1: the NVMe reads run on a host thread, one batch per Engram layer straight into its
    // pinned staging, while the plan is enqueued; each layer's upload waits only for its own rows (same bytes).
    // STRIX_DS41_COMPRESS_RING=1: commit restores every ratio-2 carry in one kernel and does not synchronise (the carries
    // are only read by the stream's later work; host reads of them download, which synchronises).
    bool engram_async = false, compress_ring = false;
    Dev<float> half_own, half_other;  // [R x H half sum][R x H/2 split shared down rows]
    Dev<std::uint8_t> shx_own, shx_other;  // split shared expert: A8 activation halves [codes R x I/2][deltas]
    // world 2, the shared expert (same kernels and bits in every form): 0 before the routed experts (default), 1 between
    // the start and the finish of the half-sum exchange, 2 split by intermediate rows over the ranks (DeviceForward's
    // split_shared). STRIX_DS41_BATCH_SHARED=overlap|split.
    int shared_mode = 0;
    // world 2: while an exchange travels, read ahead the weights of the next steps (at most prefetch_bytes) so they
    // come from the MALL afterwards. STRIX_DS41_BATCH_PREFETCH=MB (0 off, default).
    std::uint64_t prefetch_bytes = 0;
    Dev<std::uint32_t> never;
    std::vector<float> logits_host;  // [row][slice or vocab]
    std::uint64_t fixed_bytes = 0, expert_bytes = 0, state_bytes = 0, workspace = 0;
    struct LastSegment {
        std::uint32_t slot, first, count, position;  // rows [first, first + count) of the last step, from `position`
    };
    std::vector<LastSegment> last_segments;
    Dev<std::uint8_t> control;  // the serving control channel: [out][in], control_bytes each

    [[nodiscard]] bool distributed() const noexcept { return options.world == 2U; }
    [[nodiscard]] double comm_seconds() const { return distributed() ? options.communicator->stats().seconds : 0.0; }
    std::unique_ptr<ds41::WeightOverlay> overlay;  // weights_overlay (Q8_0 form)
    // DeviceForward's exact storage forms (same env switches, same defaults): routers as their BF16 copy, engram_kv as
    // its F8 rows, each only when the check at load finds every value representable. compact_saved = the bytes they
    // save against the header (footprint() stays the F32/F16 size).
    std::set<std::string> router_bf16, f8;
    std::uint64_t compact_saved = 0;
    // STRIX_DS41_BATCH_PACKED (default on since P11; =0 keeps the Q8_0 form): the overlay's Q8_0 x A8 sites and shared
    // expert held as DeviceForward's exact packed rows (ds41_packed: CB4 + table, U5 / U6), read by the same products
    // at every R <= max_rows.
    std::map<std::string, std::array<std::uint32_t, 4>> cb4;  // weights packed to CB4 and their tables
    std::map<std::string, std::uint32_t> uniform;              // weights packed to U5 / U6 (bits)
    [[nodiscard]] const std::uint8_t* dev(const std::string& name) const { return weights.at(name).p; }
    void site_weight(ds41::Q8A8View& v, const std::string& name) const {  // Q8_0 rows, or the packed rows of an overlay tensor
        v.weight = dev(name);
        const auto it = cb4.find(name);
        v.packed = it != cb4.end() ? 4U : uniform.contains(name) ? uniform.at(name) : 0U;
        if (it != cb4.end()) std::copy(it->second.begin(), it->second.end(), v.table);
    }
    [[nodiscard]] ds41::WeightFormat fmt_of(const std::string& name) const {  // storage format of a shared-expert tensor
        return cb4.contains(name) ? ds41::WeightFormat::cb4
            : uniform.contains(name) ? (uniform.at(name) == 5U ? ds41::WeightFormat::u5 : ds41::WeightFormat::u6)
                                     : ds41::WeightFormat::q8_0;
    }
    template <typename View>
    void set_table(View& v, const std::string& name) const {
        if (const auto it = cb4.find(name); it != cb4.end()) std::copy(it->second.begin(), it->second.end(), v.table);
    }
    [[nodiscard]] const float* devf(const std::string& name) const { return reinterpret_cast<const float*>(weights.at(name).p); }
    [[nodiscard]] const std::uint16_t* devh(const std::string& name) const {
        return reinterpret_cast<const std::uint16_t*>(weights.at(name).p);
    }
    std::span<const std::uint8_t> host_tensor(const std::string& name) const {
        const auto it = tensors.find(name);
        if (it == tensors.end()) throw std::invalid_argument("deepseek41 batch forward: no tensor " + name);
        if (overlay)
            if (const auto* e = overlay->find(name)) return overlay->bytes(*e);
        return file->bytes(it->second->absolute_offset, it->second->byte_size);
    }
    template <typename T>
    Dev<T> track(std::size_t count) {
        Dev<T> d(count);
        workspace += d.bytes();
        return d;
    }
    void zero_slot(Slot& s) {
        for (auto& L : s.layers) {
            check(hipMemsetAsync(L.ring.p, 0, L.ring.bytes(), stream), "ring");
            if (L.carry[0].p != nullptr) {
                if (L.kv != nullptr) {
                    check(hipMemsetAsync(L.kv, 0, L.kv_bytes, stream), "kv cache");
                    check(hipMemsetAsync(L.idx, 0, L.idx_bytes, stream), "index cache");
                }
                for (int k = 0; k < 2; ++k) {
                    check(hipMemsetAsync(L.carry[k].p, 0, L.carry[k].bytes(), stream), "carry");
                    check(hipMemsetAsync(L.carry_valid[k].p, 0, L.carry_valid[k].bytes(), stream), "carry");
                }
            }
            L.carry_cur = 0;
        }
        s.hasher->reset();
        s.position = 0;
    }
    ~Impl() {
        trellis_oa.reset();
        trellis.reset();
    }
};

BatchForward::BatchForward(const std::filesystem::path& gguf, const BatchForwardOptions& options) : impl_(std::make_unique<Impl>()) {
    const auto t_load = std::chrono::steady_clock::now();
    auto& m = *impl_;
    m.options = options;
    if (options.world != 1U && options.world != 2U) throw std::invalid_argument("deepseek41 batch forward: world must be 1 or 2");
    if (options.world == 2U && (options.communicator == nullptr || options.rank > 1U))
        throw std::invalid_argument("deepseek41 batch forward: world 2 needs a communicator and rank 0 or 1");
    if (options.slots == 0U || options.slots > 64U) throw std::invalid_argument("deepseek41 batch forward: 1..64 slots");
    if (options.gather_logits && options.world != 2U) throw std::invalid_argument("deepseek41 batch forward: gather_logits is a world-2 option");
    m.model = validate_target(read_file(gguf));
    m.plan = build_decode_plan(m.model, 0, options.layers);
    m.file = std::make_unique<Mapped>(gguf);
    for (const auto& t : m.model.manifest.tensors) m.tensors.emplace(t.name, &t);
    {
        auto path = options.weights_overlay;
        if (const char* e = std::getenv("STRIX_DS41_OVERLAY"); path.empty() && e != nullptr && *e != 0) path = e;
        if (!path.empty()) m.overlay = std::make_unique<ds41::WeightOverlay>(path, m.model.manifest);
    }
    check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking), "stream");
    const auto& c = m.model.config;
    {
        const std::uint32_t per_rank = c.n_routed_experts / options.world;
        m.expert_first = options.rank * per_rank;
        m.expert_end = m.expert_first + per_rank;
        m.expert_tp = ds41::expert_tp::decode_enabled(options.world, c.moe_intermediate_size, false);  // default on since P16
        if (m.expert_tp) {
            if (!m.distributed() || c.moe_intermediate_size != ds41::expert_tp::inter)
                throw std::invalid_argument("deepseek41 batch forward: STRIX_DS41_EXPERT_TP=1 needs two ranks and I = 2304");
            m.expert_first = 0;
            m.expert_end = c.n_routed_experts;
        }
        m.head_slice = m.distributed() ? ds41::vocab_slice(c.vocab_size, 2U, options.rank) : ds41::VocabSlice{0U, c.vocab_size};
    }
    const bool tp = m.distributed() && options.tp_attention;
    std::set<std::string> names;
    for (const auto& s : m.plan.steps)
        for (const auto& w : s.weights)
            if (w.find("engram_embd") == std::string::npos) names.insert(w);
    DirectLoader loader(gguf);
    std::unique_ptr<DirectLoader> overlay_loader;
    if (m.overlay) overlay_loader = std::make_unique<DirectLoader>(m.overlay->path());
    ds41::prepared::HipLoader prepared(gguf, m.overlay ? m.overlay->path() : std::filesystem::path{}, m.model, options.rank, options.world, m.expert_tp);
    const auto env_on = [](const char* name) { const char* v = std::getenv(name); return !(v && *v == '0'); };
    const bool use_router_bf16 = env_on("STRIX_DS41_ROUTER_BF16"), use_f16_as_f8 = env_on("STRIX_DS41_F16_AS_F8");
    const bool use_packed = env_on("STRIX_DS41_BATCH_PACKED");  // P11: +4-7.5 % at N <= 4 with c-u6, same bits
    for (const auto& n : names) {
        const auto* t = m.tensors.at(n);
        const auto* over = m.overlay ? m.overlay->find(n) : nullptr;
        if (over == nullptr && use_f16_as_f8 && n.ends_with("engram_kv.weight")) {
            const auto f16 = m.host_tensor(n);
            const auto packed = ds41::pack_f16_as_f8(
                std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(f16.data()), f16.size() / 2U), t->dimensions[0]);
            if (packed) {
                Dev<std::uint8_t> d(packed->size());
                check(hipMemcpy(d.p, packed->data(), packed->size(), hipMemcpyHostToDevice), "F8 weight upload");
                m.f8.insert(n);
                m.fixed_bytes += d.n;
                m.compact_saved += t->byte_size - d.n;
                m.weights.emplace(n, std::move(d));
                continue;
            }
        }
        if (over == nullptr && use_router_bf16 && n.ends_with("ffn_gate_inp.weight")) {  // exact BF16 copy of an F32 router
            const auto f32 = m.host_tensor(n);
            std::vector<std::uint16_t> half(f32.size() / 4U);
            bool exact = true;
            for (std::size_t i = 0; i < half.size(); ++i) {
                std::uint32_t u;
                std::memcpy(&u, f32.data() + 4U * i, 4);
                exact = exact && (u & 0xffffU) == 0U;
                half[i] = static_cast<std::uint16_t>(u >> 16U);
            }
            if (exact) {
                Dev<std::uint8_t> d(half.size() * 2U);
                check(hipMemcpy(d.p, half.data(), half.size() * 2U, hipMemcpyHostToDevice), "BF16 router upload");
                m.router_bf16.insert(n);
                m.fixed_bytes += d.n;
                m.compact_saved += t->byte_size - d.n;
                m.weights.emplace(n, std::move(d));
                continue;
            }
        }
        std::uint64_t offset = over != nullptr ? over->absolute_offset : t->absolute_offset, bytes = t->byte_size;
        if (n.ends_with("_exps.weight")) {
            const auto& cfg = m.model.config;
            const bool down = n.ends_with("ffn_down_exps.weight");
            const auto count = m.expert_tp ? ds41::expert_tp::part_bytes(bytes, cfg.n_routed_experts, down, cfg.hidden_size)
                                         : bytes / cfg.n_routed_experts * (m.expert_end - m.expert_first);
            // Allocation is also needed by the fallback; avoid allocating this speculative destination
            // unless an artifact was requested (the adapter stays inert otherwise).
            if (prepared.active()) {
                Dev<std::uint8_t> d(count);
                if (prepared.load(n, count, d.p, m.stream)) {
                    m.expert_bytes += m.expert_tp ? count / cfg.n_routed_experts : bytes / cfg.n_routed_experts;
                    m.weights.emplace(n, std::move(d));
                    continue;
                }
            }
        }
        if (m.expert_tp && n.ends_with("_exps.weight")) {  // this rank's part of every expert (ds41_expert_tp.hpp)
            const bool down = n.ends_with("ffn_down_exps.weight");
            Dev<std::uint8_t> d(ds41::expert_tp::part_bytes(bytes, c.n_routed_experts, down, c.hidden_size));
            auto& from = over != nullptr ? *overlay_loader : loader;
            ds41::expert_tp::load_part([&](std::uint64_t o, std::uint64_t b, std::uint8_t* dst) { from.load(o, b, dst); }, offset, bytes,
                                       c.n_routed_experts, down, c.hidden_size, options.rank, d.p, m.stream);
            m.expert_bytes += d.n / c.n_routed_experts;  // one expert's part, summed over the layers below
            m.weights.emplace(n, std::move(d));
            continue;
        }
        if (n.ends_with("_exps.weight")) {  // [experts][rows][row bytes]: this rank's expert range only
            const std::uint64_t per = bytes / c.n_routed_experts;
            offset += per * m.expert_first;
            bytes = per * (m.expert_end - m.expert_first);
            m.expert_bytes += per;  // gate + up + down of one expert, summed over the layers below
        } else if (n == "output.weight") {  // [vocab][row bytes]: this rank's vocabulary slice only
            const std::uint64_t per = bytes / c.vocab_size;
            offset += per * m.head_slice.first;
            bytes = per * m.head_slice.count;
        }
        const bool site = n.ends_with("attn_q_a.weight") || n.ends_with("attn_q_b.weight") || n.ends_with("attn_kv.weight") ||
                          n.ends_with("attn_output_a.weight") || n.ends_with("attn_output_b.weight") || n == "output.weight" ||
                          n.ends_with("_shexp.weight");
        std::optional<ds41::PackedChoice> choice;
        if (over != nullptr && use_packed && site) choice = ds41::choose_packing(m.overlay->bytes(*over));
        Dev<std::uint8_t> d;
        if (choice) {  // packed rows of this rank's range (the whole tensor, or its vocabulary slice)
            const auto packed = ds41::pack_rows(m.overlay->bytes(*over).subspan(offset - over->absolute_offset, bytes), t->dimensions[0],
                                                *choice, ds41::PackedLayout::interleaved);
            d = Dev<std::uint8_t>(packed.size());
            check(hipMemcpy(d.p, packed.data(), packed.size(), hipMemcpyHostToDevice), "packed weight upload");
            if (choice->kind == ds41::PackedKind::cb4) {
                std::array<std::uint32_t, 4> words{};
                std::memcpy(words.data(), choice->table.data(), 16);
                m.cb4.emplace(n, words);
            } else {
                m.uniform.emplace(n, static_cast<std::uint32_t>(choice->kind));
            }
            m.compact_saved += bytes - packed.size();
            bytes = packed.size();
        } else {
            d = Dev<std::uint8_t>(bytes);
            (over != nullptr ? *overlay_loader : loader).load(offset, bytes, d.p);
        }
        // bytes every step reads: all non-expert weights except the embedding table (one row per token) and, with
        // TP, the other rank's halves of q_b, wo_a and wo_b
        if (!n.ends_with("_exps.weight") && n != "token_embd.weight")
            m.fixed_bytes += tp && (n.ends_with("attn_q_b.weight") || n.ends_with("attn_output_a.weight") || n.ends_with("attn_output_b.weight"))
                                 ? bytes / 2U
                                 : bytes;
        m.weights.emplace(n, std::move(d));
    }
    prepared.report();
    ds41::prepared::device_hashes(m.weights, options.rank, m.stream);
    m.expert_bytes /= std::max<std::uint32_t>(options.layers, 1U);  // per layer (every layer has the same expert shapes)
    // trellis sites (default off), prepared as in DeviceForward; the Q8_0 / packed copies stay loaded (prefill)
    if (const auto spec = ds41::trellis_env_spec(); !spec.empty()) {  // STRIX_DS41_TRELLIS or STRIX_DS41_TRELLIS_MIX
        m.trellis_pack = std::make_unique<ds41::TrellisPack>(spec);
        m.trellis = std::make_unique<ds41::TrellisContext>();
        const char* sites = std::getenv("STRIX_DS41_TRELLIS_SITES");
        m.trellis_classes = ds41::parse_trellis_site_classes(sites != nullptr ? sites : "");
        if (m.trellis_classes.empty())
            std::fprintf(stderr, "ds41 trellis: STRIX_DS41_TRELLIS is set but no site class is selected (STRIX_DS41_TRELLIS_SITES empty): "
                                 "no site reads the pack\n");
        const auto prepare = [&](const std::string& name, const ds41::TrellisSlice& slice = {}) {
            const auto pack_name = ds41::trellis_pack_name(name);
            if (!pack_name) throw std::logic_error("deepseek41 batch forward: no trellis name for " + name);
            const auto& site = m.trellis->prepare(*m.trellis_pack, *pack_name, slice);
            m.trellis_sites[name] = &site;
            return &site;
        };
        for (auto& s : m.plan.steps) {
            if (s.op == Op::shared_expert && ds41::trellis_class_enabled(m.trellis_classes, s.weights[0])) {
                if (const char* t = std::getenv("STRIX_DS41_SHEXP_TP"); t != nullptr && *t == '1') {
                    // ds41-trellis-k5-v1:shexp-tp, the same slices as DeviceForward (world 2: own; one process: both)
                    const std::uint32_t inter = c.moe_intermediate_size;
                    if (m.trellis_shexp_tp.empty()) m.trellis_shexp_tp.assign(c.num_hidden_layers, {});
                    auto& slices = m.trellis_shexp_tp.at(static_cast<std::size_t>(s.layer));
                    for (std::uint32_t r = 0; r < ds41::shexp_tp_ranks; ++r) {
                        if (m.distributed() && r != options.rank) continue;
                        const auto gu = ds41::shexp_tp_gate_up_slice(inter, r), dn = ds41::shexp_tp_down_slice(inter, r);
                        ds41::ShexpTpSlices sl;
                        sl.gate = prepare(s.weights[0], gu);
                        sl.up = prepare(s.weights[1], gu);
                        sl.down = prepare(s.weights[2], dn);
                        slices.push_back(sl);
                    }
                    for (std::size_t k = 0; k < 3U; ++k) {
                        const auto& f = slices.front();
                        m.trellis_sites[s.weights[k]] = k == 0U ? f.gate : k == 1U ? f.up : f.down;
                    }
                    s.identity = ds41::trellis_shexp_step_identity(slices.front().gate->bits, slices.front().down->bits, true);
                } else {
                    const auto* gate = prepare(s.weights[0]);
                    (void)prepare(s.weights[1]);
                    const auto* down = prepare(s.weights[2]);
                    s.identity = ds41::trellis_shexp_step_identity(gate->bits, down->bits, false);
                }
            }
            if (s.op == Op::head && ds41::trellis_class_enabled(m.trellis_classes, "output.weight")) {
                for (std::uint32_t r = 0; r < 2U; ++r) {
                    if (m.distributed() && r != m.options.rank) continue;
                    const auto sl = ds41::vocab_slice(m.model.config.vocab_size, 2U, r);
                    m.trellis_head.emplace_back(sl, prepare("output.weight", {ds41::TrellisSlice::Kind::rows, sl.first, sl.count}));
                }
                s.identity = ds41::trellis_step_identity(m.trellis_head.back().second->bits, "head");
                m.headout = ds41::trellis_headout_enabled();
            }
        }
        // attention projections (docs/ds41-trellis-k5-v1.md, q_b / o_a / o_b), the same linears as DeviceForward
        m.trellis_attn.assign(c.num_hidden_layers, {});
        {
            const std::uint32_t first_rank = tp ? options.rank : 0U, ranks = tp ? 1U : 2U;
            const std::uint32_t q_rows = c.num_attention_heads / 2U * c.head_dim, o_half = c.o_groups * c.o_lora_rank / 2U;
            for (auto& s : m.plan.steps) {
                if (s.layer < 0 || (s.op != Op::q_b && s.op != Op::o_a && s.op != Op::o_b)) continue;
                const auto& name = s.weights[0];
                if (!ds41::trellis_class_enabled(m.trellis_classes, name)) continue;
                auto& ta = m.trellis_attn[static_cast<std::size_t>(s.layer)];
                for (std::uint32_t r = first_rank; r < first_rank + ranks; ++r) {
                    const char* cls = s.op == Op::q_b ? "q_b" : s.op == Op::o_a ? "o_a" : "o_b";
                    const ds41::TrellisSite* last = nullptr;
                    if (s.op == Op::q_b) {
                        ta.q_b.push_back(last = prepare(name, {ds41::TrellisSlice::Kind::rows, r * q_rows, q_rows}));
                    } else if (s.op == Op::o_b) {
                        ta.o_b.push_back(last = prepare(name, {ds41::TrellisSlice::Kind::columns, r * o_half, o_half}));
                    } else {
                        for (std::uint32_t g = r * (c.o_groups / 2U); g < (r + 1U) * (c.o_groups / 2U); ++g)
                            ta.o_a.push_back(last = &m.trellis->prepare(*m.trellis_pack, *ds41::trellis_pack_name(name, static_cast<int>(g))));
                    }
                    s.identity = ds41::trellis_step_identity(last->bits, cls);
                }
            }
        }
        m.trellis->reserve(m.stream, m.row_capacity);
        if (const int mode = ds41::trellis_oa_group_mode(); mode != 0)  // default off: the per-group calls below
            for (const auto& ta : m.trellis_attn)
                if (!ta.o_a.empty()) {
                    m.trellis_oa = std::make_unique<ds41::TrellisGroupRunner>(ta.o_a, mode, m.stream);
                    std::fprintf(stderr, "ds41 trellis: %s\n", m.trellis_oa->describe().c_str());
                    break;
                }
        m.fixed_bytes += m.trellis->device_bytes();
        const char* tp_env = std::getenv("STRIX_DS41_SHEXP_TP");
        m.trellis_announced = m.trellis->announced_identity(m.trellis_classes, tp_env != nullptr && *tp_env == '1');
        m.trellis_library = m.trellis->library_workspace_bytes() + (m.trellis_oa ? m.trellis_oa->library_workspace_bytes() : 0U);
        m.trellis_device = m.trellis->device_bytes() + m.trellis->staging_bytes() + (m.trellis_oa ? m.trellis_oa->staging_bytes() : 0U);
    }
    m.shexp_tp = !m.trellis_shexp_tp.empty();
    if (const char* t = std::getenv("STRIX_DS41_SHEXP_TP"); t != nullptr && *t == '1' && !m.shexp_tp)
        throw std::invalid_argument("deepseek41 batch forward: STRIX_DS41_SHEXP_TP=1 needs STRIX_DS41_TRELLIS with the shexp class");
    if (m.shexp_tp && m.expert_tp)  // two per-rank partial folds into the half sums: no declared identity for the pair yet
        throw std::invalid_argument("deepseek41 batch forward: STRIX_DS41_SHEXP_TP=1 with expert TP (STRIX_DS41_EXPERT_TP, default on "
                                    "since P16) is not a declared identity; set STRIX_DS41_EXPERT_TP=0 or leave SHEXP_TP unset");
    if (options.engram_direct_io) {
        std::vector<ds41::EngramTable> tables;
        for (const auto l : m.model.engram.layer_ids) {
            const auto* t = m.tensors.at("blk." + std::to_string(l) + ".engram_embd.weight");
            tables.push_back({t->absolute_offset, t->byte_size / 264U});
        }
        m.engram_reader = std::make_unique<ds41::EngramRowReader>(gguf, std::move(tables), ds41::engram_reader_threads(64U), true);  // 64 on the pair since P15
    }

    namespace attn = ds41::attn;
    m.roles = attn::released_plan();
    const std::size_t W = m.row_capacity;
    std::uint32_t source_ratio = 0;
    for (std::uint32_t l = 0; l < options.layers; ++l) {
        const auto& r = m.roles[l];
        if (r.owns_kv) m.max_count = std::max<std::size_t>(m.max_count, options.max_positions / r.ratio + 1U);
        if (r.candidate_source) source_ratio = r.ratio;
    }
    m.slots.resize(options.slots);
    for (auto& s : m.slots) {
        for (std::uint32_t l = 0; l < options.layers; ++l) {
            const auto& r = m.roles[l];
            Impl::LayerState L;
            L.ring = Dev<std::uint8_t>(std::size_t{AttentionHost::ring_capacity} * attn::kWindowEntryBytes);
            m.state_bytes += L.ring.bytes();
            if (r.owns_kv) {
                const std::size_t entries = options.max_positions / r.ratio + 1U;
                if (options.pool_positions == 0U) {
                    L.kv_cache = Dev<std::uint8_t>(entries * attn::kKvEntryBytes);
                    L.idx_cache = Dev<std::uint8_t>(entries * attn::kIdxEntryBytes);
                    L.kv = L.kv_cache.p;
                    L.idx = L.idx_cache.p;
                    L.kv_bytes = L.kv_cache.bytes();
                    L.idx_bytes = L.idx_cache.bytes();
                    m.state_bytes += L.kv_cache.bytes() + L.idx_cache.bytes();
                }
                for (int k = 0; k < 2; ++k) {
                    L.carry[k] = Dev<float>(2U * attn::kHeadDim);
                    L.carry_valid[k] = Dev<std::uint32_t>(1);
                    m.state_bytes += L.carry[k].bytes() + L.carry_valid[k].bytes();
                }
            }
            s.layers.push_back(std::move(L));
        }
        s.hasher = std::make_unique<ds41::EngramHasher>(m.model.engram);
        s.capacity = options.pool_positions == 0U ? options.max_positions : 0U;
        m.zero_slot(s);
    }
    m.state_bytes /= options.slots;
    if (options.pool_positions != 0U) {  // the shared arenas (a reservation zeroes its range)
        const std::uint64_t units = pool_units(options);
        m.pool_free[0] = units;
        m.kv_arena.resize(options.layers);
        m.idx_arena.resize(options.layers);
        for (std::uint32_t l = 0; l < options.layers; ++l)
            if (m.roles[l].owns_kv) {
                m.kv_arena[l] = Dev<std::uint8_t>(units / m.roles[l].ratio * attn::kKvEntryBytes);
                m.idx_arena[l] = Dev<std::uint8_t>(units / m.roles[l].ratio * attn::kIdxEntryBytes);
            }
    }
    m.scratch.resize(options.layers);
    for (std::uint32_t l = 0; l < options.layers; ++l)
        if (m.roles[l].owns_index) {
            m.scratch[l].idx = m.track<std::uint32_t>(W * attn::kTopK);
            m.scratch[l].idx_count = m.track<std::uint32_t>(W);
            check(hipMemsetAsync(m.scratch[l].idx.p, 0, m.scratch[l].idx.bytes(), m.stream), "top-k");
            check(hipMemsetAsync(m.scratch[l].idx_count.p, 0, m.scratch[l].idx_count.bytes(), m.stream), "top-k");
        }
    m.ckv.resize(options.layers);
    m.cscore.resize(options.layers);
    for (std::uint32_t l = 0; l < options.layers; ++l)
        if (m.roles[l].owns_kv) {
            m.ckv[l] = m.track<float>(W * attn::kHeadDim);
            m.cscore[l] = m.track<float>(W * attn::kHeadDim);
        }
    m.cand_bytes = source_ratio != 0U ? (options.max_positions / source_ratio + attn::kCandBlockSize) / attn::kCandBlockSize
                                      : (m.max_count + attn::kCandBlockSize) / attn::kCandBlockSize;
    m.candidates = m.track<std::uint8_t>(W * m.cand_bytes);
    check(hipMemsetAsync(m.candidates.p, 0, m.candidates.bytes(), m.stream), "candidates");
    m.index_scores = m.track<float>(W * m.max_count);
    m.select_keys = m.track<std::uint32_t>(W * m.max_count);
    m.iq = m.track<float>(W * attn::kIdxHeads * attn::kIdxDim);
    m.iw = m.track<float>(W * attn::kIdxHeads);
    m.index_qprep = m.track<std::uint8_t>(W * attn::hip::kIdxPrepBytes);
    m.attn_tickets = m.track<std::uint32_t>(W * attn::hip::kAttnGroupCount);
    check(hipMemsetAsync(m.attn_tickets.p, 0, m.attn_tickets.bytes(), m.stream), "attention tickets");
    m.router_tickets = m.track<std::uint32_t>(1);
    check(hipMemsetAsync(m.router_tickets.p, 0, m.router_tickets.bytes(), m.stream), "router tickets");
    m.attn_partial = m.track<float>(W * attn::kHeads * attn::kMaxChunks * (2U + attn::kHeadDim));
    m.pos64 = m.track<std::uint64_t>(W);
    m.rope_rows = m.track<attn::RopeRow>(3U * W);
    const std::size_t H = c.hidden_size, hc = c.hc_mult, I = c.moe_intermediate_size, K = c.num_experts_per_tok;
    m.streams[0] = m.track<float>(W * hc * H);
    m.streams[1] = m.track<float>(W * hc * H);
    if (options.capture_main_hidden) {
        if (options.layers < 4U) throw std::invalid_argument("deepseek41 batch forward: capture_main_hidden needs at least 4 layers");
        for (auto& cap : m.capture) cap = m.track<float>(W * hc * H);
    }
    m.x = m.track<float>(W * H);
    m.qr = m.track<float>(W * c.q_lora_rank);
    m.q = m.track<float>(W * c.num_attention_heads * c.head_dim);
    m.kv = m.track<float>(W * c.head_dim);
    m.o = m.track<float>(W * c.num_attention_heads * c.head_dim);
    m.o_low = m.track<float>(W * c.o_groups * c.o_lora_rank);
    m.y = m.track<float>(W * H);
    m.shared = m.track<float>(W * H);
    if (m.trellis) {
        m.tr_gate = m.track<float>(W * I);
        m.tr_up = m.track<float>(W * I);
        m.tr_act = m.track<float>(W * I);
        if (m.shexp_tp) m.tr_p1 = m.track<float>(W * H);  // rank 1 partial (one process); also allocated in world 2 (footprint)
    }
    m.raw = m.track<float>(W * K * H);
    m.low = m.track<float>(W * H);
    m.high = m.track<float>(W * H);
    m.h = m.track<float>(W * H);
    m.logits = m.track<float>(W * c.vocab_size);
    if (m.headout) {
        m.head_rows = Dev<float>(m.trellis_head.size() * W * m.trellis_head.front().first.count);
        m.trellis_device += std::uint64_t{m.head_rows.n} * sizeof(float);
    }
    const std::size_t cols = m.model.engram.rows_per_token();
    m.erows = m.track<std::uint8_t>(std::size_t{m.model.engram.layer_ids.size()} * W * cols * 264U);
    m.erows_pinned = Pinned<std::uint8_t>(m.erows.n);
    m.token_pinned = Pinned<std::uint32_t>(W);
    m.pos_pinned = Pinned<std::uint32_t>(W);
    m.pos64_pinned = Pinned<std::uint64_t>(W);
    m.rope_pinned = Pinned<attn::RopeRow>(3U * W);
    m.emb = m.track<float>(W * cols * 256U);
    m.ekv = m.track<float>(W * H * (hc + 1U));
    m.scores = m.track<float>(W * c.n_routed_experts);
    m.route_w = m.track<float>(W * K);
    m.ids = m.track<std::uint32_t>(W * K);
    m.image_row = m.track<std::uint8_t>(W);
    check(hipMemsetAsync(m.image_row.p, 0, W, m.stream), "image rows");
    m.token = m.track<std::uint32_t>(W);
    m.pos = m.track<std::uint32_t>(W);
    m.mix_attn = m.track<ds41::HcMix>(W);
    m.mix_ffn = m.track<ds41::HcMix>(W);
    m.mix_partial = m.track<float>(ds41::hc_mix_partial_floats(static_cast<std::uint32_t>(W), static_cast<std::uint32_t>(H)));
    m.hc_counters = m.track<std::uint32_t>(ds41::hc_site_counter_count(static_cast<std::uint32_t>(W)));
    check(hipMemsetAsync(m.hc_counters.p, 0, m.hc_counters.n * 4U, m.stream), "hc site counters");
    const std::size_t site_width = std::max<std::size_t>({H, c.q_lora_rank, std::size_t{c.num_attention_heads} * c.head_dim,
                                                          std::size_t{c.o_groups} * c.o_lora_rank});
    m.site_codes = m.track<std::int8_t>(W * site_width);
    m.o_codes = m.track<std::int8_t>(W * c.num_attention_heads * c.head_dim);
    m.o_deltas = m.track<float>(W * c.num_attention_heads * c.head_dim / 32U);
    m.o_sums = m.track<std::int32_t>(W * c.num_attention_heads * c.head_dim / 32U);
    m.site_deltas = m.track<float>(W * site_width / 32U);
    m.site_sums = m.track<std::int32_t>(W * site_width / 32U);
    m.codes = m.track<std::int8_t>(W * H);
    m.deltas = m.track<float>(W * H / 32U);
    m.sums = m.track<std::int32_t>(W * H / 32U);
    m.act_codes = m.track<std::int8_t>(W * K * I);
    m.act_deltas = m.track<float>(W * K * I / 32U);
    m.act_sums = m.track<std::int32_t>(W * K * I / 32U);
    m.sh_act_codes = m.track<std::int8_t>(W * I);
    m.sh_act_deltas = m.track<float>(W * I / 32U);
    m.sh_act_sums = m.track<std::int32_t>(W * I / 32U);
    std::uint32_t partial_blocks = 0;
    for (std::uint32_t r = 0; r < 2U; ++r)
        partial_blocks = std::max(partial_blocks, ds41::argmax_partial_blocks(ds41::vocab_slice(c.vocab_size, 2U, r).count));
    m.partial = m.track<ds41::ArgmaxCandidate>(W * partial_blocks);
    m.cand0 = m.track<ds41::ArgmaxCandidate>(W);
    m.cand1 = m.track<ds41::ArgmaxCandidate>(W);
    m.best = m.track<ds41::ArgmaxCandidate>(W);
    m.cand_own = m.track<ds41::ArgmaxCandidate>(W);
    m.cand_other = m.track<ds41::ArgmaxCandidate>(W);
    m.best_pinned = Pinned<ds41::ArgmaxCandidate>(W);
    m.half_own = m.track<float>(W * H + W * H / 2U);
    m.half_other = m.track<float>(W * H + W * H / 2U);
    m.shx_own = m.track<std::uint8_t>(W * I / 2U + W * I / 64U * 4U);
    m.shx_other = m.track<std::uint8_t>(W * I / 2U + W * I / 64U * 4U);
    if (m.distributed()) {
        const char* v = std::getenv("STRIX_DS41_BATCH_SHARED");
        const std::string mode = v ? v : "";
        m.shared_mode = mode == "overlap" ? 1 : mode == "split" ? 2 : 0;
        if (m.shared_mode == 2 && (I % 64U != 0U || H % 2U != 0U)) throw std::invalid_argument("deepseek41 batch forward: shared split geometry");
        if (m.shared_mode == 2)
            for (const auto& s : m.plan.steps)
                if (s.op == Op::shared_expert && m.trellis_site(s.weights[0]) != nullptr)
                    throw std::invalid_argument("deepseek41 batch forward: the trellis shared expert is replicated (no STRIX_DS41_BATCH_SHARED=split)");
        if (const char* pf = std::getenv("STRIX_DS41_BATCH_PREFETCH")) m.prefetch_bytes = std::strtoull(pf, nullptr, 10) << 20U;
    }
    // also STRIX_DS41_BATCH_ENGRAM_ASYNC=1 (ds41/batch-perf cd5e640, the same per-layer reads on a host thread)
    m.engram_async = [] {
        const char* v = std::getenv("STRIX_DS41_ENGRAM_ASYNC");
        const char* b = std::getenv("STRIX_DS41_BATCH_ENGRAM_ASYNC");
        return !((v && *v == '0') || (b && *b == '0'));
    }();
    // P13 defaults (after P12): HC_SITE, QA_KV, ENGRAM_ASYNC and COMPRESS_RING on, =0 turns each off.
    m.compress_ring = [] { const char* v = std::getenv("STRIX_DS41_COMPRESS_RING"); return !(v && *v == '0'); }();
    m.never = Dev<std::uint32_t>(1);
    {
        const std::size_t n = W * K + 1U;
        m.built.experts = m.track<std::uint32_t>(n);
        m.built.begins = m.track<std::uint32_t>(n);
        m.built.rows = m.track<std::uint32_t>(n);
        m.built.slots = m.track<std::uint32_t>(n);
        m.built.order = m.track<std::uint32_t>(n);
        m.built.low_counts = m.track<std::uint32_t>(n);
        m.built.counts = m.track<std::uint32_t>(2);
    }
    m.counts_pinned = Pinned<std::uint32_t>(2U * std::max<std::uint32_t>(options.layers, 1U));
    for (std::uint32_t r = 1; r <= W; ++r) {  // the shared expert as expert 0 of a one-expert window of r rows
        const auto routes = ds41::build_window_routes(std::vector<std::uint32_t>(r, 0U), r, 1U, 1U);
        Impl::SharedRoutes s;
        s.experts = Dev<std::uint32_t>(routes.experts.size());
        s.begins = Dev<std::uint32_t>(routes.begins.size());
        s.rows = Dev<std::uint32_t>(routes.rows.size());
        s.slots = Dev<std::uint32_t>(routes.slots.size());
        s.experts.upload(routes.experts, m.stream);
        s.begins.upload(routes.begins, m.stream);
        s.rows.upload(routes.rows, m.stream);
        s.slots.upload(routes.slots, m.stream);
        check(hipStreamSynchronize(m.stream), "shared routes");  // the uploads read the temporaries
        s.count = static_cast<std::uint32_t>(routes.experts.size());
        s.max_rows = routes.max_rows;
        m.shared_routes.emplace(r, std::move(s));
    }
    if (options.slice_logits || options.gather_logits)
        m.logits_host.assign(std::size_t{W} * (options.gather_logits ? c.vocab_size : m.head_slice.count), 0.0F);
    check(hipStreamSynchronize(m.stream), "setup");
    if (m.workspace != workspace_bytes_for(m.model, options) || m.state_bytes != slot_state_bytes(options) ||
        weight_bytes() + m.compact_saved != resident_weight_bytes(m.model, m.plan, options) || pool_bytes() != pool_bytes_for(options) ||
        m.trellis_device != trellis_bytes_for(m.model, m.plan, options).total())
        throw std::logic_error("deepseek41 batch forward: footprint() disagrees with the allocations");
    m.control = Dev<std::uint8_t>(2U * control_bytes);
    load_seconds_ = seconds_since(t_load);
}

std::string BatchForward::trellis_identity() const { return impl_->trellis_announced; }
std::uint64_t BatchForward::trellis_bytes() const noexcept { return impl_->trellis_device; }
std::uint64_t BatchForward::trellis_library_bytes() const noexcept { return impl_->trellis_library; }

std::string BatchForward::overlay_description() const {
    if (!impl_->overlay) return {};
    const auto packed = impl_->cb4.size() + impl_->uniform.size();
    return impl_->overlay->describe() + (packed ? ", packed sites " + std::to_string(packed) : std::string{});
}

BatchForward::~BatchForward() {
    if (impl_) {
        impl_->trellis_oa.reset();
        impl_->trellis.reset();
    }
    if (impl_ && impl_->stream != nullptr) {
        (void)hipStreamSynchronize(impl_->stream);
        (void)hipStreamDestroy(impl_->stream);
    }
}

const TargetModel& BatchForward::model() const noexcept { return impl_->model; }

std::pair<std::uint64_t, std::uint64_t> BatchForward::device_memory() const {
    std::size_t free = 0, total = 0;
    check(hipMemGetInfo(&free, &total), "memory info");
    return {free, total};
}

void BatchForward::exchange_host(std::span<const std::byte> out, std::span<std::byte> in) {
    auto& m = *impl_;
    if (!m.distributed()) throw std::logic_error("deepseek41 batch forward: exchange_host needs world 2");
    if (out.size() != in.size() || out.size() > control_bytes) throw std::invalid_argument("deepseek41 batch forward: control messages are equal and <= 64 KiB");
    check(hipMemcpyAsync(m.control.p, out.data(), out.size(), hipMemcpyHostToDevice, m.stream), "control upload");
    m.options.communicator->exchange(m.stream, m.control.p, m.control.p + control_bytes, out.size(),
                                     dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), 0xC0U, 0U});
    check(hipMemcpyAsync(in.data(), m.control.p + control_bytes, in.size(), hipMemcpyDeviceToHost, m.stream), "control download");
    check(hipStreamSynchronize(m.stream), "control");
}
void BatchForward::quiesce() {
    auto& m = *impl_;
    if (!m.distributed() || !m.options.communicator) return;
    m.options.communicator->quiesce(m.stream);
    check(hipStreamSynchronize(m.stream), "quiesce");
}
std::uint64_t BatchForward::fixed_bytes_per_step() const noexcept { return impl_->fixed_bytes; }
std::uint64_t BatchForward::expert_bytes() const noexcept { return impl_->expert_bytes; }
std::uint64_t BatchForward::state_bytes_per_slot() const noexcept { return impl_->state_bytes; }
std::uint64_t BatchForward::workspace_bytes() const noexcept { return impl_->workspace; }
std::uint64_t BatchForward::weight_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& [name, d] : impl_->weights) n += d.n;
    return n;
}

bool BatchForward::expert_tp() const noexcept { return impl_->expert_tp; }

std::map<std::string, std::pair<const void*, std::uint64_t>> BatchForward::file_form_weights() const {
    const auto& m = *impl_;
    std::map<std::string, std::pair<const void*, std::uint64_t>> out;
    for (const auto& [name, d] : m.weights)
        if (!m.f8.contains(name) && !m.router_bf16.contains(name) && !m.cb4.contains(name) && !m.uniform.contains(name))
            out.emplace(name, std::make_pair(static_cast<const void*>(d.p), static_cast<std::uint64_t>(d.bytes())));
    return out;
}

std::uint32_t BatchForward::position(std::uint32_t slot) const { return impl_->slots.at(slot).position; }
void* BatchForward::stream() const noexcept { return impl_->stream; }
std::array<const float*, 3> BatchForward::captured_streams() const {
    if (!impl_->options.capture_main_hidden) throw std::logic_error("deepseek41 batch forward: capture_main_hidden is off");
    return {impl_->capture[0].p, impl_->capture[1].p, impl_->capture[2].p};
}
std::uint32_t BatchForward::row_limit() const noexcept { return impl_->row_capacity; }
std::uint32_t BatchForward::capacity(std::uint32_t slot) const { return impl_->slots.at(slot).capacity; }
std::uint64_t BatchForward::pool_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& d : impl_->kv_arena) n += d.n;
    for (const auto& d : impl_->idx_arena) n += d.n;
    return n;
}
std::uint64_t BatchForward::pool_free_positions() const {
    std::uint64_t best = 0;
    for (const auto& [begin, len] : impl_->pool_free) best = std::max(best, len);
    return best >= 16U ? (best - 8U) / 8U * 8U : 0U;
}

void BatchForward::release(std::uint32_t slot) {
    auto& m = *impl_;
    auto& s = m.slots.at(slot);
    if (m.options.pool_positions == 0U || s.pool_units == 0U) return;
    check(hipStreamSynchronize(m.stream), "release");  // no queued work still reads the range
    auto it = m.pool_free.emplace(s.pool_begin, s.pool_units).first;
    if (it != m.pool_free.begin()) {  // merge with the free range before
        const auto prev = std::prev(it);
        if (prev->first + prev->second == it->first) {
            prev->second += it->second;
            m.pool_free.erase(it);
            it = prev;
        }
    }
    const auto next = std::next(it);
    if (next != m.pool_free.end() && it->first + it->second == next->first) {
        it->second += next->second;
        m.pool_free.erase(next);
    }
    for (auto& L : s.layers) {
        L.kv = L.idx = nullptr;
        L.kv_bytes = L.idx_bytes = 0;
    }
    s.pool_begin = s.pool_units = 0;
    s.capacity = 0;
    s.position = 0;
}

bool BatchForward::reserve(std::uint32_t slot, std::uint32_t positions) {
    auto& m = *impl_;
    namespace attn = ds41::attn;
    auto& s = m.slots.at(slot);
    if (positions > m.options.max_positions) return false;
    if (m.options.pool_positions == 0U) {
        reset(slot);
        return true;
    }
    const std::uint64_t units = (std::uint64_t{positions} + 7U) / 8U * 8U + 8U;
    release(slot);
    const auto it = std::find_if(m.pool_free.begin(), m.pool_free.end(), [&](const auto& f) { return f.second >= units; });
    if (it == m.pool_free.end()) return false;
    const std::uint64_t begin = it->first, left = it->second - units;
    m.pool_free.erase(it);
    if (left != 0U) m.pool_free[begin + units] = left;
    s.pool_begin = begin;
    s.pool_units = units;
    s.capacity = positions;
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        if (!m.roles[l].owns_kv) continue;
        auto& L = s.layers[l];
        const std::uint64_t ratio = m.roles[l].ratio, first = begin / ratio, entries = units / ratio;
        L.kv = m.kv_arena[l].p + first * attn::kKvEntryBytes;
        L.idx = m.idx_arena[l].p + first * attn::kIdxEntryBytes;
        L.kv_bytes = entries * attn::kKvEntryBytes;
        L.idx_bytes = entries * attn::kIdxEntryBytes;
    }
    reset(slot);
    return true;
}

void BatchForward::reset(std::uint32_t slot) {
    auto& m = *impl_;
    m.zero_slot(m.slots.at(slot));
    check(hipStreamSynchronize(m.stream), "reset");
}

void BatchForward::copy_slot(std::uint32_t from, std::uint32_t to) {
    auto& m = *impl_;
    auto& a = m.slots.at(from);
    auto& b = m.slots.at(to);
    if (from == to) return;
    if (a.position > b.capacity) throw std::out_of_range("deepseek41 batch forward: copy_slot: the target slot has no room for the context");
    for (std::size_t l = 0; l < a.layers.size(); ++l) {
        auto& A = a.layers[l];
        auto& B = b.layers[l];
        const auto copy = [&](const auto& src, auto& dst) {
            if (src.p != nullptr) check(hipMemcpyAsync(dst.p, src.p, src.bytes(), hipMemcpyDeviceToDevice, m.stream), "copy slot");
        };
        copy(A.ring, B.ring);
        if (A.kv != nullptr && B.kv != nullptr) {  // the used entries (the rest of B's range is rewritten before a read)
            const std::size_t entries = a.position / m.roles[l].ratio;
            check(hipMemcpyAsync(B.kv, A.kv, entries * ds41::attn::kKvEntryBytes, hipMemcpyDeviceToDevice, m.stream), "copy slot");
            check(hipMemcpyAsync(B.idx, A.idx, entries * ds41::attn::kIdxEntryBytes, hipMemcpyDeviceToDevice, m.stream), "copy slot");
        }
        for (int k = 0; k < 2; ++k) {
            copy(A.carry[k], B.carry[k]);
            copy(A.carry_valid[k], B.carry_valid[k]);
        }
        B.carry_cur = A.carry_cur;
    }
    *b.hasher = *a.hasher;
    b.position = a.position;
    check(hipStreamSynchronize(m.stream), "copy slot");
}

void BatchForward::import_state(std::uint32_t slot, const AttentionState& s) {
    auto& m = *impl_;
    namespace attn = ds41::attn;
    auto& S = m.slots.at(slot);
    const auto fail = [](const std::string& what) { throw std::invalid_argument("deepseek41 batch forward: import_state: " + what); };
    if (s.layers.size() != m.options.layers) fail("layer count differs");
    if (s.tokens.size() != s.position || s.position == 0U) fail("token history length != position, or an empty state");
    if (s.position >= S.capacity) fail("the slot has no room (reserve it first)");
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        auto& L = S.layers[l];
        const auto& T = s.layers[l];
        if (T.owns_kv != r.owns_kv || (r.owns_kv && T.ratio != r.ratio) || T.ring.size() != L.ring.bytes()) fail("layer " + std::to_string(l) + " differs");
        check(hipMemcpyAsync(L.ring.p, T.ring.data(), T.ring.size(), hipMemcpyHostToDevice, m.stream), "import ring");
        if (!r.owns_kv) continue;
        const std::size_t entries = s.position / r.ratio;
        if (T.kv.size() != entries * attn::kKvEntryBytes || T.idx_keys.size() != entries * attn::kIdxEntryBytes) fail("caches of layer " + std::to_string(l));
        check(hipMemsetAsync(L.kv, 0, L.kv_bytes, m.stream), "import kv");
        check(hipMemsetAsync(L.idx, 0, L.idx_bytes, m.stream), "import idx");
        check(hipMemcpyAsync(L.kv, T.kv.data(), T.kv.size(), hipMemcpyHostToDevice, m.stream), "import kv");
        check(hipMemcpyAsync(L.idx, T.idx_keys.data(), T.idx_keys.size(), hipMemcpyHostToDevice, m.stream), "import idx");
        for (int k = 0; k < 2; ++k) {
            check(hipMemsetAsync(L.carry[k].p, 0, L.carry[k].bytes(), m.stream), "import carry");
            check(hipMemsetAsync(L.carry_valid[k].p, 0, L.carry_valid[k].bytes(), m.stream), "import carry");
        }
        L.carry_cur = 0;
        if (r.ratio == 2U && T.carry_valid) {
            if (T.carry.size() * 4U != L.carry[0].bytes()) fail("carry of layer " + std::to_string(l));
            check(hipMemcpyAsync(L.carry[0].p, T.carry.data(), L.carry[0].bytes(), hipMemcpyHostToDevice, m.stream), "import carry");
            check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(L.carry_valid[0].p), 1, 1, m.stream), "import carry");
        }
    }
    S.hasher->reset();
    std::vector<std::uint64_t> one(std::size_t{S.hasher->layers()} * S.hasher->columns());
    for (const auto t : s.tokens) S.hasher->push(t, one);
    S.position = s.position;
    check(hipStreamSynchronize(m.stream), "import state");
}

AttentionState BatchForward::export_state(std::uint32_t slot, std::uint32_t kv_from) const {
    const auto& m = *impl_;
    namespace attn = ds41::attn;
    const auto& S = m.slots.at(slot);
    if (kv_from > S.position) throw std::invalid_argument("deepseek41 batch forward: export_state: kv_from beyond the position");
    AttentionState s;
    s.position = S.position;
    s.layers.resize(m.options.layers);
    std::vector<std::uint32_t> valid(m.options.layers, 0U);
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        const auto& L = S.layers[l];
        auto& T = s.layers[l];
        T.owns_kv = r.owns_kv;
        T.owns_index = r.owns_index;
        T.ratio = r.ratio;
        T.topk_known = false;
        T.ring.resize(L.ring.bytes());
        check(hipMemcpyAsync(T.ring.data(), L.ring.p, L.ring.bytes(), hipMemcpyDeviceToHost, m.stream), "export ring");
        if (!r.owns_kv) continue;
        if (kv_from % r.ratio != 0U) throw std::invalid_argument("deepseek41 batch forward: export_state: kv_from not a multiple of a ratio");
        const std::size_t first = kv_from / r.ratio, entries = S.position / r.ratio - first;
        T.kv.resize(entries * attn::kKvEntryBytes);
        T.idx_keys.resize(entries * attn::kIdxEntryBytes);
        if (entries != 0U) {
            check(hipMemcpyAsync(T.kv.data(), L.kv + first * attn::kKvEntryBytes, T.kv.size(), hipMemcpyDeviceToHost, m.stream), "export kv");
            check(hipMemcpyAsync(T.idx_keys.data(), L.idx + first * attn::kIdxEntryBytes, T.idx_keys.size(), hipMemcpyDeviceToHost, m.stream), "export idx");
        }
        if (r.ratio == 2U) {
            T.carry.resize(L.carry[L.carry_cur].n);
            check(hipMemcpyAsync(T.carry.data(), L.carry[L.carry_cur].p, L.carry[L.carry_cur].bytes(), hipMemcpyDeviceToHost, m.stream), "export carry");
            check(hipMemcpyAsync(&valid[l], L.carry_valid[L.carry_cur].p, 4U, hipMemcpyDeviceToHost, m.stream), "export carry");
        }
    }
    check(hipStreamSynchronize(m.stream), "export state");
    for (std::uint32_t l = 0; l < m.options.layers; ++l)
        if (m.roles[l].owns_kv && m.roles[l].ratio == 2U) s.layers[l].carry_valid = valid[l] != 0U;
    return s;
}

void BatchForward::commit(std::uint32_t slot, std::uint32_t kept) {
    auto& m = *impl_;
    const auto it = std::find_if(m.last_segments.begin(), m.last_segments.end(), [&](const auto& g) { return g.slot == slot; });
    if (it == m.last_segments.end()) throw std::invalid_argument("deepseek41 batch forward: the slot had no rows in the last step");
    const auto g = *it;
    if (kept == 0U || kept > g.count) throw std::invalid_argument("deepseek41 batch forward: commit keeps 1..R rows of the slot");
    if (kept == g.count) return;
    auto& S = m.slots[slot];
    const std::uint32_t last = g.position + kept - 1U;
    const std::size_t D = ds41::attn::kHeadDim;
    // the ratio-2 compressor's pending group after row kept - 1 (DeviceForward::commit_window, per request)
    if (m.compress_ring) {
        CarryRestores a{};
        std::uint32_t n = 0;
        const auto launch = [&] {
            if (n == 0U) return;
            restore_carries<<<n, 256, 0, m.stream>>>(a, last % 2U == 0U ? 1U : 0U);
            check(hipGetLastError(), "rollback carry");
            n = 0;
        };
        for (std::uint32_t l = 0; l < m.options.layers; ++l) {
            const auto& r = m.roles[l];
            if (!r.owns_kv || r.ratio != 2U) continue;
            auto& L = S.layers[l];
            a.layer[n++] = {reinterpret_cast<std::uint32_t*>(L.carry[L.carry_cur].p), L.carry_valid[L.carry_cur].p,
                            reinterpret_cast<const std::uint32_t*>(m.ckv[l].p + (g.first + kept - 1U) * D),
                            reinterpret_cast<const std::uint32_t*>(m.cscore[l].p + (g.first + kept - 1U) * D)};
            if (n == max_carry_restores) launch();
        }
        launch();
    }
    for (std::uint32_t l = 0; l < m.options.layers && !m.compress_ring; ++l) {
        const auto& r = m.roles[l];
        if (!r.owns_kv || r.ratio != 2U) continue;
        auto& L = S.layers[l];
        float* state = L.carry[L.carry_cur].p;
        if (last % 2U == 0U) {
            check(hipMemcpyAsync(state, m.ckv[l].p + (g.first + kept - 1U) * D, D * 4U, hipMemcpyDeviceToDevice, m.stream), "rollback carry");
            check(hipMemcpyAsync(state + D, m.cscore[l].p + (g.first + kept - 1U) * D, D * 4U, hipMemcpyDeviceToDevice, m.stream), "rollback carry");
            check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(L.carry_valid[L.carry_cur].p), 1, 1, m.stream), "rollback carry");
        } else {
            check(hipMemsetAsync(state, 0, 2U * D * 4U, m.stream), "rollback carry");
            check(hipMemsetAsync(L.carry_valid[L.carry_cur].p, 0, 4U, m.stream), "rollback carry");
        }
    }
    S.hasher->truncate(g.position + kept);
    S.position = g.position + kept;
    it->count = kept;
    if (!m.compress_ring) check(hipStreamSynchronize(m.stream), "rollback");
}

std::span<const float> BatchForward::row_logits(std::uint32_t row) const {
    const auto& m = *impl_;
    if (m.logits_host.empty()) throw std::logic_error("deepseek41 batch forward: logits are kept only with slice_logits or gather_logits");
    if (row >= stats_.rows) throw std::out_of_range("deepseek41 batch forward: no such row in the last step");
    const std::size_t n = m.options.gather_logits ? m.model.config.vocab_size : m.head_slice.count;
    return {m.logits_host.data() + std::size_t{row} * n, n};
}

const std::vector<std::uint32_t>& BatchForward::step(std::span<const BatchRow> rows) {
    auto& m = *impl_;
    const auto& c = m.model.config;
    namespace attn = ds41::attn;
    const auto R = static_cast<std::uint32_t>(rows.size());
    if (R == 0U || R > m.row_capacity) throw std::invalid_argument("deepseek41 batch forward: rows exceed the configured step limit");
    // segments: the runs of equal slots; a slot may hold one run only, at consecutive positions from its position
    struct Segment {
        std::uint32_t slot, first, count, position;
    };
    std::vector<Segment> segments;
    for (std::uint32_t r = 0; r < R; ++r) {
        if (rows[r].slot >= m.slots.size()) throw std::invalid_argument("deepseek41 batch forward: no such slot");
        if (rows[r].token >= c.vocab_size) throw std::invalid_argument("deepseek41 batch forward: token outside the vocabulary");
        if (!segments.empty() && segments.back().slot == rows[r].slot) {
            ++segments.back().count;
            continue;
        }
        for (const auto& s : segments)
            if (s.slot == rows[r].slot) throw std::invalid_argument("deepseek41 batch forward: the rows of a slot must be consecutive");
        segments.push_back({rows[r].slot, r, 1U, m.slots[rows[r].slot].position});
    }
    for (const auto& s : segments)
        if (s.count > max_rows) throw std::invalid_argument("deepseek41 batch forward: one request's segment has at most 8 rows");
    for (const auto& s : segments)
        if (s.position + s.count > m.slots[s.slot].capacity) throw std::out_of_range("deepseek41 batch forward: a request's context is full");
    const std::uint32_t H = c.hidden_size, hc = c.hc_mult, K = c.num_experts_per_tok, I = c.moe_intermediate_size;
    const std::uint32_t V = c.vocab_size, W = m.row_capacity;
    hipStream_t st = m.stream;
    const auto t_step = std::chrono::steady_clock::now();
    const double comm_at_start = m.comm_seconds();
    stats_ = BatchStepStats{};
    stats_.rows = R;
    stats_.segments = static_cast<std::uint32_t>(segments.size());

    // ---- per-row inputs: token, position, RoPE rows, Engram rows (each row from its own request's history) ----
    std::vector<std::uint32_t> row_pos(R);
    for (const auto& s : segments)
        for (std::uint32_t k = 0; k < s.count; ++k) row_pos[s.first + k] = s.position + k;
    for (std::uint32_t r = 0; r < R; ++r) {
        m.token_pinned.p[r] = rows[r].token;
        m.pos_pinned.p[r] = row_pos[r];
        m.pos64_pinned.p[r] = row_pos[r];
        m.rope_pinned.p[r] = attn::rope_row(attn::RopeKind::window, row_pos[r]);
        m.rope_pinned.p[W + r] = attn::rope_row(attn::RopeKind::compressed, row_pos[r]);
        m.rope_pinned.p[2U * W + r] = attn::rope_row(attn::RopeKind::compressed, row_pos[r] / 2U * 2U);
    }
    check(hipMemcpyAsync(m.token.p, m.token_pinned.p, R * 4U, hipMemcpyHostToDevice, st), "tokens");
    check(hipMemcpyAsync(m.pos.p, m.pos_pinned.p, R * 4U, hipMemcpyHostToDevice, st), "positions");
    check(hipMemcpyAsync(m.pos64.p, m.pos64_pinned.p, R * 8U, hipMemcpyHostToDevice, st), "positions");
    check(hipMemcpyAsync(m.rope_rows.p, m.rope_pinned.p, 3U * W * sizeof(attn::RopeRow), hipMemcpyHostToDevice, st), "rope rows");
    const attn::RopeRow* rope_window = m.rope_rows.p;
    const attn::RopeRow* rope_compressed = m.rope_rows.p + W;
    const attn::RopeRow* rope_group = m.rope_rows.p + 2U * W;
    const auto& e = m.model.engram;
    const std::uint32_t cols = e.rows_per_token(), engram_layers = static_cast<std::uint32_t>(e.layer_ids.size());
    std::uint32_t engram_in_range = 0;  // Engram layers of the range (a prefix of layer_ids)
    while (engram_in_range < engram_layers && e.layer_ids[engram_in_range] < m.options.layers) ++engram_in_range;
    std::vector<std::uint64_t> engram_rows(std::size_t{R} * engram_layers * cols);  // [row][engram layer][column]
    {
        std::vector<std::uint64_t> one(std::size_t{engram_layers} * cols);
        for (std::uint32_t r = 0; r < R; ++r) {
            m.slots[rows[r].slot].hasher->push(rows[r].token, one);
            std::copy(one.begin(), one.end(), engram_rows.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * one.size()));
        }
    }
    // STRIX_DS41_ENGRAM_ASYNC (DeviceForward::step_window): layer li's rows in [row][column] order, read straight into
    // its pinned staging; Op::engram of the layer waits for them and uploads them. The previous step ended synchronised.
    // engram_io is declared last: on unwinding it joins the reads before the requests and promises they use go away.
    const bool engram_async = m.engram_async && m.engram_reader && engram_in_range != 0U;
    std::vector<std::vector<ds41::EngramRowRequest>> layer_requests(engram_async ? engram_in_range : 0U);
    std::vector<std::promise<void>> engram_ready(layer_requests.size());
    std::vector<std::future<void>> engram_wait;
    double engram_read_seconds = 0.0;
    std::future<void> engram_io;
    if (engram_async) {
        for (std::uint32_t li = 0; li < engram_in_range; ++li) {
            for (std::uint32_t r = 0; r < R; ++r)
                for (std::uint32_t k = 0; k < cols; ++k)
                    layer_requests[li].push_back({li, engram_rows[(std::size_t{r} * engram_layers + li) * cols + k]});
            engram_wait.push_back(engram_ready[li].get_future());
        }
        engram_io = std::async(std::launch::async, [&] {
            const auto t0 = std::chrono::steady_clock::now();
            std::uint32_t li = 0;
            try {
                for (; li < engram_in_range; ++li) {
                    std::uint8_t* out = m.erows_pinned.p + std::size_t{li} * R * cols * 264U;
                    m.engram_reader->read(layer_requests[li], {out, layer_requests[li].size() * 264U});
                    engram_ready[li].set_value();
                }
            } catch (...) {
                for (; li < engram_in_range; ++li) engram_ready[li].set_exception(std::current_exception());
            }
            engram_read_seconds = seconds_since(t0);
        });
    }
    if (engram_in_range != 0U && !engram_async) {
        if (m.engram_reader) {  // every Engram layer of the range, every row, in one batch from the NVMe (O_DIRECT)
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<ds41::EngramRowRequest> requests;
            for (std::uint32_t r = 0; r < R; ++r)
                for (std::uint32_t li = 0; li < engram_in_range; ++li)
                    for (std::uint32_t k = 0; k < cols; ++k) requests.push_back({li, engram_rows[(std::size_t{r} * engram_layers + li) * cols + k]});
            m.engram_bytes.resize(requests.size() * 264U);
            m.engram_reader->read(requests, m.engram_bytes);
            stats_.engram_seconds = seconds_since(t0);
        }
        for (std::uint32_t li = 0; li < engram_in_range; ++li)  // [layer][row][column]
            for (std::uint32_t r = 0; r < R; ++r) {
                std::uint8_t* out = m.erows_pinned.p + ((std::size_t{li} * R + r) * cols) * 264U;
                if (m.engram_reader) {
                    std::copy_n(m.engram_bytes.data() + (std::size_t{r} * engram_in_range + li) * cols * 264U, std::size_t{cols} * 264U, out);
                } else {
                    const auto table = m.host_tensor("blk." + std::to_string(e.layer_ids[li]) + ".engram_embd.weight");
                    for (std::uint32_t k = 0; k < cols; ++k)
                        std::copy_n(table.data() + engram_rows[(std::size_t{r} * engram_layers + li) * cols + k] * 264U, 264, out + std::size_t{k} * 264U);
                }
            }
        check(hipMemcpyAsync(m.erows.p, m.erows_pinned.p, std::size_t{engram_in_range} * R * cols * 264U, hipMemcpyHostToDevice, st), "engram rows");
    }

    if (m.options.layer_timing) stats_.layer_seconds.assign(m.options.layers, 0.0);
    stats_.union_experts.assign(m.options.layers, 0U);
    stats_.max_rows_per_expert.assign(m.options.layers, 0U);
    auto t_layer = std::chrono::steady_clock::now();
    std::int32_t timed_layer = -1;
    const auto stream_now = [&] { return m.streams[m.cur].p; };
    const bool tp = m.distributed() && m.options.tp_attention;
    const std::uint32_t heads_local = tp ? c.num_attention_heads / 2U : c.num_attention_heads;
    const std::size_t D = attn::kHeadDim;
    const auto enqueue_shared = [&](const auto& s) {
        if (const auto* gate = m.trellis_site(s.weights[0])) {  // trellis (ds41-trellis-k5-v1:shexp): F32 x, no A8
            ds41::enqueue_trellis_linear(*gate, m.x.p, m.tr_gate.p, R, H, I, false, st);
            ds41::enqueue_trellis_linear(*m.trellis_site(s.weights[1]), m.x.p, m.tr_up.p, R, H, I, false, st);
            ds41::enqueue_swiglu_f32(m.tr_gate.p, m.tr_up.p, R, I, static_cast<float>(c.swiglu_limit), m.tr_act.p, st);
            ds41::enqueue_trellis_linear(*m.trellis_site(s.weights[2]), m.tr_act.p, m.shared.p, R, I, H, false, st);
            return;
        }
        const auto& sr = m.shared_routes.at(R);
        const auto gu_fmt = m.fmt_of(s.weights[0]), down_fmt = m.fmt_of(s.weights[2]);
        const auto gate_row = row_bytes(gu_fmt, H), down_row = row_bytes(down_fmt, I);
        ds41::GateUpView gu;
        m.set_table(gu, s.weights[0]);
        gu.gate = m.dev(s.weights[0]);
        gu.up = m.dev(s.weights[1]);
        gu.expert_stride_bytes = gate_row * I;
        gu.row_bytes = gate_row;
        gu.hidden = H;
        gu.inter = I;
        gu.routes = {sr.experts.p, sr.begins.p, sr.rows.p, sr.slots.p, sr.count, sr.max_rows};
        gu.codes = m.codes.p;
        gu.deltas = m.deltas.p;
        gu.limit = static_cast<float>(c.swiglu_limit);
        gu.act_codes = m.sh_act_codes.p;
        gu.act_deltas = m.sh_act_deltas.p;
        gu.act_sums = m.sh_act_sums.p;
        ds41::enqueue_gate_up(gu_fmt, gu, st);
        ds41::RoutesProductView dv;
        m.set_table(dv, s.weights[2]);
        dv.weights = m.dev(s.weights[2]);
        dv.expert_stride_bytes = down_row * H;
        dv.row_bytes = down_row;
        dv.width = I;
        dv.outputs = H;
        dv.routes = {sr.experts.p, sr.begins.p, sr.slots.p, sr.slots.p, sr.count, sr.max_rows};
        dv.codes = m.sh_act_codes.p;
        dv.deltas = m.sh_act_deltas.p;
        dv.out = m.shared.p;
        ds41::enqueue_routes_product(down_fmt, dv, st);
    };
    const Step* deferred_shared = nullptr;
    // STRIX_DS41_HC_SITE=1 / STRIX_DS41_QA_KV=1: as DeviceForward (one enqueue_hc_site per hyper-connection boundary,
    // q_a and kv in one launch over one A8 of the attention input), same bits. HC_SITE is off with layer timing.
    static const bool hc_site_flag = [] { const char* v = std::getenv("STRIX_DS41_HC_SITE"); return !(v && v[0] == '0'); }();
    // QA_KV is OFF in BatchForward: with it every request generated one repeated token in P12 (57328), while the
    // in-batch gate compared two wrong runs. Only STRIX_DS41_BATCH_QA_KV=1 enables it,
    // for investigating the fix; DeviceForward keeps QA_KV on (gated against the CPU host forward).
    static const bool qa_kv_flag = [] { const char* v = std::getenv("STRIX_DS41_BATCH_QA_KV"); return v && v[0] == '1'; }();
    const bool fuse_hc = hc_site_flag && !m.options.layer_timing;
    const bool fuse_qa_kv = qa_kv_flag;
    // STRIX_DS41_ATTN_A8=1: the attention combine writes o_a's input A8 and o_a's product kernel o_b's, same bits.
    static const bool attn_a8 = [] { const char* v = std::getenv("STRIX_DS41_ATTN_A8"); return v && v[0] == '1'; }();
    // STRIX_DS41_BATCH_SEGMENTS (default on since P17.2, =0 off): the stateful attention kernels of every request segment
    // in one launch per kernel (enqueue_*_segments), each segment's blocks unchanged, same bits. The segment views below
    // are sized for max_rows requests: unset, it is off with STRIX_DS41_BATCH_ROWS16=1 (16 rows); =1 with ROWS16 is refused.
    static const char* const segments_env = std::getenv("STRIX_DS41_BATCH_SEGMENTS");
    static const bool batch_segments = segments_env != nullptr && segments_env[0] != '\0'
                                           ? segments_env[0] == '1'
                                           : BatchForward::configured_row_limit() <= BatchForward::max_rows;
    if (batch_segments && BatchForward::configured_row_limit() > BatchForward::max_rows)
        throw std::invalid_argument("deepseek41 batch forward: STRIX_DS41_BATCH_SEGMENTS=1 is not qualified with STRIX_DS41_BATCH_ROWS16=1");
    struct PendingPost { bool on = false; const float* sub = nullptr; const float* residual = nullptr; const ds41::HcMix* mixes = nullptr; };
    PendingPost post;  // an expansion not yet enqueued; stream_now() is already its output
    // capture_main_hidden (as DeviceForward): the input of layers [layers - 3, layers) once the expansion is stored
    const std::uint32_t capture_first = m.options.layers >= 3U ? m.options.layers - 3U : 0U;
    std::int64_t capture_at = -1;  // the layer whose input is captured once the pending expansion is stored
    const auto stored_post = [&] {
        post.on = false;
        if (capture_at >= 0) {
            check(hipMemcpyAsync(m.capture[static_cast<std::size_t>(capture_at) - capture_first].p, stream_now(), std::size_t{R} * hc * H * 4U,
                                 hipMemcpyDeviceToDevice, st),
                  "capture");
            capture_at = -1;
        }
    };
    const Step* site_mix = nullptr;
    // STRIX_DS41_LATE_HC_LAYERS (P17.5, default all; none = off; measured R1 -0.2, R6 -0.35, batch N=2..8 -0.6..-1.5 ms): on those layers, with HC_SITE and tensor-parallel attention, the attention site
    // skips its mixes (the collapse that feeds q_a uses the previous FFN site's mixes; these are first read by
    // hc_post(attn)), and they are enqueued as a mix-only site over the same stored stream right after the o_b exchange
    // starts, so they run during it. hc_post(attn) is the latest point. Same kernels and partitions, so the same bits.
    const Step* late_mix = nullptr;
    const auto enqueue_late_mix = [&] {
        if (!late_mix) return;
        ds41::HcSiteView v;
        v.rows = R;
        v.hidden = H;
        v.stream = stream_now();
        v.fn = m.devh(late_mix->weights[0]);
        v.base = m.devf(late_mix->weights[1]);
        v.scale = m.devf(late_mix->weights[2]);
        v.iters = c.hc_sinkhorn_iters;
        v.hc_eps = static_cast<float>(c.hc_eps);
        v.mix_norm_eps = 1e-20F;
        v.mixes_out = m.mix_attn.p;
        v.partial = m.mix_partial.p;
        v.counters = m.hc_counters.p;
        ds41::enqueue_hc_site(v, st);
        late_mix = nullptr;
    };
    bool x_a8 = false;  // codes/deltas/sums hold the A8 of the current x
    bool kv_done = false;
    bool o_a_trellis = false;  // an o_a ran from the trellis pack (no A8 of o_low for a Q8 o_b)
    const auto flush_post = [&] {
        if (!post.on) return;
        ds41::enqueue_hc_expand(post.sub, post.residual, post.mixes, R, H, stream_now(), st);
        stored_post();
    };
    const auto enqueue_site = [&](const Step* mix, ds41::HcMix* mixes_out, const ds41::HcMix* collapse_mixes, const float* norm_weight,
                                  float* out, std::int8_t* codes, float* deltas, std::int32_t* sums) {
        ds41::HcSiteView v;
        v.rows = R;
        v.hidden = H;
        if (post.on) {
            v.sub = post.sub;
            v.residual = post.residual;
            v.expand_mixes = post.mixes;
        }
        v.stream = stream_now();
        if (mix) {
            v.fn = m.devh(mix->weights[0]);
            v.base = m.devf(mix->weights[1]);
            v.scale = m.devf(mix->weights[2]);
            v.iters = c.hc_sinkhorn_iters;
            v.hc_eps = static_cast<float>(c.hc_eps);
            v.mix_norm_eps = 1e-20F;
            v.mixes_out = mixes_out;
            v.partial = m.mix_partial.p;
            v.counters = m.hc_counters.p;
        }
        v.collapse_mixes = collapse_mixes;
        v.norm_weight = norm_weight;
        v.norm_eps = 1e-20F;
        v.out = out;
        v.out_stride = H;
        v.codes = codes;
        v.deltas = deltas;
        v.sums = sums;
        v.a8_stride = H;
        const bool stored = post.on;
        ds41::enqueue_hc_site(v, st);
        if (stored) stored_post();
    };
    // the weights of the steps after `from` (whole tensors in step order, the last one cut), up to prefetch_bytes;
    // the routed experts (which rows are unknown) and the embedding tables are skipped
    const auto prefetch_after = [&](const Step& from) {
        if (m.prefetch_bytes == 0U) return;
        std::uint64_t left = m.prefetch_bytes;
        for (auto i = static_cast<std::size_t>(&from - m.plan.steps.data()) + 1U; i < m.plan.steps.size() && left; ++i)
            for (const auto& w : m.plan.steps[i].weights) {
                if (!left) break;
                if (w.ends_with("_exps.weight") || w.find("engram_embd") != std::string::npos || w == "token_embd.weight") continue;
                // TP: this rank's rows of q_b and wo_a (the upper or lower half); wo_b is read by K-halves of every row
                if (tp && w.ends_with("attn_output_b.weight")) continue;
                const auto& d = m.weights.at(w);
                const bool half = tp && (w.ends_with("attn_q_b.weight") || w.ends_with("attn_output_a.weight"));
                const std::uint8_t* base = d.p + (half ? d.n / 2U * m.options.rank : 0U);
                const std::uint64_t size = std::min<std::uint64_t>(half ? d.n / 2U : d.n, left);
                if (size / 16U) touch_kernel<<<320, 256, 0, st>>>(reinterpret_cast<const uint4*>(base), size / 16U, m.never.p);
                left -= size;
            }
        check(hipGetLastError(), "prefetch");
    };

    for (const auto& s : m.plan.steps) {
        const auto layer = static_cast<std::uint32_t>(std::max(s.layer, 0));
        const std::string p = "blk." + std::to_string(layer) + ".";
        if (m.options.layer_timing && s.layer >= 0 && s.layer != timed_layer) {
            timed_layer = s.layer;
            t_layer = std::chrono::steady_clock::now();
        }
        switch (s.op) {
        case Op::embed:
            ds41::enqueue_embedding(m.devh("token_embd.weight"), m.token.p, R, H, stream_now(), st);
            m.identity_pre = true;
            break;
        case Op::engram: {
            flush_post();  // the injection updates the expanded stream
            const std::size_t li = static_cast<std::size_t>(std::find(e.layer_ids.begin(), e.layer_ids.end(), layer) - e.layer_ids.begin());
            if (engram_async) {  // this layer's rows only: the later Engram layers' reads overlap the layers before them
                engram_wait.at(li).get();
                const std::size_t at = li * R * cols * 264U;
                check(hipMemcpyAsync(m.erows.p + at, m.erows_pinned.p + at, std::size_t{R} * cols * 264U, hipMemcpyHostToDevice, st),
                      "engram rows");
            }
            ds41::enqueue_engram_rows_bf16(m.erows.p + li * R * cols * 264U, R * cols, m.emb.p, st);
            ds41::F16RowsView v;
            if (m.f8.contains(p + "engram_kv.weight")) v.weight_f8 = m.dev(p + "engram_kv.weight");
            else v.weight = m.devh(p + "engram_kv.weight");
            v.width = cols * 256U;
            v.first = 0;
            v.count = H * (hc + 1U);
            v.x = m.emb.p;
            v.rows = R;
            v.x_stride = v.width;
            v.out_stride = v.count;
            v.out = m.ekv.p;
            ds41::enqueue_f16_rows(v, st);
            ds41::enqueue_engram_inject(stream_now(), m.ekv.p, m.devf(p + "engram_q_norm.weight"), m.devf(p + "engram_k_norm.weight"), R, H,
                                        1e-20F, st);
            break;
        }
        case Op::hc_mix: {
            if (fuse_hc) {  // part of the site's enqueue_hc_site at its RMSNorm
                site_mix = &s;
                break;
            }
            ds41::HcMixView v;
            v.stream = stream_now();
            v.fn = m.devh(s.weights[0]);
            v.base = m.devf(s.weights[1]);
            v.scale = m.devf(s.weights[2]);
            v.rows = R;
            v.hidden = H;
            v.iters = c.hc_sinkhorn_iters;
            v.hc_eps = static_cast<float>(c.hc_eps);
            v.norm_eps = 1e-20F;
            v.out = s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p;
            v.partial = m.mix_partial.p;
            ds41::enqueue_hc_mixes(v, st);
            break;
        }
        case Op::hc_pre: {
            if (fuse_hc) break;  // part of the site's enqueue_hc_site at its RMSNorm
            const ds41::HcMix* mixes = s.site == "attn" ? (m.identity_pre ? nullptr : m.mix_ffn.p) : m.mix_attn.p;
            ds41::enqueue_hc_collapse(stream_now(), mixes, R, H, m.x.p, st);
            break;
        }
        case Op::rms_norm: {
            if (fuse_hc && (s.site == "attn" || s.site == "ffn")) {
                const bool at = s.site == "attn";
                const ds41::HcMix* collapse = at ? (m.identity_pre ? nullptr : m.mix_ffn.p) : m.mix_attn.p;
                // not with STRIX_DS41_BATCH_PREFETCH: its touch kernel already fills the o_b exchange window (P17.5 G6: +0.6..1.2 ms)
                const bool late = at && site_mix && m.distributed() && m.options.tp_attention && m.prefetch_bytes == 0U && late_hc_layer(layer);
                enqueue_site(late ? nullptr : site_mix, at ? m.mix_attn.p : m.mix_ffn.p, collapse, m.devf(s.weights[0]), m.x.p, m.codes.p,
                             m.deltas.p, m.sums.p);
                if (late) late_mix = site_mix;
                site_mix = nullptr;
                x_a8 = true;
                break;
            }
            float* v = s.site == "q" ? m.qr.p : s.site == "kv" ? m.kv.p : m.x.p;
            const std::uint32_t n = s.site == "q" ? c.q_lora_rank : s.site == "kv" ? c.head_dim : H;
            ds41::enqueue_rms_norm(v, m.devf(s.weights[0]), R, n, n, 1e-20F, v, st);
            break;
        }
        case Op::q_a: case Op::q_b: case Op::kv: case Op::o_a: case Op::o_b: {
            if (s.op == Op::kv && kv_done) {  // enqueued with q_a
                kv_done = false;
                break;
            }
            if (!m.trellis_attn.empty() && (s.op == Op::q_b || s.op == Op::o_a || s.op == Op::o_b)) {
                // ds41-trellis-k5-v1 q_b / o_a / o_b (as DeviceForward): F32 inputs, no A8
                const auto& ta = m.trellis_attn[layer];
                const std::uint32_t q_rows = c.num_attention_heads / 2U * c.head_dim;
                const std::uint32_t o_width = c.num_attention_heads * c.head_dim / c.o_groups, o_half = c.o_groups * c.o_lora_rank / 2U;
                if (s.op == Op::q_b && !ta.q_b.empty()) {
                    const std::size_t n = ta.q_b.size() * q_rows;
                    for (std::size_t k = 0; k < ta.q_b.size(); ++k)
                        ds41::enqueue_trellis_linear(*ta.q_b[k], m.qr.p, m.q.p + k * q_rows, R, c.q_lora_rank, n, false, st);
                    break;
                }
                if (s.op == Op::o_a && !ta.o_a.empty()) {
                    const std::size_t in_width = ta.o_a.size() * o_width, out_width = ta.o_a.size() * c.o_lora_rank;
                    if (m.trellis_oa)  // the same library calls per row, one stage + one scatter (ds41_trellis_group.hpp)
                        m.trellis_oa->enqueue(ta.o_a, m.o.p, in_width, m.o_low.p, out_width, R, st);
                    else
                        for (std::size_t g = 0; g < ta.o_a.size(); ++g)
                            ds41::enqueue_trellis_linear(*ta.o_a[g], m.o.p + g * o_width, m.o_low.p + g * c.o_lora_rank, R, in_width, out_width, false,
                                                         st);
                    o_a_trellis = true;
                    break;
                }
                if (s.op == Op::o_b && !ta.o_b.empty()) {
                    if (tp) {
                        ds41::enqueue_trellis_linear(*ta.o_b[0], m.o_low.p, m.half_own.p, R, o_half, H, false, st);
                        if (m.prefetch_bytes) {
                            m.options.communicator->start(st, m.half_own.p, std::size_t{R} * H * 4U,
                                                          dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, 0U});
                            prefetch_after(s);
                            m.options.communicator->finish(st, m.half_other.p);
                        } else {
                            m.options.communicator->exchange(st, m.half_own.p, m.half_other.p, std::size_t{R} * H * 4U,
                                                             dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, 0U});
                        }
                        const float* h0 = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                        const float* h1 = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                        const std::uint32_t n = R * H;
                        add_halves<<<(n + 255U) / 256U, 256, 0, st>>>(h0, h1, m.y.p, n);
                        check(hipGetLastError(), "add_halves");
                    } else {
                        for (std::size_t k = 0; k < 2U; ++k)
                            ds41::enqueue_trellis_linear(*ta.o_b[k], m.o_low.p + k * o_half, m.y.p, R, 2U * o_half, H, k == 1U, st);
                    }
                    break;
                }
            }
            const float* in = nullptr;
            ds41::Q8A8View v;
            m.site_weight(v, s.weights[0]);
            v.first = 0;
            v.rows = R;
            std::uint32_t in_width = 0;
            const std::uint32_t tp_groups = c.o_groups / 2U;
            if (s.op == Op::q_a) { in = m.x.p; in_width = H; v.width = H; v.count = c.q_lora_rank; v.out = m.qr.p; }
            if (s.op == Op::q_b) {
                in = m.qr.p;
                in_width = c.q_lora_rank;
                v.width = c.q_lora_rank;
                v.count = heads_local * c.head_dim;
                v.first = tp ? m.options.rank * heads_local * c.head_dim : 0U;
                v.out = m.q.p;
            }
            if (s.op == Op::kv) { in = m.x.p; in_width = H; v.width = H; v.count = c.head_dim; v.out = m.kv.p; }
            if (s.op == Op::o_a) {
                in = m.o.p;
                in_width = heads_local * c.head_dim;
                v.width = c.num_attention_heads * c.head_dim / c.o_groups;
                v.count = (tp ? tp_groups : c.o_groups) * c.o_lora_rank;
                v.first = tp ? m.options.rank * tp_groups * c.o_lora_rank : 0U;
                v.per_group = c.o_lora_rank;
                v.out = m.o_low.p;
            }
            if (s.op == Op::o_b) {  // the declared K-halves h0 + h1 (DeviceForward)
                const std::uint32_t full = c.o_groups * c.o_lora_rank, half = full / 2U;
                in = m.o_low.p;
                in_width = tp ? half : full;
                if (!attn_a8 || o_a_trellis) ds41::enqueue_quantize_a8(in, in_width, R, m.site_codes.p, m.site_deltas.p, m.site_sums.p, st);
                v.width = half;
                v.row_width = full;
                v.count = H;
                v.per_group = H;
                v.x_stride = in_width;
                v.out_stride = H;
                if (tp) {
                    v.k_offset = m.options.rank * half;
                    v.codes = m.site_codes.p;
                    v.deltas = m.site_deltas.p;
                    if (const std::uint32_t ob_k = ob_split(H); ob_k > 1U) {
                        const std::uint32_t ob_c = H / ob_k;
                        const std::size_t ob_chunk = std::size_t{R} * ob_c;
                        const auto product = [&](std::uint32_t k) {
                            auto w = v;
                            w.first = k * ob_c;
                            w.count = ob_c;
                            w.per_group = ob_c;  // one input group (the check needs per_group == count with row_width)
                            w.out_stride = ob_c;
                            w.out = m.half_own.p + k * ob_chunk;
                            ds41::enqueue_q8_0_a8(w, st);
                        };
                        const auto add = [&](std::uint32_t k) {
                            const float* own = m.half_own.p + k * ob_chunk;
                            const float* other = m.half_other.p + k * ob_chunk;
                            const auto n = static_cast<std::uint32_t>(ob_chunk);
                            add_halves_cols<<<(n + 255U) / 256U, 256, 0, st>>>(m.options.rank == 0U ? own : other, m.options.rank == 0U ? other : own, m.y.p, R, ob_c,
                                                                              k * ob_c, H);
                            check(hipGetLastError(), "add_halves_cols");
                        };
                        product(0U);
                        for (std::uint32_t k = 0; k < ob_k; ++k) {
                            m.options.communicator->start(st, m.half_own.p + k * ob_chunk, ob_chunk * 4U,
                                                          dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, 0U});
                            if (k == 0U) enqueue_late_mix();
                        if (k == 0U && m.prefetch_bytes) prefetch_after(s);
                            if (k + 1U < ob_k) product(k + 1U);
                            if (k > 0U) add(k - 1U);
                            m.options.communicator->finish(st, m.half_other.p + k * ob_chunk);
                        }
                        add(ob_k - 1U);
                        break;
                    }
                    if (const auto stage = m.options.communicator->prepare_staged_ready(st, std::size_t{R} * H * 4U); stage.data) {
                        auto* own = static_cast<float*>(stage.data);
                        v.out = own;
                        ds41::enqueue_q8_0_a8(v, st);
                        publish_ready_kernel<<<1, 64, 0, st>>>(stage.flag, stage.token);
                        check(hipGetLastError(), "publish ready");
                        m.options.communicator->start_staged_ready(st, std::size_t{R} * H * 4U, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, 0U});
                        enqueue_late_mix();
                        if (m.prefetch_bytes) prefetch_after(s);
                        const auto peer = m.options.communicator->finish_staged_ready();
                        const auto* other = static_cast<const float*>(peer.data);
                        const std::uint32_t n = R * H;
                        add_halves_ready<<<(n + 255U) / 256U, 256, 0, st>>>(m.options.rank == 0U ? own : other, m.options.rank == 0U ? other : own, m.y.p, n,
                                                                         peer.flag, peer.token);
                        check(hipGetLastError(), "add_halves_ready");
                        m.options.communicator->release_staged_ready(st);
                        break;
                    }
                    if (auto* staged = static_cast<float*>(m.options.communicator->send_staging(std::size_t{R} * H * 4U))) {
                        v.out = staged;
                        ds41::enqueue_q8_0_a8(v, st);
                        m.options.communicator->start_staged(st, std::size_t{R} * H * 4U, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, 0U});
                        enqueue_late_mix();
                        if (m.prefetch_bytes) prefetch_after(s);
                        const auto* peer = static_cast<const float*>(m.options.communicator->finish_staged(st));
                        const std::uint32_t n = R * H;
                        add_halves<<<(n + 255U) / 256U, 256, 0, st>>>(m.options.rank == 0U ? staged : peer, m.options.rank == 0U ? peer : staged, m.y.p, n);
                        check(hipGetLastError(), "add_halves");
                        m.options.communicator->release_staged(st);
                        break;
                    }
                    v.out = m.half_own.p;
                    ds41::enqueue_q8_0_a8(v, st);
                    if (m.prefetch_bytes) {
                        m.options.communicator->start(st, m.half_own.p, std::size_t{R} * H * 4U,
                                                      dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, 0U});
                        enqueue_late_mix();
                        prefetch_after(s);
                        m.options.communicator->finish(st, m.half_other.p);
                    } else {
                        m.options.communicator->start(st, m.half_own.p, std::size_t{R} * H * 4U, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, 0U});
                        enqueue_late_mix();
                        m.options.communicator->finish(st, m.half_other.p);
                    }
                    const float* h0 = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                    const float* h1 = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                    const std::uint32_t n = R * H;
                    add_halves<<<(n + 255U) / 256U, 256, 0, st>>>(h0, h1, m.y.p, n);
                    check(hipGetLastError(), "add_halves");
                } else {
                    for (std::uint32_t part = 0; part < 2U; ++part) {
                        v.k_offset = part * half;
                        v.codes = m.site_codes.p + part * half;
                        v.deltas = m.site_deltas.p + part * half / 32U;
                        v.accumulate = part == 1U;
                        v.out = m.y.p;
                        ds41::enqueue_q8_0_a8(v, st);
                    }
                }
                break;
            }
            if (v.per_group == 0U) v.per_group = v.count;
            v.x_stride = in_width;
            v.out_stride = v.count;
            if (s.op == Op::q_a || s.op == Op::kv) {
                // the attention input's A8 in codes / deltas: from the hc site, or quantized once here for q_a and kv
                if (!x_a8 && fuse_qa_kv) {
                    ds41::enqueue_quantize_a8(in, in_width, R, m.codes.p, m.deltas.p, m.sums.p, st);
                    x_a8 = true;
                }
                if (x_a8) {
                    v.codes = m.codes.p;
                    v.deltas = m.deltas.p;
                    if (fuse_qa_kv && s.op == Op::q_a) {
                        ds41::Q8A8View kv;
                        m.site_weight(kv, p + "attn_kv.weight");  // packed rows (STRIX_DS41_BATCH_PACKED) as q_a's own view
                        kv.first = 0;
                        kv.rows = R;
                        kv.width = H;
                        kv.count = c.head_dim;
                        kv.per_group = kv.count;
                        kv.codes = m.codes.p;
                        kv.deltas = m.deltas.p;
                        kv.x_stride = H;
                        kv.out_stride = kv.count;
                        kv.out = m.kv.p;
                        ds41::enqueue_q8_0_a8_pair(v, kv, st);
                        kv_done = true;
                    } else {
                        ds41::enqueue_q8_0_a8(v, st);
                    }
                    break;
                }
            }
            if (attn_a8 && s.op == Op::o_a) {  // its input's A8 from the attention combine; its output's A8 for o_b
                v.codes = m.o_codes.p;
                v.deltas = m.o_deltas.p;
                v.a8_codes = m.site_codes.p;
                v.a8_deltas = m.site_deltas.p;
                v.a8_sums = m.site_sums.p;
                ds41::enqueue_q8_0_a8(v, st);
                break;
            }
            ds41::enqueue_quantize_a8(in, in_width, R, m.site_codes.p, m.site_deltas.p, m.site_sums.p, st);
            v.codes = m.site_codes.p;
            v.deltas = m.site_deltas.p;
            ds41::enqueue_q8_0_a8(v, st);
            break;
        }
        // ---- attention: the projections over every row, the stateful kernels per request segment ----
        // (STRIX_DS41_BATCH_SEGMENTS=1: every segment's call of a kernel in one launch, the same blocks and bits)
        case Op::window_write: {
            const auto& r = m.roles[layer];
            std::array<attn::hip::WindowView, max_rows> views{};
            std::size_t nv = 0;
            for (const auto& g : segments) {
                attn::hip::WindowView v{};
                v.kv = m.kv.p + g.first * D;
                v.norm = m.devf(s.weights[0]);
                v.rope = (r.rope == attn::RopeKind::window ? rope_window : rope_compressed) + g.first;
                v.pos = m.pos64.p + g.first;
                v.rows = g.count;
                v.capacity = AttentionHost::ring_capacity;
                v.ring = m.slots[g.slot].layers[layer].ring.p;
                if (batch_segments) views[nv++] = v;
                else attn::hip::enqueue_window_write(v, st);
            }
            if (nv != 0U) attn::hip::enqueue_window_write_segments(std::span(views.data(), nv), st);
            break;
        }
        case Op::compress: {
            const auto& r = m.roles[layer];
            const auto f16_proj = [&](const std::string& w, float* out) {  // compressor projection on x
                ds41::F16RowsView v;
                v.weight = m.devh(w);
                v.width = H;
                v.first = 0;
                v.count = attn::kHeadDim;
                v.x = m.x.p;
                v.rows = R;
                v.x_stride = H;
                v.out_stride = attn::kHeadDim;
                v.out = out;
                return v;
            };
            if (r.ratio == 2U)  // one launch with STRIX_DS41_F16_PAIR=1
                ds41::enqueue_f16_rows_pair(f16_proj(p + "attn_compressor_kv.weight", m.ckv[layer].p),
                                            f16_proj(p + "attn_compressor_gate.weight", m.cscore[layer].p), st);
            else ds41::enqueue_f16_rows(f16_proj(p + "attn_compressor_kv.weight", m.ckv[layer].p), st);
            std::array<attn::hip::CompressView, max_rows> views{};
            std::size_t nv = 0;
            for (const auto& g : segments) {
                auto& L = m.slots[g.slot].layers[layer];
                attn::hip::CompressView v{};
                v.ratio = r.ratio;
                v.rows = g.count;
                v.kv = m.ckv[layer].p + g.first * D;
                v.score = m.cscore[layer].p + g.first * D;
                v.norm = m.devf(p + "attn_compressor_norm.weight");
                v.wk = m.devh(p + "indexer.attn_k.weight");
                v.knorm = m.devf(p + "indexer.k_norm.weight");
                v.rope = (r.ratio == 2U ? rope_group : rope_compressed) + g.first;
                v.pos = m.pos64.p + g.first;
                v.state_in = L.carry[L.carry_cur].p;
                v.state_valid_in = L.carry_valid[L.carry_cur].p;
                v.state_out = L.carry[1 - L.carry_cur].p;
                v.state_valid_out = L.carry_valid[1 - L.carry_cur].p;
                v.kv_cache = L.kv;
                v.idx_cache = L.idx;
                if (batch_segments) views[nv++] = v;
                else attn::hip::enqueue_compress(v, st);
                L.carry_cur = 1 - L.carry_cur;
            }
            if (nv != 0U) attn::hip::enqueue_compress_segments(std::span(views.data(), nv), st);
            break;
        }
        case Op::index: {
            const auto& r = m.roles[layer];
            auto& X = m.scratch[layer];
            const std::uint32_t ratio = m.roles[static_cast<std::size_t>(r.kv_source)].ratio;
            bool any = false;
            for (const auto& g : segments) any = any || (g.position + g.count) / ratio != 0U;
            if (any) {  // the indexer's projections over every row (a row with nothing compressed ignores them)
                ds41::F16RowsView qv;
                qv.weight = m.devh(p + "indexer.attn_q_b.weight");
                qv.width = c.q_lora_rank;
                qv.first = 0;
                qv.count = attn::kIdxHeads * attn::kIdxDim;
                qv.x = m.qr.p;
                qv.rows = R;
                qv.x_stride = c.q_lora_rank;
                qv.out_stride = qv.count;
                qv.out = m.iq.p;
                ds41::F16RowsView wv;
                wv.weight = m.devh(p + "indexer.proj.weight");
                wv.width = H;
                wv.first = 0;
                wv.count = attn::kIdxHeads;
                wv.x = m.x.p;
                wv.rows = R;
                wv.x_stride = H;
                wv.out_stride = attn::kIdxHeads;
                wv.out = m.iw.p;
                ds41::enqueue_f16_rows_pair(qv, wv, st);  // one launch with STRIX_DS41_F16_PAIR=1 (F16 q_b)
            }
            std::array<attn::hip::IndexView, max_rows> ivs{};
            std::array<attn::hip::SelectView, max_rows> svs{};
            std::size_t nv = 0;
            for (const auto& g : segments) {
                const std::uint32_t count = (g.position + g.count) / ratio;  // of the segment's last row
                if (count == 0U) {
                    check(hipMemsetAsync(X.idx_count.p + g.first, 0, g.count * 4U, st), "empty top-k");
                    continue;
                }
                if ((g.position + 1U) / ratio == 0U)
                    throw std::logic_error("deepseek41 batch forward: a multi-row segment must not start before the request's first compressed entry");
                attn::hip::IndexView iv{};
                iv.rows = g.count;
                iv.ratio = ratio;
                iv.max_count = count;
                iv.q = m.iq.p + std::size_t{g.first} * attn::kIdxHeads * attn::kIdxDim;
                iv.w = m.iw.p + std::size_t{g.first} * attn::kIdxHeads;
                iv.rope = (r.rope == attn::RopeKind::window ? rope_window : rope_compressed) + g.first;
                iv.pos = m.pos64.p + g.first;
                iv.idx_cache = m.slots[g.slot].layers[static_cast<std::size_t>(r.kv_source)].idx;
                iv.scores = m.index_scores.p + std::size_t{g.first} * m.max_count;
                iv.qprep = m.index_qprep.p + std::size_t{g.first} * attn::hip::kIdxPrepBytes;
                if (!batch_segments) attn::hip::enqueue_index_scores(iv, st);
                attn::hip::SelectView sv{};
                sv.rows = g.count;
                sv.ratio = ratio;
                sv.max_count = count;
                sv.topk = attn::kTopK;
                sv.cand_blocks = attn::kCandBlocks;
                sv.block_size = attn::kCandBlockSize;
                sv.pool = r.candidate_source ? attn::Pool::candidate_source : r.candidate_user ? attn::Pool::candidate_user : attn::Pool::plain;
                sv.pos = m.pos64.p + g.first;
                sv.scores = iv.scores;
                sv.candidates = m.candidates.p + std::size_t{g.first} * m.cand_bytes;
                sv.keys = m.select_keys.p + std::size_t{g.first} * m.max_count;
                sv.idx = X.idx.p + std::size_t{g.first} * attn::kTopK;
                sv.idx_count = X.idx_count.p + g.first;
                if (batch_segments) {
                    ivs[nv] = iv;
                    svs[nv++] = sv;
                } else {
                    attn::hip::enqueue_select(sv, st);
                }
            }
            if (nv != 0U) {
                attn::hip::enqueue_index_scores_segments(std::span(ivs.data(), nv), st);
                attn::hip::enqueue_select_segments(std::span(svs.data(), nv), st);
            }
            break;
        }
        case Op::attention: {
            const auto& r = m.roles[layer];
            std::array<attn::hip::AttentionView, max_rows> views{};
            std::size_t nv = 0;
            for (const auto& g : segments) {
                const auto& S = m.slots[g.slot];
                attn::hip::AttentionView v{};
                v.rows = g.count;
                v.capacity = AttentionHost::ring_capacity;
                v.topk = attn::kTopK;
                v.q = m.q.p + std::size_t{g.first} * heads_local * D;
                v.sink = m.devf(s.weights[0]) + (tp ? m.options.rank * heads_local : 0U);
                v.rope = (r.rope == attn::RopeKind::window ? rope_window : rope_compressed) + g.first;
                v.pos = m.pos64.p + g.first;
                v.ring = S.layers[layer].ring.p;
                if (r.ratio != 0U) {
                    v.kv_cache = S.layers[static_cast<std::size_t>(r.kv_source)].kv;
                    v.idx = m.scratch[static_cast<std::size_t>(r.index_source)].idx.p + std::size_t{g.first} * attn::kTopK;
                    v.idx_count = m.scratch[static_cast<std::size_t>(r.index_source)].idx_count.p + g.first;
                }
                v.partial = m.attn_partial.p + std::size_t{g.first} * attn::kHeads * attn::kMaxChunks * (2U + D);
                v.o = m.o.p + std::size_t{g.first} * heads_local * D;
                v.tickets = m.attn_tickets.p + std::size_t{g.first} * attn::hip::kAttnGroupCount;
                v.heads = heads_local;
                if (attn_a8) {
                    v.codes = m.o_codes.p + std::size_t{g.first} * heads_local * D;
                    v.deltas = m.o_deltas.p + std::size_t{g.first} * heads_local * D / 32U;
                    v.sums = m.o_sums.p + std::size_t{g.first} * heads_local * D / 32U;
                }
                if (batch_segments) views[nv++] = v;
                else attn::hip::enqueue_attention(v, st);
            }
            if (nv != 0U) attn::hip::enqueue_attention_segments(std::span(views.data(), nv), st);
            break;
        }
        case Op::hc_post: {
            if (s.site == "attn") enqueue_late_mix();  // if the o_b path did not
            const int next = 1 - m.cur;
            x_a8 = false;
            if (fuse_hc) {  // stored by the next site (or before the Engram injection, or by the head)
                post.on = true;
                post.sub = m.y.p;
                post.residual = stream_now();
                post.mixes = s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p;
                m.cur = next;
                if (s.site == "ffn") {
                    m.identity_pre = false;
                    if (m.options.capture_main_hidden && layer + 1U >= capture_first && layer + 1U < m.options.layers) capture_at = layer + 1U;
                }
                break;
            }
            ds41::enqueue_hc_expand(m.y.p, stream_now(), s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p, R, H, m.streams[next].p, st);
            m.cur = next;
            if (s.site == "ffn") {
                m.identity_pre = false;
                if (m.options.capture_main_hidden && layer + 1U >= capture_first && layer + 1U < m.options.layers)
                    check(hipMemcpyAsync(m.capture[layer + 1U - capture_first].p, m.streams[next].p, std::size_t{R} * hc * H * 4U,
                                         hipMemcpyDeviceToDevice, st),
                          "capture");  // the input of layer + 1
                if (m.options.layer_timing) {
                    check(hipStreamSynchronize(st), "layer end");
                    stats_.layer_seconds[layer] = seconds_since(t_layer);
                }
            }
            break;
        }
        case Op::router: {
            ds41::RouterView v;
            v.x = m.x.p;
            if (m.router_bf16.contains(s.weights[0])) v.weight_bf16 = m.devh(s.weights[0]);
            else v.weight = m.devf(s.weights[0]);
            v.bias = m.devf(s.weights[1]);
            v.bias_image = m.devf(s.weights[2]);
            v.image_rows = m.image_row.p;
            v.rows = R;
            v.hidden = H;
            v.experts = c.n_routed_experts;
            v.top_k = K;
            v.route_scale = static_cast<float>(c.routed_scaling_factor);
            v.scores = m.scores.p;
            v.ids = m.ids.p;
            v.weights = m.route_w.p;
            v.tickets = m.router_tickets.p;  // STRIX_DS41_ROUTER_FAST=1: the selection in the scores launch
            ds41::enqueue_router(v, st);
            break;
        }
        case Op::shared_expert: {
            if (!x_a8) ds41::enqueue_quantize_a8(m.x.p, H, R, m.codes.p, m.deltas.p, m.sums.p, st);  // else from the hc site
            if (m.shexp_tp) {  // ds41-trellis-k5-v1:shexp-tp (as DeviceForward): partials now, added in the combine
                const auto& slices = m.trellis_shexp_tp.at(layer);
                for (std::size_t k = 0; k < slices.size(); ++k)
                    ds41::enqueue_shexp_tp_partial(slices[k], m.x.p, R, static_cast<float>(c.swiglu_limit), m.tr_gate.p, m.tr_up.p,
                                                   m.tr_act.p, k == 0U ? m.shared.p : m.tr_p1.p, st);
                break;
            }
            if (m.shared_mode == 2) {
                // rank r: gate/up of intermediate rows [I/2 r, I/2 r + I/2) (whole A8 groups, the same products); the
                // activation halves travel while the routed experts run, the down rows follow in the combine
                const std::uint32_t hI = I / 2U;
                const auto& sr = m.shared_routes.at(R);
                const auto gu_fmt = m.fmt_of(s.weights[0]);
                const auto gate_row = row_bytes(gu_fmt, H);
                ds41::GateUpView gu;
                m.set_table(gu, s.weights[0]);
                gu.gate = m.dev(s.weights[0]) + gate_row * hI * m.options.rank;
                gu.up = m.dev(s.weights[1]) + gate_row * hI * m.options.rank;
                gu.expert_stride_bytes = gate_row * I;
                gu.row_bytes = gate_row;
                gu.hidden = H;
                gu.inter = hI;
                gu.routes = {sr.experts.p, sr.begins.p, sr.rows.p, sr.slots.p, sr.count, sr.max_rows};
                gu.codes = m.codes.p;
                gu.deltas = m.deltas.p;
                gu.limit = static_cast<float>(c.swiglu_limit);
                gu.act_codes = reinterpret_cast<std::int8_t*>(m.shx_own.p);
                gu.act_deltas = reinterpret_cast<float*>(m.shx_own.p + std::size_t{R} * hI);
                gu.act_sums = m.sh_act_sums.p;
                ds41::enqueue_gate_up(gu_fmt, gu, st);
                m.options.communicator->start(st, m.shx_own.p, std::size_t{R} * hI + std::size_t{R} * hI / 32U * 4U,
                                              dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, 1U});
                break;
            }
            if (m.shared_mode == 1) {  // enqueued between the start and the finish of the half-sum exchange
                deferred_shared = &s;
                break;
            }
            enqueue_shared(s);
            break;
        }
        case Op::routed_experts: {
            // routes of this rank's experts [expert_first, expert_end) from the router's ids: the union of every row's
            // selection, each expert once, its rows in row order
            ds41::BuildRoutesView bv;
            bv.ids = m.ids.p;
            bv.rows = R;
            bv.top_k = K;
            bv.split = m.expert_tp ? c.n_routed_experts : m.plan.expert_split;  // expert TP: every slot in the rank's fold
            bv.first = m.expert_first;
            bv.end = m.expert_end;
            bv.experts = m.built.experts.p;
            bv.begins = m.built.begins.p;
            bv.rows_out = m.built.rows.p;
            bv.slots = m.built.slots.p;
            bv.order = m.built.order.p;
            bv.low_counts = m.built.low_counts.p;
            bv.counts = m.built.counts.p;
            ds41::enqueue_build_routes(bv, st);
            check(hipMemcpyAsync(m.counts_pinned.p + 2U * layer, m.built.counts.p, 8U, hipMemcpyDeviceToHost, st), "union counts");
            const ds41::RoutesView gate_routes{m.built.experts.p, m.built.begins.p, m.built.rows.p, m.built.slots.p, R * K, R, m.built.counts.p};
            const ds41::RoutesView down_routes{m.built.experts.p, m.built.begins.p, m.built.slots.p, m.built.slots.p, R * K, R, m.built.counts.p};
            // expert TP: this rank's 1152 intermediate rows of every expert and its 5-block down rows (ds41_expert_tp.hpp)
            const std::uint32_t EI = m.expert_tp ? ds41::expert_tp::half : I;
            const auto gate_row = row_bytes(ds41::WeightFormat::iq2_xxs, H);
            const std::uint64_t down_row = m.expert_tp ? ds41::expert_tp::device_down_row_bytes() : row_bytes(ds41::WeightFormat::q2_k, I);
            const std::uint64_t gate_stride = gate_row * EI, down_stride = down_row * H;
            ds41::GateUpView gu;
            gu.gate = m.dev(s.weights[0]) - gate_stride * m.expert_first;
            gu.up = m.dev(s.weights[1]) - gate_stride * m.expert_first;
            gu.expert_stride_bytes = gate_stride;
            gu.row_bytes = gate_row;
            gu.hidden = H;
            gu.inter = EI;
            gu.routes = gate_routes;
            gu.codes = m.codes.p;
            gu.deltas = m.deltas.p;
            gu.limit = static_cast<float>(c.swiglu_limit);
            gu.act_codes = m.act_codes.p;
            gu.act_deltas = m.act_deltas.p;
            gu.act_sums = m.act_sums.p;
            ds41::enqueue_gate_up(ds41::WeightFormat::iq2_xxs, gu, st);
            ds41::RoutesProductView dv;
            dv.weights = m.dev(s.weights[2]) - down_stride * m.expert_first;
            dv.expert_stride_bytes = down_stride;
            dv.row_bytes = down_row;
            dv.width = EI;
            dv.outputs = H;
            dv.routes = down_routes;
            dv.codes = m.act_codes.p;
            dv.deltas = m.act_deltas.p;
            dv.out = m.raw.p;
            if (m.expert_tp)  // this rank's K-half parts (packed half-block rows unless STRIX_DS41_EXPERT_TP_LAYOUT=dup)
                ds41::enqueue_routes_product_groups(ds41::WeightFormat::q2_k, dv, st, ds41::expert_tp::packed_from_env());
            else ds41::enqueue_routes_product(ds41::WeightFormat::q2_k, dv, st);
            break;
        }
        case Op::combine: {
            if (m.shared_mode == 2) {
                // the peer's activation half, the full activation, this rank's down rows [H/2 r, H/2 r + H/2) after its
                // half sum in the same exchange, then (shared + low) + high on both ranks
                const std::uint32_t hI = I / 2U, hH = H / 2U;
                m.options.communicator->finish(st, m.shx_other.p);
                const auto* first = m.options.rank == 0U ? m.shx_own.p : m.shx_other.p;
                const auto* second = m.options.rank == 0U ? m.shx_other.p : m.shx_own.p;
                const auto codes = [&](const std::uint8_t* b) { return reinterpret_cast<const std::int8_t*>(b); };
                const auto deltas = [&](const std::uint8_t* b) { return reinterpret_cast<const float*>(b + std::size_t{R} * hI); };
                ds41::enqueue_assemble_a8_halves(codes(first), deltas(first), codes(second), deltas(second), R, hI, m.sh_act_codes.p,
                                                 m.sh_act_deltas.p, st);
                const auto& sr = m.shared_routes.at(R);
                const std::string down_name = p + "ffn_down_shexp.weight";
                const auto down_fmt = m.fmt_of(down_name);
                const auto down_row = row_bytes(down_fmt, I);
                ds41::RoutesProductView dv;
                m.set_table(dv, down_name);
                dv.weights = m.dev(down_name) + down_row * hH * m.options.rank;
                dv.expert_stride_bytes = down_row * H;
                dv.row_bytes = down_row;
                dv.width = I;
                dv.outputs = hH;
                dv.routes = {sr.experts.p, sr.begins.p, sr.slots.p, sr.slots.p, sr.count, sr.max_rows};
                dv.codes = m.sh_act_codes.p;
                dv.deltas = m.sh_act_deltas.p;
                dv.out = m.half_own.p + std::size_t{R} * H;
                ds41::enqueue_routes_product(down_fmt, dv, st);
                ds41::HalfSumView v;
                v.raw = m.raw.p;
                v.weights = m.route_w.p;
                v.order = m.built.order.p;
                v.low_counts = m.built.low_counts.p;
                v.rows = R;
                v.top_k = K;
                v.hidden = H;
                v.half = m.expert_tp ? 0U : m.options.rank;  // expert TP: the fold of all the row's slots (low_counts = top_k)
                v.out = m.half_own.p;
                ds41::enqueue_half_sum(v, st);
                if (m.prefetch_bytes) {
                    m.options.communicator->start(st, m.half_own.p, (std::size_t{R} * H + std::size_t{R} * hH) * 4U,
                                                  dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, 0U});
                    prefetch_after(s);
                    m.options.communicator->finish(st, m.half_other.p);
                } else {
                    m.options.communicator->exchange(st, m.half_own.p, m.half_other.p, (std::size_t{R} * H + std::size_t{R} * hH) * 4U,
                                                     dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, 0U});
                }
                const float* low = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                const float* high = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                ds41::enqueue_canonical_sum_split(low + std::size_t{R} * H, high + std::size_t{R} * H, low, high, R, H, m.y.p,
                                                  m.shared.p, st);
                break;
            }
            if (m.distributed()) {
                ds41::HalfSumView v;
                v.raw = m.raw.p;
                v.weights = m.route_w.p;
                v.order = m.built.order.p;
                v.low_counts = m.built.low_counts.p;
                v.rows = R;
                v.top_k = K;
                v.hidden = H;
                v.half = m.expert_tp ? 0U : m.options.rank;  // expert TP: the fold of all the row's slots (low_counts = top_k)
                v.out = m.half_own.p;
                ds41::enqueue_half_sum(v, st);
                if (m.shexp_tp) {  // shexp-tp: S_r + P_r travels, y = (S0 + P0) + (S1 + P1)
                    ds41::enqueue_add_inplace_f32(m.half_own.p, m.shared.p, std::size_t{R} * H, st);
                    m.options.communicator->exchange(st, m.half_own.p, m.half_other.p, std::size_t{R} * H * 4U,
                                                     dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, 0U});
                    const float* low = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                    const float* high = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                    ds41::enqueue_add_f32(low, high, std::size_t{R} * H, m.y.p, st);
                    break;
                }
                if (deferred_shared) {
                    m.options.communicator->start(st, m.half_own.p, std::size_t{R} * H * 4U,
                                                  dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, 0U});
                    enqueue_shared(*deferred_shared);
                    deferred_shared = nullptr;
                    // no read-ahead here: the shared expert already fills the half-sum transfer (ds41/batch-perf 8f869e3)
                    m.options.communicator->finish(st, m.half_other.p);
                } else if (m.prefetch_bytes) {
                    m.options.communicator->start(st, m.half_own.p, std::size_t{R} * H * 4U,
                                                  dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, 0U});
                    prefetch_after(s);
                    m.options.communicator->finish(st, m.half_other.p);
                } else {
                    m.options.communicator->exchange(st, m.half_own.p, m.half_other.p, std::size_t{R} * H * 4U,
                                                     dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, 0U});
                }
                const float* low = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                const float* high = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                ds41::enqueue_canonical_sum(m.shared.p, low, high, std::size_t{R} * H, m.y.p, st);
                break;
            }
            for (std::uint32_t half = 0; half < 2U; ++half) {
                ds41::HalfSumView v;
                v.raw = m.raw.p;
                v.weights = m.route_w.p;
                v.order = m.built.order.p;
                v.low_counts = m.built.low_counts.p;
                v.rows = R;
                v.top_k = K;
                v.hidden = H;
                v.half = half;
                v.out = half == 0U ? m.low.p : m.high.p;
                ds41::enqueue_half_sum(v, st);
            }
            if (m.shexp_tp) {  // shexp-tp, both ranks here: y = (S0 + P0) + (S1 + P1)
                ds41::enqueue_add_inplace_f32(m.low.p, m.shared.p, std::size_t{R} * H, st);
                ds41::enqueue_add_inplace_f32(m.high.p, m.tr_p1.p, std::size_t{R} * H, st);
                ds41::enqueue_add_f32(m.low.p, m.high.p, std::size_t{R} * H, m.y.p, st);
                break;
            }
            ds41::enqueue_canonical_sum(m.shared.p, m.low.p, m.high.p, std::size_t{R} * H, m.y.p, st);
            break;
        }
        case Op::head: {
            if (fuse_hc) {
                enqueue_site(nullptr, nullptr, m.identity_pre ? nullptr : m.mix_ffn.p, m.devf("output_norm.weight"), m.h.p, m.site_codes.p,
                             m.site_deltas.p, m.site_sums.p);
            } else {
                ds41::enqueue_hc_collapse(stream_now(), m.identity_pre ? nullptr : m.mix_ffn.p, R, H, m.h.p, st);
                ds41::enqueue_rms_norm(m.h.p, m.devf("output_norm.weight"), R, H, H, 1e-20F, m.h.p, st);
                ds41::enqueue_quantize_a8(m.h.p, H, R, m.site_codes.p, m.site_deltas.p, m.site_sums.p, st);
            }
            if (!m.trellis_head.empty() && m.headout) {  // same rows and bits, no staging (R1 in place, R > 1 to head_rows)
                m.head_rows_valid = R > 1U;
                for (std::size_t k = 0; k < m.trellis_head.size(); ++k) {
                    const auto& [sl, site] = m.trellis_head[k];
                    float* y = m.head_rows_valid ? m.head_rows.p + k * W * sl.count : m.logits.p + sl.first;
                    ds41::enqueue_trellis_linear_rows(*site, m.h.p, y, R, H, m.head_rows_valid ? sl.count : V, W, 5U, st);
                }
                // gate runs (gather) and world-1 kept logits read [row][vocab]: the rows go to logits too
                if (m.head_rows_valid && (m.options.gather_logits || (m.options.slice_logits && !m.distributed())))
                    for (const auto& [sl, site] : m.trellis_head) {
                        const auto [rp, rs] = m.logits_rows(sl.first, V);
                        check(hipMemcpy2DAsync(m.logits.p + sl.first, std::size_t{V} * 4U, rp, rs * 4U, std::size_t{sl.count} * 4U, R,
                                               hipMemcpyDeviceToDevice, st),
                              "head rows");
                    }
            } else if (!m.trellis_head.empty()) {  // trellis head (ds41-trellis-k5-v1:head): logits[row][slice] from F32 h
                for (const auto& [sl, site] : m.trellis_head) ds41::enqueue_trellis_linear(*site, m.h.p, m.logits.p + sl.first, R, H, V, false, st);
            } else {
                ds41::Q8A8View v;
                m.site_weight(v, "output.weight");  // this rank's vocabulary slice
                v.width = H;
                v.first = 0;
                v.count = m.head_slice.count;
                v.per_group = m.head_slice.count;
                v.codes = m.site_codes.p;
                v.deltas = m.site_deltas.p;
                v.rows = R;
                v.x_stride = H;
                v.out_stride = V;
                v.out = m.logits.p + m.head_slice.first;
                ds41::enqueue_q8_0_a8(v, st);
            }
            if (m.options.gather_logits) {  // gate runs: every row's slices exchanged so both ranks hold every logit
                const auto other = ds41::vocab_slice(c.vocab_size, 2U, 1U - m.options.rank);
                for (std::uint32_t r = 0; r < R; ++r)
                    m.options.communicator->exchange(st, m.logits.p + std::size_t{r} * V + m.head_slice.first,
                                                     m.logits.p + std::size_t{r} * V + other.first, std::size_t{other.count} * 4U,
                                                     dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), r, 0U});
                check(hipMemcpyAsync(m.logits_host.data(), m.logits.p, std::size_t{R} * V * 4U, hipMemcpyDeviceToHost, st), "logits");
            } else if (m.options.slice_logits) {
                const auto [rp, rs] =
                    m.distributed() ? m.logits_rows(m.head_slice.first, V) : std::pair<const float*, std::size_t>{m.logits.p, V};
                check(hipMemcpy2DAsync(m.logits_host.data(), std::size_t{m.head_slice.count} * 4U, rp, rs * 4U,
                                       std::size_t{m.head_slice.count} * 4U, R, hipMemcpyDeviceToHost, st),
                      "logits");
            }
            break;
        }
        case Op::argmax: {
            if (m.distributed()) {
                const auto& sl = m.head_slice;
                const auto [lp, ls] = m.logits_rows(sl.first, V);
                ds41::enqueue_argmax(lp, R, sl.count, static_cast<std::uint32_t>(ls), sl.first, m.partial.p, m.cand_own.p, st);
                m.options.communicator->exchange(st, m.cand_own.p, m.cand_other.p, std::size_t{R} * sizeof(ds41::ArgmaxCandidate),
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::argmax), 0U, 0U});
                const auto* first = m.options.rank == 0U ? m.cand_own.p : m.cand_other.p;
                const auto* second = m.options.rank == 0U ? m.cand_other.p : m.cand_own.p;
                ds41::enqueue_combine_argmax(first, second, R, m.best.p, st);
            } else {
                const auto r0 = ds41::vocab_slice(c.vocab_size, 2U, 0U), r1 = ds41::vocab_slice(c.vocab_size, 2U, 1U);
                const auto [l0, s0] = m.logits_rows(r0.first, V);
                const auto [l1, s1] = m.logits_rows(r1.first, V);
                ds41::enqueue_argmax(l0, R, r0.count, static_cast<std::uint32_t>(s0), r0.first, m.partial.p, m.cand0.p, st);
                ds41::enqueue_argmax(l1, R, r1.count, static_cast<std::uint32_t>(s1), r1.first, m.partial.p, m.cand1.p, st);
                ds41::enqueue_combine_argmax(m.cand0.p, m.cand1.p, R, m.best.p, st);
            }
            check(hipMemcpyAsync(m.best_pinned.p, m.best.p, std::size_t{R} * sizeof(ds41::ArgmaxCandidate), hipMemcpyDeviceToHost, st), "tokens");
            break;
        }
        }
    }
    if (engram_async) {
        engram_io.get();
        stats_.engram_seconds = engram_read_seconds;
    }
    check(hipStreamSynchronize(st), "step");
    tokens_.resize(R);
    for (std::uint32_t r = 0; r < R; ++r) {
        if (m.best_pinned.p[r].finite == 0U) throw std::runtime_error("deepseek41 batch forward: non-finite logits");
        tokens_[r] = m.best_pinned.p[r].index;
    }
    m.last_segments.clear();
    for (const auto& g : segments) {
        m.slots[g.slot].position += g.count;
        m.last_segments.push_back({g.slot, g.first, g.count, g.position});
    }
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        stats_.union_experts[l] = m.counts_pinned.p[2U * l];
        stats_.max_rows_per_expert[l] = m.counts_pinned.p[2U * l + 1U];
        stats_.expert_bytes += std::uint64_t{stats_.union_experts[l]} * m.expert_bytes;
    }
    stats_.exchange_seconds = m.comm_seconds() - comm_at_start;
    stats_.seconds = seconds_since(t_step);
    return tokens_;
}

}  // namespace strix::deepseek41
