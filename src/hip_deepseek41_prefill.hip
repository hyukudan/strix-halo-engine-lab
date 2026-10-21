#include "strix/deepseek41_prefill.hpp"

#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_plan.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/ds41_engram.hpp"
#include "strix/ds41_expert_tp.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/ds41_prefill.hpp"
#include "strix/ds41_weight_overlay.hpp"
#include "strix/hip_ds41_attention.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix/hip_ds41_prefill.hpp"
#include "strix/hip_ds41_rn_table.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>
#include <initializer_list>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strix::deepseek41 {
namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("deepseek41 prefill: ") + what + ": " + hipGetErrorString(e));
}

std::uint64_t fnv_bytes(const void* data, std::size_t n, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

template <typename T>
struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    bool owned = true;  // false: memory of another object (PrefillOptions::shared_weights), never freed here
    Dev() = default;
    explicit Dev(std::size_t count) : n(count) {
        check(hipMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc");
    }
    static Dev borrow(const void* data, std::size_t count) {
        Dev d;
        d.p = static_cast<T*>(const_cast<void*>(data));
        d.n = count;
        d.owned = false;
        return d;
    }
    ~Dev() {
        if (p != nullptr && owned) (void)hipFree(p);
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(std::exchange(o.p, nullptr)), n(o.n), owned(o.owned) {}
    Dev& operator=(Dev&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        std::swap(owned, o.owned);
        return *this;
    }
    void zero(hipStream_t s) { check(hipMemsetAsync(p, 0, std::max<std::size_t>(n, 1) * sizeof(T), s), "memset"); }
    template <typename C>
    void upload(const C& host, hipStream_t s, std::size_t at = 0) {
        check(hipMemcpyAsync(p + at, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice, s), "upload");
    }
};

template <typename T>
std::vector<T> download(const T* p, std::size_t n, hipStream_t s) {
    std::vector<T> out(n);
    check(hipMemcpyAsync(out.data(), p, n * sizeof(T), hipMemcpyDeviceToHost, s), "download");
    check(hipStreamSynchronize(s), "download");
    return out;
}

class Mapped {
public:
    explicit Mapped(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("deepseek41 prefill: cannot open " + path.string());
        struct stat st {};
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("deepseek41 prefill: cannot stat the model");
        size_ = static_cast<std::uint64_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) throw std::runtime_error("deepseek41 prefill: mmap failed");
        base_ = static_cast<const std::uint8_t*>(p);
    }
    ~Mapped() {
        if (base_ != nullptr) ::munmap(const_cast<std::uint8_t*>(base_), size_);
        if (fd_ >= 0) ::close(fd_);
    }
    Mapped(const Mapped&) = delete;
    Mapped& operator=(const Mapped&) = delete;
    [[nodiscard]] const std::uint8_t* at(std::uint64_t offset) const { return base_ + offset; }

private:
    int fd_ = -1;
    std::uint64_t size_ = 0;
    const std::uint8_t* base_ = nullptr;
};

// Byte ranges of the model file into device memory through two pinned staging buffers, with O_DIRECT (the weights
// never enter the page cache). The copy of one buffer runs on the loader's own stream while the other is read; a
// buffer is read into again only after its copy's event, and load() returns after the last copy completed.
class DirectLoader {
public:
    static constexpr std::uint64_t align = 4096, chunk = 128ULL << 20U;
    explicit DirectLoader(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd_ < 0) fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("deepseek41 prefill: cannot open " + path.string());
        check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "loader stream");
        for (int k = 0; k < 2; ++k) {
            check(hipHostMalloc(reinterpret_cast<void**>(&buffer_[k]), chunk + 2U * align, hipHostMallocDefault), "staging buffer");
            check(hipEventCreateWithFlags(&copied_[k], hipEventDisableTiming), "staging event");
        }
    }
    ~DirectLoader() {
        if (stream_ != nullptr) (void)hipStreamSynchronize(stream_);
        for (int k = 0; k < 2; ++k) {
            if (copied_[k] != nullptr) (void)hipEventDestroy(copied_[k]);
            if (buffer_[k] != nullptr) (void)hipHostFree(buffer_[k]);
        }
        if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
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
            std::uint8_t* buffer = buffer_[next_];
            if (pending_[next_]) check(hipEventSynchronize(copied_[next_]), "staging reuse");  // its previous copy is done
            std::uint64_t got = 0;
            while (got < lead + take) {
                const auto n = ::pread(fd_, buffer + got, want - got, static_cast<off_t>(base + got));
                if (n <= 0) throw std::runtime_error("deepseek41 prefill: short read of the weights");
                got += static_cast<std::uint64_t>(n);
            }
            check(hipMemcpyAsync(device + done, buffer + lead, take, hipMemcpyHostToDevice, stream_), "weight upload");
            check(hipEventRecord(copied_[next_], stream_), "staging event");
            pending_[next_] = true;
            next_ ^= 1;
            done += take;
        }
        check(hipStreamSynchronize(stream_), "weight upload");
        pending_[0] = pending_[1] = false;
    }

private:
    int fd_ = -1;
    hipStream_t stream_ = nullptr;
    std::uint8_t* buffer_[2] = {nullptr, nullptr};
    hipEvent_t copied_[2] = {nullptr, nullptr};
    bool pending_[2] = {false, false};
    int next_ = 0;
};

std::uint64_t row_bytes(ds41::WeightFormat f, std::uint32_t width) {
    return std::uint64_t{width} / ds41::block_elements(f) * ds41::block_bytes(f);
}
double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// Prefill routes of a dense site resident on the device (the shared expert: one expert, slot = row).
struct DeviceSiteRoutes {
    Dev<std::uint32_t> begins, entry_rows, entry_slots, batch_expert, batch_begin, batch_count, batch_total;
    ds41::PrefillRoutesView view{};
    DeviceSiteRoutes(const ds41::PrefillRoutes& r, hipStream_t s)
        : begins(r.begins.size()), entry_rows(r.entry_rows.size()), entry_slots(r.entry_slots.size()), batch_expert(r.batch_expert.size()),
          batch_begin(r.batch_begin.size()), batch_count(r.batch_count.size()), batch_total(1) {
        begins.upload(r.begins, s);
        entry_rows.upload(r.entry_rows, s);
        entry_slots.upload(r.entry_slots, s);
        batch_expert.upload(r.batch_expert, s);
        batch_begin.upload(r.batch_begin, s);
        batch_count.upload(r.batch_count, s);
        batch_total.upload(std::vector<std::uint32_t>{static_cast<std::uint32_t>(r.batch_expert.size())}, s);
        view.begins = begins.p; view.entry_rows = entry_rows.p; view.entry_slots = entry_slots.p; view.batch_expert = batch_expert.p;
        view.batch_begin = batch_begin.p; view.batch_count = batch_count.p; view.batch_total = batch_total.p;
        view.max_batches = static_cast<std::uint32_t>(r.batch_expert.size());
        view.batch_rows = r.batch_rows;
        check(hipStreamSynchronize(s), "site routes");
    }
};

// FNV of the live window of a ring (slot = position % capacity): positions max(0, T - window) .. T - 1 in order. The
// slots of older positions are never read again (a decode step or verify window reads at most window - 1 rows back),
// and a prefill on the exact cone does not write them.
std::uint64_t ring_window_hash(const std::uint8_t* ring, std::uint32_t positions) {
    namespace attn = ds41::attn;
    std::uint64_t h = 14695981039346656037ULL;
    const std::uint32_t first = positions > attn::kWindow ? positions - attn::kWindow : 0U;
    for (std::uint32_t p = first; p < positions; ++p) {
        const std::uint8_t* e = ring + std::size_t{p % AttentionHost::ring_capacity} * attn::kWindowEntryBytes;
        for (std::uint32_t i = 0; i < attn::kWindowEntryBytes; ++i) h = (h ^ e[i]) * 1099511628211ULL;
    }
    return h;
}

// Row-split exchanges with several per-row fields in one message: field k of row r (words[k] 32-bit words, rows
// consecutive in its buffer) at word off_k = words[0] + ... + words[k - 1] of packed row r (stride words).
struct PackFields {
    const std::uint32_t* src[4] = {};
    std::uint32_t* dst[4] = {};
    std::uint32_t words[4] = {};
    std::uint32_t count = 0, stride = 0;
};
__global__ void pack_rows_kernel(PackFields f, std::uint32_t* __restrict__ packed) {
    const std::size_t r = blockIdx.x;
    std::uint32_t off = 0;
    for (std::uint32_t k = 0; k < f.count; ++k) {
        for (std::uint32_t i = threadIdx.x; i < f.words[k]; i += blockDim.x) packed[r * f.stride + off + i] = f.src[k][r * f.words[k] + i];
        off += f.words[k];
    }
}
__global__ void unpack_rows_kernel(PackFields f, const std::uint32_t* __restrict__ packed) {
    const std::size_t r = blockIdx.x;
    std::uint32_t off = 0;
    for (std::uint32_t k = 0; k < f.count; ++k) {
        for (std::uint32_t i = threadIdx.x; i < f.words[k]; i += blockDim.x) f.dst[k][r * f.words[k] + i] = packed[r * f.stride + off + i];
        off += f.words[k];
    }
}

__global__ void add_halves_kernel(const float* __restrict__ h0, const float* __restrict__ h1, float* __restrict__ out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = h0[i] + h1[i];
}

// The live window of the prefill ring (capacity src_capacity) into a layer's decode ring (capacity dst_capacity):
// one block per position first .. first + count - 1, the 528-byte entry as 132 words.
__global__ void ring_copy_kernel(const std::uint32_t* __restrict__ src, std::uint32_t src_capacity, std::uint32_t* __restrict__ dst,
                                 std::uint32_t dst_capacity, std::uint32_t first) {
    constexpr unsigned words = ds41::attn::kWindowEntryBytes / 4U;
    const std::uint32_t p = first + blockIdx.x;
    if (threadIdx.x < words)
        dst[std::size_t{p % dst_capacity} * words + threadIdx.x] = src[std::size_t{p % src_capacity} * words + threadIdx.x];
}

}  // namespace

struct DevicePrefill::Impl {
    PrefillOptions options;
    TargetModel model;
    Plan plan;
    std::unique_ptr<Mapped> file;
    std::map<std::string, const ManifestTensor*> tensors;
    std::map<std::string, Dev<std::uint8_t>> weights;
    std::unique_ptr<ds41::EngramRowReader> engram_reader;
    // the routed-expert MoE of every layer (one scratch; STRIX_DS41_PREFILL_MOE_PERSIST=0: one object per layer)
    std::unique_ptr<ds41::PrefillMoe> moe;
    bool moe_persist = true;
    // row_split all-gathers on the communication stream, overlapped with independent work (default;
    // STRIX_DS41_PREFILL_ASYNC_GATHER=0: synchronous exchanges on the main stream)
    bool async_gather = true;
    // row_split: per-row fields of one all-gather packed into one message (STRIX_DS41_PREFILL_PACK=0: one exchange per field)
    bool pack = true;
    Dev<std::uint32_t> pack_send, pack_recv;
    // sites run on the dense WMMA kernel instead of the batch kernel (STRIX_DS41_PREFILL_DENSE_SITES, comma list of
    // q_a, kv, q_b; default q_a,kv, faster on the dense kernel at 512 and 2048 rows; "none" = all on the batch kernel):
    // both give the decode's bits
    bool dense_q_a = true, dense_kv = true, dense_q_b = false;
    // row_split MoE: PrefillMoe::enqueue_half_joint (one route build and expert pass over the peer's and this rank's
    // rows; opt-in, STRIX_DS41_PREFILL_JOINT=1. Default: two passes, peer rows first; P11: joint -4 % at 2K, 0 at 16K)
    bool moe_joint = false;
    // stream_experts: the rank's slice (offset, bytes) of every layer's routed-expert tensors, two layer slots and the
    // thread filling the next one
    std::filesystem::path gguf;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> expert_extent;
    struct ExpertSlot {
        std::map<std::string, Dev<std::uint8_t>> kinds;  // "gate" / "up" / "down"
        std::atomic<std::int32_t> layer{-1};  // written by the prefetch thread
    };
    ExpertSlot slots[2];
    std::thread prefetch;
    std::exception_ptr prefetch_error;
    double expert_read_seconds = 0.0, expert_wait_seconds = 0.0;
    hipStream_t stream = nullptr;
    double load_seconds = 0.0;
    std::uint32_t expert_first = 0, expert_end = 0;
    ds41::VocabSlice head_slice{};
    std::array<ds41::attn::LayerAttention, 40> roles{};
    std::uint32_t rows_done = 0;  // rows of the sequence so far (every run)
    std::unique_ptr<ds41::EngramHasher> hasher;   // the sequence's Engram history (extend continues it)
    std::uint32_t pos_base = 0;                   // position of row 0 in pos64 / rope_window / rope_comp / rope_group
    // Per-run setup kept across runs (same bits; STRIX_DS41_PREFILL_OVERLAP=0: the previous per-run form): the dense-site
    // routes by (rows, groups), the token ids on the device, a pinned buffer the Engram rows are read into (on a thread,
    // while the layers before the first Engram layer run; no stream synchronisation per Engram layer).
    bool overlap = true;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::unique_ptr<DeviceSiteRoutes>> route_cache;
    Dev<std::uint32_t> dtokens;
    std::uint8_t* engram_pinned = nullptr;
    std::size_t engram_pinned_bytes = 0;
    // world 2 with row_split: runs of fewer rows than this take the tensor-parallel form instead (same bits)
    std::uint32_t split_min_rows = 0;
    // row_split MoE: runs of at most this many rows take the joint expert pass (one read of the expert half per layer
    // instead of two; same bits). 0: only STRIX_DS41_PREFILL_JOINT=1.
    std::uint32_t joint_max_rows = 0;

    // attention state per layer: ring, the KV sources' caches, the ratio-2 carry (ping-pong), the index sources'
    // selections for every row of the prompt
    struct LayerState {
        Dev<std::uint8_t> ring, kv_cache, idx_cache;
        Dev<float> carry[2];
        Dev<std::uint32_t> carry_valid[2];
        int carry_cur = 0;
        Dev<std::uint32_t> idx, idx_count;  // [rows][topk], [rows]
    };
    std::vector<LayerState> attn;
    Dev<std::uint8_t> candidates;  // per window of the candidate source: window_rows x ceil(max_count / block) bytes
    std::size_t cand_window_bytes = 0;
    // The window ring of the layer being run, one slot per position (capacity max_positions + window_rows): every
    // window K/V of the layer is written before its attention windows, which then depend only on the layer's caches and
    // run on several lanes. Its live window goes to the layer's decode ring at the end.
    Dev<std::uint8_t> prefill_ring;
    std::uint32_t prefill_capacity = 0;
    // a stream and the scratch of one attention window per lane
    struct Lane {
        hipStream_t stream = nullptr;
        hipEvent_t done = nullptr;
        Dev<float> index_scores, attn_partial;
        Dev<std::uint32_t> attn_tickets;  // W x kAttnGroupCount, zero between calls (the last chunk block combines)
        Dev<std::uint32_t> select_keys;
        Dev<std::uint8_t> index_qprep;
    };
    std::vector<Lane> lanes;
    hipEvent_t chunk_ready = nullptr;
    // world 2: the stream the MoE half sums are staged and received on, and the events ordering it with the main one
    hipStream_t comm_stream = nullptr;
    hipEvent_t half_ready[2] = {nullptr, nullptr}, half_received = nullptr;
    // row_split: producers of the gathered rows (K/V, MoE input) and their arrival, on the communication stream
    hipEvent_t gather_ready = nullptr, gathered = nullptr;
    Dev<std::uint64_t> pos64;
    Dev<ds41::attn::RopeRow> rope_window, rope_comp, rope_group;
    // positions and RoPE rows of this rank's rows (row_split; otherwise the same as the global tables)
    Dev<std::uint64_t> lpos64;
    Dev<ds41::attn::RopeRow> lrope_window, lrope_comp, lrope_group;
    // row_split: the peer's rows received (K/V for the ring and the compressor, MoE input), the peer's half sums of
    // this rank's rows, the last row's head input
    Dev<float> peer_kv, peer_ckv, peer_cscore, peer_xdeltas, peer_route_w, half_recv, h_peer;
    Dev<std::int8_t> peer_xcodes;
    Dev<std::uint32_t> peer_ids;
    std::uint32_t local_rows_done = 0;
    bool last_row_local = true;
    // row buffers (max_positions rows unless noted)
    Dev<float> streams[2];
    int cur = 0;
    bool identity_pre = true;
    Dev<ds41::HcMix> mix_attn, mix_ffn;
    Dev<float> mix_partial;
    Dev<float> x, qr, kv, ckv, cscore, iq, iw, y, shared, scores, route_w;
    Dev<std::int8_t> xcodes, qcodes;
    Dev<float> xdeltas, qdeltas;
    Dev<std::int32_t> xsums, qsums;
    Dev<std::uint32_t> ids;
    Dev<std::uint8_t> image_rows;
    // per chunk of chunk_rows rows
    Dev<float> q, o, o_low;
    Dev<std::int8_t> ocodes, olcodes;
    Dev<float> odeltas, oldeltas;
    Dev<std::int32_t> osums, olsums;
    Dev<std::uint8_t> erows;
    Dev<float> emb, ekv;
    // shared expert activations (every row)
    Dev<std::int8_t> sh_codes;
    Dev<float> sh_deltas;
    Dev<std::int32_t> sh_sums;
    // world 2: half sums of every row; TP attention: this rank's and the peer's wo_b K-half of a chunk
    Dev<float> half_own, half_other, y_own, y_other;
    // head (last row)
    Dev<float> h, logits;
    Dev<std::int8_t> hcodes;
    Dev<float> hdeltas;
    Dev<std::int32_t> hsums;
    Dev<ds41::ArgmaxCandidate> partial, cand0, cand1, best, cand_own, cand_other;

    // pos64 and the RoPE tables of rows [0, rows) at positions [base, base + rows)
    void upload_positions(std::uint32_t base, hipStream_t s) {
        namespace attn = ds41::attn;
        const std::size_t rows = pos64.n;
        std::vector<std::uint64_t> pos(rows);
        std::vector<attn::RopeRow> rw(rows), rc(rows), rg(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            const std::uint64_t p = base + r;
            pos[r] = p;
            rw[r] = attn::rope_row(attn::RopeKind::window, p);
            rc[r] = attn::rope_row(attn::RopeKind::compressed, p);
            rg[r] = attn::rope_row(attn::RopeKind::compressed, p / 2U * 2U);
        }
        pos64.upload(pos, s);
        rope_window.upload(rw, s);
        rope_comp.upload(rc, s);
        rope_group.upload(rg, s);
        if (hipStreamSynchronize(s) != hipSuccess) throw std::runtime_error("deepseek41 prefill: rope tables");
        pos_base = base;
    }
    [[nodiscard]] bool distributed() const noexcept { return options.world == 2U; }
    std::unique_ptr<ds41::WeightOverlay> overlay;  // weights_overlay (Q8_0 form)
    [[nodiscard]] const std::uint8_t* dev(const std::string& name) const { return weights.at(name).p; }
    [[nodiscard]] const float* devf(const std::string& name) const { return reinterpret_cast<const float*>(weights.at(name).p); }
    [[nodiscard]] const std::uint16_t* devh(const std::string& name) const {
        return reinterpret_cast<const std::uint16_t*>(weights.at(name).p);
    }
};

DevicePrefill::DevicePrefill(const std::filesystem::path& gguf, const PrefillOptions& requested) : impl_(std::make_unique<Impl>()) {
    const auto t_load = std::chrono::steady_clock::now();
    auto& m = *impl_;
    // the prefill keeps the expert-parallel identity: it does not implement ds41-expert-tp2-i1152-v1 (STRIX_DS41_EXPERT_TP=1)
    if (ds41::expert_tp::enabled_from_env())
        throw std::invalid_argument("deepseek41 prefill: STRIX_DS41_EXPERT_TP=1 is not implemented in DevicePrefill (decode/batch only)");
    PrefillOptions options = requested;
    // STRIX_DS41_PREFILL_WINDOW=W: rows per attention call and row-split unit (same bits for any W)
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_WINDOW"); v != nullptr && *v != 0)
        options.window_rows = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    m.options = options;
    if (options.world != 1U && options.world != 2U) throw std::invalid_argument("deepseek41 prefill: world must be 1 or 2");
    if (options.world == 2U && (options.communicator == nullptr || options.rank > 1U))
        throw std::invalid_argument("deepseek41 prefill: world 2 needs a communicator and rank 0 or 1");
    if (options.window_rows == 0U || options.window_rows > ds41::attn::kMaxCallRows || options.chunk_rows == 0U ||
        options.chunk_rows % options.window_rows != 0U)
        throw std::invalid_argument("deepseek41 prefill: window_rows 1..kMaxCallRows dividing chunk_rows");
    if (options.attention_lanes == 0U || options.attention_lanes > 16U) throw std::invalid_argument("deepseek41 prefill: attention_lanes 1..16");
    if (options.moe_chunk_rows == 0U || options.moe_chunk_rows % options.chunk_rows != 0U)
        throw std::invalid_argument("deepseek41 prefill: moe_chunk_rows a multiple of chunk_rows");
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_PERSIST"); v != nullptr && v[0] == '0') m.moe_persist = false;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_ASYNC_GATHER"); v != nullptr && v[0] == '0') m.async_gather = false;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_PACK"); v != nullptr && v[0] == '0') m.pack = false;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_JOINT"); v != nullptr && v[0] == '1') m.moe_joint = true;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_OVERLAP"); v != nullptr && v[0] == '0') m.overlap = false;
    m.split_min_rows = options.split_min_rows;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_SPLIT_MIN_ROWS"); v != nullptr && *v != 0)
        m.split_min_rows = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    m.joint_max_rows = options.joint_max_rows;
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_JOINT_MAX_ROWS"); v != nullptr && *v != 0)
        m.joint_max_rows = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_DENSE_SITES"); v != nullptr) {
        const std::string list = std::string(",") + v + ",";
        m.dense_q_a = list.find(",q_a,") != std::string::npos;
        m.dense_kv = list.find(",kv,") != std::string::npos;
        m.dense_q_b = list.find(",q_b,") != std::string::npos;
    }
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
        m.head_slice = m.distributed() ? ds41::vocab_slice(c.vocab_size, 2U, options.rank) : ds41::VocabSlice{0U, c.vocab_size};
    }
    std::set<std::string> names;
    for (const auto& s : m.plan.steps)
        for (const auto& w : s.weights)
            if (w.find("engram_embd") == std::string::npos) names.insert(w);
    m.gguf = gguf;
    {
        DirectLoader loader(gguf);
        std::unique_ptr<DirectLoader> overlay_loader;
        if (m.overlay) overlay_loader = std::make_unique<DirectLoader>(m.overlay->path());
        for (const auto& n : names) {
            const auto* t = m.tensors.at(n);
            const auto* over = m.overlay ? m.overlay->find(n) : nullptr;
            if (over != nullptr && options.stream_experts && n.ends_with("_exps.weight"))
                throw std::invalid_argument("deepseek41 prefill: streamed experts read the model file, not an overlay's experts");
            std::uint64_t offset = over != nullptr ? over->absolute_offset : t->absolute_offset, bytes = t->byte_size;
            if (n.ends_with("_exps.weight")) {
                const std::uint64_t per = bytes / c.n_routed_experts;
                offset += per * m.expert_first;
                bytes = per * (m.expert_end - m.expert_first);
                if (options.stream_experts) {
                    m.expert_extent.emplace(n, std::make_pair(offset, bytes));
                    const std::string kind = n.find("gate") != std::string::npos ? "gate" : n.find("up") != std::string::npos ? "up" : "down";
                    for (auto& slot : m.slots) {
                        auto& d = slot.kinds[kind];
                        if (d.n < bytes) d = Dev<std::uint8_t>(bytes);
                    }
                    continue;
                }
            } else if (n == "output.weight") {
                const std::uint64_t per = bytes / c.vocab_size;
                offset += per * m.head_slice.first;
                bytes = per * m.head_slice.count;
            }
            if (const auto it = options.shared_weights.find(n); it != options.shared_weights.end() && it->second.second == bytes) {
                m.weights.emplace(n, Dev<std::uint8_t>::borrow(it->second.first, bytes));  // the same bytes, read in place
                continue;
            }
            Dev<std::uint8_t> d(bytes);
            (over != nullptr ? *overlay_loader : loader).load(offset, bytes, d.p);
            m.weights.emplace(n, std::move(d));
        }
    }
    {
        std::vector<ds41::EngramTable> tables;
        for (const auto l : m.model.engram.layer_ids) {
            const auto* t = m.tensors.at("blk." + std::to_string(l) + ".engram_embd.weight");
            tables.push_back({t->absolute_offset, t->byte_size / 264U});
        }
        m.engram_reader = std::make_unique<ds41::EngramRowReader>(gguf, std::move(tables), 96U, options.engram_direct_io);
    }

    namespace attn = ds41::attn;
    // row buffers hold max_positions rows plus four windows (row_split pads an exchange to the larger rank's rows)
    const std::size_t W = options.window_rows, C = options.chunk_rows;
    const std::size_t P = options.max_positions + 4U * W;  // positions the caches hold
    if (options.max_rows > options.max_positions) throw std::invalid_argument("deepseek41 prefill: max_rows > max_positions");
    const std::size_t T = (options.max_rows != 0U ? options.max_rows : options.max_positions) + 4U * W;  // rows of one run
    m.roles = attn::released_plan();
    std::uint32_t max_count = 1;
    for (std::uint32_t l = 0; l < options.layers; ++l) {
        const auto& r = m.roles[l];
        Impl::LayerState L;
        L.ring = Dev<std::uint8_t>(std::size_t{AttentionHost::ring_capacity} * attn::kWindowEntryBytes);
        if (r.owns_kv) {
            const std::size_t entries = P / r.ratio + 1U;
            max_count = std::max<std::uint32_t>(max_count, static_cast<std::uint32_t>(entries));
            L.kv_cache = Dev<std::uint8_t>(entries * attn::kKvEntryBytes);
            L.idx_cache = Dev<std::uint8_t>(entries * attn::kIdxEntryBytes);
            for (int k = 0; k < 2; ++k) {
                L.carry[k] = Dev<float>(2U * attn::kHeadDim);
                L.carry_valid[k] = Dev<std::uint32_t>(1);
            }
        }
        if (r.owns_index) {
            L.idx = Dev<std::uint32_t>(T * attn::kTopK);
            L.idx_count = Dev<std::uint32_t>(T);
        }
        m.attn.push_back(std::move(L));
    }
    m.cand_window_bytes = W * ((P + attn::kCandBlockSize) / attn::kCandBlockSize);
    m.candidates = Dev<std::uint8_t>((T / W + 1U) * m.cand_window_bytes);
    // a run's rows plus the window before it (restored from the layer's ring by extend) never share a slot
    m.prefill_capacity = static_cast<std::uint32_t>(T + W + attn::kWindow);
    m.prefill_ring = Dev<std::uint8_t>(std::size_t{m.prefill_capacity} * attn::kWindowEntryBytes);
    m.prefill_ring.zero(m.stream);
    check(hipEventCreateWithFlags(&m.chunk_ready, hipEventDisableTiming), "event");
    check(hipStreamCreateWithFlags(&m.comm_stream, hipStreamNonBlocking), "comm stream");
    for (auto& e : m.half_ready) check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "event");
    check(hipEventCreateWithFlags(&m.half_received, hipEventDisableTiming), "event");
    check(hipEventCreateWithFlags(&m.gather_ready, hipEventDisableTiming), "event");
    check(hipEventCreateWithFlags(&m.gathered, hipEventDisableTiming), "event");
    m.lanes.resize(options.attention_lanes);
    for (auto& ln : m.lanes) {
        check(hipStreamCreateWithFlags(&ln.stream, hipStreamNonBlocking), "lane stream");
        check(hipEventCreateWithFlags(&ln.done, hipEventDisableTiming), "lane event");
        ln.index_scores = Dev<float>(W * max_count);
        ln.select_keys = Dev<std::uint32_t>(W * max_count);
        ln.index_qprep = Dev<std::uint8_t>(W * attn::hip::kIdxPrepBytes);
        ln.attn_tickets = Dev<std::uint32_t>(std::size_t{W} * attn::hip::kAttnGroupCount);
        ln.attn_tickets.zero(m.stream);
        ln.attn_partial = Dev<float>(W * attn::kHeads * attn::kMaxChunks * (2U + attn::kHeadDim));
    }
    m.pos64 = Dev<std::uint64_t>(T);
    m.dtokens = Dev<std::uint32_t>(T);
    m.rope_window = Dev<attn::RopeRow>(T);
    m.rope_comp = Dev<attn::RopeRow>(T);
    m.rope_group = Dev<attn::RopeRow>(T);
    m.lpos64 = Dev<std::uint64_t>(T);
    m.lrope_window = Dev<attn::RopeRow>(T);
    m.lrope_comp = Dev<attn::RopeRow>(T);
    m.lrope_group = Dev<attn::RopeRow>(T);
    m.upload_positions(0U, m.stream);
    m.hasher = std::make_unique<ds41::EngramHasher>(m.model.engram);
    const std::size_t H = c.hidden_size, hc = c.hc_mult, I = c.moe_intermediate_size, K = c.num_experts_per_tok;
    const std::size_t QA = c.q_lora_rank, QH = std::size_t{c.num_attention_heads} * c.head_dim, OL = std::size_t{c.o_groups} * c.o_lora_rank;
    m.streams[0] = Dev<float>(T * hc * H);
    m.streams[1] = Dev<float>(T * hc * H);
    m.mix_attn = Dev<ds41::HcMix>(T);
    m.mix_ffn = Dev<ds41::HcMix>(T);
    m.mix_partial = Dev<float>(ds41::hc_mix_partial_floats(static_cast<std::uint32_t>(T), static_cast<std::uint32_t>(H)));
    m.x = Dev<float>(T * H);
    m.xcodes = Dev<std::int8_t>(T * H);
    m.xdeltas = Dev<float>(T * H / 32U);
    m.xsums = Dev<std::int32_t>(T * H / 32U);
    m.qr = Dev<float>(T * QA);
    m.qcodes = Dev<std::int8_t>(T * QA);
    m.qdeltas = Dev<float>(T * QA / 32U);
    m.qsums = Dev<std::int32_t>(T * QA / 32U);
    m.kv = Dev<float>(T * attn::kHeadDim);
    m.ckv = Dev<float>(T * attn::kHeadDim);
    m.cscore = Dev<float>(T * attn::kHeadDim);
    m.iq = Dev<float>(T * attn::kIdxHeads * attn::kIdxDim);
    m.iw = Dev<float>(T * attn::kIdxHeads);
    m.y = Dev<float>(T * H);
    m.shared = Dev<float>(T * H);
    m.scores = Dev<float>(T * c.n_routed_experts);
    m.route_w = Dev<float>(T * K);
    m.ids = Dev<std::uint32_t>(T * K);
    m.image_rows = Dev<std::uint8_t>(T);
    m.image_rows.zero(m.stream);
    m.q = Dev<float>(C * QH);
    m.o = Dev<float>(C * QH);
    m.ocodes = Dev<std::int8_t>(C * QH);
    m.odeltas = Dev<float>(C * QH / 32U);
    m.osums = Dev<std::int32_t>(C * QH / 32U);
    m.o_low = Dev<float>(C * OL);
    m.olcodes = Dev<std::int8_t>(C * OL);
    m.oldeltas = Dev<float>(C * OL / 32U);
    m.olsums = Dev<std::int32_t>(C * OL / 32U);
    const std::size_t cols = m.model.engram.rows_per_token();
    m.erows = Dev<std::uint8_t>(C * cols * 264U);
    m.emb = Dev<float>(C * cols * 256U);
    m.ekv = Dev<float>(C * H * (hc + 1U));
    m.sh_codes = Dev<std::int8_t>(T * I);
    m.sh_deltas = Dev<float>(T * I / 32U);
    m.sh_sums = Dev<std::int32_t>(T * I / 32U);
    if (m.distributed()) {
        m.half_own = Dev<float>(T * H);
        m.half_other = Dev<float>(T * H);
        if (options.row_split) {
            m.peer_kv = Dev<float>(T * attn::kHeadDim);
            m.peer_ckv = Dev<float>(T * attn::kHeadDim);
            m.peer_cscore = Dev<float>(T * attn::kHeadDim);
            m.peer_xcodes = Dev<std::int8_t>(T * H);
            m.peer_xdeltas = Dev<float>(T * H / 32U);
            m.peer_ids = Dev<std::uint32_t>(T * K);
            m.peer_route_w = Dev<float>(T * K);
            m.half_recv = Dev<float>(T * H);
            m.h_peer = Dev<float>(H);
            // the widest packed row: window K/V + compressor kv + score (3 x 512 floats), or the MoE input
            const std::size_t pack_words = std::max<std::size_t>(3U * attn::kHeadDim, H / 4U + H / 32U + 2U * K);
            m.pack_send = Dev<std::uint32_t>(T * pack_words);
            m.pack_recv = Dev<std::uint32_t>(T * pack_words);
        }
        if (options.tp_attention) {
            m.y_own = Dev<float>(C * H);
            m.y_other = Dev<float>(C * H);
        }
    }
    m.h = Dev<float>(H);
    m.logits = Dev<float>(c.vocab_size);
    m.hcodes = Dev<std::int8_t>(H);
    m.hdeltas = Dev<float>(H / 32U);
    m.hsums = Dev<std::int32_t>(H / 32U);
    std::uint32_t partial_blocks = 0;
    for (std::uint32_t r = 0; r < 2U; ++r)
        partial_blocks = std::max(partial_blocks, ds41::argmax_partial_blocks(ds41::vocab_slice(c.vocab_size, 2U, r).count));
    m.partial = Dev<ds41::ArgmaxCandidate>(partial_blocks);
    m.cand0 = Dev<ds41::ArgmaxCandidate>(1);
    m.cand1 = Dev<ds41::ArgmaxCandidate>(1);
    m.best = Dev<ds41::ArgmaxCandidate>(1);
    m.cand_own = Dev<ds41::ArgmaxCandidate>(1);
    m.cand_other = Dev<ds41::ArgmaxCandidate>(1);
    check(hipStreamSynchronize(m.stream), "setup");
    if (options.warmup_rows != 0U) {  // one run of a dummy prompt (first launches, scratch first touch); the next run starts fresh
        const std::uint32_t n = std::min<std::uint32_t>(options.warmup_rows, options.max_rows != 0U ? options.max_rows : options.max_positions);
        std::vector<std::uint32_t> dummy(n);
        for (std::uint32_t i = 0; i < n; ++i) dummy[i] = 1000U + i;
        (void)run(dummy);
        m.rows_done = 0;
        m.local_rows_done = 0;
        m.last_row_local = true;
    }
    m.load_seconds = seconds_since(t_load);
}

DevicePrefill::~DevicePrefill() {
    if (!impl_) return;
    if (impl_->prefetch.joinable()) impl_->prefetch.join();
    for (auto& ln : impl_->lanes) {
        if (ln.done != nullptr) (void)hipEventDestroy(ln.done);
        if (ln.stream != nullptr) (void)hipStreamDestroy(ln.stream);
    }
    if (impl_->chunk_ready != nullptr) (void)hipEventDestroy(impl_->chunk_ready);
    for (auto& e : impl_->half_ready)
        if (e != nullptr) (void)hipEventDestroy(e);
    if (impl_->half_received != nullptr) (void)hipEventDestroy(impl_->half_received);
    if (impl_->gather_ready != nullptr) (void)hipEventDestroy(impl_->gather_ready);
    if (impl_->gathered != nullptr) (void)hipEventDestroy(impl_->gathered);
    if (impl_->comm_stream != nullptr) (void)hipStreamDestroy(impl_->comm_stream);
    if (impl_->stream != nullptr) (void)hipStreamDestroy(impl_->stream);
    if (impl_->engram_pinned != nullptr) (void)hipHostFree(impl_->engram_pinned);
}

std::uint64_t DevicePrefill::weight_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& [name, d] : impl_->weights) n += d.owned ? d.n : 0U;
    return n;
}
std::uint64_t DevicePrefill::shared_weight_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& [name, d] : impl_->weights) n += d.owned ? 0U : d.n;
    return n;
}
double DevicePrefill::load_seconds() const noexcept { return impl_->load_seconds; }
std::string DevicePrefill::overlay_description() const { return impl_->overlay ? impl_->overlay->describe() : std::string{}; }

namespace {

const char* expert_kind_tensor(const std::string& kind) {
    return kind == "gate" ? "ffn_gate_exps.weight" : kind == "up" ? "ffn_up_exps.weight" : "ffn_down_exps.weight";
}

}  // namespace

PrefillResult DevicePrefill::run(std::span<const std::uint32_t> tokens) { return run_from(tokens, 0U); }

PrefillResult DevicePrefill::extend(std::span<const std::uint32_t> tokens) { return run_from(tokens, impl_->rows_done); }

std::uint32_t DevicePrefill::positions() const noexcept { return impl_->rows_done; }

PrefillResult DevicePrefill::run_from(std::span<const std::uint32_t> tokens, std::uint32_t start) {
    auto& m = *impl_;
    const ds41::RnWideOff own_kernels;  // the prefill keeps its own Q8_0 x A8 kernels (P12: the multi-row form is -1 %)
    // stream_experts: layer `layer`'s experts into slot layer % 2 (on the prefetch thread or inline)
    const auto load_experts = [&m](std::uint32_t layer) {
        const auto t0 = std::chrono::steady_clock::now();
        auto& slot = m.slots[layer % 2U];
        slot.layer = -1;
        DirectLoader loader(m.gguf);
        for (auto& [kind, d] : slot.kinds) {
            const auto& [offset, bytes] = m.expert_extent.at("blk." + std::to_string(layer) + "." + expert_kind_tensor(kind));
            loader.load(offset, bytes, d.p);
        }
        slot.layer = static_cast<std::int32_t>(layer);
        m.expert_read_seconds += seconds_since(t0);
    };
    const auto start_prefetch = [&m, load_experts](std::uint32_t layer) {
        if (layer >= m.options.layers) return;
        m.prefetch = std::thread([&m, load_experts, layer] {
            try {
                load_experts(layer);
            } catch (...) {
                m.prefetch_error = std::current_exception();
            }
        });
    };
    const auto join_prefetch = [&m] {
        if (!m.prefetch.joinable()) return;
        const auto t0 = std::chrono::steady_clock::now();
        m.prefetch.join();
        m.expert_wait_seconds += seconds_since(t0);
        if (m.prefetch_error) std::rethrow_exception(std::exchange(m.prefetch_error, nullptr));
    };
    const auto expert_weight = [&m](std::uint32_t layer, const std::string& kind) -> const std::uint8_t* {
        if (!m.options.stream_experts) return m.dev("blk." + std::to_string(layer) + "." + expert_kind_tensor(kind));
        const auto& slot = m.slots[layer % 2U];
        if (slot.layer != static_cast<std::int32_t>(layer)) throw std::logic_error("deepseek41 prefill: expert slot not loaded");
        return slot.kinds.at(kind).p;
    };
    m.expert_read_seconds = m.expert_wait_seconds = 0.0;
    if (m.options.stream_experts) {
        join_prefetch();
        if (m.slots[0].layer != 0) load_experts(0U);
        if (m.slots[1].layer != 1) start_prefetch(1U);
    }
    const auto& c = m.model.config;
    namespace attn = ds41::attn;
    const std::uint32_t T = static_cast<std::uint32_t>(tokens.size());  // rows of this run, at positions [P0, P0 + T)
    const std::uint32_t P0 = start;
    const std::uint32_t max_rows = m.options.max_rows != 0U ? m.options.max_rows : m.options.max_positions;
    if (T == 0U || T > max_rows || std::uint64_t{P0} + T > m.options.max_positions)
        throw std::invalid_argument("deepseek41 prefill: rows outside 1..max_rows or positions beyond max_positions");
    if (P0 % m.options.window_rows != 0U) throw std::invalid_argument("deepseek41 prefill: extend needs the rows so far to be a multiple of window_rows");
    if (m.pos_base != P0) m.upload_positions(P0, m.stream);
    for (const auto t : tokens)
        if (t >= c.vocab_size) throw std::invalid_argument("deepseek41 prefill: token outside the vocabulary");
    const std::uint32_t H = c.hidden_size, hc = c.hc_mult, K = c.num_experts_per_tok, I = c.moe_intermediate_size;
    const std::uint32_t QA = c.q_lora_rank, QH = c.num_attention_heads * c.head_dim, OL = c.o_groups * c.o_lora_rank;
    const std::uint32_t W = m.options.window_rows, C = m.options.chunk_rows, D = attn::kHeadDim;
    hipStream_t st = m.stream;
    PrefillResult result;
    result.layers.assign(m.options.layers, PrefillLayerTiming{});
    const auto t_run = std::chrono::steady_clock::now();

    // Rows each layer computes. Without the cone every layer runs every row. With it (the full 40 layers): encoder
    // layers every row; decoder layer L its attention input from x0 (layer 20: every row, for the compressor and the
    // index keys every decoder reads), its window K/V from k0 and its whole output from o0, as plan_suffix_replay
    // gives them, rounded down to the attention window so the windows of every layer line up (the extra rows are exact).
    struct Rows { std::uint32_t x0 = 0, k0 = 0, o0 = 0; };
    std::vector<Rows> rows(m.options.layers);
    if (m.options.cone && m.options.layers == c.num_hidden_layers) {
        ds41::SuffixReplayGeometry g;
        g.layers = c.num_hidden_layers;
        g.draft_rows = m.options.draft_rows;
        const auto plan = ds41::plan_suffix_replay(g, 0U, T);
        const auto down = [&](std::uint64_t v) { return static_cast<std::uint32_t>(v / W * W); };
        for (const auto& pl : plan.layers) {
            if (pl.layer >= m.options.layers) continue;
            auto& r = rows[pl.layer];
            r.o0 = down(pl.out_begin);
            r.k0 = down(pl.kv_begin);
            r.x0 = m.roles[pl.layer].owns_kv ? 0U : r.k0;
        }
        result.row_layers = 0;
        for (const auto& r : rows) result.row_layers += T - r.o0;
    } else {
        result.row_layers = std::uint64_t{T} * m.options.layers;
    }

    // Rows of this rank. row_split (world 2): window k of W rows belongs to rank k % 2, and a rank computes its rows
    // compacted in window order ("local rows", each with its own position); otherwise local rows = global rows.
    const bool split = m.distributed() && m.options.row_split && T >= m.split_min_rows;
    const std::uint32_t me = m.options.rank, peer = 1U - me, nw = (T + W - 1U) / W;
    const auto windows_before = [&](std::uint32_t rank, std::uint32_t k) { return split ? (k + 1U - rank) / 2U : k; };
    const auto local_of = [&](std::uint32_t rank, std::uint32_t g0) { return windows_before(rank, g0 / W) * W; };  // g0: window start
    const auto rows_of = [&](std::uint32_t rank) {
        if (!split) return T;
        std::uint32_t n = windows_before(rank, nw) * W;
        if (T % W != 0U && (nw - 1U) % 2U == rank) n -= W - T % W;
        return n;
    };
    const auto window_owner = [&](std::uint32_t k) { return split ? k % 2U : me; };
    const auto owner_window = [&](std::uint32_t k) { return split ? k / 2U : k; };  // index among its owner's windows
    const auto global_window = [&](std::uint32_t j) { return split ? 2U * j + me : j; };  // of this rank's local window j
    const std::uint32_t Tn = rows_of(me), Tp = split ? rows_of(peer) : 0U;
    const bool last_local = window_owner(nw - 1U) == me;
    std::vector<std::uint32_t> global_row(Tn);
    for (std::uint32_t l = 0; l < Tn; ++l) global_row[l] = global_window(l / W) * W + l % W;
    {
        std::vector<std::uint64_t> pos(Tn);
        std::vector<attn::RopeRow> rw(Tn), rc(Tn), rg(Tn);
        for (std::uint32_t l = 0; l < Tn; ++l) {
            const std::uint64_t p = P0 + global_row[l];
            pos[l] = p;
            rw[l] = attn::rope_row(attn::RopeKind::window, p);
            rc[l] = attn::rope_row(attn::RopeKind::compressed, p);
            rg[l] = attn::rope_row(attn::RopeKind::compressed, p / 2U * 2U);
        }
        m.lpos64.upload(pos, st);
        m.lrope_window.upload(rw, st);
        m.lrope_comp.upload(rc, st);
        m.lrope_group.upload(rg, st);
    }
    // the same ranges in this rank's local rows
    std::vector<Rows> lrows(m.options.layers);
    for (std::uint32_t l = 0; l < m.options.layers; ++l)
        lrows[l] = {local_of(me, rows[l].x0), local_of(me, rows[l].k0), local_of(me, rows[l].o0)};
    result.last_row_local = last_local;
    result.local_rows = Tn;
    // row_split all-gathers on the communication stream: rows of this rank's buffer `own` to the peer and the peer's
    // rows into `dst`, both padded to the larger count, pieces of <= 16 MB. The caller records gather_ready after the
    // producers (start_gather) and enqueues independent work on the main stream before exchange_rows, which blocks the
    // CPU until the pieces have crossed; join_gather() orders the main stream after them.
    std::uint64_t exchange_seq = 0;
    const auto start_gather = [&] {
        if (!m.async_gather) return;
        check(hipEventRecord(m.gather_ready, st), "gather ready");
        check(hipStreamWaitEvent(m.comm_stream, m.gather_ready, 0), "gather ready");
    };
    const auto exchange_rows = [&](const void* own, std::uint32_t own_rows, void* dst, std::uint32_t peer_rows, std::size_t row_bytes,
                                   std::uint32_t layer) {
        const std::uint32_t n = std::max(own_rows, peer_rows);
        const auto per = static_cast<std::uint32_t>(std::max<std::size_t>(1U, (std::size_t{16} << 20U) / row_bytes));
        hipStream_t gs = m.async_gather ? m.comm_stream : st;
        for (std::uint32_t p0 = 0; p0 < n; p0 += per) {
            const std::uint32_t pn = std::min(per, n - p0);
            m.options.communicator->start(gs, static_cast<const std::uint8_t*>(own) + std::size_t{p0} * row_bytes, std::size_t{pn} * row_bytes,
                                          dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, ++exchange_seq});
            m.options.communicator->finish(gs, static_cast<std::uint8_t*>(dst) + std::size_t{p0} * row_bytes);
        }
    };
    // the same for several fields of the same rows (own buffers at this rank's first row, destinations at the peer's),
    // packed into one message per piece
    struct Field { const void* own; void* dst; std::size_t row_bytes; };
    const auto exchange_fields = [&](std::initializer_list<Field> fields, std::uint32_t own_rows, std::uint32_t peer_rows, std::uint32_t layer) {
        if (!m.pack || fields.size() == 1U) {
            for (const auto& f : fields) exchange_rows(f.own, own_rows, f.dst, peer_rows, f.row_bytes, layer);
            return;
        }
        const std::uint32_t n = std::max(own_rows, peer_rows);
        if (n == 0U) return;
        PackFields pf;
        for (const auto& f : fields) {
            pf.src[pf.count] = static_cast<const std::uint32_t*>(f.own);
            pf.dst[pf.count] = static_cast<std::uint32_t*>(f.dst);
            pf.words[pf.count] = static_cast<std::uint32_t>(f.row_bytes / 4U);
            pf.stride += pf.words[pf.count];
            ++pf.count;
        }
        // on the stream the exchange runs on (async: the communication stream, after its wait for the producers)
        hipStream_t us = m.async_gather ? m.comm_stream : st;
        hipLaunchKernelGGL(pack_rows_kernel, dim3(n), dim3(256), 0, us, pf, m.pack_send.p);
        check(hipGetLastError(), "pack rows");
        exchange_rows(m.pack_send.p, n, m.pack_recv.p, n, std::size_t{pf.stride} * 4U, layer);
        hipLaunchKernelGGL(unpack_rows_kernel, dim3(n), dim3(256), 0, us, pf, m.pack_recv.p);
        check(hipGetLastError(), "unpack rows");
    };
    const auto join_gather = [&] {
        if (!m.async_gather) return;
        check(hipEventRecord(m.gathered, m.comm_stream), "gathered");
        check(hipStreamWaitEvent(st, m.gathered, 0), "gather join");
    };

    // fresh sequence: every layer's attention state cleared (extend keeps it)
    for (std::uint32_t l = 0; P0 == 0U && l < m.options.layers; ++l) {
        auto& L = m.attn[l];
        L.ring.zero(st);
        if (L.kv_cache.p) {
            L.kv_cache.zero(st);
            L.idx_cache.zero(st);
            for (int k = 0; k < 2; ++k) {
                L.carry[k].zero(st);
                L.carry_valid[k].zero(st);
            }
            L.carry_cur = 0;
        }
        if (L.idx.p) L.idx_count.zero(st);
    }
    m.candidates.zero(st);

    // Engram rows of every token (hash of the whole prompt), read from the NVMe: now (overlap off) or on a thread
    // joined before the first Engram layer (overlap: into the pinned buffer)
    std::vector<std::uint8_t> engram_bytes;
    const std::uint8_t* engram_data = nullptr;
    const std::uint32_t engram_cols = m.model.engram.rows_per_token();
    std::vector<std::uint32_t> engram_layers;
    std::vector<ds41::EngramRowRequest> requests;
    std::future<double> engram_job;  // declared after `requests`: destroyed (joined) first
    {
        const auto t0 = std::chrono::steady_clock::now();
        if (P0 == 0U) m.hasher->reset();
        if (m.hasher->positions() != P0) throw std::logic_error("deepseek41 prefill: Engram history out of step");
        const std::uint32_t per_token = m.hasher->layers() * engram_cols;
        std::vector<std::uint64_t> ids(std::size_t{T} * per_token), one(per_token);  // [row][layer][column] of this run
        for (std::uint32_t t = 0; t < T; ++t) {
            m.hasher->push(tokens[t], one);
            std::copy(one.begin(), one.end(), ids.begin() + static_cast<std::ptrdiff_t>(std::size_t{t} * per_token));
        }
        for (std::uint32_t li = 0; li < m.model.engram.layer_ids.size(); ++li) {
            if (m.model.engram.layer_ids[li] >= m.options.layers) continue;
            engram_layers.push_back(li);
            for (std::uint32_t t = 0; t < Tn; ++t)  // this rank's rows
                for (std::uint32_t k = 0; k < engram_cols; ++k)
                    requests.push_back({li, ids[std::size_t{global_row[t]} * per_token + li * engram_cols + k]});
        }
        const std::size_t bytes = requests.size() * 264U;
        if (m.overlap) {
            if (m.engram_pinned_bytes < bytes) {
                if (m.engram_pinned != nullptr) check(hipHostFree(m.engram_pinned), "engram pinned");
                m.engram_pinned = nullptr;
                m.engram_pinned_bytes = 0;
                check(hipHostMalloc(reinterpret_cast<void**>(&m.engram_pinned), bytes, hipHostMallocDefault), "engram pinned");
                m.engram_pinned_bytes = bytes;
            }
            engram_data = m.engram_pinned;
            if (!requests.empty())
                engram_job = std::async(std::launch::async, [&m, &requests, bytes] {
                    const auto t1 = std::chrono::steady_clock::now();
                    m.engram_reader->read(requests, std::span<std::uint8_t>(m.engram_pinned, bytes));
                    return seconds_since(t1);
                });
        } else {
            engram_bytes.resize(bytes);
            if (!requests.empty()) m.engram_reader->read(requests, engram_bytes);
            engram_data = engram_bytes.data();
        }
        result.engram_seconds = seconds_since(t0);
    }
    const auto join_engram = [&] {  // overlap: the read done (its wall time into engram_seconds, the wait into engram_wait_seconds)
        if (!engram_job.valid()) return;
        const auto t0 = std::chrono::steady_clock::now();
        result.engram_seconds += engram_job.get();
        result.engram_wait_seconds = seconds_since(t0);
    };
    const auto engram_slice = [&](std::uint32_t layer) -> const std::uint8_t* {  // [token][column] x 264 B of an Engram layer
        const auto& ids = m.model.engram.layer_ids;
        const auto li = static_cast<std::uint32_t>(std::find(ids.begin(), ids.end(), layer) - ids.begin());
        const auto at = static_cast<std::size_t>(std::find(engram_layers.begin(), engram_layers.end(), li) - engram_layers.begin());
        return engram_data + at * std::size_t{Tn} * engram_cols * 264U;
    };

    Dev<std::uint32_t> run_tokens;  // overlap off: allocated per run
    const std::uint32_t* dtokens = nullptr;
    {
        std::vector<std::uint32_t> local_tokens(Tn);
        for (std::uint32_t l = 0; l < Tn; ++l) local_tokens[l] = tokens[global_row[l]];
        if (m.overlap) {
            if (Tn != 0U) m.dtokens.upload(local_tokens, st);
            dtokens = m.dtokens.p;
        } else {
            run_tokens = Dev<std::uint32_t>(Tn);
            run_tokens.upload(local_tokens, st);
            dtokens = run_tokens.p;
        }
    }
    // dense-site routes by (rows, groups): one "expert" per group, slot = row * groups + group (overlap: kept across runs)
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::unique_ptr<DeviceSiteRoutes>> run_routes;
    auto& route_cache = m.overlap ? m.route_cache : run_routes;
    const auto site_routes = [&](std::uint32_t n, std::uint32_t groups) -> const ds41::PrefillRoutesView& {
        auto& slot = route_cache[{n, groups}];
        if (!slot) slot = std::make_unique<DeviceSiteRoutes>(ds41::prefill_site_routes(n, groups, 16U), st);
        return slot->view;
    };
    const auto stream_now = [&] { return m.streams[m.cur].p; };
    const std::size_t stream_row = std::size_t{hc} * H;
    std::int32_t timed_layer = -1;
    auto t_layer = std::chrono::steady_clock::now();
    double comm_at_layer = 0.0, blocked_at_layer = 0.0, attn_s = 0.0, moe_s = 0.0;
    const auto comm_seconds = [&] { return m.distributed() ? m.options.communicator->stats().seconds : 0.0; };
    const auto blocked_seconds = [&] { return m.distributed() ? m.options.communicator->stats().finish_wait : 0.0; };
    const auto sync_time = [&](double& acc, auto&& body) {
        if (!m.options.timing) {
            body();
            return;
        }
        check(hipStreamSynchronize(st), "timing");
        const auto t0 = std::chrono::steady_clock::now();
        body();
        check(hipStreamSynchronize(st), "timing");
        acc += seconds_since(t0);
    };
    const auto close_layer = [&] {
        if (!m.options.timing || timed_layer < 0) return;
        check(hipStreamSynchronize(st), "layer end");
        result.layers[static_cast<std::size_t>(timed_layer)] = {seconds_since(t_layer), attn_s, moe_s, comm_seconds() - comm_at_layer,
                                                                 blocked_seconds() - blocked_at_layer};
    };
    auto t_head = t_run;

    for (const auto& s : m.plan.steps) {
        const auto layer = static_cast<std::uint32_t>(std::max(s.layer, 0));
        const std::string p = "blk." + std::to_string(layer) + ".";
        const Rows R0 = s.layer >= 0 ? rows[layer] : Rows{};   // global rows
        const Rows L0 = s.layer >= 0 ? lrows[layer] : Rows{};  // this rank's rows
        const std::uint32_t o0 = L0.o0, no = Tn - o0;  // this rank's output rows of the layer
        if (s.layer >= 0 && s.layer != timed_layer) {
            close_layer();
            timed_layer = s.layer;
            t_layer = std::chrono::steady_clock::now();
            comm_at_layer = comm_seconds();
            blocked_at_layer = blocked_seconds();
            attn_s = moe_s = 0.0;
        }
        if (s.op == Op::head) {
            close_layer();
            t_head = std::chrono::steady_clock::now();
        }
        switch (s.op) {
        case Op::embed:
            if (Tn != 0U) ds41::enqueue_embedding(m.devh("token_embd.weight"), dtokens, Tn, H, stream_now(), st);  // row_split: a rank may own no row
            m.identity_pre = true;
            break;
        case Op::engram: {  // encoder layers only: every row
            join_engram();
            const std::uint8_t* erow = engram_slice(layer);
            for (std::uint32_t r0 = 0; r0 < Tn; r0 += C) {
                const std::uint32_t n = std::min(C, Tn - r0);
                check(hipMemcpyAsync(m.erows.p, erow + std::size_t{r0} * engram_cols * 264U, std::size_t{n} * engram_cols * 264U,
                                     hipMemcpyHostToDevice, st), "engram rows");
                ds41::enqueue_engram_rows_bf16(m.erows.p, n * engram_cols, m.emb.p, st);
                ds41::F16RowsView v;
                v.weight = m.devh(p + "engram_kv.weight");
                v.width = engram_cols * 256U;
                v.count = H * (hc + 1U);
                v.x = m.emb.p;
                v.rows = n;
                v.x_stride = v.width;
                v.out_stride = v.count;
                v.out = m.ekv.p;
                ds41::enqueue_f16_rows(v, st);
                ds41::enqueue_engram_inject(stream_now() + std::size_t{r0} * stream_row, m.ekv.p, m.devf(p + "engram_q_norm.weight"),
                                            m.devf(p + "engram_k_norm.weight"), n, H, 1e-20F, st);
            }
            if (!m.overlap) check(hipStreamSynchronize(st), "engram");  // overlap: pinned rows, alive until the run's end
            break;
        }
        case Op::hc_mix: {
            if (no == 0U) break;  // row_split: none of this layer's output rows is this rank's
            ds41::HcMixView v;
            v.stream = stream_now() + std::size_t{o0} * stream_row;
            v.fn = m.devh(s.weights[0]);
            v.base = m.devf(s.weights[1]);
            v.scale = m.devf(s.weights[2]);
            v.rows = no;
            v.hidden = H;
            v.iters = c.hc_sinkhorn_iters;
            v.hc_eps = static_cast<float>(c.hc_eps);
            v.norm_eps = 1e-20F;
            v.out = (s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p) + o0;
            v.partial = m.mix_partial.p;
            ds41::enqueue_hc_mixes(v, st);
            break;
        }
        case Op::hc_pre: {
            const std::uint32_t r0 = s.site == "attn" ? L0.x0 : o0;
            if (r0 == Tn) break;
            const ds41::HcMix* mixes = s.site == "attn" ? (m.identity_pre ? nullptr : m.mix_ffn.p + r0) : m.mix_attn.p + r0;
            ds41::enqueue_hc_collapse(stream_now() + std::size_t{r0} * stream_row, mixes, Tn - r0, H, m.x.p + std::size_t{r0} * H, st);
            break;
        }
        case Op::rms_norm: {
            // x in place (+ its A8 for q_a / kv / the experts); qr in place + its A8 for q_b
            const bool q = s.site == "q";
            const std::uint32_t r0 = s.site == "attn" ? L0.x0 : o0, n = q ? QA : H;
            if (r0 == Tn) break;
            ds41::NormView v;
            v.x = (q ? m.qr.p : m.x.p) + std::size_t{r0} * n;
            v.x_stride = n;
            v.weight = m.devf(s.weights[0]);
            v.rows = Tn - r0;
            v.n = n;
            v.eps = 1e-20F;
            v.out = const_cast<float*>(v.x);
            v.out_stride = n;
            v.codes = (q ? m.qcodes.p : m.xcodes.p) + std::size_t{r0} * n;
            v.deltas = (q ? m.qdeltas.p : m.xdeltas.p) + std::size_t{r0} * n / 32U;
            v.sums = (q ? m.qsums.p : m.xsums.p) + std::size_t{r0} * n / 32U;
            v.a8_stride = n;
            ds41::enqueue_norm(v, st);
            break;
        }
        case Op::q_a: case Op::kv: {
            const std::uint32_t r0 = s.op == Op::q_a ? o0 : L0.k0;
            if (r0 == Tn) break;
            if (s.op == Op::q_a ? m.dense_q_a : m.dense_kv) {
                ds41::Q8A8DensePrefillView v;
                v.weight = m.dev(s.weights[0]);
                v.width = H;
                v.count = s.op == Op::q_a ? QA : D;
                v.codes = m.xcodes.p + std::size_t{r0} * H;
                v.deltas = m.xdeltas.p + std::size_t{r0} * H / 32U;
                v.rows = Tn - r0;
                v.x_stride = H;
                v.out = s.op == Op::q_a ? m.qr.p + std::size_t{r0} * QA : m.kv.p + std::size_t{r0} * D;
                v.out_stride = v.count;
                ds41::enqueue_q8_0_a8_dense_prefill(v, st);
                break;
            }
            ds41::Q8A8PrefillView v;
            v.weight = m.dev(s.weights[0]);
            v.width = H;
            v.groups = 1;
            v.per_group = s.op == Op::q_a ? QA : D;
            v.routes = site_routes(Tn - r0, 1U);
            v.codes = m.xcodes.p + std::size_t{r0} * H;
            v.deltas = m.xdeltas.p + std::size_t{r0} * H / 32U;
            v.out = (s.op == Op::q_a ? m.qr.p + std::size_t{r0} * QA : m.kv.p + std::size_t{r0} * D);
            ds41::enqueue_q8_0_a8_prefill(v, st);
            break;
        }
        case Op::q_b: case Op::window_write: case Op::compress: case Op::index: case Op::o_a: case Op::o_b:
            break;  // inside the attention step
        case Op::attention: {
            sync_time(attn_s, [&] {
                const auto& r = m.roles[layer];
                auto& L = m.attn[layer];
                // TP: this rank's heads [head0, head0 + heads) and o-groups [group0, group0 + groups)
                const bool tp = m.distributed() && m.options.tp_attention && !split;
                const std::uint32_t heads = tp ? c.num_attention_heads / 2U : c.num_attention_heads, head0 = tp ? m.options.rank * heads : 0U;
                const std::uint32_t groups = tp ? c.o_groups / 2U : c.o_groups, group0 = tp ? m.options.rank * groups : 0U;
                const std::uint32_t QL = heads * c.head_dim, OLL = groups * c.o_lora_rank;
                const auto q8_rows = [](const std::uint8_t* w, std::size_t first, std::uint32_t width) {
                    return w + first * (width / 32U) * 34U;  // Q8_0 rows: 34 bytes per 32 weights
                };
                auto t_part = std::chrono::steady_clock::now();
                const auto mark_part = [&](std::size_t k) {  // timing: the work since the previous mark into part k
                    if (!m.options.timing) return;
                    check(hipStreamSynchronize(st), "attention part");
                    for (auto& ln : m.lanes) check(hipStreamSynchronize(ln.stream), "attention part");
                    result.attention_parts[k] += seconds_since(t_part);
                    t_part = std::chrono::steady_clock::now();
                };
                const auto f16 = [&](const std::string& w, const float* in, std::uint32_t width, std::uint32_t count, float* out, std::uint32_t r0) {
                    if (r0 == Tn) return;
                    ds41::F16RowsView v;
                    v.weight = m.devh(w);
                    v.width = width;
                    v.count = count;
                    v.x = in + std::size_t{r0} * width;
                    v.rows = Tn - r0;
                    v.x_stride = width;
                    v.out_stride = count;
                    v.out = out + std::size_t{r0} * count;
                    ds41::enqueue_f16_rows(v, st);
                };
                if (r.owns_kv) {
                    f16(p + "attn_compressor_kv.weight", m.x.p, H, D, m.ckv.p, L0.x0);
                    if (r.ratio == 2U) f16(p + "attn_compressor_gate.weight", m.x.p, H, D, m.cscore.p, L0.x0);
                }
                if (split) start_gather();  // window K/V and compressor inputs are ready; the indexer runs while they cross
                if (r.owns_index) {
                    f16(p + "indexer.attn_q_b.weight", m.qr.p, QA, attn::kIdxHeads * attn::kIdxDim, m.iq.p, o0);
                    f16(p + "indexer.proj.weight", m.x.p, H, attn::kIdxHeads, m.iw.p, o0);
                }
                // ring and compressor: global window w0 (positions from w0), K/V rows from the window's owner
                const auto ring_write = [&](const float* kv, std::uint32_t w0, std::uint32_t n, const attn::RopeRow* rope) {
                    attn::hip::WindowView v{};
                    v.kv = kv;
                    v.norm = m.devf(p + "attn_kv_a_norm.weight");
                    v.rope = rope;
                    v.pos = m.pos64.p + w0;
                    v.rows = n;
                    v.capacity = m.prefill_capacity;
                    v.ring = m.prefill_ring.p;
                    attn::hip::enqueue_window_write(v, st);
                };
                const auto compress = [&](const float* ckv, const float* cscore, std::uint32_t w0, std::uint32_t n) {
                    attn::hip::CompressView v{};
                    v.ratio = r.ratio;
                    v.rows = n;
                    v.kv = ckv;
                    v.score = cscore;
                    v.norm = m.devf(p + "attn_compressor_norm.weight");
                    v.wk = m.devh(p + "indexer.attn_k.weight");
                    v.knorm = m.devf(p + "indexer.k_norm.weight");
                    v.rope = (r.ratio == 2U ? m.rope_group.p : m.rope_comp.p) + w0;
                    v.pos = m.pos64.p + w0;
                    v.state_in = L.carry[L.carry_cur].p;
                    v.state_valid_in = L.carry_valid[L.carry_cur].p;
                    v.state_out = L.carry[1 - L.carry_cur].p;
                    v.state_valid_out = L.carry_valid[1 - L.carry_cur].p;
                    v.kv_cache = L.kv_cache.p;
                    v.idx_cache = L.idx_cache.p;
                    attn::hip::enqueue_compress(v, st);
                    L.carry_cur = 1 - L.carry_cur;
                };
                // Every window K/V of the layer (rows from k0) and every compressed entry (rows from x0: below k0 only a KV
                // source in the cone, layer 20, whose entries every decoder layer reads) in row order on the main stream;
                // a row's index scores, selection and attention read only entries of rows up to its own, so the windows
                // after this are independent of each other.
                if (split) {  // the peer's window K/V (rows from k0) and compressor inputs (rows from x0)
                    const std::uint32_t pk0 = local_of(peer, R0.k0), px0 = local_of(peer, R0.x0);
                    const Field kv{m.kv.p + std::size_t{L0.k0} * D, m.peer_kv.p + std::size_t{pk0} * D, D * 4U};
                    const Field ckv{m.ckv.p + std::size_t{L0.x0} * D, m.peer_ckv.p + std::size_t{px0} * D, D * 4U};
                    const Field cscore{m.cscore.p + std::size_t{L0.x0} * D, m.peer_cscore.p + std::size_t{px0} * D, D * 4U};
                    if (!r.owns_kv) {
                        exchange_fields({kv}, Tn - L0.k0, Tp - pk0, layer);
                    } else if (R0.x0 == R0.k0) {  // the same rows for all three (every encoder layer)
                        if (r.ratio == 2U) exchange_fields({kv, ckv, cscore}, Tn - L0.x0, Tp - px0, layer);
                        else exchange_fields({kv, ckv}, Tn - L0.x0, Tp - px0, layer);
                    } else {  // a KV source in the cone (layer 20): the window K/V of fewer rows than the compressor inputs
                        exchange_fields({kv}, Tn - L0.k0, Tp - pk0, layer);
                        if (r.ratio == 2U) exchange_fields({ckv, cscore}, Tn - L0.x0, Tp - px0, layer);
                        else exchange_fields({ckv}, Tn - L0.x0, Tp - px0, layer);
                    }
                    join_gather();
                }
                if (P0 != 0U) {  // extend: the layer's last window positions before the run, into their slots of the run's ring
                    const std::uint32_t first = P0 > attn::kWindow ? P0 - attn::kWindow : 0U;
                    ring_copy_kernel<<<P0 - first, 160, 0, st>>>(reinterpret_cast<const std::uint32_t*>(L.ring.p), AttentionHost::ring_capacity,
                                                                 reinterpret_cast<std::uint32_t*>(m.prefill_ring.p), m.prefill_capacity, first);
                    check(hipGetLastError(), "ring restore");
                }
                mark_part(0);
                const attn::RopeRow* rope_rows = r.rope == attn::RopeKind::window ? m.rope_window.p : m.rope_comp.p;
                const attn::RopeRow* lrope_rows = r.rope == attn::RopeKind::window ? m.lrope_window.p : m.lrope_comp.p;
                for (std::uint32_t w0 = R0.x0; w0 < T; w0 += W) {
                    const std::uint32_t n = std::min(W, T - w0), k = w0 / W;
                    const bool mine = window_owner(k) == me;
                    const std::size_t at = std::size_t{owner_window(k)} * W * D;
                    if (w0 >= R0.k0) ring_write((mine ? m.kv.p : m.peer_kv.p) + at, w0, n, rope_rows + w0);
                    if (r.owns_kv) compress((mine ? m.ckv.p : m.peer_ckv.p) + at, (mine ? m.cscore.p : m.peer_cscore.p) + at, w0, n);
                }
                mark_part(1);
                // TP: the peer's K-half of the chunk at `pending` (if any), then y = h0 + h1 of its rows
                std::uint32_t pending = T;
                const auto close_halves = [&] {
                    if (pending == T) return;
                    const std::uint32_t n = std::min(C, Tn - pending);
                    m.options.communicator->finish(st, m.y_other.p);
                    const float* h0 = m.options.rank == 0U ? m.y_own.p : m.y_other.p;
                    const float* h1 = m.options.rank == 0U ? m.y_other.p : m.y_own.p;
                    const std::uint32_t count = n * H;
                    add_halves_kernel<<<(count + 255U) / 256U, 256, 0, st>>>(h0, h1, m.y.p + std::size_t{pending} * H, count);
                    check(hipGetLastError(), "add halves");
                    pending = T;
                };
                for (std::uint32_t r0 = o0; r0 < Tn; r0 += C) {  // this rank's rows
                    const std::uint32_t n = std::min(C, Tn - r0);
                    if (m.dense_q_b) {
                        ds41::Q8A8DensePrefillView v;
                        v.weight = q8_rows(m.dev(p + "attn_q_b.weight"), std::size_t{head0} * c.head_dim, QA);
                        v.width = QA;
                        v.count = QL;
                        v.codes = m.qcodes.p + std::size_t{r0} * QA;
                        v.deltas = m.qdeltas.p + std::size_t{r0} * QA / 32U;
                        v.rows = n;
                        v.x_stride = QA;
                        v.out = m.q.p;
                        v.out_stride = QL;
                        ds41::enqueue_q8_0_a8_dense_prefill(v, st);
                    } else {
                        ds41::Q8A8PrefillView v;
                        v.weight = q8_rows(m.dev(p + "attn_q_b.weight"), std::size_t{head0} * c.head_dim, QA);
                        v.width = QA;
                        v.groups = 1;
                        v.per_group = QL;
                        v.routes = site_routes(n, 1U);
                        v.codes = m.qcodes.p + std::size_t{r0} * QA;
                        v.deltas = m.qdeltas.p + std::size_t{r0} * QA / 32U;
                        v.out = m.q.p;
                        ds41::enqueue_q8_0_a8_prefill(v, st);
                    }
                    mark_part(2);
                    check(hipEventRecord(m.chunk_ready, st), "chunk ready");
                    for (auto& ln : m.lanes) check(hipStreamWaitEvent(ln.stream, m.chunk_ready, 0), "lane wait");
                    for (std::uint32_t w0 = r0; w0 < r0 + n; w0 += W) {
                        // local window at w0 = global window wi (positions from wi * W)
                        const std::uint32_t Rw = std::min(W, r0 + n - w0), wi = global_window(w0 / W);
                        auto& ln = m.lanes[(w0 / W) % m.lanes.size()];
                        hipStream_t ls = ln.stream;
                        const std::uint64_t* pos = m.lpos64.p + w0;
                        const attn::RopeRow* rope = lrope_rows + w0;
                        if (r.owns_index) {
                            const std::uint32_t ratio = m.roles[static_cast<std::size_t>(r.kv_source)].ratio;
                            const std::uint32_t count = (P0 + wi * W + Rw) / ratio;  // the last row's (pos + 1) / ratio
                            if (count == 0U) {
                                check(hipMemsetAsync(L.idx_count.p + w0, 0, std::size_t{Rw} * 4U, ls), "empty top-k");
                            } else {
                                attn::hip::IndexView iv{};
                                iv.rows = Rw;
                                iv.ratio = ratio;
                                iv.max_count = count;
                                iv.q = m.iq.p + std::size_t{w0} * attn::kIdxHeads * attn::kIdxDim;
                                iv.w = m.iw.p + std::size_t{w0} * attn::kIdxHeads;
                                iv.rope = rope;
                                iv.pos = pos;
                                iv.idx_cache = m.attn[static_cast<std::size_t>(r.kv_source)].idx_cache.p;
                                iv.scores = ln.index_scores.p;
                                iv.qprep = ln.index_qprep.p;
                                attn::hip::enqueue_index_scores(iv, ls);
                                attn::hip::SelectView sv{};
                                sv.rows = Rw;
                                sv.ratio = ratio;
                                sv.max_count = count;
                                sv.topk = attn::kTopK;
                                sv.cand_blocks = attn::kCandBlocks;
                                sv.block_size = attn::kCandBlockSize;
                                sv.pool = r.candidate_source ? attn::Pool::candidate_source
                                          : r.candidate_user ? attn::Pool::candidate_user : attn::Pool::plain;
                                sv.pos = pos;
                                sv.scores = ln.index_scores.p;
                                sv.candidates = m.candidates.p + std::size_t{wi} * m.cand_window_bytes;
                                sv.keys = ln.select_keys.p;
                                sv.idx = L.idx.p + std::size_t{w0} * attn::kTopK;
                                sv.idx_count = L.idx_count.p + w0;
                                attn::hip::enqueue_select(sv, ls);
                            }
                        }
                        attn::hip::AttentionView v{};
                        v.rows = Rw;
                        v.capacity = m.prefill_capacity;
                        v.topk = attn::kTopK;
                        v.heads = heads;
                        v.q = m.q.p + std::size_t{w0 - r0} * QL;
                        v.sink = m.devf(p + "attn_sinks.weight") + head0;
                        v.rope = rope;
                        v.pos = pos;
                        v.ring = m.prefill_ring.p;
                        if (r.ratio != 0U) {
                            const auto& src = m.attn[static_cast<std::size_t>(r.index_source)];
                            v.kv_cache = m.attn[static_cast<std::size_t>(r.kv_source)].kv_cache.p;
                            v.idx = src.idx.p + std::size_t{w0} * attn::kTopK;
                            v.idx_count = src.idx_count.p + w0;
                        }
                        v.partial = ln.attn_partial.p;
                        v.tickets = ln.attn_tickets.p;
                        v.o = m.o.p + std::size_t{w0 - r0} * QL;
                        attn::hip::enqueue_attention(v, ls);
                    }
                    close_halves();  // the previous chunk's wo_b halves crossed meanwhile; y_own is free again
                    for (auto& ln : m.lanes) {
                        check(hipEventRecord(ln.done, ln.stream), "lane done");
                        check(hipStreamWaitEvent(st, ln.done, 0), "chunk join");
                    }
                    mark_part(3);
                    ds41::enqueue_quantize_a8(m.o.p, QL, n, m.ocodes.p, m.odeltas.p, m.osums.p, st);
                    {
                        ds41::Q8A8PrefillView v;
                        v.width = QH / c.o_groups;
                        v.weight = q8_rows(m.dev(p + "attn_output_a.weight"), std::size_t{group0} * c.o_lora_rank, v.width);
                        v.groups = groups;
                        v.per_group = c.o_lora_rank;
                        v.routes = site_routes(n, groups);
                        v.codes = m.ocodes.p;
                        v.deltas = m.odeltas.p;
                        v.out = m.o_low.p;
                        ds41::enqueue_q8_0_a8_prefill(v, st);
                    }
                    ds41::enqueue_quantize_a8(m.o_low.p, OLL, n, m.olcodes.p, m.oldeltas.p, m.olsums.p, st);
                    {
                        // wo_b as two K-halves h0 (columns 0..4095) + h1 (4096..8191): h0 into y, then h1 accumulated
                        ds41::Q8A8DensePrefillView v;
                        v.weight = m.dev(p + "attn_output_b.weight");
                        v.width = OL / 2U;
                        v.row_width = OL;
                        v.count = H;
                        v.rows = n;
                        v.x_stride = OLL;
                        v.out = m.y.p + std::size_t{r0} * H;
                        v.out_stride = H;
                        if (tp) {
                            // this rank's K-half, exchanged; y = h0 + h1
                            v.k_offset = m.options.rank * (OL / 2U);
                            v.codes = m.olcodes.p;
                            v.deltas = m.oldeltas.p;
                            v.out = m.y_own.p;
                            ds41::enqueue_q8_0_a8_dense_prefill(v, st);
                            m.options.communicator->start(st, m.y_own.p, std::size_t{n} * H * 4U,
                                                          dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, r0 / C});
                            pending = r0;
                        } else {
                            for (std::uint32_t part = 0; part < 2U; ++part) {
                                v.k_offset = part * (OL / 2U);
                                v.codes = m.olcodes.p + part * (OL / 2U);
                                v.deltas = m.oldeltas.p + part * (OL / 2U) / 32U;
                                v.accumulate = part == 1U;
                                ds41::enqueue_q8_0_a8_dense_prefill(v, st);
                            }
                        }
                    }
                    mark_part(4);
                }
                close_halves();
                mark_part(4);
                {
                    const std::uint32_t end = P0 + T, first = end > attn::kWindow ? end - attn::kWindow : 0U;
                    ring_copy_kernel<<<end - first, 160, 0, st>>>(reinterpret_cast<const std::uint32_t*>(m.prefill_ring.p), m.prefill_capacity,
                                                                  reinterpret_cast<std::uint32_t*>(L.ring.p), AttentionHost::ring_capacity, first);
                    check(hipGetLastError(), "ring copy");
                }
            });
            break;
        }
        case Op::hc_post: {
            const int next = 1 - m.cur;
            if (no > 0U)
                ds41::enqueue_hc_expand(m.y.p + std::size_t{o0} * H, stream_now() + std::size_t{o0} * stream_row,
                                        (s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p) + o0, no, H, m.streams[next].p + std::size_t{o0} * stream_row,
                                        st);
            m.cur = next;
            if (s.site == "ffn") {
                m.identity_pre = false;
                if (m.options.stream_hashes)  // the last row's stream, on its owner (row_split: 0 on the other rank)
                    result.stream_hashes.push_back(last_local ? fnv_bytes(download(stream_now() + std::size_t{Tn - 1U} * stream_row, stream_row, st).data(),
                                                                          stream_row * 4U)
                                                              : 0U);
            }
            break;
        }
        case Op::router: {
            if (no == 0U) {
                if (m.options.routing_loads) result.routing_loads.emplace_back(c.n_routed_experts, 0U);
                if (split) start_gather();  // nothing of this rank's crosses, but the gather still pairs with the peer's
                break;
            }
            ds41::RouterView v;
            v.x = m.x.p + std::size_t{o0} * H;
            v.weight = m.devf(s.weights[0]);
            v.bias = m.devf(s.weights[1]);
            v.bias_image = m.devf(s.weights[2]);
            v.image_rows = m.image_rows.p;
            v.rows = no;
            v.hidden = H;
            v.experts = c.n_routed_experts;
            v.top_k = K;
            v.route_scale = static_cast<float>(c.routed_scaling_factor);
            v.scores = m.scores.p;
            v.ids = m.ids.p + std::size_t{o0} * K;
            v.weights = m.route_w.p + std::size_t{o0} * K;
            ds41::enqueue_prefill_router(v, st);
            if (split) start_gather();  // the MoE input of this rank's rows is ready; the shared expert runs while it crosses
            if (m.options.routing_loads) {  // per-expert entries of this layer's output rows
                const auto ids = download(m.ids.p + std::size_t{o0} * K, std::size_t{no} * K, st);
                std::vector<std::uint32_t> load(c.n_routed_experts, 0U);
                for (const auto e : ids) ++load[e];
                result.routing_loads.push_back(std::move(load));
            }
            break;
        }
        case Op::shared_expert: {
            if (no == 0U) break;
            sync_time(moe_s, [&] {
                const auto gate_row = row_bytes(ds41::WeightFormat::q8_0, H), down_row = row_bytes(ds41::WeightFormat::q8_0, I);
                const auto& routes = site_routes(no, 1U);
                ds41::PrefillGateUpView gu;
                gu.gate = m.dev(s.weights[0]);
                gu.up = m.dev(s.weights[1]);
                gu.expert_stride_bytes = gate_row * I;
                gu.row_bytes = gate_row;
                gu.hidden = H;
                gu.inter = I;
                gu.routes = routes;
                gu.codes = m.xcodes.p + std::size_t{o0} * H;
                gu.deltas = m.xdeltas.p + std::size_t{o0} * H / 32U;
                gu.limit = static_cast<float>(c.swiglu_limit);
                gu.act_codes = m.sh_codes.p;
                gu.act_deltas = m.sh_deltas.p;
                gu.act_sums = m.sh_sums.p;
                ds41::enqueue_prefill_gate_up(ds41::WeightFormat::q8_0, gu, st);
                ds41::PrefillDownView dv;
                dv.weights = m.dev(s.weights[2]);
                dv.expert_stride_bytes = down_row * H;
                dv.row_bytes = down_row;
                dv.width = I;
                dv.outputs = H;
                dv.routes = routes;
                dv.codes = m.sh_codes.p;
                dv.deltas = m.sh_deltas.p;
                dv.out = m.shared.p + std::size_t{o0} * H;
                ds41::enqueue_prefill_down(ds41::WeightFormat::q8_0, dv, st);
            });
            break;
        }
        case Op::routed_experts:
            break;  // with the combine, chunk by chunk
        case Op::combine: {
            sync_time(moe_s, [&] {
                ds41::PrefillMoeWeights w;
                const auto gate_row = row_bytes(ds41::WeightFormat::iq2_xxs, H), down_row = row_bytes(ds41::WeightFormat::q2_k, I);
                if (m.options.stream_experts) {
                    // this layer's slot was filled meanwhile; the other slot (layer - 1, finished) takes layer + 1
                    if (m.slots[layer % 2U].layer != static_cast<std::int32_t>(layer)) join_prefetch();
                    if (m.slots[layer % 2U].layer != static_cast<std::int32_t>(layer)) load_experts(layer);
                    join_prefetch();
                    start_prefetch(layer + 1U);
                }
                w.gate = expert_weight(layer, "gate");
                w.up = expert_weight(layer, "up");
                w.down = expert_weight(layer, "down");
                w.gate_row_bytes = gate_row;
                w.down_row_bytes = down_row;
                w.hidden = H;
                w.inter = I;
                w.limit = static_cast<float>(c.swiglu_limit);
                const std::uint32_t MC = m.options.moe_chunk_rows;
                std::unique_ptr<ds41::PrefillMoe> layer_moe;
                if (!m.moe_persist || !m.moe) {
                    layer_moe = std::make_unique<ds41::PrefillMoe>(w, K, c.n_routed_experts, m.plan.expert_split, MC,
                                                                   m.distributed() ? m.expert_first : 0U, m.distributed() ? m.expert_end : 0U);
                    if (m.moe_persist) m.moe = std::move(layer_moe);
                } else {
                    m.moe->set_weights(w);  // the previous layer's work on the stream keeps its own weights
                }
                ds41::PrefillMoe& moe = layer_moe ? *layer_moe : *m.moe;
                ds41::PrefillMoeInputs in;
                in.rows = no;
                in.ids = m.ids.p + std::size_t{o0} * K;
                in.weights = m.route_w.p + std::size_t{o0} * K;
                in.codes = m.xcodes.p + std::size_t{o0} * H;
                in.deltas = m.xdeltas.p + std::size_t{o0} * H / 32U;
                in.shared = m.shared.p + std::size_t{o0} * H;
                float* y = m.y.p + std::size_t{o0} * H;
                float* own = m.half_own.p + std::size_t{o0} * H;
                float* other = m.half_other.p + std::size_t{o0} * H;
                // world 2: chunk ch's half sums cross in pieces of C rows (messages <= 21 MB) on the communication stream,
                // issued while chunk ch + 1's half runs on the main stream; then y = (shared + low) + high of chunk ch
                const auto exchange_chunk = [&](std::uint32_t ch) {
                    const std::uint32_t r0 = ch * MC, n = std::min(MC, no - r0);
                    hipStream_t cs = m.comm_stream;
                    check(hipStreamWaitEvent(cs, m.half_ready[ch % 2U], 0), "half ready");
                    for (std::uint32_t p0 = 0; p0 < n; p0 += C) {
                        const std::uint32_t pn = std::min(C, n - p0);
                        const std::size_t at = std::size_t{r0 + p0} * H;
                        m.options.communicator->start(cs, own + at, std::size_t{pn} * H * 4U,
                                                      dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, (r0 + p0) / C});
                        m.options.communicator->finish(cs, other + at);
                    }
                    check(hipEventRecord(m.half_received, cs), "half received");
                    check(hipStreamWaitEvent(st, m.half_received, 0), "half join");
                    const float* low = (m.options.rank == 0U ? own : other) + std::size_t{r0} * H;
                    const float* high = (m.options.rank == 0U ? other : own) + std::size_t{r0} * H;
                    ds41::enqueue_canonical_sum(in.shared + std::size_t{r0} * H, low, high, std::size_t{n} * H, y + std::size_t{r0} * H, st);
                };
                if (split) {
                    // the MoE input of the peer's output rows; this rank's half of the peer's rows first, sent while this
                    // rank's own rows run; the peer's half of this rank's rows comes back; y = (shared + low) + high
                    const std::uint32_t p0 = local_of(peer, R0.o0), np = Tp - p0;
                    exchange_fields({Field{m.xcodes.p + std::size_t{o0} * H, m.peer_xcodes.p + std::size_t{p0} * H, H},
                                     Field{m.xdeltas.p + std::size_t{o0} * H / 32U, m.peer_xdeltas.p + std::size_t{p0} * H / 32U, H / 32U * 4U},
                                     Field{m.ids.p + std::size_t{o0} * K, m.peer_ids.p + std::size_t{p0} * K, K * 4U},
                                     Field{m.route_w.p + std::size_t{o0} * K, m.peer_route_w.p + std::size_t{p0} * K, K * 4U}},
                                    no, np, layer);
                    join_gather();
                    ds41::PrefillMoeInputs pin;
                    pin.rows = np;
                    pin.ids = m.peer_ids.p + std::size_t{p0} * K;
                    pin.weights = m.peer_route_w.p + std::size_t{p0} * K;
                    pin.codes = m.peer_xcodes.p + std::size_t{p0} * H;
                    pin.deltas = m.peer_xdeltas.p + std::size_t{p0} * H / 32U;
                    if (m.moe_joint || T <= m.joint_max_rows) {  // one expert pass over [peer rows; own rows], the peer rows' half sums first
                        moe.enqueue_half_joint(pin, in, me, m.half_other.p, m.half_own.p, st, m.half_ready[0]);
                    } else {
                        for (std::uint32_t ch = 0; ch < moe.chunks(np); ++ch) moe.enqueue_half(pin, ch, me, m.half_other.p, st);
                        check(hipEventRecord(m.half_ready[0], st), "peer half ready");
                        for (std::uint32_t ch = 0; ch < moe.chunks(no); ++ch) moe.enqueue_half(in, ch, me, m.half_own.p, st);
                    }
                    hipStream_t cs = m.comm_stream;
                    check(hipStreamWaitEvent(cs, m.half_ready[0], 0), "peer half ready");
                    const std::uint32_t nmax = std::max(np, no);
                    for (std::uint32_t q0 = 0; q0 < nmax; q0 += C) {
                        const std::uint32_t qn = std::min(C, nmax - q0);
                        m.options.communicator->start(cs, m.half_other.p + std::size_t{q0} * H, std::size_t{qn} * H * 4U,
                                                      dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, q0 / C});
                        m.options.communicator->finish(cs, m.half_recv.p + std::size_t{q0} * H);
                    }
                    check(hipEventRecord(m.half_received, cs), "half received");
                    check(hipStreamWaitEvent(st, m.half_received, 0), "half join");
                    const float* low = me == 0U ? m.half_own.p : m.half_recv.p;
                    const float* high = me == 0U ? m.half_recv.p : m.half_own.p;
                    if (no > 0U) ds41::enqueue_canonical_sum(in.shared, low, high, std::size_t{no} * H, y, st);
                    // a per-layer scratch is freed here; with streamed experts the slot is overwritten next
                    if (!m.moe_persist || m.options.stream_experts) check(hipStreamSynchronize(st), "moe");
                    return;
                }
                const std::uint32_t chunks = moe.chunks(no);
                for (std::uint32_t ch = 0; ch < chunks; ++ch) {
                    if (!m.distributed()) {
                        moe.enqueue_combine(in, ch, y, st);
                        continue;
                    }
                    moe.enqueue_half(in, ch, m.options.rank, own, st);
                    check(hipEventRecord(m.half_ready[ch % 2U], st), "half ready");
                    if (ch > 0U) exchange_chunk(ch - 1U);
                }
                if (m.distributed() && chunks > 0U) exchange_chunk(chunks - 1U);
                // a per-layer scratch is freed here; with streamed experts the slot is overwritten next
                if (!m.moe_persist || m.options.stream_experts) check(hipStreamSynchronize(st), "moe");
            });
            break;
        }
        case Op::head: {
            const std::size_t last = Tn - 1U;  // the last row, local on its owner
            if (last_local) {
                ds41::enqueue_hc_collapse(stream_now() + last * stream_row, m.identity_pre ? nullptr : m.mix_ffn.p + last, 1, H, m.h.p, st);
                ds41::enqueue_rms_norm(m.h.p, m.devf("output_norm.weight"), 1, H, H, 1e-20F, m.h.p, st);
            }
            if (split)
                m.options.communicator->exchange(st, last_local ? m.h.p : m.h_peer.p, last_local ? m.h_peer.p : m.h.p, std::size_t{H} * 4U,
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), 3U, T - 1U});
            ds41::enqueue_quantize_a8(m.h.p, H, 1, m.hcodes.p, m.hdeltas.p, m.hsums.p, st);
            ds41::Q8A8View v;
            v.weight = m.dev("output.weight");
            v.width = H;
            v.count = m.head_slice.count;
            v.per_group = m.head_slice.count;
            v.codes = m.hcodes.p;
            v.deltas = m.hdeltas.p;
            v.rows = 1;
            v.x_stride = H;
            v.out_stride = m.head_slice.count;
            check(hipMemsetAsync(m.logits.p, 0, std::size_t{c.vocab_size} * 4U, st), "logits");
            v.out = m.logits.p + m.head_slice.first;
            ds41::enqueue_q8_0_a8(v, st);
            if (m.distributed() && m.options.gather_logits) {
                const auto other = ds41::vocab_slice(c.vocab_size, 2U, 1U - m.options.rank);
                m.options.communicator->exchange(st, m.logits.p + m.head_slice.first, m.logits.p + other.first, std::size_t{other.count} * 4U,
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), 1U, T - 1U});
            }
            break;
        }
        case Op::argmax: {
            if (m.distributed()) {
                const auto& sl = m.head_slice;
                ds41::enqueue_argmax(m.logits.p + sl.first, 1, sl.count, sl.count, sl.first, m.partial.p, m.cand_own.p, st);
                m.options.communicator->exchange(st, m.cand_own.p, m.cand_other.p, sizeof(ds41::ArgmaxCandidate),
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::argmax), 0U, T - 1U});
                const auto* first = m.options.rank == 0U ? m.cand_own.p : m.cand_other.p;
                const auto* second = m.options.rank == 0U ? m.cand_other.p : m.cand_own.p;
                ds41::enqueue_combine_argmax(first, second, 1, m.best.p, st);
            } else {
                const auto r0 = ds41::vocab_slice(c.vocab_size, 2U, 0U), r1 = ds41::vocab_slice(c.vocab_size, 2U, 1U);
                ds41::enqueue_argmax(m.logits.p + r0.first, 1, r0.count, r0.count, r0.first, m.partial.p, m.cand0.p, st);
                ds41::enqueue_argmax(m.logits.p + r1.first, 1, r1.count, r1.count, r1.first, m.partial.p, m.cand1.p, st);
                ds41::enqueue_combine_argmax(m.cand0.p, m.cand1.p, 1, m.best.p, st);
            }
            const auto best = download(m.best.p, 1, st)[0];
            if (best.finite == 0U) throw std::runtime_error("deepseek41 prefill: non-finite logits");
            result.next_token = best.index;
            result.logits = download(m.logits.p, c.vocab_size, st);
            result.head_seconds = seconds_since(t_head);
            break;
        }
        }
    }
    join_engram();
    check(hipStreamSynchronize(st), "prefill");
    if (m.options.stream_experts) {
        join_prefetch();
        start_prefetch(0U);  // the next run starts at layer 0 (slot 0 is free: layer 0 + 2k finished)
        result.expert_read_seconds = m.expert_read_seconds;
        result.expert_wait_seconds = m.expert_wait_seconds;
    }
    m.rows_done = P0 + T;
    m.local_rows_done = Tn;
    m.last_row_local = last_local;
    result.seconds = seconds_since(t_run);
    return result;
}

std::vector<std::pair<std::string, std::uint64_t>> DevicePrefill::state_hashes() const {
    const auto& m = *impl_;
    namespace attn = ds41::attn;
    std::vector<std::pair<std::string, std::uint64_t>> out;
    const std::uint32_t T = m.rows_done;
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        const auto& L = m.attn[l];
        const std::string at = "@" + std::to_string(l);
        const auto ring = download(L.ring.p, L.ring.n, m.stream);
        out.emplace_back("ring" + at, ring_window_hash(ring.data(), T));
        if (r.owns_kv) {
            const std::size_t entries = T / r.ratio;
            const auto kv = download(L.kv_cache.p, entries * attn::kKvEntryBytes, m.stream);
            const auto ix = download(L.idx_cache.p, entries * attn::kIdxEntryBytes, m.stream);
            out.emplace_back("kv" + at, fnv_bytes(kv.data(), kv.size()));
            out.emplace_back("idx_keys" + at, fnv_bytes(ix.data(), ix.size()));
            if (r.ratio == 2U) {
                const auto carry = download(L.carry[L.carry_cur].p, L.carry[L.carry_cur].n, m.stream);
                const auto valid = download(L.carry_valid[L.carry_cur].p, 1, m.stream);
                out.emplace_back("carry" + at, valid[0] != 0U ? fnv_bytes(carry.data(), carry.size() * 4U) : 0U);
            }
        }
        if (r.owns_index) {  // the last row's selection, on the rank that computed the row (row_split: 0 on the other)
            if (!m.last_row_local) {
                out.emplace_back("topk" + at, 0U);
            } else {
                const std::uint32_t last = m.local_rows_done - 1U;
                const auto count = download(L.idx_count.p + last, 1, m.stream)[0];
                const auto idx = download(L.idx.p + std::size_t{last} * attn::kTopK, count, m.stream);
                out.emplace_back("topk" + at, fnv_bytes(idx.data(), idx.size() * 4U, count));
            }
        }
    }
    return out;
}

void DevicePrefill::import_state(const AttentionState& s) {
    auto& m = *impl_;
    namespace attn = ds41::attn;
    const auto fail = [](const std::string& what) { throw std::invalid_argument("deepseek41 prefill: import_state: " + what); };
    if (s.layers.size() != m.options.layers) fail("layer count differs");
    if (s.tokens.size() != s.position || s.position == 0U) fail("token history length != position, or an empty state");
    if (s.position % m.options.window_rows != 0U || s.position >= m.options.max_positions) fail("position not a multiple of window_rows or no room left");
    hipStream_t st = m.stream;
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        auto& L = m.attn[l];
        const auto& S = s.layers[l];
        if (S.owns_kv != r.owns_kv || S.owns_index != r.owns_index || (r.owns_kv && S.ratio != r.ratio)) fail("roles of layer " + std::to_string(l));
        if (S.ring.size() != L.ring.n) fail("ring size of layer " + std::to_string(l));
        check(hipMemcpyAsync(L.ring.p, S.ring.data(), L.ring.n, hipMemcpyHostToDevice, st), "import ring");
        if (r.owns_kv) {
            const std::size_t entries = s.position / r.ratio;
            if (S.kv.size() != entries * attn::kKvEntryBytes || S.idx_keys.size() != entries * attn::kIdxEntryBytes) fail("caches of layer " + std::to_string(l));
            L.kv_cache.zero(st);
            L.idx_cache.zero(st);
            check(hipMemcpyAsync(L.kv_cache.p, S.kv.data(), S.kv.size(), hipMemcpyHostToDevice, st), "import kv");
            check(hipMemcpyAsync(L.idx_cache.p, S.idx_keys.data(), S.idx_keys.size(), hipMemcpyHostToDevice, st), "import idx");
            for (int k = 0; k < 2; ++k) {
                L.carry[k].zero(st);
                L.carry_valid[k].zero(st);
            }
            L.carry_cur = 0;
            if (r.ratio == 2U && S.carry_valid) {
                if (S.carry.size() != L.carry[0].n) fail("carry of layer " + std::to_string(l));
                check(hipMemcpyAsync(L.carry[0].p, S.carry.data(), S.carry.size() * 4U, hipMemcpyHostToDevice, st), "import carry");
                check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(L.carry_valid[0].p), 1, 1, st), "import carry");
            }
        }
    }
    m.hasher->reset();
    std::vector<std::uint64_t> one(std::size_t{m.hasher->layers()} * m.hasher->columns());
    for (const auto t : s.tokens) m.hasher->push(t, one);
    m.rows_done = s.position;
    m.local_rows_done = 0;
    m.last_row_local = false;
    check(hipStreamSynchronize(st), "import state");
}

AttentionState DevicePrefill::export_state() const {
    const auto& m = *impl_;
    namespace attn = ds41::attn;
    AttentionState s;
    const std::uint32_t T = m.rows_done;
    s.position = T;
    s.layers.resize(m.options.layers);
    if (m.overlap) {  // the same bytes with two stream synchronisations instead of several per layer
        const auto get = [&](auto& v, const auto* p, std::size_t n) {
            v.resize(n);
            if (n != 0U) check(hipMemcpyAsync(v.data(), p, n * sizeof(*p), hipMemcpyDeviceToHost, m.stream), "export");
        };
        std::vector<std::uint32_t> valid(m.options.layers, 0U), count(m.options.layers, 0U);
        const std::uint32_t last = m.local_rows_done - 1U;
        for (std::uint32_t l = 0; l < m.options.layers; ++l) {
            const auto& r = m.roles[l];
            const auto& L = m.attn[l];
            if (r.owns_kv && r.ratio == 2U)
                check(hipMemcpyAsync(&valid[l], L.carry_valid[L.carry_cur].p, 4, hipMemcpyDeviceToHost, m.stream), "export");
            if (r.owns_index && m.last_row_local) check(hipMemcpyAsync(&count[l], L.idx_count.p + last, 4, hipMemcpyDeviceToHost, m.stream), "export");
        }
        for (std::uint32_t l = 0; l < m.options.layers; ++l) {
            const auto& r = m.roles[l];
            const auto& L = m.attn[l];
            auto& S = s.layers[l];
            S.owns_kv = r.owns_kv;
            S.owns_index = r.owns_index;
            S.ratio = r.ratio;
            get(S.ring, L.ring.p, L.ring.n);
            if (r.owns_kv) {
                const std::size_t entries = T / r.ratio;
                get(S.kv, L.kv_cache.p, entries * attn::kKvEntryBytes);
                get(S.idx_keys, L.idx_cache.p, entries * attn::kIdxEntryBytes);
                if (r.ratio == 2U) get(S.carry, L.carry[L.carry_cur].p, L.carry[L.carry_cur].n);
            }
        }
        check(hipStreamSynchronize(m.stream), "export");  // the counts and the carries' flags are on the host now
        for (std::uint32_t l = 0; l < m.options.layers; ++l) {
            const auto& r = m.roles[l];
            const auto& L = m.attn[l];
            auto& S = s.layers[l];
            if (r.owns_kv && r.ratio == 2U) S.carry_valid = valid[l] != 0U;
            if (r.owns_index) {
                S.topk_known = m.last_row_local;
                if (m.last_row_local) get(S.topk, L.idx.p + std::size_t{last} * attn::kTopK, count[l]);
            }
        }
        check(hipStreamSynchronize(m.stream), "export");
        return s;
    }
    for (std::uint32_t l = 0; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        const auto& L = m.attn[l];
        auto& S = s.layers[l];
        S.owns_kv = r.owns_kv;
        S.owns_index = r.owns_index;
        S.ratio = r.ratio;
        S.ring = download(L.ring.p, L.ring.n, m.stream);
        if (r.owns_kv) {
            const std::size_t entries = T / r.ratio;
            S.kv = download(L.kv_cache.p, entries * attn::kKvEntryBytes, m.stream);
            S.idx_keys = download(L.idx_cache.p, entries * attn::kIdxEntryBytes, m.stream);
            if (r.ratio == 2U) {
                S.carry = download(L.carry[L.carry_cur].p, L.carry[L.carry_cur].n, m.stream);
                S.carry_valid = download(L.carry_valid[L.carry_cur].p, 1, m.stream)[0] != 0U;
            }
        }
        if (r.owns_index) {
            S.topk_known = m.last_row_local;
            if (m.last_row_local) {
                const std::uint32_t last = m.local_rows_done - 1U;
                const auto count = download(L.idx_count.p + last, 1, m.stream)[0];
                S.topk = download(L.idx.p + std::size_t{last} * attn::kTopK, count, m.stream);
            }
        }
    }
    return s;
}

}  // namespace strix::deepseek41
