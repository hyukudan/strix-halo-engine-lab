#include "glm/engine/glm_kvshard.hpp"
#include "glm/engine/glm_kvshard_math.hpp"
#include "glm/prefill/glm_prefill.hpp"
#include "glm/prefill/glm_kvshard_prefill.hpp"
#include "glm/engine/glm_lib.hpp"
#include "strix/dist/communicator_hip.hpp"
#include "glm/engine/glm_kvshard_exchange.hpp"
#include "glm/xfer/glm_xfer_exchange.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <vector>

// Rank merge retains separate FP32 products/additions in rank-0 then rank-1
// order. Partial attention uses the shared tuned tree256/FMA family (FAST v3).
#pragma clang fp contract(off)

namespace strix::glm::engine {
namespace {
using namespace detail;
namespace F = strix::glm::fast;
namespace K = strix::glm::kvshard;
namespace X = strix::xport::kvshard;
static_assert(kKpool == K::block_tokens);
static_assert(kKvLora == X::PartialDims::accumulator);

// Only this kernel changes the temporary position descriptor. Cache addresses
// use physical positions for append, global positions for selection/attention.
__global__ void position_kernel(unsigned* position, unsigned value) { *position = value; }

__global__ void owned_scores_kernel(F::DsaRows rows, const float* query, const float* weights,
                                    std::uint32_t* keys, unsigned rank, unsigned pools) {
    const unsigned pool = blockIdx.x * blockDim.x + threadIdx.x;
    if (pool >= pools) return;
    if (pool % 2 != rank) { keys[pool] = 0; return; }
    const float* cache = rows.slots[0].pooled + static_cast<std::size_t>(pool / 2) * kIdxDim;
    float score = 0.0F;
    for (unsigned head = 0; head < kIdxHeads; ++head) {
        float dot = 0.0F;
        for (unsigned dim = 0; dim < kIdxDim; ++dim)
            dot = __builtin_fmaf(query[head * kIdxDim + dim], cache[dim], dot);
        const float value = dot * kInvSqrt128;
        score = __builtin_fmaf(weights[head] * kInvSqrt32, value > 0.0F ? value : 0.0F, score);
    }
    keys[pool] = ordered_key(score);
}

__global__ void histogram_kernel(const std::uint32_t* keys, unsigned pools, std::uint32_t* hist) {
    const unsigned pool = blockIdx.x * blockDim.x + threadIdx.x;
    if (pool < pools) atomicAdd(hist + (keys[pool] >> 24U), 1U);
}

__global__ __launch_bounds__(512) void merge_kernel(const float* rank0, const float* rank1, float* output, unsigned* flag) {
    const unsigned head = blockIdx.x, dim = threadIdx.x, offset = head * (kKvLora + 2);
    const float value = K::merge_component(rank0[offset], rank0[offset + 1], rank0[offset + 2 + dim],
                                          rank1[offset], rank1[offset + 1], rank1[offset + 2 + dim]);
    if (!is_finite(value)) atomicOr(flag, 1U);
    output[head * kKvLora + dim] = value;
}

struct PlanEvent {
    hipEvent_t value = nullptr;
    PlanEvent() { hip_ok(hipEventCreateWithFlags(&value, hipEventDisableTiming), "kvshard plan event"); }
    ~PlanEvent() { (void)hipEventDestroy(value); }
    PlanEvent(const PlanEvent&) = delete;
    PlanEvent& operator=(const PlanEvent&) = delete;
};
// Synchronous bounded redistribution adapter for prefill windows.
// Decode uses the same Channel as the TP reductions below.
class Wire {
public:
    struct Tag { std::uint64_t layer, step; };
    struct Ticket { std::size_t bytes; std::uint64_t sequence; };
    struct Staging { void* data; };
    Wire(dist::Communicator& communicator) : comm_(communicator), tx_(9 * sizeof(std::uint64_t) + X::PartialDims::bytes) {}
    void check() const { if (failed_) throw std::runtime_error("kvshard: failed wire"); }
    Staging staging() { return {tx_.data()}; }
    void trim() {
        hip_ok(hipStreamSynchronize(stream), "kvshard trim wait");
        std::vector<unsigned char>().swap(rx_);
    }
    void reserve_bulk(std::size_t bytes) {
        // Avoid vector growth temporarily retaining two receive allocations as
        // successive unions get larger. This owner is in the admitted peak.
        rx_.reserve(std::min(bytes + 9 * sizeof(std::uint64_t), comm_.max_bytes()));
    }
    std::uint64_t prefill_budget(std::uint64_t local, std::uint64_t epoch, std::uint64_t layer, std::uint64_t start) {
        std::array<std::uint64_t, 6> frame{K::layout_id, epoch, layer, start, 8, local};
        const auto ticket = post({layer, start}, {frame.data()}, sizeof(frame), true);
        const auto* peer = static_cast<const unsigned char*>(wait(ticket, stream, true));
        std::array<std::uint64_t, 6> received{};
        std::memcpy(received.data(), peer, sizeof(received));
        if (!std::equal(frame.begin(), frame.begin() + 5, received.begin())) {
            failed_ = true;
            throw std::runtime_error("kvshard: prefill budget identity mismatch");
        }
        return std::min(local, received[5]);
    }
    Ticket post(const Tag& tag, Staging buffer, std::size_t bytes, bool) {
        check();
        if (tag.layer > std::numeric_limits<std::uint32_t>::max() || sequence_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("kvshard: wire tag overflow");
        try {
            // Host collectives validate their trailer synchronously and never
            // release a GPU consumer after a failed transfer. Do not borrow a
            // GPU-direct exchange's deferred validation/progress state.
            if (bytes > comm_.max_bytes()) throw std::runtime_error("kvshard: framed bulk exceeds communicator capacity");
            rx_.resize(bytes);
            dist::Tag physical;
            physical.kind = static_cast<std::uint32_t>(dist::TagKind::raw_slots);
            physical.layer = static_cast<std::uint32_t>(tag.layer);
            physical.request = ++sequence_;
            comm_.exchange_host({static_cast<const std::byte*>(buffer.data), bytes},
                                {reinterpret_cast<std::byte*>(rx_.data()), bytes}, physical);
            return {bytes, sequence_};
        } catch (...) { failed_ = true; throw; }
    }
    const void* wait(const Ticket& ticket, hipStream_t posting_stream, bool producer_complete = false) {
        if (ticket.sequence != sequence_ || posting_stream != stream) throw std::logic_error("kvshard: unordered wire wait");
        if (ticket.bytes != rx_.size()) throw std::logic_error("kvshard: wire receive shape");
        if (!producer_complete) hip_ok(hipStreamSynchronize(stream), "kvshard wire wait");
        return rx_.data();
    }
    // Metadata precedes every bulk transfer too: stale or mismatched selection /
    // prefill frames cannot silently enter the numerical path.
    std::vector<unsigned char> bulk(const std::array<std::uint64_t, 7>& header, const void* device, std::size_t bytes,
                                    void* staged = nullptr) {
        using Chunk = std::array<std::uint64_t, 9>;
        if (comm_.max_bytes() <= sizeof(Chunk)) throw std::runtime_error("kvshard: communicator cannot hold a bulk header");
        std::vector<unsigned char> result(bytes);
        for (std::size_t offset = 0; offset < bytes;) {
            const std::size_t count = std::min(bytes - offset, comm_.max_bytes() - sizeof(Chunk));
            Chunk chunk{};
            std::copy(header.begin(), header.end(), chunk.begin());
            chunk[7] = offset; chunk[8] = count;
            std::vector<unsigned char> frame(sizeof(chunk) + count);
            std::memcpy(frame.data(), chunk.data(), sizeof(chunk));
            try {
                hip_ok(hipMemcpyAsync(frame.data() + sizeof(chunk), static_cast<const unsigned char*>(device) + offset,
                                      count, hipMemcpyDeviceToHost, stream), "kvshard bulk stage");
                hip_ok(hipEventRecord(bulk_ready_.value, stream), "kvshard bulk stage ready");
                hip_ok(hipEventSynchronize(bulk_ready_.value), "kvshard bulk stage wait");
            } catch (...) {
                // Keep the pageable frame alive even if recording its event fails.
                (void)hipStreamSynchronize(stream);
                throw;
            }
            // Reuse the completed D2H copy when the caller also needs its own
            // payload. In particular, do not read score keys back a second time.
            if (staged != nullptr)
                std::memcpy(static_cast<unsigned char*>(staged) + offset, frame.data() + sizeof(chunk), count);
            const auto ticket = post({header[2], header[3]}, {frame.data()}, frame.size(), true);
            // The D2H event proves all device producers complete; post() has
            // synchronously validated the host transport. Retain every shape,
            // sequence and metadata check without a second full-stream wait.
            const auto* received = static_cast<const unsigned char*>(wait(ticket, stream, true));
            if (std::memcmp(received, chunk.data(), sizeof(chunk)) != 0) {
                failed_ = true;
                throw std::runtime_error("kvshard: bulk layout / epoch / layer / position / shape / offset mismatch");
            }
            std::memcpy(result.data() + offset, received + sizeof(chunk), count);
            offset += count;
        }
        return result;
    }
    hipStream_t stream = nullptr;
private:
    dist::Communicator& comm_;
    std::vector<unsigned char> tx_, rx_;
    std::uint64_t sequence_ = 0;
    bool failed_ = false;
    PlanEvent bulk_ready_;
};
// Pack bits only; the attention kernels and rank0/rank1 merge stay unchanged.
__global__ void frame_kernel(K::DecodeHeader header, const std::uint32_t* payload, unsigned words,
        const std::uint32_t* keys, unsigned pools, std::uint32_t* frame) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < K::header_words) frame[i] = reinterpret_cast<const std::uint32_t*>(&header)[i];
    else if (i < K::header_words + words) frame[i] = payload[i - K::header_words];
    else if (i < K::header_words + words + pools) frame[i] = keys[i - K::header_words - words];
}
__device__ bool valid_frame(const std::uint32_t* frame, K::DecodeHeader expected, unsigned* flag) {
    const auto* peer = reinterpret_cast<const std::uint32_t*>(frame);
    const auto* own = reinterpret_cast<const std::uint32_t*>(&expected);
    for (std::size_t i = 0; i < K::header_words; ++i) {
        if (peer[i] != own[i]) { atomicOr(flag, 1U); return false; }
    }
    return true;
}
__global__ void prepare_consume_kernel(const std::uint32_t* frame, K::DecodeHeader expected,
        float* peer_query, std::uint32_t* keys, unsigned pools, unsigned rank, unsigned* flag) {
    if (!valid_frame(frame, expected, flag)) return;
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < K::query_u32) reinterpret_cast<std::uint32_t*>(peer_query)[i] = frame[K::header_words + i];
    if (i < pools && i % 2 != rank) keys[i] = frame[K::header_words + K::query_u32 + i];
}
__global__ void validate_kernel(const std::uint32_t* frame, K::DecodeHeader expected, unsigned* flag) {
    (void)valid_frame(frame, expected, flag);
}
} // namespace

struct KvShard::Impl {
    unsigned rank;
    dist::Communicator& comm;
    bool shared_tp = true;
    std::uint64_t epoch = 0;
    Wire wire; // prefill only; decode uses the attention Channel.
    std::shared_ptr<xfer::XferExchange> exchange;
    bool budget_line_printed = false;
    float *peer_query = nullptr, *own_partial = nullptr, *peer_partial = nullptr;
    explicit Impl(dist::Communicator& communicator, unsigned rank_value)
        : rank(rank_value), comm(communicator), wire(communicator) {
        try {
            peer_query = dalloc<float>(X::PartialDims::local_heads * kKvLora);
            own_partial = dalloc<float>(X::PartialDims::local_heads * (kKvLora + 2));
            peer_partial = dalloc<float>(X::PartialDims::local_heads * (kKvLora + 2));
        } catch (...) {
            (void)hipFree(peer_query); (void)hipFree(own_partial); (void)hipFree(peer_partial);
            throw;
        }
    }
    ~Impl() { (void)hipFree(peer_query); (void)hipFree(own_partial); (void)hipFree(peer_partial); }
    xfer::Exchanger& channel(hipStream_t stream) {
        if (!exchange) throw std::logic_error("kvshard: attention Channel not installed");
        return exchange->attention_channel(stream);
    }
};

KvShard::KvShard(dist::Communicator& comm, unsigned rank) : impl_(std::make_unique<Impl>(comm, rank)) {}
KvShard::~KvShard() = default;
void KvShard::reset() {
    if (impl_->epoch == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("kvshard: epoch exhausted");
    ++impl_->epoch;
}

void KvShard::append(const DsaState&, const CallCtx& call, const F::DsaRows& rows, const float* latent,
                    const float* key, const float* gate, const dev::IndexerDev& weights, std::uint32_t* hist) {
    if (call.rows.size() != 1 || call.spec != nullptr || call.rows[0].slot != 0)
        throw std::invalid_argument("kvshard: one committed row in slot zero required");
    const auto position = call.rows[0].pos;
    if (K::owner(position) == impl_->rank) {
        position_kernel<<<1, 1, 0, call.stream>>>(const_cast<unsigned*>(rows.tpos), static_cast<unsigned>(K::local_token(position)));
        hip_ok(hipGetLastError(), "kvshard local position");
        F::dsa_store(rows, latent, key, gate, weights, hist, call.flag, call.stream);
    }
    position_kernel<<<1, 1, 0, call.stream>>>(const_cast<unsigned*>(rows.tpos), static_cast<unsigned>(position));
    hip_ok(hipGetLastError(), "kvshard global position");
}

void KvShard::bind_exchange(std::shared_ptr<xfer::XferExchange> exchange, bool shared_tp) {
    impl_->exchange = std::move(exchange);
    impl_->shared_tp = shared_tp;
}

void KvShard::prepare(const DsaState& state, const CallCtx& call, const F::DsaRows& rows,
        const float* index_query, const float* weights, std::uint32_t* keys, unsigned stride, std::uint32_t* hist,
        const float* query, const std::uint16_t* kv_b, const F::MlaWork& work, bool score_keys) {
    constexpr unsigned heads = X::PartialDims::local_heads;
    const unsigned pools = score_keys ? static_cast<unsigned>((call.rows[0].pos + 1) / kKpool) : 0U;
    if (pools > stride) throw std::out_of_range("kvshard: score stride");
    if (!impl_->shared_tp) impl_->comm.release_transport(call.stream);
    F::mla_absorb(query, kv_b, heads, work.qt, work.mkey, 1, call.stream);
    if (pools != 0) {
        owned_scores_kernel<<<(pools + 63) / 64, 64, 0, call.stream>>>(rows, index_query, weights, keys, impl_->rank, pools);
        hip_ok(hipGetLastError(), "kvshard owned scores");
    }
    auto& channel = impl_->channel(call.stream);
    const auto header = K::decode_header(impl_->epoch, state.shard_layer, call.rows[0].pos, false, pools);
    const std::size_t words = K::header_words + K::query_u32 + pools;
    frame_kernel<<<static_cast<unsigned>((words + 255) / 256), 256, 0, call.stream>>>(header,
        reinterpret_cast<const std::uint32_t*>(work.qt), static_cast<unsigned>(K::query_u32), keys, pools,
        static_cast<std::uint32_t*>(channel.tx()));
    hip_ok(hipGetLastError(), "kvshard prepare frame");
    channel.send(K::prepare_bytes(pools), {xfer::Kind::shard_prepare, state.shard_layer, call.rows[0].pos}, xfer::Publish::kernel);
    const auto* peer = static_cast<const std::uint32_t*>(channel.recv());
    const std::size_t consume_words = std::max(K::query_u32, static_cast<std::size_t>(pools));
    prepare_consume_kernel<<<static_cast<unsigned>((consume_words + 255) / 256), 256, 0, call.stream>>>(
        peer, header, impl_->peer_query, keys, pools, impl_->rank, call.flag);
    hip_ok(hipGetLastError(), "kvshard prepare consume");
    if (pools != 0) {
        hip_ok(hipMemsetAsync(hist, 0, 256 * sizeof(std::uint32_t), call.stream), "kvshard histogram clear");
        histogram_kernel<<<(pools + 255) / 256, 256, 0, call.stream>>>(keys, pools, hist);
        hip_ok(hipGetLastError(), "kvshard global histogram");
    }
}

void KvShard::attention(const DsaState& state, const CallCtx& call, const F::DsaRows& rows,
        const std::uint32_t* tokens, const std::uint32_t* count, const float*,
        const std::uint16_t* kv_b, const F::MlaWork& work, float* output) {
    constexpr unsigned heads = X::PartialDims::local_heads;
    const unsigned max_tokens = static_cast<unsigned>(std::min<std::uint64_t>(call.rows[0].pos + 1, F::kMaxTok));
    // prepare -> global selection -> peer-query partial -> rank01 merge ->
    // mla_out -> output projection -> TP reduction is a true dependency chain.
    // Neither partials nor next-layer queries can ride that same TP reduction
    // without changing arithmetic or replicating the missing queries/KV work.
    // Send the peer-query partial first; local-query work runs while it travels.
    F::mla_partial(rows, tokens, count, impl_->peer_query, heads, work, impl_->peer_partial, impl_->rank, call.flag, call.stream, max_tokens);
    auto& channel = impl_->channel(call.stream);
    const auto header = K::decode_header(impl_->epoch, state.shard_layer, call.rows[0].pos, true, 0);
    constexpr std::size_t words = K::header_words + K::partial_u32;
    frame_kernel<<<static_cast<unsigned>((words + 255) / 256), 256, 0, call.stream>>>(header,
        reinterpret_cast<const std::uint32_t*>(impl_->peer_partial), static_cast<unsigned>(K::partial_u32), nullptr, 0,
        static_cast<std::uint32_t*>(channel.tx()));
    hip_ok(hipGetLastError(), "kvshard partial frame");
    channel.send(K::partial_bytes, {xfer::Kind::shard_partial, state.shard_layer, call.rows[0].pos}, xfer::Publish::kernel);
    F::mla_partial(rows, tokens, count, work.qt, heads, work, impl_->own_partial, impl_->rank, call.flag, call.stream, max_tokens);
    const auto* peer_frame = static_cast<const std::uint32_t*>(channel.recv());
    validate_kernel<<<1, 1, 0, call.stream>>>(peer_frame, header, call.flag);
    hip_ok(hipGetLastError(), "kvshard partial validation");
    const auto* remote = reinterpret_cast<const float*>(peer_frame + K::header_words);
    const float* rank0 = impl_->rank == 0 ? impl_->own_partial : remote;
    const float* rank1 = impl_->rank == 0 ? remote : impl_->own_partial;
    merge_kernel<<<heads, 512, 0, call.stream>>>(rank0, rank1, work.ob, call.flag);
    hip_ok(hipGetLastError(), "kvshard rank01 merge");
    F::mla_out(work.ob, kv_b, heads, output, 1, call.stream);
    if (!impl_->shared_tp) {
        impl_->exchange->complete();
        hip_ok(hipStreamSynchronize(call.stream), "kvshard attention transport handoff");
    }
}

namespace {
template<class T> struct Buffer {
    T* data = nullptr;
    bool pinned;
    explicit Buffer(std::size_t count, bool host = false) : pinned(host) {
        if (pinned) hip_ok(hipHostMalloc(reinterpret_cast<void**>(&data), count * sizeof(T), hipHostMallocDefault), "kvshard pinned plan");
        else data = dalloc<T>(count);
    }
    ~Buffer() { if (pinned) (void)hipHostFree(data); else (void)hipFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct PlanStream {
    hipStream_t value = nullptr;
    PlanStream() { hip_ok(hipStreamCreateWithFlags(&value, hipStreamNonBlocking), "kvshard plan stream"); }
    ~PlanStream() { (void)hipStreamDestroy(value); }
    PlanStream(const PlanStream&) = delete;
    PlanStream& operator=(const PlanStream&) = delete;
};
// Construct after all owners, before submissions: also drains both streams on
// validation/transport exceptions, before pinned/device owners can be released.
struct PlanDrain {
    hipStream_t compute, copy;
    ~PlanDrain() { (void)hipStreamSynchronize(copy); (void)hipStreamSynchronize(compute); }
};
// Pack only the owned half of a global pool window. Unowned entries are zero
// and never read an uninitialised compact cache address.
__global__ void pool_window_kernel(const float* compact, float* packed, unsigned begin, unsigned count, unsigned rank) {
    const unsigned element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= count * kIdxDim) return;
    const unsigned pool = begin + element / kIdxDim, dim = element % kIdxDim;
    packed[element] = pool % 2 == rank ? compact[static_cast<std::size_t>(pool / 2) * kIdxDim + dim] : 0.0F;
}
__global__ void merge_pool_window_kernel(const float* local, const float* peer, float* merged,
                                        unsigned begin, unsigned count, unsigned rank) {
    const unsigned element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= count * kIdxDim) return;
    merged[element] = (begin + element / kIdxDim) % 2 == rank ? local[element] : peer[element];
}
__global__ void selected_latent_kernel(const std::uint16_t* compact, const std::uint32_t* tokens,
                                      unsigned count, std::uint16_t* packed, unsigned rank) {
    const unsigned element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= count * kKvLora) return;
    const unsigned item = element / kKvLora, dim = element % kKvLora;
    const unsigned token = tokens[item];
    packed[element] = K::owner(token) == rank
        ? compact[static_cast<std::size_t>(K::local_token(token)) * kKvLora + dim] : std::uint16_t{};
}
__global__ void merge_selected_kernel(const std::uint16_t* local, const std::uint16_t* peer, std::uint16_t* merged,
                                     const std::uint32_t* tokens, unsigned count, unsigned rank) {
    const unsigned element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= count * kKvLora) return;
    const unsigned item = element / kKvLora;
    merged[element] = K::owner(tokens[item]) == rank ? local[element] : peer[element];
}
} // namespace

void KvShard::prefill(PrefillPath& path, const DsaLayerWeights& weights, DsaState& state,
                      const CallCtx& call, const float* input, float* output) {
    if (state.shard != this) throw std::invalid_argument("kvshard: prefill owner mismatch");
    path.dsa(weights, state, call, input, output);
}

void KvShard::prefill_rows(const DsaLayerWeights& weights, const DsaState& state, const CallCtx& call,
                           const PrefillWork& work, std::uint64_t margin) {
    if (call.rows.empty() || call.spec != nullptr || state.shard != this)
        throw std::invalid_argument("kvshard: prefill rows / owner");
    const unsigned count = static_cast<unsigned>(call.rows.size());
    const auto start = call.rows[0].pos;
    for (unsigned row = 0; row < count; ++row)
        if (call.rows[row].slot != 0 || call.rows[row].pos != start + row || call.rows[row].pos >= state.cap)
            throw std::invalid_argument("kvshard: consecutive prefill rows in slot zero required");
    Lib::admit(pf::kKvShardPrefillTemporaryBytes, margin, "kvshard bounded prefill temporary");
    const unsigned storage = static_cast<unsigned>(K::storage_tokens(state.cap));
    const dev::DsaSlot owned{state.latent, state.ikey, state.igate, state.pooled, storage};
    const dev::IndexerDev iw{weights.wk, weights.gate, weights.weights_proj, weights.k_norm_w, weights.k_norm_b, weights.ape};
    // Each pool is owned wholly by one rank. The existing store and pool
    // arithmetic runs at compact addresses, including partial-pool chunk carry.
    pf::dsa_append_rows(owned, static_cast<unsigned>(start), work.latent, work.key, work.gate,
                        iw, call.flag, count, call.stream, true, impl_->rank);
    impl_->wire.stream = call.stream;
    impl_->wire.trim();
    { // Pool scratch is gone before allocating the selected-latent images.
        constexpr unsigned pool_elements = pf::kIndexerPoolWindow * kIdxDim;
        Buffer<float> pool_local(pool_elements), pool_peer(pool_elements), pool_full(pool_elements);
        auto pool_window = [&](unsigned begin, unsigned pools) -> const float* {
            pool_window_kernel<<<(pools * kIdxDim + 255) / 256, 256, 0, call.stream>>>(
                state.pooled, pool_local.data, begin, pools, impl_->rank);
            hip_ok(hipGetLastError(), "kvshard prefill pool pack");
            const std::size_t bytes = static_cast<std::size_t>(pools) * kIdxDim * sizeof(float);
            auto remote = impl_->wire.bulk({K::layout_id, impl_->epoch, state.shard_layer, start, 6, bytes, begin}, pool_local.data, bytes);
            hip_ok(hipMemcpyAsync(pool_peer.data, remote.data(), bytes, hipMemcpyHostToDevice, call.stream), "kvshard prefill pool peer");
            merge_pool_window_kernel<<<(pools * kIdxDim + 255) / 256, 256, 0, call.stream>>>(
                pool_local.data, pool_peer.data, pool_full.data, begin, pools, impl_->rank);
            hip_ok(hipGetLastError(), "kvshard prefill pool merge");
            // The pageable source and receive frame remain alive through H2D/merge.
            hip_ok(hipStreamSynchronize(call.stream), "kvshard prefill pool lifetime");
            return pool_full.data;
        };
        pf::indexer_select_rows_bounded(owned, static_cast<unsigned>(start), work.index_query, work.weights,
                                        work.keys, work.tokens, work.counts, count, call.stream, 0, pool_window);
        impl_->wire.trim();
    }
    // Both ranks choose identical geometry from the same fail-closed admission.
    // The startup 64 MiB charge stays unchanged even when there is no extra spare.
    std::uint64_t device_free = 0, mem_available = 0;
    const auto spare = Lib::admit(pf::kKvShardPrefillTemporaryBytes, margin, "kvshard prefill budget", &device_free, &mem_available);
    const auto budget = impl_->wire.prefill_budget(std::min(spare, pf::kKvShardPrefillMaximumBytes),
                                                  impl_->epoch, state.shard_layer, start);
    if (!impl_->budget_line_printed) {
        std::printf("rank %u kv-budget layout=sharded dsa_layer_kv_bytes=%llu pf_path=%llu pf_window_budget=%llu device_free=%llu mem_available=%llu margin=%llu\n",
            impl_->rank, static_cast<unsigned long long>(K::storage_tokens(state.cap) * kvshard::bytes_per_token),
            static_cast<unsigned long long>(work.path_bytes), static_cast<unsigned long long>(budget),
            static_cast<unsigned long long>(device_free), static_cast<unsigned long long>(mem_available),
            static_cast<unsigned long long>(margin));
        std::fflush(stdout);
        impl_->budget_line_printed = true;
    }
    const auto limits = pf::kvshard_prefill_limits(budget, std::min(state.cap, start + count), count, pf::kMaxTokens);
    const std::size_t latent_elements = static_cast<std::size_t>(limits.latents) * kKvLora;
    const std::size_t token_elements = static_cast<std::size_t>(limits.rows) * pf::kMaxTokens;
    const std::size_t upload_elements = limits.latents + token_elements;
    impl_->wire.reserve_bulk(latent_elements * sizeof(std::uint16_t));
    PlanStream copy;
    Buffer<std::uint16_t> latent_local(latent_elements), latent_peer(latent_elements), latent_full(latent_elements);
    std::array<Buffer<std::uint32_t>, 2> ids{Buffer<std::uint32_t>(upload_elements), Buffer<std::uint32_t>(upload_elements)};
    std::array<Buffer<std::uint32_t>, 2> uploads{Buffer<std::uint32_t>(upload_elements, true), Buffer<std::uint32_t>(upload_elements, true)};
    std::array<Buffer<std::uint32_t>, 2> pages{Buffer<std::uint32_t>(token_elements + limits.rows, true),
                                              Buffer<std::uint32_t>(token_elements + limits.rows, true)};
    std::array<PlanEvent, 2> page_ready, ids_ready, consumed;
    PlanEvent peer_copied;
    std::vector<unsigned char> remote;
    PlanDrain drain{call.stream, copy.value};
    std::array<bool, 2> used{false, false};
    auto read_page = [&](unsigned begin, unsigned slot) {
        const unsigned rows = std::min(limits.rows, count - begin);
        hip_ok(hipMemcpyAsync(pages[slot].data, work.tokens + static_cast<std::size_t>(begin) * pf::kMaxTokens,
            static_cast<std::size_t>(rows) * pf::kMaxTokens * sizeof(std::uint32_t), hipMemcpyDeviceToHost, copy.value),
            "kvshard prefill selection page");
        hip_ok(hipMemcpyAsync(pages[slot].data + token_elements, work.counts + begin,
            static_cast<std::size_t>(rows) * sizeof(std::uint32_t), hipMemcpyDeviceToHost, copy.value),
            "kvshard prefill selection counts");
        hip_ok(hipEventRecord(page_ready[slot].value, copy.value), "kvshard selection ready");
    };
    // indexer/pool trim above completed the producer once for the layer.
    read_page(0, 0);
    if (limits.rows < count) read_page(limits.rows, 1);
    unsigned sequence = 0, page = 0;
    for (unsigned begin = 0; begin < count; ++page) {
        const unsigned page_slot = page % 2;
        const unsigned rows = std::min(limits.rows, count - begin);
        hip_ok(hipEventSynchronize(page_ready[page_slot].value), "kvshard selection page wait");
        for (unsigned row = 0; row < rows;) {
            // Previous attention is already enqueued: plan w+1 on the CPU while
            // w executes. Only address remapping changes, never a row's list.
            auto window = pf::kvshard_prefill_window(pages[page_slot].data + static_cast<std::size_t>(row) * pf::kMaxTokens,
                pages[page_slot].data + token_elements + row, rows - row, pf::kMaxTokens, state.cap, limits);
            const unsigned slot = sequence++ % 2;
            if (used[slot]) {
                // Host upload lifetime ends at H2D; device slot lifetime ends at
                // MLA's last consumer. Neither wait drains the compute stream.
                hip_ok(hipEventSynchronize(ids_ready[slot].value), "kvshard upload source reuse");
                hip_ok(hipStreamWaitEvent(copy.value, consumed[slot].value, 0), "kvshard device IDs reuse");
            }
            std::copy(window.latents.begin(), window.latents.end(), uploads[slot].data);
            std::copy(window.tokens.begin(), window.tokens.end(), uploads[slot].data + limits.latents);
            hip_ok(hipMemcpyAsync(ids[slot].data, uploads[slot].data, window.latents.size() * sizeof(std::uint32_t),
                hipMemcpyHostToDevice, copy.value), "kvshard prefill union IDs");
            hip_ok(hipMemcpyAsync(ids[slot].data + limits.latents, uploads[slot].data + limits.latents,
                window.tokens.size() * sizeof(std::uint32_t), hipMemcpyHostToDevice, copy.value), "kvshard prefill remapped tokens");
            hip_ok(hipEventRecord(ids_ready[slot].value, copy.value), "kvshard IDs ready");
            hip_ok(hipStreamWaitEvent(call.stream, ids_ready[slot].value, 0), "kvshard IDs consume");
            const unsigned latents = static_cast<unsigned>(window.latents.size());
            selected_latent_kernel<<<(latents * kKvLora + 255) / 256, 256, 0, call.stream>>>(
                state.latent, ids[slot].data, latents, latent_local.data, impl_->rank);
            hip_ok(hipGetLastError(), "kvshard prefill selected pack");
            // Release only the preceding receive H2D source, not its MLA work.
            if (!remote.empty()) hip_ok(hipEventSynchronize(peer_copied.value), "kvshard peer source reuse");
            std::vector<unsigned char>().swap(remote);
            const std::size_t bytes = static_cast<std::size_t>(latents) * kKvLora * sizeof(std::uint16_t);
            remote = impl_->wire.bulk({K::layout_id, impl_->epoch, state.shard_layer, start + begin + row, 7, bytes, window.rows},
                                       latent_local.data, bytes);
            hip_ok(hipMemcpyAsync(latent_peer.data, remote.data(), bytes, hipMemcpyHostToDevice, call.stream), "kvshard prefill selected peer");
            hip_ok(hipEventRecord(peer_copied.value, call.stream), "kvshard peer copied");
            merge_selected_kernel<<<(latents * kKvLora + 255) / 256, 256, 0, call.stream>>>(
                latent_local.data, latent_peer.data, latent_full.data, ids[slot].data, latents, impl_->rank);
            hip_ok(hipGetLastError(), "kvshard prefill selected merge");
            const dev::DsaSlot selected{latent_full.data, nullptr, nullptr, nullptr, limits.latents};
            const std::size_t offset = static_cast<std::size_t>(begin + row) * weights.heads * kMlaDim;
            if (work.attention_window == 0)
                pf::mla_attention_rows(selected, ids[slot].data + limits.latents, work.counts + begin + row, work.query + offset,
                    weights.kv_b, weights.heads, work.qt, work.scores, work.obar, work.output + offset, window.rows, call.stream);
            else
                pf::mla_attention_rows(selected, ids[slot].data + limits.latents, work.counts + begin + row, work.query + offset,
                    weights.kv_b, weights.heads, work.qt, work.scores, work.obar, work.output + offset, window.rows, call.stream,
                    work.attention_window);
            hip_ok(hipEventRecord(consumed[slot].value, call.stream), "kvshard IDs consumed");
            used[slot] = true;
            row += window.rows;
        }
        begin += rows;
        // The CPU has consumed this page; prefetch two pages ahead while the
        // final window runs. Each selected token crosses D2H exactly once.
        if (count - begin > limits.rows) read_page(begin + limits.rows, page_slot);
    }
    impl_->wire.trim(); // one final release, not one full-stream wait per window
}
} // namespace strix::glm::engine
