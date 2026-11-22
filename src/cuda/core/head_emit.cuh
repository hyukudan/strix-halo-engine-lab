#pragma once

#include "head_argmax.cuh"
#include "head_emit.hpp"

#include <math_constants.h>

#include <type_traits>

namespace strix::cuda::core::head {

namespace greedy = ::strix::cuda::core::head_argmax;
using Launch = greedy::Launch;

struct EmitIds {
    std::uint32_t* out;
    ha::Pick* workspace;
    bool refuse_nonfinite = false;
    std::uint32_t chunks = 0;
};

struct EmitPick {  // value, ID and nonfinite count for engine adapters
    ha::Pick* out;
    ha::Pick* workspace;
    std::uint32_t chunks = 0;

    __device__ void operator()(std::uint32_t row, const ha::Pick& pick) const { out[row] = pick; }
};

struct EmitTop2 {
    std::uint32_t* out;
    std::uint32_t* second;
    float* gap;  // optional
    ha::Pick2* workspace;
    std::uint32_t chunks = 0;
};

struct EmitCandidates {
    Candidate* out;       // device buffer of rows * top entries, no host readback required
    std::uint32_t top;    // 1..64, ordered by descending value then ascending ID
    RowLse* lse = nullptr;  // optional compact output from the same passes
};

struct EmitLse { RowLse* out; };

// Incumbent draft confidence: lowest shortlist slot wins. Apply any shortlist-to-token map
// after the pick. This uses the draft's 1024 lanes and __expf, distinct from EmitLse.
struct EmitDraftConfidence { DraftChoice* out; };

namespace detail {

template <bool Two>
__global__ void draft_remap(std::uint32_t* top, const std::uint32_t* group_of, std::uint32_t rows, std::uint32_t group,
                            std::uint32_t* feed_out, std::uint32_t* top2, std::uint32_t* feed_out2,
                            const float* gap, std::uint32_t* feed_gap) {
    if constexpr (Two) {
        if (gap && feed_gap && threadIdx.x + 1 == rows) *feed_gap = __float_as_uint(gap[threadIdx.x]);
    }
    if (threadIdx.x < rows) {
        std::uint32_t t = top[threadIdx.x];
        if (group_of) {
            t = group_of[t / group] * group + t % group;
            top[threadIdx.x] = t;
        }
        if (feed_out && threadIdx.x + 1 == rows) *feed_out = t;
        if (Two && top2) {
            std::uint32_t u = top2[threadIdx.x];
            if (group_of && u != ha::kNone) {
                u = group_of[u / group] * group + u % group;
                top2[threadIdx.x] = u;
            }
            if (feed_out2 && threadIdx.x + 1 == rows) *feed_out2 = u;
        }
    }
}

inline bool valid_rows(Rows& v) {
    if (v.stride == 0U) v.stride = v.count;
    return v.logits && v.count != 0U && v.stride >= v.count && v.count <= ha::kNone - v.first &&
           (std::uint64_t{v.rows - 1U} * v.stride + v.count) <= SIZE_MAX / sizeof(float);
}

inline bool valid_greedy(Rows& v, std::uint32_t chunks) {
    // The incumbent uses a grid.y for rows and uint32_t chunk/scan arithmetic.
    return valid_rows(v) && v.rows <= 65535U && chunks <= ha::kMaxChunks &&
           v.count <= ha::kNone - ha::kMaxChunks - greedy::detail::kThreads;
}

// Lifted from src/cuda/logprob_partials.cu. Preserve the input order per lane and the fold tree.
// For the shortlist-confidence trait the fold geometry and exponential follow qwen_mtp_draft.cu.
constexpr unsigned kCandidatePool = 2048;
constexpr unsigned kLogprobThreads = ::strix::kLogprobPartialLanes;
constexpr unsigned kDraftThreads = 1024;

struct RowStats {
    float maximum;
    unsigned nonfinite;
};

template <unsigned Threads, unsigned Batch, class Visit>
__device__ __forceinline__ void scan_row(const float* row, std::uint32_t count, Visit visit) {
    for (std::uint64_t base = threadIdx.x; base < count; base += Threads * Batch) {
        float values[Batch];
#pragma unroll
        for (unsigned u = 0; u < Batch; ++u) {
            const auto i = base + u * Threads;
            values[u] = i < count ? row[i] : 0.0F;
        }
#pragma unroll
        for (unsigned u = 0; u < Batch; ++u) {
            const auto i = base + u * Threads;
            if (i < count) visit(values[u], static_cast<std::uint32_t>(i));
        }
    }
}

template <unsigned Threads, class T, class Combine>
__device__ __forceinline__ void fold(T* values, T local, Combine combine) {
    const unsigned t = threadIdx.x;
    values[t] = local;
    __syncthreads();
    for (unsigned half = Threads / 2U; half > 0U; half /= 2U) {
        if (t < half) values[t] = combine(values[t], values[t + half]);
        __syncthreads();
    }
}

__device__ __forceinline__ bool ahead(Candidate a, Candidate b) {
    return ha::better(a.value, a.id, b.value, b.id);
}

__device__ __forceinline__ bool finite(float x) {
    return (__float_as_uint(x) & 0x7f800000U) != 0x7f800000U;
}

template <bool Candidates, bool Lse>
__device__ __forceinline__ void emit_scoring(const float* row, Rows v, EmitCandidates emit) {
    __shared__ Candidate lane[Candidates ? kLogprobThreads : 1U], pool[Candidates ? kCandidatePool : 1U];
    __shared__ Candidate threshold;
    __shared__ unsigned pool_count;
    __shared__ float values[Lse ? kLogprobThreads : 1U];
    __shared__ RowStats stats[Lse ? kLogprobThreads : 1U];
    const unsigned t = threadIdx.x;
    Candidate best = kNoCandidate;
    float maximum = -CUDART_INF_F;
    unsigned bad = 0U;
    scan_row<kLogprobThreads, 32>(row, v.count, [&](float x, std::uint32_t i) {
        if (!finite(x)) {
            if constexpr (Lse) ++bad;
            return;
        }
        if constexpr (Lse) maximum = fmaxf(maximum, x);
        if constexpr (Candidates) {
            const Candidate entry{x, v.first + i};
            if (ahead(entry, best)) best = entry;
        }
    });
    if constexpr (Lse) {
        fold<kLogprobThreads>(stats, RowStats{maximum, bad}, [](RowStats a, RowStats b) {
            return RowStats{fmaxf(a.maximum, b.maximum), a.nonfinite + b.nonfinite};
        });
        maximum = stats[0].maximum;
    }
    Candidate cut = kNoCandidate;
    if constexpr (Candidates) {
        // The Kth lane winner is a lower bound on the row's Kth entry. Retain all entries at or ahead of it.
        lane[t] = best;
        if (t == 0U) { threshold = kNoCandidate; pool_count = 0U; }
        __syncthreads();
        unsigned rank = 0;
        for (unsigned i = 0; i < kLogprobThreads; ++i) rank += ahead(lane[i], best) ? 1U : 0U;
        if (best.id != ha::kNone && rank == emit.top - 1U) threshold = best;
        __syncthreads();
        cut = threshold;
    }
    float sum = 0.0F;
    scan_row<kLogprobThreads, 32>(row, v.count, [&](float x, std::uint32_t i) {
        if (!finite(x)) return;
        if constexpr (Lse) sum += expf(x - maximum);
        if constexpr (Candidates) {
            const Candidate entry{x, v.first + i};
            if (!ahead(cut, entry)) {
                const auto slot = atomicAdd(&pool_count, 1U);
                if (slot < kCandidatePool) pool[slot] = entry;
            }
        }
    });
    if constexpr (Lse) {
        fold<kLogprobThreads>(values, sum, [](float a, float b) { return a + b; });
        if (t == 0U) emit.lse[blockIdx.x] = {maximum, values[0], maximum + logf(values[0]), stats[0].nonfinite};
    }
    if constexpr (Candidates) {
        if constexpr (!Lse) __syncthreads();  // the first barrier of the sum fold also publishes the pool
        const unsigned count = pool_count;
        const unsigned want = count < emit.top ? count : emit.top;
        auto* out = emit.out + std::size_t{blockIdx.x} * emit.top;
        if (t >= want && t < emit.top) out[t] = kNoCandidate;
        if (count <= kCandidatePool) {
            for (unsigned slot = t; slot < count; slot += kLogprobThreads) {
                unsigned position = 0;
                for (unsigned i = 0; i < count; ++i) position += ahead(pool[i], pool[slot]) ? 1U : 0U;
                if (position < want) out[position] = pool[slot];
            }
            return;
        }

        // Pool overflow: rescan exactly, restricted to entries behind the preceding winner.
        Candidate previous = kNoCandidate;
        for (unsigned rank_out = 0; rank_out < want; ++rank_out) {
            best = kNoCandidate;
            for (std::uint64_t i = t; i < v.count; i += kLogprobThreads) {
                const Candidate entry{row[i], v.first + static_cast<std::uint32_t>(i)};
                if (finite(entry.value) && (rank_out == 0U || ahead(previous, entry)) && ahead(entry, best)) best = entry;
            }
            fold<kLogprobThreads>(lane, best, [](Candidate a, Candidate b) { return ahead(b, a) ? b : a; });
            previous = lane[0];
            if (t == 0U) out[rank_out] = previous;
            __syncthreads();
        }
    }
}

template <class Emit, bool ScoreLse = false,
          unsigned Threads = std::is_same_v<Emit, EmitDraftConfidence> ? kDraftThreads : kLogprobThreads>
__global__ __launch_bounds__(Threads)
void reduce_rows(Rows v, Emit emit) {
    const float* row = v.logits + std::size_t{blockIdx.x} * v.stride;
    if constexpr (std::is_same_v<Emit, EmitCandidates>) {
        emit_scoring<true, ScoreLse>(row, v, emit);
    } else if constexpr (std::is_same_v<Emit, EmitLse>) {
        emit_scoring<false, true>(row, v, {nullptr, 0U, emit.out});
    } else {
        __shared__ Candidate picks[Threads];
        __shared__ float values[Threads];
        Candidate best{-CUDART_INF_F, ha::kNone};
        scan_row<Threads, 16>(row, v.count, [&](float x, std::uint32_t i) {
            if (x > best.value) best = {x, v.first + i};
        });
        fold<Threads>(picks, best, [](Candidate a, Candidate b) { return ahead(b, a) ? b : a; });
        const float maximum = picks[0].value;
        float sum = 0.0F;
        scan_row<Threads, 16>(row, v.count, [&](float x, std::uint32_t) { sum += __expf(x - maximum); });
        fold<Threads>(values, sum, [](float a, float b) { return a + b; });
        if (threadIdx.x == 0U) {
            const auto id = picks[0].id;
            emit.out[blockIdx.x] = {id, id == ha::kNone ? 0.0F : 1.0F / values[0]};
        }
    }
}

}  // namespace detail

template <Launch Schedule = Launch::plain>
cudaError_t enqueue(Rows v, EmitIds emit, cudaStream_t stream) {
    if (v.rows == 0U) return cudaSuccess;
    if (!detail::valid_greedy(v, emit.chunks) || !emit.out || !emit.workspace) return cudaErrorInvalidValue;
    const greedy::View pick{v.logits, v.stride, v.count, v.rows, v.first, emit.chunks, emit.workspace};
    if (emit.refuse_nonfinite) return greedy::pick_rows<Schedule>(pick, greedy::EmitIndexOrNone{emit.out}, stream);
    return greedy::pick_rows<Schedule>(pick, greedy::EmitIndex{emit.out}, stream);
}

template <Launch Schedule = Launch::plain>
cudaError_t enqueue(Rows v, EmitPick emit, cudaStream_t stream) {
    if (v.rows == 0U) return cudaSuccess;
    if (!detail::valid_greedy(v, emit.chunks) || !emit.out || !emit.workspace) return cudaErrorInvalidValue;
    const greedy::View pick{v.logits, v.stride, v.count, v.rows, v.first, emit.chunks, emit.workspace};
    return greedy::pick_rows<Schedule>(pick, emit, stream);
}

template <Launch Schedule = Launch::plain>
cudaError_t enqueue(Rows v, EmitTop2 emit, cudaStream_t stream) {
    if (v.rows == 0U) return cudaSuccess;
    if (!detail::valid_greedy(v, emit.chunks) || !emit.out || !emit.second || !emit.workspace) return cudaErrorInvalidValue;
    const greedy::View pick{v.logits, v.stride, v.count, v.rows, v.first, emit.chunks, nullptr};
    return greedy::pick2_rows<Schedule>(pick, emit.workspace, greedy::EmitIndex2{emit.out, emit.second, emit.gap}, stream);
}

template <class Emit>
    requires (std::is_same_v<Emit, EmitCandidates> || std::is_same_v<Emit, EmitLse> ||
              std::is_same_v<Emit, EmitDraftConfidence>)
cudaError_t enqueue(Rows v, Emit emit, cudaStream_t stream) {
    if (v.rows == 0U) return cudaSuccess;
    if (!detail::valid_rows(v) || !emit.out || v.rows > 0x7fffffffU) return cudaErrorInvalidValue;
    if constexpr (std::is_same_v<Emit, EmitCandidates>) {
        if (emit.top == 0U || emit.top > kMaxCandidates) return cudaErrorInvalidValue;
        if (emit.lse) {
            detail::reduce_rows<Emit, true><<<v.rows, detail::kLogprobThreads, 0, stream>>>(v, emit);
            return cudaGetLastError();
        }
    }
    constexpr unsigned threads = std::is_same_v<Emit, EmitDraftConfidence> ? detail::kDraftThreads : detail::kLogprobThreads;
    detail::reduce_rows<<<v.rows, threads, 0, stream>>>(v, emit);
    return cudaGetLastError();
}

// Post-pick trait for compact group IDs and/or device feed. The map rewrites IDs in place.
// Each feed pointer is optional and receives the last row only. A gap is copied as bits.
template <bool Two = false>
cudaError_t enqueue_draft_remap(std::uint32_t* top, const std::uint32_t* group_of, std::uint32_t rows, std::uint32_t group,
                                std::uint32_t* feed_out, cudaStream_t stream, std::uint32_t* top2 = nullptr,
                                std::uint32_t* feed_out2 = nullptr, const float* gap = nullptr,
                                std::uint32_t* feed_gap = nullptr) {
    if (rows == 0U) return cudaSuccess;
    if (rows > 32U || !top || (group_of && group == 0U)) return cudaErrorInvalidValue;
    detail::draft_remap<Two><<<1, 32, 0, stream>>>(top, group_of, rows, group, feed_out, top2, feed_out2, gap, feed_gap);
    return cudaGetLastError();
}

}  // namespace strix::cuda::core::head
