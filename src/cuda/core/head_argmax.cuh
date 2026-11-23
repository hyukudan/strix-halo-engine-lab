#pragma once
// CUDA body of the greedy head pick (contract: include/strix/head_argmax.hpp). Two launches per call:
// partial: grid (chunks, rows), each block scans one contiguous chunk of one row (many loads in flight across the GPU
// instead of one block per row walking ~1 MB); final: one block per row merges the chunk picks and hands the row's Pick
// to the engine's Emit adapter (its output words). Identity: any chunk count gives the same Pick (merge is associative
// and commutative; within a thread indices only grow, so the strict compare keeps the first maximum).
// Launch form is a trait of the caller (Launch): plain <<<>>> launches, or programmatic dependent launch (contract in
// src/cuda/core/common.cuh) where both kernels wait on their predecessor before their first dependent read.
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "common.cuh"
#include "step_rows.cuh"
#include "strix/head_argmax.hpp"

namespace strix::cuda::core::head_argmax {

namespace ha = ::strix::head_argmax;

struct View {
    const float* logits = nullptr;
    std::uint64_t stride = 0;       // floats between rows (0 = count)
    std::uint32_t count = 0;        // logits per row
    std::uint32_t rows = 0;
    std::uint32_t first_index = 0;  // global index of position 0 (vocabulary slices)
    std::uint32_t chunks = 0;       // 0 = ha::default_chunks(count); <= ha::kMaxChunks
    ha::Pick* partial = nullptr;    // ha::workspace_bytes(rows, chunks) of device memory
};

// plain: both kernels launched with <<<>>> (stream order). pdl: both launched with launch_pdl (programmatic dependent
// launch when pdl_enabled()); the partial kernel triggers its dependents and waits before reading the logits, the
// final kernel waits before reading the partial picks and does not trigger early, so a dependent kernel's own pdl_wait
// covers the emitted words.
// Pick type is a trait too: ha::Pick (top-1, every engine's head pick) or ha::Pick2 (runner-up trait: the same top-1 plus
// the second in the same order; used where a drafter asks for runner-ups). The Pick instantiations are the top-1 bodies
// unchanged.
enum class Launch { plain, pdl };

namespace detail {
inline constexpr unsigned kThreads = 256;

__device__ __forceinline__ ha::Pick2 shfl_xor(const ha::Pick2& p, unsigned d) {
    ha::Pick2 o;
    o.first.value = __shfl_xor_sync(0xffffffffU, p.first.value, d);
    o.first.index = __shfl_xor_sync(0xffffffffU, p.first.index, d);
    o.first.nonfinite = __shfl_xor_sync(0xffffffffU, p.first.nonfinite, d);
    o.value2 = __shfl_xor_sync(0xffffffffU, p.value2, d);
    o.index2 = __shfl_xor_sync(0xffffffffU, p.index2, d);
    return o;
}

__device__ __forceinline__ ha::Pick2 block_merge(ha::Pick2 p) {  // runner-up trait of block_merge below
    for (unsigned d = 16U; d; d >>= 1U) p = ha::merge2(p, shfl_xor(p, d));
    __shared__ ha::Pick2 warp_pick2[kThreads / 32U];
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    if (lane == 0U) warp_pick2[warp] = p;
    __syncthreads();
    if (warp == 0U) {
        p = lane < kThreads / 32U ? warp_pick2[lane] : ha::Pick2{};
        for (unsigned d = 16U; d; d >>= 1U) p = ha::merge2(p, shfl_xor(p, d));
    }
    return p;  // valid in thread 0
}

__device__ __forceinline__ ha::Pick block_merge(ha::Pick p) {
    for (unsigned d = 16U; d; d >>= 1U) {
        ha::Pick o;
        o.value = __shfl_xor_sync(0xffffffffU, p.value, d);
        o.index = __shfl_xor_sync(0xffffffffU, p.index, d);
        o.nonfinite = __shfl_xor_sync(0xffffffffU, p.nonfinite, d);
        p = ha::merge(p, o);
    }
    __shared__ ha::Pick warp_pick[kThreads / 32U];
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    if (lane == 0U) warp_pick[warp] = p;
    __syncthreads();
    if (warp == 0U) {
        p = lane < kThreads / 32U ? warp_pick[lane] : ha::Pick{};
        for (unsigned d = 16U; d; d >>= 1U) {
            ha::Pick o;
            o.value = __shfl_xor_sync(0xffffffffU, p.value, d);
            o.index = __shfl_xor_sync(0xffffffffU, p.index, d);
            o.nonfinite = __shfl_xor_sync(0xffffffffU, p.nonfinite, d);
            p = ha::merge(p, o);
        }
    }
    return p;  // valid in thread 0
}

template <Launch L, class P = ha::Pick>
__global__ __launch_bounds__(kThreads) void partial_kernel(View v) {
    if constexpr (L == Launch::pdl) {
        ::strix::cuda_core::pdl_trigger();
        (void)::strix::cuda_core::pdl_wait();  // the logits are the predecessor's output
    }
    constexpr bool Masked = false;
    [[maybe_unused]] const StepRows<1> binding{};
#include "head_argmax_partial.inc"
}

template <Launch L, class Emit, class P = ha::Pick>
__global__ __launch_bounds__(kThreads) void final_kernel(View v, Emit emit) {
    if constexpr (L == Launch::pdl) (void)::strix::cuda_core::pdl_wait();  // partial picks of the partial kernel
    constexpr bool Masked = false;
    [[maybe_unused]] const StepRows<1> binding{};
#include "head_argmax_final.inc"
}
template <unsigned Width, class P = ha::Pick2>
__global__ __launch_bounds__(kThreads) void partial_step_kernel(View v, StepRows<Width> binding) {
    constexpr bool Masked = true;
#include "head_argmax_partial.inc"
}

template <unsigned Width, class Emit, class P = ha::Pick2>
__global__ __launch_bounds__(kThreads) void final_step_kernel(View v, Emit emit, StepRows<Width> binding) {
    constexpr bool Masked = true;
#include "head_argmax_final.inc"
}

inline bool prepare_split_view(View& v) {
    if (v.stride == 0U) v.stride = v.count;
    if (v.chunks == 0U) v.chunks = ha::default_chunks(v.count);
    return v.logits && v.partial && v.count != 0U && v.chunks <= ha::kMaxChunks && v.stride >= v.count;
}
}  // namespace detail

// The caller orders the partial launch before the final launch.
template <Launch L = Launch::plain, class P = ha::Pick>
cudaError_t pick_partial(View v, cudaStream_t stream) {
    if (v.rows == 0U) return cudaSuccess;
    if (!detail::prepare_split_view(v)) return cudaErrorInvalidValue;
    const dim3 grid(v.chunks, v.rows), block(detail::kThreads);
    if constexpr (L == Launch::pdl) {
        return ::strix::cuda_core::launch_pdl(detail::partial_kernel<L, P>, grid, block, stream, v);
    } else {
        detail::partial_kernel<L, P><<<grid, block, 0, stream>>>(v);
        return cudaGetLastError();
    }
}

template <Launch L = Launch::plain, class P = ha::Pick, class Emit>
cudaError_t pick_final(View v, Emit emit, cudaStream_t stream) {
    if (v.rows == 0U) return cudaSuccess;
    if (!detail::prepare_split_view(v)) return cudaErrorInvalidValue;
    const dim3 grid(v.rows), block(detail::kThreads);
    if constexpr (L == Launch::pdl) {
        return ::strix::cuda_core::launch_pdl(detail::final_kernel<L, Emit, P>, grid, block, stream, v, emit);
    } else {
        detail::final_kernel<L, Emit, P><<<grid, block, 0, stream>>>(v, emit);
        return cudaGetLastError();
    }
}

// Enqueue the pick of every row; emit(row, pick) runs once per row on the device. rows == 0 enqueues nothing.
template <Launch L = Launch::plain, class P = ha::Pick, class Emit>
cudaError_t pick_rows(View v, Emit emit, cudaStream_t stream) {
    if (v.rows == 0U) return cudaSuccess;
    if (v.stride == 0U) v.stride = v.count;
    if (v.chunks == 0U) v.chunks = ha::default_chunks(v.count);
    if (!v.logits || !v.partial || v.count == 0U || v.chunks > ha::kMaxChunks || v.stride < v.count)
        return cudaErrorInvalidValue;
    const dim3 partial_grid(v.chunks, v.rows), block(detail::kThreads);
    if constexpr (L == Launch::pdl) {
        namespace cc = ::strix::cuda_core;
        if (auto e = cc::launch_pdl(detail::partial_kernel<L, P>, partial_grid, block, stream, v); e != cudaSuccess) return e;
        return cc::launch_pdl(detail::final_kernel<L, Emit, P>, dim3(v.rows), block, stream, v, emit);
    } else {
        detail::partial_kernel<L, P><<<partial_grid, block, 0, stream>>>(v);
        detail::final_kernel<L, Emit, P><<<v.rows, block, 0, stream>>>(v, emit);
        return cudaGetLastError();
    }
}

// Common adapters.
struct EmitIndex {  // out[row] = index (kNone when the row has no finite logit)
    std::uint32_t* out;
    __device__ void operator()(std::uint32_t row, const ha::Pick& p) const { out[row] = p.index; }
};
struct EmitIndexOrNone {  // out[row] = index, or kNone when any logit of the row is non-finite
    std::uint32_t* out;
    __device__ void operator()(std::uint32_t row, const ha::Pick& p) const { out[row] = p.nonfinite ? ha::kNone : p.index; }
};

// Runner-up trait: the same two launches with ha::Pick2 partials. partial2 = ha::workspace2_bytes(rows, chunks) of device
// memory (v.partial is overwritten). Emit receives the row's Pick2; its first is the top-1 Pick bit for bit.
template <Launch L = Launch::plain, class Emit>
cudaError_t pick2_rows(View v, ha::Pick2* partial2, Emit emit, cudaStream_t stream) {
    v.partial = reinterpret_cast<ha::Pick*>(partial2);  // carried in the partial word pointer, read back as Pick2
    return pick_rows<L, ha::Pick2>(v, emit, stream);
}
// out[row] = top-1 index (== EmitIndex), out2[row] = runner-up index (kNone if none); gap (optional): top-1 value -
// runner-up value (log p1 / p2 for logits; NaN when there is no runner-up).
struct EmitIndex2 {
    std::uint32_t* out;
    std::uint32_t* out2;
    float* gap;
    __device__ void operator()(std::uint32_t row, const ha::Pick2& p) const {
        out[row] = p.first.index;
        out2[row] = p.index2;
        if (gap) gap[row] = p.index2 != ha::kNone ? __fsub_rn(p.first.value, p.value2) : __int_as_float(0x7fc00000);
    }
};

}  // namespace strix::cuda::core::head_argmax
