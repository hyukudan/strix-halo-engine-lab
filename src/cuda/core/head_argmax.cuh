#pragma once
// CUDA body of the greedy head pick (contract: include/strix/head_argmax.hpp). Two launches per call:
// partial: grid (chunks, rows), each block scans one contiguous chunk of one row (many loads in flight across the GPU
// instead of one block per row walking ~1 MB); final: one block per row merges the chunk picks and hands the row's Pick
// to the engine's Emit adapter (its output words). Identity: any chunk count gives the same Pick (merge is associative
// and commutative; within a thread indices only grow, so the strict compare keeps the first maximum).
#include <cuda_runtime.h>

#include <cstdint>

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

namespace detail {
inline constexpr unsigned kThreads = 256;

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

template <int = 0>
__global__ __launch_bounds__(kThreads) void partial_kernel(View v) {
    const std::uint32_t row = blockIdx.y, chunk = blockIdx.x;
    std::uint32_t lo = 0, hi = 0;
    ha::chunk_range(v.count, v.chunks, chunk, lo, hi);
    const float* x = v.logits + row * v.stride;
    ha::Pick p;
#pragma unroll 4
    for (std::uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
        const float y = x[i];
        const std::uint32_t g = v.first_index + i;
        if (!isfinite(y)) ++p.nonfinite;
        else if (ha::better(y, g, p.value, p.index)) { p.value = y; p.index = g; }
    }
    p = block_merge(p);
    if (threadIdx.x == 0U) v.partial[std::size_t{row} * v.chunks + chunk] = p;
}

template <class Emit>
__global__ __launch_bounds__(kThreads) void final_kernel(View v, Emit emit) {
    const std::uint32_t row = blockIdx.x;
    ha::Pick p;
    for (std::uint32_t c = threadIdx.x; c < v.chunks; c += kThreads) p = ha::merge(p, v.partial[std::size_t{row} * v.chunks + c]);
    p = block_merge(p);
    if (threadIdx.x == 0U) emit(row, p);
}
}  // namespace detail

// Enqueue the pick of every row; emit(row, pick) runs once per row on the device.
template <class Emit>
cudaError_t pick_rows(View v, Emit emit, cudaStream_t stream) {
    if (v.stride == 0U) v.stride = v.count;
    if (v.chunks == 0U) v.chunks = ha::default_chunks(v.count);
    if (!v.logits || !v.partial || v.count == 0U || v.rows == 0U || v.chunks > ha::kMaxChunks || v.stride < v.count)
        return cudaErrorInvalidValue;
    detail::partial_kernel<<<dim3(v.chunks, v.rows), detail::kThreads, 0, stream>>>(v);
    detail::final_kernel<<<v.rows, detail::kThreads, 0, stream>>>(v, emit);
    return cudaGetLastError();
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

}  // namespace strix::cuda::core::head_argmax
