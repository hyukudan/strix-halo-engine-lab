#pragma once

#include "attn_core/select_spec.hpp"

#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

#include <cstddef>

namespace strix::cuda::core::select {
namespace detail {

inline constexpr unsigned pick_threads = 256;

__device__ __forceinline__ std::uint32_t float_key(float value, bool canonical_zero = true) {
    auto bits = __float_as_uint(value);
    if (canonical_zero && (bits & 0x7fffffffU) == 0U) bits = 0;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}

using Scan = cub::BlockScan<unsigned, pick_threads>;
struct PickScratch {
    Scan::TempStorage scan;
    unsigned hist[256];
    unsigned prefix, need, eligible;
};

struct Key {
    std::uint32_t rank;
    bool eligible;
};

// All keys pass through the same radix cutoff and stable position scan. Integer
// histograms make the tie decision independent of CUDA scheduling order.
template <class Read, class Emit>
__device__ unsigned pick(PickScratch& sh, unsigned size, unsigned top_k, bool can_exclude, Read read, Emit emit) {
    const auto tid = threadIdx.x;
    if (!top_k) return 0;
    if (size <= top_k && !can_exclude) {
        for (unsigned i = tid; i < size; i += pick_threads) emit(i, i);
        return size;
    }
    unsigned prefix = 0, mask = 0, need = top_k, total = top_k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        sh.hist[tid] = 0;
        __syncthreads();
        for (unsigned i = tid; i < size; i += pick_threads) {
            const auto k = read(i);
            if (k.eligible && (k.rank & mask) == prefix)
                atomicAdd(&sh.hist[(k.rank >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned left = need;
            sh.eligible = 0;
            for (unsigned d = 256; d != 0; --d) {
                const unsigned n = sh.hist[d - 1U];
                if (n >= left) {
                    sh.prefix = prefix | ((d - 1U) << shift);
                    sh.need = left;
                    sh.eligible = top_k;
                    break;
                }
                left -= n;
                sh.eligible += n;
            }
        }
        __syncthreads();
        total = sh.eligible;
        if (total == top_k) {
            prefix = sh.prefix;
            need = sh.need;
            mask |= 255U << shift;
        }
        __syncthreads();
        if (total < top_k) break;
    }

    unsigned equal_before = 0, out_before = top_k - total;
    for (unsigned base = 0; base < size; base += pick_threads) {
        const unsigned i = base + tid;
        const Key k = i < size ? read(i) : Key{0, false};
        const bool equal = k.eligible && k.rank == prefix;
        unsigned equal_rank, equals;
        Scan(sh.scan).ExclusiveSum(static_cast<unsigned>(equal), equal_rank, equals);
        const bool take = k.eligible &&
            (total < top_k || k.rank > prefix || (equal && equal_before + equal_rank < need));
        __syncthreads();
        unsigned rank, taken;
        Scan(sh.scan).ExclusiveSum(static_cast<unsigned>(take), rank, taken);
        if (take) emit(i, out_before + rank);
        __syncthreads();
        equal_before += equals;
        out_before += taken;
    }
    return total;
}

__device__ inline void make_pool(PickScratch& scratch, const PickRow& row) {
    const unsigned blocks = 1U + (row.groups - 1U) / row.pool_width;
    const float negative_infinity = __uint_as_float(0xff800000U);
    for (unsigned b = threadIdx.x; b < blocks; b += pick_threads) {
        float best = negative_infinity;
        const unsigned begin = b * row.pool_width, end = begin + min(row.groups - begin, row.pool_width);
        for (unsigned i = begin; i < end; ++i) best = fmaxf(best, row.scores[i]);
        row.pool_keys[b] = b + 1U == blocks ? 0xff800000U : float_key(best, false);
        row.candidates[b] = 0;
    }
    __syncthreads();
    const unsigned bottom = float_key(negative_infinity, false);
    pick(scratch, blocks, row.pool_top, false,
         [&](unsigned i) { return Key{row.pool_keys[i], true}; },
         [&](unsigned i, unsigned) { if (row.pool_keys[i] > bottom) row.candidates[i] = 1; });
    __syncthreads();
}

template <KeyOrder Order>
__global__ __launch_bounds__(pick_threads) void pick_kernel(const PickRow* rows, PickSpec spec) {
    __shared__ PickScratch scratch;
    const PickRow row = rows[blockIdx.x];
#include "cuda/core/select_pick_body.inc"
}

} // namespace detail

// rows is a device array and its storage stays live through execution. top_k,
// expansion and any enabled pool width must be positive. Source/user pools
// use raw_float. A mask aliases only its row's scores, and only for eligible_float.
// A shard join uses ordered keys with IDs ascending across the input array,
// retaining at least the global budget (or all entries) from each shard.
inline cudaError_t enqueue_pick(const PickRow* rows, unsigned row_count, PickSpec spec, cudaStream_t stream) {
    if (row_count == 0) return cudaSuccess;
    if (!rows || row_count > 2147483647U || spec.top_k == 0 || spec.expansion == 0) return cudaErrorInvalidValue;
    using detail::pick_kernel;
    constexpr auto threads = detail::pick_threads;
    switch (spec.order) {
    case KeyOrder::ordered: pick_kernel<KeyOrder::ordered><<<row_count, threads, 0, stream>>>(rows, spec); break;
    case KeyOrder::canonical_float: pick_kernel<KeyOrder::canonical_float><<<row_count, threads, 0, stream>>>(rows, spec); break;
    case KeyOrder::raw_float: pick_kernel<KeyOrder::raw_float><<<row_count, threads, 0, stream>>>(rows, spec); break;
    case KeyOrder::eligible_float: pick_kernel<KeyOrder::eligible_float><<<row_count, threads, 0, stream>>>(rows, spec); break;
    default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

} // namespace strix::cuda::core::select
