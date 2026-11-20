#pragma once
// HIP body of the exact bfloat16 TP-join wire (strix/pair/bf16_wire.hpp): the packer (FP32 words -> their high halves,
// kInexactFlag on a word whose low half is not zero) and the placer (the swapped two slots of halves -> FP32 columns of
// the one-node rows, slot q -> columns [q count, (q + 1) count)). Both are stream-ordered around the gather.
// Home: shared (backend core, HIP). Bound by: Hy4 TP step joins and verify rows under TP (src/hy4/hip).
#include "strix/pair/bf16_wire.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strix::xport::bf16_wire {
namespace contract = strix::pair::bf16_wire;

template <unsigned kBlock>
__global__ __launch_bounds__(kBlock) void pack_kernel(const float* __restrict__ src, std::uint16_t* __restrict__ dst, std::size_t n,
                                                       unsigned* flag) {
    bool exact = true;
    for (std::size_t i = std::size_t{blockIdx.x} * kBlock + threadIdx.x; i < n; i += std::size_t{gridDim.x} * kBlock) {
        const unsigned b = __float_as_uint(src[i]);
        exact = exact && (b & 0xFFFFU) == 0U;
        dst[i] = static_cast<std::uint16_t>(b >> 16U);
    }
    if (!exact && flag != nullptr) atomicOr(flag, contract::kInexactFlag);
}

struct PlaceArgs {
    const std::uint16_t* wire;  // the swapped buffer: slot 0 | slot 1, `slot` halves each
    std::size_t slot, at, pitch, count;
    unsigned rows;
    float* full;                // [rows][cols]
    std::size_t cols;
};
template <unsigned kBlock>
__global__ __launch_bounds__(kBlock) void place_kernel(PlaceArgs a) {
    const std::size_t per = a.count * a.rows;
    for (std::size_t i = std::size_t{blockIdx.x} * kBlock + threadIdx.x; i < 2U * per; i += std::size_t{gridDim.x} * kBlock) {
        const std::size_t q = i / per, k = i - q * per, r = k / a.count, j = k - r * a.count;
        a.full[r * a.cols + q * a.count + j] = __uint_as_float(unsigned{a.wire[q * a.slot + a.at + r * a.pitch + j]} << 16U);
    }
}

inline unsigned blocks_for(std::size_t n) { return static_cast<unsigned>(std::clamp<std::size_t>((n + 255U) / 256U, 1U, 1024U)); }

// src[n] FP32 -> dst[n] halves (flag: nullptr or OR-ed with kInexactFlag when a word is not a bfloat16).
inline hipError_t enqueue_pack(const float* src, std::uint16_t* dst, std::size_t n, unsigned* flag, hipStream_t s) {
    if (n == 0U) return hipSuccess;
    pack_kernel<256U><<<blocks_for(n), 256U, 0, s>>>(src, dst, n, flag);
    return hipGetLastError();
}
inline hipError_t enqueue_place(const PlaceArgs& a, hipStream_t s) {
    const std::size_t n = 2U * a.count * a.rows;
    if (n == 0U) return hipSuccess;
    place_kernel<256U><<<blocks_for(n), 256U, 0, s>>>(a);
    return hipGetLastError();
}

}  // namespace strix::xport::bf16_wire
