#pragma once
// HIP body of the shared RoPE op (contract: rope.hpp). One thread per (vector, pair), grid.y = rows. Lifted unchanged
// from the DS41 HIP layer (Round::none instantiation = the former rope_kernel, ISA identical).
#include "ops/rope/rope.hpp"

#include <hip/hip_runtime.h>

namespace strix::ops::rope {

template <Round R>
__global__ __launch_bounds__(256) void rope_kernel(View v) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (index >= v.vectors * v.pairs) return;
    const unsigned p = index % v.pairs, vec = index / v.pairs;
    float* x = v.x + static_cast<std::size_t>(r) * v.row_stride + static_cast<std::size_t>(vec) * v.vector_stride + v.offset;
    const std::size_t t = static_cast<std::size_t>(v.positions[r]) * v.pairs + p;
    const float c = v.cos[t], s = v.inverse ? -v.sin[t] : v.sin[t];
    const float2 ab = *reinterpret_cast<const float2*>(x + 2 * p);
    float2 y;
    if constexpr (R == Round::none) {
        y.x = ab.x * c - ab.y * s;
        y.y = ab.x * s + ab.y * c;
    } else {
#pragma clang fp contract(off)
        const float ac = ab.x * c, bs = ab.y * s, as = ab.x * s, bc = ab.y * c;
        y.x = ::strix::numeric::round_to_bf16(ac - bs);
        y.y = ::strix::numeric::round_to_bf16(as + bc);
    }
    *reinterpret_cast<float2*>(x + 2 * p) = y;
}

// The caller checks the launch (each binder keeps its own error path).
template <Round R>
inline void launch(const View& v, hipStream_t stream) {
    require_view(v);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rope_kernel<R>), dim3((v.vectors * v.pairs + 255U) / 256U, v.rows), dim3(256), 0,
                       stream, v);
}

}  // namespace strix::ops::rope
