#pragma once
// HIP bodies of the shared gated elementwise ops (contract: glu.hpp). One thread per element, 256-thread blocks.
// swiglu_kernel<DeclaredExp, Round::none> is the former DS41 trellis swiglu_f32_kernel (same signature and body).
// The caller checks the launch (each binder keeps its own error path).
#include "ops/glu/glu.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace strix::ops::glu {

template <class Exp, Round R>
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, std::size_t n, float limit,
                              float* __restrict__ act) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) act[i] = swiglu<Exp, R>(gate[i], up[i], limit);
}

// y may alias x.
template <class Exp, Round R>
__global__ void sigmoid_gate_kernel(const float* x, const float* __restrict__ gate, std::size_t n, float* y) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sigmoid_gate<Exp, R>(x[i], gate[i]);
}

// y may alias x. bias: [width] (row-broadcast) or nullptr.
template <Round R>
__global__ void epilogue_kernel(const float* x, const float* __restrict__ bias, std::size_t width, std::size_t n, float* y) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) y[i] = epilogue<R>(x[i], bias, bias ? i % width : 0U);
}

inline unsigned blocks_of(std::size_t n) { return static_cast<unsigned>((n + 255U) / 256U); }

// act[i] = swiglu(gate[i], up[i], limit), i < n (n = rows x width, contiguous).
template <class Exp, Round R>
inline void launch_swiglu(const float* gate, const float* up, std::size_t n, float limit, float* act, hipStream_t stream) {
    if (n == 0U) return;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(swiglu_kernel<Exp, R>), dim3(blocks_of(n)), dim3(256), 0, stream, gate, up, n, limit,
                       act);
}

// y[i] = sigmoid_gate(x[i], gate[i]), i < n; in place when y == x.
template <class Exp, Round R>
inline void launch_sigmoid_gate(const float* x, const float* gate, std::size_t n, float* y, hipStream_t stream) {
    if (n == 0U) return;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(sigmoid_gate_kernel<Exp, R>), dim3(blocks_of(n)), dim3(256), 0, stream, x, gate, n, y);
}

// y[i] = epilogue(x[i], bias, i % width), i < n (n = rows x width); in place when y == x.
template <Round R>
inline void launch_epilogue(const float* x, const float* bias, std::size_t width, std::size_t n, float* y, hipStream_t stream) {
    if (n == 0U) return;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(epilogue_kernel<R>), dim3(blocks_of(n)), dim3(256), 0, stream, x, bias, width, n, y);
}

}  // namespace strix::ops::glu
