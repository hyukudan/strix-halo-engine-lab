#pragma once
// HIP bodies of the shared gated elementwise ops (contract: glu.hpp). One thread per element, 256-thread blocks.
// swiglu_kernel<DeclaredExp, Round::none> is the former DS41 trellis swiglu_f32_kernel (same signature and body).
// The caller checks the launch (each binder keeps its own error path).
#include "glu.hpp"
#include "../a8/a8_quant_device.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace strix::ops::glu {

template <class Exp, Round R, Form F = Form::div>
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, std::size_t n, float limit,
                              float* __restrict__ act) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) act[i] = swiglu<Exp, R, F>(gate[i], up[i], limit);
}

template <class Math, Round R>
__global__ void softcapped_swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, std::size_t n,
                                         float gate_cap, float up_cap, float* __restrict__ act) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) act[i] = softcapped_swiglu<Math, R>(gate[i], up[i], gate_cap, up_cap);
}

// A8 store policy: one wave32 per contiguous group, eight groups per block.
template <class Exp, Round R, Form F, class Epilogue>
__global__ __launch_bounds__(256) void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                                     unsigned group_count, float limit, std::int8_t* __restrict__ codes,
                                                     float* __restrict__ deltas, std::int32_t* __restrict__ sums) {
    const unsigned group = blockIdx.x * 8U + threadIdx.x / 32U;
    if (group >= group_count) return;
    const unsigned lane = threadIdx.x % 32U;
    const std::size_t i = std::size_t{group} * 32U + lane;
    const auto q = Epilogue::template apply<Exp, R, F>(gate[i], up[i], limit, [](float x) { return a8::quantize_group(x); });
    codes[i] = static_cast<std::int8_t>(q.code);
    if (lane == 0U) {
        deltas[group] = q.delta;
        sums[group] = q.sum;
    }
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
template <class Exp, Round R, Form F = Form::div>
inline void launch_swiglu(const float* gate, const float* up, std::size_t n, float limit, float* act, hipStream_t stream) {
    if (n == 0U) return;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(swiglu_kernel<Exp, R, F>), dim3(blocks_of(n)), dim3(256), 0, stream, gate, up, n, limit,
                       act);
}

// act[i] = softcapped_swiglu(gate[i], up[i], caps), i < n.
template <class Math, Round R>
inline void launch_softcapped_swiglu(const float* gate, const float* up, std::size_t n, float gate_cap, float up_cap,
                                     float* act, hipStream_t stream) {
    if (n == 0U) return;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(softcapped_swiglu_kernel<Math, R>), dim3(blocks_of(n)), dim3(256), 0, stream, gate,
                       up, n, gate_cap, up_cap, act);
}

// Caller admits wave32 and supplies complete groups in tightly packed rows.
template <class Exp, Round R, Form F = Form::div, class Epilogue = A8Epilogue>
inline void launch_swiglu_a8(const float* gate, const float* up, unsigned groups, float limit, std::int8_t* codes,
                             float* deltas, std::int32_t* sums, hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(swiglu_kernel<Exp, R, F, Epilogue>), dim3((groups + 7U) / 8U), dim3(256), 0,
                       stream, gate, up, groups, limit, codes, deltas, sums);
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
