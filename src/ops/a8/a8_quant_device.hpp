#pragma once
// Shared A8 activation quantisation (HIP): int8 codes per group of 32 consecutive values, delta = amax / 127 (1 for an all-zero
// or tiny group), integer code sum per group. The bits of ds41::quantize_a8 (strix/ds41_formats.hpp) and of
// ds41::enqueue_quantize_a8 (src/hip_ds41_formats.hip quantize_a8_kernel, which calls quantize_group below).
// Producers that already hold the FP32 values of a group in one wave32 (lane = column within the group) store the A8 rows
// themselves instead of writing FP32 rows for a separate quantiser launch:
//  - NormStore: the rmsnorm Store hook (ops/norm/norm_device.hpp rmsnorm_row: thread t handles columns t + L k, so the 32
//    lanes of a wave hold one group per iteration); the norm's FP32 output row is still written;
//  - quantize_rows_kernel<Pre>: the quantiser of an elementwise pre-op (the shared glu functions sigmoid_gate / swiglu) of
//    two FP32 rows; same launch shape as quantize_a8_kernel (one wave per (group, row)).
#include "ops/glu/glu.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::ops::a8 {

struct Group {
    int code;
    float delta;
    int sum;
};

// x: this lane's value; all 32 lanes of the wave call (shuffles over the wave32).
__device__ __forceinline__ Group quantize_group(float x) {
    float amax = fabsf(x);
    for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    const float rounded = tiny ? 0.0F : rintf(x / delta);
    const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
    int sum = code;
    for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
    return {code, delta, sum};
}

// A8 rows: codes [rows][pitch], deltas / sums [rows][pitch / 32] (pitch % 32 == 0, pitch >= width).
struct Rows {
    std::int8_t* codes;
    float* deltas;
    std::int32_t* sums;
    unsigned pitch;
};

// Column `col` (lane = col % 32) of `row`.
__device__ __forceinline__ void store_group(const Rows& o, unsigned row, unsigned col, float x) {
    const Group g = quantize_group(x);
    o.codes[static_cast<std::size_t>(row) * o.pitch + col] = static_cast<std::int8_t>(g.code);
    if (col % 32U == 0U) {
        const std::size_t at = static_cast<std::size_t>(row) * (o.pitch / 32U) + col / 32U;
        o.deltas[at] = g.delta;
        o.sums[at] = g.sum;
    }
}

// rmsnorm Store hook (rmsnorm_row calls store(base + i, y) with base = row * n): needs n % 32 == 0 and a block of a
// multiple of 32 threads (every lane of a wave takes the same trip count).
struct NormStore {
    Rows out;
    unsigned n;
    __device__ void operator()(std::size_t idx, float y) const {
        store_group(out, static_cast<unsigned>(idx / n), static_cast<unsigned>(idx % n), y);
    }
};

// Elementwise pre-ops: the shared glu functions on two FP32 rows (index i = row * width + column).
template <class Exp, glu::Round R>
struct SigmoidGate {
    const float* x;
    const float* gate;
    __device__ float operator()(std::size_t i) const { return glu::sigmoid_gate<Exp, R>(x[i], gate[i]); }
};
template <class Exp, glu::Round R>
struct Swiglu {
    const float* gate;
    const float* up;
    float limit;
    __device__ float operator()(std::size_t i) const { return glu::swiglu<Exp, R>(gate[i], up[i], limit); }
};

// A8 of pre(i) over rows x width (width % 32 == 0): one wave per (group, row), as quantize_a8_kernel.
template <class Pre>
__global__ __launch_bounds__(32) void quantize_rows_kernel(Pre pre, unsigned width, Rows out) {
    const unsigned col = 32U * blockIdx.x + threadIdx.x, row = blockIdx.y;
    store_group(out, row, col, pre(static_cast<std::size_t>(row) * width + col));
}

template <class Pre>
inline void launch_quantize_rows(const Pre& pre, unsigned width, unsigned rows, const Rows& out, hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(quantize_rows_kernel<Pre>), dim3(width / 32U, rows), dim3(32), 0, stream, pre, width, out);
}

}  // namespace strix::ops::a8
