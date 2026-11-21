#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include "core/bf16_round_device.hpp"
#include "ops/norm/row_reduction.hpp"
#include "strix/attn/cache_rows.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

namespace strix::core::prepared_rows_device {

#if defined(__HIPCC__)
using Status = hipError_t;
using Stream = hipStream_t;
inline constexpr auto invalid = hipErrorInvalidValue;
inline Status last_error() { return hipGetLastError(); }
#else
using Status = cudaError_t;
using Stream = cudaStream_t;
inline constexpr auto invalid = cudaErrorInvalidValue;
inline Status last_error() { return cudaGetLastError(); }
#endif

template <class Transform>
__global__ void elements_kernel(std::size_t size, Transform transform) {
    const std::size_t start = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
    for (std::size_t i = start; i < size; i += stride) transform(i);
}

template <class Transform>
Status elements(std::size_t size, Transform transform, Stream stream) {
    if (size == 0 || size > std::numeric_limits<std::size_t>::max() / sizeof(float)) return invalid;
    constexpr unsigned threads = 256;
    constexpr unsigned max_blocks = 65535;
    const auto blocks = (size - 1) / threads + 1;
    elements_kernel<<<blocks < max_blocks ? static_cast<unsigned>(blocks) : max_blocks, threads, 0, stream>>>(
        size, transform);
    return last_error();
}

struct Bf16Round {
    float* row;
    __device__ void operator()(std::size_t i) const { row[i] = round_bf16(row[i]); }
};

// Test the stored representation, matching a subsequent finite scan.
__device__ inline void reject_nonfinite(float value, unsigned* failure) {
    if (!attn::cache_rows::finite(value)) atomicOr(failure, 1U);
}

struct CheckedBf16Round {
    float* row;
    unsigned* failure;
    __device__ void operator()(std::size_t i) const {
        const float value = round_bf16(row[i]);
        row[i] = value;
        reject_nonfinite(value, failure);
    }
};

struct Bf16SigmoidProduct {
    const float* input;
    const float* gate_logits;
    float* output;
    __device__ void operator()(std::size_t i) const {
        const float gate = round_bf16(gate_logits[i]);
        const float sigmoid = __fdiv_rn(1.0F, __fadd_rn(1.0F, expf(-gate)));
        output[i] = round_bf16(__fmul_rn(round_bf16(input[i]), sigmoid));
    }
};

struct Bf16Softcap {
    float* row;
    float cap;
    __device__ void operator()(std::size_t i) const {
        const float value = round_bf16(row[i]);
        row[i] = isfinite(value) ? __fmul_rn(cap, tanhf(__fdiv_rn(value, cap))) : value;
    }
};

// A short projection into a large output row. Each lane owns one output
// and reads along contiguous columns in the checkpoint's [K][N] matrix.
// Keep the ascending K FMA chain and round once, after the projection.
template <unsigned Inputs, bool CheckFinite = false>
struct ShortBf16Projection {
    const float* input;
    const std::uint16_t* weights;
    float* output;
    unsigned outputs;
    unsigned* failure = nullptr;

    __device__ void operator()(std::size_t i) const {
        const auto* row = input + (i / outputs) * Inputs;
        const auto column = i % outputs;
        float result = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < Inputs; ++k) {
            const auto bits = weights[std::size_t{k} * outputs + column];
            const float operand = weight::scalar_float::u2f(std::uint32_t{bits} << 16U);
            result = __builtin_fmaf(row[k], operand, result);
        }
        const float rounded = round_bf16(result);
        output[i] = rounded;
        if constexpr (CheckFinite) reject_nonfinite(rounded, failure);
    }
};

// Widen canonical [lag][channel] weights, or convert an input stored in
// [channel][tap] order with the earliest relative time first.
struct WidenConvolutionBf16 {
    const std::uint16_t* input;
    float* output;
    unsigned channels;
    unsigned taps;
    bool channel_taps;
    __device__ void operator()(std::size_t i) const {
        const auto from = channel_taps ? (i % channels) * taps + (taps - 1U - i / channels) : i;
        output[i] = weight::scalar_float::u2f(std::uint32_t{input[from]} << 16U);
    }
};

struct LogPositionScale {
    unsigned first = 0;
    unsigned floor = 0;  // Zero disables the positional scale.
    float alpha = 0.0F;
};

struct Bf16HeadClipArgs {
    const float* input;
    const std::uint16_t* gain;  // [group][width], optional
    float* output;
    unsigned rows;
    unsigned groups;
    unsigned width;
    float epsilon;
    float threshold;
    LogPositionScale position;
};

// Use the shared LDS reduction with an L2 clip epilogue on BF16-rounded inputs. The
// optional per-head gain follows the BF16 projection boundary and precedes
// clipping. The clip output rounds to BF16 before optional position scaling.
template <unsigned Threads>
__global__ void clip_kernel(Bf16HeadClipArgs args) {
    __shared__ float reductions[Threads];
    const auto group = blockIdx.x;
    const auto lane = threadIdx.x;
    for (unsigned row = blockIdx.y; row < args.rows; row += gridDim.y) {
        const auto base = (std::size_t{row} * args.groups + group) * args.width;
        float value = 0.0F;
        if (lane < args.width) {
            const auto gain_offset = std::size_t{group} * args.width + lane;
            const float gain = args.gain
                ? weight::scalar_float::u2f(std::uint32_t{args.gain[gain_offset]} << 16U) : 1.0F;
            value = __fmul_rn(round_bf16(args.input[base + lane]), gain);
        }
        const float square = ops::norm::lds_halving_sum<Threads>(reductions, __fmul_rn(value, value));
        const float scale = fminf(1.0F, __fdiv_rn(args.threshold, __fadd_rn(__fsqrt_rn(square), args.epsilon)));
        if (lane < args.width) {
            float clipped = round_bf16(__fmul_rn(value, scale));
            const unsigned absolute = args.position.first + row;
            if (args.position.floor != 0 && absolute >= args.position.floor) {
                const float ratio = __fdiv_rn(static_cast<float>(absolute + 1U), static_cast<float>(args.position.floor));
                const float log_scale = __fadd_rn(1.0F, __fmul_rn(args.position.alpha, logf(ratio)));
                clipped = round_bf16(__fmul_rn(clipped, log_scale));
            }
            args.output[base + lane] = clipped;
        }
    }
}

// Contiguous rows. In-place operation is safe when input and output match exactly.
inline Status l2_clip_bf16(const Bf16HeadClipArgs& args, Stream stream) {
    constexpr unsigned max_rows = std::numeric_limits<unsigned>::max() - 65535U;
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.input || !args.output || args.rows == 0 || args.rows > max_rows || args.groups == 0 ||
        args.groups > 0x7FFFFFFFU || args.width == 0 || args.width > 256 ||
        args.groups > max_floats / args.rows / args.width ||
        !(args.epsilon > 0.0F) || !std::isfinite(args.epsilon) ||
        !(args.threshold > 0.0F) || !std::isfinite(args.threshold) ||
        (args.position.floor != 0 && (!std::isfinite(args.position.alpha) ||
                                     args.rows > std::numeric_limits<unsigned>::max() - args.position.first))) {
        return invalid;
    }
    const dim3 grid(args.groups, args.rows < 65535U ? args.rows : 65535U);
    if (args.width <= 32U) clip_kernel<32><<<grid, 32, 0, stream>>>(args);
    else if (args.width <= 128U) clip_kernel<128><<<grid, 128, 0, stream>>>(args);
    else clip_kernel<256><<<grid, 256, 0, stream>>>(args);
    return last_error();
}

}  // namespace strix::core::prepared_rows_device
