#pragma once

#include "core/pointwise.hpp"
#include "ops/norm/norm_device.hpp"
#include "strix/attn/cache_rows.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <cstddef>

namespace strix::core::row_transforms {

using pointwise::Status;
using pointwise::Stream;
using pointwise::invalid;
using pointwise::elements;
using weight::scalar_float::bf16_round;

struct Bf16Round {
    float* row;
    __device__ void operator()(std::size_t i) const { row[i] = bf16_round(row[i]); }
};

// Test the stored representation, matching a subsequent finite scan.
__device__ inline void reject_nonfinite(float value, unsigned* failure) {
    if (!attn::cache_rows::finite(value)) atomicOr(failure, 1U);
}

struct CheckedBf16Round {
    float* row;
    unsigned* failure;
    __device__ void operator()(std::size_t i) const {
        const float value = bf16_round(row[i]);
        row[i] = value;
        reject_nonfinite(value, failure);
    }
};

struct Bf16SigmoidProduct {
    const float* input;
    const float* gate_logits;
    float* output;
    __device__ void operator()(std::size_t i) const {
        const float gate = bf16_round(gate_logits[i]);
        const float sigmoid = __fdiv_rn(1.0F, __fadd_rn(1.0F, expf(-gate)));
        output[i] = bf16_round(__fmul_rn(bf16_round(input[i]), sigmoid));
    }
};

struct Bf16Softcap {
    float* row;
    float cap;
    __device__ void operator()(std::size_t i) const {
        const float value = bf16_round(row[i]);
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
        const float rounded = bf16_round(result);
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

using ops::norm::Bf16HeadClipArgs;
using ops::norm::LogPositionScale;
using ops::norm::l2_clip_bf16;

}  // namespace strix::core::row_transforms
