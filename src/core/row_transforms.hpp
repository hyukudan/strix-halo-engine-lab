#pragma once

#include "core/pointwise.hpp"
#include "ops/norm/norm_device.hpp"
#include "strix/attn/cache_rows.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

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

// A short projection into a large output row. Each lane owns `Columns` adjacent
// outputs and reads along contiguous columns in the checkpoint's [K][N] matrix.
// Keep the ascending K FMA chain and round once, after the projection.
// Columns = 4 (element i = outputs 4i .. 4i + 3 of the flat [row][output] image)
// reads each weight row piece as one 8-byte load and the input row as 16-byte
// loads and stores 16 bytes: per output the same operands in the same chain,
// so the same bits as Columns = 1. Needs outputs % 4 == 0, input and output
// 16-byte aligned and weights 8-byte aligned (callers check fits()).
template <unsigned Inputs, bool CheckFinite = false, unsigned Columns = 1>
struct ShortBf16Projection {
    static_assert(Columns == 1U || (Columns == 4U && Inputs % 4U == 0U));
    const float* input;
    const std::uint16_t* weights;
    float* output;
    unsigned outputs;
    unsigned* failure = nullptr;

    // Element count of a launch over `rows` rows of `outputs` outputs.
    static constexpr std::size_t launch_size(std::size_t rows, unsigned width) { return rows * width / Columns; }
    static bool fits(const float* in, const std::uint16_t* w, const float* out, unsigned width) {
        if constexpr (Columns == 1U) return true;
        return width % Columns == 0U && reinterpret_cast<std::uintptr_t>(in) % 16U == 0U &&
               reinterpret_cast<std::uintptr_t>(out) % 16U == 0U && reinterpret_cast<std::uintptr_t>(w) % 8U == 0U;
    }

    __device__ void operator()(std::size_t i) const {
        if constexpr (Columns == 1U) {
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
        } else {
            const std::size_t first = i * Columns;
            const auto* row = input + (first / outputs) * Inputs;
            const auto column = first % outputs;
            float x[Inputs];
#pragma unroll
            for (unsigned k = 0; k < Inputs; k += 4U) {
                const float4 piece = *reinterpret_cast<const float4*>(row + k);
                x[k] = piece.x;
                x[k + 1U] = piece.y;
                x[k + 2U] = piece.z;
                x[k + 3U] = piece.w;
            }
            float result[Columns] = {};
#pragma unroll
            for (unsigned k = 0; k < Inputs; ++k) {
                // Little-endian words: weight (k, column + j) is the low (even j) or high (odd j) half of word j / 2.
                const uint2 bits = *reinterpret_cast<const uint2*>(weights + std::size_t{k} * outputs + column);
                const std::uint32_t words[2] = {bits.x, bits.y};
#pragma unroll
                for (unsigned j = 0; j < Columns; ++j) {
                    const std::uint32_t word = words[j / 2U];
                    const float operand = weight::scalar_float::u2f((j % 2U) != 0U ? word & 0xFFFF0000U : word << 16U);
                    result[j] = __builtin_fmaf(x[k], operand, result[j]);
                }
            }
            float rounded[Columns];
#pragma unroll
            for (unsigned j = 0; j < Columns; ++j) {
                rounded[j] = bf16_round(result[j]);
                if constexpr (CheckFinite) reject_nonfinite(rounded[j], failure);
            }
            *reinterpret_cast<float4*>(output + first) = make_float4(rounded[0], rounded[1], rounded[2], rounded[3]);
        }
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
