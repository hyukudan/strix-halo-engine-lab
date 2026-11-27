#pragma once

#include "core/pointwise.hpp"
#include "ops/norm/norm_device.hpp"
#include "strix/attn/cache_rows.hpp"
#include "strix/numeric/declared.hpp"
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

struct CopyBf16Round {
    const float* input;
    float* output;
    __device__ void operator()(std::size_t i) const { output[i] = bf16_round(input[i]); }
};

struct Copy {
    const float* input;
    float* output;
    __device__ void operator()(std::size_t i) const { output[i] = input[i]; }
};

// out[i] = a[i] + b[i] in FP32 (one rounding; out may alias a).
struct Add {
    const float* a;
    const float* b;
    float* out;
    __device__ void operator()(std::size_t i) const { out[i] = __fadd_rn(a[i], b[i]); }
};

struct WidenRepeatedBf16Row {
    const std::uint16_t* input;
    float* output;
    unsigned width;
    __device__ void operator()(std::size_t i) const { output[i] = attn::cache_rows::decode(input[i % width]); }
};

// Encode an FP32 row into a slice of a wider BF16 row.
struct StoreBf16Slice {
    const float* input;
    std::uint16_t* output;
    unsigned width;
    unsigned stride;

    __device__ void operator()(std::size_t i) const {
        output[(i / width) * stride + i % width] = attn::cache_rows::encode(input[i]);
    }
};

// StoreBf16Slice whose first destination row is read on the device (`*first_row`, e.g. a step descriptor's
// position), so a captured step graph stores each replay at its own row: output row (*first_row + i / width), the
// same bytes StoreBf16Slice writes at output + *first_row * stride.
struct StoreBf16SliceAtRow {
    const float* input;
    std::uint16_t* output;
    unsigned width;
    unsigned stride;
    const std::uint32_t* first_row;

    __device__ void operator()(std::size_t i) const {
        output[(std::size_t{*first_row} + i / width) * stride + i % width] = attn::cache_rows::encode(input[i]);
    }
};

template <unsigned Taps>
struct MixBf16Rows {
    const std::uint16_t* input;
    const float* coefficients;
    float* output;
    unsigned width;

    __device__ void operator()(std::size_t i) const {
        const auto* first = input + (i / width) * width * Taps + i % width;
        float sum = 0.0F;
        for (unsigned tap = 0; tap < Taps; ++tap)
            sum = __fadd_rn(sum, __fmul_rn(coefficients[tap], attn::cache_rows::decode(first[tap * width])));
        output[i] = sum;
    }
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
    // Element i of the gated product, reading the gate logit before the input (the incumbent access order: the
    // standalone and fused consumers keep their machine code).
    __device__ static float value(const float* input, const float* gate_logits, std::size_t i) {
        const float gate = bf16_round(gate_logits[i]);
        const float sigmoid = __fdiv_rn(1.0F, __fadd_rn(1.0F, expf(-gate)));
        return bf16_round(__fmul_rn(bf16_round(input[i]), sigmoid));
    }
    __device__ void operator()(std::size_t i) const { output[i] = value(input, gate_logits, i); }
};

// The exp of a sigmoid gate: the device libm (default, the incumbent bits) or the declared exp (numeric::declared_exp,
// the host reference's bits on every platform; core/bf16_reference.hpp sigmoid_product_reference declared_exp).
struct NativeSigmoidExp {
    __device__ static float exp(float x) { return expf(x); }
};
struct DeclaredSigmoidExp {
    __device__ static float exp(float x) { return ::strix::numeric::declared_exp(x); }
};

// Bf16SigmoidProduct with the sigmoid itself rounded to BF16 before the product (core/bf16_reference.hpp
// SigmoidGatePrecision::bf16_sigmoid): B(B(x) * B(sigmoid(B(gate)))).
template <class Exp = NativeSigmoidExp>
struct Bf16RoundedSigmoidProduct {
    const float* input;
    const float* gate_logits;
    float* output;
    __device__ void operator()(std::size_t i) const {
        const float gate = bf16_round(gate_logits[i]);
        const float sigmoid = bf16_round(__fdiv_rn(1.0F, __fadd_rn(1.0F, Exp::exp(-gate))));
        output[i] = bf16_round(__fmul_rn(bf16_round(input[i]), sigmoid));
    }
};

// Mix across the head axis with a folded [heads][heads] matrix (attn_core/value_head_mix_reference.hpp
// apply_folded_value_head_mix_reference): out[row][h][d] = fma-chain over g = 0 .. heads - 1 ascending from +0 of
// folded[h][g] * in[row][g][d]. Element i of the flat [row][head][width] output; input and output are disjoint.
struct FoldedHeadMix {
    const float* input;
    const float* folded;
    float* output;
    unsigned heads;
    unsigned width;
    __device__ void operator()(std::size_t i) const {
        const std::size_t d = i % width, h = (i / width) % heads, row = i / (std::size_t{width} * heads);
        const float* in = input + row * heads * width + d;
        float sum = 0.0F;
        for (unsigned g = 0; g < heads; ++g) sum = __builtin_fmaf(folded[h * heads + g], in[std::size_t{g} * width], sum);
        output[i] = sum;
    }
};

// Copy a [rows][vectors][width] block between two layouts (row / vector strides in elements; the caller offsets the
// pointers): the permutation the per-head and per-group projections of a row-major activation need (a projection sweep
// reads tight rows, so a head's or group's columns of all rows are gathered first, and scattered back after).
struct StridedCopy {
    const float* input;
    float* output;
    unsigned width;
    unsigned vectors;
    std::size_t in_row, in_vector, out_row, out_vector;
    __device__ void operator()(std::size_t i) const {
        const std::size_t d = i % width, v = (i / width) % vectors, r = i / (std::size_t{width} * vectors);
        output[r * out_row + v * out_vector + d] = input[r * in_row + v * in_vector + d];
    }
};

// The same gated value stored as the BF16 operand image a BF16 projection makes of Bf16SigmoidProduct's FP32 output
// (bf16_encode of that value, core/fma_bf16_rows.hpp enqueue_bf16_operand_image): no FP32 plane and no image pass.
struct Bf16SigmoidProductImage {
    const float* input;
    const float* gate_logits;
    std::uint16_t* image;
    __device__ void operator()(std::size_t i) const {
        image[i] = ::strix::weight::scalar_float::bf16_encode(Bf16SigmoidProduct::value(input, gate_logits, i));
    }
};

// The BF16 head boundary then softcap. One expression for every consumer (Bf16Softcap and the screened greedy head,
// core/hip_screened_head.hpp, which relies on it being monotone non-decreasing; its device test proves that).
__device__ __forceinline__ float bf16_softcap(float y, float cap) {
    const float value = bf16_round(y);
    return isfinite(value) ? __fmul_rn(cap, tanhf(__fdiv_rn(value, cap))) : value;
}

struct Bf16Softcap {
    float* row;
    float cap;
    __device__ void operator()(std::size_t i) const { row[i] = bf16_softcap(row[i], cap); }
};

// Linear and quadratic logit hinges (strix/core/logit_epilogue.hpp, host core/multimax_reference.hpp
// multimax_f32_reference): the four hinges read the same x, every subtract, multiply and add rounds on its own (no
// contraction), ReLU keeps NaN and maps x <= 0 to +0; with preserve_mask a -inf logit stays -inf.
struct LogitHinges {
    float* row;
    float t0, t1, t2, t3, g0, g1, g2, g3;
    bool preserve_mask;
    __device__ static float relu(float v) { return v <= 0.0F ? 0.0F : v; }
    __device__ void operator()(std::size_t i) const {
        const float x = row[i];
        if (preserve_mask && x == -__builtin_inff()) return;
        const float lower = relu(__fsub_rn(t0, x)), upper = relu(__fsub_rn(x, t1));
        const float lower_square = relu(__fsub_rn(t2, x)), upper_square = relu(__fsub_rn(x, t3));
        const float a = __fmul_rn(g0, lower), b = __fmul_rn(g1, upper);
        const float c = __fmul_rn(g2, __fmul_rn(lower_square, lower_square));
        const float d = __fmul_rn(g3, __fmul_rn(upper_square, upper_square));
        row[i] = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(x, a), b), c), d);
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
// ReverseOrder walks K from the last input to the first (both forms).
template <unsigned Inputs, bool CheckFinite = false, unsigned Columns = 1, bool ReverseOrder = false>
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
            for (unsigned step = 0; step < Inputs; ++step) {
                const auto k = ReverseOrder ? Inputs - 1U - step : step;
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
            for (unsigned step = 0; step < Inputs; ++step) {
                const auto k = ReverseOrder ? Inputs - 1U - step : step;
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
using ops::norm::l2_clip_bf16_rows;

}  // namespace strix::core::row_transforms
