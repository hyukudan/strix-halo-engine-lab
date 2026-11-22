#pragma once

#include "core/gemv_rows.hpp"
#include "moe_core/microscaled_r1_cuda_plan.hpp"
#include "strix/weight/formats/microscaled.hpp"

#include <cuda_runtime.h>

namespace strix::moe_core::microscaled_r1::cuda_body {

namespace packed = weight::microscaled;
namespace scalar = weight::scalar_float;

__device__ __forceinline__ float low_bf16(unsigned pair) { return scalar::u2f(pair << 16U); }
__device__ __forceinline__ float high_bf16(unsigned pair) { return scalar::u2f(pair & 0xffff0000U); }

__device__ __forceinline__ float2 input_pair(const float* input) {
    return {scalar::bf16_round(input[0]), scalar::bf16_round(input[1])};
}

__device__ __forceinline__ float2 input_pair(const std::uint16_t* input) {
    return {low_bf16(input[0]), low_bf16(input[1])};
}

__device__ __forceinline__ float2 aligned_input_pair(const float* input) {
    const auto pair = __ldg(reinterpret_cast<const float2*>(input));
    return {scalar::bf16_round(pair.x), scalar::bf16_round(pair.y)};
}

__device__ __forceinline__ float2 aligned_input_pair(const std::uint16_t* input) {
    const auto pair = __ldg(reinterpret_cast<const unsigned*>(input));
    return {low_bf16(pair), high_bf16(pair)};
}

__device__ __forceinline__ unsigned e4m3_half_pair(unsigned code) {
    unsigned result;
    asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f16x2.e4m3x2 %0, lo; }"
        : "=r"(result) : "r"(code));
    return result;
}

__device__ __forceinline__ unsigned e2m1_half_pair_portable(unsigned bits) {
    // E2M1 values have zero in the low byte of FP16. Select the two high
    // magnitude bytes and copy each sign bit, including the sign of zero.
    const unsigned select = ((bits & 0x07U) << 4U) | ((bits & 0x70U) << 8U);
    const unsigned signs = ((bits & 0x08U) << 12U) | ((bits & 0x80U) << 24U);
    return __byte_perm(0x3e3c3800U, 0x46444240U, select) | signs;
}

__device__ __forceinline__ unsigned e2m1_half_pair(unsigned bits) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM100_ALL)
    unsigned halves;
    asm("{ .reg .b8 byte; cvt.u8.u32 byte, %1; cvt.rn.f16x2.e2m1x2 %0, byte; }"
        : "=r"(halves) : "r"(bits & 0xffU));
    return halves;
#else
    return e2m1_half_pair_portable(bits);
#endif
}

__device__ __forceinline__ unsigned scale_and_round(unsigned halves, float scale) {
    float lo, hi;
    asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
        : "=f"(lo), "=f"(hi) : "r"(halves));
    unsigned result;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(__fmul_rn(hi, scale)), "f"(__fmul_rn(lo, scale)));
    return result;
}

template <class Operand>
__device__ __forceinline__ unsigned decode(const Operand& weights, unsigned bits, std::uint8_t scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    if constexpr (Operand::kNvfp4) {
        if (scale <= 0x7eU && (scalar::f2u(weights.matrix.outer_scale) & 0x7fffffffU) < 0x7f800000U) {
            unsigned halves = e2m1_half_pair(bits);
            const unsigned scales = __byte_perm(e4m3_half_pair(scale), 0U, 0x1010U);
            // E2M1 times finite E4M3 has at most six significant bits.
            // Its nonzero magnitude is in [2^-10, 2688], so this product is exact in FP16.
            asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(halves) : "r"(scales));
            return scale_and_round(halves, weights.matrix.outer_scale);
        }
    } else {
        if (scale != 0xffU && (bits & 0x007fU) != 0x007fU && (bits & 0x7f00U) != 0x7f00U) {
            return scale_and_round(e4m3_half_pair(bits), Operand::decode_scale(scale));
        }
    }
#endif
    // Preserve the incumbent scalar decoder's handling of special values and older targets.
    return weights.decode_pair(reinterpret_cast<const std::uint8_t*>(&bits), Operand::decode_scale(scale));
}

__device__ __forceinline__ void accumulate(unsigned pair, float2 input, float& sum) {
    sum = __fmaf_rn(low_bf16(pair), input.x, sum);
    sum = __fmaf_rn(high_bf16(pair), input.y, sum);
}

// The packed dot for `count` inputs (count <= Inputs) against the same weight rows. Every code word and block scale is
// loaded and decoded once and accumulated into each input's own accumulators: per input the lane chunks, decode, FMA
// order and butterfly are those of a one-input dot, so out[n] is bitwise the dot of inputs[n] alone (the expert union of
// a rows launch reads each expert once for all its routed rows). The aligned run path and the short / unaligned scalar
// path give the same bits (same per-lane K sequence 2 lane + 64 j, decode == decode_pair), so one alignment decision
// for all inputs keeps every input's bits.
template <packed::MatrixFormat Format, unsigned Columns, unsigned Inputs, unsigned Outputs, class Input>
__device__ __forceinline__ void dot_inputs(const packed::PackedMatrix& matrix, unsigned col, unsigned stride,
                                           unsigned live, const Input* const (&inputs)[Inputs], unsigned count,
                                           unsigned width, unsigned lane, float (&out)[Inputs][Outputs]) {
    using Operand = packed::PackedOperand<Format>;
    using Run = CudaRun<Operand::kNvfp4 ? 1U : 2U, Operand::kBlockColumns>;
    constexpr unsigned values_per_byte = Operand::kNvfp4 ? 2U : 1U;
    constexpr unsigned warp_mask = 0xffffffffU;
    const Operand weights{matrix};
    typename Operand::Tile16Address rows[Outputs];
    bool aligned = matrix.columns % (Run::kWordBytes * values_per_byte) == 0U &&
                   reinterpret_cast<std::uintptr_t>(matrix.codes) % Run::kWordBytes == 0U &&
                   reinterpret_cast<std::uintptr_t>(inputs[0]) % (2U * sizeof(Input)) == 0U;
#pragma unroll
    for (unsigned n = 1; n < Inputs; ++n) {
        if (n < count) aligned = aligned && reinterpret_cast<std::uintptr_t>(inputs[n]) % (2U * sizeof(Input)) == 0U;
    }
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) {
        const unsigned offset = i % Columns < live ? i % Columns : 0U;
        rows[i] = weights.tile16_address(col + offset + (i / Columns) * stride, 0U);
#pragma unroll
        for (unsigned n = 0; n < Inputs; ++n) out[n][i] = 0.0F;
    }

    const unsigned full = aligned ? width / Run::kColumns * Run::kColumns : 0U;
    for (unsigned base = 0; base < full; base += Run::kColumns) {
        unsigned codes[Outputs], scales[Outputs];
#pragma unroll
        for (unsigned i = 0; i < Outputs; ++i) {
            const auto* run = rows[i].codes + base / values_per_byte;
            codes[i] = __ldg(reinterpret_cast<const unsigned*>(run) + lane);
            scales[i] = __ldg(rows[i].scale_code + base / Operand::kBlockColumns + (lane & (Run::kScales - 1U)));
        }
#pragma unroll 1
        for (unsigned stripe = 0; stripe < Run::kPairsPerWord; ++stripe) {
            float2 x[Inputs];
#pragma unroll
            for (unsigned n = 0; n < Inputs; ++n) {
                if (n < count) x[n] = aligned_input_pair(inputs[n] + base + stripe * Run::kStripeColumns + 2U * lane);
            }
#pragma unroll
            for (unsigned i = 0; i < Outputs; ++i) {
                const unsigned word = __shfl_sync(warp_mask, codes[i], Run::code_lane(stripe, lane));
                const auto scale = static_cast<std::uint8_t>(__shfl_sync(warp_mask, scales[i], Run::scale_lane(stripe, lane)));
                const unsigned pair = decode(weights, word >> Run::code_shift(lane), scale);
#pragma unroll
                for (unsigned n = 0; n < Inputs; ++n) {
                    if (n < count) accumulate(pair, x[n], out[n][i]);
                }
            }
        }
    }

    // Short and unaligned rows use the same order and exact scalar operand decoder.
    for (unsigned k = full + 2U * lane; k < width; k += Run::kStripeColumns) {
        float2 x[Inputs];
#pragma unroll
        for (unsigned n = 0; n < Inputs; ++n) {
            if (n < count) x[n] = input_pair(inputs[n] + k);
        }
#pragma unroll
        for (unsigned i = 0; i < Outputs; ++i) {
            const auto scale = Operand::decode_scale(rows[i].scale_code[k / Operand::kBlockColumns]);
            const unsigned pair = weights.decode_pair(rows[i].codes + k / values_per_byte, scale);
#pragma unroll
            for (unsigned n = 0; n < Inputs; ++n) {
                if (n < count) accumulate(pair, x[n], out[n][i]);
            }
        }
    }
#pragma unroll
    for (unsigned n = 0; n < Inputs; ++n) {
#pragma unroll
        for (unsigned i = 0; i < Outputs; ++i) {
            if (n < count) out[n][i] = core::gemv_rows::butterfly_sum(out[n][i]);
        }
    }
}

// One input: the incumbent decode dot.
template <packed::MatrixFormat Format, unsigned Columns, unsigned Outputs, class Input>
__device__ __forceinline__ void dot_rows(const packed::PackedMatrix& matrix, unsigned col, unsigned stride,
                                         unsigned live, const Input* input, unsigned width, unsigned lane,
                                         float (&out)[Outputs]) {
    const Input* const inputs[1] = {input};
    float one[1][Outputs];
    dot_inputs<Format, Columns>(matrix, col, stride, live, inputs, 1U, width, lane, one);
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) out[i] = one[0][i];
}

} // namespace strix::moe_core::microscaled_r1::cuda_body
