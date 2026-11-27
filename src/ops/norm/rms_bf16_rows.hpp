#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#include "norm_presets.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <limits>
#endif

namespace strix::ops::norm {

// The optional BF16 scale precedes the single BF16 rounding of the FP32 result.
// Buffers may coincide if their row strides match, or must be disjoint.
struct Bf16RmsRows {
    const float* input{};
    const std::uint16_t* scale{};
    float* output{};
    unsigned rows{};
    unsigned width{};
    std::size_t input_row{};
    std::size_t output_row{};
};

enum class RmsRowsReduction { parallel, serial_reference };

#if defined(__HIPCC__)

struct Bf16ScaleSqrtDivTraits : DeclaredF32Traits<OutputForm::bf16_once> {
    using Scale = std::uint16_t;
    __device__ static float inv_sqrt(float value) { return __fdiv_rn(1.0F, __fsqrt_rn(value)); }
    __device__ static float load(const Scale* scale, ScaleFormat, unsigned i) {
        return weight::scalar_float::u2f(std::uint32_t{scale[i]} << 16U);
    }
};

// The RMS factor of the FP32 row, the BF16 image of the row as the numerator: B((B(x) * r) * w) (a collapse kept in FP32
// for its variance and stored BF16 for the norm, e.g. core/bf16_reference.hpp rms_norm_bf16_reference round_numerator).
struct Bf16RoundedNumeratorTraits : Bf16ScaleSqrtDivTraits {
    __device__ static float numerator(float v) { return weight::scalar_float::bf16_round(v); }
};

// The unweighted RMS rows: FP32 out = x * r, no scale (pass scale = nullptr) and no rounding (the host's per-head
// attn::head_rms_reference: the same r = 1 / sqrt(ss / n + eps), the product stored as FP32 for the caller's next step).
// Used through enqueue_rms_bf16_rows_rounded_numerator<Bf16ScaleSqrtDivPlainTraits> (any Traits is admitted).
struct Bf16ScaleSqrtDivPlainTraits : DeclaredF32Traits<OutputForm::plain> {
    using Scale = std::uint16_t;
    __device__ static float inv_sqrt(float value) { return __fdiv_rn(1.0F, __fsqrt_rn(value)); }
    __device__ static float load(const Scale* scale, ScaleFormat, unsigned i) {
        return weight::scalar_float::u2f(std::uint32_t{scale[i]} << 16U);
    }
};

// The same rows with the declared square root (strix::numeric::declared_sqrt: the correctly rounded root through double
// precision) in the factor: __fsqrt_rn is not correctly rounded over the whole range on every device, the declared root
// is, and it is the one the host reference of a declared-arithmetic model computes. 1 / root is the IEEE quotient.
template <class Base>
struct DeclaredSqrtOf : Base {
    __device__ static float inv_sqrt(float value) { return __fdiv_rn(1.0F, ::strix::numeric::declared_sqrt(value)); }
};
using Bf16DeclaredSqrtTraits = DeclaredSqrtOf<Bf16ScaleSqrtDivTraits>;
using Bf16DeclaredSqrtPlainTraits = DeclaredSqrtOf<Bf16ScaleSqrtDivPlainTraits>;

namespace detail {
template <class Traits, unsigned J>
void launch_rms_wave_rows(const Bf16RmsRows& args, float epsilon, hipStream_t stream) {
    const unsigned blocks = (args.rows + kWaveRows - 1U) / kWaveRows;
    rmsnorm_wave_rows_kernel<Traits, J><<<dim3(blocks), 32U * kWaveRows, 0, stream>>>(
        args.input, args.input_row, args.scale, 0, args.width, args.rows, epsilon, args.output, args.output_row);
}

template <class Traits>
hipError_t enqueue_rms_bf16_rows(const Bf16RmsRows& args, float epsilon, RmsRowsReduction reduction,
                                 hipStream_t stream) {
    constexpr unsigned lanes = 256;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.input || !args.output || args.rows == 0 || args.rows > 0x7FFFFFFFU || args.width == 0 ||
        args.width > std::numeric_limits<unsigned>::max() - lanes ||
        args.input_row < args.width || args.output_row < args.width ||
        args.rows > max_elements / args.input_row || args.rows > max_elements / args.output_row ||
        (args.input == args.output && args.input_row != args.output_row)) return hipErrorInvalidValue;

    if (reduction == RmsRowsReduction::serial_reference) {
        rmsnorm_strided_kernel<Traits, 1><<<dim3(args.rows), 1, 0, stream>>>(
            args.input, args.input_row, args.scale, 0, args.width, epsilon, args.output, args.output_row);
    } else if (reduction == RmsRowsReduction::parallel && args.width <= 32U) {
        launch_rms_wave_rows<Traits, 1>(args, epsilon, stream);
    } else if (reduction == RmsRowsReduction::parallel && args.width <= 64U) {
        launch_rms_wave_rows<Traits, 2>(args, epsilon, stream);
    } else if (reduction == RmsRowsReduction::parallel && args.width <= 128U) {
        launch_rms_wave_rows<Traits, 4>(args, epsilon, stream);
    } else if (reduction == RmsRowsReduction::parallel && args.width <= lanes) {
        launch_rms_wave_rows<Traits, 8>(args, epsilon, stream);
    } else if (reduction == RmsRowsReduction::parallel) {
        rmsnorm_strided_kernel<Traits, lanes><<<dim3(args.rows), lanes, 0, stream>>>(
            args.input, args.input_row, args.scale, 0, args.width, epsilon, args.output, args.output_row);
    } else {
        return hipErrorInvalidValue;
    }
    return hipGetLastError();
}
}  // namespace detail

// Parallel uses the lane-strided FMA and LDS halving tree (rows of at most 256 values: one wave per row with the same
// tree and bits, rmsnorm_wave_rows_kernel). One lane retains the ascending reference sum for device qualification.
template <class Traits = Bf16ScaleSqrtDivTraits>
inline hipError_t enqueue_rms_bf16_rows(const Bf16RmsRows& args, float epsilon, RmsRowsReduction reduction,
                                       hipStream_t stream) {
    return detail::enqueue_rms_bf16_rows<Traits>(args, epsilon, reduction, stream);
}
// The same reductions with Bf16RoundedNumeratorTraits. A template, so its kernels are instantiated only where it is
// called (the objects of the other includers keep their machine code).
template <class Traits = Bf16RoundedNumeratorTraits>
hipError_t enqueue_rms_bf16_rows_rounded_numerator(const Bf16RmsRows& args, float epsilon,
                                                          RmsRowsReduction reduction, hipStream_t stream) {
    return detail::enqueue_rms_bf16_rows<Traits>(args, epsilon, reduction, stream);
}

#endif

}  // namespace strix::ops::norm
