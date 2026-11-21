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

// Parallel uses the lane-strided FMA and LDS halving tree. One lane retains
// the ascending reference sum for device qualification.
inline hipError_t enqueue_rms_bf16_rows(const Bf16RmsRows& args, float epsilon, RmsRowsReduction reduction,
                                       hipStream_t stream) {
    constexpr unsigned lanes = 256;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.input || !args.output || args.rows == 0 || args.rows > 0x7FFFFFFFU || args.width == 0 ||
        args.width > std::numeric_limits<unsigned>::max() - lanes ||
        args.input_row < args.width || args.output_row < args.width ||
        args.rows > max_elements / args.input_row || args.rows > max_elements / args.output_row ||
        (args.input == args.output && args.input_row != args.output_row)) return hipErrorInvalidValue;

    using Traits = Bf16ScaleSqrtDivTraits;
    if (reduction == RmsRowsReduction::serial_reference) {
        rmsnorm_strided_kernel<Traits, 1><<<dim3(args.rows), 1, 0, stream>>>(
            args.input, args.input_row, args.scale, 0, args.width, epsilon, args.output, args.output_row);
    } else if (reduction == RmsRowsReduction::parallel) {
        rmsnorm_strided_kernel<Traits, lanes><<<dim3(args.rows), lanes, 0, stream>>>(
            args.input, args.input_row, args.scale, 0, args.width, epsilon, args.output, args.output_row);
    } else {
        return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

#endif

}  // namespace strix::ops::norm
