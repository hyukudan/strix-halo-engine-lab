#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <limits>
#endif

namespace strix::moe_core::grouped_rows {

struct GatherArgs {
    const float* input;                  // Source token rows, FP32.
    std::size_t input_stride;            // Elements.
    const std::uint32_t* assignment_rows; // Expert-major assignment -> source token.
    unsigned assignments;
    unsigned width;
    std::uint16_t* grouped;              // Expert-major activation rows, BF16.
    std::size_t grouped_stride;          // Elements.
    const unsigned* failure = nullptr;  // Optional device plan failure word.
};

struct CombineArgs {
    const float* grouped;                // Expert-major down-projection rows, FP32.
    std::size_t grouped_stride;
    const std::uint32_t* combine_rows;    // [token][expert in ascending order] -> grouped row.
    const float* assignment_weights;     // One FP32 route weight per grouped row.
    unsigned rows;
    unsigned top_k;
    unsigned width;
    float* output;                       // Token rows, BF16-rounded values stored as FP32.
    std::size_t output_stride;
    const unsigned* failure = nullptr;
};

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace detail {

inline constexpr unsigned kThreads = 256U;
inline constexpr unsigned kGridRows = 65535U;

#if defined(__HIPCC__)
using Error = hipError_t;
using Stream = hipStream_t;
inline constexpr Error success = hipSuccess;
inline constexpr Error invalid = hipErrorInvalidValue;
inline Error launch_error() { return hipGetLastError(); }
#else
using Error = cudaError_t;
using Stream = cudaStream_t;
inline constexpr Error success = cudaSuccess;
inline constexpr Error invalid = cudaErrorInvalidValue;
inline Error launch_error() { return cudaGetLastError(); }
#endif

template <bool PreserveNaN>
__device__ __forceinline__ float rounded_bf16(float value) {
    namespace bf16 = weight::scalar_float;
    if constexpr (PreserveNaN) {
        if ((bf16::f2u(value) & 0x7fffffffU) > 0x7f800000U) return value;
    }
    return bf16::bf16_round(value);
}

template <bool PreserveNaN>
__device__ __forceinline__ float weighted_bf16(float weight, float projection) {
    return __fmul_rn(weight, rounded_bf16<PreserveNaN>(projection));
}

static __global__ __launch_bounds__(kThreads) void gather_kernel(GatherArgs args) {
    if (args.failure && *args.failure != 0U) return;
    const auto column = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (column >= args.width) return;
    for (std::size_t row = blockIdx.y; row < args.assignments; row += gridDim.y) {
        const std::size_t token = args.assignment_rows[row];
        const float value = args.input[token * args.input_stride + column];
        args.grouped[row * args.grouped_stride + column] = weight::scalar_float::bf16_encode(value);
    }
}

static __global__ __launch_bounds__(kThreads) void combine_kernel(CombineArgs args) {
    if (args.failure && *args.failure != 0U) return;
    const auto column = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (column >= args.width) return;
    for (std::size_t row = blockIdx.y; row < args.rows; row += gridDim.y) {
        float sum = 0.0F;
        for (unsigned slot = 0; slot < args.top_k; ++slot) {
            const std::size_t expert_row = args.combine_rows[row * args.top_k + slot];
            const float term = weighted_bf16<true>(args.assignment_weights[expert_row],
                                                   args.grouped[expert_row * args.grouped_stride + column]);
            sum = __fadd_rn(sum, term);
        }
        args.output[row * args.output_stride + column] = rounded_bf16<true>(sum);
    }
}

inline dim3 grid(unsigned width, unsigned rows) {
    return dim3((width - 1U) / kThreads + 1U, std::min(rows, kGridRows));
}

}  // namespace detail

// The validated grouping plan supplies assignment_rows and an optional
// asynchronous failure word. Keep buffers disjoint and live through the stream.
inline detail::Error gather(const GatherArgs& args, detail::Stream stream) {
    if (args.assignments == 0U) return detail::success;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.input || !args.assignment_rows || !args.grouped || args.width == 0U ||
        args.input_stride < args.width || args.grouped_stride < args.width ||
        args.assignments > max_elements / args.input_stride ||
        args.assignments > max_elements / args.grouped_stride) return detail::invalid;
    detail::gather_kernel<<<detail::grid(args.width, args.assignments), detail::kThreads, 0, stream>>>(args);
    return detail::launch_error();
}

// The plan keeps the route's original FP32 weights and orders its combine
// by expert ID. Round each down operand to BF16, accumulate separate FP32
// products and adds in plan order, and round the final output once to BF16.
inline detail::Error combine(const CombineArgs& args, detail::Stream stream) {
    if (args.rows == 0U) return detail::success;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.grouped || !args.combine_rows || !args.assignment_weights || !args.output ||
        args.top_k == 0U || args.width == 0U || args.grouped_stride < args.width || args.output_stride < args.width ||
        args.rows > std::numeric_limits<std::uint32_t>::max() / args.top_k ||
        args.rows > max_elements / args.top_k / args.grouped_stride ||
        args.rows > max_elements / args.output_stride) return detail::invalid;
    detail::combine_kernel<<<detail::grid(args.width, args.rows), detail::kThreads, 0, stream>>>(args);
    return detail::launch_error();
}
#endif

}  // namespace strix::moe_core::grouped_rows
