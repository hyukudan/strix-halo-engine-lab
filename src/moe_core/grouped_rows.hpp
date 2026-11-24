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

#if defined(__HIPCC__)
// CombineArgs on the BF16 image of the down rows (core::f16_tile_gemm::Bf16NanExactStore): grouped[i] is
// bf16_encode(down sum i), exact[i] is the FP32 sum itself wherever it is a NaN (any other element is never read),
// and *nan_seen is nonzero once a launch of this stream stored a NaN sum. Half the read of CombineArgs.
struct CombineImageArgs {
    const std::uint16_t* grouped;
    const float* exact;
    const unsigned* nan_seen;
    std::size_t grouped_stride;          // Elements, of grouped and of exact.
    const std::uint32_t* combine_rows;
    const float* assignment_weights;
    unsigned rows;
    unsigned top_k;
    unsigned width;
    float* output;
    std::size_t output_stride;
    const unsigned* failure = nullptr;
};
#endif

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

#if defined(__HIPCC__)
// gather_kernel with row r = token r: the BF16 image of the token rows.
static __global__ __launch_bounds__(kThreads) void token_rows_kernel(GatherArgs args) {
    if (args.failure && *args.failure != 0U) return;
    const auto column = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (column >= args.width) return;
    for (std::size_t row = blockIdx.y; row < args.assignments; row += gridDim.y)
        args.grouped[row * args.grouped_stride + column] =
            weight::scalar_float::bf16_encode(args.input[row * args.input_stride + column]);
}
#endif

// How the combine reads one down operand. Fp32Operand is the FP32 sum itself; ImageOperand (HIP) is the BF16 image of
// the sum with the FP32 sum read whole where it is a NaN: the same operand bits as Fp32Operand for every sum, so the
// product, the adds in plan order and the final round see the same operands in the same order, NaN signs and payloads
// included (a product of two NaNs keeps one of them by operand position: tests/moe_core/combine_image_hip.hip pins it).
struct Fp32Operand {
    using Args = CombineArgs;
    struct State {};
    __device__ __forceinline__ static State begin(const Args&) { return {}; }
    __device__ __forceinline__ static float load(const Args& args, State, std::size_t at) { return args.grouped[at]; }
};

#if defined(__HIPCC__)
struct ImageOperand {
    using Args = CombineImageArgs;
    struct State { unsigned seen; };
    // Uniform: the slow read exists only after a NaN sum was stored.
    __device__ __forceinline__ static State begin(const Args& args) { return {*args.nan_seen}; }
    __device__ __forceinline__ static float load(const Args& args, State state, std::size_t at) {
        const std::uint32_t image = args.grouped[at];
        float projection = weight::scalar_float::u2f(image << 16U);  // bf16_round(x) is bf16_encode(x) << 16
        if (state.seen != 0U && (image & 0x7fffU) > 0x7f80U) projection = args.exact[at];
        return projection;
    }
};
#endif

// The one combine: failure guard, grid traversal, expert order, weight-first product, adds and final round.
template <class Operand>
static __global__ __launch_bounds__(kThreads) void combine_kernel(typename Operand::Args args) {
    if (args.failure && *args.failure != 0U) return;
    const auto column = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (column >= args.width) return;
    const auto state = Operand::begin(args);
    for (std::size_t row = blockIdx.y; row < args.rows; row += gridDim.y) {
        float sum = 0.0F;
        for (unsigned slot = 0; slot < args.top_k; ++slot) {
            const std::size_t expert_row = args.combine_rows[row * args.top_k + slot];
            const float term = weighted_bf16<true>(args.assignment_weights[expert_row],
                                                   Operand::load(args, state, expert_row * args.grouped_stride + column));
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

#if defined(__HIPCC__)
// The BF16 image of `assignments` token rows (assignment_rows unused): row r holds bf16_encode of token r, the value
// gather() would copy to every assignment of token r. For a grouped GEMM that reads its input rows through the
// assignment map (grouped_microscaled::IndirectArgs) instead of a gathered copy.
inline detail::Error token_rows(const GatherArgs& args, detail::Stream stream) {
    if (args.assignments == 0U) return detail::success;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.input || !args.grouped || args.width == 0U || args.input_stride < args.width ||
        args.grouped_stride < args.width || args.assignments > max_elements / args.input_stride ||
        args.assignments > max_elements / args.grouped_stride) return detail::invalid;
    detail::token_rows_kernel<<<detail::grid(args.width, args.assignments), detail::kThreads, 0, stream>>>(args);
    return detail::launch_error();
}
#endif

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
    detail::combine_kernel<detail::Fp32Operand><<<detail::grid(args.width, args.rows), detail::kThreads, 0, stream>>>(args);
    return detail::launch_error();
}

#if defined(__HIPCC__)
// combine() reading the BF16 image of the down rows: the same output bits for every down sum (see
// ImageOperand). The caller's down launches store through core::f16_tile_gemm::Bf16NanExactStore on this stream.
inline detail::Error combine_image(const CombineImageArgs& args, detail::Stream stream) {
    if (args.rows == 0U) return detail::success;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!args.grouped || !args.exact || !args.nan_seen || !args.combine_rows || !args.assignment_weights ||
        !args.output || args.top_k == 0U || args.width == 0U || args.grouped_stride < args.width ||
        args.output_stride < args.width || args.rows > std::numeric_limits<std::uint32_t>::max() / args.top_k ||
        args.rows > max_elements / args.top_k / args.grouped_stride ||
        args.rows > max_elements / args.output_stride) return detail::invalid;
    detail::combine_kernel<detail::ImageOperand><<<detail::grid(args.width, args.rows), detail::kThreads, 0, stream>>>(args);
    return detail::launch_error();
}
#endif
#endif

}  // namespace strix::moe_core::grouped_rows
