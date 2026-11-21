#pragma once

#include "moe_core/device_grouped_plan.hpp"
#include "moe_core/expert_activation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIPCC__)
#include "strix/hip_wave32_dpp.hpp"
#elif defined(__CUDACC__)
#include "cuda/core/reduce.cuh"
#endif

namespace strix::moe_core::microscaled_r1 {

namespace packed = strix::weight::microscaled;

// One token, already routed in selection order. Matrices refer to an admitted
// layer's device projections. The optional device failure word is the route
// status. Keep all inputs, output and TopK * intermediate BF16 activations
// alive through stream completion. The intermediate buffer must be disjoint.
struct Args {
    const device_grouped::ExpertWeights* matrices;
    const unsigned* ids;
    const float* weights;
    const unsigned* failure;
    const float* input;
    std::uint16_t* activation;
    float* output;
    unsigned experts, hidden, intermediate;
    float gate_cap, up_cap;
};

template <packed::MatrixFormat Format, unsigned TopK>
inline bool fits(const Args& a) {
    static_assert(TopK > 0U && TopK <= 32U);
    constexpr auto k = packed::PackedOperand<Format>::kBlockColumns;
    constexpr auto max_columns = std::numeric_limits<unsigned>::max() / 2U;
    return a.matrices && a.ids && a.weights && a.input && a.activation && a.output &&
           a.experts >= TopK && a.hidden >= k && a.hidden <= max_columns && a.hidden % k == 0U &&
           a.intermediate >= k && a.intermediate <= max_columns && a.intermediate % k == 0U &&
           std::uint64_t{TopK} * a.intermediate <= std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) &&
           std::isfinite(a.gate_cap) && a.gate_cap > 0.0F && std::isfinite(a.up_cap) && a.up_cap > 0.0F;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace detail {

namespace scalar = weight::scalar_float;
namespace activation = expert_activation::detail;

inline constexpr unsigned kWarp = 32;
inline constexpr unsigned kWarps = 8;
inline constexpr unsigned kThreads = kWarp * kWarps;

__device__ __forceinline__ float widen(std::uint16_t value) {
    return scalar::u2f(std::uint32_t{value} << 16U);
}

__device__ __forceinline__ float input_bf16(std::uint16_t value) { return widen(value); }
__device__ __forceinline__ float input_bf16(float value) { return activation::round_linear(value); }

__device__ __forceinline__ float warp_sum(float value) {
#if defined(__HIPCC__)
    value += hip_wave32_dpp::xor32<16>(value);
    value += hip_wave32_dpp::xor32<8>(value);
    value += hip_wave32_dpp::xor32<4>(value);
    value += hip_wave32_dpp::xor32<2>(value);
    return value + hip_wave32_dpp::xor32<1>(value);
#else
    return cuda::core::butterfly(value);
#endif
}

// One warp owns a matrix output. Pair loading and reductions have fixed
// order across backends. The GPU accumulation tree differs from scalar GEMV.
template <packed::MatrixFormat Format, unsigned Outputs, class Input>
__device__ __forceinline__ void dot_rows(const packed::PackedMatrix& matrix, unsigned col, unsigned stride,
                                         const Input* input, unsigned width, unsigned lane, float (&out)[Outputs]) {
    using Operand = packed::PackedOperand<Format>;
    const Operand weights{matrix};
    typename Operand::Tile16Address rows[Outputs];
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) {
        rows[i] = weights.tile16_address(col + i * stride, 0);
        out[i] = 0.0F;
    }
    for (unsigned k = 2U * lane; k < width; k += 2U * kWarp) {
        const float a0 = input_bf16(input[k]);
        const float a1 = input_bf16(input[k + 1U]);
        const auto code = Operand::kNvfp4 ? k / 2U : k;
        const auto scale = k / Operand::kBlockColumns;
#pragma unroll
        for (unsigned i = 0; i < Outputs; ++i) {
            const auto pair = weights.decode_pair(rows[i].codes + code, Operand::decode_scale(rows[i].scale_code[scale]));
            out[i] = __fmaf_rn(widen(static_cast<std::uint16_t>(pair)), a0, out[i]);
            out[i] = __fmaf_rn(widen(static_cast<std::uint16_t>(pair >> 16U)), a1, out[i]);
        }
    }
#pragma unroll
    for (unsigned i = 0; i < Outputs; ++i) out[i] = warp_sum(out[i]);
}

// Paired gate and up GEMV share the input loads. Store only the BF16
// activation made with the same pre-activation BF16 rounding as grouped GEMM.
template <packed::MatrixFormat Format>
__global__ __launch_bounds__(kThreads) void gate_up(Args args) {
    if (args.failure && *args.failure != 0U) return;
    const unsigned slot = blockIdx.y;
    const unsigned expert = args.ids[slot];
    const unsigned col = blockIdx.x * kWarps + threadIdx.x / kWarp;
    if (expert >= args.experts || col >= args.intermediate) return;
    const unsigned lane = threadIdx.x % kWarp;
    float pair[2];
    dot_rows<Format>(args.matrices[expert].gate_up, col, args.intermediate, args.input, args.hidden, lane, pair);
    if (lane == 0U) {
        const moe::SoftcappedSwiglu<activation::DeviceMath> swiglu{args.gate_cap, args.up_cap};
        const float value = swiglu(activation::round_linear(pair[0]), activation::round_linear(pair[1]));
        args.activation[std::size_t{slot} * args.intermediate + col] = scalar::bf16_encode(value);
    }
}

// One CTA owns a hidden output column. Each warp projects down one or more
// routed experts. Store weighted terms in ascending expert order, then
// consume them with separate FP32 adds. No down-projection image is needed.
template <packed::MatrixFormat Format, unsigned TopK>
__global__ __launch_bounds__(kThreads) void down_combine(Args args) {
    if (args.failure && *args.failure != 0U) return;
    __shared__ float terms[TopK];
    const unsigned col = blockIdx.x;
    const unsigned lane = threadIdx.x % kWarp;
    for (unsigned slot = threadIdx.x / kWarp; slot < TopK; slot += kWarps) {
        const unsigned expert = args.ids[slot];
        if (expert >= args.experts) continue;
        const auto* input = args.activation + std::size_t{slot} * args.intermediate;
        float projected[1];
        dot_rows<Format>(args.matrices[expert].down, col, 0U, input, args.intermediate, lane, projected);
        if (lane == 0U) {
            unsigned rank = 0;
#pragma unroll
            for (unsigned i = 0; i < TopK; ++i) {
                const auto other = args.ids[i];
                rank += other < expert || (other == expert && i < slot);
            }
            terms[rank] = __fmul_rn(args.weights[slot], activation::round_linear(projected[0]));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
#pragma unroll
        for (unsigned slot = 0; slot < TopK; ++slot) {
            if (args.ids[slot] >= args.experts) return;
        }
        float result = 0.0F;
#pragma unroll
        for (unsigned slot = 0; slot < TopK; ++slot) result = __fadd_rn(result, terms[slot]);
        args.output[col] = activation::round_linear(result);
    }
}

} // namespace detail

// Decode-only plan. The caller admits a wave32 HIP device or CUDA sm_80+
// and supplies prevalidated weights and a checked route for the token.
template <packed::MatrixFormat Format, unsigned TopK>
#if defined(__HIPCC__)
inline hipError_t launch(const Args& args, hipStream_t stream) {
    if (!fits<Format, TopK>(args)) return hipErrorInvalidValue;
#else
inline cudaError_t launch(const Args& args, cudaStream_t stream) {
    if (!fits<Format, TopK>(args)) return cudaErrorInvalidValue;
#endif
    constexpr auto threads = detail::kThreads;
    const auto columns = (args.intermediate - 1U) / detail::kWarps + 1U;
    detail::gate_up<Format><<<dim3(columns, TopK), threads, 0, stream>>>(args);
#if defined(__HIPCC__)
    if (const auto status = hipGetLastError(); status != hipSuccess) return status;
#else
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
#endif
    detail::down_combine<Format, TopK><<<args.hidden, threads, 0, stream>>>(args);
#if defined(__HIPCC__)
    return hipGetLastError();
#else
    return cudaGetLastError();
#endif
}
#endif

} // namespace strix::moe_core::microscaled_r1
