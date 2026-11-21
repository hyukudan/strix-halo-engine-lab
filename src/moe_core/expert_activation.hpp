#pragma once

#include "strix/moe/activation_policy.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace strix::moe_core::expert_activation {

// The first GEMM emits rows [gate | up] in FP32 accumulation storage.
// The activation writes BF16 operands for the expert's down GEMM.
// Strides are in elements. Input and output allocations must not overlap.
struct Args {
    const float* gate_up{};
    std::uint16_t* output{};
    std::size_t rows{};
    std::size_t width{};
    std::size_t gate_up_row{};
    std::size_t output_row{};
    const unsigned* failure{};
};

inline bool fits(const Args& args, float gate_cap, float up_cap) {
    if (args.rows == 0) return true;
    constexpr auto max_input = std::numeric_limits<std::size_t>::max() / sizeof(float);
    constexpr auto max_output = std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t);
    return args.gate_up && args.output && args.width != 0 && args.width <= max_input / 2U &&
           args.gate_up_row >= 2U * args.width && args.output_row >= args.width &&
           args.rows <= max_input / args.gate_up_row && args.rows <= max_output / args.output_row &&
           std::isfinite(gate_cap) && gate_cap > 0.0F && std::isfinite(up_cap) && up_cap > 0.0F;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace detail {

struct DeviceMath {
    __host__ __device__ static float exp(float value) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
        return ::expf(value);
#else
        return std::exp(value);
#endif
    }
    __host__ __device__ static float tanh(float value) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
        return ::tanhf(value);
#else
        return std::tanh(value);
#endif
    }
};

__device__ __forceinline__ float round_linear(float value) {
    return weight::scalar_float::bf16_round(value);
}

__device__ __forceinline__ std::uint16_t activate_bf16(const moe::SoftcappedSwiglu<DeviceMath>& activation,
                                                      float gate, float up) {
    return weight::scalar_float::bf16_encode(activation(round_linear(gate), round_linear(up)));
}

static __global__ void activate(const Args args, const moe::SoftcappedSwiglu<DeviceMath> activation) {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
    if (args.failure && *args.failure != 0U) return;
    const auto first = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const auto step = std::size_t{gridDim.x} * blockDim.x;
    for (std::size_t row = blockIdx.y; row < args.rows; row += gridDim.y) {
        const auto* gate = args.gate_up + row * args.gate_up_row;
        const auto* up = gate + args.width;
        auto* out = args.output + row * args.output_row;
        for (std::size_t col = first; col < args.width; col += step)
            out[col] = activate_bf16(activation, gate[col], up[col]);
    }
}

}  // namespace detail

// Apply the model's input/output BF16 stores around the FP32 activation in
// one launch. Compile with strict floating-point math for the division order.
#if defined(__HIPCC__)
inline hipError_t launch(const Args& args, float gate_cap, float up_cap, hipStream_t stream) {
    if (args.rows == 0) return hipSuccess;
    if (!fits(args, gate_cap, up_cap)) return hipErrorInvalidValue;
#else
inline cudaError_t launch(const Args& args, float gate_cap, float up_cap, cudaStream_t stream) {
    if (args.rows == 0) return cudaSuccess;
    if (!fits(args, gate_cap, up_cap)) return cudaErrorInvalidValue;
#endif

    constexpr unsigned threads = 128;
    constexpr std::size_t grid_limit = 65535;
    const auto columns = static_cast<unsigned>(std::min((args.width - 1U) / threads + 1U, grid_limit));
    const auto rows = static_cast<unsigned>(std::min(args.rows, grid_limit));
    detail::activate<<<dim3(columns, rows), threads, 0, stream>>>(args, {gate_cap, up_cap});
#if defined(__HIPCC__)
    return hipGetLastError();
#else
    return cudaGetLastError();
#endif
}

#endif

}  // namespace strix::moe_core::expert_activation
