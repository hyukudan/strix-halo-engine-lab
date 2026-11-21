#pragma once

#include <hip/hip_runtime.h>

#include "core/short_convolution.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace strix::core::hip_short_conv {

struct Args {
    std::size_t rows;
    const float* input;            // [row][channel]
    const float* static_weights;   // [lag][channel], current tap first
    const float* dynamic_weights;  // [row][group][lag], or nullptr with no dynamic groups
    float* history;                // [taps - 1][channel], oldest first
    float* output;                 // [row][channel]
};

namespace detail {

__device__ inline void commit_channel(const ShortConvolution& plan, const Args& args, std::size_t channel) {
    const auto history_rows = plan.taps - 1;
    const auto carried = args.rows < history_rows ? args.rows : history_rows;
    const auto kept = history_rows - carried;
    for (std::size_t h = 0; h < kept; ++h) {
        args.history[h * plan.channels + channel] = args.history[(h + carried) * plan.channels + channel];
    }
    for (std::size_t h = kept; h < history_rows; ++h) {
        const auto row = args.rows - (history_rows - h);
        args.history[h * plan.channels + channel] = args.input[row * plan.channels + channel];
    }
}

template<bool Dynamic, bool Decode>
__global__ void convolve(const ShortConvolution plan, const Args args, const std::size_t group_width) {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
    const std::size_t first_channel = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t channel_stride = std::size_t(gridDim.x) * blockDim.x;
    for (std::size_t channel = first_channel; channel < plan.channels; channel += channel_stride) {
        std::size_t group = 0;
        if constexpr (Dynamic) group = channel / group_width;
        for (std::size_t row = blockIdx.y; row < args.rows; row += gridDim.y) {
            const float* projected_taps = nullptr;
            if constexpr (Dynamic) projected_taps = args.dynamic_weights + (row * plan.dynamic_groups + group) * plan.taps;
            float sum = 0.0F;
            for (std::size_t lag = 0; lag < plan.taps; ++lag) {
                const float sample = lag <= row
                    ? args.input[(row - lag) * plan.channels + channel]
                    : args.history[(plan.taps - 1 - (lag - row)) * plan.channels + channel];
                float weight = args.static_weights[lag * plan.channels + channel];
                if constexpr (Dynamic) weight += projected_taps[lag];
                sum += sample * weight;
            }
            const auto index = row * plan.channels + channel;
            const float residual = plan.residual_weight * args.input[index];
            args.output[index] = __fdiv_rn(sum + residual, plan.output_divisor);
        }
        if constexpr (Decode) commit_channel(plan, args, channel);
    }
}

static __global__ void commit_prefill(const ShortConvolution plan, const Args args) {
    const std::size_t first_channel = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t channel_stride = std::size_t(gridDim.x) * blockDim.x;
    for (std::size_t channel = first_channel; channel < plan.channels; channel += channel_stride) {
        commit_channel(plan, args, channel);
    }
}

}  // namespace detail

// Contiguous, unpadded device buffers with the same shapes and non-overlap
// contract as causal_short_convolution. History is updated on the same stream.
// Decode commits it in each writer thread, prefill starts the commit only after
// all output rows have finished reading the prior history.
inline hipError_t launch(const ShortConvolution& plan, const Args& args, hipStream_t stream) {
    if (args.rows == 0) return hipSuccess;
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!valid_short_convolution_plan(plan) || args.input == nullptr || args.static_weights == nullptr ||
        args.output == nullptr || (plan.taps > 1 && args.history == nullptr) ||
        (plan.dynamic_groups == 0) != (args.dynamic_weights == nullptr) ||
        args.rows > max_elements / plan.channels || plan.taps > max_elements / plan.channels ||
        (plan.dynamic_groups != 0 && args.rows > max_elements / plan.taps / plan.dynamic_groups)) {
        return hipErrorInvalidValue;
    }

    constexpr unsigned threads = 256;
    constexpr std::size_t grid_limit = 65535;
    const auto channel_blocks = static_cast<unsigned>(std::min((plan.channels - 1) / threads + 1, grid_limit));
    const auto row_blocks = static_cast<unsigned>(std::min(args.rows, grid_limit));
    const dim3 grid(channel_blocks, row_blocks);
    const auto group_width = plan.dynamic_groups == 0 ? 0 : plan.channels / plan.dynamic_groups;
    const bool decode = args.rows == 1;
    if (plan.dynamic_groups == 0) {
        if (decode) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<false, true>), grid, dim3(threads), 0, stream,
                               plan, args, group_width);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<false, false>), grid, dim3(threads), 0, stream,
                               plan, args, group_width);
        }
    } else if (decode) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<true, true>), grid, dim3(threads), 0, stream,
                           plan, args, group_width);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<true, false>), grid, dim3(threads), 0, stream,
                           plan, args, group_width);
    }
    const auto status = hipGetLastError();
    if (status != hipSuccess || decode || plan.taps == 1) return status;

    hipLaunchKernelGGL(detail::commit_prefill, dim3(channel_blocks), dim3(threads), 0, stream, plan, args);
    return hipGetLastError();
}

}  // namespace strix::core::hip_short_conv
