#pragma once

#include <hip/hip_runtime.h>

#include "core/short_convolution_device.hpp"

namespace strix::core::hip_short_conv {

using short_conv_device::Args;
namespace detail = short_conv_device::detail;

// Contiguous, unpadded device buffers with the same shapes and non-overlap
// contract as causal_short_convolution. History is updated on the same stream.
// Decode commits it in each writer thread, prefill starts the commit only after
// all output rows have finished reading the prior history.
template <bool RoundOutput>
inline hipError_t launch_with_store(const ShortConvolution& plan, const Args& args, hipStream_t stream) {
    if (args.rows == 0) return hipSuccess;
    if (!short_conv_device::valid(plan, args)) return hipErrorInvalidValue;

    const auto config = short_conv_device::launch_plan(plan, args);
    constexpr unsigned threads = short_conv_device::LaunchPlan::threads;
    const dim3 grid(config.channel_blocks, config.row_blocks);
    const bool decode = args.rows == 1;
    if (plan.dynamic_groups == 0) {
        if (decode) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<false, true, RoundOutput>), grid, dim3(threads), 0, stream,
                               plan, args, config.group_width);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<false, false, RoundOutput>), grid, dim3(threads), 0, stream,
                               plan, args, config.group_width);
        }
    } else if (decode) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<true, true, RoundOutput>), grid, dim3(threads), 0, stream,
                           plan, args, config.group_width);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<true, false, RoundOutput>), grid, dim3(threads), 0, stream,
                           plan, args, config.group_width);
    }
    const auto status = hipGetLastError();
    if (status != hipSuccess || decode || plan.taps == 1) return status;

    hipLaunchKernelGGL(detail::commit_prefill, dim3(config.channel_blocks), dim3(threads), 0, stream, plan, args);
    return hipGetLastError();
}

inline hipError_t launch(const ShortConvolution& plan, const Args& args, hipStream_t stream) {
    return launch_with_store<false>(plan, args, stream);
}

}  // namespace strix::core::hip_short_conv
