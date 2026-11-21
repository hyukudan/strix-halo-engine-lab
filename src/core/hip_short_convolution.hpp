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
inline hipError_t launch(const ShortConvolution& plan, const Args& args, hipStream_t stream) {
    if (args.rows == 0) return hipSuccess;
    if (!short_conv_device::valid(plan, args)) return hipErrorInvalidValue;

    const auto launch = short_conv_device::launch_plan(plan, args);
    constexpr unsigned threads = short_conv_device::LaunchPlan::threads;
    const dim3 grid(launch.channel_blocks, launch.row_blocks);
    const bool decode = args.rows == 1;
    if (plan.dynamic_groups == 0) {
        if (decode) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<false, true>), grid, dim3(threads), 0, stream,
                               plan, args, launch.group_width);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<false, false>), grid, dim3(threads), 0, stream,
                               plan, args, launch.group_width);
        }
    } else if (decode) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<true, true>), grid, dim3(threads), 0, stream,
                           plan, args, launch.group_width);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(detail::convolve<true, false>), grid, dim3(threads), 0, stream,
                           plan, args, launch.group_width);
    }
    const auto status = hipGetLastError();
    if (status != hipSuccess || decode || plan.taps == 1) return status;

    hipLaunchKernelGGL(detail::commit_prefill, dim3(launch.channel_blocks), dim3(threads), 0, stream, plan, args);
    return hipGetLastError();
}

}  // namespace strix::core::hip_short_conv
