#pragma once

#include <cuda_runtime.h>

#include "core/short_convolution_device.hpp"

namespace strix::cuda_core::short_conv {

using core::short_conv_device::Args;

// Same contiguous device-buffer contract as the HIP binding. The existing
// CUDA core compile policy disables FMA contraction and FTZ for parity with the
// scalar reference. History is committed after the dependent reads on stream.
inline cudaError_t launch(const core::ShortConvolution& plan, const Args& args, cudaStream_t stream) {
    if (args.rows == 0) return cudaSuccess;
    namespace device = core::short_conv_device;
    if (!device::valid(plan, args)) return cudaErrorInvalidValue;

    const auto config = device::launch_plan(plan, args);
    constexpr unsigned threads = device::LaunchPlan::threads;
    const dim3 grid(config.channel_blocks, config.row_blocks);
    const bool decode = args.rows == 1;
    const bool dynamic = plan.dynamic_groups != 0;
    if (!dynamic && decode) {
        device::detail::convolve<false, true><<<grid, threads, 0, stream>>>(plan, args, config.group_width);
    } else if (!dynamic) {
        device::detail::convolve<false, false><<<grid, threads, 0, stream>>>(plan, args, config.group_width);
    } else if (decode) {
        device::detail::convolve<true, true><<<grid, threads, 0, stream>>>(plan, args, config.group_width);
    } else {
        device::detail::convolve<true, false><<<grid, threads, 0, stream>>>(plan, args, config.group_width);
    }

    const auto status = cudaGetLastError();
    if (status != cudaSuccess || decode || plan.taps == 1) return status;
    device::detail::commit_prefill<<<config.channel_blocks, threads, 0, stream>>>(plan, args);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core::short_conv
