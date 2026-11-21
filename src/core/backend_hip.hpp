#pragma once
#include "backend.hpp"
#include <hip/hip_runtime.h>

namespace strix::core {
// Creation-time probe only; unknown HIP errors propagate to the caller.
hipError_t query_fused_output_resources(int expected_device,
    size_t shared_bytes,bool* supported);
hipError_t launch_linear_planned(const DeviceWeights& weights, const float* input,
    float* output, int rows, void* workspace, uint32_t* counters, hipStream_t stream,
    float alpha, float beta, const LinearExecution& execution);
hipError_t launch_linear(const DeviceWeights& weights, const float* input,
    float* output, int rows, void* workspace, hipStream_t stream,
    float alpha, float beta, bool prefer_wmma);
// launch_linear with the output transform folded into the core for the cases above.
hipError_t launch_linear_completed(const DeviceWeights& weights, const float* input,
    float* output, int rows, void* workspace, uint32_t* counters, hipStream_t stream,
    float alpha, float beta, bool prefer_wmma);
}
