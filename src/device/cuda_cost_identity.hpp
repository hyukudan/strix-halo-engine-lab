#pragma once

#include "strix/device/cost_identity.hpp"

#include <cuda_runtime.h>

#include <string>

namespace strix::device {

// Cost-table key of the current CUDA device, or empty when the runtime cannot describe it (no table is used then).
// The clock is read as an attribute (cudaDeviceProp::clockRate is gone from recent toolkits).
[[nodiscard]] inline std::string cuda_cost_device_identity() {
    int device = 0, runtime = 0, driver = 0, clock = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, device) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime) != cudaSuccess || cudaDriverGetVersion(&driver) != cudaSuccess)
        return {};
    return cost_device_identity({"sm_" + std::to_string(properties.major) + std::to_string(properties.minor),
                                 properties.name, properties.multiProcessorCount, properties.totalGlobalMem, clock,
                                 "cuda", runtime, driver});
}

}  // namespace strix::device
