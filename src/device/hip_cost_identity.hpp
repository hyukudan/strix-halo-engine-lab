#pragma once

#include "strix/device/cost_identity.hpp"

#include <hip/hip_runtime.h>

// See strix/device/hip_enumerator.hpp: keep HIP's empty host __noinline__ out of standard headers.
#if defined(__GNUC__) && !defined(__clang__) && defined(__noinline__)
#undef __noinline__
#endif

#include <string>

namespace strix::device {

// Cost-table key of the current HIP device, or empty when the runtime cannot describe it (no table is used then).
[[nodiscard]] inline std::string hip_cost_device_identity() {
    int device = 0, runtime = 0, driver = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess ||
        hipRuntimeGetVersion(&runtime) != hipSuccess || hipDriverGetVersion(&driver) != hipSuccess)
        return {};
    return cost_device_identity({properties.gcnArchName, properties.name, properties.multiProcessorCount,
                                 properties.totalGlobalMem, properties.clockRate, "hip", runtime, driver});
}

}  // namespace strix::device
