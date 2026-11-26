#pragma once

#include "strix/hip/arch.hpp"
#include <hip/hip_runtime_api.h>

namespace strix::hip::arch {

// Only successful properties queries update the calling thread's cache.
// HIP ordinals are process-local; a device switch refreshes the cached target.
inline std::string_view current_device_arch_name() {
    int ordinal = -1;
    if (hipGetDevice(&ordinal) != hipSuccess || ordinal < 0) return {};
    struct CachedTarget {
        int ordinal = -1;
        Target target = Target::Unknown;
    };
    static thread_local CachedTarget cached;
    if (cached.ordinal != ordinal) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, ordinal) != hipSuccess) return {};
        cached = {ordinal, target_from_name(properties.gcnArchName)};
    }
    return cached.target == Target::Unknown ? std::string_view{} : std::string_view(caps(cached.target).name);
}

}  // namespace strix::hip::arch
