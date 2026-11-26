#pragma once

#include "../hip_check.hpp"
#include "arch.hpp"

namespace strix::hip::arch {

namespace detail {

struct DeviceCaps {
    int ordinal = -1;
    std::string arch;
    std::string_view wave32_refusal;
    std::string_view rdna3_refusal;
};

inline const DeviceCaps& current_device_caps() {
    int device = -1;
    strix::hip_check(hipGetDevice(&device), "HIP launch device");
    // HIP ordinals are process-local and architectures are immutable. Refresh when this thread switches devices.
    static thread_local DeviceCaps current;
    if (current.ordinal != device) {
        hipDeviceProp_t properties{};
        strix::hip_check(hipGetDeviceProperties(&properties, device), "HIP launch device properties");
        const auto wave_size = static_cast<unsigned>(properties.warpSize);
        const std::string_view name(properties.gcnArchName);
        current = {device, std::string(name), launch_refusal(name, wave_size, LaunchRequirement::Wave32),
                   launch_refusal(name, wave_size, LaunchRequirement::Rdna3Layout)};
    }
    return current;
}

}  // namespace detail

inline void require_current_device(LaunchRequirement requirement) {
    const auto& current = detail::current_device_caps();
    const auto reason = requirement == LaunchRequirement::Rdna3Layout ? current.rdna3_refusal : current.wave32_refusal;
    if (!reason.empty()) throw std::runtime_error("HIP launch refused on " + current.arch + ": " + std::string(reason));
}

}  // namespace strix::hip::arch

// Keep HIP's own launch syntax, including its parenthesized template names.
#define STRIX_HIP_LAUNCH(requirement, ...) \
    (::strix::hip::arch::require_current_device(::strix::hip::arch::LaunchRequirement::requirement), (__VA_ARGS__))
#define STRIX_HIP_LAUNCH_GGL(requirement, kernel, ...) do { \
    ::strix::hip::arch::require_current_device(::strix::hip::arch::LaunchRequirement::requirement); \
    hipLaunchKernelGGL((kernel), __VA_ARGS__); \
} while (false)
#define STRIX_HIP_EXT_LAUNCH_GGL(requirement, kernel, ...) do { \
    ::strix::hip::arch::require_current_device(::strix::hip::arch::LaunchRequirement::requirement); \
    hipExtLaunchKernelGGL((kernel), __VA_ARGS__); \
} while (false)
