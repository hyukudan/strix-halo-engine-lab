#pragma once

#include "strix/linux_direct_transport_driver.hpp"
#if defined(__linux__)
#include "strix/linux_direct_gpu_stream_uapi.h"
#endif

namespace strix::transport {
struct DirectGpuStreamDriverObservationPolicy {
    // Trusted serialized privileged ConfigFS owner has selected gpu_direct and
    // excludes mode changes from preflight through close. This is permission
    // to open a provisioned route, not hardware or readiness qualification.
    bool configured_gpu_direct = false;
    std::string expected_kernel_release, expected_driver_build_id;
};
// Actual observations only; QUERY readiness remains unavailable even when the
// ABI is observed. The configured extension mode opens disabled rings. Opening
// an arbitrary stock/copied-mode stream can itself activate paths, before any
// QUERY/ENOTTY; unknown routes must not be opened to infer their compatibility.
struct DirectGpuStreamDriverObservation {
    DirectStreamDeviceIdentity route;
    std::uint64_t open_generation = 0;
    std::string kernel_release, driver_build_id, error;
    DirectNativeQueryStatus status = DirectNativeQueryStatus::unavailable;
    bool query_observed = false, fd_close_requested = false, fd_retained = false;
#if defined(__linux__)
    sgu_query query{};
#endif
};
[[nodiscard]] DirectGpuStreamDriverObservation observe_direct_gpu_stream_driver(
    const DirectStreamDeviceIdentity&, const DirectGpuStreamDriverObservationPolicy&,
    LinuxDirectGpuStreamCalls = LinuxDirectGpuStreamCalls::native());
[[nodiscard]] std::string direct_gpu_stream_driver_observation_json(const DirectGpuStreamDriverObservation&);
} // namespace strix::transport
