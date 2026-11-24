#pragma once

#include <stdexcept>
#include <string>

namespace strix {

// A device-memory allocation refused by the accelerator runtime (cudaErrorMemoryAllocation / hipErrorOutOfMemory from
// a device malloc). Callers decide by type, never by message text. `status` is the backend's numeric status. It is a
// std::runtime_error with the unchanged "<operation>: <runtime error string>" message.
class DeviceAllocationError : public std::runtime_error {
public:
    DeviceAllocationError(const std::string& message, int status_code) : std::runtime_error(message), status(status_code) {}
    int status;
};

// The operation whose runtime status failed: a device-memory allocation, or anything else (copies, launches, pinned
// host staging, stream or graph calls).
enum class RuntimeOperation { general, device_allocation };

// The exception for a failed runtime call (device::native::check_device_allocation on CUDA and HIP). An out-of-memory
// status is typed DeviceAllocationError only for a device-memory allocation; any other failure, a pinned host staging
// buffer refused for lack of memory included, is the generic std::runtime_error with the same message, so a retry
// with fewer device rows never answers a host-side refusal.
[[noreturn]] inline void throw_runtime_failure(const std::string& message, int status, bool out_of_memory,
                                               RuntimeOperation operation) {
    if (out_of_memory && operation == RuntimeOperation::device_allocation) throw DeviceAllocationError(message, status);
    throw std::runtime_error(message);
}

}  // namespace strix
