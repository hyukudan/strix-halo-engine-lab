#pragma once

#include <stdexcept>
#include <string>

namespace strix {

// A device allocation refused by the accelerator runtime (cudaErrorMemoryAllocation / hipErrorOutOfMemory). The
// runtime status check throws it, so callers decide by type, never by message text. `status` is the backend's
// numeric status. It is a std::runtime_error with the unchanged "<operation>: <runtime error string>" message.
class DeviceAllocationError : public std::runtime_error {
public:
    DeviceAllocationError(const std::string& message, int status_code) : std::runtime_error(message), status(status_code) {}
    int status;
};

}  // namespace strix
