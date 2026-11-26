#pragma once

#include <cstddef>
#include <span>

namespace strix::pair {

// Each slot stays owned until both its copy and blocking wire operation finish.
struct StateStaging {
    virtual ~StateStaging() = default;
    virtual std::span<std::byte> buffer(std::size_t slot) = 0;
    virtual void copy_device(bool save, void* device, std::span<std::byte> host) = 0;
};

} // namespace strix::pair
