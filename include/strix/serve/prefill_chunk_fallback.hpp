#pragma once

#include <cstdint>
#include <exception>
#include <ostream>

#include "strix/device_allocation_error.hpp"

namespace strix::serve {

// Loads a decoder at `chunk` rows. When the chunk is the family's implicit default (not set by the user) and the load
// fails with a typed device allocation failure (strix::DeviceAllocationError, thrown only for an out-of-memory status of
// a device-memory allocation, device::native::check_device_allocation: the row-dependent device buffers; a pinned host
// staging buffer refused for lack of memory is a generic runtime error), the load is retried once at
// `fallback` rows and one line says so. Nothing else is retried: not a message that merely contains "out of memory",
// and not std::bad_alloc, a host allocation failure that fewer device rows do not address.
// An explicit chunk, or a chunk already at or below the fallback, is never changed: the error propagates.
template <class Load>
auto load_with_chunk_fallback(std::uint32_t chunk, bool explicit_chunk, std::uint32_t fallback, std::ostream& log,
                              Load&& load) {
    if (explicit_chunk || fallback == 0U || chunk <= fallback) return load(chunk);
    try {
        return load(chunk);
    } catch (const DeviceAllocationError&) {
        log << "strix-prefill-chunk " << chunk << " -> " << fallback << " (device memory)\n";
    }
    return load(fallback);
}

} // namespace strix::serve
