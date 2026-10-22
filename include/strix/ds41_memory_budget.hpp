#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

namespace strix::ds41 {

struct HostMemoryRoom {
    std::uint64_t free_commit = 0, free_physical = 0;
    bool known = false;
};

inline HostMemoryRoom host_memory_room() {
#ifdef _WIN32
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) return {s.ullAvailPageFile, s.ullAvailPhys, true};
#endif
    return {};
}

// Bounded CPU tail: request hash IDs, state capture/serialization/retry copies, HTTP queue and runtime margin.
// Device and pinned allocations are already resident when this reservation is checked at admission.
inline std::uint64_t long_request_host_reserve(std::uint32_t context_positions) {
    const auto n = std::uint64_t{context_positions};
    const auto state = 894U * n + (4ULL << 20U);
    return 6U * state + 384U * n + (3ULL << 30U);
}

inline bool host_commit_fits(const HostMemoryRoom& room, std::uint64_t allocation, std::uint64_t floor) {
    // Subtract before comparing: do not wrap on an invalid/extreme requested allocation.
    return !room.known || (room.free_commit >= floor && allocation <= room.free_commit - floor);
}

}  // namespace strix::ds41
