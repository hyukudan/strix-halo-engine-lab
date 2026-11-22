#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace strix::exec {

using BufferId = std::uint32_t;
inline constexpr BufferId no_buffer = std::numeric_limits<BufferId>::max();

struct BufferLifetime {
    BufferId id;
    std::uint32_t first_launch;
    std::uint32_t last_launch;
    std::uint64_t bytes;
};

struct ArenaAllocation {
    BufferLifetime lifetime;
    std::uint64_t offset;
};

struct ArenaPlan {
    std::uint64_t alignment = 1;
    std::uint64_t total_bytes = 0;
    std::uint64_t naive_nonaliased_bytes = 0;
    std::vector<ArenaAllocation> allocations;
};

class Arena {
public:
    // Inclusive lifetimes in serial launch order. Equal-start allocations
    // are placed by decreasing size, then by id. Ids must be unique.
    [[nodiscard]] static ArenaPlan plan(std::span<const BufferLifetime> lifetimes,
                                        std::uint64_t alignment = 256);
};

}  // namespace strix::exec
