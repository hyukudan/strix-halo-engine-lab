#pragma once
#include "strix/prefix_state_cache.hpp"
#include <type_traits>

namespace strix {
// Additive storage snapshot. Pool fields stay zero until a resident backend exists.
struct StateCacheSnapshotV1 {
    std::uint32_t schema = 1;
    bool published = false;
    std::uint64_t sequence = 0, published_steady_ns = 0;
    bool prefix_enabled = false, session_enabled = false, pool_present = false;
    PrefixStateCacheStats prefix;
    struct Named {
        std::uint64_t hits = 0, misses = 0, refusals = 0, parked = 0, no_store_skips = 0;
    } named;
    struct Tier {
        std::uint64_t payload_bytes = 0, segments = 0, checkpoints = 0, budget_bytes = 0;
    } resident, host, disk;
    struct Pool {
        std::uint64_t capacity_positions = 0, used_positions = 0, reserved_positions = 0;
        std::uint64_t busy_positions = 0, held_positions = 0, free_positions = 0;
        std::uint64_t largest_free_extent = 0, committed_positions = 0;
        std::uint64_t arena_bytes = 0, fixed_bytes = 0, workspace_bytes = 0, move_staging_bytes = 0;
        std::uint64_t pending_waiters = 0, invalid_positions = 0;
    } pool;
};
static_assert(std::is_trivially_copyable_v<StateCacheSnapshotV1>);
} // namespace strix
