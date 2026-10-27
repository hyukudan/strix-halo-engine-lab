#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace strix::decode_cli {
// Pure admission policy; callers observe memory only when demand requires growth.
// Grow when every built slot is busy and work waits, or (spare) as soon as the last free slot is taken, so a request
// arriving during a long prompt finds a built slot (nested admission needs one inside the block).
inline bool http_slot_should_grow(bool queued, std::size_t active, std::size_t built,
    std::size_t capacity, bool memory_ok, bool spare = false) {
    return (queued || spare) && active == built && built < capacity && memory_ok;
}
inline std::uint32_t http_eager_slots(std::uint32_t capacity, std::uint32_t eager = 4) {
    return eager ? std::min(capacity, eager) : capacity;
}
inline bool http_slot_memory_fits(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available,
    std::optional<std::uint64_t> gtt_available, std::uint64_t headroom) {
    if (!available || !gtt_available) return false;
    const auto bytes = std::min(*available, *gtt_available);
    if (bytes < headroom || bytes - headroom < retained) return false;
    const auto remaining = bytes - headroom - retained;
    return remaining >= active_peak;
}
struct HttpSlotCapacity {
    std::uint32_t slots = 1;
    std::string limiting = "missing memory observations";
};
// Observations are remaining capacity after the primary stack, before the probe.
// Retained bytes charge only extra stacks; active peaks charge every runnable slot.
inline HttpSlotCapacity resolve_http_slot_capacity(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available, std::optional<std::uint64_t> gtt_available,
    std::uint64_t headroom, std::uint32_t cap = 0) {
    if (!available || !gtt_available || !retained) return {};
    if (active_peak > std::numeric_limits<std::uint64_t>::max() - retained)
        return {1, "cost overflow"};
    const auto per_slot = retained + active_peak;
    const auto count = [&](std::uint64_t bytes) {
        if (bytes < headroom || bytes - headroom < active_peak) return UINT64_C(1);
        const auto extra = (bytes - headroom - active_peak) / per_slot;
        return UINT64_C(1) + std::min(extra, static_cast<std::uint64_t>(UINT32_MAX) - 1U);
    };
    const auto uma = count(*available), gtt = count(*gtt_available);
    auto slots = std::min({uma, gtt, static_cast<std::uint64_t>(UINT32_MAX)});
    std::string limiting = uma <= gtt ? "MemAvailable budget" : "GTT budget";
    if (cap && cap <= slots) { slots = cap; limiting = "requested cap"; }
    return {static_cast<std::uint32_t>(slots), std::move(limiting)};
}
} // namespace strix::decode_cli
