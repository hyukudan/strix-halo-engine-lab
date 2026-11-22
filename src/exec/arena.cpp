#include "strix/exec/arena.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace strix::exec {
namespace {

std::uint64_t add_bytes(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error("arena byte count overflow");
    return left + right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : add_bytes(value, alignment - remainder);
}

}  // namespace

ArenaPlan Arena::plan(std::span<const BufferLifetime> lifetimes, std::uint64_t alignment) {
    if (alignment == 0) throw std::invalid_argument("arena alignment must be nonzero");
    ArenaPlan result;
    result.alignment = alignment;
    result.allocations.reserve(lifetimes.size());
    std::unordered_set<BufferId> ids;
    for (const auto& lifetime : lifetimes) {
        if (lifetime.id == no_buffer || lifetime.bytes == 0 || lifetime.first_launch > lifetime.last_launch)
            throw std::invalid_argument("invalid arena buffer lifetime");
        if (!ids.insert(lifetime.id).second) throw std::invalid_argument("duplicate arena buffer id");
        result.naive_nonaliased_bytes =
            add_bytes(result.naive_nonaliased_bytes, align_up(lifetime.bytes, alignment));
        result.allocations.push_back({lifetime, 0});
    }

    std::sort(result.allocations.begin(), result.allocations.end(), [](const auto& left, const auto& right) {
        const auto& a = left.lifetime;
        const auto& b = right.lifetime;
        if (a.first_launch != b.first_launch) return a.first_launch < b.first_launch;
        if (a.bytes != b.bytes) return a.bytes > b.bytes;
        return a.id < b.id;
    });
    std::vector<const ArenaAllocation*> active;
    for (auto& allocation : result.allocations) {
        std::erase_if(active, [&](const auto* other) {
            return other->lifetime.last_launch < allocation.lifetime.first_launch;
        });
        // Active allocations stay sorted by offset, so the first aligned gap wins.
        std::uint64_t candidate = 0;
        auto gap = active.begin();
        for (; gap != active.end(); ++gap) {
            const auto& other = **gap;
            if (add_bytes(candidate, allocation.lifetime.bytes) <= other.offset) break;
            candidate = align_up(add_bytes(other.offset, other.lifetime.bytes), alignment);
        }
        allocation.offset = candidate;
        result.total_bytes = std::max(result.total_bytes, add_bytes(candidate, allocation.lifetime.bytes));
        active.insert(gap, &allocation);
    }
    result.total_bytes = align_up(result.total_bytes, alignment);
    return result;
}

}  // namespace strix::exec
