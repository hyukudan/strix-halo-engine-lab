#pragma once

#include "strix/serve/prefix_directory.hpp"

#include <algorithm>
#include <cstdint>
#include <span>

namespace strix::serve {

// In-flight prefix join (shared serve admission policy; engines bind it, the boundary rule is the family's).
// A request whose prompt shares a prefix with a prompt that is still prefilling waits until the producer's state at
// a join boundary exists, then restores that state instead of prefilling the same rows again. The wait never blocks
// a lane: the waiter is only left out of the prefill turn, and any end of the producer (completion, cancel, failure,
// slot reuse) releases it to its own prefill. Exact: the restored state is a checkpoint of the producer's own rows.
// Directory protocol: a producer is announced when its prefill starts (not while it waits itself) and withdrawn when
// its prefill ends; a waiter holds a lease on the producer's entry until it is released.

struct PrefixJoinWait {
    std::uint64_t generation = 0, boundary = 0;
    // current: the producer slot's request generation now (slot reuse releases the waiter). available: the
    // producer's state at boundary exists (family rule: an asynchronous capture once the position has passed the
    // boundary, a synchronous capture once the position reaches it).
    [[nodiscard]] bool pending(std::uint64_t current, bool active, bool prefilling, bool available) const noexcept {
        return boundary != 0 && current == generation && active && prefilling && !available;
    }
};

struct PrefixJoinChoice {
    std::uint64_t entry = 0;     // producer's directory entry; 0 = no join
    std::uint32_t lane = 0;      // producer lane as announced
    std::uint64_t boundary = 0;  // join boundary in tokens
    explicit operator bool() const noexcept { return entry != 0; }
};

// Producer with the largest boundary above floor_tokens (the rows the waiter already restored) among the in-flight
// entries that share a prefix with tokens; ties keep the older entry. boundary_of(match, lane) is the family trait:
// the join boundary that producer can serve now, 0 when it cannot (its plan, its position, its request kind).
template <class Boundary>
PrefixJoinChoice choose_prefix_join(const PrefixDirectory& directory, const PrefixKey& key,
                                    std::span<const std::uint32_t> tokens, std::uint64_t floor_tokens,
                                    Boundary&& boundary_of) {
    PrefixJoinChoice best;
    for (const PrefixMatch& m : directory.inflight_matches(key, tokens)) {
        const std::uint32_t lane = directory.status(m.entry).lane;
        const std::uint64_t boundary = boundary_of(m, lane);
        if (boundary > floor_tokens && boundary > best.boundary) best = PrefixJoinChoice{m.entry, lane, boundary};
    }
    return best;
}

// Boundary trait for block-grid checkpoints (a checkpoint can be taken at any multiple of block and also carries the
// token at its position): the deepest multiple P of block with P < usable (the token at P is shared), P <= n - 2 for
// both prompts (each keeps a suffix row), and P beyond the producer's frontier (the rows it has run or launched).
// 0 when no such P exists.
[[nodiscard]] inline std::uint64_t prefix_join_block_boundary(std::uint64_t usable, std::uint64_t waiter_tokens,
                                                              std::uint64_t producer_tokens,
                                                              std::uint64_t producer_frontier,
                                                              std::uint32_t block) noexcept {
    if (block == 0 || usable == 0 || waiter_tokens < 2 || producer_tokens < 2) return 0;
    const std::uint64_t top = std::min({usable - 1, waiter_tokens - 2, producer_tokens - 2});
    const std::uint64_t p = top / block * block;
    return p > producer_frontier ? p : 0;
}

}  // namespace strix::serve
