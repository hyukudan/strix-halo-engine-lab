#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strix::serve {

inline constexpr unsigned kMaxQueueOvertakes = 8;

// Stable shortest-first, arrival-order ties, one eight-overtake bound.
// An ineligible aged owner blocks admission rather than losing its starvation bound.
// count limits MiniMax cancellation batches; returning count means no eligible owner.
struct AllWaitingEligible {
    template <class Entry> bool operator()(const Entry&) const { return true; }
};
template <class Container, class Size, class Skips, class Eligible = AllWaitingEligible>
std::size_t select_waiting(Container& entries, bool short_first, Size size, Skips skips,
                         std::size_t count = static_cast<std::size_t>(-1), Eligible eligible = {}) {
    count = std::min(count, entries.size());
    if (count == 0) return 0;
    if (!short_first) return eligible(entries[0]) ? 0 : count;
    std::size_t best = count;
    for (std::size_t i = 0; i < count; ++i) {
        if (skips(entries[i]) >= kMaxQueueOvertakes) {
            if (!eligible(entries[i])) return count;
            best = i;
            break;
        }
        if (eligible(entries[i]) && (best == count || size(entries[i]) < size(entries[best]))) best = i;
    }
    if (best != count)
        for (std::size_t i = 0; i < best; ++i) ++skips(entries[i]);
    return best;
}

}  // namespace strix::serve
