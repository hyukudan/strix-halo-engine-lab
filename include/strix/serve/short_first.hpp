#pragma once

#include <cstddef>
#include <cstdint>

namespace strix::serve {

// Stable shortest-first with at most eight overtakes of any older waiting entry.
// Call only for eligible work, in arrival order; counters travel with their owners.
template <class Container, class Size, class Skips>
std::size_t select_waiting(Container& entries, bool short_first, Size size, Skips skips) {
    std::size_t best = 0;
    if (!short_first || entries.empty()) return best;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (skips(entries[i]) >= 8U) { best = i; break; }
        if (size(entries[i]) < size(entries[best])) best = i;
    }
    for (std::size_t i = 0; i < best; ++i) ++skips(entries[i]);
    return best;
}

}  // namespace strix::serve
