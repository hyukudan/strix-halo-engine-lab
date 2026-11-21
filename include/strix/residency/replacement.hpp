#pragma once

#include <optional>

namespace strix::residency {

// A missing rank protects an entry. Lower ranks win, with ties settled by
// traversal order. The minimum rank lets an already ordered caller stop at
// its first eligible entry.
template<class Iterator, class Rank, class Priority>
[[nodiscard]] Iterator select_victim(Iterator first, Iterator last, Rank minimum, Priority priority) {
    auto victim = last;
    std::optional<Rank> best;
    for (auto it = first; it != last; ++it) {
        const auto rank = priority(*it);
        if (!rank) continue;
        if (*rank == minimum) return it;
        if (!best || *rank < *best) {
            victim = it;
            best = rank;
        }
    }
    return victim;
}

template<class Iterator, class Eligible>
[[nodiscard]] Iterator select_oldest(Iterator oldest, Iterator end, Eligible eligible) {
    return select_victim(oldest, end, 0, [&](const auto& entry) -> std::optional<int> {
        if (eligible(entry)) return 0;
        return std::nullopt;
    });
}

}  // namespace strix::residency
