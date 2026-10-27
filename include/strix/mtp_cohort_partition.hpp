#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

namespace strix {
// Indices into the live member list, never replacement slot IDs. Each member
// appears once, and bounded controller searches see at most eight requests.
inline std::vector<std::vector<std::size_t>> mtp_cohort_partition(std::size_t count, std::size_t start) {
    std::vector<std::vector<std::size_t>> groups;
    if (!count) return groups;
    start %= count;
    for (std::size_t offset = 0; offset < count;) {
        auto& group = groups.emplace_back();
        const auto size = std::min<std::size_t>(8U, count - offset);
        for (std::size_t j = 0; j < size; ++j) group.push_back((start + offset + j) % count);
        offset += size;
    }
    return groups;
}
} // namespace strix
