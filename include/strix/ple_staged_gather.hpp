#pragma once

#include "strix/ple_lookup.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix::ple_detail {

// The caller checks worker success, slot, position, source and byte extent before
// calling. Only a successfully finished independent row-0 lookup may be omitted
// from deferred. Rejection leaves the inline gather's work list untouched.
[[nodiscard]] inline std::size_t accept_staged_gather(
    std::span<const PleHashResult> hashes, std::uint32_t heads,
    std::span<const std::uint32_t> staged_ids, bool first_row_consumed,
    std::vector<std::size_t>& deferred) {
    if (hashes.empty() || !heads || heads > PleHashContract::maximum_heads ||
        hashes.size() > staged_ids.size() / heads)
        return 0;
    const std::size_t first_deferred = first_row_consumed ? 1U : 0U;
    if (deferred.size() != hashes.size() - first_deferred) return 0;
    for (std::size_t i = 0; i < deferred.size(); ++i)
        if (deferred[i] != i + first_deferred) return 0;
    // Compare every row, including row 0 even if it was consumed independently.
    for (std::size_t r = 0; r < hashes.size(); ++r) {
        if (hashes[r].row_count != heads) return 0;
        for (std::uint32_t h = 0; h < heads; ++h)
            if (hashes[r].row_ids[h] != staged_ids[r * heads + h]) return 0;
    }
    deferred.clear();
    return hashes.size();
}

}  // namespace strix::ple_detail
