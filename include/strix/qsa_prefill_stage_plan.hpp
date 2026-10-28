#pragma once
#include "strix/ctx_exact_memory.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace strix {
// Up to capacity/64 existing groups per lease, alternating the owner's two slots.
// Tails retain the incumbent 32/16/8-row fallback decomposition.
struct QsaPrefillStagePlan {
    static_assert(QsaScoreRingGeometry::slots == 2U);
    std::uint32_t rows, score_rows, capacity;
    QsaPrefillStagePlan(std::uint32_t count, unsigned call_score_rows, std::uint32_t slot_rows=64U)
        : rows(count), score_rows(call_score_rows), capacity(slot_rows) {
        if (!qsa_score_ring_rows_valid(capacity) || !rows || rows % 8U || (score_rows != 16U && score_rows != 32U && score_rows != 64U))
            throw std::invalid_argument("QSA stage slice geometry");
    }
    std::uint32_t slices() const { return (rows - 1U) / capacity + 1U; }
    std::uint32_t first(std::uint32_t slice) const {
        if (slice >= slices()) throw std::out_of_range("QSA stage slice");
        return slice * capacity;
    }
    std::uint32_t count(std::uint32_t slice) const { return std::min(capacity, rows - first(slice)); }
};
inline unsigned qsa_auto_score_rows(std::uint32_t call_max_blocks, bool allow64) {
    if (allow64 && call_max_blocks >= 65536U) return 64U;
    return call_max_blocks > 4096U ? 32U : 16U;
}
} // namespace strix
