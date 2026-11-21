#pragma once

#include "inkling2_mini_config.hpp"
#include "inkling2_mini_schedule.hpp"
#include "strix/attn/cache_rows.hpp"

#include <cstdint>

namespace strix::inkling2 {

constexpr bool uses_sliding_attention(LayerVisit visit) {
    return !is_full_attention(visit.physical_layer);
}

struct MiniVisitStatePlan {
    attn::cache_rows::Rows kv;
    unsigned kv_rows;
    std::array<unsigned, 4> convolution_widths;

    constexpr std::uint64_t kv_bytes() const {
        return std::uint64_t{kv_rows} * kv.width * sizeof(std::uint16_t) * 2U;
    }
};

// An entire chunk may be written before the first query reads it.
// The ring must hold the previous window tail alongside those new rows.
constexpr MiniVisitStatePlan visit_state_plan(LayerVisit visit, unsigned max_context, unsigned chunk_rows) {
    if (max_context == 0 || max_context > kMaxContext || chunk_rows == 0 || chunk_rows > max_context) {
        throw std::invalid_argument("invalid Inkling 2 Mini context/chunk size");
    }
    const unsigned sliding_rows = attn::cache_rows::window_ring(kSlidingWindow, chunk_rows);
    const unsigned ring = uses_sliding_attention(visit) && sliding_rows < max_context ? sliding_rows : 0U;
    return {{kKvWidth, ring}, ring == 0 ? max_context : ring, {kKvWidth, kKvWidth, kHiddenWidth, kHiddenWidth}};
}

}  // namespace strix::inkling2
