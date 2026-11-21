#pragma once

#include "inkling2_mini_config.hpp"
#include "inkling2_mini_schedule.hpp"
#include "strix/attn/cache_rows.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::inkling2 {

constexpr bool uses_sliding_attention(LayerVisit visit) {
    return !is_full_attention(visit.physical_layer);
}

struct MiniVisitStatePlan {
    attn::cache_rows::Rows kv{};
    unsigned kv_rows{};
    std::array<unsigned, 4> convolution_widths{};
    unsigned window{};  // The semantic attention window, independent of backing slots.
    std::uint64_t kv_offset{};  // Elements from the start of each separate K and V arena.
    std::array<std::uint64_t, 4> convolution_offsets{};  // FP32 elements from the convolution arena.

    constexpr std::uint64_t kv_elements() const {
        return std::uint64_t{kv_rows} * kv.width;
    }

    constexpr std::uint64_t kv_bytes() const {
        return kv_elements() * sizeof(std::uint16_t) * 2U;
    }

    constexpr std::uint64_t convolution_elements(unsigned stream) const {
        return std::uint64_t{kConvTaps - 1U} * convolution_widths.at(stream);
    }
};

// An entire chunk may be written before the first query reads it.
// The ring must hold the previous window tail alongside those new rows.
constexpr MiniVisitStatePlan visit_state_plan(LayerVisit visit, unsigned max_context, unsigned max_chunk_rows) {
    if (!valid_mini_visit(visit) || max_context == 0 || max_context > kMaxContext ||
        max_chunk_rows == 0 || max_chunk_rows > max_context) {
        throw std::invalid_argument("invalid Inkling 2 Mini visit, context or maximum chunk size");
    }
    const unsigned window = uses_sliding_attention(visit) ? kSlidingWindow : 0U;
    const unsigned sliding_rows = window != 0 ? attn::cache_rows::window_ring(window, max_chunk_rows) : 0U;
    const unsigned ring = sliding_rows < max_context ? sliding_rows : 0U;
    return {{kKvWidth, ring}, ring == 0 ? max_context : ring, {kKvWidth, kKvWidth, kHiddenWidth, kHiddenWidth}, window};
}

// Bind once per session. All appends, from multirow prefill to one-row decode,
// address the same logical visit, ring and arena slices until session reset.
class MiniSessionStatePlan {
public:
    constexpr MiniSessionStatePlan(unsigned max_context, unsigned max_chunk_rows)
        : max_context_(max_context), max_chunk_rows_(max_chunk_rows) {
        for (const auto visit : mini_schedule()) {
            auto plan = visit_state_plan(visit, max_context_, max_chunk_rows_);
            plan.kv_offset = kv_elements_;
            kv_elements_ += plan.kv_elements();
            for (unsigned stream = 0; stream < plan.convolution_offsets.size(); ++stream) {
                plan.convolution_offsets[stream] = convolution_elements_;
                convolution_elements_ += plan.convolution_elements(stream);
            }
            visits_[visit.logical_layer] = plan;
        }
    }

    constexpr bool admits(unsigned first_position, unsigned count) const {
        return count != 0 && count <= max_chunk_rows_ && first_position < max_context_ &&
               count <= max_context_ - first_position;
    }

    constexpr const MiniVisitStatePlan& visit(LayerVisit visit) const {
        if (!valid_mini_visit(visit)) throw std::invalid_argument("invalid Inkling 2 Mini state visit");
        return visits_[visit.logical_layer];
    }

    constexpr std::uint64_t kv_elements_per_plane() const { return kv_elements_; }
    constexpr std::uint64_t kv_bytes() const { return kv_elements_ * sizeof(std::uint16_t) * 2U; }
    constexpr std::uint64_t convolution_elements() const { return convolution_elements_; }
    constexpr std::uint64_t convolution_bytes() const { return convolution_elements_ * sizeof(float); }

private:
    unsigned max_context_;
    unsigned max_chunk_rows_;
    std::uint64_t kv_elements_ = 0;
    std::uint64_t convolution_elements_ = 0;
    std::array<MiniVisitStatePlan, kLogicalVisits> visits_{};
};

struct MiniSessionKvPlanes {
    std::span<std::uint16_t> keys;
    std::span<std::uint16_t> values;
};

struct MiniVisitKvCache {
    std::uint16_t* keys{};
    std::uint16_t* values{};
    unsigned slots{};
    bool wrapped{};
};

// Both backends address the same slices of the admitted session cache.
// The key and value planes must be disjoint.
inline MiniVisitKvCache mini_visit_kv_cache(const MiniSessionStatePlan& session, LayerVisit visit,
                                             MiniSessionKvPlanes planes) {
    if (!valid_mini_visit(visit) || planes.keys.size() < session.kv_elements_per_plane() ||
        planes.values.size() < session.kv_elements_per_plane() || planes.keys.data() == planes.values.data()) {
        return {};
    }
    const auto& state = session.visit(visit);
    return {planes.keys.data() + state.kv_offset, planes.values.data() + state.kv_offset,
            state.kv_rows, state.kv.ring != 0U};
}

}  // namespace strix::inkling2
