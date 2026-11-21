#pragma once

#include "inkling2_mini_config.hpp"
#include "inkling2_mini_schedule.hpp"
#include "strix/session/kv_row_append.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::inkling2 {

constexpr bool uses_sliding_attention(LayerVisit visit) {
    return !is_full_attention(visit.physical_layer);
}

using MiniVisitStatePlan = session::VisitCacheLayout<4>;

constexpr MiniVisitStatePlan visit_state_plan(LayerVisit visit, unsigned max_context, unsigned max_chunk_rows) {
    if (!valid_mini_visit(visit) || max_context == 0 || max_context > kMaxContext ||
        max_chunk_rows == 0 || max_chunk_rows > max_context) {
        throw std::invalid_argument("invalid Inkling 2 Mini visit, context or maximum chunk size");
    }
    const unsigned window = uses_sliding_attention(visit) ? kSlidingWindow : 0U;
    return session::make_visit_cache_layout(max_context, max_chunk_rows, kKvWidth, window,
                                            std::array{kKvWidth, kKvWidth, kHiddenWidth, kHiddenWidth},
                                            kConvTaps - 1U);
}

struct MiniCacheTraits {
    using Visit = LayerVisit;
    static constexpr auto schedule() { return mini_schedule(); }
    static constexpr unsigned index(Visit visit) { return visit.logical_layer; }
    static constexpr bool valid(Visit visit) { return valid_mini_visit(visit); }
    static constexpr MiniVisitStatePlan layout(Visit visit, unsigned context, unsigned max_chunk) {
        return visit_state_plan(visit, context, max_chunk);
    }
};

using MiniSessionStatePlan = session::VisitCachePlan<MiniCacheTraits>;
using MiniSessionKvPlanes = session::KvRowPlanes;
using MiniVisitKvCache = session::VisitKvRows;

inline MiniVisitKvCache mini_visit_kv_cache(const MiniSessionStatePlan& plan, LayerVisit visit,
                                           MiniSessionKvPlanes planes) {
    return session::slice_kv_visit(plan, visit, planes);
}

}  // namespace strix::inkling2
