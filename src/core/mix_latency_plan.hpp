#pragma once

#include "core/residual_mix_plan.hpp"

namespace strix::core::residual_mix {

struct LatencyGrid {
    unsigned tiles;
    unsigned queries;
    unsigned banks;
};

[[nodiscard]] inline LatencyGrid latency_grid(const MixArgs& args, unsigned tile_width, unsigned max_width) {
    if (tile_width == 0U || args.rows != 1U || args.width > max_width || !fits(args))
        return {};
    return {(args.width - 1U) / tile_width + 1U, args.query_count, args.bank_count};
}

} // namespace strix::core::residual_mix
