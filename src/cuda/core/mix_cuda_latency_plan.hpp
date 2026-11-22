#pragma once

#include "core/residual_mix.hpp"

namespace strix::cuda::mix {

inline constexpr unsigned kLatencyTile = 512U;
inline constexpr unsigned kLatencyBankThreads = 128U;
inline constexpr unsigned kLatencyMaxWidth = 4096U;

struct LatencyGrid {
    unsigned tiles;
    unsigned queries;
    unsigned banks;
};

[[nodiscard]] inline LatencyGrid latency_grid(const ::strix::core::residual_mix::MixArgs& args) {
    if (args.rows != 1U || args.width > kLatencyMaxWidth || !::strix::core::residual_mix::fits(args))
        return {};
    return {(args.width - 1U) / kLatencyTile + 1U, args.query_count, args.bank_count};
}

} // namespace strix::cuda::mix
