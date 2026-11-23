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

// grid.queries counts the jobs of every set (blockIdx.y = job).
[[nodiscard]] inline LatencyGrid latency_batch_grid(const ::strix::core::residual_mix::BankBatchArgs& args) {
    if (args.width > kLatencyMaxWidth || !::strix::core::residual_mix::fits(args)) return {};
    const auto jobs = ::strix::core::residual_mix::job_count(args);
    if (jobs > 65535U) return {};
    return {(args.width - 1U) / kLatencyTile + 1U, jobs, args.bank_count};
}

} // namespace strix::cuda::mix
