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

// Row-owner transition (residual.transition) as ONE thread block cluster: block r owns tile r of kLatencyTile values
// of the row (raw readout, add store, normalize and a8g32 of that tile), every block recomputes the bank scores of
// the whole row (the tiled readout's redundant schedule), and the RMS sum of a query is taken in the rms.bf16 lane
// order (lane t of kTransitionNormLanes adds x[t + kTransitionNormLanes k] in ascending k) by reading the peers'
// tiles through distributed shared memory. Tiles are multiples of 32 values apart, so a8g32 groups never straddle.
inline constexpr unsigned kTransitionNormLanes = 256U;
inline constexpr unsigned kTransitionMaxClusterTiles = 8U;  // the portable cluster size
static_assert(kLatencyMaxWidth <= kTransitionMaxClusterTiles * kLatencyTile);
static_assert(kLatencyTile % 32U == 0U && kLatencyTile % kTransitionNormLanes == 0U);
static_assert(::strix::core::residual_mix::kTransitionMaxWidth == kLatencyMaxWidth);

struct TransitionPlan {
    unsigned tiles;   // blocks = cluster size; 0 = not admitted
    unsigned slices;  // block = kLatencyBankThreads x slices (one slice per bank, at least the norm lanes)
};

[[nodiscard]] constexpr TransitionPlan transition_plan(unsigned width, unsigned banks) {
    if (width == 0U || width > kLatencyMaxWidth || banks == 0U || banks > ::strix::core::residual_mix::kMaxBanks) return {};
    const unsigned min_slices = kTransitionNormLanes / kLatencyBankThreads;
    return {(width - 1U) / kLatencyTile + 1U, banks < min_slices ? min_slices : banks};
}

// Owner block (cluster rank) and offset inside its tile of row value `index` (the kernel and the host test share them).
#if defined(__CUDACC__)
#define STRIX_TRANSITION_HD __host__ __device__
#else
#define STRIX_TRANSITION_HD
#endif
[[nodiscard]] STRIX_TRANSITION_HD constexpr unsigned transition_owner(unsigned index) { return index / kLatencyTile; }
[[nodiscard]] STRIX_TRANSITION_HD constexpr unsigned transition_offset(unsigned index) { return index % kLatencyTile; }
#undef STRIX_TRANSITION_HD

} // namespace strix::cuda::mix
