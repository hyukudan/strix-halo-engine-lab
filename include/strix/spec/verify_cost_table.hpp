#pragma once

// Measured verify walls of the shared speculator, persisted across runs through the shared CostTableCache.
//
// The Arbiter prices a verify window it has not measured from the windows it has (one known window: +10 % per row,
// and the measured ladder lets the first lookup pass use at most the next window). A sparse MoE pays far more per
// row (measured on one HIP MoE target: 2 rows = 1.43 x 1 row), so on text where lookup proposals are weak the first
// pass of every run is a loss the run then learns from. Seeding the walls a previous run of the same binary, model,
// device and policy measured prices that pass before it runs, so auto drafts only where the measured price pays.
// Seeds give the shape of the cost curve; the run's own walls rescale them (Arbiter::known_ms). Scheduling only:
// emitted ids never change.

#include "strix/cost_table_cache.hpp"
#include "strix/spec/block_drafter.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

namespace strix::spec {

[[nodiscard]] inline std::string verify_cost_entry(std::uint32_t rows) { return "verify.r" + std::to_string(rows); }

// Seeds every warm entry for 1 .. max_rows into the drafter; returns the number of rows seeded.
inline std::uint32_t seed_verify_costs(const CostTableCache& table, BlockDrafter& drafter, std::uint32_t max_rows) {
    std::uint32_t seeded = 0;
    for (std::uint32_t rows = 1; rows <= max_rows; ++rows)
        if (const auto e = table.warm(verify_cost_entry(rows)); e && e->ms > 0.0) {
            drafter.seed_verify_ms(rows, e->ms);
            ++seeded;
        }
    return seeded;
}

// Records the run's mean verify wall per row count (every row count that ran at least once); returns the count.
inline std::uint32_t record_verify_costs(CostTableCache& table, const SpecStats& stats) {
    std::uint32_t recorded = 0;
    const auto n = std::min(stats.rows_passes.size(), stats.rows_ms.size());
    for (std::size_t rows = 1; rows < n; ++rows) {
        const auto passes = stats.rows_passes[rows];
        if (passes == 0 || !(stats.rows_ms[rows] > 0.0)) continue;
        table.record(verify_cost_entry(std::uint32_t(rows)), {stats.rows_ms[rows] / double(passes), passes});
        ++recorded;
    }
    return recorded;
}

// Recheck off: every run re-times the windows it uses (the anchor row always) and its walls rescale the seeds, so a
// rotating cold entry would only re-pay the pass this table exists to remove. The key (executable hash, device,
// model, policy) keeps a different build or target from reading the table.
[[nodiscard]] inline CostTableOptions verify_cost_table_options() {
    auto options = cost_table_options_from_env();
    options.recheck_fraction = 0.0;
    return options;
}

}  // namespace strix::spec
