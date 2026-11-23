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
#include <optional>
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

// Records the run's warm mean verify wall per row count (SpecStats::warm_ms: without the width's first pass, which can
// carry graph capture or lazy allocation; a row count that ran only once is not recorded); returns the count. A cold
// first wall stored as a seed would price that width (the plain row: every alternative) in every later run.
inline std::uint32_t record_verify_costs(CostTableCache& table, const SpecStats& stats) {
    std::uint32_t recorded = 0;
    for (std::size_t rows = 1; rows < stats.rows_passes.size(); ++rows) {
        const double ms = stats.warm_ms(rows);
        if (!(ms > 0.0)) continue;
        table.record(verify_cost_entry(std::uint32_t(rows)), {ms, stats.rows_passes[rows] - 1U});
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

// A prior must carry its own scale anchor: the run always times the one-row window, and Arbiter::known_ms rescales a
// seed only from a window that is both measured and seeded. A table without a warm verify.r1 would price the run's
// wider windows at another context's raw walls; one without a warm wider window (2 .. max_rows) gives no shape.
[[nodiscard]] inline bool verify_cost_prior_usable(const CostTableCache& table, std::uint32_t max_rows) {
    const auto anchor = table.warm(verify_cost_entry(1U));
    if (!anchor || !(anchor->ms > 0.0)) return false;
    for (std::uint32_t rows = 2U; rows <= max_rows && rows != 0U; ++rows)
        if (const auto e = table.warm(verify_cost_entry(rows)); e && e->ms > 0.0) return true;
    return false;
}

// A run whose own context bucket was never timed (load() != loaded) borrows the curve SHAPE of the nearest timed and
// usable (verify_cost_prior_usable) bucket of the same key (engine, binary, device, model, policy): bucket - 1,
// bucket + 1, bucket - 2, ... up to `reach` buckets away, inside 0 .. kCostTableMaxContextBucket. The seeds give shape
// only; the run's own walls rescale them (Arbiter::known_ms), and the run records into its own bucket, never into the
// prior's. Read only (no lock, no write). Scheduling only: ids never change.
[[nodiscard]] inline std::optional<CostTableCache> open_nearest_verify_cost_prior(const CostTableCache& table,
                                                                                 std::uint32_t max_rows,
                                                                                 std::uint32_t reach = 3U) {
    if (table.load_status() == CostTableLoad::loaded || table.file().empty()) return std::nullopt;
    const std::int64_t own = table.key().context_bucket, last = kCostTableMaxContextBucket;
    if (own > last) return std::nullopt;
    const auto directory = table.file().parent_path();
    // Beyond the whole bucket range no candidate is valid, so the search is bounded whatever reach says.
    const std::int64_t bound = std::min<std::int64_t>(reach, last);
    for (std::int64_t distance = 1; distance <= bound; ++distance)
        for (const std::int64_t candidate : {own - distance, own + distance}) {
            if (candidate < 0 || candidate > last) continue;
            auto key = table.key();
            key.context_bucket = static_cast<std::uint32_t>(candidate);
            std::optional<CostTableCache> prior;
            prior.emplace(directory, std::move(key), verify_cost_table_options());
            if (prior->load() == CostTableLoad::loaded && verify_cost_prior_usable(*prior, max_rows)) return prior;
        }
    return std::nullopt;
}

}  // namespace strix::spec
