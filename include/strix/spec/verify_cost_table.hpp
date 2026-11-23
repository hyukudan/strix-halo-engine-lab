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
//
// Frozen prices (table_prices_run): once a table prices every verify window the run may use (and a chain draft depth
// when a chain drafter runs), the run decides from the table alone (BlockConfig::fixed_costs). Learned in-run walls
// made the decisions a function of host-clock noise and of the previous writer's walls: measured on CUDA Inkling 2 Mini
// code, the same binary and prompt ran 97 then 157 passes (the learned drafter ran 57 then 8 passes), so an A/B of two
// builds read the drafting path, not the kernels. Every run still records its own warm walls for the next one.

#include "strix/cost_table_cache.hpp"
#include "strix/spec/block_drafter.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
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

[[nodiscard]] inline std::string draft_cost_entry(std::uint32_t steps) { return "draft.k" + std::to_string(steps); }

// Seeds every warm chain draft wall for 1 .. max_k into the drafter; returns the number of depths seeded.
inline std::uint32_t seed_draft_costs(const CostTableCache& table, BlockDrafter& drafter, std::uint32_t max_k) {
    std::uint32_t seeded = 0;
    for (std::uint32_t k = 1; k <= max_k; ++k)
        if (const auto e = table.warm(draft_cost_entry(k)); e && e->ms > 0.0) {
            drafter.seed_chain_ms(k, e->ms);
            ++seeded;
        }
    return seeded;
}

// Records the run's warm mean chain wall per depth (BlockDrafter::measured_chain_ms); returns the count.
inline std::uint32_t record_draft_costs(CostTableCache& table, const BlockDrafter& drafter, std::uint32_t max_k) {
    std::uint32_t recorded = 0;
    for (std::uint32_t k = 1; k <= max_k; ++k)
        if (const auto w = drafter.measured_chain_ms(k); w.samples > 0U && w.ms > 0.0) {
            table.record(draft_cost_entry(k), {w.ms, w.samples});
            ++recorded;
        }
    return recorded;
}

// True when the table prices the whole run: a warm verify wall for every arbiter window up to max_rows and, with a
// chain drafter (chain_max_k > 0), a warm draft wall for at least one depth up to chain_max_k (the nearest one extends
// the others). The caller then decides from the table alone (BlockConfig::fixed_costs, context cost off).
[[nodiscard]] inline bool table_prices_run(const CostTableCache& table, std::span<const std::uint32_t> windows,
                                           std::uint32_t max_rows, std::uint32_t chain_max_k) {
    bool any = false;
    for (const auto rows : windows) {
        if (rows > max_rows) break;
        const auto e = table.warm(verify_cost_entry(rows));
        if (!e || !(e->ms > 0.0)) return false;
        any = true;
    }
    if (!any) return false;
    if (chain_max_k == 0U) return true;
    for (std::uint32_t k = 1; k <= chain_max_k; ++k)
        if (const auto e = table.warm(draft_cost_entry(k)); e && e->ms > 0.0) return true;
    return false;
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
