#pragma once
// Work model of exact speculation on the qwen35 HIP dense binding (Strix Halo gfx1151, 27B IU4 pack): the fixed costs
// the shared controller prices every decision with (strix/spec/block_drafter.hpp BlockConfig::fixed_costs), the same
// mechanism and arm shape as the CUDA dense app's speculative arm (apps/cuda_qwen35_spec_main.cu), with this device's
// numbers. Decisions are a pure function of (this model, acceptance history): no wall clock enters them.
//   verify  V(r) = the measured wall of r's IU4 row bucket (1/2/4/8/16): the dense core computes the whole bucket
//           (src/core/dense_linear_iu4.hpp), rows <= 8 are one weight stream, 9-16 two 8-row calls.
//   draft   first position (flush: catch-up of the accepted rows + head + host sync) and each further position:
//           MTP layer (Q8_0, 0.45 GB) + the draft head's rows (full IU4 head 0.71 GB x fraction f) at ~190 GB/s.
// Learned walls (the previous form, `fixed = false`) froze depths at their first sample: on the reference device a 3-row window
// costs 127 ms under the old partial-bucket core, so the first k = 2 decision priced every non-bucket depth out
// (measured receipt of the reference device: code k ~ 1, E 1.90). Host-compilable (no HIP): tests/hip_dense_spec_costs_tests.cpp.
// Cost source (HipDenseCosts): "calibrated" (default) = the walls of this device and model, measured once per process at
// decode start (hip_spec_defaults.hpp hip_dense_measure_costs: verify windows of each row bucket after the prompt); the
// draft prices follow from the model's MTP-layer and head bytes at the stream rate V(1) implies (one row streams every
// weight once). "table" = the compiled receipt of the reference device below, a FALLBACK only (calibration off or invalid), reported as
// such on the run's spec config line. The same flags and resolver serve every entry point (HipDenseSpecArgs).
// Exact tensor-parallel decode (--tp-rank) is the one forced use of the table: every rank must choose the same verify
// windows, so it runs the table, labelled "table (forced: ranks must agree)"; an explicit calibrated / learned request
// with --tp-rank is refused (a TP-calibrated table agreed by both ranks is the follow-up).
// Home: qwen35 HIP binding (shared resolver of the family); the controller itself is shared (src/spec).
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "strix/spec/block_drafter.hpp"

namespace strix::qwen35 {

// FALLBACK table (not a policy; forced only under exact TP decode, see above): verify wall per full row bucket, ms (measured receipt of the reference device, commit full, ctx 256+).
inline constexpr std::uint32_t kHipDenseBucketRows[] = {1U, 2U, 4U, 8U, 16U};
inline constexpr double kHipDenseBucketMsScalar[] = {73.141, 73.747, 76.526, 82.637, 126.841};  // STRIX_IU4_SCALAR_X on (default)
inline constexpr double kHipDenseBucketMsVector[] = {76.646, 76.934, 89.104, 118.866, 214.443};  // STRIX_IU4_SCALAR_X=0
// Draft position cost: step = A + B f (f = draft head rows / vocab), flush = step + catch-up. Byte estimates
// (hip_dense_dec3-STATE.md section 4); the run's stats line reports the measured draft wall for recalibration.
inline constexpr double kHipDenseDraftA = 3.3, kHipDenseDraftB = 3.6, kHipDenseDraftCatchUp = 3.75;

[[nodiscard]] constexpr std::uint32_t hip_dense_row_bucket(std::uint32_t rows) {
    return rows <= 1U ? 1U : rows <= 2U ? 2U : rows <= 4U ? 4U : rows <= 8U ? 8U : 16U;
}

[[nodiscard]] constexpr double hip_dense_verify_ms(std::uint32_t rows, bool scalar_x) {
    const std::uint32_t b = hip_dense_row_bucket(rows);
    const std::uint32_t i = b == 1U ? 0U : b == 2U ? 1U : b == 4U ? 2U : b == 8U ? 3U : 4U;
    return scalar_x ? kHipDenseBucketMsScalar[i] : kHipDenseBucketMsVector[i];
}

[[nodiscard]] constexpr double hip_dense_draft_step_ms(double head_fraction) {
    return kHipDenseDraftA + kHipDenseDraftB * head_fraction;
}

// The work model's walls and where they come from.
enum class HipDenseCostSource { calibrated, table };
struct HipDenseCosts {
    HipDenseCostSource source = HipDenseCostSource::table;
    double bucket_ms[5] = {};                       // verify wall per row bucket 1 / 2 / 4 / 8 / 16
    double draft_a = 0, draft_b = 0, catch_up = 0;  // draft step = a + b f (f = head fraction); flush = step + catch_up
};
[[nodiscard]] constexpr std::uint32_t hip_dense_bucket_index(std::uint32_t rows) {
    const std::uint32_t b = hip_dense_row_bucket(rows);
    return b == 1U ? 0U : b == 2U ? 1U : b == 4U ? 2U : b == 8U ? 3U : 4U;
}
[[nodiscard]] constexpr HipDenseCosts hip_dense_table_costs(bool scalar_x) {
    HipDenseCosts c;
    for (std::uint32_t i = 0; i < 5U; ++i) c.bucket_ms[i] = scalar_x ? kHipDenseBucketMsScalar[i] : kHipDenseBucketMsVector[i];
    c.draft_a = kHipDenseDraftA;
    c.draft_b = kHipDenseDraftB;
    c.catch_up = kHipDenseDraftCatchUp;
    return c;
}
// Costs from measured bucket walls (ms, rows 1 / 2 / 4 / 8 / 16) and the loaded model's bytes. A larger window cannot be
// cheaper than a smaller one: a smaller wall measured above is noise, so each bucket takes the running maximum. The
// stream rate is weight_bytes / V(1); a draft position streams the MTP layer plus f of the head at that rate, and the
// flush streams the MTP layer once more (catch-up of the accepted rows). Any non-finite or non-positive wall, or
// weight_bytes 0, makes the measurement invalid: the table is returned (source table).
[[nodiscard]] inline HipDenseCosts hip_dense_calibrated_costs(const double (&walls)[5], std::uint64_t weight_bytes,
                                                              std::uint64_t mtp_bytes, std::uint64_t head_bytes, bool scalar_x) {
    HipDenseCosts c = hip_dense_table_costs(scalar_x);
    for (const double w : walls)
        if (!(std::isfinite(w) && w > 0.0)) return c;
    if (weight_bytes == 0U) return c;
    c.source = HipDenseCostSource::calibrated;
    for (std::uint32_t i = 0; i < 5U; ++i) c.bucket_ms[i] = i == 0U ? walls[0] : std::fmax(walls[i], c.bucket_ms[i - 1U]);
    const double bytes_per_ms = double(weight_bytes) / c.bucket_ms[0];
    if (mtp_bytes != 0U) {
        c.draft_a = double(mtp_bytes) / bytes_per_ms;
        c.draft_b = double(head_bytes) / bytes_per_ms;
        c.catch_up = c.draft_a;
    }
    return c;
}
[[nodiscard]] constexpr double hip_dense_verify_ms(const HipDenseCosts& c, std::uint32_t rows) {
    return c.bucket_ms[hip_dense_bucket_index(rows)];
}

// ONE set of speculation flags for every entry point of the family (strix-hip-qwen35-decode, strix-hip-qwen35-spec):
//   --mtp auto|off|none|FILE, --draft-rows on|off, --draft-pack F1,F2, --spec-costs calibrated|table|learned,
//   --cost-kv auto|online|off|KV (context cost of the calibrated / table model, hip_dense_cost_kv)
//   (fixed = table, the previous name). learned = the previous learned form (cap 7), kept as the rollback.
enum class HipDenseCostMode { calibrated, table, learned };
struct HipDenseSpecArgs {
    std::string mtp = "auto";
    bool draft_rows = true;
    std::string draft_packs;
    HipDenseCostMode costs = HipDenseCostMode::calibrated;
    std::string cost_kv = "auto";  // --cost-kv auto|online|off|KV: context cost of the table / calibrated model (hip_dense_cost_kv)
    bool costs_explicit = false;  // --spec-costs was given (exact TP decode refuses an explicit non-table choice)
};
[[nodiscard]] inline const char* hip_dense_cost_mode_name(HipDenseCostMode m) {
    return m == HipDenseCostMode::calibrated ? "calibrated" : m == HipDenseCostMode::table ? "table" : "learned";
}
// Consumes `flag` (and its value through next()) when it is one of the shared flags; false otherwise.
template <class Next>
bool parse_hip_dense_spec_arg(const std::string& flag, Next&& next, HipDenseSpecArgs& a) {
    if (flag == "--mtp") {
        a.mtp = next();
    } else if (flag == "--draft-rows") {
        const std::string v = next();
        if (v != "on" && v != "off") throw std::invalid_argument("--draft-rows on|off");
        a.draft_rows = v == "on";
    } else if (flag == "--draft-pack") {
        a.draft_packs = next();
    } else if (flag == "--spec-costs") {
        const std::string v = next();
        a.costs_explicit = true;
        if (v == "calibrated") a.costs = HipDenseCostMode::calibrated;
        else if (v == "table" || v == "fixed") a.costs = HipDenseCostMode::table;
        else if (v == "learned") a.costs = HipDenseCostMode::learned;
        else throw std::invalid_argument("--spec-costs calibrated|table|learned");
    } else if (flag == "--cost-kv") {
        a.cost_kv = next();
        spec::ArbiterConfig probe;
        if (a.cost_kv != "auto") spec::set_context_cost(probe, a.cost_kv);  // validates now: auto|online|off|KV
    } else {
        return false;
    }
    return true;
}

// The binding's block config. fixed = true: the work model above (cap 15 = the engine's 16-row window, like sub15d;
// the model, not the cap, decides how deep). fixed = false: the previous learned form (cap 7, seeds below).
// max_rows = the engine's window rows (16); chain = an MTP layer is loaded.
[[nodiscard]] inline spec::BlockConfig hip_dense_spec_block_config(std::uint32_t max_rows, bool chain, bool fixed,
                                                                  double head_fraction, const HipDenseCosts& costs) {
    spec::BlockConfig bc;
    bc.max_block = max_rows;
    bc.chain = bc.hybrid = chain;
    if (fixed) {
        bc.fixed_costs = true;
        bc.chain_max_k = max_rows > 1U ? max_rows - 1U : 1U;
        bc.chain_seed_step_ms = costs.draft_a + costs.draft_b * head_fraction;
        bc.chain_seed_flush_ms = bc.chain_seed_step_ms + costs.catch_up;
    } else {
        bc.chain_max_k = 7U;
        bc.chain_seed_step_ms = head_fraction < 1.0 ? 4.0 : 9.0;
    }
    return bc;
}
[[nodiscard]] inline spec::BlockConfig hip_dense_spec_block_config(std::uint32_t max_rows, bool chain, bool fixed,
                                                                  double head_fraction) {
    return hip_dense_spec_block_config(max_rows, chain, fixed, head_fraction, hip_dense_table_costs(true));
}

// Learned form's verify seeds (its previous table: shape only, walls replace it).
inline constexpr std::uint32_t kHipDenseLearnedSeedRows[] = {1U, 2U, 4U, 8U, 12U, 16U};
inline constexpr double kHipDenseLearnedSeedMs[] = {79.0, 81.3, 92.0, 123.7, 200.4, 220.7};

// Seeds the drafter's verify costs: every window 1..max_rows from the bucket model (fixed), or the learned form's shape.
template <class Drafter>
void hip_dense_seed_verify(Drafter& drafter, std::uint32_t max_rows, bool fixed, const HipDenseCosts& costs) {
    if (fixed) {
        for (std::uint32_t r = 1; r <= max_rows; ++r) drafter.seed_verify_ms(r, hip_dense_verify_ms(costs, r));
        return;
    }
    for (std::size_t i = 0; i < sizeof(kHipDenseLearnedSeedRows) / sizeof(kHipDenseLearnedSeedRows[0]); ++i)
        if (kHipDenseLearnedSeedRows[i] <= max_rows) drafter.seed_verify_ms(kHipDenseLearnedSeedRows[i], kHipDenseLearnedSeedMs[i]);
}
template <class Drafter>
void hip_dense_seed_verify(Drafter& drafter, std::uint32_t max_rows, bool fixed, bool scalar_x) {
    hip_dense_seed_verify(drafter, max_rows, fixed, hip_dense_table_costs(scalar_x));
}

}  // namespace strix::qwen35
