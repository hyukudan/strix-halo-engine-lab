#pragma once
// Work model of exact speculation on the qwen35 HIP dense binding (Strix Halo gfx1151, 27B IU4 pack): the fixed costs
// the shared controller prices every decision with (strix/spec/block_drafter.hpp BlockConfig::fixed_costs), the same
// mechanism and arm shape as the CUDA dense app's sub15d (apps/cuda_qwen35_spec_main.cu, cc_decode3), with this node's
// numbers. Decisions are a pure function of (this model, acceptance history): no wall clock enters them.
//   verify  V(r) = the measured wall of r's IU4 row bucket (1/2/4/8/16): the dense core computes the whole bucket
//           (src/core/dense_linear_iu4.hpp), rows <= 8 are one weight stream, 9-16 two 8-row calls.
//   draft   first position (flush: catch-up of the accepted rows + head + host sync) and each further position:
//           MTP layer (Q8_0, 0.45 GB) + the draft head's rows (full IU4 head 0.71 GB x fraction f) at ~190 GB/s.
// Learned walls (the previous form, `fixed = false`) froze depths at their first sample: on this node a 3-row window
// costs 127 ms under the old partial-bucket core, so the first k = 2 decision priced every non-bucket depth out
// (receipt HDD2-T1: code k ~ 1, E 1.90). Host-compilable (no HIP): tests/hip_dense_spec_costs_tests.cpp.
// Home: qwen35 HIP binding (shared resolver of the family); the controller itself is shared (src/spec).
#include <cstdint>

#include "strix/spec/block_drafter.hpp"

namespace strix::qwen35 {

// Verify wall per full row bucket, ms (HDD2-T1 cost table, cluster1, commit full, ctx 256+).
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

// The binding's block config. fixed = true: the work model above (cap 15 = the engine's 16-row window, like sub15d;
// the model, not the cap, decides how deep). fixed = false: the previous learned form (cap 7, seeds below).
// max_rows = the engine's window rows (16); chain = an MTP layer is loaded.
[[nodiscard]] inline spec::BlockConfig hip_dense_spec_block_config(std::uint32_t max_rows, bool chain, bool fixed,
                                                                  double head_fraction) {
    spec::BlockConfig bc;
    bc.max_block = max_rows;
    bc.chain = bc.hybrid = chain;
    if (fixed) {
        bc.fixed_costs = true;
        bc.chain_max_k = max_rows > 1U ? max_rows - 1U : 1U;
        bc.chain_seed_step_ms = hip_dense_draft_step_ms(head_fraction);
        bc.chain_seed_flush_ms = bc.chain_seed_step_ms + kHipDenseDraftCatchUp;
    } else {
        bc.chain_max_k = 7U;
        bc.chain_seed_step_ms = head_fraction < 1.0 ? 4.0 : 9.0;
    }
    return bc;
}

// Learned form's verify seeds (its previous table: shape only, walls replace it).
inline constexpr std::uint32_t kHipDenseLearnedSeedRows[] = {1U, 2U, 4U, 8U, 12U, 16U};
inline constexpr double kHipDenseLearnedSeedMs[] = {79.0, 81.3, 92.0, 123.7, 200.4, 220.7};

// Seeds the drafter's verify costs: every window 1..max_rows from the bucket model (fixed), or the learned form's shape.
template <class Drafter>
void hip_dense_seed_verify(Drafter& drafter, std::uint32_t max_rows, bool fixed, bool scalar_x) {
    if (fixed) {
        for (std::uint32_t r = 1; r <= max_rows; ++r) drafter.seed_verify_ms(r, hip_dense_verify_ms(r, scalar_x));
        return;
    }
    for (std::size_t i = 0; i < sizeof(kHipDenseLearnedSeedRows) / sizeof(kHipDenseLearnedSeedRows[0]); ++i)
        if (kHipDenseLearnedSeedRows[i] <= max_rows) drafter.seed_verify_ms(kHipDenseLearnedSeedRows[i], kHipDenseLearnedSeedMs[i]);
}

}  // namespace strix::qwen35
