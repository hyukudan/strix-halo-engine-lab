#pragma once
// Decode / verify weight-read geometry (backend-neutral; the decode step plan's companion to
// strix/decode_projection_group.hpp). A window of `bucket` rows (1..16) multiplies every weight word by each row.
// The kernel reads each word once per SWEEP: rows that share a sweep share its bytes, its scale loads and its
// weight decode. The bound on rows per sweep is a register budget of the backend kernel, given as traits:
//   - accumulator_rows: rows whose accumulators one wave holds through a whole sweep without spilling;
//   - operand_rows: rows whose activation operands it holds at once (more rows take another operand load and
//     wait per k-tile inside the same sweep, no extra weight bytes).
// Rows are independent (same operands, same per-row op order whatever the grouping), so the geometry never
// changes bits; it only decides how many times the weights are streamed. Backends: HIP dense IU4
// (core/dense_linear_iu4.hpp dispatch_bucket, core/w4a16.hpp wave_rows); a CUDA tall GEMV binds the same functions
// with its own traits.
namespace strix::decode {

struct WeightSweepTraits {
    unsigned accumulator_rows;
    unsigned operand_rows;
};

constexpr unsigned sweep_rows(unsigned bucket, WeightSweepTraits t) {
    return bucket < t.accumulator_rows ? bucket : t.accumulator_rows;
}
constexpr unsigned weight_sweeps(unsigned bucket, WeightSweepTraits t) {
    const unsigned per = sweep_rows(bucket, t);
    return per == 0U ? 0U : (bucket + per - 1U) / per;
}
// Operand loads (and their waits) per k-tile inside one sweep.
constexpr unsigned operand_chunks(unsigned bucket, WeightSweepTraits t) {
    const unsigned rows = sweep_rows(bucket, t);
    return t.operand_rows == 0U ? 0U : (rows + t.operand_rows - 1U) / t.operand_rows;
}

// HIP W4A16 IU4 small-M core (gfx1151) register budgets:
//   scalar activations (UniformX, SGPR operands, 8 words per row): 8 operand rows = 64 SGPRs of the 106 limit;
//   16 accumulator rows (acc + group dot, 4 VGPRs per row) in that form: one stream of the weights instead of two
//   for a 16-row window; its VGPR / scratch readout on the node gates it.
//   vector activations (pair_activations in VGPRs): one 16-row sweep spilled (private segment 144 B), so 8.
inline constexpr WeightSweepTraits kHipIu4ScalarX{16U, 8U};
inline constexpr WeightSweepTraits kHipIu4VectorX{8U, 8U};

}  // namespace strix::decode
