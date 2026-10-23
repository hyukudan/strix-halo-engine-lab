#pragma once

// deepseek41 prefill: the routed experts of T prompt rows at once, and the exact suffix replay plan.
//
// Routes. Every (row, k) assignment is one slot = row * top_k + k, as in the decode window. The
// prefill CSR is dense over expert ids: expert e owns entries begins[e] .. begins[e + 1] in
// ascending slot (= ascending row, an expert appears at most once per row); entry i reads activation
// row rows[i] and writes slot slots[i]. The entries of each expert are cut into batches of at most
// batch_rows consecutive entries, in ascending expert id; batch b is (batch_expert[b],
// batch_begin[b], batch_count[b]). order / low_counts are the combine's (ds41_moe.hpp): per row its
// top_k slots in ascending expert id and how many of them have id < split.
//
// Arithmetic. The prefill expert kernels produce, for every (entry, output), the same bits as the
// decode products of ds41_formats.hpp (same integer group dots, same per-group FP32 expressions,
// same lane classes and the same butterfly tree), so a prefill row equals the row computed alone by
// the decode window, whatever T, the batch or the other rows. No new numerical identity.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strix::ds41 {

struct PrefillPass {
    std::uint32_t begin = 0, rows = 0;
};
// Consecutive global positions, bounded row buffers and aligned starts. Rebalance a short final pass so that a
// decoder suffix can fit entirely in it. The final row count need not be aligned; all preceding counts are.
[[nodiscard]] std::vector<PrefillPass> plan_prefill_passes(std::uint32_t total, std::uint32_t pass_rows,
                                                        std::uint32_t alignment, std::uint32_t min_final_rows = 0);

struct PrefillRoutes {
    std::uint32_t rows = 0, top_k = 0, experts = 0, batch_rows = 0;
    std::vector<std::uint32_t> begins;                                 // [experts + 1]
    std::vector<std::uint32_t> entry_rows, entry_slots;                // [rows * top_k]
    std::vector<std::uint32_t> batch_expert, batch_begin, batch_count; // [batches]
    std::vector<std::uint32_t> order, low_counts;                      // [rows * top_k], [rows]
};
// Upper bound of the batch count for any routing of rows * top_k slots over `experts` experts.
[[nodiscard]] constexpr std::uint32_t prefill_max_batches(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                                                          std::uint32_t batch_rows) noexcept {
    return (rows * top_k + batch_rows - 1U) / batch_rows + experts;
}
// ids[row * top_k + k] < experts, distinct within a row. batch_rows is 16 or 32. With an expert range
// [first, end) (end 0: all) only those experts own entries; begins and batch_expert are relative to
// first; order / low_counts still cover every slot.
[[nodiscard]] PrefillRoutes build_prefill_routes(std::span<const std::uint32_t> ids, std::uint32_t rows, std::uint32_t top_k,
                                                 std::uint32_t experts, std::uint32_t split, std::uint32_t batch_rows,
                                                 std::uint32_t first = 0, std::uint32_t end = 0);

// ---- Exact suffix replay (CED prefill) ----
//
// Each layer keeps its own sliding window of `window` K/V rows (the decode ring), computed from that
// layer's own input. A layer's output for row s reads the window rows s - window + 1 .. s of the same
// layer. The compressed KV and index keys every decoder layer reads come from the first decoder layer
// (the KV source), whose input is the encoder output, so they exist for every row once the encoder
// has run. Hence, after the encoder layers ran on every new row, the decoder layers need only a cone
// of trailing rows to leave the decode state (every layer's window ring, the KV-source caches) and
// the last row's logits exactly as a full prefill would.
struct SuffixReplayGeometry {
    std::uint32_t layers = 40;          // decoder layers are first_decoder .. layers - 1
    std::uint32_t first_decoder = 20;   // the KV source of every decoder layer
    std::uint32_t window = 128;         // per-layer sliding window (rows attended, current row included)
    // Rows whose inputs of the last draft_layers layers the drafter needs (window seed), 0 for none.
    std::uint32_t draft_rows = 128;
    std::uint32_t draft_layers = 3;     // inputs of layers layers - draft_layers .. layers - 1
};
struct SuffixReplayLayer {
    std::uint32_t layer = 0;
    // Rows [out_begin, total) run the whole layer (attention output, MoE, ...); rows
    // [kv_begin, total) need this layer's window K/V (projection of the layer input). Encoder layers
    // run every new row: out_begin = kv_begin = cached.
    std::uint64_t out_begin = 0, kv_begin = 0;
};
struct SuffixReplayPlan {
    std::uint64_t cached = 0, total = 0;
    std::vector<SuffixReplayLayer> layers;
    // Rows of the first decoder layer's input (= encoder output) that must be available, including
    // rows before `cached` that only a ring entry can provide (see ring_rows).
    std::uint64_t encoder_output_begin = 0;
    // Window-ring rows of the cached prefix each layer reads (cached - window + 1 .. cached - 1 at
    // most): they come from the ring saved with the prefix, never recomputed.
    std::uint64_t ring_rows = 0;
    std::uint64_t full_row_layers = 0;    // rows x layers of a full prefill of the new rows
    std::uint64_t replay_row_layers = 0;  // rows x layers this plan runs (out rows)
};
// cached = rows already in the state (0 for a fresh prompt), total = rows after this prefill.
[[nodiscard]] SuffixReplayPlan plan_suffix_replay(const SuffixReplayGeometry& geometry, std::uint64_t cached, std::uint64_t total);
// Brute-force check of a plan: the backward dependency closure of the decode state, logits and
// drafter seed over the per-layer windows, row by row. Returns an empty string when the plan
// computes exactly the needed rows (nothing missing, nothing extra), else a description.
[[nodiscard]] std::string verify_suffix_replay(const SuffixReplayGeometry& geometry, const SuffixReplayPlan& plan);

} // namespace strix::ds41
