#pragma once

// Backend-neutral dense site geometry and selection thresholds (host and device constexpr, no SDK types).
//
// A latent-attention model with a grouped output projection, a gated shared FFN, a vocabulary head and a
// tensor-parallel split of two ranks has a fixed set of dense (columns x rows) sites. Their shapes are functions of
// the model dimensions (the generated constants of models/<id>.toml, strix/model/constants.hpp); the engine fills a
// Geometry from those constants and every shape-keyed decision below follows from it, so a different model changes
// the data and not the code. Thresholds that are tuning (row floors, pass sizes) live in DensePolicy, one place.
//
// Every function is pure and constexpr: a call with the geometry of the model that introduced it returns exactly
// the literal the engine used to compile in (tests/dense_site_policy_tests.cpp compares them one by one).

#include <cstdint>

namespace strix::core::dense_sites {

struct Geometry {
    std::uint32_t hidden = 0;        // residual width: input of q_a / kv / shared w1 / w3, output of o_b / shared w2
    std::uint32_t q_lora = 0;        // q_a output = q_b input
    std::uint32_t kv_rows = 0;       // kv projection rows (n_kv_heads * head_dim)
    std::uint32_t heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t o_groups = 0;
    std::uint32_t o_lora = 0;        // rows of one o_a group
    std::uint32_t intermediate = 0;  // shared FFN width
    std::uint32_t vocab = 0;
    std::uint32_t tp_world = 1;      // ranks of the tensor-parallel split (the split halves heads, o_b K, FFN, vocab)
    std::uint32_t engram_columns = 0;  // engram projection input width ((n-gram orders - 1) * heads * head_dim)
    std::uint32_t engram_rows = 0;     // engram projection output rows
};

[[nodiscard]] constexpr bool valid(const Geometry& g) noexcept {
    return g.hidden != 0U && g.q_lora != 0U && g.kv_rows != 0U && g.heads != 0U && g.head_dim != 0U && g.o_groups != 0U &&
           g.o_lora != 0U && g.intermediate != 0U && g.vocab != 0U && g.tp_world != 0U &&
           (g.heads * g.head_dim) % (g.o_groups * g.tp_world) == 0U && (g.o_groups * g.o_lora) % g.tp_world == 0U &&
           g.intermediate % g.tp_world == 0U && g.vocab % g.tp_world == 0U && (g.heads * g.head_dim) % g.tp_world == 0U;
}

// ---- derived dimensions ----
[[nodiscard]] constexpr std::uint32_t q_b_rows_tp(const Geometry& g) noexcept { return g.heads * g.head_dim / g.tp_world; }
[[nodiscard]] constexpr std::uint32_t o_a_group_width(const Geometry& g) noexcept { return g.heads * g.head_dim / g.o_groups; }
[[nodiscard]] constexpr std::uint32_t o_a_row_width(const Geometry& g) noexcept { return g.o_groups * g.o_lora; }
[[nodiscard]] constexpr std::uint32_t o_b_width_tp(const Geometry& g) noexcept { return o_a_row_width(g) / g.tp_world; }
[[nodiscard]] constexpr std::uint32_t ffn_half(const Geometry& g) noexcept { return g.intermediate / g.tp_world; }
[[nodiscard]] constexpr std::uint32_t vocab_half(const Geometry& g) noexcept { return g.vocab / g.tp_world; }

// ---- tuning thresholds (not model data) ----
struct DensePolicy {
    // dense q_b / o_a engage from this many rows (below it the batch kernel keeps them)
    std::uint32_t dense_min_rows = 256U;
    // rows of one pass of the FAST shared-expert dense gate / up
    std::uint32_t shared_fast_rows = 2048U;
    // tall Q8xA8 selector (site benchmark, every form bitwise equal): row floors and output-count limits
    std::uint32_t tall_min_rows = 256U;
    std::uint32_t tall_col_first_rows = 6144U;     // q_a / kv: column-first form from this many rows
    std::uint32_t tall_wide_count_max = 1536U;     // hidden-wide input with at most this many outputs (q_a / kv)
    std::uint32_t tall_narrow_width_max = 1536U;   // narrow input (q_b K) ...
    std::uint32_t tall_narrow_count_min = 4096U;   // ... with at least this many outputs
    std::uint32_t tall_o_a_col_first_rows = 512U;  // o_a groups: column-first from this many rows
    std::uint32_t tall_earlier_rows = 3072U;       // earlier rule: 1 x 4 from this many rows ...
    std::uint32_t tall_earlier_rows_mid = 2048U;   // ... or this many with ...
    std::uint32_t tall_earlier_mid_count = 1024U;  // ... at least this many outputs
    std::uint32_t tall_earlier_wide_count = 4096U; // earlier rule: 1 x 2 from tall_min_rows with this many outputs and
    std::uint32_t tall_earlier_wide_width = 4096U; // at least this input width
    // engram F16 prefill fused form: row window
    std::uint32_t engram_min_rows = 32U;
    std::uint32_t engram_max_rows = 8192U;
};

// ---- tensor-parallel shapes of the trellis dense route ----
// True for the (columns, rows, bits) of a site the tensor-parallel dense path serves: q_b rank rows, o_b K halves,
// q_a / kv / shared-w1 half rows, shared-w2 half columns and (head_rows) the vocabulary head and its half.
[[nodiscard]] constexpr bool tp_shape(const Geometry& g, std::uint32_t columns, std::uint32_t rows, std::uint32_t bits,
                                      bool head_rows) noexcept {
    if (bits != 4U && bits != 5U) return false;
    return (columns == g.q_lora && rows == q_b_rows_tp(g)) || (columns == o_b_width_tp(g) && rows == g.hidden) ||
           (columns == g.hidden && (rows == ffn_half(g) || rows == g.q_lora || rows == g.kv_rows)) ||
           (columns == ffn_half(g) && rows == g.hidden) ||
           (head_rows && columns == g.hidden && (rows == vocab_half(g) || rows == g.vocab));
}

// The q_b rank-rows site of the 5-bit class (excluded from R8 grouped submit).
[[nodiscard]] constexpr bool q_b_rank_rows(const Geometry& g, std::uint32_t columns, std::uint32_t rows) noexcept {
    return columns == g.q_lora && rows == q_b_rows_tp(g);
}

// Grouped-submit counts of a 5-bit site (bit c-1 set = a submission of c rows is admitted); every other shape takes
// every count 1..8. Measured identity table of the trellis library.
[[nodiscard]] constexpr std::uint32_t trellis_submit_sizes(const Geometry& g, std::uint32_t columns, std::uint32_t rows,
                                                           std::uint32_t bits) noexcept {
    constexpr std::uint32_t all = 0xffU;
    if (bits != 5U) return all;
    if (columns == o_a_group_width(g) && rows == g.o_lora) return (1U << 0U) | (1U << 3U) | (1U << 7U);  // 1, 4, 8
    if (columns == g.hidden && rows == g.intermediate) return 0x7eU;                                                          // 2..7
    if (columns == g.intermediate && rows == g.hidden) return 0xfeU;                                       // 2..8
    return all;
}

// q_a / kv fused pair: both read the residual, with these row counts.
[[nodiscard]] constexpr bool qa_kv_pair(const Geometry& g, std::uint32_t qa_columns, std::uint32_t kv_columns,
                                        std::uint32_t qa_rows, std::uint32_t kv_rows) noexcept {
    return qa_columns == g.hidden && kv_columns == g.hidden && qa_rows == g.q_lora && kv_rows == g.kv_rows;
}

// ---- engram projection ----
[[nodiscard]] constexpr bool engram_prefill_shape(const Geometry& g, const DensePolicy& p, std::uint32_t width,
                                                  std::uint32_t count, std::uint32_t first, std::uint32_t rows) noexcept {
    return width == g.engram_columns && count == g.engram_rows && first == 0U && rows >= p.engram_min_rows &&
           rows <= p.engram_max_rows;
}

// ---- tall Q8xA8 form ----
// Form code of the auto selector: 814 / 914 / 822 (Lean 4 forms) from the production attention sites, 0 = none (the
// earlier rule applies). rows >= tall_min_rows only.
[[nodiscard]] constexpr unsigned tall_lean_pick(const Geometry& g, const DensePolicy& p, std::uint32_t rows,
                                                std::uint32_t width, std::uint32_t count, std::uint32_t row_width) noexcept {
    if (rows < p.tall_min_rows) return 0U;
    if (width == g.hidden && count <= p.tall_wide_count_max) return rows >= p.tall_col_first_rows ? 914U : 814U;
    if (width <= p.tall_narrow_width_max && count >= p.tall_narrow_count_min) return 814U;
    if (width == o_a_group_width(g) && count == g.o_lora) return rows >= p.tall_o_a_col_first_rows ? 914U : 822U;
    if (row_width == o_a_row_width(g) && width == o_a_group_width(g)) return 822U;
    return 0U;
}

// The earlier rule (tile code 14 / 12, 0 = wide form) used when no lean form is picked.
[[nodiscard]] constexpr unsigned tall_earlier_pick(const DensePolicy& p, std::uint32_t rows, std::uint32_t width,
                                                   std::uint32_t count) noexcept {
    if (rows >= p.tall_earlier_rows || (rows >= p.tall_earlier_rows_mid && count >= p.tall_earlier_mid_count)) return 14U;
    if (rows >= p.tall_min_rows && count >= p.tall_earlier_wide_count && width >= p.tall_earlier_wide_width) return 12U;
    return 0U;
}

// ---- tensor-parallel packed Q2_K expert rows ----
// Layout of a rank's packed row of the FFN half (two half rows = whole 256-element blocks; the rank-1 split falls
// inside the middle block). `block_bytes` / `block_width` are the Q2_K block constants (84 bytes per 256 elements).
struct Q2kHalfLayout {
    std::uint32_t full_row;    // bytes of the whole row (both halves)
    std::uint32_t part_row;    // bytes of rank 0's part (first half, rounded up to whole blocks)
    std::uint32_t last;        // bytes before the split block
    std::uint32_t packed_row;  // packed per-rank row (whole blocks before the split, then half of the split block)
    std::uint32_t group_width; // elements per scale group
};
[[nodiscard]] constexpr Q2kHalfLayout q2k_half_layout(std::uint32_t half, std::uint32_t block_bytes = 84U,
                                                      std::uint32_t block_width = 256U,
                                                      std::uint32_t split_block_half_bytes = 44U) noexcept {
    const std::uint32_t blocks = 2U * half / block_width;
    return {blocks * block_bytes, (blocks + 1U) / 2U * block_bytes, blocks / 2U * block_bytes,
            blocks / 2U * block_bytes + split_block_half_bytes, 32U};
}

} // namespace strix::core::dense_sites
