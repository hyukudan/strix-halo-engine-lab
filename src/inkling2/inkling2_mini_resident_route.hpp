#pragma once

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// Decode experts on the packed one-row form (moe_core::microscaled_r1). The CLI and HIP serving resolve to this
// value, so a served lane and the CLI share one EXACT identity; --grouped-decode keeps the grouped tile kernel.
inline constexpr bool kMiniPackedDecodeDefault = true;

// Cohort decode head (HIP, BF16 head): the score rows of a cohort step are projected by the shared tight-row head
// (core::fma_rows::enqueue_bf16_rows), which reads the head once for every row of a launch. Up to
// core::gemv_rows::max_rows rows a launch keeps each row on its one-row body (same chunk order and wave sum, so the
// same bits); more rows would select the WMMA tile, so a larger cohort runs as consecutive launches of at most
// kMiniHeadRows rows. Rows are contiguous: chunk `first` writes from element `first * vocabulary`.
inline constexpr unsigned kMiniHeadRows = 16U;

struct MiniHeadChunk {
    unsigned first;
    unsigned rows;
};

// Chunk starting at row `first` (< rows) of a cohort head of `rows` rows.
constexpr MiniHeadChunk mini_head_chunk(unsigned rows, unsigned first) {
    return {first, rows - first < kMiniHeadRows ? rows - first : kMiniHeadRows};
}

enum class MiniAppendKind { prefill, decode, verify };

struct MiniResidentRoute {
    unsigned attention_rows;
    bool packed_moe;
    bool step_graph;
};

// With packed decode, every decode and verify row (a lone token, a verify window or a cohort of lanes of any size)
// runs the packed form, whose row bits equal a one-row launch: a lane's tokens do not depend on the cohort size.
// Prefill chunks of more than one row keep the grouped tile kernel.
constexpr MiniResidentRoute mini_resident_route(MiniAppendKind kind, unsigned rows,
                                                bool packed_decode, bool allow_graph) {
    const bool verify = kind == MiniAppendKind::verify;
    return {
        verify ? 1U : rows,
        packed_decode && (rows == 1U || kind != MiniAppendKind::prefill),
        allow_graph && rows == 1U && kind != MiniAppendKind::prefill,
    };
}

}  // namespace strix::inkling2
