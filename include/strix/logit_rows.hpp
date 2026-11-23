#pragma once
// Row pitch of multi-row logits / score buffers (cohort decode lanes, verify windows, batched heads), backend-neutral.
//
// The dense row projections that write a head row admit only outputs aligned to kLogitRowAlignmentBytes (CUDA
// bf16 rows, HIP BF16 rows). A vocabulary that is not a multiple of that alignment in elements (e.g. 200058 FP32
// logits = 800232 bytes) puts every odd packed row off the boundary, so the head refuses row 1. Rows therefore start
// every logit_row_pitch(vocabulary) elements: row r holds the same `vocabulary` values, only its address changes.
// A pitch equal to the vocabulary (aligned vocabularies) leaves the packed layout unchanged.
#include <cstddef>

namespace strix::head {

inline constexpr std::size_t kLogitRowAlignmentBytes = 16U;

// Elements between the starts of two rows: the vocabulary rounded up so every row starts on `alignment_bytes`.
// Returns 0 for an invalid request (zero sizes, alignment not a power of two or not a multiple of the element).
constexpr std::size_t logit_row_pitch(std::size_t vocabulary, std::size_t element_bytes = sizeof(float),
                                      std::size_t alignment_bytes = kLogitRowAlignmentBytes) {
    if (vocabulary == 0U || element_bytes == 0U || alignment_bytes == 0U ||
        (alignment_bytes & (alignment_bytes - 1U)) != 0U || alignment_bytes % element_bytes != 0U)
        return 0U;
    const std::size_t per_alignment = alignment_bytes / element_bytes;
    return (vocabulary + per_alignment - 1U) / per_alignment * per_alignment;
}

// Elements a buffer of `rows` rows needs (the last row holds only `vocabulary` values).
constexpr std::size_t logit_rows_elements(std::size_t rows, std::size_t vocabulary, std::size_t pitch) {
    return rows == 0U ? 0U : (rows - 1U) * pitch + vocabulary;
}

constexpr std::size_t logit_row_offset(std::size_t row, std::size_t pitch) { return row * pitch; }

// Rows of one head issued as row projections (verify windows, cohort lanes). A row projection stores its row r at
// r x outputs, so a head that projects `pitch` outputs per row (live <= pitch <= stored head rows) puts row r at
// logit_row_offset(r, pitch): the live outputs are those of a one-row launch (each output is computed alone) and
// the pitch - live pad outputs are further stored head rows that no consumer reads. A row family keeps its one-row
// arithmetic only up to its own row cap, so the rows run as consecutive launches of at most `max_rows`; launch
// `first` writes from logit_row_offset(first, pitch).
struct LogitRowsLaunch {
    std::size_t first;
    std::size_t rows;
};

// The launch that starts at row `first` (< rows) of `rows` rows; {first, 0} for an invalid request.
constexpr LogitRowsLaunch logit_rows_launch(std::size_t rows, std::size_t first, std::size_t max_rows) {
    if (max_rows == 0U || first >= rows) return {first, 0U};
    return {first, rows - first < max_rows ? rows - first : max_rows};
}

// A pitched rows head can project `pitch` outputs per row of a head with `stored` rows, keep `live` outputs per row,
// and fit a buffer of `elements` values (the last row stores its pad as well: rows x pitch).
constexpr bool pitched_rows_fit(std::size_t rows, std::size_t live, std::size_t pitch, std::size_t stored,
                                std::size_t elements) {
    return rows != 0U && live != 0U && live <= pitch && pitch <= stored && rows <= elements / pitch;
}

}  // namespace strix::head
