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

}  // namespace strix::head
