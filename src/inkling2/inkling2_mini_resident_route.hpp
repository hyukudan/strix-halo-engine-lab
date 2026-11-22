#pragma once

#include "inkling2_mini_config.hpp"

#include <cstddef>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

// Device score rows of several lanes (cohort decode) or verify rows. The dense head writes each row
// through a projection that admits only 16-byte aligned outputs; the live vocabulary (200058 floats)
// is not a multiple of four, so contiguous rows put every odd row 8 bytes off and the head refuses it
// (invalid argument). Rows therefore start every kMiniLogitRowPitch floats; row r holds the same
// kOutputVocabulary scores as before, only its address changes.
inline constexpr std::size_t kMiniLogitRowAlignmentBytes = 16U;
inline constexpr std::size_t kMiniLogitRowPitch =
    (std::size_t{kOutputVocabulary} * sizeof(float) + kMiniLogitRowAlignmentBytes - 1U) /
    kMiniLogitRowAlignmentBytes * kMiniLogitRowAlignmentBytes / sizeof(float);

constexpr std::size_t mini_logit_rows_elements(unsigned rows) {
    return rows == 0U ? 0U : std::size_t{rows - 1U} * kMiniLogitRowPitch + kOutputVocabulary;
}

constexpr std::size_t mini_logit_row_offset(unsigned row) { return std::size_t{row} * kMiniLogitRowPitch; }

enum class MiniAppendKind { prefill, decode, verify };

struct MiniResidentRoute {
    unsigned attention_rows;
    bool packed_moe;
    bool step_graph;
};

constexpr MiniResidentRoute mini_resident_route(MiniAppendKind kind, unsigned rows,
                                                bool packed_decode, bool allow_graph) {
    const bool verify = kind == MiniAppendKind::verify;
    return {
        verify ? 1U : rows,
        packed_decode && (rows == 1U || verify),
        allow_graph && rows == 1U && kind != MiniAppendKind::prefill,
    };
}

}  // namespace strix::inkling2
