#pragma once
// Bit-safe cuts of a backend prompt chunk: the shared rule of bounded prompt slices (serve::CohortConfig::
// prompt_slice_decoding). Header-only and dependency-free so a decoder can state its rule.
#include <cstdint>

namespace strix::serve {

// Where a backend prompt chunk may be cut without changing bits (CohortBackend::prompt_piece_rule): at offsets that
// are multiples of `align` rows counted from the chunk start, with every piece longer than `short_rows` rows (a chunk
// that short keeps its own form and is never cut). align 0 = the chunk cannot be cut.
struct PromptPieceRule {
    std::uint32_t align = 0;
    std::uint32_t short_rows = 0;
    [[nodiscard]] constexpr bool cuttable() const noexcept { return align != 0U; }
};

// Rows of the next piece of a backend chunk with `chunk_left` rows unread (its read offset is a cut point), about
// `bound` rows. A piece never leaves `short_rows` rows or fewer behind: such a remainder rides in the piece when the
// piece still fits `max_call` (the most rows one prefill call runs as ONE chunk), else the piece shrinks by whole
// `align` steps. chunk_left when no cut keeps the forms.
[[nodiscard]] constexpr std::uint64_t prompt_piece_rows(std::uint64_t chunk_left, std::uint64_t bound,
                                                        std::uint64_t max_call, PromptPieceRule rule) {
    if (!rule.cuttable() || chunk_left <= bound) return chunk_left;
    const std::uint64_t step = bound / rule.align * rule.align;
    if (step <= rule.short_rows) return chunk_left;
    const std::uint64_t rest = chunk_left - step;
    if (rest > rule.short_rows) return step;
    if (chunk_left <= max_call) return chunk_left;
    const std::uint64_t back = (rule.short_rows + 1U - rest + rule.align - 1U) / rule.align * rule.align;
    return step > back + rule.short_rows ? step - back : chunk_left;
}

// A piece under a PRICED grant (CohortConfig::prompt_gap_ms): rows of the next piece of a backend chunk with
// `chunk_left` rows unread (its read offset is a cut point). A bit-safe piece is the whole rest, or a multiple of
// `align` rows longer than `short_rows` that leaves more than `short_rows` rows behind. The piece is the LARGEST
// bit-safe piece <= grant; only when none exists it is the SMALLEST bit-safe piece and `over` is set (the caller
// records the overshoot). Unlike prompt_piece_rows no remainder rides above the grant.
struct PromptPieceChoice {
    std::uint64_t rows = 0;
    bool over = false;
};
[[nodiscard]] constexpr PromptPieceChoice prompt_piece_within(std::uint64_t chunk_left, std::uint64_t grant,
                                                              PromptPieceRule rule) {
    if (chunk_left <= grant) return {chunk_left, false};
    if (!rule.cuttable() || chunk_left <= 2U * std::uint64_t{rule.short_rows} + 1U) return {chunk_left, true};
    const std::uint64_t top = chunk_left - rule.short_rows - 1U;  // the largest piece leaving > short_rows behind
    const std::uint64_t fit = (grant < top ? grant : top) / rule.align * rule.align;
    if (fit > rule.short_rows) return {fit, false};
    const std::uint64_t least = (std::uint64_t{rule.short_rows} / rule.align + 1U) * rule.align;
    return {least <= top ? least : chunk_left, true};
}

}  // namespace strix::serve
