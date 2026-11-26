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

// Mixed service: the policy data a server binds into CohortConfig::prompt_slice_decoding. A prompt admitted while
// another lane decodes runs in pieces of `piece_rows` rows, cut where the decoder's PromptPieceRule keeps the bits, so
// a decoding lane waits about one piece and its decode round. Fixed rows, not a priced grant: the bits never depend
// on the size, and a fixed size holds no learned state (CohortConfig::prompt_gap_ms learns a row cost per pass).
struct MixedServicePieces {
    std::uint32_t piece_rows = 0;  // 0 = whole chunks beside decoding lanes
    // Decode owed per ms of prompt while lanes decode (CohortConfig::prompt_share): 1 = after a piece the decoding lanes
    // run decode-only steps for as long as the piece took; 0 = one decode round per piece (the gap is one piece and
    // one round either way, the prompt's wall is not doubled).
    double prompt_share = 1.0;
};
// The measured sizes, by regime:
// - short: 128 rows, the incumbent mixed-service bound (Inkling 2 Mini served routes on gfx1151). RTX PRO 6000,
//   Inkling 2 Mini q35: a 145-157 ms decode gap, but an 8K prompt beside a decoding lane took 9.7 s (a ~124 ms fixed
//   cost per prefill pass, one pass per 128 rows).
// - long: 1024 rows, for a device whose prefill pass carries a large fixed cost. RTX PRO 6000, Inkling 2 Mini held-back
//   served route, ticket q41: decode gap 190 ms while an 8K prompt is admitted (priced grants, q39: 849-853 ms), the
//   8K prompt 2.5-2.7 s beside the lane with prompt share 1 (the prefill call is synchronous: each ~165 ms piece
//   owed ~165 ms of decode-only steps, 100-113 decode events during the admission), C8 aggregate 192-196 tok/s over
//   3 reps. Share 0: one decode round per piece.
inline constexpr MixedServicePieces kMixedServiceShortPieces{128U, 1.0};
inline constexpr MixedServicePieces kMixedServiceLongPieces{1024U, 0.0};

// Rows of a piece beside decoding lanes for prompt calls of `chunk` rows: the policy's rows where the rule can cut and
// the chunk is larger, else 0 (whole chunks).
[[nodiscard]] constexpr std::uint32_t mixed_service_piece_rows(MixedServicePieces policy, PromptPieceRule rule,
                                                               std::uint64_t chunk) noexcept {
    return rule.cuttable() && policy.piece_rows != 0U && chunk > policy.piece_rows ? policy.piece_rows : 0U;
}

}  // namespace strix::serve
