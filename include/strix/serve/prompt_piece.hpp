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

// Rows of a gap-bounded piece (CohortConfig::prompt_gap_ms): the most rows r with
// fixed_ms + row_ms * r + decode_ms <= gap_ms, where fixed_ms is the backend's cost of any prefill pass (e.g. all
// expert weights re-streamed), row_ms the observed cost of one row of the previous pass and decode_ms the last decode
// round, i.e. the gap a decoding lane sees across one piece and one decode round. Clamped to [floor, cap]: floor is
// the qualified fixed piece (prompt_slice_decoding), cap the unbounded call. Bits never depend on it: the cut still
// goes through prompt_piece_rows.
[[nodiscard]] constexpr std::uint64_t gap_piece_rows(double gap_ms, double fixed_ms, double row_ms, double decode_ms,
                                                     std::uint64_t floor, std::uint64_t cap) noexcept {
    if (!(row_ms > 0.0) || !(gap_ms > 0.0) || cap <= floor) return floor < cap ? floor : cap;
    const double room = (gap_ms - fixed_ms - decode_ms) / row_ms;
    if (!(room > static_cast<double>(floor))) return floor;
    if (room >= static_cast<double>(cap)) return cap;
    return static_cast<std::uint64_t>(room);
}

}  // namespace strix::serve
