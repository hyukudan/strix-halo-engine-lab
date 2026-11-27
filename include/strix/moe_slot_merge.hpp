#pragma once
// Expert-slot merge inside the producing down launch (backend-neutral contract).
//
// A routed-expert down step produces one partial per (slot, output row): rawdots[slot][row]. The incumbent
// merged them in a separate one-CTA kernel: per output, current = +0, then for slot 0..top_k-1 in order
// current = fma(route_weight[slot], rawdots[slot][row], current); then a finite test over every output, and
// the commit only when the whole step is clean. Here the producing launch merges instead: every block of a
// row tile stores its partials, fences and takes the tile's arrival ticket; the block that takes the LAST
// ticket of a tile merges that tile's rows in slot order (the order comes from the slot index, never from
// the arrival order, so the bits equal the incumbent's for any schedule). The tile that completes last then
// decides the step (status | nonfinite), commits every output and publishes the record. Ticket words live in
// the caller's cleared workspace and are returned to zero by the launch itself.
#include <cstdint>

#include "strix/numeric/bf16_bits.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_SLOT_MERGE_HD __host__ __device__
#else
#define STRIX_SLOT_MERGE_HD
#endif

namespace strix::moe_slot_merge {

// Output rows owned by one down block (one row tile of the ticket).
inline constexpr std::uint32_t tile_rows = 64U;
// Most slots one merge stages in registers (the runtime admits top_k <= 10).
inline constexpr std::uint32_t max_slots = 16U;
// Status bit the incumbent tail ORs in when any output fails the commit threshold.
inline constexpr std::uint32_t nonfinite_status = 8U;

[[nodiscard]] STRIX_SLOT_MERGE_HD constexpr std::uint32_t tiles(std::uint32_t hidden) noexcept {
    return hidden / tile_rows;
}
// Layout of the ticket words: [tile arrivals x tiles][tiles done][nonfinite flag].
[[nodiscard]] STRIX_SLOT_MERGE_HD constexpr std::uint32_t tile_done_word(std::uint32_t hidden) noexcept {
    return tiles(hidden);
}
[[nodiscard]] STRIX_SLOT_MERGE_HD constexpr std::uint32_t nonfinite_word(std::uint32_t hidden) noexcept {
    return tiles(hidden) + 1U;
}
[[nodiscard]] STRIX_SLOT_MERGE_HD constexpr std::uint32_t ticket_words(std::uint32_t hidden) noexcept {
    return tiles(hidden) + 2U;
}
// An atomic add returned `previous`; the arrival that completes `arrivals` is the last one.
[[nodiscard]] STRIX_SLOT_MERGE_HD constexpr bool last_arrival(std::uint32_t previous, std::uint32_t arrivals) noexcept {
    return previous + 1U == arrivals;
}
// Commit threshold of the incumbent tail: |x| at or above it (inf / nan included) fails; a bf16 commit also
// fails finite values that round to inf.
[[nodiscard]] STRIX_SLOT_MERGE_HD constexpr bool fails_commit(std::uint32_t bits, bool bf16_commit) noexcept {
    return (bits & 0x7fffffffU) >= (bf16_commit ? 0x7f7f8000U : 0x7f800000U);
}
// bf16 round-to-nearest-even of the f32 bits, without NaN canonicalisation (the incumbent activation store).
[[nodiscard]] STRIX_SLOT_MERGE_HD constexpr std::uint16_t bf16_rne(std::uint32_t bits) noexcept {
    return ::strix::numeric::bf16_bits::rne(bits);
}

} // namespace strix::moe_slot_merge
