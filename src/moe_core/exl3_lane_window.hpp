#pragma once
// Lane window of the shared EXL3 mul1 tile reader (core/trellis_codec.hpp LaneGeom / TileStates), as host constexpr so
// the reader traits and the host tests share one formula. Lane t of a 16 x 16 tile owns the eight states 8t .. 8t + 7;
// its window starts at ring bit (8t + 1) * Bits + 256 * Bits - 16 and state J is the 16 bits at window offset J * Bits.
// LaneGeom loads the three words that start at the window's word; with shift = start mod 32 the last state ends at
// bit shift + 7 * Bits + 16 of the first word, so a third word is read only when that end passes 64.
//   Bits 1..4: every lane's states lie in the first two words (K2: shift 18 / 2; K3: <= 27; K4: 20).
//   Bits 5..8: some lane needs the third word.
// Host only (no HIP types): included by moe_core/exl3_trellis.hpp and by the host tests.
namespace strix::moe_core::lane_window {

[[nodiscard]] constexpr unsigned start_bit(int bits, unsigned lane) {
    return (lane * 8U + 1U) * static_cast<unsigned>(bits) + 256U * static_cast<unsigned>(bits) - 16U;
}
[[nodiscard]] constexpr unsigned shift(int bits, unsigned lane) { return start_bit(bits, lane) & 31U; }
// One past the last window bit lane `lane` reads, counted from the first word's most significant bit.
[[nodiscard]] constexpr unsigned end_bit(int bits, unsigned lane) { return shift(bits, lane) + 7U * static_cast<unsigned>(bits) + 16U; }
// Words lane `lane` needs (1, 2 or 3; e.g. the K2 odd lanes end at bit 32 and read one word).
[[nodiscard]] constexpr unsigned words(int bits, unsigned lane) { return (end_bit(bits, lane) + 31U) / 32U; }
// True when some lane of the wave needs the third word: the reader must load it for every lane.
[[nodiscard]] constexpr bool needs_third_word(int bits) {
    for (unsigned lane = 0; lane < 32U; ++lane)
        if (words(bits, lane) == 3U) return true;
    return false;
}

}  // namespace strix::moe_core::lane_window

namespace strix::moe_core::split_policy {
// Host form of moe_core::vector_splits (exl3_trellis.hpp, the strix-trellis <= 8-row split policy) for host planners
// (ds41_k2_experts.cpp). exl3_trellis.hpp static_asserts that both agree on every 128-multiple shape up to 16384.
[[nodiscard]] constexpr unsigned vector_splits(unsigned input, unsigned output) {
    return output <= 1024 ? 32U : (output <= 4096 && input >= 4096 ? 16U : 4U);
}
}  // namespace strix::moe_core::split_policy
