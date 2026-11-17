#pragma once
// Host-only decision of the selected-flash pipelined loads (Pf, sparse_flash_wmma.hpp), usable from plain C++ without
// HIP headers. Pf is bitwise (same operands, same order): every choice here changes speed only, never the result.
#include <cstdint>

namespace strix::attn_core::sparse {

// Auto threshold of the T 32 core: Pf hides latency only when there are key blocks to overlap, and at T 32 its extra
// VGPRs (gfx1151 Pf on / off 238 / 199) cost occupancy on short lists. Receipts (DS41 T 32, ABAB, bitwise): lists 640
// -4.3..-4.7 %, 228 +3.3 %, <= 128 +12..15 % -> on from kPfMinKeys.
// T 16 has no such cost (Pf on / off 210 / 253 VGPRs, Pf uses fewer) and its only receipt is GLM T 16 lists <= 2,051
// flash ~-19 %; no short-list T 16 measurement exists, so auto keeps T 16 on at every bound.
inline constexpr unsigned kPfMinKeys = 320;

// mode: -1 auto, 0 forced off, 1 forced on (STRIX_ATTN_FLASH_PF). T: key block of the launch. max_keys: host upper
// bound of the launch's longest key list (~0U = unknown, no bound).
constexpr bool flash_pf_auto(unsigned T, unsigned max_keys) { return T <= 16U || max_keys >= kPfMinKeys; }
constexpr bool flash_pf_decide(int mode, unsigned T, unsigned max_keys) {
    return mode < 0 ? flash_pf_auto(T, max_keys) : mode == 1;
}
inline const char* flash_pf_reason_text(int mode, unsigned T, unsigned max_keys) {
    if (mode >= 0) return mode == 1 ? "on (STRIX_ATTN_FLASH_PF=1)" : "off (STRIX_ATTN_FLASH_PF=0)";
    if (T <= 16U) return "on (auto: T 16, no occupancy cost)";
    return max_keys >= kPfMinKeys ? "on (auto: key lists >= 320)" : "off (auto: key lists < 320, nothing to overlap)";
}

// Host bound of a causal selected list: a query row at position p reads distinct keys <= p, at most list_cap of them,
// so every row of a launch whose last position is end - 1 has at most min(end, list_cap) keys. Never derived from
// device data (no sync).
constexpr unsigned causal_list_bound(std::uint64_t end, unsigned list_cap) {
    return end < list_cap ? static_cast<unsigned>(end) : list_cap;
}

}  // namespace strix::attn_core::sparse
