#pragma once
// Byte spans the f16x K-quant down decoder (moe_core/hip_moe_f16x_kquant_down.inc) reads from one Q6_K block. Host-visible so
// the read discipline is tested without a device: every read of a block stays inside the 4-byte-aligned words that hold
// the block's own bytes, and never past the block's last byte, so a tensor needs no tail slack after its last block.
// A Q6_K block is 210 bytes (ql 128, qh 64, int8 scales 16, FP16 d at 208) and starts on an even byte; odd = address & 2.
// The ql / qh / scale words are read dword-aligned from base = block - odd (9 dwords each, realigned with alignbit): for
// odd = 2 they start 2 bytes before the block, inside the dword that holds the block's first bytes. d is read as one
// 16-bit word at block + 208 (2-byte aligned), not as the dword at base + 208, which for odd = 0 covers bytes 210..211
// past the block.
#include <cstdint>

namespace strix::moe_core::q6k_reads {

inline constexpr unsigned block_bytes = 210U;
inline constexpr unsigned dwords = 9U;  // ql / qh: 8 words of 32 weights + 1 for the alignbit shift

// [begin, end) relative to the block start (may be -2 for odd = 2: the leading bytes of the block's own dword).
struct Span { int begin, end; };

[[nodiscard]] constexpr unsigned sub_n(unsigned step) { return step % 4U / 2U; }
[[nodiscard]] constexpr unsigned sub_j(unsigned step, unsigned half) { return 2U * (step % 2U) + half; }

// Offsets from base = block - odd (what the device code adds to its dword-aligned pointer).
[[nodiscard]] constexpr unsigned ql_from_base(unsigned step, unsigned half) { return 64U * sub_n(step) + 32U * half; }
[[nodiscard]] constexpr unsigned qh_from_base(unsigned step) { return 128U + 32U * sub_n(step); }
[[nodiscard]] constexpr unsigned sc_at(unsigned odd, unsigned step, unsigned half) {
    return odd + 192U + 8U * sub_n(step) + 2U * sub_j(step, half);
}
[[nodiscard]] constexpr unsigned sc_word_from_base(unsigned odd, unsigned step, unsigned half) { return sc_at(odd, step, half) & ~3U; }
inline constexpr unsigned d_from_block = 208U;  // 16-bit load

[[nodiscard]] constexpr Span ql_span(unsigned odd, unsigned step, unsigned half) {
    const int b = static_cast<int>(ql_from_base(step, half)) - static_cast<int>(odd);
    return {b, b + static_cast<int>(4U * dwords)};
}
[[nodiscard]] constexpr Span qh_span(unsigned odd, unsigned step) {
    const int b = static_cast<int>(qh_from_base(step)) - static_cast<int>(odd);
    return {b, b + static_cast<int>(4U * dwords)};
}
[[nodiscard]] constexpr Span sc_span(unsigned odd, unsigned step, unsigned half) {
    const int b = static_cast<int>(sc_word_from_base(odd, step, half)) - static_cast<int>(odd);
    return {b, b + 4};
}
[[nodiscard]] constexpr Span d_span() { return {static_cast<int>(d_from_block), static_cast<int>(d_from_block) + 2}; }

// The read discipline: a span may start up to `odd` bytes before the block (same aligned dword as the block's first
// byte) and must end at or before the block's last byte.
[[nodiscard]] constexpr bool span_ok(Span s, unsigned odd) {
    return s.begin >= -static_cast<int>(odd) && s.begin < s.end && s.end <= static_cast<int>(block_bytes);
}
[[nodiscard]] constexpr bool all_reads_in_block() {
    for (unsigned odd = 0U; odd <= 2U; odd += 2U)
        for (unsigned step = 0U; step < 4U; ++step)
            for (unsigned half = 0U; half < 2U; ++half) {
                if (!span_ok(ql_span(odd, step, half), odd) || !span_ok(qh_span(odd, step), odd) ||
                    !span_ok(sc_span(odd, step, half), odd) || !span_ok(d_span(), 0U))
                    return false;
                // the two scale bytes never straddle the word read
                if ((sc_at(odd, step, half) & 3U) == 3U) return false;
            }
    return true;
}
static_assert(all_reads_in_block(), "Q6_K down decode reads outside its block");

}  // namespace strix::moe_core::q6k_reads
