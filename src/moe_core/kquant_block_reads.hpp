#pragma once
// Byte spans the f16x K-quant down decoder (moe_core/hip_moe_f16x_kquant_down.inc) reads from one Q6_K block. Host-visible so
// the read discipline is tested without a device: every read of a block stays inside the 4-byte-aligned words that hold
// the block's own bytes, and never past the block's last byte, so a tensor needs no tail slack after its last block.
// A Q6_K block is 210 bytes (ql 128, qh 64, int8 scales 16, FP16 d at 208) and starts on an even byte; odd = address & 2.
// The decoder reads a step PAIR at once: the 64-K steps 2m and 2m+1 of a superblock use the same ql bytes (low / high
// nibble), the same qh bytes (bit pairs j = 2(s%2)+h) and the adjacent scale bytes 192+8n..199+8n (n = (s%4)/2).
// The ql / qh / scale words are read dword-aligned from base = block - odd and realigned with alignbit: for odd = 2 they
// start 2 bytes before the block, inside the dword that holds the block's first bytes. The 8 scale bytes are one 8-byte
// read at base + 192 + 8n plus one tail dword, at base + 200 + 8n for odd = 2 (the last two scale bytes) and a re-read of
// base + 192 + 8n for odd = 0 (base + 200 + 8n would cover bytes 210..211 past the block for n = 1). d is read as one
// 16-bit word at block + 208 (2-byte aligned), not as the dword at base + 208, which for odd = 0 covers bytes 210..211.
#include <cstdint>

namespace strix::moe_core::q6k_reads {
inline constexpr unsigned block_bytes = 210U;
inline constexpr unsigned qh_dwords = 9U;  // 8 words of 32 weights + 1 for the alignbit shift
// ql dwords of a pair read covering `halves` consecutive halves (32 weights each) + 1 for the alignbit shift.
[[nodiscard]] constexpr unsigned ql_dwords(unsigned halves) { return 8U * halves + 1U; }
// [begin, end) relative to the block start (may be -2 for odd = 2: the leading bytes of the block's own dword).
struct Span { int begin, end; };
[[nodiscard]] constexpr unsigned sub_n(unsigned step) { return step % 4U / 2U; }
[[nodiscard]] constexpr unsigned sub_j(unsigned step, unsigned half) { return 2U * (step % 2U) + half; }
// Offsets from base = block - odd (what the device code adds to its dword-aligned pointer).
[[nodiscard]] constexpr unsigned ql_from_base(unsigned step, unsigned first_half) { return 64U * sub_n(step) + 32U * first_half; }
[[nodiscard]] constexpr unsigned qh_from_base(unsigned step) { return 128U + 32U * sub_n(step); }
[[nodiscard]] constexpr unsigned sc_pair_from_base(unsigned step) { return 192U + 8U * sub_n(step); }  // 8-byte read
[[nodiscard]] constexpr unsigned sc_tail_from_base(unsigned odd, unsigned step) { return sc_pair_from_base(step) + (odd ? 8U : 0U); }
// Byte of the 8 realigned scale bytes that holds the scale of sub-block 2j (l < 16); the next byte holds 2j + 1.
[[nodiscard]] constexpr unsigned sc_byte(unsigned step, unsigned half) { return 2U * sub_j(step, half); }
inline constexpr unsigned d_from_block = 208U;  // 16-bit load
[[nodiscard]] constexpr Span from_base(unsigned odd, unsigned at, unsigned bytes) {
    const int b = static_cast<int>(at) - static_cast<int>(odd);
    return {b, b + static_cast<int>(bytes)};
}
[[nodiscard]] constexpr Span ql_span(unsigned odd, unsigned step, unsigned first_half, unsigned halves) {
    return from_base(odd, ql_from_base(step, first_half), 4U * ql_dwords(halves));
}
[[nodiscard]] constexpr Span qh_span(unsigned odd, unsigned step) { return from_base(odd, qh_from_base(step), 4U * qh_dwords); }
[[nodiscard]] constexpr Span sc_pair_span(unsigned odd, unsigned step) { return from_base(odd, sc_pair_from_base(step), 8U); }
[[nodiscard]] constexpr Span sc_tail_span(unsigned odd, unsigned step) { return from_base(odd, sc_tail_from_base(odd, step), 4U); }
[[nodiscard]] constexpr Span d_span() { return {static_cast<int>(d_from_block), static_cast<int>(d_from_block) + 2}; }
// The read discipline: a span may start up to `odd` bytes before the block (same aligned dword as the block's first
// byte) and must end at or before the block's last byte.
[[nodiscard]] constexpr bool span_ok(Span s, unsigned odd) {
    return s.begin >= -static_cast<int>(odd) && s.begin < s.end && s.end <= static_cast<int>(block_bytes);
}
// Every pair read (even steps; one half from either half, or both halves) stays in the block, and the realigned scale
// bytes (pair read + tail, shifted by odd bytes) are exactly the block's bytes 192+8n..199+8n.
[[nodiscard]] constexpr bool all_reads_in_block() {
    for (unsigned odd = 0U; odd <= 2U; odd += 2U)
        for (unsigned step = 0U; step < 4U; step += 2U) {
            for (unsigned halves = 1U; halves <= 2U; ++halves)
                for (unsigned first = 0U; first + halves <= 2U; ++first)
                    if (!span_ok(ql_span(odd, step, first, halves), odd)) return false;
            if (!span_ok(qh_span(odd, step), odd) || !span_ok(sc_pair_span(odd, step), odd) ||
                !span_ok(sc_tail_span(odd, step), odd) || !span_ok(d_span(), 0U))
                return false;
            const Span p = sc_pair_span(odd, step), t = sc_tail_span(odd, step);
            if (p.begin + static_cast<int>(odd) != static_cast<int>(192U + 8U * sub_n(step))) return false;
            if (odd != 0U && t.begin != p.end) return false;  // odd = 2: the tail continues the pair read
        }
    return true;
}
static_assert(all_reads_in_block(), "Q6_K down decode reads outside its block");

}  // namespace strix::moe_core::q6k_reads
