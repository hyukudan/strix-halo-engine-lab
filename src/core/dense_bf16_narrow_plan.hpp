#pragma once

namespace strix::core {

// Split complete incumbent m16 tiles. Both halves keep their original MMA
// fragment positions. This explicitly selected form covers one activation row.
[[nodiscard]] constexpr unsigned bf16_narrow_blocks(unsigned in, unsigned out, unsigned input_alignment,
                                                    unsigned output_tile) noexcept {
    if (input_alignment == 0U || output_tile == 0U || output_tile % 2U != 0U ||
        in == 0U || in % input_alignment != 0U || out == 0U || out % output_tile != 0U)
        return 0U;
    return out / (output_tile / 2U);
}

// Widest projection a dense dispatch binds to the split form without an explicit request: the measured narrow class
// (the low-rank controller's rank-64 down projection, whose incumbent grid has 4 blocks). Wider projections keep the
// incumbent until measured.
inline constexpr unsigned kBf16NarrowSelectOutputs = 64U;

// Whether a short-rows rounded MMA3 dispatch (the incumbent rounded MMA3 rows kernel over `projections` projections
// and `rows` rows) takes the output-split form: exactly one row and one rounded projection of a shape the split
// plan admits, at most kBf16NarrowSelectOutputs outputs. The split is the incumbent's arithmetic (same K32 blocks,
// lo/mid/hi chain and fragment slots), so either form gives the same bits; everything else keeps the incumbent.
[[nodiscard]] constexpr bool bf16_narrow_selected(unsigned rows, unsigned projections, bool round_bf16, unsigned in,
                                                  unsigned out, unsigned input_alignment,
                                                  unsigned output_tile) noexcept {
    return rows == 1U && projections == 1U && round_bf16 && out <= kBf16NarrowSelectOutputs &&
           bf16_narrow_blocks(in, out, input_alignment, output_tile) != 0U;
}

} // namespace strix::core
