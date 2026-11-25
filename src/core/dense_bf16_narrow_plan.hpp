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

} // namespace strix::core
