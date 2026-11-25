#pragma once

#include <cstddef>
#include <limits>

namespace strix::core {

struct F32InputMajorGrid {
    unsigned row_tiles = 0, output_tiles = 0;
    [[nodiscard]] constexpr bool valid() const noexcept { return output_tiles != 0; }
};

// The caller supplies the tile shape and its backend's grid limits.
template <class Tile>
[[nodiscard]] constexpr F32InputMajorGrid plan_f32_input_major_rows(
    std::size_t rows, unsigned in, unsigned out, unsigned out_stride,
    unsigned max_row_tiles, unsigned max_output_tiles) noexcept {
    static_assert(Tile::rows > 0 && Tile::outputs > 0 && Tile::k > 0);
    if (in == 0 || in % Tile::k != 0 || out == 0 || out_stride < out) return {};
    const auto row_tiles = rows == 0 ? 0 : (rows - 1) / Tile::rows + 1;
    const auto output_tiles = (out - 1) / Tile::outputs + 1;
    constexpr auto max_floats = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (row_tiles > max_row_tiles || output_tiles > max_output_tiles ||
        rows > max_floats / in || rows > max_floats / out_stride || in > max_floats / out) return {};
    return {static_cast<unsigned>(row_tiles), output_tiles};
}

}  // namespace strix::core
