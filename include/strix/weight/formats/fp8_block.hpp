#pragma once

#include "strix/weight/formats/scalar_float.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace strix::weight {

// Row-major E4M3 finite-only matrix with one scale per rectangular tile.
// The scale plane uses ceil(rows / BlockRows) x ceil(columns / BlockColumns).
// Code rows are unpadded, including at the matrix edges.
template <std::size_t BlockRows, std::size_t BlockColumns, class Scale>
class BlockScaledE4M3 {
    static_assert(BlockRows > 0 && BlockColumns > 0);
    static_assert(std::is_same_v<Scale, float> || std::is_same_v<Scale, std::uint8_t>,
                  "the scale plane is FP32 or E8M0 bytes; any other type would convert silently and decode wrong values");
public:
    constexpr static std::size_t scale_rows(std::size_t rows) {
        return rows / BlockRows + (rows % BlockRows != 0U);
    }
    constexpr static std::size_t scale_columns(std::size_t columns) {
        return columns / BlockColumns + (columns % BlockColumns != 0U);
    }

    BlockScaledE4M3(std::span<const std::uint8_t> codes, std::span<const Scale> scales,
                   std::size_t rows, std::size_t columns)
        : codes_(codes), scales_(scales), rows_(rows), columns_(columns), scale_columns_(scale_columns(columns)) {
        if (rows == 0U || columns == 0U || codes.size() / columns != rows || codes.size() % columns != 0U ||
            scales.size() / scale_columns_ != scale_rows(rows) || scales.size() % scale_columns_ != 0U)
            throw std::invalid_argument("E4M3 block matrix: code and scale shapes disagree");
    }

    void decode_row(std::size_t row, std::span<float> out) const {
        if (row >= rows_ || out.size() != columns_)
            throw std::invalid_argument("E4M3 block matrix: invalid row");
        const auto codes = codes_.subspan(row * columns_, columns_);
        const auto scales = scales_.subspan((row / BlockRows) * scale_columns_, scale_columns_);
        for (std::size_t col = 0; col < columns_; ++col)
            out[col] = scalar_float::e4m3(codes[col]) * expand_scale(scales[col / BlockColumns]);
    }

private:
    static float expand_scale(float value) { return value; }
    static float expand_scale(std::uint8_t value) { return scalar_float::e8m0(value); }

    std::span<const std::uint8_t> codes_;
    std::span<const Scale> scales_;
    std::size_t rows_, columns_, scale_columns_;
};

} // namespace strix::weight
