#pragma once

#include "strix/weight/formats/microscaled.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strix::moe_core::grouped_microscaled {

namespace packed = strix::weight::microscaled;
inline constexpr unsigned kMaxTiles = 65535U;

// One expert and up to one backend row tile of its routed activations.
// Larger groups provide additional descriptors with advanced row pointers.
// Input and output strides are in elements, and buffers must not overlap.
struct TileRows {
    packed::PackedMatrix weights;
    const std::uint16_t* input{};
    float* output{};
    std::size_t input_row{};
    std::size_t output_row{};
    unsigned rows{};
};

struct Args {
    const TileRows* tiles{};
    unsigned count{};
    unsigned inputs{};
    unsigned outputs{};
    const unsigned* tile_count{};  // Optional device count, bounded by count.
    const unsigned* failure{};     // Optional device plan failure word.
};

inline bool fits_args(const Args& args, unsigned k, unsigned n) {
    return args.tiles && args.count >= 1U && args.count <= kMaxTiles &&
           args.inputs >= k && args.inputs % k == 0U && args.outputs >= n && args.outputs % n == 0U;
}

template <packed::MatrixFormat Format>
inline bool fits_rows(const TileRows& tile, unsigned inputs, unsigned outputs, unsigned k, unsigned m) {
    const auto& w = tile.weights;
    constexpr auto code_alignment = packed::PackedOperand<Format>::kCodes;
    constexpr auto max_input = std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t);
    constexpr auto max_output = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!w.codes || !w.block_scales || w.rows != outputs || w.columns != inputs ||
        inputs < k || outputs == 0 || inputs % k != 0 ||
        reinterpret_cast<std::uintptr_t>(w.codes) % code_alignment != 0 ||
        !tile.input || !tile.output || tile.rows == 0 || tile.rows > m ||
        tile.input_row < inputs || tile.output_row < outputs || tile.input_row % 8U != 0 ||
        reinterpret_cast<std::uintptr_t>(tile.input) % 16U != 0 ||
        tile.rows > max_input / tile.input_row || tile.rows > max_output / tile.output_row ||
        outputs > max_input / inputs) return false;
    if (w.row_order != packed::RowOrder::contiguous &&
        (w.row_order != packed::RowOrder::interleaved_halves || w.rows % 2U != 0)) return false;
    if constexpr (Format == packed::MatrixFormat::nvfp4) {
        return std::isfinite(w.outer_scale) && w.outer_scale >= 0.0F;
    }
    return true;
}

}  // namespace strix::moe_core::grouped_microscaled
