#pragma once
// FAST W4A4 operand class of the grouped expert projection (Home: shared, backend-neutral moe_core contract; the sm_120 body is
// src/cuda/core/arch/sm120/grouped_w4a4.cuh, a backend without a block-scaled FP4 tensor instruction declines by name).
//
// Same plan, tiles and outputs as grouped_microscaled (TileRows of <= 128 routed rows of one expert, FP32 [rows][outputs]); only
// the activation operand changes: instead of BF16 rows each tile row reads an NVFP4 image of the same assignment row, made once
// per projection input by the family's activation quantiser (one FP32 scale per row, rs = amax / 2688; per 16 inputs an E4M3
// scale and 16 E2M1 codes; the rule of strix/cuda/core/nvfp4_tile.hpp, stored canonically: codes [rows][K/2] low nibble first,
// scales [rows][K/16]). NVFP4 weights are consumed as stored (PackedMatrix codes / block scales), with no decode.
//
// Arithmetic identity "grouped-nvfp4-w4a4-f32-v1": per (row, output) the exact E2M1 x E4M3 products are summed in FP32 by the
// tensor core over 64-input blocks in ascending order (order inside a block not specified by the ISA, deterministic for one
// plan), then y = acc * (outer_scale * rs[row]) with IEEE single rounding. Bits differ from the BF16 tile: FAST class only.
#include "moe_core/grouped_microscaled_rows.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::moe_core::grouped_w4a4 {

namespace packed = strix::weight::microscaled;
using grouped_microscaled::TileRows;

inline constexpr const char* kArithmeticId = "grouped-nvfp4-w4a4-f32-v1";
inline constexpr float kRowScaleDiv = 2688.0F;  // 6 * 448: a block scale never saturates
inline constexpr unsigned kTileRows = 64U;      // routed rows per CTA
// Rows per plan tile: a tile above kTileRows rows runs as row slices of kTileRows (grid y). Slices start on multiples of 64,
// so every row keeps its n8 position and its bits are those of the 64-row plan.
inline constexpr unsigned kMaxTileRows = 128U;
inline constexpr unsigned kTileOutputs = 128U;
inline constexpr unsigned kInputBlock = 128U;   // inputs per pipeline stage (two 64-input MMA blocks)
inline constexpr unsigned kGroup = 16U;

// Activation operand class of a grouped projection (backend-neutral trait; bf16 = grouped_microscaled, the default).
enum class Activations : std::uint8_t { bf16, nvfp4 };

// NVFP4 image of assignment rows (device addresses). Row a holds assignment a of the plan.
struct Image {
    std::uint8_t* codes{};
    std::uint8_t* scales{};
    float* row_scale{};
    std::size_t code_row{};   // bytes
    std::size_t scale_row{};  // bytes
};

// Bytes of an image of `rows` rows of `width` inputs at offset 0 of a 256-byte aligned buffer (codes, scales, row scales).
struct ImageLayout {
    std::size_t codes{}, scales{}, row_scale{}, bytes{};
};
constexpr std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1U) / a * a; }
constexpr ImageLayout image_layout(std::size_t rows, unsigned width) {
    ImageLayout l{};
    l.codes = 0U;
    l.scales = align_up(rows * (width / 2U), 256U);
    l.row_scale = align_up(l.scales + rows * (width / kGroup), 256U);
    l.bytes = align_up(l.row_scale + rows * sizeof(float), 256U);
    return l;
}
inline Image make_image(std::uint8_t* base, std::size_t rows, unsigned width) {
    const auto l = image_layout(rows, width);
    return {base + l.codes, base + l.scales, reinterpret_cast<float*>(base + l.row_scale), width / 2U, width / kGroup};
}

// A grouped projection over the plan's tiles whose input rows are taken from `input`: tile t reads image rows
// first + r, first = (tiles[t].input - row_base) / tiles[t].input_row (the tile's BF16 address is used as an index only and is
// never dereferenced).
struct Args {
    grouped_microscaled::Args tiles;
    const std::uint16_t* row_base{};
    Image input;
};

// Quantiser launch: dst row a = quantise(source row (index ? index[a] : a)), rows x width.
template <class T>
struct QuantizeArgs {
    const T* source{};
    std::size_t source_row{};         // elements
    const std::uint32_t* index{};     // optional
    unsigned rows{};
    unsigned width{};
    Image output;
    const unsigned* failure{};
};

inline bool fits_image(const Image& image, unsigned width) {
    return image.codes && image.scales && image.row_scale && width % kInputBlock == 0U &&
           image.code_row >= width / 2U && image.code_row % 16U == 0U && image.scale_row >= width / kGroup &&
           image.scale_row % 8U == 0U && reinterpret_cast<std::uintptr_t>(image.codes) % 16U == 0U &&
           reinterpret_cast<std::uintptr_t>(image.scales) % 8U == 0U &&
           reinterpret_cast<std::uintptr_t>(image.row_scale) % 4U == 0U;
}

inline bool fits(const Args& args) {
    return grouped_microscaled::fits_args(args.tiles, kInputBlock, kTileOutputs) && args.row_base &&
           args.tiles.max_tile_rows <= kMaxTileRows &&
           fits_image(args.input, args.tiles.inputs);
}

// Admission of one tile: an NVFP4 matrix whose code and scale rows start on 16- / 8-byte boundaries (stage copies of 64 code
// bytes and 8 scale bytes per row), plus the grouped_microscaled tile rules.
inline bool fits_tile(const TileRows& tile, unsigned inputs, unsigned outputs) {
    const auto& w = tile.weights;
    return outputs % kTileOutputs == 0U && inputs % kInputBlock == 0U &&
           grouped_microscaled::fits_rows<packed::MatrixFormat::nvfp4>(tile, inputs, outputs, kInputBlock, kMaxTileRows) &&
           reinterpret_cast<std::uintptr_t>(w.codes) % 16U == 0U &&
           reinterpret_cast<std::uintptr_t>(w.block_scales) % 8U == 0U;
}

}  // namespace strix::moe_core::grouped_w4a4
