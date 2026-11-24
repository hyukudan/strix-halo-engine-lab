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
#include "moe_core/nvfp4_act_rule.hpp"

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

// Activation operand class of a grouped projection pair (backend-neutral trait; bf16 = grouped_microscaled, the default).
//   nvfp4          both projections on the NVFP4 image, group scale rule nearest@1 (identity grouped-nvfp4-w4a4-f32-v1).
//   nvfp4_mse      both projections on the NVFP4 image, scale rule mse@1 (moe_core/nvfp4_act_rule.hpp); same GEMM identity.
//   nvfp4_mse_gate_up W4A4 gate-up (mse@1) and the down projection on BF16 activations (the grouped_microscaled down of bf16):
//                  removes the down input's share of the activation error at the cost of the down GEMM's FP4 rate.
//   nvfp4_mse_r2_gate_up  gate-up input as TWO NVFP4 terms (two-term rule r2@1 of moe_core/nvfp4_act_rule.hpp, mse@1 each),
//                  down input one term (mse@1); both projections on the FP4 MMA. The gate-up input carries most of the
//                  activation error that moves the next layer's routes (w4a4_q DIAG 11-24: 68 % of the per-layer route
//                  injection); two terms cost two FP4 passes of that projection instead of the BF16 tile's four.
//   nvfp4_mse_r2   both projection inputs as two NVFP4 terms (mse@1 each).
// Two terms: the image row holds [term 1 | term 2] (codes 2 x K/2 bytes, scales 2 x K/16, one row scale), and the projection
// runs the K-concatenation [q1 | q2] against [W | W]: same instruction, same stored weights, 2 x K/128 pipeline stages.
enum class Activations : std::uint8_t { bf16, nvfp4, nvfp4_mse, nvfp4_mse_gate_up, nvfp4_mse_r2_gate_up, nvfp4_mse_r2 };
using ScaleRule = nvfp4_act::ScaleRule;
inline constexpr unsigned kMaxTerms = 2U;
constexpr bool w4a4_gate_up(Activations a) { return a != Activations::bf16; }
constexpr bool w4a4_down(Activations a) { return a != Activations::bf16 && a != Activations::nvfp4_mse_gate_up; }
constexpr ScaleRule scale_rule(Activations a) { return a == Activations::nvfp4 ? ScaleRule::nearest : ScaleRule::mse; }
constexpr unsigned terms_gate_up(Activations a) {
    return a == Activations::nvfp4_mse_r2_gate_up || a == Activations::nvfp4_mse_r2 ? 2U : 1U;
}
constexpr unsigned terms_down(Activations a) { return a == Activations::nvfp4_mse_r2 ? 2U : 1U; }
constexpr const char* identity(Activations a) {
    return a == Activations::bf16 ? "bf16" : a == Activations::nvfp4 ? "nvfp4-nearest@1"
         : a == Activations::nvfp4_mse ? "nvfp4-mse@1" : a == Activations::nvfp4_mse_gate_up ? "nvfp4-mse-gate-up@1"
         : a == Activations::nvfp4_mse_r2_gate_up ? "nvfp4-mse-r2-gate-up@1" : "nvfp4-mse-r2@1";
}

// NVFP4 image of assignment rows (device addresses). Row a holds assignment a of the plan.
struct Image {
    std::uint8_t* codes{};
    std::uint8_t* scales{};
    float* row_scale{};
    std::size_t code_row{};   // bytes
    std::size_t scale_row{};  // bytes
};

// Bytes of an image of `rows` rows of `width` inputs in `terms` terms at offset 0 of a 256-byte aligned buffer (codes, scales,
// row scales). Term t of a row starts at code byte t * width / 2 and scale byte t * width / 16 of that row.
struct ImageLayout {
    std::size_t codes{}, scales{}, row_scale{}, bytes{};
};
constexpr std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1U) / a * a; }
constexpr ImageLayout image_layout(std::size_t rows, unsigned width, unsigned terms = 1U) {
    ImageLayout l{};
    l.codes = 0U;
    l.scales = align_up(rows * terms * (width / 2U), 256U);
    l.row_scale = align_up(l.scales + rows * terms * (width / kGroup), 256U);
    l.bytes = align_up(l.row_scale + rows * sizeof(float), 256U);
    return l;
}
inline Image make_image(std::uint8_t* base, std::size_t rows, unsigned width, unsigned terms = 1U) {
    const auto l = image_layout(rows, width, terms);
    return {base + l.codes, base + l.scales, reinterpret_cast<float*>(base + l.row_scale), terms * (width / 2U),
            terms * (width / kGroup)};
}

// A grouped projection over the plan's tiles whose input rows are taken from `input`: tile t reads image rows
// first + r, first = (tiles[t].input - row_base) / tiles[t].input_row (the tile's BF16 address is used as an index only and is
// never dereferenced).
// terms 2: image rows hold [q1 | q2] of tiles.inputs each; the projection sums both against the same weights.
struct Args {
    grouped_microscaled::Args tiles;
    const std::uint16_t* row_base{};
    Image input;
    unsigned terms{1U};
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
    unsigned terms{1U};  // 2: term 2 = the image of the residual (nvfp4_act::residual) at the same row scale and scale rule
};
constexpr bool valid_terms(unsigned terms) { return terms >= 1U && terms <= kMaxTerms; }

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
           valid_terms(args.terms) && args.tiles.inputs <= ~0U / args.terms &&
           fits_image(args.input, args.tiles.inputs * args.terms);
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
