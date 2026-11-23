#pragma once

#include "strix/weight/formats/microscaled.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_GROUPED_ROWS_HD __host__ __device__
#else
#define STRIX_GROUPED_ROWS_HD
#endif

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

// Weight decode form of a tile launch (backend-neutral trait). reference: PackedOperand::decode_bf16. shift: NVFP4 tiles
// use PackedOperand::decode_bf16_shift, bitwise the same BF16 operands with fewer instructions (MXFP8 keeps decode_bf16).
enum class Decode : std::uint8_t { reference, shift };

// Schedule of a tile launch (backend-neutral trait; never changes bits). staged: the incumbent K16 tile with a register
// lookahead of one step. tall: the slab ring of moe_core/grouped_tall_plan.hpp (128-row CTAs, one weight decode per CTA
// and stage, asynchronous copies, one barrier per stage); refused where the backend has no tall body or the shape does not
// fit. automatic: tall where it fits, else staged. A backend with a single schedule ignores the trait.
enum class Schedule : std::uint8_t { automatic, staged, tall };

struct Args {
    const TileRows* tiles{};
    unsigned count{};
    unsigned inputs{};
    unsigned outputs{};
    const unsigned* tile_count{};  // Optional device count, bounded by count.
    const unsigned* failure{};     // Optional device plan failure word.
    unsigned max_tile_rows{};      // Host-known upper bound on live tile rows; zero means unknown.
    // Optional rebase of activation pointers into a separate region: p -> rebase_to + (p - rebase_from). The plain
    // binding applies it to each tile's input (the down projection reading the region); the activated gate-up binding
    // applies it to pair_rows[t].input, where tile t stores its BF16 activations (pair_rows = the down tiles).
    const std::uint16_t* rebase_from{};
    std::uint16_t* rebase_to{};
    const TileRows* pair_rows{};
    float gate_cap{};
    float up_cap{};
    Decode decode{Decode::reference};
    Schedule schedule{Schedule::automatic};
};

// Args of a gate-up whose tiles read their input rows in place: a tile whose input is gather_from + first x input_row
// (the address of a gathered copy that is never written) reads its row r from gather_from + gather_rows[first + r] x
// input_row, the BF16 image of the token rows (grouped_rows::token_rows), row for row the gather's copy.
struct IndirectArgs : Args {
    const std::uint32_t* gather_rows{};
    const std::uint16_t* gather_from{};
};

// Row plan of a grouped tile launch (backend-neutral; the CUDA tile and tall launches and the HIP tile launch bind it):
// grid y = row slices of cta_rows rows over the tallest descriptor tile, whose live rows the launch bounds by
// max_tile_rows (0 = unknown) and the plan by its descriptor capacity tile_capacity. A slice past a shorter tile's rows
// leaves before any barrier. Example: 129 live rows at 128 rows per CTA take 2 slices, an unknown bound with a 256-row
// capacity also 2.
STRIX_GROUPED_ROWS_HD constexpr unsigned tile_row_slices(unsigned max_tile_rows, unsigned tile_capacity,
                                                         unsigned cta_rows) {
    const unsigned rows = max_tile_rows == 0U || max_tile_rows > tile_capacity ? tile_capacity : max_tile_rows;
    return cta_rows == 0U ? 0U : rows / cta_rows + (rows % cta_rows != 0U ? 1U : 0U);
}

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
    } else {
        return true;
    }
}

}  // namespace strix::moe_core::grouped_microscaled
