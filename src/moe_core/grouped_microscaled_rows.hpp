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
// fit. narrow: decode-height CTAs (few rows, few waves, a small stage) for launches whose live tile rows fit one of them,
// so a 1..32-row expert tile neither stages nor holds LDS for 256 rows and more CTAs stay resident per compute unit;
// refused where the shape does not fit its stage; a backend without a narrow body runs staged. automatic: the backend's
// resolve_schedule below. A backend with a single schedule ignores the trait.
enum class Schedule : std::uint8_t { automatic, staged, tall, narrow };

// Resolution of Schedule::automatic (backend-neutral; the backend supplies its CTA heights and whether the launch shape
// fits the K stage shared by its tall and narrow bodies): narrow when the host-known bound on live tile rows fits one
// narrow CTA, or when the host-known mean live rows per tile does (a taller tile then runs ceil(max / narrow_rows) narrow
// row slices, the short slices exit; measured faster than the tall CTA while the mean stays within one narrow CTA);
// narrow_rows 0 = no narrow body; else tall where it fits, else staged. Every schedule stores the same bits, so the
// choice never changes a row's result (batch invariance holds across it).
STRIX_GROUPED_ROWS_HD constexpr Schedule resolve_schedule(Schedule requested, unsigned max_tile_rows, unsigned narrow_rows,
                                                          bool stage_fits, unsigned mean_tile_rows = 0U) {
    if (requested != Schedule::automatic) return requested;
    if (!stage_fits) return Schedule::staged;
    const auto fits_narrow = [narrow_rows](unsigned rows) { return rows != 0U && rows <= narrow_rows; };
    return fits_narrow(max_tile_rows) || fits_narrow(mean_tile_rows) ? Schedule::narrow : Schedule::tall;
}

// Operand form of a tile launch (backend-neutral trait). bf16_tile: the tile family above (Decode / Schedule). small_f16
// (FAST, bit-changing): tiles of any row count on the small-M schedule of the f16 tile family
// (core/f16_tile_gemm_small.hpp with core/microscaled_f16_staging.hpp): the codes are FP16 WMMA operands held in
// registers (one weight read per row block, no LDS stage for B), one FP32 multiplier per scale group on the accumulator,
// the outer scale at the store. Rows with any |BF16 input| > 65280 (or nonfinite input) use packed BF16 WMMA.
// Row r's bits depend on its own inputs and the shape only (not on the tile's other rows,
// its position in the tile, the row tile count or the row block count), so one form serves every row count of a launch
// class (a lone decode row, a cohort, a verify window) with the same bits per row. Needs max_tile_rows >= 1, inputs %
// 128 == 0, outputs % 64 == 0, tiles x row blocks within the y grid (fits_small, checked for both legs of a visit
// before any of it is submitted) and every expert matrix's codes 16-byte and scale plane 4-byte aligned
// (fits_small_matrix, admitted on the host before descriptors can select the form); the activated (epilogue) launches
// refuse it. Each row block reads the tile's weights again, so a tall tile (prefill) belongs to the
// BF16 tile: the caller selects the form per launch class.
enum class Form : std::uint8_t { bf16_tile, small_f16 };

// FP16 overflows at 65520 (RNE). BF16's adjacent magnitudes are 65280 (0x477f) and 65536 (0x4780).
STRIX_GROUPED_ROWS_HD constexpr bool small_f16_needs_bf16(std::uint16_t bits) { return (bits & 0x7FFFU) >= 0x4780U; }

// Row plan of the small form: CTAs of 16 row_tiles rows (row_tiles 1 for tiles of <= 16 rows, else 2), row_blocks of them
// per tile (y grid = tiles x row_blocks); {0, 0} refuses (no rows).
struct SmallRows {
    unsigned row_tiles, row_blocks;
};
STRIX_GROUPED_ROWS_HD constexpr SmallRows small_rows(unsigned max_tile_rows) {
    if (max_tile_rows == 0U) return {0U, 0U};
    const unsigned row_tiles = max_tile_rows <= 16U ? 1U : 2U;
    const unsigned block = 16U * row_tiles;
    return {row_tiles, max_tile_rows / block + (max_tile_rows % block != 0U ? 1U : 0U)};
}

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
    Form form{Form::bf16_tile};
    // Host-known mean live rows per tile (ceil(assignments / experts)); zero means unknown. Schedule choice only.
    unsigned mean_tile_rows{};
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

// Shape and grid of a small-form launch (Form::small_f16): the one predicate its launcher and the dispatcher's
// preflight of a whole visit share. Each CTA covers kSmallOutputs outputs (one projection panel) of one tile's row
// block over K in kSmallInputs steps; the y grid holds tiles x row blocks.
inline constexpr unsigned kSmallInputs = 128U, kSmallOutputs = 64U, kSmallGridY = 65535U;
inline bool fits_small(const Args& args) {
    const auto plan = small_rows(args.max_tile_rows);
    return args.tiles && args.count >= 1U && args.count <= kMaxTiles && plan.row_tiles != 0U &&
           args.inputs >= kSmallInputs && args.inputs % kSmallInputs == 0U && args.outputs >= kSmallOutputs &&
           args.outputs % kSmallOutputs == 0U && args.count <= kSmallGridY / plan.row_blocks;
}

// An expert matrix the small form may read (host admission, before descriptors select the form): the shape of
// fits_small, 16-byte code rows and a 4-byte aligned scale plane. The form loads scale bytes as 4-byte words (NVFP4:
// four groups; MXFP8: two groups, or one 2-byte pair), and with inputs % 128 every row's codes and scales keep the
// alignment of the plane's start. The BF16 tile's own admission (fits_rows) does not require it.
inline bool fits_small_matrix(const packed::PackedMatrix& w, unsigned inputs, unsigned outputs) {
    return w.codes && w.block_scales && w.rows == outputs && w.columns == inputs && inputs >= kSmallInputs &&
           inputs % kSmallInputs == 0U && outputs >= kSmallOutputs && outputs % kSmallOutputs == 0U &&
           reinterpret_cast<std::uintptr_t>(w.codes) % 16U == 0U &&
           reinterpret_cast<std::uintptr_t>(w.block_scales) % 4U == 0U;
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
