#pragma once

#include "core/f16_tile_gemm.hpp"
#include "moe_core/expert_activation.hpp"
#include "moe_core/grouped_microscaled_rows.hpp"

namespace strix::moe_core::grouped_microscaled {

namespace gemm = strix::core::f16_tile_gemm;

// Rows per descriptor tile of the HIP plan: one CTA of the tall schedule (below). Inputs that are not a multiple of its
// stage, and Schedule::staged, run the incumbent 128-row CTA over row slices of the same tiles (same bits).
inline constexpr unsigned kRowsPerTile = 256U;

inline bool fits(const Args& args) {
    return fits_args(args, gemm::kBK, gemm::kBN);
}

template <packed::MatrixFormat Format>
inline bool fits_tile(const TileRows& tile, unsigned inputs, unsigned outputs) {
    return fits_rows<Format>(tile, inputs, outputs, gemm::kBK, kRowsPerTile);
}

#if defined(__HIPCC__)

// Tall schedule (EXACT, placement only): 256 rows x 128 outputs per CTA, 16 waves of 32 x 64 wave tiles, K stage 64, so
// each decoded weight stage feeds twice the rows of the incumbent (gemm::IncumbentSchedule: 128 rows, 8 waves, K32).
using TallSchedule = gemm::TileSchedule<2, 8, 64>;
static_assert(TallSchedule::kRows == kRowsPerTile);

template <packed::MatrixFormat Format>
struct TileBinding {
    using Args = grouped_microscaled::Args;
    using Store = gemm::RawStore;
    struct Tile {
        gemm::Args shape;
        Store store;
        packed::PackedOperand<Format> weights;
    };

    __device__ static __forceinline__ bool active(const Args& args, unsigned tile) {
        return (!args.failure || *args.failure == 0U) && (!args.tile_count || tile < *args.tile_count);
    }

    __device__ static __forceinline__ Tile bind(const Args& args, unsigned tile) {
        const TileRows rows = args.tiles[tile];
        gemm::Args shape{};
        shape.a16 = args.rebase_to ? args.rebase_to + (rows.input - args.rebase_from) : rows.input;
        shape.a_row = rows.input_row;
        shape.y_row = rows.output_row;
        shape.rows = rows.rows;
        shape.K = args.inputs;
        shape.N = args.outputs;
        return {shape, {rows.output, rows.output_row, 0U}, {rows.weights}};
    }
};

// Gate-up with the expert activation as the tile epilogue: the CTA owns gate columns [64 x, 64 x + 64) and the matching
// up columns (+ outputs / 2), and stores activate_bf16(gate sum, up sum) -- the bits the FP32 [gate | up] image plus
// expert_activation::launch would store -- at row r, column c of the tile's activation rows.
struct ActivationPair {
    static constexpr bool kPaired = true;
    unsigned half;
    std::uint16_t* output;
    std::size_t output_row;
    float gate_cap, up_cap;
    __device__ void put(unsigned row, unsigned column, float gate, float up) const {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif
        const moe::SoftcappedSwiglu<expert_activation::detail::DeviceMath> swiglu{gate_cap, up_cap};
        output[std::size_t{row} * output_row + column] = expert_activation::detail::activate_bf16(swiglu, gate, up);
    }
};

template <packed::MatrixFormat Format>
struct ActivatedTileBinding {
    using Args = grouped_microscaled::Args;
    using Store = gemm::RawStore;
    struct Tile {
        gemm::Args shape;
        Store store;
        packed::PackedOperand<Format> weights;
        ActivationPair pair;
    };

    __device__ static __forceinline__ bool active(const Args& args, unsigned tile) {
        return TileBinding<Format>::active(args, tile);
    }

    __device__ static __forceinline__ Tile bind(const Args& args, unsigned tile) {
        const TileRows rows = args.tiles[tile];
        const TileRows paired = args.pair_rows[tile];
        gemm::Args shape{};
        shape.a16 = rows.input;
        shape.a_row = rows.input_row;
        shape.y_row = rows.output_row;
        shape.rows = rows.rows;
        shape.K = args.inputs;
        shape.N = args.outputs;
        return {shape, {rows.output, rows.output_row, 0U}, {rows.weights},
                {args.outputs / 2U, args.rebase_to + (paired.input - args.rebase_from), paired.input_row, args.gate_cap,
                 args.up_cap}};
    }
};

// A binding (TileBinding or ActivatedTileBinding) with its tiles' A rows read through IndirectArgs::gather_rows.
template <class Base>
struct IndirectRows {
    using Args = IndirectArgs;
    using Store = typename Base::Store;
    struct Tile : Base::Tile {
        gemm::MappedRows rows;
    };

    __device__ static __forceinline__ bool active(const Args& args, unsigned tile) { return Base::active(args, tile); }

    __device__ static __forceinline__ Tile bind(const Args& args, unsigned tile) {
        Tile bound{Base::bind(args, tile), {}};
        const TileRows rows = args.tiles[tile];
        const auto first = static_cast<std::size_t>(rows.input - args.gather_from) / rows.input_row;
        bound.shape.a16 = args.gather_from;
        bound.rows.map = args.gather_rows + first;
        return bound;
    }
};

// Schedule selection of a tile launch: tall where the inputs are a multiple of its stage (refused for Schedule::tall
// otherwise); Schedule::staged or a narrower K: the incumbent over ceil(max tile rows / 128) row slices.
template <class Binding, class A>
inline hipError_t launch_tiles(const A& args, hipStream_t stream) {
    const bool tall = args.inputs % TallSchedule::kStageK == 0U;
    if (args.schedule == Schedule::tall && !tall) return hipErrorInvalidValue;
    if (args.schedule != Schedule::staged && tall)
        return gemm::gemm_packed_bf16<Binding, TallSchedule>(args, args.outputs, args.count, stream);
    const unsigned rows = args.max_tile_rows == 0U || args.max_tile_rows > kRowsPerTile ? kRowsPerTile : args.max_tile_rows;
    return gemm::gemm_packed_bf16<Binding>(args, args.outputs, args.count, stream, (rows + gemm::kBM - 1U) / gemm::kBM);
}

inline bool fits_indirect(const IndirectArgs& args) {
    return fits(args) && args.gather_rows && args.gather_from &&
           reinterpret_cast<std::uintptr_t>(args.gather_from) % 16U == 0U;
}

// Host admission checks the tile descriptors before upload with fits_tile().
// Invoke on a stream for a supported gfx11 device after device admission.
template <packed::MatrixFormat Format>
inline hipError_t launch(const Args& args, hipStream_t stream) {
    if (!fits(args)) return hipErrorInvalidValue;
    return launch_tiles<TileBinding<Format>>(args, stream);
}

// Same tiles; needs pair_rows (the down tiles of the same plan, row for row), the rebase pointers and outputs / 2 a
// multiple of 64 (one CTA per 64 gate columns: outputs / kBN CTAs).
template <packed::MatrixFormat Format>
inline hipError_t launch_activated(const Args& args, hipStream_t stream) {
    if (!fits(args) || !args.pair_rows || !args.rebase_from || !args.rebase_to || (args.outputs / 2U) % 64U != 0U ||
        !(args.gate_cap > 0.0F) || !(args.up_cap > 0.0F))
        return hipErrorInvalidValue;
    return launch_tiles<ActivatedTileBinding<Format>>(args, stream);
}

// launch / launch_activated with the input rows read in place (IndirectArgs): the same bits as on the gathered copy.
template <packed::MatrixFormat Format>
inline hipError_t launch_indirect(const IndirectArgs& args, hipStream_t stream) {
    if (!fits_indirect(args)) return hipErrorInvalidValue;
    return launch_tiles<IndirectRows<TileBinding<Format>>>(args, stream);
}

template <packed::MatrixFormat Format>
inline hipError_t launch_activated_indirect(const IndirectArgs& args, hipStream_t stream) {
    if (!fits_indirect(args) || !args.pair_rows || !args.rebase_from || !args.rebase_to || (args.outputs / 2U) % 64U != 0U ||
        !(args.gate_cap > 0.0F) || !(args.up_cap > 0.0F))
        return hipErrorInvalidValue;
    return launch_tiles<IndirectRows<ActivatedTileBinding<Format>>>(args, stream);
}

#endif

}  // namespace strix::moe_core::grouped_microscaled
