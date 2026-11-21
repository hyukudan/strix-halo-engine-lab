#pragma once

#include "core/f16_tile_gemm.hpp"
#include "moe_core/grouped_microscaled_rows.hpp"

namespace strix::moe_core::grouped_microscaled {

namespace gemm = strix::core::f16_tile_gemm;

inline bool fits(const Args& args) {
    return fits_args(args, gemm::kBK, gemm::kBN);
}

template <packed::MatrixFormat Format>
inline bool fits_tile(const TileRows& tile, unsigned inputs, unsigned outputs) {
    return fits_rows<Format>(tile, inputs, outputs, gemm::kBK, gemm::kBM);
}

#if defined(__HIPCC__)

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
        shape.a16 = rows.input;
        shape.a_row = rows.input_row;
        shape.y_row = rows.output_row;
        shape.rows = rows.rows;
        shape.K = args.inputs;
        shape.N = args.outputs;
        return {shape, {rows.output, rows.output_row, 0U}, {rows.weights}};
    }
};

// Host admission checks the tile descriptors before upload with fits_tile().
// Invoke on a stream for a supported gfx11 device after device admission.
template <packed::MatrixFormat Format>
inline hipError_t launch(const Args& args, hipStream_t stream) {
    if (!fits(args)) return hipErrorInvalidValue;
    return gemm::gemm_packed_bf16<TileBinding<Format>>(args, args.outputs, args.count, stream);
}

#endif

}  // namespace strix::moe_core::grouped_microscaled
