#pragma once

#include "cuda/core/grouped_microscaled.cuh"
#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "inkling2_mini_schedule.hpp"
#include "moe_core/expert_activation.hpp"
#include "moe_core/grouped_rows.hpp"

namespace strix::inkling2 {

namespace mini_cuda_moe_detail {

namespace grouped = cuda::core::grouped_microscaled;
using weight::microscaled::MatrixFormat;

inline bool fits_tile(unsigned physical_layer, MiniExpertProjection projection, const grouped::TileRows& tile) {
    const auto geometry = mini_expert_geometry(projection);
    if (physical_layer >= kPhysicalLayers || geometry.inputs == 0 || tile.weights.row_order != mini_expert_row_order(projection)) return false;
    if (mini_expert_format(physical_layer) == MatrixFormat::nvfp4) {
        return grouped::fits_tile<MatrixFormat::nvfp4>(tile, geometry.inputs, geometry.outputs);
    }
    return grouped::fits_tile<MatrixFormat::mxfp8>(tile, geometry.inputs, geometry.outputs);
}

#if defined(__CUDACC__) && !defined(__HIPCC__)
inline cudaError_t project(unsigned physical_layer, MiniExpertProjection projection,
                           const grouped::TileRows* tiles, unsigned count, cudaStream_t stream) {
    const auto geometry = mini_expert_geometry(projection);
    if (physical_layer >= kPhysicalLayers || geometry.inputs == 0) return cudaErrorInvalidValue;
    const grouped::Args args{tiles, count, geometry.inputs, geometry.outputs};
    if (mini_expert_format(physical_layer) == MatrixFormat::nvfp4) return grouped::launch<MatrixFormat::nvfp4>(args, stream);
    return grouped::launch<MatrixFormat::mxfp8>(args, stream);
}
#endif

}  // namespace mini_cuda_moe_detail

struct MiniCudaMoeBackend {
    static constexpr unsigned rows_per_tile = mini_cuda_moe_detail::grouped::kRowsPerTile;
    static constexpr auto fits = mini_cuda_moe_detail::grouped::fits;
    static constexpr auto fits_tile = mini_cuda_moe_detail::fits_tile;

#if defined(__CUDACC__) && !defined(__HIPCC__)
    using Status = cudaError_t;
    using Stream = cudaStream_t;
    static constexpr auto success = cudaSuccess;
    static constexpr auto invalid_value = cudaErrorInvalidValue;
    static constexpr auto gather = moe_core::grouped_rows::gather;
    static constexpr auto combine = moe_core::grouped_rows::combine;
    static constexpr auto activate = moe_core::expert_activation::launch;
    static constexpr auto project = mini_cuda_moe_detail::project;
#endif
};

}  // namespace strix::inkling2
