#pragma once

#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_schedule.hpp"
#include "moe_core/grouped_microscaled.hpp"

namespace strix::inkling2 {

// The host validates each descriptor before upload. All tiles in the launch
// belong to one physical layer and one projection.
inline bool fits_mini_hip_expert_tile(unsigned physical_layer, MiniExpertProjection projection,
                                     const moe_core::grouped_microscaled::TileRows& tile) {
    namespace grouped = moe_core::grouped_microscaled;
    using weight::microscaled::MatrixFormat;
    const auto geometry = mini_expert_geometry(projection);
    if (physical_layer >= kPhysicalLayers || geometry.inputs == 0 || tile.weights.row_order != mini_expert_row_order(projection)) return false;
    if (mini_expert_format(physical_layer) == MatrixFormat::nvfp4) {
        return grouped::fits_tile<MatrixFormat::nvfp4>(tile, geometry.inputs, geometry.outputs);
    }
    return grouped::fits_tile<MatrixFormat::mxfp8>(tile, geometry.inputs, geometry.outputs);
}

#if defined(__HIPCC__)
inline hipError_t enqueue_mini_hip_expert_projection(unsigned physical_layer, MiniExpertProjection projection,
                                                     const moe_core::grouped_microscaled::TileRows* device_tiles,
                                                     unsigned count, hipStream_t stream) {
    namespace grouped = moe_core::grouped_microscaled;
    using weight::microscaled::MatrixFormat;
    const auto geometry = mini_expert_geometry(projection);
    if (physical_layer >= kPhysicalLayers || geometry.inputs == 0) return hipErrorInvalidValue;
    const grouped::Args args{device_tiles, count, geometry.inputs, geometry.outputs};
    if (mini_expert_format(physical_layer) == MatrixFormat::nvfp4) return grouped::launch<MatrixFormat::nvfp4>(args, stream);
    return grouped::launch<MatrixFormat::mxfp8>(args, stream);
}
#endif

}  // namespace strix::inkling2
