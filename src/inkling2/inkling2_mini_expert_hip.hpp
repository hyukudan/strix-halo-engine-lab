#pragma once

#include "inkling2_mini_moe_dispatch.hpp"
#include "moe_core/grouped_dispatch_hip.hpp"

namespace strix::inkling2 {

using MiniHipMoeBackend = mini_grouped::HipBackend;

inline constexpr auto fits_mini_hip_expert_tile = mini_grouped::fits_tile<MiniMoeConfig, MiniHipMoeBackend>;

#if defined(__HIPCC__)
inline hipError_t enqueue_mini_hip_expert_projection(
    unsigned layer, MiniExpertProjection projection, const mini_grouped::TileRows* tiles,
    unsigned count, hipStream_t stream, const unsigned* tile_count = nullptr, const unsigned* failure = nullptr) {
    const auto geometry = mini_expert_geometry(projection);
    if (layer >= kPhysicalLayers || geometry.inputs == 0U) return hipErrorInvalidValue;
    return mini_grouped::project<MiniMoeConfig, MiniHipMoeBackend>(
        layer, {tiles, count, geometry.inputs, geometry.outputs, tile_count, failure}, stream);
}
#endif

}  // namespace strix::inkling2
