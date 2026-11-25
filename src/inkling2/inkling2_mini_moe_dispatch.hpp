#pragma once

#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_schedule.hpp"
#include "moe_core/grouped_dispatch.hpp"

namespace strix::inkling2 {

namespace mini_grouped = moe_core::grouped_dispatch;

struct MiniMoeConfig {
    using Projection = MiniExpertProjection;
    static constexpr unsigned hidden = kHiddenWidth;
    static constexpr unsigned intermediate = kExpertWidth;
    static constexpr unsigned experts = kExperts;
    static constexpr unsigned top_k = kExpertsPerToken;
    static constexpr unsigned layers = kPhysicalLayers;
    static constexpr unsigned max_rows = kMaxContext;
    static constexpr auto format = mini_expert_format;
    static constexpr auto row_order = mini_expert_row_order;

#if defined(__HIPCC__) || defined(__CUDACC__)
    struct Caps {
        static constexpr float gate = kMlpGateSoftcap;
        static constexpr float up = kMlpUpSoftcap;
    };
    using Activation = mini_grouped::SoftcappedSwigluBf16<Caps>;
    using Combine = mini_grouped::Bf16Combine;
#endif
};

// The narrow expert image (lever moe.experts.mxfp8-nvfp4@1): the checkpoint's MXFP8 layers are stored as NVFP4 at load
// (weight/formats/microscaled_encode.hpp transcode_slab_to_nvfp4), so every physical layer of that image reads as NVFP4.
// The checkpoint configuration above stays the one every other load dispatches with.
inline constexpr weight::microscaled::MatrixFormat mini_narrow_expert_format(unsigned) {
    return weight::microscaled::MatrixFormat::nvfp4;
}
struct MiniMoeNarrowConfig : MiniMoeConfig {
    static constexpr auto format = mini_narrow_expert_format;
};

using MiniMoeWorkspace = mini_grouped::Workspace;
using MiniMoeExpert = mini_grouped::ExpertWeights;
using MiniMoeDevicePlan = mini_grouped::DevicePlan;
using MiniMoeResidentRouteBuffers = mini_grouped::RouteBuffers;

inline std::size_t mini_moe_workspace_elements(unsigned rows) {
    static_assert(2U * kExpertWidth == kHiddenWidth);
    return mini_grouped::workspace_elements<MiniMoeConfig>(rows).fp32;
}

// BF16 activation rows of the gate-up epilogue form (HIP).
inline std::size_t mini_moe_activation_elements(unsigned rows) {
    return mini_grouped::workspace_elements<MiniMoeConfig>(rows).activation;
}

inline std::size_t mini_moe_nvfp4_image_bytes(unsigned rows, moe_core::grouped_w4a4::Activations activations) {
    return mini_grouped::workspace_elements<MiniMoeConfig>(rows, activations).nvfp4;
}

template<class Backend>
inline constexpr auto prepare_mini_moe = mini_grouped::prepare<MiniMoeConfig, Backend>;

template<class Backend>
inline constexpr auto enqueue_mini_moe = mini_grouped::enqueue<MiniMoeConfig, Backend>;

}  // namespace strix::inkling2
