#pragma once

#include "inkling2_mini_expert_hip.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "moe_core/grouped_rows.hpp"
#include "moe_core/expert_activation.hpp"

namespace strix::inkling2 {

struct MiniHipMoeBackend {
    static constexpr unsigned rows_per_tile = moe_core::grouped_microscaled::gemm::kBM;
    static constexpr auto fits = moe_core::grouped_microscaled::fits;
    static constexpr auto fits_tile = fits_mini_hip_expert_tile;

#if defined(__HIPCC__)
    using Status = hipError_t;
    using Stream = hipStream_t;
    static constexpr auto success = hipSuccess;
    static constexpr auto invalid_value = hipErrorInvalidValue;
    static constexpr auto gather = moe_core::grouped_rows::gather;
    static constexpr auto combine = moe_core::grouped_rows::combine;
    static constexpr auto activate = moe_core::expert_activation::launch;
    static constexpr auto project = enqueue_mini_hip_expert_projection;
#endif
};

}  // namespace strix::inkling2
