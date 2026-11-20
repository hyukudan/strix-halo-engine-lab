#pragma once

// GLM adapter data for the shared stored-INT4 layer policy (src/moe_core/int4_layer_policy.hpp): the defaults of
// STRIX_GLM_FAST_I4_EXL3_LAYERS and STRIX_GLM_FAST_I4_SCOPE. Host-only, so the host tests check the defaults the
// engine uses.
//
// Layer default 0-24: the routed gate/up experts of MoE ordinals 25-41 (17 layers) are INT4. FAST gate v2 against
// the exact reference (4 windows x 255 rows, one-FP16-ulp null, FAST prefill STRIX_GLM_PF_WMMA=1 underneath):
// ordinals 15-41 (27 layers) fail two windows on KL beyond 1.5 x the null and drop top-1 to 0.898; ordinals 25-41
// pass every window that FAST alone passes. The pooled absolute bars are not met by the exact null either
// (KL 0.141), so neither set is a qualified FAST lever (see docs/toggles.md).
namespace strix::glm::engine::fast_i4 {

inline constexpr const char* kExl3LayersDefault = "0-24";
inline constexpr const char* kScopeDefault = "gate_up";

}  // namespace strix::glm::engine::fast_i4
