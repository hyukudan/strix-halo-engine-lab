#pragma once
// F9 `recur` HIP home: the gated delta decode step for one row (gated_delta_policy.hpp has the formula and the policies).
// One block per (value head, tile of kStepValueTile value columns), kStepThreads lanes, lane = key index; the state tile
// is read once into registers, the two key reductions are LDS halving trees over all lanes for the tile's columns.
// gated_delta_step_body.inc is the former recurrent_step_kernel body (src/hip_gdn_launcher.hip), lifted verbatim; the
// four policy hooks replace its gate, lane products and update. ScalarGate keeps that kernel's expressions, so the GDN wrapper
// compiles to the same machine code; PerKeyChannelGate is the KDA step (per-key-channel decay).
#include <hip/hip_runtime.h>

#include "attn_core/recurrent_state.hpp"
#include "ops/recur/gated_delta_policy.hpp"
#include "strix/recurrent_state_type.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ops::recur {

namespace rstate = ::strix::attn_core::rstate;

// Launch: grid value_heads * (head_dim / kStepValueTile), block kStepThreads, head_dim == kStepThreads.
template <RecurrentStateType S, class Policy>
__global__ __launch_bounds__(kStepThreads) void gated_delta_step_kernel(
    const float* qkv, const float* beta, const float* gate, float* state_raw, float* core, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t head_dim, std::uint32_t* graph_status, Policy policy) {
    // graph_status != nullptr: the step is skipped when the graph's sticky status is already set (GDN graph validation).
#include "ops/recur/gated_delta_step_body.inc"
}

}  // namespace strix::ops::recur
