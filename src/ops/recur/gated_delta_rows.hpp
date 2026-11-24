#pragma once
// F9 `recur` HIP home: the gated delta step over the rows of one window in one launch (gated_delta_policy.hpp has the
// formula and the policies; gated_delta_step.hpp the single-row form). The retained state tile stays in registers from row
// to row; row r runs exactly the operations of the single-row step on the qkv / beta / gate / core of row r, in the same
// order, so the bits equal `rows` separate launches. `state` is loaded once before any store and the final tile goes to
// `state_out` (== state for in place). The reduction buffer's physical index is the form F (gdn_recurrent_lds_index.hpp:
// same elements, pairs and order for every form). gated_delta_rows_body.inc is the former recurrent_rows_kernel body.
#include <hip/hip_runtime.h>

#include "ops/recur/gated_delta_policy.hpp"
#include "strix/gdn_recurrent_lds_index.hpp"

#include <cstdint>

namespace strix::ops::recur {

namespace gdn_recurrent_lds = ::strix::gdn_recurrent_lds;

// Launch: grid value_heads * (head_dim / kStepValueTile), block kStepThreads, head_dim == kStepThreads.
template <gdn_recurrent_lds::Form F, class Policy>
__global__ __launch_bounds__(kStepThreads) void gated_delta_rows_kernel(
    const float* qkv, const float* beta, const float* gate, const float* state, float* state_out, float* core,
    std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t head_dim, std::uint32_t rows,
    std::uint64_t qkv_stride, std::uint64_t core_stride, Policy policy) {
#include "ops/recur/gated_delta_rows_body.inc"
}

}  // namespace strix::ops::recur
