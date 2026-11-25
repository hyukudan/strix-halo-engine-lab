#pragma once

// CUDA operand ABI of the BF16 dense-row groups: 16-byte operands (vector loads), input width in multiples of eight
// floats, at most four projections per group, and the MMA3 tile (K32 x N16). The validation itself is the shared
// core::valid_bf16_row_group; this adapter only supplies the CUDA limits. Host-compilable (no CUDA headers).
#include "core/bf16_dense_row_plan.hpp"
#include "strix/core/rows.hpp"

namespace strix::cuda_core {

inline constexpr std::size_t kMaxBf16RowProjections = 4;

using Bf16DenseRowGroup = core::Bf16DenseRowGroup;

inline constexpr core::Bf16GroupLimits kBf16GroupLimits{16U, 8U, kMaxBf16RowProjections, core::rows::kBlockIn,
                                                        core::rows::kTileOut};

namespace detail {

inline bool bf16_mma3_tile(unsigned in, unsigned out) { return core::bf16_tile_eligible(in, out, kBf16GroupLimits); }

inline bool valid_bf16_dense_row_group(const Bf16DenseRowGroup& group, unsigned rows, bool allow_round_bf16) {
    return core::valid_bf16_row_group(group, rows, allow_round_bf16, kBf16GroupLimits);
}

}  // namespace detail

}  // namespace strix::cuda_core
