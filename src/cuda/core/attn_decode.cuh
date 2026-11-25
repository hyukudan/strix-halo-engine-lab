#pragma once

#include "attn_core/decode_plan.hpp"
#include "strix/cuda_core/attention.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace strix::cuda::core::attn {

inline constexpr std::uint32_t kNativeSplitHeadDim = 256;

struct DecodeInput {
    // Q is [row][head][dim], or [row][head][Q | gate] for sigmoid_interleaved.
    // K/V use the plan layout. Perform any rotary or value norm before append.
    const float* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* relative_bias = nullptr;  // [row][head][lag]
    const float* sinks = nullptr;          // [row][head]
    const float* gate = nullptr;           // separate gate for sigmoid_bf16 / sigmoid_strided
    std::uint32_t first_position = 0;
    // Optional device override, for replay. An out-of-range row leaves output
    // untouched. The caller must keep the live K/V tail valid on every launch.
    const std::uint32_t* device_position = nullptr;
    // Native split operand views. Default Q is packed (or interleaved according
    // to the plan); the default separate gate is packed independently of Q.
    cuda_core::GroupedView q_view{};
    cuda_core::GroupedView gate_view{};
    // Mapped positions and physical cache indices use the production ABI: the
    // caller guarantees they refer to resident, in-range rows. E4M3 requires
    // a device position or position map and a single non-tree cache.
    cuda_core::RowMap row_map{};
    const cuda_core::KvShadowE4m3* shadow = nullptr;
};

// Enqueue a plan made with kNativeSplitHeadDim. Output is contiguous
// [row][head][dim]. The caller owns scratch and keeps all buffers alive until
// the stream completes. Prepared MMA requires 16-byte aligned K/V planes.
cudaError_t decode(const DecodePlan& plan, const DecodeInput& input, float* output,
                   float* scratch, std::size_t scratch_floats, cudaStream_t stream);

// Position-canonical BF16 global-max decode, with packed Q and no output gate.
// RowMap may reorder rows, select independent cache planes, and remap a row's
// own key to a physical slot. The caller keeps the mapped K/V slots resident.
// An out-of-range device position leaves its output row untouched.
cudaError_t decode_position_split(const DecodePlan& plan, const DecodeInput& input, float* output,
                                  float* scratch, std::size_t scratch_floats, cudaStream_t stream);

namespace detail {

// Optional checked device replay or a by-value starting position. Mapped
// positions have their own resident-row contract.
struct PlanPosition {
    const std::uint32_t* device;
    std::uint32_t host, last_start;
};

cudaError_t decode_split(const cuda_core::AttentionDims& dims,
                         const float* q, const cuda_core::GroupedView& q_view,
                         const float* gate, const cuda_core::GroupedView& gate_view,
                         const __nv_bfloat16* k, const __nv_bfloat16* v, PlanPosition position,
                         std::uint32_t rows, float* out, float* scratch, cudaStream_t stream,
                         const cuda_core::RowMap& row_map);

cudaError_t decode_prepared(const cuda_core::AttentionDims& dims,
                            const cuda_core::PreparedBf16Flash& prepared, const float* q,
                            const __nv_bfloat16* k, const __nv_bfloat16* v, PlanPosition position,
                            std::uint32_t rows, const float* gate, float* out, cudaStream_t stream);

}  // namespace detail

}  // namespace strix::cuda::core::attn
