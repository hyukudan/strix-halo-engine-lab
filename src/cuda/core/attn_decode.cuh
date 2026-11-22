#pragma once

#include "attn_plan.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace strix::cuda::core::attn {

struct DecodeInput {
    // Q is [row][head][dim], or [row][head][Q | gate] for sigmoid_interleaved.
    // K/V use the plan layout. Perform any rotary or value norm before append.
    const float* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* relative_bias = nullptr;  // [row][head][lag]
    const float* sinks = nullptr;          // [row][head]
    const float* gate = nullptr;           // [row][head][dim] for sigmoid_bf16
    std::uint32_t first_position = 0;
    // Optional device override, for replay. An out-of-range row leaves output
    // untouched. The caller must keep the live K/V tail valid on every launch.
    const std::uint32_t* device_position = nullptr;
};

// Enqueue a plan returned by make_decode_plan. Output is contiguous
// [row][head][dim]. The caller owns scratch and keeps all buffers alive until
// the stream completes. Prepared MMA requires 16-byte aligned K/V planes.
cudaError_t decode(const DecodePlan& plan, const DecodeInput& input, float* output,
                   float* scratch, std::size_t scratch_floats, cudaStream_t stream);

}  // namespace strix::cuda::core::attn
