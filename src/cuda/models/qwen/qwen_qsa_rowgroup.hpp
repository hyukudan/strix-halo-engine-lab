#pragma once

#include <cstdint>

struct CUstream_st;
struct float2;

namespace strix::cuda::amdmath {
struct Tables;
}

namespace strix::cuda::qwen_qsa {
struct RowBuffers;
struct LayerWeights;
struct LayerState;
struct Workspace;

// Prepare the Qwen policy on the stream's current CUDA context before capture.
void prepare_attention_mma();

// The prepare call must have succeeded on this context.
void enqueue_attention_mma(const float* q_gate, std::uint32_t rows, std::uint64_t first_position,
    const std::uint32_t* selected, const std::uint16_t* kv, void* workspace, float* attention_output,
    CUstream_st* stream);

void layer_core_prepared(const RowBuffers& b, std::uint32_t rows, std::uint64_t first_position,
    const LayerWeights& w, const LayerState& state, const float2* cs, const amdmath::Tables& tables,
    Workspace& ws, CUstream_st* stream, bool mma_attention);

}  // namespace strix::cuda::qwen_qsa
