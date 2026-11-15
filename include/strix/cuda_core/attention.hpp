#pragma once
#include <cuda_bf16.h>

#include "strix/cuda_core/types.hpp"

namespace strix::cuda_core {

struct AttentionDims {
    std::uint32_t heads = 0, kv_heads = 0, head_dim = 256;  // GQA: q head h reads kv head h / (heads / kv_heads)
    float scale = 1.0f;
    std::uint32_t max_positions = 0;  // KV cache capacity per kv head
};

// KV cache BF16, layout [kv_head][max_positions][head_dim] (one buffer each for K and V).
// pos0 is a DEVICE scalar (graph capture: the step's start position changes without re-capture).
// Stores rows k/v ([rows][kv_heads][head_dim] FP32, contiguous) at positions pos0 + r, rounding RNE to BF16.
cudaError_t kv_store(const AttentionDims& d, const float* k, const float* v, __nv_bfloat16* k_cache,
                     __nv_bfloat16* v_cache, const std::uint32_t* pos0_dev, std::uint32_t rows, cudaStream_t stream);

// Floats of scratch attention_decode needs: rows * heads * max_positions.
inline std::size_t attention_scratch_floats(const AttentionDims& d, std::uint32_t rows) {
    return static_cast<std::size_t>(rows) * d.heads * d.max_positions;
}

// Row r (position pos0 + r) attends to cache positions 0 .. pos0 + r (causal inside the window; the window's own K/V
// must have been stored first). q: FP32 view (rows x heads x head_dim). gate (optional, same shape, its own strides):
// out *= sigmoid(gate). ORDER per (row, head): score_p = fl(dot) * scale with the dot over the dims as 32 lanes x
// head_dim/32 consecutive dims (sequential per lane, then xor butterfly 16..1); m = max; e_p = expf(s_p - m);
// denominator = thread t adds e[t + head_dim i] ascending, then the warp-order tree of norms.hpp;
// out_d = (sum_p e_p * v[p][d], p ascending) / denom. Independent of rows and launch shape. head_dim 256 instantiated.
cudaError_t attention_decode(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate,
                             const GroupedView& gv, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                             const std::uint32_t* pos0_dev, std::uint32_t rows, float* out /*[rows][heads][head_dim]*/,
                             float* scratch, cudaStream_t stream);

}  // namespace strix::cuda_core
