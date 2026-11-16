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
                     __nv_bfloat16* v_cache, const std::uint32_t* pos0_dev, std::uint32_t rows, cudaStream_t stream, const RowMap& rm = {});

constexpr std::uint32_t kAttnChunk = 256;  // positions per reduction chunk
[[nodiscard]] __host__ __device__ inline std::uint32_t attn_max_chunks(const AttentionDims& d) { return (d.max_positions + kAttnChunk - 1) / kAttnChunk; }

// Floats of scratch attention_decode needs for `rows`: scores + chunk max + chunk denominators + chunk value partials.
inline std::size_t attention_scratch_floats(const AttentionDims& d, std::uint32_t rows) {
    const std::size_t rh = static_cast<std::size_t>(rows) * d.heads, mc = attn_max_chunks(d);
    return rh * d.max_positions + 2 * rh * mc + rh * mc * d.head_dim;
}

// Row r (position pos0 + r) attends to cache positions 0 .. pos0 + r (causal inside the window; the window's own K/V
// must have been stored first). q: FP32 view (rows x heads x head_dim). gate (optional, same shape, its own strides):
// out *= sigmoid(gate). ORDER per (row, head), positions split into 256-position chunks by absolute position:
// score_p = fl(dot) * scale, dot = lane owns 8 consecutive dims sequentially then xor butterfly 16..1; m = max over all positions
// (exact); e_p = expf(s_p - m); chunk denominator = block tree (thread t holds position chunk*256 + t, zero past the length) of norms.hpp;
// chunk value partial_d = sum over the chunk positions ascending of fl(e_p * v[p][d]); denom = chunk denominators added ascending,
// acc = chunk partials added ascending, out = acc / denom. Independent of rows and launch shape. head_dim 256, group <= 8.
cudaError_t attention_decode(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate,
                             const GroupedView& gv, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                             const std::uint32_t* pos0_dev, std::uint32_t rows, float* out /*[rows][heads][head_dim]*/,
                             float* scratch, cudaStream_t stream, const RowMap& rm = {});

// PREFILL attention (FAST class, NOT bitwise equal to attention_decode): causal flash attention over the BF16 KV cache for a chunk of
// `rows` query rows at positions pos0 .. pos0 + rows - 1 (the chunk's own K/V must already be stored by kv_store). Q is rounded to bf16,
// scores and the online softmax are FP32, P is rounded to bf16 for the PV mma (FP32 accumulate); out = o * sigmoid(gate) like
// attention_decode. head_dim 256, GQA group <= 8. q/gate/out use the same grouped views as attention_decode.
cudaError_t attention_prefill_flash(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                                    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, std::uint32_t pos0, std::uint32_t rows,
                                    float* out, cudaStream_t stream);
// p[i] = base + step * i for i < n (small device helper used to address attention windows).
cudaError_t iota_u32(std::uint32_t* p, std::uint32_t n, std::uint32_t base, std::uint32_t step, cudaStream_t stream);

}  // namespace strix::cuda_core
