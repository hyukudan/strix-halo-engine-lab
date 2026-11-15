// GQA decode attention over a BF16 KV cache, rows 1..16 with causal masking inside the window. See attention.hpp.
#include "strix/cuda_core/attention.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

__global__ void kv_store_kernel(AttentionDims d, const float* k, const float* v, __nv_bfloat16* kc, __nv_bfloat16* vc,
                                std::uint32_t pos0) {
    const std::uint32_t r = blockIdx.y, h = blockIdx.x;
    const std::size_t src = (static_cast<std::size_t>(r) * d.kv_heads + h) * d.head_dim;
    const std::size_t dst = (static_cast<std::size_t>(h) * d.max_positions + pos0 + r) * d.head_dim;
    for (std::uint32_t i = threadIdx.x; i < d.head_dim; i += blockDim.x) {
        kc[dst + i] = __float2bfloat16_rn(k[src + i]);
        vc[dst + i] = __float2bfloat16_rn(v[src + i]);
    }
}

template <int D>
__global__ void __launch_bounds__(D) attn_kernel(AttentionDims dims, const float* q, GroupedView qv, const float* gate,
                                                  GroupedView gv, const __nv_bfloat16* kc, const __nv_bfloat16* vc,
                                                  std::uint32_t pos0, float* out, float* scratch) {
    constexpr int EPL = D / 32;  // dims per lane
    constexpr int NW = D / 32;
    __shared__ float qs[D];
    __shared__ float red[32];
    const std::uint32_t h = blockIdx.x, r = blockIdx.y;
    const std::uint32_t kvh = h / (dims.heads / dims.kv_heads);
    const std::uint32_t len = pos0 + r + 1;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    qs[t] = q[static_cast<std::size_t>(r) * qv.rs() + static_cast<std::size_t>(h) * qv.gs() + t];
    __syncthreads();
    float* sc = scratch + (static_cast<std::size_t>(r) * dims.heads + h) * dims.max_positions;
    const __nv_bfloat16* kb = kc + static_cast<std::size_t>(kvh) * dims.max_positions * D;
    const __nv_bfloat16* vb = vc + static_cast<std::size_t>(kvh) * dims.max_positions * D;

    // pass 1: scores, one warp per position
    for (std::uint32_t p = w; p < len; p += NW) {
        const __nv_bfloat16* kp = kb + static_cast<std::size_t>(p) * D + lane * EPL;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < EPL; ++j) dot = __fadd_rn(dot, __fmul_rn(qs[lane * EPL + j], __bfloat162float(kp[j])));
        dot = warp_sum_xor(dot);
        if (lane == 0) sc[p] = __fmul_rn(dot, dims.scale);
    }
    __syncthreads();
    // max
    float m = __int_as_float(0xff800000);
    for (std::uint32_t p = t; p < len; p += D) m = fmaxf(m, sc[p]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if (lane == 0) red[w] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < NW; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    // exp + denominator
    float part = 0.0f;
    for (std::uint32_t p = t; p < len; p += D) {
        const float e = cc_expf(__fsub_rn(sc[p], m));
        sc[p] = e;
        part = __fadd_rn(part, e);
    }
    part = warp_sum_xor(part);
    if (lane == 0) red[w] = part;
    __syncthreads();
    float denom = red[0];
    for (int i = 1; i < NW; ++i) denom = __fadd_rn(denom, red[i]);
    // pass 3: value accumulation, thread = output dim
    float acc = 0.0f;
    const __nv_bfloat16* vp = vb + t;
#pragma unroll 4
    for (std::uint32_t p = 0; p < len; ++p) acc = __fadd_rn(acc, __fmul_rn(sc[p], __bfloat162float(vp[static_cast<std::size_t>(p) * D])));
    float o = __fdiv_rn(acc, denom);
    if (gate != nullptr)
        o = __fmul_rn(o, cc_sigmoid(gate[static_cast<std::size_t>(r) * gv.rs() + static_cast<std::size_t>(h) * gv.gs() + t]));
    out[(static_cast<std::size_t>(r) * dims.heads + h) * D + t] = o;
}

}  // namespace

cudaError_t kv_store(const AttentionDims& d, const float* k, const float* v, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                     std::uint32_t pos0, std::uint32_t rows, cudaStream_t stream) {
    if (pos0 + rows > d.max_positions || rows == 0) return cudaErrorInvalidValue;
    kv_store_kernel<<<dim3(d.kv_heads, rows), 256, 0, stream>>>(d, k, v, k_cache, v_cache, pos0);
    return cudaGetLastError();
}

cudaError_t attention_decode(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate,
                             const GroupedView& gv, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                             std::uint32_t pos0, std::uint32_t rows, float* out, float* scratch, cudaStream_t stream) {
    if (d.head_dim != 256 || rows == 0 || rows > kMaxRows || d.kv_heads == 0 || d.heads % d.kv_heads != 0 ||
        pos0 + rows > d.max_positions)
        return cudaErrorInvalidValue;
    attn_kernel<256><<<dim3(d.heads, rows), 256, 0, stream>>>(d, q, qv, gate, gv, k_cache, v_cache, pos0, out, scratch);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
