// Gated DeltaNet decode: causal depthwise conv (k=4) + SiLU, l2norm q/k, delta rule with sequential rows, gated RMSNorm.
#include "strix/cuda_core/gdn.hpp"

#include <cmath>

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

// One thread per channel; rows sequential. tap order oldest first: ((0 + w0 s0) + w1 s1) + w2 s2) + w3 x.
__global__ void conv_silu_kernel(const float* __restrict__ w, float* conv_state, float* qkv, std::uint32_t channels,
                                 std::uint32_t rows) {
    const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float s0 = conv_state[c * 3], s1 = conv_state[c * 3 + 1], s2 = conv_state[c * 3 + 2];
    for (std::uint32_t r = 0; r < rows; ++r) {
        float* p = qkv + static_cast<std::size_t>(r) * channels + c;
        const float x = *p;
        float sum = __fmul_rn(w0, s0);
        sum = __fadd_rn(sum, __fmul_rn(w1, s1));
        sum = __fadd_rn(sum, __fmul_rn(w2, s2));
        sum = __fadd_rn(sum, __fmul_rn(w3, x));
        *p = cc_silu(sum);
        s0 = s1;
        s1 = s2;
        s2 = x;
    }
    conv_state[c * 3] = s0;
    conv_state[c * 3 + 1] = s1;
    conv_state[c * 3 + 2] = s2;
}

// Block = one warp = 32 value columns of one value head; the warp holds S[:, cols] (128 floats per lane) in registers.
__global__ void __launch_bounds__(32) delta_kernel(float* state, const float* qkv, const float* a_raw, const float* b_raw,
                                                   const float* ssm_a, const float* dt_bias, float* o, GdnDims d,
                                                   std::uint32_t rows) {
    constexpr int K = 128;
    __shared__ float ks[K], qs[K];
    const std::uint32_t h = blockIdx.y, tile = blockIdx.x, lane = threadIdx.x;
    const std::uint32_t kh = h % d.key_heads;
    const std::uint32_t col = tile * 32 + lane;
    const std::uint32_t channels = d.conv_channels();
    float* sp = state + static_cast<std::size_t>(h) * K * d.value_dim + col;
    float S[K];
#pragma unroll
    for (int k = 0; k < K; ++k) S[k] = sp[static_cast<std::size_t>(k) * d.value_dim];
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* row = qkv + static_cast<std::size_t>(r) * channels;
#pragma unroll
        for (int i = 0; i < K / 32; ++i) {
            qs[lane + 32 * i] = row[kh * K + lane + 32 * i];
            ks[lane + 32 * i] = row[d.qk_width() + kh * K + lane + 32 * i];
        }
        __syncwarp();
        const float braw = b_raw[static_cast<std::size_t>(r) * d.value_heads + h];
        const float araw = a_raw[static_cast<std::size_t>(r) * d.value_heads + h];
        const float beta = cc_sigmoid(braw);
        const float xin = __fadd_rn(araw, dt_bias[h]);
        const float sp_ = xin > 20.0f ? xin : cc_log1pf(cc_expf(xin));
        const float g = __fmul_rn(ssm_a[h], sp_);
        const float decay = cc_expf(g);
        const float vcol = row[2 * d.qk_width() + h * d.value_dim + col];
        float kv = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            S[k] = __fmul_rn(S[k], decay);
            kv = __fadd_rn(kv, __fmul_rn(S[k], ks[k]));
        }
        const float dl = __fmul_rn(__fsub_rn(vcol, kv), beta);
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            S[k] = __fadd_rn(S[k], __fmul_rn(ks[k], dl));
            acc = __fadd_rn(acc, __fmul_rn(S[k], qs[k]));
        }
        o[(static_cast<std::size_t>(r) * d.value_heads + h) * d.value_dim + col] = acc;
        __syncwarp();
    }
#pragma unroll
    for (int k = 0; k < K; ++k) sp[static_cast<std::size_t>(k) * d.value_dim] = S[k];
}

}  // namespace

cudaError_t gdn_decode(const GdnDims& d, const GdnLayer& layer, const GdnState& state, float* qkv, const float* z,
                       const float* a_raw, const float* b_raw, float* out, std::uint32_t rows, float eps,
                       cudaStream_t stream) {
    if (d.key_dim != 128 || d.value_dim != 128 || d.conv_kernel != 4 || rows == 0 || rows > kMaxRows) return cudaErrorInvalidValue;
    const std::uint32_t C = d.conv_channels();
    conv_silu_kernel<<<(C + 127) / 128, 128, 0, stream>>>(layer.conv_w, state.conv, qkv, C, rows);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    GroupedView qk{rows, d.key_heads, d.key_dim, C, d.key_dim};
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.key_dim)));
    if (auto e = l2norm(qkv, qk, qkv, qk, 1e-6f, qscale, stream); e != cudaSuccess) return e;
    if (auto e = l2norm(qkv + d.qk_width(), qk, qkv + d.qk_width(), qk, 1e-6f, 1.0f, stream); e != cudaSuccess) return e;
    delta_kernel<<<dim3(d.value_dim / 32, d.value_heads), 32, 0, stream>>>(state.delta, qkv, a_raw, b_raw, layer.ssm_a,
                                                                           layer.dt_bias, out, d, rows);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    GroupedView ov{rows, d.value_heads, d.value_dim, 0, 0};
    return gated_rmsnorm(out, ov, layer.norm_gain, z, ov, GateKind::swish, out, ov, eps, stream);
}

}  // namespace strix::cuda_core
