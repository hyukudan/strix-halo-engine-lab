// Chunked gated-delta-rule prefill, FAST contract gdn-chunk16-f32-v1 (see include/strix/cuda_core/gdn_chunk.hpp for the algebra).
// K1 chunk_prep_kernel  grid (chunks, value heads): A, T = (I + A)^-1, W, U~, P, G of one chunk; all chunks in parallel.
// K2 chunk_scan_kernel  grid (value heads, 128 / 32 value tiles) = 192 blocks: the chunks in order, state tile [128 k][32 v] on chip.
// K3 chunk_norm_kernel  grid (value heads, rows): gated RMSNorm of o (same formula as the serial kernel).
// FP32 FFMA everywhere (no operand rounding); the order of the sums differs from the serial recurrence (FAST class).
#include "strix/cuda_core/gdn.hpp"
#include "strix/cuda_core/gdn_chunk.hpp"

#include "../common.cuh"
#include "gated_norm.cuh"

namespace strix::cuda_core {
namespace {

namespace gc = gdn_chunk;
constexpr int C = static_cast<int>(gc::kChunk);
constexpr int D = static_cast<int>(gc::kDim);
constexpr int VT = static_cast<int>(gc::kVTile);
using L = gc::Layout;

__device__ __forceinline__ float softplus_g(float x) { return x > 20.0f ? x : cc_log1pf(cc_expf(x)); }

// ---- K1: one (chunk, value head) per block, thread t = key/value column t.
__global__ void __launch_bounds__(128) chunk_prep_kernel(const float* __restrict__ qkvc, const float* __restrict__ a_raw,
                                                         const float* __restrict__ b_raw, const float* __restrict__ ssm_a,
                                                         const float* __restrict__ dt_bias, float* __restrict__ ws, GdnDims d,
                                                         std::uint32_t rows) {
    __shared__ float ks[C][D + 1], qs[C][D + 1], vs[C][D];  // +1: conflict-free row-pair dots
    __shared__ float A[C][C], T[C][C + 1];
    __shared__ float G[C], beta[C], kb[C];
    const std::uint32_t c = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const std::uint32_t kh = h % d.key_heads, ch = d.conv_channels(), vh = d.value_heads;
#pragma unroll 4
    for (int i = 0; i < C; ++i) {
        const std::uint32_t r = c * C + i;
        float q = 0.f, k = 0.f, v = 0.f;
        if (r < rows) {
            const float* row = qkvc + static_cast<std::size_t>(r) * ch;
            q = row[kh * D + t];
            k = row[d.qk_width() + kh * D + t];
            v = row[2 * d.qk_width() + h * D + t];
        }
        qs[i][t] = q;
        ks[i][t] = k;
        vs[i][t] = v;
    }
    if (t < C) {
        const std::uint32_t r = c * C + t;
        float g = 0.f, b = 0.f;  // padding rows: decay 1, beta 0
        if (r < rows) {
            g = __fmul_rn(ssm_a[h], softplus_g(__fadd_rn(a_raw[static_cast<std::size_t>(r) * vh + h], dt_bias[h])));
            b = cc_sigmoid(b_raw[static_cast<std::size_t>(r) * vh + h]);
        }
        G[t] = g;
        beta[t] = b;
    }
    __syncthreads();
    if (t == 0) {
        float s = 0.f;
        for (int i = 0; i < C; ++i) G[i] = s = s + G[i];
    }
    __syncthreads();
    float* blk = ws + gc::block_offset(c, h, vh);
    {  // Gram entries: thread -> row i, columns j0, j0 + 1
        const int i = static_cast<int>(t >> 3), j0 = static_cast<int>(t & 7) * 2;
        float kk0 = 0.f, kk1 = 0.f, qk0 = 0.f, qk1 = 0.f;
#pragma unroll 8
        for (int e = 0; e < D; ++e) {
            const float ki = ks[i][e], qi = qs[i][e], a0 = ks[j0][e], a1 = ks[j0 + 1][e];
            kk0 = fmaf(ki, a0, kk0);
            kk1 = fmaf(ki, a1, kk1);
            qk0 = fmaf(qi, a0, qk0);
            qk1 = fmaf(qi, a1, qk1);
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int j = j0 + q;
            const float dec = j <= i ? cc_expf(G[i] - G[j]) : 0.f;  // <= 1 on and below the diagonal
            A[i][j] = j < i ? beta[i] * dec * (q ? kk1 : kk0) : 0.f;
            blk[L::pt + static_cast<std::size_t>(j) * C + i] = dec * (q ? qk1 : qk0);  // Pt[j][i] = P[i][j], 0 above the diagonal
        }
    }
    if (t < C) kb[t] = beta[t] * cc_expf(G[t]);
    __syncthreads();
    if (t < 32) {  // T = (I + A)^-1, forward substitution by rows (warp 0; lanes >= C idle)
        for (int i = 0; i < C; ++i) {
            if (t < C) {
                float s = t == static_cast<std::uint32_t>(i) ? 1.f : 0.f;
                for (int j = 0; j < i; ++j) s = fmaf(-A[i][j], T[j][t], s);
                T[i][t] = s;
            }
            __syncwarp();
        }
    }
    __syncthreads();
    float wcol[C];
#pragma unroll
    for (int i = 0; i < C; ++i) {  // W[i][t] = sum_j T[i][j] kb[j] k_j[t], U~[i][t] = sum_j T[i][j] beta[j] v_j[t] (j <= i)
        float w = 0.f, u = 0.f;
        for (int j = 0; j <= i; ++j) {
            const float tij = T[i][j];
            w = fmaf(tij * kb[j], ks[j][t], w);
            u = fmaf(tij * beta[j], vs[j][t], u);
        }
        wcol[i] = w;
        blk[L::u + static_cast<std::size_t>(i) * D + t] = u;
    }
    float4* wt = reinterpret_cast<float4*>(blk + L::wt + static_cast<std::size_t>(t) * C);  // Wt[t][0..15]
#pragma unroll
    for (int q = 0; q < C / 4; ++q) wt[q] = make_float4(wcol[4 * q], wcol[4 * q + 1], wcol[4 * q + 2], wcol[4 * q + 3]);
    if (t < C) blk[L::g + t] = G[t];
}

// ---- K2: one (value head, 32-column value tile) per block; chunks in order. Shared memory 45 KB (two blocks per SM fit).
struct ScanSmem {
    float S[D][VT];       // state tile, row = key
    float Wt[D][C];       // W^T of the chunk; after step 1 it holds the upper half's partial sums [64][16]
    float Qt[D][C + 4];   // q^T (+4: float4-aligned, conflict-free transposing stores)
    float Ks[C][D];       // k rows
    float Us[C][VT];      // U~ tile -> U -> U scaled by exp(G_last - G_i)
    float Pt[C][C];
    float Gs[C];
};
struct ScanPre {  // next chunk's operands, loaded to registers during step 3
    float4 wt[4];
    float q[C], k[C];
    float4 u;
    float2 p;
    float g;
};

__global__ void __launch_bounds__(128) chunk_scan_kernel(const float* __restrict__ qkvc, const float* __restrict__ ws, float* base,
                                                         float* __restrict__ out, GdnDims d, std::uint32_t rows,
                                                         const std::uint32_t* parity_dev) {
    __shared__ __align__(16) ScanSmem sm;
    const std::uint32_t h = blockIdx.x, v0 = blockIdx.y * VT, t = threadIdx.x;
    const std::uint32_t kh = h % d.key_heads, ch = d.conv_channels(), vh = d.value_heads, nch = gc::chunks(rows);
    const std::size_t side = static_cast<std::size_t>(vh) * D * D;
    const std::uint32_t par = *parity_dev;
    const float* s_in = base + par * side + static_cast<std::size_t>(h) * D * D;
    float* s_out = base + (1u - par) * side + static_cast<std::size_t>(h) * D * D;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const int idx = q * 128 + static_cast<int>(t), k = idx >> 3, v4 = (idx & 7) * 4;
        *reinterpret_cast<float4*>(&sm.S[k][v4]) = *reinterpret_cast<const float4*>(s_in + static_cast<std::size_t>(k) * D + v0 + v4);
    }
    ScanPre pre;
    auto fetch = [&](std::uint32_t c) {
        const float* blk = ws + gc::block_offset(c, h, vh);
        const float4* wt = reinterpret_cast<const float4*>(blk + L::wt);
#pragma unroll
        for (int q = 0; q < 4; ++q) pre.wt[q] = wt[q * 128 + t];
#pragma unroll
        for (int i = 0; i < C; ++i) {
            const std::uint32_t r = c * C + i;
            const float* row = qkvc + static_cast<std::size_t>(r < rows ? r : 0) * ch;
            pre.q[i] = r < rows ? row[kh * D + t] : 0.f;
            pre.k[i] = r < rows ? row[d.qk_width() + kh * D + t] : 0.f;
        }
        pre.u = *reinterpret_cast<const float4*>(blk + L::u + static_cast<std::size_t>(t >> 3) * D + v0 + (t & 7) * 4);
        pre.p = *reinterpret_cast<const float2*>(blk + L::pt + 2 * t);
        pre.g = t < C ? blk[L::g + t] : 0.f;
    };
    auto store = [&]() {
        float4* wt = reinterpret_cast<float4*>(&sm.Wt[0][0]);
#pragma unroll
        for (int q = 0; q < 4; ++q) wt[q * 128 + t] = pre.wt[q];
#pragma unroll
        for (int q = 0; q < C / 4; ++q)
            *reinterpret_cast<float4*>(&sm.Qt[t][4 * q]) = make_float4(pre.q[4 * q], pre.q[4 * q + 1], pre.q[4 * q + 2], pre.q[4 * q + 3]);
#pragma unroll
        for (int i = 0; i < C; ++i) sm.Ks[i][t] = pre.k[i];
        *reinterpret_cast<float4*>(&sm.Us[t >> 3][(t & 7) * 4]) = pre.u;
        *reinterpret_cast<float2*>(&sm.Pt[0][0] + 2 * t) = pre.p;
        if (t < C) sm.Gs[t] = pre.g;
    };
    fetch(0);
    store();
    __syncthreads();
    const int hf = static_cast<int>(t >> 6), ig = static_cast<int>((t & 63) >> 4), cg = static_cast<int>(t & 15);
    for (std::uint32_t c = 0; c < nch; ++c) {
        // step 1: W S and Q S over this half's 64 keys (rows 4ig..4ig+3, columns 2cg, 2cg+1)
        float aw[4][2] = {}, aq[4][2] = {};
#pragma unroll 4
        for (int kk = 0; kk < D / 2; ++kk) {
            const int k = hf * (D / 2) + kk;
            const float4 w = *reinterpret_cast<const float4*>(&sm.Wt[k][4 * ig]);
            const float4 q = *reinterpret_cast<const float4*>(&sm.Qt[k][4 * ig]);
            const float2 s = *reinterpret_cast<const float2*>(&sm.S[k][2 * cg]);
            const float wv[4] = {w.x, w.y, w.z, w.w}, qv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                aw[a][0] = fmaf(wv[a], s.x, aw[a][0]);
                aw[a][1] = fmaf(wv[a], s.y, aw[a][1]);
                aq[a][0] = fmaf(qv[a], s.x, aq[a][0]);
                aq[a][1] = fmaf(qv[a], s.y, aq[a][1]);
            }
        }
        __syncthreads();  // every Wt read done: Wt becomes the partial buffer
        float* part = &sm.Wt[0][0] + static_cast<std::size_t>(t & 63) * 16;
        if (hf == 1) {
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                part[4 * a + 0] = aw[a][0];
                part[4 * a + 1] = aw[a][1];
                part[4 * a + 2] = aq[a][0];
                part[4 * a + 3] = aq[a][1];
            }
        }
        __syncthreads();
        float o[4][2];
        if (hf == 0) {  // U = U~ - W S, O1 = exp(G_i) Q S
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const int i = 4 * ig + a;
                const float gi = cc_expf(sm.Gs[i]);
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const float ws_ = aw[a][b] + part[4 * a + b], qs_ = aq[a][b] + part[4 * a + 2 + b];
                    sm.Us[i][2 * cg + b] = sm.Us[i][2 * cg + b] - ws_;
                    o[a][b] = gi * qs_;
                }
            }
        }
        __syncthreads();
        if (hf == 0) {  // O = O1 + P U (Pt is zero above the diagonal), store o rows
#pragma unroll 4
            for (int j = 0; j < C; ++j) {
                const float4 p = *reinterpret_cast<const float4*>(&sm.Pt[j][4 * ig]);
                const float2 u = *reinterpret_cast<const float2*>(&sm.Us[j][2 * cg]);
                const float pv[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    o[a][0] = fmaf(pv[a], u.x, o[a][0]);
                    o[a][1] = fmaf(pv[a], u.y, o[a][1]);
                }
            }
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const std::uint32_t r = c * C + 4 * ig + a;
                if (r < rows)
                    *reinterpret_cast<float2*>(out + static_cast<std::size_t>(r) * d.v_width() + h * D + v0 + 2 * cg) =
                        make_float2(o[a][0], o[a][1]);
            }
        }
        __syncthreads();
        const float glast = sm.Gs[C - 1], dlast = cc_expf(glast);
        {  // U rows scaled by exp(G_last - G_i)
            const int i = static_cast<int>(t >> 3), v4 = static_cast<int>(t & 7) * 4;
            const float f = cc_expf(glast - sm.Gs[i]);
            float4 u = *reinterpret_cast<float4*>(&sm.Us[i][v4]);
            u.x *= f;
            u.y *= f;
            u.z *= f;
            u.w *= f;
            *reinterpret_cast<float4*>(&sm.Us[i][v4]) = u;
        }
        __syncthreads();
        if (c + 1 < nch) fetch(c + 1);  // in flight during step 3
        {  // step 3: S = exp(G_last) S + K^T U, thread -> keys 8kg..8kg+7, columns 4vg..4vg+3
            const int kg = static_cast<int>(t >> 3), vg = static_cast<int>(t & 7);
            float s[8][4];
#pragma unroll
            for (int a = 0; a < 8; ++a) {
                const float4 x = *reinterpret_cast<const float4*>(&sm.S[8 * kg + a][4 * vg]);
                s[a][0] = dlast * x.x;
                s[a][1] = dlast * x.y;
                s[a][2] = dlast * x.z;
                s[a][3] = dlast * x.w;
            }
#pragma unroll 4
            for (int i = 0; i < C; ++i) {
                const float4 k0 = *reinterpret_cast<const float4*>(&sm.Ks[i][8 * kg]);
                const float4 k1 = *reinterpret_cast<const float4*>(&sm.Ks[i][8 * kg + 4]);
                const float4 u = *reinterpret_cast<const float4*>(&sm.Us[i][4 * vg]);
                const float kv[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w}, uv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int a = 0; a < 8; ++a)
#pragma unroll
                    for (int b = 0; b < 4; ++b) s[a][b] = fmaf(kv[a], uv[b], s[a][b]);
            }
#pragma unroll
            for (int a = 0; a < 8; ++a)
                *reinterpret_cast<float4*>(&sm.S[8 * kg + a][4 * vg]) = make_float4(s[a][0], s[a][1], s[a][2], s[a][3]);
        }
        __syncthreads();
        if (c + 1 < nch) store();
        __syncthreads();
    }
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const int idx = q * 128 + static_cast<int>(t), k = idx >> 3, v4 = (idx & 7) * 4;
        *reinterpret_cast<float4*>(s_out + static_cast<std::size_t>(k) * D + v0 + v4) = *reinterpret_cast<const float4*>(&sm.S[k][v4]);
    }
}

// ---- K3: gated RMSNorm (swish gate) of one (row, value head); same formula as the serial kernel's epilogue. The arithmetic is
// gated_norm.cuh, shared with the FAST4 folded quantiser (nvfp4_tile.cuh GatedHeadNorm), which relies on warp q holding positions
// 32 q .. 32 q + 31 in lane order.
__global__ void __launch_bounds__(128) chunk_norm_kernel(float* __restrict__ out, const float* __restrict__ z,
                                                         const float* __restrict__ gain, GdnDims d, float eps) {
    static_assert(static_cast<unsigned>(D) == kGatedHead, "chunk_norm: one CTA = one gated-norm head");
    __shared__ float red[4];
    const std::uint32_t h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const std::size_t o = static_cast<std::size_t>(r) * d.v_width() + h * D + t;
    const float acc = out[o];
    const float s = warp_sum_xor(__fmul_rn(acc, acc));
    if ((t & 31) == 0) red[t >> 5] = s;
    __syncthreads();
    out[o] = gated_head_value(acc, gated_head_inv(red[0], red[1], red[2], red[3], eps), gain[t], z[o]);
}

}  // namespace

std::size_t gdn_chunk_workspace_floats(const GdnDims& d, std::uint32_t rows) { return gc::workspace_floats(d.value_heads, rows); }

cudaError_t gdn_chunk_delta(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                            float* out, std::uint32_t rows, float eps, const std::uint32_t* parity_dev, const GdnChunkWork& w,
                            cudaStream_t stream) {
    if (d.key_dim != gc::kDim || d.value_dim != gc::kDim || rows == 0 || rows > 65535u || w.ws == nullptr ||
        w.floats < gc::workspace_floats(d.value_heads, rows))
        return cudaErrorInvalidValue;
    chunk_prep_kernel<<<dim3(gc::chunks(rows), d.value_heads), 128, 0, stream>>>(win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a,
                                                                                 layer.dt_bias, w.ws, d, rows);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    chunk_scan_kernel<<<dim3(d.value_heads, gc::kDim / gc::kVTile), 128, 0, stream>>>(win.qkv_c, w.ws, state.delta_base, out, d, rows,
                                                                                     parity_dev);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    if (w.defer_norm) return cudaSuccess;
    return gdn_chunk_norm(d, layer.norm_gain, z, out, rows, eps, stream);
}

cudaError_t gdn_chunk_norm(const GdnDims& d, const float* norm_gain, const float* z, float* out, std::uint32_t rows, float eps,
                           cudaStream_t stream) {
    if (d.value_dim != gc::kDim || rows == 0 || rows > 65535u || norm_gain == nullptr || z == nullptr || out == nullptr)
        return cudaErrorInvalidValue;
    chunk_norm_kernel<<<dim3(d.value_heads, rows), 128, 0, stream>>>(out, z, norm_gain, d, eps);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
