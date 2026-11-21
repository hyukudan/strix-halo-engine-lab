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

// ---- K2 tf32 (gdn-chunk16-tf32-v2): one (value head, 64-column value tile) per CTA; warp w owns value columns 16 w .. 16 w + 15
// with ALL 128 keys, so its FP32 state S^T[16 v][128 k] is 16 accumulator fragments (64 registers) for the whole launch and no
// product needs a cross-warp sum. Per chunk and warp, every product in split TF32 (3 mma each, gdn_chunk.hpp Scan): X^T = S^T W^T
// and Y^T = S^T Q^T (2 x 96 mma), O^T = exp(G) Y^T + U^T P^T (12 mma), S^T = exp(G_last) S^T + U'^T K (96 mma): 300 mma. Every A
// operand is a C fragment as it is (gdn_chunk.hpp tc::perm). The next chunk's operands stream into the other shared-memory stage
// by cp.async during this chunk: one barrier per chunk.
namespace tcs = gc::tc;
constexpr int WTS = static_cast<int>(tcs::kWtStride), QS = static_cast<int>(tcs::kQStride), KS = static_cast<int>(tcs::kKStride),
              US = static_cast<int>(tcs::kUStride), PTS = static_cast<int>(tcs::kPtStride);
constexpr int SWT = static_cast<int>(tcs::Stage::wt), SQ = static_cast<int>(tcs::Stage::q), SK = static_cast<int>(tcs::Stage::k),
              SU = static_cast<int>(tcs::Stage::u), SPT = static_cast<int>(tcs::Stage::pt), SG = static_cast<int>(tcs::Stage::g),
              SFL = static_cast<int>(tcs::Stage::floats);

__device__ __forceinline__ std::uint32_t to_tf32(float x) {
    std::uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void mma_tf32(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// split TF32 operand (gdn_chunk.hpp tf32-v2): x = hi + lo with hi = rna(x), lo = rna(x - hi)
struct Tf32Pair {
    std::uint32_t hi, lo;
};
__device__ __forceinline__ Tf32Pair split_tf32(float x) {
    const std::uint32_t hi = to_tf32(x);
    return {hi, to_tf32(x - __uint_as_float(hi))};
}
// d += A B in split TF32: the two cross terms first, the hi * hi term last (lo * lo, relative 2^-22, dropped)
__device__ __forceinline__ void mma_tf32x3(float (&d)[4], const std::uint32_t (&ah)[4], const std::uint32_t (&al)[4], Tf32Pair b0,
                                           Tf32Pair b1) {
    mma_tf32(d, al, b0.hi, b1.hi);
    mma_tf32(d, ah, b0.lo, b1.lo);
    mma_tf32(d, ah, b0.hi, b1.hi);
}
// the same with the cross terms in their own accumulator ds (the caller adds d + ds after a long reduction)
__device__ __forceinline__ void mma_tf32x3_apart(float (&d)[4], float (&ds)[4], const std::uint32_t (&ah)[4], const std::uint32_t (&al)[4],
                                                 Tf32Pair b0, Tf32Pair b1) {
    mma_tf32(ds, al, b0.hi, b1.hi);
    mma_tf32(ds, ah, b0.lo, b1.lo);
    mma_tf32(d, ah, b0.hi, b1.hi);
}
// 16-byte cp.async; valid == false fills the 16 bytes with zeros (src must still be a mapped address)
__device__ __forceinline__ void cp16z(float* dst, const float* src, bool valid) {
    const auto s = static_cast<std::uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

__global__ void __launch_bounds__(128) chunk_scan_tc_kernel(const float* __restrict__ qkvc, const float* __restrict__ ws, float* base,
                                                            float* __restrict__ out, GdnDims d, std::uint32_t rows,
                                                            const std::uint32_t* parity_dev) {
    extern __shared__ __align__(16) float smem[];
    const int t = static_cast<int>(threadIdx.x), lane = t & 31, warp = t >> 5, g = lane >> 2, tq = lane & 3;
    const std::uint32_t h = blockIdx.x, kh = h % d.key_heads, ch = d.conv_channels(), vh = d.value_heads, nch = gc::chunks(rows);
    const int c0 = static_cast<int>(blockIdx.y * tcs::kCtaCols);          // first value column of the CTA tile
    const int vw = static_cast<int>(warp * static_cast<int>(tcs::kVCols));  // first value column of this warp inside the tile
    const std::size_t side = static_cast<std::size_t>(vh) * D * D;
    const std::uint32_t par = *parity_dev;
    const float* s_in = base + par * side + static_cast<std::size_t>(h) * D * D;
    float* s_out = base + (1u - par) * side + static_cast<std::size_t>(h) * D * D;

    // one operand stage: Wt, q, k rows (zero past `rows`), this CTA's U~ columns, Pt, G; one cp.async group
    auto load = [&](std::uint32_t c, float* st) {
        const float* blk = ws + gc::block_offset(c, h, vh);
        for (int n = t; n < D * 4; n += 128) {
            const int k = n >> 2, f = (n & 3) * 4;
            cp16z(st + SWT + k * WTS + f, blk + L::wt + k * C + f, true);
        }
        for (int n = t; n < C * (D / 4); n += 128) {
            const int i = n >> 5, f = (n & 31) * 4;
            const std::uint32_t r = c * C + static_cast<std::uint32_t>(i);
            const bool in = r < rows;
            const float* row = qkvc + static_cast<std::size_t>(in ? r : 0u) * ch;
            cp16z(st + SQ + i * QS + f, row + kh * D + f, in);
            cp16z(st + SK + i * KS + f, row + d.qk_width() + kh * D + f, in);
        }
        for (int n = t; n < C * 16; n += 128) {
            const int i = n >> 4, f = (n & 15) * 4;
            cp16z(st + SU + i * US + f, blk + L::u + i * D + c0 + f, true);
        }
        if (t < C * 4) cp16z(st + SPT + (t >> 2) * PTS + (t & 3) * 4, blk + L::pt + (t >> 2) * C + (t & 3) * 4, true);
        if (t < C / 4) cp16z(st + SG + t * 4, blk + L::g + t * 4, true);
        asm volatile("cp.async.commit_group;" ::: "memory");
    };

    float acc[D / 8][4];  // block nt, element e = S[key 8 nt + c_col(e)][value c0 + vw + c_row(e)]
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            acc[nt][e] = s_in[static_cast<std::size_t>(8 * nt + tcs::c_col(lane, e)) * D + c0 + vw + tcs::c_row(lane, e)];
    load(0, smem);
    for (std::uint32_t c = 0; c < nch; ++c) {
        const float* st = smem + static_cast<int>(c & 1u) * SFL;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();  // stage c landed for every thread; every read of the other stage (chunk c - 1) is done
        if (c + 1 < nch) load(c + 1, smem + static_cast<int>((c + 1) & 1u) * SFL);

        // X^T = S^T W^T, Y^T = S^T Q^T (split TF32): k-slot t <-> key 8 nt + 2t, k-slot t + 4 <-> key 8 nt + 2t + 1 (tc::perm).
        // The cross terms go to their own accumulators xs / ys: 8 independent 32-deep chains instead of 4 48-deep ones.
        float x[2][4] = {}, y[2][4] = {}, xs[2][4] = {}, ys[2][4] = {};
#pragma unroll
        for (int nt = 0; nt < D / 8; ++nt) {
            std::uint32_t ah[4], al[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const Tf32Pair s = split_tf32(acc[nt][tcs::a_from_c(e)]);
                ah[e] = s.hi;
                al[e] = s.lo;
            }
            const int k0 = 8 * nt + 2 * tq;
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int i = 8 * ni + g;
                mma_tf32x3_apart(x[ni], xs[ni], ah, al, split_tf32(st[SWT + k0 * WTS + i]), split_tf32(st[SWT + (k0 + 1) * WTS + i]));
                const float2 qq = *reinterpret_cast<const float2*>(st + SQ + i * QS + k0);
                mma_tf32x3_apart(y[ni], ys[ni], ah, al, split_tf32(qq.x), split_tf32(qq.y));
            }
        }
        // U^T = U~^T - X^T, O^T = exp(G_i) Y^T at the C fragment (rows = values, columns = chunk rows i)
        float u[2][4], o[2][4];
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = 8 * ni + tcs::c_col(lane, e);
                u[ni][e] = st[SU + i * US + vw + tcs::c_row(lane, e)] - (x[ni][e] + xs[ni][e]);
                o[ni][e] = cc_expf(st[SG + i]) * (y[ni][e] + ys[ni][e]);
            }
        // O^T += U^T P^T (split TF32): U's columns are the k index (rows j), so its C fragment is the A operand
#pragma unroll
        for (int kj = 0; kj < 2; ++kj) {
            std::uint32_t ah[4], al[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const Tf32Pair s = split_tf32(u[kj][tcs::a_from_c(e)]);
                ah[e] = s.hi;
                al[e] = s.lo;
            }
            const int j0 = 8 * kj + 2 * tq;
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int i = 8 * ni + g;
                mma_tf32x3(o[ni], ah, al, split_tf32(st[SPT + j0 * PTS + i]), split_tf32(st[SPT + (j0 + 1) * PTS + i]));
            }
        }
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const std::uint32_t r = c * C + static_cast<std::uint32_t>(8 * ni + tcs::c_col(lane, e));
                if (r < rows) out[static_cast<std::size_t>(r) * d.v_width() + h * D + static_cast<std::uint32_t>(c0 + vw + tcs::c_row(lane, e))] = o[ni][e];
            }
        // S^T = exp(G_last) S^T + U'^T K (split TF32) with U'[j] = exp(G_last - G_j) U[j]
        const float glast = st[SG + C - 1], dlast = cc_expf(glast);
        std::uint32_t uh[2][4], ul[2][4];
#pragma unroll
        for (int kj = 0; kj < 2; ++kj) {
#pragma unroll
            for (int e = 0; e < 4; ++e) u[kj][e] *= cc_expf(glast - st[SG + 8 * kj + tcs::c_col(lane, e)]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const Tf32Pair s = split_tf32(u[kj][tcs::a_from_c(e)]);
                uh[kj][e] = s.hi;
                ul[kj][e] = s.lo;
            }
        }
#pragma unroll
        for (int nt = 0; nt < D / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[nt][e] *= dlast;
            const int kk = 8 * nt + g;
#pragma unroll
            for (int kj = 0; kj < 2; ++kj) {
                const int j0 = 8 * kj + 2 * tq;
                mma_tf32x3(acc[nt], uh[kj], ul[kj], split_tf32(st[SK + j0 * KS + kk]), split_tf32(st[SK + (j0 + 1) * KS + kk]));
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            s_out[static_cast<std::size_t>(8 * nt + tcs::c_col(lane, e)) * D + c0 + vw + tcs::c_row(lane, e)] = acc[nt][e];
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
        w.floats < gc::workspace_floats(d.value_heads, rows) || w.scan > static_cast<std::uint8_t>(gc::Scan::tf32))
        return cudaErrorInvalidValue;
    chunk_prep_kernel<<<dim3(gc::chunks(rows), d.value_heads), 128, 0, stream>>>(win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a,
                                                                                 layer.dt_bias, w.ws, d, rows);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    if (w.scan == static_cast<std::uint8_t>(gc::Scan::tf32)) {
        static_assert(gc::kDim % gc::tc::kCtaCols == 0 && gc::tc::kWarps * 32 == 128, "tf32 scan geometry");
        static const cudaError_t attr = cudaFuncSetAttribute(chunk_scan_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                             static_cast<int>(gc::tc::kSmemBytes));
        if (attr != cudaSuccess) return attr;
        chunk_scan_tc_kernel<<<dim3(d.value_heads, gc::kDim / gc::tc::kCtaCols), 128, gc::tc::kSmemBytes, stream>>>(
            win.qkv_c, w.ws, state.delta_base, out, d, rows, parity_dev);
    } else {
        chunk_scan_kernel<<<dim3(d.value_heads, gc::kDim / gc::kVTile), 128, 0, stream>>>(win.qkv_c, w.ws, state.delta_base, out, d,
                                                                                         rows, parity_dev);
    }
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
