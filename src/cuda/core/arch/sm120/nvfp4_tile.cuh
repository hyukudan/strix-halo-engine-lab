#pragma once
// sm_120 kernels of the NVFP4 tile GEMM (contract, layouts and identity: strix/cuda/core/nvfp4_tile.hpp).
//
// Tile: CTA = 128 weight rows (A, mma m) x 256 activation rows (B, mma n), K 128 per stage, 3-stage cp.async ring, 8 warps as
// 2 (weight) x 4 (rows), warp tile 64 x 64 = 4 x 8 m16n8k64 MMAs per 64-input block, 128 FP32 accumulators per thread.
// Work per MAC: smem->register 0.021 B (LDS.128 A fragment + LDS.64 B fragment + one u32 scale each), L2->smem 0.0066 B
// ((1/128 + 1/256) x 0.5625 B), so the kernel stays under both data pipes at the FP4 tensor rate (2,048 MAC/clk/SM).
// Rasterisation: activation tiles vary fastest, so co-resident CTAs share one weight tile and each weight is read from DRAM about
// once per call; the 256-row activation tiles (<= 12 MB per 4K x 5120 input) stay in L2.
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "strix/cuda/core/nvfp4_tile.hpp"
#include "../../../../cuda_core/common.cuh"

namespace strix::cuda::core::nvfp4::tile::sm120 {
constexpr std::uint32_t kThreads = 256, kStages = 3, kSlotsA = 8, kSlotsB = 32;
constexpr std::uint32_t kAStage = kSlotsA * kWRec, kBStage = kSlotsB * kXRec, kStage = kAStage + kBStage;
constexpr std::uint32_t kSmem = kStages * kStage;  // 82,944 B
constexpr std::uint32_t kChunksA = kAStage / 16, kChunks = kStage / 16;
static_assert(kWRec % 16 == 0 && kXRec % 16 == 0 && kSmem <= 99 * 1024, "stage layout");

// Operand order of the validated operator (arch/sm120/nvfp4.cuh): a = {(g, k 8t), (g + 8, k 8t), (g, k 32 + 8t), (g + 8, ...)},
// b = {(n g, k 8t), (n g, k 32 + 8t)}, scale A from row g + 8 (lane & 1), scale B from column g, selectors {0, 0}.
__device__ __forceinline__ void mma(float (&d)[4], const uint4& a, const uint2& b, std::uint32_t sa, std::uint32_t sb) {
#if __CUDA_ARCH__ == 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(sa), "r"(sb));
#else
    asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp16(std::uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__global__ void __launch_bounds__(kThreads, 1) gemm_kernel(const GemmArgs a) {
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t KS = a.in / kK, tiles_n = a.rows_pad / kCtaRows, wtiles = a.out / 16, ntiles = a.rows_pad / 8;
    const std::uint32_t tm = blockIdx.x / tiles_n, tn = blockIdx.x % tiles_n;
    const bool glu = a.epi == Epi::glu;
    const std::uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp & 1, wn = warp >> 1;
    const std::uint32_t sbase = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
    // one stage = the 8 weight records and the 32 activation records of input block ks, copied verbatim
    auto load = [&](std::uint32_t stage, std::uint32_t ks) {
        const std::uint32_t st = sbase + stage * kStage;
        for (std::uint32_t c = tid; c < kChunks; c += kThreads) {
            if (c < kChunksA) {
                const std::uint32_t s = c / (kWRec / 16), off = (c % (kWRec / 16)) * 16;
                const std::uint32_t t = glu ? glu_slot_tile(tm, s) : a_slot_tile(tm, s);
                const std::uint8_t* W = glu && (s & 1) ? a.w2 : a.w;
                const bool ok = t < wtiles;
                cp16(st + s * kWRec + off, W + (static_cast<std::size_t>(ok ? t : 0) * KS + ks) * kWRec + off, ok);
            } else {
                const std::uint32_t c2 = c - kChunksA, j = c2 / (kXRec / 16), off = (c2 % (kXRec / 16)) * 16;
                const std::uint32_t t = tn * kSlotsB + j;
                const bool ok = t < ntiles;
                cp16(st + kAStage + j * kXRec + off, a.x + (static_cast<std::size_t>(ok ? t : 0) * KS + ks) * kXRec + off, ok);
            }
        }
    };
    float acc[4][8][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int z = 0; z < 4; ++z) acc[i][j][z] = 0.f;
#pragma unroll
    for (std::uint32_t s = 0; s + 1 < kStages; ++s) {
        if (s < KS) load(s, s);
        cp_commit();
    }
    for (std::uint32_t ks = 0; ks < KS; ++ks) {
        cp_wait<kStages - 2>();
        __syncthreads();  // stage ks landed for every thread; stage (ks - 1) % kStages is free
        if (ks + kStages - 1 < KS) load((ks + kStages - 1) % kStages, ks + kStages - 1);
        cp_commit();
        const unsigned char* sA = smem + (ks % kStages) * kStage;
        const unsigned char* sB = sA + kAStage;
#pragma unroll
        for (std::uint32_t kb = 0; kb < 2; ++kb) {
            uint4 af[4];
            std::uint32_t sa[4];
#pragma unroll
            for (std::uint32_t mt = 0; mt < 4; ++mt) {
                const unsigned char* rec = sA + (wm * 4 + mt) * kWRec;
                af[mt] = *reinterpret_cast<const uint4*>(rec + a_frag(kb, lane));
                sa[mt] = *reinterpret_cast<const std::uint32_t*>(rec + a_scale(kb, lane));
            }
#pragma unroll
            for (std::uint32_t nt = 0; nt < 8; ++nt) {
                const unsigned char* rec = sB + (wn * 8 + nt) * kXRec;
                const uint2 bf = *reinterpret_cast<const uint2*>(rec + b_frag(kb, lane));
                const std::uint32_t sb = *reinterpret_cast<const std::uint32_t*>(rec + b_scale(kb, lane));
#pragma unroll
                for (std::uint32_t mt = 0; mt < 4; ++mt) mma(acc[mt][nt], af[mt], bf, sa[mt], sb);
            }
        }
    }
    cp_wait<0>();
    const std::uint32_t g = lane >> 2, t = lane & 3;
#pragma unroll
    for (std::uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
        for (std::uint32_t z = 0; z < 4; ++z) {
            const std::uint32_t r = act_row(tn, wn, nt, t, z);
            if (r >= a.rows) continue;
            const float rs = a.rs[r];
            const std::size_t yrow = static_cast<std::size_t>(r) * a.out;
            if (glu) {
#pragma unroll
                for (std::uint32_t p = 0; p < 2; ++p) {
                    const std::uint32_t ch = glu_channel(tm, wm, p, g, z);
                    if (ch >= a.out) continue;
                    const float gate = __fmul_rn(acc[2 * p][nt][z], __fmul_rn(a.gs, rs));
                    const float up = __fmul_rn(acc[2 * p + 1][nt][z], __fmul_rn(a.gs2, rs));
                    a.y[yrow + ch] = __fmul_rn(strix::cuda_core::cc_silu(gate), up);
                }
            } else {
#pragma unroll
                for (std::uint32_t mt = 0; mt < 4; ++mt) {
                    const std::uint32_t o = out_row(tm, wm, mt, g, z);
                    if (o >= a.out) continue;
                    float v = __fmul_rn(acc[mt][nt][z], __fmul_rn(a.gs, rs));
                    if (a.epi == Epi::residual) v = __fadd_rn(a.aux[yrow + o], v);
                    else if (a.epi == Epi::silu_mul) v = __fmul_rn(strix::cuda_core::cc_silu(a.aux[yrow + o]), v);
                    a.y[yrow + o] = v;
                }
            }
        }
}

// one CTA per row; rows >= `rows` (padding up to rows_pad) are written as zero codes / scales / rs.
__global__ void __launch_bounds__(256) quantize_kernel(const float* __restrict__ x, std::uint32_t rows, std::uint32_t in,
                                                       std::uint8_t* __restrict__ xt, float* __restrict__ rs) {
    __shared__ float red[8];
    const std::uint32_t r = blockIdx.x, groups = in / 16;
    if (r >= rows) {
        for (std::uint32_t q = threadIdx.x; q < groups; q += blockDim.x) {
            *reinterpret_cast<std::uint32_t*>(xt + x_code_byte(r, 16 * q, in)) = 0u;
            *reinterpret_cast<std::uint32_t*>(xt + x_code_byte(r, 16 * q + 8, in)) = 0u;
            xt[x_scale_byte(r, q, in)] = 0;
        }
        if (threadIdx.x == 0) rs[r] = 0.f;
        return;
    }
    const float4* row = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in);
    float amax = 0.f;
    for (std::uint32_t i = threadIdx.x; i < in / 4; i += blockDim.x) {
        const float4 v = row[i];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) amax = fmaxf(amax, red[w]);
    const float ts = __fdiv_rn(amax, kRowScaleDiv);
    if (threadIdx.x == 0) rs[r] = ts;
    for (std::uint32_t q = threadIdx.x; q < groups; q += blockDim.x) {
        float v[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 f = row[4 * q + i];
            v[4 * i] = f.x;
            v[4 * i + 1] = f.y;
            v[4 * i + 2] = f.z;
            v[4 * i + 3] = f.w;
        }
        float a16 = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) a16 = fmaxf(a16, fabsf(v[i]));
        __nv_fp8_e4m3 sc;
        sc.__x = 0;
        float scale = 0.f;
        if (ts > 0.f) {
            sc.__x = __nv_cvt_float_to_fp8(fminf(448.f, __fdiv_rn(__fdiv_rn(a16, 6.f), ts)), __NV_SATFINITE, __NV_E4M3);
            scale = __fmul_rn(static_cast<float>(sc), ts);
        }
        std::uint32_t word[2] = {0u, 0u};
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const float u = scale != 0.f ? fminf(6.f, fmaxf(-6.f, __fdiv_rn(v[2 * p], scale))) : 0.f;
            const float w = scale != 0.f ? fminf(6.f, fmaxf(-6.f, __fdiv_rn(v[2 * p + 1], scale))) : 0.f;
            const std::uint32_t b = __nv_cvt_float2_to_fp4x2(make_float2(u, w), __NV_E2M1, cudaRoundNearest);
            word[p >> 2] |= (b & 0xffu) << (8 * (p & 3));
        }
        *reinterpret_cast<std::uint32_t*>(xt + x_code_byte(r, 16 * q, in)) = word[0];
        *reinterpret_cast<std::uint32_t*>(xt + x_code_byte(r, 16 * q + 8, in)) = word[1];
        xt[x_scale_byte(r, q, in)] = sc.__x;
    }
}

// one thread per source byte: a byte permutation from the decode record layout to the W tile layout.
__global__ void __launch_bounds__(256) repack_kernel(const std::uint8_t* __restrict__ src, std::size_t bytes, std::uint32_t in,
                                                     std::uint8_t* __restrict__ dst) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < bytes) dst[repack_dst(i, in)] = src[i];
}
}  // namespace strix::cuda::core::nvfp4::tile::sm120
