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
#include "../../../../cuda_core/gdn/gated_norm.cuh"

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
    // one stage = the 8 weight records and the 32 activation records of input block ks, copied verbatim. Thread tid copies the
    // 16-byte chunks c = tid + i * kThreads (i < kPer, c < kChunks). Which record / tile / source a chunk reads does not depend on ks,
    // so its source base, ks stride, shared offset and predicates are computed ONCE here; a stage then costs kPer cp.async plus one
    // multiply-add each. (Recomputing them per stage in a rolled, divergent loop was ~1,860 clk per 128-K stage serialised after the
    // barrier: the tensor pipe sat at 52 %, ncu rtx_dense2 11-18.) Same bytes to the same shared offsets: the result is bitwise equal.
    constexpr std::uint32_t kPer = (kChunks + kThreads - 1) / kThreads;
    static_assert(kSlotsA == kStageWRecs && kSlotsB == kStageXRecs && kChunks == kStageChunks, "producer contract (nvfp4_tile.hpp)");
    const std::uint8_t* src[kPer];
    std::uint32_t dst[kPer], step[kPer];
    bool live[kPer], okc[kPer];
#pragma unroll
    for (std::uint32_t i = 0; i < kPer; ++i) {
        const std::uint32_t c = tid + i * kThreads;
        const StageChunk q = stage_chunk(c, tm, tn, glu, KS, wtiles, ntiles);
        live[i] = c < kChunks;
        okc[i] = live[i] && q.ok;
        src[i] = (q.which == 2 ? a.x : q.which == 1 ? a.w2 : a.w) + q.base;
        dst[i] = q.dst;
        step[i] = q.step;
    }
    auto load = [&](std::uint32_t stage, std::uint32_t ks) {
        const std::uint32_t st = sbase + stage * kStage;
#pragma unroll
        for (std::uint32_t i = 0; i < kPer; ++i)
            if (live[i]) cp16(st + dst[i], src[i] + static_cast<std::size_t>(ks) * step[i], okc[i]);
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
    // Epilogue loads first, stores after. y may alias aux (the residual GEMMs run in place) and, as plain pointers, rs too, so the
    // compiler may not move a load above an earlier store: written per element, the residual path was 128 LDG -> FADD -> STG chains
    // per thread and rs was re-read after every store, each paying a full memory latency while the CTA held its SM (+0.19-0.20 ms
    // per residual launch, ~20K clk per store/glu tile; rtx_dense3 11-18). Every output element is read (aux) and written (y) only
    // by the thread that owns its accumulator (gemm_args_ok: aux == y or disjoint, rs disjoint from y), so this thread's loads may
    // all go before its stores: the row scales once, then per nt the 16 aux values. Same operations on the same values: bitwise.
    const std::uint32_t g = lane >> 2, t = lane & 3;
    const bool use_aux = a.epi == Epi::residual || a.epi == Epi::silu_mul;
    float rsv[8][2];
#pragma unroll
    for (std::uint32_t nt = 0; nt < 8; ++nt)
#pragma unroll
        for (std::uint32_t j = 0; j < 2; ++j) {
            const std::uint32_t r = act_row(tn, wn, nt, t, j);
            rsv[nt][j] = r < a.rows ? a.rs[r] : 0.f;
        }
#pragma unroll
    for (std::uint32_t nt = 0; nt < 8; ++nt) {
        float av[4][4];  // [z][mt]: this thread's residual / silu_mul operands of tile column nt
#pragma unroll
        for (std::uint32_t z = 0; z < 4; ++z)
#pragma unroll
            for (std::uint32_t mt = 0; mt < 4; ++mt) {
                const std::uint32_t r = act_row(tn, wn, nt, t, z), o = out_row(tm, wm, mt, g, z);
                av[z][mt] = use_aux && r < a.rows && o < a.out ? a.aux[static_cast<std::size_t>(r) * a.out + o] : 0.f;
            }
#pragma unroll
        for (std::uint32_t z = 0; z < 4; ++z) {
            const std::uint32_t r = act_row(tn, wn, nt, t, z);
            if (r >= a.rows) continue;
            const float rs = rsv[nt][z & 1];
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
                    if (a.epi == Epi::residual) v = __fadd_rn(av[z][mt], v);
                    else if (a.epi == Epi::silu_mul) v = __fmul_rn(strix::cuda_core::cc_silu(av[z][mt]), v);
                    a.y[yrow + o] = v;
                }
            }
        }
    }
}

// Padding row r (rows..rows_pad-1): zero codes / scales / rs.
__device__ __forceinline__ void zero_row(std::uint32_t r, std::uint32_t in, std::uint8_t* __restrict__ xt, float* __restrict__ rs) {
    for (std::uint32_t q = threadIdx.x; q < in / 16; q += blockDim.x) {
        *reinterpret_cast<std::uint32_t*>(xt + x_code_byte(r, 16 * q, in)) = 0u;
        *reinterpret_cast<std::uint32_t*>(xt + x_code_byte(r, 16 * q + 8, in)) = 0u;
        xt[x_scale_byte(r, q, in)] = 0;
    }
    if (threadIdx.x == 0) rs[r] = 0.f;
}

// THE activation quantiser of the family (one rule for every producer): 16 values of row r, group q, row scale ts -> e4m3 group scale
// (a16 / 6 / ts, saturated at 448) and 16 e2m1 codes (v / (scale * ts), clamped to +-6, round to nearest even).
__device__ __forceinline__ void quantize_group(const float4* g4, float ts, std::uint32_t r, std::uint32_t q, std::uint32_t in,
                                               std::uint8_t* __restrict__ xt) {
    float v[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 f = g4[i];
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

// one CTA per row; rows >= `rows` (padding up to rows_pad) are written as zero codes / scales / rs.
__global__ void __launch_bounds__(256) quantize_kernel(const float* __restrict__ x, std::uint32_t rows, std::uint32_t in,
                                                       std::uint8_t* __restrict__ xt, float* __restrict__ rs) {
    __shared__ float red[8];
    const std::uint32_t r = blockIdx.x, groups = in / 16;
    if (r >= rows) {
        zero_row(r, in, xt, rs);
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
    for (std::uint32_t q = threadIdx.x; q < groups; q += blockDim.x) quantize_group(row + 4 * q, ts, r, q, in, xt);
}

// Row normalisers folded into the quantiser (norm_quantize_kernel's Norm trait). apply() turns thread t's elements v[i] = x[t + 256 i]
// (idx < in; padding entries are 0) into the normalised values with the arithmetic of the norm kernel they replace, so the f32 row
// equals that kernel's output bit for bit and the exact max + quantize_group give the same codes / scales / rs as norm -> quantise.
//
// RmsRowNorm: cuda_core norm_kernel<rms> verbatim (src/cuda_core/norms/norms.cu): ascending per-thread p += x*x (padding adds +0),
// block_sum256, inv = 1 / sqrt(ss / n + eps), y = (x * inv) * gain (gain may be null).
struct RmsRowNorm {
    const float* gain;
    float eps;
    template <int NREG>
    __device__ __forceinline__ void apply(float (&v)[NREG], std::uint32_t, std::uint32_t in, float* red) const {
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < NREG; ++i) p = __fadd_rn(p, __fmul_rn(v[i], v[i]));
        const float ss = ::strix::cuda_core::block_sum256(p, red);
        const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(in)), eps)));
#pragma unroll
        for (int i = 0; i < NREG; ++i) {
            const std::uint32_t idx = threadIdx.x + 256u * i;
            if (idx < in) {
                const float y = __fmul_rn(v[i], inv);
                v[i] = gain != nullptr ? __fmul_rn(y, gain[idx]) : y;
            }
        }
    }
};

// GatedHeadNorm: the GDN value-head gated RMSNorm of gdn_chunk.cu chunk_norm_kernel (one CTA per (head, row), 128 threads). Here warp
// w at step i holds the 32 consecutive elements 256 i + 32 w + lane = head 2 i + (w >> 2), quarter w & 3, lane = position & 31: the
// same warp_sum_xor input in the same lane order as chunk_norm's warp (w & 3) of that head, so the same partial; the four partials
// and the inv / gain / silu chain are cuda_core gated_head_inv / gated_head_value (gdn/gated_norm.cuh), shared with chunk_norm.
// z: the gate rows (stride in), gain: kGatedHead values. Requires in % kGatedHead == 0 and in <= 64 heads.
struct GatedHeadNorm {
    const float* z;
    const float* gain;
    float eps;
    template <int NREG>
    __device__ __forceinline__ void apply(float (&v)[NREG], std::uint32_t r, std::uint32_t in, float*) const {
        namespace cc = ::strix::cuda_core;
        static_assert(cc::kGatedHead == 128 && kGatedNormHead == cc::kGatedHead, "two heads per 256-thread step; one head width");
        __shared__ float part[2 * NREG][4];
        const std::uint32_t lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
#pragma unroll
        for (int i = 0; i < NREG; ++i) {
            if (256u * i + 32u * w < in) {  // warp-uniform: in % 128 == 0
                const float s = cc::warp_sum_xor(__fmul_rn(v[i], v[i]));
                if (lane == 0) part[2 * i + (w >> 2)][w & 3u] = s;
            }
        }
        __syncthreads();
        const float* zr = z + static_cast<std::size_t>(r) * in;
#pragma unroll
        for (int i = 0; i < NREG; ++i) {
            const std::uint32_t idx = threadIdx.x + 256u * i;
            if (idx < in) {
                const float* q = part[idx / cc::kGatedHead];
                v[i] = cc::gated_head_value(v[i], cc::gated_head_inv(q[0], q[1], q[2], q[3], eps), gain[idx % cc::kGatedHead], zr[idx]);
            }
        }
    }
};

// A row normaliser (Norm, above) folded into the quantiser: the FAST4 A operand straight from the un-normalised rows, no f32
// normalised row in memory. One CTA per row, NREG * 256 >= in; dynamic smem in * 4 bytes holds the normalised row for the 16-group
// pass.
template <int NREG, class Norm>
__global__ void __launch_bounds__(256) norm_quantize_kernel(const float* __restrict__ x, const Norm nm, std::uint32_t rows,
                                                            std::uint32_t in, std::uint8_t* __restrict__ xt, float* __restrict__ rs) {
    extern __shared__ float4 nq_row4[];
    __shared__ float red[8];
    const std::uint32_t r = blockIdx.x, groups = in / 16;
    if (r >= rows) {
        zero_row(r, in, xt, rs);
        return;
    }
    float* const yrow = reinterpret_cast<float*>(nq_row4);
    const float* xp = x + static_cast<std::size_t>(r) * in;
    float v[NREG];
#pragma unroll
    for (int i = 0; i < NREG; ++i) {
        const std::uint32_t idx = threadIdx.x + 256u * i;
        v[i] = idx < in ? xp[idx] : 0.0f;
    }
    nm.apply(v, r, in, red);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < NREG; ++i) {
        const std::uint32_t idx = threadIdx.x + 256u * i;
        if (idx < in) {
            yrow[idx] = v[i];
            amax = fmaxf(amax, fabsf(v[i]));
        }
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();  // red and the whole normalised row are visible after this
    amax = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) amax = fmaxf(amax, red[w]);
    const float ts = __fdiv_rn(amax, kRowScaleDiv);
    if (threadIdx.x == 0) rs[r] = ts;
    for (std::uint32_t q = threadIdx.x; q < groups; q += blockDim.x) quantize_group(nq_row4 + 4 * q, ts, r, q, in, xt);
}

// one thread per source byte: a byte permutation from the decode record layout to the W tile layout.
__global__ void __launch_bounds__(256) repack_kernel(const std::uint8_t* __restrict__ src, std::size_t bytes, std::uint32_t in,
                                                     std::uint8_t* __restrict__ dst) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < bytes) dst[repack_dst(i, in)] = src[i];
}
}  // namespace strix::cuda::core::nvfp4::tile::sm120
