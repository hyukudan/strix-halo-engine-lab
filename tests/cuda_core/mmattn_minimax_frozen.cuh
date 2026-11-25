#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Frozen from src/cuda/models/minimax/mm_rtx_kernels.cu:22–226 at a18968a99.
// The constants and SeqView below mirror mm_rtx_select.hpp:13–22.
// Retain the two original kernels and their arithmetic independently of K4c.
namespace mmattn_frozen {

inline constexpr unsigned kBlock = 128, kTopK = 16;

struct SeqView {
    std::uint32_t row0, rows, cap;
    std::uint16_t *k, *v, *ik;
    std::uint32_t pos0;
};

constexpr float kScale = 0.08838834764831845F;

__device__ __forceinline__ float u2f(std::uint32_t b) { return __uint_as_float(b); }
__device__ __forceinline__ float bf16_to_float(std::uint16_t b) { return u2f(static_cast<std::uint32_t>(b) << 16U); }
__device__ __forceinline__ float fmaxs(float a, float b) { return a < b ? b : a; }  // std::max(a, b)
__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000U); }
__device__ float exp_f(float x) {  // strix/minimax_msa.hpp exp_f
    if (x < -87.0F) return 0.0F;
    if (x > 88.0F) return u2f(0x7f800000U);
    const float n = rintf(x * 1.44269504088896341F);
    float r = __fmaf_rn(n, -0.693145751953125F, x);
    r = __fmaf_rn(n, -1.42860676533018e-06F, r);
    float p = 2.48015873e-05F;
    p = __fmaf_rn(p, r, 1.98412698e-04F);
    p = __fmaf_rn(p, r, 1.38888889e-03F);
    p = __fmaf_rn(p, r, 8.33333333e-03F);
    p = __fmaf_rn(p, r, 4.16666667e-02F);
    p = __fmaf_rn(p, r, 1.66666667e-01F);
    p = __fmaf_rn(p, r, 0.5F);
    p = __fmaf_rn(p, r, 1.0F);
    p = __fmaf_rn(p, r, 1.0F);
    return p * u2f(static_cast<std::uint32_t>(static_cast<int>(n) + 127) << 23U);
}
__device__ __forceinline__ float shx(float v, int d) { return __shfl_xor_sync(0xffffffffU, v, d, 32); }
// Butterfly of strix/minimax_msa.hpp (v = v + v[lane ^ d], d = 16..1): every lane ends with lanes[0]'s value.
__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v = v + shx(v, d);
    return v;
}
// Sparse attention: one warp per (row, query head h); lane l owns dims 4l..4l+3.
__global__ void attend_kernel(const SeqView* seqs, const std::uint32_t* row_seq, std::size_t rows, unsigned layer, const float* q,
                              const std::uint32_t* sel, const std::uint32_t* sel_count, float* o) {
    const std::size_t warp = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x & 31U;
    if (warp >= rows * 64) return;
    const std::size_t row = warp / 64;
    const unsigned h = static_cast<unsigned>(warp % 64), g = h / 16;
    const SeqView sq = seqs[row_seq[row]];
    const std::uint64_t count = row - sq.row0 + sq.pos0 + 1;
    float qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = q[(row * 64 + h) * 128 + 4 * lane + i];
    const std::uint32_t* blocks = sel + (row * 4 + g) * kTopK;
    const unsigned chunks = sel_count[row * 4 + g];
    float cm[kTopK], cl[kTopK], cacc[kTopK][4];
    for (unsigned c = 0; c < chunks; ++c) {
        float m = neg_inf(), l = 0.0F, acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        bool first = true;
        const std::uint64_t lo = static_cast<std::uint64_t>(blocks[c]) * kBlock;
        const std::uint64_t hi = count < lo + kBlock ? count : lo + kBlock;
        for (std::uint64_t t0 = lo; t0 < hi; t0 += 16) {
            const unsigned tn = static_cast<unsigned>((hi - t0) < 16 ? (hi - t0) : 16);
            float s[16];
            float tm = neg_inf();
            for (unsigned j = 0; j < tn; ++j) {
                const std::uint16_t* kc = sq.k + ((static_cast<std::size_t>(layer) * sq.cap + t0 + j) * 4 + g) * 128 + 4 * lane;
                float part = 0.0F;
#pragma unroll
                for (int i = 0; i < 4; ++i) part = __fmaf_rn(qv[i], bf16_to_float(kc[i]), part);
                s[j] = butterfly(part) * kScale;
                tm = fmaxs(tm, s[j]);
            }
            const float mn = fmaxs(m, tm);
            const float corr = first ? 0.0F : exp_f(m - mn);
            l = l * corr;
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] = acc[i] * corr;
            for (unsigned j = 0; j < tn; ++j) {
                const float p = exp_f(s[j] - mn);
                l = l + p;
                const std::uint16_t* vc = sq.v + ((static_cast<std::size_t>(layer) * sq.cap + t0 + j) * 4 + g) * 128 + 4 * lane;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] = __fmaf_rn(p, bf16_to_float(vc[i]), acc[i]);
            }
            m = mn;
            first = false;
        }
        cm[c] = m;
        cl[c] = l;
#pragma unroll
        for (int i = 0; i < 4; ++i) cacc[c][i] = acc[i];
    }
    float big = neg_inf();
    for (unsigned c = 0; c < chunks; ++c) big = fmaxs(big, cm[c]);
    float L = 0.0F, e[kTopK];
    for (unsigned c = 0; c < chunks; ++c) {
        e[c] = exp_f(cm[c] - big);
        L = __fmaf_rn(cl[c], e[c], L);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = 0.0F;
        for (unsigned c = 0; c < chunks; ++c) a = __fmaf_rn(cacc[c][i], e[c], a);
        o[(row * 64 + h) * 128 + 4 * lane + i] = a / L;
    }
}

// Sparse attention, shared chunks + key-parallel scores (same arithmetic as attend_kernel): CTA = (row, KV group g), 16
// warps = the group's 16 query heads; each selected block's K and V rows (BF16) are staged once in shared memory. Lane j
// computes the whole score of key j (the reference's 32 lane chains + tree, in registers) and its p = exp_f(s - m');
// the online-softmax state then advances key by key in the reference order, p broadcast by one shuffle, while lane l
// keeps the PV accumulators of dims 4l..4l+3. K rows are padded to 132 BF16 (264 B): lane j reads row j, and at the
// natural 256-B stride every lane hit the same banks (32-way conflicts on each score load); V is read row-wise.
constexpr unsigned kKStride = 132;
__global__ __launch_bounds__(512) void attend_shared_kernel(const SeqView* seqs, const std::uint32_t* row_seq, unsigned layer, const float* q,
                                                            const std::uint32_t* sel, const std::uint32_t* sel_count, float* o,
                                                            int diag) {
    extern __shared__ uint4 smem16[];
    std::uint16_t* kc = reinterpret_cast<std::uint16_t*>(smem16);  // [128][kKStride] BF16
    std::uint16_t* vc = kc + 128 * kKStride;                        // [128][128] BF16
    float* qs = reinterpret_cast<float*>(vc + 128 * 128);           // [16 heads][128]
    const std::size_t row = blockIdx.x;
    const unsigned g = blockIdx.y, w = threadIdx.x / 32, lane = threadIdx.x & 31U, h = 16 * g + w;
    const SeqView sq = seqs[row_seq[row]];
    const std::uint64_t count = row - sq.row0 + sq.pos0 + 1;
    for (unsigned i = lane; i < 128; i += 32) qs[w * 128 + i] = q[(row * 64 + h) * 128 + i];
    const float* qh = qs + w * 128;
    const std::uint32_t* blocks = sel + (row * 4 + g) * kTopK;
    const unsigned chunks = sel_count[row * 4 + g];
    float cm[kTopK], cl[kTopK], cacc[kTopK][4];
    for (unsigned c = 0; c < chunks; ++c) {
        const std::uint64_t lo = static_cast<std::uint64_t>(blocks[c]) * kBlock;
        const std::uint64_t hi = count < lo + kBlock ? count : lo + kBlock;
        const unsigned keys = static_cast<unsigned>(hi - lo);
        __syncthreads();  // the previous chunk is consumed (and qs is written before the first)
        for (unsigned i = threadIdx.x; diag != 2 && i < keys * 16; i += 512) {  // 16 uint4 (8 BF16 each) per key row
            const unsigned j = i / 16, part = i % 16;
            const std::size_t at = ((static_cast<std::size_t>(layer) * sq.cap + lo + j) * 4 + g) * 128;
            const uint4 kv = reinterpret_cast<const uint4*>(sq.k + at)[part];
            uint2* kd = reinterpret_cast<uint2*>(kc + j * kKStride + 8 * part);  // 8-B aligned rows
            kd[0] = make_uint2(kv.x, kv.y);
            kd[1] = make_uint2(kv.z, kv.w);
            reinterpret_cast<uint4*>(vc)[j * 16 + part] = reinterpret_cast<const uint4*>(sq.v + at)[part];
        }
        __syncthreads();
        float m = neg_inf(), l = 0.0F, acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        bool first = true;
        for (unsigned g0 = 0; diag != 1 && g0 < keys; g0 += 32) {  // 32 keys = two softmax tiles of 16
            const unsigned key = g0 + lane;
            float sc = neg_inf();
            if (key < keys) {
                const std::uint16_t* kr = kc + key * kKStride;
                float cc[32];
#pragma unroll
                for (int ll = 0; ll < 32; ++ll) {
                    const uint2 k2 = *reinterpret_cast<const uint2*>(kr + 4 * ll);
                    float a = __fmaf_rn(qh[4 * ll], u2f(k2.x << 16U), 0.0F);
                    a = __fmaf_rn(qh[4 * ll + 1], u2f(k2.x & 0xffff0000U), a);
                    a = __fmaf_rn(qh[4 * ll + 2], u2f(k2.y << 16U), a);
                    a = __fmaf_rn(qh[4 * ll + 3], u2f(k2.y & 0xffff0000U), a);
                    cc[ll] = a;
                }
#pragma unroll
                for (int d = 16; d >= 1; d >>= 1)
#pragma unroll
                    for (int ll = 0; ll < d; ++ll) cc[ll] = cc[ll] + cc[ll + d];
                sc = cc[0] * kScale;
            }
            for (unsigned half = 0; half < 2; ++half) {
                const unsigned t0 = g0 + 16 * half;
                if (t0 >= keys) break;
                const unsigned tn = (keys - t0) < 16 ? (keys - t0) : 16;
                // tile max over the tile's keys (lanes 16 half .. 16 half + tn - 1); comparison only
                float tm = (lane / 16 == half && lane % 16 < tn) ? sc : neg_inf();
#pragma unroll
                for (int d = 16; d >= 1; d >>= 1) tm = fmaxs(tm, shx(tm, d));
                const float mn = fmaxs(m, tm);
                const float corr = first ? 0.0F : exp_f(m - mn);
                l = l * corr;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] = acc[i] * corr;
                const float pl = (lane / 16 == half && lane % 16 < tn) ? exp_f(sc - mn) : 0.0F;
                for (unsigned j = 0; j < tn; ++j) {
                    const float pj = __shfl_sync(0xffffffffU, pl, 16 * half + j, 32);
                    l = l + pj;
                    const uint2 v2 = *reinterpret_cast<const uint2*>(vc + (t0 + j) * 128 + 4 * lane);
                    acc[0] = __fmaf_rn(pj, u2f(v2.x << 16U), acc[0]);
                    acc[1] = __fmaf_rn(pj, u2f(v2.x & 0xffff0000U), acc[1]);
                    acc[2] = __fmaf_rn(pj, u2f(v2.y << 16U), acc[2]);
                    acc[3] = __fmaf_rn(pj, u2f(v2.y & 0xffff0000U), acc[3]);
                }
                m = mn;
                first = false;
            }
        }
        cm[c] = m;
        cl[c] = l;
#pragma unroll
        for (int i = 0; i < 4; ++i) cacc[c][i] = acc[i];
    }
    float big = neg_inf();
    for (unsigned c = 0; c < chunks; ++c) big = fmaxs(big, cm[c]);
    float L = 0.0F, e[kTopK];
    for (unsigned c = 0; c < chunks; ++c) {
        e[c] = exp_f(cm[c] - big);
        L = __fmaf_rn(cl[c], e[c], L);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = 0.0F;
        for (unsigned c = 0; c < chunks; ++c) a = __fmaf_rn(cacc[c][i], e[c], a);
        o[(row * 64 + h) * 128 + 4 * lane + i] = a / L;
    }
}
}  // namespace mmattn_frozen
