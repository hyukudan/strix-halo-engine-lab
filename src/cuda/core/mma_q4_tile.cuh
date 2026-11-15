// Shared CUDA core: f16 tensor-core tile GEMM, FP32 activations x Q4_0-style weights, FP32 accumulation (FAST forms only).
//   C[m][n] = sum_kb scale[n][kb] * ( sum_{k in block kb} f16(A[m][k]) * code[n][k] ),  code = nibble - 8 (exact in f16).
// mma.sync m16n8k16 f16 -> f32, A and B fragments through ldmatrix, per-32-K block FP32 scale epilogue (acc += s * block
// partial), so the weights are never rounded; only the activations are rounded to f16 on load.
// Weight block layout (18 B, 2-byte aligned): fp16 d, 16 bytes; byte j low nibble = element j, high nibble = element j + 16.
// Tile: BM = 128 rows x BN = 16 * NT8 outputs per CTA of 256 threads (8 warps: 4 along M x 2 along N, warp tile 32 x 8 * NT8),
// K step 32, register prefetch + double-buffered shared memory. Rows past `rows` are clamped on load (caller masks the store).
// The model file supplies pointers and the epilogue; this header has no model dependency (a B trait for another weight
// format only replaces the W half of load / store).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "ptx.cuh"

namespace strix::cuda::core::mma_q4 {

constexpr int kBM = 128, kBK = 32, kLd = 40;  // smem row stride in halves (80 B: conflict-free ldmatrix)

template <int NT8>
struct Tile {
    static constexpr int kBN = 16 * NT8;
    static_assert(kBN == 128 || kBN == 64, "weight loader: two threads per output row");
    struct Smem {
        __half a[2][kBM * kLd];
        __half w[2][kBN * kLd];
        float s[2][kBN];
    };
    struct Regs { float4 a[4]; unsigned short w[9]; };

    // Global -> registers for K step kb. x: A row base of the stream (row stride ld floats, K offset applied by caller);
    // wq: weight block 0 of the stream for output 0 (row stride row_bytes); n0: first output of this CTA.
    __device__ static __forceinline__ void load(Regs& r, const float* x, std::uint64_t ld, unsigned m0, unsigned rows, int kb,
                                                const std::uint8_t* wq, std::uint64_t row_bytes, unsigned n0) {
        const unsigned t = threadIdx.x, row = t >> 1U, kh = t & 1U;
        const float* p = x + static_cast<std::uint64_t>(min(m0 + row, rows - 1U)) * ld + kb * kBK + kh * 16U;
#pragma unroll
        for (int i = 0; i < 4; ++i) r.a[i] = *reinterpret_cast<const float4*>(p + 4 * i);
        const unsigned nl = (t % (kBN * 2U)) >> 1U;
        const std::uint8_t* b = wq + static_cast<std::uint64_t>(n0 + nl) * row_bytes + static_cast<std::uint64_t>(kb) * 18U;
#pragma unroll
        for (int i = 0; i < 9; ++i) r.w[i] = *reinterpret_cast<const unsigned short*>(b + 2 * i);
    }
    // Registers -> shared (FP32 -> f16 for A; nibbles -> exact f16 codes for W; scale -> FP32 for the epilogue).
    __device__ static __forceinline__ void store(const Regs& r, Smem& sm, int buf) {
        const unsigned t = threadIdx.x, row = t >> 1U, kh = t & 1U;
        uint4 pa[2];
        unsigned* pw = reinterpret_cast<unsigned*>(pa);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            __half2 lo = __floats2half2_rn(r.a[i].x, r.a[i].y), hi = __floats2half2_rn(r.a[i].z, r.a[i].w);
            pw[2 * i] = *reinterpret_cast<unsigned*>(&lo);
            pw[2 * i + 1] = *reinterpret_cast<unsigned*>(&hi);
        }
        uint4* da = reinterpret_cast<uint4*>(&sm.a[buf][row * kLd + kh * 16U]);
        da[0] = pa[0];
        da[1] = pa[1];
        // W: thread t -> output t / 2, half kh: elements kh * 16 .. +15 are nibble kh of bytes 0..15.
        if (t < kBN * 2U) {
            const unsigned nl = t >> 1U;
            uint4 pv[2];
            unsigned* ww = reinterpret_cast<unsigned*>(pv);
            const unsigned sh = kh * 4U;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned u = static_cast<unsigned>(r.w[1 + i]) >> sh;
                const unsigned word = 0x64006400U | (u & 15U) | (((u >> 8U) & 15U) << 16U);
                const __half2 v = __hsub2(*reinterpret_cast<const __half2*>(&word), __half2half2(__ushort_as_half(0x6408)));
                ww[i] = *reinterpret_cast<const unsigned*>(&v);
            }
            uint4* dw = reinterpret_cast<uint4*>(&sm.w[buf][nl * kLd + kh * 16U]);
            dw[0] = pv[0];
            dw[1] = pv[1];
            if (!kh) sm.s[buf][nl] = __half2float(__ushort_as_half(r.w[0]));
        }
    }
    // One K step (2 x k16) on shared buffer `buf`: partial = A * codes in fresh accumulators, acc += scale * partial.
    __device__ static __forceinline__ void compute(float (&acc)[2][NT8][4], const Smem& sm, int buf) {
        namespace px = strix::cuda::core::ptx;
        const unsigned lane = threadIdx.x & 31U, wid = threadIdx.x >> 5U, wm = wid & 3U, wn = wid >> 2U;
        unsigned af[2][2][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                px::ldsm_x4(af[ks][mi], &sm.a[buf][(wm * 32U + mi * 16U + (lane & 15U)) * kLd + ks * 16U + (lane >> 4U) * 8U]);
#pragma unroll
        for (int jp = 0; jp < NT8 / 2; ++jp) {
            unsigned bf[2][4];
            const unsigned nrow = wn * (NT8 * 8U) + jp * 16U + ((lane >> 4U) * 8U) + (lane & 7U);
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
                px::ldsm_x4(bf[ks], &sm.w[buf][nrow * kLd + ks * 16U + ((lane >> 3U) & 1U) * 8U]);
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = jp * 2 + jj;
                const float2 sc = *reinterpret_cast<const float2*>(&sm.s[buf][wn * (NT8 * 8U) + j * 8U + 2U * (lane & 3U)]);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    float t[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks)
                        px::mma_f16(t, af[ks][mi][0], af[ks][mi][1], af[ks][mi][2], af[ks][mi][3], bf[ks][2 * jj], bf[ks][2 * jj + 1]);
                    acc[mi][j][0] = __fmaf_rn(sc.x, t[0], acc[mi][j][0]);
                    acc[mi][j][1] = __fmaf_rn(sc.y, t[1], acc[mi][j][1]);
                    acc[mi][j][2] = __fmaf_rn(sc.x, t[2], acc[mi][j][2]);
                    acc[mi][j][3] = __fmaf_rn(sc.y, t[3], acc[mi][j][3]);
                }
            }
        }
    }
    // acc = sum over `kblocks` K steps of one stream. Ends with a barrier (shared buffers reusable by the next stream).
    __device__ static void run(float (&acc)[2][NT8][4], Smem& sm, const float* x, std::uint64_t ld, unsigned m0, unsigned rows,
                               const std::uint8_t* wq, std::uint64_t row_bytes, unsigned n0, int kblocks) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int j = 0; j < NT8; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.0F;
        Regs r;
        load(r, x, ld, m0, rows, 0, wq, row_bytes, n0);
        store(r, sm, 0);
        __syncthreads();
        for (int kb = 0; kb < kblocks; ++kb) {
            const int cur = kb & 1;
            if (kb + 1 < kblocks) load(r, x, ld, m0, rows, kb + 1, wq, row_bytes, n0);
            compute(acc, sm, cur);
            if (kb + 1 < kblocks) store(r, sm, cur ^ 1);
            __syncthreads();
        }
    }
    // Tile-local output column / row of accumulator element e (0..3) of fragment (mi, j) for this thread.
    __device__ static __forceinline__ unsigned col(int j, int e) {
        return ((threadIdx.x >> 5U) >> 2U) * (NT8 * 8U) + j * 8U + 2U * (threadIdx.x & 3U) + (e & 1);
    }
    __device__ static __forceinline__ unsigned row(int mi, int e) {
        return ((threadIdx.x >> 5U) & 3U) * 32U + mi * 16U + ((threadIdx.x & 31U) >> 2U) + (e >> 1) * 8U;
    }
};

}  // namespace strix::cuda::core::mma_q4
