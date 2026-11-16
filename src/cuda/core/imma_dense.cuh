#pragma once
// Exact A8 dense scheduling shared by backend sources. This is the incumbent
// IMMA body, moved once; the FAST F16 mma::Tile retains its distinct arithmetic.
#include "imma_tile.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
namespace strix::cuda::core::dense_a8 {
constexpr int kBM = 128, kBN = 128, kKB = 4, kThreads = 256, kRowWords = 36;
constexpr int kStageA = kBM * kRowWords * 4, kStageB = kBN * kRowWords * 4, kStageD = kKB * kBM * 4;
constexpr int kStage = kStageA + kStageB + kStageD;
constexpr int kSmem = 2 * kStage;

// Empty production observer is erased at compile time. The capture build also
// instantiates the identical arithmetic with register-partial stores.
struct NoCapture {
    __device__ __forceinline__ static void record(unsigned, unsigned, unsigned, float) {}
};
#ifdef STRIX_WFMT_CAPTURE
struct PartialCaptureConfig { float* output; unsigned rows, outputs; };
__device__ PartialCaptureConfig partial_capture_config;
struct PartialCapture {
    __device__ __forceinline__ static void record(unsigned block, unsigned row, unsigned output, float value) {
        const auto c = partial_capture_config;
        if (row < c.rows && output < c.outputs)
            c.output[(static_cast<std::uint64_t>(block) * c.rows + row) * c.outputs + output] = value;
    }
};
#endif

struct KSeg { const unsigned char* w; void* out; std::uint64_t stride; std::uint32_t begin, end; };
template <unsigned MaxSegments> struct Site {
    KSeg seg[MaxSegments];
    std::uint32_t nseg, width, outputs;
    const float* gate;
    std::uint64_t gate_stride;
};

template <class Site>
__device__ __forceinline__ unsigned find_seg(const Site& s, std::uint32_t n) {
    unsigned m = 0;
    while (m + 1U < s.nseg && n >= s.seg[m].end) ++m;
    return m;
}
__device__ __forceinline__ void cp_async4(void* smem, const void* g, bool ok) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(g), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* g, bool ok) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }

__device__ __forceinline__ void imma(int (&d)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
    constexpr int kMagic = 0x4B400000;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(kMagic));
}



template <auto S, class Source, class Numeric, class Capture = NoCapture>
__global__ __launch_bounds__(kThreads) void project_kernel(typename Numeric::Site site, std::uint32_t rows, const std::int8_t* codes,
                                                          const float* deltas, std::uint32_t pitch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int wm = warp & 1, wn = warp >> 1;
    const std::uint32_t m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
    const std::uint32_t width = site.width, nblocks = width / 32U, row_bytes = nblocks * Source::kBlockBytes;
    const std::uint32_t stages = (nblocks + kKB - 1U) / kKB;

    // Weight loader: thread -> output row ln of the tile, half of its words.
    const int ln = tid >> 1, half = tid & 1;
    const std::uint32_t gn = n0 + static_cast<std::uint32_t>(ln);
    const bool nvalid = gn < site.outputs;
    const unsigned char* wrow = site.seg[0].w;
    if (nvalid) {
        const KSeg& sg = site.seg[find_seg(site, gn)];
        wrow = sg.w + static_cast<std::uint64_t>(gn - sg.begin) * row_bytes;
    }
    const auto load_stage = [&](std::uint32_t s, int buf) {
        unsigned char* base = smem + buf * kStage;
        auto* A = reinterpret_cast<std::uint32_t*>(base);
        auto* B = reinterpret_cast<std::uint32_t*>(base + kStageA);
        auto* D = reinterpret_cast<float*>(base + kStageA + kStageB);
        const std::uint32_t kb0 = s * kKB, nb = min(static_cast<std::uint32_t>(kKB), nblocks - kb0);
        const std::uint32_t chunks = nb * 2U;
        for (std::uint32_t i = static_cast<std::uint32_t>(tid); i < kBM * chunks; i += kThreads) {
            const std::uint32_t r = i / chunks, c = i % chunks, gr = m0 + r;
            const bool ok = gr < rows;
            const std::int8_t* src = ok ? codes + static_cast<std::uint64_t>(gr) * width + kb0 * 32U + c * 16U : codes;
            cp_async16(A + r * kRowWords + c * 4U, src, ok);
        }
        const std::uint32_t words = 17U * nb / 2U, h0 = half ? (words + 1U) / 2U : 0U, h1 = half ? words : (words + 1U) / 2U;
        const auto* wsrc = reinterpret_cast<const std::uint32_t*>(wrow + static_cast<std::uint64_t>(kb0) * Source::kBlockBytes);
        for (std::uint32_t w = h0; w < h1; ++w)
            cp_async4(B + ln * kRowWords + w, nvalid ? wsrc + w : reinterpret_cast<const std::uint32_t*>(site.seg[0].w), nvalid);
        for (std::uint32_t i = static_cast<std::uint32_t>(tid); i < nb * kBM; i += kThreads) {
            const std::uint32_t j = i / kBM, r = i % kBM, gr = m0 + r;
            const bool ok = gr < rows;
            cp_async4(D + j * kBM + r, ok ? deltas + static_cast<std::uint64_t>(kb0 + j) * pitch + gr : deltas, ok);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[a][b][e] = 0.0F;

    load_stage(0U, 0);
    cp_async_commit();
    for (std::uint32_t s = 0; s < stages; ++s) {
        if (s + 1U < stages) load_stage(s + 1U, static_cast<int>((s + 1U) & 1U));
        cp_async_commit();
        cp_async_wait1();
        __syncthreads();
        const unsigned char* base = smem + (s & 1U) * kStage;
        const auto* A = reinterpret_cast<const std::uint32_t*>(base);
        const auto* B = reinterpret_cast<const std::uint32_t*>(base + kStageA);
        const auto* D = reinterpret_cast<const float*>(base + kStageA + kStageB);
        const std::uint32_t nb = min(static_cast<std::uint32_t>(kKB), nblocks - s * kKB);
        for (std::uint32_t j = 0; j < nb; ++j) {   // blocks ascending: the HIP chain order
            std::uint32_t a[4][4];
            float dl[4][2];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int r = wm * 64 + mt * 16 + gid;
                a[mt][0] = A[r * kRowWords + j * 8U + tig];
                a[mt][1] = A[(r + 8) * kRowWords + j * 8U + tig];
                a[mt][2] = A[r * kRowWords + j * 8U + 4U + tig];
                a[mt][3] = A[(r + 8) * kRowWords + j * 8U + 4U + tig];
                dl[mt][0] = D[j * kBM + r];
                dl[mt][1] = D[j * kBM + r + 8];
            }
            // Block j of the stage starts at byte 34 j: its scale there, its codes at 34 j + 2 (even j: two bytes into a
            // word, realigned with a byte permute; odd j: word aligned).
            const bool even = (j & 1U) == 0U;
            const std::uint32_t qword = even ? 17U * j / 2U : (17U * j + 1U) / 2U;
            const std::uint32_t sword = even ? 17U * j / 2U : (17U * j - 1U) / 2U;
            std::uint32_t b[4][2];
            float dw[4][2];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                Source::raw136(B + (wn * 32 + nt * 8 + gid) * kRowWords, j, tig, b[nt]);
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const std::uint32_t sw = B[(wn * 32 + nt * 8 + 2 * tig + c) * kRowWords + sword];
                    const unsigned short h = static_cast<unsigned short>(even ? (sw & 0xffffU) : (sw >> 16U));
                    dw[nt][c] = Source::scale_bits(h);   // exact
                }
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int d[4];
                    imma(d, a[mt], b[nt]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float dot = __fsub_rn(__int_as_float(d[e]), 12582912.0F);   // exact float(dot_b)
                        const float p = __fmul_rn(dw[nt][e & 1], dl[mt][e >> 1]);         // v_mul_f32 dw * delta
                        acc[mt][nt][e] = __fmaf_rn(p, dot, acc[mt][nt][e]);               // v_fmac_f32
                        Capture::record(s * kKB + j, m0 + wm * 64 + mt * 16 + gid + 8 * (e >> 1),
                                        n0 + wn * 32 + nt * 8 + 2 * tig + (e & 1), acc[mt][nt][e]);
                    }
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const std::uint32_t row = m0 + static_cast<std::uint32_t>(wm * 64 + mt * 16 + gid + 8 * hh);
            if (row >= rows) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const std::uint32_t n = n0 + static_cast<std::uint32_t>(wn * 32 + nt * 8 + 2 * tig + c);
                    if (n < site.outputs) Numeric::template store<S>(site, row, n, acc[mt][nt][2 * hh + c]);
                }
        }
}

// ---------------------------------------------------------------------------------------------------------------------
// Projection, tile2 (STRIX_QRTX_DENSE_TILE2, bitwise): the same CTA 128 x 128 / warp 64 x 32 / per-(row, output) FP32 chain
// as project_kernel, with the shared-memory traffic and the occupancy reworked. Stage = 2 K blocks (even + odd, so every
// raw Q8_0 offset is static), 45 KB double buffer -> two CTAs per SM (128 regs). A fragments by ldmatrix.x4 (core::imma);
// the 128 + 128 per-stage scales/deltas are staged by the loader in the exact order the warps read them, so a warp reads
// them as 2 + 2 LDS.128 instead of 16 LDS.32. Requires width / 32 even (checked by the launcher).
constexpr int kT2Rw = 20;                                  // row pitch in words (80 B): ldmatrix and LDS conflict-free
constexpr int kT2A = kBM * kT2Rw * 4, kT2B = kBN * kT2Rw * 4, kT2D = 2 * kBM * 4, kT2S = 2 * kBN * 4;
constexpr int kT2Stage = kT2A + kT2B + kT2D + kT2S;
constexpr int kT2Smem = 2 * kT2Stage;

// Remap (lever qwen_dense_t2_remap, bitwise: same bytes into the same shared-memory slots): the weight loader below and
// no 2-CTA register cap. Measured (8192 x 2560 x 16384, sm_120 at 2.6 GHz): 3.148 -> 2.267 ms; without any load the
// kernel runs 1.529 ms, without the FP32 chain 3.040 ms: the stage is bound by the loader, not by the chain or the MMA.
// The old loader (17 lanes per row, 9 pointers hoisted per thread) spilled 112 B under the 128-register cap of
// launch_bounds(256, 2) and reloaded them each stage; one CTA per SM with 249 registers was measured faster than two
// capped CTAs (2.267 vs 2.410 ms) and than a 4-stage ring with one barrier per stage (2.506 ms).
template <auto S, bool Remap, class Source, class Numeric, class Capture = NoCapture>
__global__ __launch_bounds__(kThreads, Remap ? 1 : 2) void project_tile2_kernel(typename Numeric::Site site, std::uint32_t rows, const std::int8_t* codes,
                                                                    const float* deltas, std::uint32_t pitch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int wm = warp & 1, wn = warp >> 1;
    const std::uint32_t m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
    const std::uint32_t width = site.width, nblocks = width / 32U, row_bytes = nblocks * Source::kBlockBytes;
    const std::uint32_t stages = nblocks / 2U;

    // Weight rows: every segment begin is a multiple of kBN (launcher check), so the tile lies in one segment and a row is
    // wbase + row * row_bytes. Raw bytes loader: consecutive lanes take consecutive words of a row (68 B per stage and row,
    // 17 words), so one warp instruction touches 2-3 rows instead of 16. Scale loader: thread -> output tid & 127, block tid >> 7.
    const KSeg& tseg = site.seg[find_seg(site, n0)];
    const unsigned char* wbase = tseg.w + static_cast<std::uint64_t>(n0 - tseg.begin) * row_bytes;
    const std::uint32_t nrows = n0 < site.outputs ? min(static_cast<std::uint32_t>(kBN), site.outputs - n0) : 0U;
    const int sn = tid & 127, sj = tid >> 7;
    const bool svalid = static_cast<std::uint32_t>(sn) < nrows;
    const unsigned char* srow = wbase + static_cast<std::uint64_t>(sn) * row_bytes;
    const std::uint32_t sgr = m0 + static_cast<std::uint32_t>(sn);   // delta row of the scale/delta loaders
    const bool dvalid = sgr < rows;
    // Destination slots: S[j][wn][tig][nt][c] words, D[j][wm][gid][mt * 2 + h] floats (the reader's order).
    const int s_off = (((sj * 4 + (sn >> 5)) * 4 + ((sn >> 1) & 3)) * 4 + ((sn >> 3) & 3)) * 2 + (sn & 1);
    const int d_off = ((sj * 2 + (sn >> 6)) * 8 + (sn & 7)) * 8 + ((sn >> 4) & 3) * 2 + ((sn >> 3) & 1);

    const auto load_stage = [&](std::uint32_t s, int buf) {
        unsigned char* base = smem + buf * kT2Stage;
        auto* A = reinterpret_cast<std::uint32_t*>(base);
        auto* B = reinterpret_cast<std::uint32_t*>(base + kT2A);
        auto* D = reinterpret_cast<float*>(base + kT2A + kT2B);
        auto* SC = reinterpret_cast<std::uint32_t*>(base + kT2A + kT2B + kT2D);
        const std::uint32_t kb0 = s * 2U;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const std::uint32_t i = static_cast<std::uint32_t>(tid + k * kThreads), r = i >> 2, c = i & 3U, gr = m0 + r;
            const bool ok = gr < rows;
            cp_async16(A + r * kT2Rw + c * 4U, ok ? codes + static_cast<std::uint64_t>(gr) * width + kb0 * 32U + c * 16U : codes, ok);
        }
        const unsigned char* wstage = wbase + static_cast<std::uint64_t>(kb0) * Source::kBlockBytes;
        if constexpr (Remap) {
            // 16 lanes per row take words 0..15 (64 contiguous bytes), rows (tid >> 4) + 16 k; word 16 of row tid (tid < 128).
            // Stage-invariant addresses: one row pointer stepped by 16 rows, shared-memory offsets are immediates.
        {
            const std::uint32_t r0 = static_cast<std::uint32_t>(tid) >> 4U, w = static_cast<std::uint32_t>(tid) & 15U;
            const unsigned char* g = wstage + static_cast<std::uint64_t>(r0) * row_bytes + w * 4U;
            std::uint32_t* sb = B + r0 * kT2Rw + w;
            const std::uint64_t step = static_cast<std::uint64_t>(row_bytes) * 16U;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const bool ok = r0 + 16U * k < nrows;
                cp_async4(sb + 16 * kT2Rw * k, ok ? reinterpret_cast<const std::uint32_t*>(g + step * k) : reinterpret_cast<const std::uint32_t*>(site.seg[0].w), ok);
            }
            if (tid < 128) cp_async4(B + sn * kT2Rw + 16, svalid ? reinterpret_cast<const std::uint32_t*>(srow + static_cast<std::uint64_t>(kb0) * Source::kBlockBytes) + 16
                                                                 : reinterpret_cast<const std::uint32_t*>(site.seg[0].w), svalid);
        }
        } else {
        std::uint32_t brow = static_cast<std::uint32_t>(tid) / 17U, bw = static_cast<std::uint32_t>(tid) % 17U;
#pragma unroll
        for (int k = 0; k < 9; ++k) {   // 128 rows x 17 words = 2176 = 8.5 x 256
            if (k < 8 || tid < 128) {
                const bool ok = brow < nrows;
                cp_async4(B + brow * kT2Rw + bw, ok ? reinterpret_cast<const std::uint32_t*>(wstage + static_cast<std::uint64_t>(brow) * row_bytes) + bw
                                                    : reinterpret_cast<const std::uint32_t*>(site.seg[0].w), ok);
            }
            bw += 1U; brow += 15U;   // + 256 = 15 rows + 1 word
            if (bw >= 17U) { bw -= 17U; ++brow; }
        }
        }
        // scale of block kb0 + sj: low half of word 0 (even block) or high half of word 8 (odd block) of the stage's raw bytes
        const auto* ssrc = reinterpret_cast<const std::uint32_t*>(srow + static_cast<std::uint64_t>(kb0) * Source::kBlockBytes) + (sj ? 8 : 0);
        cp_async4(SC + s_off, svalid ? ssrc : reinterpret_cast<const std::uint32_t*>(site.seg[0].w), svalid);
        cp_async4(D + d_off, dvalid ? deltas + static_cast<std::uint64_t>(kb0 + sj) * pitch + sgr : deltas, dvalid);
    };

    float acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[a][b][e] = 0.0F;

    load_stage(0U, 0);
    cp_async_commit();
    for (std::uint32_t s = 0; s < stages; ++s) {
#ifndef T2_NOLOAD
        if (s + 1U < stages) load_stage(s + 1U, static_cast<int>((s + 1U) & 1U));
#endif
        cp_async_commit();
        cp_async_wait1();
        __syncthreads();
        const unsigned char* base = smem + (s & 1U) * kT2Stage;
        const auto* B = reinterpret_cast<const std::uint32_t*>(base + kT2A);
        const auto* D = reinterpret_cast<const float*>(base + kT2A + kT2B);
        const auto* SC = reinterpret_cast<const std::uint32_t*>(base + kT2A + kT2B + kT2D);
#pragma unroll
        for (int j = 0; j < 2; ++j) {   // blocks ascending: the HIP chain order
            std::uint32_t a[4][4];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
                core::imma::load_a_frag(a[mt], base + (wm * 64 + mt * 16) * (kT2Rw * 4) + j * 32, kT2Rw * 4, lane);
            const float4 d0 = *reinterpret_cast<const float4*>(D + ((j * 2 + wm) * 8 + gid) * 8);
            const float4 d1 = *reinterpret_cast<const float4*>(D + ((j * 2 + wm) * 8 + gid) * 8 + 4);
            const float dl[4][2] = {{d0.x, d0.y}, {d0.z, d0.w}, {d1.x, d1.y}, {d1.z, d1.w}};
            const uint4 s0 = *reinterpret_cast<const uint4*>(SC + ((j * 4 + wn) * 4 + tig) * 8);
            const uint4 s1 = *reinterpret_cast<const uint4*>(SC + ((j * 4 + wn) * 4 + tig) * 8 + 4);
            const std::uint32_t sw[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
            std::uint32_t b[4][2];
            float dw[4][2];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const std::uint32_t* W = B + (wn * 32 + nt * 8 + gid) * kT2Rw + tig;
                if (j == 0) { b[nt][0] = __byte_perm(W[0], W[1], 0x5432); b[nt][1] = __byte_perm(W[4], W[5], 0x5432); }
                else { b[nt][0] = W[9]; b[nt][1] = W[13]; }
#pragma unroll
                for (int c = 0; c < 2; ++c) dw[nt][c] = j == 0 ? Source::scale_bits(static_cast<unsigned short>(sw[nt * 2 + c] & 0xffffU)) : Source::scale_bits(static_cast<unsigned short>(sw[nt * 2 + c] >> 16U));
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int d[4];
                    core::imma::mma_seeded(d, a[mt], b[nt]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
#ifdef T2_NOEPI
                        acc[mt][nt][e] = __int_as_float(__float_as_int(acc[mt][nt][e]) ^ d[e]);
#else
                        acc[mt][nt][e] = core::imma::block_acc(d[e], dw[nt][e & 1], dl[mt][e >> 1], acc[mt][nt][e]);
                        Capture::record(s * 2 + j, m0 + wm * 64 + mt * 16 + gid + 8 * (e >> 1),
                                        n0 + wn * 32 + nt * 8 + 2 * tig + (e & 1), acc[mt][nt][e]);
#endif
                    }
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const std::uint32_t row = m0 + static_cast<std::uint32_t>(wm * 64 + mt * 16 + gid + 8 * hh);
            if (row >= rows) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const std::uint32_t n = n0 + static_cast<std::uint32_t>(wn * 32 + nt * 8 + 2 * tig + c);
                    if (n < site.outputs) Numeric::template store<S>(site, row, n, acc[mt][nt][2 * hh + c]);
                }
        }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode projection (rows <= 8): the same per-(row, output) chain acc = fma(dw * delta, float(dot_b), acc) over blocks
// b ascending, laid out for few rows. A quad of lanes owns one output: lane q takes codes 8q..8q+7 of each block and
// holds partial integer dots for all RP rows (RP = rows padded to 1, 2, 4 or 8). The exact group dots are completed by
// integer adds across the quad (order-free): for RP <= 2 a full xor reduction (every lane then runs the chains, lane 0
// stores); for RP >= 4 a transpose-reduce (xor 2 then xor 1, each level keeping half of the rows), so lane q finishes
// rows (q >> 1) * RP/2 + (q & 1) * RP/4 .. + RP/4 and runs only their chains. Activations sit in shared memory.
constexpr unsigned kRowsThreads = 128, kRowsQuads = kRowsThreads / 4;
template <auto S, unsigned R, unsigned RP, class Source, class Numeric, class Capture = NoCapture>
__global__ __launch_bounds__(kRowsThreads) void project_rows_kernel(typename Numeric::Site site, const std::int8_t* codes, const float* deltas,
                                                                   std::uint32_t pitch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t width = site.width, nblocks = width / 32U, row_bytes = nblocks * Source::kBlockBytes;
    auto* A = reinterpret_cast<std::uint32_t*>(smem);                        // [RP][width / 4]
    auto* D = reinterpret_cast<float*>(smem + RP * width);                   // [RP][nblocks]
    for (std::uint32_t i = threadIdx.x; i < RP * width / 4U; i += kRowsThreads) {
        const std::uint32_t r = i / (width / 4U), w = i % (width / 4U);
        A[i] = r < R ? reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::uint64_t>(r) * width)[w] : 0U;
    }
    for (std::uint32_t i = threadIdx.x; i < RP * nblocks; i += kRowsThreads)
        D[i] = i / nblocks < R ? deltas[static_cast<std::uint64_t>(i % nblocks) * pitch + i / nblocks] : 0.0F;
    __syncthreads();
    const unsigned q = threadIdx.x & 3U;
    const std::uint32_t n = blockIdx.x * kRowsQuads + threadIdx.x / 4U;
    const bool valid = n < site.outputs;
    const unsigned char* wrow = site.seg[0].w;
    if (valid) {
        const KSeg& sg = site.seg[find_seg(site, n)];
        wrow = sg.w + static_cast<std::uint64_t>(n - sg.begin) * row_bytes;
    }
    constexpr unsigned OWN = RP >= 4U ? RP / 4U : RP;                        // rows whose chains this lane runs
    const unsigned first = RP >= 4U ? (q >> 1U) * (RP / 2U) + (q & 1U) * (RP / 4U) : 0U;
    float acc[OWN];
#pragma unroll
    for (unsigned r = 0; r < OWN; ++r) acc[r] = 0.0F;
#pragma unroll 4
    for (std::uint32_t b = 0; b < nblocks; ++b) {
        const auto* h = reinterpret_cast<const unsigned short*>(wrow + Source::kBlockBytes * b);   // 2-byte aligned
        const float dw = Source::scale_bits(valid ? h[0] : static_cast<unsigned short>(0));
        const unsigned short* c = h + 1U + 4U * q;
        const std::uint32_t w0 = valid ? Source::word(c, 0) : 0U;
        const std::uint32_t w1 = valid ? Source::word(c, 1) : 0U;
        int dot[RP];
#pragma unroll
        for (unsigned r = 0; r < RP; ++r) {
            const std::uint32_t* a = A + r * (width / 4U) + b * 8U + 2U * q;
            dot[r] = __dp4a(static_cast<int>(w1), static_cast<int>(a[1]), __dp4a(static_cast<int>(w0), static_cast<int>(a[0]), 0));
        }
        if constexpr (RP >= 4U) {
            const bool hi2 = (q & 2U) != 0U, hi1 = (q & 1U) != 0U;
#pragma unroll
            for (unsigned i = 0; i < RP / 2U; ++i) {
                const int keep = hi2 ? dot[i + RP / 2U] : dot[i], send = hi2 ? dot[i] : dot[i + RP / 2U];
                dot[i] = keep + __shfl_xor_sync(0xffffffffU, send, 2);
            }
#pragma unroll
            for (unsigned i = 0; i < RP / 4U; ++i) {
                const int keep = hi1 ? dot[i + RP / 4U] : dot[i], send = hi1 ? dot[i] : dot[i + RP / 4U];
                dot[i] = keep + __shfl_xor_sync(0xffffffffU, send, 1);
            }
        } else {
#pragma unroll
            for (unsigned r = 0; r < RP; ++r) {
                dot[r] += __shfl_xor_sync(0xffffffffU, dot[r], 1);
                dot[r] += __shfl_xor_sync(0xffffffffU, dot[r], 2);
            }
        }
#pragma unroll
        for (unsigned i = 0; i < OWN; ++i) {
            const float p = __fmul_rn(dw, D[(first + i) * nblocks + b]);    // v_mul_f32 dw * delta
            acc[i] = __fmaf_rn(p, static_cast<float>(dot[i]), acc[i]);      // v_fmac_f32 (float(dot) exact, |dot| < 2^24)
            if (RP >= 4U || q == 0U) Capture::record(b, first + i, n, acc[i]);
        }
    }
    if (!valid || (RP < 4U && q != 0U)) return;
#pragma unroll
    for (unsigned i = 0; i < OWN; ++i)
        if (first + i < R) Numeric::template store<S>(site, first + i, n, acc[i]);
}

template <auto S, unsigned R, class Source, class Numeric, class Capture = NoCapture> cudaError_t launch_rows_r(cudaStream_t stream, const typename Numeric::Site& k, const std::int8_t* codes,
                                                        const float* deltas, std::uint32_t pitch) {
    constexpr unsigned RP = R <= 1U ? 1U : R <= 2U ? 2U : R <= 4U ? 4U : 8U;
    const std::uint32_t smem = RP * k.width + RP * (k.width / 32U) * 4U;
    static std::once_flag once;
    static cudaError_t attr = cudaSuccess;
    std::call_once(once, [] {
        attr = cudaFuncSetAttribute(project_rows_kernel<S, R, RP, Source, Numeric, Capture>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 6144 + 8 * 192 * 4);
    });
    if (attr != cudaSuccess) return attr;
    project_rows_kernel<S, R, RP, Source, Numeric, Capture><<<(k.outputs + kRowsQuads - 1U) / kRowsQuads, kRowsThreads, smem, stream>>>(k, codes, deltas, pitch);
    return cudaGetLastError();
}
template <auto S, class Source, class Numeric, class Capture = NoCapture> cudaError_t launch_rows(cudaStream_t stream, const typename Numeric::Site& k, std::uint32_t rows, const std::int8_t* codes,
                                           const float* deltas, std::uint32_t pitch) {
    switch (rows) {
    case 1: return launch_rows_r<S, 1, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 2: return launch_rows_r<S, 2, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 3: return launch_rows_r<S, 3, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 4: return launch_rows_r<S, 4, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 5: return launch_rows_r<S, 5, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 6: return launch_rows_r<S, 6, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 7: return launch_rows_r<S, 7, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    default: return launch_rows_r<S, 8, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode projection, K-split form (rows <= 8; default, STRIX_QRTX_ROWS_KSPLIT=0 = project_rows_kernel). Same chain per
// (row, output) as project_rows_kernel: acc = fma(dw * delta, float(dot_b), acc) over blocks b ascending, so the outputs
// are bit-identical. Only the integer block dots (exact, order-free) are computed K-parallel: warp w owns blocks
// 8w .. 8w + 7 (quad j = block 8w + j, lane q = codes 8q .. 8q + 7 as above) and keeps their activation codes in
// registers, then walks the CTA's kKsOut outputs reading 272 contiguous weight bytes per output (the old mapping read 8
// rows per warp instruction) and leaves the finished dots in shared memory. One thread per (output, row) then runs the
// FP32 chain in block order. A CTA covers kKsOut outputs, so 2,560-output projections fill the card (320 CTAs, was 80)
// and the activations are read once per CTA from L1 instead of being staged per 32 outputs.
constexpr unsigned kKsOut = 8;

template <auto S, unsigned R, unsigned RP, class Source, class Numeric, class Capture = NoCapture>
__global__ __launch_bounds__(1024) void project_rows_ks_kernel(typename Numeric::Site site, const std::int8_t* codes, const float* deltas,
                                                              std::uint32_t pitch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t width = site.width, nblocks = width / 32U, row_bytes = nblocks * Source::kBlockBytes;
    auto* DOT = reinterpret_cast<int*>(smem);                                         // [nblocks][kKsOut][RP]
    auto* DW = reinterpret_cast<float*>(smem + nblocks * kKsOut * RP * 4U);          // [nblocks][kKsOut]
    auto* D = DW + nblocks * kKsOut;                                                  // [nblocks][RP]
    for (std::uint32_t i = threadIdx.x; i < RP * nblocks; i += blockDim.x)
        D[i] = i % RP < R ? deltas[static_cast<std::uint64_t>(i / RP) * pitch + i % RP] : 0.0F;
    const unsigned lane = threadIdx.x & 31U, q = lane & 3U;
    const std::uint32_t blk = (threadIdx.x >> 5U) * 8U + (lane >> 2U);
    const bool active = blk < nblocks;
    std::uint32_t a0[RP], a1[RP];
#pragma unroll
    for (unsigned r = 0; r < RP; ++r) {
        const auto* a = reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::uint64_t>(r) * width) + blk * 8U + 2U * q;
        a0[r] = active && r < R ? a[0] : 0U;
        a1[r] = active && r < R ? a[1] : 0U;
    }
    constexpr unsigned OWN = RP >= 4U ? RP / 4U : RP;
    const unsigned first = RP >= 4U ? (q >> 1U) * (RP / 2U) + (q & 1U) * (RP / 4U) : 0U;
    const std::uint32_t n0 = blockIdx.x * kKsOut;
#pragma unroll 2
    for (unsigned o = 0; o < kKsOut; ++o) {
        const std::uint32_t n = n0 + o;
        const bool valid = active && n < site.outputs;
        const unsigned char* wrow = site.seg[0].w;
        if (n < site.outputs) {
            const KSeg& sg = site.seg[find_seg(site, n)];
            wrow = sg.w + static_cast<std::uint64_t>(n - sg.begin) * row_bytes;
        }
        const auto* h = reinterpret_cast<const unsigned short*>(wrow + Source::kBlockBytes * (active ? blk : 0U));
        const float dw = Source::scale_bits(valid ? h[0] : static_cast<unsigned short>(0));
        const unsigned short* c = h + 1U + 4U * q;
        const std::uint32_t w0 = valid ? Source::word(c, 0) : 0U;
        const std::uint32_t w1 = valid ? Source::word(c, 1) : 0U;
        int dot[RP];
#pragma unroll
        for (unsigned r = 0; r < RP; ++r)
            dot[r] = __dp4a(static_cast<int>(w1), static_cast<int>(a1[r]), __dp4a(static_cast<int>(w0), static_cast<int>(a0[r]), 0));
        if constexpr (RP >= 4U) {
            const bool hi2 = (q & 2U) != 0U, hi1 = (q & 1U) != 0U;
#pragma unroll
            for (unsigned i = 0; i < RP / 2U; ++i) {
                const int keep = hi2 ? dot[i + RP / 2U] : dot[i], send = hi2 ? dot[i] : dot[i + RP / 2U];
                dot[i] = keep + __shfl_xor_sync(0xffffffffU, send, 2);
            }
#pragma unroll
            for (unsigned i = 0; i < RP / 4U; ++i) {
                const int keep = hi1 ? dot[i + RP / 4U] : dot[i], send = hi1 ? dot[i] : dot[i + RP / 4U];
                dot[i] = keep + __shfl_xor_sync(0xffffffffU, send, 1);
            }
        } else {
#pragma unroll
            for (unsigned r = 0; r < RP; ++r) {
                dot[r] += __shfl_xor_sync(0xffffffffU, dot[r], 1);
                dot[r] += __shfl_xor_sync(0xffffffffU, dot[r], 2);
            }
        }
        if (active && (RP >= 4U || q == 0U)) {
            int* out = DOT + (blk * kKsOut + o) * RP + first;
#pragma unroll
            for (unsigned i = 0; i < OWN; ++i) out[i] = dot[i];
            if (q == 0U) DW[blk * kKsOut + o] = dw;
        }
    }
    __syncthreads();
    for (std::uint32_t t = threadIdx.x; t < kKsOut * R; t += blockDim.x) {
        const std::uint32_t o = t / R, r = t % R, n = n0 + o;
        if (n >= site.outputs) continue;
        float acc = 0.0F;
        for (std::uint32_t b = 0; b < nblocks; ++b) {
            const float p = __fmul_rn(DW[b * kKsOut + o], D[b * RP + r]);                       // v_mul_f32 dw * delta
            acc = __fmaf_rn(p, static_cast<float>(DOT[(b * kKsOut + o) * RP + r]), acc);        // v_fmac_f32
            Capture::record(b, r, n, acc);
        }
        Numeric::template store<S>(site, r, n, acc);
    }
}

template <auto S, unsigned R, class Source, class Numeric, class Capture = NoCapture> cudaError_t launch_rows_ks_r(cudaStream_t stream, const typename Numeric::Site& k, const std::int8_t* codes,
                                                           const float* deltas, std::uint32_t pitch) {
    constexpr unsigned RP = R <= 1U ? 1U : R <= 2U ? 2U : R <= 4U ? 4U : 8U;
    const std::uint32_t nb = k.width / 32U, smem = nb * (kKsOut * RP + kKsOut + RP) * 4U;
    static std::once_flag once;
    static cudaError_t attr = cudaSuccess;
    std::call_once(once, [] {
        attr = cudaFuncSetAttribute(project_rows_ks_kernel<S, R, RP, Source, Numeric, Capture>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    256 * (kKsOut * 8 + kKsOut + 8) * 4);
    });
    if (attr != cudaSuccess) return attr;
    project_rows_ks_kernel<S, R, RP, Source, Numeric, Capture><<<(k.outputs + kKsOut - 1U) / kKsOut, 32U * ((nb + 7U) / 8U), smem, stream>>>(k, codes, deltas, pitch);
    return cudaGetLastError();
}
template <auto S, class Source, class Numeric, class Capture = NoCapture> cudaError_t launch_rows_ks(cudaStream_t stream, const typename Numeric::Site& k, std::uint32_t rows, const std::int8_t* codes,
                                              const float* deltas, std::uint32_t pitch) {
    switch (rows) {
    case 1: return launch_rows_ks_r<S, 1, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 2: return launch_rows_ks_r<S, 2, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 3: return launch_rows_ks_r<S, 3, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 4: return launch_rows_ks_r<S, 4, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 5: return launch_rows_ks_r<S, 5, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 6: return launch_rows_ks_r<S, 6, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    case 7: return launch_rows_ks_r<S, 7, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    default: return launch_rows_ks_r<S, 8, Source, Numeric, Capture>(stream, k, codes, deltas, pitch);
    }
}

template <class Site>
inline bool segments_tile_aligned(const Site& k) {
    for (std::uint32_t m = 0; m < k.nseg; ++m)
        if (k.seg[m].begin % static_cast<std::uint32_t>(kBN) != 0U) return false;
    return true;
}

template <auto S, class Source, class Numeric, class Capture = NoCapture> cudaError_t launch_project(cudaStream_t stream, const typename Numeric::Site& k, std::uint32_t rows, const std::int8_t* codes,
                                              const float* deltas, std::uint32_t pitch) {
    if (rows <= 8U && k.width <= 6144U && Numeric::decode_rows_enabled())
        return Numeric::rows_ksplit_enabled() ? launch_rows_ks<S, Source, Numeric, Capture>(stream, k, rows, codes, deltas, pitch)
                                     : launch_rows<S, Source, Numeric, Capture>(stream, k, rows, codes, deltas, pitch);
    const dim3 grid((k.outputs + kBN - 1U) / kBN, (rows + kBM - 1U) / kBM);
    if (Numeric::dense_tile2_enabled() && (k.width / 32U) % 2U == 0U && segments_tile_aligned(k)) {
        static std::once_flag once2;
        static cudaError_t attr2 = cudaSuccess;
        std::call_once(once2, [] {
            attr2 = cudaFuncSetAttribute(project_tile2_kernel<S, true, Source, Numeric, Capture>, cudaFuncAttributeMaxDynamicSharedMemorySize, kT2Smem);
            if (attr2 == cudaSuccess)
                attr2 = cudaFuncSetAttribute(project_tile2_kernel<S, false, Source, Numeric, Capture>, cudaFuncAttributeMaxDynamicSharedMemorySize, kT2Smem);
        });
        if (attr2 != cudaSuccess) return attr2;
        if (Numeric::dense_t2_remap_enabled()) project_tile2_kernel<S, true, Source, Numeric, Capture><<<grid, kThreads, kT2Smem, stream>>>(k, rows, codes, deltas, pitch);
        else project_tile2_kernel<S, false, Source, Numeric, Capture><<<grid, kThreads, kT2Smem, stream>>>(k, rows, codes, deltas, pitch);
        return cudaGetLastError();
    }
    static std::once_flag once;
    static cudaError_t attr = cudaSuccess;
    std::call_once(once, [] { attr = cudaFuncSetAttribute(project_kernel<S, Source, Numeric, Capture>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem); });
    if (attr != cudaSuccess) return attr;
    project_kernel<S, Source, Numeric, Capture><<<grid, kThreads, kSmem, stream>>>(k, rows, codes, deltas, pitch);
    return cudaGetLastError();
}

} // namespace strix::cuda::core::dense_a8
