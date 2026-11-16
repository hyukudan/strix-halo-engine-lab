// CUDA dense A8 projections and shared expert, bit-identical to the HIP dense-a8-v1 prefill. See
// include/strix/cuda/qwen_dense.hpp for the contract; HIP references are given per kernel.
//
// ISA facts relied on (gfx1151 disassembly of the HIP translation units, engine bf60d9d2):
//  quantizer  hip_gdn_launcher.s, _ZN5strix7q8_rows24dense_a8_quantize_kernelINS_12_GLOBAL__N_116GdnInputRowsSiteILj256EEEEEvT_PaPf
//             (:257360): v_cmp_class_f32 0x207 (non-finite), v_max3_f32 |x| from 0, DPP row_xmask:1/2 quad max/or,
//             tiny = v_cmp_lt 0 / v_cmp_gt 0x4000000 (2^-119) (:257498-257499), IEEE v_div_scale/fmas/fixup amax/127
//             (:257598-257613) and x/delta (:257543-257558), v_rndne_f32 (:257560), v_maxmin_f32 -127/127 (:257561),
//             v_cvt_i32_f32 (:257563). float_denorm_mode_32 3 (IEEE denormals) in every kernel descriptor.
//  projection hip_gdn_launcher.s, ..project_rows_dense_a8_wmma_lds2_kernelINS_..GdnInputRowsSiteILj256EEELj8ELb0E.. (:274436):
//             v_wmma_i32_16x16x16_iu8 from the 0x4B400000 seed (:274851-274858), v_add_f32 0xcb400000 (exact float(dot)),
//             v_mul_f32 dw*delta, v_fmac_f32 acc += p*dot (:274854-274892), v_cndmask 0x7fc00000 canonical NaN (:279359...).
//  shared down store hip_shared_expert_launcher.s (:114776-114786): asm v_mul_f32 gate, value; then BF16 (v_add3 0x7fff).
//  activation hip_shared_expert_launcher.s, _ZN5strix12_GLOBAL__N_136rows_shared_activation_scalar_kernelEmmmmPKvjj (:165-462):
//             expf(-g) inlined (v_mul -log2e (:202), v_fma, v_fmamk lo, v_rndne, v_sub, v_add, v_exp, v_ldexp (:214),
//             selects 0x42ce8ed0 / 0xc2b17218) == amd_expf(-g) by sign symmetry; v_add 1.0 (:220); IEEE div g/den
//             (:221-236); v_mul u*q (:238); scalar dot v_dual_fmac_f32 sum = w * x + sum over features tid, tid+256, ...
//             ascending (:281); LDS tree strides 128..1, v_add_f32 (:301-401); 1/(1 + expf(-s)) (:416-457, IEEE div).
#include "strix/env_flags.hpp"
#include "strix/cuda/qwen_dense.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strix::cuda::qwen_dense {
namespace {

// STRIX_QRTX_DECODE_KERNELS=0: keep the tiled kernels at rows <= 8 too (reference seam for the decode-shaped ones).
bool decode_rows_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_DECODE_KERNELS", true);
    return on;
}

constexpr int kBM = 128, kBN = 128, kKB = 4, kThreads = 256, kRowWords = 36;
constexpr int kStageA = kBM * kRowWords * 4, kStageB = kBN * kRowWords * 4, kStageD = kKB * kBM * 4;
constexpr int kStage = kStageA + kStageB + kStageD;
constexpr int kSmem = 2 * kStage;

__device__ __forceinline__ std::uint32_t bf16_bits(float v) {
    const std::uint32_t b = __float_as_uint(v);
    return (b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U;   // HIP float_to_bf16, no NaN special case
}
__device__ __forceinline__ float bf16_float(std::uint32_t h) { return __uint_as_float(h << 16U); }

// v_mul_f32 under IEEE mode: a NaN operand is returned quieted (src0 first) [assumption on RDNA3 NaN propagation];
// CUDA would return 0x7fffffff, whose BF16 rounding is 0x8000 (-0), so the NaN path is emulated explicitly.
__device__ __forceinline__ float amd_mul(float a, float b) {
    if (a != a) return __uint_as_float(__float_as_uint(a) | 0x00400000U);
    if (b != b) return __uint_as_float(__float_as_uint(b) | 0x00400000U);
    return __fmul_rn(a, b);
}

// ---------------------------------------------------------------------------------------------------------------------
// Quantizer: HIP dense_a8_quantize_kernel / dense_a8_quantize8 (src/hip_q8_rows_gemv.hpp :1107-1144). Four threads per
// 32-feature block, 8 features each; the quad max/or is order-free (fmaxf, |).
template <Input I>
__global__ __launch_bounds__(256) void quantize_kernel(std::uint32_t rows, std::uint32_t width, const void* input,
                                                       std::uint64_t stride, std::int8_t* codes, float* deltas,
                                                       std::uint32_t pitch) {
    const std::uint32_t groups = width / 8U;
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
    const bool valid = item < static_cast<std::uint64_t>(rows) * groups;
    const std::uint32_t r = valid ? static_cast<std::uint32_t>(item / groups) : 0U;
    const std::uint32_t g = valid ? static_cast<std::uint32_t>(item % groups) : 0U;
    float x[8];
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) {
        const std::uint64_t at = static_cast<std::uint64_t>(r) * stride + 8U * g + l;
        if (!valid) x[l] = 0.0F;
        else if constexpr (I == Input::bf16) x[l] = bf16_float(static_cast<const std::uint16_t*>(input)[at]);
        else x[l] = static_cast<const float*>(input)[at];
    }
    int bad = 0;
    float amax = 0.0F;
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) { bad |= !isfinite(x[l]); amax = fmaxf(amax, fabsf(x[l])); }
    bad |= __shfl_xor_sync(0xffffffffU, bad, 1); bad |= __shfl_xor_sync(0xffffffffU, bad, 2);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 1)); amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 2));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = bad ? __uint_as_float(kCanonicalNan) : (amax == 0.0F || tiny) ? 1.0F : __fdiv_rn(amax, 127.0F);
    unsigned word[2] = {0U, 0U};
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) {
        const float rounded = bad || tiny ? 0.0F : rintf(__fdiv_rn(x[l], delta));
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        word[l / 4U] |= (static_cast<unsigned>(code) & 0xffU) << (8U * (l % 4U));
    }
    if (!valid) return;
    *reinterpret_cast<uint2*>(codes + static_cast<std::uint64_t>(r) * width + 8U * g) = make_uint2(word[0], word[1]);
    if (!(g & 3U)) deltas[static_cast<std::uint64_t>(g / 4U) * pitch + r] = delta;
}

// ---------------------------------------------------------------------------------------------------------------------
// Projection. CTA = 128 rows x 128 outputs, 8 warps (2 x 4), warp = 64 rows x 32 outputs = 4 x 4 IMMA m16n8k32 tiles.
// K = 32 = one Q8_0 block per IMMA, so the IMMA result is exactly dot_b (seeded with 0x4B400000 like the HIP WMMA; then
// float(dot) = bits - 1.5*2^23, one exact subtraction). Every thread owns 64 (row, output) accumulators and walks the
// blocks in ascending order: the HIP chain acc = fma(dw * delta, float(dot), acc), unchanged by tiling.
// Pipeline: stages of 4 blocks (codes 128 x 128 B, raw Q8_0 bytes 128 x 136 B, deltas 4 x 128), cp.async double buffer.
struct KSeg { const unsigned char* w; void* out; std::uint64_t stride; std::uint32_t begin, end; };
struct KSite {
    KSeg seg[kMaxSegments];
    std::uint32_t nseg, width, outputs;
    const float* gate;
    std::uint64_t gate_stride;
};

__device__ __forceinline__ unsigned find_seg(const KSite& s, std::uint32_t n) {
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

template <Store S>
__device__ __forceinline__ void store_value(const KSite& site, std::uint32_t row, std::uint32_t n, float acc) {
    const float y = acc != acc ? __uint_as_float(kCanonicalNan) : acc;   // dense_a8_canonical (:1101)
    const KSeg& sg = site.seg[find_seg(site, n)];
    const std::uint64_t at = static_cast<std::uint64_t>(row) * sg.stride + (n - sg.begin);
    if constexpr (S == Store::f32) static_cast<float*>(sg.out)[at] = y;
    else if constexpr (S == Store::f32_round_bf16) static_cast<float*>(sg.out)[at] = bf16_float(bf16_bits(y));
    else if constexpr (S == Store::bf16) static_cast<std::uint16_t*>(sg.out)[at] = static_cast<std::uint16_t>(bf16_bits(y));
    else {
        const float gated = amd_mul(site.gate[static_cast<std::uint64_t>(row) * site.gate_stride], y);
        static_cast<std::uint16_t*>(sg.out)[at] = static_cast<std::uint16_t>(bf16_bits(gated));
    }
}

template <Store S>
__global__ __launch_bounds__(kThreads) void project_kernel(KSite site, std::uint32_t rows, const std::int8_t* codes,
                                                          const float* deltas, std::uint32_t pitch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int wm = warp & 1, wn = warp >> 1;
    const std::uint32_t m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
    const std::uint32_t width = site.width, nblocks = width / 32U, row_bytes = nblocks * 34U;
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
        const auto* wsrc = reinterpret_cast<const std::uint32_t*>(wrow + static_cast<std::uint64_t>(kb0) * 34U);
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
                const std::uint32_t* W = B + (wn * 32 + nt * 8 + gid) * kRowWords + qword + tig;
                if (even) { b[nt][0] = __byte_perm(W[0], W[1], 0x5432); b[nt][1] = __byte_perm(W[4], W[5], 0x5432); }
                else { b[nt][0] = W[0]; b[nt][1] = W[4]; }
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const std::uint32_t sw = B[(wn * 32 + nt * 8 + 2 * tig + c) * kRowWords + sword];
                    const unsigned short h = static_cast<unsigned short>(even ? (sw & 0xffffU) : (sw >> 16U));
                    dw[nt][c] = __half2float(__ushort_as_half(h));   // exact
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
                    if (n < site.outputs) store_value<S>(site, row, n, acc[mt][nt][2 * hh + c]);
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
template <Store S, unsigned R, unsigned RP>
__global__ __launch_bounds__(kRowsThreads) void project_rows_kernel(KSite site, const std::int8_t* codes, const float* deltas,
                                                                   std::uint32_t pitch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t width = site.width, nblocks = width / 32U, row_bytes = nblocks * 34U;
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
        const auto* h = reinterpret_cast<const unsigned short*>(wrow + 34U * b);   // 2-byte aligned
        const float dw = __half2float(__ushort_as_half(valid ? h[0] : static_cast<unsigned short>(0)));
        const unsigned short* c = h + 1U + 4U * q;
        const std::uint32_t w0 = valid ? static_cast<std::uint32_t>(c[0]) | (static_cast<std::uint32_t>(c[1]) << 16U) : 0U;
        const std::uint32_t w1 = valid ? static_cast<std::uint32_t>(c[2]) | (static_cast<std::uint32_t>(c[3]) << 16U) : 0U;
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
        }
    }
    if (!valid || (RP < 4U && q != 0U)) return;
#pragma unroll
    for (unsigned i = 0; i < OWN; ++i)
        if (first + i < R) store_value<S>(site, first + i, n, acc[i]);
}

template <Store S, unsigned R> cudaError_t launch_rows_r(cudaStream_t stream, const KSite& k, const std::int8_t* codes,
                                                        const float* deltas, std::uint32_t pitch) {
    constexpr unsigned RP = R <= 1U ? 1U : R <= 2U ? 2U : R <= 4U ? 4U : 8U;
    const std::uint32_t smem = RP * k.width + RP * (k.width / 32U) * 4U;
    static std::once_flag once;
    static cudaError_t attr = cudaSuccess;
    std::call_once(once, [] {
        attr = cudaFuncSetAttribute(project_rows_kernel<S, R, RP>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 6144 + 8 * 192 * 4);
    });
    if (attr != cudaSuccess) return attr;
    project_rows_kernel<S, R, RP><<<(k.outputs + kRowsQuads - 1U) / kRowsQuads, kRowsThreads, smem, stream>>>(k, codes, deltas, pitch);
    return cudaGetLastError();
}
template <Store S> cudaError_t launch_rows(cudaStream_t stream, const KSite& k, std::uint32_t rows, const std::int8_t* codes,
                                           const float* deltas, std::uint32_t pitch) {
    switch (rows) {
    case 1: return launch_rows_r<S, 1>(stream, k, codes, deltas, pitch);
    case 2: return launch_rows_r<S, 2>(stream, k, codes, deltas, pitch);
    case 3: return launch_rows_r<S, 3>(stream, k, codes, deltas, pitch);
    case 4: return launch_rows_r<S, 4>(stream, k, codes, deltas, pitch);
    case 5: return launch_rows_r<S, 5>(stream, k, codes, deltas, pitch);
    case 6: return launch_rows_r<S, 6>(stream, k, codes, deltas, pitch);
    case 7: return launch_rows_r<S, 7>(stream, k, codes, deltas, pitch);
    default: return launch_rows_r<S, 8>(stream, k, codes, deltas, pitch);
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
bool rows_ksplit_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_ROWS_KSPLIT", true);
    return on;
}
template <Store S, unsigned R, unsigned RP, unsigned KO>
__global__ __launch_bounds__(1024) void project_rows_ks_kernel(KSite site, const std::int8_t* codes, const float* deltas,
                                                              std::uint32_t pitch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t width = site.width, nblocks = width / 32U, row_bytes = nblocks * 34U;
    auto* DOT = reinterpret_cast<int*>(smem);                                         // [nblocks][KO][RP]
    auto* DW = reinterpret_cast<float*>(smem + nblocks * KO * RP * 4U);          // [nblocks][KO]
    auto* D = DW + nblocks * KO;                                                  // [nblocks][RP]
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
    const std::uint32_t n0 = blockIdx.x * KO;
#pragma unroll 2
    for (unsigned o = 0; o < KO; ++o) {
        const std::uint32_t n = n0 + o;
        const bool valid = active && n < site.outputs;
        const unsigned char* wrow = site.seg[0].w;
        if (n < site.outputs) {
            const KSeg& sg = site.seg[find_seg(site, n)];
            wrow = sg.w + static_cast<std::uint64_t>(n - sg.begin) * row_bytes;
        }
        const auto* h = reinterpret_cast<const unsigned short*>(wrow + 34U * (active ? blk : 0U));
        const float dw = __half2float(__ushort_as_half(valid ? h[0] : static_cast<unsigned short>(0)));
        const unsigned short* c = h + 1U + 4U * q;
        const std::uint32_t w0 = valid ? static_cast<std::uint32_t>(c[0]) | (static_cast<std::uint32_t>(c[1]) << 16U) : 0U;
        const std::uint32_t w1 = valid ? static_cast<std::uint32_t>(c[2]) | (static_cast<std::uint32_t>(c[3]) << 16U) : 0U;
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
            int* out = DOT + (blk * KO + o) * RP + first;
#pragma unroll
            for (unsigned i = 0; i < OWN; ++i) out[i] = dot[i];
            if (q == 0U) DW[blk * KO + o] = dw;
        }
    }
    __syncthreads();
    for (std::uint32_t t = threadIdx.x; t < KO * R; t += blockDim.x) {
        const std::uint32_t o = t / R, r = t % R, n = n0 + o;
        if (n >= site.outputs) continue;
        float acc = 0.0F;
        for (std::uint32_t b = 0; b < nblocks; ++b) {
            const float p = __fmul_rn(DW[b * KO + o], D[b * RP + r]);                       // v_mul_f32 dw * delta
            acc = __fmaf_rn(p, static_cast<float>(DOT[(b * KO + o) * RP + r]), acc);        // v_fmac_f32
        }
        store_value<S>(site, r, n, acc);
    }
}

template <Store S, unsigned R, unsigned KO> cudaError_t launch_rows_ks_o(cudaStream_t stream, const KSite& k, const std::int8_t* codes,
                                                           const float* deltas, std::uint32_t pitch) {
    constexpr unsigned RP = R <= 1U ? 1U : R <= 2U ? 2U : R <= 4U ? 4U : 8U;
    const std::uint32_t nb = k.width / 32U, smem = nb * (KO * RP + KO + RP) * 4U;
    static std::once_flag once;
    static cudaError_t attr = cudaSuccess;
    std::call_once(once, [] {
        attr = cudaFuncSetAttribute(project_rows_ks_kernel<S, R, RP, KO>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    256 * (8 * 8 + 8 + 8) * 4);
    });
    if (attr != cudaSuccess) return attr;
    project_rows_ks_kernel<S, R, RP, KO><<<(k.outputs + KO - 1U) / KO, 32U * ((nb + 7U) / 8U), smem, stream>>>(k, codes, deltas, pitch);
    return cudaGetLastError();
}
template <Store S, unsigned R> cudaError_t launch_rows_ks_r(cudaStream_t stream, const KSite& k, const std::int8_t* codes,
                                                           const float* deltas, std::uint32_t pitch) {
    static const int ko = [] { const char* e = strix::envcfg::text("STRIX_QRTX_KSO"); return e ? std::atoi(e) : 0; }();
    const std::uint32_t nb = k.width / 32U;
    const int pick = ko ? ko : (nb >= 128U ? 4 : 8);
    return pick == 4 ? launch_rows_ks_o<S, R, 4>(stream, k, codes, deltas, pitch)
         : pick == 2 ? launch_rows_ks_o<S, R, 2>(stream, k, codes, deltas, pitch)
                     : launch_rows_ks_o<S, R, 8>(stream, k, codes, deltas, pitch);
}
template <Store S> cudaError_t launch_rows_ks(cudaStream_t stream, const KSite& k, std::uint32_t rows, const std::int8_t* codes,
                                              const float* deltas, std::uint32_t pitch) {
    switch (rows) {
    case 1: return launch_rows_ks_r<S, 1>(stream, k, codes, deltas, pitch);
    case 2: return launch_rows_ks_r<S, 2>(stream, k, codes, deltas, pitch);
    case 3: return launch_rows_ks_r<S, 3>(stream, k, codes, deltas, pitch);
    case 4: return launch_rows_ks_r<S, 4>(stream, k, codes, deltas, pitch);
    case 5: return launch_rows_ks_r<S, 5>(stream, k, codes, deltas, pitch);
    case 6: return launch_rows_ks_r<S, 6>(stream, k, codes, deltas, pitch);
    case 7: return launch_rows_ks_r<S, 7>(stream, k, codes, deltas, pitch);
    default: return launch_rows_ks_r<S, 8>(stream, k, codes, deltas, pitch);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode projection, streamed form (rows <= 8; default, STRIX_QRTX_ROWS_CP=0 = project_rows_ks_kernel). Same integer dots
// and the same per-(output,row) FP32 chain as the K-split kernel (so the bits are identical); only the data movement
// changes: persistent CTAs keep their activation codes in registers, and the weight rows of OUT outputs per group are
// streamed with 16-byte cp.async through an NST-deep shared ring (rows are nb * 34 bytes, nb % 8 == 0 -> 16-byte
// multiples, the host checks the segment bases), so several groups of weight bytes are in flight at all times. DOT/DW are
// double buffered: the chain of group i - 1 (warp 0) overlaps the dot phase of group i.
// Selection (rtxmax2 receipts, qwen_rows_glue_exact_test --time on the RTX PRO 6000, us/call old ks -> cp):
//   small sites win (shared gate/up 2.7 MB 5.65 -> 3.76 at 1 row, shared down 1.4 MB 3.60 -> 2.98), large sites lose
//   (gdn in 44 MB 36.1 -> 43.6 at 8 rows, gdn out 16.7 MB 15.2 -> 20.2 at 1 row) except K = 6144 at 8 rows (24.3 -> 22.5).
//   STRIX_QRTX_ROWS_CP: unset/"auto" = by site (weight bytes <= STRIX_QRTX_ROWS_CP_MB, default 4, or K >= 6144 at 8 rows),
//   1 = every fitting site, 0 = never (old K-split kernel everywhere). Bits are identical in every mode.
int rows_cp_mode() {
    static const int mode = [] {
        const char* e = strix::envcfg::text("STRIX_QRTX_ROWS_CP");
        if (e == nullptr || *e == 0 || std::strcmp(e, "auto") == 0) return 2;
        return std::atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}
bool rows_cp_enabled() { return rows_cp_mode() != 0; }
bool rows_cp_wanted(const KSite& k, unsigned rows) {
    const int mode = rows_cp_mode();
    if (mode != 2) return mode == 1;
    static const double limit = [] {
        const char* e = strix::envcfg::text("STRIX_QRTX_ROWS_CP_MB");
        return (e != nullptr && *e != 0 ? std::atof(e) : 4.0) * 1048576.0;
    }();
    const double bytes = static_cast<double>(k.outputs) * (k.width / 32U) * 34.0;
    return bytes <= limit || (rows >= 8U && k.width >= 6144U);
}
template <Store S, unsigned R, unsigned RP, unsigned OUT>
__global__ __launch_bounds__(1024) void project_rows_cp_kernel(KSite site, const std::int8_t* codes, const float* deltas,
                                                              std::uint32_t pitch, std::uint32_t nst) {
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t width = site.width, nb = width / 32U, row_bytes = nb * 34U, stage_bytes = OUT * row_bytes;
    auto* DOT = reinterpret_cast<int*>(smem + nst * stage_bytes);       // [2][nb][OUT][RP]
    auto* DW = reinterpret_cast<float*>(DOT + 2U * nb * OUT * RP);      // [2][nb][OUT]
    auto* D = DW + 2U * nb * OUT;                                       // [nb][RP]
    for (std::uint32_t i = threadIdx.x; i < RP * nb; i += blockDim.x)
        D[i] = i % RP < R ? deltas[static_cast<std::uint64_t>(i / RP) * pitch + i % RP] : 0.0F;
    const unsigned lane = threadIdx.x & 31U, q = lane & 3U;
    const std::uint32_t blk = (threadIdx.x >> 5U) * 8U + (lane >> 2U);
    const bool active = blk < nb;
    std::uint32_t a0[RP], a1[RP];
#pragma unroll
    for (unsigned r = 0; r < RP; ++r) {
        const auto* a = reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::uint64_t>(r) * width) + blk * 8U + 2U * q;
        a0[r] = active && r < R ? a[0] : 0U;
        a1[r] = active && r < R ? a[1] : 0U;
    }
    constexpr unsigned OWN = RP >= 4U ? RP / 4U : RP;
    const unsigned first = RP >= 4U ? (q >> 1U) * (RP / 2U) + (q & 1U) * (RP / 4U) : 0U;
    const std::uint32_t groups = (site.outputs + OUT - 1U) / OUT;
    const std::uint32_t ng = (groups - blockIdx.x + gridDim.x - 1U) / gridDim.x;
    const std::uint32_t cpr = row_bytes / 16U, chunks = OUT * cpr;

    const auto issue = [&](std::uint32_t i) {
        if (i < ng) {
            const std::uint32_t n0 = (blockIdx.x + i * gridDim.x) * OUT;
            unsigned char* dst = smem + (i % nst) * stage_bytes;
            for (std::uint32_t c = threadIdx.x; c < chunks; c += blockDim.x) {
                const std::uint32_t o = c / cpr, n = n0 + o;
                const bool ok = n < site.outputs;
                const unsigned char* src = site.seg[0].w;
                if (ok) {
                    const KSeg& sg = site.seg[find_seg(site, n)];
                    src = sg.w + static_cast<std::uint64_t>(n - sg.begin) * row_bytes + (c - o * cpr) * 16U;
                }
                cp_async16(dst + c * 16U, src, ok);
            }
        }
        cp_async_commit();
    };
    const auto chain = [&](std::uint32_t i) {
        const std::uint32_t t = threadIdx.x, buf = i & 1U;
        if (t >= OUT * R) return;
        const std::uint32_t o = t / R, r = t % R, n = (blockIdx.x + i * gridDim.x) * OUT + o;
        if (n >= site.outputs) return;
        const float* dw = DW + buf * nb * OUT + o;
        const int* dt = DOT + buf * nb * OUT * RP + o * RP + r;
        float acc = 0.0F;
#pragma unroll 8
        for (std::uint32_t b = 0; b < nb; ++b) {
            const float p = __fmul_rn(dw[b * OUT], D[b * RP + r]);                  // v_mul_f32 dw * delta
            acc = __fmaf_rn(p, static_cast<float>(dt[b * OUT * RP]), acc);          // v_fmac_f32
        }
        store_value<S>(site, r, n, acc);
    };

    for (std::uint32_t i = 0; i + 1U < nst; ++i) issue(i);
    for (std::uint32_t i = 0; i < ng; ++i) {
        if (nst >= 3U) asm volatile("cp.async.wait_group 1;\n" ::);
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        issue(i + nst - 1U);
        const std::uint32_t buf = i & 1U;
        const unsigned char* W = smem + (i % nst) * stage_bytes;
        int dot[OUT][RP];
        float dwv[OUT];
#pragma unroll
        for (unsigned o = 0; o < OUT; ++o) {
            const auto* h = reinterpret_cast<const unsigned short*>(W + o * row_bytes + 34U * (active ? blk : 0U));
            dwv[o] = __half2float(__ushort_as_half(active ? h[0] : static_cast<unsigned short>(0)));
            const unsigned short* c = h + 1U + 4U * q;
            const std::uint32_t w0 = active ? static_cast<std::uint32_t>(c[0]) | (static_cast<std::uint32_t>(c[1]) << 16U) : 0U;
            const std::uint32_t w1 = active ? static_cast<std::uint32_t>(c[2]) | (static_cast<std::uint32_t>(c[3]) << 16U) : 0U;
#pragma unroll
            for (unsigned r = 0; r < RP; ++r)
                dot[o][r] = __dp4a(static_cast<int>(w1), static_cast<int>(a1[r]), __dp4a(static_cast<int>(w0), static_cast<int>(a0[r]), 0));
        }
#pragma unroll
        for (unsigned o = 0; o < OUT; ++o) {
            if constexpr (RP >= 4U) {
                const bool hi2 = (q & 2U) != 0U, hi1 = (q & 1U) != 0U;
#pragma unroll
                for (unsigned k = 0; k < RP / 2U; ++k) {
                    const int keep = hi2 ? dot[o][k + RP / 2U] : dot[o][k], send = hi2 ? dot[o][k] : dot[o][k + RP / 2U];
                    dot[o][k] = keep + __shfl_xor_sync(0xffffffffU, send, 2);
                }
#pragma unroll
                for (unsigned k = 0; k < RP / 4U; ++k) {
                    const int keep = hi1 ? dot[o][k + RP / 4U] : dot[o][k], send = hi1 ? dot[o][k] : dot[o][k + RP / 4U];
                    dot[o][k] = keep + __shfl_xor_sync(0xffffffffU, send, 1);
                }
            } else {
#pragma unroll
                for (unsigned r = 0; r < RP; ++r) {
                    dot[o][r] += __shfl_xor_sync(0xffffffffU, dot[o][r], 1);
                    dot[o][r] += __shfl_xor_sync(0xffffffffU, dot[o][r], 2);
                }
            }
            if (active && (RP >= 4U || q == 0U)) {
                int* out = DOT + ((buf * nb + blk) * OUT + o) * RP + first;
#pragma unroll
                for (unsigned k = 0; k < OWN; ++k) out[k] = dot[o][k];
                if (q == 0U) DW[(buf * nb + blk) * OUT + o] = dwv[o];
            }
        }
        if (i > 0U) chain(i - 1U);
    }
    __syncthreads();
    if (ng) chain(ng - 1U);
}

template <Store S, unsigned R, unsigned OUT> cudaError_t launch_rows_cp_o(cudaStream_t stream, const KSite& k, const std::int8_t* codes,
                                                                          const float* deltas, std::uint32_t pitch) {
    constexpr unsigned RP = R <= 1U ? 1U : R <= 2U ? 2U : R <= 4U ? 4U : 8U;
    const std::uint32_t nb = k.width / 32U, threads = 32U * ((nb + 7U) / 8U), stage = OUT * nb * 34U;
    const std::uint32_t fixed = (2U * nb * OUT * RP + 2U * nb * OUT + nb * RP) * 4U;
    constexpr std::uint32_t kBudget = 99U * 1024U;
    const std::uint32_t nst = fixed + 3U * stage <= kBudget ? 3U : 2U;
    const std::uint32_t smem = fixed + nst * stage;
    static std::once_flag once;
    static cudaError_t attr = cudaSuccess;
    std::call_once(once, [] {
        attr = cudaFuncSetAttribute(project_rows_cp_kernel<S, R, RP, OUT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024 - 1024);
    });
    if (attr != cudaSuccess) return attr;
    int device = 0, sms = 0, per_sm = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, project_rows_cp_kernel<S, R, RP, OUT>, static_cast<int>(threads), smem);
    cudaGetLastError();
    if (per_sm < 1) per_sm = 1;
    const std::uint32_t groups = (k.outputs + OUT - 1U) / OUT;
    const std::uint32_t grid = std::min<std::uint32_t>(groups, static_cast<std::uint32_t>(sms * per_sm));
    project_rows_cp_kernel<S, R, RP, OUT><<<grid, threads, smem, stream>>>(k, codes, deltas, pitch, nst);
    return cudaGetLastError();
}
template <Store S, unsigned R> cudaError_t launch_rows_cp_r(cudaStream_t stream, const KSite& k, const std::int8_t* codes,
                                                           const float* deltas, std::uint32_t pitch) {
    return k.width / 32U <= 96U ? launch_rows_cp_o<S, R, 4>(stream, k, codes, deltas, pitch)
                                : launch_rows_cp_o<S, R, 2>(stream, k, codes, deltas, pitch);
}
template <Store S> cudaError_t launch_rows_cp(cudaStream_t stream, const KSite& k, std::uint32_t rows, const std::int8_t* codes,
                                              const float* deltas, std::uint32_t pitch) {
    switch (rows) {
    case 1: return launch_rows_cp_r<S, 1>(stream, k, codes, deltas, pitch);
    case 2: return launch_rows_cp_r<S, 2>(stream, k, codes, deltas, pitch);
    case 3: return launch_rows_cp_r<S, 3>(stream, k, codes, deltas, pitch);
    case 4: return launch_rows_cp_r<S, 4>(stream, k, codes, deltas, pitch);
    case 5: return launch_rows_cp_r<S, 5>(stream, k, codes, deltas, pitch);
    case 6: return launch_rows_cp_r<S, 6>(stream, k, codes, deltas, pitch);
    case 7: return launch_rows_cp_r<S, 7>(stream, k, codes, deltas, pitch);
    default: return launch_rows_cp_r<S, 8>(stream, k, codes, deltas, pitch);
    }
}
bool rows_cp_fits(const KSite& k) {
    if (!rows_cp_enabled() || (k.width / 32U) % 8U != 0U || k.width > 6144U) return false;
    for (std::uint32_t m = 0; m < k.nseg; ++m)
        if ((reinterpret_cast<std::uintptr_t>(k.seg[m].w) & 15U) != 0U) return false;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// rows_shared_activation_scalar_kernel (src/hip_shared_expert_launcher.hip:535-562), 256 threads, blockIdx.y = row.
__global__ __launch_bounds__(256) void shared_activation_kernel(float* workspace_base, std::uint64_t workspace_stride,
                                                                const std::uint16_t* input_base, std::uint64_t input_stride,
                                                                const float* scalar, std::uint32_t intermediate,
                                                                std::uint32_t hidden, amdmath::Tables tables) {
    __shared__ float scratch[256];
    float* workspace = workspace_base + blockIdx.y * workspace_stride;
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < intermediate) {
        const float gate = workspace[index];
        const float up = workspace[intermediate + index];
        const float den = __fadd_rn(1.0F, amdmath::amd_expf(tables, -gate));
        workspace[index] = __fmul_rn(__fdiv_rn(gate, den), up);
    }
    if (blockIdx.x != 0) return;
    const std::uint16_t* input = input_base + blockIdx.y * input_stride;
    float sum = 0.0F;
    for (std::uint32_t feature = threadIdx.x; feature < hidden; feature += 256U)
        sum = __fmaf_rn(scalar[feature], bf16_float(input[feature]), sum);   // v_dual_fmac_f32 (contracted in the ISA)
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = 128U; stride > 0U; stride >>= 1U) {   // HIP :555-558, v_add_f32 upper + lower
        if (threadIdx.x < stride) scratch[threadIdx.x] = __fadd_rn(scratch[threadIdx.x + stride], scratch[threadIdx.x]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        workspace[intermediate] = __fdiv_rn(1.0F, __fadd_rn(1.0F, amdmath::amd_expf(tables, -scratch[0])));
}

template <Store S> cudaError_t launch_project(cudaStream_t stream, const KSite& k, std::uint32_t rows, const std::int8_t* codes,
                                              const float* deltas, std::uint32_t pitch) {
    if (rows <= 8U && k.width <= 6144U && decode_rows_enabled())
        return rows_ksplit_enabled() ? (rows_cp_fits(k) && rows_cp_wanted(k, rows) ? launch_rows_cp<S>(stream, k, rows, codes, deltas, pitch)
                                                        : launch_rows_ks<S>(stream, k, rows, codes, deltas, pitch))
                                     : launch_rows<S>(stream, k, rows, codes, deltas, pitch);
    static std::once_flag once;
    static cudaError_t attr = cudaSuccess;
    std::call_once(once, [] { attr = cudaFuncSetAttribute(project_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem); });
    if (attr != cudaSuccess) return attr;
    const dim3 grid((k.outputs + kBN - 1U) / kBN, (rows + kBM - 1U) / kBM);
    project_kernel<S><<<grid, kThreads, kSmem, stream>>>(k, rows, codes, deltas, pitch);
    return cudaGetLastError();
}

}  // namespace

cudaError_t quantize(cudaStream_t stream, std::uint32_t rows, std::uint32_t width, Input type, const void* input,
                     std::uint64_t input_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch) {
    if (!rows) return cudaSuccess;
    if (width == 0U || width % 64U != 0U || pitch < rows || !input || !codes || !deltas) return cudaErrorInvalidValue;
    const std::uint64_t items = static_cast<std::uint64_t>(rows) * (width / 8U);
    const unsigned blocks = static_cast<unsigned>((items + 255U) / 256U);
    if (type == Input::bf16) quantize_kernel<Input::bf16><<<blocks, 256, 0, stream>>>(rows, width, input, input_stride, codes, deltas, pitch);
    else quantize_kernel<Input::f32><<<blocks, 256, 0, stream>>>(rows, width, input, input_stride, codes, deltas, pitch);
    return cudaGetLastError();
}

cudaError_t project(cudaStream_t stream, const Site& site, std::uint32_t rows, const std::int8_t* codes, const float* deltas,
                    std::uint32_t pitch) {
    if (!rows) return cudaSuccess;
    if (site.width == 0U || site.width % 64U != 0U || site.segments == 0U || site.segments > kMaxSegments || pitch < rows ||
        (reinterpret_cast<std::uintptr_t>(codes) & 15U) != 0U)
        return cudaErrorInvalidValue;
    if (site.store == Store::bf16_gated && !site.gate) return cudaErrorInvalidValue;
    KSite k{};
    std::uint32_t n = 0;
    for (std::uint32_t m = 0; m < site.segments; ++m) {
        const Segment& s = site.segment[m];
        if (!s.weight || !s.output || !s.outputs || (reinterpret_cast<std::uintptr_t>(s.weight) & 3U) != 0U) return cudaErrorInvalidValue;
        k.seg[m] = {static_cast<const unsigned char*>(s.weight), s.output, s.output_stride, n, n + s.outputs};
        n += s.outputs;
    }
    k.nseg = site.segments; k.width = site.width; k.outputs = n; k.gate = site.gate; k.gate_stride = site.gate_stride;
    switch (site.store) {
    case Store::f32: return launch_project<Store::f32>(stream, k, rows, codes, deltas, pitch);
    case Store::f32_round_bf16: return launch_project<Store::f32_round_bf16>(stream, k, rows, codes, deltas, pitch);
    case Store::bf16: return launch_project<Store::bf16>(stream, k, rows, codes, deltas, pitch);
    case Store::bf16_gated: return launch_project<Store::bf16_gated>(stream, k, rows, codes, deltas, pitch);
    }
    return cudaErrorInvalidValue;
}

cudaError_t quantize_project(cudaStream_t stream, const Site& site, std::uint32_t rows, Input type, const void* input,
                             std::uint64_t input_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch) {
    const cudaError_t e = quantize(stream, rows, site.width, type, input, input_stride, codes, deltas, pitch);
    return e != cudaSuccess ? e : project(stream, site, rows, codes, deltas, pitch);
}

cudaError_t shared_activation(cudaStream_t stream, std::uint32_t rows, float* workspace, std::uint64_t workspace_stride,
                              const std::uint16_t* input, std::uint64_t input_stride, const float* scalar,
                              std::uint32_t intermediate, std::uint32_t hidden, const amdmath::Tables& tables) {
    if (!rows) return cudaSuccess;
    if (!intermediate || workspace_stride < 2ULL * intermediate || !workspace || !input || !scalar) return cudaErrorInvalidValue;
    const dim3 grid((intermediate + 255U) / 256U, rows);
    shared_activation_kernel<<<grid, 256, 0, stream>>>(workspace, workspace_stride, input, input_stride, scalar, intermediate,
                                                       hidden, tables);
    return cudaGetLastError();
}

cudaError_t shared_expert(cudaStream_t stream, std::uint32_t rows, const SharedExpert& w, const std::uint16_t* input,
                          std::uint64_t input_stride, float* workspace, std::uint64_t workspace_stride, std::uint16_t* output,
                          std::uint64_t output_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch,
                          const amdmath::Tables& tables) {
    Site gate_up{};
    gate_up.store = Store::f32;
    gate_up.width = w.hidden;
    gate_up.segments = 2;
    gate_up.segment[0] = {w.gate_q8, w.intermediate, workspace, workspace_stride};
    gate_up.segment[1] = {w.up_q8, w.intermediate, workspace + w.intermediate, workspace_stride};
    cudaError_t e = quantize_project(stream, gate_up, rows, Input::bf16, input, input_stride, codes, deltas, pitch);
    if (e != cudaSuccess) return e;
    e = shared_activation(stream, rows, workspace, workspace_stride, input, input_stride, w.scalar, w.intermediate, w.hidden, tables);
    if (e != cudaSuccess) return e;
    Site down{};
    down.store = Store::bf16_gated;
    down.width = w.intermediate;
    down.segments = 1;
    down.segment[0] = {w.down_q8, w.hidden, output, output_stride};
    down.gate = workspace + w.intermediate;   // the scalar overwrote up[0] (HIP site.gate[r], :1874)
    down.gate_stride = workspace_stride;
    return quantize_project(stream, down, rows, Input::f32, workspace, workspace_stride, codes, deltas, pitch);
}

}  // namespace strix::cuda::qwen_dense
