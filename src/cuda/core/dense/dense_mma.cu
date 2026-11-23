// Dense rows 1..32 on tensor cores: BF16 mma.sync m16n8k16 (FP32 accumulate). The FP32 activation is split exactly into three
// bf16 terms (hi + mid + lo = x, products with bf16 weights exact in FP32), accumulated lo, mid, hi. Fixed order (independent of
// the row count: rows ride the N dimension, every output element is its own accumulator): the K axis is cut into blocks of 32;
// warp w (of 8) owns groups of U consecutive blocks (U = 4 bf16, 8 q8_0): blocks U(8i+w) .. U(8i+w)+U-1 for i = 0, 1, ... (blocks ascending); inside a block two k16 steps, each with its terms; the 8 warp partials are summed in
// ascending warp order. q8_0: int8 values are exact in bf16; each 32-block partial starts from zero and is folded into the
// running sum as tot = tot + fl(d * part), blocks ascending per warp. Thread-level k permutation: thread q of a quad owns 8
// consecutive k of the 32-block (same permutation for A and B, so the sum is unchanged).
#include "dense_mma.hpp"
#include "rows_bf16_narrow.cuh"
#include "strix/attn/group_indexer.hpp"
#include "strix/core/rows.hpp"
#include "strix/weight/registry.hpp"

#include "../common.cuh"
#include "../step_rows.cuh"

namespace strix::cuda_core {
namespace {

struct MmaArgs {
    const void* w[kMmaMaxMulti];
    float* y[kMmaMaxMulti];
    std::uint32_t out[kMmaMaxMulti];
    std::uint32_t tile_end[kMmaMaxMulti];
};

namespace rows = core::rows;

// Keep the native kernel parameter type and its layout stable.
struct MmaGroupArgs : MmaGroupTable {};
static_assert(sizeof(MmaGroupArgs) == sizeof(MmaGroupTable));

__device__ __forceinline__ void mma16816(float* c, const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// low half = lo_val, high half = hi_val (round to nearest even)
__device__ __forceinline__ std::uint32_t pack_bf16(float lo_val, float hi_val) {
    std::uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi_val), "f"(lo_val));
    return r;
}

// x terms of 8 consecutive k for one activation row: t[term][word] (term 0 = hi, 1 = mid, 2 = lo), word w = elements 2w, 2w+1.
template <int TERMS>
__device__ __forceinline__ void split8(const float* xp, bool valid, std::uint32_t (&t)[3][4]) {
    float v[8];
    if (valid) {
        const float4 a = reinterpret_cast<const float4*>(xp)[0], b = reinterpret_cast<const float4*>(xp)[1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = 0.0f;
    }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        t[0][w] = pack_bf16(v[2 * w], v[2 * w + 1]);
        t[1][w] = t[2][w] = 0;
        if constexpr (TERMS > 1) {
            const float r0 = __fsub_rn(v[2 * w], __uint_as_float(t[0][w] << 16));
            const float r1 = __fsub_rn(v[2 * w + 1], __uint_as_float(t[0][w] & 0xFFFF0000u));
            t[1][w] = pack_bf16(r0, r1);
            if constexpr (TERMS > 2)
                t[2][w] = pack_bf16(__fsub_rn(r0, __uint_as_float(t[1][w] << 16)), __fsub_rn(r1, __uint_as_float(t[1][w] & 0xFFFF0000u)));
        }
    }
}

// Raw weight data of one k32 block for the thread's two rows (g and g + 8).
template <WeightFormat F>
struct WLoad {
    uint4 u[2];             // bf16: 8 elements as 4 words
    std::uint32_t q[2][2];  // q8: 8 int8 as two words
    float d[2];             // q8 scales
};

template <WeightFormat F>
__device__ __forceinline__ void wload(WLoad<F>& l, const void* base, std::uint32_t in, std::uint32_t o0, int g, int qd, std::uint32_t kb,
                                      const unsigned char* sbase, int ub) {
    if constexpr (F == WeightFormat::bf16) {
        const uint4* p = reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(base) + static_cast<std::size_t>(o0 + g) * in) + kb * 4 + qd;
        l.u[0] = __ldg(p);
        l.u[1] = __ldg(p + in);  // row + 8: 8 * in elements = in uint4
    } else {
        const std::size_t rb = static_cast<std::size_t>(in >> 5) * 34;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(
                sbase != nullptr ? sbase + (g + 8 * h) * 272 + ub * 34 : static_cast<const unsigned char*>(base) + (o0 + g + 8 * h) * rb + kb * 34);
            l.d[h] = __half2float(__ushort_as_half(*p));
            const std::uint16_t* q = p + 1 + 4 * qd;
            const std::uint32_t v0 = q[0], v1 = q[1], v2 = q[2], v3 = q[3];
            l.q[h][0] = v0 | (v1 << 16);
            l.q[h][1] = v2 | (v3 << 16);
        }
    }
}

// the 4 words of the thread's 8 elements of row half h
template <WeightFormat F>
__device__ __forceinline__ void a_words(const WLoad<F>& l, int h, std::uint32_t (&w)[4]) {
    if constexpr (F == WeightFormat::bf16) {
        const uint4 u = l.u[h];
        w[0] = u.x; w[1] = u.y; w[2] = u.z; w[3] = u.w;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const std::uint32_t v = l.q[h][i >> 1] >> (16 * (i & 1));
            w[i] = pack_bf16(static_cast<float>(static_cast<std::int8_t>(v & 0xFF)), static_cast<float>(static_cast<std::int8_t>((v >> 8) & 0xFF)));
        }
    }
}

template <WeightFormat F, int NT, int TERMS, int U, bool SplitOutputs = false>
__device__ __forceinline__ void mma_dot(const void* base, std::uint32_t in, std::uint32_t o0, const float* x, std::uint32_t rows, int warp,
                                        int g, int qd, float (&tot)[NT][4], uint4* stg, int lane, bool upper = false) {
    const std::uint32_t nkb = in >> 5;
    const bool staged = F == WeightFormat::q8_0 && (in & 255) == 0;  // rows 16 B aligned: stage 16 rows x 8 blocks (272 B each) through smem
    for (std::uint32_t kb0 = warp * U; kb0 < nkb; kb0 += 8 * U) {
        const unsigned char* sbase = nullptr;
        if constexpr (F == WeightFormat::q8_0) {
            if (staged) {
                const std::size_t rb = static_cast<std::size_t>(nkb) * 34;
                uint4 v[9];
#pragma unroll
                for (int t = 0; t < 9; ++t) {
                    const int j = lane + 32 * t, r = j / 17, c = j % 17;
                    if (j < 272) v[t] = __ldg(reinterpret_cast<const uint4*>(static_cast<const unsigned char*>(base) + (o0 + r) * rb + kb0 * 34) + c);
                }
                __syncwarp();
#pragma unroll
                for (int t = 0; t < 9; ++t)
                    if (lane + 32 * t < 272) stg[lane + 32 * t] = v[t];
                __syncwarp();
                sbase = reinterpret_cast<const unsigned char*>(stg);
            }
        }
        WLoad<F> l[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const std::uint32_t kb = kb0 + u;
            if (kb < nkb) {
                if constexpr (SplitOutputs) {
                    static_assert(F == WeightFormat::bf16);
                    const auto* p = reinterpret_cast<const uint4*>(
                        static_cast<const std::uint16_t*>(base) + static_cast<std::size_t>(o0 + g) * in) + kb * 4 + qd;
                    const uint4 weights = __ldg(p);
                    l[u].u[0] = upper ? uint4{} : weights;
                    l[u].u[1] = upper ? weights : uint4{};
                } else {
                    wload<F>(l[u], base, in, o0, g, qd, kb, sbase, u);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const std::uint32_t kb = kb0 + u;
            if (kb >= nkb) break;
            std::uint32_t aw[2][4];
            a_words<F>(l[u], 0, aw[0]);
            a_words<F>(l[u], 1, aw[1]);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                std::uint32_t t[3][4];
                const std::uint32_t r = nt * 8 + g;
                split8<TERMS>(x + static_cast<std::size_t>(r) * in + kb * 32 + 8 * qd, r < rows, t);
                float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float* c = (F == WeightFormat::bf16) ? tot[nt] : part;
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const std::uint32_t a[4] = {aw[0][2 * s], aw[1][2 * s], aw[0][2 * s + 1], aw[1][2 * s + 1]};
#pragma unroll
                    for (int tt = TERMS - 1; tt >= 0; --tt) mma16816(c, a, t[tt][2 * s], t[tt][2 * s + 1]);
                }
                if constexpr (F != WeightFormat::bf16) {
                    tot[nt][0] = __fadd_rn(tot[nt][0], __fmul_rn(l[u].d[0], part[0]));
                    tot[nt][1] = __fadd_rn(tot[nt][1], __fmul_rn(l[u].d[0], part[1]));
                    tot[nt][2] = __fadd_rn(tot[nt][2], __fmul_rn(l[u].d[1], part[2]));
                    tot[nt][3] = __fadd_rn(tot[nt][3], __fmul_rn(l[u].d[1], part[3]));
                }
            }
        }
    }
}

__device__ __forceinline__ int warp_id_of(int t) { return t >> 5; }

// Keep the arithmetic in each entry's scope. An inlined device function
// changed nvcc's thread-index range analysis and the legacy SASS.
// The staging area is reused for reduction. Element e = nt * 128 + o * 8 + c,
// where o is the output within the tile and c is the activation row within nt.
#define STRIX_MMA_TILE_BODY \
    float a0[NT][4], a1[NT][4]; \
_Pragma("unroll") \
    for (int nt = 0; nt < NT; ++nt) \
_Pragma("unroll") \
        for (int i = 0; i < 4; ++i) a0[nt][i] = a1[nt][i] = 0.0f; \
    pdl_wait(); \
    mma_dot<F, NT, TERMS, U>(wsel, in, o0, x, rows, warp, g, qd, a0, stg, lane); \
    if constexpr (FUSED) { __syncwarp(); mma_dot<F, NT, TERMS, U>(w1, in, o0, x, rows, warp, g, qd, a1, stg, lane); } \
    __syncthreads(); \
_Pragma("unroll") \
    for (int nt = 0; nt < NT; ++nt) { \
        const int e0 = nt * 128 + g * 8 + 2 * qd, e2 = nt * 128 + (g + 8) * 8 + 2 * qd; \
        red[0][warp][e0] = a0[nt][0]; red[0][warp][e0 + 1] = a0[nt][1]; \
        red[0][warp][e2] = a0[nt][2]; red[0][warp][e2 + 1] = a0[nt][3]; \
        if constexpr (FUSED) { \
            red[1][warp][e0] = a1[nt][0]; red[1][warp][e0 + 1] = a1[nt][1]; \
            red[1][warp][e2] = a1[nt][2]; red[1][warp][e2 + 1] = a1[nt][3]; \
        } \
    } \
    __syncthreads(); \
    for (int e = threadIdx.x; e < NT * 128; e += 256) { \
        float s = red[0][0][e], u = 0.0f; \
        if constexpr (FUSED) u = red[1][0][e]; \
_Pragma("unroll") \
        for (int w = 1; w < 8; ++w) { \
            s = __fadd_rn(s, red[0][w][e]); \
            if constexpr (FUSED) u = __fadd_rn(u, red[1][w][e]); \
        } \
        const std::uint32_t r = (e >> 7) * 8 + (e & 7), o = o0 + ((e & 127) >> 3); \
        if (r >= rows) continue; \
        const std::size_t idx = static_cast<std::size_t>(r) * out + o; \
        float v = s; \
        if constexpr (FUSED) v = __fmul_rn(cc_silu(s), u); \
        else if (resid != nullptr) v = __fadd_rn(resid[idx], s); \
        y[idx] = RoundBf16 ? attn::group_indexer::bf16_round(v) : v; \
    }

template <WeightFormat F, int NT, int TERMS, bool FUSED, bool RoundBf16 = false>
__global__ void __launch_bounds__(256) mma_kernel(MmaArgs ma, const void* w1, const float* x, const float* resid,
                                                  std::uint32_t in, std::uint32_t rows) {
#include "rows_mma_body.inc"
}

template <unsigned Width, bool RoundBf16>
__global__ void __launch_bounds__(256) mma_step_kernel(MmaArgs ma, const float* x, unsigned in,
                                                       cuda::core::StepRows<Width> binding) {
    const unsigned row = blockIdx.y;
    if (!binding.valid(row)) return;
    x += std::size_t{row} * in;
    for (unsigned j = 0; j < kMmaMaxMulti; ++j)
        if (ma.y[j]) ma.y[j] += std::size_t{row} * ma.out[j];
    constexpr auto F = WeightFormat::bf16;
    constexpr int NT = 1, TERMS = 3;
    constexpr bool FUSED = false;
    constexpr unsigned rows = 1;
    constexpr const void* w1 = nullptr;
    constexpr const float* resid = nullptr;
#include "rows_mma_body.inc"
}

template <WeightFormat F, int NT>
__global__ void __launch_bounds__(256) mma_kernel(const __grid_constant__ MmaGroupArgs args, std::uint32_t rows) {
    constexpr int U = F == WeightFormat::bf16 ? 4 : 8;
    constexpr int TERMS = 3;
    constexpr bool FUSED = false;
    constexpr const void* w1 = nullptr;
    constexpr std::size_t kRed = 8 * NT * 128 * sizeof(float), kStg = F == WeightFormat::q8_0 ? 8 * 272 * 16 : 0;
    __shared__ __align__(16) unsigned char smem[kRed > kStg ? kRed : kStg];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, qd = lane & 3;
    auto* red = reinterpret_cast<float(*)[8][NT * 128]>(smem);
    auto* stg = reinterpret_cast<uint4*>(smem) + warp * 272;
    pdl_trigger();
    unsigned member;
    std::uint32_t local;
    decode::locate(args.tiles.first, args.tiles.count, blockIdx.x, member, local);
    const auto& job = args.jobs[member];
    const auto* wsel = job.w;
    const auto* x = job.x;
    auto* y = job.y;
    const auto* resid = job.residual;
    const auto in = job.in, out = job.out, o0 = local * core::rows::kTileOut;
    const bool RoundBf16 = F == WeightFormat::bf16 && job.round_bf16;
    STRIX_MMA_TILE_BODY
}

#undef STRIX_MMA_TILE_BODY

#include "rows_bf16_narrow_body.inc"

template <WeightFormat F>
cudaError_t grouped_by_nt(const MmaGroupArgs& args, std::uint32_t row_count, cudaStream_t stream) {
    const dim3 grid(args.tiles.total()), block(256);
    if (row_count <= 8) return launch_pdl(mma_kernel<F, 1>, grid, block, stream, args, row_count);
    if (row_count <= 16) return launch_pdl(mma_kernel<F, 2>, grid, block, stream, args, row_count);
    if (row_count <= 24) return launch_pdl(mma_kernel<F, 3>, grid, block, stream, args, row_count);
    return launch_pdl(mma_kernel<F, 4>, grid, block, stream, args, row_count);
}

template <WeightFormat F, int NT, int TERMS, bool FUSED, bool RoundBf16 = false>
cudaError_t go(const MmaCall& c, const MmaArgs& ma, std::uint32_t tiles, cudaStream_t s) {
    return launch_pdl(mma_kernel<F, NT, TERMS, FUSED, RoundBf16>, dim3(tiles), dim3(256), s, ma, c.w1, c.x, c.residual, c.in, c.rows);
}
template <WeightFormat F, int TERMS, bool FUSED, bool RoundBf16 = false>
cudaError_t by_nt(const MmaCall& c, const MmaArgs& ma, std::uint32_t tiles, cudaStream_t s) {
    if (c.rows <= 8) return go<F, 1, TERMS, FUSED, RoundBf16>(c, ma, tiles, s);
    if (c.rows <= 16) return go<F, 2, TERMS, FUSED, RoundBf16>(c, ma, tiles, s);
    if (c.rows <= 24) return go<F, 3, TERMS, FUSED, RoundBf16>(c, ma, tiles, s);
    return go<F, 4, TERMS, FUSED, RoundBf16>(c, ma, tiles, s);
}
template <WeightFormat F, bool FUSED>
cudaError_t by_terms(const MmaCall& c, const MmaArgs& ma, std::uint32_t tiles, cudaStream_t s) {
    if (c.terms <= 1) return by_nt<F, 1, FUSED>(c, ma, tiles, s);
    if (c.terms == 2) return by_nt<F, 2, FUSED>(c, ma, tiles, s);
    return by_nt<F, 3, FUSED>(c, ma, tiles, s);
}

}  // namespace

template <bool RoundBf16>
static cudaError_t launch_mma(const MmaCall& c, cudaStream_t s) {
    MmaArgs ma{};
    std::uint32_t tiles = 0;
    for (int j = 0; j < c.n; ++j) {
        ma.w[j] = c.w[j];
        ma.y[j] = c.y[j];
        ma.out[j] = c.out[j];
        tiles += c.out[j] / 16;
        ma.tile_end[j] = tiles;
    }
    for (int j = c.n; j < kMmaMaxMulti; ++j) ma.tile_end[j] = tiles;
    if constexpr (RoundBf16) return by_nt<WeightFormat::bf16, 3, false, true>(c, ma, tiles, s);
    const bool fused = c.w1 != nullptr;
    if (c.format == WeightFormat::bf16)
        return fused ? by_terms<WeightFormat::bf16, true>(c, ma, tiles, s) : by_terms<WeightFormat::bf16, false>(c, ma, tiles, s);
    return fused ? by_terms<WeightFormat::q8_0, true>(c, ma, tiles, s) : by_terms<WeightFormat::q8_0, false>(c, ma, tiles, s);
}

bool dense_mma_supported(const MmaCall& c) {
    if (c.format == WeightFormat::f32 || c.in % 32 != 0 || c.rows == 0 || c.rows > 32) return false;
    for (int j = 0; j < c.n; ++j)
        if (c.out[j] % 16 != 0) return false;
    return true;
}

cudaError_t dense_mma_launch(const MmaCall& c, cudaStream_t s) {
    return launch_mma<false>(c, s);
}

cudaError_t dense_bf16_mma3_r1(const MmaGroupTable& args, cudaStream_t stream) {
    if (!valid_bf16_mma3_r1(args)) return cudaErrorInvalidValue;
    return grouped_by_nt<WeightFormat::bf16>(MmaGroupArgs{args}, 1, stream);
}

cudaError_t dense_mma3_rows_grouped(const rows::Plan& plan, cudaStream_t stream) {
    if (plan.launches().empty() || plan.prologue() != 0) return cudaErrorInvalidValue;
    const auto aligned = [](const void* p, std::uintptr_t bytes) {
        return reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
    };
    for (const auto& launch : plan.launches()) {
        const auto& weight = launch.jobs[0].w;
        const bool is_bf16 = weight.has_format(weight::kBf16WeightStorage);
        if (!is_bf16 && !weight.has_format(weight::kQ8WeightStorage)) return cudaErrorInvalidValue;
        for (unsigned j = 0; j < launch.tiles.count; ++j) {
            const auto& job = launch.jobs[j];
            const std::uintptr_t weight_align = is_bf16 || job.in % 256 == 0 ? 16 : 2;
            if (!aligned(job.w.data, weight_align) || !aligned(job.x.data, 16) || !aligned(job.y.data, alignof(float)))
                return cudaErrorInvalidValue;
            if (job.epilogue != 0 &&
                !(job.epilogue == kRowsAddResidual && job.aux[0].data && aligned(job.aux[0].data, alignof(float))) &&
                !(job.epilogue == kRowsRoundBf16 && is_bf16))
                return cudaErrorInvalidValue;
        }
    }

    for (const auto& launch : plan.launches()) {
        const MmaGroupArgs args{mma_group_args(launch)};
        const auto status = launch.jobs[0].w.has_format(weight::kBf16WeightStorage)
            ? grouped_by_nt<WeightFormat::bf16>(args, plan.rows(), stream)
            : grouped_by_nt<WeightFormat::q8_0>(args, plan.rows(), stream);
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

cudaError_t dense_bf16_mma3_rows_round(const DenseWeight* ws, float* const* ys, int n, const float* x,
                                      std::uint32_t rows, cudaStream_t stream) {
    if (!ws || !ys || !x || n < 1 || n > kMmaMaxMulti || ws[0].in == 0U) return cudaErrorInvalidValue;
    MmaCall call{};
    call.n = n;
    call.format = WeightFormat::bf16;
    call.in = ws[0].in;
    call.rows = rows;
    call.terms = 3;
    call.x = x;
    for (int i = 0; i < n; ++i) {
        if (ws[i].format != call.format || ws[i].in != call.in || ws[i].out == 0U || !ws[i].data || !ys[i])
            return cudaErrorInvalidValue;
        call.w[i] = ws[i].data;
        call.y[i] = ys[i];
        call.out[i] = ws[i].out;
    }
    if (!dense_mma_supported(call)) return cudaErrorInvalidValue;
    return launch_mma<true>(call, stream);
}

cudaError_t dense_bf16_mma3_r1_output_split(const DenseWeight& weight, float* output,
                                           const float* input, cudaStream_t stream) {
    const auto blocks = bf16_narrow_blocks(weight.in, weight.out);
    const auto aligned = [](const void* p, std::uintptr_t bytes) {
        return p && reinterpret_cast<std::uintptr_t>(p) % bytes == 0U;
    };
    if (blocks == 0U || weight.format != WeightFormat::bf16 || !aligned(weight.data, 16U) ||
        !aligned(input, 16U) || !aligned(output, alignof(float)) || output == input ||
        weight.out > static_cast<std::size_t>(-1) / sizeof(std::uint16_t) / weight.in)
        return cudaErrorInvalidValue;
    return launch_pdl(mma3_r1_output_split, dim3(blocks), dim3(kBf16NarrowThreads), stream,
                      weight.data, weight.in, input, output);
}

cudaError_t dense_bf16_mma3_step(const float* input, unsigned in,
    std::span<const core::Bf16RowProjection> projections, bool round,
    const exec::StepBlock* descriptor, dim3 grid, cudaStream_t stream) {
    MmaArgs args{};
    unsigned tiles = 0;
    for (std::size_t i = 0; i < projections.size(); ++i) {
        const auto& p = projections[i];
        args.w[i] = p.weights;
        args.y[i] = p.output;
        args.out[i] = p.outputs;
        tiles += p.outputs / core::rows::kTileOut;
        args.tile_end[i] = tiles;
    }
    for (std::size_t i = projections.size(); i < kMmaMaxMulti; ++i) args.tile_end[i] = tiles;
    return cuda::core::dispatch_step_rows(grid.y, [&]<unsigned Width> {
        const cuda::core::StepRows<Width> binding{descriptor};
        if (round) mma_step_kernel<Width, true><<<grid, 256, 0, stream>>>(args, input, in, binding);
        else mma_step_kernel<Width, false><<<grid, 256, 0, stream>>>(args, input, in, binding);
        return cudaGetLastError();
    });
}

}  // namespace strix::cuda_core
