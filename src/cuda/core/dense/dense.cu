// Dense rows 1..16: BF16 / Q8_0 / F32 weights, FP32 accumulation in the fixed order documented in dense.hpp.
#include "../head_argmax.cuh"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/nvfp4.hpp"

#include "../common.cuh"
#include "dense_mma.hpp"
#include "row_plan.hpp"
#include "rows_bf16_epilogue.cuh"

#include <cmath>

namespace strix::cuda_core {
namespace {

// acc <- acc + w * x in the documented order. With CC_DENSE_FMA the product is not rounded separately (one fused rounding).
__device__ __forceinline__ float mac(float acc, float w, float x) {
#ifdef CC_DENSE_FMA
    return __fmaf_rn(w, x, acc);
#else
    return __fadd_rn(acc, __fmul_rn(w, x));
#endif
}

// ORO consecutive output rows per warp: the x values of a chunk are loaded once and applied to every weight row (identical
// per-(output, row) arithmetic and order as ORO = 1).
template <int R, int ORO>
__device__ __forceinline__ void dot_bf16(const std::uint16_t* wrow, std::size_t row_stride, const float* x, std::uint32_t in,
                                         std::uint32_t rows, int lane, float (&acc)[ORO][R]) {
    const std::uint32_t nch = in >> 3;
#pragma unroll 2
    for (std::uint32_t c = lane; c < nch; c += 32) {
        float w[ORO][8];
#pragma unroll
        for (int oi = 0; oi < ORO; ++oi) {
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(wrow + oi * row_stride) + c);
            w[oi][0] = bf16_lo(u.x); w[oi][1] = bf16_hi(u.x); w[oi][2] = bf16_lo(u.y); w[oi][3] = bf16_hi(u.y);
            w[oi][4] = bf16_lo(u.z); w[oi][5] = bf16_hi(u.z); w[oi][6] = bf16_lo(u.w); w[oi][7] = bf16_hi(u.w);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (static_cast<std::uint32_t>(r) < rows) {
                const float4* xp = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in + c * 8);
                const float4 a = xp[0], b2 = xp[1];
                const float xv[8] = {a.x, a.y, a.z, a.w, b2.x, b2.y, b2.z, b2.w};
#pragma unroll
                for (int oi = 0; oi < ORO; ++oi) {
                    float sacc = acc[oi][r];
#pragma unroll
                    for (int j = 0; j < 8; ++j) sacc = mac(sacc, w[oi][j], xv[j]);
                    acc[oi][r] = sacc;
                }
            }
        }
    }
}

template <int R>
__device__ __forceinline__ void dot_f32(const float* wrow, const float* x, std::uint32_t in, std::uint32_t rows, int lane,
                                        float (&acc)[R]) {
    const std::uint32_t nch = in >> 3;
    for (std::uint32_t c = lane; c < nch; c += 32) {
        const float4 wa = reinterpret_cast<const float4*>(wrow)[c * 2], wb = reinterpret_cast<const float4*>(wrow)[c * 2 + 1];
        const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (static_cast<std::uint32_t>(r) < rows) {
                const float4* xp = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in + c * 8);
                const float4 a = xp[0], b = xp[1];
                const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                float s = acc[r];
#pragma unroll
                for (int j = 0; j < 8; ++j) s = __fadd_rn(s, __fmul_rn(w[j], xv[j]));
                acc[r] = s;
            }
        }
    }
}

template <int R>
__device__ __forceinline__ void dot_q8(const unsigned char* wrow, const float* x, std::uint32_t in, std::uint32_t rows,
                                       int lane, float (&acc)[R]) {
    const std::uint32_t nb = in >> 5;
    for (std::uint32_t b = lane; b < nb; b += 32) {
        const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(wrow + static_cast<std::size_t>(b) * 34);
        const float d = __half2float(__ushort_as_half(__ldg(p)));
        float q[32];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const std::uint32_t v = __ldg(p + 1 + j);
            q[2 * j] = static_cast<float>(static_cast<std::int8_t>(v & 0xFF));
            q[2 * j + 1] = static_cast<float>(static_cast<std::int8_t>(v >> 8));
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (static_cast<std::uint32_t>(r) < rows) {
                const float4* xp = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in + b * 32);
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float4 a = xp[k];
                    s = __fadd_rn(s, __fmul_rn(q[4 * k], a.x));
                    s = __fadd_rn(s, __fmul_rn(q[4 * k + 1], a.y));
                    s = __fadd_rn(s, __fmul_rn(q[4 * k + 2], a.z));
                    s = __fadd_rn(s, __fmul_rn(q[4 * k + 3], a.w));
                }
                acc[r] = __fadd_rn(acc[r], __fmul_rn(d, s));
            }
        }
    }
}

template <WeightFormat F, int R, int ORO>
__device__ __forceinline__ void dot_any(const void* w, std::uint32_t o, const float* x, std::uint32_t in, std::uint32_t rows,
                                        int lane, float (&acc)[ORO][R]) {
    if constexpr (F == WeightFormat::bf16) {
        dot_bf16<R, ORO>(static_cast<const std::uint16_t*>(w) + static_cast<std::size_t>(o) * in, in, x, in, rows, lane, acc);
    } else {
        static_assert(ORO == 1, "only bf16 has multi-row warps");
        float(&a1)[R] = acc[0];
        if constexpr (F == WeightFormat::f32)
            dot_f32<R>(static_cast<const float*>(w) + static_cast<std::size_t>(o) * in, x, in, rows, lane, a1);
        else
            dot_q8<R>(static_cast<const unsigned char*>(w) + static_cast<std::size_t>(o) * (in >> 5) * 34, x, in, rows, lane, a1);
    }
}

// Up to four weights sharing x (and format/in) are served by ONE launch: warp g of the grid owns global output row g, mapped to
// (weight j, row o). The per-row arithmetic does not depend on the mapping.
constexpr int kMaxMulti = 4;
constexpr int kWarpsPerBlock = 8;  // 128-thread blocks: finer grid -> less wave-quantisation tail on 188 SMs
struct MultiArgs {
    const void* w[kMaxMulti];
    float* y[kMaxMulti];
    std::uint32_t out[kMaxMulti];
    std::uint32_t row_end[kMaxMulti];  // cumulative rows
    int n;
};

// L2 prefetch of the first 4 KiB of this warp's weight row while the predecessor kernel is still running (weights do not depend on it).
template <WeightFormat F>
__device__ __forceinline__ void prefetch_row(const void* w, std::uint32_t o, std::uint32_t in, int lane) {
    const std::size_t row_bytes = F == WeightFormat::bf16 ? static_cast<std::size_t>(in) * 2
                                  : F == WeightFormat::f32 ? static_cast<std::size_t>(in) * 4
                                                           : static_cast<std::size_t>(in >> 5) * 34;
    const char* p = static_cast<const char*>(w) + static_cast<std::size_t>(o) * row_bytes + static_cast<std::size_t>(lane) * 128;
    if (static_cast<std::size_t>(lane) * 128 < row_bytes) asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}


template <WeightFormat F, int R, bool FUSED, int ORO, bool RowGrid = false>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    dense_kernel(MultiArgs ma, const void* w1, const float* x, const float* resid, std::uint32_t in, std::uint32_t rows) {
#define STRIX_DENSE_STORE(value) (value)
#include "rows_body.inc"
#undef STRIX_DENSE_STORE
}

template <Bf16Fp32Store Store>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    dense_bf16_fp32_r1_store(MultiArgs ma, const float* x, std::uint32_t in, float cap) {
    constexpr auto F = WeightFormat::bf16;
    constexpr int R = 1, ORO = 1;
    constexpr bool FUSED = false, RowGrid = false;
    constexpr const void* w1 = nullptr;
    constexpr const float* resid = nullptr;
    std::uint32_t rows = 1;
    const Bf16Store<Store> store{cap};
#define STRIX_DENSE_STORE(value) store(value)
#include "rows_body.inc"
#undef STRIX_DENSE_STORE
}

// Mapped-rows trait of the same R=1 family (screened greedy head, pass 2): the identical rows_body.inc, so every output
// runs the incumbent instruction sequence (bf16 chunk loads, the documented mac order, warp_sum_xor, Bf16Store); only the
// group -> row map differs. Group g computes row ids[g] (all rows 0..rows-1 when *count > capacity) of the head and stores
// y[g]. Warps loop over the groups, so any grid gives the same y. ids / count are the predecessor's output: wait first.
template <Bf16Fp32Store Store>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    dense_bf16_fp32_r1_mapped(const std::uint16_t* weights, const float* x, float* ystore, const std::uint32_t* ids,
                              const std::uint32_t* count_ptr, std::uint32_t capacity, std::uint32_t head_rows,
                              std::uint32_t in, float cap) {
    constexpr auto F = WeightFormat::bf16;
    constexpr int R = 1, ORO = 1;
    constexpr bool FUSED = false, RowGrid = false;
    constexpr const void* w1 = nullptr;
    constexpr const float* resid = nullptr;
    std::uint32_t rows = 1;
    pdl_trigger();
    pdl_wait();
    const std::uint32_t count = *count_ptr;
    const bool all = count > capacity;
    const std::uint32_t limit = all ? head_rows : count;
    MultiArgs ma{};
    ma.n = 1;
    ma.w[0] = weights;
    ma.y[0] = ystore;
    ma.out[0] = limit;
    for (auto& end : ma.row_end) end = limit;
    const Bf16Store<Store> store{cap};
    const std::uint32_t stride = gridDim.x * kWarpsPerBlock;
    for (std::uint32_t group = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);; group += stride) {
#define STRIX_DENSE_MAPPED
#define STRIX_DENSE_MAPPED_GROUP group
#define STRIX_DENSE_MAPPED_DONE break
#define STRIX_DENSE_MAPPED_ROW(g) (all ? (g) : ids[(g)])
#define STRIX_DENSE_STORE(value) store(value)
#include "rows_body.inc"
#undef STRIX_DENSE_STORE
#undef STRIX_DENSE_MAPPED_ROW
#undef STRIX_DENSE_MAPPED_DONE
#undef STRIX_DENSE_MAPPED_GROUP
#undef STRIX_DENSE_MAPPED
    }
}

template <WeightFormat F, int R, bool FUSED, int ORO, bool RowGrid = false>
cudaError_t launch(const MultiArgs& ma, const void* w1, const float* x, const float* resid, std::uint32_t in, std::uint32_t rows,
                   cudaStream_t s) {
    const std::uint32_t total = ma.row_end[kMaxMulti - 1];
    if constexpr (RowGrid) {
        const dim3 grid((total - 1) / kWarpsPerBlock + 1, (rows - 1) / R + 1);
        return launch_pdl(dense_kernel<F, R, FUSED, ORO, true>, grid, dim3(kWarpsPerBlock * 32), s,
                          ma, w1, x, resid, in, rows);
    } else {
        return launch_pdl(dense_kernel<F, R, FUSED, ORO>, dim3((total + kWarpsPerBlock - 1) / kWarpsPerBlock),
                          dim3(kWarpsPerBlock * 32), s, ma, w1, x, resid, in, rows);
    }
}

// Output rows per warp by row count (bf16 only): more rows share each x load as the row count grows.
template <WeightFormat F, int R, bool FUSED>
cudaError_t launch_oro(const MultiArgs& ma, std::uint32_t oro, const void* w1, const float* x, const float* resid, std::uint32_t in,
                       std::uint32_t rows, cudaStream_t s) {
    if constexpr (F == WeightFormat::bf16) {
        if constexpr (R >= 4 && R <= 8) {
            if (oro == 4) return launch<F, R, FUSED, 4>(ma, w1, x, resid, in, rows, s);
        }
        if constexpr (R >= 16 && !(FUSED && R > 16)) {
            if (oro == 2) return launch<F, R, FUSED, 2>(ma, w1, x, resid, in, rows, s);
        }
    }
    return launch<F, R, FUSED, 1>(ma, w1, x, resid, in, rows, s);
}

template <WeightFormat F, bool FUSED>
cudaError_t by_rows(const MultiArgs& ma, std::uint32_t oro, const void* w1, const float* x, const float* resid, std::uint32_t in,
                    std::uint32_t rows, cudaStream_t s) {
    if (rows <= 1) return launch_oro<F, 1, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    if (rows <= 2) return launch_oro<F, 2, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    if (rows <= 4) return launch_oro<F, 4, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    if (rows <= 8) return launch_oro<F, 8, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    if (rows <= 16) return launch_oro<F, 16, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    return launch_oro<F, 32, FUSED>(ma, oro, w1, x, resid, in, rows, s);
}

bool valid(const DenseWeight& w, std::uint32_t rows) {
    if (w.data == nullptr || rows == 0 || rows > kMaxRows || w.out == 0) return false;
    return w.format == WeightFormat::q8_0 ? (w.in % 32 == 0) : (w.in % 8 == 0);
}

// ma.row_end is given in ROWS here; it is converted to groups of `oro` rows when every weight has out % oro == 0.
template <bool FUSED>
cudaError_t dispatch(MultiArgs ma, const DenseWeight& a, const void* w1, const float* x, const float* resid,
                     std::uint32_t rows, cudaStream_t s) {
    std::uint32_t oro = 1;
    if (a.format == WeightFormat::bf16 && rows >= 4) {
        const bool small = rows <= 8;  // row bucket 4 / 8 -> 4 outputs per warp; bucket 16 / 32 -> 2 (32 only unfused)
        oro = small ? 4 : ((FUSED && rows > 16) ? 1 : 2);
        for (int j = 0; j < ma.n; ++j) {
            if (ma.out[j] % oro != 0) oro = 1;
        }
        if (oro > 1) {
            for (int j = 0; j < kMaxMulti; ++j) ma.row_end[j] /= oro;
        }
    }
    switch (a.format) {
        case WeightFormat::bf16: return by_rows<WeightFormat::bf16, FUSED>(ma, oro, w1, x, resid, a.in, rows, s);
        case WeightFormat::q8_0: return by_rows<WeightFormat::q8_0, FUSED>(ma, 1, w1, x, resid, a.in, rows, s);
        case WeightFormat::f32: return by_rows<WeightFormat::f32, FUSED>(ma, 1, w1, x, resid, a.in, rows, s);
    }
    return cudaErrorInvalidValue;
}

__global__ void embedding_kernel(const unsigned char* table, WeightFormat fmt, std::uint32_t in, const std::uint32_t* ids,
                                 float* out) {
    pdl_trigger();
    pdl_wait();
    const std::size_t id = ids[blockIdx.x];
    float* o = out + static_cast<std::size_t>(blockIdx.x) * in;
    for (std::uint32_t i = threadIdx.x; i < in; i += blockDim.x) {
        float v;
        if (fmt == WeightFormat::bf16) {
            v = __uint_as_float(static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(table)[id * in + i]) << 16);
        } else if (fmt == WeightFormat::f32) {
            v = reinterpret_cast<const float*>(table)[id * in + i];
        } else {
            const unsigned char* blk = table + (id * (in >> 5) + (i >> 5)) * 34;
            const std::uint16_t hb = static_cast<std::uint16_t>(blk[0] | (blk[1] << 8));
            v = __fmul_rn(__half2float(__ushort_as_half(hb)), static_cast<float>(static_cast<std::int8_t>(blk[2 + (i & 31)])));
        }
        o[i] = v;
    }
}

}  // namespace

void set_programmatic_launch(bool on) { pdl_enabled() = on; }

namespace {
DenseArith g_arith = DenseArith::mma3;
}
void set_dense_arith(DenseArith a) { g_arith = a; }
DenseArith dense_arith() { return g_arith; }

cudaError_t dense_rows(const DenseWeight& w, const float* x, float* y, std::uint32_t rows, const float* residual,
                       cudaStream_t stream, DenseArith arithmetic) {
    if (w.format == WeightFormat::nvfp4) return nvfp4_dense_rows(w, x, y, rows, residual, stream);
    return dense_rows_multi(&w, &y, 1, x, rows, residual, stream, arithmetic);
}

cudaError_t dense_bf16_fp32_r1_epilogue(const std::uint16_t* weights, const float* input, float* output,
                                       unsigned in, unsigned out, Bf16Fp32Store store, float cap,
                                       cudaStream_t stream) {
    if (!weights || !input || !output || out == 0 || in == 0 || in % 8U != 0)
        return cudaErrorInvalidValue;
    MultiArgs args{};
    args.n = 1;
    args.w[0] = weights;
    args.y[0] = output;
    args.out[0] = out;
    for (auto& end : args.row_end) end = out;
    const dim3 grid((out - 1U) / kWarpsPerBlock + 1U), block(kWarpsPerBlock * 32);
    switch (store) {
    case Bf16Fp32Store::round:
        return launch_pdl(dense_bf16_fp32_r1_store<Bf16Fp32Store::round>, grid, block, stream, args, input, in, 0.0F);
    case Bf16Fp32Store::softcap:
        if (!(cap > 0.0F) || !std::isfinite(cap)) return cudaErrorInvalidValue;
        return launch_pdl(dense_bf16_fp32_r1_store<Bf16Fp32Store::softcap>, grid, block, stream, args, input, in, cap);
    }
    return cudaErrorInvalidValue;
}

cudaError_t dense_bf16_fp32_r1_mapped_epilogue(const std::uint16_t* weights, const float* input, float* output,
                                               const std::uint32_t* ids, const std::uint32_t* count,
                                               unsigned capacity, unsigned rows, unsigned in, Bf16Fp32Store store,
                                               float cap, unsigned blocks, cudaStream_t stream) {
    if (!weights || !input || !output || !ids || !count || capacity == 0 || rows == 0 || in == 0 || in % 8U != 0 ||
        blocks == 0)
        return cudaErrorInvalidValue;
    const dim3 grid(blocks), block(kWarpsPerBlock * 32);
    switch (store) {
    case Bf16Fp32Store::round:
        dense_bf16_fp32_r1_mapped<Bf16Fp32Store::round><<<grid, block, 0, stream>>>(weights, input, output, ids, count,
                                                                                    capacity, rows, in, 0.0F);
        return cudaGetLastError();
    case Bf16Fp32Store::softcap:
        if (!(cap > 0.0F) || !std::isfinite(cap)) return cudaErrorInvalidValue;
        dense_bf16_fp32_r1_mapped<Bf16Fp32Store::softcap><<<grid, block, 0, stream>>>(weights, input, output, ids, count,
                                                                                      capacity, rows, in, cap);
        return cudaGetLastError();
    }
    return cudaErrorInvalidValue;
}

cudaError_t dense_rows_multi(const DenseWeight* ws, float* const* ys, int n, const float* x, std::uint32_t rows,
                             const float* residual, cudaStream_t stream, DenseArith arithmetic) {
    if (n < 1 || n > kMaxMulti || (residual != nullptr && n != 1)) return cudaErrorInvalidValue;
    if (ws[0].format == WeightFormat::nvfp4) {  // W4A16 GEMV: one launch for jobs sharing x, bitwise equal to separate calls
        Nvfp4Job jobs[kMaxMulti];
        for (int j = 0; j < n; ++j) {
            if (ws[j].format != WeightFormat::nvfp4 || ws[j].in != ws[0].in) return cudaErrorInvalidValue;
            jobs[j] = Nvfp4Job{ws[j], ys[j], residual};
        }
        return nvfp4_dense_rows_multi(jobs, static_cast<std::uint32_t>(n), x, rows, stream);
    }
    MultiArgs ma{};
    ma.n = n;
    std::uint32_t total = 0;
    for (int j = 0; j < n; ++j) {
        if (!valid(ws[j], rows) || ws[j].format != ws[0].format || ws[j].in != ws[0].in) return cudaErrorInvalidValue;
        ma.w[j] = ws[j].data;
        ma.y[j] = ys[j];
        ma.out[j] = ws[j].out;
        total += ws[j].out;
        ma.row_end[j] = total;
    }
    for (int j = n; j < kMaxMulti; ++j) ma.row_end[j] = total;
    if (arithmetic != DenseArith::fp32) {
        MmaCall mc{};
        mc.n = n; mc.format = ws[0].format; mc.in = ws[0].in; mc.rows = rows; mc.x = x; mc.residual = residual;
        mc.terms = arithmetic == DenseArith::mma3 ? 3 : arithmetic == DenseArith::mma2 ? 2 : 1;
        for (int j = 0; j < n; ++j) { mc.w[j] = ws[j].data; mc.y[j] = ys[j]; mc.out[j] = ws[j].out; }
        if (dense_mma_supported(mc)) return dense_mma_launch(mc, stream);
        // A sibling's fallback must not change another matrix's arithmetic.
        if (n > 1) {
            DenseWeight grouped[2][kMaxMulti];
            float* outputs[2][kMaxMulti];
            int counts[2]{};
            for (int j = 0; j < n; ++j) {
                MmaCall one = mc;
                one.n = 1; one.out[0] = ws[j].out;
                const int route = dense_mma_supported(one) ? 1 : 0;
                outputs[route][counts[route]] = ys[j];
                grouped[route][counts[route]++] = ws[j];
            }
            if (counts[1]) {
                for (int route = 0; route < 2; ++route) {
                    if (!counts[route]) continue;
                    const auto error = dense_rows_multi(grouped[route], outputs[route], counts[route], x, rows,
                                                        nullptr, stream, arithmetic);
                    if (error != cudaSuccess) return error;
                }
                return cudaSuccess;
            }
        }
    }
    return dispatch<false>(ma, ws[0], nullptr, x, residual, rows, stream);
}

cudaError_t dense_gate_up_silu_rows(const DenseWeight& gate, const DenseWeight& up, const float* x, float* y,
                                    std::uint32_t rows, cudaStream_t stream, DenseArith arithmetic) {
    if (gate.format == WeightFormat::nvfp4) return nvfp4_gate_up_silu_rows(gate, up, x, y, rows, stream);
    if (!valid(gate, rows) || !valid(up, rows) || gate.format != up.format || gate.in != up.in || gate.out != up.out)
        return cudaErrorInvalidValue;
    MultiArgs ma{};
    ma.n = 1;
    ma.w[0] = gate.data;
    ma.y[0] = y;
    ma.out[0] = gate.out;
    for (int j = 0; j < kMaxMulti; ++j) ma.row_end[j] = gate.out;
    if (arithmetic != DenseArith::fp32) {
        MmaCall mc{};
        mc.n = 1; mc.format = gate.format; mc.in = gate.in; mc.rows = rows; mc.x = x; mc.w1 = up.data;
        mc.terms = arithmetic == DenseArith::mma3 ? 3 : arithmetic == DenseArith::mma2 ? 2 : 1;
        mc.w[0] = gate.data; mc.y[0] = y; mc.out[0] = gate.out;
        if (dense_mma_supported(mc)) return dense_mma_launch(mc, stream);
    }
    return dispatch<true>(ma, gate, up.data, x, nullptr, rows, stream);
}

cudaError_t embedding_rows(const DenseWeight& table, const std::uint32_t* ids, float* out, std::uint32_t rows,
                           cudaStream_t stream) {
    if (table.data == nullptr || rows == 0) return cudaErrorInvalidValue;
    return launch_pdl(embedding_kernel, dim3(rows), dim3(256), stream, static_cast<const unsigned char*>(table.data), table.format, table.in, ids, out);
}

cudaError_t argmax_rows(const float* logits, std::uint32_t vocab, std::uint32_t rows, std::uint32_t* out, void* workspace,
                        cudaStream_t stream) {
    if (workspace == nullptr || rows == 0 || rows > kMaxRows) return cudaErrorInvalidValue;
    namespace hp = strix::cuda::core::head_argmax;  // the shared head pick (strix/head_argmax.hpp)
    hp::View v;
    v.logits = logits;
    v.count = vocab;
    v.rows = rows;
    v.partial = static_cast<strix::head_argmax::Pick*>(workspace);
    return hp::pick_rows<hp::Launch::pdl>(v, hp::EmitIndex{out}, stream);  // main's launch form for this core (PDL)
}

cudaError_t argmax2_rows(const float* logits, std::uint32_t vocab, std::uint32_t rows, std::uint32_t* out, std::uint32_t* out2,
                         void* workspace, cudaStream_t stream, float* gap) {
    if (workspace == nullptr || out2 == nullptr || rows == 0 || rows > kMaxRows) return cudaErrorInvalidValue;
    namespace hp = strix::cuda::core::head_argmax;  // runner-up trait of the same shared head pick
    hp::View v;
    v.logits = logits;
    v.count = vocab;
    v.rows = rows;
    return hp::pick2_rows<hp::Launch::pdl>(v, static_cast<strix::head_argmax::Pick2*>(workspace), hp::EmitIndex2{out, out2, gap},
                                           stream);
}

cudaError_t dense_fp32_rows_grid(const DenseWeight& w, const float* x, float* y, std::uint32_t rows,
                                cudaStream_t stream) {
    if (w.format != WeightFormat::bf16 && w.format != WeightFormat::f32) return cudaErrorInvalidValue;
    if (rows <= kMaxRows) return dense_rows(w, x, y, rows, nullptr, stream, DenseArith::fp32);
    if (rows > kFp32GridMaxRows || !valid(w, 1) || w.in == 0 || !x || !y) return cudaErrorInvalidValue;
    MultiArgs ma{};
    ma.n = 1;
    ma.w[0] = w.data;
    ma.y[0] = y;
    ma.out[0] = w.out;
    const auto enqueue = [&]<WeightFormat F, int Oro>() {
        for (auto& end : ma.row_end) end = w.out / Oro;
        return launch<F, kFp32GridRows, false, Oro, true>(ma, nullptr, x, nullptr, w.in, rows, stream);
    };
    if (w.format == WeightFormat::f32) return enqueue.template operator()<WeightFormat::f32, 1>();
    // Two outputs reuse activation reads without the stack traffic of the four-output grid tile.
    if (w.out % 2U == 0) return enqueue.template operator()<WeightFormat::bf16, 2>();
    return enqueue.template operator()<WeightFormat::bf16, 1>();
}

}  // namespace strix::cuda_core
