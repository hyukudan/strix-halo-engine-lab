// Dense rows 1..16: BF16 / Q8_0 / F32 weights, FP32 accumulation in the fixed order documented in dense.hpp.
#include "strix/cuda_core/dense.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

template <int R>
__device__ __forceinline__ void dot_bf16(const std::uint16_t* wrow, const float* x, std::uint32_t in, std::uint32_t rows,
                                         int lane, float (&acc)[R]) {
    const std::uint32_t nch = in >> 3;
    const uint4* wv = reinterpret_cast<const uint4*>(wrow);
#pragma unroll 4
    for (std::uint32_t c = lane; c < nch; c += 32) {
        const uint4 u = __ldg(wv + c);
        const float w[8] = {bf16_lo(u.x), bf16_hi(u.x), bf16_lo(u.y), bf16_hi(u.y),
                            bf16_lo(u.z), bf16_hi(u.z), bf16_lo(u.w), bf16_hi(u.w)};
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

template <WeightFormat F, int R>
__device__ __forceinline__ void dot_any(const void* w, std::uint32_t o, const float* x, std::uint32_t in, std::uint32_t rows,
                                        int lane, float (&acc)[R]) {
    if constexpr (F == WeightFormat::bf16) {
        dot_bf16<R>(static_cast<const std::uint16_t*>(w) + static_cast<std::size_t>(o) * in, x, in, rows, lane, acc);
    } else if constexpr (F == WeightFormat::f32) {
        dot_f32<R>(static_cast<const float*>(w) + static_cast<std::size_t>(o) * in, x, in, rows, lane, acc);
    } else {
        dot_q8<R>(static_cast<const unsigned char*>(w) + static_cast<std::size_t>(o) * (in >> 5) * 34, x, in, rows, lane, acc);
    }
}

template <WeightFormat F, int R, bool FUSED>
__global__ void __launch_bounds__(256) dense_kernel(const void* w0, const void* w1, const float* x, float* y,
                                                    const float* resid, std::uint32_t out, std::uint32_t in,
                                                    std::uint32_t rows) {
    const std::uint32_t o = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (o >= out) return;
    const int lane = threadIdx.x & 31;
    float a0[R], a1[R];
#pragma unroll
    for (int r = 0; r < R; ++r) a0[r] = a1[r] = 0.0f;
    dot_any<F, R>(w0, o, x, in, rows, lane, a0);
    if constexpr (FUSED) dot_any<F, R>(w1, o, x, in, rows, lane, a1);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float g = warp_sum_xor(a0[r]);
        float v = g;
        if constexpr (FUSED) {
            const float u = warp_sum_xor(a1[r]);
            v = __fmul_rn(cc_silu(g), u);
        } else if (resid != nullptr) {
            const std::size_t idx = static_cast<std::size_t>(r) * out + o;
            if (static_cast<std::uint32_t>(r) < rows) v = __fadd_rn(resid[idx], g);
        }
        if (lane == 0 && static_cast<std::uint32_t>(r) < rows) y[static_cast<std::size_t>(r) * out + o] = v;
    }
}

template <WeightFormat F, int R, bool FUSED>
cudaError_t launch(const void* w0, const void* w1, const float* x, float* y, const float* resid, std::uint32_t out,
                   std::uint32_t in, std::uint32_t rows, cudaStream_t s) {
    dense_kernel<F, R, FUSED><<<(out + 7) / 8, 256, 0, s>>>(w0, w1, x, y, resid, out, in, rows);
    return cudaGetLastError();
}

template <WeightFormat F, bool FUSED>
cudaError_t by_rows(const void* w0, const void* w1, const float* x, float* y, const float* resid, std::uint32_t out,
                    std::uint32_t in, std::uint32_t rows, cudaStream_t s) {
    if (rows <= 1) return launch<F, 1, FUSED>(w0, w1, x, y, resid, out, in, rows, s);
    if (rows <= 2) return launch<F, 2, FUSED>(w0, w1, x, y, resid, out, in, rows, s);
    if (rows <= 4) return launch<F, 4, FUSED>(w0, w1, x, y, resid, out, in, rows, s);
    if (rows <= 8) return launch<F, 8, FUSED>(w0, w1, x, y, resid, out, in, rows, s);
    return launch<F, 16, FUSED>(w0, w1, x, y, resid, out, in, rows, s);
}

bool valid(const DenseWeight& w, std::uint32_t rows) {
    if (w.data == nullptr || rows == 0 || rows > kMaxRows || w.out == 0) return false;
    return w.format == WeightFormat::q8_0 ? (w.in % 32 == 0) : (w.in % 8 == 0);
}

template <bool FUSED>
cudaError_t dispatch(const DenseWeight& a, const DenseWeight* b, const float* x, float* y, const float* resid,
                     std::uint32_t rows, cudaStream_t s) {
    const void* w1 = b ? b->data : nullptr;
    switch (a.format) {
        case WeightFormat::bf16: return by_rows<WeightFormat::bf16, FUSED>(a.data, w1, x, y, resid, a.out, a.in, rows, s);
        case WeightFormat::q8_0: return by_rows<WeightFormat::q8_0, FUSED>(a.data, w1, x, y, resid, a.out, a.in, rows, s);
        case WeightFormat::f32: return by_rows<WeightFormat::f32, FUSED>(a.data, w1, x, y, resid, a.out, a.in, rows, s);
    }
    return cudaErrorInvalidValue;
}

__global__ void embedding_kernel(const unsigned char* table, WeightFormat fmt, std::uint32_t in, const std::uint32_t* ids,
                                 float* out) {
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

__global__ void argmax_kernel(const float* logits, std::uint32_t vocab, std::uint32_t* out) {
    const float* l = logits + static_cast<std::size_t>(blockIdx.x) * vocab;
    float best = __int_as_float(0xff800000);
    std::uint32_t bi = 0xFFFFFFFFu;
    for (std::uint32_t i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = l[i];
        if (v > best || bi == 0xFFFFFFFFu) {
            best = v;
            bi = i;
        }
    }
    __shared__ float sv[1024];
    __shared__ std::uint32_t si[1024];
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const std::uint32_t oi = si[threadIdx.x + s];
            const float mv = sv[threadIdx.x];
            const std::uint32_t mi = si[threadIdx.x];
            const bool take = mi == 0xFFFFFFFFu || (oi != 0xFFFFFFFFu && (ov > mv || (ov == mv && oi < mi)));
            if (take) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = si[0];
}

}  // namespace

cudaError_t dense_rows(const DenseWeight& w, const float* x, float* y, std::uint32_t rows, const float* residual,
                       cudaStream_t stream) {
    if (!valid(w, rows)) return cudaErrorInvalidValue;
    return dispatch<false>(w, nullptr, x, y, residual, rows, stream);
}

cudaError_t dense_gate_up_silu_rows(const DenseWeight& gate, const DenseWeight& up, const float* x, float* y,
                                    std::uint32_t rows, cudaStream_t stream) {
    if (!valid(gate, rows) || !valid(up, rows) || gate.format != up.format || gate.in != up.in || gate.out != up.out)
        return cudaErrorInvalidValue;
    return dispatch<true>(gate, &up, x, y, nullptr, rows, stream);
}

cudaError_t embedding_rows(const DenseWeight& table, const std::uint32_t* ids, float* out, std::uint32_t rows,
                           cudaStream_t stream) {
    if (table.data == nullptr || rows == 0) return cudaErrorInvalidValue;
    embedding_kernel<<<rows, 256, 0, stream>>>(static_cast<const unsigned char*>(table.data), table.format, table.in, ids, out);
    return cudaGetLastError();
}

cudaError_t argmax_rows(const float* logits, std::uint32_t vocab, std::uint32_t rows, std::uint32_t* out,
                        cudaStream_t stream) {
    argmax_kernel<<<rows, 1024, 0, stream>>>(logits, vocab, out);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
