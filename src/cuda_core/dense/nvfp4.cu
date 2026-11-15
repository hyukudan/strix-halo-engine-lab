// TURBO weights: NVFP4 (e2m1 + e4m3/16 + FP32 global) converter and decode GEMV rows 1..32. Layout/orders: nvfp4.hpp.
// Compiled for sm_120a only (cvt.e2m1x2 / cvt.e4m3x2 PTX); see cmake/CudaCore.cmake.
#include "strix/cuda_core/nvfp4.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

__device__ __forceinline__ void e2m1_word(std::uint32_t w, std::uint32_t (&o)[4]) {
    asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        " cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
        : "r"(w));
}
// two e4m3 bytes (low 16 bits of v) -> f16x2 (low = byte 0)
__device__ __forceinline__ std::uint32_t e4m3x2_to_f16x2(std::uint32_t v) {
    std::uint32_t r;
    asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(r) : "r"(v));
    return r;
}
__device__ __forceinline__ std::uint32_t f32_to_e4m3(float v) {
    std::uint32_t r;
    asm("{\n .reg .b16 h;\n cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, %1;\n cvt.u32.u16 %0, h;\n}" : "=r"(r) : "f"(v));
    return r & 0xFFu;
}
// (hi, lo) -> byte, hi in the upper nibble
__device__ __forceinline__ std::uint32_t f32x2_to_e2m1x2(float hi, float lo) {
    std::uint32_t r;
    asm("{\n .reg .b8 h;\n cvt.rn.satfinite.e2m1x2.f32 h, %1, %2;\n cvt.u32.u8 %0, h;\n}" : "=r"(r) : "f"(hi), "f"(lo));
    return r & 0xFFu;
}
__device__ __forceinline__ std::uint32_t pack_f16x2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const std::uint32_t*>(&h);
}
__device__ __forceinline__ std::uint32_t hmul2(std::uint32_t a, std::uint32_t b) {
    std::uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ std::uint32_t prmt(std::uint32_t a, std::uint32_t sel) {
    std::uint32_t r;
    asm("prmt.b32 %0, %1, %1, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
__device__ __forceinline__ void mma16816(float (&c)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
                                         std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------------------------------------------- GEMV
constexpr int kWarpsPerBlock = 4;
std::uint32_t g_target_warps = 640;  // split-K target (depends on the shape only, never on rows)

struct LoadRegs {
    uint4 d0, d1, sc;
};

// Weight stream: read once, keep out of L1 so the activation slab stays resident.
__device__ __forceinline__ uint4 ld_stream(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ LoadRegs load_record(const unsigned char* rec, int lane) {
    LoadRegs r;
    r.d0 = ld_stream(rec + lane * 16);
    r.d1 = ld_stream(rec + 512 + lane * 16);
    r.sc = ld_stream(rec + 1024 + (lane >> 2) * 16);
    return r;
}

constexpr int kSlab = 4;  // K chunks (of 128) of activations staged in shared memory per barrier pair

// xs layout (fragment ready): [chunk in slab][rb][group j][lane] -> uint2 {b0, b1} (f16x2 pairs of that lane).
template <int RB, int MT>
__device__ __forceinline__ void compute_chunk(const LoadRegs (&L)[MT], const bool (&on)[MT], const uint2* xs, int lane,
                                              float (&acc)[MT][RB][4]) {
#pragma unroll
    for (int jp = 0; jp < 4; ++jp) {  // group pair (2jp, 2jp+1)
        std::uint32_t a[MT][2][4];     // [mt][group in pair][a0..a3]
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (!on[mt]) continue;
            const std::uint32_t w0 = jp == 0 ? L[mt].d0.x : jp == 1 ? L[mt].d0.y : jp == 2 ? L[mt].d0.z : L[mt].d0.w;
            const std::uint32_t w1 = jp == 0 ? L[mt].d1.x : jp == 1 ? L[mt].d1.y : jp == 2 ? L[mt].d1.z : L[mt].d1.w;
            const std::uint32_t s0w = jp < 2 ? L[mt].sc.x : L[mt].sc.y;
            const std::uint32_t s1w = jp < 2 ? L[mt].sc.z : L[mt].sc.w;
            const std::uint32_t sh = (jp & 1) * 16;
            const std::uint32_t s0 = e4m3x2_to_f16x2(s0w >> sh);
            const std::uint32_t s1 = e4m3x2_to_f16x2(s1w >> sh);
            std::uint32_t o0[4], o1[4];
            e2m1_word(w0, o0);
            e2m1_word(w1, o1);
            const std::uint32_t s0a = prmt(s0, 0x1010), s0b = prmt(s0, 0x3232);
            const std::uint32_t s1a = prmt(s1, 0x1010), s1b = prmt(s1, 0x3232);
            // group 2jp: a0 = (h0,p0) a1 = (h1,p0) a2 = (h0,p1) a3 = (h1,p1)
            a[mt][0][0] = hmul2(o0[0], s0a);
            a[mt][0][1] = hmul2(o1[0], s1a);
            a[mt][0][2] = hmul2(o0[1], s0a);
            a[mt][0][3] = hmul2(o1[1], s1a);
            a[mt][1][0] = hmul2(o0[2], s0b);
            a[mt][1][1] = hmul2(o1[2], s1b);
            a[mt][1][2] = hmul2(o0[3], s0b);
            a[mt][1][3] = hmul2(o1[3], s1b);
        }
#pragma unroll
        for (int gi = 0; gi < 2; ++gi) {
            uint2 b[RB];
#pragma unroll
            for (int rb = 0; rb < RB; ++rb) b[rb] = xs[(rb * 8 + 2 * jp + gi) * 32 + lane];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if (!on[mt]) continue;
#pragma unroll
                for (int rb = 0; rb < RB; ++rb)
                    mma16816(acc[mt][rb], a[mt][gi][0], a[mt][gi][1], a[mt][gi][2], a[mt][gi][3], b[rb].x, b[rb].y);
            }
        }
    }
}

// grid (ceil(ntiles / (4 MT)), ks, nw): the 4 warps of a block take 4 MT consecutive output tiles and share the
// activation slab through shared memory; blockIdx.y selects the K range. part = [z][ks][rows][out] FP32; when DIRECT
// (ks == 1, single weight) the epilogue y = acc * gscale (+ resid) is written straight to y.
template <int RB, int MT, bool DIRECT>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    fp4_gemv(const unsigned char* w0, const unsigned char* w1, const float* x, float* part, float* y, const float* resid,
             float gs0, std::uint32_t out, std::uint32_t in, std::uint32_t rows, std::uint32_t ks) {
    extern __shared__ uint2 xs_dyn[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const std::uint32_t kidx = blockIdx.y;
    const unsigned char* w = blockIdx.z == 0 ? w0 : w1;
    const std::uint32_t nch = in >> 7, ntiles = out >> 4;
    const std::uint32_t c0 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(kidx) * nch / ks);
    const std::uint32_t c1 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(kidx + 1) * nch / ks);
    bool on[MT];
    const unsigned char* base[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const std::uint32_t m = (blockIdx.x * kWarpsPerBlock + warp) * MT + mt;
        on[mt] = m < ntiles;
        base[mt] = w + static_cast<std::size_t>(on[mt] ? m : 0) * nch * kNvfp4RecordBytes;
    }
    float acc[MT][RB][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) acc[mt][rb][0] = acc[mt][rb][1] = acc[mt][rb][2] = acc[mt][rb][3] = 0.f;

    LoadRegs cur[MT], nxt[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        if (on[mt] && c0 < c1) cur[mt] = load_record(base[mt] + static_cast<std::size_t>(c0) * kNvfp4RecordBytes, lane);
    for (std::uint32_t c = c0; c < c1; ++c) {
        const std::uint32_t cs = (c - c0) % kSlab;
        if (cs == 0) {
            __syncthreads();
            const std::uint32_t nc = min(static_cast<std::uint32_t>(kSlab), c1 - c);
            const std::uint32_t total = nc * RB * 8 * 32;
            for (std::uint32_t e = threadIdx.x; e < total; e += kWarpsPerBlock * 32) {
                const std::uint32_t ln = e & 31, j = (e >> 5) & 7, rb = (e >> 8) % RB, cc_ = e / (256 * RB);
                const std::uint32_t n = rb * 8 + (ln >> 2), k0 = ((c + cc_) << 7) + j * 16 + (ln & 3) * 2;
                float2 p0 = make_float2(0.f, 0.f), p1 = p0;
                if (n < rows) {
                    const float* xr = x + static_cast<std::size_t>(n) * in + k0;
                    p0 = *reinterpret_cast<const float2*>(xr);
                    p1 = *reinterpret_cast<const float2*>(xr + 8);
                }
                xs_dyn[(cc_ * RB + rb) * 256 + j * 32 + ln] = make_uint2(pack_f16x2(p0.x, p0.y), pack_f16x2(p1.x, p1.y));
            }
            __syncthreads();
        }
        if (c + 1 < c1) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                if (on[mt]) nxt[mt] = load_record(base[mt] + static_cast<std::size_t>(c + 1) * kNvfp4RecordBytes, lane);
        }
        compute_chunk<RB, MT>(cur, on, xs_dyn + cs * RB * 256, lane, acc);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) cur[mt] = nxt[mt];
    }
    // epilogue: C fragment (c0,c1: row g, cols 2t,2t+1; c2,c3: row g+8)
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        if (!on[mt]) continue;
        const std::uint32_t m = (blockIdx.x * kWarpsPerBlock + warp) * MT + mt;
#pragma unroll
        for (int rb = 0; rb < RB; ++rb)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const std::uint32_t n = static_cast<std::uint32_t>(rb * 8 + 2 * t + (q & 1));
                const std::uint32_t o = m * 16 + static_cast<std::uint32_t>(g + ((q >> 1) << 3));
                if (n >= rows) continue;
                if constexpr (DIRECT) {
                    float v = __fmul_rn(acc[mt][rb][q], gs0);
                    if (resid != nullptr) v = __fadd_rn(resid[static_cast<std::size_t>(n) * out + o], v);
                    y[static_cast<std::size_t>(n) * out + o] = v;
                } else {
                    part[((static_cast<std::size_t>(blockIdx.z) * ks + kidx) * rows + n) * out + o] = acc[mt][rb][q];
                }
            }
    }
}

template <bool FUSED>
__global__ void fp4_reduce(const float* part, float* y, const float* resid, float gs0, float gs1, std::uint32_t out,
                           std::uint32_t rows, std::uint32_t ks) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<std::size_t>(out) * rows) return;
    const std::size_t stride = static_cast<std::size_t>(out) * rows;
    float s = part[idx];
    for (std::uint32_t k = 1; k < ks; ++k) s = __fadd_rn(s, part[k * stride + idx]);
    float v = __fmul_rn(s, gs0);
    if constexpr (FUSED) {
        const std::size_t off = static_cast<std::size_t>(ks) * stride;
        float u = part[off + idx];
        for (std::uint32_t k = 1; k < ks; ++k) u = __fadd_rn(u, part[off + k * stride + idx]);
        v = __fmul_rn(cc_silu(v), __fmul_rn(u, gs1));
    } else if (resid != nullptr) {
        v = __fadd_rn(resid[idx], v);
    }
    y[idx] = v;
}

float* g_ws = nullptr;
std::size_t g_ws_bytes = 0;

template <int RB, int MT, bool DIRECT>
cudaError_t launch_gemv(const unsigned char* w0, const unsigned char* w1, const float* x, float* part, float* y,
                        const float* resid, float gs0, float gs1, std::uint32_t out, std::uint32_t in, std::uint32_t rows,
                        std::uint32_t ks, std::uint32_t nw, cudaStream_t s) {
    const dim3 grid((out / 16 + kWarpsPerBlock * MT - 1) / (kWarpsPerBlock * MT), ks, nw);
    fp4_gemv<RB, MT, DIRECT><<<grid, kWarpsPerBlock * 32, kSlab * RB * 2048, s>>>(w0, w1, x, part, y, resid, gs0, out, in, rows, ks);
    return cudaGetLastError();
}

template <bool DIRECT>
cudaError_t by_rb(const unsigned char* w0, const unsigned char* w1, const float* x, float* part, float* y,
                  const float* resid, float gs0, float gs1, std::uint32_t out, std::uint32_t in, std::uint32_t rows,
                  std::uint32_t ks, std::uint32_t nw, cudaStream_t s) {
    if (rows <= 8) return launch_gemv<1, 4, DIRECT>(w0, w1, x, part, y, resid, gs0, gs1, out, in, rows, ks, nw, s);
    if (rows <= 16) return launch_gemv<2, 4, DIRECT>(w0, w1, x, part, y, resid, gs0, gs1, out, in, rows, ks, nw, s);
    if (rows <= 24) return launch_gemv<3, 2, DIRECT>(w0, w1, x, part, y, resid, gs0, gs1, out, in, rows, ks, nw, s);
    return launch_gemv<4, 2, DIRECT>(w0, w1, x, part, y, resid, gs0, gs1, out, in, rows, ks, nw, s);
}

bool ok(const DenseWeight& w, std::uint32_t rows) {
    return w.data != nullptr && w.format == WeightFormat::nvfp4 && rows >= 1 && rows <= kNvfp4MaxRows &&
           nvfp4_shape_ok(w.out, w.in);
}

cudaError_t ensure_ws(std::size_t bytes) {
    if (bytes <= g_ws_bytes) return cudaSuccess;
    if (g_ws) cudaFree(g_ws);
    g_ws = nullptr;
    g_ws_bytes = 0;
    const cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&g_ws), bytes);
    if (e == cudaSuccess) g_ws_bytes = bytes;
    return e;
}

// ----------------------------------------------------------------------------------------------------- converter
__global__ void amax_kernel_bf16(const std::uint16_t* src, std::size_t n, std::uint32_t* amax) {
    float m = 0.f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        m = fmaxf(m, fabsf(__uint_as_float(static_cast<std::uint32_t>(src[i]) << 16)));
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) atomicMax(amax, __float_as_uint(m));
}
__global__ void amax_kernel_f32(const float* src, std::size_t n, std::uint32_t* amax) {
    float m = 0.f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        m = fmaxf(m, fabsf(src[i]));
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) atomicMax(amax, __float_as_uint(m));
}

// one thread per (output row o, group G of 16 inputs)
template <bool BF16>
__global__ void convert_kernel(const void* src, std::uint32_t out, std::uint32_t in, const std::uint32_t* amax_bits,
                               unsigned char* dst, float* gs_out) {
    const std::size_t ngroups = static_cast<std::size_t>(out) * (in >> 4);
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float amax_t = __uint_as_float(*amax_bits);
    const float gs = amax_t > 0.f ? __fdiv_rn(amax_t, 6.0f * 448.0f) : 1.0f;
    if (idx == 0) *gs_out = gs;
    if (idx >= ngroups) return;
    const std::uint32_t G = static_cast<std::uint32_t>(idx % (in >> 4));
    const std::uint32_t o = static_cast<std::uint32_t>(idx / (in >> 4));
    float w[16];
    if constexpr (BF16) {
        const uint4* p = reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(src) + static_cast<std::size_t>(o) * in + G * 16);
        const uint4 a = __ldg(p), b = __ldg(p + 1);
        const std::uint32_t u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[2 * i] = bf16_lo(u[i]);
            w[2 * i + 1] = bf16_hi(u[i]);
        }
    } else {
        const float4* p = reinterpret_cast<const float4*>(static_cast<const float*>(src) + static_cast<std::size_t>(o) * in + G * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = __ldg(p + i);
            w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
        }
    }
    float am = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(w[i]));
    const std::uint32_t sb = f32_to_e4m3(__fdiv_rn(am, __fmul_rn(6.0f, gs)));
    const float sd = __half2float(__ushort_as_half(static_cast<unsigned short>(e4m3x2_to_f16x2(sb) & 0xFFFFu)));
    const float den = __fmul_rn(sd, gs);
    std::uint32_t code[16];
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
        if (sd > 0.f) {
            const std::uint32_t b = f32x2_to_e2m1x2(__fdiv_rn(w[i + 1], den), __fdiv_rn(w[i], den));
            code[i] = b & 0xF;
            code[i + 1] = b >> 4;
        } else {
            code[i] = code[i + 1] = 0;
        }
    }
    const std::uint32_t m = o >> 4, r = o & 15, h = r >> 3, gg = r & 7, c = G >> 3, j = G & 7;
    unsigned char* rec = dst + (static_cast<std::size_t>(m) * (in >> 7) + c) * kNvfp4RecordBytes;
    rec[1024 + gg * 16 + h * 8 + j] = static_cast<unsigned char>(sb);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const std::uint32_t lane = static_cast<std::uint32_t>(gg * 4 + t);
        // pair p=0: inputs 2t,2t+1 ; p=1: inputs 2t+8, 2t+9
        rec[h * 512 + lane * 16 + 4 * (j >> 1) + 2 * (j & 1) + 0] = static_cast<unsigned char>(code[2 * t] | (code[2 * t + 1] << 4));
        rec[h * 512 + lane * 16 + 4 * (j >> 1) + 2 * (j & 1) + 1] = static_cast<unsigned char>(code[2 * t + 8] | (code[2 * t + 9] << 4));
    }
}

template <bool BF16>
cudaError_t quantize(const void* src, std::uint32_t out, std::uint32_t in, void* dst, float* gscale, cudaStream_t s) {
    if (!nvfp4_shape_ok(out, in) || src == nullptr || dst == nullptr || gscale == nullptr) return cudaErrorInvalidValue;
    std::uint32_t* d_amax = nullptr;
    float* d_gs = nullptr;
    cudaError_t e = cudaMallocAsync(reinterpret_cast<void**>(&d_amax), 8, s);
    if (e != cudaSuccess) return e;
    d_gs = reinterpret_cast<float*>(d_amax + 1);
    cudaMemsetAsync(d_amax, 0, 4, s);
    const std::size_t n = static_cast<std::size_t>(out) * in;
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 4096));
    if constexpr (BF16) amax_kernel_bf16<<<blocks, 256, 0, s>>>(static_cast<const std::uint16_t*>(src), n, d_amax);
    else amax_kernel_f32<<<blocks, 256, 0, s>>>(static_cast<const float*>(src), n, d_amax);
    const std::size_t ng = static_cast<std::size_t>(out) * (in >> 4);
    convert_kernel<BF16><<<static_cast<unsigned>((ng + 127) / 128), 128, 0, s>>>(src, out, in, d_amax, static_cast<unsigned char*>(dst), d_gs);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaMemcpyAsync(gscale, d_gs, 4, cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    cudaFreeAsync(d_amax, s);
    return e;
}

}  // namespace

void nvfp4_set_split_target(std::uint32_t warps) { g_target_warps = warps ? warps : 640; }

std::uint32_t nvfp4_split_k(std::uint32_t out, std::uint32_t in) {
    const std::uint32_t nch = in >> 7, groups = std::max<std::uint32_t>(1, (out / 16 + 15) / 16);
    return std::clamp<std::uint32_t>((g_target_warps + groups / 2) / groups, 1, nch);
}

cudaError_t nvfp4_quantize_bf16(const void* bf16, std::uint32_t out, std::uint32_t in, void* dst, float* gscale,
                                cudaStream_t s) {
    return quantize<true>(bf16, out, in, dst, gscale, s);
}
cudaError_t nvfp4_quantize_f32(const float* src, std::uint32_t out, std::uint32_t in, void* dst, float* gscale,
                               cudaStream_t s) {
    return quantize<false>(src, out, in, dst, gscale, s);
}

cudaError_t nvfp4_reserve_workspace(std::size_t bytes) { return ensure_ws(bytes); }

cudaError_t nvfp4_dense_rows(const DenseWeight& w, const float* x, float* y, std::uint32_t rows, const float* residual,
                             cudaStream_t s) {
    if (!ok(w, rows)) return cudaErrorInvalidValue;
    const std::uint32_t ks = nvfp4_split_k(w.out, w.in);
    const auto* wd = static_cast<const unsigned char*>(w.data);
    if (ks == 1) return by_rb<true>(wd, wd, x, nullptr, y, residual, w.gscale, 1.0f, w.out, w.in, rows, 1, 1, s);
    cudaError_t e = ensure_ws(static_cast<std::size_t>(ks) * rows * w.out * 4);
    if (e != cudaSuccess) return e;
    e = by_rb<false>(wd, wd, x, g_ws, y, residual, w.gscale, 1.0f, w.out, w.in, rows, ks, 1, s);
    if (e != cudaSuccess) return e;
    const std::size_t n = static_cast<std::size_t>(w.out) * rows;
    fp4_reduce<false><<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(g_ws, y, residual, w.gscale, 1.0f, w.out, rows, ks);
    return cudaGetLastError();
}

cudaError_t nvfp4_gate_up_silu_rows(const DenseWeight& gate, const DenseWeight& up, const float* x, float* y,
                                    std::uint32_t rows, cudaStream_t s) {
    if (!ok(gate, rows) || !ok(up, rows) || gate.out != up.out || gate.in != up.in) return cudaErrorInvalidValue;
    const std::uint32_t ks = nvfp4_split_k(gate.out, gate.in);
    cudaError_t e = ensure_ws(static_cast<std::size_t>(2) * ks * rows * gate.out * 4);
    if (e != cudaSuccess) return e;
    e = by_rb<false>(static_cast<const unsigned char*>(gate.data), static_cast<const unsigned char*>(up.data), x, g_ws,
                     nullptr, nullptr, gate.gscale, up.gscale, gate.out, gate.in, rows, ks, 2, s);
    if (e != cudaSuccess) return e;
    const std::size_t n = static_cast<std::size_t>(gate.out) * rows;
    fp4_reduce<true><<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(g_ws, y, nullptr, gate.gscale, up.gscale, gate.out, rows, ks);
    return cudaGetLastError();
}

// ------------------------------------------------------------------------------------------------- host reference
float nvfp4_host_e4m3(unsigned char b) {
    const int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
    if (e == 15 && m == 7) v = std::nanf("");
    return (b & 0x80) ? -v : v;
}

static const float kE2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

unsigned char nvfp4_host_quantize_group(const float* w16, float gs, unsigned char* codes16) {
    float am = 0.f;
    for (int i = 0; i < 16; ++i) am = std::max(am, std::fabs(w16[i]));
    const float target = am / (6.0f * gs);
    // e4m3 RN-even with saturation at 448: search the 126 finite non-negative codes for the nearest (ties to even code)
    int best = 0;
    float bd = std::fabs(target - 0.0f);
    for (int c = 1; c < 127; ++c) {
        const float d = std::fabs(target - nvfp4_host_e4m3(static_cast<unsigned char>(c)));
        if (d < bd || (d == bd && (c & 1) == 0)) {
            bd = d;
            best = c;
        }
    }
    const float sd = nvfp4_host_e4m3(static_cast<unsigned char>(best));
    const float den = sd * gs;
    for (int i = 0; i < 16; ++i) {
        if (!(sd > 0.f)) {
            codes16[i] = 0;
            continue;
        }
        const float v = w16[i] / den, a = std::fabs(v);
        int c = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
        if (std::signbit(v)) c |= 8;
        codes16[i] = static_cast<unsigned char>(c);
    }
    return static_cast<unsigned char>(best);
}

float nvfp4_host_dequant(const unsigned char* tiles, std::uint32_t in, std::uint32_t o, std::uint32_t k) {
    const std::uint32_t m = o >> 4, r = o & 15, h = r >> 3, g = r & 7, c = k >> 7, kl = k & 127, j = kl >> 4, kk = kl & 15;
    const std::uint32_t t = (kk & 7) >> 1, p = kk >> 3, lane = g * 4 + t;
    const unsigned char* rec = tiles + (static_cast<std::size_t>(m) * (in >> 7) + c) * kNvfp4RecordBytes;
    const unsigned char byte = rec[h * 512 + lane * 16 + 4 * (j >> 1) + 2 * (j & 1) + p];
    const unsigned nib = (kk & 1) ? (byte >> 4) : (byte & 15);
    const float v = kE2m1[nib & 7] * ((nib & 8) ? -1.0f : 1.0f);
    return v * nvfp4_host_e4m3(rec[1024 + g * 16 + h * 8 + j]);
}

}  // namespace strix::cuda_core
