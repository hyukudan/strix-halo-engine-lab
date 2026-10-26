#pragma once
// Small FP32 kernels of the MiniMax engine. The first six are copied verbatim from the minimax-m3 layer gates
// (tests/minimax_gate_common.hpp) so the engine reproduces the G4 harness bit for bit; include from ONE translation unit.
#include "mm_common.hpp"

#include <hip/hip_runtime.h>

namespace strix::minimax::engine::detail {

inline void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

namespace {

__global__ void rmsnorm_gemma_kernel(const float* x, const float* w, float* y, unsigned d) {
    __shared__ float red[256];
    const float* xr = x + static_cast<std::size_t>(blockIdx.x) * d;
    float* yr = y + static_cast<std::size_t>(blockIdx.x) * d;
    float s = 0.0F;
    for (unsigned i = threadIdx.x; i < d; i += 256) s = __builtin_fmaf(xr[i], xr[i], s);
    red[threadIdx.x] = s;
    __syncthreads();
    for (unsigned k = 128; k > 0; k >>= 1U) {
        if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
        __syncthreads();
    }
    const float r = 1.0F / __builtin_sqrtf(red[0] / static_cast<float>(d) + kEps);
    for (unsigned i = threadIdx.x; i < d; i += 256) yr[i] = (xr[i] * r) * (1.0F + w[i]);
}
// y[t][o] = sum_d x[t][d] * wt[d][o]; wt is the transposed weight [d][n] in FP32. Block: one token, 128 outputs.
__global__ void dense_t_kernel(const float* x, const float* wt, float* y, unsigned d, unsigned n) {
    __shared__ float xs[kHidden];
    const float* xr = x + static_cast<std::size_t>(blockIdx.x) * d;
    for (unsigned i = threadIdx.x; i < d; i += 128) xs[i] = xr[i];
    __syncthreads();
    const unsigned o = blockIdx.y * 128 + threadIdx.x;
    if (o >= n) return;
    float acc = 0.0F;
    for (unsigned i = 0; i < d; ++i) acc = __builtin_fmaf(xs[i], wt[static_cast<std::size_t>(i) * n + o], acc);
    y[static_cast<std::size_t>(blockIdx.x) * n + o] = acc;
}
// dense_t_kernel's arithmetic (per output: acc = 0; acc = fmaf(x[t][i], w[i][o], acc) for i ascending), so the same
// bits, but TT tokens per block share every weight read, and the weights stay F16 (the file's format; widening a
// half to float is exact). Block: TT tokens x 128 outputs; x staged through LDS in 64-input tiles.
template <unsigned TT>
__global__ void dense_t16_kernel(const float* x, const _Float16* wt, float* y, unsigned d, unsigned n, unsigned rows) {
    __shared__ float xs[TT][64];
    const unsigned t0 = blockIdx.x * TT;
    const unsigned o = blockIdx.y * 128 + threadIdx.x;
    float acc[TT];
    for (unsigned t = 0; t < TT; ++t) acc[t] = 0.0F;
    for (unsigned i0 = 0; i0 < d; i0 += 64) {
        __syncthreads();
        for (unsigned j = threadIdx.x; j < TT * 64; j += 128) {
            const unsigned t = j / 64, i = j % 64;
            xs[t][i] = t0 + t < rows ? x[static_cast<std::size_t>(t0 + t) * d + i0 + i] : 0.0F;
        }
        __syncthreads();
        if (o < n)
            for (unsigned i = 0; i < 64; ++i) {
                const float w = static_cast<float>(wt[static_cast<std::size_t>(i0 + i) * n + o]);
                for (unsigned t = 0; t < TT; ++t) acc[t] = __builtin_fmaf(xs[t][i], w, acc[t]);
            }
    }
    if (o < n)
        for (unsigned t = 0; t < TT; ++t)
            if (t0 + t < rows) y[static_cast<std::size_t>(t0 + t) * n + o] = acc[t];
}
// Decode form of the index projections (<= 8 rows), both matrices in ONE launch, same per-output chain as
// dense_t_kernel (acc = 0; acc = fmaf(x[t][i], w[i][o], acc), i ascending), so the same bits. The dense_t forms give a
// 1-row call only 2 + 1 blocks, so ~4.5 MB of weights per layer were streamed by a handful of waves (measurement:
// ~1.3 ms per layer). Here the weights are block-major, wb[ob][i][32] (32 outputs per block, every 64-input tile is
// 4 KB contiguous); all 256 threads prefetch the next tile (+ the rows' 64 inputs) into registers while wave 0 runs the
// chains on the current tile in LDS (double buffer). Blocks [0, nb_q) do the q matrix, the rest the k matrix.
constexpr unsigned kDecOut = 32, kDecTile = 64;
__global__ __launch_bounds__(256) void index_proj_decode_kernel(const float* x, const _Float16* wq, float* yq, unsigned nq,
                                                                const _Float16* wk, float* yk, unsigned nk, unsigned d,
                                                                unsigned rows) {
    __shared__ _Float16 ws[2][kDecTile * kDecOut];  // 2 x 4 KB
    __shared__ float xs[2][8][kDecTile];            // 2 x 2 KB
    const unsigned nbq = nq / kDecOut;
    const bool is_q = blockIdx.x < nbq;
    const unsigned ob = is_q ? blockIdx.x : blockIdx.x - nbq, n = is_q ? nq : nk;
    const _Float16* wb = (is_q ? wq : wk) + static_cast<std::size_t>(ob) * d * kDecOut;
    float* y = is_q ? yq : yk;
    const unsigned tid = threadIdx.x;
    // per tile: 2048 halves = 256 threads x 8 halves (one 16-byte load each); x: rows x 64 floats (<= 512)
    auto load_w = [&](unsigned tile, uint4& r) { r = reinterpret_cast<const uint4*>(wb + static_cast<std::size_t>(tile) * kDecTile * kDecOut)[tid]; };
    auto load_x = [&](unsigned tile, float (&r)[2]) {
        for (unsigned k = 0; k < 2; ++k) {
            const unsigned j = tid + k * 256, t = j / kDecTile, i = j % kDecTile;
            r[k] = t < rows ? x[static_cast<std::size_t>(t) * d + tile * kDecTile + i] : 0.0F;
        }
    };
    auto store = [&](unsigned buf, const uint4& rw, const float (&rx)[2]) {
        reinterpret_cast<uint4*>(ws[buf])[tid] = rw;
        for (unsigned k = 0; k < 2; ++k) {
            const unsigned j = tid + k * 256, t = j / kDecTile, i = j % kDecTile;
            if (t < 8) xs[buf][t][i] = rx[k];
        }
    };
    const unsigned tiles = d / kDecTile;
    float acc[8];
    for (unsigned t = 0; t < 8; ++t) acc[t] = 0.0F;
    uint4 rw;
    float rx[2];
    load_w(0, rw);
    load_x(0, rx);
    for (unsigned tile = 0; tile < tiles; ++tile) {
        const unsigned buf = tile & 1U;
        store(buf, rw, rx);
        __syncthreads();
        if (tile + 1 < tiles) {  // next tile in flight while wave 0 computes this one
            load_w(tile + 1, rw);
            load_x(tile + 1, rx);
        }
        if (tid < kDecOut)
            for (unsigned i = 0; i < kDecTile; ++i) {
                const float w = static_cast<float>(ws[buf][i * kDecOut + tid]);
                for (unsigned t = 0; t < 8; ++t) acc[t] = __builtin_fmaf(xs[buf][t][i], w, acc[t]);
            }
        __syncthreads();
    }
    if (tid < kDecOut)
        for (unsigned t = 0; t < rows; ++t) y[static_cast<std::size_t>(t) * n + ob * kDecOut + tid] = acc[t];
}
__global__ void add_kernel(float* h, const float* y, std::size_t n) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i < n) h[i] = h[i] + y[i];
}
__global__ void swiglu_kernel(const float* g, const float* u, float* a, std::size_t n) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i >= n) return;
    const float gg = g[i] < kLimit ? g[i] : kLimit;
    const float uu = u[i] < -kLimit ? -kLimit : (u[i] > kLimit ? kLimit : u[i]);
    a[i] = gg * (1.0F / (1.0F + expf(-kAlpha * gg))) * (uu + 1.0F);
}
__global__ void gather_kernel(const float* x, const std::uint32_t* rows, float* out, unsigned d) {
    const float* src = x + static_cast<std::size_t>(rows[blockIdx.x]) * d;
    float* dst = out + static_cast<std::size_t>(blockIdx.x) * d;
    for (unsigned i = threadIdx.x; i < d; i += 256) dst[i] = src[i];
}
__global__ void scatter_add_kernel(const float* in, const std::uint32_t* rows, const float* w, float* y, unsigned d) {
    float* dst = y + static_cast<std::size_t>(rows[blockIdx.x]) * d;
    const float* src = in + static_cast<std::size_t>(blockIdx.x) * d;
    const float s = w[blockIdx.x];
    for (unsigned i = threadIdx.x; i < d; i += 256) dst[i] = __builtin_fmaf(s, src[i], dst[i]);
}
__global__ void embed_kernel(const std::uint16_t* table, const std::uint32_t* tokens, float* out, unsigned d) {
    const std::uint16_t* src = table + static_cast<std::size_t>(tokens[blockIdx.x]) * d;
    float* dst = out + static_cast<std::size_t>(blockIdx.x) * d;
    for (unsigned i = threadIdx.x; i < d; i += 256) dst[i] = __uint_as_float(static_cast<std::uint32_t>(src[i]) << 16U);
}

}  // namespace

inline unsigned grid(std::size_t n) { return static_cast<unsigned>((n + 255) / 256); }

}  // namespace strix::minimax::engine::detail
