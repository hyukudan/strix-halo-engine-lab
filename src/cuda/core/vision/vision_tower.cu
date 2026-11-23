// Shared CUDA core: FP32 kernels of the vision tower (see vision_tower.hpp).
#include "vision_tower.hpp"

#include <cmath>
#include <cstdint>

namespace strix::cuda_core::vision {
namespace {

constexpr unsigned kBm = 64, kBn = 64, kBk = 16, kPad = 4;

// 64 x 64 x 16 tile, 256 threads, 4 x 4 outputs per thread, fixed ascending k order.
__global__ __launch_bounds__(256) void gemm_nt_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
                                                      float* __restrict__ y, unsigned m, unsigned n, unsigned k, unsigned ldy, bool accumulate) {
    __shared__ float As[kBk][kBm + kPad];
    __shared__ float Bs[kBk][kBn + kPad];
    const unsigned tid = threadIdx.x, tx = tid % 16U, ty = tid / 16U;
    const std::size_t row0 = static_cast<std::size_t>(blockIdx.y) * kBm, col0 = static_cast<std::size_t>(blockIdx.x) * kBn;
    float acc[4][4] = {};
    for (unsigned k0 = 0; k0 < k; k0 += kBk) {
        for (unsigned i = 0; i < 4U; ++i) {
            const unsigned e = tid + i * 256U, r = e / kBk, kk = e % kBk;
            const std::size_t gr = row0 + r, gc = col0 + r;
            As[kk][r] = gr < m && k0 + kk < k ? x[gr * k + k0 + kk] : 0.0F;
            Bs[kk][r] = gc < n && k0 + kk < k ? w[gc * k + k0 + kk] : 0.0F;
        }
        __syncthreads();
#pragma unroll
        for (unsigned kk = 0; kk < kBk; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) {
                a[i] = As[kk][ty + 16U * i];
                b[i] = Bs[kk][tx + 16U * i];
            }
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (unsigned i = 0; i < 4U; ++i)
        for (unsigned j = 0; j < 4U; ++j) {
            const std::size_t r = row0 + ty + 16U * i, c = col0 + tx + 16U * j;
            if (r >= m || c >= n) continue;
            const float v = acc[i][j] + (bias ? bias[c] : 0.0F);
            float* dst = &y[r * ldy + c];
            *dst = accumulate ? *dst + v : v;
        }
}

__device__ float block_sum(float v, float* sh) {
    for (unsigned o = 16; o; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFU, v, o);
    if ((threadIdx.x & 31U) == 0) sh[threadIdx.x >> 5U] = v;
    __syncthreads();
    float t = 0.0F;
    for (unsigned i = 0; i < blockDim.x / 32U; ++i) t += sh[i];
    __syncthreads();
    return t;
}

// One block per row; two passes over the row (mean, then centred square sum); each thread rewrites only what it read.
__global__ __launch_bounds__(256) void norm_kernel(const float* x, float* y, unsigned width, const float* w, const float* b, float eps, unsigned kind) {
    __shared__ float sh[8];
    const float* xr = x + static_cast<std::size_t>(blockIdx.x) * width;
    float* yr = y + static_cast<std::size_t>(blockIdx.x) * width;
    float mean = 0.0F;
    if (kind == 0U) {
        float s = 0.0F;
        for (unsigned i = threadIdx.x; i < width; i += blockDim.x) s += xr[i];
        mean = block_sum(s, sh) / static_cast<float>(width);
    }
    float q = 0.0F;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        const float d = xr[i] - mean;
        q += d * d;
    }
    const float inv = 1.0F / sqrtf(block_sum(q, sh) / static_cast<float>(width) + eps);
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + (b ? b[i] : 0.0F);
}

__global__ __launch_bounds__(256) void add_kernel(float* x, const float* a, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) x[i] += a[i];
}

__global__ __launch_bounds__(256) void rope_kernel(float* qkv, const float* cs, const float* sn, unsigned rows, unsigned heads, unsigned d, bool adjacent) {
    const unsigned half = d / 2U;
    const std::size_t total = static_cast<std::size_t>(rows) * 2U * heads * half;
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const unsigned p = static_cast<unsigned>(i % half);
    std::size_t t = i / half;
    const unsigned h = static_cast<unsigned>(t % heads);
    t /= heads;
    const unsigned part = static_cast<unsigned>(t % 2U);  // 0: q, 1: k
    const std::size_t r = t / 2U;
    float* v = qkv + r * 3U * heads * d + static_cast<std::size_t>(part) * heads * d + static_cast<std::size_t>(h) * d;
    const unsigned a = adjacent ? 2U * p : p, b = adjacent ? 2U * p + 1U : p + half;
    const float c = cs[r * half + p], s = sn[r * half + p], va = v[a], vb = v[b];
    v[a] = va * c - vb * s;
    v[b] = vb * c + va * s;
}

// One warp per query row and head; keys and values staged 32 at a time; online softmax in FP32.
constexpr unsigned kWarps = 8, kKeys = 32, kMaxD = 128;
__global__ __launch_bounds__(kWarps * 32) void attention_kernel(const float* qkv, float* ao, unsigned row0, unsigned n, unsigned heads, unsigned d) {
    extern __shared__ float sm[];
    const unsigned dp = d + 1U;  // padded row: no bank conflicts on the per-lane key dot
    float* Ks = sm;
    float* Vs = Ks + kKeys * dp;
    float* Qs = Vs + kKeys * dp;
    const unsigned warp = threadIdx.x / 32U, lane = threadIdx.x % 32U, h = blockIdx.y;
    const std::size_t width = static_cast<std::size_t>(heads) * d, stride = 3U * width;
    const unsigned qi = blockIdx.x * kWarps + warp;
    const bool live = qi < n;
    const float* qrow = qkv + (row0 + (live ? qi : 0U)) * stride + static_cast<std::size_t>(h) * d;
    for (unsigned i = lane; i < d; i += 32U) Qs[warp * dp + i] = live ? qrow[i] : 0.0F;
    float acc[kMaxD / 32U] = {};
    float top = __int_as_float(0xff800000), sum = 0.0F;
    const float scale = 1.0F / sqrtf(static_cast<float>(d));
    for (unsigned j0 = 0; j0 < n; j0 += kKeys) {
        const unsigned nk = min(kKeys, n - j0);
        __syncthreads();
        for (unsigned e = threadIdx.x; e < kKeys * d; e += blockDim.x) {
            const unsigned j = e / d, c = e % d;
            const bool ok = j < nk;
            const float* kr = qkv + (row0 + j0 + (ok ? j : 0U)) * stride + width + static_cast<std::size_t>(h) * d;
            Ks[j * dp + c] = ok ? kr[c] : 0.0F;
            Vs[j * dp + c] = ok ? kr[width + c] : 0.0F;
        }
        __syncthreads();
        if (!live) continue;
        float s = __int_as_float(0xff800000);
        if (lane < nk) {
            float dot = 0.0F;
            for (unsigned c = 0; c < d; ++c) dot = fmaf(Qs[warp * dp + c], Ks[lane * dp + c], dot);
            s = dot * scale;
        }
        float cm = s;
        for (unsigned o = 16; o; o >>= 1) cm = fmaxf(cm, __shfl_xor_sync(0xFFFFFFFFU, cm, o));
        const float next = fmaxf(top, cm), alpha = expf(top - next);
        const float pj = lane < nk ? expf(s - next) : 0.0F;
        float ps = pj;
        for (unsigned o = 16; o; o >>= 1) ps += __shfl_xor_sync(0xFFFFFFFFU, ps, o);
        sum = sum * alpha + ps;
        top = next;
        for (unsigned t = 0; t < kMaxD / 32U; ++t) acc[t] *= alpha;
        for (unsigned j = 0; j < nk; ++j) {
            const float p = __shfl_sync(0xFFFFFFFFU, pj, j);
            for (unsigned t = 0; t < kMaxD / 32U; ++t) {
                const unsigned c = lane + 32U * t;
                if (c < d) acc[t] = fmaf(p, Vs[j * dp + c], acc[t]);
            }
        }
    }
    if (!live) return;
    float* out = ao + (row0 + qi) * width + static_cast<std::size_t>(h) * d;
    for (unsigned t = 0; t < kMaxD / 32U; ++t) {
        const unsigned c = lane + 32U * t;
        if (c < d) out[c] = acc[t] / sum;
    }
}

__global__ __launch_bounds__(256) void act_kernel(const float* mid, float* act, unsigned rows, unsigned width, unsigned kind, float limit, bool gate_first) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * width) return;
    const std::size_t r = i / width, j = i % width;
    if (kind == 2U) {
        const float* m = mid + r * 2U * width;
        float gate = m[gate_first ? j : width + j], up = m[gate_first ? width + j : j];
        if (limit > 0.0F) {
            gate = fminf(gate, limit);
            up = fminf(fmaxf(up, -limit), limit);
        }
        act[i] = gate / (1.0F + expf(-gate)) * up;
        return;
    }
    const float v = mid[i];
    act[i] = kind == 0U ? 0.5F * v * (1.0F + erff(v * 0.70710678118654752440F))
                        : 0.5F * v * (1.0F + tanhf(0.79788456080286535588F * (v + 0.044715F * v * v * v)));
}

unsigned blocks_of(std::size_t count) { return static_cast<unsigned>((count + 255U) / 256U); }

}  // namespace

cudaError_t gemm_nt(const float* x, const float* w, const float* bias, float* y, unsigned m, unsigned n, unsigned k, unsigned ldy,
                    bool accumulate, cudaStream_t s) {
    if (!m || !n) return cudaSuccess;
    const dim3 grid((n + kBn - 1U) / kBn, (m + kBm - 1U) / kBm);
    gemm_nt_kernel<<<grid, 256, 0, s>>>(x, w, bias, y, m, n, k, ldy, accumulate);
    return cudaGetLastError();
}
cudaError_t norm_rows(const float* x, float* y, unsigned rows, unsigned width, const float* w, const float* b, float eps, Norm kind, cudaStream_t s) {
    if (!rows) return cudaSuccess;
    norm_kernel<<<rows, 256, 0, s>>>(x, y, width, w, b, eps, static_cast<unsigned>(kind));
    return cudaGetLastError();
}
cudaError_t add_rows(float* x, const float* add, std::size_t count, cudaStream_t s) {
    if (!count) return cudaSuccess;
    add_kernel<<<blocks_of(count), 256, 0, s>>>(x, add, count);
    return cudaGetLastError();
}
cudaError_t rope_qkv(float* qkv, const float* cs, const float* sn, unsigned rows, unsigned heads, unsigned d, bool adjacent, cudaStream_t s) {
    const std::size_t total = static_cast<std::size_t>(rows) * 2U * heads * (d / 2U);
    if (!total) return cudaSuccess;
    rope_kernel<<<blocks_of(total), 256, 0, s>>>(qkv, cs, sn, rows, heads, d, adjacent);
    return cudaGetLastError();
}
cudaError_t attention_tile(const float* qkv, float* ao, unsigned row0, unsigned n, unsigned heads, unsigned d, cudaStream_t s) {
    if (!n) return cudaSuccess;
    if (d > kMaxD || d == 0) return cudaErrorInvalidValue;
    const std::size_t smem = sizeof(float) * (2U * kKeys + kWarps) * (d + 1U);
    const dim3 grid((n + kWarps - 1U) / kWarps, heads);
    attention_kernel<<<grid, kWarps * 32U, smem, s>>>(qkv, ao, row0, n, heads, d);
    return cudaGetLastError();
}
cudaError_t activate_rows(const float* mid, float* act, unsigned rows, unsigned width, Act kind, float swiglu_limit, bool gate_first, cudaStream_t s) {
    const std::size_t total = static_cast<std::size_t>(rows) * width;
    if (!total) return cudaSuccess;
    act_kernel<<<blocks_of(total), 256, 0, s>>>(mid, act, rows, width, static_cast<unsigned>(kind), swiglu_limit, gate_first);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core::vision
