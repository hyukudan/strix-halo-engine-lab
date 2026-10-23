// deepseek41 CUDA layer core: RMSNorm (with the hyper-connection collapse and A8 fused), hc mixes with the Sinkhorn
// split, and the hyper-connection expansion. Bitwise the host reference of ds41_layer.hpp.

#include "ds41_cuda_common.cuh"

#include "strix/cuda_ds41_core.hpp"

namespace strix::ds41::cuda {
namespace {

using detail::chunk;
using detail::full_mask;
using detail::require;

constexpr std::uint32_t kNormThreads = 256, kMaxNorm = 8192, kMaxChunks = kMaxNorm / chunk + 1;

// grid (parts, rows). Every block builds the whole input row in shared memory (the collapse is recomputed per block)
// and the chunk sums of squares; part b writes the outputs of [b * span, (b + 1) * span) (span a multiple of 32, so
// whole A8 groups). One part per row when the norm runs in place.
__device__ __forceinline__ void norm_body(std::uint32_t r, std::uint32_t part_index, const float* __restrict__ x,
            std::uint32_t x_stride, const float* __restrict__ hc, const float* __restrict__ pre,
            std::uint32_t pre_stride, const float* __restrict__ weight, std::uint32_t n, float eps, float* out, std::uint32_t out_stride,
            float* __restrict__ collapsed, std::int8_t* __restrict__ codes, float* __restrict__ deltas, std::int32_t* __restrict__ sums,
            std::uint32_t a8_stride, std::uint32_t span) {
    extern __shared__ float xs[];
    __shared__ float part[kMaxChunks];
    __shared__ float rr;
    const std::uint32_t tid = threadIdx.x, lane = tid & 31U, warp = tid / 32U, warps = blockDim.x / 32U;
    const std::uint32_t lo = part_index * span, hi = lo + span < n ? lo + span : n;
    float p[4] = {1.0F, 0.0F, 0.0F, 0.0F};
    if (hc && pre)
        for (int c = 0; c < 4; ++c) p[c] = pre[std::size_t{r} * pre_stride + c];
    for (std::uint32_t i = tid; i < n; i += blockDim.x) {
        float y;
        if (hc) {
            const float* s = hc + std::size_t{r} * 4U * n + i;
            y = 0.0F;
            for (int c = 0; c < 4; ++c) y = y + p[c] * s[std::size_t(c) * n];
            if (collapsed && i >= lo && i < hi) collapsed[std::size_t{r} * n + i] = y;
        } else {
            y = x[std::size_t{r} * x_stride + i];
        }
        xs[i] = y;
    }
    __syncthreads();
    if (!weight) return;
    const std::uint32_t chunks = (n + chunk - 1U) / chunk;
    for (std::uint32_t c = warp; c < chunks; c += warps) {
        const float* xc = xs + c * chunk;
        const std::uint32_t len = n - c * chunk < chunk ? n - c * chunk : chunk;
        const float s = detail::lane_dot([&](std::uint32_t i) { return xc[i]; }, [&](std::uint32_t i) { return xc[i]; }, len, lane);
        if (lane == 0) part[c] = s;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0F;
        for (std::uint32_t c = 0; c < chunks; ++c) total = total + part[c];
        const float mean = total / static_cast<float>(n);
        rr = declared_rsqrt(mean + eps);
    }
    __syncthreads();
    const float rv = rr;
    for (std::uint32_t i = lo + tid; i < hi; i += blockDim.x) {
        const float y = weight[i] * (xs[i] * rv);
        xs[i] = y;
        if (out) out[std::size_t{r} * out_stride + i] = y;
    }
    if (!codes) return;
    __syncthreads();
    detail::a8_row(xs + lo, hi - lo, codes + std::size_t{r} * a8_stride + lo, deltas + (std::size_t{r} * a8_stride + lo) / 32U,
                   sums + (std::size_t{r} * a8_stride + lo) / 32U);
}

__global__ void __launch_bounds__(kNormThreads) norm_kernel(NormView n, std::uint32_t span) {
    norm_body(blockIdx.y, blockIdx.x, n.x, n.x_stride ? n.x_stride : n.n, n.hc_stream, n.pre, n.pre_stride,
              n.weight, n.n, n.eps, n.out, n.out_stride ? n.out_stride : n.n, n.collapsed,
              n.a8.codes, n.a8.deltas, n.a8.sums, n.a8.stride ? n.a8.stride : n.n, span);
}

// ---- hc mixes ------------------------------------------------------------------------------------------------------
constexpr std::uint32_t kDots = hc_mix_count + 1U;  // 24 fn rows + the sum of squares

// grid (chunks, rows), kDots warps: warp m < 24 the chunk dot of fn row m with the stream, warp 24 the stream with itself.
__global__ void __launch_bounds__(kDots * 32)
hc_partial_kernel(const float* __restrict__ stream, const std::uint16_t* __restrict__ fn, std::uint32_t total_n,
                  float* __restrict__ partial) {
    const std::uint32_t c = blockIdx.x, r = blockIdx.y, chunks = gridDim.x, m = threadIdx.x / 32U, lane = threadIdx.x & 31U;
    const std::uint32_t c0 = c * chunk, len = total_n - c0 < chunk ? total_n - c0 : chunk;
    const float* s = stream + std::size_t{r} * total_n + c0;
    float v;
    if (m < hc_mix_count) {
        const std::uint16_t* w = fn + std::size_t{m} * total_n + c0;
        v = detail::lane_dot([&](std::uint32_t i) { return f16_to_float(w[i]); }, [&](std::uint32_t i) { return s[i]; }, len, lane);
    } else {
        v = detail::lane_dot([&](std::uint32_t i) { return s[i]; }, [&](std::uint32_t i) { return s[i]; }, len, lane);
    }
    if (lane == 0) partial[(std::size_t{r} * kDots + m) * chunks + c] = v;
}

// One warp per row: lanes 0..24 sum their dot's chunks in ascending order; then the split (hc_split_sinkhorn), with
// lane t = 4 j + k (t < 16) holding comb[j][k]; row and column sums are taken in ascending index by every lane.
__device__ __forceinline__ void hc_split_body(std::uint32_t r, const float* __restrict__ partial, std::uint32_t chunks, std::uint32_t total_n,
                                const float* __restrict__ base, const float* __restrict__ scale, std::uint32_t iters, float hc_eps,
                                float norm_eps, float* __restrict__ mixes) {
    const std::uint32_t lane = threadIdx.x;
    float total = 0.0F;
    if (lane < kDots) {
        const float* p = partial + (std::size_t{r} * kDots + lane) * chunks;
#pragma unroll 16
        for (std::uint32_t c = 0; c < chunks; ++c) total = total + p[c];
    }
    const float sq = __shfl_sync(full_mask, total, hc_mix_count);
    const float rs = declared_rsqrt(sq / static_cast<float>(total_n) + norm_eps);
    const float m = total * rs;  // lane m < 24: mixes[m]
    float* out = mixes + std::size_t{r} * hc_mix_count;
    if (lane < 4U) out[lane] = declared_sigmoid(m * scale[0] + base[lane]) + hc_eps;
    else if (lane < 8U) out[lane] = 2.0F * declared_sigmoid(m * scale[1] + base[lane]);
    // comb: lane t < 16 takes mixes[8 + t]
    const std::uint32_t t = lane & 15U, j = t / 4U, k = t % 4U;
    float cv = __shfl_sync(full_mask, m, 8U + t);
    cv = cv * scale[2] + base[8U + t];
    // row softmax: max over the row (first then ascending), exp(c - max), ascending sum, c / sum + eps
    float row[4];
    for (std::uint32_t q = 0; q < 4U; ++q) row[q] = __shfl_sync(full_mask, cv, 4U * j + q);
    float mx = row[0];
    for (std::uint32_t q = 1; q < 4U; ++q) mx = row[q] > mx ? row[q] : mx;
    cv = declared_exp(cv - mx);
    for (std::uint32_t q = 0; q < 4U; ++q) row[q] = __shfl_sync(full_mask, cv, 4U * j + q);
    float sum = 0.0F;
    for (std::uint32_t q = 0; q < 4U; ++q) sum = sum + row[q];
    cv = cv / sum + hc_eps;
    auto columns = [&] {
        float s = 0.0F;
        for (std::uint32_t q = 0; q < 4U; ++q) s = s + __shfl_sync(full_mask, cv, 4U * q + k);
        cv = cv / (s + hc_eps);
    };
    auto rows = [&] {
        float s = 0.0F;
        for (std::uint32_t q = 0; q < 4U; ++q) s = s + __shfl_sync(full_mask, cv, 4U * j + q);
        cv = cv / (s + hc_eps);
    };
    columns();
    for (std::uint32_t it = 1; it < iters; ++it) { rows(); columns(); }
    if (lane < 16U) out[8U + lane] = cv;
}

__global__ void hc_split_kernel(HcMixView m, std::uint32_t chunks) {
    hc_split_body(blockIdx.x, m.partial, chunks, 4U * m.hidden, m.base, m.scale, m.iters, m.hc_eps, m.norm_eps, m.mixes);
}

// Split and norm use independent pre coefficients. Block 0 does the Sinkhorn split; the other blocks
// do exactly the legacy norm parts. No block consumes another block's writes in this launch.
__global__ void __launch_bounds__(kNormThreads) hc_split_norm_kernel(HcMixView m, NormView n, std::uint32_t chunks) {
    if (blockIdx.x == 0U) {
        if (threadIdx.x < 32U)
            hc_split_body(blockIdx.y, m.partial, chunks, 4U * m.hidden, m.base, m.scale, m.iters, m.hc_eps, m.norm_eps, m.mixes);
    } else {
        norm_body(blockIdx.y, blockIdx.x - 1U, n.x, n.x_stride ? n.x_stride : n.n, n.hc_stream, n.pre, n.pre_stride,
                  n.weight, n.n, n.eps, n.out, n.out_stride ? n.out_stride : n.n, n.collapsed,
                  n.a8.codes, n.a8.deltas, n.a8.sums, n.a8.stride ? n.a8.stride : n.n, chunk);
    }
}

// ---- hc expand -----------------------------------------------------------------------------------------------------
__global__ void hc_expand_kernel(const float* __restrict__ sub, std::uint32_t sub_stride, const float* residual,
                                 const float* __restrict__ mixes, std::uint32_t rows, std::uint32_t hidden, float* out, float* pre_carry) {
    const std::size_t idx = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (idx >= std::size_t{rows} * hidden) return;
    const std::uint32_t r = static_cast<std::uint32_t>(idx / hidden), i = static_cast<std::uint32_t>(idx % hidden);
    const float* mix = mixes + std::size_t{r} * hc_mix_count;
    if (pre_carry && i == 0U)
        for (int c = 0; c < 4; ++c) pre_carry[std::size_t{r} * 4U + c] = mix[c];
    const float sv = sub[std::size_t{r} * sub_stride + i];
    float res[4];
    for (int j = 0; j < 4; ++j) res[j] = residual[(std::size_t{r} * 4U + j) * hidden + i];
    float y[4];
    for (int k = 0; k < 4; ++k) {
        float s = 0.0F;
        for (int j = 0; j < 4; ++j) s = s + mix[8 + j * 4 + k] * res[j];
        y[k] = mix[4 + k] * sv + s;
    }
    for (int k = 0; k < 4; ++k) out[(std::size_t{r} * 4U + k) * hidden + i] = y[k];
}

}  // namespace

void enqueue_norm(const NormView& v, cudaStream_t stream) {
    require(v.rows > 0 && v.n > 0 && v.n % 32U == 0 && v.n <= kMaxNorm && (v.x || v.hc_stream), "norm geometry");
    require(v.weight || (!v.out && !v.a8.codes), "norm outputs need a weight");
    require(!v.hc_stream || !v.pre || v.pre_stride >= 4U, "norm pre stride");
    // in place (out aliases x): one block per row; otherwise one block per 640-value part of the row
    const bool in_place = v.x && !v.hc_stream && v.out == v.x;
    const std::uint32_t span = in_place ? v.n : chunk, parts = (v.n + span - 1U) / span;
    norm_kernel<<<dim3(parts, v.rows), kNormThreads, v.n * sizeof(float), stream>>>(v, span);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

std::size_t hc_mix_workspace_bytes(std::uint32_t rows, std::uint32_t hidden) {
    return std::size_t{rows} * kDots * ((4U * hidden + chunk - 1U) / chunk) * sizeof(float);
}

void enqueue_hc_mixes(const HcMixView& v, cudaStream_t stream) {
    const std::uint32_t total_n = 4U * v.hidden, chunks = (total_n + chunk - 1U) / chunk;
    require(v.rows > 0 && v.hidden > 0 && v.iters >= 1 && v.partial && v.mixes, "hc mix geometry");
    hc_partial_kernel<<<dim3(chunks, v.rows), kDots * 32U, 0, stream>>>(v.stream, v.fn, total_n, v.partial);
    STRIX_CUDA_CHECK(cudaGetLastError());
    hc_split_kernel<<<v.rows, 32, 0, stream>>>(v, chunks);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_hc_mixes_norm(const HcMixView& m, const NormView& n, cudaStream_t stream) {
    require(m.rows > 0 && m.hidden > 0 && m.iters >= 1 && m.partial && m.mixes && m.stream && m.fn && m.base && m.scale,
            "hc mix norm geometry");
    require(n.rows == m.rows && n.n == m.hidden && n.n % 32U == 0 && n.n <= kMaxNorm &&
                n.hc_stream == m.stream && n.pre && n.pre != m.mixes && n.pre_stride >= 4U && n.weight,
            "hc mix norm needs independent pre coefficients and matching stream");
    const std::uint32_t total_n = 4U * m.hidden, chunks = (total_n + chunk - 1U) / chunk;
    hc_partial_kernel<<<dim3(chunks, m.rows), kDots * 32U, 0, stream>>>(m.stream, m.fn, total_n, m.partial);
    STRIX_CUDA_CHECK(cudaGetLastError());
    hc_split_norm_kernel<<<dim3(1U + (n.n + chunk - 1U) / chunk, n.rows), kNormThreads,
                           n.n * sizeof(float), stream>>>(m, n, chunks);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_hc_expand(const HcExpandView& v, cudaStream_t stream) {
    require(v.rows > 0 && v.hidden > 0, "hc expand geometry");
    const std::size_t n = std::size_t{v.rows} * v.hidden;
    hc_expand_kernel<<<static_cast<unsigned>((n + 255U) / 256U), 256, 0, stream>>>(v.sub, v.sub_stride ? v.sub_stride : v.hidden,
                                                                                   v.residual, v.mixes, v.rows, v.hidden, v.out, v.pre_carry);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda
