// MiniMax RTX prefill oracle from wave10 6fb09ae7c,
// src/cuda/models/minimax/mm_rtx_kernels.cu:15,24-47,98-114.
namespace frozen {

__device__ __forceinline__ float u2f(std::uint32_t b) { return __uint_as_float(b); }
__device__ float exp_f(float x) {
    if (x < -87.0F) return 0.0F;
    if (x > 88.0F) return u2f(0x7f800000U);
    const float n = rintf(x * 1.44269504088896341F);
    float r = __fmaf_rn(n, -0.693145751953125F, x);
    r = __fmaf_rn(n, -1.42860676533018e-06F, r);
    float p = 2.48015873e-05F;
    p = __fmaf_rn(p, r, 1.98412698e-04F);
    p = __fmaf_rn(p, r, 1.38888889e-03F);
    p = __fmaf_rn(p, r, 8.33333333e-03F);
    p = __fmaf_rn(p, r, 4.16666667e-02F);
    p = __fmaf_rn(p, r, 1.66666667e-01F);
    p = __fmaf_rn(p, r, 0.5F);
    p = __fmaf_rn(p, r, 1.0F);
    p = __fmaf_rn(p, r, 1.0F);
    return p * u2f(static_cast<std::uint32_t>(static_cast<int>(n) + 127) << 23U);
}
__device__ __forceinline__ float shx(float v, int d) { return __shfl_xor_sync(0xffffffffU, v, d, 32); }
__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v = v + shx(v, d);
    return v;
}

__global__ void router_logits_kernel(const float* y, const float* rw, const float* bias, std::size_t rows, float* scratch) {
    const std::size_t warp = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x & 31U;
    if (warp >= rows * kExperts) return;
    const std::size_t r = warp / kExperts;
    const unsigned e = static_cast<unsigned>(warp % kExperts);
    const float* yr = y + r * kHidden;
    const float* we = rw + static_cast<std::size_t>(e) * kHidden;
    float acc = 0.0F;
    for (unsigned j = lane; j < kHidden; j += 32) acc = __fmaf_rn(yr[j], we[j], acc);
    acc = butterfly(acc);
    if (lane == 0) {
        const float s = 1.0F / (1.0F + exp_f(-acc));
        scratch[(r * kExperts + e) * 2] = s;
        scratch[(r * kExperts + e) * 2 + 1] = s + bias[e];
    }
}

}  // namespace frozen
