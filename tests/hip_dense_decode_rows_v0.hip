// See tests/hip_dense_decode_rows_v0.hpp. Compiled with STRIX_HIP_EXACT_FP_OPTIONS only (never the test's own flags).
#include "hip_dense_decode_rows_v0.hpp"

namespace dd {  // frozen copies of the shared helpers v0 uses (attn_core/dense_decode.hpp as on main)
using Params = rows_v0::Params;
inline constexpr unsigned waves = 8U;
__device__ __forceinline__ std::uint16_t to_bf16(float x) {
    const auto u = __builtin_bit_cast(std::uint32_t, x);
    if ((u & 0x7fffffffU) > 0x7f800000U)
        return std::uint16_t((u >> 16U) | 0x40U);
    return std::uint16_t((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ float from_bf16(std::uint32_t h) { return __builtin_bit_cast(float, h << 16U); }
}  // namespace dd

namespace {
__device__ __forceinline__ std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16U; x *= 0x7feb352dU; x ^= x >> 15U; x *= 0x846ca68bU; x ^= x >> 16U;
    return x;
}
__device__ __forceinline__ float unit(std::size_t i, std::uint32_t seed) {  // [-2, 2)
    return float(mix(std::uint32_t(i) ^ mix(std::uint32_t(i >> 32U) + seed)) >> 8U) * (4.0F / 16777216.0F) - 2.0F;
}
__global__ void fill_bf16(std::uint16_t* x, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        x[i] = dd::to_bf16(unit(i, seed));
}
__global__ void fill_f32(float* x, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        x[i] = unit(i, seed);
}

namespace v0 {  // frozen: attend_kernel as on main da843ffe3 .. 953449df5 (grid (rows, heads)), source unchanged
template <unsigned Dim>
__global__ void __launch_bounds__(32U * dd::waves) attend_kernel(dd::Params p, const float* q_n, const float* q_in,
                                                                 const std::uint16_t* k_cache, const std::uint16_t* v_cache,
                                                                 float* ctx) {
    constexpr unsigned per_lane = Dim / 32U, waves = dd::waves;
    __shared__ float m_s[waves], l_s[waves];
    __shared__ float acc_s[waves][Dim];
    const unsigned row = blockIdx.x, head = blockIdx.y, wave = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const unsigned group = p.heads / p.kv_heads, kh = head / group;
    const unsigned last = p.pos0 + row;
    float q[per_lane];
    for (unsigned e = 0; e < per_lane; ++e)
        q[e] = q_n[(std::size_t(row) * p.heads + head) * Dim + lane * per_lane + e];
    float m = -INFINITY, l = 0.0F, acc[per_lane] = {};
    for (unsigned pos = wave; pos <= last; pos += waves) {
        const std::size_t at = (std::size_t(pos) * p.kv_heads + kh) * Dim + lane * per_lane;
        float s = 0.0F;
        for (unsigned e = 0; e < per_lane; ++e)
            s = __builtin_fmaf(q[e], dd::from_bf16(k_cache[at + e]), s);
#pragma unroll
        for (unsigned o = 16U; o > 0U; o >>= 1U)
            s += __shfl_xor(s, int(o), 32);
        s *= p.scale;
        const float mn = fmaxf(m, s), corr = expf(m - mn), w = expf(s - mn);
        l = l * corr + w;
        for (unsigned e = 0; e < per_lane; ++e)
            acc[e] = acc[e] * corr + w * dd::from_bf16(v_cache[at + e]);
        m = mn;
    }
    for (unsigned e = 0; e < per_lane; ++e)
        acc_s[wave][lane * per_lane + e] = acc[e];
    if (lane == 0) {
        m_s[wave] = m;
        l_s[wave] = l;
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < Dim; d += 32U * waves) {
        float M = -INFINITY;
        for (unsigned w = 0; w < waves; ++w)
            M = fmaxf(M, m_s[w]);
        float L = 0.0F, O = 0.0F;
        for (unsigned w = 0; w < waves; ++w) {
            const float f = m_s[w] == -INFINITY ? 0.0F : expf(m_s[w] - M);
            L += l_s[w] * f;
            O += acc_s[w][d] * f;
        }
        float o = O / L;
        if (p.gated) {
            const float g = q_in[(std::size_t(row) * p.heads + head) * p.q_stride + Dim + d];
            o *= 1.0F / (1.0F + expf(-g));
        }
        ctx[(std::size_t(row) * p.heads + head) * Dim + d] = o;
    }
}
}  // namespace v0
}  // namespace

namespace rows_v0 {
void fill(float* qn, std::size_t nq, float* qin, std::size_t nqin, std::uint16_t* k, std::uint16_t* v, std::size_t nkv) {
    fill_f32<<<256, 256>>>(qn, nq, 11U);
    fill_f32<<<256, 256>>>(qin, nqin, 12U);
    fill_bf16<<<1024, 256>>>(k, nkv, 13U);
    fill_bf16<<<1024, 256>>>(v, nkv, 14U);
}
void attend(const dd::Params& p, const float* qn, const float* qin, const std::uint16_t* k, const std::uint16_t* v,
            float* ctx, hipStream_t s) {
    v0::attend_kernel<256U><<<dim3(p.rows, p.heads), dim3(32U * dd::waves), 0, s>>>(p, qn, qin, k, v, ctx);
}
}  // namespace rows_v0
