#pragma once

// Shared HIP hyper-connection (HC) core: the device bodies of include/strix/hc/hc_contract.hpp, one kernel family
// for every model with a multi-stream residual. The variant is the contract's Spec, a template parameter:
//   streams          number of residual copies (N);
//   site             sublayer (pre, post [, comb]) or head (pre only);
//   comb             sinkhorn (N x N comb rows, Sinkhorn on the lanes of one wave; N == 4) or identity (no stage);
//   fn               F16 or F32 mix weights;
//   store            FP32 or the bfloat16 rounding of every collapse / expand output;
//   post_magnitude, post_eps   compile-time constants of the post coefficient (post_eps 0: no add, as DS41).
// hidden, the Sinkhorn iterations, the pre/comb eps (hc_eps) and the RMS eps (norm_eps) are runtime arguments, so
// Spec::hidden / sinkhorn_iters / pre_eps / comb_eps / norm_eps are not read here.
//
// The arithmetic is the contract's declared order (chunked lane dots of 640, butterflies 16..1, scale after the dot,
// ascending sums from 0). Callers compile with -ffp-contract=off -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt.
// Bindings: DeepSeek V4.1 (src/hip_ds41_layer.hip: F16 fn, sinkhorn, FP32 store; its fused decode site schedules call the
// bodies below), a Spec with F32 fn, identity comb and bf16 store (tests/hip_hc_core_tests.hip).
// Mix rows are written to a row type M with arrays pre[N] (+ post[N], + comb[N * N] when the Spec has them), e.g.
// MixRow<S> below or a layout-equal model type (::strix::ds41::HcMix).

#include "strix/ds41_layer.hpp"  // declared arithmetic (declared_sigmoid / rsqrt / exp, f16_to_float, round_to_bf16)
#include "strix/hc/hc_contract.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace strix::core::hc {

using strix::hc::Comb;
using strix::hc::FnType;
using strix::hc::Spec;
using strix::hc::Store;

inline constexpr unsigned wave = 32;
inline constexpr unsigned chunk = static_cast<unsigned>(::strix::ds41::dot_chunk);  // elements per wave chunk of the long dots
inline constexpr unsigned chunk_k = chunk / wave;                          // values per lane in a chunk
inline constexpr unsigned mix_tile = 8;                                    // rows per mix-partial block

template <Spec S> inline constexpr unsigned mix_rows = S.mix_rows();
template <Spec S> inline constexpr unsigned mix_waves = S.mix_rows() + 1U;  // one wave per mix row + the squares wave
template <Spec S> using FnElem = std::conditional_t<S.fn == FnType::f16, std::uint16_t, float>;

// The mix row of a Spec (pre, then post, then comb, FP32), for binders without a model type of their own.
template <Spec S, bool Post = (S.post_rows() != 0U), bool HasComb = (S.comb_rows() != 0U)> struct MixRow;
template <Spec S> struct MixRow<S, false, false> {
    float pre[S.streams];
};
template <Spec S> struct MixRow<S, true, false> {
    float pre[S.streams];
    float post[S.streams];
};
template <Spec S> struct MixRow<S, true, true> {
    float pre[S.streams];
    float post[S.streams];
    float comb[S.streams * S.streams];
};

// The sublayer input an expansion reads: p[i], or, with rank partials, the canonical rank sum (p[i] + low[i]) + high[i]
// read in place (FP32 adds only; low and high both set or both null).
// A site dispatched with both terms always present can share the reduction without testing low on each load.
template <bool TermsPresent> struct SubSumView {
    const float* __restrict__ p;
    const float* __restrict__ low;
    const float* __restrict__ high;
    __device__ __forceinline__ float operator[](std::size_t i) const {
        const float s = p[i];
        if constexpr (TermsPresent) return (s + low[i]) + high[i];
        return low ? (s + low[i]) + high[i] : s;
    }
    __device__ __forceinline__ explicit operator bool() const { return p != nullptr; }
};

using SubSum = SubSumView<false>;
using CanonicalSubSum = SubSumView<true>;

__device__ __forceinline__ float butterfly(float v) {
    for (int d = 16; d; d >>= 1) v = v + __shfl_xor(v, d, wave);
    return v;
}
template <Spec S> __device__ __forceinline__ float fn_value(const FnElem<S>* __restrict__ fn, std::size_t i) {
    if constexpr (S.fn == FnType::f16) return ::strix::ds41::f16_to_float(fn[i]);
    else return fn[i];
}
template <Spec S> __device__ __forceinline__ float store_value(float v) {
    if constexpr (S.store == Store::bf16) return ::strix::ds41::round_to_bf16(v);
    else return v;
}
template <Spec S> __device__ __forceinline__ float post_value(float sigmoid) {
    const float v = S.post_magnitude * sigmoid;
    if constexpr (S.post_eps != 0.0F) return v + S.post_eps;
    else return v;
}

// Copy k of element i of row r after the expansion: store(post[k] * sub[i] + s), s = residual[k][i] (identity) or
// 0 + sum_j comb[j][k] * residual[j][i] in ascending j (sinkhorn).
template <Spec S, class M, class Sub>
__device__ __forceinline__ float copy_value(Sub sub, const float* __restrict__ residual, const M* __restrict__ em, unsigned r,
                                            unsigned k, unsigned i, unsigned hidden) {
    static_assert(S.post_rows() != 0U, "expansion needs a sublayer site");
    const float* res = residual + static_cast<std::size_t>(r) * S.streams * hidden;
    const M& m = em[r];
    if constexpr (S.comb == Comb::sinkhorn) {
        float s = 0.0F;
#pragma unroll
        for (unsigned j = 0; j < S.streams; ++j) s = s + m.comb[j * S.streams + k] * res[j * hidden + i];
        return store_value<S>(m.post[k] * sub[static_cast<std::size_t>(r) * hidden + i] + s);
    } else {
        const float s = res[k * hidden + i];
        return store_value<S>(m.post[k] * sub[static_cast<std::size_t>(r) * hidden + i] + s);
    }
}

// Mix partials of chunk c for the rows of a tile: wave m < mix_rows the chunk's lane dot of fn row m with the stream,
// wave mix_rows the chunk's sum of squares; the wave's fn values stay in registers over the tile.
// partial[(r * mix_waves + m) * chunks + c].
template <Spec S>
__device__ __forceinline__ void mix_partial_block(const float* __restrict__ stream, const FnElem<S>* __restrict__ fn,
                                                  float* __restrict__ partial, unsigned hidden, unsigned rows, unsigned chunks,
                                                  unsigned c, unsigned r0) {
    const unsigned m = threadIdx.x / wave, l = threadIdx.x % wave, n = S.streams * hidden;
    const unsigned c0 = c * chunk, len = c0 + chunk < n ? chunk : n - c0, k_count = len / wave;
    const bool squares = m == mix_rows<S>;
    float f[chunk_k];
#pragma unroll
    for (unsigned k = 0; k < chunk_k; ++k)
        f[k] = !squares && k < k_count ? fn_value<S>(fn, static_cast<std::size_t>(m) * n + c0 + k * wave + l) : 0.0F;
    for (unsigned r = r0; r < r0 + mix_tile && r < rows; ++r) {
        const float* x = stream + static_cast<std::size_t>(r) * n + c0 + l;
        float xv[chunk_k];
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k) xv[k] = k < k_count ? x[k * wave] : 0.0F;
        float acc = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < chunk_k; ++k)
            if (k < k_count) acc = acc + (squares ? xv[k] : f[k]) * xv[k];
        acc = butterfly(acc);
        if (l == 0) partial[(static_cast<std::size_t>(r) * mix_waves<S> + m) * chunks + c] = acc;
    }
}

// One wave per row: the ascending chunk sums (lane t <= mix_rows holds dot t), the RMS factor, the split and, for
// Comb::sinkhorn, the Sinkhorn on the lanes: lane t (and t + 16) holds comb entry (j, k) = (t / 4, t % 4); every row or
// column sum is gathered from its four lanes and added in ascending order from 0, as on the host.
template <Spec S, class M>
__device__ __forceinline__ void mix_final_wave(const float* __restrict__ partial, const float* __restrict__ base,
                                               const float* __restrict__ scale, unsigned n, unsigned chunks, unsigned iters,
                                               float hc_eps, float norm_eps, M* __restrict__ out, unsigned r) {
    static_assert(S.comb_rows() == 0U || S.streams == 4U, "the lane Sinkhorn holds a 4 x 4 comb in 16 lanes");
    static_assert(mix_waves<S> <= wave, "one lane per mix row");
    const unsigned t = threadIdx.x % wave;
    float d = 0.0F;
    if (t < mix_waves<S>) {
        const float* p = partial + (static_cast<std::size_t>(r) * mix_waves<S> + t) * chunks;
        for (unsigned c = 0; c < chunks; ++c) d = d + p[c];
    }
    constexpr unsigned h = S.streams;
    const float rr = ::strix::ds41::declared_rsqrt(__shfl(d, static_cast<int>(mix_rows<S>), wave) / static_cast<float>(n) + norm_eps);
    M& o = out[r];
    if constexpr (S.post_rows() == 0U) {
        if (t < h) o.pre[t] = ::strix::ds41::declared_sigmoid((d * rr) * scale[0] + base[t]) + hc_eps;
    } else {
        if (t < h) o.pre[t] = ::strix::ds41::declared_sigmoid((d * rr) * scale[0] + base[t]) + hc_eps;
        else if (t < 2 * h) o.post[t - h] = post_value<S>(::strix::ds41::declared_sigmoid((d * rr) * scale[1] + base[t]));
    }
    if constexpr (S.comb_rows() != 0U) {
        const unsigned e = t & 15U, j = e / h, k = e % h;
        float v = (__shfl(d, static_cast<int>(2 * h + e), wave) * rr) * scale[2] + base[2 * h + e];
        auto at = [&](float x, unsigned jj, unsigned kk) { return __shfl(x, static_cast<int>(jj * h + kk), wave); };
        float mx = at(v, j, 0);
        for (unsigned kk = 1; kk < h; ++kk) { const float y = at(v, j, kk); mx = y > mx ? y : mx; }
        v = ::strix::ds41::declared_exp(v - mx);
        float sum = 0.0F;
        for (unsigned kk = 0; kk < h; ++kk) sum = sum + at(v, j, kk);
        v = v / sum + hc_eps;
        for (unsigned it = 0; it < iters; ++it) {
            if (it) {
                float rs = 0.0F;
                for (unsigned kk = 0; kk < h; ++kk) rs = rs + at(v, j, kk);
                v = v / (rs + hc_eps);
            }
            float cs = 0.0F;
            for (unsigned jj = 0; jj < h; ++jj) cs = cs + at(v, jj, k);
            v = v / (cs + hc_eps);
        }
        if (t < 16) o.comb[t] = v;
    }
}

// ---- kernels -------------------------------------------------------------------------------------------------
template <Spec S>
__global__ __launch_bounds__(32 * mix_waves<S>) void mix_partial_kernel(const float* __restrict__ stream,
                                                                        const FnElem<S>* __restrict__ fn, unsigned hidden,
                                                                        unsigned rows, unsigned chunks, float* __restrict__ partial) {
    mix_partial_block<S>(stream, fn, partial, hidden, rows, chunks, blockIdx.x, blockIdx.y * mix_tile);
}
template <Spec S, class M>
__global__ __launch_bounds__(32) void mix_final_kernel(const float* __restrict__ partial, const float* __restrict__ base,
                                                       const float* __restrict__ scale, unsigned n, unsigned chunks, unsigned iters,
                                                       float hc_eps, float norm_eps, M* __restrict__ out) {
    mix_final_wave<S>(partial, base, scale, n, chunks, iters, hc_eps, norm_eps, out, blockIdx.x);
}
// out[r][i] = store(0 + pre[0] * x[0][i] + ... ascending); null mixes: pre = (1, 0, ..., 0) (a first layer's copy 0).
template <Spec S, class M>
__global__ __launch_bounds__(256) void collapse_kernel(const float* __restrict__ stream, const M* __restrict__ mixes,
                                                       unsigned hidden, unsigned long long total, float* __restrict__ out) {
    const auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const auto r = index / hidden, i = index % hidden;
    const float* x = stream + r * S.streams * hidden;
    float y = 0.0F;
    for (unsigned c = 0; c < S.streams; ++c) {
        const float p = mixes ? mixes[r].pre[c] : (c == 0 ? 1.0F : 0.0F);
        y = y + p * x[c * hidden + i];
    }
    out[index] = store_value<S>(y);
}
template <Spec S, class M, class Sub>
__global__ __launch_bounds__(256) void expand_kernel(Sub sub, const float* __restrict__ residual, const M* __restrict__ mixes,
                                                     unsigned hidden, unsigned long long total, float* __restrict__ out) {
    const auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;  // total = rows * streams * hidden
    const auto per_row = static_cast<unsigned long long>(S.streams) * hidden;
    const auto r = static_cast<unsigned>(index / per_row), rest = static_cast<unsigned>(index % per_row);
    out[index] = copy_value<S>(sub, residual, mixes, r, rest / hidden, rest % hidden, hidden);
}

// ---- host launchers (no checks beyond geometry: the binder validates its views; returns hipGetLastError()) ----
template <Spec S> constexpr std::size_t mix_partial_floats(std::uint32_t rows, std::uint32_t hidden) {
    const std::size_t chunks = (std::size_t{hidden} * S.streams + chunk - 1) / chunk;
    return std::size_t{rows} * mix_waves<S> * chunks;
}
inline unsigned blocks_for(unsigned long long total, unsigned threads) {
    return static_cast<unsigned>((total + threads - 1) / threads);
}
// Mixes of `rows` rows of streams [rows][N * hidden]: partial needs mix_partial_floats<S>(rows, hidden) floats.
// iters is read only for Comb::sinkhorn. hidden * N must be a multiple of 32.
template <Spec S, class M>
hipError_t enqueue_mixes(const float* stream, const FnElem<S>* fn, const float* base, const float* scale, unsigned rows,
                         unsigned hidden, unsigned iters, float hc_eps, float norm_eps, M* out, float* partial, hipStream_t s) {
    const unsigned n = hidden * S.streams, chunks = (n + chunk - 1U) / chunk;
    hipLaunchKernelGGL((mix_partial_kernel<S>), dim3(chunks, (rows + mix_tile - 1U) / mix_tile), dim3(wave * mix_waves<S>), 0, s,
                       stream, fn, hidden, rows, chunks, partial);
    hipLaunchKernelGGL((mix_final_kernel<S, M>), dim3(rows), dim3(wave), 0, s, partial, base, scale, n, chunks, iters, hc_eps,
                       norm_eps, out);
    return hipGetLastError();
}
template <Spec S, class M>
hipError_t enqueue_collapse(const float* stream, const M* mixes, unsigned rows, unsigned hidden, float* out, hipStream_t s) {
    const auto total = static_cast<unsigned long long>(rows) * hidden;
    hipLaunchKernelGGL((collapse_kernel<S, M>), dim3(blocks_for(total, 256)), dim3(256), 0, s, stream, mixes, hidden, total, out);
    return hipGetLastError();
}
template <Spec S, class M, class Sub = SubSum>
hipError_t enqueue_expand(Sub sub, const float* residual, const M* mixes, unsigned rows, unsigned hidden, float* out,
                          hipStream_t s) {
    const auto total = static_cast<unsigned long long>(rows) * S.streams * hidden;
    hipLaunchKernelGGL((expand_kernel<S, M, Sub>), dim3(blocks_for(total, 256)), dim3(256), 0, s, sub, residual, mixes, hidden,
                       total, out);
    return hipGetLastError();
}

} // namespace strix::core::hc
