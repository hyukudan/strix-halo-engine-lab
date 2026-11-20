#pragma once
// Shared HIP hyper-connection (HC / mHC) row expansion, bandwidth-organised: one row of P::kStreams residual streams of
// P::kHidden floats per work-group. Out[i] = P::expand(mix, i, X[0 .. S-1], y) per column, optionally followed by the
// next site's norm factor rr = P::finish(tree of the per-lane sums of squares).
//
// The reference organisation is "virtual lane" l in [0, 256) owning columns d = b * 256 + l (b = 0 .. H/256-1): lane l's
// fmaf chain runs over stream 0 in ascending b, then streams 1 .. S-1 in ascending b; the 256 lane partials enter
// P::tree256_lane0. This kernel keeps that ownership and every operation order; only the placement changes:
//   kLanes  virtual lanes per thread (1, 2 or 4): thread t owns lanes t*kLanes .. t*kLanes+kLanes-1 (contiguous columns,
//           one kLanes-wide access per load / store);
//   mix     the row's coefficients are loaded once into registers (P::load_mix); every pointer is restrict, so all X / y
//           loads of a step may issue before its stores (callers guarantee disjoint X / Xo / y / mix / rr);
//   kNT     cache policy only (values unchanged): 0 none, 1 streaming Xo stores, 2 also streaming X loads (each row is
//           read once; keeps the streamed rows from evicting a concurrent kernel's working set).
// Policy P (model arithmetic, compiled under the binding TU's FP flags, e.g. -ffp-contract=off):
//   kStreams, kHidden, kMixStride, kNorm, kMinWavesPerEu; struct Mix; Mix load_mix(const float* __restrict__);
//   float expand(const Mix&, unsigned i, const float* x /*kStreams*/, float y); float pair(float y0, float y1);
//   bool finite(float); float tree256_lane0(const float* s, unsigned lane /*< 32*/); float finish(float ss).
// Bindings: GLM-5.3 prefill (src/glm/prefill/glm_hc_expand_bw.inc: lane-chain order + lds tree, fused next-site norm).
// DeepSeek V4.1's expansion (src/core/hc_core.hpp, contract order: chunked lane dots) has another arithmetic contract.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::core::hc::rows {

template <unsigned N> struct LaneVec;
template <> struct LaneVec<1> { using T = float; };
template <> struct LaneVec<2> { typedef float T __attribute__((ext_vector_type(2))); };
template <> struct LaneVec<4> { typedef float T __attribute__((ext_vector_type(4))); };

template <unsigned kLanes, bool kNT> __device__ __forceinline__ void load_lanes(const float* __restrict__ p, float (&o)[kLanes]) {
    if constexpr (kLanes == 1) {
        o[0] = kNT ? __builtin_nontemporal_load(p) : *p;
    } else {
        using V = typename LaneVec<kLanes>::T;
        const V v = kNT ? __builtin_nontemporal_load(reinterpret_cast<const V*>(p)) : *reinterpret_cast<const V*>(p);
#pragma unroll
        for (unsigned k = 0; k < kLanes; ++k) o[k] = v[k];
    }
}
template <unsigned kLanes, bool kNT> __device__ __forceinline__ void store_lanes(float* __restrict__ p, const float (&o)[kLanes]) {
    if constexpr (kLanes == 1) {
        if constexpr (kNT) __builtin_nontemporal_store(o[0], p);
        else *p = o[0];
    } else {
        using V = typename LaneVec<kLanes>::T;
        V v;
#pragma unroll
        for (unsigned k = 0; k < kLanes; ++k) v[k] = o[k];
        if constexpr (kNT) __builtin_nontemporal_store(v, reinterpret_cast<V*>(p));
        else *reinterpret_cast<V*>(p) = v;
    }
}

// One row per work-group of 256 / kLanes threads. amdgpu_waves_per_eu(P::kMinWavesPerEu) caps the VGPR budget (the
// hoisted loads would otherwise take every VGPR and spill).
template <class P, bool kPair, unsigned kLanes, unsigned kNT>
__global__ __launch_bounds__(256 / kLanes) __attribute__((amdgpu_waves_per_eu(P::kMinWavesPerEu))) void expand_rows_kernel(
    const float* __restrict__ X, const float* __restrict__ mix, const float* __restrict__ y, const float* __restrict__ y1,
    float* __restrict__ Xo, float* __restrict__ rr_out, unsigned* flag) {
    constexpr unsigned S = P::kStreams, H = P::kHidden, B = H / 256;
    static_assert(kLanes == 1 || kLanes == 2 || kLanes == 4);
    static_assert(H % 256 == 0 && S >= 1);
    __shared__ float s[256];
    const unsigned t = threadIdx.x, l0 = t * kLanes, r = blockIdx.x;
    const typename P::Mix M = P::load_mix(mix + static_cast<std::size_t>(r) * P::kMixStride);
    const float* __restrict__ Xr = X + static_cast<std::size_t>(r) * S * H;
    float* __restrict__ Xor = Xo + static_cast<std::size_t>(r) * S * H;
    const float* __restrict__ yr = y + static_cast<std::size_t>(r) * H;
    float yv[B][kLanes];
#pragma unroll
    for (unsigned b = 0; b < B; ++b) {
        load_lanes<kLanes, false>(yr + b * 256 + l0, yv[b]);
        if constexpr (kPair) {
            float q[kLanes];
            load_lanes<kLanes, false>(y1 + static_cast<std::size_t>(r) * H + b * 256 + l0, q);
#pragma unroll
            for (unsigned k = 0; k < kLanes; ++k) yv[b][k] = P::pair(yv[b][k], q[k]);
        }
    }
    float a[kLanes];
#pragma unroll
    for (unsigned k = 0; k < kLanes; ++k) a[k] = 0.0F;
    float v[P::kNorm && S > 1 ? S - 1 : 1][P::kNorm ? B : 1][kLanes];
    bool bad = false;
#pragma unroll
    for (unsigned b = 0; b < B; ++b) {
        float x[S][kLanes];
#pragma unroll
        for (unsigned j = 0; j < S; ++j) load_lanes<kLanes, (kNT >= 2)>(Xr + j * H + b * 256 + l0, x[j]);
#pragma unroll
        for (unsigned i = 0; i < S; ++i) {
            float o[kLanes];
#pragma unroll
            for (unsigned k = 0; k < kLanes; ++k) {
                float xs[S];
#pragma unroll
                for (unsigned j = 0; j < S; ++j) xs[j] = x[j][k];
                o[k] = P::expand(M, i, xs, yv[b][k]);
                bad = bad || !P::finite(o[k]);
                if constexpr (P::kNorm) {
                    if (i == 0) a[k] = __builtin_fmaf(o[k], o[k], a[k]);
                    else v[i - 1][b][k] = o[k];
                }
            }
            store_lanes<kLanes, (kNT >= 1)>(Xor + i * H + b * 256 + l0, o);
        }
    }
    if (flag && bad) atomicOr(flag, 1U);
    if constexpr (P::kNorm) {
#pragma unroll
        for (unsigned i = 0; i + 1 < S; ++i)
#pragma unroll
            for (unsigned b = 0; b < B; ++b)
#pragma unroll
                for (unsigned k = 0; k < kLanes; ++k) a[k] = __builtin_fmaf(v[i][b][k], v[i][b][k], a[k]);
#pragma unroll
        for (unsigned k = 0; k < kLanes; ++k) s[l0 + k] = a[k];
        __syncthreads();
        if (t < 32) {
            const float ss = P::tree256_lane0(s, t);
            if (t == 0) rr_out[r] = P::finish(ss);
        }
    }
}

template <class P, bool kPair, unsigned kLanes, unsigned kNT>
inline void launch_expand_rows(const float* X, const float* mix, const float* y, const float* y1, float* Xo, float* rr, unsigned* flag,
                               unsigned rows, hipStream_t s) {
    if (rows == 0) return;
    expand_rows_kernel<P, kPair, kLanes, kNT><<<rows, 256 / kLanes, 0, s>>>(X, mix, y, y1, Xo, rr, flag);
}

// Host contract of the restrict pointers: written ranges (Xo, rr) disjoint from every other range, vector-aligned rows.
inline bool ranges_disjoint(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto pa = reinterpret_cast<std::uintptr_t>(a), pb = reinterpret_cast<std::uintptr_t>(b);
    return a == nullptr || b == nullptr || pa + an <= pb || pb + bn <= pa;
}
inline bool aligned_to(const void* p, std::size_t bytes) { return reinterpret_cast<std::uintptr_t>(p) % bytes == 0; }
template <class P>
inline bool expand_rows_admissible(unsigned lanes, const float* X, const float* mix, const float* y, const float* y1, const float* Xo,
                                   const float* rr, unsigned rows) {
    const std::size_t nx = std::size_t{rows} * P::kStreams * P::kHidden * sizeof(float), ny = std::size_t{rows} * P::kHidden * sizeof(float);
    const std::size_t nm = std::size_t{rows} * P::kMixStride * sizeof(float), nr = std::size_t{rows} * sizeof(float);
    const std::size_t vec = std::size_t{lanes} * sizeof(float);
    const bool disjoint = ranges_disjoint(Xo, nx, X, nx) && ranges_disjoint(Xo, nx, y, ny) && ranges_disjoint(Xo, nx, y1, ny) &&
                          ranges_disjoint(Xo, nx, mix, nm) && ranges_disjoint(Xo, nx, rr, nr) && ranges_disjoint(rr, nr, X, nx) &&
                          ranges_disjoint(rr, nr, y, ny) && ranges_disjoint(rr, nr, y1, ny) && ranges_disjoint(rr, nr, mix, nm);
    return disjoint && aligned_to(X, vec) && aligned_to(y, vec) && (y1 == nullptr || aligned_to(y1, vec)) && aligned_to(Xo, vec);
}

}  // namespace strix::core::hc::rows
