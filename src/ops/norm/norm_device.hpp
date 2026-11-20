#pragma once

#include "norm.hpp"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <type_traits>

namespace strix::ops::norm {

// No second store (the plain norm).
struct NoStore {
    __device__ void operator()(std::size_t, float) const {}
};

// One row of the norm: L lane-strided FMA chains, then the engine's LDS halving tree. Traits adapters must inline
// without changing evaluation order.
// Residual (decode launch fusion): x[i] = x[i] + r[i] is stored back before the reduction, the same FP32 add as a
// separate residual-add kernel, so x and y keep their bits and one launch per norm is saved.
// Store (decode launch fusion): store(base + i, y[i]) after the F32 store, so a consumer's input conversion (e.g. the
// prepared FP16 rows of dense_iu4) is produced here instead of by its own launch. NoStore = the plain norm.
template <class Traits, unsigned L, bool Residual, class Store>
__device__ __forceinline__ void rmsnorm_row(float* s, const float* xr, float* xw, const float* rr,
                                            const typename Traits::Scale* w, typename Traits::ScaleFormat wt, unsigned n,
                                            float eps, float* yr, std::size_t base, const Store& store) {
    static_assert(Traits::reduction_order == ReductionOrder::lane_strided_fma_lds_halving);
    float a = 0.0F;
    for (unsigned i = threadIdx.x; i < n; i += L) {
        if constexpr (Residual) {
            const float v = xr[i] + rr[i];
            xw[i] = v;
            a = __builtin_fmaf(v, v, a);
        } else {
            a = __builtin_fmaf(xr[i], xr[i], a);
        }
    }
    const float ss = Traits::template reduce<L>(s, a);
    const float r = Traits::inv_sqrt(ss / static_cast<float>(n) + eps);
    constexpr OutputForm form = output_form_of<Traits>();  // the Traits' output rounding (norm.hpp OutputForm)
    for (unsigned i = threadIdx.x; i < n; i += L) {
        const float v = Residual ? xw[i] : xr[i];  // this thread's own store of the first loop
        float y = 0.0F;
        if constexpr (form == OutputForm::plain) y = w ? (v * r) * Traits::load(w, wt, i) : v * r;
        else y = finish<form>(v * r, w ? Traits::load(w, wt, i) : 1.0F, w != nullptr);
        yr[i] = y;
        if constexpr (!std::is_same_v<Store, NoStore>) {
            asm volatile("" : "+v"(y));  // y rounded to FP32 before the consumer's conversion (no mixed-precision fold)
            store(base + i, y);
        }
    }
}

// One block per row.
template <class Traits, unsigned L>
__global__ void rmsnorm_kernel(const typename Traits::Input* x, const typename Traits::Scale* w,
                              typename Traits::ScaleFormat wt, unsigned n, float eps,
                              typename Traits::Output* y) {
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    __shared__ float s[L];
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * n;
    rmsnorm_row<Traits, L, false>(s, x + base, nullptr, nullptr, w, wt, n, eps, y + base, base, NoStore{});
}

// The norm of the first n values of rows at a row stride (x row r at x + r * x_stride, y row r at y + r * y_stride; y may
// be x): the plain kernel's row function on the same values, without gathering the rows to a tight buffer. One block per row.
template <class Traits, unsigned L>
__global__ void rmsnorm_strided_kernel(const typename Traits::Input* x, std::size_t x_stride, const typename Traits::Scale* w,
                                       typename Traits::ScaleFormat wt, unsigned n, float eps, typename Traits::Output* y,
                                       std::size_t y_stride) {
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    __shared__ float s[L];
    const std::size_t row = blockIdx.x;
    rmsnorm_row<Traits, L, false>(s, x + row * x_stride, nullptr, nullptr, w, wt, n, eps, y + row * y_stride, row * n, NoStore{});
}

// x += r (Residual), then the norm of x into y and store(row * n + i, y) per element. One block per row.
template <class Traits, unsigned L, bool Residual, class Store>
__global__ void rmsnorm_fused_kernel(float* x, const float* r, const typename Traits::Scale* w,
                                     typename Traits::ScaleFormat wt, unsigned n, float eps, float* y, Store store) {
    static_assert(std::is_same_v<typename Traits::Input, float>);
    static_assert(std::is_same_v<typename Traits::Output, float>);
    __shared__ float s[L];
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * n;
    rmsnorm_row<Traits, L, Residual>(s, x + base, x + base, Residual ? r + base : nullptr, w, wt, n, eps, y + base, base,
                                     store);
}

}  // namespace strix::ops::norm
