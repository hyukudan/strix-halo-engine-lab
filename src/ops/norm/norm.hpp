#pragma once

#include "strix/ds41_layer.hpp"  // round_to_bf16: the declared BF16 RNE shared by the declared-arithmetic models

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_OPS_NORM_HD __host__ __device__
#else
#define STRIX_OPS_NORM_HD
#endif

namespace strix::ops::norm {

// lane_strided_fma_lds_halving: lane l of L sums values l, l + L, ... then the LDS halving tree (row_reduction.hpp);
// wave_chunk_serial: per-wave chunks, xor butterfly per chunk, chunk sums added serially (norm_numeric_traits.hpp).
// The square accumulation of either order is SquareForm (norm_numeric_traits.hpp; default fma, the name's "fma").
enum class ReductionOrder { lane_strided_fma_lds_halving, wave_chunk_serial };

// Output form, an optional trait (`static constexpr OutputForm output_form`; absent = plain):
//   plain:       y = (x*r)*scale, unit scale y = x*r (the existing binders: bits unchanged);
//   bf16_once:   y = R((x*r)*scale), unit scale y = R(x*r);
//   bf16_double: y = R(scale * R(x*r)), unit scale y = R(x*r), R = BF16 RNE: the BF16 reference module's norm
//                (normalise in BF16, then the BF16 weight product). Binders: models whose reference norm rounds through BF16.
enum class OutputForm { plain, bf16_double, bf16_once };

template <class Traits>
STRIX_OPS_NORM_HD constexpr OutputForm output_form_of() {
    if constexpr (requires { Traits::output_form; }) return Traits::output_form;
    else return OutputForm::plain;
}

// One output element: xr = x * r; scale = the decoded weight (has_scale false: unit scale).
template <OutputForm F>
STRIX_OPS_NORM_HD inline float finish(float xr, float scale, bool has_scale) {
    if constexpr (F == OutputForm::plain) {
        return has_scale ? xr * scale : xr;
    } else if constexpr (F == OutputForm::bf16_once) {
        return ds41::round_to_bf16(has_scale ? xr * scale : xr);
    } else {
        const float n = ds41::round_to_bf16(xr);
        return has_scale ? ds41::round_to_bf16(scale * n) : n;
    }
}

// Contiguous [rows][n] RMSNorm, shared scale[n] (or null for unit scale),
// no persistent state. Input/output may alias. The engine owns launch/error
// handling, scale decoding and the declared reciprocal-square-root operation.
// Epsilon is added AFTER FP32 sumsq/n; output is (x*r)*scale, never x*(r*scale).
// The reduction is Traits::reduction_order (ReductionOrder above); each order has its own row body.
template <class Traits>
struct Norm {
    using Input = typename Traits::Input;
    using Output = typename Traits::Output;
    using Scale = typename Traits::Scale;
    using ScaleFormat = typename Traits::ScaleFormat;
    using Stream = typename Traits::Stream;
    static constexpr float epsilon = Traits::epsilon;
    static constexpr ReductionOrder reduction_order = Traits::reduction_order;

    static constexpr unsigned lanes(unsigned n) { return Traits::lanes(n); }

    // Explicit epsilon retains callers' existing arithmetic identity. Geometry,
    // pointer validity and stream lifetime remain the engine's responsibility.
    static void run(const Input* x, const Scale* w, ScaleFormat wt, unsigned n,
                    float eps, Output* y, unsigned rows, Stream stream) {
        Traits::launch(x, w, wt, n, eps, y, rows, stream);
    }
};

}  // namespace strix::ops::norm
