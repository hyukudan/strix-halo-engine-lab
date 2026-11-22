#pragma once
// Norm device presets over the shared row body (norm_device.hpp): the LDS halving reduction of
// ReductionOrder::lane_strided_fma_lds_halving as one function, and a Traits preset for FP32 rows with the declared square
// root. A binding with this arithmetic uses the preset instead of a private Traits copy.
// Home: shared (HIP norm device home). Bound by: RMSNorm sites of the bf16_double form.
#include "norm_device.hpp"
#include "row_reduction.hpp"
#include "strix/numeric/declared.hpp"  // declared_sqrt: the declared square root of the declared-arithmetic models

#include <hip/hip_runtime.h>

namespace strix::ops::norm {

// Device Traits preset: FP32 rows and FP32 weights, the reduction above, r = 1 / declared_sqrt(ss / n + eps), output
// form F (norm.hpp OutputForm). A binding whose norm is this arithmetic uses the preset instead of its own Traits.
template <OutputForm F>
struct DeclaredF32Traits {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;
    static constexpr auto reduction_order = ReductionOrder::lane_strided_fma_lds_halving;
    static constexpr auto output_form = F;
    template <unsigned L>
    __device__ __forceinline__ static float reduce(float* s, float part) { return lds_halving_sum<L>(s, part); }
    __device__ __forceinline__ static float inv_sqrt(float x) { return 1.0F / ::strix::numeric::declared_sqrt(x); }
    __device__ __forceinline__ static float load(const float* w, int, unsigned i) { return w[i]; }
};

}  // namespace strix::ops::norm
