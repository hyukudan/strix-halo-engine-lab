#pragma once
// Norm device presets over the shared row body (norm_device.hpp): the LDS halving reduction of
// ReductionOrder::lane_strided_fma_lds_halving as one function, and a Traits preset for FP32 rows with the declared square
// root. A binding with this arithmetic uses the preset instead of a private Traits copy.
// Home: shared (HIP norm device home). Bound by: RMSNorm sites of the bf16_double form; ChunkedF32Traits: the
// wave-chunk norms (bindings follow in their model changes).
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

// Device Traits preset of ReductionOrder::wave_chunk_serial (norm_numeric_traits.hpp) for FP32 rows and FP32 weights:
// Chunk values per wave chunk (0 = the whole row on one wave), squares in form Q, r = 1 / declared_sqrt(ss / n + eps)
// (Rsqrt = DeclaredRsqrt) or the correctly rounded u^(-1/2) with the non-finite guard (Rsqrt = CorrectlyRoundedRsqrt),
// output plain (x * r) * w. The DS41 chunked norms are <640, mul_add, DeclaredRsqrt>; the one-wave correctly rounded norms
// are <0, mul_add, CorrectlyRoundedRsqrt>.
struct DeclaredRsqrt {
    __device__ __forceinline__ static float inv_sqrt(float x) { return 1.0F / ds41::declared_sqrt(x); }
};
struct CorrectlyRoundedRsqrt {
    __device__ __forceinline__ static float inv_sqrt(float x) { return inv_sqrt_correctly_rounded_finite(x); }
};

template <unsigned Chunk, SquareForm Q, class Rsqrt>
struct ChunkedF32Traits {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;
    static constexpr auto reduction_order = ReductionOrder::wave_chunk_serial;
    static constexpr unsigned wave_chunk = Chunk;
    static constexpr auto square_form = Q;
    __device__ __forceinline__ static float inv_sqrt(float x) { return Rsqrt::inv_sqrt(x); }
    __device__ __forceinline__ static float load(const float* w, int, unsigned i) { return w[i]; }
};

}  // namespace strix::ops::norm
