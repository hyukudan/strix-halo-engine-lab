#pragma once
// Stock Q8_0 Linear sites of the shared q8_rows direct form (hip_q8_rows_gemv.hpp): FP16 scale x int8 code coefficients,
// FMA accumulation, FP32 input and output. A binding fills a site and launches it; the kernel is the shared one.
// - LinearF32: one input row; optional stored output (round = true): the shared glu::epilogue<bf16> (one rounding of
//   dot + bias) applied in the store, the same device function on the same FP32 value as a separate epilogue launch.
// - LinearRowsF32<R>: the R-row window of LinearF32 (out[r * outputs + o]); R = 4 / 8 run the VOPD coefficient kernels.
// - HeadsF32: H per-head GEMVs in ONE launch (strix/decode_head_rows.hpp head rows), out[(o / head_outputs) * out_stride
//   + o % head_outputs]; e.g. MLA absorb of K_b / V_b.
// Home: shared (HIP kernel sites beside the q8_rows core). Bound by: bindings with stock Q8_0 projections and MLA absorb.
#include "hip_q8_rows_gemv.hpp"
#include "ops/glu/glu.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::q8_rows {

struct StockQ8Arithmetic {
    // fma is the fused single-rounding v_fmac_f32 of the R1 kernel, so 4- and 8-row windows of FP32 inputs take the
    // shared 4 x 4 VOPD coefficient kernels (hip_q8_rows_gemv.hpp launch) like every other fused site: same chains and
    // trees, the bits of LinearF32 on each row.
    static constexpr bool fused_fma = true;
    __device__ static float coefficient(unsigned scale, std::int8_t code) {
        return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(scale))) * static_cast<float>(code);
    }
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(c, x, acc); }
    __device__ static float add(float lower, float upper) { return lower + upper; }
};

struct HeadsF32 : StockQ8Arithmetic {
    using Input = float;
    static constexpr unsigned rows = 1U;
    static constexpr bool head_input = true;
    const void* input[1];
    Segments segments;
    std::uint32_t width, outputs, head_outputs, input_stride, out_stride;
    float* out;
    __device__ void store(std::uint32_t o, unsigned, float value) const {
        out[std::size_t{o / head_outputs} * out_stride + o % head_outputs] = value;
    }
};

struct LinearF32 : StockQ8Arithmetic {
    using Input = float;
    static constexpr unsigned rows = 1U;
    const void* input[1];
    Segments segments;
    std::uint32_t width, outputs;
    float* out;
    const float* bias = nullptr;
    bool round = false;
    __device__ void store(std::uint32_t o, unsigned, float value) const {
        out[o] = round ? ops::glu::epilogue<ops::glu::Round::bf16>(value, bias, bias ? o : 0U) : value;
    }
};

// LinearF32 over R input rows in one launch (verify windows): row r reads input[r] and stores out[r * outputs + o] with
// the same optional epilogue; each (output, row) value has the bits of LinearF32 on that row.
template <unsigned R>
struct LinearRowsF32 : StockQ8Arithmetic {
    using Input = float;
    static constexpr unsigned rows = R;
    const void* input[R];
    Segments segments;
    std::uint32_t width, outputs;
    float* out;
    const float* bias = nullptr;
    bool round = false;
    __device__ void store(std::uint32_t o, unsigned r, float value) const {
        out[std::size_t{r} * outputs + o] = round ? ops::glu::epilogue<ops::glu::Round::bf16>(value, bias, bias ? o : 0U) : value;
    }
};

// H heads of head_outputs rows each, stored consecutively in w (Q8_0, width inputs per row); head h reads x + h * x_stride.
inline void launch_heads_f32(const std::uint8_t* w, std::uint32_t heads, std::uint32_t width, std::uint32_t head_outputs,
                             const float* x, std::uint32_t x_stride, float* y, std::uint32_t y_stride, hipStream_t s) {
    HeadsF32 site{};
    site.input[0] = x;
    site.segments = segments({{w, heads * head_outputs}});
    site.width = width;
    site.outputs = heads * head_outputs;
    site.head_outputs = head_outputs;
    site.input_stride = x_stride;
    site.out_stride = y_stride;
    site.out = y;
    launch_heads(site, s);
}

}  // namespace strix::q8_rows
