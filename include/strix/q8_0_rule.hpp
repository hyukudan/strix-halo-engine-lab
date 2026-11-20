#pragma once
// ggml's quantize_row_q8_0_ref on one group of 32 values, shared by host code and HIP / CUDA kernels that build Q8_0
// rows at load (draft heads, transcodes): d = amax / 127 (FP32, round to nearest), id = d != 0 ? 1 / d : 0,
// q[j] = roundf(v[j] * id) (ties away from zero). The block stores d as FP16 (the caller's conversion: its value is the
// one every Q8_0 reader multiplies by) followed by the 32 int8 codes. Finite inputs.
#include <cstdint>
#include <math.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_Q8R_HD __host__ __device__
#else
#define STRIX_Q8R_HD
#endif

namespace strix::q8_0 {

inline constexpr unsigned group_values = 32;
inline constexpr unsigned block_bytes = 34;  // FP16 d + 32 int8

struct Group {
    float d = 0.0F;  // FP32 scale (store as FP16)
    std::int8_t q[group_values]{};
};

STRIX_Q8R_HD inline Group quantize_group(const float* v) {
    float amax = 0.0F;
    for (unsigned j = 0; j < group_values; ++j) amax = fmaxf(amax, fabsf(v[j]));
    Group g;
    g.d = amax / 127.0F;
    const float id = g.d != 0.0F ? 1.0F / g.d : 0.0F;
    for (unsigned j = 0; j < group_values; ++j) g.q[j] = static_cast<std::int8_t>(static_cast<int>(roundf(v[j] * id)));
    return g;
}

// BF16 bits -> FP32 (exact).
STRIX_Q8R_HD inline float bf16_value(std::uint16_t bits) {
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(static_cast<std::uint32_t>(bits) << 16U));
}

}  // namespace strix::q8_0

#undef STRIX_Q8R_HD
