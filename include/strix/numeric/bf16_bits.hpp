#pragma once

// BF16 storage codec of the activation path, one definition for host, HIP and CUDA: widening is the
// 16-bit shift, narrowing is round to nearest even on the FP32 bits without NaN canonicalisation (the
// incumbent activation store of the HIP launchers and of the MoE slot merge). A path that must quiet
// NaN binds weight::scalar_float::bf16_encode or numeric::round_to_bf16 instead.

#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_BF16_BITS_HD __host__ __device__
#else
#define STRIX_BF16_BITS_HD
#endif

namespace strix::numeric::bf16_bits {

// Round to nearest even of the FP32 bits `bits`, as BF16 bits.
STRIX_BF16_BITS_HD constexpr std::uint16_t rne(std::uint32_t bits) noexcept {
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

STRIX_BF16_BITS_HD inline float bf16_to_float(std::uint16_t bits) noexcept {
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(bits) << 16U);
}

STRIX_BF16_BITS_HD inline std::uint16_t float_to_bf16(float value) noexcept {
    return rne(__builtin_bit_cast(std::uint32_t, value));
}

// The FP32 value of float_to_bf16(value).
STRIX_BF16_BITS_HD inline float round_bf16(float value) noexcept { return bf16_to_float(float_to_bf16(value)); }

}  // namespace strix::numeric::bf16_bits
