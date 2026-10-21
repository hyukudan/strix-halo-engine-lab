#pragma once

// Low-bit weight and cache formats of the deepseek41 model files and the declared arithmetic of
// their decode products. Host side: format tables, reference decoders, the A8 activation contract
// and the reference of every device product, evaluated in the device's exact operation order so
// that host and device results are bitwise comparable.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix::ds41 {

// GGUF block geometry: IQ2_XXS (type 16), Q2_K (type 10), MXFP4 (type 39), Q8_0 (type 8, shared expert).
inline constexpr std::size_t super_block_elements = 256;
inline constexpr std::size_t iq2_xxs_block_bytes = 66;
inline constexpr std::size_t q2_k_block_bytes = 84;
inline constexpr std::size_t mxfp4_block_elements = 32;
inline constexpr std::size_t mxfp4_block_bytes = 17;
inline constexpr std::size_t q8_0_block_elements = 32;
inline constexpr std::size_t q8_0_block_bytes = 34;
// Engram row container: 256 E4M3 codes then 8 E8M0 scales, one per 32 channels.
inline constexpr std::size_t engram_row_elements = 256;
inline constexpr std::size_t engram_row_bytes = 264;
// A8 activation groups and the lanes of one product wave.
inline constexpr std::size_t a8_group = 32;
inline constexpr std::size_t product_lanes = 32;

// CB4 (device kernels only, requantization candidates): 18-byte blocks {f16 d, 16 bytes of 4-bit indices, byte j =
// weight j low nibble, weight j + 16 high nibble} into a 16-entry int8 table given with the launch; a block is the
// Q8_0 block {d, table[index]}, so its products are those of that Q8_0 block.
inline constexpr std::size_t cb4_block_elements = 32;
inline constexpr std::size_t cb4_block_bytes = 18;
// U5 / U6 (device kernels only): 22 / 26-byte blocks {f16 d, 16 bytes of low nibbles as CB4, then the high bits:
// U5 one 32-bit word, bit j = bit 4 of weight j; U6 one 64-bit word, bits 2j..2j+1 = bits 4..5 of weight j}; the
// 5 / 6-bit value v gives the Q8_0 code v - 16 / v - 32 (the uniform codebooks of the requantization tool).
inline constexpr std::size_t u5_block_bytes = 22;
inline constexpr std::size_t u6_block_bytes = 26;

enum class WeightFormat : std::uint32_t { iq2_xxs = 0, q2_k = 1, mxfp4 = 2, q8_0 = 3, cb4 = 4, u5 = 5, u6 = 6 };
[[nodiscard]] constexpr std::size_t block_elements(WeightFormat format) noexcept {
    return format == WeightFormat::mxfp4 ? mxfp4_block_elements
        : format == WeightFormat::q8_0   ? q8_0_block_elements
        : format == WeightFormat::cb4 || format == WeightFormat::u5 || format == WeightFormat::u6 ? cb4_block_elements
                                         : super_block_elements;
}
[[nodiscard]] constexpr std::size_t block_bytes(WeightFormat format) noexcept {
    return format == WeightFormat::iq2_xxs ? iq2_xxs_block_bytes
        : format == WeightFormat::q2_k     ? q2_k_block_bytes
        : format == WeightFormat::mxfp4    ? mxfp4_block_bytes
        : format == WeightFormat::cb4      ? cb4_block_bytes
        : format == WeightFormat::u5       ? u5_block_bytes
        : format == WeightFormat::u6       ? u6_block_bytes
                                           : q8_0_block_bytes;
}

// IQ2_XXS grid: 256 entries of eight magnitudes in {8, 25, 43}, one byte each (byte j = element j).
extern const std::array<std::uint64_t, 256> iq2_xxs_grid;
// Sign byte of a 7-bit IQ2_XXS sign index: bit j set = element j negative; bit 7 is the parity of
// the index, so every byte has an even number of negative elements.
[[nodiscard]] constexpr std::uint8_t iq2_xxs_sign_byte(std::uint32_t index) noexcept {
    index &= 127U;
    std::uint32_t parity = index;
    parity ^= parity >> 4U; parity ^= parity >> 2U; parity ^= parity >> 1U;
    return static_cast<std::uint8_t>(index | ((parity & 1U) << 7U));
}
// E2M1 values times two (the MXFP4 integer form): codes 0-7 positive, 8-15 negative.
inline constexpr std::array<std::int8_t, 16> e2m1_doubled{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Scalar formats. E8M0: 2^(e - 127), with e = 0 giving 2^-127 and e = 255 giving NaN.
[[nodiscard]] float e8m0_to_float(std::uint8_t code) noexcept;
// Half of E8M0 (the scale of doubled E2M1 values): 2^(e - 128); e = 255 gives NaN.
[[nodiscard]] float e8m0_half_to_float(std::uint8_t code) noexcept;
// E4M3 (finite-only variant): bias 7, subnormals, S.1111.111 = NaN, no infinities.
[[nodiscard]] float e4m3_to_float(std::uint8_t code) noexcept;
// Round to nearest even with saturation to +-448; NaN maps to 0x7f.
[[nodiscard]] std::uint8_t float_to_e4m3(float value) noexcept;
[[nodiscard]] float e2m1_to_float(std::uint8_t nibble) noexcept;
// Every E4M3 code as FP32 (exact).
[[nodiscard]] const std::array<float, 256>& e4m3_table() noexcept;

// Reference decoders (FP32 values in element order).
void decode_iq2_xxs(std::span<const std::uint8_t, iq2_xxs_block_bytes> block, std::span<float, super_block_elements> out);
void decode_q2_k(std::span<const std::uint8_t, q2_k_block_bytes> block, std::span<float, super_block_elements> out);
void decode_mxfp4(std::span<const std::uint8_t, mxfp4_block_bytes> block, std::span<float, mxfp4_block_elements> out);
void decode_q8_0(std::span<const std::uint8_t, q8_0_block_bytes> block, std::span<float, q8_0_block_elements> out);
void decode_engram_row(std::span<const std::uint8_t, engram_row_bytes> row, std::span<float, engram_row_elements> out);

// Integer forms of one 32-weight group. IQ2_XXS: value = scale * code, codes in {+-8, +-25, +-43}.
struct Iq2XxsGroup { std::array<std::int8_t, 32> codes{}; float scale = 0.0F; };
[[nodiscard]] Iq2XxsGroup iq2_xxs_group(std::span<const std::uint8_t, iq2_xxs_block_bytes> block, unsigned group);
// MXFP4: value = scale * code with doubled E2M1 codes and scale = 2^(e - 128).
struct Mxfp4Group { std::array<std::int8_t, 32> codes{}; float scale = 0.0F; };
[[nodiscard]] Mxfp4Group mxfp4_group(std::span<const std::uint8_t, mxfp4_block_bytes> block);
// Q8_0: value = scale * code with int8 codes and the block's FP16 scale.
struct Q8Group { std::array<std::int8_t, 32> codes{}; float scale = 0.0F; };
[[nodiscard]] Q8Group q8_0_group(std::span<const std::uint8_t, q8_0_block_bytes> block);
// Q2_K: value = d * (sc & 15) * q - dmin * (sc >> 4), two 16-element halves per 32-element group.
struct Q2kGroup { std::array<std::uint8_t, 32> codes{}; std::uint8_t scale[2]{}; float d = 0.0F, dmin = 0.0F; };
[[nodiscard]] Q2kGroup q2_k_group(std::span<const std::uint8_t, q2_k_block_bytes> block, unsigned group);

// A8 activations: groups of 32, delta = amax / 127 (correctly rounded), code = rint(x / delta)
// clamped to +-127; an all-zero group, or one with amax below 2^-119, has delta 1 and codes 0.
// sums[g] is the sum of the group's codes.
struct A8Activation {
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
    std::vector<std::int32_t> sums;
};
[[nodiscard]] A8Activation quantize_a8(std::span<const float> values);

// Declared product of one weight row with one A8 activation row (width = row elements, a multiple
// of 256 for IQ2_XXS/Q2_K and of 32 for MXFP4). Group g (32 weights) contributes
//   IQ2_XXS: float(sum codes*x) * (scale * delta_g)
//   MXFP4:   float(sum codes*x) * (scale * delta_g)
//   Q2_K:    (d * float(A) - dmin * float(B)) * delta_g, A = sum_h (sc_h & 15) * sum q*x,
//            B = sum_h (sc_h >> 4) * sum x over the group's two halves h
// lane l accumulates groups l, l + 32, ... in ascending order from 0, and the lanes are combined
// by the butterfly v[l] += v[l ^ d] for d = 16, 8, 4, 2, 1; the result is lane 0.
[[nodiscard]] float product_iq2_xxs(std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width);
[[nodiscard]] float product_q2_k(std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width);
[[nodiscard]] float product_mxfp4(std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width);
// Q8_0 (shared expert): as MXFP4, with the block scale.
[[nodiscard]] float product_q8_0(std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width);

} // namespace strix::ds41
