// PackedOperand<nvfp4>::decode_bf16_shift is bitwise decode_bf16: every block-scale code x every code byte (in every byte
// position) for special outer scales (zeros, signs, infinities, NaN, subnormal and overflow boundaries, the guard edges),
// then random outer scales over the whole FP32 range and over the model range, with random code words (one and two words).
#include "strix/weight/formats/microscaled.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

namespace packed = strix::weight::microscaled;
using Operand = packed::PackedOperand<packed::MatrixFormat::nvfp4>;

float bits_float(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

struct Counts {
    std::uint64_t checked = 0, shifted = 0, failures = 0;
};

template <std::size_t Words>
void check(const Operand& operand, const std::uint32_t (&codes)[Words], std::uint8_t scale, Counts& counts) {
    std::uint8_t bytes[4U * Words];
    for (std::size_t i = 0; i < 4U * Words; ++i) bytes[i] = static_cast<std::uint8_t>(codes[i / 4U] >> (8U * (i % 4U)));
    std::uint32_t expected[4U * Words], actual[4U * Words];
    operand.decode_bf16(bytes, scale, expected);
    operand.decode_bf16_shift(codes, scale, actual);
    ++counts.checked;
    std::uint32_t low[2]{}, high[2]{};
    if (operand.shift_table(scale, low, high)) ++counts.shifted;
    if (std::memcmp(expected, actual, sizeof expected) != 0) {
        if (++counts.failures <= 5U) {
            float outer = operand.matrix.outer_scale;
            std::uint32_t outer_bits;
            std::memcpy(&outer_bits, &outer, sizeof outer_bits);
            std::cerr << "decode_bf16_shift MISMATCH outer 0x" << std::hex << outer_bits << " scale 0x"
                      << unsigned{scale} << " codes 0x" << codes[0] << " expected 0x" << expected[0] << " actual 0x"
                      << actual[0] << std::dec << "\n";
        }
    }
}

Operand operand_with(float outer) {
    Operand operand{};
    operand.matrix.outer_scale = outer;
    return operand;
}

}  // namespace

int main() {
    Counts counts;
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> special = {1.0F, 0.5F, 2.0F, 1.0F / 448.0F, 1.0F / 6.0F, 1.0F / 2688.0F, 0.0F, -0.0F, -1.0F,
                                  inf, -inf, std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::denorm_min(), 1e-30F, 1e30F, 1e-40F};
    // Guard edges: products whose exponent field sits at 1..3 and 250..253 for the extreme block scales.
    for (std::uint32_t e : {1U, 2U, 3U, 4U, 10U, 11U, 12U, 13U, 14U, 240U, 241U, 242U, 243U, 248U, 249U, 250U, 251U,
                            252U, 253U, 254U})
        for (std::uint32_t mantissa : {0U, 1U, 0x3fffffU, 0x400000U, 0x7fffffU, 0x7f8000U, 0x7f7fffU, 0x0080U})
            special.push_back(bits_float((e << 23U) | mantissa));

    // Exhaustive over block-scale codes and code bytes (all four byte positions at once) for the special scales.
    for (const float outer : special) {
        const auto operand = operand_with(outer);
        for (unsigned scale = 0; scale < 256U; ++scale)
            for (std::uint32_t b = 0; b < 256U; ++b) {
                const std::uint32_t one[1] = {b * 0x01010101U};
                check(operand, one, static_cast<std::uint8_t>(scale), counts);
            }
    }

    // Random outer scales: full FP32 bit range, then the checkpoint range, each with every block-scale code.
    std::mt19937 random(0x5eed1234U);
    for (unsigned trial = 0; trial < 24000U; ++trial) {
        const float outer = trial % 2U == 0U
                                ? bits_float(static_cast<std::uint32_t>(random()))
                                : bits_float(0x30000000U + static_cast<std::uint32_t>(random() % 0x10000000U));
        const auto operand = operand_with(outer);
        for (unsigned scale = 0; scale < 256U; ++scale) {
            for (unsigned word = 0; word < 6U; ++word) {
                const std::uint32_t one[1] = {static_cast<std::uint32_t>(random())};
                check(operand, one, static_cast<std::uint8_t>(scale), counts);
            }
            const std::uint32_t two[2] = {static_cast<std::uint32_t>(random()), static_cast<std::uint32_t>(random())};
            check(operand, two, static_cast<std::uint8_t>(scale), counts);
        }
    }

    // The checkpoint outer-scale range must use the shift form for every finite nonzero block scale code.
    {
        const auto operand = operand_with(1.0F / 2688.0F);
        std::uint32_t low[2], high[2];
        for (unsigned scale = 1; scale <= 0x7eU; ++scale)
            if (!operand.shift_table(static_cast<std::uint8_t>(scale), low, high)) {
                std::cerr << "decode_bf16_shift: model-range scale 0x" << std::hex << scale << std::dec
                          << " left the shift domain\n";
                ++counts.failures;
            }
    }

    if (counts.failures != 0U || counts.shifted == 0U) {
        std::cerr << "microscaled shift decode: FAIL " << counts.failures << " of " << counts.checked << "\n";
        return 1;
    }
    std::cout << "microscaled shift decode bitwise: PASS " << counts.checked << " tiles (" << counts.shifted
              << " in the shift domain)\n";
    return 0;
}
