#include "core/linear_microscaled_reference.hpp"
#include "core/linear_reference.hpp"
#include "strix/weight/formats/microscaled.hpp"
#include "strix/moe/combine.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace {

template<class Decode>
bool rejects(const Decode& decode) {
    try {
        decode();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

template <strix::weight::microscaled::MatrixFormat Format>
bool packed_tiles() {
    using namespace strix::weight::microscaled;
    constexpr bool nvfp4 = Format == MatrixFormat::nvfp4;
    constexpr std::size_t rows = 6, columns = 64;
    constexpr std::size_t code_bytes = nvfp4 ? columns / 2U : columns;
    constexpr std::size_t scale_bytes = columns / (nvfp4 ? 16U : 32U);
    std::array<std::uint8_t, rows * code_bytes> codes{};
    std::array<std::uint8_t, rows * scale_bytes> scales{};
    for (std::size_t i = 0; i < codes.size(); ++i) {
        codes[i] = static_cast<std::uint8_t>(nvfp4 ? (17U * i + i / code_bytes) % 256U
                                                 : (i * 37U + i / code_bytes) % 120U + (i % 2U) * 128U);
    }
    for (std::size_t i = 0; i < scales.size(); ++i)
        scales[i] = static_cast<std::uint8_t>(nvfp4 ? 38U + i : 121U + i);

    for (const auto order : {RowOrder::contiguous, RowOrder::interleaved_halves}) {
        const PackedOperand<Format> packed{{codes.data(), scales.data(), rows, columns, 13.0F / 31.0F, order}};
        constexpr std::array<std::size_t, rows> interleaved{0, 2, 4, 1, 3, 5};
        std::array<float, columns> input{};
        for (std::size_t column = 0; column < input.size(); ++column)
            input[column] = static_cast<float>(static_cast<int>(column % 13U) - 6) / 9.0F;
        std::array<float, rows> projected{};
        strix::core::microscaled_matvec_reference<Format>(packed.matrix, input, projected);
        for (std::size_t row = 0; row < rows; ++row) {
            const auto physical = order == RowOrder::contiguous ? row : interleaved[row];
            const auto row_codes = std::span(codes).subspan(physical * code_bytes, code_bytes);
            const auto row_scales = std::span(scales).subspan(physical * scale_bytes, scale_bytes);
            std::array<float, columns> expected{};
            if constexpr (nvfp4) decode_nvfp4_row(row_codes, row_scales, 13.0F, 31.0F, expected);
            else decode_mxfp8_row(row_codes, row_scales, expected);
            for (std::size_t k = 0; k < columns; k += 16U) {
                std::uint32_t words[8];
                packed.decode_bf16(packed.load_tile16(row, k), words);
                for (std::size_t col = 0; col < 16; ++col) {
                    const auto want = std::bit_cast<std::uint32_t>(strix::moe::combine_round_bf16(expected[k + col])) >> 16U;
                    const auto got = (words[col / 2U] >> (16U * (col % 2U))) & 0xffffU;
                    if (got != want) return false;
                }
            }
            for (auto& value : expected) value = strix::moe::combine_round_bf16(value);
            if (std::bit_cast<std::uint32_t>(projected[row]) !=
                std::bit_cast<std::uint32_t>(strix::core::dot_f32(expected, input))) return false;
        }
    }
    return true;
}

bool packed_ties() {
    using namespace strix::weight::microscaled;
    using NVFP4 = PackedOperand<MatrixFormat::nvfp4>;
    constexpr std::array<std::uint8_t, 8> codes{0x82U};
    std::array<std::uint8_t, 1> scales{0x38U};
    NVFP4 packed{{codes.data(), scales.data(), 1U, 16U, 1.00390625F, RowOrder::contiguous}};
    std::uint32_t words[8];
    packed.decode_bf16(packed.load_tile16(0, 0), words);
    if (words[0] != 0x80003f80U) return false;  // +1.00390625 is a tie that rounds down, and -0 stays signed.
    packed.matrix.outer_scale = 1.01171875F;
    packed.decode_bf16(packed.load_tile16(0, 0), words);
    if (words[0] != 0x80003f82U) return false;  // Tie rounds up to the next even mantissa.

    scales[0] = 0x80U;  // Invalid negative scale must stay observable as NaN in the device path.
    packed.decode_bf16(packed.load_tile16(0, 0), words);
    constexpr std::array<float, 16> input{};
    std::array<float, 1> output{};
    return (words[0] & 0x7fff7fffU) == 0x7fc07fc0U &&
           rejects([&] { strix::core::microscaled_matvec_reference<MatrixFormat::nvfp4>(packed.matrix, input, output); });
}

} // namespace

int main() {
    using strix::weight::microscaled::decode_mxfp8_row;
    using strix::weight::microscaled::decode_nvfp4_row;

    std::array<std::uint8_t, 16> fp4{};
    fp4[0] = 0x9fU;  // -6, -0.5
    fp4[1] = 0x81U;  // +0.5, -0
    fp4[7] = 0x21U;  // +0.5, +1, last packed byte of the first block
    fp4[8] = 0xbaU;  // -1, -1.5, first packed byte of the next block
    const std::array<std::uint8_t, 2> fp4_scales{0x30U, 0x40U};  // +0.5, +2
    std::array<float, 32> fp4_output{};
    decode_nvfp4_row(fp4, fp4_scales, 3.0F, 2.0F, fp4_output);
    if (fp4_output[0] != -4.5F || fp4_output[1] != -0.375F || fp4_output[2] != 0.375F ||
        fp4_output[3] != 0.0F || !std::signbit(fp4_output[3]) ||
        fp4_output[14] != 0.375F || fp4_output[15] != 0.75F ||
        fp4_output[16] != -3.0F || fp4_output[17] != -4.5F) return 1;
    decode_nvfp4_row(fp4, fp4_scales, 6.0F, 2.0F, fp4_output);
    if (fp4_output[0] != -9.0F || fp4_output[16] != -6.0F) return 2;
    decode_nvfp4_row(fp4, fp4_scales, 0.0F, 2.0F, fp4_output);
    if (fp4_output[0] != 0.0F || fp4_output[16] != 0.0F) return 3;
    const std::array<std::uint8_t, 2> invalid_fp4_scales{0x80U, 0x7fU};
    if (!rejects([&] { decode_nvfp4_row(fp4, fp4_scales, 3.0F, 0.0F, fp4_output); }) ||
        !rejects([&] { decode_nvfp4_row(fp4, invalid_fp4_scales, 3.0F, 2.0F, fp4_output); }) ||
        !rejects([&] { decode_nvfp4_row(std::span{fp4}.first(15), fp4_scales, 3.0F, 2.0F, fp4_output); })) return 4;

    std::array<std::uint8_t, 64> fp8{};
    fp8[0] = 0xb8U;   // -1
    fp8[31] = 0x01U;  // +2^-9, last element of the first block
    fp8[32] = 0x3cU;  // +1.5, first element of the next block
    fp8[33] = 0x80U;  // -0
    std::array<std::uint8_t, 2> fp8_scales{126U, 128U};  // +0.5, +2
    std::array<float, 64> fp8_output{};
    decode_mxfp8_row(fp8, fp8_scales, fp8_output);
    if (fp8_output[0] != -0.5F || fp8_output[31] != 0x1.0p-10F || fp8_output[32] != 3.0F ||
        fp8_output[33] != 0.0F || !std::signbit(fp8_output[33])) return 5;
    fp8_scales[0] = 0U;
    decode_mxfp8_row(fp8, fp8_scales, fp8_output);
    if (fp8_output[0] != -0x1.0p-127F) return 6;
    fp8[0] = 0x7fU;
    if (!rejects([&] { decode_mxfp8_row(fp8, fp8_scales, fp8_output); })) return 7;
    fp8[0] = 0x7eU;
    fp8_scales[0] = 254U;
    if (!rejects([&] { decode_mxfp8_row(fp8, fp8_scales, fp8_output); })) return 8;
    fp8_scales[0] = std::numeric_limits<std::uint8_t>::max();
    if (!rejects([&] { decode_mxfp8_row(fp8, fp8_scales, fp8_output); }) ||
        !rejects([&] { decode_mxfp8_row(fp8, std::span{fp8_scales}.first(1), fp8_output); })) return 9;
    using strix::weight::microscaled::MatrixFormat;
    if (!packed_tiles<MatrixFormat::nvfp4>()) return 10;
    if (!packed_tiles<MatrixFormat::mxfp8>()) return 11;
    if (!packed_ties()) return 12;
    return 0;
}
