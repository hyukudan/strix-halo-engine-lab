#include "strix/weight/formats/microscaled.hpp"

#include <array>
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
    return 0;
}
