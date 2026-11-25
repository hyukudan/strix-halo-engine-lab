#include "strix/weight/formats/microscaled.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

int main() {
    using strix::weight::microscaled::decode_mxfp4_row;
    std::array<std::uint8_t, 32> codes{};
    constexpr std::array<std::uint8_t, 8> values{0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe};
    std::ranges::copy(values, codes.begin());
    codes[16] = 0xf3;
    const std::array<std::uint8_t, 2> scales{127, 128};
    std::array<float, 64> output{};
    decode_mxfp4_row(codes, scales, output);
    constexpr std::array expected{0.F, .5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F,
                                 -0.F, -.5F, -1.F, -1.5F, -2.F, -3.F, -4.F, -6.F};
    if (!std::equal(expected.begin(), expected.end(), output.begin()) ||
        !std::signbit(output[8]) || output[32] != 3.F || output[33] != -12.F ||
        output[31] != 0.F || output[63] != 0.F) {
        std::cerr << "incorrect MXFP4 values or column/block order\n";
        return 1;
    }
    auto invalid_scale = scales;
    invalid_scale.back() = 255;
    bool refused_scale = false;
    try { decode_mxfp4_row(codes, invalid_scale, output); }
    catch (const std::invalid_argument&) { refused_scale = true; }
    if (!refused_scale) {
        std::cerr << "MXFP4 accepted a NaN scale\n";
        return 1;
    }
    try { decode_mxfp4_row(std::span(codes).first(31), scales, output); }
    catch (const std::invalid_argument&) { return 0; }
    std::cerr << "MXFP4 accepted a truncated code row\n";
    return 1;
}
