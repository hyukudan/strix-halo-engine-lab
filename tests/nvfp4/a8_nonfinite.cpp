// Host rejection contract; device parity remains blocked on the converter scan fix.
#include "strix/cuda/core/nvfp4_a8.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

int main() {
    namespace a8 = strix::cuda::core::nvfp4_a8;
    const std::array<std::uint16_t, 5> scales{0x7e00, 0xfe00, 0x7c01, 0x7c00, 0xfc00};
    for (const auto bits : scales) {
        for (const std::uint8_t code : {std::uint8_t{0}, std::uint8_t{1}}) {
            std::array<std::uint8_t, 34> q8{};
            q8.fill(code);
            std::memcpy(q8.data(), &bits, sizeof(bits));
            std::array<std::uint8_t, 16> codes{};
            std::array<std::uint8_t, 2> groups{};
            float tensor_scale = 0;
            try {
                a8::reference_convert_q8_0(q8.data(), 1, 32, codes.data(), groups.data(), &tensor_scale);
            } catch (const std::invalid_argument&) {
                continue;
            }
            std::fputs("nonfinite Q8_0 scale was accepted\n", stderr);
            return 1;
        }
    }
    return 0;
}
