#include "strix/weight/formats/ieee_half.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>

int main() {
    // Construct the IEEE value arithmetically, independently of the decoder's
    // exponent-bit assembly and CUDA conversion instruction.
    for (unsigned h = 0; h < 65536U; ++h) {
        const auto exponent = (h >> 10U) & 31U;
        const auto mantissa = h & 1023U;
        const auto sign = (h & 32768U) << 16U;
        float reference;
        if (exponent == 31U) {
            reference = std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13U));
        } else {
            const auto magnitude = exponent == 0U ? std::ldexp(static_cast<float>(mantissa), -24)
                : std::ldexp(static_cast<float>(1024U + mantissa), static_cast<int>(exponent) - 25);
            reference = std::copysign(magnitude, h & 32768U ? -1.0F : 1.0F);
        }
        const auto actual = strix::weight::ieee_half::decode(static_cast<std::uint16_t>(h));
        if (std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(reference)) {
            std::cerr << "IEEE half mismatch at bit pattern " << h << '\n';
            return 1;
        }
        using strix::weight::ieee_half::NanMode;
        const auto quiet = strix::weight::ieee_half::decode(static_cast<std::uint16_t>(h), NanMode::quiet);
        const auto quiet_bits = std::bit_cast<std::uint32_t>(reference) |
            (exponent == 31U && mantissa != 0U ? 0x00400000U : 0U);
        if (std::bit_cast<std::uint32_t>(quiet) != quiet_bits) {
            std::cerr << "IEEE half quiet policy mismatch at bit pattern " << h << '\n';
            return 1;
        }
    }
    std::cout << "IEEE half decoder: all 65536 patterns match the independent oracle\n";
}
