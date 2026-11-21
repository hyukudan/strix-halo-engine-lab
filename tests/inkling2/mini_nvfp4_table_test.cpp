#include "mini_nvfp4_table_cases.hpp"

int main() {
    using namespace mini_nvfp4_table_cases;
    for (const auto bits : outer_bits) {
        for (unsigned scale = 0; scale < scale_count; ++scale) {
            const auto result = check_block(strix::weight::scalar_float::u2f(bits), static_cast<std::uint8_t>(scale));
            if (!report(result, bits, scale)) return 1;
        }
    }
    return 0;
}
