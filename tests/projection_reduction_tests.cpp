#include "strix/projection_reduction.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }
template <typename Call> void rejected(Call call) {
    try { call(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid arithmetic rejection");
}
} // namespace

int main() try {
    std::array<float, 256> values{};
    // Sensitivity control: a different reduction tree loses the unit term.
    values[0] = 0x1p24F;
    values[1] = -0x1p24F;
    values[128] = 1.0F;
    expect(strix::projection_reduce_lds(values) == 0.0F, "fixed-tree reference changed");
    float sequential = 0;
    for (float value : values) sequential += value;
    expect(sequential == 1.0F, "reduction sensitivity fixture no longer sensitive");
    std::uint32_t random = 0x31415926U;
    for (unsigned trial = 0; trial < 1024; ++trial) {
        for (float& value : values) {
            random = random * 1664525U + 1013904223U;
            // Finite mixed magnitudes/signs, including subnormal and zero cases.
            const auto sign = random & 0x80000000U;
            const auto exponent = ((random >> 24U) % 180U) << 23U;
            value = std::bit_cast<float>(sign | exponent | (random & 0x007fffffU));
        }
        expect(bits(strix::projection_reduce_lds(values)) ==
               bits(strix::projection_reduce_shuffle_tail(values)), "shuffle changed arithmetic tree");
    }
    values.fill(-0.0F);
    expect(bits(strix::projection_reduce_shuffle_tail(values)) == 0x80000000U,
           "negative-zero tree changed");
    values.fill(0.0F);
    values[0] = std::numeric_limits<float>::denorm_min();
    expect(bits(strix::projection_reduce_shuffle_tail(values)) == 1U, "subnormal lost");
    values[1] = std::numeric_limits<float>::quiet_NaN();
    rejected([&] { static_cast<void>(strix::projection_reduce_lds(values)); });
    std::vector<float> w(768, 0.0F), x(768, 0.0F);
    // At lane zero: exact FMA retains a residual that separate multiply/add drops.
    w[0] = -1.0F; x[0] = 1.0F;
    w[256] = 1.0F + 0x1p-23F; x[256] = 1.0F - 0x1p-23F;
    expect(strix::projection_dot_reference(w, x) == -0x1p-46F, "FMA fixture changed");
    volatile float product = w[256] * x[256];
    expect(product - 1.0F == 0.0F, "FMA negative control insensitive");
    rejected([&] { static_cast<void>(strix::projection_dot_reference({}, {})); });
    x[1] = std::numeric_limits<float>::infinity();
    rejected([&] { static_cast<void>(strix::projection_dot_reference(w, x)); });
    std::cout << "projection reduction references passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
