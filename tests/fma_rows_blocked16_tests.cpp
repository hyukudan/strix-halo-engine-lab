#include "core/fma_rows.hpp"

#include <array>
#include <cstdio>
#include <limits>

namespace {
using namespace strix::core::fma_rows;
using Math = Reduction<Reduce::blocked16_tree>;
using Input = std::array<float, 4096>;
using Weights = std::array<std::uint16_t, 4096>;

std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }

float reference_widen(std::uint16_t h) {
    const unsigned exponent = (h / 1024U) % 32U, significand = h % 1024U;
    if (exponent == 31)
        return std::bit_cast<float>((static_cast<std::uint32_t>(h & 0x8000U) << 16U) |
                                    0x7f800000U | (significand << 13U));
    const float magnitude = std::ldexp(static_cast<float>(significand + (exponent == 0 ? 0U : 1024U)),
                                       exponent == 0 ? -24 : static_cast<int>(exponent) - 25);
    return (h & 0x8000U) != 0 ? -magnitude : magnitude;
}

float reference(const Input& x, const Weights& w) {
    float p[16] = {};
    for (unsigned j = 0; j < 16; ++j)
        for (unsigned k = j * 256; k < (j + 1) * 256; ++k)
            p[j] = std::fma(x[k], reference_widen(w[k]), p[j]);
    for (unsigned h : {8U, 4U, 2U, 1U})
        for (unsigned j = 0; j < h; ++j) p[j] = p[j] + p[j + h];
    return p[0];
}

bool check(const char* name, const Input& x, const Weights& w) {
    float partials[Math::blocks] = {};
    float lanes[Math::blocks] = {};
    for (unsigned j = 0; j < Math::blocks; ++j) {
        for (unsigned c = 0; c < Math::inputs_per_block; c += 8)
            for (unsigned k = 0; k < 8; ++k) {
                const unsigned at = j * Math::inputs_per_block + c + k;
                partials[j] = Math::step(x[at], Math::widen(w[at]), partials[j]);
            }
        lanes[j] = partials[j];
    }
    Math::levels([&](unsigned h) {
        const auto snapshot = std::to_array(lanes);
        for (unsigned j = 0; j < h; ++j) lanes[j] += snapshot[j + h];
    });
    const auto expected = bits(reference(x, w));
    if (bits(Math::tree(partials)) == expected && bits(lanes[0]) == expected) return true;
    std::fprintf(stderr, "blocked16 %s differs from the scalar reference\n", name);
    return false;
}
}  // namespace

int main() {
    static_assert(Math::blocks == 16 && Math::inputs_per_block == 256);
    for (unsigned i = 0; i <= std::numeric_limits<std::uint16_t>::max(); ++i) {
        const auto h = static_cast<std::uint16_t>(i);
        if (bits(Math::widen(h)) != bits(reference_widen(h))) {
            std::fprintf(stderr, "blocked16 widening mismatch at half %04x\n", i);
            return 1;
        }
    }

    Input x{};
    Weights w{};
    for (unsigned i = 0; i < x.size(); ++i) {
        x[i] = std::ldexp(static_cast<float>(static_cast<int>((i * 269U) % 997U) - 498), static_cast<int>(i % 29U) - 23);
        w[i] = static_cast<std::uint16_t>((i * 1137U) % 0x7c00U);
        if ((i & 7U) == 4U) w[i] |= 0x8000U;
    }
    if (!check("mixed finite", x, w)) return 1;

    x.fill(0.0F);
    w.fill(0x3c00U);
    x[0] = 0x1p30F;
    x[256] = 1.0F;
    x[2048] = -0x1p30F;
    if (reference(x, w) != 1.0F || !check("cross-half tree", x, w)) return 1;

    x.fill(0.0F);
    x[0] = 0x1p30F;
    x[1] = 1.0F;
    x[2] = -0x1p30F;
    if (bits(reference(x, w)) != bits(0.0F) || !check("ascending FMA chain", x, w)) return 1;

    x.fill(0.0F);
    w.fill(0x3c00U);
    x[0] = -1.0F;
    x[1] = 1.0F + 0x1p-14F;
    w[1] = 0x3c01U;
    if (reference(x, w) != 0x1.1004p-10F || !check("fused rounding", x, w)) return 1;

    x.fill(-0.0F);
    if (!check("positive-zero seed", x, w)) return 1;
    std::puts("blocked16_tree: exhaustive half widening and five bitwise scalar cases passed");
    return 0;
}
