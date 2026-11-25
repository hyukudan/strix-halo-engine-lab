// Host check of the HIP device steps of a8g32-v1 (src/ops/a8/a8_group_device.hpp group_scale / code_of, compiled on a CPU
// through the scalar shim in tests/ds41_integer_host): applied to a group the way any wave topology applies them (the maximum
// over the 32 values, then the code of each), they give the bits of the ONE CPU reference strix::core::a8g32::quantize_group on
// random and adversarial groups: zeros, tiny values, the 2^-119 boundary, ties, clamp edges, signed zeros, infinities, NaN.
// It does not exercise the wave shuffles; that the exchanged maximum and integer sum are order-free is the device identity
// argument, and the machine code of every bound translation unit is compared base against head separately.
#include "ops/a8/a8_group_device.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
std::uint32_t bits(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    return u;
}

using Group = std::array<float, 32>;

// The device steps applied to a group: the maximum is the order-free fmaxf reduction (NaN ignored), then code_of per value.
bool same(const Group& x, const char* what) {
    std::array<std::int8_t, 32> want_codes{};
    const auto want = strix::core::a8g32::quantize_group(x.data(), want_codes.data());
    float amax = std::fabs(x[0]);
    for (std::size_t j = 1; j < 32; ++j) amax = fmaxf(amax, std::fabs(x[j]));
    // The reference seeds the maximum from +0; a lane that holds NaN first keeps a NaN seed here, so seed from +0 as the
    // reference and the wave bodies (fmaxf(0, |y|)) do.
    amax = fmaxf(0.0F, amax);
    const auto scale = strix::ops::a8::group_scale(amax);
    std::int32_t sum = 0;
    bool ok = bits(scale.delta) == bits(want.delta);
    for (std::size_t j = 0; j < 32; ++j) {
        const int code = strix::ops::a8::code_of(x[j], scale);
        ok = ok && code == want_codes[j];
        sum += code;
    }
    ok = ok && sum == want.sum;
    expect(ok, what);
    return ok;
}

}  // namespace

int main() {
    namespace a8 = strix::core::a8g32;
    static_assert(strix::ops::a8::code_max == a8::code_max && strix::ops::a8::tiny_below == a8::tiny_below);

    Group g{};
    same(g, "zero group");
    for (float tiny : {1e-38F, 0x1p-120F, 0x1p-119F, 0x1p-118F, 1e-45F}) {
        g.fill(0.0F);
        g[3] = tiny;
        g[9] = -tiny / 2.0F;
        same(g, "tiny boundary");
    }
    g.fill(-0.0F);
    same(g, "signed zeros");
    for (std::size_t j = 0; j < 32; ++j) g[j] = (j % 2 ? -1.0F : 1.0F) * (0.5F + static_cast<float>(j));  // ties
    same(g, "rounding ties");
    g.fill(127.0F);
    g[0] = -127.0F;
    g[1] = 1e30F;
    same(g, "clamp edge");
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    g.fill(1.0F);
    g[7] = inf;
    same(g, "infinity");
    g[7] = -inf;
    same(g, "negative infinity");
    g.fill(1.0F);
    g[0] = nan;
    same(g, "NaN first lane");
    g[5] = nan;
    g[6] = 3.0F;
    same(g, "NaN with finite values");
    g.fill(nan);
    same(g, "all NaN");

    std::mt19937 rng(12345U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (int round = 0; round < 20000; ++round) {
        const float scale = std::ldexp(1.0F, static_cast<int>(rng() % 90U) - 60);
        for (auto& v : g) v = normal(rng) * scale;
        if (round % 7 == 0) g[rng() % 32U] = 0.0F;
        if (round % 11 == 0)
            for (auto& v : g) v = std::nearbyint(v / scale * 8.0F) * scale / 8.0F;  // lattice values: many ties
        same(g, "random group");
    }
    if (failures == 0) std::printf("a8g32 device steps: bitwise equal to the CPU reference\n");
    return failures == 0 ? 0 : 1;
}
