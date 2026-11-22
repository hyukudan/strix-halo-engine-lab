// Host checks of the projected relative bias (attn_core/relative_bias.hpp):
// 1. relative_bias_chain over packed k-contiguous BF16 pairs is the ascending FMA chain over the BF16 weights
//    (the producer's per-output chain) followed by bf16_round.
// 2. relative_bias_bounded is sound: when it returns true, every logit of those features is finite for any weights
//    with |w| <= w_max (the flash then skips the exhaustive non-finite scan the producer stage did).
// 3. Non-finite features or weights make it return false.
#include "attn_core/relative_bias.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

namespace {
namespace ab = strix::attn_core;
namespace sf = strix::weight::scalar_float;
constexpr unsigned kWidth = 16;

float from_bf16(std::uint16_t b) { return sf::u2f(std::uint32_t{b} << 16U); }
std::uint16_t to_bf16(float x) { return sf::bf16_encode(x); }

float reference(const float* x, const std::uint16_t* w) {
    float r = 0.0F;
    for (unsigned k = 0; k < kWidth; ++k) r = std::fma(x[k], from_bf16(w[k]), r);
    return sf::bf16_round(r);
}

float chain(const float* x, const std::uint16_t* w) {
    std::uint32_t pairs[kWidth / 2U];
    for (unsigned i = 0; i < kWidth / 2U; ++i) pairs[i] = std::uint32_t{w[2U * i]} | (std::uint32_t{w[2U * i + 1U]} << 16U);
    return ab::relative_bias_chain<kWidth>(x, pairs);
}

float magnitude(const std::uint16_t* w) {
    float m = 0.0F;
    for (unsigned k = 0; k < kWidth; ++k) m = std::fmax(m, std::fabs(from_bf16(w[k])));
    return m;
}

bool same_bits(float a, float b) {
    std::uint32_t x = 0, y = 0;
    std::memcpy(&x, &a, sizeof x);
    std::memcpy(&y, &b, sizeof y);
    return x == y;
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    std::mt19937 rng(20261122U);
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::uniform_int_distribution<int> exponent(-20, 127);

    // 1 + 2: random magnitudes up to and past the bound; every bounded case must give finite logits.
    unsigned bounded = 0, unbounded = 0;
    for (unsigned trial = 0; trial < 200000U; ++trial) {
        float x[kWidth];
        std::uint16_t w[kWidth];
        const float xs = std::ldexp(1.0F, exponent(rng)), ws = std::ldexp(1.0F, exponent(rng) / 4);
        const bool same_sign = trial % 4U == 0U;
        for (unsigned k = 0; k < kWidth; ++k) {
            x[k] = xs * (same_sign ? std::fabs(unit(rng)) : unit(rng));
            w[k] = to_bf16(ws * (same_sign ? std::fabs(unit(rng)) : unit(rng)));
        }
        const float value = chain(x, w);
        expect(same_bits(value, reference(x, w)), "chain == ascending FMA chain over BF16 weights + bf16_round");
        if (ab::relative_bias_bounded<kWidth>(x, magnitude(w))) {
            ++bounded;
            expect(std::isfinite(value), "bounded features give a finite logit");
        } else {
            ++unbounded;
        }
    }
    expect(bounded > 1000U && unbounded > 1000U, "the sweep covers both sides of the bound");

    // Edge: all products positive and equal, total just below / above 2^126.
    {
        float x[kWidth];
        std::uint16_t w[kWidth];
        for (unsigned k = 0; k < kWidth; ++k) w[k] = to_bf16(1.0F);
        for (unsigned k = 0; k < kWidth; ++k) x[k] = std::ldexp(0.99F, 122);  // 16 x 0.99 x 2^122 < 2^126
        expect(ab::relative_bias_bounded<kWidth>(x, 1.0F), "sum just below 2^126 is bounded");
        expect(std::isfinite(chain(x, w)), "just below 2^126 is finite");
        for (unsigned k = 0; k < kWidth; ++k) x[k] = std::ldexp(1.01F, 122);
        expect(!ab::relative_bias_bounded<kWidth>(x, 1.0F), "sum just above 2^126 is not bounded");
    }

    // 3: non-finite operands are never bounded; zero weights with finite features are.
    {
        float x[kWidth] = {};
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        expect(ab::relative_bias_bounded<kWidth>(x, 0.0F), "zero features, zero weights");
        expect(!ab::relative_bias_bounded<kWidth>(x, inf), "infinite weight");
        expect(!ab::relative_bias_bounded<kWidth>(x, nan), "NaN weight");
        x[5] = inf;
        expect(!ab::relative_bias_bounded<kWidth>(x, 0.0F), "infinite feature, zero weights (inf x 0)");
        expect(!ab::relative_bias_bounded<kWidth>(x, 1.0F), "infinite feature");
        x[5] = nan;
        expect(!ab::relative_bias_bounded<kWidth>(x, 1.0F), "NaN feature");
        // The flash's w_max: largest BF16 magnitude bits; an infinite or NaN weight maps to inf / NaN.
        expect(std::isinf(sf::u2f(std::uint32_t{0x7f80U} << 16U)) && std::isnan(sf::u2f(std::uint32_t{0x7fc1U} << 16U)),
               "magnitude bits >= 0x7f80 are not finite");
    }

    std::printf("relative bias bound: %u bounded, %u unbounded, %d failures\n", bounded, unbounded, failures);
    return failures == 0 ? 0 : 1;
}
