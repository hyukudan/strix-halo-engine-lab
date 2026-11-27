#include "strix/moe/declared_exp.hpp"
#include "../src/ops/norm/norm_numeric_traits.hpp"

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>

#if defined(_M_X64) || defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace {
using strix::moe::declared_math::ExpForm;
using strix::moe::declared_math::declared_exp;
using strix::ops::norm::SquareForm;
using strix::ops::norm::square_step;

struct Case {
    std::uint32_t input;
    std::uint32_t expected;
};

constexpr std::array cases{
    Case{0x42b0cccdU, 0x7f395e0fU},  // 88.4, where the frozen exp returns infinity
    Case{0x42b17217U, 0x7f7fff84U},  // last finite float argument before overflow
    Case{0x42b17218U, 0x7f800000U},
    Case{0xc2cfcccdU, 0x00000001U},  // -103.9, below the frozen exp's cutoff
    Case{0xc2d00000U, 0x00000000U},
    Case{0xc2b00000U, 0x0041edc4U},
    Case{0xc2aeac50U, 0x007fffe6U},
    Case{0xbf800000U, 0x3ebc5ab2U},
    Case{0x3f800000U, 0x402df854U},
    Case{0x00000000U, 0x3f800000U},
    Case{0x80000000U, 0x3f800000U},
    Case{0x7f800000U, 0x7f800000U},
    Case{0xff800000U, 0x00000000U},
    Case{0x7fc00231U, 0x7fc00231U},
};

bool forms_have_declared_bits() {
    for (const auto& fixture : cases) {
        const volatile float x = std::bit_cast<float>(fixture.input);
        const auto actual = std::bit_cast<std::uint32_t>(declared_exp<ExpForm::fixed_rne_v2>(x));
        if (actual != fixture.expected) {
            std::cerr << std::hex << "exp input=" << fixture.input << " actual=" << actual
                      << " expected=" << fixture.expected << '\n';
            return false;
        }
    }
    return true;
}

bool same_bits_all_rounding_modes() {
    const int original = std::fegetround();
    bool ok = true;
    for (const int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        ok = std::fesetround(mode) == 0 && forms_have_declared_bits() && ok;
    }
    return std::fesetround(original) == 0 && ok;
}

bool exp_accuracy() {
    // Exact binary inputs across ordinary, subnormal and overflow ranges.
    // The approximation promises neither a correctly rounded exp nor libm identity.
    for (int i = -104 * 1024; i < 89 * 1024; ++i) {
        const float x = static_cast<float>(i) / 1024.0F;
        const auto actual = std::bit_cast<std::uint32_t>(declared_exp<ExpForm::fixed_rne_v2>(x));
        const auto oracle = std::bit_cast<std::uint32_t>(static_cast<float>(std::exp(static_cast<double>(x))));
        if (actual > oracle + 1U || oracle > actual + 1U) return false;
    }
    return true;
}

// declared_tanh: within 4 ulps of tanh (binary64 oracle rounded to binary32) on a dense sweep, tiny magnitudes and
// the branch borders; odd; +-1 saturation; NaN propagates. The shared GLU trait binds the same functions.
std::uint32_t ulp_distance(float a, float b) {
    const auto key = [](float v) {
        const auto u = std::bit_cast<std::uint32_t>(v);
        return (u & 0x80000000U) ? 0x80000000U - (u & 0x7fffffffU) : 0x80000000U + u;
    };
    const auto x = key(a), y = key(b);
    return x > y ? x - y : y - x;
}
bool tanh_accuracy() {
    using strix::moe::declared_math::declared_tanh;
    const auto close = [](float x) {
        const float oracle = static_cast<float>(std::tanh(static_cast<double>(x)));
        return ulp_distance(declared_tanh(x), oracle) <= 4U && declared_tanh(-x) == -declared_tanh(x);
    };
    for (int i = -12 * 8192; i <= 12 * 8192; ++i)
        if (!close(static_cast<float>(i) / 8192.0F)) return false;
    for (int k = 2; k < 140; ++k)
        if (!close(std::ldexp(1.0F, -k)) || !close(std::ldexp(1.5F, -k))) return false;
    for (const float b : {0.25F, 9.25F})
        if (!close(std::nextafter(b, 0.0F)) || !close(b) || !close(std::nextafter(b, 20.0F))) return false;
    const strix::moe::SoftcappedSwiglu<strix::moe::declared_math::DeclaredGluMath> situ{7.0F, 7.0F};
    return declared_tanh(30.0F) == 1.0F && declared_tanh(-30.0F) == -1.0F && std::isnan(declared_tanh(std::nanf(""))) &&
           declared_tanh(0.0F) == 0.0F && std::isfinite(situ(1.5F, -2.0F));
}

bool distinct_square_forms() {
    // At the addition midpoint, the square's discarded low bit determines
    // whether the result rounds up if multiply and add are fused.
    const volatile float a = std::bit_cast<float>(0x33800000U);
    const volatile float v = std::bit_cast<float>(0x3f800001U);
    return std::bit_cast<std::uint32_t>(square_step<SquareForm::mul_add_uncontracted>(a, v)) == 0x3f800002U &&
           std::bit_cast<std::uint32_t>(square_step<SquareForm::fma>(a, v)) == 0x3f800003U;
}
}  // namespace

int main() {
    const volatile float upper = 88.4F;
    const volatile float lower = -103.9F;
    const bool frozen = std::isinf(declared_exp(upper)) && declared_exp(lower) == 0.0F &&
                        std::isinf(declared_exp<ExpForm::frozen_v1>(upper));
    bool ok = frozen && exp_accuracy() && tanh_accuracy() && distinct_square_forms() && same_bits_all_rounding_modes();
#if defined(_M_X64) || defined(__SSE2__)
    const auto original = _mm_getcsr();
    _mm_setcsr(original | 0x8040U);  // Flush results and treat inputs as zero.
    ok = same_bits_all_rounding_modes() && ok;
    _mm_setcsr(original);
#endif
    if (!ok) std::cerr << "declared exp or square-form contract failed\n";
    return ok ? 0 : 1;
}
