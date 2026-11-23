#include "strix/moe/route_policy.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {

using Dot = strix::moe::RouteStridedFmaXor32;
using Math = strix::moe::RouteCodyWaiteExpV1;

void check_bits(float actual, float expected, const char* why) {
    if (std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected))
        throw std::runtime_error(why);
}

void test_dot() {
    std::array<float, 33> hidden{}, weight{};
    weight.fill(1.0F);
    hidden[0] = 16777216.0F;
    hidden[1] = 1.0F;
    hidden[16] = -16777216.0F;
    check_bits(Dot::dot_ref(hidden, weight), 1.0F, "XOR reduction order");

    hidden.fill(0.0F);
    constexpr float eps = std::numeric_limits<float>::epsilon();
    hidden[0] = -1.0F;
    hidden[32] = 1.0F + eps;
    weight[32] = 1.0F - eps;
    check_bits(Dot::dot_ref(hidden, weight), -eps * eps, "strided fused accumulation");
    check_bits(Dot::dot_ref(std::array{3.0F}, std::array{-2.0F}), -6.0F, "partial warp");
}

void test_scores() {
    const auto project = [](float value, float bias) {
        return strix::moe::route_project_sigmoid_ref<Dot, Math>(std::array{value}, std::array{1.0F}, bias);
    };
    const auto zero = project(0.0F, -0.75F);
    check_bits(zero.unbiased, 0.5F, "unbiased score at zero");
    check_bits(zero.biased, -0.25F, "bias added after sigmoid");
    check_bits(project(90.0F, 0.0F).unbiased, 1.0F, "negative exp cutoff");
    check_bits(project(-90.0F, 0.0F).unbiased, 0.0F, "positive exp cutoff");
    check_bits(Math::exp(-87.0F), 0x1.666d0ep-126F, "lower cutoff is inclusive");
    if (!std::isfinite(Math::exp(88.0F))) throw std::runtime_error("upper cutoff is inclusive");
    if (!std::isnan(Math::exp(std::numeric_limits<float>::quiet_NaN())))
        throw std::runtime_error("NaN exp");
}

}  // namespace

int main() {
    try {
        test_dot();
        test_scores();
        std::puts("k3r host strided FMA, XOR and score planes PASS");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "k3r host FAIL: %s\n", error.what());
        return 1;
    }
}
