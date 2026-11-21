#include "inkling2/inkling2_mini_moe.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>

namespace {

struct HostMath {
    static float exp(float value) { return std::exp(value); }
    static float tanh(float value) { return std::tanh(value); }
};

}  // namespace

int main() {
    using namespace strix::inkling2;
    std::array<float, 2U * kMiniRoute.experts> logits{};
    std::array<float, kMiniRoute.experts> bias{};
    logits[4] = std::log(3.0F);
    logits[kMiniRoute.experts + 6] = std::log(3.0F);
    logits[kMiniRoute.experts + 127] = -std::log(3.0F);
    bias[127] = 1.0F;
    bias[10] = -1.0F;
    const auto routes = select_mini_routes(logits, bias);
    constexpr std::array<std::uint32_t, 16> ids{127, 4, 0, 1, 2, 3, 5, 6, 127, 6, 0, 1, 2, 3, 4, 5};
    if (!std::ranges::equal(routes.experts, ids)) return 1;
    const auto weights = std::span(routes.weights).first(kMiniRoute.top_k);
    if (!(std::abs(weights[0] / weights[1] - 2.0F / 3.0F) <= 1e-6F)) return 2;
    if (!(std::abs(std::accumulate(weights.begin(), weights.end(), 0.0F) - 8.0F) <= 1e-6F)) return 3;

    constexpr auto activation = kMiniExpertActivation<HostMath>;
    if (activation(0.0F, 3.0F) != 0.0F) return 4;
    if (!(std::abs(activation(100.0F, 1000.0F) - 100.0F) <= 1e-5F)) return 5;
    if (!(std::abs(activation(100.0F, -1000.0F) + 100.0F) <= 1e-5F)) return 6;
    const float negative_gate = activation(-12.0F, 30.0F);
    // Independent evaluated value, with sigmoid on -12 rather than the capped gate.
    if (!(std::abs(negative_gate - (-0.000509679F)) <= 1e-8F)) return 7;

    const auto second = std::span(routes.weights).subspan(kMiniRoute.top_k);
    constexpr std::array second_expected{0.5F, 1.5F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    const auto near = [](float a, float b) { return std::abs(a - b) <= 1e-6F; };
    if (!std::ranges::equal(second, second_expected, near)) return 8;

    const auto rejects = [](std::span<const float> route_logits, std::span<const float> route_bias) {
        try {
            (void)select_mini_routes(route_logits, route_bias);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    if (!rejects(std::span(logits).first(logits.size() - 1U), bias) ||
        !rejects(logits, std::span(bias).first(bias.size() - 1U))) return 9;
    logits.back() = std::numeric_limits<float>::infinity();
    if (!rejects(logits, bias)) return 10;
    return 0;
}
