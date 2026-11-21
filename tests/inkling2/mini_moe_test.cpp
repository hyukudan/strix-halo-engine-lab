#include "inkling2/inkling2_mini_moe.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace {

struct HostMath {
    static float exp(float value) { return std::exp(value); }
    static float tanh(float value) { return std::tanh(value); }
};

}  // namespace

int main() {
    using namespace strix::inkling2;
    std::array<float, kMiniRoute.experts> logits{};
    std::array<float, kMiniRoute.experts> scores{};
    std::array<float, kMiniRoute.experts> bias{};
    std::array<std::uint32_t, kMiniRoute.top_k> ids{};
    std::array<float, kMiniRoute.top_k> weights{};
    logits[4] = std::log(3.0F);
    bias[127] = 1.0F;
    bias[10] = -1.0F;
    strix::moe::route_scores_ref<HostMath>(kMiniRoute.score, logits, scores);
    strix::moe::route_select_ref(kMiniRoute, scores, bias, ids, weights);
    if (ids != std::array<std::uint32_t, 8>{127, 4, 0, 1, 2, 3, 5, 6}) return 1;
    if (!(std::abs(weights[0] / weights[1] - 2.0F / 3.0F) <= 1e-6F)) return 2;
    if (!(std::abs(std::accumulate(weights.begin(), weights.end(), 0.0F) - 8.0F) <= 1e-6F)) return 3;

    constexpr auto activation = kMiniExpertActivation<HostMath>;
    if (activation(0.0F, 3.0F) != 0.0F) return 4;
    if (!(std::abs(activation(100.0F, 1000.0F) - 100.0F) <= 1e-5F)) return 5;
    if (!(std::abs(activation(100.0F, -1000.0F) + 100.0F) <= 1e-5F)) return 6;
    const float negative_gate = activation(-12.0F, 30.0F);
    // Independent evaluated value, with sigmoid on -12 rather than the capped gate.
    if (!(std::abs(negative_gate - (-0.000509679F)) <= 1e-8F)) return 7;
    return 0;
}
