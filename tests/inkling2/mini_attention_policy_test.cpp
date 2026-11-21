#include "attn_core/relative_bias.hpp"

#include <array>
#include <cmath>
#include <limits>

int main() {
    using strix::attn_core::AttentionSinkRows;
    using strix::attn_core::RelativeBiasRows;

    // Two projected rows, two heads, four positions per relative bias row.
    constexpr float ln2 = 0.6931471805599453F;
    constexpr std::array<float, 16> relative{
        0.0F, ln2, -ln2, 2.0F * ln2,
        ln2, 0.0F, -2.0F * ln2, 3.0F * ln2,
        2.0F * ln2, -ln2, -3.0F * ln2, 0.0F,
        3.0F * ln2, 2.0F * ln2, ln2, -ln2,
    };
    constexpr std::array<float, 4> sinks{0.0F, -ln2, ln2, 2.0F * ln2};
    const RelativeBiasRows bias{relative.data(), 2U, 4U};
    const AttentionSinkRows sink{sinks.data(), 2U};
    constexpr unsigned pos = 1000000U;
    if (bias.score_bias(1U, 1U, pos, pos) != 3.0F * ln2) return 1;
    if (bias.score_bias(1U, 0U, pos, pos - 3U) != 0.0F) return 2;
    if (bias.score_bias(0U, 1U, pos, pos - 3U) != 3.0F * ln2) return 3;
    if (bias.score_bias(0U, 1U, pos, pos - 4U) != 0.0F) return 4;
    if (bias.score_bias(0U, 1U, pos, pos + 1U) != 0.0F) return 5;
    if (bias.score_bias(0U, 1U, 0U, std::numeric_limits<unsigned>::max()) != 0.0F) return 6;

    constexpr RelativeBiasRows disabled{nullptr, 2U, 0U};
    static_assert(disabled.score_bias(0U, 0U, 0U, 0U) == 0.0F);
    if (sink.sink(0U, 1U) != -ln2 || sink.sink(1U, 1U) != 2.0F * ln2) return 7;

    // For row 0, head 0, scaled QK logits ln(2), -ln(2), 0 receive bias 0, ln(2), 0.
    // Their unnormalized weights are 2, 1, 1. The zero-value sink contributes 1 to the denominator.
    constexpr std::array<unsigned, 3> keys{pos, pos - 1U, pos - 4U};
    constexpr std::array<float, 3> dots{2.0F * ln2, -2.0F * ln2, 0.0F};
    constexpr std::array<float, 3> values{3.0F, 6.0F, 8.0F};
    double numerator = 0.0;
    double denominator = std::exp(sink.sink(0U, 0U));
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const float logit = 0.5F * dots[i] + bias.score_bias(0U, 0U, pos, keys[i]);
        const double weight = std::exp(logit);
        numerator += weight * values[i];
        denominator += weight;
    }
    return std::abs(numerator / denominator - 4.0) <= 1e-6 ? 0 : 8;
}
