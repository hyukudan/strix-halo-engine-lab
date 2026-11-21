#include "inkling2/inkling2_mini_moe_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

int main() {
    using namespace strix::inkling2;
    using strix::moe::ExpertProjection;

    std::array<float, kHiddenWidth> hidden{};
    hidden[0] = 1.003F;
    std::vector<float> router(static_cast<std::size_t>(kExperts) * kHiddenWidth);
    router[4U * kHiddenWidth] = std::log(3.0F);
    std::array<float, kExperts> bias;
    bias.fill(-1.0F);
    for (unsigned id = 1; id <= 7; ++id) bias[id] = 1.0F;
    bias[127] = 2.0F;

    std::vector<std::pair<unsigned, ExpertProjection>> calls;
    bool valid = true;
    const auto project = [&](std::uint32_t expert, ExpertProjection kind,
                             std::span<const float> input, std::span<float> out) {
        calls.emplace_back(expert, kind);
        std::ranges::fill(out, 0.0F);
        if (kind == ExpertProjection::gate_up) {
            valid &= input.size() == kHiddenWidth && out.size() == 2U * kExpertWidth && input.front() == 1.0F;
            out[0] = 0.9011F;
            out[kExpertWidth] = 1.6011F;
        } else {
            valid &= input.size() == kExpertWidth && out.size() == kHiddenWidth &&
                input.front() == 1.0078125F && input.back() == 0.0F;
            if (expert == 4U) out[0] = 3.151F;
            if (expert == 127U) out[0] = -0.7751F;
        }
    };

    std::array<float, kHiddenWidth> output{};
    mini_moe_reference(hidden, router, bias, project, output);
    if (!valid || calls.size() != 2U * kExpertsPerToken || output[0] != 3.734375F || output[1] != 0.0F) return 1;
    // Expert 127 wins selection. The combine still visits expert IDs in ascending order.
    constexpr std::array<unsigned, kExpertsPerToken> expected{1, 2, 3, 4, 5, 6, 7, 127};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (calls[2 * i] != std::pair{expected[i], ExpertProjection::gate_up} ||
            calls[2 * i + 1] != std::pair{expected[i], ExpertProjection::down}) return 2;
    }

    // Model reader supplies FP32 dequantized rows. The checkpoint projection
    // rounds their values to BF16 before multiplying with the activation.
    std::array<float, 2> projected{};
    constexpr std::array<float, 2> small_input{1.0F, 2.0F};
    project_mini_expert_rows(small_input, projected, [](std::size_t row, std::span<float> weights) {
        weights[0] = row == 0 ? 1.003F : -1.003F;
        weights[1] = 2.009F;
    });
    if (projected != std::array{5.03125F, 3.03125F}) return 3;

    // Failure in an expert source leaves the caller's output intact.
    output.fill(42.0F);
    try {
        mini_moe_reference(hidden, router, bias,
                           [](auto, auto, auto, auto) { throw std::runtime_error("missing expert slice"); },
                           output);
        return 4;
    } catch (const std::runtime_error&) {
        if (!std::ranges::all_of(output, [](float v) { return v == 42.0F; })) return 5;
    }
    return 0;
}
