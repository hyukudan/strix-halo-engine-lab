#pragma once

#include "core/residual_device.hpp"
#include "inkling2_mini_forward.hpp"
#include "inkling2_mini_resident_forward.hpp"
#include "inkling2_mini_resident_weights.hpp"

namespace strix::inkling2 {

constexpr std::size_t mini_controller_elements(unsigned rows) {
    return std::size_t{rows} * (4U * kHiddenWidth + kControllerRank);
}

// Operations binds the backend's shared grouped BF16-weight projections and
// RMS launcher. The controller keeps FP32 rows and explicitly rounds only
// the normalized down operands, latent and output norm to BF16.
template <class Operations>
void enqueue_mini_layer_input(
    const MiniResidentWeights& weights, LayerVisit visit, MessagePath path, unsigned rows,
    std::span<const float> raw, std::span<const std::span<const float>> previous_bank,
    std::span<float> output, std::span<float> scratch, Operations& operations) {
    namespace residual = core::residual_device;
    const auto row_elements = std::size_t{rows} * kHiddenWidth;
    if (!valid_mini_visit(visit) || rows == 0 || rows > kMaxContext ||
        raw.size() != row_elements || output.size() != row_elements)
        throw std::invalid_argument("Inkling 2 Mini: invalid layer input rows");
    const auto scale = weights.bf16(mini_layer_parameter(visit.physical_layer, mini_input_norm_suffix(visit, path)));
    if (visit.pass == 0U) {
        operations.rms(raw.data(), scale.data(), output.data(), rows);
        return;
    }
    if (previous_bank.size() != 2U || previous_bank[0].size() != row_elements ||
        previous_bank[1].size() != row_elements || scratch.size() < mini_controller_elements(rows))
        throw std::invalid_argument("Inkling 2 Mini: invalid previous-pass controller buffers");

    auto message = scratch.first(row_elements);
    auto operands = scratch.subspan(row_elements, 2U * row_elements);
    // The two projection outputs reuse the concatenated down-input storage.
    auto forget = operands.first(row_elements);
    auto gain = operands.last(row_elements);
    auto conditioned = scratch.subspan(3U * row_elements, row_elements);
    auto latent = scratch.subspan(4U * row_elements, std::size_t{rows} * kControllerRank);

    const bool key_value = path == MessagePath::key || path == MessagePath::value;
    const auto queries = weights.f32(key_value ? "middle_cycle_cross_pass_kv_queries"
                                                : "middle_cycle_cross_pass_queries");
    const unsigned query = 2U * (visit.physical_layer - kCoreFirst) +
                           (path == MessagePath::value || path == MessagePath::moe ? 1U : 0U);
    residual::MixArgs mix{};
    mix.banks[0] = previous_bank[0].data();
    mix.banks[1] = previous_bank[1].data();
    mix.queries = queries.data() + query * kHiddenWidth;
    mix.output[0] = message.data();
    mix.rows = rows;
    mix.width = kHiddenWidth;
    mix.bank_count = 2U;
    mix.query_count = 1U;
    mix.normalized = true;
    mix.epsilon = 1e-6F;
    operations.check(residual::launch_mix(mix, operations.stream()));
    const auto normalize = [&](const float* input, float* out) {
        operations.check(residual::launch_normalize(
            {input, out, rows, kHiddenWidth, 1e-6F, kHiddenWidth, 2U * kHiddenWidth, true}, operations.stream()));
    };
    normalize(raw.data(), operands.data());
    normalize(message.data(), operands.data() + kHiddenWidth);

    const auto stem = mini_layer_parameter(
        visit.physical_layer, std::string("controllers.") + std::string(mini_controller_path(path)) + ".");
    using Projection = typename Operations::Projection;
    const std::array down{Projection{weights.bf16(stem + "down_weight").data(), kControllerRank, latent.data()}};
    operations.dense(operands.data(), 2U * kHiddenWidth, rows, down);
    operations.check(residual::launch_round({latent.data(), latent.data(), latent.size()}, operations.stream()));
    const std::array gates{
        Projection{weights.bf16(stem + "forget_weight").data(), kHiddenWidth, forget.data()},
        Projection{weights.bf16(stem + "gain_weight").data(), kHiddenWidth, gain.data()},
    };
    operations.dense(latent.data(), kControllerRank, rows, gates);
    operations.check(residual::launch_controller(
        {raw.data(), message.data(), forget.data(), gain.data(), conditioned.data(), row_elements}, operations.stream()));
    operations.rms(conditioned.data(), scale.data(), output.data(), rows);
}

}  // namespace strix::inkling2
