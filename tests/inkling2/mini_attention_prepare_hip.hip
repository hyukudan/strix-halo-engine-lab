#include "inkling2/inkling2_mini_attention_prepare_hip.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

namespace {

using namespace strix::inkling2;
namespace bf16 = strix::weight::scalar_float;
constexpr unsigned rows = 2;
constexpr unsigned max_projection_elements = kHiddenWidth * kQueryWidth;

struct Fixture {
    alignas(16) std::array<std::uint16_t, max_projection_elements> zeros{};
    alignas(16) std::array<std::uint16_t, kKvWidth * kHiddenWidth> value_weight{};
    alignas(16) std::array<std::uint16_t, kHiddenWidth * kQueryWidth> output_weight{};
    alignas(16) std::array<float, rows * kHiddenWidth> input{}, output{}, post{};
    alignas(16) std::array<float, rows * kQueryWidth> query{}, gate_logits{}, attended{}, gated{};
    alignas(16) std::array<float, rows * kKvWidth> raw_keys{}, raw_values{}, keys{}, values{};
    alignas(16) std::array<float, rows * kKvHeads * kConvTaps> dynamic_keys{}, dynamic_values{};
    alignas(16) std::array<float, rows * kQueryHeads * kRelativeWidth> relative_features{};
    alignas(16) std::array<float, rows * kQueryHeads * kRelativeExtent> relative_logits{};
    alignas(16) std::array<float, rows * kQueryHeads> sinks{};
    alignas(16) std::array<float, kConvTaps * kHiddenWidth> lags{};
    alignas(16) std::array<std::uint16_t, kRelativeWidth * kRelativeExtent> relative_weight{};
    alignas(16) std::array<float, (kConvTaps - 1U) * kKvWidth> key_history{}, value_history{};
    alignas(16) std::array<float, (kConvTaps - 1U) * kHiddenWidth> post_history{};
    alignas(16) std::array<std::uint16_t, rows * kKvWidth> key_cache{}, value_cache{};
    alignas(16) std::array<std::uint16_t, kHeadWidth> padding{};
    std::array<std::uint16_t, 8> tap_words{};
    std::array<float, 8> bound_taps{};
    unsigned failure{};
};

void require_hip(hipError_t status) {
    if (status == hipSuccess) return;
    std::cerr << hipGetErrorString(status) << '\n';
    std::exit(1);
}

bool test_weight_binding(Fixture& f, hipStream_t stream) {
    constexpr std::array raw{0.0F, 1.0F, 2.0F, 3.0F, -4.0F, -5.0F, -6.0F, -7.0F};
    constexpr std::array reordered{3.0F, -7.0F, 2.0F, -6.0F, 1.0F, -5.0F, 0.0F, -4.0F};
    std::ranges::transform(raw, f.tap_words.begin(), bf16::bf16_encode);
    for (const bool channel_taps : {true, false}) {
        require_hip(strix::attn_core::enqueue_bind_convolution_weights(
            f.tap_words, f.bound_taps.data(), 2, kConvTaps, channel_taps, stream));
        require_hip(hipStreamSynchronize(stream));
        if (f.bound_taps != (channel_taps ? reordered : raw)) return false;
    }
    return true;
}

bool run(Fixture& f, LayerVisit visit, hipStream_t stream) {
    f.key_history.fill(0.0F);
    f.value_history.fill(0.0F);
    f.post_history.fill(0.0F);
    f.failure = 0;
    const MiniProjectionInputs input{f.input.data(), f.input.data(), f.input.data()};
    MiniProjectionWeights weights;
    weights.matrix.fill(f.zeros.data());
    weights.matrix[static_cast<unsigned>(MiniAttentionProjection::value)] = f.value_weight.data();
    weights.matrix[static_cast<unsigned>(MiniAttentionProjection::output)] = f.output_weight.data();
    weights.matrix[static_cast<unsigned>(MiniAttentionProjection::relative_bias)] = f.relative_weight.data();
    weights.query_gain = f.zeros.data();
    weights.relative_gain = f.zeros.data();
    weights.key_convolution = f.lags.data();
    weights.value_convolution = f.lags.data();
    const MiniProjectionWorkspace work{f.raw_keys.data(), f.raw_values.data(), f.dynamic_keys.data(),
        f.dynamic_values.data(), f.relative_features.data(),
        f.key_history.data(), f.value_history.data()};
    const MiniPreparedAttention prepared{f.query.data(), f.keys.data(), f.values.data(),
        f.relative_logits.data(), f.sinks.data(), f.gate_logits.data()};
    const MiniAttentionOutputRows output{f.attended.data(), f.gated.data(), f.output.data()};
    const MiniHipAttentionCache cache{f.key_cache.data(), f.value_cache.data(), f.padding.data(), rows, false};
    require_hip(enqueue_mini_hip_attention_forward(visit, 0, rows, input, weights, work, prepared,
                                                   output, cache, &f.failure, stream));
    require_hip(enqueue_mini_hip_post_convolution(rows, f.lags.data(), f.output.data(),
                                                 f.post_history.data(), f.post.data(), stream));
    require_hip(hipStreamSynchronize(stream));

    const auto first_value = f.output[0];
    const auto second_value = f.output[kHiddenWidth];
    const bool kv = std::ranges::all_of(f.key_cache, [](auto word) { return word == 0; }) &&
                    std::ranges::all_of(f.value_cache, [](auto word) { return word == bf16::bf16_encode(1.0F); });
    const bool valid = f.failure == 0 && kv && f.value_history[kKvWidth] == 1.0F &&
           f.value_history[2U * kKvWidth] == 1.0F && f.key_history[2U * kKvWidth] == 0.0F &&
           std::abs(first_value - 0.25F) <= 2.0e-3F && std::abs(second_value - (1.0F / 3.0F)) <= 4.0e-3F &&
           f.output[1] == 0.0F && f.post_history[kHiddenWidth] == first_value &&
           f.post_history[2U * kHiddenWidth] == second_value &&
           std::abs(f.post[0] - (first_value / std::sqrt(5.0F))) <= 1.0e-3F;

    // The last relative column is masked from both rows. Its producer must
    // still reject the NaN written by 0 * Inf before the visit can commit.
    constexpr auto last_distance = kRelativeExtent - 1U;
    f.relative_weight[last_distance] = bf16::bf16_encode(std::numeric_limits<float>::infinity());
    f.failure = 0;
    require_hip(enqueue_mini_hip_attention_forward(visit, 0, rows, input, weights, work, prepared,
                                                   output, cache, &f.failure, stream));
    require_hip(hipStreamSynchronize(stream));
    f.relative_weight[last_distance] = 0;
    return valid && f.failure == 1U && !std::isfinite(f.relative_logits[last_distance]);
}

}  // namespace

int main() {
    void* storage = nullptr;
    require_hip(hipMallocManaged(&storage, sizeof(Fixture)));
    auto* fixture = std::construct_at(static_cast<Fixture*>(storage));
    for (unsigned row = 0; row < rows; ++row) fixture->input[row * kHiddenWidth] = 1.0F;
    for (unsigned col = 0; col < kKvWidth; ++col)
        fixture->value_weight[col * kHiddenWidth] = bf16::bf16_encode(1.0F);
    fixture->output_weight[0] = bf16::bf16_encode(1.0F);
    hipStream_t stream{};
    require_hip(hipStreamCreate(&stream));
    constexpr auto schedule = mini_schedule();
    const bool pass = test_weight_binding(*fixture, stream) &&
                      run(*fixture, schedule[0], stream) && run(*fixture, schedule[5], stream);
    if (!pass) std::cerr << "Mini HIP prepared attention or its convolution history differs from the fixture\n";
    require_hip(hipStreamDestroy(stream));
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    return pass ? 0 : 1;
}
