#include "inkling2/inkling2_mini_attention_prepare.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <span>
#include <stdexcept>

namespace {

namespace mini = strix::inkling2;
using Projection = mini::MiniAttentionProjection;

void expect(bool matches, const char* message) {
    if (!matches) throw std::runtime_error(message);
}

struct Projections {
    unsigned key_tap_calls = 0;
    unsigned value_tap_calls = 0;
    unsigned relative_head_calls = 0;
    unsigned output_calls = 0;

    void operator()(Projection projection, std::span<const float> input, std::span<float> output) {
        using namespace mini;
        switch (projection) {
            case Projection::query:
                expect(input[0] == 3.0F, "Q projection used a different normalized stream");
                expect(output.size() == kQueryWidth, "wrong Q projection shape");
                std::ranges::fill(output, 0.25F);
                std::ranges::fill(output.first(kHeadWidth), 16.0F);
                break;
            case Projection::key:
                expect(input[0] == 1.0F, "K projection used a different normalized stream");
                expect(output.size() == kKvWidth, "wrong K projection shape");
                std::ranges::fill(output, input[0]);
                break;
            case Projection::value:
                expect(input[0] == 2.0F, "V projection used a different normalized stream");
                expect(output.size() == kKvWidth, "wrong V projection shape");
                std::ranges::fill(output, 0.0F);
                for (unsigned head = 0; head < kKvHeads; ++head) output[head * kHeadWidth] = input[0];
                break;
            case Projection::key_taps:
                ++key_tap_calls;
                expect(input[0] == 1.0F && output.size() == kKvHeads * kConvTaps, "wrong K tap shape or stream");
                std::ranges::fill(output, 0.0F);
                for (unsigned head = 0; head < kKvHeads; ++head) {
                    output[head * kConvTaps + 1U] = 0.5F * static_cast<float>(head + 1U);
                }
                break;
            case Projection::value_taps:
                ++value_tap_calls;
                expect(input[0] == 2.0F && output.size() == kKvHeads * kConvTaps, "wrong V tap shape or stream");
                std::ranges::fill(output, 0.0F);
                output[1] = 1.0F;
                break;
            case Projection::relative:
                expect(input[0] == 3.0F && output.size() == kQueryHeads * kRelativeWidth, "wrong relative shape or stream");
                std::ranges::fill(output, 0.5F);
                std::ranges::fill(output.first(kRelativeWidth), 6.0F);
                break;
            case Projection::relative_bias:
                ++relative_head_calls;
                expect(input.size() == kRelativeWidth && output.size() == kRelativeExtent, "wrong relative-bias shape");
                for (std::size_t distance = 0; distance < output.size(); ++distance) {
                    output[distance] = input[0] - 0.25F * static_cast<float>(distance % 4U);
                }
                break;
            case Projection::sink:
                expect(input[0] == 3.0F && output.size() == kQueryHeads, "wrong sink shape or stream");
                std::ranges::fill(output, 0.3333F);
                break;
            case Projection::output_gate:
                expect(input[0] == 3.0F && output.size() == kQueryWidth, "wrong gate shape or stream");
                std::ranges::fill(output, 0.0F);
                output[1] = 1.0F;
                break;
            case Projection::output:
                ++output_calls;
                expect(input.size() == kQueryWidth && output.size() == kHiddenWidth, "wrong O projection shape");
                expect(input[0] == 1.046875F && input[1] == 1.53125F, "gate must precede O projection in BF16");
                std::ranges::fill(output, 0.3333F);
                break;
        }
    }
};

void test_static_layout() {
    constexpr std::array raw{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
    constexpr std::array expected{3.0F, 7.0F, 2.0F, 6.0F, 1.0F, 5.0F, 0.0F, 4.0F};
    expect(std::ranges::equal(mini::mini_short_conv_weights(raw, 2), expected), "static tap layout is reversed");
}

void test_projection() {
    using namespace mini;
    constexpr auto schedule = mini_schedule();
    constexpr auto slide_visit = schedule[6];
    constexpr auto full_visit = schedule[11];
    std::array<float, kHiddenWidth> q_input{3.0F}, k_input{1.0F}, v_input{2.0F};
    std::array<float, kQueryWidth> q_scale;
    q_scale.fill(0.5F);
    std::ranges::fill(std::span<float>(q_scale).first(kHeadWidth), 2.0F);
    std::array<float, kQueryHeads * kRelativeWidth> relative_scale;
    relative_scale.fill(0.5F);
    std::array<float, kConvTaps * kKvWidth> key_lags{}, value_lags{};
    std::ranges::fill(std::span<float>(key_lags).first(kKvWidth), 1.0F);
    const MiniAttentionPrepWeights weights{q_scale, relative_scale, key_lags, value_lags};
    std::array<float, (kConvTaps - 1U) * kKvWidth> k_history{}, v_history{};
    Projections project;

    const auto sliding = prepare_mini_attention(slide_visit, 16383, q_input, k_input, v_input,
                                               weights, k_history, v_history, project);
    expect(project.key_tap_calls == 0 && project.value_tap_calls == 0, "sliding attention projected dynamic taps");
    expect(sliding.query[0] == 2.0F && sliding.query[kHeadWidth] == 0.125F, "scaled head clip is incorrect");
    expect(sliding.key[0] == 0.89453125F, "static K residual normalization is incorrect");
    expect(sliding.value[0] == 11.3125F && sliding.value[1] == 0.0F, "V RMS norm is incorrect");
    expect(sliding.relative_logits[1] == 1.75F && sliding.relative_logits[kRelativeExtent] == 0.25F,
           "relative head or lag is incorrect");
    expect(sliding.sinks[0] == 0.333984375F, "sink projection should be rounded to BF16");
    expect(sliding.output_gate[0] == 0.5F && std::abs(sliding.output_gate[1] - 0.7310586F) <= 1e-7F,
           "sigmoid output gate is incorrect");
    expect(k_history[2U * kKvWidth] == 1.0F && v_history[2U * kKvWidth] == 2.0F,
           "convolution histories must save raw projections");

    auto latest_key_raw = std::span<float>(k_history).last(kKvWidth);
    std::ranges::fill(latest_key_raw, 4.0F);
    auto latest_value_raw = std::span<float>(v_history).last(kKvWidth);
    std::ranges::fill(latest_value_raw, 0.0F);
    latest_value_raw[1] = -2.0F;
    const auto full = prepare_mini_attention(full_visit, 16383, q_input, k_input, v_input,
                                            weights, k_history, v_history, project);
    expect(project.key_tap_calls == 1 && project.value_tap_calls == 1, "full attention did not project both dynamic taps");
    expect(project.relative_head_calls == 2U * kQueryHeads, "relative output needs one independent row per head");
    expect(full.query[0] == 2.140625F && full.relative_logits[0] == 2.140625F,
           "full attention log scaling should follow clip and rounding");
    expect(full.key[0] == 1.3359375F && full.key[kHeadWidth] == 2.0F, "dynamic K tap layout or clipping is incorrect");
    expect(full.value[0] == 8.0F && full.value[1] == -8.0F, "dynamic V tap layout or normalization is incorrect");
    expect(k_history[2U * kKvWidth] == 1.0F && v_history[2U * kKvWidth + 1U] == 0.0F,
           "dynamic convolution histories must save raw projections");

    std::array<float, kQueryWidth> attended{2.1F, 2.1F};
    std::array<float, kHiddenWidth> output;
    project_mini_attention_output(attended, full.output_gate, project, output);
    expect(project.output_calls == 1 && output[0] == 0.333984375F, "O projection should produce a BF16 hidden row");
}

} // namespace

int main() {
    try {
        test_static_layout();
        test_projection();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
