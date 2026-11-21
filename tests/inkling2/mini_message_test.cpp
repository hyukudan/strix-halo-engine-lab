#include "core/residual_reference.hpp"
#include "inkling2/inkling2_mini_messages.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <span>
#include <stdexcept>

namespace {

namespace ref = strix::core::residual_reference;

constexpr float kEpsilon = 1.0e-6F;

void close(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("result shape mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!(std::abs(actual[i] - expected[i]) <= 2.0e-6F)) {
            throw std::runtime_error("result differs from the residual oracle");
        }
    }
}

template <class F>
void rejects(F run) {
    try {
        run();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid shape accepted");
}

void test_message_attention() {
    // One token with two previous-pass states and four asymmetric learned queries (Q/K/V/MLP).
    // Expected values evaluate RMS over each state, then scaled dot product, softmax and PV in double.
    constexpr std::array states{3.0F, 4.0F, 0.0F, 2.0F};
    constexpr std::array queries{0.0F, 0.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.75F, 0.25F};
    constexpr std::array expected{
        0.424264052F, 1.272792007F, 0.585462694F, 1.219059182F,
        0.263065409F, 1.326524831F, 0.620323751F, 1.207438842F,
    };
    const auto bank = ref::normalize_bank(states, 2, kEpsilon);
    close(ref::mix(bank, queries, 2), expected);

    // A large positive score must remain finite, and one bank row always has probability one.
    constexpr std::array large_query{1000.0F, 0.0F};
    const auto first = std::span<const float>(bank).first(2);
    close(ref::mix(bank, large_query, 2), first);
    close(ref::mix(first, large_query, 2), first);

    rejects([&] { (void)ref::normalize_bank(states, 3, kEpsilon); });
    rejects([&] { (void)ref::normalize_bank(states, 2, 0.0F); });
    rejects([&] { (void)ref::mix({}, queries, 2); });
}

void test_low_rank_gate() {
    constexpr std::array current{2.0F, 0.5F};
    constexpr std::array message{0.2F, -3.0F};
    constexpr std::array down{0.25F, -0.75F, 0.5F, 0.125F, -0.5F, 0.375F, 0.25F, -0.25F};
    constexpr std::array forget{0.25F, -0.5F, -0.5F, 0.375F};
    constexpr std::array gain{0.75F, 0.125F, -0.5F, -0.5F};
    const ref::LowRankGateWeights weights{down, forget, gain};

    // Independent row RMS followed by (1 - tanh(F*z))*current + tanh(G*z)*message.
    // The BF16 arm additionally rounds the normalized inputs and the down projection output.
    constexpr std::array fp32_expected{1.829998116F, 0.187424568F};
    constexpr std::array bf16_expected{1.829464266F, 0.186526240F};
    close(ref::apply_low_rank_gate(current, message, weights, kEpsilon, ref::DownProjectionPrecision::fp32),
          fp32_expected);
    close(ref::apply_low_rank_gate(current, message, weights, kEpsilon, ref::DownProjectionPrecision::bf16),
          bf16_expected);

    constexpr std::array<float, 8> zero_down{};
    const ref::LowRankGateWeights zero_latent{zero_down, forget, gain};
    close(ref::apply_low_rank_gate(current, message, zero_latent, kEpsilon, ref::DownProjectionPrecision::bf16),
          current);

    const ref::LowRankGateWeights short_down{std::span<const float>(down).first(7), forget, gain};
    rejects([&] {
        (void)ref::apply_low_rank_gate(current, message, short_down, kEpsilon, ref::DownProjectionPrecision::bf16);
    });
}

void test_mini_checkpoint_query_order() {
    using namespace strix::inkling2;
    std::vector<float> states(2U * kHiddenWidth);
    states[0] = 1.0F;
    states[kHiddenWidth + 1U] = 1.0F;
    std::vector<float> attention_moe(2U * kHiddenWidth * MiniLayers::core_width);
    std::vector<float> key_value(attention_moe.size());
    attention_moe[0] = 1.0F;
    attention_moe[kHiddenWidth] = -2.0F;
    key_value[0] = -1.0F;
    key_value[kHiddenWidth] = 2.0F;
    const MiniMessageBank messages(states, attention_moe, key_value);
    constexpr auto visits = mini_schedule();
    constexpr auto repeat = visits[kCoreFirst + MiniLayers::core_width];
    const float m0 = messages.message(repeat, MessagePath::query)[0];
    const float m1 = messages.message(repeat, MessagePath::key)[0];
    const float m2 = messages.message(repeat, MessagePath::value)[0];
    const float m3 = messages.message(repeat, MessagePath::moe)[0];
    const float next = messages.message(visits[repeat.logical_layer + 1U], MessagePath::query)[0];
    if (!(m3 < m1 && m1 < next && next < m0 && m0 < m2)) {
        throw std::runtime_error("unexpected Inkling 2 Mini checkpoint message query order");
    }
    rejects([&] { (void)messages.message(visits[kCoreFirst], MessagePath::query); });
}

} // namespace

int main() {
    try {
        test_message_attention();
        test_low_rank_gate();
        test_mini_checkpoint_query_order();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
