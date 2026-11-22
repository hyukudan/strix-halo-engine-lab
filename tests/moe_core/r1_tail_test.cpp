// Host test of the R1 down-combine tails: a fused tail equals the nodes it replaces, bitwise, and the plan refuses
// illegal tails. The tail is driven exactly as the kernel drives it: thread 0 stages one CTA's DownColumns channels,
// then finishes each live channel with the stored combined value.

#include "moe_core/microscaled_r1.hpp"
#include "moe_core/r1_tail.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {

namespace r1 = strix::moe_core::microscaled_r1;
namespace tail = strix::moe_core::r1_tail;
using strix::weight::scalar_float::bf16_round;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool same_bits(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) return false;
    return true;
}

std::vector<float> bf16_values(std::mt19937& rng, std::size_t count, float scale) {
    std::normal_distribution<float> dist(0.0F, scale);
    std::vector<float> values(count);
    for (auto& v : values) v = bf16_round(dist(rng));
    return values;
}

// Drives a tail over all channels like the kernel: CTA b owns channels [b * Columns, b * Columns + live).
template <unsigned Columns, class Tail>
void run_tail(const Tail& t, unsigned row, std::span<const float> moe, std::size_t channels) {
    for (std::size_t first = 0; first < channels; first += Columns) {
        const auto live = static_cast<unsigned>(channels - first);
        const auto staged = t.template stage<Columns>(row, static_cast<unsigned>(first), live);
        for (unsigned i = 0; i < Columns && i < live; ++i)
            t.template finish<Columns>(staged, i, row, static_cast<unsigned>(first + i), moe[first + i]);
    }
}

// Incumbent chain: causal_conv decode node (round) then residual add node (round), as Inkling's MoE tail.
template <unsigned Columns, unsigned Taps>
void conv_residual_case(std::size_t channels, std::size_t taps, float residual_weight, bool round_conv, bool round_sum,
                        bool with_residual, unsigned seed) {
    std::mt19937 rng(seed);
    const strix::core::ShortConvolution plan{channels, taps, 0U, residual_weight, 2.2360679774997898F};
    const auto weights = bf16_values(rng, taps * channels, 0.5F);
    std::vector<float> history_ref((taps - 1) * channels, 0.0F), history_fused = history_ref;
    for (unsigned step = 0; step < 6; ++step) {
        auto moe = bf16_values(rng, channels, 1.0F);
        const auto residual = bf16_values(rng, channels, 2.0F);
        if (step == 2) {  // signed zeros and a large magnitude stay on the identical path
            moe[0] = -0.0F;
            moe[1] = 0.0F;
            moe[channels - 1] = 3.0e38F;
        }
        std::vector<float> conv_ref(channels), sum_ref(channels);
        strix::core::causal_short_convolution(plan, moe, weights, {}, history_ref, conv_ref);
        for (std::size_t c = 0; c < channels; ++c) {
            if (round_conv) conv_ref[c] = bf16_round(conv_ref[c]);
            const float added = residual[c] + conv_ref[c];
            sum_ref[c] = round_sum ? bf16_round(added) : added;
        }
        std::vector<float> conv_fused(channels, -1.0F), sum_fused(channels, -1.0F);
        const tail::ConvResidualTail<Taps> t{plan,
                                                weights.data(),
                                                history_fused.data(),
                                                conv_fused.data(),
                                                with_residual ? residual.data() : nullptr,
                                                with_residual ? sum_fused.data() : nullptr,
                                                round_conv,
                                                round_sum};
        check(tail::admits(t, 1U, static_cast<unsigned>(channels)), "conv tail admitted");
        run_tail<Columns>(t, 0U, moe, channels);
        check(same_bits(conv_fused, conv_ref), "conv output bits");
        check(same_bits(history_fused, history_ref), "history bits");
        if (with_residual) check(same_bits(sum_fused, sum_ref), "residual sum bits");
    }
}

void residual_case() {
    std::mt19937 rng(7);
    constexpr unsigned rows = 3, hidden = 70;
    constexpr std::size_t stride = hidden + 3;
    auto stream = bf16_values(rng, rows * stride, 2.0F);
    const auto before = stream;
    std::vector<std::vector<float>> moe(rows);
    for (auto& m : moe) m = bf16_values(rng, hidden, 1.0F);
    const tail::ResidualTail t{stream.data(), stream.data(), stride, true};  // in place, like a residual stream
    check(tail::admits(t, rows, hidden), "residual tail admitted at rows 3");
    for (unsigned row = 0; row < rows; ++row) run_tail<4>(t, row, moe[row], hidden);
    bool ok = true;
    for (unsigned row = 0; row < rows; ++row)
        for (std::size_t c = 0; c < stride; ++c) {
            const float want = c < hidden ? bf16_round(before[row * stride + c] + moe[row][c]) : before[row * stride + c];
            ok = ok && std::bit_cast<std::uint32_t>(want) == std::bit_cast<std::uint32_t>(stream[row * stride + c]);
        }
    check(ok, "residual tail bits (pad untouched)");
}

void legality() {
    std::vector<float> w(4 * 64), h(3 * 64), conv(64), res(64), sum(64), out(64);
    const strix::core::ShortConvolution plan{64, 4, 0U, 0.0F, 2.0F};
    const tail::ConvResidualTail<4> good{plan, w.data(), h.data(), conv.data(), res.data(), sum.data(), true, true};
    check(tail::admits(good, 1U, 64U), "legal conv tail");
    check(!tail::admits(good, 2U, 64U), "conv tail refuses a rows launch");
    check(!tail::admits(good, 1U, 128U), "conv tail refuses channels != hidden");
    auto many = good;
    many.plan.taps = 5;
    check(!tail::admits(many, 1U, 64U), "conv tail refuses taps != Taps");
    auto dynamic = good;
    dynamic.plan.dynamic_groups = 2;
    check(!tail::admits(dynamic, 1U, 64U), "conv tail refuses dynamic weights");
    auto alias = good;
    alias.sum = h.data();
    check(!tail::admits(alias, 1U, 64U), "conv tail refuses history aliasing the sum");
    auto empty = good;
    empty.conv_output = nullptr;
    empty.residual = nullptr;
    check(!tail::admits(empty, 1U, 64U), "conv tail refuses a tail with no store");
    check(!tail::admits(tail::ResidualTail{res.data(), sum.data(), 63U, false}, 1U, 64U), "residual stride < hidden");

    r1::Args args{};
    args.hidden = 64;
    args.output = out.data();
    check(r1::admits_tail(args, good, 1U), "admits_tail legal");
    auto into_output = good;
    into_output.conv_output = out.data();
    check(!r1::admits_tail(args, into_output, 1U), "admits_tail refuses conv over the stored MoE output");
    check(!r1::admits_tail(args, tail::ResidualTail{res.data(), out.data(), 64U, true}, 1U),
          "admits_tail refuses a sum over the stored MoE output");
    check(r1::admits_tail(args, tail::NoTail{}, 8U), "NoTail always admitted");
}

}  // namespace

int main() {
    // Inkling shape class: 4 taps, divisor sqrt(5), rounded conv and sum; live < Columns on the last CTA (66 % 4).
    conv_residual_case<4, 4>(66, 4, 0.0F, true, true, true, 1U);
    conv_residual_case<2, 4>(2304, 4, 0.0F, true, true, true, 2U);
    conv_residual_case<4, 4>(66, 4, 0.25F, false, false, true, 3U);  // residual weight and unrounded stores
    conv_residual_case<4, 3>(66, 3, 0.5F, true, true, false, 4U);    // three taps, conv output only
    conv_residual_case<1, 1>(9, 1, 1.0F, true, true, true, 5U);      // one tap: no history
    residual_case();
    legality();
    if (failures != 0) {
        std::fprintf(stderr, "r1_tail_test: %d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("r1_tail_test: PASS\n");
    return EXIT_SUCCESS;
}
