// Host test of the R1 down-combine tails: a fused tail equals the nodes it replaces, bitwise, and the plan refuses
// illegal tails. The tail is driven exactly as the kernel drives it: thread 0 stages one CTA's DownColumns channels,
// then finishes each live channel with the stored combined value.

#include "moe_core/microscaled_r1.hpp"
#include "moe_core/r1_tail.hpp"
#include "strix/attn/group_indexer.hpp"

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
// The residual add node rounds with residual_mix's AddRound, the group_indexer BF16 round (keeps the NaN payload).
float add_node_round(float x) { return strix::attn::group_indexer::bf16_round(x); }
std::uint32_t bits(float x) { return std::bit_cast<std::uint32_t>(x); }
float from_bits(std::uint32_t u) { return std::bit_cast<float>(u); }

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
            sum_ref[c] = round_sum ? add_node_round(added) : added;
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
            const float want =
                c < hidden ? add_node_round(before[row * stride + c] + moe[row][c]) : before[row * stride + c];
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

// Finding: the overlap rule compared base pointers only. A tail store one element into the MoE output (or any partial
// overlap between tail buffers) passed and raced the CTAs that own the neighbouring channels.
void partial_overlap() {
    std::vector<float> w(4 * 64), h(3 * 64), conv(64), res(2 * 64), sum(64), out(3 * 64), in(2 * 64);
    std::vector<std::uint16_t> act(2 * 8 * 32);
    r1::Args args{};
    args.hidden = 64;
    args.intermediate = 32;
    args.output = out.data();
    args.input = in.data();
    args.activation = act.data();
    const tail::ResidualTail into_output{res.data(), out.data() + 1, 64U, true};
    check(tail::admits(into_output, 1U, 64U), "residual tail alone is legal");
    check(!r1::admits_tail(args, into_output, 1U), "admits_tail refuses a sum one element into the MoE output");
    check(!r1::admits_tail(args, tail::ResidualTail{res.data(), out.data() + 64 + 63, 64U, true}, 2U),
          "admits_tail refuses a sum overlapping the last element of MoE output row 1");
    check(r1::admits_tail(args, tail::ResidualTail{res.data(), out.data() + 64, 64U, true}, 1U),
          "a sum just past the one-row MoE output is legal");
    check(!r1::admits_tail(args, tail::ResidualTail{res.data(), in.data() + 3, 64U, true}, 1U),
          "admits_tail refuses a sum inside the input rows");
    float* const in_activation = reinterpret_cast<float*>(act.data() + 40);
    check(r1::admits_tail(args, tail::ResidualTail{res.data(), in_activation, 64U, true}, 1U),
          "activation unknown without top_k");
    check(!r1::admits_tail(args, tail::ResidualTail{res.data(), in_activation, 64U, true}, 1U, 8U),
          "admits_tail refuses a sum inside the activation rows (top_k known)");
    check(!tail::admits(tail::ResidualTail{res.data(), res.data() + 1, 63U, true}, 1U, 63U),
          "residual tail refuses a sum shifted by one over its residual");
    check(tail::admits(tail::ResidualTail{res.data(), res.data(), 64U, true}, 1U, 64U), "exact in-place sum is legal");

    const strix::core::ShortConvolution plan{64, 4, 0U, 0.0F, 2.0F};
    const tail::ConvResidualTail<4> good{plan, w.data(), h.data(), conv.data(), res.data(), sum.data(), true, true};
    auto conv_shift = good;
    conv_shift.conv_output = h.data() + 2 * 64 + 5;
    check(!tail::admits(conv_shift, 1U, 64U), "conv tail refuses conv output overlapping the history");
    auto weights_shift = good;
    weights_shift.weights = h.data() + 1;
    check(!tail::admits(weights_shift, 1U, 64U), "conv tail refuses weights overlapping the history");
    auto sum_shift = good;
    sum_shift.sum = conv.data() + 7;
    check(!tail::admits(sum_shift, 1U, 64U), "conv tail refuses a sum overlapping the conv output");
    auto conv_into_output = good;
    conv_into_output.conv_output = out.data() + 3;
    check(!r1::admits_tail(args, conv_into_output, 1U), "admits_tail refuses conv output inside the MoE output");
    auto history_into_output = good;
    history_into_output.history = out.data() + 1;
    check(!r1::admits_tail(args, history_into_output, 1U), "admits_tail refuses history inside the MoE output");
    auto in_place = good;
    in_place.sum = res.data();
    check(tail::admits(in_place, 1U, 64U), "conv tail: exact in-place residual sum is legal");
}

// Finding: the tail stages its inputs before the down-combine store. A residual that is (or overlaps) the MoE output, or
// a conv-tail residual that is the conv output, read stale values where the separate nodes read the stored row. The
// driver runs one CTA as the kernel does (stage, store, finish) to show the hazard the plan now refuses.
void stale_read() {
    constexpr unsigned hidden = 8;
    std::vector<float> out(hidden, 11.0F), moe(hidden, 0.0F), res(hidden), sum(hidden);
    const tail::ResidualTail aliased{out.data(), sum.data(), hidden, false};
    const auto staged = aliased.stage<hidden>(0U, 0U, hidden);
    for (unsigned c = 0; c < hidden; ++c) out[c] = moe[c];
    for (unsigned c = 0; c < hidden; ++c) aliased.finish<hidden>(staged, c, 0U, c, moe[c]);
    check(sum[0] == 11.0F && out[0] + moe[0] == 0.0F, "fused driver reads the stale output, the add the stored one");
    r1::Args args{};
    args.hidden = hidden;
    args.output = out.data();
    check(!r1::admits_tail(args, aliased, 1U), "admits_tail refuses a residual that is the MoE output");
    check(!r1::admits_tail(args, tail::ResidualTail{out.data() + 2, sum.data(), hidden, false}, 1U),
          "admits_tail refuses a residual overlapping the MoE output");

    std::vector<float> w(4 * hidden), h(3 * hidden), conv(hidden);
    const strix::core::ShortConvolution plan{hidden, 4, 0U, 1.0F, 2.0F};
    tail::ConvResidualTail<4> t{plan, w.data(), h.data(), conv.data(), conv.data(), sum.data(), true, true};
    check(!tail::admits(t, 1U, hidden), "conv tail refuses a residual that is the conv output");
    t.residual = conv.data() + 1;
    check(!tail::admits(t, 1U, hidden), "conv tail refuses a residual overlapping the conv output");
    t.residual = h.data() + hidden;
    check(!tail::admits(t, 1U, hidden), "conv tail refuses a residual inside the history");
    t.residual = res.data();
    check(tail::admits(t, 1U, hidden), "conv tail with a separate residual is legal");
    t.residual = out.data();
    check(tail::admits(t, 1U, hidden), "conv tail alone does not see the MoE output");
    check(!r1::admits_tail(args, t, 1U), "admits_tail refuses a conv-tail residual that is the MoE output");
}

// Finding: the tail rounded the residual sum with the weight-format BF16 encoder, whose NaN result (0x7fff0000) differs
// from the residual add node's (payload kept, 0x7fffffff). The device add of a NaN gives the canonical 0x7fffffff.
void nan_payload() {
    constexpr unsigned hidden = 4;
    const float nan = from_bits(0x7fffffffU);
    std::vector<float> res{nan, 1.0F, 2.0F, 3.0F}, sum(hidden), moe{0.5F, 0.25F, 0.0F, -1.0F};
    const tail::ResidualTail t{res.data(), sum.data(), hidden, true};
    run_tail<4>(t, 0U, moe, hidden);
    check(bits(sum[0]) == 0x7fffffffU, "residual tail NaN payload equals the add node (0x7fffffff)");
    check(bits(tail::residual_add(nan, 1.0F, true)) == 0x7fffffffU, "residual_add keeps the NaN payload");
    check(bits(tail::residual_add(nan, 1.0F, false)) == 0x7fffffffU, "unrounded residual_add keeps the NaN");

    std::vector<float> w(4 * hidden, 0.5F), h(3 * hidden, 0.0F), conv(hidden), csum(hidden);
    const strix::core::ShortConvolution plan{hidden, 4, 0U, 1.0F, 2.0F};
    const tail::ConvResidualTail<4> ct{plan, w.data(), h.data(), conv.data(), res.data(), csum.data(), true, true};
    run_tail<4>(ct, 0U, moe, hidden);
    check(bits(csum[0]) == 0x7fffffffU, "conv tail residual-sum NaN payload equals the add node");
}

// Finding: ResidualTail admitted any stride >= hidden; stride SIZE_MAX with two rows wrapped row * stride.
void stride_overflow() {
    std::vector<float> res(64), sum(64);
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    check(!tail::admits(tail::ResidualTail{res.data(), sum.data(), max, false}, 2U, 64U),
          "residual tail refuses stride SIZE_MAX with two rows");
    check(!tail::admits(tail::ResidualTail{res.data(), sum.data(), max / sizeof(float), false}, 2U, 64U),
          "residual tail refuses a byte range that overflows");
    check(!tail::admits(tail::ResidualTail{res.data(), sum.data(), max / 8U, false}, 3U, 64U),
          "residual tail refuses a range past the address space");
    check(tail::admits(tail::ResidualTail{res.data(), sum.data(), max, false}, 1U, 64U),
          "one row never multiplies the stride");
    float* const high = reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max() - 64U);
    check(!tail::admits(tail::ResidualTail{res.data(), high, 64U, false}, 1U, 64U),
          "residual tail refuses a sum range that wraps the address space");
    const strix::core::ShortConvolution wide{max / 2U, 4, 0U, 0.0F, 2.0F};
    const tail::ConvResidualTail<4> huge{wide, res.data(), sum.data(), nullptr, nullptr, nullptr, true, true};
    check(tail::footprint(huge, 1U, 0U).writes[0].valid == false, "conv footprint flags taps * channels overflow");
}

// Finding: the conv tail copied the shared convolution arithmetic. It now calls core::causal_conv's per-channel
// helpers; pin those helpers to the host reference of the convolution node.
void shared_conv_helpers() {
    std::mt19937 rng(11);
    constexpr std::size_t channels = 5, taps = 4;
    const strix::core::ShortConvolution plan{channels, taps, 0U, 0.75F, 3.0F};
    const auto weights = bf16_values(rng, taps * channels, 0.5F);
    const auto history = bf16_values(rng, (taps - 1) * channels, 1.0F);
    const auto input = bf16_values(rng, channels, 1.0F);
    auto history_ref = history;
    std::vector<float> ref(channels);
    strix::core::causal_short_convolution(plan, input, weights, {}, history_ref, ref);
    namespace conv = strix::core::causal_conv;
    std::vector<float> got(channels), history_got = history;
    for (std::size_t c = 0; c < channels; ++c) {
        const float taps_sum = conv::weighted_taps<true>(
            taps, [&](std::size_t lag) { return lag == 0 ? input[c] : history[(taps - 1 - lag) * channels + c]; },
            [&](std::size_t lag) { return weights[lag * channels + c]; });
        got[c] = conv::epilogue(plan, taps_sum, input[c]);
        conv::commit_history(taps - 1, std::size_t{1}, channels, c, history.data(), input.data(), history_got.data());
    }
    check(same_bits(got, ref), "causal_conv helpers equal the convolution reference");
    check(same_bits(history_got, history_ref), "causal_conv commit_history equals the reference history");
    check(bits(conv::round_output(from_bits(0x7fffffffU))) == bits(bf16_round(from_bits(0x7fffffffU))),
          "conv output rounding is the convolution node's BF16 encoder");
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
    partial_overlap();
    stale_read();
    nan_payload();
    stride_overflow();
    shared_conv_helpers();
    if (failures != 0) {
        std::fprintf(stderr, "r1_tail_test: %d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("r1_tail_test: PASS\n");
    return EXIT_SUCCESS;
}
