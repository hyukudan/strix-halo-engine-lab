// Host checks of the A4W4 projection-call layout (core/iu4a4_projection_layout.hpp): admission limits, segment
// boundaries on whole 16-column tiles, the owner of every column, the image byte sizes, and call matching against the
// image seal (pitch, input width, rounding policy, weight rule).
#include "core/iu4a4_projection_layout.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

namespace {
namespace ip = strix::core::iu4a4_projections;
int fails = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
}  // namespace

int main() {
    // Inkling 2 Mini attention call: Q 3,072 + relative 384 + gate 3,072 + K 1,024 + V 1,024 from 2,304 inputs.
    static constexpr std::array<unsigned, 5> attention{3072U, 384U, 3072U, 1024U, 1024U};
    constexpr ip::Layout a = ip::make_layout(2304U, attention);
    static_assert(a.admitted() && a.count == 5U && a.n == 8576U && a.in == 2304U);
    static_assert(a.first_col[0] == 0U && a.first_col[1] == 3072U && a.first_col[2] == 3456U && a.first_col[3] == 6528U &&
                  a.first_col[4] == 7552U && a.first_col[5] == 8576U && a.first_col[8] == 8576U);
    static_assert(a.segment(0U) == 0U && a.segment(3071U) == 0U && a.segment(3072U) == 1U && a.segment(3455U) == 1U &&
                  a.segment(3456U) == 2U && a.segment(8575U) == 4U);
    static_assert(a.weight_code_bytes() == std::size_t{8576U} * 2304U / 2U && a.weight_step_bytes() == std::size_t{8576U} * 18U * 4U);
    // Every column's owner matches the segment ranges, and every 16-column tile lies inside one segment.
    for (unsigned c = 0; c < a.n; ++c) {
        const unsigned s = a.segment(c);
        expect(c >= a.first_col[s] && c < a.first_col[s + 1U], "owner range");
        if (c % 16U == 0U) expect(a.segment(c + 15U) == s, "tile inside one segment");
    }
    for (unsigned s = 0; s < a.count; ++s) expect(a.outputs(s) == attention[s], "segment outputs");
    // Refusals: no segment, too many, a segment off the 16-column grid or empty, inputs off the 128 block or too long.
    const std::vector<unsigned> none;
    const std::vector<unsigned> nine(9U, 16U), eight(8U, 16U);
    const std::array<unsigned, 2> odd{3072U, 24U}, empty{16U, 0U};
    static constexpr std::array<unsigned, 1> one{2304U};
    expect(!ip::make_layout(2304U, none).admitted(), "no segment refused");
    expect(!ip::make_layout(2304U, nine).admitted(), "nine segments refused");
    expect(ip::make_layout(2304U, eight).admitted() && ip::make_layout(2304U, eight).n == 128U, "eight segments admitted");
    expect(!ip::make_layout(2304U, odd).admitted(), "24-output segment refused");
    expect(!ip::make_layout(2304U, empty).admitted(), "empty segment refused");
    expect(!ip::make_layout(2300U, one).admitted(), "inputs off the block refused");
    expect(!ip::make_layout(0U, one).admitted(), "zero inputs refused");
    expect(!ip::make_layout(ip::kMaxInputs + 128U, one).admitted(), "inputs past the contract refused");
    expect(ip::make_layout(ip::kMaxInputs, one).admitted(), "inputs at the contract limit admitted");
    const std::array<unsigned, 2> overflow{0xFFFFFFF0U, 32U};
    expect(!ip::make_layout(128U, overflow).admitted(), "column total overflow refused");
    // The output call: O 3,072 -> 2,304, one segment.
    constexpr ip::Layout o = ip::make_layout(3072U, one);
    static_assert(o.admitted() && o.count == 1U && o.n == 2304U && o.segment(2303U) == 0U);
    // Call matching: image 0 = [Q, rel, gate, K, V] (sink not in it, natural pitch, incumbent rounding), image 1 = [O]
    // (weights at a pitch of 3,200 BF16, no rounding); both rule 1 (v1).
    struct P {
        const void* weights;
        unsigned outputs;
        bool round_bf16;
        std::size_t row_bytes;
        bool round_separate;
        constexpr P(const void* weights_, unsigned outputs_, bool round_, std::size_t pitch_ = 0U, bool separate_ = false)
            : weights(weights_), outputs(outputs_), round_bf16(round_), row_bytes(pitch_), round_separate(separate_) {}
    };
    static const int q = 0, rel = 0, sink = 0, gate = 0, k = 0, v = 0, out = 0, other = 0;
    const std::array<ip::SegmentKey, 6> keys{{{&q, 0U, 0U}, {&rel, 0U, 1U}, {&gate, 0U, 2U}, {&k, 0U, 3U}, {&v, 0U, 4U}, {&out, 1U, 0U}}};
    const std::array<ip::Layout, 2> layouts{a, o};
    ip::Seal sa{}, so{};
    sa.in = 2304U;
    sa.round = ip::RoundPolicy::incumbent;
    sa.rule = 1U;
    for (unsigned s = 0; s < a.count; ++s) sa.row_bytes[s] = 2304U * 2U;
    so.in = 3072U;
    so.round = ip::RoundPolicy::none;
    so.rule = 1U;
    so.row_bytes[0] = 3200U * 2U;
    const std::array<ip::Seal, 2> seals{sa, so};
    unsigned index[ip::kMaxSegments]{};
    const auto match = [&](unsigned width, unsigned rule, const auto& call) {
        return ip::match_call<P>(keys, layouts, seals, width, rule, call, index);
    };
    const std::array<P, 6> call_a{{{&q, 3072U, true}, {&rel, 384U, true}, {&sink, 24U, true}, {&gate, 3072U, true},
                                   {&k, 1024U, true}, {&v, 1024U, true}}};
    expect(match(2304U, 1U, call_a) == 0U && index[0] == 0U && index[1] == 1U && index[2] == 3U && index[3] == 4U &&
               index[4] == 5U, "attention call matches image 0, sink left out");
    const std::array<P, 6> call_a_pitch{{{&q, 3072U, true, 4608U}, {&rel, 384U, true}, {&sink, 24U, true}, {&gate, 3072U, true},
                                         {&k, 1024U, true, 4608U}, {&v, 1024U, true}}};
    expect(match(2304U, 1U, call_a_pitch) == 0U, "an explicit pitch equal to the sealed natural pitch matches");
    const std::array<P, 1> call_o{{{&out, 2304U, false, 6400U}}};
    expect(match(3072U, 1U, call_o) == 1U && index[0] == 0U, "output call at the sealed pitch matches image 1");
    // Sealed fields: pitch, input width, rounding policy, weight rule. Any difference refuses the image.
    const std::array<P, 1> o_natural{{{&out, 2304U, false}}};
    expect(match(3072U, 1U, o_natural) == ip::kNoImage, "same weights + outputs at the natural pitch (image at 3,200) refused");
    const std::array<P, 1> o_other_pitch{{{&out, 2304U, false, 6656U}}};
    expect(match(3072U, 1U, o_other_pitch) == ip::kNoImage, "same weights + outputs at another pitch refused");
    const std::array<P, 5> a_pitched{{{&q, 3072U, true}, {&rel, 384U, true}, {&gate, 3072U, true}, {&k, 1024U, true, 4864U},
                                      {&v, 1024U, true}}};
    expect(match(2304U, 1U, a_pitched) == ip::kNoImage, "one segment at a pitch other than the sealed one refused");
    expect(match(2560U, 1U, call_a) == ip::kNoImage, "input width other than the sealed one refused");
    expect(match(3072U, 2U, call_o) == ip::kNoImage, "weight rule other than the sealed one refused");
    const std::array<P, 1> o_rounded{{{&out, 2304U, true, 6400U}}};
    expect(match(3072U, 1U, o_rounded) == ip::kNoImage, "rounded call on an image sealed for FP32 stores refused");
    const std::array<P, 5> a_plain{{{&q, 3072U, false}, {&rel, 384U, false}, {&gate, 3072U, false}, {&k, 1024U, false},
                                    {&v, 1024U, false}}};
    expect(match(2304U, 1U, a_plain) == ip::kNoImage, "FP32 call on an image sealed for incumbent rounding refused");
    const std::array<P, 5> a_separate{{{&q, 3072U, true, 0U, true}, {&rel, 384U, true, 0U, true}, {&gate, 3072U, true, 0U, true},
                                       {&k, 1024U, true, 0U, true}, {&v, 1024U, true, 0U, true}}};
    expect(match(2304U, 1U, a_separate) == ip::kNoImage, "separate-round call on an incumbent-round image refused");
    const std::array<P, 1> o_bad_policy{{{&out, 2304U, false, 6400U, true}}};
    expect(match(3072U, 1U, o_bad_policy) == ip::kNoImage, "round_separate without round_bf16 refused");
    const std::span<const ip::Seal> first_seal(seals.data(), 1U);
    expect(ip::match_call<P>(keys, layouts, first_seal, 3072U, 1U, call_o, index) == ip::kNoImage, "an image without a seal refused");
    ip::RoundPolicy policy = ip::RoundPolicy::none;
    expect(ip::round_policy(true, true, policy) && policy == ip::RoundPolicy::separate, "policy separate");
    expect(ip::round_policy(true, false, policy) && policy == ip::RoundPolicy::incumbent, "policy incumbent");
    expect(ip::round_policy(false, false, policy) && policy == ip::RoundPolicy::none, "policy none");
    expect(!ip::round_policy(false, true, policy), "round_separate alone is not a policy");
    const std::array<P, 4> q_only{{{&q, 3072U, true}, {&rel, 384U, true}, {&sink, 24U, true}, {&gate, 3072U, true}}};
    expect(match(2304U, 1U, q_only) == ip::kNoImage, "a call missing segments K, V keeps its route");
    const std::array<P, 6> swapped{{{&q, 3072U, true}, {&rel, 384U, true}, {&gate, 3072U, true}, {&v, 1024U, true},
                                    {&k, 1024U, true}, {&sink, 24U, true}}};
    expect(match(2304U, 1U, swapped) == ip::kNoImage, "segments out of order refused");
    const std::array<P, 5> mixed{{{&q, 3072U, true}, {&rel, 384U, false}, {&gate, 3072U, true}, {&k, 1024U, true}, {&v, 1024U, true}}};
    expect(match(2304U, 1U, mixed) == ip::kNoImage, "mixed rounding classes refused");
    const std::array<P, 5> narrow{{{&q, 2048U, true}, {&rel, 384U, true}, {&gate, 3072U, true}, {&k, 1024U, true}, {&v, 1024U, true}}};
    expect(match(2304U, 1U, narrow) == ip::kNoImage, "outputs other than the segment's refused");
    const std::array<P, 2> unknown{{{&other, 2304U, false}, {nullptr, 16U, false}}};
    expect(match(3072U, 1U, unknown) == ip::kNoImage, "unknown weights keep their route");
    std::printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
