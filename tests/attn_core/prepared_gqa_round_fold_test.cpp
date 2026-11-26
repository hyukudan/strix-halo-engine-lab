// The opt-in round fold of the shared prepared GQA stage (attn_core::detail::PreparedGqaFoldsRounds): a folding
// backend receives no plain round of raw K/V, the dynamic taps, the value rows or the output projection; those rounds
// move into convolve<RoundOutput, RoundInput> and the projection store, and failure-checking rounds stay. A backend
// without the opt-in sees exactly the incumbent sequence.
#include "attn_core/prepared_gqa.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
namespace ac = strix::attn_core;
using strix::core::Bf16RowProjection;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Shape {
    static constexpr unsigned kMaxContext = 64, kHiddenWidth = 32, kQueryWidth = 32, kHeadWidth = 16, kKvWidth = 16,
                              kQueryHeads = 2, kKvHeads = 1, kConvTaps = 3, kRelativeWidth = 16;
    static constexpr float kQkClip = 1, kRelativeClip = 1, kFullConvDivisor = 1, kLocalConvDivisor = 1;
    static unsigned projection_width(ac::PreparedGqaProjection role) {
        return role == ac::PreparedGqaProjection::sink ? 2U : 16U;
    }
};

struct Conv {
    const float* input;
    const float* dynamic;
    float* output;
    bool round_output, round_input;
};

template <bool Fold> struct Probe {
    using Status = bool;
    static constexpr bool success = true, invalid = false;
    static constexpr bool kFoldRounds = Fold;
    bool admit_store_round = true;
    bool fold_on = true;
    mutable std::vector<const float*> rounds;
    mutable std::vector<bool> checked;
    mutable std::vector<Conv> convs;
    mutable std::vector<Bf16RowProjection> projections;
    struct Store {
        const float* input;
        float* output;
        bool round, input_rounded;  // input_rounded: a plain round of `input` was launched before the store
    };
    mutable std::vector<Store> stores;

    bool fold_rounds() const { return fold_on; }
    bool store_round(unsigned rows) const { return admit_store_round && rows <= 16U; }
    bool linear(const float*, unsigned, unsigned, std::span<const Bf16RowProjection> views) const {
        projections.insert(projections.end(), views.begin(), views.end());
        return true;
    }
    bool round(float* p, std::size_t, unsigned* failure = nullptr) const {
        rounds.push_back(p);
        checked.push_back(failure != nullptr);
        return true;
    }
    bool store_rows(const float* input, std::size_t, float* output, bool round) const {
        stores.push_back({input, output, round, rounded(input)});
        return true;
    }
    template <bool RoundOutput = false, bool RoundInput = false>
    bool convolve(const strix::core::ShortConvolution&, unsigned, const float* input, const float*,
                  const float* dynamic, float*, float* output) const {
        convs.push_back({input, dynamic, output, RoundOutput, RoundInput});
        return true;
    }
    template <class... T> bool clip(T&&...) const { return true; }
    template <class... T> bool rms_values(T&&...) const { return true; }
    template <class... T> bool relative_logits(T&&...) const { return true; }
    template <class... T> bool gate(T&&...) const { return true; }

    bool rounded(const float* p) const { return std::ranges::find(rounds, p) != rounds.end(); }
};

struct Buffers {
    alignas(16) float a[24][64]{};
    alignas(16) std::uint16_t w[32 * 32]{};
    ac::PreparedGqaWeights weights{};
    ac::PreparedGqaInputs in{};
    ac::PreparedGqaWorkspace work{};
    ac::PreparedGqaRows out{};
    Buffers() {
        weights.matrix.fill(w);
        weights.query_gain = weights.relative_gain = w;
        weights.key_convolution = weights.value_convolution = a[0];
        in = {a[0], a[0], a[0]};
        work = {a[3], a[4], a[5], a[6], a[7], a[8], a[9]};
        out = {a[10], a[11], a[12], a[13], a[14], a[15]};
    }
};

template <bool Fold> void run(bool full, bool admit, unsigned rows, bool on = true, bool tape = false) {
    Buffers b;
    if (tape) {
        b.work.key_tape = b.a[19];
        b.work.value_tape = b.a[20];
    }
    Probe<Fold> probe;
    probe.admit_store_round = admit;
    probe.fold_on = on;
    const bool folded = Fold && on;
    unsigned failure{};
    float* attended = b.a[16];
    float* gated = b.a[17];
    float* output = b.a[18];
    const ac::PreparedGqaOutputRows output_rows{attended, gated, output};
    check(ac::enqueue_prepared_gqa_forward<Shape>(probe, full, 0, rows, b.in, b.weights, b.work, b.out, output_rows, &failure,
              [] { return true; }), "forward failed");
    const auto& w = b.work;
    check(probe.convs.size() == 2U && probe.convs[0].input == w.raw_keys && probe.convs[1].input == w.raw_values,
          "K then V convolution");
    check(probe.convs[0].output == b.out.key && probe.convs[1].output == b.out.value, "convolution outputs");
    check(probe.convs[0].dynamic == (full ? w.dynamic_keys : nullptr) &&
          probe.convs[1].dynamic == (full ? w.dynamic_values : nullptr), "dynamic taps");
    check(!probe.convs[0].round_output, "key output is clipped, never rounded in the store");
    check(probe.convs[0].round_input == folded && probe.convs[1].round_input == folded, "input rounds fold into the conv");
    check(probe.convs[1].round_output == folded, "value round folds into the conv store");
    for (const float* p : {w.raw_keys, w.raw_values, b.out.value})
        check(probe.rounded(p) != folded, "plain round present iff the backend does not fold");
    if (full)
        for (const float* p : {w.dynamic_keys, w.dynamic_values})
            check(probe.rounded(p) != folded, "dynamic-tap round present iff the backend does not fold");
    // The sink round checks a failure word: it is never folded.
    const auto sink = std::ranges::find(probe.rounds, static_cast<const float*>(b.out.sinks));
    check(sink != probe.rounds.end() && probe.checked[static_cast<std::size_t>(sink - probe.rounds.begin())],
          "failure-checking sink round kept");
    const bool store = folded && admit && rows <= 16U;
    check(!probe.projections.empty() && probe.projections.back().output == output, "output projection last");
    check(probe.projections.back().round_bf16 == store, "output projection rounds in the store iff admitted");
    check(probe.projections.back().round_separate == store, "the folded store is the separate round's policy");
    check(probe.rounded(output) != store, "output round launched iff not stored rounded");
    for (std::size_t i = 0; i + 1 < probe.projections.size(); ++i)
        check(!probe.projections[i].round_bf16, "input projections keep their stores");
    std::size_t plain = 0;
    for (std::size_t i = 0; i < probe.rounds.size(); ++i) plain += probe.checked[i] ? 0U : 1U;
    const std::size_t incumbent = 4U + (full ? 2U : 0U);  // raw K, raw V, value, output (+ dynamic K, V)
    check(plain == (folded ? (store ? 0U : 1U) : incumbent), "plain round count");
    // Tape: the rows each history commit stores. A folding convolution rounds its reads (store rounds them); the
    // incumbent sequence rounds the activation first (store copies it). Without a tape nothing is stored.
    if (!tape) {
        check(probe.stores.empty(), "no tape, no store");
        return;
    }
    check(probe.stores.size() == 2U, "one tape store per key/value convolution");
    const float* inputs[2]{w.raw_keys, w.raw_values};
    const float* tapes[2]{w.key_tape, w.value_tape};
    for (std::size_t i = 0; i < 2U; ++i) {
        const auto& st = probe.stores[i];
        check(st.input == inputs[i] && st.output == tapes[i], "tape of the convolution input");
        check(st.round == folded, "tape rounds iff the convolution rounds its reads");
        check(st.input_rounded == !folded, "unfolded tape copies the already rounded rows");
    }
}
}  // namespace

int main() {
    try {
        for (const bool full : {false, true}) {
            for (const unsigned rows : {1U, 16U, 17U}) {
                run<false>(full, true, rows);
                run<true>(full, true, rows);
                run<true>(full, false, rows);
                run<true>(full, true, rows, false);  // off switch: the incumbent sequence
                run<false>(full, true, rows, true, true);
                run<true>(full, true, rows, true, true);
                run<true>(full, true, rows, false, true);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL prepared_gqa_round_fold: %s\n", e.what());
        return 1;
    }
    std::puts("prepared_gqa_round_fold: PASS");
    return 0;
}
