// Host test of the shared DFlash row functors (src/spec/dflash_rows.hpp), the bodies the device forward launches:
// the staged top-k equals the host reference's canonical selection (value descending, then the lower id; -0 == +0;
// non-finite logits raise the failure word and never enter), the grouped convolution equals its formula, and the
// selector's edge scores + greedy path follow the chosen predecessor (anchor at step 0, softcap after the correction).

#include "spec/dflash_rows.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

using namespace strix::spec::dflash::row_ops;

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

template <class F>
void run(std::size_t count, F f) {
    for (std::size_t i = 0; i < count; ++i) f(i);
}

std::uint16_t to_bf16(float x) {
    std::uint32_t u = 0;
    const float r = bf16_round(x);
    static_assert(sizeof(u) == sizeof(r));
    std::memcpy(&u, &r, sizeof(u));
    return static_cast<std::uint16_t>(u >> 16U);
}

// The staged top-k of rows x vocab logits as the device runs it: 1024 interleaved chunks -> 64 -> 4 -> 1 lists.
void staged_top_k(const std::vector<float>& logits, unsigned rows, unsigned vocab, unsigned k, std::vector<float>& values,
                  std::vector<std::uint32_t>& ids, unsigned& failure) {
    std::vector<float> a(std::size_t{rows} * kTopChunks * k), b(a.size());
    std::vector<std::uint32_t> ai(a.size()), bi(a.size());
    failure = 0U;
    run(std::size_t{rows} * kTopChunks, TopChunks{logits.data(), vocab, k, a.data(), ai.data(), &failure});
    run(std::size_t{rows} * 64U, MergeTop{a.data(), ai.data(), kTopChunks, 16U, k, b.data(), bi.data()});
    run(std::size_t{rows} * 4U, MergeTop{b.data(), bi.data(), 64U, 16U, k, a.data(), ai.data()});
    values.assign(std::size_t{rows} * k, 0.0F);
    ids.assign(values.size(), 0U);
    run(rows, MergeTop{a.data(), ai.data(), 4U, 4U, k, values.data(), ids.data()});
}

void test_top_k() {
    for (const unsigned k : {16U, 4U, 1U}) {
        const unsigned rows = 3U, vocab = 5003U;  // not a multiple of the chunk count
        std::mt19937 rng(7U + k);
        std::uniform_int_distribution<int> level(-40, 40);  // many ties
        std::vector<float> logits(std::size_t{rows} * vocab);
        for (auto& x : logits) x = static_cast<float>(level(rng)) * 0.25F;
        logits[17] = -0.0F;
        logits[vocab + 5] = 1e9F;  // a clear winner
        logits[2 * vocab + 4000] = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> values;
        std::vector<std::uint32_t> ids;
        unsigned failure = 0;
        staged_top_k(logits, rows, vocab, k, values, ids, failure);
        expect(failure == 1U, "a NaN logit raises the failure word");
        for (unsigned r = 0; r < rows; ++r) {
            const float* row = logits.data() + std::size_t{r} * vocab;
            std::vector<std::uint32_t> order(vocab);
            std::iota(order.begin(), order.end(), 0U);
            // The host reference's rule (src/spec/dflash_drafter.cpp), non-finite entries excluded.
            order.erase(std::remove_if(order.begin(), order.end(), [&](std::uint32_t i) { return !std::isfinite(row[i]); }),
                        order.end());
            std::partial_sort(order.begin(), order.begin() + k, order.end(), [row](std::uint32_t l, std::uint32_t rr) {
                return row[l] > row[rr] || (row[l] == row[rr] && l < rr);
            });
            for (unsigned j = 0; j < k; ++j) {
                expect(ids[std::size_t{r} * k + j] == order[j], "staged top-k ids equal the canonical selection");
                expect(values[std::size_t{r} * k + j] == row[order[j]], "staged top-k values equal the logits");
            }
        }
    }
    // All finite: no failure.
    std::vector<float> logits(2000, 1.0F);
    std::vector<float> values;
    std::vector<std::uint32_t> ids;
    unsigned failure = 7U;
    staged_top_k(logits, 1U, 2000U, 16U, values, ids, failure);
    expect(failure == 0U, "finite logits leave the failure word clear");
    for (unsigned j = 0; j < 16U; ++j) expect(ids[j] == j, "all-equal logits select the lowest ids in order");
}

void test_conv() {
    const unsigned rows = 8U, width = 64U, taps = 2U, group = 16U, groups = width / group;
    std::mt19937 rng(11U);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::vector<float> in(std::size_t{rows} * width), delta(std::size_t{rows} * 2U * taps * groups), out(in.size());
    std::vector<std::uint16_t> base(std::size_t{2U} * taps * width);
    for (auto& x : in) x = u(rng);
    for (auto& x : delta) x = u(rng);
    for (auto& x : base) x = to_bf16(u(rng));
    for (unsigned side = 0; side < 2U; ++side) {
        run(in.size(), GroupedConv{in.data(), base.data(), delta.data(), out.data(), width, taps, group, side});
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned c = 0; c < width; ++c) {
                float acc = 0.0F;
                for (unsigned t = 0; t < taps && t <= r; ++t)
                    acc = acc + (decode_bf16(base[(std::size_t{side} * taps + t) * width + c]) +
                                 delta[std::size_t{r} * 2U * taps * groups + (side * taps + t) * groups + c / group]) *
                                    in[std::size_t{r - t} * width + c];
                expect(out[std::size_t{r} * width + c] == bf16_round(acc), "grouped convolution equals its formula");
            }
        // Row 0 reads only itself (no row before the anchor row).
        expect(out[0] == bf16_round((decode_bf16(base[std::size_t{side} * taps * width]) +
                                     delta[side * taps * groups]) * in[0]), "row 0 sees tap 0 only");
    }
}

void test_selector() {
    const unsigned steps = 7U, K = 16U, rank = 32U, vocab = 300U;
    const std::uint32_t anchor = 123U;
    std::mt19937 rng(5U);
    std::uniform_real_distribution<float> u(-2.0F, 2.0F);
    std::uniform_int_distribution<std::uint32_t> id(0U, vocab - 1U);
    std::vector<float> unary(std::size_t{steps} * K), hidden(std::size_t{steps} * rank);
    std::vector<std::uint32_t> cand(unary.size());
    std::vector<std::uint16_t> pred(std::size_t{vocab} * rank), succ(pred.size());
    for (auto& x : unary) x = u(rng) * 20.0F;
    for (auto& x : hidden) x = u(rng);
    for (auto& x : cand) x = id(rng);
    for (auto& x : pred) x = to_bf16(u(rng));
    for (auto& x : succ) x = to_bf16(u(rng));
    for (const bool before : {false, true}) {
        const float cap = 30.0F;
        std::vector<float> scores(std::size_t{steps} * K * K);
        run(scores.size(), EdgeScores{unary.data(), cand.data(), hidden.data(), pred.data(), succ.data(), anchor, vocab, K,
                                      rank, cap, before, scores.data()});
        std::vector<std::uint32_t> picks(steps);
        run(1U, GreedyPath{scores.data(), cand.data(), steps, K, picks.data()});
        // Independent replay of the reference path.
        std::uint32_t prev_index = 0U;
        for (unsigned j = 0; j < steps; ++j) {
            const std::uint32_t prev = j == 0U ? anchor : cand[std::size_t{j - 1U} * K + prev_index];
            unsigned best = 0U;
            float best_score = -std::numeric_limits<float>::infinity();
            for (unsigned c = 0; c < K; ++c) {
                float dot = 0.0F;
                for (unsigned q = 0; q < rank; ++q)
                    dot = dot + (decode_bf16(pred[std::size_t{prev} * rank + q]) * hidden[std::size_t{j} * rank + q]) *
                                    decode_bf16(succ[std::size_t{cand[std::size_t{j} * K + c]} * rank + q]);
                float s = unary[std::size_t{j} * K + c];
                if (before) s = cap * std::tanh(s / cap);
                float e = s + dot;
                if (!before) e = cap * std::tanh(e / cap);
                expect(scores[(std::size_t{j} * K + (j == 0U ? 0U : prev_index)) * K + c] == e, "edge score formula");
                if (c == 0U || e > best_score) {
                    best = c;
                    best_score = e;
                }
            }
            expect(picks[j] == cand[std::size_t{j} * K + best], "greedy path follows the chosen predecessor");
            prev_index = best;
        }
        // Step 0 has one predecessor (the anchor): the other rows are never read and score -inf.
        expect(std::isinf(scores[K]) && scores[K] < 0.0F, "step 0 rows past the anchor are excluded");
    }
    // Anchor ids beyond the codebooks clamp to the last row (as the host reference).
    std::vector<float> scores(std::size_t{1U} * K * K), clamped(scores.size());
    run(scores.size(), EdgeScores{unary.data(), cand.data(), hidden.data(), pred.data(), succ.data(), vocab + 50U, vocab, K,
                                  rank, 30.0F, false, scores.data()});
    run(clamped.size(), EdgeScores{unary.data(), cand.data(), hidden.data(), pred.data(), succ.data(), vocab - 1U, vocab, K,
                                   rank, 30.0F, false, clamped.data()});
    expect(scores == clamped, "anchor beyond the codebooks reads the last row");
}

}  // namespace

int main() {
    test_top_k();
    test_conv();
    test_selector();
    if (failures) {
        std::fprintf(stderr, "spec_dflash_rows_tests: %d failures\n", failures);
        return 1;
    }
    std::printf("spec_dflash_rows_tests: PASS\n");
    return 0;
}
