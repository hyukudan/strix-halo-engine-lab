// Contract of the shared greedy head pick (include/strix/head_argmax.hpp): the chunked two-stage form the device bodies
// run (partial pick per chunk in index order, then a merge of the chunk picks in any order) equals the plain reference
// for every chunk count, with planted ties, +0 / -0, non-finite logits and a vocabulary offset; and the engine
// adapters' words follow from one Pick (Qwen: index or none on non-finite; DS41: value / index / finite; GLM: token /
// nonfinite / logit bits).
#include "strix/head_argmax.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace ha = strix::head_argmax;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}
bool same(const ha::Pick& a, const ha::Pick& b) {
    std::uint32_t va = 0, vb = 0;
    std::memcpy(&va, &a.value, 4);
    std::memcpy(&vb, &b.value, 4);
    return a.index == b.index && a.nonfinite == b.nonfinite && (a.index == ha::kNone || va == vb);
}
// Device-shaped evaluation: per chunk, `threads` lanes each scan a strided subsequence (index order within a lane),
// lanes merged by a butterfly; chunks merged in a shuffled order.
ha::Pick chunked(const std::vector<float>& x, std::uint32_t first, std::uint32_t chunks, std::uint32_t threads,
                 std::mt19937& rng) {
    const auto count = static_cast<std::uint32_t>(x.size());
    std::vector<ha::Pick> parts(chunks);
    for (std::uint32_t c = 0; c < chunks; ++c) {
        std::uint32_t lo = 0, hi = 0;
        ha::chunk_range(count, chunks, c, lo, hi);
        std::vector<ha::Pick> lane(threads);
        for (std::uint32_t t = 0; t < threads; ++t)
            for (std::uint32_t i = lo + t; i < hi; i += threads) {
                const float y = x[i];
                if (!std::isfinite(y)) ++lane[t].nonfinite;
                else if (ha::better(y, first + i, lane[t].value, lane[t].index)) { lane[t].value = y; lane[t].index = first + i; }
            }
        for (std::uint32_t d = threads / 2U; d; d /= 2U)
            for (std::uint32_t t = 0; t < d; ++t) lane[t] = ha::merge(lane[t], lane[t + d]);
        parts[c] = lane[0];
    }
    std::shuffle(parts.begin(), parts.end(), rng);
    ha::Pick p;
    for (const auto& q : parts) p = ha::merge(p, q);
    return p;
}
// Plain first-max over finite values, written independently of the header's helpers.
ha::Pick naive(const std::vector<float>& x, std::uint32_t first) {
    ha::Pick p;
    bool have = false;
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) { ++p.nonfinite; continue; }
        if (!have || x[j] > p.value) { have = true; p.value = x[j]; p.index = static_cast<std::uint32_t>(first + j); }
    }
    return p;
}
}  // namespace

int main() {
    std::mt19937 rng(20261119U);
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const std::uint32_t chunk_set[] = {1, 2, 3, 7, 64, 121, 128};
    for (int trial = 0; trial < 40; ++trial) {
        const std::uint32_t count = trial < 4 ? static_cast<std::uint32_t>(1 + trial) : 1000U + static_cast<std::uint32_t>(rng() % 20000U);
        const std::uint32_t first = (trial % 3 == 0) ? 0U : static_cast<std::uint32_t>(rng() % 200000U);
        std::vector<float> x(count);
        std::uniform_real_distribution<float> u(-8.0F, 8.0F);
        for (auto& v : x) v = std::round(u(rng) * 4.0F) / 4.0F;  // coarse grid: many equal values
        if (count > 10) {
            const float top = 9.5F;
            for (int k = 0; k < 4; ++k) x[rng() % count] = top;  // planted ties at the maximum
            if (trial % 4 == 1) { x[rng() % count] = nan; x[rng() % count] = -inf; }
            if (trial % 4 == 2) x[rng() % count] = inf;
        }
        const ha::Pick ref = ha::reference(x, first);
        check(same(ref, naive(x, first)), "reference == naive first max");
        for (const std::uint32_t chunks : chunk_set)
            for (const std::uint32_t threads : {32U, 256U})
                check(same(chunked(x, first, chunks, threads, rng), ref), "chunked == reference");
    }
    {  // +0 == -0: the lower index wins whichever sign comes first
        std::vector<float> x = {-1.0F, -0.0F, 0.0F, -2.0F};
        check(ha::reference(x).index == 1U, "-0 before +0 -> index 1");
        x = {0.0F, -0.0F};
        check(ha::reference(x).index == 0U, "+0 before -0 -> index 0");
    }
    {  // all non-finite: no index, counted
        const std::vector<float> x = {nan, inf, -inf};
        const ha::Pick p = ha::reference(x, 5);
        check(p.index == ha::kNone && p.nonfinite == 3U, "all non-finite");
        check(same(chunked(x, 5, 2, 32, rng), p), "all non-finite chunked");
    }
    {  // adapters: Qwen (index or none), DS41 (value, index, finite), GLM (token, nonfinite, logit)
        const std::vector<float> x = {1.0F, 3.0F, nan, 3.0F};
        const ha::Pick p = ha::reference(x);
        check(p.index == 1U && p.nonfinite == 1U && p.value == 3.0F, "pick with a NaN");
        const std::uint32_t qwen = p.nonfinite ? ha::kNone : p.index;
        check(qwen == ha::kNone, "Qwen word: non-finite row -> none");
        const std::uint32_t ds41_finite = p.nonfinite == 0U ? 1U : 0U;
        check(ds41_finite == 0U && p.index == 1U, "DS41 candidate: index of the finite max, finite = 0");
    }
    check(ha::default_chunks(248320) == ha::kMaxChunks && ha::default_chunks(10) == 1U, "default chunks");
    check(ha::workspace_bytes(16) == 16U * 128U * 12U, "workspace bytes");
    {  // chunk_range covers [0, count) exactly once for every chunk count, including chunks > count
        bool ok = true;
        for (std::uint32_t count : {1U, 5U, 1000U, 248320U})
            for (std::uint32_t chunks : {1U, 3U, 128U}) {
                std::uint32_t next = 0;
                for (std::uint32_t c = 0; c < chunks; ++c) {
                    std::uint32_t lo = 0, hi = 0;
                    ha::chunk_range(count, chunks, c, lo, hi);
                    if (lo > hi || (lo < hi && lo != next)) ok = false;
                    if (lo < hi) next = hi;
                }
                if (next != count) ok = false;
            }
        check(ok, "chunk_range partition");
    }
    {  // Split-vocabulary pair words (MiniMax host head_argmax; device mirror in xport argmax_exchange).
        const std::uint32_t half = 1000;
        // The rule the MiniMax host scan used before binding: first max per half (`l[i] > l[best]`), then
        // `v1 > v0 ? id1 : id0` across ranks. Finite rows only (it threw on the rank that saw a non-finite logit).
        const auto legacy = [](const std::vector<float>& l0, const std::vector<float>& l1, std::uint32_t id0_1) {
            std::uint32_t b0 = 0, b1 = 0;
            for (std::uint32_t i = 0; i < l0.size(); ++i) if (l0[i] > l0[b0]) b0 = i;
            for (std::uint32_t i = 0; i < l1.size(); ++i) if (l1[i] > l1[b1]) b1 = i;
            return l1[b1] > l0[b0] ? id0_1 + b1 : b0;
        };
        const auto pair = [&](const std::vector<float>& l0, const std::vector<float>& l1) {
            // Both ranks call pair_pick(rank 0's word, rank 1's word), so they agree by construction.
            return ha::pair_pick(ha::pair_word(ha::reference(l0, 0)), ha::pair_word(ha::reference(l1, half)));
        };
        std::uniform_int_distribution<int> small(-4, 4);
        bool same = true;
        for (int t = 0; t < 2000; ++t) {
            const auto ut = static_cast<std::size_t>(t);
            const std::size_t n0 = 1 + ut % 37, n1 = 1 + ut % 23;
            std::vector<float> l0(n0), l1(n1);
            for (auto& v : l0) v = static_cast<float>(small(rng)) * 0.5F;  // many planted ties within and across halves
            for (auto& v : l1) v = static_cast<float>(small(rng)) * 0.5F;
            if (ut % 5 == 0) l0[ut % n0] = -0.0F;
            if (ut % 7 == 0) l1[ut % n1] = 0.0F;
            same &= pair(l0, l1) == legacy(l0, l1, half);
        }
        check(same, "pair pick == legacy MiniMax host rule on finite rows (ties, +0/-0)");
        check(pair({1.0F, 2.0F}, {2.0F, 1.0F}) == 1U, "pair tie across halves -> rank 0");
        check(pair({-0.0F}, {0.0F}) == 0U, "pair +0 == -0 -> rank 0");
        check(pair({1.0F, 2.0F}, {2.5F, 2.5F}) == half, "pair rank 1 larger -> its first max");
        check(pair({1.0F, nan}, {2.0F}) == ha::kNone, "pair NaN on rank 0 fails both");
        check(pair({1.0F}, {inf, 0.0F}) == ha::kNone, "pair +inf on rank 1 fails both");
        check(pair({-inf}, {1.0F}) == ha::kNone, "pair -inf fails both");
        const auto w = ha::pair_word(ha::reference(std::vector<float>{3.5F, 7.25F, 7.25F}, 40));
        float v = 0.0F;
        std::memcpy(&v, &w.value_bits, 4);
        check(w.index == 41U && v == 7.25F, "pair word carries value bits + global index");
    }
    std::printf("%s head_argmax: %d failures\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
