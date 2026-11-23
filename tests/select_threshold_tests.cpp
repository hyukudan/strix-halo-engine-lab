// Host tests of the "Threshold" select schedule (src/attn_core/select_threshold.hpp, statement in
// src/attn_core/select_threshold_rule.hpp). Integer only, so the host statement is the device result.
// Invariants:
//  1. order_key is the frozen pre-lift statement on every tested bit pattern and is monotone in float order
//     (equal floats incl. -0 / +0 -> keys ordered by bits, -inf below every finite value).
//  2. Both threshold schedules (whole-bin early end, legacy four passes) + the ordered compaction select the k largest
//     keys, equal keys by the smaller position, in ascending position: the rule of select.hpp's other schedules.
//  3. The two-level slice / merge form gives the one-level positions for every slice size tested.
//  4. Candidate-block pass: the selected blocks are the cand_blocks largest block maxima (smaller block first among
//     equals), the newest block always among them, blocks whose maximum is -inf never.
#include "attn_core/select_threshold_rule.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace rule = strix::attn_core::select::threshold::rule;

namespace {
int g_failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Frozen pre-lift statement of the private order key.
std::uint32_t frozen_order_key(float x) {
    std::uint32_t b = 0;
    std::memcpy(&b, &x, sizeof b);
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}

std::vector<std::uint32_t> reference_top_k(const std::vector<std::uint32_t>& keys, std::uint32_t k) {
    std::vector<std::uint32_t> pos(keys.size());
    std::iota(pos.begin(), pos.end(), 0U);
    if (keys.size() <= k) return pos;
    std::stable_sort(pos.begin(), pos.end(), [&](std::uint32_t a, std::uint32_t b) { return keys[a] > keys[b]; });
    pos.resize(k);
    std::sort(pos.begin(), pos.end());
    return pos;
}

std::vector<std::uint32_t> random_keys(std::mt19937& rng, std::uint32_t n, unsigned mode) {
    std::vector<std::uint32_t> keys(n);
    std::uniform_int_distribution<std::uint32_t> any(1U, 0xFFFFFFFFU);
    std::uniform_int_distribution<std::uint32_t> few(0U, 7U);
    std::normal_distribution<float> normal(0.0F, 2.0F);
    for (auto& key : keys) {
        if (mode == 0U) key = any(rng);                                   // distinct keys, all bytes live
        else if (mode == 1U) key = 0x80000000U + few(rng);                 // heavy ties in the last byte
        else if (mode == 2U) key = (few(rng) << 24U) | 0x00ABCD00U | 1U;   // ties sharing three lower bytes
        else key = rule::order_key(normal(rng));                           // float scores
    }
    return keys;
}

void test_order_key(std::mt19937& rng) {
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> xs = {-inf, -1.0e30F, -1.0F, -1.0e-40F, -0.0F, 0.0F, 1.0e-40F, 1.0F, 1.0e30F, inf};
    std::uniform_real_distribution<float> u(-1.0e6F, 1.0e6F);
    for (int i = 0; i < 4096; ++i) xs.push_back(u(rng));
    for (const float x : xs) CHECK(rule::order_key(x) == frozen_order_key(x));
    for (const float a : xs)
        for (std::size_t j = 0; j < 64; ++j) {
            const float b = xs[j];
            if (a < b) CHECK(rule::order_key(a) < rule::order_key(b));
            if (a > b) CHECK(rule::order_key(a) > rule::order_key(b));
        }
    CHECK(rule::order_key(-inf) < rule::order_key(-std::numeric_limits<float>::max()));
}

template <bool WholeBinEnd>
void check_select(const std::vector<std::uint32_t>& keys, std::uint32_t k) {
    const auto n = static_cast<std::uint32_t>(keys.size());
    std::vector<std::uint32_t> idx(std::max(n, k) + 1U, 0xFFFFFFFFU);
    const std::uint32_t count = rule::select_positions<WholeBinEnd>(keys.data(), n, k, idx.data());
    const auto ref = reference_top_k(keys, k);
    CHECK(count == ref.size());
    CHECK(std::equal(ref.begin(), ref.end(), idx.begin()));
}

void test_select(std::mt19937& rng) {
    for (unsigned mode = 0; mode < 4; ++mode)
        for (const std::uint32_t n : {1U, 7U, 64U, 300U, 1024U, 1500U, 5000U})
            for (const std::uint32_t k : {1U, 3U, 32U, 512U, 2048U}) {
                const auto keys = random_keys(rng, n, mode);
                check_select<true>(keys, k);
                check_select<false>(keys, k);
                if (n > k) {   // both schedules: same keys taken
                    const auto t1 = rule::radix_threshold<true>(keys.data(), n, k);
                    const auto t0 = rule::radix_threshold<false>(keys.data(), n, k);
                    std::uint32_t c1 = 0, c0 = 0;
                    rule::ordered_select(keys.data(), n, t1, [&](std::uint32_t, bool s, std::uint32_t) { c1 += s ? 1U : 0U; });
                    rule::ordered_select(keys.data(), n, t0, [&](std::uint32_t, bool s, std::uint32_t) { c0 += s ? 1U : 0U; });
                    CHECK(c1 == k);
                    CHECK(c0 == k);
                    CHECK(t1.need <= t0.need);
                }
            }
}

template <bool WholeBinEnd>
void test_two_level_one(std::mt19937& rng) {
    for (unsigned mode = 0; mode < 4; ++mode)
        for (const std::uint32_t n : {50U, 513U, 4097U})
            for (const std::uint32_t k : {1U, 7U, 64U})
                for (const std::uint32_t slice : {8U, 64U, 1000U, 4096U}) {
                    const auto keys = random_keys(rng, n, mode);
                    const std::uint32_t slices = (n + slice - 1) / slice;
                    std::vector<std::uint32_t> sk(static_cast<std::size_t>(slices) * k), sp(sk.size()), idx(std::max(n, k));
                    const std::uint32_t count =
                        rule::select_two_level<WholeBinEnd>(keys.data(), n, k, slice, sk.data(), sp.data(), idx.data());
                    const auto ref = reference_top_k(keys, k);
                    CHECK(count == ref.size());
                    CHECK(std::equal(ref.begin(), ref.end(), idx.begin()));
                }
}

void test_candidate_blocks(std::mt19937& rng) {
    const float ninf = -std::numeric_limits<float>::infinity();
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_int_distribution<int> coin(0, 3);
    for (const std::uint32_t n : {1U, 9U, 64U, 1000U, 3000U})
        for (const std::uint32_t bs : {8U, 64U})
            for (const std::uint32_t cb : {1U, 4U, 16U}) {
                std::vector<float> scores(n);
                for (auto& s : scores) s = coin(rng) == 0 ? ninf : normal(rng);
                const std::uint32_t nb = (n + bs - 1) / bs;
                std::vector<std::uint32_t> bkeys(nb);
                std::vector<std::uint8_t> cand(nb, 2U);
                rule::candidate_blocks<true>(scores.data(), n, bs, cb, bkeys.data(), cand.data());
                std::vector<std::uint8_t> cand0(nb, 2U);
                std::vector<std::uint32_t> bkeys0(nb);
                rule::candidate_blocks<false>(scores.data(), n, bs, cb, bkeys0.data(), cand0.data());
                CHECK(cand == cand0);
                CHECK(cand[nb - 1] == 1U);   // the newest block always
                const auto top = reference_top_k(bkeys, cb);
                std::vector<std::uint8_t> want(nb, 0U);
                for (const std::uint32_t b : top) want[b] = bkeys[b] > rule::order_key(ninf) ? 1U : 0U;
                CHECK(cand == want);
            }
}
} // namespace

int main() {
    std::mt19937 rng(20261122U);
    test_order_key(rng);
    test_select(rng);
    test_two_level_one<true>(rng);
    test_two_level_one<false>(rng);
    test_candidate_blocks(rng);
    if (g_failures != 0) {
        std::fprintf(stderr, "select_threshold_tests: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("select_threshold_tests: PASS\n");
    return 0;
}
