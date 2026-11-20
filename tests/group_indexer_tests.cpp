// Host tests of the compressed group indexer contract (include/strix/attn/group_indexer.hpp).
// Invariants: MODEL_B gated pooling vs an independent double formula; the shared gated pool channel is bit-identical to
// GLM bf16 pool channel (shared since the pool lift) on the host; masking of the own / later / incomplete groups; declared tie rule and the set it
// can differ from a torch.topk reference by; expansion vs the reference's clamp + causal mask; dense below 2048.
#include "strix/attn/group_indexer.hpp"

#include "strix/ds41_msvc_builtins.hpp"  // GLM math uses GCC builtins; no-op outside MSVC
#include "glm/common/glm_math.hpp"
#include "strix/attn/group_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace gi = strix::attn::group_indexer;
// Gated-pool AlwaysOwn indexer shape exercised here (ratio 4, 16 heads x 128, 512 groups); a model binding keeps its
// own Shape in its model package.
constexpr gi::Shape kGatedApe{4U, 16U, 128U, 512U, gi::Pooling::GatedSoftmaxApe, gi::OwnGroup::AlwaysOwn,
                              gi::ScoreScale::InHeadWeight};

namespace {
int g_failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

std::uint32_t bits(float x) { return __builtin_bit_cast(std::uint32_t, x); }
std::uint16_t bf16_bits(float x) { return static_cast<std::uint16_t>(bits(gi::bf16_round(x)) >> 16U); }
float exp_glm(float x) { return strix::glm::declared_exp(x); }

void test_shape_rules() {
    const gi::Shape& h = kGatedApe;
    CHECK(gi::visible_groups(h, 0) == 0 && gi::visible_groups(h, 3) == 0 && gi::visible_groups(h, 4) == 1);
    CHECK(gi::pick_count(h) == 511 && gi::max_tokens(h) == 2048);
    CHECK(gi::dense(h, 2047) && !gi::dense(h, 2048));  // the reference's kv_len <= 2048 switch, per query
    const gi::Shape& g = gi::kGlmDsa;
    CHECK(gi::visible_groups(g, 3) == 1 && gi::pick_count(g) == 512 && gi::max_tokens(g) == 2051);
    CHECK(gi::dense(g, 2050) && !gi::dense(g, 2051));
    CHECK(gi::tail_begin(h, 11) == 8 && gi::tail_begin(g, 11) == 12);  // own complete group: MODEL_B tail, GLM scored
    CHECK(gi::group_rope_slot(h, 5) == 20 && gi::closed_groups(h, 23) == 5 && gi::token_ring_slots(h, 16) == 19);
    CHECK(bits(gi::head_weight_scale(h)) == bits(static_cast<float>(0.25 * 0.08838834764831845)));
}

void test_pool_gated_ape(std::mt19937& rng) {
    const gi::Shape& s = kGatedApe;
    const unsigned d = s.head_dim, r = s.ratio;
    std::normal_distribution<float> n01(0.0F, 1.0F);
    std::vector<float> keys(r * d), wv(d * d), wg(d * d), ape(r * d), norm(d), out(d);
    for (int rep = 0; rep < 4; ++rep) {
        for (float& x : keys) x = gi::bf16_round(n01(rng));
        for (float& x : wv) x = n01(rng) * 0.09F;
        for (float& x : wg) x = n01(rng) * 0.09F * static_cast<float>(rep + 1);
        for (float& x : ape) x = n01(rng);
        for (float& x : norm) x = 1.0F + 0.1F * n01(rng);
        gi::pool_group_gated_ape(s, keys.data(), gi::GatedApeWeights{wv.data(), wg.data(), ape.data(), norm.data(), 1e-5F},
                                 exp_glm, out.data());
        // Independent formula in double.
        std::vector<double> pooled(d);
        for (unsigned o = 0; o < d; ++o) {
            double v[4], gl[4], m = -1e300, z = 0.0, acc = 0.0;
            for (unsigned j = 0; j < r; ++j) {
                v[j] = 0.0;
                gl[j] = ape[j * d + o];
                for (unsigned i = 0; i < d; ++i) {
                    v[j] += static_cast<double>(wv[o * d + i]) * keys[j * d + i];
                    gl[j] += static_cast<double>(wg[o * d + i]) * keys[j * d + i];
                }
                m = std::max(m, gl[j]);
            }
            for (unsigned j = 0; j < r; ++j) z += std::exp(gl[j] - m);
            for (unsigned j = 0; j < r; ++j) acc += std::exp(gl[j] - m) / z * v[j];
            pooled[o] = acc;
        }
        double ss = 0.0;
        for (const double x : pooled) ss += x * x;
        const double inv = 1.0 / std::sqrt(ss / d + 1e-5);
        for (unsigned o = 0; o < d; ++o) {
            const double want = norm[o] * pooled[o] * inv;
            CHECK((bits(out[o]) & 0xFFFFU) == 0U);                            // stored as bf16
            CHECK(std::fabs(out[o] - want) <= std::fabs(want) * 0x1p-7 + 1e-6);  // within one bf16 rounding
        }
    }
}

void test_glm_pool_identity(std::mt19937& rng) {
    using strix::glm::kIdxDim;
    using strix::glm::kKpool;
    static_assert(kKpool == 4U, "GLM pools 4 slots");
    std::normal_distribution<float> n01(0.0F, 1.0F);
    std::vector<std::uint16_t> key(kKpool * kIdxDim), gate(kKpool * kIdxDim);
    std::vector<float> ape(kKpool * kIdxDim);
    unsigned same = 0, total = 0;
    for (int rep = 0; rep < 64; ++rep) {
        const float spread = rep % 8 == 0 ? 40.0F : 3.0F;  // also saturated softmax rows
        for (auto& x : key) x = bf16_bits(n01(rng) * 4.0F);
        for (auto& x : gate) x = bf16_bits(n01(rng) * spread);
        for (float& x : ape) x = n01(rng);
        for (unsigned c = 0; c < kIdxDim; ++c) {
            float logits[4], values[4];
            for (unsigned j = 0; j < 4; ++j) {
                logits[j] = strix::glm::bf16_to_float(gate[j * kIdxDim + c]) + ape[j * kIdxDim + c];
                values[j] = strix::glm::bf16_to_float(key[j * kIdxDim + c]);
            }
            const float shared = gi::gated_pool_channel<4>(logits, values, exp_glm);
            const float glm = strix::attn::group_pool::gated_bf16_channel<4>(key.data(), gate.data(), ape.data(), c, kIdxDim, exp_glm);
            same += bits(shared) == bits(glm) ? 1U : 0U;
            ++total;
        }
    }
    CHECK(same == total);
}

void test_score(std::mt19937& rng) {
    const gi::Shape& s = kGatedApe;
    std::normal_distribution<float> n01(0.0F, 1.0F);
    std::vector<float> q(s.heads * s.head_dim), k(s.head_dim), w(s.heads);
    for (float& x : q) x = gi::bf16_round(n01(rng));
    for (float& x : k) x = gi::bf16_round(n01(rng));
    for (float& x : w) x = n01(rng) * gi::head_weight_scale(s);
    double want = 0.0;
    for (unsigned h = 0; h < s.heads; ++h) {
        double dot = 0.0;
        for (unsigned i = 0; i < s.head_dim; ++i) dot += static_cast<double>(q[h * s.head_dim + i]) * k[i];
        want += w[h] * std::max(dot, 0.0);
    }
    const float got = gi::score_group(s, q.data(), k.data(), w.data(), 123.0F);  // c ignored for InHeadWeight
    CHECK(std::fabs(got - want) <= 1e-4 * (std::fabs(want) + 1.0));
    // Every head dot negative, negative head weights: the score is +0, never -0.
    for (float& x : k) x = -std::fabs(x);
    for (float& x : q) x = std::fabs(x) + 0.01F;
    for (float& x : w) x = -std::fabs(x);
    CHECK(bits(gi::score_group(s, q.data(), k.data(), w.data(), 1.0F)) == 0U);
    CHECK(gi::score_key(-0.0F) == gi::score_key(0.0F) && gi::score_key(-1.0F) < gi::score_key(0.0F) &&
          gi::score_key(0.0F) < gi::score_key(1e-30F) && gi::score_key(2.0F) < gi::score_key(3.0F));
}

// The reference (modeling_hy_v4_flash.py:425-434, 657-662): scores of ALL num_groups = n / 4 complete groups of the
// context, groups >= own = t / 4 masked to -inf, topk(min(511, num_groups)) with an unspecified order among equal values
// (here: higher index first, the adversarial choice), + own group, expanded, clamp(max = n - 1), AND causal (<= t).
std::vector<std::uint32_t> reference_tokens(const std::vector<float>& all_scores, std::uint32_t n, std::uint32_t t) {
    const std::uint32_t groups = n / 4U, own = t / 4U;
    std::vector<std::pair<float, std::uint32_t>> v;
    for (std::uint32_t g = 0; g < groups; ++g)
        v.emplace_back(g >= own ? -std::numeric_limits<float>::infinity() : all_scores[g], g);
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second > b.second; });
    std::set<std::uint32_t> set;
    const std::uint32_t k = std::min(511U, groups);
    std::vector<std::uint32_t> recalled;
    for (std::uint32_t i = 0; i < k; ++i) recalled.push_back(v[i].second);
    recalled.push_back(own);
    for (const std::uint32_t g : recalled)
        for (std::uint32_t j = 0; j < 4U; ++j) {
            const std::uint32_t p = std::min(g * 4U + j, n - 1U);
            if (p <= t) set.insert(p);
        }
    return {set.begin(), set.end()};
}

std::vector<std::uint32_t> ours(const std::vector<float>& all_scores, std::uint32_t t) {
    const gi::Shape& s = kGatedApe;
    const unsigned visible = gi::visible_groups(s, t);
    return gi::expand_tokens(s, gi::select_groups(s, all_scores.data(), visible), t);
}

void test_selection_vs_reference(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    const std::uint32_t n = 3001;  // last group (3000) incomplete: never pooled, only an own group
    std::vector<float> all(n / 4U);
    const std::uint32_t ts[] = {0, 1, 2047, 2048, 2049, 2050, 2051, 2052, 2600, 2999, 3000};
    for (const std::uint32_t t : ts) {
        for (float& x : all) x = u(rng);  // distinct with overwhelming probability
        const auto a = ours(all, t), b = reference_tokens(all, n, t);
        CHECK(a == b);
        CHECK(std::is_sorted(a.begin(), a.end()) && std::adjacent_find(a.begin(), a.end()) == a.end());
        CHECK(!a.empty() && a.back() == t && a.size() <= gi::max_tokens(kGatedApe));
        if (gi::dense(kGatedApe, t)) CHECK(a.size() == t + 1U);
        if (!gi::dense(kGatedApe, t) && t % 4U == 3U) CHECK(a.size() == 2048U);
    }
    // Scores from the shared score arithmetic for one query (later groups carry garbage that must be masked).
    const gi::Shape& s = kGatedApe;
    std::normal_distribution<float> n01(0.0F, 1.0F);
    std::vector<float> q(s.heads * s.head_dim), w(s.heads), keys((n / 4U) * s.head_dim);
    for (float& x : q) x = gi::bf16_round(n01(rng));
    for (float& x : w) x = n01(rng) * gi::head_weight_scale(s);
    for (float& x : keys) x = gi::bf16_round(n01(rng));
    gi::score_row(s, q.data(), w.data(), 0.0F, keys.data(), n / 4U, all.data());
    for (const std::uint32_t t : {2600U, 3000U}) CHECK(ours(all, t) == reference_tokens(all, n, t));
}

void test_ties(std::mt19937& rng) {
    const std::uint32_t n = 4096, t = 4095;  // 1023 visible groups, 511 picked
    std::vector<float> all(n / 4U);
    std::uniform_real_distribution<float> u(1.0F, 2.0F);
    for (float& x : all) x = u(rng);
    // 500 clear winners, then 30 groups tied at the cut value: 11 of them are picked.
    std::vector<std::uint32_t> perm(1023);
    for (std::uint32_t i = 0; i < 1023; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (std::uint32_t i = 0; i < 500; ++i) all[perm[i]] = 10.0F + static_cast<float>(i);
    std::vector<std::uint32_t> tied(perm.begin() + 500, perm.begin() + 530);
    for (const std::uint32_t g : tied) all[g] = 5.0F;
    std::sort(tied.begin(), tied.end());
    const auto groups = gi::select_groups(kGatedApe, all.data(), 1023);
    CHECK(groups.size() == 511U);
    for (std::size_t i = 0; i < tied.size(); ++i)  // declared rule: the lowest 11 tied indices
        CHECK(std::binary_search(groups.begin(), groups.end(), tied[i]) == (i < 11U));
    // Identity risk: a reference that breaks the tie the other way selects the same count but different groups.
    const auto a = ours(all, t), b = reference_tokens(all, n, t);
    CHECK(a.size() == b.size() && a != b);
    std::vector<std::uint32_t> diff;
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
    for (const std::uint32_t p : diff) CHECK(all[p / 4U] == 5.0F);  // only tied groups differ
    // Ties wholly above the cut select the same set under any tie order.
    for (const std::uint32_t g : tied) all[g] = 1e6F;
    CHECK(ours(all, t) == reference_tokens(all, n, t));
}
}  // namespace

int main() {
    std::mt19937 rng(20261118U);
    test_shape_rules();
    test_pool_gated_ape(rng);
    test_glm_pool_identity(rng);
    test_score(rng);
    test_selection_vs_reference(rng);
    test_ties(rng);
    if (g_failures != 0) {
        std::fprintf(stderr, "group_indexer_tests: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("group_indexer_tests: PASS\n");
    return 0;
}
