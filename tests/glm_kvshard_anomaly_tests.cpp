// Host reproduction of GLM's score ownership, selection and attention arithmetic.
#include <cmath>
#if defined(_MSC_VER) && !defined(__clang__)
// MSVC provides the other builtins used by glm_math.hpp, but not these two.
#define __builtin_rintf std::rintf
#define __builtin_sqrtf std::sqrt
#define __builtin_fmaf std::fma
#endif
#include "glm/engine/glm_kvshard_math.hpp"
#include "glm/attn/glm_dsa_policy.hpp"
#include "glm/common/glm_math.hpp"
#include "strix/attn/group_pool.hpp"
#include "glm/engine/glm_kvshard_dump.hpp"
#include "glm/engine/glm_kvshard_layout.hpp"
#include "xport/kvshard_oracle.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <numeric>
#include <chrono>
#include <cstdlib>

namespace G = strix::glm;
namespace K = strix::glm::kvshard;
namespace O = strix::xport::kvshard;
namespace {
unsigned checks = 0;
float max_attention_error = 0.0F, max_oracle_error = 0.0F, max_old_attention_error = 0.0F;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint64_t> select(const std::vector<std::uint32_t>& keys, std::uint64_t position) {
    std::vector<std::size_t> pools(keys.size());
    std::iota(pools.begin(), pools.end(), std::size_t{0});
    std::sort(pools.begin(), pools.end(), [&](std::size_t a, std::size_t b) {
        return keys[a] == keys[b] ? a < b : keys[a] > keys[b];
    });
    pools.resize(std::min(pools.size(), std::size_t{512}));
    std::sort(pools.begin(), pools.end());
    std::vector<std::uint64_t> tokens;
    for (const auto pool : pools)
        for (unsigned j = 0; j < 4; ++j) tokens.push_back(pool * 4 + j);
    for (std::uint64_t token = keys.size() * 4; token <= position; ++token) tokens.push_back(token);
    return tokens;
}

void selection(std::uint64_t position) {
    const auto pools = static_cast<std::size_t>((position + 1) / 4);
    std::vector<std::uint32_t> full(pools);
    std::array<std::vector<std::uint32_t>, 2> compact;
    for (auto& rank : compact) rank.resize(static_cast<std::size_t>(K::storage_tokens((pools + 1) * 4) / 4), 0xffffffffU);
    std::vector<O::Candidate> candidates;
    for (std::size_t pool = 0; pool < pools; ++pool) {
        const float score = static_cast<float>(static_cast<int>((pool * 31) % 17) - 8);
        full[pool] = G::ordered_key(score);
        compact[pool % 2][pool / 2] = full[pool];
        candidates.push_back({pool, pool * 4 + 3, score});
    }
    const auto chosen = O::select_global(candidates, position, 512);
    std::vector<std::uint64_t> expected;
    for (const auto pool : chosen)
        for (unsigned j = 0; j < 4; ++j) expected.push_back(pool * 4 + j);
    for (std::uint64_t token = pools * 4; token <= position; ++token) expected.push_back(token);
    for (unsigned rank = 0; rank < 2; ++rank) {
        std::vector<std::uint32_t> local(pools, 0xffffffffU), peer(pools, 0xffffffffU);
        for (std::size_t pool = 0; pool < pools; ++pool)
            (pool % 2 == rank ? local : peer)[pool] = compact[pool % 2][pool / 2];
        const auto merged = K::merge_score_keys(local, peer, rank);
        require(merged == full, "score exchange lost an owned pool");
        const auto tokens = select(merged, position);
        require(tokens == expected, "global selection, ties or mandatory tail differs");
        if ((position + 1) % 4 != 0) require(tokens.back() == position, "current token missing from tail");
        require(std::count(tokens.begin(), tokens.end(), position) <= 1, "current token duplicated");
    }
}

float score(const float* query, const float* pooled, const float* weights) {
    float result = 0.0F;
    for (unsigned head = 0; head < G::kIdxHeads; ++head) {
        float dot = 0.0F;
        for (unsigned dim = 0; dim < G::kIdxDim; ++dim)
            dot = std::fma(query[head * G::kIdxDim + dim], pooled[dim], dot);
        const float scaled = dot * G::kInvSqrt128;
        result = std::fma(weights[head] * G::kInvSqrt32, scaled > 0.0F ? scaled : 0.0F, result);
    }
    return result;
}

void score_keys() {
    constexpr std::size_t pools = 2048, width = G::kIdxDim;
    std::vector<float> full(pools * width);
    std::array<std::vector<float>, 2> compact;
    for (auto& rank : compact) rank.resize(pools / 2 * width);
    for (std::size_t p = 0; p < pools; ++p)
        for (std::size_t d = 0; d < width; ++d) {
            const float value = static_cast<float>(static_cast<int>((p * 7 + d * 11) % 37) - 18) / 32.0F;
            full[p * width + d] = value;
            compact[p % 2][(p / 2) * width + d] = value;
        }
    std::array<float, G::kIdxHeads * G::kIdxDim> query{};
    std::array<float, G::kIdxHeads> weights{};
    for (std::size_t i = 0; i < query.size(); ++i)
        query[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 16.0F;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(i + 1) / 32.0F;
    for (std::size_t p = 0; p < pools; ++p) {
        const auto reference = G::ordered_key(score(query.data(), full.data() + p * width, weights.data()));
        const auto owned = G::ordered_key(score(query.data(), compact[p % 2].data() + (p / 2) * width, weights.data()));
        require(reference == owned, "owned score arithmetic differs");
    }
}

float tree(std::array<float, 256> lanes) {
    for (std::size_t stride = 128; stride != 0; stride /= 2)
        for (std::size_t lane = 0; lane < stride; ++lane) lanes[lane] = lanes[lane] + lanes[lane + stride];
    return lanes[0];
}
#include "glm_dsa_policy_fixture.hpp"

void attention(std::uint64_t position) {
    std::vector<std::uint32_t> keys(static_cast<std::size_t>((position + 1) / 4));
    for (std::size_t p = 0; p < keys.size(); ++p) keys[p] = G::ordered_key(static_cast<float>((p * 31) % 17));
    const auto tokens = select(keys, position);
    std::vector<O::Row> rows;
    std::array<std::vector<O::Row>, 2> owned;
    for (const auto token : tokens) {
        O::Row row{};
        row.id = row.token = token;
        row.value.resize(512);
        float dot = 0.0F;
        for (std::size_t dim = 0; dim < 512; ++dim) {
            const float value = static_cast<float>(static_cast<int>((token * 7 + dim * 13) % 61) - 30) / 16.0F;
            row.value[dim] = G::bf16_to_float(G::float_to_bf16(value));
            const float query = static_cast<float>(static_cast<int>(dim % 19) - 9) / 16.0F;
            dot = std::fma(query, row.value[dim], dot);
        }
        row.score = dot * G::kMlaScale;
        owned[K::owner(token)].push_back(row);
        rows.push_back(std::move(row));
    }
    const auto a = policy_attention<true>(rows, 0), b = policy_attention<true>(rows, 1);
    const auto replica = policy_attention<false>(rows, 0);
    const auto oracle = O::normalize(O::merge_rank0_rank1(O::accumulate(owned[0], 512), O::accumulate(owned[1], 512)));
    float maximum = -std::numeric_limits<float>::infinity();
    for (const auto& row : rows) maximum = std::max(maximum, row.score);
    std::vector<float> weights(rows.size());
    std::array<float, 256> lanes{};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        weights[i] = G::declared_exp(rows[i].score - maximum);
        lanes[i % 256] = lanes[i % 256] + weights[i];
    }
    const float denominator = tree(lanes);
    for (auto& weight : weights) weight = weight / denominator;
    for (std::size_t dim = 0; dim < 512; ++dim) {
        lanes.fill(0.0F);
        for (std::size_t i = 0; i < rows.size(); ++i)
            lanes[i % 256] = std::fma(weights[i], rows[i].value[dim], lanes[i % 256]);
        const float reference = tree(lanes);
        require(__builtin_bit_cast(std::uint32_t, replica.a[dim]) == __builtin_bit_cast(std::uint32_t, reference),
                "replicated policy changed baseline output bits");
        const float result = K::merge_component(a.m, a.l, a.a[dim], b.m, b.l, b.a[dim]);
        max_attention_error = std::max(max_attention_error, std::abs(result - reference));
        max_oracle_error = std::max(max_oracle_error, std::abs(result - oracle[dim]));
        max_old_attention_error = std::max(max_old_attention_error, std::abs(oracle[dim] - reference));
        require(std::abs(result - oracle[dim]) <= 1e-5F * (1.0F + std::abs(oracle[dim])),
                "shared partials plus rank01 merge differ from independent oracle");
        require(std::isfinite(result) && std::abs(result - reference) <= 1e-5F * (1.0F + std::abs(reference)),
                "GLM attention exceeds local reordering allowance");
    }
}

void exponent_contract() {
    unsigned old_differences = 0;
    for (int i = 1; i <= 8192; ++i) {
        const float x = -static_cast<float>(i) / 997.0F;
        const float expected = G::declared_exp(x);
        require(K::partial_weight(x, 0.0F) == expected, "partial exponential differs from GLM");
        old_differences += std::exp(x) == expected ? 0U : 1U;
        require(K::merge_component(x, 1.0F, 1.0F, 0.0F, 1.0F, 0.0F) == expected / (expected + 1.0F),
                "merge rescale exponential differs from GLM");
    }
    const float empty = -std::numeric_limits<float>::infinity();
    require(K::merge_component(empty, 0, 0, empty, 0, 0) == 0, "empty ranks produce NaN");
    require(K::merge_component(empty, 0, 0, 3, 2, 4) == 2, "empty rank zero");
    require(K::merge_component(3, 2, 4, empty, 0, 0) == 2, "empty rank one");
    require(old_differences > 0, "old expf contract defect not reproduced");
    std::printf("exponential contract: old std::exp differs on %u/8192 samples\n", old_differences);
}

void selection_discontinuity() {
    // A local attention error bound is not an end-to-end logit error bound.
    std::vector<O::Candidate> before;
    for (std::uint64_t p = 0; p < 513; ++p) before.push_back({p, p * 4 + 3, p < 511 ? 2.0F : 1.0F});
    auto after = before;
    after.back().selection_score = std::nextafter(1.0F, 2.0F);
    const auto left = O::select_global(before, 2051, 512), right = O::select_global(after, 2051, 512);
    require(left.back() == 511 && right.back() == 512, "one-ulp selection discontinuity missing");
    auto attend = [](const std::vector<std::uint64_t>& selected) {
        std::vector<O::Row> rows;
        for (const auto p : selected) rows.push_back({p, p * 4, p, p >= 511 ? 0.0F : -104.0F, {p == 511 ? -0.1F : 0.1F}});
        return O::normalize(O::accumulate(rows, 1))[0];
    };
    const float jump = std::abs(attend(left) - attend(right));
    require(jump > 0.19F, "selection did not amplify tiny score change");
    std::printf("synthetic selection discontinuity: score delta %.9g -> output delta %.9g (not model logits)\n",
                static_cast<double>(after.back().selection_score - 1.0F), static_cast<double>(jump));
}
#include "glm_kvshard_decode_fixture.hpp"
} // namespace

int main() {
    try {
        exponent_contract(); score_keys(); incremental_decode(); decode_dump_format(); policy_empty_ranks(); policy_stress();
        for (std::uint64_t p = 2046; p <= 2053; ++p) selection(p);
        for (std::uint64_t p = 8192; p < 8320; ++p) selection(p);
        for (const auto p : {3ULL, 4ULL, 2047ULL, 2048ULL, 2051ULL, 8192ULL, 8195ULL, 8199ULL, 65535ULL, 65538ULL, 262142ULL}) {
            selection(p); attention(p);
        }
        selection_discontinuity();
        std::printf("GLM shard host: %u checks; fixed attention abs %.9g; old attention abs %.9g; fixed-vs-oracle abs %.9g; device gates NOT RUN\n",
                    checks, static_cast<double>(max_attention_error), static_cast<double>(max_old_attention_error),
                    static_cast<double>(max_oracle_error));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GLM shard host FAILED: %s\n", error.what());
        return 1;
    }
}
