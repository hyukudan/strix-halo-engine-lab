// Host tests of the "Radix3" select schedule (src/attn_core/select_radix3.hpp, statement in select_radix3_rule.hpp).
// Invariants:
//  1. order_key equals the frozen pre-lift statement (+0 == -0, monotone) on every tested bit pattern.
//  2. The three-pass radix + ordered compaction + zero fill + expansion emit the k first blocks of (key descending, block
//     ascending) among blocks with score > -inf (NaN never), in ascending block order, each block's `ratio` tokens, then
//     the partial-block tail; fewer than k such blocks put block 0 first (zero fill); blocks <= k selects every token.
//  3. The candidate-group form gives the same tokens whenever it does not decline.
#include "attn_core/select_radix3_rule.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace rule = strix::attn_core::select::radix3::rule;

namespace {
int g_failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Frozen pre-lift statement of the private key.
std::uint32_t frozen_key(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}

// Independent statement: serial slot loop of the pre-lift reference (lowest-index maximal score > -inf, block 0 when none).
std::vector<std::uint32_t> serial_reference(std::vector<float> s, std::uint32_t k, std::uint32_t ratio, std::uint32_t visible) {
    const std::uint32_t blocks = visible / ratio;
    std::vector<std::uint32_t> out;
    if (blocks <= k) { for (std::uint32_t t = 0; t < visible; ++t) out.push_back(t); return out; }
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<std::uint32_t> pick;
    for (std::uint32_t slot = 0; slot < k; ++slot) {
        float best = ninf;
        std::uint32_t best_block = 0;
        for (std::uint32_t b = 0; b < blocks; ++b)
            if (s[b] > best || (s[b] == best && b < best_block)) { best = s[b]; best_block = b; }
        pick.push_back(best_block);
        s[best_block] = ninf;
    }
    std::sort(pick.begin(), pick.end());
    for (const auto b : pick)
        for (std::uint32_t o = 0; o < ratio; ++o) out.push_back(b * ratio + o);
    for (std::uint32_t t = blocks * ratio; t < visible; ++t) out.push_back(t);
    return out;
}

std::vector<float> make_scores(std::mt19937& rng, std::uint32_t blocks, unsigned mode) {
    std::vector<float> s(blocks);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::uniform_int_distribution<int> few(-3, 3), pick(0, 9);
    const float ninf = -std::numeric_limits<float>::infinity();
    for (auto& v : s) {
        if (mode == 0U) v = normal(rng);
        else if (mode == 1U) v = static_cast<float>(few(rng));                       // heavy ties
        else if (mode == 2U) v = pick(rng) < 6 ? ninf : normal(rng);                  // mostly -inf
        else v = pick(rng) < 3 ? (pick(rng) < 5 ? 0.0F : -0.0F) : static_cast<float>(few(rng));  // +0 / -0 ties
    }
    return s;
}

void test_order_key(std::mt19937& rng) {
    std::vector<float> xs = {-std::numeric_limits<float>::infinity(), -1.0e30F, -1.0F, -1.0e-40F, -0.0F, 0.0F, 1.0e-40F, 1.0F,
                             1.0e30F, std::numeric_limits<float>::infinity()};
    std::uniform_real_distribution<float> u(-1.0e6F, 1.0e6F);
    for (int i = 0; i < 4096; ++i) xs.push_back(u(rng));
    for (const float x : xs) CHECK(rule::order_key(x) == frozen_key(x));
    CHECK(rule::order_key(0.0F) == rule::order_key(-0.0F));
    for (const float a : xs)
        for (std::size_t j = 0; j < 64; ++j) {
            const float b = xs[j];
            if (a < b) CHECK(rule::order_key(a) <= rule::order_key(b));
            if (a > b) CHECK(rule::order_key(a) >= rule::order_key(b));
        }
}

void test_select(std::mt19937& rng) {
    const std::uint32_t ks[] = {1U, 5U, 16U, 64U, 300U};
    const std::uint32_t blocks_list[] = {17U, 100U, 513U, 2049U, 5000U, 12000U};
    for (unsigned mode = 0; mode < 4U; ++mode)
        for (const auto k : ks)
            for (const auto blocks : blocks_list) {
                if (blocks <= k) continue;
                const std::uint32_t ratio = 4U, tail = blocks % 3U;
                const std::uint32_t visible = blocks * ratio + tail;
                const auto s = make_scores(rng, blocks, mode);
                const auto got = rule::select_tokens(s, k, ratio, visible);
                // Rows with fewer than k finite blocks follow the zero-fill rule, checked separately below.
                std::uint32_t finite = 0;
                for (const float v : s) finite += rule::counts(v) ? 1U : 0U;
                if (finite >= k) CHECK(got == serial_reference(s, k, ratio, visible));
                CHECK(got.size() == static_cast<std::size_t>(k) * ratio + tail);
            }
}

void test_zero_fill(std::mt19937& rng) {
    const float ninf = -std::numeric_limits<float>::infinity();
    for (const std::uint32_t finite : {0U, 1U, 3U, 7U}) {
        const std::uint32_t k = 8U, blocks = 300U, ratio = 4U;
        std::vector<float> s(blocks, ninf);
        std::vector<std::uint32_t> at(blocks);
        std::iota(at.begin(), at.end(), 0U);
        std::shuffle(at.begin(), at.end(), rng);
        at.resize(finite);
        std::sort(at.begin(), at.end());
        for (const auto b : at) s[b] = 1.0F + static_cast<float>(b % 5U);
        s[2] = std::numeric_limits<float>::quiet_NaN();  // NaN never wins; it is not a finite block
        const auto got = rule::select_tokens(s, k, ratio, blocks * ratio);
        CHECK(got.size() == static_cast<std::size_t>(k) * ratio);
        std::vector<std::uint32_t> want;
        std::uint32_t used = 0;
        for (const auto b : at) used += b == 2U ? 0U : 1U;
        for (std::uint32_t z = 0; z < (k - used) * ratio; ++z) want.push_back(z % ratio);
        for (const auto b : at)
            if (b != 2U) for (std::uint32_t o = 0; o < ratio; ++o) want.push_back(b * ratio + o);
        CHECK(got == want);
    }
}

void test_candidates(std::mt19937& rng) {
    for (unsigned mode = 0; mode < 4U; ++mode)
        for (const std::uint32_t k : {4U, 32U, 128U})
            for (const std::uint32_t blocks : {200U, 1000U, 4099U, 9000U}) {
                const std::uint32_t ratio = 4U, visible = blocks * ratio + 1U;
                const auto s = make_scores(rng, blocks, mode);
                std::vector<std::uint32_t> got;
                const bool ok = rule::select_tokens_candidates(s, k, ratio, visible, 4U, 6144U / 4U, got);
                if (ok) CHECK(got == rule::select_tokens(s, k, ratio, visible));
            }
}
}  // namespace

int main() {
    std::mt19937 rng(20261122U);
    test_order_key(rng);
    test_select(rng);
    test_zero_fill(rng);
    test_candidates(rng);
    if (g_failures != 0) {
        std::fprintf(stderr, "select_radix3_tests: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("select_radix3_tests: PASS\n");
    return 0;
}
