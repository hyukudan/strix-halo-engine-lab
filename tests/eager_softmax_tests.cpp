// Host rule of the shared eager selected softmax (src/attn_core/eager_softmax.hpp): lane striding, LDS halving-tree
// pairing, failure checks and rounded probabilities.
#include "attn_core/eager_softmax_rule.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace es = strix::attn_core::eager_softmax;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::uint32_t bits(float x) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

float bf16_round(float x) {  // round to nearest even BF16, quiet NaN
    std::uint32_t u = bits(x);
    if ((u & 0x7fffffffU) > 0x7f800000U)
        u = (u | 0x00400000U) & 0xffff0000U;
    else
        u = (u + 0x7fffU + ((u >> 16U) & 1U)) & 0xffff0000U;
    float y = 0.0F;
    std::memcpy(&y, &u, sizeof y);
    return y;
}

float exp_f(float x) { return std::exp(x); }
float identity(float x) { return x; }

}  // namespace

int main() {
    // Pairing: 4 lanes a, b, c, d -> (a + c) + (b + d), not ((a + b) + c) + d.
    {
        const float a = 1.0e8F, b = 1.0F, c = -1.0e8F, d = 1.0F;
        const float tree = es::halving_tree({a, b, c, d}, [](float x, float y) { return x + y; });
        check(bits(tree) == bits((a + c) + (b + d)), "halving tree pairs lane l with l + L/2");
        check(bits(tree) != bits(((a + b) + c) + d), "halving tree differs from the serial order on this input");
    }
    // Lane striding: with 2 lanes and 4 scores lane 0 sums s0, s2 and lane 1 sums s1, s3 before the tree.
    {
        const float s[4] = {0.0F, -30.0F, -0.5F, -30.0F};
        const auto r = es::softmax_row_rule<2U>(s, 4U, true, exp_f, identity);
        const float l0 = 0.0F + std::exp(0.0F) + std::exp(-0.5F), l1 = 0.0F + std::exp(-30.0F) + std::exp(-30.0F);
        check(!r.failed && bits(r.denominator) == bits(l0 + l1), "lane partials are strided and ascending");
        check(bits(r.probabilities[2]) == bits(std::exp(-0.5F) / (l0 + l1)), "probability = exp(s - max) / sum");
    }
    // Maximum by fmax: a NaN score is skipped by the maximum, poisons the sum, and the check fires.
    {
        const float s[3] = {1.0F, std::numeric_limits<float>::quiet_NaN(), 2.0F};
        const auto r = es::softmax_row_rule<256U>(s, 3U, true, exp_f, identity);
        check(r.maximum == 2.0F, "fmax maximum ignores NaN");
        check(r.failed, "NaN denominator fails the row");
        check(!es::softmax_row_rule<256U>(s, 3U, false, exp_f, identity).failed, "checks off never fail");
    }
    // Empty row and all -inf row: non-finite maximum.
    {
        const float s[2] = {-INFINITY, -INFINITY};
        check(es::softmax_row_rule<256U>(s, 0U, true, exp_f, identity).failed, "empty row fails");
        check(es::softmax_row_rule<256U>(s, 2U, true, exp_f, identity).failed, "all -inf row fails");
    }
    // Wider than the block: 1000 scores over 256 lanes; BF16 probabilities, top score has the largest one and the
    // unrounded probabilities sum to 1 within FP32 accumulation error.
    {
        std::vector<float> s(1000);
        for (std::size_t i = 0; i < s.size(); ++i) s[i] = std::sin(static_cast<float>(i) * 0.37F) * 4.0F;
        s[517] = 9.0F;
        const auto r = es::softmax_row_rule<256U>(s.data(), 1000U, true, exp_f, bf16_round);
        const auto u = es::softmax_row_rule<256U>(s.data(), 1000U, true, exp_f, identity);
        check(!r.failed && r.maximum == 9.0F, "maximum over strided lanes");
        bool rounded = true, top = true;
        double total = 0.0;
        for (std::size_t i = 0; i < s.size(); ++i) {
            rounded = rounded && (bits(r.probabilities[i]) & 0xffffU) == 0U && r.probabilities[i] == bf16_round(u.probabilities[i]);
            top = top && r.probabilities[i] <= r.probabilities[517];
            total += static_cast<double>(u.probabilities[i]);
        }
        check(rounded, "stored probability = round(unrounded probability)");
        check(top, "largest score has the largest probability");
        check(std::fabs(total - 1.0) < 1.0e-5, "probabilities sum to one");
    }
    if (failures != 0) return 1;
    std::printf("eager softmax rule tests passed\n");
    return 0;
}
