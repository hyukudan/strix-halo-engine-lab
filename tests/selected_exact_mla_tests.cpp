// Shared EXACT materialised rows-selected latent attention (src/attn_core/selected_exact_mla.hpp), host statement:
// the binary-counter tree of the lanes o-bar == the 256-lane halving tree bit for bit, and the lane-major probability
// layout of softmax_kernel<S, true> is a bijection of the kRounds * 256 slots.
#include "attn_core/selected_exact_mla_rule.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace rule = strix::attn_core::selected_exact::rule;

namespace {

int failures = 0;

void expect(bool ok, const char* what, unsigned a, unsigned b) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s (%u, %u)\n", what, a, b);
}

std::uint32_t bits(float x) { return std::bit_cast<std::uint32_t>(x); }

void tree_identity() {
    std::mt19937 rng(20261122U);
    std::uniform_real_distribution<float> mant(-1.0F, 1.0F);
    std::uniform_int_distribution<int> expo(-30, 30);
    const unsigned sizes[] = {1, 2, 31, 255, 256, 257, 511, 1024, 2048, 2051};
    for (unsigned trial = 0; trial < 64; ++trial)
        for (const unsigned n : sizes) {
            std::vector<float> p(n), c(n), w(n);
            for (unsigned t = 0; t < n; ++t) {
                p[t] = std::ldexp(std::fabs(mant(rng)), expo(rng) / 3);
                c[t] = std::ldexp(mant(rng), expo(rng));
                w[t] = std::ldexp(std::fabs(mant(rng)), -expo(rng) / 2);
            }
            const auto ob = rule::obar_lanes(p, c);
            expect(bits(rule::counter_tree(ob)) == bits(rule::halving_tree(ob)), "o-bar counter tree == halving tree", trial, n);
            const auto sum = rule::sum_lanes(w);
            expect(bits(rule::counter_tree(sum)) == bits(rule::halving_tree(sum)), "sum counter tree == halving tree", trial, n);
        }
    // Cancellation and signed zeros: the pairing alone decides the bits.
    std::array<float, rule::kLanes> s{};
    for (unsigned l = 0; l < rule::kLanes; ++l) s[l] = l % 3 == 0 ? 1.0e30F : l % 3 == 1 ? -1.0e30F : 1.0F;
    expect(bits(rule::counter_tree(s)) == bits(rule::halving_tree(s)), "cancellation counter tree == halving tree", 0, 0);
    s.fill(-0.0F);
    expect(bits(rule::counter_tree(s)) == bits(rule::halving_tree(s)), "signed zero counter tree == halving tree", 0, 0);
}

void layout_bijection() {
    const unsigned caps[] = {256, 2051, 2304};
    for (const unsigned cap : caps) {
        const unsigned slots = rule::rounds(cap) * rule::kLanes;
        std::vector<unsigned> hit(slots, 0);
        for (unsigned t = 0; t < slots; ++t) {
            const unsigned s = rule::lane_major_slot(t, cap);
            expect(s < slots, "slot in range", cap, t);
            if (s < slots) ++hit[s];
            // lane k = bitrev8(t % 256) owns rounds contiguous slots, round t / 256 inside them
            expect(s / rule::rounds(cap) == rule::bitrev8(t % rule::kLanes) && s % rule::rounds(cap) == t / rule::kLanes,
                   "slot = rounds * k + round", cap, t);
        }
        for (unsigned s = 0; s < slots; ++s) expect(hit[s] == 1, "slot written exactly once", cap, s);
    }
    expect(rule::rounds(2051) == 9, "a 2051-entry list takes 9 rounds", rule::rounds(2051), 9);
    for (unsigned x = 0; x < 256; ++x) expect(rule::bitrev8(rule::bitrev8(x)) == x, "bitrev8 is an involution", x, 0);
}

}  // namespace

int main() {
    tree_identity();
    layout_bijection();
    if (failures != 0) {
        std::fprintf(stderr, "selected_exact_mla_tests: %d failures\n", failures);
        return 1;
    }
    std::printf("selected_exact_mla_tests: OK\n");
    return 0;
}
