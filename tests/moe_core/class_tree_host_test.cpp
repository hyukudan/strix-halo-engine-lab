// Host test of the lane-class tree contract (src/moe_core/class_tree.hpp): the in-register carry-stack fold and the
// cross-wave placements of the tree (classes 4q + k per wave, q + 8k per wave, two passes over class bit 0) produce the
// bits of the definition reference(), for finite values of mixed magnitude, signed zeros and infinities.
#include "moe_core/class_tree.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

namespace ct = strix::moe_core::class_tree;
constexpr unsigned kN = ct::kClasses;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

std::uint32_t bits(float v) {
    std::uint32_t u = 0U;
    std::memcpy(&u, &v, sizeof u);
    return u;
}

struct Rng {
    std::uint64_t s;
    std::uint32_t next() {
        s ^= s << 13U;
        s ^= s >> 7U;
        s ^= s << 17U;
        return static_cast<std::uint32_t>(s >> 32U);
    }
};

// Mixed magnitudes (cancellation and absorption), signed zeros and +inf (no -inf: no NaN is produced).
float sample(Rng& r) {
    const std::uint32_t kind = r.next() % 64U;
    if (kind == 0U) return 0.0F;
    if (kind == 1U) return -0.0F;
    if (kind == 2U) return std::numeric_limits<float>::infinity();
    const float mant = 1.0F + static_cast<float>(r.next() & 0xffffffU) * 0x1p-24F;
    const int exp = static_cast<int>(r.next() % 61U) - 30;
    const float v = std::ldexp(mant, exp);
    return (r.next() & 1U) != 0U ? -v : v;
}

// The wave butterfly v[l] += v[l ^ d] (all lanes at once), lane 0.
float butterfly(const float (&x)[kN]) {
    float v[kN];
    for (unsigned l = 0; l < kN; ++l) v[l] = x[l];
    for (unsigned d = 16U; d != 0U; d /= 2U) {
        float n[kN];
        for (unsigned l = 0; l < kN; ++l) n[l] = v[l] + v[l ^ d];
        for (unsigned l = 0; l < kN; ++l) v[l] = n[l];
    }
    return v[0];
}

// Eight waves, wave q holds classes 4q + k; LDS levels s = 4, 2, 1 (class bits 4, 3, 2), then (k0 + k2) + (k1 + k3).
float classes_per_wave_4(const float (&x)[kN]) {
    float leaf[8][4];
    for (unsigned q = 0; q < 8U; ++q)
        for (unsigned k = 0; k < 4U; ++k) leaf[q][k] = x[4U * q + k];
    for (unsigned s = 4U; s != 0U; s /= 2U)
        for (unsigned q = 0; q < s; ++q)
            for (unsigned k = 0; k < 4U; ++k) leaf[q][k] = leaf[q][k] + leaf[q + s][k];
    const float c0 = leaf[0][0] + leaf[0][2], c1 = leaf[0][1] + leaf[0][3];
    return c0 + c1;
}

// Eight waves, wave q holds classes q + 8k: ((q + q+16) + (q+8 + q+24)) in registers, then wave bits s = 4, 2, 1.
float classes_strided_8(const float (&x)[kN]) {
    float v[8];
    for (unsigned q = 0; q < 8U; ++q) {
        const float c0 = x[q] + x[q + 16U], c1 = x[q + 8U] + x[q + 24U];
        v[q] = c0 + c1;
    }
    for (unsigned s = 4U; s != 0U; s /= 2U)
        for (unsigned q = 0; q < s; ++q) v[q] = v[q] + v[q + s];
    return v[0];
}

// Two passes p = class bit 0; wave q holds p + 2q + 16k: level 1 in registers, wave bits s = 4, 2, 1, then even + odd.
float two_passes(const float (&x)[kN]) {
    float out = 0.0F;
    for (unsigned p = 0; p < 2U; ++p) {
        float v[8];
        for (unsigned q = 0; q < 8U; ++q) v[q] = x[p + 2U * q] + x[p + 2U * q + 16U];
        for (unsigned s = 4U; s != 0U; s /= 2U)
            for (unsigned q = 0; q < s; ++q) v[q] = v[q] + v[q + s];
        out = p == 0U ? v[0] : out + v[0];
    }
    return out;
}

float sequential(const float (&x)[kN]) {
    float s = 0.0F;
    for (unsigned c = 0; c < kN; ++c) s = s + x[c];
    return s;
}

constexpr unsigned kMT = 2U, kE = 3U;

// The in-register placement over MT x E independent trees; stack starts as NaN (every level is written before read).
void stage_fold(const float (&x)[kMT][kE][kN], float (&v)[kMT][kE]) {
    float stack[ct::kLevels][kMT][kE];
    for (auto& level : stack)
        for (auto& row : level)
            for (float& e : row) e = std::numeric_limits<float>::quiet_NaN();
    for (unsigned i = 0; i < kN; ++i) {
        float p[kMT][kE];
        for (unsigned t = 0; t < kMT; ++t)
            for (unsigned e = 0; e < kE; ++e) p[t][e] = x[t][e][ct::class_of(i)];
        ct::fold(i, p, stack, v);
    }
}

} // namespace

int main() {
    // Visiting order: a permutation, positions 2m / 2m + 1 are classes c / c + 16, the 5-bit reversal.
    {
        unsigned seen = 0U;
        bool rev = true, pairs = true;
        for (unsigned i = 0; i < kN; ++i) {
            const unsigned c = ct::class_of(i);
            seen |= 1U << c;
            unsigned r = 0U;
            for (unsigned b = 0; b < 5U; ++b) r |= ((i >> b) & 1U) << (4U - b);
            rev = rev && c == r;
            if ((i & 1U) != 0U) pairs = pairs && c == ct::class_of(i - 1U) + 16U;
        }
        check(seen == 0xffffffffU, "class_of is a permutation of 0..31");
        check(rev, "class_of(i) == rev5(i)");
        check(pairs, "positions 2m, 2m + 1 are classes c, c + 16");
        check(ct::class_of(1U) == 16U && ct::class_of(2U) == 8U && ct::class_of(3U) == 24U && ct::class_of(31U) == 31U,
              "class_of spot values");
    }

    Rng rng{0x9e3779b97f4a7c15ULL};
    unsigned order_sensitive = 0U;
    constexpr unsigned kCases = 20000U;
    bool fold_ok = true, bfly_ok = true, w4_ok = true, s8_ok = true, p2_ok = true;
    for (unsigned n = 0; n < kCases; ++n) {
        float x[kMT][kE][kN];
        for (auto& row : x)
            for (auto& tree : row)
                for (float& c : tree) c = sample(rng);
        float v[kMT][kE];
        stage_fold(x, v);
        for (unsigned t = 0; t < kMT; ++t)
            for (unsigned e = 0; e < kE; ++e) {
                const float(&tree)[kN] = x[t][e];
                const std::uint32_t ref = bits(ct::reference(tree));
                fold_ok = fold_ok && bits(v[t][e]) == ref;
                bfly_ok = bfly_ok && bits(butterfly(tree)) == ref;
                w4_ok = w4_ok && bits(classes_per_wave_4(tree)) == ref;
                s8_ok = s8_ok && bits(classes_strided_8(tree)) == ref;
                p2_ok = p2_ok && bits(two_passes(tree)) == ref;
                if (bits(sequential(tree)) != ref) ++order_sensitive;
            }
    }
    check(fold_ok, "carry-stack fold (bit-reversed walk) == reference");
    check(bfly_ok, "wave butterfly lane 0 == reference");
    check(w4_ok, "classes 4q + k per wave, LDS levels 4,2,1 then (0+2)+(1+3) == reference");
    check(s8_ok, "classes q + 8k per wave, (q+q16)+(q8+q24) then wave levels == reference");
    check(p2_ok, "two passes over class bit 0, even + odd == reference");
    // The data must be able to tell orders apart, or the equalities above prove nothing.
    check(order_sensitive > kCases / 4U, "inputs distinguish the tree from a sequential sum");

    if (failures != 0) {
        std::printf("class_tree_host_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("class_tree_host_test: PASS (%u trees, %u order-sensitive)\n", kCases * kMT * kE, order_sensitive);
    return 0;
}
