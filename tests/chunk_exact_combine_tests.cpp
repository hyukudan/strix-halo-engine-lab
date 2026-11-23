// Shared EXACT chunk-partial combine (src/attn_core/chunk_exact_combine.hpp): the host rule equals, bit for bit,
// (1) a host emulation of the device lane form (lane cc holds chunk cc's m, l; readlane broadcast; acc preloaded with
// zero tails) and (2) the per-thread scalar form (exp evaluated once per chunk into e[cc]). Random partials, every
// cnt 1 .. kMaxChunks, poisoned entries past cnt, exponent spreads that underflow e.
#include "attn_core/chunk_exact_combine_rule.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr unsigned kDim = 128, kMaxChunks = 32, kPart = 2 + kDim, kDpl = kDim / 32;

float exp_decl(float x) { return std::exp(x); }

void lane_form(const float* pt, unsigned cnt, float* o) {
    float mc[32], lc[32], ec_own[32];
    for (unsigned lane = 0; lane < 32; ++lane) {
        mc[lane] = lane < cnt ? pt[lane * kPart] : -INFINITY;
        lc[lane] = lane < cnt ? pt[lane * kPart + 1] : 0.0F;
    }
    float big = -INFINITY;
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        if (cc < cnt) big = big < mc[cc] ? mc[cc] : big;
    for (unsigned lane = 0; lane < 32; ++lane) ec_own[lane] = exp_decl(mc[lane] - big);
    float L = 0.0F, e[kMaxChunks];
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        if (cc < cnt) {
            e[cc] = ec_own[cc];
            L = std::fma(lc[cc], e[cc], L);
        }
    for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned i = 0; i < kDpl; ++i) {
            float a = 0.0F;
            for (unsigned cc = 0; cc < kMaxChunks; ++cc) {
                const float ac = cc < cnt ? pt[cc * kPart + 2 + lane * kDpl + i] : 0.0F;
                if (cc < cnt) a = std::fma(ac, e[cc], a);
            }
            o[lane * kDpl + i] = a / L;
        }
}

void scalar_form(const float* pt, unsigned cnt, float* o) {
    float big = -INFINITY;
    for (unsigned cc = 0; cc < cnt; ++cc) big = big < pt[cc * kPart] ? pt[cc * kPart] : big;
    float L = 0.0F, e[kMaxChunks];
    for (unsigned cc = 0; cc < cnt; ++cc) {
        e[cc] = exp_decl(pt[cc * kPart] - big);
        L = std::fma(pt[cc * kPart + 1], e[cc], L);
    }
    for (unsigned d = 0; d < kDim; ++d) {
        float a = 0.0F;
        for (unsigned cc = 0; cc < cnt; ++cc) a = std::fma(pt[cc * kPart + 2 + d], e[cc], a);
        o[d] = a / L;
    }
}

bool same(const float* a, const float* b) { return std::memcmp(a, b, kDim * sizeof(float)) == 0; }

} // namespace

int main() {
    std::mt19937 rng(20261122U);
    std::vector<float> pt(kMaxChunks * kPart);
    float r[kDim], l[kDim], s[kDim];
    unsigned cases = 0, bad = 0;
    for (unsigned trial = 0; trial < 400; ++trial) {
        const float spread = trial % 4 == 0 ? 120.0F : (trial % 4 == 1 ? 8.0F : 1.0F);
        std::uniform_real_distribution<float> mdist(-spread, spread), ldist(1.0F, 128.0F), adist(-64.0F, 64.0F);
        for (unsigned cnt = 1; cnt <= kMaxChunks; ++cnt) {
            for (unsigned cc = 0; cc < kMaxChunks; ++cc) {
                float* p = pt.data() + cc * kPart;
                if (cc >= cnt) {  // never read for results
                    for (unsigned j = 0; j < kPart; ++j) p[j] = NAN;
                    continue;
                }
                p[0] = mdist(rng);
                p[1] = ldist(rng);
                for (unsigned d = 0; d < kDim; ++d) p[2 + d] = adist(rng);
            }
            strix::attn_core::chunk_exact::combine_rule(pt.data(), cnt, kDim, exp_decl, r);
            lane_form(pt.data(), cnt, l);
            scalar_form(pt.data(), cnt, s);
            ++cases;
            if (!same(r, l) || !same(r, s)) ++bad;
        }
    }
    // Empty row (cnt == 0): the rule returns before touching the output (canary stays; no 0 / 0 NaN).
    float canary[kDim];
    for (unsigned d = 0; d < kDim; ++d) r[d] = canary[d] = 1234.5F;
    strix::attn_core::chunk_exact::combine_rule(pt.data(), 0U, kDim, exp_decl, r);
    if (!same(r, canary)) {
        std::printf("chunk_exact_combine: empty row overwrote the output\n");
        return 1;
    }
    std::printf("chunk_exact_combine: %u cases, %u mismatches\n", cases, bad);
    return bad == 0 && cases == 400U * kMaxChunks ? 0 : 1;
}
