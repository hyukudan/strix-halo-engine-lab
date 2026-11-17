// Host test of the exact QSA fused attention's index and order invariants (src/cuda/models/qwen/qsa_fused_order.hpp,
// qwen_qsa.cu attention_fused_kernel), for every selected count 1..2051:
//  1. the softmax denominator: one warp (lane lam holds reference lanes lam + 32 j, then a shuffle-down tree) gives
//     the same bits as softmax_kernel_body's 256 lanes + LDS tree on adversarial e values; a plain sequential sum must
//     differ for some counts (the check can see order changes);
//  2. the e pass visits each slot exactly once, from the lane that also owns it in the weight pass (i % 32 == lam);
//  3. the score phase (slot = 12 i + 2 sp + half, i < ceil(count / 12)) covers each slot < count exactly once.
#include "qsa_fused_order.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace fo = strix::cuda::qwen_qsa::fused_order;

namespace {
constexpr unsigned kMaxSelected = 2051;   // qwen_qsa.hpp kMaxSelected (2048 + kRatio - 1)
int failures = 0;
unsigned owner_bad = 0;

std::uint32_t bits(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return u; }

float reference_den(const std::vector<float>& e, unsigned count) {   // softmax_kernel_body, literally
    float scratch[256];
    for (unsigned v = 0; v < 256U; ++v) {
        float s = 0.0F;
        for (unsigned i = v; i < count; i += 256U) s = s + e[i];
        scratch[v] = s;
    }
    for (unsigned stride = 128U; stride > 0U; stride >>= 1U)
        for (unsigned v = 0; v < stride; ++v) scratch[v] = scratch[v] + scratch[v + stride];
    return scratch[0];
}

float fused_den(const std::vector<float>& e, unsigned count, std::vector<unsigned>* visits) {
    float x[32];
    const auto add = [](float a, float b) { return a + b; };
    for (unsigned lam = 0; lam < 32U; ++lam)
        x[lam] = fo::lane_partial(lam, count, add, [&](unsigned i) {
            if (visits) { ++(*visits)[i]; if (i % 32U != lam) ++owner_bad; }
            return e[i];
        });
    for (unsigned d = 16U; d > 0U; d >>= 1U) {   // __shfl_down_sync: lanes past the end read their own value
        float y[32];
        for (unsigned lam = 0; lam < 32U; ++lam) y[lam] = x[lam] + (lam + d < 32U ? x[lam + d] : x[lam]);
        std::memcpy(x, y, sizeof x);
    }
    return x[0];
}
}  // namespace

int main() {
    std::mt19937 rng(20261117);
    std::uniform_real_distribution<float> u(0.0F, 1.0F);
    std::vector<float> e(kMaxSelected);
    unsigned den_bad = 0, seq_differs = 0, visit_bad = 0, score_bad = 0;
    for (int trial = 0; trial < 3; ++trial) {
        for (auto& x : e) {   // e = expf(s - max) in (0, 1]: wide exponent spread so every rounding step matters
            const float m = u(rng);
            x = trial == 0 ? m : (trial == 1 ? std::ldexp(1.0F + m, -static_cast<int>(rng() % 40U)) : (rng() % 7U == 0 ? 1.0F : m * 1e-3F));
        }
        for (unsigned count = 1; count <= kMaxSelected; ++count) {
            std::vector<unsigned> visits(count, 0U);
            const float ref = reference_den(e, count);
            const float got = fused_den(e, count, trial == 0 ? &visits : nullptr);
            if (bits(ref) != bits(got)) {
                if (den_bad++ < 5) std::printf("den mismatch trial %d count %u: ref %08x got %08x\n", trial, count, bits(ref), bits(got));
            }
            float seq = 0.0F;
            for (unsigned i = 0; i < count; ++i) seq = seq + e[i];
            if (bits(seq) != bits(ref)) ++seq_differs;
            if (trial == 0) {
                for (unsigned i = 0; i < count; ++i) visit_bad += visits[i] != 1U;
                std::vector<unsigned> cover(count, 0U);
                const unsigned iters = (count + 11U) / 12U;
                for (unsigned i = 0; i < iters; ++i)
                    for (unsigned sp = 0; sp < 6U; ++sp)
                        for (unsigned half = 0; half < 2U; ++half) {
                            const unsigned slot = 12U * i + 2U * sp + half;
                            if (slot < count) ++cover[slot];
                        }
                for (unsigned i = 0; i < count; ++i) score_bad += cover[i] != 1U;
            }
        }
    }
    const auto report = [](const char* what, bool ok) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) ++failures; };
    report("denominator bits == softmax_kernel_body (3 x 2051 counts)", den_bad == 0U);
    report("sequential sum differs for some counts (check is order-sensitive)", seq_differs > 100U);
    report("e pass: every slot once, owner lane == i % 32", visit_bad == 0U && owner_bad == 0U);
    report("score phase covers every slot < count once", score_bad == 0U);
    std::printf("den mismatches %u, sequential-sum differences %u of %u\n", den_bad, seq_differs, 3U * kMaxSelected);
    std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
