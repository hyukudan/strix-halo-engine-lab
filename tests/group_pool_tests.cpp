// Host tests of the shared group-pool statements (include/strix/attn/group_pool.hpp), the code the HIP block core
// (src/attn_core/group_pool.hpp) and GLM's pool kernels run.
// Invariants:
//  1. MODEL_B GatedSoftmaxApe: the lane composition of the block core (gated_ape_channel per lane, one rms_inv chain,
//     rms_out) is bit-identical to the host oracle pool_group_gated_ape (src/attn/group_indexer_ref.cpp).
//  2. GLM GatedSoftmax: gated_bf16_channel (decode site) and gated_pool_channel on loaded arrays (prefill / spec sites)
//     are bit-identical to the statements GLM's private dsa_pool_channel ran before the lift (frozen copy below).
//  3. Tie rule: GLM's ordered_key is score_key on every bit pattern tested, and score_key orders scores exactly as float
//     comparison (equal scores incl. -0 / +0 -> equal keys), which is what select.hpp's "equal key -> lower index" uses.
#include "strix/attn/group_pool.hpp"

#include "strix/ds41_msvc_builtins.hpp"  // GLM math uses GCC builtins; no-op outside MSVC
#include "glm/common/glm_math.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace gi = strix::attn::group_indexer;
// Gated-pool AlwaysOwn indexer shape exercised here (ratio 4, 16 heads x 128, 512 groups); a model binding keeps its
// own Shape in its model package.
constexpr gi::Shape kGatedApe{4U, 16U, 128U, 512U, gi::Pooling::GatedSoftmaxApe, gi::OwnGroup::AlwaysOwn,
                              gi::ScoreScale::InHeadWeight};
namespace gp = strix::attn::group_pool;

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
float exp_declared(float x) { return strix::glm::declared_exp(x); }

// The statements of the deleted src/glm/common/glm_dsa_pool.hpp (frozen: the reference the lift must keep).
float legacy_glm_pool_channel(const std::uint16_t* key, const std::uint16_t* gate, const float* ape, unsigned channel) {
    constexpr unsigned kKpool = 4, kIdxDim = 128;
    float logits[kKpool], values[kKpool], exponent[kKpool], probability[kKpool];
    for (unsigned j = 0; j < kKpool; ++j) {
        const unsigned offset = j * kIdxDim + channel;
        logits[j] = strix::glm::bf16_to_float(gate[offset]) + ape[offset];
        values[j] = strix::glm::bf16_to_float(key[offset]);
    }
    float maximum = logits[0];
    for (unsigned j = 1; j < kKpool; ++j) maximum = logits[j] > maximum ? logits[j] : maximum;
    for (unsigned j = 0; j < kKpool; ++j) exponent[j] = strix::glm::declared_exp(logits[j] - maximum);
    const float sum = ((exponent[0] + exponent[1]) + exponent[2]) + exponent[3];
    for (unsigned j = 0; j < kKpool; ++j) probability[j] = exponent[j] / sum;
    return ((probability[0] * values[0] + probability[1] * values[1]) + probability[2] * values[2]) +
           probability[3] * values[3];
}

void test_modelb_lanes_vs_oracle(std::mt19937& rng) {
    const gi::Shape s = kGatedApe;
    constexpr unsigned R = 4, D = 128;
    static_assert(kGatedApe.ratio == R && kGatedApe.head_dim == D, "MODEL_B pools 4 slots of 128");
    std::normal_distribution<float> n01(0.0F, 1.0F);
    std::vector<float> keys(R * D), wv(D * D), wg(D * D), ape(R * D), norm(D), lane_x(D), oracle(D);
    unsigned same = 0, total = 0;
    for (int rep = 0; rep < 48; ++rep) {
        const float gate_spread = rep % 6 == 0 ? 2.0F : 0.1F;  // rep % 6 == 0: saturated softmax channels
        for (float& x : keys) x = gi::bf16_round(n01(rng));   // cached token keys are bf16
        for (float& x : wv) x = n01(rng) * 0.09F;
        for (float& x : wg) x = n01(rng) * gate_spread;
        for (float& x : ape) x = n01(rng);
        for (float& x : norm) x = 1.0F + 0.1F * n01(rng);
        const gi::GatedApeWeights w{wv.data(), wg.data(), ape.data(), norm.data(), 1e-5F};
        gi::pool_group_gated_ape(s, keys.data(), w, exp_declared, oracle.data());
        const gp::F32Ape view{w, D};
        for (unsigned o = 0; o < D; ++o) lane_x[o] = gp::gated_ape_channel<R>(keys.data(), D, view, o, exp_declared);
        const float r = gp::rms_inv(lane_x.data(), D, view.eps());
        for (unsigned o = 0; o < D; ++o) {
            const float out = gp::rms_out(view.norm(o), lane_x[o], r);
            same += bits(out) == bits(oracle[o]) ? 1U : 0U;
            ++total;
            CHECK(bits(out) == bits(gi::bf16_round(out)));  // bf16-valued
        }
    }
    std::printf("modelb lanes vs oracle: %u / %u bit-identical\n", same, total);
    CHECK(same == total);
}

void test_glm_legacy_identity(std::mt19937& rng) {
    constexpr unsigned R = 4, D = 128;
    std::normal_distribution<float> n01(0.0F, 1.0F);
    std::vector<std::uint16_t> key(R * D), gate(R * D);
    std::vector<float> ape(R * D);
    unsigned same = 0, total = 0;
    for (int rep = 0; rep < 256; ++rep) {
        const float spread = rep % 8 == 0 ? 60.0F : 3.0F;
        for (auto& x : key) x = bf16_bits(n01(rng) * 4.0F);
        for (auto& x : gate) x = bf16_bits(n01(rng) * spread);
        for (float& x : ape) x = n01(rng);
        for (unsigned c = 0; c < D; ++c) {
            const float legacy = legacy_glm_pool_channel(key.data(), gate.data(), ape.data(), c);
            const float decode = gp::gated_bf16_channel<R>(key.data(), gate.data(), ape.data(), c, D, exp_declared);
            float l4[R], kv[R];  // the prefill / spec sites: loads stay in the engine, arithmetic is shared
            for (unsigned j = 0; j < R; ++j) {
                l4[j] = strix::glm::bf16_to_float(gate[j * D + c]) + ape[j * D + c];
                kv[j] = strix::glm::bf16_to_float(key[j * D + c]);
            }
            const float rows = gi::gated_pool_channel<R>(l4, kv, [](float x) { return strix::glm::declared_exp(x); });
            same += (bits(decode) == bits(legacy) && bits(rows) == bits(legacy)) ? 1U : 0U;
            ++total;
        }
    }
    std::printf("glm pool sites vs legacy statements: %u / %u bit-identical\n", same, total);
    CHECK(same == total);
}

void test_tie_keys(std::mt19937& rng) {
    std::vector<float> special = {0.0F, -0.0F, 1.0F, -1.0F, std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::denorm_min(),
                                  -std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
                                  -std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};
    unsigned bad = 0;
    for (float x : special) bad += strix::glm::ordered_key(x) == gi::score_key(x) ? 0U : 1U;
    std::uniform_int_distribution<std::uint32_t> u32;
    for (int i = 0; i < 1 << 20; ++i) {
        const float x = __builtin_bit_cast(float, u32(rng));
        bad += strix::glm::ordered_key(x) == gi::score_key(x) ? 0U : 1U;
    }
    CHECK(bad == 0U);
    // Order: key(a) < key(b) <=> a < b; key(a) == key(b) <=> a == b (finite and infinite scores; NaN is rejected upstream).
    std::normal_distribution<float> n01(0.0F, 1.0F);
    unsigned order_bad = 0;
    for (int i = 0; i < 1 << 18; ++i) {
        float a = n01(rng), b = i % 4 == 0 ? a : n01(rng);
        if (i % 97 == 0) { a = 0.0F; b = -0.0F; }
        const std::uint32_t ka = gi::score_key(a), kb = gi::score_key(b);
        order_bad += ((a < b) == (ka < kb) && (a == b) == (ka == kb)) ? 0U : 1U;
    }
    CHECK(order_bad == 0U);
    std::printf("tie keys: ordered_key == score_key mismatches %u, order mismatches %u\n", bad, order_bad);
}
}  // namespace

int main() {
    std::mt19937 rng(20261118U);
    test_modelb_lanes_vs_oracle(rng);
    test_glm_legacy_identity(rng);
    test_tie_keys(rng);
    if (g_failures != 0) {
        std::fprintf(stderr, "group_pool_tests: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("group_pool_tests: PASS\n");
    return 0;
}
