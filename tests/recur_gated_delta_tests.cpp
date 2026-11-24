// F9 recur gated_delta step policies (src/ops/recur/gated_delta_policy.hpp): host invariants.
// 1. PerKeyChannelGate (KDA) through the shared step order and the shared GDN operand layout (rows flattened into value
//    heads) equals, bit for bit, a transcription of the per-(row, head) KDA decode update (state f32 [k][v], per column:
//    s = S * exp(g[k]); u = tree(0 + s k); d = (v - u) beta; s = fmaf(k, d, s); o = tree(0 + s q)) on states, outputs
//    and the non-finite flag, including signed zeros and subnormals.
// 2. The zero seed is load-bearing: plain products give a different sign of zero on an all -0 column.
// 3. ScalarGate equals the GDN transcription (scalar exp gate, plain products, s + k d) under the same host flags.
// Build: -ffp-contract=off -fno-fast-math (the device binders' contracts are proven by ISA comparison, not here).
#include "ops/recur/gated_delta_policy.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

namespace recur = strix::ops::recur;
using strix::moe::declared_math::declared_exp;

constexpr std::uint32_t kD = 128;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

std::uint32_t bits(float x) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (bits(a[i]) != bits(b[i])) return false;
    return true;
}

struct Rng {
    std::uint64_t s;
    std::uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(s >> 33U);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() & 0xFFFFFFU) / 16777216.0F; }
    // Mostly ordinary values; some +-0, subnormals and tiny products that round to -0.
    float value(float scale) {
        const std::uint32_t c = next() % 32U;
        if (c == 0) return 0.0F;
        if (c == 1) return -0.0F;
        if (c == 2) return (next() & 1U ? 1.0F : -1.0F) * 1.0e-39F;
        if (c == 3) return (next() & 1U ? 1.0F : -1.0F) * 1.0e-30F;
        return uniform(-scale, scale);
    }
};

float halving(float* t, std::uint32_t n) {
    for (std::uint32_t h = n / 2U; h > 0; h >>= 1U)
        for (std::uint32_t l = 0; l < h; ++l) t[l] = t[l] + t[l + h];
    return t[0];
}

// KDA decode update, per (row, head), scratch[row][head] = q | k | v | g (4 x D), beta[row][head], S[row][head][k][v].
void kda_update_transcription(std::uint32_t rows, std::uint32_t heads, std::vector<float>& S, const std::vector<float>& scratch,
                              const std::vector<float>& beta, std::vector<float>& o, unsigned& flag) {
    float t[kD];
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t h = 0; h < heads; ++h) {
            const float* sc = scratch.data() + (static_cast<std::size_t>(r) * heads + h) * 4 * kD;
            const float bt = beta[static_cast<std::size_t>(r) * heads + h];
            float* Sh = S.data() + (static_cast<std::size_t>(r) * heads + h) * kD * kD;
            for (std::uint32_t vv = 0; vv < kD; ++vv) {
                for (std::uint32_t kk = 0; kk < kD; ++kk) {
                    const float sp = Sh[kk * kD + vv] * declared_exp(sc[3 * kD + kk]);
                    t[kk] = 0.0F + sp * sc[kD + kk];
                }
                const float u = halving(t, kD);
                const float dlt = (sc[2 * kD + vv] - u) * bt;
                for (std::uint32_t kk = 0; kk < kD; ++kk) {
                    const float sp = Sh[kk * kD + vv] * declared_exp(sc[3 * kD + kk]);
                    const float sn = std::fmaf(sc[kD + kk], dlt, sp);
                    if (!std::isfinite(sn)) flag |= 1U;
                    Sh[kk * kD + vv] = sn;
                    t[kk] = 0.0F + sn * sc[kk];
                }
                o[(static_cast<std::size_t>(r) * heads + h) * kD + vv] = halving(t, kD);
            }
        }
}

// The same step through the shared policy and the GDN layout: value head vh = row * heads + head, key heads = value heads.
template <class Policy>
void shared_step(const Policy& policy, std::uint32_t rows, std::uint32_t heads, std::vector<float>& S,
                 const std::vector<float>& scratch, const std::vector<float>& beta, bool gate_is_factor, std::vector<float>& o) {
    const std::uint32_t vh_count = rows * heads;
    std::vector<float> qkv(static_cast<std::size_t>(3) * vh_count * kD), gate(static_cast<std::size_t>(vh_count) * kD);
    for (std::uint32_t vh = 0; vh < vh_count; ++vh)
        for (std::uint32_t i = 0; i < kD; ++i) {
            const float* sc = scratch.data() + static_cast<std::size_t>(vh) * 4 * kD;
            qkv[static_cast<std::size_t>(vh) * kD + i] = sc[i];
            qkv[(static_cast<std::size_t>(vh_count) + vh) * kD + i] = sc[kD + i];
            qkv[(2 * static_cast<std::size_t>(vh_count) + vh) * kD + i] = sc[2 * kD + i];
            gate[static_cast<std::size_t>(vh) * kD + i] = gate_is_factor ? declared_exp(sc[3 * kD + i]) : sc[3 * kD + i];
        }
    for (std::uint32_t vh = 0; vh < vh_count; ++vh) {
        const std::uint32_t kh = vh % vh_count;
        recur::gated_delta_step_ref(policy, qkv.data() + static_cast<std::size_t>(kh) * kD,
                                    qkv.data() + (static_cast<std::size_t>(vh_count) + kh) * kD,
                                    qkv.data() + (2 * static_cast<std::size_t>(vh_count) + vh) * kD, beta[vh], gate.data(), vh, kD,
                                    S.data() + static_cast<std::size_t>(vh) * kD * kD, o.data() + static_cast<std::size_t>(vh) * kD);
    }
}

void test_kda_equals_transcription(std::uint64_t seed, bool poison) {
    const std::uint32_t rows = 2, heads = 3;
    Rng rng{seed};
    std::vector<float> S(static_cast<std::size_t>(rows) * heads * kD * kD), scratch(static_cast<std::size_t>(rows) * heads * 4 * kD),
        beta(static_cast<std::size_t>(rows) * heads);
    for (auto& x : S) x = rng.value(2.0F);
    for (std::size_t rh = 0; rh < beta.size(); ++rh) {
        float* sc = scratch.data() + rh * 4 * kD;
        for (std::uint32_t i = 0; i < 3 * kD; ++i) sc[i] = rng.value(1.0F);
        for (std::uint32_t i = 0; i < kD; ++i) sc[3 * kD + i] = -5.0F * rng.uniform(0.0F, 1.0F);  // low * sigmoid(.)
        beta[rh] = rng.uniform(0.0F, 1.0F);
    }
    // One all-zero-term column per head: k = -0 on every lane; one value -0 (the sign-of-zero path).
    for (std::size_t rh = 0; rh < beta.size(); ++rh) scratch[rh * 4 * kD + 2 * kD + 5] = -0.0F;
    if (poison) S[7] = std::numeric_limits<float>::infinity();
    std::vector<float> S_ref = S, S_sh = S, S_fac = S, o_ref(beta.size() * kD), o_sh(o_ref.size()), o_fac(o_ref.size());
    unsigned flag_ref = 0, flag_sh = 0, flag_fac = 0;
    kda_update_transcription(rows, heads, S_ref, scratch, beta, o_ref, flag_ref);
    shared_step(recur::PerKeyChannelGate<recur::DeclaredExp>{&flag_sh}, rows, heads, S_sh, scratch, beta, false, o_sh);
    shared_step(recur::PerKeyChannelGate<recur::FactorExp>{&flag_fac}, rows, heads, S_fac, scratch, beta, true, o_fac);
    check(same_bits(S_ref, S_sh) && same_bits(o_ref, o_sh), "KDA (declared exp) state/out bitwise");
    check(same_bits(S_ref, S_fac) && same_bits(o_ref, o_fac), "KDA (precomputed factor) state/out bitwise");
    check(flag_ref == flag_sh && flag_ref == flag_fac && flag_ref == (poison ? 1U : 0U), "KDA non-finite flag");
}

// State -0, key +0, value -0: plain products give u = -0, d = (-0 - -0) b = +0, S' = fmaf(+0, +0, -0) = +0; the seeded
// terms give u = +0, d = (-0 - +0) b = -0, S' = fmaf(+0, -0, -0) = -0. The seed is part of the KDA bits.
void test_zero_seed_is_load_bearing() {
    std::vector<float> q(kD, 1.0F), k(kD, 0.0F), v(kD, -0.0F), gate(kD, -1.0F), s1(static_cast<std::size_t>(kD) * kD, -0.0F);
    std::vector<float> s2 = s1, o1(kD), o2(kD);
    struct Plain : recur::PerKeyChannelGate<recur::DeclaredExp> {
        float lane_term(float s, float x) const { return s * x; }
    };
    recur::gated_delta_step_ref(recur::PerKeyChannelGate<recur::DeclaredExp>{}, q.data(), k.data(), v.data(), 0.5F, gate.data(), 0, kD,
                                s1.data(), o1.data());
    recur::gated_delta_step_ref(Plain{}, q.data(), k.data(), v.data(), 0.5F, gate.data(), 0, kD, s2.data(), o2.data());
    check(bits(s1[0]) == bits(-0.0F) && bits(s2[0]) == bits(0.0F), "zero seed decides the sign of a zero state");
    check(same_bits(o1, o2), "outputs of the all-zero column agree (+0)");
}

// ScalarGate: the GDN step transcription (scalar exp gate per value head, plain products, retained += k * delta).
void test_scalar_gate_equals_gdn() {
    const std::uint32_t key_heads = 2, value_heads = 4;
    Rng rng{99};
    std::vector<float> qkv(static_cast<std::size_t>(2 * key_heads + value_heads) * kD), beta(value_heads), gate(value_heads),
        S(static_cast<std::size_t>(value_heads) * kD * kD);
    for (auto& x : qkv) x = rng.value(1.0F);
    for (auto& x : S) x = rng.value(2.0F);
    for (std::uint32_t vh = 0; vh < value_heads; ++vh) {
        beta[vh] = rng.uniform(0.0F, 1.0F);
        gate[vh] = rng.uniform(-3.0F, 0.0F);
    }
    std::vector<float> S_ref = S, o_ref(static_cast<std::size_t>(value_heads) * kD), o_sh(o_ref.size());
    float t[kD], retained[kD];
    for (std::uint32_t vh = 0; vh < value_heads; ++vh) {
        const std::uint32_t kh = vh % key_heads;
        const float* qh = qkv.data() + static_cast<std::size_t>(kh) * kD;
        const float* kh_ = qkv.data() + (static_cast<std::size_t>(key_heads) + kh) * kD;
        const float* vh_ = qkv.data() + (2 * static_cast<std::size_t>(key_heads) + vh) * kD;
        float* Sh = S_ref.data() + static_cast<std::size_t>(vh) * kD * kD;
        for (std::uint32_t col = 0; col < kD; ++col) {
            for (std::uint32_t l = 0; l < kD; ++l) {
                retained[l] = Sh[l * kD + col] * expf(gate[vh]);
                t[l] = retained[l] * kh_[l];
            }
            const float delta = (vh_[col] - halving(t, kD)) * beta[vh];
            for (std::uint32_t l = 0; l < kD; ++l) {
                retained[l] += kh_[l] * delta;
                Sh[l * kD + col] = retained[l];
                t[l] = retained[l] * qh[l];
            }
            o_ref[static_cast<std::size_t>(vh) * kD + col] = halving(t, kD);
        }
        recur::gated_delta_step_ref(recur::ScalarGate{}, qh, kh_, vh_, beta[vh], gate.data(), vh, kD,
                                    S.data() + static_cast<std::size_t>(vh) * kD * kD, o_sh.data() + static_cast<std::size_t>(vh) * kD);
    }
    check(same_bits(S_ref, S) && same_bits(o_ref, o_sh), "ScalarGate equals the GDN transcription");
}

// 4. Rows form addressing (gated_delta_rows_body.inc): row r reads qkv + r * qkv_stride, beta + r * value_heads and
//    gate + r * policy.gate_row_stride and carries the state tile from row to row; the result equals `rows` consecutive
//    single-row steps on the gathered operands, for both policies (key heads == value heads here).
template <class Policy>
void test_rows_form(const Policy& policy, std::uint64_t seed) {
    constexpr std::uint32_t rows = 5, heads = 2;
    const std::uint32_t gate_stride = static_cast<std::uint32_t>(policy.gate_row_stride(heads, kD));
    const std::uint64_t qkv_stride = 3ULL * heads * kD + 7U;  // padded row stride
    Rng rng{seed};
    std::vector<float> qkv(rows * qkv_stride), beta(rows * heads), gate(static_cast<std::size_t>(rows) * gate_stride);
    for (auto& x : qkv) x = rng.value(1.0F);
    for (auto& x : beta) x = rng.uniform(0.0F, 1.0F);
    for (auto& x : gate) x = rng.uniform(-2.0F, -0.01F);
    std::vector<float> state(static_cast<std::size_t>(heads) * kD * kD);
    for (auto& x : state) x = rng.value(0.5F);
    std::vector<float> chained = state, carried = state;
    std::vector<float> out_chain(static_cast<std::size_t>(rows) * heads * kD), out_rows = out_chain;
    for (std::uint32_t r = 0; r < rows; ++r) {  // reference: one single-row step per row on its own operands
        const float* q = qkv.data() + r * qkv_stride;
        for (std::uint32_t h = 0; h < heads; ++h)
            recur::gated_delta_step_ref(policy, q + h * kD, q + (heads + h) * kD, q + (2U * heads + h) * kD,
                                        beta[static_cast<std::size_t>(r) * heads + h],
                                        gate.data() + static_cast<std::size_t>(r) * gate_stride, h, kD,
                                        chained.data() + static_cast<std::size_t>(h) * kD * kD,
                                        out_chain.data() + (static_cast<std::size_t>(r) * heads + h) * kD);
    }
    // rows form: the same addressing as the fragment, the state of a head is carried across rows in one buffer.
    for (std::uint32_t h = 0; h < heads; ++h)
        for (std::uint32_t r = 0; r < rows; ++r) {
            const float* q = qkv.data() + r * qkv_stride;
            const float* g = gate.data() + static_cast<std::size_t>(r) * policy.gate_row_stride(heads, kD);
            recur::gated_delta_step_ref(policy, q + h * kD, q + (heads + h) * kD, q + (2U * heads + h) * kD,
                                        beta[static_cast<std::size_t>(r) * heads + h], g, h, kD,
                                        carried.data() + static_cast<std::size_t>(h) * kD * kD,
                                        out_rows.data() + (static_cast<std::size_t>(r) * heads + h) * kD);
        }
    check(same_bits(chained, carried), "rows form state");
    check(same_bits(out_chain, out_rows), "rows form outputs");
}

// Policy hooks on the host branch (portable math): fused update, sticky flag, declared gate row strides.
void test_policy_host_hooks() {
    const recur::ScalarGate scalar;
    check(scalar.gate_row_stride(4, kD) == 4U, "ScalarGate gate row stride is value heads");
    unsigned flag = 0;
    const recur::PerKeyChannelGate<recur::DeclaredExp> kda{&flag};
    check(kda.gate_row_stride(4, kD) == 4ULL * kD, "PerKeyChannelGate gate row stride is value heads * head_dim");
    kda.observe(1.0F);
    check(flag == 0U, "finite state leaves the flag clear");
    kda.observe(std::numeric_limits<float>::infinity());
    check(flag == 1U, "infinite state sets the flag");
    flag = 0;
    kda.observe(std::numeric_limits<float>::quiet_NaN());
    check(flag == 1U, "NaN state sets the flag");
    const float key = 1.0F + 0x1.0p-12F, delta = 1.0F - 0x1.0p-12F, state = -1.0F;  // product rounds away its low bits
    check(bits(kda.update(key, delta, state)) == bits(std::fma(key, delta, state)), "PerKeyChannelGate update is fused");
}

}  // namespace

int main() {
    test_kda_equals_transcription(1, false);
    test_kda_equals_transcription(2, false);
    test_kda_equals_transcription(3, true);
    test_zero_seed_is_load_bearing();
    test_scalar_gate_equals_gdn();
    test_policy_host_hooks();
    test_rows_form(recur::ScalarGate{}, 11);
    test_rows_form(recur::PerKeyChannelGate<recur::DeclaredExp>{}, 12);
    if (failures != 0) return 1;
    std::printf("recur gated_delta policy tests: OK\n");
    return 0;
}
