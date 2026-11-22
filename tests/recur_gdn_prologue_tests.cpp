// F9 gate prologue (src/ops/recur/gated_delta_prologue.hpp) and F7 coded-history short conv decode form
// (src/core/short_convolution_ring.hpp): host invariants.
// 1. gate_prologue::softplus / sigmoid and gated_delta_gate_body (SoftplusGate) equal, bit for bit, a transcription of the
//    former GDN prologue (beta = bf16(1 / (1 + expf(-beta))); gate = a * softplus(gate + dt)) over a sweep that crosses the
//    +-20 softplus branches, signed zeros, subnormals, infinities and NaN.
// 2. ring_push + SiluRound over several rows equal the transcription of the former conv update (shift, BF16 encode of
//    the input, sum from +0 over taps, SiLU, BF16 rounding), including the history left behind.
// Build: -ffp-contract=off -fno-fast-math (device contraction is proven by ISA comparison of the binders, not here).
#include "core/short_convolution_ring.hpp"
#include "ops/recur/gated_delta_prologue.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

namespace recur = strix::ops::recur;
namespace conv = strix::core::causal_conv;

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
float from_bits(std::uint32_t u) {
    float x = 0.0F;
    std::memcpy(&x, &u, sizeof x);
    return x;
}
bool same(float a, float b) { return bits(a) == bits(b); }

// The GDN BF16 codec: round to nearest even by carry, no NaN quieting (the binder's encoder).
struct Bf16Carry {
    using Slot = std::uint16_t;
    static std::uint16_t encode(float value) {
        const auto b = bits(value);
        return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
    }
    static float decode(std::uint16_t slot) { return from_bits(static_cast<std::uint32_t>(slot) << 16U); }
    static float round(float value) { return decode(encode(value)); }
};
struct F32Load {
    using Kind = int;
    static float load(const void* p, Kind /*kind*/, std::uint64_t i) { return static_cast<const float*>(p)[i]; }
};

// Transcriptions of the former private text.
float old_softplus(float value) {
    if (value > 20.0F) return value;
    if (value < -20.0F) return expf(value);
    return log1pf(expf(value));
}
void old_gate(float* beta, float* gate, const float* dt, const float* a, std::uint32_t head) {
    beta[head] = Bf16Carry::round(1.0F / (1.0F + expf(-beta[head])));
    const float biased = gate[head] + dt[head];
    gate[head] = a[head] * old_softplus(biased);
}
void old_conv(float* qkv, std::uint16_t* conv_state, const float* w, std::uint32_t kernel_size, std::uint32_t channel) {
    auto* state = conv_state + static_cast<std::uint64_t>(channel) * kernel_size;
    for (std::uint32_t index = 0; index + 1U < kernel_size; ++index) state[index] = state[index + 1U];
    state[kernel_size - 1U] = Bf16Carry::encode(qkv[channel]);
    float sum = 0.0F;
    for (std::uint32_t tap = 0; tap < kernel_size; ++tap)
        sum += Bf16Carry::decode(state[tap]) * w[static_cast<std::uint64_t>(channel) * kernel_size + tap];
    const float silu = sum / (1.0F + expf(-sum));
    qkv[channel] = Bf16Carry::round(silu);
}
// The shared pieces in the order of ring_update (the device form calls weighted_taps<false>: from +0, tap order).
void new_conv(float* qkv, std::uint16_t* conv_state, const float* w, std::uint32_t kernel_size, std::uint32_t channel) {
    auto* state = conv_state + static_cast<std::uint64_t>(channel) * kernel_size;
    conv::ring_push<Bf16Carry>(state, kernel_size, qkv[channel]);
    float sum = 0.0F;
    for (std::uint32_t tap = 0; tap < kernel_size; ++tap)
        sum += Bf16Carry::decode(state[tap]) * F32Load::load(w, 0, static_cast<std::uint64_t>(channel) * kernel_size + tap);
    qkv[channel] = conv::SiluRound<Bf16Carry>::apply(sum);
}

std::vector<float> sweep() {
    std::vector<float> v{0.0F, -0.0F, 1.0F, -1.0F, 20.0F, -20.0F, std::nextafter(20.0F, 30.0F),
                         std::nextafter(-20.0F, -30.0F), std::nextafter(20.0F, 0.0F), std::nextafter(-20.0F, 0.0F),
                         88.0F, -88.0F, 1.0e-40F, -1.0e-40F, std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::denorm_min(), 3.0e38F, -3.0e38F};
    std::uint32_t s = 0x9e3779b9U;
    for (int i = 0; i < 4000; ++i) {
        s = s * 1664525U + 1013904223U;
        v.push_back((static_cast<float>(s >> 8U) / 16777216.0F - 0.5F) * 64.0F);
    }
    return v;
}

void gate_tests() {
    const auto xs = sweep();
    bool sp = true, sg = true;
    for (const float x : xs) {
        sp = sp && same(recur::gate_prologue::softplus(x), old_softplus(x));
        sg = sg && same(recur::gate_prologue::sigmoid(x), 1.0F / (1.0F + expf(-x)));
    }
    check(sp, "softplus == former text");
    check(sg, "sigmoid == former text");

    const auto n = static_cast<std::uint32_t>(xs.size());
    std::vector<float> beta0(xs), gate0(n), dt(n), a(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        gate0[i] = xs[(i * 7U + 3U) % n];
        dt[i] = xs[(i * 13U + 5U) % n] * 0.25F;
        a[i] = -std::fabs(xs[(i * 31U + 1U) % n]) * 0.125F;
    }
    auto beta1 = beta0, gate1 = gate0, beta2 = beta0, gate2 = gate0;
    for (std::uint32_t h = 0; h < n; ++h) {
        recur::gated_delta_gate_body<recur::SoftplusGate, Bf16Carry, F32Load>(beta1.data(), gate1.data(), dt.data(), 0,
                                                                              a.data(), 0, h);
        old_gate(beta2.data(), gate2.data(), dt.data(), a.data(), h);
    }
    bool ok = true;
    for (std::uint32_t h = 0; h < n; ++h) ok = ok && same(beta1[h], beta2[h]) && same(gate1[h], gate2[h]);
    check(ok, "gated_delta_gate_body<SoftplusGate> == former transform_beta_gate_body");
}

void conv_tests() {
    const auto xs = sweep();
    for (const std::uint32_t taps : {1U, 2U, 4U, 5U}) {
        constexpr std::uint32_t channels = 96, rows = 9;
        std::vector<float> w(static_cast<std::size_t>(channels) * taps);
        for (std::size_t i = 0; i < w.size(); ++i) w[i] = xs[(i * 11U + 2U) % xs.size()] * 0.03125F;
        std::vector<std::uint16_t> s1(static_cast<std::size_t>(channels) * taps), s2;
        for (std::size_t i = 0; i < s1.size(); ++i) s1[i] = Bf16Carry::encode(xs[(i * 17U + 9U) % xs.size()] * 0.0625F);
        s2 = s1;
        bool ok = true;
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<float> q1(channels);
            for (std::uint32_t c = 0; c < channels; ++c) q1[c] = xs[(r * channels + c) * 5U % xs.size()];
            auto q2 = q1;
            for (std::uint32_t c = 0; c < channels; ++c) {
                new_conv(q1.data(), s1.data(), w.data(), taps, c);
                old_conv(q2.data(), s2.data(), w.data(), taps, c);
            }
            for (std::uint32_t c = 0; c < channels; ++c) ok = ok && same(q1[c], q2[c]);
        }
        ok = ok && s1 == s2;
        check(ok, "ring_push + SiluRound == former conv update (outputs and history)");
    }
}

}  // namespace

int main() {
    gate_tests();
    conv_tests();
    if (failures != 0) {
        std::printf("recur gdn prologue tests: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("recur gdn prologue tests: PASS\n");
    return 0;
}
