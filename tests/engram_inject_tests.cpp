// Shared Engram injection contract (include/strix/engram/inject.hpp): host reference of both forms.
//  - folded: bitwise equal to the DeepSeek V4.1 host code it replaced (transcribed below from src/ds41_layer.cpp before
//    the lift) and to the re-bound ds41::engram_inject / engram_gates;
//  - normalized_bf16: bitwise equal to the model-B CPU oracle's Engram update (transcribed from
//    src/modelb/ref/modelb_host_ref.cpp, Engram block after the projection), including sign(0), masked rows and small hidden.
#include "strix/ds41_layer.hpp"
#include "strix/engram/inject.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

struct Rng {
    std::uint64_t s;
    float next(float lo, float hi) {
        s ^= s << 13U;
        s ^= s >> 7U;
        s ^= s << 17U;
        const auto u = static_cast<float>(s >> 40U) / static_cast<float>(1U << 24U);
        return strix::ds41::round_to_bf16(lo + (hi - lo) * u);
    }
    std::vector<float> vec(std::size_t n, float lo, float hi) {
        std::vector<float> v(n);
        for (auto& x : v) x = next(lo, hi);
        return v;
    }
};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

constexpr std::uint32_t C = 4;

// ---- DS41 host code before the lift (src/ds41_layer.cpp engram_gates + engram_inject) -------------------------------
void ds41_before(std::vector<float>& stream, const std::vector<float>& kv, const std::vector<float>& qn, const std::vector<float>& kn,
                 float eps, bool gate_off) {
    using namespace strix::ds41;
    const std::size_t hidden = stream.size() / C;
    const float inv_sqrt_hidden = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hidden)));
    std::array<float, C> gates{};
    for (std::uint32_t c = 0; c < C; ++c) {
        const float* h = stream.data() + c * hidden;
        const float* key = kv.data() + c * hidden;
        const float* q = qn.data() + c * hidden;
        const float* k = kn.data() + c * hidden;
        const auto sq = [hidden](const float* x) {
            return chunked_dot_of([x](std::size_t i) { return x[i]; }, [x](std::size_t i) { return x[i]; }, hidden);
        };
        const float rh = declared_rsqrt(sq(h) / static_cast<float>(hidden) + eps);
        const float rk = declared_rsqrt(sq(key) / static_cast<float>(hidden) + eps);
        const float dot_sum = chunked_dot_of([&](std::size_t i) { return h[i] * (q[i] * k[i]); }, [&](std::size_t i) { return key[i]; }, hidden);
        gates[c] = engram_gate_of((dot_sum * (rh * rk)) * inv_sqrt_hidden);
    }
    if (gate_off) gates.fill(0.0F);
    const float* value = kv.data() + C * hidden;
    for (std::uint32_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < hidden; ++i) stream[c * hidden + i] = stream[c * hidden + i] + gates[c] * value[i];
}

// ---- model-B oracle Engram update (modelb_host_ref.cpp, after `projected`): kv blocks 0..3 keys, 4 value --------------------
float bf(float x) { return strix::ds41::round_to_bf16(x); }
std::vector<float> oracle_rms(const float* x, std::size_t n, float eps) {
    const float r = 1.0F / strix::ds41::declared_sqrt(
                               strix::ds41::chunked_dot_of([x](std::size_t i) { return x[i]; }, [x](std::size_t i) { return x[i]; }, n) /
                                   static_cast<float>(n) + eps);
    std::vector<float> y(n);
    for (std::size_t i = 0; i < n; ++i) y[i] = bf(x[i] * r);
    return y;
}
void modelb_oracle(std::vector<float>& x, const std::vector<float>& kv, const std::vector<float>& gate_w, const std::vector<float>& hc_scale,
                const std::vector<float>& pre_gamma, float eps, bool masked) {
    if (masked) return;
    const std::size_t h = pre_gamma.size();
    const auto unit = oracle_rms(kv.data() + 4 * h, h, eps);
    std::vector<float> m(h);
    for (std::size_t i = 0; i < h; ++i) m[i] = bf(pre_gamma[i] * unit[i]);
    const float inv_sqrt_hidden = static_cast<float>(1.0 / std::sqrt(static_cast<double>(h)));
    for (std::uint32_t copy = 0; copy < 4U; ++copy) {
        const auto key = oracle_rms(kv.data() + copy * h, h, eps);
        const auto query = oracle_rms(x.data() + copy * h, h, eps);
        std::vector<float> term(h);
        for (std::size_t i = 0; i < h; ++i) term[i] = bf(bf(key[i] * gate_w[copy * h + i]) * query[i]);
        const float sum = strix::ds41::chunked_dot_of([&](std::size_t i) { return term[i]; }, [](std::size_t) { return 1.0F; }, h);
        const float d = bf(sum * inv_sqrt_hidden);
        const float sign = d > 0.0F ? 1.0F : (d < 0.0F ? -1.0F : 0.0F);
        const float g = bf(strix::ds41::declared_sigmoid(sign * strix::ds41::declared_sqrt(std::max(std::fabs(d), 1e-6F))));
        for (std::size_t i = 0; i < h; ++i) {
            const float out = bf(bf(g * m[i]) + bf(hc_scale[copy * h + i] * m[i]));
            x[copy * h + i] = bf(x[copy * h + i] + out);
        }
    }
}

void folded_case(std::uint32_t hidden, std::uint64_t seed, bool gate_off) {
    Rng r{seed};
    auto stream = r.vec(C * hidden, -3.0F, 3.0F);
    const auto kv = r.vec((C + 1) * hidden, -2.0F, 2.0F);
    const auto qn = r.vec(C * hidden, 0.5F, 1.5F), kn = r.vec(C * hidden, 0.5F, 1.5F);
    auto before = stream, shared = stream, rebound = stream;
    ds41_before(before, kv, qn, kn, 1e-20F, gate_off);
    strix::engram::inject_row_reference(strix::engram::InjectForm::folded, C, hidden, 1e-20F, shared.data(), kv.data(), qn.data(),
                                        kn.data(), gate_off);
    strix::ds41::engram_inject(rebound, kv, qn, kn, 1e-20F, gate_off);
    check(same_bits(before, shared), "folded: shared reference == DS41 code before the lift");
    check(same_bits(before, rebound), "folded: re-bound ds41::engram_inject == before");
    check(!same_bits(before, stream) || gate_off, "folded: the update changes the stream");
}

void normalized_case(std::uint32_t hidden, std::uint64_t seed, bool masked, bool zero_query) {
    Rng r{seed};
    auto stream = r.vec(C * hidden, -3.0F, 3.0F);
    if (zero_query) std::fill(stream.begin(), stream.begin() + hidden, 0.0F);  // copy 0: d = 0 -> sign(0) = 0 -> sigmoid(0)
    const auto kv = r.vec((C + 1) * hidden, -2.0F, 2.0F);
    const auto gate_w = r.vec(C * hidden, -1.5F, 1.5F), scale = r.vec(C * hidden, -0.5F, 0.5F), gamma = r.vec(hidden, 0.5F, 1.5F);
    std::vector<float> w1(scale);
    w1.insert(w1.end(), gamma.begin(), gamma.end());
    const float eps = 0.0078125F;
    auto oracle = stream, shared = stream;
    modelb_oracle(oracle, kv, gate_w, scale, gamma, eps, masked);
    std::array<float, C> gates{};
    strix::engram::inject_row_reference(strix::engram::InjectForm::normalized_bf16, C, hidden, eps, shared.data(), kv.data(),
                                        gate_w.data(), w1.data(), masked, gates.data());
    check(same_bits(oracle, shared), "normalized_bf16: shared reference == model-B oracle update");
    if (masked) check(same_bits(stream, shared), "normalized_bf16: masked row unchanged");
    if (zero_query) check(gates[0] == 0.5F, "normalized_bf16: d = 0 gives sigmoid(0) (sign(0) = 0)");
}

}  // namespace

int main() {
    for (const std::uint32_t hidden : {8U, 33U, 640U, 1000U, 3072U, 4096U, 5120U})
        for (std::uint64_t seed = 1; seed <= 3; ++seed) {
            folded_case(hidden, 0x9e3779b97f4a7c15ULL * seed + hidden, seed == 2);
            normalized_case(hidden, 0xc2b2ae3d27d4eb4fULL * seed + hidden, seed == 2, seed == 3);
        }
    // multi-row view: masked rows skipped, other rows = single-row reference
    {
        const std::uint32_t hidden = 3072, rows = 3;
        Rng r{77};
        auto stream = r.vec(std::size_t{rows} * C * hidden, -3.0F, 3.0F);
        const auto kv = r.vec(std::size_t{rows} * (C + 1) * hidden, -2.0F, 2.0F);
        const auto w0 = r.vec(C * hidden, -1.5F, 1.5F), w1 = r.vec((C + 1) * hidden, 0.25F, 1.25F);
        const std::uint8_t off[rows] = {0, 1, 0};
        auto expect = stream;
        for (std::uint32_t row = 0; row < rows; ++row)
            strix::engram::inject_row_reference(strix::engram::InjectForm::normalized_bf16, C, hidden, 0.0078125F,
                                                expect.data() + std::size_t{row} * C * hidden, kv.data() + std::size_t{row} * (C + 1) * hidden,
                                                w0.data(), w1.data(), off[row] != 0U);
        strix::engram::InjectView v;
        v.form = strix::engram::InjectForm::normalized_bf16;
        v.stream = stream.data();
        v.kv = kv.data();
        v.w0 = w0.data();
        v.w1 = w1.data();
        v.gate_off = off;
        v.rows = rows;
        v.hidden = hidden;
        v.eps = 0.0078125F;
        strix::engram::inject_reference(v);
        check(same_bits(expect, stream), "view: rows independent, masked row skipped");
        check(strix::engram::inject_threads(v) == 32U * 4U * 5U, "view: model-B hidden 3072 -> 20 waves");
        v.hidden = 5121;
        check(!strix::engram::inject_supported(v), "view: hidden 5121 (9 chunks x 4 copies) refused");
        v.hidden = 4096;
        check(strix::engram::inject_threads(v) == 32U * 4U * 7U, "view: DS41 hidden 4096 -> 28 waves (as before)");
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("engram inject tests: all passed\n");
    return 0;
}
