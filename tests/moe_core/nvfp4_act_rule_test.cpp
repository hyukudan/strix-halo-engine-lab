// Host test of the NVFP4 activation scale rules (moe_core/nvfp4_act_rule.hpp) and the grouped W4A4 activation classes:
// E4M3 / E2M1 decode and rounding, mse@1 never worse than nearest@1 per group, ties keep the nearest code, zero groups, and a
// pooled error reduction on deterministic Gaussian and Laplace rows; the two-term rule r2@1 (E2M1 decode, residual of an exact
// group is zero, per element the second term never increases the error, pooled error energy) and the r2 classes / image layout.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "moe_core/grouped_w4a4.hpp"
#include "moe_core/nvfp4_act_rule.hpp"

namespace {

namespace act = strix::moe_core::nvfp4_act;
namespace gw = strix::moe_core::grouped_w4a4;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// E4M3 round to nearest (ties to the even code), saturated at 448: the host mirror of the device conversion of nearest@1.
std::uint8_t e4m3_nearest(float x) {
    if (!(x > 0.F)) return 0U;
    if (x >= 448.F) return static_cast<std::uint8_t>(act::kMaxCode);
    unsigned best = 0U;
    for (unsigned c = 1U; c <= act::kMaxCode; ++c) {
        const float d = std::fabs(act::e4m3_value(c) - x), b = std::fabs(act::e4m3_value(best) - x);
        if (d < b || (d == b && (c & 1U) == 0U)) best = c;
    }
    return static_cast<std::uint8_t>(best);
}

std::uint8_t nearest_code(const float (&v)[16], float ts) {
    float a16 = 0.F;
    for (float x : v) a16 = std::fmax(a16, std::fabs(x));
    return ts > 0.F ? e4m3_nearest(std::fmin(448.F, (a16 / 6.F) / ts)) : std::uint8_t{0};
}

float scale_of(std::uint8_t code, float ts) { return code ? act::e4m3_value(code) * ts : 0.F; }

struct Rng {
    std::uint64_t s;
    double uniform() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<double>(s >> 11) + 0.5) / 9007199254740992.0;
    }
    float gauss() { return static_cast<float>(std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307179586 * uniform())); }
    float laplace() {
        const double u = uniform() - 0.5;
        return static_cast<float>(u < 0 ? std::log(1.0 + 2.0 * u) : -std::log(1.0 - 2.0 * u));
    }
};

void decode() {
    check(act::e4m3_value(0x7EU) == 448.F, "e4m3 0x7e = 448");
    check(act::e4m3_value(0x38U) == 1.F, "e4m3 0x38 = 1");
    check(act::e4m3_value(0x08U) == 0.015625F, "e4m3 0x08 = 2^-6 (smallest normal)");
    check(act::e4m3_value(0x01U) == 0.001953125F, "e4m3 0x01 = 2^-9 (smallest subnormal)");
    bool monotonic = true;
    for (unsigned c = 1U; c <= act::kMaxCode; ++c) monotonic = monotonic && act::e4m3_value(c) > act::e4m3_value(c - 1U);
    check(monotonic, "e4m3 codes 0 .. 0x7e increase");
    const float in[] = {0.25F, 0.26F, 0.75F, 1.25F, 1.75F, 2.5F, 3.5F, 5.F, 5.01F, 6.F};
    const float out[] = {0.F, 0.5F, 1.F, 1.F, 2.F, 2.F, 4.F, 4.F, 6.F, 6.F};
    bool rne = true;
    for (unsigned i = 0; i < 10U; ++i) rne = rne && act::e2m1_round(in[i]) == out[i];
    check(rne, "e2m1 round to nearest, ties to the even code");
    check(act::search_offset(0) == 1 && act::search_offset(5) == 6 && act::search_offset(6) == -1 && act::search_offset(7) == -2,
          "candidate offsets +1..+6, -1, -2");
}

void edges() {
    float zero[16] = {};
    check(act::search_scale_code(zero, 1.F, 0U) == 0U, "zero group keeps code 0");
    float v[16];
    for (unsigned i = 0; i < 16U; ++i) v[i] = static_cast<float>(i) - 7.5F;
    check(act::search_scale_code(v, 0.F, 0U) == 0U, "ts = 0 keeps code 0");
    // Exactly representable at the nearest code (error 0): no candidate is strictly better, the nearest code stays.
    const float grid[8] = {0.F, 0.5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
    const float ts = 0.0078125F;  // 2^-7: every product exact
    float exact[16];
    for (unsigned i = 0; i < 16U; ++i) exact[i] = (i & 1U ? -1.F : 1.F) * grid[i % 8U] * act::e4m3_value(0x38U) * ts;
    const auto c0 = nearest_code(exact, ts);
    check(c0 == 0x38U && act::group_error(exact, scale_of(c0, ts)) == 0.F, "exact group: nearest code 1.0, error 0");
    check(act::search_scale_code(exact, ts, c0) == c0, "exact group: tie keeps the nearest code");
}

template <class Draw>
double pooled(Draw draw, const char* what, double min_gain) {
    Rng rng{0x5eedULL};
    double near = 0.0, mse = 0.0;
    bool never_worse = true, valid = true;
    for (unsigned row = 0; row < 512U; ++row) {
        float r[256];
        float amax = 0.F;
        for (float& x : r) {
            x = draw(rng);
            amax = std::fmax(amax, std::fabs(x));
        }
        const float ts = amax / gw::kRowScaleDiv;
        for (unsigned g = 0; g < 16U; ++g) {
            float v[16];
            std::memcpy(v, r + 16U * g, sizeof v);
            const auto c0 = nearest_code(v, ts);
            const auto c = act::search_scale_code(v, ts, c0);
            const float e0 = act::group_error(v, scale_of(c0, ts)), e = act::group_error(v, scale_of(c, ts));
            never_worse = never_worse && e <= e0;
            valid = valid && c >= 1U && c <= act::kMaxCode;
            near += e0;
            mse += e;
        }
    }
    const double gain = 1.0 - mse / near;
    std::printf("%s: group error nearest %.6g mse %.6g (-%.1f %%)\n", what, near, mse, 100.0 * gain);
    check(never_worse, "mse@1 group error <= nearest@1 group error for every group");
    check(valid, "mse@1 codes are positive finite E4M3 codes");
    check(gain >= min_gain, what);
    return gain;
}

// Host mirror of one group's image under a scale rule (the device encode_values / encode_values_mse): E4M3 code, codes word.
std::uint8_t encode_group(const float (&v)[16], float ts, bool mse, std::uint32_t (&word)[2]) {
    auto code = nearest_code(v, ts);
    if (mse) code = act::search_scale_code(v, ts, code);
    const float scale = scale_of(code, ts);
    const float grid[8] = {0.F, 0.5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
    word[0] = word[1] = 0U;
    for (unsigned i = 0; i < 16U; ++i) {
        const float u = scale != 0.F ? std::fmin(6.F, std::fmax(-6.F, v[i] / scale)) : 0.F;
        const float a = act::e2m1_round(std::fabs(u));
        unsigned m = 0U;
        while (grid[m] != a) ++m;
        const unsigned nibble = (u < 0.F && a > 0.F ? 8U : 0U) | m;
        word[i >> 3U] |= nibble << (4U * (i & 7U));
    }
    return code;
}

void two_term_rule() {
    const float grid[8] = {0.F, 0.5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
    bool table = true;
    for (unsigned n = 0; n < 16U; ++n) table = table && act::e2m1_value(n) == ((n & 8U) ? -grid[n & 7U] : grid[n & 7U]);
    check(table, "e2m1_value decodes the 16 codes (bit 3 sign)");
    const float ts = 0.0078125F;
    float exact[16];
    for (unsigned i = 0; i < 16U; ++i) exact[i] = (i & 1U ? -1.F : 1.F) * grid[i % 8U] * act::e4m3_value(0x38U) * ts;
    std::uint32_t word[2];
    const auto c = encode_group(exact, ts, false, word);
    float r[16];
    act::residual(exact, scale_of(c, ts), word, r);
    bool zero = true;
    for (float x : r) zero = zero && x == 0.F;
    check(zero, "r2@1: the residual of an exactly representable group is 0");
    float none[16] = {};
    act::residual(none, 0.F, word, r);
    check(r[0] == 0.F && r[15] == 0.F, "r2@1: zero scale leaves the residual equal to the group");
}

// Two-term pooled error energy (term 2 at the same row scale and rule) relative to one term, and the per-element invariant.
template <class Draw>
double two_terms(Draw draw, const char* what, bool mse, double max_ratio) {
    Rng rng{0x7e57ULL};
    double one = 0.0, two = 0.0;
    bool element_ok = true;
    for (unsigned row = 0; row < 256U; ++row) {
        float x[256];
        float amax = 0.F;
        for (float& v : x) {
            v = draw(rng);
            amax = std::fmax(amax, std::fabs(v));
        }
        const float ts = amax / gw::kRowScaleDiv;
        for (unsigned g = 0; g < 16U; ++g) {
            float v[16], r[16], r2[16];
            std::memcpy(v, x + 16U * g, sizeof v);
            std::uint32_t w1[2], w2[2];
            const auto c1 = encode_group(v, ts, mse, w1);
            act::residual(v, scale_of(c1, ts), w1, r);
            const auto c2 = encode_group(r, ts, mse, w2);
            act::residual(r, scale_of(c2, ts), w2, r2);  // what both terms together leave
            for (unsigned i = 0; i < 16U; ++i) {
                element_ok = element_ok && std::fabs(r2[i]) <= std::fabs(r[i]);
                one += static_cast<double>(r[i]) * r[i];
                two += static_cast<double>(r2[i]) * r2[i];
            }
        }
    }
    const double ratio = two / one;
    std::printf("%s: error energy one term %.6g, two terms %.6g (ratio %.4f, amplitude %.3f)\n", what, one, two, ratio,
                std::sqrt(ratio));
    check(element_ok, "r2@1: per element the second term never increases the error");
    check(ratio <= max_ratio, what);
    return ratio;
}

void classes() {
    check(!gw::w4a4_gate_up(gw::Activations::bf16) && !gw::w4a4_down(gw::Activations::bf16), "bf16: no W4A4 projection");
    check(gw::w4a4_gate_up(gw::Activations::nvfp4) && gw::w4a4_down(gw::Activations::nvfp4) &&
              gw::scale_rule(gw::Activations::nvfp4) == gw::ScaleRule::nearest, "nvfp4: both projections, nearest@1");
    check(gw::w4a4_down(gw::Activations::nvfp4_mse) && gw::scale_rule(gw::Activations::nvfp4_mse) == gw::ScaleRule::mse,
          "nvfp4_mse: both projections, mse@1");
    check(gw::w4a4_gate_up(gw::Activations::nvfp4_mse_gate_up) && !gw::w4a4_down(gw::Activations::nvfp4_mse_gate_up),
          "nvfp4_mse_gate_up: gate-up W4A4, down BF16");
    check(gw::terms_gate_up(gw::Activations::nvfp4_mse) == 1U && gw::terms_down(gw::Activations::nvfp4_mse) == 1U &&
              gw::terms_gate_up(gw::Activations::nvfp4_mse_gate_up) == 1U, "one-term classes keep one term");
    check(gw::w4a4_gate_up(gw::Activations::nvfp4_mse_r2_gate_up) && gw::w4a4_down(gw::Activations::nvfp4_mse_r2_gate_up) &&
              gw::terms_gate_up(gw::Activations::nvfp4_mse_r2_gate_up) == 2U &&
              gw::terms_down(gw::Activations::nvfp4_mse_r2_gate_up) == 1U &&
              gw::scale_rule(gw::Activations::nvfp4_mse_r2_gate_up) == gw::ScaleRule::mse,
          "nvfp4_mse_r2_gate_up: two-term gate-up, one-term down, mse@1");
    check(gw::w4a4_down(gw::Activations::nvfp4_mse_r2) && gw::terms_gate_up(gw::Activations::nvfp4_mse_r2) == 2U &&
              gw::terms_down(gw::Activations::nvfp4_mse_r2) == 2U, "nvfp4_mse_r2: two terms on both projections");
    check(std::strcmp(gw::identity(gw::Activations::nvfp4_mse_r2_gate_up), "nvfp4-mse-r2-gate-up@1") == 0 &&
              std::strcmp(gw::identity(gw::Activations::nvfp4_mse_r2), "nvfp4-mse-r2@1") == 0 &&
              std::strcmp(gw::identity(gw::Activations::nvfp4_mse_gate_up), "nvfp4-mse-gate-up@1") == 0, "r2 identities");
    check(!gw::valid_terms(0U) && gw::valid_terms(1U) && gw::valid_terms(2U) && !gw::valid_terms(3U), "terms 1 or 2");
    alignas(256) static std::uint8_t buf[gw::image_layout(64, 2304, 2).bytes];
    static_assert(gw::image_layout(64, 2304, 1).bytes == gw::image_layout(64, 2304).bytes, "one term: unchanged layout");
    static_assert(gw::image_layout(64, 2304, 2).scales == 64U * 2304U, "two terms: codes K bytes per row");
    const auto two = gw::make_image(buf, 64, 2304, 2), one = gw::make_image(buf, 64, 2304);
    check(two.code_row == 2304U && two.scale_row == 288U && one.code_row == 1152U && one.scale_row == 144U,
          "image rows hold terms x K/2 code bytes and terms x K/16 scale bytes");
    check(gw::fits_image(two, 2U * 2304U) && !gw::fits_image(one, 2U * 2304U), "a two-term projection needs a two-term image");
    check(std::strcmp(gw::identity(gw::Activations::nvfp4_mse), "nvfp4-mse@1") == 0 &&
              std::strcmp(act::identity(act::ScaleRule::nearest), "nvfp4-act-scale-nearest@1") == 0, "identities");
}

// Review regression: one group {1e30, 15 x 4.5e29} with the row max 1e30 has nearest code 0x7E (448); 0x7D (416) cuts
// the squared error from ~0.0375 amax^2 to ~0.00816 amax^2. Plain FP32 scores overflow to +inf at 1e30 (and underflow
// toward 0 at tiny magnitudes), so the search must pick 0x7D at every finite magnitude, identically to the unit group.
void magnitudes() {
    const float scales[] = {1.F, 1e30F, 1e-30F, 2e38F};
    for (const float amax : scales) {
        float v[16];
        v[0] = amax;
        for (int i = 1; i < 16; ++i) v[i] = 0.45F * amax;
        const float ts = amax / 6.F / 448.F;
        const auto c0 = nearest_code(v, ts);
        check(c0 == 0x7EU, "magnitude group: nearest code is 0x7E");
        check(act::search_scale_code(v, ts, c0) == 0x7DU, "magnitude group: mse@1 picks 0x7D at every finite scale");
    }
    float unit[16];
    for (int i = 0; i < 16; ++i) unit[i] = static_cast<float>(i % 5) * 0.37F - 0.6F;
    check(act::score_normalizer(unit) == 1.F, "in-band group: normalizer is exactly 1 (plain FP32 scores)");
}
}  // namespace

int main() {
    decode();
    edges();
    magnitudes();
    classes();
    pooled([](Rng& r) { return r.gauss(); }, "gaussian rows: mse@1 cuts the group error by >= 15 %", 0.15);
    pooled([](Rng& r) { return r.laplace(); }, "laplace rows: mse@1 cuts the group error by >= 10 %", 0.10);
    two_term_rule();
    two_terms([](Rng& r) { return r.gauss(); }, "gaussian rows: two terms (nearest) leave <= 2 % of one term's error energy",
              false, 0.02);
    two_terms([](Rng& r) { return r.gauss(); }, "gaussian rows: two terms (mse@1) leave <= 2 % of one term's error energy",
              true, 0.02);
    two_terms([](Rng& r) { return r.laplace(); }, "laplace rows: two terms (mse@1) leave <= 2 % of one term's error energy",
              true, 0.02);
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("nvfp4_act_rule_test: PASS");
    return EXIT_SUCCESS;
}
