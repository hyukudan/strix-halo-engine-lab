// Host test of the NVFP4 activation scale rules (moe_core/nvfp4_act_rule.hpp) and the grouped W4A4 activation classes:
// E4M3 / E2M1 decode and rounding, mse@1 never worse than nearest@1 per group, ties keep the nearest code, zero groups, and a
// pooled error reduction on deterministic Gaussian and Laplace rows; the two-term rule r2@1 (E2M1 decode, residual of an exact
// group is zero, per element the second term never increases the error, pooled error energy) and the r2 classes / image layout.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include "moe_core/grouped_dispatch.hpp"
#include "moe_core/grouped_w4a4.hpp"
#include "moe_core/nvfp4_act_rule.hpp"

namespace {

namespace act = strix::moe_core::nvfp4_act;
namespace gw = strix::moe_core::grouped_w4a4;
namespace gd = strix::moe_core::grouped_dispatch;

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
    for (unsigned n = 0; n < 16U; ++n) {
        const float want = (n & 8U) ? -grid[n & 7U] : grid[n & 7U];
        const float got = strix::weight::scalar_float::e2m1(static_cast<std::uint8_t>(n));
        table = table && got == want && std::signbit(got) == ((n & 8U) != 0U);
    }
    check(table, "the shared E2M1 decoder the residual uses gives the 16 codes (bit 3 sign, code 8 = -0)");
    {  // act::residual against an independent expectation: literal code values and one FP32 FMA, fixed packed vectors
        const float lit[16] = {0.F, 0.5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F, -0.F, -0.5F, -1.F, -1.5F, -2.F, -3.F, -4.F, -6.F};
        const std::uint32_t words[3][2] = {{0x76543210U, 0xFEDCBA98U}, {0x89ABCDEFU, 0x01234567U}, {0x80808080U, 0x08080808U}};
        const float scales[3] = {0.0078125F, 1.25F, 3.0e-3F};
        bool same = true;
        for (unsigned w = 0; w < 3U; ++w)
            for (const float scale : scales) {
                float v[16], r[16];
                for (unsigned i = 0; i < 16U; ++i) v[i] = (i % 3U == 0U ? -0.F : 0.37F * static_cast<float>(i) - 2.F) * scale * 2.F;
                act::residual(v, scale, words[w], r);
                for (unsigned i = 0; i < 16U; ++i) {
                    const float want = -std::fma(lit[(words[w][i / 8U] >> (4U * (i % 8U))) & 15U], scale, -v[i]);
                    std::uint32_t a, b;
                    std::memcpy(&a, &r[i], 4);
                    std::memcpy(&b, &want, 4);
                    same = same && a == b;
                }
            }
        check(same, "residual bitwise (sign of zero included) the literal-value FP32 FMA on fixed packed vectors");
    }
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
    check(gw::w4a4_gate_up(gw::Activations::nvfp4_r2) && gw::w4a4_down(gw::Activations::nvfp4_r2) &&
              gw::terms_gate_up(gw::Activations::nvfp4_r2) == 2U && gw::terms_down(gw::Activations::nvfp4_r2) == 2U &&
              gw::scale_rule(gw::Activations::nvfp4_r2) == gw::ScaleRule::nearest &&
              std::strcmp(gw::identity(gw::Activations::nvfp4_r2), "nvfp4-r2@1") == 0,
          "nvfp4_r2: two nearest@1 terms on both projections");
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

// Review regression (#1228 r19 P2): one-term callers reserve and admit exactly the incumbent one-term image (the larger of the
// hidden and intermediate one-term layouts), so a buffer sized for the incumbent path still runs W4A4; only a two-term class
// needs its own larger image. Shapes of a 2304 / 1152 top-8 layer; the quantiser admission checks the base width first.
struct ImageConfig {
    static constexpr unsigned hidden = 2304U, intermediate = 1152U, top_k = 8U, experts = 64U, max_rows = 8192U;
};

void workspace_images() {
    const std::size_t incumbent[] = {
        std::max(gw::image_layout(8U, 2304U).bytes, gw::image_layout(8U, 1152U).bytes),
        std::max(gw::image_layout(8U * 8192U, 2304U).bytes, gw::image_layout(8U * 8192U, 1152U).bytes)};
    const unsigned rows[] = {1U, 8192U};
    const gw::Activations one_term[] = {gw::Activations::bf16, gw::Activations::nvfp4, gw::Activations::nvfp4_mse,
                                        gw::Activations::nvfp4_mse_gate_up};
    bool reserve = gd::workspace_requirement<ImageConfig>(1U).nvfp4 == incumbent[0] &&
                   gd::workspace_elements<ImageConfig>(8192U).nvfp4 == incumbent[1];
    for (unsigned i = 0; i < 2U; ++i)
        for (const auto a : one_term)
            reserve = reserve && gd::workspace_requirement<ImageConfig>(rows[i], a).nvfp4 == incumbent[i];
    check(reserve, "one-term classes reserve exactly the incumbent one-term image");
    const auto r2g = gd::workspace_requirement<ImageConfig>(8192U, gw::Activations::nvfp4_mse_r2_gate_up).nvfp4;
    const auto r2 = gd::workspace_requirement<ImageConfig>(8192U, gw::Activations::nvfp4_mse_r2).nvfp4;
    check(r2g == gw::image_layout(8U * 8192U, 2304U, 2U).bytes && r2 == r2g && r2g > incumbent[1],
          "two-term classes reserve the two-term gate-up image");

    // Admission: an aligned buffer of exactly the incumbent bytes runs W4A4 for every one-term class; one byte less does not;
    // a two-term class refuses it (its image needs more); a misaligned base never fits.
    std::vector<std::uint8_t> storage(incumbent[0] + 512U);
    auto* base = storage.data() + (256U - reinterpret_cast<std::uintptr_t>(storage.data()) % 256U) % 256U;
    const std::span<const std::uint8_t> exact(base, incumbent[0]), short_by_one(base, incumbent[0] - 1U),
        misaligned(base + 16, incumbent[0]);
    bool admit = true;
    for (const auto a : one_term)
        admit = admit && gd::nvfp4_image_fits<ImageConfig>(exact, 1U, a) &&
                !gd::nvfp4_image_fits<ImageConfig>(short_by_one, 1U, a) &&
                !gd::nvfp4_image_fits<ImageConfig>(misaligned, 1U, a);
    check(admit, "a buffer sized exactly for the incumbent one-term image is admitted (one byte less is not)");
    check(!gd::nvfp4_image_fits<ImageConfig>(exact, 1U, gw::Activations::nvfp4_mse_r2_gate_up) &&
              !gd::nvfp4_image_fits<ImageConfig>(exact, 1U, gw::Activations::nvfp4_mse_r2),
          "the incumbent one-term image does not admit a two-term class");
    check(!gd::nvfp4_image_fits<ImageConfig>(exact, 0U, gw::Activations::nvfp4), "zero rows never fit");

    // Quantiser admission (#1228 r19 P3): base width K % 128 before the term-expanded strides.
    alignas(256) static std::uint8_t img[gw::image_layout(8U, 256U, 2U).bytes];
    const auto k64 = gw::make_image(img, 8U, 64U, 2U);  // 64-byte code rows, 8-byte scale rows: K x terms = 128
    const auto k128 = gw::make_image(img, 8U, 128U, 2U), k128_one = gw::make_image(img, 8U, 128U);
    check(gw::fits_image(k64, 128U), "K = 64 two-term strides alone pass the term-expanded image rule");
    check(!gw::fits_quantize(k64, 64U, 2U) && !gw::fits_quantize(k64, 64U, 1U), "K = 64 is refused at any term count");
    check(gw::fits_quantize(k128, 128U, 2U) && gw::fits_quantize(k128_one, 128U, 1U), "K = 128 fits with its own image");
    check(!gw::fits_quantize(k128_one, 128U, 2U), "a one-term image is too narrow for two terms");
    check(!gw::fits_quantize(k128, 0U, 1U) && !gw::fits_quantize(k128, 128U, 0U) && !gw::fits_quantize(k128, 128U, 3U),
          "zero width and term counts outside 1..2 are refused");
    check(!gw::fits_quantize(k128, 0x80000080U, 2U), "a term-expanded width that overflows unsigned is refused");
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

// Host stand-ins for the dispatch (#1247 r27 P2): a backend and a configuration that count every submission, so the test sees
// whether enqueue binds the selected class, refuses it, or runs the BF16 tile.
struct Submissions {
    unsigned quantize = 0U, w4a4 = 0U, gather = 0U, bf16 = 0U, activation = 0U, combine = 0U;
    [[nodiscard]] unsigned total() const { return quantize + w4a4 + gather + bf16 + activation + combine; }
};
Submissions submitted;

struct CountingBackend {
    using Status = int;
    using Stream = int;
    static constexpr Status success = 0, invalid_value = 1;
    static constexpr unsigned rows_per_tile = 128U;
    static bool fits(const strix::moe_core::grouped_microscaled::Args&) { return true; }
    static Status gather(const strix::moe_core::grouped_rows::GatherArgs&, Stream) {
        ++submitted.gather;
        return success;
    }
    template<strix::weight::microscaled::MatrixFormat>
    static Status launch_projection(const strix::moe_core::grouped_microscaled::Args&, Stream) {
        ++submitted.bf16;
        return success;
    }
    static Status quantize_w4a4(const gw::QuantizeArgs<float>&, Stream, gw::ScaleRule) {
        ++submitted.quantize;
        return success;
    }
    static Status quantize_w4a4(const gw::QuantizeArgs<std::uint16_t>&, Stream, gw::ScaleRule) {
        ++submitted.quantize;
        return success;
    }
    static Status launch_w4a4(const gw::Args&, Stream) {
        ++submitted.w4a4;
        return success;
    }
};

struct CountingConfig {
    static constexpr unsigned hidden = 256U, intermediate = 128U, top_k = 2U, experts = 4U, layers = 2U, max_rows = 16U;
    // Layer 0 NVFP4, layer 1 MXFP8.
    static constexpr strix::weight::microscaled::MatrixFormat format(unsigned layer) {
        return layer == 0U ? strix::weight::microscaled::MatrixFormat::nvfp4 : strix::weight::microscaled::MatrixFormat::mxfp8;
    }
    struct Activation {
        static constexpr float gate_cap = 0.F, up_cap = 0.F;  // read by enqueue's epilogue form (this backend has none)
        static int launch(const strix::moe_core::expert_activation::Args&, int) {
            ++submitted.activation;
            return 0;
        }
    };
    struct Combine {
        static int launch(const strix::moe_core::grouped_rows::CombineArgs&, int) {
            ++submitted.combine;
            return 0;
        }
    };
};

// A selected W4A4 class on an NVFP4 layer binds or is refused before any submission; it never runs the BF16 tile silently.
void selected_class_binds_or_refuses() {
    constexpr unsigned rows = 4U, assignments = rows * CountingConfig::top_k;
    const std::vector<std::uint32_t> assignment_rows{0U, 1U, 2U, 3U, 3U, 2U, 1U, 0U};
    const std::vector<std::uint32_t> combine_rows{0U, 7U, 1U, 6U, 2U, 5U, 3U, 4U};
    const std::vector<float> weights(assignments, 0.5F);
    const std::vector<gd::TileRows> tiles(1U);
    const unsigned tile_count = 1U, failure = 0U;
    std::vector<float> input(std::size_t{rows} * CountingConfig::hidden), output(input.size());
    const auto need = gd::workspace_requirement<CountingConfig>(rows, gw::Activations::nvfp4_r2);
    const auto one_term = gd::workspace_requirement<CountingConfig>(rows).nvfp4;
    std::vector<std::uint16_t> bf16(need.bf16);
    std::vector<float> fp32(need.fp32);
    std::vector<std::uint8_t> storage(need.nvfp4 + 256U);
    auto* base = storage.data() + (256U - reinterpret_cast<std::uintptr_t>(storage.data()) % 256U) % 256U;
    check(one_term < need.nvfp4, "the two-term class needs more image bytes than the one-term reservation");
    const auto run = [&](unsigned layer, gw::Activations activations, std::size_t image_bytes) {
        submitted = {};
        gd::DevicePlan plan{layer, rows, assignment_rows, combine_rows, weights, tiles, tiles, &tile_count, &failure};
        plan.activations = activations;
        const gd::Workspace workspace{bf16, fp32, {}, std::span<std::uint8_t>(base, image_bytes)};
        return gd::enqueue<CountingConfig, CountingBackend>(plan, input, output, workspace, 0);
    };
    for (const auto a : {gw::Activations::nvfp4_r2, gw::Activations::nvfp4_mse_r2, gw::Activations::nvfp4_mse_r2_gate_up}) {
        check(run(0U, a, one_term) == CountingBackend::invalid_value && submitted.total() == 0U,
              "a two-term class on a one-term image is refused before any submission (no silent BF16 tile)");
        check(run(0U, a, need.nvfp4) == CountingBackend::success && submitted.w4a4 >= 1U && submitted.gather == 0U,
              "a two-term class on its own image runs the W4A4 tile");
    }
    check(run(0U, gw::Activations::nvfp4, one_term) == CountingBackend::success && submitted.quantize == 2U &&
              submitted.w4a4 == 2U && submitted.bf16 == 0U && submitted.gather == 0U,
          "the one-term class binds on the one-term image (both projections W4A4)");
    check(run(0U, gw::Activations::nvfp4, 0U) == CountingBackend::invalid_value && submitted.total() == 0U,
          "a W4A4 class with no image is refused on an NVFP4 layer");
    check(run(1U, gw::Activations::nvfp4_r2, 0U) == CountingBackend::success && submitted.w4a4 == 0U &&
              submitted.gather == 1U && submitted.bf16 == 2U,
          "an MXFP8 layer keeps the deliberate BF16 route under a W4A4 class");
    check(run(0U, gw::Activations::bf16, 0U) == CountingBackend::success && submitted.w4a4 == 0U && submitted.bf16 == 2U,
          "the BF16 class runs the BF16 tile");
}
}  // namespace

int main() {
    decode();
    edges();
    magnitudes();
    classes();
    workspace_images();
    selected_class_binds_or_refuses();
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
