// Hyper-connection contract tests (include/strix/hc): DS41 parameters reproduce the DS41 host reference
// (ds41_layer.hpp) bit for bit; the Hy-V4 Flash identity-comb trait matches a direct FP64 evaluation within the
// stated bounds; the identity trait is a skipped stage, not an identity matrix.

#include "hc_spec_fixture.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/hc/hc_ref.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what);
    }
}

bool same_bits(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<float> uniform(std::mt19937& g, std::size_t n, float lo, float hi, bool bf16 = false) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = bf16 ? strix::ds41::round_to_bf16(d(g)) : d(g);
    return v;
}

// Finite F16 weights with magnitudes in [2^-9, 2^-1): sign, exponent 6..14, random mantissa.
std::vector<std::uint16_t> f16_weights(std::mt19937& g, std::size_t n) {
    std::uniform_int_distribution<std::uint32_t> mant(0, 0x3ff), ex(6, 14), sign(0, 1);
    std::vector<std::uint16_t> v(n);
    for (auto& h : v) h = static_cast<std::uint16_t>((sign(g) << 15U) | (ex(g) << 10U) | mant(g));
    return v;
}

// DS41 (comb = sinkhorn, F16 fn, FP32 store) against ds41::hc_* for several hidden sizes (partial 640 chunks).
void ds41_bitwise(std::mt19937& g) {
    namespace ds = strix::ds41;
    for (const std::uint32_t hidden : {160U, 1536U, 4096U}) {
        strix::hc::Spec spec;
        spec.streams = ds::hc_copies;
        spec.hidden = hidden;
        spec.comb = strix::hc::Comb::sinkhorn;
        spec.sinkhorn_iters = 20;
        spec.fn = strix::hc::FnType::f16;
        spec.pre_eps = spec.comb_eps = 1e-6F;
        spec.norm_eps = 1e-20F;  // deepseek41_host_forward.cpp hc_mix site
        check(spec.mix_rows() == ds::hc_mix_count && spec.scale_count() == 3U, "ds41 mix geometry");
        const std::size_t n = spec.flat();
        const auto stream = uniform(g, n, -3.0F, 3.0F);
        const auto fn = f16_weights(g, n * spec.mix_rows());
        const auto base = uniform(g, spec.mix_rows(), -1.0F, 1.0F);
        const auto scale = uniform(g, 3, 0.5F, 2.0F);
        const ds::HcSiteWeights dw{fn, base, scale};
        const strix::hc::SiteWeights hw{fn, {}, base, scale};

        const auto ref_raw = ds::hc_raw_mixes(stream, dw, spec.norm_eps);
        const auto raw = strix::hc::raw_mixes(spec, stream, hw);
        check(same_bits(ref_raw, raw), "ds41 raw mixes bitwise");
        for (const std::uint32_t iters : {1U, 20U}) {
            spec.sinkhorn_iters = iters;
            const auto ref = ds::hc_split_sinkhorn(ref_raw, base, scale, iters, spec.pre_eps);
            const auto mix = strix::hc::split(spec, raw, base, scale);
            check(same_bits(ref.pre, mix.pre) && same_bits(ref.post, mix.post) && same_bits(ref.comb, mix.comb),
                  "ds41 split + sinkhorn bitwise");
        }
        const auto ref = ds::hc_mixes(stream, dw, 20, spec.pre_eps, spec.norm_eps);
        const auto mix = strix::hc::mixes(spec, stream, hw);
        std::vector<float> x_ref(hidden), x(hidden);
        ds::hc_collapse(stream, std::span<const float, ds::hc_copies>(ref.pre), x_ref);
        strix::hc::collapse(spec, stream, mix.pre, x);
        check(same_bits(x_ref, x), "ds41 collapse bitwise");
        const auto first = ds::hc_identity_pre();
        ds::hc_collapse(stream, first, x_ref);
        strix::hc::collapse(spec, stream, first, x);
        check(same_bits(x_ref, x), "ds41 first-layer collapse bitwise");
        const auto sub = uniform(g, hidden, -4.0F, 4.0F);
        std::vector<float> out_ref(n), out(n);
        ds::hc_expand(sub, stream, ref, out_ref);
        strix::hc::expand(spec, sub, stream, mix, out);
        check(same_bits(out_ref, out), "ds41 expand bitwise");
    }
}

// Hy-V4 Flash against FP64. Bounds (stated): raw mix |err| <= 1e-5 * r * sum|fn * x| + 1e-7 (FP32 chunked dot of
// 12288 terms: lane chain 20 + butterfly 5 + chunk chain 19 roundings, worst ~(44 + 4) * 2^-24 = 2.9e-6 relative to
// r * sum|fn * x|, so 1e-5 is ~3.5x the worst case); pre/post |err| <= magnitude * (0.25 * |scale| * mix bound + 4e-7)
// (sigmoid slope + declared_exp ulps); collapse/expand (given the oracle's FP32 coefficients) within one bfloat16
// rounding of the FP64 value plus 2^-22 of the summed magnitudes (FP32 roundings under cancellation).
void modelb_fp64(std::mt19937& g) {
    for (const auto site : {strix::hc::Site::sublayer, strix::hc::Site::head}) {
        const auto spec = hc_fixture::identity_bf16_spec(site);
        check(spec.valid() && spec.comb_rows() == 0U, "modelb spec");
        check(spec.mix_rows() == (site == strix::hc::Site::head ? 4U : 8U), "modelb mix rows");
        check(spec.scale_count() == (site == strix::hc::Site::head ? 1U : 2U), "modelb scale count");
        const std::size_t n = spec.flat(), rows = spec.mix_rows();
        const auto stream = uniform(g, n, -2.0F, 2.0F, true);
        const auto fn = uniform(g, rows * n, -0.02F, 0.02F);
        const auto base = uniform(g, rows, -1.0F, 1.0F);
        const auto scale = uniform(g, spec.scale_count(), 0.5F, 2.0F);
        const strix::hc::SiteWeights w{{}, fn, base, scale};
        const auto raw = strix::hc::raw_mixes(spec, stream, w);
        const auto mix = strix::hc::split(spec, raw, base, scale);

        double ss = 0.0;
        for (const float v : stream) ss += double{v} * double{v};
        const double r = 1.0 / std::sqrt(ss / static_cast<double>(n) + double{spec.norm_eps});
        double worst_mix = 0.0, worst_coef = 0.0;
        for (std::size_t m = 0; m < rows; ++m) {
            double d = 0.0, a = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                d += double{fn[m * n + i]} * double{stream[i]};
                a += std::fabs(double{fn[m * n + i]} * double{stream[i]});
            }
            const double mref = d * r, mtol = 1e-5 * r * a + 1e-7;
            const double merr = std::fabs(double{raw[m]} - mref);
            worst_mix = std::fmax(worst_mix, merr / mtol);
            check(merr <= mtol, "modelb raw mix vs fp64");
            const bool is_pre = m < spec.streams;
            const double s = double{scale[is_pre ? 0U : 1U]};
            const double mag = is_pre ? 1.0 : double{spec.post_magnitude};
            const double eps = double{is_pre ? spec.pre_eps : spec.post_eps};
            const double cref = mag / (1.0 + std::exp(-(mref * s + double{base[m]}))) + eps;
            const float got = is_pre ? mix.pre[m] : mix.post[m - spec.streams];
            const double ctol = mag * (0.25 * std::fabs(s) * mtol + 4e-7);
            const double cerr = std::fabs(double{got} - cref);
            worst_coef = std::fmax(worst_coef, cerr / ctol);
            check(cerr <= ctol, "modelb pre/post vs fp64");
        }
        std::printf("modelb %s: worst raw-mix err %.2e of bound, worst coef err %.2e of bound\n",
                    site == strix::hc::Site::head ? "head" : "sublayer", worst_mix, worst_coef);

        const auto bf16_close = [](float got, double ref, double magnitude) {
            return std::fabs(double{got} - ref) <= std::ldexp(std::fabs(ref), -8) * 1.01 + std::ldexp(magnitude, -22);
        };
        std::vector<float> x(spec.hidden);
        strix::hc::collapse(spec, stream, mix.pre, x);
        bool ok = true;
        for (std::size_t i = 0; i < spec.hidden; ++i) {
            double y = 0.0, a = 0.0;
            for (std::uint32_t c = 0; c < spec.streams; ++c) {
                const double t = double{mix.pre[c]} * double{stream[c * spec.hidden + i]};
                y += t;
                a += std::fabs(t);
            }
            ok = ok && bf16_close(x[i], y, a) && x[i] == strix::ds41::round_to_bf16(x[i]);
        }
        check(ok, "modelb collapse vs fp64 (one bf16 rounding)");
        if (site == strix::hc::Site::head) continue;

        const auto sub = uniform(g, spec.hidden, -4.0F, 4.0F, true);
        std::vector<float> out(n);
        strix::hc::expand(spec, sub, stream, mix, out);
        ok = true;
        for (std::uint32_t k = 0; k < spec.streams; ++k)
            for (std::size_t i = 0; i < spec.hidden; ++i) {
                const std::size_t at = k * spec.hidden + i;
                const double p = double{mix.post[k]} * double{sub[i]};
                const double ref = p + double{stream[at]};
                ok = ok && bf16_close(out[at], ref, std::fabs(p) + std::fabs(double{stream[at]})) &&
                     out[at] == strix::ds41::round_to_bf16(mix.post[k] * sub[i] + stream[at]);
            }
        check(ok, "modelb expand vs fp64 and declared order");
    }
}

// The identity trait skips the comb: an infinite residual copy stays infinite in its own stream, while an identity
// comb matrix would give 0 * inf = NaN in every other stream (why identity is a trait, not a matrix).
void identity_is_a_skip() {
    auto spec = hc_fixture::identity_bf16_spec(strix::hc::Site::sublayer);
    spec.hidden = 2;
    spec.store = strix::hc::Store::f32;
    strix::hc::Mix mix;
    mix.pre = {1.0F, 1.0F, 1.0F, 1.0F};
    mix.post = {1.0F, 1.0F, 1.0F, 1.0F};
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> residual = {inf, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
    const std::vector<float> sub = {0.5F, 0.25F};
    std::vector<float> out(8);
    strix::hc::expand(spec, sub, residual, mix, out);
    check(std::isinf(out[0]) && out[2] == 2.5F && out[7] == 7.25F, "identity trait = skipped stage");
    std::vector<float> eye(16, 0.0F);
    for (std::uint32_t k = 0; k < 4; ++k) eye[k * 4 + k] = 1.0F;
    check(std::isnan(strix::hc::expand_at(0.5F, residual.data(), 2, 0, 1, 1.0F, eye.data(), 4, strix::hc::Store::f32)),
          "identity matrix differs (0 * inf)");
}

void geometry_rejected() {
    auto spec = hc_fixture::identity_bf16_spec(strix::hc::Site::sublayer);
    bool threw = false;
    try {
        const std::vector<float> stream(10);
        (void)strix::hc::raw_mixes(spec, stream, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "bad geometry throws");
    spec.comb = strix::hc::Comb::sinkhorn;
    check(!spec.valid(), "sinkhorn without iterations is invalid");
}

} // namespace

int main() {
    std::mt19937 g(20261118U);
    ds41_bitwise(g);
    modelb_fp64(g);
    identity_is_a_skip();
    geometry_rejected();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("hc contract tests PASS\n");
    return 0;
}
