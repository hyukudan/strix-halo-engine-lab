// Hyper-connection contract tests (include/strix/hc): DS41 parameters reproduce the DS41 host reference
// (ds41_layer.hpp) bit for bit; the Hy-V4 Flash identity-comb trait matches a direct FP64 evaluation within the
// stated bounds; the identity trait is a skipped stage, not an identity matrix.

#include "hc_spec_fixture.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/hc/hc_ref.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
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
        std::vector<float> transposed(fn.size());
        for (std::size_t m = 0; m < rows; ++m)
            for (std::size_t i = 0; i < n; ++i) transposed[i * rows + m] = fn[m * n + i];
        check(same_bits(raw, strix::hc::raw_mixes(spec, stream,
            {{}, transposed, base, scale, strix::hc::FnLayout::feature_major})), "F32 feature-major mixes bitwise");
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

void mix_epsilon_position() {
    strix::hc::Spec spec{.streams = 1, .hidden = 1, .site = strix::hc::Site::head, .norm_eps = 0.25F};
    constexpr std::array<float, 1> row{0.25F}, weight{1.0F};
    const strix::hc::SiteWeights weights{{}, weight, {}, {}};
    const float inside = strix::hc::raw_mixes(spec, row, weights).front();
    spec.norm_eps_outside_sqrt = true;
    const float outside = strix::hc::raw_mixes(spec, row, weights).front();
    check(std::fabs(inside - 0.25F / std::sqrt(0.3125F)) < 1e-6F, "mix epsilon inside RMS square root");
    check(outside == 0.5F, "mix epsilon outside RMS square root");
}

void collapse_variance_boundary() {
    using namespace strix::hc;
    const Spec spec{.streams = 1, .hidden = 1, .site = Site::head, .store = Store::bf16};
    constexpr std::array residual{1.0F}, pre{1.00390625F}, gain{1.0F};
    std::array<float, 1> output{};
    collapse_rms_bf16(spec, residual, pre, gain, 1e-6F, CollapseVariance::before_store, output);
    check(output[0] == 0.99609375F, "variance before store, BF16 numerator");
    collapse_rms_bf16(spec, residual, pre, gain, 1e-6F, CollapseVariance::after_store, output);
    check(output[0] == 1.0F, "variance and numerator after store");
}

void outside_root_sublayer_cases(std::mt19937& random) {
    using namespace strix::hc;
    constexpr Spec spec{.streams = 4, .hidden = 4096, .sinkhorn_iters = 20,
                        .store = Store::bf16, .pre_eps = 1e-6F, .comb_eps = 1e-6F,
                        .norm_eps = 0.5F, .norm_eps_outside_sqrt = true};
    constexpr float layer_epsilon = 1e-6F;
    constexpr std::size_t hidden = spec.hidden, flat = spec.flat(), rows = spec.mix_rows();
    constexpr std::array scales{1.0F, 1.0F, 1.0F};
    const auto base = uniform(random, rows, -0.1F, 0.1F);
    const auto gain = uniform(random, hidden, -1.0F, 1.0F, true);
    auto weight = uniform(random, rows * flat, -0.02F, 0.02F);
    auto pinned_post = std::span(weight).subspan(spec.streams * flat, flat);
    std::ranges::fill(pinned_post, 0.0F);
    pinned_post.front() = 1.5F;
    const SiteWeights weights{{}, weight, base, scales};
    enum class Input { full, broadcast, post };
    for (const auto count : {1U, 8U, 128U}) {
        for (const auto entry : {Input::full, Input::broadcast, Input::post}) {
            double root_gap = 0.0;
            for (unsigned token = 0; token < count; ++token) {
                auto residual = uniform(random, flat, -1.75F, 1.75F, true);
                residual.front() = 1.25F;
                if (entry == Input::broadcast)
                    for (std::size_t stream = 1; stream < spec.streams; ++stream)
                        std::copy_n(residual.begin(), hidden, residual.begin() + std::ptrdiff_t(stream * hidden));
                if (entry == Input::post) {
                    auto branch = uniform(random, hidden, -1.0F, 1.0F, true);
                    Mix previous{{}, uniform(random, spec.streams, 0.0F, 1.0F),
                                 uniform(random, spec.comb_rows(), 0.0F, 1.0F)};
                    branch.front() = 1.25F;
                    previous.post.front() = 0.75F;
                    for (std::size_t source = 0; source < spec.streams; ++source)
                        previous.comb[source * spec.streams] = source == 0U ? 0.5F : 0.0F;
                    std::vector<float> next(flat);
                    expand(spec, branch, residual, previous, next);
                    for (std::size_t stream = 0; stream < spec.streams; ++stream)
                        for (std::size_t i = 0; i < hidden; ++i) {
                            double wanted = double(branch[i]) * previous.post[stream];
                            for (std::size_t source = 0; source < spec.streams; ++source)
                                wanted += double(residual[source * hidden + i]) *
                                    previous.comb[source * spec.streams + stream];
                            wanted = store_value(static_cast<float>(wanted), Store::bf16);
                            check(std::fabs(next[stream * hidden + i] - wanted) <= 0.01 + 0.01 * std::fabs(wanted),
                                  "post branch and comb column belong to destination stream");
                        }
                    residual = std::move(next);
                }
                const auto got = mixes(spec, residual, weights);
                const auto sigmoid = [](double value) { return 1.0 / (1.0 + std::exp(-value)); };
                double squares = 0.0;
                for (auto value : residual) squares += double(value) * value;
                const double mean_square = squares / flat;
                const double inv = 1.0 / (std::sqrt(mean_square) + spec.norm_eps);
                const double inside = 1.0 / std::sqrt(mean_square + spec.norm_eps);
                std::array<double, rows> projected{};
                for (std::size_t mix = 0; mix < rows; ++mix)
                    for (std::size_t i = 0; i < flat; ++i)
                        projected[mix] += double(residual[i]) * weight[mix * flat + i];
                std::array<double, spec.streams> pre{};
                for (std::size_t stream = 0; stream < spec.streams; ++stream) {
                    pre[stream] = sigmoid(projected[stream] * inv + base[stream]) + spec.pre_eps;
                    const auto i = stream + spec.streams;
                    const double post = 2.0 * sigmoid(projected[i] * inv + base[i]);
                    root_gap = std::max(root_gap, std::fabs(post - 2.0 * sigmoid(projected[i] * inside + base[i])));
                    check(std::fabs(got.post[stream] - post) <= 0.005, "outside-root post coefficient");
                }
                std::array<std::array<double, spec.streams>, spec.streams> comb{};
                for (std::size_t source = 0; source < spec.streams; ++source) {
                    double total = 0.0;
                    for (std::size_t dest = 0; dest < spec.streams; ++dest) {
                        const auto i = 2U * spec.streams + source * spec.streams + dest;
                        comb[source][dest] = std::exp(projected[i] * inv + base[i]);
                        total += comb[source][dest];
                    }
                    for (auto& value : comb[source]) value = value / total + spec.comb_eps;
                }
                for (unsigned pass = 0; pass < spec.sinkhorn_iters; ++pass) {
                    if (pass != 0U)
                        for (auto& row : comb) {
                            double total = spec.comb_eps;
                            for (auto value : row) total += value;
                            for (auto& value : row) value /= total;
                        }
                    for (std::size_t dest = 0; dest < spec.streams; ++dest) {
                        double total = spec.comb_eps;
                        for (const auto& row : comb) total += row[dest];
                        for (auto& row : comb) row[dest] /= total;
                    }
                }
                for (std::size_t source = 0; source < spec.streams; ++source)
                    for (std::size_t dest = 0; dest < spec.streams; ++dest)
                        check(std::fabs(got.comb[source * spec.streams + dest] - comb[source][dest]) <= 0.005,
                              "outside-root residual mixing matrix");
                std::vector<float> normalized(hidden), oracle(hidden);
                collapse_rms_bf16(spec, residual, got.pre, gain, layer_epsilon,
                                  CollapseVariance::before_store, normalized);
                double layer_squares = 0.0;
                for (std::size_t i = 0; i < hidden; ++i) {
                    double sum = 0.0;
                    for (std::size_t stream = 0; stream < spec.streams; ++stream)
                        sum += double(residual[stream * hidden + i]) * pre[stream];
                    oracle[i] = store_value(static_cast<float>(sum), Store::bf16);
                    layer_squares += double(oracle[i]) * oracle[i];
                }
                const double inverse = 1.0 / std::sqrt(layer_squares / hidden + layer_epsilon);
                for (std::size_t i = 0; i < hidden; ++i) {
                    const float wanted = store_value(static_cast<float>(oracle[i] * inverse * gain[i]), Store::bf16);
                    check(std::fabs(normalized[i] - wanted) <= 0.05 + 0.01 * std::fabs(wanted),
                          "outside-root collapse and layer RMS");
                }
            }
            check(root_gap > 0.05, "outside-root case distinguishes the inside-root formula");
        }
    }
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

void interleave_is_feature_major() {
    strix::hc::Spec spec{};
    spec.streams = 3;
    spec.hidden = 2;
    const float streams[] = {0, 1, 10, 11, 20, 21};
    const auto ordered = strix::hc::interleave_features(spec, streams);
    const std::vector<float> expected{0, 10, 20, 1, 11, 21};
    check(ordered == expected, "feature interleave order");
}

int main() {
    std::mt19937 g(20261118U);
    ds41_bitwise(g);
    modelb_fp64(g);
    identity_is_a_skip();
    mix_epsilon_position();
    collapse_variance_boundary();
    outside_root_sublayer_cases(g);
    geometry_rejected();
    interleave_is_feature_major();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("hc contract tests PASS\n");
    return 0;
}
