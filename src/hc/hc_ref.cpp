#include "strix/hc/hc_ref.hpp"
#include "core/bf16_reference.hpp"

#include <stdexcept>

namespace strix::hc {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

void sinkhorn(const Spec& spec, std::vector<float>& c) {
    const std::uint32_t h = spec.streams;
    const float eps = spec.comb_eps;
    for (std::uint32_t j = 0; j < h; ++j) {  // row softmax + eps
        float mx = c[j * h];
        for (std::uint32_t k = 1; k < h; ++k) mx = c[j * h + k] > mx ? c[j * h + k] : mx;
        float sum = 0.0F;
        for (std::uint32_t k = 0; k < h; ++k) {
            c[j * h + k] = ds41::declared_exp(c[j * h + k] - mx);
            sum = sum + c[j * h + k];
        }
        for (std::uint32_t k = 0; k < h; ++k) c[j * h + k] = c[j * h + k] / sum + eps;
    }
    std::vector<float> sum(h);
    const auto columns = [&] {
        for (std::uint32_t k = 0; k < h; ++k) {
            sum[k] = 0.0F;
            for (std::uint32_t j = 0; j < h; ++j) sum[k] = sum[k] + c[j * h + k];
        }
        for (std::uint32_t j = 0; j < h; ++j)
            for (std::uint32_t k = 0; k < h; ++k) c[j * h + k] = c[j * h + k] / (sum[k] + eps);
    };
    const auto rows = [&] {
        for (std::uint32_t j = 0; j < h; ++j) {
            sum[j] = 0.0F;
            for (std::uint32_t k = 0; k < h; ++k) sum[j] = sum[j] + c[j * h + k];
        }
        for (std::uint32_t j = 0; j < h; ++j)
            for (std::uint32_t k = 0; k < h; ++k) c[j * h + k] = c[j * h + k] / (sum[j] + eps);
    };
    columns();
    for (std::uint32_t it = 1; it < spec.sinkhorn_iters; ++it) {
        rows();
        columns();
    }
}

} // namespace

std::vector<float> raw_mixes(const Spec& spec, std::span<const float> streams, const SiteWeights& w) {
    require(spec.valid(), "hc: invalid spec");
    const std::size_t n = spec.flat();
    const std::size_t rows = spec.mix_rows();
    require(streams.size() == n, "hc: stream geometry");
    require(spec.fn == FnType::f16 ? w.fn_f16.size() == rows * n : w.fn_f32.size() == rows * n, "hc: fn geometry");
    require(w.layout == FnLayout::mix_major || spec.fn == FnType::f32, "hc: feature-major fn requires F32");
    const float mean_square =
        ds41::chunked_dot_of([&](std::size_t i) { return streams[i]; }, [&](std::size_t i) { return streams[i]; }, n) /
        static_cast<float>(n);
    const float r = mix_rms_inverse(mean_square, spec.norm_eps, spec.norm_eps_outside_sqrt);
    std::vector<float> mix(rows);
    for (std::size_t m = 0; m < rows; ++m) {
        float d;
        if (spec.fn == FnType::f16) {
            const std::uint16_t* row = w.fn_f16.data() + m * n;
            d = ds41::chunked_dot_of([&](std::size_t i) { return ds41::f16_to_float(row[i]); },
                                     [&](std::size_t i) { return streams[i]; }, n);
        } else if (w.layout == FnLayout::feature_major) {
            d = ds41::chunked_dot_of([&](std::size_t i) { return w.fn_f32[i * rows + m]; },
                                     [&](std::size_t i) { return streams[i]; }, n);
        } else {
            const float* row = w.fn_f32.data() + m * n;
            d = ds41::chunked_dot_of([&](std::size_t i) { return row[i]; }, [&](std::size_t i) { return streams[i]; }, n);
        }
        mix[m] = d * r;
    }
    return mix;
}

Mix split(const Spec& spec, std::span<const float> raw, std::span<const float> base, std::span<const float> scale) {
    require(spec.valid(), "hc: invalid spec");
    const std::uint32_t h = spec.streams;
    require(raw.size() == spec.mix_rows() && base.size() == spec.mix_rows() && scale.size() == spec.scale_count(),
            "hc: split geometry");
    Mix out;
    out.pre.resize(h);
    for (std::uint32_t j = 0; j < h; ++j) out.pre[j] = pre_coef(raw[j], scale[0], base[j], spec.pre_eps);
    if (spec.post_rows() == 0U) return out;
    out.post.resize(h);
    for (std::uint32_t j = 0; j < h; ++j)
        out.post[j] = post_coef(raw[h + j], scale[1], base[h + j], spec.post_magnitude, spec.post_eps);
    if (spec.comb_rows() == 0U) return out;
    out.comb.resize(std::size_t{h} * h);
    for (std::uint32_t m = 0; m < h * h; ++m) out.comb[m] = raw[2U * h + m] * scale[2] + base[2U * h + m];
    sinkhorn(spec, out.comb);
    return out;
}

Mix mixes(const Spec& spec, std::span<const float> streams, const SiteWeights& w) {
    return split(spec, raw_mixes(spec, streams, w), w.base, w.scale);
}

void collapse(const Spec& spec, std::span<const float> streams, std::span<const float> pre, std::span<float> out) {
    require(spec.valid() && streams.size() == spec.flat() && pre.size() == spec.streams && out.size() == spec.hidden,
            "hc: collapse geometry");
    for (std::size_t i = 0; i < spec.hidden; ++i)
        out[i] = collapse_at(streams.data(), spec.hidden, i, pre.data(), spec.streams, spec.store);
}

void collapse_rms_bf16(const Spec& spec, std::span<const float> streams, std::span<const float> pre,
                       std::span<const float> gain, float epsilon, CollapseVariance variance,
                       std::span<float> out) {
    const bool delayed_store = spec.store == Store::bf16 && variance == CollapseVariance::before_store;
    auto collapse_spec = spec;
    if (delayed_store) collapse_spec.store = Store::f32;
    collapse(collapse_spec, streams, pre, out);
    core::rms_norm_bf16_reference(out, gain, epsilon, delayed_store);
}

void expand(const Spec& spec, std::span<const float> sub, std::span<const float> residual, const Mix& mix,
            std::span<float> out) {
    require(spec.valid() && spec.site == Site::sublayer, "hc: expand needs a sublayer spec");
    require(sub.size() == spec.hidden && residual.size() == spec.flat() && out.size() == spec.flat() &&
                mix.post.size() == spec.streams && mix.comb.size() == std::size_t{spec.comb_rows()},
            "hc: expand geometry");
    const float* comb = mix.comb.empty() ? nullptr : mix.comb.data();
    for (std::uint32_t k = 0; k < spec.streams; ++k)
        for (std::size_t i = 0; i < spec.hidden; ++i)
            out[k * spec.hidden + i] = expand_at(sub[i], residual.data(), spec.hidden, i, k, mix.post[k], comb,
                                                 spec.streams, spec.store);
}

} // namespace strix::hc
