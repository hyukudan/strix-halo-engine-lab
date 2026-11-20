// Shared HIP hyper-connection core (src/core/hc_core.hpp) against the host oracle (include/strix/hc/hc_ref.hpp),
// bit for bit, on synthetic data (no weights):
//   identity bf16 sublayer (hc_fixture::identity_bf16_spec: 4 streams, identity comb, F32 fn, bf16 store, post eps) and head
//   (pre-only mixes + collapse), hidden 3072 (19.2 chunks of 640: a partial last chunk), 9 rows (two mix tiles),
//   expansion from a canonical rank sum (SubSum low/high, the TP2 form);
//   DeepSeek V4.1 (Sinkhorn comb 20 iterations, F16 fn, FP32 store), hidden 1536, with the null-mixes collapse (copy 0).
// Every mix, collapse and expand value is compared with memcmp; the first mismatch is printed.

#include "hc_spec_fixture.hpp"
#include "core/hc_core.hpp"
#include "strix/hc/hc_ref.hpp"
#include "strix/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace hc = strix::hc;
namespace hcc = strix::core::hc;

struct Lcg {
    std::uint64_t s;
    std::uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(s >> 32);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() >> 8) * 0x1p-24F; }
    // A finite F16 of magnitude 2^-8 .. 2^-1 with a random sign and mantissa.
    std::uint16_t f16() {
        const std::uint32_t r = next();
        return static_cast<std::uint16_t>(((r & 1U) << 15) | ((7U + (r >> 1) % 7U) << 10) | ((r >> 8) & 0x3ffU));
    }
};

template <class T> struct Dev {
    T* p = nullptr;
    explicit Dev(std::size_t n) { strix::hip_check(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T) + 4), "hipMalloc"); }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) {
        strix::hip_check(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> get(std::size_t n) const {
        std::vector<T> v(n);
        strix::hip_check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
        return v;
    }
};

int failures = 0;
bool same(const char* what, const char* name, std::size_t row, const float* a, const float* b, std::size_t n) {
    if (std::memcmp(a, b, n * sizeof(float)) == 0) return true;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t x, y;
        std::memcpy(&x, a + i, 4);
        std::memcpy(&y, b + i, 4);
        if (x != y) {
            std::printf("hc-core %s %s row %zu [%zu]: device %.9g (0x%08x) oracle %.9g (0x%08x)\n", name, what, row, i,
                        static_cast<double>(a[i]), x, static_cast<double>(b[i]), y);
            break;
        }
    }
    ++failures;
    return false;
}

// One Spec end to end: mixes of every row, collapse with the device mixes (or copy 0), expand when a sublayer site.
template <hc::Spec S>
void run_case(const char* name, const hc::Spec& ref_spec, unsigned rows, unsigned iters, float hc_eps, bool copy0, bool rank_sum,
              std::uint64_t seed) {
    using M = hcc::MixRow<S>;
    using Fn = hcc::FnElem<S>;
    const unsigned hidden = ref_spec.hidden, n = S.streams * hidden, mr = S.mix_rows();
    Lcg g{seed};
    std::vector<float> stream(std::size_t{rows} * n), base(mr), scale(ref_spec.scale_count());
    for (float& x : stream) x = hc::store_value(g.uniform(-2.0F, 2.0F), S.store);
    std::vector<Fn> fn(std::size_t{mr} * n);
    std::vector<float> fn_f(S.fn == hc::FnType::f32 ? fn.size() : 0);
    std::vector<std::uint16_t> fn_h(S.fn == hc::FnType::f16 ? fn.size() : 0);
    for (std::size_t i = 0; i < fn.size(); ++i) {
        if constexpr (S.fn == hc::FnType::f16) fn[i] = fn_h[i] = g.f16();
        else fn[i] = fn_f[i] = g.uniform(-0.05F, 0.05F);
    }
    for (float& x : base) x = g.uniform(-1.0F, 1.0F);
    for (float& x : scale) x = g.uniform(0.5F, 1.5F);
    hc::SiteWeights w;
    w.fn_f16 = fn_h;
    w.fn_f32 = fn_f;
    w.base = base;
    w.scale = scale;

    Dev<float> d_stream(stream), d_base(base), d_scale(scale), d_partial(hcc::mix_partial_floats<S>(rows, hidden));
    Dev<Fn> d_fn(fn);
    Dev<M> d_mix(rows);
    hipStream_t s = nullptr;
    strix::hip_check(hcc::enqueue_mixes<S>(d_stream.p, d_fn.p, d_base.p, d_scale.p, rows, hidden, iters, hc_eps, ref_spec.norm_eps,
                                           d_mix.p, d_partial.p, s),
                     "mixes");
    Dev<float> d_collapse(std::size_t{rows} * hidden);
    strix::hip_check(hcc::enqueue_collapse<S>(d_stream.p, copy0 ? static_cast<const M*>(nullptr) : d_mix.p, rows, hidden,
                                              d_collapse.p, s),
                     "collapse");
    std::vector<float> sub(std::size_t{rows} * hidden), low(sub.size()), high(sub.size()), residual(stream.size());
    for (float& x : sub) x = g.uniform(-1.0F, 1.0F);
    for (float& x : low) x = g.uniform(-0.5F, 0.5F);
    for (float& x : high) x = g.uniform(-0.25F, 0.25F);
    for (float& x : residual) x = hc::store_value(g.uniform(-2.0F, 2.0F), S.store);
    Dev<float> d_sub(sub), d_low(low), d_high(high), d_res(residual), d_expand(stream.size());
    if constexpr (S.post_rows() != 0U) {
        const hcc::SubSum terms{d_sub.p, rank_sum ? d_low.p : nullptr, rank_sum ? d_high.p : nullptr};
        strix::hip_check(hcc::enqueue_expand<S>(terms, d_res.p, d_mix.p, rows, hidden, d_expand.p, s), "expand");
        if (rank_sum) {
            Dev<float> canonical(stream.size());
            const hcc::CanonicalSubSum required{d_sub.p, d_low.p, d_high.p};
            strix::hip_check(hcc::enqueue_expand<S>(required, d_res.p, d_mix.p, rows, hidden, canonical.p, s), "canonical expand");
            const auto expected = d_expand.get(stream.size());
            const auto actual = canonical.get(stream.size());
            if (std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) != 0)
                throw std::runtime_error("canonical expand differs from optional rank sum");
        }
    }
    strix::hip_check(hipDeviceSynchronize(), "sync");
    const auto mixes = d_mix.get(rows);
    const auto collapsed = d_collapse.get(sub.size());
    const auto expanded = d_expand.get(stream.size());

    unsigned ok_rows = 0;
    for (unsigned r = 0; r < rows; ++r) {
        const std::span<const float> x(stream.data() + std::size_t{r} * n, n);
        const hc::Mix ref = hc::mixes(ref_spec, x, w);
        bool ok = same("pre", name, r, mixes[r].pre, ref.pre.data(), S.streams);
        if constexpr (S.post_rows() != 0U) ok = same("post", name, r, mixes[r].post, ref.post.data(), S.streams) && ok;
        if constexpr (S.comb_rows() != 0U) ok = same("comb", name, r, mixes[r].comb, ref.comb.data(), S.comb_rows()) && ok;

        std::vector<float> pre0(S.streams, 0.0F), out(hidden);
        pre0[0] = 1.0F;
        hc::collapse(ref_spec, x, copy0 ? std::span<const float>(pre0) : std::span<const float>(ref.pre), out);
        ok = same("collapse", name, r, collapsed.data() + std::size_t{r} * hidden, out.data(), hidden) && ok;

        if constexpr (S.post_rows() != 0U) {
            std::vector<float> sub_r(hidden), e(n);
            for (unsigned i = 0; i < hidden; ++i) {
                const std::size_t at = std::size_t{r} * hidden + i;
                sub_r[i] = rank_sum ? (sub[at] + low[at]) + high[at] : sub[at];
            }
            hc::expand(ref_spec, sub_r, std::span<const float>(residual.data() + std::size_t{r} * n, n), ref, e);
            ok = same("expand", name, r, expanded.data() + std::size_t{r} * n, e.data(), n) && ok;
        }
        ok_rows += ok ? 1U : 0U;
    }
    std::printf("hc-core %s rows %u hidden %u mix_rows %u: %u/%u rows bitwise (mixes, collapse%s)\n", name, rows, hidden, mr,
                ok_rows, rows, S.post_rows() != 0U ? ", expand" : "");
}

constexpr hc::Spec modelb_sub = hc_fixture::identity_bf16_spec(hc::Site::sublayer);
constexpr hc::Spec modelb_head = hc_fixture::identity_bf16_spec(hc::Site::head);
// DeepSeek V4.1 mHC as bound in src/hip_ds41_layer.hip (hc_spec), with the runtime fields the oracle reads.
constexpr hc::Spec ds41_device{.streams = 4, .hidden = 0, .site = hc::Site::sublayer, .comb = hc::Comb::sinkhorn,
                               .sinkhorn_iters = 0, .fn = hc::FnType::f16, .store = hc::Store::f32, .post_magnitude = 2.0F};
hc::Spec ds41_ref(std::uint32_t hidden, std::uint32_t iters) {
    hc::Spec s = ds41_device;
    s.hidden = hidden;
    s.sinkhorn_iters = iters;
    s.pre_eps = 1e-6F;
    s.comb_eps = 1e-6F;
    s.norm_eps = 1e-20F;
    return s;
}

} // namespace

int main() {
    try {
        run_case<modelb_sub>("modelb-sublayer", modelb_sub, 9, 0, modelb_sub.pre_eps, false, false, 1);
        run_case<modelb_sub>("modelb-sublayer-ranksum", modelb_sub, 3, 0, modelb_sub.pre_eps, false, true, 2);
        run_case<modelb_head>("modelb-head", modelb_head, 9, 0, modelb_head.pre_eps, false, false, 3);
        run_case<ds41_device>("ds41", ds41_ref(1536, 20), 9, 20, 1e-6F, false, true, 4);
        run_case<ds41_device>("ds41-copy0", ds41_ref(1536, 1), 2, 1, 1e-6F, true, false, 5);
    } catch (const std::exception& e) {
        std::printf("hc-core: error %s\n", e.what());
        return 2;
    }
    std::printf("hc-core: %s (%d mismatching value groups)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
