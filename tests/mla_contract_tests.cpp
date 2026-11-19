// Shared MLA attention contract: host oracle tests (absorbed == unabsorbed reference form, window / list selection,
// learned sink, sigmoid gate, prefill chunk == decode rows, presets and the HIP core fit).
#include "strix/attn/mla_contract.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace mla = strix::attn::mla;
// List (DSA) and window (SWA) layers with learned sink and sigmoid gate; a model binding keeps its own LayerSpec values
// in its model package.
constexpr mla::LayerSpec list_spec() {
    return {{32U, 512U, 64U, 192U, 256U}, {mla::Select::List, 2048U}, mla::Sink::Learned, mla::Gate::SigmoidElementwise, 1.0e7};
}
constexpr mla::LayerSpec window_spec() {
    return {{64U, 512U, 64U, 192U, 256U}, {mla::Select::Window, 512U}, mla::Sink::Learned, mla::Gate::SigmoidElementwise, 1.0e4};
}

namespace {

int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

struct Rng {
    std::uint64_t s;
    float next() {  // uniform [-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<double>(s >> 40U) / 8388608.0 - 1.0);
    }
};

struct Data {
    std::vector<float> kv_b, sinks, latent, krope;
    mla::ref::Weights w() const { return {kv_b.data(), sinks.empty() ? nullptr : sinks.data()}; }
    mla::ref::Cache c(unsigned positions) const { return {latent.data(), krope.data(), positions}; }
};

Data make(const mla::LayerSpec& s, unsigned positions, std::uint64_t seed) {
    Rng r{seed};
    Data d;
    const float wscale = 1.0F / std::sqrt(static_cast<float>(s.g.kv_lora));
    d.kv_b.resize(static_cast<std::size_t>(s.g.kv_b_rows()) * s.g.kv_lora);
    for (float& x : d.kv_b) x = r.next() * wscale * 4.0F;
    d.sinks.resize(s.g.heads);
    for (float& x : d.sinks) x = r.next() * 3.0F;
    d.latent.resize(static_cast<std::size_t>(positions) * s.g.kv_lora);
    for (float& x : d.latent) x = r.next();
    d.krope.resize(static_cast<std::size_t>(positions) * s.g.rope + 1U);
    for (float& x : d.krope) x = r.next();
    return d;
}

std::vector<float> fill(std::size_t n, std::uint64_t seed, float scale) {
    Rng r{seed};
    std::vector<float> v(n);
    for (float& x : v) x = r.next() * scale;
    return v;
}

double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num = std::fmax(num, std::fabs(a[i] - b[i]));
        den = std::fmax(den, std::fabs(b[i]));
    }
    return den > 0.0 ? num / den : num;
}

// absorbed (1)-(6) == unabsorbed reference form for one row
void absorbed_equals_reference(const mla::LayerSpec& s, unsigned t, std::vector<std::uint32_t> list, std::uint64_t seed) {
    const Data d = make(s, t + 1U, seed);
    const std::vector<float> q = fill(static_cast<std::size_t>(s.g.heads) * s.g.head_qk(), seed + 1U, 2.0F);
    const std::vector<float> g = fill(static_cast<std::size_t>(s.g.heads) * s.g.v, seed + 2U, 3.0F);
    std::vector<double> ya(static_cast<std::size_t>(s.g.heads) * s.g.v), yu(ya.size());
    const auto n = static_cast<unsigned>(list.size());
    mla::ref::attend_row(s, d.w(), d.c(t + 1U), q.data(), t, list.data(), n, g.data(), ya.data());
    mla::ref::attend_row_unabsorbed(s, d.w(), d.c(t + 1U), q.data(), t, list.data(), n, g.data(), yu.data());
    const double e = max_rel(ya, yu);
    std::printf("absorbed vs reference heads %u d_qk %u t %u: max rel %.3e\n", s.g.heads, s.g.d_qk(), t, e);
    CHECK(e < 1e-12);
}

void presets() {
    const mla::LayerSpec dsa = list_spec(), swa = window_spec(), glm = mla::glm5_dsa();
    CHECK(dsa.g.d_qk() == 576U && dsa.g.d_v() == 512U && dsa.g.head_qk() == 256U && dsa.g.heads == 32U);
    CHECK(swa.g.d_qk() == 576U && swa.g.heads == 64U && swa.sel.kind == mla::Select::Window && swa.sel.budget == 512U);
    CHECK(glm.g.d_qk() == 512U && glm.g.rope == 0U && glm.g.nope == 256U);
    CHECK(dsa.scale() == 0.0625 && swa.scale() == 0.0625 && glm.scale() == 0.0625);
    CHECK(mla::hip_core_fits(dsa, 2U) && mla::hip_core_fits(swa, 2U) && mla::hip_core_fits(dsa, 1U));
    CHECK(!mla::hip_core_fits(swa, 1U) && mla::hip_core_fits(glm, 2U) && !mla::hip_core_fits(dsa, 3U));
}

void window_selection() {
    const mla::LayerSpec s = window_spec();
    std::vector<std::uint32_t> out(512);
    CHECK(mla::ref::select(s, 0U, nullptr, 0U, out.data()) == 1U && out[0] == 0U);
    CHECK(mla::ref::select(s, 510U, nullptr, 0U, out.data()) == 511U && out[510] == 510U);
    CHECK(mla::ref::select(s, 511U, nullptr, 0U, out.data()) == 512U && out[0] == 0U && out[511] == 511U);
    CHECK(mla::ref::select(s, 512U, nullptr, 0U, out.data()) == 512U && out[0] == 1U && out[511] == 512U);
    CHECK(mla::ref::select(s, 1000U, nullptr, 0U, out.data()) == 512U && out[0] == 489U && out[511] == 1000U);
    static_assert(mla::window_count(5U, 512U) == 6U && mla::window_key(1000U, 512U, 0U) == 489U);
}

void list_validation() {
    const mla::LayerSpec s = list_spec();
    std::vector<std::uint32_t> out(2048);
    const std::uint32_t bad_order[] = {3U, 2U}, after[] = {1U, 9U}, ok[] = {0U, 4U, 7U};
    int thrown = 0;
    try { (void)mla::ref::select(s, 7U, bad_order, 2U, out.data()); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)mla::ref::select(s, 7U, after, 2U, out.data()); } catch (const std::invalid_argument&) { ++thrown; }
    CHECK(thrown == 2);
    CHECK(mla::ref::select(s, 7U, ok, 3U, out.data()) == 3U && out[2] == 7U);
}

// small geometry with the real structure (rope tail, sink, gate) for windows past their length
mla::LayerSpec small(mla::Select kind, unsigned budget) {
    return {{4U, 32U, 8U, 12U, 16U}, {kind, budget}, mla::Sink::Learned, mla::Gate::SigmoidElementwise, 1.0e4};
}

void sink_and_gate() {
    // q = 0 -> every score 0; sink z = ln n -> p_j = 1 / (2n): exactly half the sink-free output
    mla::LayerSpec s = small(mla::Select::Window, 8U);
    const unsigned t = 20U, n = 8U;
    Data d = make(s, t + 1U, 7U);
    for (float& z : d.sinks) z = static_cast<float>(std::log(static_cast<double>(n)));
    const std::vector<float> q(static_cast<std::size_t>(s.g.heads) * s.g.head_qk(), 0.0F);
    const std::vector<float> g = fill(static_cast<std::size_t>(s.g.heads) * s.g.v, 8U, 3.0F);
    std::vector<double> y1(static_cast<std::size_t>(s.g.heads) * s.g.v), y0(y1.size()), yg(y1.size());
    mla::ref::attend_row(s, d.w(), d.c(t + 1U), q.data(), t, nullptr, 0U, g.data(), y1.data());
    mla::LayerSpec s0 = s;
    s0.sink = mla::Sink::None;
    mla::ref::attend_row(s0, d.w(), d.c(t + 1U), q.data(), t, nullptr, 0U, g.data(), y0.data());
    double e = 0.0;
    for (std::size_t i = 0; i < y1.size(); ++i) e = std::fmax(e, std::fabs(y1[i] - 0.5 * y0[i]));
    std::printf("sink ln(n) halves the output: max abs %.3e\n", e);
    CHECK(e < 1e-6 * (1.0 + std::fabs(y0[0])));
    // gate logits 0 -> sigmoid 0.5 -> half the ungated output
    const std::vector<float> g0(g.size(), 0.0F);
    mla::LayerSpec su = s0;
    su.gate = mla::Gate::None;
    mla::ref::attend_row(s0, d.w(), d.c(t + 1U), q.data(), t, nullptr, 0U, g0.data(), yg.data());
    mla::ref::attend_row(su, d.w(), d.c(t + 1U), q.data(), t, nullptr, 0U, nullptr, y0.data());
    e = 0.0;
    for (std::size_t i = 0; i < yg.size(); ++i) e = std::fmax(e, std::fabs(yg[i] - 0.5 * y0[i]));
    CHECK(e == 0.0);
    // a sink at -1e30 is the sink-free softmax
    for (float& z : d.sinks) z = -1.0e30F;
    const std::vector<float> qr = fill(q.size(), 9U, 2.0F);
    mla::ref::attend_row(s, d.w(), d.c(t + 1U), qr.data(), t, nullptr, 0U, g.data(), y1.data());
    mla::ref::attend_row(s0, d.w(), d.c(t + 1U), qr.data(), t, nullptr, 0U, g.data(), y0.data());
    CHECK(max_rel(y1, y0) < 1e-15);
}

void chunk_equals_rows() {
    for (const mla::Select kind : {mla::Select::List, mla::Select::Window}) {
        const mla::LayerSpec s = small(kind, 8U);
        const unsigned t0 = 13U, rows = 3U, stride = 8U;
        const Data d = make(s, t0 + rows, 11U);
        const std::size_t qs = static_cast<std::size_t>(s.g.heads) * s.g.head_qk(), ys = static_cast<std::size_t>(s.g.heads) * s.g.v;
        const std::vector<float> q = fill(rows * qs, 12U, 2.0F), g = fill(rows * ys, 13U, 3.0F);
        const std::vector<std::uint32_t> lists = {0U, 5U, 13U, 0U, 0U, 0U, 0U, 0U, 1U, 2U, 3U, 14U, 0U, 0U, 0U, 0U,
                                                  7U, 8U, 9U, 10U, 11U, 12U, 13U, 15U};
        const unsigned list_n[] = {3U, 4U, 8U};
        std::vector<double> yc(rows * ys), yr(ys);
        mla::ref::attend_chunk(s, d.w(), d.c(t0 + rows), q.data(), t0, rows, lists.data(), stride, list_n, g.data(), yc.data());
        for (unsigned r = 0; r < rows; ++r) {
            const bool list = kind == mla::Select::List;
            mla::ref::attend_row(s, d.w(), d.c(t0 + rows), q.data() + r * qs, t0 + r, list ? lists.data() + r * stride : nullptr,
                                 list ? list_n[r] : 0U, g.data() + r * ys, yr.data());
            for (std::size_t i = 0; i < ys; ++i) CHECK(yc[r * ys + i] == yr[i]);
        }
        // the window rows past the window agree with the reference form too
        if (kind == mla::Select::Window) absorbed_equals_reference(s, 30U, {}, 21U);
    }
}

void empty_list_is_zero() {
    const mla::LayerSpec s = small(mla::Select::List, 8U);
    const Data d = make(s, 4U, 31U);
    const std::vector<float> q = fill(static_cast<std::size_t>(s.g.heads) * s.g.head_qk(), 32U, 2.0F);
    const std::vector<float> g = fill(static_cast<std::size_t>(s.g.heads) * s.g.v, 33U, 1.0F);
    std::vector<double> y(static_cast<std::size_t>(s.g.heads) * s.g.v, 1.0);
    mla::ref::attend_row(s, d.w(), d.c(4U), q.data(), 3U, nullptr, 0U, g.data(), y.data());
    for (const double x : y) CHECK(x == 0.0);
}

}  // namespace

int main() {
    try {
        presets();
        window_selection();
        list_validation();
        sink_and_gate();
        chunk_equals_rows();
        empty_list_is_zero();
        // real geometries: MODEL_B DSA (list, 32 heads, 576/512), MODEL_B SWA (window, 64 heads), GLM-5 NoPE (8 of 64 heads)
        absorbed_equals_reference(list_spec(), 40U, {0U, 3U, 4U, 9U, 17U, 22U, 23U, 30U, 31U, 32U, 38U, 39U, 40U}, 101U);
        absorbed_equals_reference(window_spec(), 15U, {}, 202U);
        mla::LayerSpec glm = mla::glm5_dsa();
        glm.g.heads = 8U;
        absorbed_equals_reference(glm, 25U, {1U, 2U, 8U, 16U, 24U, 25U}, 303U);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("mla contract tests PASS\n");
    return 0;
}
