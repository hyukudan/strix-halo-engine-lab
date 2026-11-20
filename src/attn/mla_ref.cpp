// Host oracle of the shared MLA attention contract (include/strix/attn/mla_contract.hpp), fp64 accumulation.
#include "strix/attn/mla_contract.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace strix::attn::mla {

double LayerSpec::scale() const { return 1.0 / std::sqrt(static_cast<double>(g.head_qk())); }

namespace ref {
namespace {

using std::size_t;

void check(const LayerSpec& s, const Weights& w) {
    if (s.g.heads == 0U || s.g.kv_lora == 0U || s.g.v == 0U || w.kv_b == nullptr)
        throw std::invalid_argument("mla ref: empty geometry or missing kv_b");
    if (s.sink == Sink::Learned && w.sinks == nullptr) throw std::invalid_argument("mla ref: learned sink needs sinks");
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// key row of position p as the cache stores it: [latent | krope]
double key_dot(const LayerSpec& s, const Cache& c, const double* qt, std::uint32_t p) {
    const float* lat = c.latent + static_cast<size_t>(p) * s.g.kv_lora;
    double acc = 0.0;
    for (unsigned l = 0; l < s.g.kv_lora; ++l) acc += qt[l] * static_cast<double>(lat[l]);
    const float* kr = c.krope + static_cast<size_t>(p) * s.g.rope;
    for (unsigned i = 0; i < s.g.rope; ++i) acc += qt[s.g.kv_lora + i] * static_cast<double>(kr[i]);
    return acc;
}

// softmax weights of one head over n scores (+ optional sink) -> p (sum of p < 1 with a sink)
void softmax(const double* sc, unsigned n, bool sink, double z, double* p) {
    double m = sink ? z : -HUGE_VAL;
    for (unsigned j = 0; j < n; ++j) m = sc[j] > m ? sc[j] : m;
    double den = sink ? std::exp(z - m) : 0.0;
    for (unsigned j = 0; j < n; ++j) {
        p[j] = std::exp(sc[j] - m);
        den += p[j];
    }
    for (unsigned j = 0; j < n; ++j) p[j] /= den;
}

}  // namespace

unsigned select(const LayerSpec& s, unsigned t, const std::uint32_t* list, unsigned list_n, std::uint32_t* out) {
    if (s.sel.budget == 0U) throw std::invalid_argument("mla ref: zero selection budget");
    if (s.sel.kind == Select::Window) {
        const unsigned n = window_count(t, s.sel.budget);
        for (unsigned j = 0; j < n; ++j) out[j] = window_key(t, n, j);
        return n;
    }
    if (list_n > s.sel.budget) throw std::invalid_argument("mla ref: list longer than the budget");
    for (unsigned j = 0; j < list_n; ++j) {
        if (list[j] > t) throw std::invalid_argument("mla ref: list position after the query");
        if (j > 0U && list[j] <= list[j - 1U]) throw std::invalid_argument("mla ref: list not ascending / unique");
        out[j] = list[j];
    }
    return list_n;
}

void absorb(const LayerSpec& s, const Weights& w, const float* q, double* qt) {
    check(s, w);
    const Geometry& g = s.g;
    for (unsigned h = 0; h < g.heads; ++h) {
        const float* qh = q + static_cast<size_t>(h) * g.head_qk();
        double* th = qt + static_cast<size_t>(h) * g.d_qk();
        for (unsigned l = 0; l < g.kv_lora; ++l) {
            double acc = 0.0;
            for (unsigned d = 0; d < g.nope; ++d)
                acc += static_cast<double>(qh[d]) * w.kv_b[(static_cast<size_t>(h) * (g.nope + g.v) + d) * g.kv_lora + l];
            th[l] = acc;
        }
        for (unsigned i = 0; i < g.rope; ++i) th[g.kv_lora + i] = qh[g.nope + i];
    }
}

void latent_attend(const LayerSpec& s, const Weights& w, const Cache& c, const double* qt, const std::uint32_t* pos,
                   unsigned n, double* o) {
    check(s, w);
    const Geometry& g = s.g;
    for (unsigned j = 0; j < n; ++j)
        if (pos[j] >= c.positions) throw std::invalid_argument("mla ref: position outside the cache");
    std::vector<double> sc(n), p(n);
    const double scale = s.scale();
    for (unsigned h = 0; h < g.heads; ++h) {
        double* oh = o + static_cast<size_t>(h) * g.kv_lora;
        for (unsigned l = 0; l < g.kv_lora; ++l) oh[l] = 0.0;
        if (n == 0U) continue;
        const double* th = qt + static_cast<size_t>(h) * g.d_qk();
        for (unsigned j = 0; j < n; ++j) sc[j] = scale * key_dot(s, c, th, pos[j]);
        softmax(sc.data(), n, s.sink == Sink::Learned, s.sink == Sink::Learned ? static_cast<double>(w.sinks[h]) : 0.0, p.data());
        for (unsigned j = 0; j < n; ++j) {
            const float* lat = c.latent + static_cast<size_t>(pos[j]) * g.kv_lora;
            for (unsigned l = 0; l < g.kv_lora; ++l) oh[l] += p[j] * static_cast<double>(lat[l]);
        }
    }
}

void out_project(const LayerSpec& s, const Weights& w, const double* o, const float* gate, double* y) {
    check(s, w);
    const Geometry& g = s.g;
    if (s.gate != Gate::None && gate == nullptr) throw std::invalid_argument("mla ref: gate logits required");
    for (unsigned h = 0; h < g.heads; ++h)
        for (unsigned e = 0; e < g.v; ++e) {
            const float* wv = w.kv_b + (static_cast<size_t>(h) * (g.nope + g.v) + g.nope + e) * g.kv_lora;
            const double* oh = o + static_cast<size_t>(h) * g.kv_lora;
            double acc = 0.0;
            for (unsigned l = 0; l < g.kv_lora; ++l) acc += oh[l] * static_cast<double>(wv[l]);
            if (s.gate == Gate::SigmoidElementwise) acc *= sigmoid(gate[static_cast<size_t>(h) * g.v + e]);
            y[static_cast<size_t>(h) * g.v + e] = acc;
        }
}

void attend_row(const LayerSpec& s, const Weights& w, const Cache& c, const float* q, unsigned t, const std::uint32_t* list,
                unsigned list_n, const float* gate, double* y) {
    std::vector<std::uint32_t> pos(s.sel.budget);
    const unsigned n = select(s, t, list, list_n, pos.data());
    std::vector<double> qt(static_cast<size_t>(s.g.heads) * s.g.d_qk()), o(static_cast<size_t>(s.g.heads) * s.g.kv_lora);
    absorb(s, w, q, qt.data());
    latent_attend(s, w, c, qt.data(), pos.data(), n, o.data());
    out_project(s, w, o.data(), gate, y);
}

void attend_row_unabsorbed(const LayerSpec& s, const Weights& w, const Cache& c, const float* q, unsigned t,
                           const std::uint32_t* list, unsigned list_n, const float* gate, double* y) {
    check(s, w);
    const Geometry& g = s.g;
    std::vector<std::uint32_t> pos(s.sel.budget);
    const unsigned n = select(s, t, list, list_n, pos.data());
    for (unsigned j = 0; j < n; ++j)
        if (pos[j] >= c.positions) throw std::invalid_argument("mla ref: position outside the cache");
    if (s.gate != Gate::None && gate == nullptr) throw std::invalid_argument("mla ref: gate logits required");
    std::vector<double> k(static_cast<size_t>(n) * g.head_qk()), v(static_cast<size_t>(n) * g.v), sc(n), p(n);
    const double scale = s.scale();
    for (unsigned h = 0; h < g.heads; ++h) {
        const size_t r0 = static_cast<size_t>(h) * (g.nope + g.v);
        for (unsigned j = 0; j < n; ++j) {  // expand: k = [W_k latent | k_rope], v = W_v latent
            const float* lat = c.latent + static_cast<size_t>(pos[j]) * g.kv_lora;
            for (unsigned d = 0; d < g.nope + g.v; ++d) {
                double acc = 0.0;
                for (unsigned l = 0; l < g.kv_lora; ++l) acc += static_cast<double>(w.kv_b[(r0 + d) * g.kv_lora + l]) * lat[l];
                if (d < g.nope) k[static_cast<size_t>(j) * g.head_qk() + d] = acc;
                else v[static_cast<size_t>(j) * g.v + (d - g.nope)] = acc;
            }
            for (unsigned i = 0; i < g.rope; ++i)
                k[static_cast<size_t>(j) * g.head_qk() + g.nope + i] = c.krope[static_cast<size_t>(pos[j]) * g.rope + i];
        }
        const float* qh = q + static_cast<size_t>(h) * g.head_qk();
        for (unsigned j = 0; j < n; ++j) {
            double acc = 0.0;
            for (unsigned d = 0; d < g.head_qk(); ++d) acc += static_cast<double>(qh[d]) * k[static_cast<size_t>(j) * g.head_qk() + d];
            sc[j] = acc * scale;
        }
        if (n > 0U)
            softmax(sc.data(), n, s.sink == Sink::Learned, s.sink == Sink::Learned ? static_cast<double>(w.sinks[h]) : 0.0, p.data());
        for (unsigned e = 0; e < g.v; ++e) {
            double acc = 0.0;
            for (unsigned j = 0; j < n; ++j) acc += p[j] * v[static_cast<size_t>(j) * g.v + e];
            if (s.gate == Gate::SigmoidElementwise) acc *= sigmoid(gate[static_cast<size_t>(h) * g.v + e]);
            y[static_cast<size_t>(h) * g.v + e] = acc;
        }
    }
}

void attend_chunk(const LayerSpec& s, const Weights& w, const Cache& c, const float* q, unsigned t0, unsigned rows,
                  const std::uint32_t* lists, unsigned list_stride, const unsigned* list_n, const float* gate, double* y) {
    const size_t qs = static_cast<size_t>(s.g.heads) * s.g.head_qk(), ys = static_cast<size_t>(s.g.heads) * s.g.v;
    for (unsigned r = 0; r < rows; ++r) {
        const bool list = s.sel.kind == Select::List;
        attend_row(s, w, c, q + r * qs, t0 + r, list ? lists + static_cast<size_t>(r) * list_stride : nullptr,
                   list ? list_n[r] : 0U, gate == nullptr ? nullptr : gate + r * ys, y + r * ys);
    }
}

}  // namespace ref
}  // namespace strix::attn::mla
