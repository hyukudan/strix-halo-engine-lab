// e4b-eagle host tests: format round trip, contract semantics against an independent double-precision reference,
// g4-v1 recipe properties, state/protocol rules and exactness of the verification cycle against OFF.
#include "common.hpp"
#include "mock.hpp"
#include "gemma4/eagle/session.hpp"
#include "gemma4/kernels/ops.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>

using namespace eagle_test;
namespace eg = strix::gemma4::eagle;

namespace {
// ---------------------------------------------------------------- independent reference (CONTRACT.md literally)
// Double precision, plain softmax with the attention scale on the scores (not folded), libm tanh/exp, explicit
// key sets. It shares only the decoded weights, the RoPE table values and the embedding bf16 boundary.
class Naive final : public Drafter {
public:
    Naive(const Weights& w, eg::RopeView rope, LinearView emb) : w_(w), c_(w.config), rope_(rope), emb_(emb) {
        auto dec = [](const eg::Matrix& m) {
            std::vector<double> out(std::size_t(m.outputs) * m.inputs);
            const auto v = m.view();
            for (std::uint32_t o = 0; o < m.outputs; ++o)
                for (std::uint32_t i = 0; i < m.inputs; ++i) out[std::size_t(o) * m.inputs + i] = gpu::coefficient(v, o, i);
            return out;
        };
        fc_ = dec(w.fc); if (c_.concat) in_ = dec(w.in_proj);
        q_ = dec(w.q); k_ = dec(w.k); v_ = dec(w.v); o_ = dec(w.o); gate_ = dec(w.gate); up_ = dec(w.up); down_ = dec(w.down); head_ = dec(w.head);
    }
    const Config& config() const override { return c_; }
    std::uint64_t frontier() const override { return frontier_; }
    const eg::Timings& timings() const override { return t_; }
    void reset() override { rows_.clear(); frontier_ = 0; }
    void stage(const Rows& r) override {
        pending_.clear();
        const auto H = c_.hidden;
        for (std::size_t i = 0; i < r.positions.size(); ++i) {
            Row row;
            row.taps.assign(r.taps.begin() + std::ptrdiff_t(i * 5 * H), r.taps.begin() + std::ptrdiff_t((i + 1) * 5 * H));
            if (c_.needs_final()) row.hidden.assign(r.hidden.begin() + std::ptrdiff_t(i * H), r.hidden.begin() + std::ptrdiff_t((i + 1) * H));
            row.next = r.next_tokens[i];
            pending_.emplace_back(r.positions[i], std::move(row));
        }
    }
    void publish() override {
        for (auto& [p, row] : pending_) {
            compute(row, p);
            rows_[p] = std::move(row);
        }
        frontier_ = pending_.back().first + 1;
        pending_.clear();
    }
    void discard() noexcept override { pending_.clear(); }
    // margin[d] = top1 - top2 capped logit of depth d (for tolerance decisions).
    std::vector<double> margin;
    Draft draft(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) override {
        Draft out;
        out.depth = k;
        margin.assign(k, 0.0);
        const auto r = position - 1;
        const auto& last = rows_.at(r);
        if (last.next != anchor) throw std::runtime_error("naive anchor");
        std::vector<std::vector<double>> chain_k, chain_v;
        std::vector<double> state = last.x;
        auto run = [&](const std::vector<double>& a, std::uint64_t p, std::uint32_t depth) {
            const auto q = query(a, p);
            const auto att = attend(q, p, r, chain_k, chain_v);
            auto& s = state;
            auto o = mat(o_, att, c_.hidden);
            const auto b = norm(o, w_.post_attn_norm);
            for (std::size_t i = 0; i < s.size(); ++i) s[i] += b[i];
            const auto y = norm(s, w_.pre_ffn_norm);
            auto g = mat(gate_, y, c_.ffn);
            const auto u = mat(up_, y, c_.ffn);
            for (std::size_t i = 0; i < g.size(); ++i) {
                const double x = g[i];
                g[i] = 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))) * u[i];
            }
            const auto d = norm(mat(down_, g, c_.hidden), w_.post_ffn_norm);
            for (std::size_t i = 0; i < s.size(); ++i) s[i] += d[i];
            auto z = mat(head_, norm(s, w_.final_norm), c_.vocab);
            std::size_t best = 0;
            for (std::size_t i = 0; i < z.size(); ++i) {
                z[i] = c_.softcap * std::tanh(z[i] / c_.softcap);
                if (z[i] > z[best]) best = i;
            }
            double second = -1e300, sum = 0;
            for (std::size_t i = 0; i < z.size(); ++i) {
                if (i != best) second = std::max(second, z[i]);
                sum += std::exp(z[i] - z[best]);
            }
            out.draft_ids[depth] = std::uint32_t(best);
            out.tokens[depth] = w_.d2t[best];
            out.prob[depth] = float(1.0 / sum);
            margin[depth] = z[best] - second;
        };
        run(last.a, r, 0);
        for (std::uint32_t depth = 2; depth <= k; ++depth) {
            const auto p = r + depth - 1;
            const auto x = combine(state, out.tokens[depth - 2]);
            state = x;
            const auto a = norm(x, w_.input_norm);
            chain_k.push_back(key(a, p));
            chain_v.push_back(value(a));
            run(a, p, depth - 1);
        }
        return out;
    }
private:
    struct Row {
        std::vector<float> taps, hidden;
        std::uint32_t next = 0;
        std::vector<double> x, a, k, v;
    };
    std::vector<double> mat(const std::vector<double>& m, const std::vector<double>& x, std::uint32_t outputs) const {
        std::vector<double> y(outputs, 0.0);
        const auto in = x.size();
        for (std::uint32_t o = 0; o < outputs; ++o)
            for (std::size_t i = 0; i < in; ++i) y[o] += m[o * in + i] * x[i];
        return y;
    }
    std::vector<double> norm(const std::vector<double>& x, const std::vector<float>& w) const {
        double ss = 0;
        for (const auto a : x) ss += a * a;
        const double r = 1.0 / std::sqrt(ss / double(x.size()) + 1e-6);
        std::vector<double> y(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * r * (w.empty() ? 1.0 : double(w[i]));
        return y;
    }
    void rope(std::vector<double>& h, std::uint64_t p) const {
        const float* t = rope_.table.data() + p * 256;
        for (std::size_t i = 0; i < 128; ++i) {
            const double a = h[i], b = h[i + 128];
            h[i] = a * t[2 * i] - b * t[2 * i + 1];
            h[i + 128] = b * t[2 * i] + a * t[2 * i + 1];
        }
    }
    std::vector<double> heads(std::vector<double> x, std::uint32_t count, const std::vector<float>& w, std::uint64_t p, bool rotate) const {
        for (std::uint32_t h = 0; h < count; ++h) {
            std::vector<double> part(x.begin() + h * 256, x.begin() + (h + 1) * 256);
            part = norm(part, w);
            if (rotate) rope(part, p);
            std::copy(part.begin(), part.end(), x.begin() + h * 256);
        }
        return x;
    }
    std::vector<double> key(const std::vector<double>& a, std::uint64_t p) const { return heads(mat(k_, a, c_.kv_width()), c_.kv_heads, w_.k_norm, p, true); }
    std::vector<double> value(const std::vector<double>& a) const { return heads(mat(v_, a, c_.kv_width()), c_.kv_heads, {}, 0, false); }
    std::vector<double> query(const std::vector<double>& a, std::uint64_t p) const { return heads(mat(q_, a, c_.q_width()), c_.heads, w_.q_norm, p, true); }
    std::vector<double> combine(const std::vector<double>& h, std::uint32_t token) const {
        std::vector<double> e(c_.hidden);
        for (std::uint32_t i = 0; i < c_.hidden; ++i) e[i] = embedding_value(gpu::coefficient(emb_, token, i), 50.5F);
        const auto hn = norm(h, w_.h_norm), en = norm(e, w_.e_norm);
        if (!c_.concat) {
            std::vector<double> x(c_.hidden);
            for (std::size_t i = 0; i < x.size(); ++i) x[i] = hn[i] + en[i];
            return x;
        }
        std::vector<double> cat(hn);
        cat.insert(cat.end(), en.begin(), en.end());
        return mat(in_, cat, c_.hidden);
    }
    void compute(Row& row, std::uint64_t p) const {
        std::vector<double> tn;
        for (std::uint32_t j = 0; j < c_.tap_count(); ++j) {
            const auto slot = c_.tap_slot(j);
            const float* src = slot < 5 ? row.taps.data() + slot * c_.hidden : row.hidden.data();
            const auto part = norm(std::vector<double>(src, src + c_.hidden), w_.tap_norm[j]);
            tn.insert(tn.end(), part.begin(), part.end());
        }
        row.x = combine(mat(fc_, tn, c_.hidden), row.next);
        row.a = norm(row.x, w_.input_norm);
        row.k = key(row.a, p);
        row.v = value(row.a);
    }
    // Keys: committed depth-1 K/V of positions p' <= r with q - p' < 512, chain K/V r+1..q-1, and itself.
    std::vector<double> attend(const std::vector<double>& q, std::uint64_t p, std::uint64_t r,
                               const std::vector<std::vector<double>>& ck, const std::vector<std::vector<double>>& cv) const {
        std::vector<std::pair<const std::vector<double>*, const std::vector<double>*>> keys;
        for (std::uint64_t j = p >= 511 ? p - 511 : 0; j <= p; ++j) {
            if (j <= r) { const auto& row = rows_.at(j); keys.emplace_back(&row.k, &row.v); }
            else keys.emplace_back(&ck.at(j - r - 1), &cv.at(j - r - 1));
        }
        std::vector<double> out(c_.q_width(), 0.0);
        for (std::uint32_t h = 0; h < c_.heads; ++h) {
            const auto g = h / (c_.heads / c_.kv_heads);
            std::vector<double> s(keys.size());
            double m = -1e300;
            for (std::size_t j = 0; j < keys.size(); ++j) {
                double dot = 0;
                for (std::uint32_t i = 0; i < 256; ++i) dot += q[h * 256 + i] * (*keys[j].first)[g * 256 + i];
                s[j] = dot * double(c_.attention_scale);
                m = std::max(m, s[j]);
            }
            double den = 0;
            for (auto& x : s) { x = std::exp(x - m); den += x; }
            for (std::size_t j = 0; j < keys.size(); ++j)
                for (std::uint32_t i = 0; i < 256; ++i) out[h * 256 + i] += s[j] / den * (*keys[j].second)[g * 256 + i];
        }
        return out;
    }
    const Weights& w_;
    const Config& c_;
    eg::RopeView rope_;
    LinearView emb_;
    std::vector<double> fc_, in_, q_, k_, v_, o_, gate_, up_, down_, head_;
    std::map<std::uint64_t, Row> rows_;
    std::vector<std::pair<std::uint64_t, Row>> pending_;
    std::uint64_t frontier_ = 0;
    eg::Timings t_;
};

struct Fixture {
    Config c;
    Weights w;
    eg::Matrix emb;
    std::vector<float> rope;
    explicit Fixture(Config config, std::uint64_t seed = 7, WeightFormat f = WeightFormat::q8_0)
        : c(std::move(config)), w(Weights::random(c, seed, f)), emb(embedding_table(c, seed + 1)), rope(eg::make_rope_table(1024)) {}
    eg::RopeView rope_view() const { return {rope, 1024}; }
};

void contract_vs_naive(const Config& config, const char* name) {
    Fixture fx(config);
    eg::HostDrafter host(fx.w, fx.rope_view(), fx.emb.view());
    Naive naive(fx.w, fx.rope_view(), fx.emb.view());
    Scenario s{fx.c, 3, 37, 640};
    std::size_t drafts = 0, compared = 0;
    double worst = 0;
    lockstep(s, {&host, &naive}, [&](const Draft& h, const Draft& n, std::size_t, const std::string& label) {
        ++drafts;
        for (std::uint32_t d = 0; d < h.depth; ++d) {
            if (h.tokens[d] != n.tokens[d]) {
                // A different argmax is only admissible on a near tie of the exact reference; the chain then diverges.
                EAGLE_CHECK(naive.margin[d] < 1e-3, "%s %s depth %u token %u vs %u margin %g", name, label.c_str(), d + 1,
                            h.tokens[d], n.tokens[d], naive.margin[d]);
                break;
            }
            ++compared;
            const double err = std::fabs(double(h.prob[d]) - double(n.prob[d]));
            worst = std::max(worst, err);
            EAGLE_CHECK(err < 2e-3, "%s %s depth %u prob %g vs %g", name, label.c_str(), d + 1, double(h.prob[d]), double(n.prob[d]));
        }
    });
    std::printf("contract vs naive (%s): %zu drafts, %zu depths compared, max |dprob| %.3g\n", name, drafts, compared, worst);
    EAGLE_CHECK(compared > 500, "%s too few compared depths %zu", name, compared);
}

void format_round_trip() {
    for (const auto f : {WeightFormat::q8_0, WeightFormat::bf16}) {
        Fixture fx(small_config({5, 23, 35}, false, 1.0F), 11, f);
        const auto path = std::filesystem::temp_directory_path() / ("e4b-eagle-test-" + std::to_string(int(f)) + ".gguf");
        fx.w.save(path);
        const auto back = Weights::load(path);
        std::filesystem::remove(path);
        EAGLE_CHECK(back.config.taps == fx.w.config.taps && back.config.concat == fx.w.config.concat &&
                    back.config.attention_scale == fx.w.config.attention_scale && back.config.vocab == fx.w.config.vocab, "config");
        auto same = [](const eg::Matrix& a, const eg::Matrix& b) { return a.bytes == b.bytes && a.format == b.format && a.outputs == b.outputs; };
        EAGLE_CHECK(same(back.fc, fx.w.fc) && same(back.q, fx.w.q) && same(back.k, fx.w.k) && same(back.v, fx.w.v) &&
                    same(back.o, fx.w.o) && same(back.gate, fx.w.gate) && same(back.up, fx.w.up) && same(back.down, fx.w.down) &&
                    same(back.head, fx.w.head) && back.in_proj.bytes.empty(), "matrices");
        EAGLE_CHECK(back.tap_norm == fx.w.tap_norm && back.final_norm == fx.w.final_norm && back.q_norm == fx.w.q_norm &&
                    back.d2t == fx.w.d2t, "vectors");
    }
    Fixture fx(small_config());
    const auto path = std::filesystem::temp_directory_path() / "e4b-eagle-test-concat.gguf";
    fx.w.save(path);
    const auto back = Weights::load(path);
    std::filesystem::remove(path);
    EAGLE_CHECK(back.in_proj.bytes == fx.w.in_proj.bytes && back.config.needs_final(), "concat round trip");
    EAGLE_CHECK(throws([] { (void)Weights::load("/nonexistent/e4b-eagle.gguf"); }), "missing file");
    auto bad = small_config({7, 23, 35});
    EAGLE_CHECK(throws([&] { bad.validate(); }), "tap 7 is not exported by the target");
    auto scale = small_config();
    scale.attention_scale = 0.1F;
    EAGLE_CHECK(throws([&] { scale.validate(); }), "non power-of-two scale");
}

// Folding a power-of-two attention scale into the Q norm is bit-identical to scaling scores (normal range).
void scale_fold() {
    Rng r{5};
    const auto table = eg::make_rope_table(64);
    std::size_t checked = 0;
    for (int trial = 0; trial < 200; ++trial) {
        float q[256], k[256], w[256], ws[256], qa[256], qb[256];
        for (int i = 0; i < 256; ++i) { q[i] = 3 * r.uniform(); k[i] = r.uniform(); w[i] = 1 + 0.3F * r.uniform(); ws[i] = w[i] * 0.0625F; }
        const float* t = table.data() + std::size_t(r.below(64)) * 256;
        auto run = [&](const float* weight, float* out) {
            rms_norm(q, weight, out, 256, norm_eps);
            for (int i = 0; i < 128; ++i) {
                const float a = out[i], b = out[i + 128];
                out[i] = a * t[2 * i] - b * t[2 * i + 1];
                out[i + 128] = b * t[2 * i] + a * t[2 * i + 1];
            }
        };
        run(w, qa); run(ws, qb);
        const float scaled = lane_dot(qa, k, 256) * 0.0625F, folded = lane_dot(qb, k, 256);
        EAGLE_CHECK(bit_cast<std::uint32_t>(scaled) == bit_cast<std::uint32_t>(folded), "trial %d %a vs %a", trial, double(scaled), double(folded));
        ++checked;
    }
    std::printf("scale fold: %zu trials bit-identical\n", checked);
}

void select_recipe() {
    Rng r{9};
    for (const std::uint32_t V : {1U, 7U, 1024U, 1025U, 5000U}) {
        std::vector<float> z(V);
        for (auto& x : z) x = 40 * r.uniform();
        const auto [id, prob] = eg::select(z, 30.0F);
        std::uint32_t best = 0;
        double sum = 0;
        for (std::uint32_t i = 0; i < V; ++i) if (std::tanh(z[i] / 30.0) > std::tanh(z[best] / 30.0)) best = i;
        for (std::uint32_t i = 0; i < V; ++i) sum += std::exp(30 * std::tanh(z[i] / 30.0) - 30 * std::tanh(z[best] / 30.0));
        EAGLE_CHECK(id == best, "V %u argmax %u vs %u", V, id, best);
        EAGLE_CHECK(std::fabs(prob - 1 / sum) < 1e-5, "V %u prob %g vs %g", V, double(prob), 1 / sum);
    }
    std::vector<float> tie(3000, 1.0F);
    tie[2047] = 2.0F; tie[1023] = 2.0F;
    EAGLE_CHECK(eg::select(tie, 30.0F).first == 1023, "ties prefer the low draft id");
    std::vector<float> bad(10, 0.0F);
    bad[3] = std::numeric_limits<float>::quiet_NaN();
    EAGLE_CHECK(throws([&] { (void)eg::select(bad, 30.0F); }), "nonfinite logits refused");
}

// Accepted drafted positions are recomputed from target taps: ring K/V depend only on the committed row inputs, not
// on chunking or on what the chain computed for that position.
void prime_invariance() {
    Fixture fx(small_config());
    eg::HostDrafter a(fx.w, fx.rope_view(), fx.emb.view()), b(fx.w, fx.rope_view(), fx.emb.view());
    Scenario s{fx.c, 4, 2, 0};
    std::vector<std::uint32_t> next(24);
    Rng r{2};
    for (auto& t : next) t = r.below(fx.c.target_vocab);
    for (std::uint64_t f = 0; f < 24; f += 8) a.prime(make_rows(s, f, std::span(next).subspan(f, 8)).rows());
    for (std::uint64_t f = 0; f < 24; ++f) {
        if (f == 20) (void)b.draft(next[19], 20, 7); // chain K/V for 21..26 must never leak into the ring
        b.prime(make_rows(s, f, std::span(next).subspan(f, 1)).rows());
    }
    for (std::uint64_t p = 0; p < 24; ++p)
        EAGLE_CHECK(a.key(p) == b.key(p) && a.value(p) == b.value(p) && !a.key(p).empty(), "position %llu", (unsigned long long)p);
    const auto da = a.draft(next[23], 24, 7), db = b.draft(next[23], 24, 7);
    EAGLE_CHECK(da.tokens == db.tokens && da.prob == db.prob, "draft after different chunking");
}

void protocol_rules() {
    Fixture fx(small_config());
    eg::HostDrafter d(fx.w, fx.rope_view(), fx.emb.view());
    Scenario s{fx.c, 6, 2, 0};
    std::vector<std::uint32_t> next{5, 6, 7, 8, 9, 10, 11, 12, 13};
    EAGLE_CHECK(throws([&] { (void)d.draft(5, 0, 1); }), "draft before any committed row");
    d.prime(make_rows(s, 100, std::span(next).first(4)).rows()); // initial tail may start after zero
    EAGLE_CHECK(d.frontier() == 104, "frontier %llu", (unsigned long long)d.frontier());
    EAGLE_CHECK(throws([&] { (void)d.draft(8, 104, 2); }), "incomplete context tail (positions < 100 missing)");
    d.reset();
    d.prime(make_rows(s, 0, std::span(next).first(4)).rows());
    EAGLE_CHECK(throws([&] { d.stage(make_rows(s, 5, std::span(next).first(1)).rows()); }), "gap");
    EAGLE_CHECK(throws([&] { d.stage(make_rows(s, 3, std::span(next).first(1)).rows()); }), "rollback");
    EAGLE_CHECK(throws([&] { (void)d.draft(9, 4, 2); }), "anchor mismatch");
    EAGLE_CHECK(throws([&] { (void)d.draft(8, 5, 2); }), "position mismatch");
    EAGLE_CHECK(throws([&] { (void)d.draft(8, 4, 8); }), "depth > 7");
    const auto before = d.draft(8, 4, 3);
    d.stage(make_rows(s, 4, std::span(next).subspan(4, 2)).rows());
    EAGLE_CHECK(throws([&] { (void)d.draft(8, 4, 3); }), "draft while staged");
    d.discard();
    const auto after = d.draft(8, 4, 3);
    EAGLE_CHECK(before.tokens == after.tokens && before.prob == after.prob && d.frontier() == 4, "discard leaves state unchanged");
}

std::vector<std::uint32_t> off_decode(const std::vector<std::uint32_t>& prompt, std::size_t budget) {
    MockTarget t;
    MockStream s;
    for (std::size_t i = 0; i + 1 < prompt.size(); ++i) {
        auto tx = s.begin();
        RowPlan p{0, 0, i, prompt[i], tx.get(), 0, true, false};
        (void)t.forward_rows({&p, 1});
        tx->commit(1);
    }
    std::vector<std::uint32_t> out;
    std::uint32_t anchor = prompt.back();
    while (out.size() < budget) {
        auto tx = s.begin();
        RowPlan p{0, 0, s.tokens.size(), anchor, tx.get(), 0, true, false};
        const auto f = t.forward_rows({&p, 1});
        tx->commit(1);
        anchor = std::uint32_t(std::max_element(f.logits.begin(), f.logits.end()) - f.logits.begin());
        out.push_back(anchor);
        if (anchor == t.eos) break;
    }
    return out;
}
struct OnRun {
    std::vector<std::uint32_t> ids;
    dflash::Acceptance stats;
    std::uint64_t target_rows = 0;
};
OnRun on_decode(const std::vector<std::uint32_t>& prompt, std::size_t budget, int kind) {
    MockTarget t;
    MockStream s;
    Fixture fx(small_config(kind == 2 ? std::vector<std::int32_t>{5, 23, 35} : std::vector<std::int32_t>{11, 23, eg::final_tap}, kind != 2), 21);
    std::unique_ptr<Drafter> d;
    if (kind == 0) d = std::make_unique<Oracle>(t, s, fx.c);
    else d = std::make_unique<eg::HostDrafter>(fx.w, fx.rope_view(), fx.emb.view());
    const std::uint32_t eos[] = {t.eos};
    auto anchor = eg::prefill(t, [&] { return s.begin(); }, *d, prompt);
    OnRun run;
    for (std::uint32_t cycle = 0; run.ids.size() < budget; ++cycle) {
        auto tx = s.begin();
        const auto r = eg::run_cycle(t, *tx, *d, anchor, 1 + cycle % 7, budget - run.ids.size(), eos, &run.stats);
        run.ids.insert(run.ids.end(), r.verdict.emitted.begin(), r.verdict.emitted.end());
        anchor = r.next_anchor;
        EAGLE_CHECK(d->frontier() == s.tokens.size(), "drafter frontier follows target commits");
        if (r.verdict.stopped) break;
    }
    run.target_rows = t.rows;
    return run;
}
void cycle_exactness() {
    Rng r{77};
    std::size_t sequences = 0, accepted = 0, attempted = 0;
    for (int trial = 0; trial < 6; ++trial) {
        std::vector<std::uint32_t> prompt(1 + r.below(30));
        for (auto& t : prompt) t = 2 + r.below(1000);
        const std::size_t budget = 20 + r.below(120);
        const auto off = off_decode(prompt, budget);
        for (int kind = 0; kind < 3; ++kind) {
            const auto on = on_decode(prompt, budget, kind);
            EAGLE_CHECK(on.ids == off, "trial %d drafter %d: ids differ from OFF (%zu vs %zu tokens)", trial, kind, on.ids.size(), off.size());
            accepted += on.stats.accepted; attempted += on.stats.attempted;
            ++sequences;
        }
    }
    std::printf("cycle exactness: %zu sequences identical to OFF (oracle+2 real drafters), accepted %zu/%zu proposals\n",
                sequences, accepted, attempted);
    EAGLE_CHECK(accepted > 0, "oracle acceptance path exercised");
}

void session_adapter() {
    Fixture fx(small_config());
    auto factory = eg::host_drafts(fx.w, fx.rope_view(), fx.emb.view());
    auto session = factory();
    auto* eagle_session = dynamic_cast<eg::Session*>(session.get());
    EAGLE_CHECK(eagle_session != nullptr, "factory returns an eagle session");
    if (!eagle_session) return;
    Scenario s{fx.c, 8, 2, 0};
    std::vector<std::uint32_t> next{3, 4, 5};
    const auto rows = make_rows(s, 0, next);
    EAGLE_CHECK(throws([&] { session->stage(rows.taps, rows.positions); }), "legacy stage refuses without next tokens");
    EAGLE_CHECK(session->propose(9, 0) == (std::array<std::uint32_t, 7>{}), "no committed rows: empty proposal");
    EAGLE_CHECK(eagle_session->wants_hidden(), "tap set C wants the final hidden row");
    EAGLE_CHECK(eagle_session->wants_taps(), "EAGLE sessions consume target taps (prompt prefill must keep them)");
    eagle_session->stage_rows(rows.taps, rows.hidden, rows.next, rows.positions);
    session->publish();
    const auto p = eagle_session->propose_window(5, 3, 3);
    EAGLE_CHECK(p[0] == eagle_session->last_draft().tokens[0] && p[3] == 0 && eagle_session->last_draft().depth == 3, "window 3");
    session->discard();
}
} // namespace

int main() {
    try {
        format_round_trip();
        scale_fold();
        select_recipe();
        prime_invariance();
        protocol_rules();
        contract_vs_naive(small_config(), "taps C concat scale 1/16");
        contract_vs_naive(small_config({5, 23, 35}, false, 1.0F), "taps A sum scale 1");
        contract_vs_naive(small_config({5, 11, 17, 23, 35}), "taps dflash concat");
        cycle_exactness();
        session_adapter();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(eagle_test::failures ? "FAILED %d checks\n" : "ALL OK\n", eagle_test::failures);
    return eagle_test::failures ? 1 : 0;
}
