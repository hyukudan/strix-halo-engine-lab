// Hy4-flash DFly2 drafter host reference (contract: hy4_dfly_ref.hpp).
#include "hy4/ref/hy4_dfly_ref.hpp"

#include "strix/ds41_layer.hpp"
#include "strix/weight_archive.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace strix::hy4::dfly {
namespace {

using Vec = std::vector<float>;
constexpr const char* kPrefix = "model.dfly_layers.";

float bf(float x) { return ds41::round_to_bf16(x); }
float sigmoid(float x) { return ds41::declared_sigmoid(x); }
float silu(float g) { return g / (1.0F + ds41::declared_exp(-g)); }

float dot(const float* a, const float* b, std::size_t n) {
    float s = 0.0F;
    for (std::size_t i = 0; i < n; ++i) s = s + a[i] * b[i];
    return s;
}

// y[r] = bf(W[r] . x) for every row of W [rows][x.size()], rows split over the host threads (order-free per row).
Vec linear(const Vec& w, std::span<const float> x, std::size_t rows) {
    const std::size_t cols = x.size();
    if (w.size() != rows * cols) throw std::invalid_argument("dfly ref: linear shape");
    Vec y(rows);
    const auto work = [&](std::size_t first, std::size_t last) {
        for (std::size_t r = first; r < last; ++r) y[r] = bf(dot(w.data() + r * cols, x.data(), cols));
    };
    const std::size_t threads = rows * cols >= (std::size_t{1} << 20) ? std::max(1U, std::thread::hardware_concurrency()) : 1U;
    if (threads == 1) {
        work(0, rows);
        return y;
    }
    std::vector<std::thread> pool;
    const std::size_t step = (rows + threads - 1) / threads;
    for (std::size_t first = 0; first < rows; first += step) pool.emplace_back(work, first, std::min(rows, first + step));
    for (auto& t : pool) t.join();
    return y;
}

// RMSNorm with the model's double rounding: bf(w * bf(x * r)).
Vec rms(std::span<const float> x, const Vec& w, float eps) {
    float ss = 0.0F;
    for (const float v : x) ss = ss + v * v;
    const float r = 1.0F / ds41::declared_sqrt(ss / static_cast<float>(x.size()) + eps);
    Vec y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = bf(w[i] * bf(x[i] * r));
    return y;
}

void rope(float* x, std::uint32_t dims, std::uint32_t pos, float theta) {  // interleaved pairs, BF16 table (oracle form)
    for (std::uint32_t i = 0; i < dims / 2U; ++i) {
        const float inv = 1.0F / std::pow(theta, static_cast<float>(2U * i) / static_cast<float>(dims));
        const float angle = static_cast<float>(pos) * inv;
        const float c = bf(static_cast<float>(std::cos(static_cast<double>(angle))));
        const float s = bf(static_cast<float>(std::sin(static_cast<double>(angle))));
        const float a = x[2U * i], b = x[2U * i + 1U];
        x[2U * i] = bf(a * c - b * s);
        x[2U * i + 1U] = bf(b * c + a * s);
    }
}

Vec cat(std::span<const float> a, std::span<const float> b) {
    Vec y(a.begin(), a.end());
    y.insert(y.end(), b.begin(), b.end());
    return y;
}

// hidden_correction: down(silu(gate z) * up z), z = cat(hidden_norm(h), embed_norm(e)) (the publisher's order).
Vec correction(const Geometry& g, Weights& w, std::span<const float> e, std::span<const float> h) {
    const auto z = cat(rms(h, w.tensor("hidden_correction.hidden_norm.weight"), g.eps),
                       rms(e, w.tensor("hidden_correction.embed_norm.weight"), g.eps));
    const auto gt = linear(w.tensor("hidden_correction.gate_proj.weight"), z, g.hidden);
    const auto up = linear(w.tensor("hidden_correction.up_proj.weight"), z, g.hidden);
    Vec m(g.hidden);
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = bf(bf(silu(gt[i])) * up[i]);
    return linear(w.tensor("hidden_correction.down_proj.weight"), m, g.hidden);
}

std::string layer(std::uint32_t i, const char* name) { return std::to_string(i) + "." + name; }

class ShardWeights final : public Weights {
public:
    explicit ShardWeights(const std::filesystem::path& path) : file_(weights::read_safetensors(path)) {}
    const Vec& tensor(const std::string& name) override {
        if (const auto it = cache_.find(name); it != cache_.end()) return it->second;
        const std::string full = kPrefix + name;
        for (const auto& t : file_.tensors) {
            if (t.name != full) continue;
            if (t.dtype != "BF16") throw std::runtime_error("dfly shard: not BF16: " + full);
            Vec v(static_cast<std::size_t>(t.bytes / 2U));
            const auto* src = file_.storage->data() + t.offset;
            for (std::size_t i = 0; i < v.size(); ++i) {
                std::uint16_t lo = 0;
                std::memcpy(&lo, src + 2U * i, 2U);
                const std::uint32_t bits = std::uint32_t{lo} << 16U;
                std::memcpy(&v[i], &bits, 4U);
            }
            return cache_.emplace(name, std::move(v)).first->second;
        }
        throw std::runtime_error("dfly shard: missing tensor " + full);
    }
    Vec row(const std::string&, std::uint64_t) override { throw std::logic_error("dfly shard: row() unused"); }

private:
    weights::SafeFile file_;
    std::map<std::string, Vec> cache_;
};

class SyntheticWeights final : public Weights {
public:
    SyntheticWeights(const Geometry& g, std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        const auto add = [&](const std::string& name, std::size_t n, float scale) {
            std::uniform_real_distribution<float> u(-std::fabs(scale), std::fabs(scale));
            Vec v(n);
            for (auto& x : v) x = bf(scale < 0.0F ? 1.0F + 0.1F * u(rng) : u(rng));
            t_.emplace(name, std::move(v));
        };
        const std::size_t h = g.hidden, H = g.heads;
        for (std::uint32_t i = 0; i < g.layers; ++i) {
            add(layer(i, "input_layernorm.weight"), h, -0.1F);
            add(layer(i, "post_attention_layernorm.weight"), h, -0.1F);
            add(layer(i, "self_attn.q_a_proj.weight"), g.q_lora * h, 0.2F);
            add(layer(i, "self_attn.q_a_layernorm.weight"), g.q_lora, -0.1F);
            add(layer(i, "self_attn.q_b_proj.weight"), H * g.qk() * g.q_lora, 0.2F);
            add(layer(i, "self_attn.kv_a_proj_with_mqa.weight"), std::size_t{g.latent()} * h, 0.2F);
            add(layer(i, "self_attn.kv_a_layernorm.weight"), g.kv_lora, -0.1F);
            add(layer(i, "self_attn.kv_b_proj.weight"), H * (g.nope + g.v_head) * g.kv_lora, 0.2F);
            add(layer(i, "self_attn.learnable_sink_param"), H, 1.0F);
            add(layer(i, "self_attn.linear_gate.weight"), H * g.v_head * h, 0.2F);
            add(layer(i, "self_attn.o_proj.weight"), h * H * g.v_head, 0.2F);
            add(layer(i, "self_attn.post_proj_norm.weight"), h, -0.1F);
            add(layer(i, "mlp.gate_proj.weight"), std::size_t{g.ff} * h, 0.2F);
            add(layer(i, "mlp.up_proj.weight"), std::size_t{g.ff} * h, 0.2F);
            add(layer(i, "mlp.down_proj.weight"), h * g.ff, 0.2F);
        }
        add("mtp_fusion.enorm.weight", h, -0.1F);
        add("mtp_fusion.hnorm.weight", h, -0.1F);
        add("mtp_fusion.eh_proj.weight", h * 2U * h, 0.2F);
        add("mtp_fusion.gate", h, 0.2F);
        add("hidden_correction.embed_norm.weight", h, -0.1F);
        add("hidden_correction.hidden_norm.weight", h, -0.1F);
        add("hidden_correction.gate_proj.weight", h * 2U * h, 0.2F);
        add("hidden_correction.up_proj.weight", h * 2U * h, 0.2F);
        add("hidden_correction.down_proj.weight", h * h, 0.2F);
        add("confidence_head.proj.weight", h, 0.2F);
        add("confidence_head.proj.bias", 1, 0.2F);
        add("final_norm.weight", h, -0.1F);
    }
    const Vec& tensor(const std::string& name) override {
        const auto it = t_.find(name);
        if (it == t_.end()) throw std::runtime_error("dfly synthetic: missing tensor " + name);
        return it->second;
    }
    Vec row(const std::string&, std::uint64_t) override { throw std::logic_error("dfly synthetic: row() unused"); }

private:
    std::map<std::string, Vec> t_;
};

}  // namespace

Geometry Geometry::official() { return Geometry{}; }

Geometry Geometry::tiny() {
    Geometry g;
    g.hidden = 32; g.heads = 2; g.q_lora = 16; g.kv_lora = 16; g.nope = 8; g.rope = 4; g.v_head = 8; g.ff = 24;
    g.layers = 2; g.block = 4; g.mask_id = 5;
    return g;
}

Variant parse_variant(const std::string& text) {
    Variant v;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        if (item.empty()) continue;
        const auto eq = item.find('=');
        const std::string k = item.substr(0, eq), val = eq == std::string::npos ? std::string{} : item.substr(eq + 1);
        const auto bad = [&] { throw std::invalid_argument("dfly variant: " + item); };
        if (k == "h") v.hidden = val == "post" ? HiddenForm::post_norm : val == "pre" ? HiddenForm::pre_norm : (bad(), v.hidden);
        else if (k == "fus") v.fusion = val == "eh" ? FusionForm::eh : val == "e+g" ? FusionForm::e_plus_gate
                                       : val == "mix" ? FusionForm::sigmoid_mix : (bad(), v.fusion);
        else if (k == "hc") v.correction = val == "off" ? Correction::off : val == "h+" ? Correction::hidden_residual
                                           : val == "mask+" ? Correction::mask_residual : val == "slot0+" ? Correction::slot0_residual
                                           : val == "out+" ? Correction::output_residual
                                           : val == "chain" ? Correction::chain : (bad(), v.correction);
        else if (k == "mask") v.mask = val == "causal" ? Mask::causal : val == "bidir" ? Mask::bidirectional : (bad(), v.mask);
        else if (k == "cand") v.candidates = static_cast<std::uint32_t>(std::stoul(val));
        else bad();
    }
    return v;
}

std::unique_ptr<Weights> shard_weights(const std::filesystem::path& shard) { return std::make_unique<ShardWeights>(shard); }
std::unique_ptr<Weights> synthetic_weights(const Geometry& g, std::uint64_t seed) {
    return std::make_unique<SyntheticWeights>(g, seed);
}

Block draft(const Geometry& g, const Variant& v, Weights& w, Target& target, std::uint32_t anchor, std::uint32_t pos,
            std::span<const float> h_in, std::uint32_t rows) {
    if (rows < 2U || rows > g.block) throw std::invalid_argument("dfly ref: rows outside 2..block");
    if (pos == 0U) throw std::invalid_argument("dfly ref: the anchor needs a previous target row");
    if (h_in.size() != g.hidden) throw std::invalid_argument("dfly ref: hidden size");
    const std::size_t H = g.hidden, R = rows, NH = g.heads, qk = g.qk(), kl = g.kv_lora, vd = g.v_head;
    Vec h(h_in.begin(), h_in.end());
    const auto e0s = target.embed(anchor);
    const Vec e0(e0s.begin(), e0s.end());
    const auto ems = target.embed(g.mask_id);
    const Vec em(ems.begin(), ems.end());
    if (v.correction == Correction::hidden_residual) {
        const auto c = correction(g, w, e0, h);
        for (std::size_t i = 0; i < H; ++i) h[i] = bf(h[i] + c[i]);
    }
    // slot-0 fusion
    const auto eh = linear(w.tensor("mtp_fusion.eh_proj.weight"),
                           cat(rms(e0, w.tensor("mtp_fusion.enorm.weight"), g.eps), rms(h, w.tensor("mtp_fusion.hnorm.weight"), g.eps)), H);
    const auto& gate0 = w.tensor("mtp_fusion.gate");
    std::vector<Vec> x(R);
    x[0].resize(H);
    for (std::size_t i = 0; i < H; ++i) {
        switch (v.fusion) {
        case FusionForm::eh: x[0][i] = eh[i]; break;
        case FusionForm::e_plus_gate: x[0][i] = bf(e0[i] + bf(gate0[i] * eh[i])); break;
        case FusionForm::sigmoid_mix: {
            const float s = sigmoid(gate0[i]);
            x[0][i] = bf(s * eh[i] + (1.0F - s) * e0[i]);
            break;
        }
        }
    }
    if (v.correction == Correction::slot0_residual) {
        const auto c = correction(g, w, e0, h);
        for (std::size_t i = 0; i < H; ++i) x[0][i] = bf(x[0][i] + c[i]);
    }
    Vec fill = em;
    if (v.correction == Correction::mask_residual) {
        const auto c = correction(g, w, em, h);
        for (std::size_t i = 0; i < H; ++i) fill[i] = bf(em[i] + c[i]);
    }
    for (std::size_t j = 1; j < R; ++j) x[j] = fill;
    const float scale = 1.0F / ds41::declared_sqrt(static_cast<float>(qk));
    for (std::uint32_t i = 0; i < g.layers; ++i) {
        const auto& kvb = w.tensor(layer(i, "self_attn.kv_b_proj.weight"));  // [NH][nope + vd][kl]
        const auto& sinks = w.tensor(layer(i, "self_attn.learnable_sink_param"));
        // per row: query (absorbed: q_lat[head] = Wk[head]^T q_nope), roped query tail, gate, own latent row
        std::vector<Vec> qlat(R), qrot(R), gate(R), own(R);
        for (std::size_t j = 0; j < R; ++j) {
            const auto xn = rms(x[j], w.tensor(layer(i, "input_layernorm.weight")), g.eps);
            gate[j] = linear(w.tensor(layer(i, "self_attn.linear_gate.weight")), xn, NH * vd);
            const auto qa = rms(linear(w.tensor(layer(i, "self_attn.q_a_proj.weight")), xn, g.q_lora),
                                w.tensor(layer(i, "self_attn.q_a_layernorm.weight")), g.eps);
            auto q = linear(w.tensor(layer(i, "self_attn.q_b_proj.weight")), qa, NH * qk);
            const auto p = pos + static_cast<std::uint32_t>(j);
            qlat[j].assign(NH * kl, 0.0F);
            qrot[j].resize(NH * g.rope);
            for (std::size_t hd = 0; hd < NH; ++hd) {
                rope(q.data() + hd * qk + g.nope, g.rope, p, g.theta);
                std::copy_n(q.begin() + static_cast<std::ptrdiff_t>(hd * qk + g.nope), g.rope, qrot[j].begin() + static_cast<std::ptrdiff_t>(hd * g.rope));
                for (std::size_t k = 0; k < kl; ++k) {
                    float s = 0.0F;
                    for (std::size_t d = 0; d < g.nope; ++d) s = s + kvb[(hd * (g.nope + vd) + d) * kl + k] * q[hd * qk + d];
                    qlat[j][hd * kl + k] = s;
                }
            }
            auto ckv = linear(w.tensor(layer(i, "self_attn.kv_a_proj_with_mqa.weight")), xn, g.latent());
            own[j] = rms(std::span<const float>(ckv.data(), kl), w.tensor(layer(i, "self_attn.kv_a_layernorm.weight")), g.eps);
            rope(ckv.data() + kl, g.rope, p, g.theta);
            own[j].insert(own[j].end(), ckv.begin() + static_cast<std::ptrdiff_t>(kl), ckv.end());
        }
        // keys: target rows 0 .. pos-1, then block rows (causal: row j sees block rows <= j)
        std::vector<std::span<const float>> keys;
        for (std::uint32_t p = 0; p < pos; ++p) keys.push_back(target.latent(i, p));
        for (std::size_t j = 0; j < R; ++j) keys.emplace_back(own[j]);
        std::vector<Vec> nx(R);
        for (std::size_t j = 0; j < R; ++j) {
            const std::size_t visible = pos + (v.mask == Mask::causal ? j + 1U : R);
            Vec o(NH * vd);
            Vec s(visible);
            for (std::size_t hd = 0; hd < NH; ++hd) {
                float m = sinks[hd];
                for (std::size_t a = 0; a < visible; ++a) {
                    const auto k = keys[a];
                    s[a] = (dot(qlat[j].data() + hd * kl, k.data(), kl) + dot(qrot[j].data() + hd * g.rope, k.data() + kl, g.rope)) * scale;
                    m = std::max(m, s[a]);
                }
                float z = 0.0F;
                for (auto& e : s) { e = ds41::declared_exp(e - m); z = z + e; }
                z = z + ds41::declared_exp(sinks[hd] - m);
                Vec ol(kl, 0.0F);  // attention-weighted latent, then the own kv_b value rows
                for (std::size_t a = 0; a < visible; ++a) {
                    const float p = s[a] / z;
                    for (std::size_t k = 0; k < kl; ++k) ol[k] = ol[k] + p * keys[a][k];
                }
                for (std::size_t e = 0; e < vd; ++e) {
                    const float val = bf(dot(kvb.data() + (hd * (g.nope + vd) + g.nope + e) * kl, ol.data(), kl));
                    o[hd * vd + e] = bf(val * bf(sigmoid(gate[j][hd * vd + e])));
                }
            }
            const auto att = rms(linear(w.tensor(layer(i, "self_attn.o_proj.weight")), o, H), w.tensor(layer(i, "self_attn.post_proj_norm.weight")), g.eps);
            nx[j].resize(H);
            for (std::size_t c = 0; c < H; ++c) nx[j][c] = bf(x[j][c] + att[c]);
            const auto xf = rms(nx[j], w.tensor(layer(i, "post_attention_layernorm.weight")), g.eps);
            const auto gt = linear(w.tensor(layer(i, "mlp.gate_proj.weight")), xf, g.ff);
            const auto up = linear(w.tensor(layer(i, "mlp.up_proj.weight")), xf, g.ff);
            Vec mm(g.ff);
            for (std::size_t c = 0; c < g.ff; ++c) mm[c] = bf(bf(silu(gt[c])) * up[c]);
            const auto down = linear(w.tensor(layer(i, "mlp.down_proj.weight")), mm, H);
            for (std::size_t c = 0; c < H; ++c) nx[j][c] = bf(nx[j][c] + down[c]);
        }
        x = std::move(nx);
    }
    Block out;
    const auto& cw = w.tensor("confidence_head.proj.weight");
    const float cb = w.tensor("confidence_head.proj.bias").at(0);
    Vec logits;
    if (v.correction == Correction::output_residual) {
        for (std::size_t j = 0; j < R; ++j) {
            const auto c = correction(g, w, j == 0 ? e0 : em, x[j]);
            for (std::size_t i = 0; i < H; ++i) x[j][i] = bf(x[j][i] + c[i]);
        }
    }
    std::uint32_t prev = anchor;
    Vec base;
    std::vector<std::uint32_t> order;
    for (std::size_t j = 0; j < R; ++j) {
        auto y = rms(x[j], w.tensor("final_norm.weight"), g.eps);
        out.final_rows.insert(out.final_rows.end(), y.begin(), y.end());
        if (j == 0) continue;
        if (v.correction == Correction::chain) {
            if (v.candidates > 0U) target.logits(y, base);
            const auto pe = target.embed(prev);
            const auto c = correction(g, w, pe, y);
            for (std::size_t i = 0; i < H; ++i) y[i] = bf(y[i] + c[i]);
        }
        target.logits(y, logits);
        std::uint32_t pick = 0;
        if (v.correction == Correction::chain && v.candidates > 0U && v.candidates < base.size()) {
            // the N best uncorrected ids (value desc., lower id on ties), then the corrected argmax among them
            order.resize(base.size());
            for (std::uint32_t t = 0; t < order.size(); ++t) order[t] = t;
            const auto better = [&](std::uint32_t a, std::uint32_t b) { return base[a] > base[b] || (base[a] == base[b] && a < b); };
            std::partial_sort(order.begin(), order.begin() + v.candidates, order.end(), better);
            pick = order[0];
            for (std::uint32_t k = 1; k < v.candidates; ++k) {
                const auto t = order[k];
                if (logits[t] > logits[pick] || (logits[t] == logits[pick] && t < pick)) pick = t;
            }
        } else {
            pick = static_cast<std::uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        out.proposals.push_back(pick);
        out.confidence.push_back(sigmoid(dot(cw.data(), y.data(), H) + cb));
        prev = pick;
    }
    return out;
}

}  // namespace strix::hy4::dfly
