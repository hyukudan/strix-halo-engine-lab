#include "reference.hpp"
#include "validation.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <cfenv>
#include <limits>
#include <stdexcept>

namespace strix::gemma4::prefill {
namespace {
void finite(std::span<const float> x) {
    for (float a : x) if (!std::isfinite(a)) throw std::runtime_error("nonfinite prefill reference execution");
}
std::vector<float> vector(LinearView w) {
    std::vector<float> out(w.inputs);
    for (unsigned i = 0; i < w.inputs; ++i) out[i] = weight_at(w, 0, i);
    return out;
}
void normalize(std::span<const float> x, LinearView weight, std::vector<float>& out, unsigned width) {
    const auto scale = weight.data ? vector(weight) : std::vector<float>{};
    out.resize(x.size());
    for (std::size_t i = 0; i < x.size(); i += width) {
        if (!std::isfinite(lane_dot(x.data() + i, x.data() + i, width))) throw std::runtime_error("reference norm overflow");
        rms_norm(x.data() + i, scale.empty() ? nullptr : scale.data(), out.data() + i, width, norm_eps);
    }
    finite(out);
}
} // namespace
struct Reference::Impl {
    struct Cache {
        unsigned width = 0, slots = 0;
        bool global = false;
        std::vector<float> k, v;
        std::vector<std::uint64_t> tags;
    };
    HostWeights w;
    RopeTables tables;
    unsigned admitted = 0;
    std::uint64_t frontier = 0;
    std::vector<Cache> cache;
    AttentionMode attention = AttentionMode::canonical;
    void linear(LinearView matrix, std::span<const float> x, unsigned rows, std::vector<float>& out, bool wmma) {
        if (wmma) out = emulate_linear(matrix, x, rows);
        else if (matrix.format == WeightFormat::exl3_mul1) {
            out.resize(std::size_t(rows) * matrix.outputs);
            exl3::linear(matrix, x.data(), rows, out.data());
        }
        else {
            out.resize(std::size_t(rows) * matrix.outputs);
            for (unsigned r = 0; r < rows; ++r)
                for (unsigned n = 0; n < matrix.outputs; ++n)
                    out[std::size_t(r) * matrix.outputs + n] = chunk16_dot([&](unsigned i) { return weight_at(matrix, n, i); }, x.data() + std::size_t(r) * matrix.inputs, matrix.inputs);
        }
        finite(out);
    }
    void head(std::vector<float>& x, LinearView weight, unsigned count, unsigned dim, unsigned rows, bool global) {
        std::vector<float> norm;
        normalize(x, weight, norm, dim); x = std::move(norm);
        const auto table = global ? tables.global : tables.sliding;
        for (unsigned r = 0; r < rows; ++r) {
            const auto base = std::size_t(frontier + r) * (global ? 512U : 256U);
            for (unsigned h = 0; h < count; ++h)
                for (unsigned i = 0; i < std::min(dim / 2, global ? 64U : 128U); ++i) {
                    const auto index = (std::size_t(r) * count + h) * dim + i;
                    const float a = x[index], b = x[index + dim / 2], cs = table[base + 2 * i], sn = table[base + 2 * i + 1];
                    x[index] = a * cs - b * sn; x[index + dim / 2] = b * cs + a * sn;
                }
        }
        finite(x);
    }
};
Reference::Reference(HostWeights weights, RopeTables tables, unsigned admitted, AttentionMode attention) : impl_(std::make_unique<Impl>()) {
    detail::validate_weights(weights);
    detail::validate_tables(tables);
    if (std::fegetround() != FE_TONEAREST || !admitted || admitted > weights.config.max_positions || admitted > tables.positions ||
        tables.sliding.size() != std::size_t(tables.positions) * 256 || tables.global.size() != std::size_t(tables.positions) * 512 ||
        weights.layer.size() != weights.config.layers)
        throw std::invalid_argument("prefill reference configuration");
    volatile float minimum = bit_cast<float>(1U), one = 1.0F;
    if (bit_cast<unsigned>(float(minimum * one)) != 1U) throw std::invalid_argument("prefill reference requires gradual underflow");
    auto& b = *impl_; b.w = std::move(weights); b.tables = std::move(tables); b.admitted = admitted;
    b.attention = attention;
    b.cache.resize(b.w.config.layers);
    for (unsigned l = 0; l < b.w.config.layers; ++l)
        if (b.w.config.layer[l].producer == l) {
            auto& c = b.cache[l];
            c.width = b.w.config.kv_heads * b.w.config.layer[l].head_dim;
            c.global = b.w.config.layer[l].attention == AttentionKind::global;
            c.slots = c.global ? admitted : b.w.config.window;
            c.k.resize(std::size_t(c.slots) * c.width); c.v.resize(c.k.size()); c.tags.resize(c.slots, UINT64_MAX);
        }
}
Reference::~Reference() = default;
std::uint64_t Reference::frontier() const { return impl_->frontier; }
std::string Reference::digest() const {
    const auto& b = *impl_;
    Sha256Fast h;
    h.update({reinterpret_cast<const std::byte*>(&b.frontier), sizeof(b.frontier)});
    for (const auto& c : b.cache) if (c.width) {
        const auto first = c.global || b.frontier < b.w.config.window ? 0 : b.frontier - b.w.config.window;
        for (auto p = first; p < b.frontier; ++p) {
            const auto slot = c.global ? p : p % c.slots;
            if (c.tags[slot] != p) throw std::runtime_error("reference KV tag missing");
            h.update({reinterpret_cast<const std::byte*>(&p), sizeof(p)});
            h.update({reinterpret_cast<const std::byte*>(c.k.data() + slot * c.width), c.width * sizeof(float)});
            h.update({reinterpret_cast<const std::byte*>(c.v.data() + slot * c.width), c.width * sizeof(float)});
        }
    }
    return model_content_identity_hex(h.finalize());
}
ChunkResult Reference::forward(std::span<const unsigned> tokens, bool matrix, bool all_logits) {
    auto& b = *impl_; const auto& w = b.w; const auto& c = w.config; const auto n = unsigned(tokens.size());
    if (!n || tokens.size() > chunk_rows || b.frontier > b.admitted || tokens.size() > b.admitted - b.frontier)
        throw std::invalid_argument("prefill reference rows");
    for (const auto token : tokens) if (token >= c.vocab) throw std::invalid_argument("prefill reference token");
    ChunkResult result; result.rows = n; result.vocab = c.vocab; result.tap_width = 5 * c.hidden;
    result.taps.resize(std::size_t(n) * result.tap_width);
    std::vector<float> residual(std::size_t(n) * c.hidden), x, q, k, v, att, out, gate, up, ple;
    std::vector<std::vector<float>> overlay_k(c.layers), overlay_v(c.layers);
    for (unsigned r = 0; r < n; ++r)
        for (unsigned i = 0; i < c.hidden; ++i) residual[std::size_t(r) * c.hidden + i] = embedding_value(weight_at(w.embedding, tokens[r], i), embedding_scale);
    finite(residual); b.linear(w.ple_model, residual, n, ple, matrix);
    for (auto& a : ple) a = a * model_ple_scale;
    normalize(ple, w.ple_model_norm, x, c.ple_dim); ple = std::move(x);
    for (unsigned r = 0; r < n; ++r)
        for (unsigned i = 0; i < c.layers * c.ple_dim; ++i) {
            auto& a = ple[std::size_t(r) * c.layers * c.ple_dim + i];
            a = (a + embedding_value(weight_at(w.ple_table, tokens[r], i), ple_embedding_scale)) * ple_combine_scale;
        }
    finite(ple);
    const auto add = [&](LinearView norm, LinearView scalar = {}) {
        normalize(out, norm, x, c.hidden);
        for (std::size_t i = 0; i < residual.size(); ++i) {
            const float a = residual[i] + x[i]; residual[i] = scalar.data ? a * weight_at(scalar, 0, 0) : a;
        }
        finite(residual);
    };
    for (unsigned l = 0; l < c.layers; ++l) {
        const auto& y = w.layer[l]; const auto& a = c.layer[l]; const auto d = a.head_dim;
        const auto qw = c.heads * d, kw = c.kv_heads * d; const bool global = a.attention == AttentionKind::global;
        normalize(residual, y.input_norm, x, c.hidden); b.linear(y.q, x, n, q, matrix); b.head(q, y.q_norm, c.heads, d, n, global);
        if (a.producer == l) {
            b.linear(y.k, x, n, k, matrix); b.linear(y.v, x, n, v, matrix); b.head(k, y.k_norm, c.kv_heads, d, n, global);
            normalize(v, {}, out, d); overlay_k[l] = std::move(k); overlay_v[l] = std::move(out);
        }
        const auto read = [&](std::uint64_t p, bool value) -> const float* {
            if (p >= b.frontier) return (value ? overlay_v[a.producer] : overlay_k[a.producer]).data() + std::size_t(p - b.frontier) * kw;
            const auto& cache = b.cache[a.producer]; const auto slot = global ? p : p % cache.slots;
            if (cache.tags[slot] != p) throw std::runtime_error("reference attention KV tag missing");
            return (value ? cache.v : cache.k).data() + slot * kw;
        };
        att.resize(std::size_t(n) * qw);
        for (unsigned r = 0; r < n; ++r) {
            const auto position = b.frontier + r, first = global || position < c.window ? 0 : position - c.window + 1;
            for (unsigned h = 0; h < c.heads; ++h) {
                if (matrix && b.attention == AttentionMode::wmma) {
                    const auto value = emulate_attention(std::span(q).subspan(std::size_t(r) * qw + h * d, d), position, global ? 0 : c.window,
                        [&](std::uint64_t p, bool is_value) { return read(p, is_value) + (h / group) * d; });
                    std::copy(value.begin(), value.end(), att.begin() + std::ptrdiff_t(std::size_t(r) * qw + h * d));
                    continue;
                }
                std::vector<float> scores(std::size_t(position - first + 1));
                float maximum = -std::numeric_limits<float>::infinity();
                for (auto p = first; p <= position; ++p) {
                    const float score = lane_dot(q.data() + std::size_t(r) * qw + h * d, read(p, false) + (h / group) * d, d);
                    if (!std::isfinite(score)) throw std::runtime_error("reference attention score overflow");
                    scores[std::size_t(p - first)] = score; maximum = std::max(maximum, score);
                }
                for (auto& score : scores) {
                    const float difference = score - maximum;
                    if (!std::isfinite(difference)) throw std::runtime_error("reference attention difference overflow");
                    score = declared_exp(difference);
                }
                float denominator = 0.0F;
                std::vector<float> numerator(d, 0.0F);
                for (auto chunk = first / attention_chunk; chunk <= position / attention_chunk; ++chunk) {
                    const auto start = std::max(first, chunk * attention_chunk), end = std::min(position + 1, (chunk + 1) * attention_chunk);
                    float part = 0.0F;
                    for (auto p = start; p < end; ++p) part = part + scores[std::size_t(p - first)];
                    denominator = denominator + part;
                    for (unsigned i = 0; i < d; ++i) {
                        float sum = 0.0F;
                        for (auto p = start; p < end; ++p) sum = sum + scores[std::size_t(p - first)] * read(p, true)[(h / group) * d + i];
                        numerator[i] = numerator[i] + sum;
                    }
                }
                for (unsigned i = 0; i < d; ++i) att[std::size_t(r) * qw + h * d + i] = numerator[i] / denominator;
            }
        }
        finite(att); b.linear(y.o, att, n, out, matrix); add(y.post_attention_norm);
        normalize(residual, y.pre_ffn_norm, x, c.hidden); b.linear(y.gate, x, n, gate, matrix); b.linear(y.up, x, n, up, matrix);
        for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = gelu_tanh(gate[i]) * up[i];
        finite(gate); b.linear(y.down, gate, n, out, matrix); add(y.post_ffn_norm);
        b.linear(y.ple_gate, residual, n, gate, matrix);
        for (unsigned r = 0; r < n; ++r)
            for (unsigned i = 0; i < c.ple_dim; ++i) gate[std::size_t(r) * c.ple_dim + i] = gelu_tanh(gate[std::size_t(r) * c.ple_dim + i]) * ple[(std::size_t(r) * c.layers + l) * c.ple_dim + i];
        finite(gate); b.linear(y.ple_projection, gate, n, out, matrix); add(y.post_ple_norm, y.scalar);
        for (unsigned t = 0; t < 5; ++t) if (tap_layers[t] == l)
            for (unsigned r = 0; r < n; ++r) std::copy_n(residual.data() + std::size_t(r) * c.hidden, c.hidden, result.taps.data() + std::size_t(r) * result.tap_width + t * c.hidden);
    }
    normalize(residual, w.final_norm, x, c.hidden);
    b.linear(w.head.data ? w.head : w.embedding, all_logits ? std::span<const float>(x) : std::span<const float>(x).last(c.hidden), all_logits ? n : 1, out, matrix);
    for (auto& a : out) a = softcap(a);
    finite(out); result.last_logits.assign(out.end() - c.vocab, out.end());
    if (all_logits) result.all_logits = std::move(out);
    // Publish only after every linear and nonlinear stage has passed.
    for (unsigned l = 0; l < c.layers; ++l) if (c.layer[l].producer == l) {
        auto& cache = b.cache[l];
        for (unsigned r = 0; r < n; ++r) {
            const auto position = b.frontier + r, slot = cache.global ? position : position % cache.slots;
            std::copy_n(overlay_k[l].data() + std::size_t(r) * cache.width, cache.width, cache.k.data() + slot * cache.width);
            std::copy_n(overlay_v[l].data() + std::size_t(r) * cache.width, cache.width, cache.v.data() + slot * cache.width);
            cache.tags[slot] = position;
        }
    }
    b.frontier += n; return result;
}
Result Reference::run(std::span<const unsigned> tokens) {
    Result result; result.frontier = frontier();
    for (const auto& chunk : plan(frontier(), tokens.size(), impl_->admitted)) {
        auto out = forward(tokens.first(chunk.rows)); tokens = tokens.subspan(chunk.rows);
        result.last_logits = std::move(out.last_logits);
        result.tail_taps.insert(result.tail_taps.end(), out.taps.begin(), out.taps.end());
        for (unsigned r = 0; r < chunk.rows; ++r) result.tail_positions.push_back(chunk.position + r);
        const auto extra = result.tail_positions.size() > 1024 ? result.tail_positions.size() - 1024 : 0;
        result.tail_positions.erase(result.tail_positions.begin(), result.tail_positions.begin() + std::ptrdiff_t(extra));
        result.tail_taps.erase(result.tail_taps.begin(), result.tail_taps.begin() + std::ptrdiff_t(extra * out.tap_width));
        result.frontier = frontier();
    }
    return result;
}
} // namespace strix::gemma4::prefill
