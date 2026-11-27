#include "strix/spec/dflash_drafter.hpp"

#include "ops/rope/rope.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight/tensor_access.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace strix::spec::dflash {
namespace {

using Vec = std::vector<float>;

const serve::Json& member(const serve::Json& object, std::string_view key) {
    const auto* value = object.find(key);
    if (!value || value->is_null()) throw std::invalid_argument("DFlash config: missing " + std::string(key));
    return *value;
}

std::uint32_t u32(const serve::Json& object, std::string_view key) {
    const auto value = member(object, key).as_int();
    if (value < 0 || value > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("DFlash config: out of range " + std::string(key));
    return static_cast<std::uint32_t>(value);
}

std::uint32_t u32_or(const serve::Json& object, std::string_view key, std::uint32_t fallback) {
    const auto* value = object.find(key);
    return value && !value->is_null() ? u32(object, key) : fallback;
}

float f32_or(const serve::Json& object, std::string_view key, float fallback) {
    const auto* value = object.find(key);
    return value && !value->is_null() ? static_cast<float>(value->as_double()) : fallback;
}

// Output rows [begin, end) of `count` split over `threads` workers; each output is written by exactly one worker,
// so the thread count never changes a bit.
template <class F>
void parallel_rows(std::uint32_t count, unsigned threads, F&& fn, std::uint32_t grain = 64U) {
    const unsigned workers = std::max(1U, std::min<unsigned>(threads, (count + grain - 1U) / grain));
    if (workers == 1U) {
        fn(0U, count);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(workers);
    const std::uint32_t chunk = (count + workers - 1U) / workers;
    for (unsigned w = 0; w < workers; ++w) {
        const std::uint32_t begin = std::min(count, w * chunk), end = std::min(count, begin + chunk);
        if (begin < end) pool.emplace_back([&fn, begin, end] { fn(begin, end); });
    }
    for (auto& t : pool) t.join();
}

// Fixed-order FP32 dot product (eight interleaved partial sums, then a fixed tree).
float dot(const float* a, const float* b, std::uint32_t n) {
    float s[8]{};
    std::uint32_t i = 0;
    for (; i + 8U <= n; i += 8U)
        for (unsigned l = 0; l < 8U; ++l) s[l] += a[i + l] * b[i + l];
    for (; i < n; ++i) s[0] += a[i] * b[i];
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

// y[r][o] = W[o] . x[r]; W is [out][in] row-major.
void linear(std::span<const float> w, std::uint32_t out, std::uint32_t in, const float* x, std::uint32_t rows, float* y,
            unsigned threads) {
    if (w.size() != std::size_t{out} * in) throw std::logic_error("DFlash: linear shape");
    parallel_rows(out, threads, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t o = begin; o < end; ++o) {
            const float* row = w.data() + std::size_t{o} * in;
            for (std::uint32_t r = 0; r < rows; ++r) y[std::size_t{r} * out + o] = dot(row, x + std::size_t{r} * in, in);
        }
    });
}

void rms_norm(const float* x, std::span<const float> gain, float eps, float* y) {
    const auto n = static_cast<std::uint32_t>(gain.size());
    double sum = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    const auto scale = static_cast<float>(1.0 / std::sqrt(sum / static_cast<double>(n) + static_cast<double>(eps)));
    for (std::uint32_t i = 0; i < n; ++i) y[i] = x[i] * scale * gain[i];
}

// NeoX RoPE (pair i = (i, i + half)); angles from the shared native table (no scaling, FP32).
void rope_half(float* x, std::uint32_t dims, std::uint32_t position, float theta) {
    const std::uint32_t half = dims / 2U;
    thread_local Vec c, s;
    c.resize(half);
    s.resize(half);
    ops::rope::build_native_table(theta, half, position, 1U, ops::rope::Round::none, c.data(), s.data());
    for (std::uint32_t i = 0; i < half; ++i) {
        const float a = x[i], b = x[i + half];
        x[i] = a * c[i] - b * s[i];
        x[i + half] = b * c[i] + a * s[i];
    }
}

float silu(float g) { return g / (1.0F + std::exp(-g)); }

std::string layer_name(std::uint32_t layer, std::string_view rest) {
    return "layers." + std::to_string(layer) + "." + std::string(rest);
}

class MapWeights final : public Weights {
public:
    std::map<std::string, Vec, std::less<>> tensors;
    [[nodiscard]] std::span<const float> tensor(const std::string& name) const override {
        const auto it = tensors.find(name);
        if (it == tensors.end()) throw std::invalid_argument("DFlash: no tensor " + name);
        return it->second;
    }
};

std::uint64_t count_of(const std::vector<std::uint64_t>& extents) {
    std::uint64_t n = 1;
    for (const auto e : extents) n *= e;
    return n;
}

}  // namespace

Config Config::from_json_text(std::string_view text) {
    const auto root = serve::parse_json(text);
    Config c;
    c.hidden = u32(root, "hidden_size");
    c.heads = u32(root, "num_attention_heads");
    c.kv_heads = u32(root, "num_key_value_heads");
    c.head_dim = u32_or(root, "head_dim", c.heads ? c.hidden / c.heads : 0U);
    c.intermediate = u32(root, "intermediate_size");
    c.layers = u32(root, "num_hidden_layers");
    c.vocab = u32(root, "vocab_size");
    c.eps = f32_or(root, "rms_norm_eps", 1e-6F);
    if (const auto* rope = root.find("rope_parameters"); rope && rope->is_object()) c.rope_theta = f32_or(*rope, "rope_theta", c.rope_theta);
    c.rope_theta = f32_or(root, "rope_theta", c.rope_theta);
    if (const auto* causal = root.find("is_causal"); causal && causal->is_bool() && causal->as_bool())
        throw std::invalid_argument("DFlash config: causal block rows are not implemented");
    if (const auto* act = root.find("hidden_act"); act && act->is_string() && act->as_string() != "silu")
        throw std::invalid_argument("DFlash config: hidden_act " + act->as_string());
    if (const auto* use = root.find("use_sliding_window"); use && use->is_bool() && use->as_bool())
        c.window = u32(root, "sliding_window");
    const auto& d = member(root, "dflash_config");
    c.block = u32(d, "block_size");
    c.mask_token = u32(d, "mask_token_id");
    for (const auto& v : member(d, "target_layer_ids").as_array()) c.target_layers.push_back(static_cast<std::uint32_t>(v.as_int()));
    c.conv_taps = u32_or(d, "conv_kernel_size", 0U);
    c.conv_group = c.conv_taps ? u32(d, "conv_group_size") : 0U;
    c.selector_rank = u32_or(d, "selector_rank", 0U);
    c.selector_top_k = c.selector_rank ? u32(d, "selector_top_k") : 0U;
    c.softcap = f32_or(d, "final_logit_softcapping", 0.0F);
    if (const auto* mode = d.find("selector_softcap_mode"); mode && mode->is_string()) {
        if (mode->as_string() == "after_correction") c.softcap_after_correction = true;
        else if (mode->as_string() == "before_correction") c.softcap_after_correction = false;
        else throw std::invalid_argument("DFlash config: selector_softcap_mode " + mode->as_string());
    } else {
        c.softcap_after_correction = false;  // the published default
    }
    c.embed_scale = f32_or(d, "input_embedding_scale", 1.0F);
    if (!c.hidden || !c.heads || !c.kv_heads || !c.head_dim || c.heads % c.kv_heads || c.head_dim % 2U || !c.layers ||
        c.block < 2U || c.target_layers.empty() || c.mask_token >= c.vocab || (c.conv_taps && (!c.conv_group || c.hidden % c.conv_group)) ||
        (c.selector_rank && (c.selector_top_k < 1U || c.selector_top_k > 64U)))
        throw std::invalid_argument("DFlash config: inconsistent geometry");
    return c;
}

std::vector<std::pair<std::string, std::vector<std::uint64_t>>> inventory(const Config& c) {
    const std::uint64_t h = c.hidden, q = std::uint64_t{c.heads} * c.head_dim, kv = std::uint64_t{c.kv_heads} * c.head_dim;
    std::vector<std::pair<std::string, std::vector<std::uint64_t>>> out{
        {"fc.weight", {h, c.tap_width()}}, {"hidden_norm.weight", {h}}, {"norm.weight", {h}}};
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        out.push_back({layer_name(l, "input_layernorm.weight"), {h}});
        out.push_back({layer_name(l, "post_attention_layernorm.weight"), {h}});
        out.push_back({layer_name(l, "self_attn.q_proj.weight"), {q, h}});
        out.push_back({layer_name(l, "self_attn.k_proj.weight"), {kv, h}});
        out.push_back({layer_name(l, "self_attn.v_proj.weight"), {kv, h}});
        out.push_back({layer_name(l, "self_attn.o_proj.weight"), {h, q}});
        out.push_back({layer_name(l, "self_attn.q_norm.weight"), {c.head_dim}});
        out.push_back({layer_name(l, "self_attn.k_norm.weight"), {c.head_dim}});
        out.push_back({layer_name(l, "mlp.gate_proj.weight"), {c.intermediate, h}});
        out.push_back({layer_name(l, "mlp.up_proj.weight"), {c.intermediate, h}});
        out.push_back({layer_name(l, "mlp.down_proj.weight"), {h, c.intermediate}});
        if (c.conv_taps)
            for (const auto* conv : {"attention_conv.", "mlp_conv."}) {
                out.push_back({layer_name(l, std::string(conv) + "base_kernel"), {2U, c.conv_taps, h}});
                out.push_back({layer_name(l, std::string(conv) + "kernel_projection.weight"),
                               {2ULL * c.conv_taps * c.conv_groups(), h}});
            }
    }
    if (c.selector_rank) {
        out.push_back({"candidate_selector.hidden_projection.weight", {c.selector_rank, h}});
        out.push_back({"candidate_selector.predecessor_codebook", {c.vocab, c.selector_rank}});
        out.push_back({"candidate_selector.successor_codebook", {c.vocab, c.selector_rank}});
    }
    return out;
}

Config load_config(const std::filesystem::path& dir) {
    std::ifstream file(dir / "config.json", std::ios::binary);
    if (!file) throw std::runtime_error("DFlash: cannot read " + (dir / "config.json").string());
    std::ostringstream text;
    text << file.rdbuf();
    return Config::from_json_text(text.str());
}

Schema schema(const Config& config) {
    Schema out;
    for (const auto& [name, extents] : inventory(config)) out.specs.push_back({.name = name, .wire_dtype = "BF16", .extents = extents});
    return out;
}

weight::TensorLookup open_lookup(const Config& config, const std::filesystem::path& dir) {
    const auto owned = schema(config);  // the schema view must outlive the open
    weight::TensorLookup lookup;
    if (const auto status = lookup.open_safetensors((dir / "model.safetensors").string(), owned.view()); !status)
        throw std::runtime_error("DFlash weights: " + status.detail);
    return lookup;
}

weight::ResidentSlicePlan resident_plan(const Config& config, const weight::TensorLookup& lookup, std::size_t alignment) {
    // NamedTensor holds a view of the name: `names` must outlive make(), which copies what it keeps.
    const auto names = inventory(config);
    std::vector<weight::ResidentSlicePlan::NamedTensor> tensors;
    tensors.reserve(names.size());
    for (const auto& [name, extents] : names) tensors.push_back({name, lookup.find(name)});
    return weight::ResidentSlicePlan::make(std::move(tensors), alignment);
}

std::unique_ptr<Weights> load_weights(const Config& config, const std::filesystem::path& dir) {
    const auto names = inventory(config);
    const auto lookup = open_lookup(config, dir);
    auto weights = std::make_unique<MapWeights>();
    for (const auto& [name, extents] : names) {
        Vec v(count_of(extents));
        if (const auto status = lookup.read_floats(name, v); !status)
            throw std::runtime_error("DFlash weights: " + name + ": " + status.detail);
        weights->tensors.emplace(name, std::move(v));
    }
    return weights;
}

std::unique_ptr<Weights> synthetic_weights(const Config& config, std::uint64_t seed) {
    auto weights = std::make_unique<MapWeights>();
    std::mt19937_64 rng(seed);
    for (const auto& [name, extents] : inventory(config)) {
        Vec v(count_of(extents));
        const bool gain = extents.size() == 1U;
        const float scale = gain ? 0.25F : 1.0F / std::sqrt(static_cast<float>(extents.back()));
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto& x : v) x = gain ? 1.0F + dist(rng) : dist(rng);
        weights->tensors.emplace(name, std::move(v));
    }
    return weights;
}

struct Drafter::Impl {
    unsigned threads = 1;
    std::vector<Vec> keys, values;  // per layer [position][kv width]
};

Drafter::Drafter(Config config, std::unique_ptr<Weights> weights, Target& target, unsigned threads)
    : config_(std::move(config)), weights_(std::move(weights)), target_(target), impl_(std::make_unique<Impl>()) {
    if (!weights_) throw std::invalid_argument("DFlash: no weights");
    for (const auto& [name, extents] : inventory(config_))
        if (weights_->tensor(name).size() != count_of(extents)) throw std::invalid_argument("DFlash: tensor size " + name);
    if (config_.selector_rank && target_.vocab_out() > config_.vocab) throw std::invalid_argument("DFlash: head wider than the codebooks");
    impl_->threads = threads ? threads : std::max(1U, std::thread::hardware_concurrency());
    impl_->keys.resize(config_.layers);
    impl_->values.resize(config_.layers);
}

Drafter::~Drafter() = default;

BlockPolicy Drafter::policy(const Config& c) {
    BlockPolicy p;
    p.block = c.block;
    p.fill_token = c.mask_token;
    p.kv = KvSource::own_ring;
    p.ring = c.window;
    p.mask = BlockMask::bidirectional;
    p.draft_layers = std::min<std::uint32_t>(c.layers, static_cast<std::uint32_t>(max_draft_layers));
    p.target_layer_count = static_cast<std::uint32_t>(std::min(c.target_layers.size(), max_draft_layers));
    for (std::uint32_t i = 0; i < p.target_layer_count; ++i) p.target_layers[i] = static_cast<std::uint16_t>(c.target_layers[i]);
    return p;
}

void Drafter::append_context(std::span<const float> taps, std::uint32_t rows) {
    const auto& c = config_;
    const std::uint32_t tw = c.tap_width(), h = c.hidden, kvw = c.kv_heads * c.head_dim;
    if (taps.size() != std::size_t{rows} * tw) throw std::invalid_argument("DFlash: tap rows shape");
    constexpr std::uint32_t chunk = 32U;
    Vec fused(std::size_t{chunk} * h), normed(std::size_t{chunk} * h), k(std::size_t{chunk} * kvw), v(std::size_t{chunk} * kvw);
    const auto fc = weights_->tensor("fc.weight");
    const auto hidden_norm = weights_->tensor("hidden_norm.weight");
    for (std::uint32_t first = 0; first < rows; first += chunk) {
        const std::uint32_t n = std::min(chunk, rows - first);
        linear(fc, h, tw, taps.data() + std::size_t{first} * tw, n, fused.data(), impl_->threads);
        for (std::uint32_t r = 0; r < n; ++r)
            rms_norm(fused.data() + std::size_t{r} * h, hidden_norm, c.eps, normed.data() + std::size_t{r} * h);
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            linear(weights_->tensor(layer_name(l, "self_attn.k_proj.weight")), kvw, h, normed.data(), n, k.data(), impl_->threads);
            linear(weights_->tensor(layer_name(l, "self_attn.v_proj.weight")), kvw, h, normed.data(), n, v.data(), impl_->threads);
            const auto k_gain = weights_->tensor(layer_name(l, "self_attn.k_norm.weight"));
            for (std::uint32_t r = 0; r < n; ++r)
                for (std::uint32_t head = 0; head < c.kv_heads; ++head) {
                    float* row = k.data() + std::size_t{r} * kvw + std::size_t{head} * c.head_dim;
                    rms_norm(row, k_gain, c.eps, row);
                    rope_half(row, c.head_dim, positions_ + first + r, c.rope_theta);
                }
            impl_->keys[l].insert(impl_->keys[l].end(), k.begin(), k.begin() + static_cast<std::ptrdiff_t>(std::size_t{n} * kvw));
            impl_->values[l].insert(impl_->values[l].end(), v.begin(), v.begin() + static_cast<std::ptrdiff_t>(std::size_t{n} * kvw));
        }
    }
    positions_ += rows;
}

void Drafter::truncate(std::uint32_t positions) {
    if (positions > positions_) throw std::invalid_argument("DFlash: truncate beyond the context");
    const std::size_t kvw = std::size_t{config_.kv_heads} * config_.head_dim;
    for (std::uint32_t l = 0; l < config_.layers; ++l) {
        impl_->keys[l].resize(std::size_t{positions} * kvw);
        impl_->values[l].resize(std::size_t{positions} * kvw);
    }
    positions_ = positions;
}

namespace {

// Grouped convolution over the block rows (row r reads rows r - t, t < taps, of the same block).
void conv(const Config& c, const float* in, std::span<const float> base, const float* delta, std::uint32_t side,
          std::uint32_t rows, float* out) {
    const std::uint32_t h = c.hidden, groups = c.conv_groups(), taps = c.conv_taps;
    const std::size_t dw = std::size_t{2U} * taps * groups;
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t ch = 0; ch < h; ++ch) {
            const std::uint32_t g = ch / c.conv_group;
            float acc = 0.0F;
            for (std::uint32_t t = 0; t < taps && t <= r; ++t) {
                const float coef = base[(std::size_t{side} * taps + t) * h + ch] +
                                   delta[std::size_t{r} * dw + (std::size_t{side} * taps + t) * groups + g];
                acc += coef * in[std::size_t{r - t} * h + ch];
            }
            out[std::size_t{r} * h + ch] = acc;
        }
}

std::uint32_t argmax(const float* v, std::uint32_t n) {
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;  // ties keep the lower index
    return best;
}

void softmax(const float* s, std::uint32_t n, double temperature, std::vector<double>& p) {
    p.resize(n);
    const double t = temperature > 0.0 ? temperature : 1.0;
    double m = -std::numeric_limits<double>::infinity(), sum = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) m = std::max(m, static_cast<double>(s[i]) / t);
    for (std::uint32_t i = 0; i < n; ++i) sum += (p[i] = std::exp(static_cast<double>(s[i]) / t - m));
    for (auto& x : p) x /= sum;
}

}  // namespace

void Drafter::draft(std::uint32_t anchor, std::uint32_t k, Draft& out, Path path, double temperature, std::uint64_t seed) {
    const auto& c = config_;
    if (k == 0 || k >= c.block) throw std::invalid_argument("DFlash: proposals outside the block");
    if (positions_ == 0) throw std::invalid_argument("DFlash: no context before the anchor");
    const std::uint32_t R = c.block, h = c.hidden, qw = c.heads * c.head_dim, kvw = c.kv_heads * c.head_dim;
    const unsigned threads = impl_->threads;
    const std::uint32_t P = positions_;
    Vec x(std::size_t{R} * h), n(std::size_t{R} * h), t(std::size_t{R} * h), q(std::size_t{R} * qw), kb(std::size_t{R} * kvw),
        vb(std::size_t{R} * kvw), att(std::size_t{R} * qw), a(std::size_t{R} * h), g(std::size_t{R} * c.intermediate),
        u(std::size_t{R} * c.intermediate);
    const std::size_t dw = std::size_t{2U} * c.conv_taps * c.conv_groups();
    Vec delta(std::size_t{R} * std::max<std::size_t>(dw, 1U));
    for (std::uint32_t r = 0; r < R; ++r) {
        auto row = std::span(x).subspan(std::size_t{r} * h, h);
        target_.embed(r == 0 ? anchor : c.mask_token, row);
        for (auto& e : row) e *= c.embed_scale;
    }
    // conv_pre: filter `t` in place (side 0) after computing its coefficients; conv_post: filter `y` (side 1).
    auto conv_pre = [&](std::uint32_t l, const char* which, float* rows) {
        const std::string prefix = std::string(which);
        linear(weights_->tensor(layer_name(l, prefix + "kernel_projection.weight")), static_cast<std::uint32_t>(dw), h, rows, R,
               delta.data(), threads);
        Vec filtered(std::size_t{R} * h);
        conv(c, rows, weights_->tensor(layer_name(l, prefix + "base_kernel")), delta.data(), 0U, R, filtered.data());
        std::copy(filtered.begin(), filtered.end(), rows);
    };
    auto conv_post = [&](std::uint32_t l, const char* which, float* rows) {
        Vec filtered(std::size_t{R} * h);
        conv(c, rows, weights_->tensor(layer_name(l, std::string(which) + "base_kernel")), delta.data(), 1U, R, filtered.data());
        std::copy(filtered.begin(), filtered.end(), rows);
    };
    const float scale = 1.0F / std::sqrt(static_cast<float>(c.head_dim));
    const std::uint32_t group = c.heads / c.kv_heads;
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const auto in_gain = weights_->tensor(layer_name(l, "input_layernorm.weight"));
        for (std::uint32_t r = 0; r < R; ++r) rms_norm(&x[std::size_t{r} * h], in_gain, c.eps, &n[std::size_t{r} * h]);
        if (c.conv_taps) conv_pre(l, "attention_conv.", n.data());
        linear(weights_->tensor(layer_name(l, "self_attn.q_proj.weight")), qw, h, n.data(), R, q.data(), threads);
        linear(weights_->tensor(layer_name(l, "self_attn.k_proj.weight")), kvw, h, n.data(), R, kb.data(), threads);
        linear(weights_->tensor(layer_name(l, "self_attn.v_proj.weight")), kvw, h, n.data(), R, vb.data(), threads);
        const auto q_gain = weights_->tensor(layer_name(l, "self_attn.q_norm.weight"));
        const auto k_gain = weights_->tensor(layer_name(l, "self_attn.k_norm.weight"));
        for (std::uint32_t r = 0; r < R; ++r) {
            for (std::uint32_t hd = 0; hd < c.heads; ++hd) {
                float* row = &q[std::size_t{r} * qw + std::size_t{hd} * c.head_dim];
                rms_norm(row, q_gain, c.eps, row);
                rope_half(row, c.head_dim, P + r, c.rope_theta);
            }
            for (std::uint32_t hd = 0; hd < c.kv_heads; ++hd) {
                float* row = &kb[std::size_t{r} * kvw + std::size_t{hd} * c.head_dim];
                rms_norm(row, k_gain, c.eps, row);
                rope_half(row, c.head_dim, P + r, c.rope_theta);
            }
        }
        const auto& ck = impl_->keys[l];
        const auto& cv = impl_->values[l];
        parallel_rows(R * c.heads, threads, [&](std::uint32_t begin, std::uint32_t end) {
            std::vector<double> w;
            for (std::uint32_t item = begin; item < end; ++item) {
                const std::uint32_t r = item / c.heads, hd = item % c.heads, kh = hd / group;
                const float* qr = &q[std::size_t{r} * qw + std::size_t{hd} * c.head_dim];
                const std::uint32_t pos = P + r;
                const std::uint32_t first = c.window && pos >= c.window ? pos - c.window + 1U : 0U;
                const std::uint32_t ctx = P > first ? P - first : 0U;
                w.resize(std::size_t{ctx} + R);
                double m = -std::numeric_limits<double>::infinity();
                for (std::uint32_t j = 0; j < ctx; ++j)
                    m = std::max(m, w[j] = scale * dot(qr, &ck[std::size_t{first + j} * kvw + std::size_t{kh} * c.head_dim], c.head_dim));
                for (std::uint32_t j = 0; j < R; ++j)
                    m = std::max(m, w[ctx + j] = scale * dot(qr, &kb[std::size_t{j} * kvw + std::size_t{kh} * c.head_dim], c.head_dim));
                double sum = 0.0;
                for (auto& e : w) sum += (e = std::exp(e - m));
                float* o = &att[std::size_t{r} * qw + std::size_t{hd} * c.head_dim];
                std::vector<double> acc(c.head_dim, 0.0);
                for (std::uint32_t j = 0; j < ctx + R; ++j) {
                    const float* vr = j < ctx ? &cv[std::size_t{first + j} * kvw + std::size_t{kh} * c.head_dim]
                                              : &vb[std::size_t{j - ctx} * kvw + std::size_t{kh} * c.head_dim];
                    for (std::uint32_t d = 0; d < c.head_dim; ++d) acc[d] += w[j] * static_cast<double>(vr[d]);
                }
                for (std::uint32_t d = 0; d < c.head_dim; ++d) o[d] = static_cast<float>(acc[d] / sum);
            }
        }, 4U);
        linear(weights_->tensor(layer_name(l, "self_attn.o_proj.weight")), h, qw, att.data(), R, a.data(), threads);
        if (c.conv_taps) conv_post(l, "attention_conv.", a.data());
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += a[i];
        const auto post_gain = weights_->tensor(layer_name(l, "post_attention_layernorm.weight"));
        for (std::uint32_t r = 0; r < R; ++r) rms_norm(&x[std::size_t{r} * h], post_gain, c.eps, &n[std::size_t{r} * h]);
        if (c.conv_taps) conv_pre(l, "mlp_conv.", n.data());
        linear(weights_->tensor(layer_name(l, "mlp.gate_proj.weight")), c.intermediate, h, n.data(), R, g.data(), threads);
        linear(weights_->tensor(layer_name(l, "mlp.up_proj.weight")), c.intermediate, h, n.data(), R, u.data(), threads);
        for (std::size_t i = 0; i < g.size(); ++i) g[i] = silu(g[i]) * u[i];
        linear(weights_->tensor(layer_name(l, "mlp.down_proj.weight")), h, c.intermediate, g.data(), R, a.data(), threads);
        if (c.conv_taps) conv_post(l, "mlp_conv.", a.data());
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += a[i];
    }
    const auto final_gain = weights_->tensor("norm.weight");
    for (std::uint32_t r = 0; r < R; ++r) rms_norm(&x[std::size_t{r} * h], final_gain, c.eps, &t[std::size_t{r} * h]);

    // Head on rows 1 .. k (proposal j reads row j + 1).
    const std::uint32_t vocab = target_.vocab_out();
    Vec logits(std::size_t{k} * vocab);
    target_.head(std::span<const float>(t).subspan(h, std::size_t{k} * h), k, logits);
    out.tokens.clear();
    out.q.clear();
    out.candidates.clear();
    out.scores.clear();
    std::vector<double> p;
    std::mt19937_64 rng(seed);
    if (!c.selector_rank) {
        for (std::uint32_t j = 0; j < k; ++j) {
            const float* row = &logits[std::size_t{j} * vocab];
            std::uint32_t token = argmax(row, vocab);
            softmax(row, vocab, temperature, p);
            if (temperature > 0.0) token = static_cast<std::uint32_t>(std::discrete_distribution<std::uint32_t>(p.begin(), p.end())(rng));
            out.tokens.push_back(token);
            out.q.push_back(static_cast<float>(p[token]));
        }
        return;
    }
    const std::uint32_t K = c.selector_top_k, rank = c.selector_rank;
    std::vector<std::uint32_t> cand(std::size_t{k} * K);
    Vec unary(std::size_t{k} * K), hsel(std::size_t{k} * rank);
    for (std::uint32_t j = 0; j < k; ++j) {
        const float* row = &logits[std::size_t{j} * vocab];
        std::vector<std::uint32_t> ids(vocab);
        for (std::uint32_t i = 0; i < vocab; ++i) ids[i] = i;
        std::partial_sort(ids.begin(), ids.begin() + K, ids.end(), [row](std::uint32_t lhs, std::uint32_t rhs) {
            return row[lhs] > row[rhs] || (row[lhs] == row[rhs] && lhs < rhs);
        });
        for (std::uint32_t i = 0; i < K; ++i) {
            cand[std::size_t{j} * K + i] = ids[i];
            float s = row[ids[i]];
            if (c.softcap > 0.0F && !c.softcap_after_correction) s = c.softcap * std::tanh(s / c.softcap);
            unary[std::size_t{j} * K + i] = s;
        }
    }
    linear(weights_->tensor("candidate_selector.hidden_projection.weight"), rank, h, &t[h], k, hsel.data(), threads);
    const auto pred = weights_->tensor("candidate_selector.predecessor_codebook");
    const auto succ = weights_->tensor("candidate_selector.successor_codebook");
    // s[j][p][c]: step j, predecessor index p (step 0: the anchor, one row), candidate c.
    Vec s(std::size_t{k} * K * K);
    Vec weighted(rank);
    for (std::uint32_t j = 0; j < k; ++j)
        for (std::uint32_t pi = 0; pi < (j == 0 ? 1U : K); ++pi) {
            const std::uint32_t prev = j == 0 ? std::min(anchor, c.vocab - 1U) : cand[std::size_t{j - 1U} * K + pi];
            for (std::uint32_t qi = 0; qi < rank; ++qi) weighted[qi] = pred[std::size_t{prev} * rank + qi] * hsel[std::size_t{j} * rank + qi];
            for (std::uint32_t ci = 0; ci < K; ++ci) {
                float e = unary[std::size_t{j} * K + ci] + dot(weighted.data(), &succ[std::size_t{cand[std::size_t{j} * K + ci]} * rank], rank);
                if (c.softcap > 0.0F && c.softcap_after_correction) e = c.softcap * std::tanh(e / c.softcap);
                s[(std::size_t{j} * K + pi) * K + ci] = e;
            }
        }
    std::vector<std::uint32_t> pick(k);
    if (path == Path::viterbi && temperature <= 0.0) {
        // MAP path of the locally normalised chain (log-softmax per predecessor row).
        std::vector<double> score(K), next(K);
        std::vector<std::uint32_t> back(std::size_t{k} * K);
        softmax(&s[0], K, 1.0, p);
        for (std::uint32_t ci = 0; ci < K; ++ci) score[ci] = std::log(p[ci]);
        for (std::uint32_t j = 1; j < k; ++j) {
            std::fill(next.begin(), next.end(), -std::numeric_limits<double>::infinity());
            for (std::uint32_t pi = 0; pi < K; ++pi) {
                softmax(&s[(std::size_t{j} * K + pi) * K], K, 1.0, p);
                for (std::uint32_t ci = 0; ci < K; ++ci) {
                    const double v = score[pi] + std::log(p[ci]);
                    if (v > next[ci]) {
                        next[ci] = v;
                        back[std::size_t{j} * K + ci] = pi;
                    }
                }
            }
            score = next;
        }
        pick[k - 1U] = static_cast<std::uint32_t>(std::max_element(score.begin(), score.end()) - score.begin());
        for (std::uint32_t j = k - 1U; j > 0; --j) pick[j - 1U] = back[std::size_t{j} * K + pick[j]];
    } else {
        std::uint32_t prev = 0;
        for (std::uint32_t j = 0; j < k; ++j) {
            const float* row = &s[(std::size_t{j} * K + (j == 0 ? 0U : prev)) * K];
            softmax(row, K, temperature, p);
            prev = temperature > 0.0 ? static_cast<std::uint32_t>(std::discrete_distribution<std::uint32_t>(p.begin(), p.end())(rng))
                                     : argmax(row, K);
            pick[j] = prev;
        }
    }
    for (std::uint32_t j = 0; j < k; ++j) {
        const float* row = &s[(std::size_t{j} * K + (j == 0 ? 0U : pick[j - 1U])) * K];
        softmax(row, K, temperature, p);
        out.tokens.push_back(cand[std::size_t{j} * K + pick[j]]);
        out.q.push_back(static_cast<float>(p[pick[j]]));
        out.candidates.insert(out.candidates.end(), cand.begin() + static_cast<std::ptrdiff_t>(std::size_t{j} * K),
                              cand.begin() + static_cast<std::ptrdiff_t>(std::size_t{j + 1U} * K));
        out.scores.insert(out.scores.end(), row, row + K);
    }
}

Proposal proposal(const Draft& draft, std::uint32_t step, std::uint32_t top_k, double temperature) {
    Proposal out;
    if (top_k == 0U || draft.candidates.empty()) return out;
    if (step >= draft.tokens.size() || draft.candidates.size() < std::size_t{step + 1U} * top_k ||
        draft.scores.size() < std::size_t{step + 1U} * top_k)
        throw std::invalid_argument("DFlash: proposal step outside the draft");
    std::vector<double> p;
    softmax(&draft.scores[std::size_t{step} * top_k], top_k, temperature, p);
    out.tokens.assign(draft.candidates.begin() + static_cast<std::ptrdiff_t>(std::size_t{step} * top_k),
                      draft.candidates.begin() + static_cast<std::ptrdiff_t>(std::size_t{step + 1U} * top_k));
    out.probabilities = std::move(p);
    return out;
}

void Drafter::forward(const BlockRequest& req, std::vector<std::uint32_t>& proposals, std::vector<float>& confidence) {
    if (req.ids.empty() || req.anchor_pos != positions_ || req.prefix != 0)
        throw std::invalid_argument("DFlash: invalid proposal frontier (known prefixes are not part of this drafter)");
    Draft d;
    draft(req.ids.front(), req.k, d);
    proposals = d.tokens;
    confidence = d.q;
}

}  // namespace strix::spec::dflash
