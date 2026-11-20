#include "strix/hy4/presets.hpp"
#include "hy4/ref/hy4_host_ref.hpp"

#include "strix/ds41_layer.hpp"
#include "strix/ds41_scalar_codecs.hpp"
#include "strix/attn/group_indexer.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hc/hc_ref.hpp"
#include "strix/moe/route_policy.hpp"
#include "strix/weight_archive.hpp"
#include "ops/glu/glu.hpp"
#include "ops/rope/rope.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace strix::hy4 {
namespace {

using Vec = std::vector<float>;

unsigned g_threads = 1;  // set_ref_threads; every row is one independent dot, so the split never changes a bit
std::size_t g_min_work = std::size_t{1} << 20;

// fn(r) for r in [0, n): contiguous row ranges on up to g_threads threads when the work is large enough.
template <class Fn>
void for_rows(std::size_t n, std::size_t work_per_row, Fn&& fn) {
    const std::size_t threads = std::min<std::size_t>(g_threads, n);
    if (threads <= 1U || n * work_per_row < g_min_work) {
        for (std::size_t r = 0; r < n; ++r) fn(r);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t)
        pool.emplace_back([&fn, n, t, threads] {
            for (std::size_t r = n * t / threads; r < n * (t + 1U) / threads; ++r) fn(r);
        });
    for (auto& th : pool) th.join();
}

float bf(float x) { return ds41::round_to_bf16(x); }
// The shared declared dot (lane-strided chunks, ascending): the order every reduction of this oracle uses.
float dot(const float* a, const float* b, std::size_t n) {
    return ds41::chunked_dot_of([a](std::size_t i) { return a[i]; }, [b](std::size_t i) { return b[i]; }, n);
}
float sigmoid(float x) { return ds41::declared_sigmoid(x); }

std::string blk(std::uint32_t layer, const char* name) { return "blk." + std::to_string(layer) + "." + name; }

// y = W x, W [rows][cols]; BF16 output (a Linear of the reference) or FP32.
Vec linear(const Vec& w, const Vec& x, std::size_t rows, bool round = true) {
    const auto cols = x.size();
    if (w.size() != rows * cols) throw std::invalid_argument("hy4 ref: weight shape mismatch");
    Vec y(rows);
    for_rows(rows, cols, [&](std::size_t r) {
        const float v = dot(w.data() + r * cols, x.data(), cols);
        y[r] = round ? bf(v) : v;
    });
    return y;
}
// Rows [first_row, first_row + rows) read one by one through Weights::row (decoded, not cached): the routed experts,
// whose FP32 copy of the whole tensor would not fit in host memory on the real pack. Same values, same dot order.
Vec linear_rows(Weights& w, const std::string& name, std::size_t first_row, const Vec& x, std::size_t rows) {
    Vec y(rows);
    for_rows(rows, x.size(), [&](std::size_t r) {
        const auto wr = w.row(name, first_row + r);
        if (wr.size() != x.size()) throw std::invalid_argument("hy4 ref: weight row width mismatch");
        y[r] = bf(dot(wr.data(), x.data(), x.size()));
    });
    return y;
}
// The model's RMSNorm: y = bf16(w * bf16(x * r)), r = 1 / sqrt(sum(x^2) / n + eps); w == nullptr: y = bf16(x * r).
Vec rms_norm(const float* x, std::size_t n, const float* w, float eps) {
    const float r = 1.0F / ds41::declared_sqrt(dot(x, x, n) / static_cast<float>(n) + eps);
    Vec y(n);
    for (std::size_t i = 0; i < n; ++i) y[i] = w ? bf(w[i] * bf(x[i] * r)) : bf(x[i] * r);
    return y;
}
Vec rms_norm(const Vec& x, const Vec* w, float eps) { return rms_norm(x.data(), x.size(), w ? w->data() : nullptr, eps); }

// Interleaved RoPE over x[0, dims): pair i = (2i, 2i+1), angle pos * theta^(-2i/dims) in FP32, cos/sin rounded to BF16
// (the shared native table and BF16 host rotation, src/ops/rope).
void rope(float* x, std::size_t dims, std::uint32_t pos, float theta) {
    const auto pairs = static_cast<std::uint32_t>(dims / 2U);
    Vec cos_t(pairs), sin_t(pairs);
    ops::rope::build_native_table(theta, pairs, pos, 1U, ops::rope::Round::bf16, cos_t.data(), sin_t.data());
    const std::uint32_t row = 0;  // the table holds the single position
    ops::rope::View v;
    v.x = x;
    v.rows = 1;
    v.pairs = pairs;
    v.cos = cos_t.data();
    v.sin = sin_t.data();
    v.positions = &row;
    ops::rope::reference_apply(v, ops::rope::Round::bf16);
}

// Hyper-connection of one site over the 4 copy-major streams: the shared hc contract (src/hc) with the Hy4 spec.
using HcMix = hc::Mix;
hc::Spec hc_spec(std::size_t hidden, bool sublayer, float rms_eps, float hc_eps) {
    auto spec = strix::hy4::preset::hc_spec(sublayer ? hc::Site::sublayer : hc::Site::head);
    spec.hidden = static_cast<std::uint32_t>(hidden);
    spec.pre_eps = spec.post_eps = hc_eps;
    spec.norm_eps = rms_eps;
    return spec;
}
HcMix hc_mix(const Vec& streams, const Vec& fn, const Vec& base, const Vec& scale, std::uint32_t mixes, float rms_eps,
             float hc_eps) {
    return hc::mixes(hc_spec(streams.size() / 4U, mixes == 8U, rms_eps, hc_eps), streams, {{}, fn, base, scale});
}
Vec hc_collapse(const Vec& streams, const HcMix& mix, std::size_t hidden) {
    Vec u(hidden);
    hc::collapse(hc_spec(hidden, true, 0.0F, 0.0F), streams, mix.pre, u);
    return u;
}
void hc_update(Vec& streams, const HcMix& mix, const Vec& f) {
    Vec out(streams.size());
    hc::expand(hc_spec(f.size(), true, 0.0F, 0.0F), f, streams, mix, out);
    streams = std::move(out);
}

std::uint64_t positive_mod(std::int64_t value, std::uint64_t modulus) {
    const auto m = static_cast<std::int64_t>(modulus);
    const auto r = value % m;
    return static_cast<std::uint64_t>(r < 0 ? r + m : r);
}

}  // namespace

void Config::validate() const {
    const auto bad = [](const char* what) { throw std::invalid_argument(std::string("hy4 config: ") + what); };
    if (indexed.size() != layers || heads.size() != layers || dense.size() != layers) bad("per-layer arrays");
    if (qk_rope % 2U || index_dim % 4U || hidden == 0U || top_k > experts) bad("dimensions");
    if (nextn > 1U || (nextn == 1U && nextn_heads == 0U)) bad("next-token block (one block with heads > 0)");
    const auto eh = engram_heads();
    if (engram_mult.size() != engram_layers.size() * ngram || head_offsets.size() != engram_layers.size() * eh ||
        head_sizes.size() != engram_layers.size() * eh || token_map.empty())
        bad("engram arrays");
    for (const auto l : engram_layers)
        if (l >= layers) bad("engram layer id");
}

Config Config::tiny() {
    Config c;
    c.hidden = 64; c.vocab = 300; c.layers = 3;
    c.q_lora = 32; c.kv_lora = 16; c.qk_nope = 8; c.qk_rope = 8; c.v_head = 12;
    c.window = 4;
    c.index_heads = 2; c.index_dim = 8; c.index_ratio = 4; c.index_groups = 2;  // dense while kv <= 8
    c.experts = 8; c.top_k = 2; c.latent = 16; c.expert_ff = 12; c.shared_ff = 12; c.dense_ff = 40;
    c.indexed = {1, 0, 1}; c.heads = {2, 4, 2}; c.dense = {1, 0, 0};
    c.engram_layers = {1, 2}; c.engram_dim = 8; c.ngram = 3; c.heads_per_ngram = 2;
    c.engram_mult = {1000003, 2000029, 3000017, 4000037, 5000011, 6000007};
    const std::vector<std::uint64_t> primes = {13, 17, 19, 23, 29, 31, 37, 41};
    std::uint64_t offset = 0;
    for (const auto p : primes) { c.head_offsets.push_back(offset); c.head_sizes.push_back(p); offset += p; }
    c.pad_token = 2;
    c.token_map.resize(c.vocab);
    for (std::uint32_t t = 0; t < c.vocab; ++t) c.token_map[t] = static_cast<std::int32_t>(t % 97U);
    return c;
}

Config Config::truncated(std::uint32_t n) const {
    if (n == 0U || n > layers) throw std::invalid_argument("hy4 config: truncated depth outside 1..layers");
    Config c = *this;
    c.layers = n;
    c.nextn = 0;  // the next-token block follows the full trunk only
    c.indexed.resize(n);
    c.heads.resize(n);
    c.dense.resize(n);
    const auto eh = engram_heads();
    std::size_t keep = 0;
    while (keep < engram_layers.size() && engram_layers[keep] < n) ++keep;  // engram_layers ascending (pack order)
    c.engram_layers.resize(keep);
    c.engram_mult.resize(keep * ngram);
    c.head_offsets.resize(keep * eh);
    c.head_sizes.resize(keep * eh);
    c.validate();
    return c;
}

Config Config::from_gguf(const std::filesystem::path& first_part) {
    gguf::Limits limits;
    limits.max_retained_array_elements = 200000;
    // Metadata only: tensor data ranges are not checked here (a partial download still has its whole header);
    // pack_weights and the pack check read the data against the real file sizes.
    std::ifstream input(first_part, std::ios::binary);
    if (!input) throw std::runtime_error("hy4 pack: cannot open " + first_part.string());
    const auto info = gguf::read(input, std::numeric_limits<std::uint64_t>::max(), limits);
    const auto entry = [&](const std::string& key) -> const gguf::MetadataEntry& {
        const auto* e = info.find_metadata("hy_v4_flash." + key);
        if (!e) throw std::runtime_error("hy4 pack: missing metadata hy_v4_flash." + key);
        return *e;
    };
    const auto u32 = [&](const std::string& key) {
        const auto& e = entry(key);
        if (e.unsigned_value) return static_cast<std::uint32_t>(*e.unsigned_value);
        if (e.signed_value) return static_cast<std::uint32_t>(*e.signed_value);
        throw std::runtime_error("hy4 pack: " + key + " is not an integer");
    };
    const auto f32 = [&](const std::string& key) {
        const auto& e = entry(key);
        if (!e.floating_value) throw std::runtime_error("hy4 pack: " + key + " is not a float");
        return static_cast<float>(*e.floating_value);
    };
    const auto uarr = [&](const std::string& key) {
        const auto& e = entry(key);
        std::vector<std::uint64_t> out(e.unsigned_array.begin(), e.unsigned_array.end());
        for (const auto v : e.signed_array) out.push_back(static_cast<std::uint64_t>(v));
        return out;
    };
    Config c;
    const auto blocks = u32("block_count");
    const auto nextn = u32("nextn_predict_layers");
    c.layers = blocks - nextn;  // the trailing next-token block is not part of the text forward pass
    c.hidden = u32("embedding_length");
    c.vocab = u32("vocab_size");
    c.rms_eps = f32("attention.layer_norm_rms_epsilon");
    c.hc_eps = f32("hyper_connection.epsilon");
    c.q_lora = u32("attention.q_lora_rank");
    c.kv_lora = u32("attention.kv_lora_rank");
    c.qk_rope = u32("rope.dimension_count");
    c.qk_nope = u32("attention.key_length_mla") - c.qk_rope;
    c.v_head = u32("attention.value_length_mla");
    c.rope_theta_indexed = f32("rope.freq_base");
    c.rope_theta_sliding = f32("rope.freq_base_swa");
    c.window = u32("attention.sliding_window");
    c.index_heads = u32("attention.indexer.head_count");
    c.index_dim = u32("attention.indexer.key_length");
    c.index_ratio = u32("attention.indexer.kpool");
    c.index_groups = u32("attention.indexer.top_k") / c.index_ratio;
    c.experts = u32("expert_count");
    c.top_k = u32("expert_used_count");
    c.latent = u32("expert_latent_length");
    c.expert_ff = u32("expert_feed_forward_length");
    c.shared_ff = c.expert_ff * u32("expert_shared_count");
    c.dense_ff = u32("feed_forward_length");
    c.routed_scale = f32("expert_weights_scale");
    const auto& clamp = entry("swiglu_clamp_exp");
    if (!clamp.floating_array.empty()) c.swiglu_limit = static_cast<float>(clamp.floating_array.back());
    const auto types = uarr("attention.indexer.types");
    const auto heads = uarr("attention.head_count");
    const auto leading_dense = u32("leading_dense_block_count");
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        c.indexed.push_back(l < types.size() && types[l] ? 1U : 0U);
        c.heads.push_back(static_cast<std::uint32_t>(heads.at(l)));
        c.dense.push_back(l < leading_dense ? 1U : 0U);
    }
    c.nextn = nextn;
    if (nextn == 1U) {
        c.nextn_heads = static_cast<std::uint32_t>(heads.at(c.layers));
        c.nextn_indexed = c.layers < types.size() ? (types[c.layers] ? 1U : 0U) : 1U;  // the block carries the indexer
    }
    for (const auto l : uarr("ple.layers")) c.engram_layers.push_back(static_cast<std::uint32_t>(l));
    c.ngram = u32("ple.ngram_size");
    c.heads_per_ngram = u32("ple.heads_per_ngram");
    c.engram_dim = u32("embedding_length_per_layer_input");
    for (const auto m : uarr("ple.layer_multipliers")) c.engram_mult.push_back(static_cast<std::int64_t>(m));
    c.head_offsets = uarr("ple.head_offsets");
    c.head_sizes = uarr("ple.head_vocab_sizes");
    c.pad_token = u32("engram.pad_token_id");
    for (const auto t : uarr("engram.token_map")) c.token_map.push_back(static_cast<std::int32_t>(t));
    c.validate();
    return c;
}

// ---- weights --------------------------------------------------------------------------------------------------
namespace {

class SyntheticWeights final : public Weights {
public:
    SyntheticWeights(const Config& c, std::uint64_t seed, std::string filter) : seed_(seed), filter_(std::move(filter)) {
        const std::uint64_t h = c.hidden, ql = c.q_lora, kl = c.kv_lora;
        add("token_embd.weight", {h, c.vocab}, 1.0F);
        for (std::uint32_t l = 0; l < c.layers + c.nextn; ++l) {
            const bool text = l < c.layers;  // else the next-token block
            const std::uint64_t H = text ? c.heads[l] : c.nextn_heads;
            for (const char* norm : {"attn_norm.weight", "attn_q_a_norm.weight", "post_attention_norm.weight", "ffn_norm.weight",
                                     "post_ffw_norm.weight"})
                add(blk(l, norm), {std::string(norm) == "attn_q_a_norm.weight" ? ql : h}, -1.0F);
            add(blk(l, "attn_kv_a_norm.weight"), {kl}, -1.0F);
            add(blk(l, "attn_q_a.weight"), {h, ql}, 0.15F);
            add(blk(l, "attn_q_b.weight"), {ql, H * c.qk_head()}, 0.2F);
            add(blk(l, "attn_kv_a_mqa.weight"), {h, kl + c.qk_rope}, 0.15F);
            add(blk(l, "attn_k_b.weight"), {c.qk_nope, kl, H}, 0.25F);
            add(blk(l, "attn_v_b.weight"), {kl, c.v_head, H}, 0.25F);
            add(blk(l, "attn_sinks.weight"), {H}, 1.0F);
            add(blk(l, "attn_gate.weight"), {h, H * c.v_head}, 0.15F);
            add(blk(l, "attn_output.weight"), {H * c.v_head, h}, 0.12F);
            for (const char* site : {"hc_attn", "hc_ffn"}) {
                if (!text) break;
                add(blk(l, (std::string(site) + "_fn.weight").c_str()), {4U * h, 8}, 0.05F);
                add(blk(l, (std::string(site) + "_base.weight").c_str()), {8}, 0.5F);
                add(blk(l, (std::string(site) + "_scale.weight").c_str()), {2}, 0.5F);
            }
            if (!text) {
                add(blk(l, "nextn.eh_proj.weight"), {2U * h, h}, 0.1F);
                for (const char* norm : {"nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"}) add(blk(l, norm), {h}, -1.0F);
            }
            if (text ? c.indexed[l] != 0U : c.nextn_indexed != 0U) {
                add(blk(l, "indexer.attn_k.weight"), {h, c.index_dim}, 0.15F);
                add(blk(l, "indexer.k_norm.weight"), {c.index_dim}, -1.0F);
                add(blk(l, "indexer.attn_q_b.weight"), {ql, c.index_heads * c.index_dim}, 0.2F);
                add(blk(l, "indexer.proj.weight"), {h, c.index_heads}, 0.2F);
                add(blk(l, "indexer_compressor_value.weight"), {c.index_dim, c.index_dim}, 0.3F);
                add(blk(l, "indexer_compressor_gate.weight"), {c.index_dim, c.index_dim}, 0.3F);
                add(blk(l, "indexer_compressor_ape.weight"), {c.index_dim, c.index_ratio}, 0.3F);
                add(blk(l, "indexer_compressor_norm.weight"), {c.index_dim}, -1.0F);
            }
            if (text && c.dense[l]) {
                add(blk(l, "ffn_gate.weight"), {h, c.dense_ff}, 0.15F);
                add(blk(l, "ffn_up.weight"), {h, c.dense_ff}, 0.15F);
                add(blk(l, "ffn_down.weight"), {c.dense_ff, h}, 0.1F);
            } else {
                add(blk(l, "ffn_gate_inp.weight"), {h, c.experts}, 0.2F);
                add(blk(l, "exp_probs_b.bias"), {c.experts}, 0.1F);
                add(blk(l, "ffn_routed_down.weight"), {h, c.latent}, 0.15F);
                add(blk(l, "ffn_gate_exps.weight"), {c.latent, c.expert_ff, c.experts}, 0.3F);
                add(blk(l, "ffn_up_exps.weight"), {c.latent, c.expert_ff, c.experts}, 0.3F);
                add(blk(l, "ffn_down_exps.weight"), {c.expert_ff, c.latent, c.experts}, 0.3F);
                add(blk(l, "ffn_routed_up.weight"), {c.latent, h}, 0.2F);
                add(blk(l, "ffn_gate_shexp.weight"), {h, c.shared_ff}, 0.15F);
                add(blk(l, "ffn_up_shexp.weight"), {h, c.shared_ff}, 0.15F);
                add(blk(l, "ffn_down_shexp.weight"), {c.shared_ff, h}, 0.2F);
            }
        }
        const std::uint64_t ec = std::uint64_t{c.engram_heads()} * c.engram_dim;
        for (const auto l : c.engram_layers) {
            add(blk(l, "engram_kv.weight"), {ec, 5U * h}, 0.2F);
            add(blk(l, "engram_kv_b.weight"), {h, 5}, 0.1F);
            add(blk(l, "engram_lookup_norm.weight"), {c.engram_dim, c.engram_heads()}, -1.0F);
            add(blk(l, "engram_gate.weight"), {h, 4}, -1.0F);
            add(blk(l, "engram_hc_scale.weight"), {h, 4}, 0.2F);
            add(blk(l, "engram_pre_gamma.weight"), {h}, -1.0F);
        }
        std::uint64_t rows = 0;
        for (const auto s : c.head_sizes) rows += s;
        add("per_layer_token_embd.weight", {c.engram_dim, rows}, 1.0F);
        add("output_hc_fn.weight", {4U * h, 4}, 0.05F);
        add("output_hc_base.weight", {4}, 0.5F);
        add("output_hc_scale.weight", {1}, 0.5F);
        add("output_norm.weight", {h}, -1.0F);
        add("output.weight", {h, c.vocab}, 0.15F);
    }
    const Vec& tensor(const std::string& name) override {
        const auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::out_of_range("hy4 synthetic weights: no tensor " + name);
        return it->second;
    }
    Vec row(const std::string& name, std::uint64_t r) override {
        const auto& t = tensor(name);
        const auto width = widths_.at(name);
        return {t.begin() + static_cast<std::ptrdiff_t>(r * width), t.begin() + static_cast<std::ptrdiff_t>((r + 1U) * width)};
    }

private:
    // amplitude < 0: a norm weight 1 + U(-0.1, 0.1); else U(-a, a) rounded to BF16 (values a BF16 / quantised pack holds).
    void add(const std::string& name, std::vector<std::uint64_t> extents, float amplitude) {
        std::uint64_t n = 1;
        for (const auto e : extents) n *= e;
        const bool reseed = !filter_.empty() && name.find(filter_) != std::string::npos;
        std::mt19937_64 rng(seed_ + (reseed ? 1U : 0U) + std::hash<std::string>{}(name));
        std::uniform_real_distribution<float> u(-1.0F, 1.0F);
        Vec v(n);
        for (auto& x : v) x = bf(amplitude < 0.0F ? 1.0F + 0.1F * u(rng) : amplitude * u(rng));
        widths_[name] = extents.front();
        tensors_[name] = std::move(v);
    }
    std::uint64_t seed_;
    std::string filter_;
    std::map<std::string, Vec> tensors_;
    std::map<std::string, std::uint64_t> widths_;
};

class PackWeights final : public Weights {
public:
    explicit PackWeights(const std::vector<std::filesystem::path>& parts) {
        for (const auto& part : parts) {
            const auto info = gguf::read_file(part);
            for (auto& [name, t] : weights::load_gguf(part, info)) {
                const auto found = std::find_if(info.tensors.begin(), info.tensors.end(),
                                                [&n = name](const gguf::TensorInfo& ti) { return ti.name == n; });
                if (found != info.tensors.end() && info.is_f8_e4m3(*found)) e4m3_.insert(name);
                archive_.emplace(name, std::move(t));
            }
        }
    }
    const Vec& tensor(const std::string& name) override {
        if (const auto it = cache_.find(name); it != cache_.end()) return it->second;
        const auto& t = archive_.at(name);
        const auto width = t.shape.back();
        std::uint64_t rows = 1;
        for (std::size_t i = 0; i + 1U < t.shape.size(); ++i) rows *= t.shape[i];
        Vec all;
        all.reserve(rows * width);
        for (std::uint64_t r = 0; r < rows; ++r) {
            const auto v = row(name, r);
            all.insert(all.end(), v.begin(), v.end());
        }
        return cache_.emplace(name, std::move(all)).first->second;
    }
    Vec row(const std::string& name, std::uint64_t r) override {
        const auto& t = archive_.at(name);
        if (!e4m3_.contains(name)) return weights::decode_row(t, r);
        const auto width = t.shape.back();
        const auto bytes = t.bytes().subspan(r * width, width);
        Vec out(width);
        for (std::size_t i = 0; i < width; ++i) out[i] = ds41::codec::e4m3(static_cast<std::uint8_t>(bytes[i]));
        return out;
    }

private:
    weights::Archive archive_;
    std::set<std::string> e4m3_;
    std::map<std::string, Vec> cache_;
};

}  // namespace

std::unique_ptr<Weights> synthetic_weights(const Config& config, std::uint64_t seed, const std::string& reseed_filter) {
    config.validate();
    return std::make_unique<SyntheticWeights>(config, seed, reseed_filter);
}
std::unique_ptr<Weights> pack_weights(const std::vector<std::filesystem::path>& parts) {
    return std::make_unique<PackWeights>(parts);
}

// ---- shared helpers -------------------------------------------------------------------------------------------
Route route(const std::vector<float>& logits, const std::vector<float>& bias, std::uint32_t top_k, float scale) {
    const auto n = logits.size();
    Vec score(n);
    for (std::size_t e = 0; e < n; ++e) score[e] = sigmoid(logits[e]);
    auto policy = strix::hy4::preset::route;  // sigmoid score + bias selection, ordered keys, ids ascending, renormalised
    policy.experts = static_cast<std::uint32_t>(n);
    policy.top_k = top_k;
    policy.scale = scale;
    Route out;
    out.ids.resize(top_k);
    out.weights.resize(top_k);
    moe::route_select_ref(policy, score, bias, out.ids, out.weights);
    return out;
}

void set_ref_threads(unsigned threads, std::size_t min_work) noexcept {
    g_threads = threads == 0U ? 1U : threads;
    g_min_work = min_work;
}

std::vector<std::uint64_t> engram_rows(const Config& c, std::uint32_t ordinal, const std::vector<std::uint32_t>& window) {
    if (window.size() != c.ngram) throw std::invalid_argument("hy4 engram: window size");
    std::vector<std::int64_t> p(c.ngram);
    for (std::size_t s = 0; s < c.ngram; ++s) {
        const auto id = window[s];
        const auto mapped = c.token_map.at(id < c.token_map.size() ? id : c.pad_token);
        // int64 product with wrap-around (two's complement), as the reference's int64 tensor arithmetic
        p[s] = static_cast<std::int64_t>(static_cast<std::uint64_t>(static_cast<std::int64_t>(mapped)) *
                                         static_cast<std::uint64_t>(c.engram_mult[ordinal * c.ngram + s]));
    }
    std::vector<std::uint64_t> rows;
    std::int64_t running = p[0];
    const auto eh = c.engram_heads();
    for (std::uint32_t order = 2; order <= c.ngram; ++order) {
        running ^= p[order - 1U];
        for (std::uint32_t h = 0; h < c.heads_per_ngram; ++h) {
            const auto head = (order - 2U) * c.heads_per_ngram + h;
            const auto index = ordinal * eh + head;
            rows.push_back(c.head_offsets[index] + positive_mod(running, c.head_sizes[index]));
        }
    }
    return rows;
}

// ---- the forward pass -----------------------------------------------------------------------------------------
struct HostRef::State {
    Config c;
    Weights& w;
    std::uint32_t pos = 0;
    std::vector<std::uint32_t> tokens;
    // per layer: per position, the expanded keys [heads][qk_head] and values [heads][v_head]
    std::vector<std::vector<Vec>> keys, values;
    std::vector<std::vector<Vec>> token_keys, group_keys;  // indexed layers
    // taps (read only, no bit change): per layer per position the latent row [kv_lora normed | qk_rope roped], and
    // the last step's hc_head output before / after output_norm
    std::vector<std::vector<Vec>> latents;
    Vec head_in, head_out;
    Vec logits;
    std::uint32_t next = 0;

    State(Config config, Weights& weights) : c(std::move(config)), w(weights) {
        c.validate();
        const std::size_t blocks = std::size_t{c.layers} + c.nextn;  // text layers + the next-token block
        keys.resize(blocks);
        values.resize(blocks);
        token_keys.resize(blocks);
        group_keys.resize(blocks);
        latents.resize(blocks);
    }
    // per-block attention / FFN form: the text layers' arrays, else the next-token block's
    std::uint32_t heads_of(std::uint32_t l) const { return l < c.layers ? c.heads[l] : c.nextn_heads; }
    bool indexed_of(std::uint32_t l) const { return l < c.layers ? c.indexed[l] != 0U : c.nextn_indexed != 0U; }
    bool dense_of(std::uint32_t l) const { return l < c.layers && c.dense[l] != 0U; }
    const Vec& t(const std::string& name) const { return w.tensor(name); }

    void engram(std::uint32_t layer, std::uint32_t ordinal, Vec& x) {
        const auto h = c.hidden;
        const auto pad = c.pad_token;
        const std::uint32_t q = pos;
        if (tokens[q] == pad) return;  // masked: the output is zero
        std::vector<std::uint32_t> window(c.ngram);
        for (std::uint32_t s = 0; s < c.ngram; ++s) window[s] = q >= s ? tokens[q - s] : pad;
        const auto rows = engram_rows(c, ordinal, window);
        const auto& lookup_norm = t(blk(layer, "engram_lookup_norm.weight"));
        Vec e;
        for (std::size_t head = 0; head < rows.size(); ++head) {
            auto row = w.row("per_layer_token_embd.weight", rows[head]);
            for (auto& v : row) v = bf(v);
            const auto normed = rms_norm(row.data(), row.size(), lookup_norm.data() + head * c.engram_dim, c.engram_norm_eps);
            e.insert(e.end(), normed.begin(), normed.end());
        }
        // engram_kv rows: keys of copies 0..3, then the value (forward-pass Q3); engram_kv_b: the five biases
        const auto& kv = t(blk(layer, "engram_kv.weight"));
        const auto& kv_b = t(blk(layer, "engram_kv_b.weight"));
        const auto projected = [&](std::size_t block) {  // Linear with bias: one BF16 rounding of (dot + bias)
            Vec y(h);
            for (std::size_t i = 0; i < h; ++i) y[i] = bf(dot(kv.data() + (block * h + i) * e.size(), e.data(), e.size()) + kv_b[block * h + i]);
            return y;
        };
        const auto value = projected(4);
        const auto& pre_gamma = t(blk(layer, "engram_pre_gamma.weight"));
        const auto& gate_w = t(blk(layer, "engram_gate.weight"));
        const auto& hc_scale = t(blk(layer, "engram_hc_scale.weight"));
        const auto unit = rms_norm(value, nullptr, c.engram_norm_eps);
        Vec m(h);
        for (std::size_t i = 0; i < h; ++i) m[i] = bf(pre_gamma[i] * unit[i]);
        const float inv_sqrt_hidden = static_cast<float>(1.0 / std::sqrt(static_cast<double>(h)));
        for (std::uint32_t copy = 0; copy < 4U; ++copy) {
            const auto key = rms_norm(projected(copy), nullptr, c.engram_norm_eps);
            const auto query = rms_norm(x.data() + copy * h, h, nullptr, c.engram_norm_eps);
            Vec term(h);
            for (std::size_t i = 0; i < h; ++i) term[i] = bf(bf(key[i] * gate_w[copy * h + i]) * query[i]);
            const float sum = ds41::chunked_dot_of([&](std::size_t i) { return term[i]; }, [](std::size_t) { return 1.0F; }, h);
            const float d = bf(sum * inv_sqrt_hidden);
            const float sign = d > 0.0F ? 1.0F : (d < 0.0F ? -1.0F : 0.0F);
            const float g = bf(sigmoid(sign * ds41::declared_sqrt(std::max(std::fabs(d), 1e-6F))));
            for (std::size_t i = 0; i < h; ++i) {
                const float out = bf(bf(g * m[i]) + bf(hc_scale[copy * h + i] * m[i]));
                x[copy * h + i] = bf(x[copy * h + i] + out);
            }
        }
    }

    Vec attention(std::uint32_t l, const Vec& xn) {
        const auto H = heads_of(l), qk = c.qk_head(), nope = c.qk_nope, kl = c.kv_lora, vd = c.v_head;
        const std::uint32_t q = pos;
        const bool idx = indexed_of(l);
        const float theta = idx ? c.rope_theta_indexed : c.rope_theta_sliding;
        const auto gate = linear(t(blk(l, "attn_gate.weight")), xn, std::size_t{H} * vd);
        const auto qr = rms_norm(linear(t(blk(l, "attn_q_a.weight")), xn, c.q_lora), &t(blk(l, "attn_q_a_norm.weight")), c.rms_eps);
        auto qv = linear(t(blk(l, "attn_q_b.weight")), qr, std::size_t{H} * qk);
        for (std::uint32_t head = 0; head < H; ++head) rope(qv.data() + head * qk + nope, c.qk_rope, q, theta);
        auto ckv = linear(t(blk(l, "attn_kv_a_mqa.weight")), xn, std::size_t{kl} + c.qk_rope);
        const auto lat = rms_norm(ckv.data(), kl, t(blk(l, "attn_kv_a_norm.weight")).data(), c.rms_eps);
        rope(ckv.data() + kl, c.qk_rope, q, theta);
        Vec tap(lat);
        tap.insert(tap.end(), ckv.begin() + kl, ckv.end());
        latents[l].push_back(std::move(tap));
        // expanded per-head keys and values (attn_k_b: per head kv_lora rows of qk_nope; attn_v_b: v_head rows of kv_lora)
        const auto& kb = t(blk(l, "attn_k_b.weight"));
        const auto& vb = t(blk(l, "attn_v_b.weight"));
        Vec k(std::size_t{H} * qk), v(std::size_t{H} * vd);
        for (std::uint32_t head = 0; head < H; ++head) {
            for (std::uint32_t d = 0; d < nope; ++d)
                k[head * qk + d] = bf(ds41::chunked_dot_of([&](std::size_t i) { return kb[(std::size_t{head} * kl + i) * nope + d]; },
                                                           [&](std::size_t i) { return lat[i]; }, kl));
            for (std::uint32_t r = 0; r < c.qk_rope; ++r) k[head * qk + nope + r] = ckv[kl + r];
            for (std::uint32_t e = 0; e < vd; ++e) v[head * vd + e] = bf(dot(vb.data() + (std::size_t{head} * vd + e) * kl, lat.data(), kl));
        }
        keys[l].push_back(std::move(k));
        values[l].push_back(std::move(v));
        // visible keys
        std::vector<std::uint32_t> visible;
        if (!idx) {
            for (std::uint32_t j = q + 1U > c.window ? q + 1U - c.window : 0U; j <= q; ++j) visible.push_back(j);
        } else {
            visible = indexer_select(l, xn, qr);
        }
        const float scale = 1.0F / ds41::declared_sqrt(static_cast<float>(qk));
        const auto& sinks = t(blk(l, "attn_sinks.weight"));
        Vec o(std::size_t{H} * vd);
        for (std::uint32_t head = 0; head < H; ++head) {
            Vec s(visible.size());
            float m = sinks[head];
            for (std::size_t a = 0; a < visible.size(); ++a) {
                s[a] = dot(qv.data() + head * qk, keys[l][visible[a]].data() + head * qk, qk) * scale;
                m = std::max(m, s[a]);
            }
            float z = 0.0F;
            for (auto& x : s) { x = ds41::declared_exp(x - m); z = z + x; }
            z = z + ds41::declared_exp(sinks[head] - m);
            for (std::uint32_t e = 0; e < vd; ++e) {
                float acc = 0.0F;
                for (std::size_t a = 0; a < visible.size(); ++a) acc = acc + bf(s[a] / z) * values[l][visible[a]][head * vd + e];
                o[head * vd + e] = bf(bf(acc) * bf(sigmoid(gate[head * vd + e])));
            }
        }
        return rms_norm(linear(t(blk(l, "attn_output.weight")), o, c.hidden), &t(blk(l, "post_attention_norm.weight")), c.rms_eps);
    }

    std::vector<std::uint32_t> indexer_select(std::uint32_t l, const Vec& xn, const Vec& qr) {
        const std::uint32_t q = pos, ratio = c.index_ratio, dim = c.index_dim;
        token_keys[l].push_back(rms_norm(linear(t(blk(l, "indexer.attn_k.weight")), xn, dim), &t(blk(l, "indexer.k_norm.weight")), c.rms_eps));
        if ((q + 1U) % ratio == 0U) {  // group q / ratio is complete: pool it once (FP32), norm, rotate at its first slot
            const auto& wv = t(blk(l, "indexer_compressor_value.weight"));
            const auto& wg = t(blk(l, "indexer_compressor_gate.weight"));
            const auto& ape = t(blk(l, "indexer_compressor_ape.weight"));
            const std::uint32_t first = q + 1U - ratio;
            std::vector<Vec> val, gate;
            for (std::uint32_t s = 0; s < ratio; ++s) {
                val.push_back(linear(wv, token_keys[l][first + s], dim, false));
                auto g = linear(wg, token_keys[l][first + s], dim, false);
                for (std::uint32_t ch = 0; ch < dim; ++ch) g[ch] = g[ch] + ape[s * dim + ch];
                gate.push_back(std::move(g));
            }
            Vec pooled(dim);
            for (std::uint32_t ch = 0; ch < dim; ++ch) {
                float m = gate[0][ch];
                for (std::uint32_t s = 1; s < ratio; ++s) m = std::max(m, gate[s][ch]);
                float z = 0.0F, acc = 0.0F;
                std::vector<float> e(ratio);
                for (std::uint32_t s = 0; s < ratio; ++s) { e[s] = ds41::declared_exp(gate[s][ch] - m); z = z + e[s]; }
                for (std::uint32_t s = 0; s < ratio; ++s) acc = acc + val[s][ch] * (e[s] / z);
                pooled[ch] = acc;
            }
            const auto& norm_w = t(blk(l, "indexer_compressor_norm.weight"));
            const float r = 1.0F / ds41::declared_sqrt(dot(pooled.data(), pooled.data(), dim) / static_cast<float>(dim) + c.rms_eps);
            Vec key(dim);
            for (std::uint32_t ch = 0; ch < dim; ++ch) key[ch] = bf(norm_w[ch] * (pooled[ch] * r));
            rope(key.data() + dim / 2U, dim / 2U, first, c.rope_theta_indexed);
            group_keys[l].push_back(std::move(key));
        }
        const attn::group_indexer::Shape shape{ratio, c.index_heads, dim, c.index_groups, attn::group_indexer::Pooling::GatedSoftmaxApe,
                                               attn::group_indexer::OwnGroup::AlwaysOwn, attn::group_indexer::ScoreScale::InHeadWeight};
        if (attn::group_indexer::dense(shape, q)) {  // dense fallback
            std::vector<std::uint32_t> visible;
            for (std::uint32_t j = 0; j <= q; ++j) visible.push_back(j);
            return visible;
        }
        const std::uint32_t own = q / ratio;  // groups 0 .. own-1 are complete and compete
        auto iq = linear(t(blk(l, "indexer.attn_q_b.weight")), qr, std::size_t{c.index_heads} * dim);
        for (std::uint32_t head = 0; head < c.index_heads; ++head) rope(iq.data() + head * dim + dim / 2U, dim / 2U, q, c.rope_theta_indexed);
        auto hw = linear(t(blk(l, "indexer.proj.weight")), xn, c.index_heads, false);
        const float head_scale = (1.0F / ds41::declared_sqrt(static_cast<float>(c.index_heads))) *
                                 (1.0F / ds41::declared_sqrt(static_cast<float>(dim)));
        for (auto& x : hw) x = x * head_scale;
        Vec score(own);
        for (std::uint32_t g = 0; g < own; ++g) {
            float s = 0.0F;
            for (std::uint32_t head = 0; head < c.index_heads; ++head)
                s = s + hw[head] * std::max(0.0F, dot(iq.data() + head * dim, group_keys[l][g].data(), dim));
            score[g] = s;
        }
        // the shared selection (top pick_count by score, lower group on ties, ascending) and expansion (+ own tail)
        return attn::group_indexer::expand_tokens(shape, attn::group_indexer::select_groups(shape, score.data(), own), q);
    }

    Vec ffn(std::uint32_t l, const Vec& xf) {
        const auto& closing = t(blk(l, "post_ffw_norm.weight"));
        const auto swiglu = [](const Vec& g, const Vec& u, float limit) {
            Vec a(g.size());
            for (std::size_t i = 0; i < g.size(); ++i) a[i] = ops::glu::swiglu<ops::glu::DeclaredExp, ops::glu::Round::bf16>(g[i], u[i], limit);
            return a;
        };
        if (dense_of(l)) {
            const auto a = swiglu(linear(t(blk(l, "ffn_gate.weight")), xf, c.dense_ff), linear(t(blk(l, "ffn_up.weight")), xf, c.dense_ff),
                                   ops::glu::no_clamp);
            return rms_norm(linear(t(blk(l, "ffn_down.weight")), a, c.hidden), &closing, c.rms_eps);
        }
        const auto r = route(linear(t(blk(l, "ffn_gate_inp.weight")), xf, c.experts, false), t(blk(l, "exp_probs_b.bias")), c.top_k,
                             c.routed_scale);
        const auto z = linear(t(blk(l, "ffn_routed_down.weight")), xf, c.latent);
        const auto wg = blk(l, "ffn_gate_exps.weight");
        const auto wu = blk(l, "ffn_up_exps.weight");
        const auto wd = blk(l, "ffn_down_exps.weight");
        Vec acc(c.latent, 0.0F);
        for (std::size_t k = 0; k < r.ids.size(); ++k) {
            const std::size_t e = r.ids[k];
            auto g = linear_rows(w, wg, e * c.expert_ff, z, c.expert_ff);
            auto u = linear_rows(w, wu, e * c.expert_ff, z, c.expert_ff);
            const auto d = linear_rows(w, wd, e * c.latent, swiglu(g, u, c.swiglu_limit), c.latent);
            for (std::size_t i = 0; i < c.latent; ++i) acc[i] = bf(acc[i] + bf(d[i] * r.weights[k]));
        }
        const auto routed = linear(t(blk(l, "ffn_routed_up.weight")), acc, c.hidden);
        const auto shared = linear(t(blk(l, "ffn_down_shexp.weight")),
                                   swiglu(linear(t(blk(l, "ffn_gate_shexp.weight")), xf, c.shared_ff),
                                          linear(t(blk(l, "ffn_up_shexp.weight")), xf, c.shared_ff),
                                          ops::glu::no_clamp),
                                   c.hidden);
        Vec sum(c.hidden);
        for (std::size_t i = 0; i < c.hidden; ++i) sum[i] = bf(routed[i] + shared[i]);
        return rms_norm(sum, &closing, c.rms_eps);
    }

    const Vec& step(std::uint32_t token) {
        if (token >= c.vocab) throw std::invalid_argument("hy4 ref: token outside the vocabulary");
        tokens.push_back(token);
        const auto h = c.hidden;
        const auto& embd = t("token_embd.weight");
        Vec x(4U * std::size_t{h});
        for (std::size_t copy = 0; copy < 4U; ++copy)
            for (std::size_t i = 0; i < h; ++i) x[copy * h + i] = bf(embd[std::size_t{token} * h + i]);
        for (std::uint32_t l = 0; l < c.layers; ++l) {
            const auto found = std::find(c.engram_layers.begin(), c.engram_layers.end(), l);
            if (found != c.engram_layers.end()) engram(l, static_cast<std::uint32_t>(found - c.engram_layers.begin()), x);
            const auto am = hc_mix(x, t(blk(l, "hc_attn_fn.weight")), t(blk(l, "hc_attn_base.weight")), t(blk(l, "hc_attn_scale.weight")), 8U,
                                   c.rms_eps, c.hc_eps);
            const auto xn = rms_norm(hc_collapse(x, am, h), &t(blk(l, "attn_norm.weight")), c.rms_eps);
            hc_update(x, am, attention(l, xn));
            const auto fm = hc_mix(x, t(blk(l, "hc_ffn_fn.weight")), t(blk(l, "hc_ffn_base.weight")), t(blk(l, "hc_ffn_scale.weight")), 8U,
                                   c.rms_eps, c.hc_eps);
            const auto xf = rms_norm(hc_collapse(x, fm, h), &t(blk(l, "ffn_norm.weight")), c.rms_eps);
            hc_update(x, fm, ffn(l, xf));
        }
        const auto hm = hc_mix(x, t("output_hc_fn.weight"), t("output_hc_base.weight"), t("output_hc_scale.weight"), 4U, c.rms_eps, c.hc_eps);
        head_in = hc_collapse(x, hm, h);
        head_out = rms_norm(head_in, &t("output_norm.weight"), c.rms_eps);
        logits = linear(t("output.weight"), head_out, c.vocab, false);
        next = static_cast<std::uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        ++pos;
        return logits;
    }

    // ---- the next-token (MTP) block: block id c.layers, its own K/V; `pos` is borrowed for the step only ----
    std::uint32_t nextn_positions() const { return c.nextn ? static_cast<std::uint32_t>(keys[c.layers].size()) : 0U; }

    NextnStep nextn_step(std::uint32_t token, const Vec& hid, std::uint32_t position) {
        if (c.nextn != 1U) throw std::invalid_argument("hy4 ref: the configuration has no next-token block");
        if (token >= c.vocab) throw std::invalid_argument("hy4 ref: token outside the vocabulary");
        const auto h = c.hidden, L = c.layers;
        if (hid.size() != h) throw std::invalid_argument("hy4 ref: next-token hidden width");
        if (position != nextn_positions()) throw std::invalid_argument("hy4 ref: next-token position != cached positions");
        const auto er = w.row("token_embd.weight", token);
        Vec e(h);
        for (std::size_t i = 0; i < h; ++i) e[i] = bf(er[i]);
        Vec cat = rms_norm(e, &t(blk(L, "nextn.enorm.weight")), c.rms_eps);
        const auto hn = rms_norm(hid, &t(blk(L, "nextn.hnorm.weight")), c.rms_eps);
        cat.insert(cat.end(), hn.begin(), hn.end());
        Vec x = linear(t(blk(L, "nextn.eh_proj.weight")), cat, h);
        const std::uint32_t saved = pos;
        pos = position;  // attention / indexer read the query position from pos
        try {
            const auto a = attention(L, rms_norm(x, &t(blk(L, "attn_norm.weight")), c.rms_eps));
            for (std::size_t i = 0; i < h; ++i) x[i] = bf(x[i] + a[i]);
            const auto f = ffn(L, rms_norm(x, &t(blk(L, "ffn_norm.weight")), c.rms_eps));
            for (std::size_t i = 0; i < h; ++i) x[i] = bf(x[i] + f[i]);
        } catch (...) {
            pos = saved;
            throw;
        }
        pos = saved;
        NextnStep out;
        out.normed = rms_norm(x, &t(blk(L, "nextn.shared_head_norm.weight")), c.rms_eps);
        out.logits = linear(t("output.weight"), out.normed, c.vocab, false);
        out.argmax = static_cast<std::uint32_t>(std::max_element(out.logits.begin(), out.logits.end()) - out.logits.begin());
        out.hidden = std::move(x);
        return out;
    }

    void nextn_truncate(std::uint32_t n) {
        if (c.nextn != 1U || n >= nextn_positions()) return;
        const auto L = c.layers;
        keys[L].resize(n);
        values[L].resize(n);
        latents[L].resize(n);
        if (token_keys[L].size() > n) token_keys[L].resize(n);
        const std::size_t groups = n / c.index_ratio;  // group g is pooled once position (g + 1) * ratio - 1 is cached
        if (group_keys[L].size() > groups) group_keys[L].resize(groups);
    }
};

HostRef::HostRef(Config config, Weights& weights) : state_(std::make_unique<State>(std::move(config), weights)) {}
HostRef::~HostRef() = default;
const std::vector<float>& HostRef::step(std::uint32_t token) { return state_->step(token); }
std::uint32_t HostRef::position() const noexcept { return state_->pos; }
std::uint32_t HostRef::next_token() const noexcept { return state_->next; }
const Config& HostRef::config() const noexcept { return state_->c; }
const std::vector<float>& HostRef::latent_row(std::uint32_t layer, std::uint32_t position) const {
    if (layer >= state_->latents.size() || position >= state_->latents[layer].size())
        throw std::out_of_range("hy4 ref: latent_row outside the fed positions");
    return state_->latents[layer][position];
}
HostRef::NextnStep HostRef::nextn_step(std::uint32_t token, const std::vector<float>& h, std::uint32_t position) {
    return state_->nextn_step(token, h, position);
}
void HostRef::nextn_truncate(std::uint32_t positions) { state_->nextn_truncate(positions); }
std::uint32_t HostRef::nextn_positions() const noexcept { return state_->nextn_positions(); }
const std::vector<float>& HostRef::last_hidden(bool normed) const {
    if (state_->pos == 0U) throw std::out_of_range("hy4 ref: last_hidden before the first step");
    return normed ? state_->head_out : state_->head_in;
}

}  // namespace strix::hy4
