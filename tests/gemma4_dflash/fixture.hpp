#pragma once

#include "gemma4/dflash/dflash.hpp"

#include <bit>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace fixture {
using namespace strix::gemma4::dflash;

inline void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    check(caught, "expected exception");
}

inline Shape shape() { return Shape{16, 2, 4, 1, 8, 32}; }

inline Bf16Matrix matrix(std::uint32_t outputs, std::uint32_t width, std::uint32_t seed) {
    Bf16Matrix m{outputs, width, std::vector<std::uint16_t>(std::size_t{outputs} * width)};
    for (std::size_t i = 0; i < m.bits.size(); ++i) {
        const float f = static_cast<float>(static_cast<int>((i * 7 + seed) % 17) - 8) / 64.0F;
        m.bits[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(f) >> 16);
    }
    return m;
}

inline Weights weights(Shape s = shape()) {
    Weights w;
    w.shape = s;
    w.fc = matrix(s.hidden, static_cast<std::uint32_t>(s.tap_width()), 1);
    w.context_norm.assign(s.hidden, 1.0F);
    w.final_norm.assign(s.hidden, 0.75F);
    w.layer.resize(s.layers);
    for (std::uint32_t l = 0; l < s.layers; ++l) {
        auto& y = w.layer[l];
        y.q = matrix(static_cast<std::uint32_t>(s.q_width()), s.hidden, 2 + l);
        y.k = matrix(static_cast<std::uint32_t>(s.kv_width()), s.hidden, 3 + l);
        y.v = matrix(static_cast<std::uint32_t>(s.kv_width()), s.hidden, 4 + l);
        y.o = matrix(s.hidden, static_cast<std::uint32_t>(s.q_width()), 5 + l);
        y.gate = matrix(s.inter, s.hidden, 6 + l);
        y.up = matrix(s.inter, s.hidden, 7 + l);
        y.down = matrix(s.hidden, s.inter, 8 + l);
        y.input_norm.assign(s.hidden, 0.875F);
        y.ffn_norm.assign(s.hidden, 0.625F);
        y.q_norm.assign(s.head_dim, 0.75F);
        y.k_norm.assign(s.head_dim, 1.125F);
    }
    w.checkpoint_embedding_scale = 50.59644317626953F;
    w.validate();
    return w;
}

struct Tables {
    std::vector<float> cs, sn;
    std::uint32_t positions, dim;
    Tables(std::uint32_t n, std::uint32_t d) : cs(std::size_t{n} * (d / 2), 1.0F), sn(cs.size()), positions(n), dim(d) {}
    [[nodiscard]] RopeView view() const { return {cs, sn, positions, dim}; }
};

template<class T> void put(std::ostream& out, T value) {
    static_assert(std::is_integral_v<T>);
    for (std::size_t i = 0; i < sizeof(T); ++i) out.put(static_cast<char>((value >> (8 * i)) & 255));
}
inline void string(std::ostream& out, const std::string& s) {
    put(out, static_cast<std::uint64_t>(s.size()));
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// Tiny real GGUF, including retained array metadata. Mutation names exercise admission failures.
inline void gguf_file(const std::filesystem::path& path, const Weights& w, const std::string& mutation = {}) {
    struct Tensor { std::string name; std::vector<std::uint64_t> dims; std::uint32_t type; std::vector<std::uint8_t> data; };
    std::vector<Tensor> tensors;
    auto linear = [&](const std::string& name, const Bf16Matrix& m) {
        Tensor t{name + ".weight", {m.width, m.outputs}, 30, {}};
        for (auto x : m.bits) { t.data.push_back(static_cast<std::uint8_t>(x)); t.data.push_back(static_cast<std::uint8_t>(x >> 8)); }
        tensors.push_back(std::move(t));
    };
    auto norm = [&](const std::string& name, const std::vector<float>& v) {
        Tensor t{name + ".weight", {v.size()}, 0, {}};
        for (auto x : v) {
            auto b = std::bit_cast<std::uint32_t>(x);
            for (std::size_t i = 0; i < 4; ++i) t.data.push_back(static_cast<std::uint8_t>(b >> (8 * i)));
        }
        tensors.push_back(std::move(t));
    };
    linear("fc", w.fc); norm("enc.output_norm", w.context_norm); norm("output_norm", w.final_norm);
    for (std::size_t i = 0; i < w.layer.size(); ++i) {
        const auto& l = w.layer[i]; const std::string p = "blk." + std::to_string(i) + ".";
        linear(p + "attn_q", l.q); linear(p + "attn_k", l.k); linear(p + "attn_v", l.v);
        linear(p + "attn_output", l.o); linear(p + "ffn_gate", l.gate); linear(p + "ffn_up", l.up);
        linear(p + "ffn_down", l.down); norm(p + "attn_norm", l.input_norm); norm(p + "ffn_norm", l.ffn_norm);
        norm(p + "attn_q_norm", l.q_norm); norm(p + "attn_k_norm", l.k_norm);
    }
    if (mutation == "missing") tensors.pop_back();
    if (mutation == "extra") tensors.push_back({"output.weight", {1}, 0, {0, 0, 0, 0}});
    if (mutation == "shape") ++tensors[0].dims[1];
    if (mutation == "dtype") tensors[0].type = 1;
    if (mutation == "nan") { tensors[0].data[0] = 0xc0; tensors[0].data[1] = 0x7f; }
    if (mutation == "norm_nan") { auto& d = tensors[1].data; d[0] = 0; d[1] = 0; d[2] = 0xc0; d[3] = 0x7f; }
    std::ofstream out(path, std::ios::binary);
    out.write("GGUF", 4); put(out, std::uint32_t{3}); put(out, static_cast<std::uint64_t>(tensors.size()));
    put(out, std::uint64_t{21});
    auto u = [&](const std::string& k, std::uint32_t v) { string(out, k); put(out, std::uint32_t{4}); put(out, v); };
    auto f = [&](const std::string& k, float v) { string(out, k); put(out, std::uint32_t{6}); put(out, std::bit_cast<std::uint32_t>(v)); };
    auto text = [&](const std::string& k, const std::string& v) { string(out, k); put(out, std::uint32_t{8}); string(out, v); };
    text("general.architecture", "dflash");
    text("general.license", "fixture");
    u("dflash.block_count", w.shape.layers); u("dflash.embedding_length", w.shape.hidden);
    u("dflash.feed_forward_length", w.shape.inter); u("dflash.attention.head_count", w.shape.heads);
    u("dflash.attention.head_count_kv", w.shape.kv_heads); u("dflash.attention.key_length", w.shape.head_dim);
    u("dflash.attention.value_length", w.shape.head_dim); u("dflash.block_size", block);
    u("dflash.context_length", 131072); u("dflash.attention.sliding_window", mutation == "window" ? 512 : window);
    f("dflash.rope.freq_base", 1000000.0F); f("dflash.attention.layer_norm_rms_epsilon", 1e-6F);
    f("dflash.embedding_scale", w.checkpoint_embedding_scale);
    f("dflash.logit_scale", 1.25F); f("dflash.final_logit_softcapping", 6.0F);
    u("dflash.mask_token_id", mutation == "mask" ? 5 : mask_id);
    string(out, "dflash.attention.causal"); put(out, std::uint32_t{7}); put(out, std::uint8_t{0});
    string(out, "dflash.attention.sliding_window_pattern"); put(out, std::uint32_t{9}); put(out, std::uint32_t{7});
    put(out, static_cast<std::uint64_t>(w.shape.layers));
    for (std::uint32_t i = 0; i < w.shape.layers; ++i) put(out, std::uint8_t{1});
    string(out, "dflash.target_layers"); put(out, std::uint32_t{9});
    put(out, mutation == "unsigned_taps" ? std::uint32_t{4} : std::uint32_t{5}); put(out, std::uint64_t{5});
    for (auto i : tap_layers) put(out, i + (mutation == "taps" ? 0U : 1U));
    std::uint64_t offset = 0;
    for (const auto& t : tensors) {
        string(out, t.name); put(out, static_cast<std::uint32_t>(t.dims.size()));
        for (auto d : t.dims) put(out, d);
        put(out, t.type); put(out, offset);
        offset += (t.data.size() + 31U) / 32U * 32U;
    }
    while (static_cast<std::uint64_t>(out.tellp()) % 32 != 0) out.put(0);
    for (const auto& t : tensors) {
        out.write(reinterpret_cast<const char*>(t.data.data()), static_cast<std::streamsize>(t.data.size()));
        for (std::size_t i = t.data.size(); i % 32 != 0; ++i) out.put(0);
    }
    check(static_cast<bool>(out), "could not write fixture");
}
} // namespace fixture
