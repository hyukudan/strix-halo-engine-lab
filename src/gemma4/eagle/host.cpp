// e4b-eagle: loader/writer, host g4-v1 reference drafter and the greedy verification cycle helper.
#include "eagle.hpp"
#include "gemma4/kernels/ops.hpp"
#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace strix::gemma4::eagle {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("eagle: " + message);
}
bool finite_bits(float x) { return (bit_cast<std::uint32_t>(x) & 0x7f800000U) != 0x7f800000U; }
void finite(std::span<const float> values, const char* what) {
    for (const float x : values) require(finite_bits(x), std::string("nonfinite ") + what);
}
bool power_of_two(float x) {
    if (!(x > 0.0F) || !finite_bits(x)) return false;
    const auto bits = bit_cast<std::uint32_t>(x);
    return (bits & 0x7fffffU) == 0 && (bits >> 23U) != 0;
}

std::uint16_t half_bits(float f) {
    const auto x = bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (x >> 16U) & 0x8000U;
    std::uint32_t mant = x & 0x7fffffU;
    const int exp = int((x >> 23U) & 0xffU);
    if (exp == 255) return std::uint16_t(sign | 0x7c00U | (mant ? 0x200U : 0U));
    const int e = exp - 127 + 15;
    if (e >= 31) return std::uint16_t(sign | 0x7c00U);
    if (e <= 0) {
        if (e < -10) return std::uint16_t(sign);
        mant |= 0x800000U;
        const auto shift = std::uint32_t(14 - e);
        std::uint32_t half = mant >> shift;
        const std::uint32_t rem = mant & ((1U << shift) - 1U), mid = 1U << (shift - 1U);
        if (rem > mid || (rem == mid && (half & 1U))) ++half;
        return std::uint16_t(sign | half);
    }
    std::uint32_t half = (std::uint32_t(e) << 10U) | (mant >> 13U);
    const std::uint32_t rem = mant & 0x1fffU;
    if (rem > 0x1000U || (rem == 0x1000U && (half & 1U))) ++half;
    return std::uint16_t(sign | half);
}

// ------------------------------------------------------------------ g4-v1 host operators
void linear(LinearView w, const float* x, std::size_t rows, float* out) {
    std::vector<float> decoded(w.inputs);
    for (std::uint32_t o = 0; o < w.outputs; ++o) {
        for (std::uint32_t i = 0; i < w.inputs; ++i) decoded[i] = gpu::coefficient(w, o, i);
        for (std::size_t r = 0; r < rows; ++r)
            out[r * w.outputs + o] = chunk16_dot([&](std::uint32_t i) { return decoded[i]; }, x + r * w.inputs, w.inputs);
    }
    finite({out, rows * w.outputs}, "linear output");
}
void norm(const float* x, const float* w, float* out, std::uint32_t width) {
    require(finite_bits(lane_dot(x, x, width)), "nonfinite norm reduction");
    rms_norm(x, w, out, width, norm_eps);
    finite({out, width}, "norm output");
}
// In-place head RMSNorm (+ optional NeoX RoPE over all head_dim/2 pairs from the sliding table row).
void head_norm(float* v, const float* w, std::uint32_t dim, const float* rope_row) {
    norm(v, w, v, dim);
    if (!rope_row) return;
    for (std::uint32_t i = 0; i < dim / 2U; ++i) {
        const float a = v[i], b = v[i + dim / 2U], cs = rope_row[2U * i], sn = rope_row[2U * i + 1U];
        v[i] = a * cs - b * sn;
        v[i + dim / 2U] = b * cs + a * sn;
    }
}
float capped(float x, float cap) { return declared_tanh(x / cap) * cap; }
} // namespace

// ------------------------------------------------------------------ config / packing
bool Config::needs_final() const { return std::find(taps.begin(), taps.end(), final_tap) != taps.end(); }
std::uint32_t Config::tap_slot(std::uint32_t j) const {
    require(j < taps.size(), "tap index");
    if (taps[j] == final_tap) return engine_taps;
    for (std::uint32_t s = 0; s < engine_taps; ++s)
        if (std::int32_t(tap_layers[s]) == taps[j]) return s;
    throw std::runtime_error("eagle: tap layer is not exported by the target (taps 5/11/17/23/35 or final)");
}
void Config::validate() const {
    require(hidden > 0 && hidden <= gemma4::hidden && hidden % 32 == 0, "hidden width");
    require(head_dim == 256, "head_dim must be 256 (target sliding RoPE table)");
    require(kv_heads > 0 && heads == 4 * kv_heads && heads <= 32, "heads must be 4x kv heads (device attention group)");
    require(ffn > 0 && ffn % 32 == 0 && ffn <= 16384, "ffn width");
    require(window == gemma4::window, "window must be 512");
    require(vocab > 0 && vocab <= target_vocab && target_vocab <= gemma4::vocab, "draft vocabulary");
    require(target_hidden == hidden, "drafter width must equal the target hidden width");
    require(!taps.empty() && taps.size() <= engine_taps + 1, "tap count");
    std::set<std::int32_t> unique(taps.begin(), taps.end());
    require(unique.size() == taps.size(), "duplicate taps");
    for (std::uint32_t j = 0; j < taps.size(); ++j) (void)tap_slot(j);
    require(power_of_two(attention_scale), "attention scale must be a power of two");
    require(softcap > 0.0F && finite_bits(softcap), "softcap");
    require(eps == norm_eps, "RMSNorm epsilon must be 1e-6");
    require(rope_theta == 10000.0F, "RoPE theta must be 1e4 (target sliding table)");
    require(embedding_scale == gemma4::embedding_scale, "embedding scale must be bf16(sqrt(2560)) = 50.5");
}
std::size_t row_bytes(WeightFormat format, std::uint32_t inputs) {
    switch (format) {
    case WeightFormat::f32: return 4ULL * inputs;
    case WeightFormat::bf16: return 2ULL * inputs;
    case WeightFormat::q8_0: require(inputs % 32 == 0, "q8_0 width"); return (inputs / 32ULL) * 34ULL;
    default: throw std::runtime_error("eagle: matrices must be F32, BF16 or Q8_0");
    }
}
std::size_t Matrix::row_bytes() const { return eagle::row_bytes(format, inputs); }
LinearView Matrix::view() const { return {bytes.data(), outputs, inputs, row_bytes(), format}; }
Matrix pack(std::span<const float> values, std::uint32_t outputs, std::uint32_t inputs, WeightFormat format) {
    require(values.size() == std::size_t(outputs) * inputs, "pack shape");
    Matrix m{{}, outputs, inputs, format};
    const auto stride = m.row_bytes();
    m.bytes.assign(stride * outputs, 0);
    for (std::uint32_t o = 0; o < outputs; ++o) {
        const float* x = values.data() + std::size_t(o) * inputs;
        auto* row = m.bytes.data() + std::size_t(o) * stride;
        if (format == WeightFormat::f32) std::memcpy(row, x, 4ULL * inputs);
        else if (format == WeightFormat::bf16)
            for (std::uint32_t i = 0; i < inputs; ++i) {
                const auto b = bf16_bits(x[i]);
                row[2 * i] = std::uint8_t(b & 0xffU); row[2 * i + 1] = std::uint8_t(b >> 8U);
            }
        else
            for (std::uint32_t blk = 0; blk < inputs / 32; ++blk) {
                float amax = 0.0F;
                for (std::uint32_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[blk * 32 + i]));
                const float d = amax / 127.0F, id = d != 0.0F ? 1.0F / d : 0.0F;
                const auto h = half_bits(d);
                auto* b = row + blk * 34ULL;
                b[0] = std::uint8_t(h & 0xffU); b[1] = std::uint8_t(h >> 8U);
                for (std::uint32_t i = 0; i < 32; ++i)
                    b[2 + i] = std::uint8_t(std::int8_t(std::clamp(std::lround(x[blk * 32 + i] * id), -127L, 127L)));
            }
    }
    return m;
}

// ------------------------------------------------------------------ weights
std::vector<float> Weights::q_norm_effective() const {
    std::vector<float> out(q_norm.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = q_norm[i] * config.attention_scale;
    return out;
}
void Weights::validate() const {
    const auto& c = config;
    c.validate();
    const auto H = c.hidden;
    auto matrix = [&](const Matrix& m, std::uint32_t outputs, std::uint32_t inputs, const char* name) {
        require(m.outputs == outputs && m.inputs == inputs && m.bytes.size() == std::size_t(outputs) * m.row_bytes(),
                std::string("matrix shape ") + name);
        const auto view = m.view();
        if (m.format == WeightFormat::q8_0) {
            for (std::uint32_t row = 0; row < outputs; ++row)
                for (std::uint32_t b = 0; b < inputs / 32; ++b)
                    require(finite_bits(fp16_value(load16(view.data + row * view.row_bytes + b * 34ULL))),
                            std::string("nonfinite q8 scale ") + name);
        } else {
            for (std::uint32_t row = 0; row < outputs; ++row)
                for (std::uint32_t i = 0; i < inputs; ++i)
                    require(finite_bits(gpu::coefficient(view, row, i)), std::string("nonfinite weight ") + name);
        }
    };
    auto vec = [&](const std::vector<float>& w, std::uint32_t width, const char* name) {
        require(w.size() == width, std::string("norm shape ") + name);
        finite(w, name);
    };
    require(tap_norm.size() == c.tap_count(), "tap norm count");
    for (const auto& t : tap_norm) vec(t, H, "tap_norm");
    matrix(fc, H, c.tap_count() * H, "fc");
    vec(h_norm, H, "h_norm"); vec(e_norm, H, "e_norm");
    if (c.concat) matrix(in_proj, H, 2 * H, "in_proj");
    else require(in_proj.bytes.empty(), "sum combine has no in_proj");
    vec(input_norm, H, "input_norm");
    matrix(q, c.q_width(), H, "q"); matrix(k, c.kv_width(), H, "k"); matrix(v, c.kv_width(), H, "v");
    require(q.format == k.format && k.format == v.format, "q/k/v must share one format (fused device QKV)");
    vec(q_norm, c.head_dim, "q_norm"); vec(k_norm, c.head_dim, "k_norm");
    finite(q_norm_effective(), "scaled q_norm");
    matrix(o, H, c.q_width(), "o");
    vec(post_attn_norm, H, "post_attn_norm"); vec(pre_ffn_norm, H, "pre_ffn_norm");
    matrix(gate, c.ffn, H, "gate"); matrix(up, c.ffn, H, "up"); matrix(down, H, c.ffn, "down");
    require(gate.format == up.format, "gate/up format");
    vec(post_ffn_norm, H, "post_ffn_norm"); vec(final_norm, H, "final_norm");
    matrix(head, c.vocab, H, "head");
    require(d2t.size() == c.vocab, "d2t size");
    std::vector<bool> seen(c.target_vocab, false);
    for (const auto id : d2t) {
        require(id < c.target_vocab && !seen[id], "d2t ids must be unique target ids");
        seen[id] = true;
    }
}

namespace {
class Loader {
public:
    explicit Loader(const std::filesystem::path& path) : input_(path, std::ios::binary) {
        gguf::Limits limits;
        limits.max_retained_array_elements = 64;
        info_ = gguf::read_file(path, limits);
        require(bool(input_), "cannot open drafter file");
    }
    Weights load() {
        Weights w;
        auto& c = w.config;
        require(text("general.architecture") == "e4b-eagle", "unsupported architecture (want e4b-eagle)");
        require(number("eagle.version") == 1, "unsupported eagle.version");
        c.hidden = u32("eagle.embedding_length");
        c.ffn = u32("eagle.feed_forward_length");
        c.heads = u32("eagle.attention.head_count");
        c.kv_heads = u32("eagle.attention.head_count_kv");
        c.head_dim = u32("eagle.attention.key_length");
        require(u32("eagle.attention.value_length") == c.head_dim, "value_length must equal key_length");
        c.window = u32("eagle.attention.sliding_window");
        c.attention_scale = float(number("eagle.attention.scale"));
        c.eps = float(number("eagle.attention.layer_norm_rms_epsilon"));
        c.rope_theta = float(number("eagle.rope.freq_base"));
        c.vocab = u32("eagle.vocab_size");
        c.softcap = float(number("eagle.final_logit_softcapping"));
        c.target_hidden = u32("eagle.target_embedding_length");
        c.target_vocab = u32("eagle.target_vocab_size");
        c.embedding_scale = float(number("eagle.embedding_scale"));
        const auto combine = text("eagle.combine");
        require(combine == "concat" || combine == "sum", "eagle.combine must be concat or sum");
        c.concat = combine == "concat";
        const auto* aux = info_.find_metadata("eagle.aux");
        require(!aux || (aux->boolean_value && !*aux->boolean_value), "PLE aux adapter is not supported by engine v1");
        const auto* taps = info_.find_metadata("eagle.target_layers");
        require(taps && (!taps->signed_array.empty() || !taps->unsigned_array.empty()), "missing eagle.target_layers");
        c.taps.clear();
        if (!taps->signed_array.empty())
            for (const auto t : taps->signed_array) c.taps.push_back(std::int32_t(t));
        else
            for (const auto t : taps->unsigned_array) c.taps.push_back(std::int32_t(t));
        c.validate();
        w.license = text("general.license", false);
        w.license_name = text("general.license.name", false);
        w.license_link = text("general.license.link", false);
        const auto H = c.hidden;
        for (std::uint32_t j = 0; j < c.tap_count(); ++j) w.tap_norm.push_back(norm("tap_norm." + std::to_string(j), H));
        w.fc = matrix("fc", H, c.tap_count() * H);
        w.h_norm = norm("h_norm", H); w.e_norm = norm("e_norm", H);
        if (c.concat) w.in_proj = matrix("in_proj", H, 2 * H);
        w.input_norm = norm("blk.0.attn_norm", H);
        w.q = matrix("blk.0.attn_q", c.q_width(), H);
        w.k = matrix("blk.0.attn_k", c.kv_width(), H);
        w.v = matrix("blk.0.attn_v", c.kv_width(), H);
        w.q_norm = norm("blk.0.attn_q_norm", c.head_dim);
        w.k_norm = norm("blk.0.attn_k_norm", c.head_dim);
        w.o = matrix("blk.0.attn_output", H, c.q_width());
        w.post_attn_norm = norm("blk.0.post_attention_norm", H);
        w.pre_ffn_norm = norm("blk.0.ffn_norm", H);
        w.gate = matrix("blk.0.ffn_gate", c.ffn, H);
        w.up = matrix("blk.0.ffn_up", c.ffn, H);
        w.down = matrix("blk.0.ffn_down", H, c.ffn);
        w.post_ffn_norm = norm("blk.0.post_ffw_norm", H);
        w.final_norm = norm("output_norm", H);
        w.head = matrix("output", c.vocab, H);
        {
            const auto& t = tensor("d2t", {c.vocab}, false);
            require(t.ggml_type == 26, "d2t must be I32");
            const auto data = bytes(t);
            w.d2t.resize(c.vocab);
            for (std::size_t i = 0; i < c.vocab; ++i) {
                std::int32_t value = 0;
                std::memcpy(&value, data.data() + 4 * i, 4);
                require(value >= 0, "negative d2t id");
                w.d2t[i] = std::uint32_t(value);
            }
        }
        require(used_.size() == info_.tensors.size(), "unexpected tensor in drafter file");
        w.tensor_bytes = tensor_bytes_;
        w.validate();
        return w;
    }
private:
    std::string text(const std::string& key, bool mandatory = true) const {
        const auto* e = info_.find_metadata(key);
        if (!mandatory && !e) return {};
        require(e && e->string_value, "missing/string metadata " + key);
        return *e->string_value;
    }
    double number(const std::string& key) const {
        const auto* e = info_.find_metadata(key);
        require(e != nullptr, "missing numeric metadata " + key);
        if (e->floating_value) return *e->floating_value;
        if (e->unsigned_value) return double(*e->unsigned_value);
        if (e->signed_value) return double(*e->signed_value);
        throw std::runtime_error("eagle: invalid numeric metadata " + key);
    }
    std::uint32_t u32(const std::string& key) const {
        const double x = number(key);
        require(x >= 0 && x <= 4294967295.0 && x == std::floor(x), "invalid integer metadata " + key);
        return std::uint32_t(x);
    }
    const gguf::TensorInfo& tensor(const std::string& name, const std::vector<std::uint64_t>& shape, bool weight = true) {
        const std::string full = weight ? name + ".weight" : name;
        const auto it = std::find_if(info_.tensors.begin(), info_.tensors.end(), [&](const auto& t) { return t.name == full; });
        require(it != info_.tensors.end(), "missing tensor " + full);
        require(it->dimensions == shape, "shape mismatch " + full);
        used_.insert(full);
        tensor_bytes_ += it->byte_size;
        return *it;
    }
    std::vector<std::uint8_t> bytes(const gguf::TensorInfo& t) {
        std::vector<std::uint8_t> data(std::size_t(t.byte_size));
        input_.seekg(std::streamoff(t.absolute_offset));
        input_.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size()));
        require(bool(input_), "truncated tensor " + t.name);
        return data;
    }
    Matrix matrix(const std::string& name, std::uint32_t outputs, std::uint32_t inputs) {
        const auto& t = tensor(name, {inputs, outputs});
        WeightFormat format{};
        if (t.ggml_type == 8) format = WeightFormat::q8_0;
        else if (t.ggml_type == 30) format = WeightFormat::bf16;
        else if (t.ggml_type == 0) format = WeightFormat::f32;
        else throw std::runtime_error("eagle: matrix must be Q8_0 (BF16/F32 fixtures) " + name);
        Matrix m{bytes(t), outputs, inputs, format};
        require(m.bytes.size() == std::size_t(outputs) * m.row_bytes(), "matrix byte size " + name);
        return m;
    }
    std::vector<float> norm(const std::string& name, std::uint32_t width) {
        const auto& t = tensor(name, {width});
        require(t.ggml_type == 0, "norm must be F32 " + name);
        const auto data = bytes(t);
        std::vector<float> out(width);
        std::memcpy(out.data(), data.data(), 4ULL * width);
        return out;
    }
    std::ifstream input_;
    gguf::FileInfo info_;
    std::set<std::string> used_;
    std::uint64_t tensor_bytes_ = 0;
};

// Minimal GGUF v3 writer (little endian host assumed, as everywhere in this engine).
class Writer {
public:
    void str(const std::string& key, const std::string& value) { kv(key, 8); put_string(meta_, value); ++count_; }
    void u32(const std::string& key, std::uint32_t value) { kv(key, 4); put(meta_, value); ++count_; }
    void f32(const std::string& key, float value) { kv(key, 6); put(meta_, value); ++count_; }
    void boolean(const std::string& key, bool value) { kv(key, 7); put(meta_, std::uint8_t(value ? 1 : 0)); ++count_; }
    void i32s(const std::string& key, std::span<const std::int32_t> values) {
        kv(key, 9); put(meta_, std::uint32_t(5)); put(meta_, std::uint64_t(values.size()));
        for (const auto x : values) put(meta_, x);
        ++count_;
    }
    void tensor(const std::string& name, std::vector<std::uint64_t> dims, std::uint32_t type, const void* data, std::size_t size) {
        tensors_.push_back({name, std::move(dims), type, std::vector<std::uint8_t>(static_cast<const std::uint8_t*>(data),
                                                                                     static_cast<const std::uint8_t*>(data) + size)});
    }
    void save(const std::filesystem::path& path) const {
        std::vector<std::uint8_t> out;
        out.insert(out.end(), {'G', 'G', 'U', 'F'});
        put(out, std::uint32_t(3)); put(out, std::uint64_t(tensors_.size())); put(out, std::uint64_t(count_));
        out.insert(out.end(), meta_.begin(), meta_.end());
        std::uint64_t offset = 0;
        for (const auto& t : tensors_) {
            put_string(out, t.name); put(out, std::uint32_t(t.dims.size()));
            for (const auto d : t.dims) put(out, d);
            put(out, t.type); put(out, offset);
            offset += (t.data.size() + 31U) / 32U * 32U;
        }
        out.resize((out.size() + 31U) / 32U * 32U, 0);
        for (const auto& t : tensors_) {
            out.insert(out.end(), t.data.begin(), t.data.end());
            out.resize((out.size() + 31U) / 32U * 32U, 0);
        }
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
        require(bool(f), "cannot write drafter file");
    }
private:
    struct T { std::string name; std::vector<std::uint64_t> dims; std::uint32_t type; std::vector<std::uint8_t> data; };
    template <class X> static void put(std::vector<std::uint8_t>& o, X x) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(&x);
        o.insert(o.end(), p, p + sizeof(X));
    }
    static void put_string(std::vector<std::uint8_t>& o, const std::string& s) {
        put(o, std::uint64_t(s.size())); o.insert(o.end(), s.begin(), s.end());
    }
    void kv(const std::string& key, std::uint32_t type) { put_string(meta_, key); put(meta_, type); }
    std::vector<std::uint8_t> meta_;
    std::vector<T> tensors_;
    std::uint64_t count_ = 0;
};
std::uint32_t ggml_type(WeightFormat f) {
    return f == WeightFormat::q8_0 ? 8U : f == WeightFormat::bf16 ? 30U : 0U;
}
struct Random {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }
    float uniform() { return float(double(next() >> 11U) * 0x1p-53 * 2.0 - 1.0); } // [-1, 1)
};
} // namespace

Weights Weights::load(const std::filesystem::path& path) { return Loader(path).load(); }

void Weights::save(const std::filesystem::path& path) const {
    validate();
    const auto& c = config;
    Writer w;
    w.str("general.architecture", "e4b-eagle");
    if (!license.empty()) w.str("general.license", license);
    if (!license_name.empty()) w.str("general.license.name", license_name);
    if (!license_link.empty()) w.str("general.license.link", license_link);
    w.u32("eagle.version", 1); w.u32("eagle.embedding_length", c.hidden); w.u32("eagle.feed_forward_length", c.ffn);
    w.u32("eagle.attention.head_count", c.heads); w.u32("eagle.attention.head_count_kv", c.kv_heads);
    w.u32("eagle.attention.key_length", c.head_dim); w.u32("eagle.attention.value_length", c.head_dim);
    w.u32("eagle.attention.sliding_window", c.window); w.f32("eagle.attention.scale", c.attention_scale);
    w.f32("eagle.attention.layer_norm_rms_epsilon", c.eps); w.f32("eagle.rope.freq_base", c.rope_theta);
    w.u32("eagle.vocab_size", c.vocab); w.f32("eagle.final_logit_softcapping", c.softcap);
    w.i32s("eagle.target_layers", c.taps); w.str("eagle.combine", c.concat ? "concat" : "sum");
    w.boolean("eagle.aux", false);
    w.u32("eagle.target_embedding_length", c.target_hidden); w.u32("eagle.target_vocab_size", c.target_vocab);
    w.f32("eagle.embedding_scale", c.embedding_scale);
    auto norm = [&](const std::string& name, const std::vector<float>& values) {
        w.tensor(name + ".weight", {values.size()}, 0, values.data(), values.size() * 4);
    };
    auto matrix = [&](const std::string& name, const Matrix& m) {
        w.tensor(name + ".weight", {m.inputs, m.outputs}, ggml_type(m.format), m.bytes.data(), m.bytes.size());
    };
    for (std::size_t j = 0; j < tap_norm.size(); ++j) norm("tap_norm." + std::to_string(j), tap_norm[j]);
    matrix("fc", fc); norm("h_norm", h_norm); norm("e_norm", e_norm);
    if (c.concat) matrix("in_proj", in_proj);
    norm("blk.0.attn_norm", input_norm);
    matrix("blk.0.attn_q", q); matrix("blk.0.attn_k", k); matrix("blk.0.attn_v", v);
    norm("blk.0.attn_q_norm", q_norm); norm("blk.0.attn_k_norm", k_norm);
    matrix("blk.0.attn_output", o);
    norm("blk.0.post_attention_norm", post_attn_norm); norm("blk.0.ffn_norm", pre_ffn_norm);
    matrix("blk.0.ffn_gate", gate); matrix("blk.0.ffn_up", up); matrix("blk.0.ffn_down", down);
    norm("blk.0.post_ffw_norm", post_ffn_norm); norm("output_norm", final_norm);
    matrix("output", head);
    std::vector<std::int32_t> ids(d2t.begin(), d2t.end());
    w.tensor("d2t", {ids.size()}, 26, ids.data(), ids.size() * 4);
    w.save(path);
}

Weights Weights::random(const Config& c, std::uint64_t seed, WeightFormat matrices, float head_std) {
    c.validate();
    Weights w;
    w.config = c;
    Random rng{seed};
    auto mat = [&](std::uint32_t outputs, std::uint32_t inputs, float scale) {
        std::vector<float> x(std::size_t(outputs) * inputs);
        for (auto& a : x) a = rng.uniform() * scale;
        return pack(x, outputs, inputs, matrices);
    };
    auto vec = [&](std::uint32_t width) {
        std::vector<float> x(width);
        for (auto& a : x) a = 1.0F + 0.25F * rng.uniform();
        return x;
    };
    const auto H = c.hidden;
    const float unit = 1.7320508F; // uniform[-1,1) * sqrt(3) / sqrt(fan_in) -> unit variance
    auto s = [&](std::uint32_t fan_in) { return unit / std::sqrt(float(fan_in)); };
    for (std::uint32_t j = 0; j < c.tap_count(); ++j) w.tap_norm.push_back(vec(H));
    w.fc = mat(H, c.tap_count() * H, s(c.tap_count() * H));
    w.h_norm = vec(H); w.e_norm = vec(H);
    if (c.concat) w.in_proj = mat(H, 2 * H, s(2 * H));
    w.input_norm = vec(H);
    w.q = mat(c.q_width(), H, s(H)); w.k = mat(c.kv_width(), H, s(H)); w.v = mat(c.kv_width(), H, s(H));
    w.q_norm = vec(c.head_dim); w.k_norm = vec(c.head_dim);
    w.o = mat(H, c.q_width(), s(c.q_width()));
    w.post_attn_norm = vec(H); w.pre_ffn_norm = vec(H);
    w.gate = mat(c.ffn, H, s(H)); w.up = mat(c.ffn, H, s(H)); w.down = mat(H, c.ffn, s(c.ffn));
    w.post_ffn_norm = vec(H); w.final_norm = vec(H);
    w.head = mat(c.vocab, H, head_std * unit);
    std::vector<std::uint32_t> ids(c.target_vocab);
    std::iota(ids.begin(), ids.end(), 0U);
    for (std::uint32_t i = 0; i < c.vocab; ++i) std::swap(ids[i], ids[i + rng.next() % (c.target_vocab - i)]);
    w.d2t.assign(ids.begin(), ids.begin() + c.vocab);
    w.license_name = "synthetic-test-weights";
    w.validate();
    return w;
}

void RopeView::validate() const {
    require(positions > 0 && positions <= gemma4::max_positions && table.size() == std::size_t(positions) * 256,
            "RoPE table must be the target sliding layout [positions][128][cos,sin]");
    finite(table, "RoPE table");
}
std::vector<float> make_rope_table(std::uint32_t positions, float theta) {
    std::vector<float> t(std::size_t(positions) * 256);
    for (std::uint32_t p = 0; p < positions; ++p)
        for (std::uint32_t i = 0; i < 128; ++i) {
            const double f = 1.0 / std::pow(double(theta), double(2 * i) / 256.0), a = double(p) * f;
            t[std::size_t(p) * 256 + 2 * i] = float(std::cos(a));
            t[std::size_t(p) * 256 + 2 * i + 1] = float(std::sin(a));
        }
    return t;
}

std::pair<std::uint32_t, float> select(std::span<const float> raw, float cap) {
    require(!raw.empty(), "empty draft vocabulary");
    constexpr std::uint32_t T = select_threads;
    const auto V = std::uint32_t(raw.size());
    std::vector<float> bv(T, -std::numeric_limits<float>::infinity());
    std::vector<std::uint32_t> bi(T, UINT32_MAX);
    for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t i = t; i < V; i += T) {
            require(finite_bits(raw[i]), "nonfinite draft logit");
            const float x = capped(raw[i], cap);
            if (better(x, i, bv[t], bi[t])) { bv[t] = x; bi[t] = i; }
        }
    for (std::uint32_t s = T / 2; s; s /= 2)
        for (std::uint32_t t = 0; t < s; ++t)
            if (better(bv[t + s], bi[t + s], bv[t], bi[t])) { bv[t] = bv[t + s]; bi[t] = bi[t + s]; }
    const float best = bv[0];
    std::vector<float> sum(T, 0.0F);
    for (std::uint32_t t = 0; t < T; ++t)
        for (std::uint32_t i = t; i < V; i += T) sum[t] = sum[t] + declared_exp(capped(raw[i], cap) - best);
    for (std::uint32_t s = T / 2; s; s /= 2)
        for (std::uint32_t t = 0; t < s; ++t) sum[t] = sum[t] + sum[t + s];
    return {bi[0], 1.0F / sum[0]};
}

bool enabled_from_environment() {
    const char* v = std::getenv("STRIX_G4_EAGLE");
    return !v || std::string_view(v) != "0";
}
Kind kind_from_environment() {
    const char* v = std::getenv("STRIX_G4_DRAFT_KIND");
    if (!v || std::string_view(v).empty() || std::string_view(v) == "dflash") return Kind::dflash;
    if (std::string_view(v) == "eagle") return Kind::eagle;
    if (std::string_view(v) == "mtp") return Kind::mtp;
    throw std::invalid_argument("STRIX_G4_DRAFT_KIND must be dflash, eagle or mtp");
}

// ------------------------------------------------------------------ host reference drafter
struct HostDrafter::Impl {
    const Weights& w;
    const Config& c;
    RopeView rope;
    LinearView embedding;
    std::vector<float> qn;
    std::vector<float> ring_k, ring_v;
    std::vector<std::uint64_t> tags;
    std::uint64_t frontier = 0;
    bool initialized = false, has_last = false;
    std::vector<float> last_x, last_a;
    std::uint32_t last_next = 0;
    // staged (unpublished) rows
    bool staged = false;
    std::vector<float> sk, sv, sx, sa;
    std::vector<std::uint64_t> spos;
    std::uint32_t snext = 0;
    Timings times;

    Impl(const Weights& weights, RopeView r, LinearView e)
        : w(weights), c(weights.config), rope(r), embedding(e), qn(weights.q_norm_effective()) {
        w.validate(); rope.validate();
        require(embedding.data && embedding.inputs == c.hidden && embedding.outputs >= c.target_vocab &&
                embedding.row_bytes >= row_bytes(embedding.format, embedding.inputs), "target embedding geometry");
        ring_k.assign(std::size_t(c.window) * c.kv_width(), 0.0F); ring_v = ring_k;
        tags.assign(c.window, UINT64_MAX);
    }
    const float* rope_row(std::uint64_t p) const {
        require(p < rope.positions, "position outside RoPE table");
        return rope.table.data() + std::size_t(p) * 256;
    }
    // h = fc(concat_j RMSNorm_j(tap_j))
    void fuse(const float* taps, const float* hidden, float* h) const {
        const auto H = c.hidden;
        std::vector<float> tn(std::size_t(c.tap_count()) * H);
        for (std::uint32_t j = 0; j < c.tap_count(); ++j) {
            const auto slot = c.tap_slot(j);
            const float* src = slot < engine_taps ? taps + std::size_t(slot) * H : hidden;
            norm(src, w.tap_norm[j].data(), tn.data() + std::size_t(j) * H, H);
        }
        linear(w.fc.view(), tn.data(), 1, h);
    }
    // x = in_proj([RMSNorm_h(h); RMSNorm_e(e(token))]) or RMSNorm_h(h) + RMSNorm_e(e)
    void combine(const float* h, std::uint32_t token, float* x) const {
        const auto H = c.hidden;
        require(token < c.target_vocab, "token outside target vocabulary");
        std::vector<float> e(H), cat(2ULL * H);
        for (std::uint32_t i = 0; i < H; ++i) e[i] = embedding_value(gpu::coefficient(embedding, token, i), c.embedding_scale);
        norm(h, w.h_norm.data(), cat.data(), H);
        norm(e.data(), w.e_norm.data(), cat.data() + H, H);
        if (c.concat) linear(w.in_proj.view(), cat.data(), 1, x);
        else for (std::uint32_t i = 0; i < H; ++i) x[i] = cat[i] + cat[H + i];
    }
    // a = RMSNorm(x); k/v heads at position p (k: norm+RoPE, v: unscaled norm)
    void kv(const float* x, std::uint64_t p, float* a, float* k, float* v) const {
        norm(x, w.input_norm.data(), a, c.hidden);
        linear(w.k.view(), a, 1, k);
        linear(w.v.view(), a, 1, v);
        for (std::uint32_t h = 0; h < c.kv_heads; ++h) {
            head_norm(k + std::size_t(h) * c.head_dim, w.k_norm.data(), c.head_dim, rope_row(p));
            head_norm(v + std::size_t(h) * c.head_dim, nullptr, c.head_dim, nullptr);
        }
    }
    void query(const float* a, std::uint64_t p, float* q) const {
        linear(w.q.view(), a, 1, q);
        for (std::uint32_t h = 0; h < c.heads; ++h) head_norm(q + std::size_t(h) * c.head_dim, qn.data(), c.head_dim, rope_row(p));
    }
    // Keys at absolute positions [max(0, p-511), p]: <= r from the committed ring, > r from the chain overlay.
    void attend(const float* q, std::uint64_t p, std::uint64_t r, const std::vector<float>& ok,
                const std::vector<float>& ov, float* out) const {
        const auto d = c.head_dim, kw = c.kv_width();
        const auto first = p < c.window ? 0 : p - c.window + 1;
        auto at = [&](std::uint64_t j, bool value) -> const float* {
            if (j > r) {
                require(j - r - 1 < max_depth, "chain overlay position");
                return (value ? ov : ok).data() + std::size_t(j - r - 1) * kw;
            }
            require(tags[j % c.window] == j, "incomplete drafter context tail");
            return (value ? ring_v : ring_k).data() + std::size_t(j % c.window) * kw;
        };
        for (std::uint32_t h = 0; h < c.heads; ++h) {
            const auto g = h / 4U;
            std::vector<float> scores(std::size_t(p - first + 1));
            float maximum = -std::numeric_limits<float>::infinity();
            for (auto j = first; j <= p; ++j) {
                const float s = lane_dot(q + std::size_t(h) * d, at(j, false) + std::size_t(g) * d, d);
                require(finite_bits(s), "nonfinite attention score");
                scores[std::size_t(j - first)] = s;
                maximum = std::max(maximum, s);
            }
            for (auto& s : scores) s = declared_exp(s - maximum);
            float denominator = 0.0F;
            std::vector<float> numerator(d, 0.0F);
            for (auto chunk = first / attention_chunk; chunk <= p / attention_chunk; ++chunk) {
                const auto b = std::max<std::uint64_t>(first, chunk * attention_chunk);
                const auto e = std::min<std::uint64_t>(p + 1, (chunk + 1) * attention_chunk);
                float part = 0.0F;
                for (auto j = b; j < e; ++j) part = part + scores[std::size_t(j - first)];
                denominator = denominator + part;
                for (std::uint32_t i = 0; i < d; ++i) {
                    float acc = 0.0F;
                    for (auto j = b; j < e; ++j) acc = acc + scores[std::size_t(j - first)] * at(j, true)[std::size_t(g) * d + i];
                    numerator[i] = numerator[i] + acc;
                }
            }
            for (std::uint32_t i = 0; i < d; ++i) out[std::size_t(h) * d + i] = numerator[i] / denominator;
        }
        finite({out, c.q_width()}, "attention output");
    }
    // Residual epilogues, FFN, final norm, head and selection. resid is updated in place to the depth state s.
    std::pair<std::uint32_t, float> tail(float* resid, const float* att) const {
        const auto H = c.hidden;
        std::vector<float> out(H), b(H), y(H), g(c.ffn), u(c.ffn), logits(c.vocab);
        linear(w.o.view(), att, 1, out.data());
        norm(out.data(), w.post_attn_norm.data(), b.data(), H);
        for (std::uint32_t i = 0; i < H; ++i) resid[i] = resid[i] + b[i];
        finite({resid, H}, "residual");
        norm(resid, w.pre_ffn_norm.data(), y.data(), H);
        linear(w.gate.view(), y.data(), 1, g.data());
        linear(w.up.view(), y.data(), 1, u.data());
        for (std::uint32_t i = 0; i < c.ffn; ++i) g[i] = gelu_tanh(g[i]) * u[i];
        finite(g, "GeGLU");
        linear(w.down.view(), g.data(), 1, out.data());
        norm(out.data(), w.post_ffn_norm.data(), b.data(), H);
        for (std::uint32_t i = 0; i < H; ++i) resid[i] = resid[i] + b[i];
        finite({resid, H}, "residual");
        norm(resid, w.final_norm.data(), y.data(), H);
        linear(w.head.view(), y.data(), 1, logits.data());
        return select(logits, c.softcap);
    }
};

HostDrafter::HostDrafter(const Weights& weights, RopeView rope, LinearView target_embedding)
    : impl_(std::make_unique<Impl>(weights, rope, target_embedding)) {}
HostDrafter::~HostDrafter() = default;
const Config& HostDrafter::config() const { return impl_->c; }
std::uint64_t HostDrafter::frontier() const { return impl_->frontier; }
const Timings& HostDrafter::timings() const { return impl_->times; }
void HostDrafter::reset() {
    auto& d = *impl_;
    std::fill(d.tags.begin(), d.tags.end(), UINT64_MAX);
    std::fill(d.ring_k.begin(), d.ring_k.end(), 0.0F); std::fill(d.ring_v.begin(), d.ring_v.end(), 0.0F);
    d.frontier = 0; d.initialized = false; d.has_last = false; d.staged = false; d.times = {};
}
void HostDrafter::stage(const Rows& rows) {
    auto& d = *impl_;
    const auto start = Clock::now();
    const auto n = rows.positions.size();
    const auto H = d.c.hidden, kw = d.c.kv_width();
    require(!d.staged, "previous stage not published/discarded");
    require(n >= 1 && n <= max_rows && rows.taps.size() == n * engine_taps * H && rows.next_tokens.size() == n &&
            (d.c.needs_final() ? rows.hidden.size() == n * H : true), "stage geometry");
    require(!d.initialized || rows.positions[0] == d.frontier, "duplicate, gap or rollback in drafter stage");
    for (std::size_t r = 0; r < n; ++r)
        require(rows.positions[r] == rows.positions[0] + r && rows.positions[r] < d.rope.positions, "stage positions");
    finite(rows.taps, "taps"); finite(rows.hidden, "hidden rows");
    d.sk.assign(n * kw, 0.0F); d.sv = d.sk; d.sx.assign(H, 0.0F); d.sa = d.sx;
    std::vector<float> h(H), x(H), a(H);
    for (std::size_t r = 0; r < n; ++r) {
        d.fuse(rows.taps.data() + r * engine_taps * H, d.c.needs_final() ? rows.hidden.data() + r * H : nullptr, h.data());
        d.combine(h.data(), rows.next_tokens[r], x.data());
        d.kv(x.data(), rows.positions[r], a.data(), d.sk.data() + r * kw, d.sv.data() + r * kw);
        if (r + 1 == n) { d.sx = x; d.sa = a; }
    }
    d.spos.assign(rows.positions.begin(), rows.positions.end());
    d.snext = rows.next_tokens[n - 1];
    d.staged = true;
    d.times.stage_ms = elapsed(start);
}
void HostDrafter::publish() {
    auto& d = *impl_;
    require(d.staged, "no staged rows");
    const auto kw = d.c.kv_width();
    for (std::size_t r = 0; r < d.spos.size(); ++r) {
        const auto slot = std::size_t(d.spos[r] % d.c.window);
        std::copy_n(d.sk.data() + r * kw, kw, d.ring_k.data() + slot * kw);
        std::copy_n(d.sv.data() + r * kw, kw, d.ring_v.data() + slot * kw);
        d.tags[slot] = d.spos[r];
    }
    d.last_x = d.sx; d.last_a = d.sa; d.last_next = d.snext; d.has_last = true;
    d.frontier = d.spos.back() + 1; d.initialized = true; d.staged = false;
}
void HostDrafter::discard() noexcept { impl_->staged = false; }
Draft HostDrafter::draft(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) {
    auto& d = *impl_;
    const auto start = Clock::now();
    const auto H = d.c.hidden, kw = d.c.kv_width();
    require(!d.staged && d.has_last && position == d.frontier && position > 0, "draft position must equal the committed frontier");
    require(anchor == d.last_next, "anchor differs from the last committed row's next token");
    require(k >= 1 && k <= max_depth && position + k - 1 < d.rope.positions && position + k <= gemma4::max_positions, "draft depth");
    const auto r = position - 1;
    Draft out;
    std::vector<float> resid = d.last_x, a(H), q(d.c.q_width()), att(d.c.q_width()), ok(std::size_t(max_depth) * kw, 0.0F),
                       ov(ok.size(), 0.0F), x(H);
    // depth 1: committed row r (target features, already in the ring)
    d.query(d.last_a.data(), r, q.data());
    d.attend(q.data(), r, r, ok, ov, att.data());
    auto [id, prob] = d.tail(resid.data(), att.data());
    out.draft_ids[0] = id; out.tokens[0] = d.w.d2t[id]; out.prob[0] = prob;
    for (std::uint32_t depth = 2; depth <= k; ++depth) {
        const auto p = r + depth - 1;
        d.combine(resid.data(), out.tokens[depth - 2], x.data());
        resid = x;
        d.kv(x.data(), p, a.data(), ok.data() + std::size_t(depth - 2) * kw, ov.data() + std::size_t(depth - 2) * kw);
        d.query(a.data(), p, q.data());
        d.attend(q.data(), p, r, ok, ov, att.data());
        std::tie(id, prob) = d.tail(resid.data(), att.data());
        out.draft_ids[depth - 1] = id; out.tokens[depth - 1] = d.w.d2t[id]; out.prob[depth - 1] = prob;
    }
    out.depth = k;
    d.times.draft_ms = elapsed(start);
    return out;
}
std::vector<float> HostDrafter::key(std::uint64_t position) const {
    const auto& d = *impl_;
    const auto slot = std::size_t(position % d.c.window);
    if (d.tags[slot] != position) return {};
    return {d.ring_k.begin() + std::ptrdiff_t(slot * d.c.kv_width()), d.ring_k.begin() + std::ptrdiff_t((slot + 1) * d.c.kv_width())};
}
std::vector<float> HostDrafter::value(std::uint64_t position) const {
    const auto& d = *impl_;
    const auto slot = std::size_t(position % d.c.window);
    if (d.tags[slot] != position) return {};
    return {d.ring_v.begin() + std::ptrdiff_t(slot * d.c.kv_width()), d.ring_v.begin() + std::ptrdiff_t((slot + 1) * d.c.kv_width())};
}

// ------------------------------------------------------------------ cycle helpers
namespace {
std::vector<std::uint32_t> greedy_rows(const ForwardResult& f, std::size_t rows) {
    std::vector<std::uint32_t> greedy(rows, 0);
    for (std::size_t r = 0; r < rows; ++r) {
        float best = -std::numeric_limits<float>::infinity();
        for (std::uint32_t id = 0; id < f.vocab; ++id) {
            const float x = f.logits[r * f.vocab + id];
            require(finite_bits(x), "nonfinite target logit");
            if (better(x, id, best, greedy[r])) { best = x; greedy[r] = id; }
        }
    }
    return greedy;
}
void check_forward(const ForwardResult& f, std::size_t rows, std::uint32_t hidden, bool want_hidden) {
    require(f.rows == rows && f.vocab > 0 && f.logits.size() == rows * f.vocab && f.tap_width == engine_taps * hidden &&
            f.taps.size() == rows * f.tap_width && (!want_hidden || f.hidden.size() == rows * hidden),
            "invalid target forward result");
    finite(f.taps, "target taps");
}
} // namespace

CycleResult run_cycle(Target& target, KvTxn& transaction, Drafter& drafter, std::uint32_t anchor, std::uint32_t k,
                      std::size_t budget, std::span<const std::uint32_t> eos, dflash::Acceptance* stats,
                      std::uint64_t stream, std::uint64_t generation) {
    CycleResult result;
    result.next_anchor = anchor;
    if (!budget) return result;
    const auto& c = drafter.config();
    const auto position = transaction.frontier();
    require(drafter.frontier() == position, "target/draft frontier mismatch");
    k = dflash::admit_window(std::min(k, max_depth), position, budget, enabled_from_environment(), false);
    if (position == 0) k = 0; // no committed row yet (one-token prompt): plain target step
    result.k = k;
    if (k) {
        const auto start = Clock::now();
        result.draft = drafter.draft(anchor, position, k);
        result.draft_ms = elapsed(start);
    }
    const bool hidden = c.needs_final();
    std::vector<RowPlan> plan(k + 1);
    for (std::uint32_t r = 0; r <= k; ++r)
        plan[r] = RowPlan{stream, generation, position + r, r ? result.draft.tokens[r - 1] : anchor, &transaction, r, true, hidden};
    auto start = Clock::now();
    const auto forward = target.forward_rows(plan);
    result.verify_ms = elapsed(start);
    check_forward(forward, plan.size(), c.hidden, hidden);
    const auto greedy = greedy_rows(forward, plan.size());
    dflash::Acceptance staged = stats ? *stats : dflash::Acceptance{};
    result.verdict = dflash::verify_greedy(std::span<const std::uint32_t>(result.draft.tokens).first(k), greedy, budget, eos, &staged);
    const auto processed = std::size_t(result.verdict.processed);
    std::vector<std::uint64_t> positions(processed);
    std::iota(positions.begin(), positions.end(), position);
    const Rows rows{std::span(forward.taps).first(processed * forward.tap_width),
                    hidden ? std::span(forward.hidden).first(processed * c.hidden) : std::span<const float>{},
                    std::span<const std::uint32_t>(result.verdict.emitted).first(processed), positions};
    start = Clock::now();
    drafter.stage(rows);
    result.stage_ms = elapsed(start);
    start = Clock::now();
    try {
        transaction.commit(processed);
        drafter.publish();
    } catch (...) { drafter.discard(); throw; }
    result.commit_ms = elapsed(start);
    if (stats) *stats = staged;
    result.next_anchor = result.verdict.emitted.back();
    return result;
}

std::uint32_t prefill(Target& target, const std::function<std::unique_ptr<KvTxn>()>& begin, Drafter& drafter,
                      std::span<const std::uint32_t> prompt, std::uint64_t stream, std::uint64_t generation) {
    require(!prompt.empty(), "empty prompt");
    const auto& c = drafter.config();
    const bool hidden = c.needs_final();
    std::size_t done = 0;
    const std::size_t n = prompt.size() - 1;
    while (done < n) {
        auto tx = begin();
        const auto position = tx->frontier();
        const auto rows = std::min<std::size_t>(max_rows, n - done);
        std::vector<RowPlan> plan(rows);
        std::vector<std::uint64_t> positions(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            plan[r] = RowPlan{stream, generation, position + r, prompt[done + r], tx.get(), r, true, hidden};
            positions[r] = position + r;
        }
        const auto forward = target.forward_rows(plan);
        check_forward(forward, rows, c.hidden, hidden);
        drafter.stage({forward.taps, hidden ? std::span<const float>(forward.hidden) : std::span<const float>{},
                       prompt.subspan(done + 1, rows), positions});
        try {
            tx->commit(rows);
            drafter.publish();
        } catch (...) { drafter.discard(); throw; }
        done += rows;
    }
    return prompt.back();
}

} // namespace strix::gemma4::eagle
