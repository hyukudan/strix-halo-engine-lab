// Host-only DFlash loader/reference and target-owned greedy verification.
#include "dflash.hpp"
#include "gemma4/exl3/format.hpp"
#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace strix::gemma4::dflash {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("dflash: " + message);
}

void finite(std::span<const float> values) {
    for (float x : values) require(std::isfinite(x), "nonfinite tensor or execution value");
}

class Loader {
public:
    Loader(const std::filesystem::path& path, Shape shape) : input_(path, std::ios::binary) {
        shape.validate();
        gguf::Limits limits;
        limits.max_retained_array_elements = 64;
        info_ = gguf::read_file(path, limits);
        require(static_cast<bool>(input_), "cannot open checkpoint");
        weights_.shape = shape;
    }

    Weights load() {
        const Shape& s = weights_.shape;
        require(text("general.architecture") == "dflash", "unsupported architecture");
        integer("dflash.block_count", s.layers);
        integer("dflash.embedding_length", s.hidden);
        integer("dflash.feed_forward_length", s.inter);
        integer("dflash.attention.head_count", s.heads);
        integer("dflash.attention.head_count_kv", s.kv_heads);
        integer("dflash.attention.key_length", s.head_dim);
        integer("dflash.attention.value_length", s.head_dim);
        integer("dflash.block_size", block);
        integer("dflash.attention.sliding_window", window);
        require(number("dflash.rope.freq_base") == 1000000.0, "unsupported RoPE theta");
        require(static_cast<float>(number("dflash.attention.layer_norm_rms_epsilon")) == 1e-6F,
                "unsupported RMSNorm epsilon");
        const auto* causal = info_.find_metadata("dflash.attention.causal");
        require(causal && causal->boolean_value && !*causal->boolean_value, "block must be non-causal");
        const auto* pattern = info_.find_metadata("dflash.attention.sliding_window_pattern");
        require(pattern && pattern->unsigned_array.size() == s.layers, "missing sliding-window pattern");
        for (auto v : pattern->unsigned_array) require(v == 1, "all draft layers must use sliding context");
        const auto* taps = info_.find_metadata("dflash.target_layers");
        require(taps && (taps->unsigned_array.size() == tap_count || taps->signed_array.size() == tap_count),
                "missing target taps");
        for (std::size_t i = 0; i < tap_count; ++i)
            require(taps->signed_array.empty() ? taps->unsigned_array[i] == tap_layers[i] + 1 :
                    taps->signed_array[i] == static_cast<std::int64_t>(tap_layers[i] + 1), "target tap mapping mismatch");
        if (info_.find_metadata("dflash.mask_token_id")) integer("dflash.mask_token_id", mask_id);
        if (info_.find_metadata("dflash.rope.dimension_count")) integer("dflash.rope.dimension_count", s.head_dim);
        if (info_.find_metadata("dflash.rope.scaling.factor"))
            require(number("dflash.rope.scaling.factor") == 1.0, "RoPE scaling is unsupported");
        if (info_.find_metadata("dflash.attention.scale"))
            require(static_cast<float>(number("dflash.attention.scale")) == attention_scale,
                    "attention scale override is unsupported");
        weights_.checkpoint_embedding_scale = static_cast<float>(number("dflash.embedding_scale"));
        if (info_.find_metadata("dflash.logit_scale"))
            weights_.logit_scale = static_cast<float>(number("dflash.logit_scale"));
        if (info_.find_metadata("dflash.final_logit_softcapping"))
            weights_.logit_cap = static_cast<float>(number("dflash.final_logit_softcapping"));
        weights_.license = text("general.license", false);
        weights_.license_name = text("general.license.name", false);
        weights_.license_link = text("general.license.link", false);
        weights_.fc = matrix("fc", s.hidden, static_cast<std::uint32_t>(s.tap_width()));
        weights_.context_norm = norm("enc.output_norm", s.hidden);
        weights_.final_norm = norm("output_norm", s.hidden);
        weights_.layer.resize(s.layers);
        for (std::uint32_t l = 0; l < s.layers; ++l) {
            LayerWeights& w = weights_.layer[l];
            const std::string p = "blk." + std::to_string(l) + ".";
            w.q = matrix(p + "attn_q", static_cast<std::uint32_t>(s.q_width()), s.hidden);
            w.k = matrix(p + "attn_k", static_cast<std::uint32_t>(s.kv_width()), s.hidden);
            w.v = matrix(p + "attn_v", static_cast<std::uint32_t>(s.kv_width()), s.hidden);
            w.o = matrix(p + "attn_output", s.hidden, static_cast<std::uint32_t>(s.q_width()));
            w.gate = matrix(p + "ffn_gate", s.inter, s.hidden);
            w.up = matrix(p + "ffn_up", s.inter, s.hidden);
            w.down = matrix(p + "ffn_down", s.hidden, s.inter);
            w.input_norm = norm(p + "attn_norm", s.hidden);
            w.ffn_norm = norm(p + "ffn_norm", s.hidden);
            w.q_norm = norm(p + "attn_q_norm", s.head_dim);
            w.k_norm = norm(p + "attn_k_norm", s.head_dim);
        }
        require(used_.size() == info_.tensors.size(), "unexpected tensor (bias, head, embedding or unsupported layer)");
        weights_.validate();
        return std::move(weights_);
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
        if (e->unsigned_value) return static_cast<double>(*e->unsigned_value);
        if (e->signed_value) return static_cast<double>(*e->signed_value);
        throw std::runtime_error("dflash: invalid numeric metadata " + key);
    }
    void integer(const std::string& key, std::uint32_t expected) const {
        require(number(key) == expected, "metadata mismatch " + key);
    }
    const gguf::TensorInfo& tensor(const std::string& base, const std::vector<std::uint64_t>& shape) {
        const std::string name = base + ".weight";
        const auto it = std::find_if(info_.tensors.begin(), info_.tensors.end(),
                                     [&](const auto& t) { return t.name == name; });
        require(it != info_.tensors.end(), "missing tensor " + name);
        require(it->dimensions == shape, "shape mismatch " + name);
        used_.insert(name);
        weights_.tensor_bytes += it->byte_size;
        return *it;
    }
    std::vector<std::uint8_t> bytes(const gguf::TensorInfo& t) {
        std::vector<std::uint8_t> data(static_cast<std::size_t>(t.byte_size));
        input_.seekg(static_cast<std::streamoff>(t.absolute_offset));
        input_.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        require(static_cast<bool>(input_), "truncated tensor " + t.name);
        return data;
    }
    Bf16Matrix matrix(const std::string& name, std::uint32_t outputs, std::uint32_t width) {
        const auto& t = tensor(name, {width, outputs});
        require(t.ggml_type == 30, "matrix must be BF16 " + name);
        const auto data = bytes(t);
        Bf16Matrix m{outputs, width, std::vector<std::uint16_t>(std::size_t{outputs} * width)};
        for (std::size_t i = 0; i < m.bits.size(); ++i)
            m.bits[i] = static_cast<std::uint16_t>(data[2 * i] | (std::uint16_t{data[2 * i + 1]} << 8));
        return m;
    }
    std::vector<float> norm(const std::string& name, std::uint32_t width) {
        const auto& t = tensor(name, {width});
        require(t.ggml_type == 0, "norm must retain FP32 storage " + name);
        const auto data = bytes(t);
        std::vector<float> out(width);
        for (std::size_t i = 0; i < width; ++i) {
            std::uint32_t bits = 0;
            for (std::size_t b = 0; b < 4; ++b) bits |= std::uint32_t{data[4 * i + b]} << (8 * b);
            out[i] = std::bit_cast<float>(bits);
        }
        return out;
    }
    std::ifstream input_;
    gguf::FileInfo info_;
    Weights weights_;
    std::set<std::string> used_;
};

void check_matrix(const Bf16Matrix& m, std::size_t outputs, std::size_t width) {
    require(m.outputs == outputs && m.width == width && m.bits.size() == outputs * width,
            "invalid matrix dimensions");
    for (std::uint16_t b : m.bits) require((b & 0x7f80U) != 0x7f80U, "nonfinite BF16 weight");
}

void check_norm(std::span<const float> w, std::size_t width) {
    require(w.size() == width, "invalid norm dimensions");
    finite(w);
}

} // namespace

void Shape::validate() const {
    require(hidden > 0 && hidden <= 2560 && layers > 0 && layers <= 5 && heads > 0 && heads <= 32 &&
            kv_heads > 0 && heads == 4 * kv_heads && head_dim > 0 && head_dim <= 128 &&
            head_dim % 2 == 0 && inter > 0 && inter <= 5120, "invalid draft shape");
}

void Weights::validate() const {
    shape.validate();
    const Shape& s = shape;
    check_matrix(fc, s.hidden, s.tap_width());
    check_norm(context_norm, s.hidden);
    check_norm(final_norm, s.hidden);
    require(layer.size() == s.layers, "layer count mismatch");
    for (const auto& w : layer) {
        check_matrix(w.q, s.q_width(), s.hidden);
        check_matrix(w.k, s.kv_width(), s.hidden);
        check_matrix(w.v, s.kv_width(), s.hidden);
        check_matrix(w.o, s.hidden, s.q_width());
        check_matrix(w.gate, s.inter, s.hidden);
        check_matrix(w.up, s.inter, s.hidden);
        check_matrix(w.down, s.hidden, s.inter);
        check_norm(w.input_norm, s.hidden);
        check_norm(w.ffn_norm, s.hidden);
        check_norm(w.q_norm, s.head_dim);
        check_norm(w.k_norm, s.head_dim);
    }
    require(std::isfinite(logit_scale) && logit_scale > 0.0F && std::isfinite(logit_cap) && logit_cap >= 0.0F,
            "invalid draft logit policy");
    require(std::isfinite(checkpoint_embedding_scale), "invalid embedding scale metadata");
}

DraftFile DraftFile::open(const std::filesystem::path& path) { return open_fixture(path, Shape{}); }
DraftFile DraftFile::open_fixture(const std::filesystem::path& path, const Shape& shape) {
    DraftFile file;
    file.weights_ = Loader(path, shape).load();
    return file;
}

void RopeView::validate() const {
    require(positions > 0 && positions <= position_limit && head_dim > 0 && head_dim % 2 == 0 &&
            cosine.size() == std::size_t{positions} * (head_dim / 2) && sine.size() == cosine.size(),
            "invalid RoPE table dimensions");
    finite(cosine);
    finite(sine);
}

RopeFile RopeFile::open(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[4]{};
    in.read(magic, 4);
    require(std::string(magic, 4) == "G4DR", "invalid draft RoPE table magic");
    auto word = [&]() {
        std::uint32_t v = 0;
        for (unsigned i = 0; i < 4; ++i) {
            const int c = in.get();
            require(c != std::char_traits<char>::eof(), "truncated draft RoPE table");
            v |= static_cast<std::uint32_t>(c) << (8 * i);
        }
        return v;
    };
    require(word() == 1, "unsupported draft RoPE table version");
    RopeFile file;
    file.positions = word(); file.head_dim = word();
    require(file.positions > 0 && file.positions <= position_limit && file.head_dim > 0 &&
            file.head_dim <= 128 && file.head_dim % 2 == 0, "unsupported draft RoPE table dimensions");
    file.cosine.resize(std::size_t{file.positions} * (file.head_dim / 2));
    file.sine.resize(file.cosine.size());
    for (float& x : file.cosine) x = std::bit_cast<float>(word());
    for (float& x : file.sine) x = std::bit_cast<float>(word());
    require(in.get() == std::char_traits<char>::eof(), "trailing draft RoPE table bytes");
    file.view().validate();
    return file;
}

bool context_visible(std::uint64_t query, std::uint64_t anchor, std::uint64_t key) {
    return key < anchor && (query < window || key > query - window);
}

std::vector<std::uint64_t> visible_positions(std::uint64_t anchor, std::uint32_t row) {
    require(row < block && anchor <= position_limit - block, "invalid full draft block position");
    const std::uint64_t q = anchor + row;
    const std::uint64_t first = q >= window ? q - window + 1 : 0;
    std::vector<std::uint64_t> result;
    for (std::uint64_t p = first; p < anchor; ++p) result.push_back(p);
    for (std::uint64_t p = anchor; p < anchor + block; ++p) result.push_back(p);
    return result;
}

Verdict verify_greedy(std::span<const std::uint32_t> proposals, std::span<const std::uint32_t> greedy,
                      std::size_t budget, std::span<const std::uint32_t> eos, Acceptance* stats) {
    require(proposals.size() <= max_drafts && greedy.size() == proposals.size() + 1, "invalid verify window");
    Verdict v;
    while (v.emitted.size() < budget) {
        const std::size_t i = v.accepted;
        bool accepted = false;
        if (i < proposals.size()) {
            ++v.eligible;
            accepted = proposals[i] == greedy[i];
        }
        const std::uint32_t token = accepted ? proposals[i] : greedy[i];
        v.emitted.push_back(token);
        if (accepted) ++v.accepted;
        if (std::find(eos.begin(), eos.end(), token) != eos.end()) {
            v.stopped = true;
            break;
        }
        if (!accepted) break;
    }
    v.processed = static_cast<std::uint32_t>(v.emitted.size());
    if (stats) {
        ++stats->cycles;
        stats->attempted += proposals.size();
        stats->eligible += v.eligible;
        stats->accepted += v.accepted;
        stats->emitted += v.emitted.size();
        for (std::size_t d = 0; d < v.eligible; ++d) {
            ++stats->depth_eligible[d];
            if (d < v.accepted) ++stats->depth_accepted[d];
            else ++stats->rejection_depth[d];
        }
        if (!proposals.empty() && v.accepted == proposals.size()) ++stats->all_accepted;
    }
    return v;
}

std::uint32_t admit_window(std::uint32_t requested, std::uint64_t position, std::size_t budget,
                           bool enabled, bool serial_policy, double conservative_speedup, bool automatic) {
    require(requested <= max_drafts && position < position_limit, "invalid window request");
    if (!enabled || serial_policy || budget <= 1 || position > position_limit - block) return 0;
    if (automatic && (!std::isfinite(conservative_speedup) || conservative_speedup < 1.05)) return 0;
    return static_cast<std::uint32_t>(std::min<std::size_t>(requested, budget - 1));
}

bool enabled_from_environment() {
    const char* value = std::getenv("STRIX_G4_DFLASH");
    return !value || std::string(value) != "0";
}

namespace {

void linear(const Bf16Matrix& w, std::span<const float> x, std::size_t rows, std::span<float> out) {
    require(x.size() == rows * w.width && out.size() == rows * w.outputs &&
            w.bits.size() == std::size_t{w.width} * w.outputs, "linear shape mismatch");
    for (std::uint32_t o = 0; o < w.outputs; ++o) {
        const auto* row = w.bits.data() + std::size_t{o} * w.width;
        for (std::size_t r = 0; r < rows; ++r)
            out[r * w.outputs + o] = gemma4::chunk16_dot(
                [&](std::uint32_t i) { return gemma4::bf16_value(row[i]); }, x.data() + r * w.width, w.width);
    }
    finite(out);
}

void norm_rows(std::span<const float> x, std::span<const float> w, std::span<float> out) {
    require(!w.empty() && x.size() % w.size() == 0 && out.size() == x.size(), "norm shape mismatch");
    for (std::size_t r = 0; r < x.size() / w.size(); ++r) {
        const float* row = x.data() + r * w.size();
        require(std::isfinite(gemma4::lane_dot(row, row, static_cast<std::uint32_t>(w.size()))), "nonfinite norm reduction");
        gemma4::rms_norm(x.data() + r * w.size(), w.data(), out.data() + r * w.size(), static_cast<std::uint32_t>(w.size()));
    }
    finite(out);
}

void norm_rope(std::span<float> values, std::span<const float> weight, RopeView rope, std::uint64_t position) {
    require(position < rope.positions && weight.size() == rope.head_dim && values.size() % rope.head_dim == 0,
            "head norm/RoPE shape mismatch");
    const std::size_t half = rope.head_dim / 2;
    for (std::size_t h = 0; h < values.size() / rope.head_dim; ++h) {
        float* v = values.data() + h * rope.head_dim;
        require(std::isfinite(gemma4::lane_dot(v, v, rope.head_dim)), "nonfinite head norm reduction");
        gemma4::rms_norm(v, weight.data(), v, rope.head_dim);
        gemma4::apply_rope(v, rope.head_dim, static_cast<std::uint32_t>(half),
                           rope.cosine.data() + position * half, rope.sine.data() + position * half);
    }
    finite(values);
}

std::size_t row_bytes(WeightFormat format, std::uint32_t width) {
    switch (format) {
    case WeightFormat::bf16: return std::size_t{width} * 2;
    case WeightFormat::f32: return std::size_t{width} * 4;
    case WeightFormat::q8_0: require(width % 32 == 0, "Q8 row alignment"); return std::size_t{width / 32} * 34;
    case WeightFormat::q4_k: require(width % 256 == 0, "Q4 row alignment"); return std::size_t{width / 256} * 144;
    case WeightFormat::q6_k: require(width % 256 == 0, "Q6 row alignment"); return std::size_t{width / 256} * 210;
    case WeightFormat::exl3_mul1: break;
    }
    throw std::runtime_error("dflash: unsupported target weight format");
}

void check_view(LinearView view) {
    if (view.format == WeightFormat::exl3_mul1) { exl3::validate(view); return; }
    require(view.data && view.inputs > 0 && view.outputs > 0 && view.row_bytes >= row_bytes(view.format, view.inputs),
            "invalid target linear view");
}

float coefficient(LinearView view, std::uint32_t row, std::uint32_t i) {
    const auto* p = view.data + std::size_t{row} * view.row_bytes;
    switch (view.format) {
    case WeightFormat::bf16: return gemma4::bf16_value(gemma4::load16(p + 2 * i));
    case WeightFormat::f32: {
        p += 4 * i;
        return std::bit_cast<float>(std::uint32_t{p[0]} | (std::uint32_t{p[1]} << 8) |
                                    (std::uint32_t{p[2]} << 16) | (std::uint32_t{p[3]} << 24));
    }
    case WeightFormat::q8_0: return gemma4::q8_0_value(p + std::size_t{i / 32} * 34, i % 32);
    case WeightFormat::q4_k: return gemma4::q4_k_value(p + std::size_t{i / 256} * 144, i % 256);
    case WeightFormat::q6_k: return gemma4::q6_k_value(p + std::size_t{i / 256} * 210, i % 256);
    case WeightFormat::exl3_mul1: break;
    }
    throw std::runtime_error("dflash: unsupported coefficient format");
}

} // namespace

DraftContext::DraftContext(Shape shape) : shape_(shape) {
    shape_.validate();
    k_.resize(std::size_t{shape.layers} * window * shape.kv_width());
    v_.resize(k_.size());
    tags_.fill(-1);
}

bool DraftContext::contains(std::uint64_t position) const {
    return position < position_limit && tags_[position % window] == static_cast<std::int64_t>(position);
}
std::span<const float> DraftContext::key(std::uint32_t layer, std::uint64_t position) const {
    require(layer < shape_.layers && contains(position), "missing committed draft key");
    return std::span<const float>(k_).subspan((std::size_t{layer} * window + position % window) * shape_.kv_width(), shape_.kv_width());
}
std::span<const float> DraftContext::value(std::uint32_t layer, std::uint64_t position) const {
    require(layer < shape_.layers && contains(position), "missing committed draft value");
    return std::span<const float>(v_).subspan((std::size_t{layer} * window + position % window) * shape_.kv_width(), shape_.kv_width());
}

void DraftContext::append(const Weights& w, RopeView rope, std::span<const float> taps,
                           std::span<const std::uint64_t> positions) {
    require(w.shape == shape_ && w.layer.size() == shape_.layers && rope.head_dim == shape_.head_dim &&
            taps.size() == positions.size() * shape_.tap_width(), "context append shape mismatch");
    if (positions.empty()) return;
    rope.validate();
    finite(taps);
    require(!initialized_ || positions.front() == frontier_, "duplicate, gap or rollback in committed context");
    for (std::size_t i = 0; i < positions.size(); ++i)
        require(positions[i] < position_limit && positions[i] < rope.positions &&
                positions[i] == positions.front() + i, "invalid committed positions");
    // Only the recent prefill tail is needed; staging all retained layers makes failure atomic.
    const std::size_t skip = positions.size() > window ? positions.size() - window : 0;
    positions = positions.subspan(skip);
    taps = taps.subspan(skip * shape_.tap_width());
    const std::size_t n = positions.size(), kw = shape_.kv_width();
    std::vector<float> projected(n * shape_.hidden), c(projected.size());
    linear(w.fc, taps, n, projected);
    norm_rows(projected, w.context_norm, c);
    std::vector<float> keys(std::size_t{shape_.layers} * n * kw), values(keys.size());
    for (std::uint32_t l = 0; l < shape_.layers; ++l) {
        auto kl = std::span<float>(keys).subspan(std::size_t{l} * n * kw, n * kw);
        auto vl = std::span<float>(values).subspan(std::size_t{l} * n * kw, n * kw);
        linear(w.layer[l].k, c, n, kl);
        linear(w.layer[l].v, c, n, vl); // deliberately no V norm
        for (std::size_t r = 0; r < n; ++r) norm_rope(kl.subspan(r * kw, kw), w.layer[l].k_norm, rope, positions[r]);
    }
    for (std::uint32_t l = 0; l < shape_.layers; ++l)
        for (std::size_t r = 0; r < n; ++r) {
            const std::size_t source = (std::size_t{l} * n + r) * kw;
            const std::size_t dest = (std::size_t{l} * window + positions[r] % window) * kw;
            std::copy_n(keys.data() + source, kw, k_.data() + dest);
            std::copy_n(values.data() + source, kw, v_.data() + dest);
        }
    for (auto p : positions) tags_[p % window] = static_cast<std::int64_t>(p);
    frontier_ = positions.back() + 1;
    initialized_ = true;
}

std::vector<float> block_forward(const Weights& w, const DraftContext& context, RopeView rope,
                                 std::span<const float> embedded, std::uint64_t position,
                                 ForwardDiagnostics* diagnostics) {
    const Shape& s = w.shape;
    require(context.shape() == s && context.frontier() == position && w.layer.size() == s.layers &&
            embedded.size() == std::size_t{block} * s.hidden && position <= position_limit - block &&
            rope.head_dim == s.head_dim && position + block <= rope.positions, "invalid draft block");
    rope.validate();
    finite(embedded);
    if (diagnostics) diagnostics->gates.clear();
    const auto all_positions = visible_positions(position, 0);
    for (auto p : all_positions) if (p < position) require(context.contains(p), "incomplete committed context tail");
    const std::size_t qw = s.q_width(), kw = s.kv_width();
    std::vector<float> residual(embedded.begin(), embedded.end()), x(residual.size()), output(residual.size()),
        q(block * qw), kb(block * kw), vb(block * kw), att(block * qw), gate(std::size_t{block} * s.inter), up(gate.size());
    for (std::uint32_t l = 0; l < s.layers; ++l) {
        const auto& lw = w.layer[l];
        norm_rows(residual, lw.input_norm, x);
        linear(lw.q, x, block, q); linear(lw.k, x, block, kb); linear(lw.v, x, block, vb);
        for (std::uint32_t r = 0; r < block; ++r) {
            norm_rope(std::span<float>(q).subspan(r * qw, qw), lw.q_norm, rope, position + r);
            norm_rope(std::span<float>(kb).subspan(r * kw, kw), lw.k_norm, rope, position + r);
        }
        for (std::uint32_t r = 0; r < block; ++r) {
            const auto visible = visible_positions(position, r);
            auto key = [&](std::uint64_t p) {
                return p < position ? context.key(l, p) : std::span<const float>(kb).subspan((p - position) * kw, kw);
            };
            auto value = [&](std::uint64_t p) {
                return p < position ? context.value(l, p) : std::span<const float>(vb).subspan((p - position) * kw, kw);
            };
            for (std::uint32_t h = 0; h < s.heads; ++h) {
                const std::size_t g = h / 4, hd = s.head_dim;
                const float* query = q.data() + r * qw + h * hd;
                std::vector<float> scores(visible.size());
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t j = 0; j < visible.size(); ++j) {
                    scores[j] = gemma4::lane_dot(query, key(visible[j]).data() + g * hd, s.head_dim) * attention_scale;
                    maximum = std::max(maximum, scores[j]);
                }
                finite(scores);
                for (float& score : scores) {
                    const float delta = score - maximum;
                    require(std::isfinite(delta), "nonfinite softmax difference");
                    score = gemma4::declared_exp(delta);
                }
                std::vector<float> numerator(hd, 0.0F), partial(hd, 0.0F);
                float denominator = 0.0F, part = 0.0F;
                std::uint64_t chunk = visible.front() / gemma4::attention_chunk;
                auto merge = [&] {
                    denominator = denominator + part;
                    for (std::size_t d = 0; d < hd; ++d) numerator[d] = numerator[d] + partial[d];
                    part = 0.0F;
                    std::fill(partial.begin(), partial.end(), 0.0F);
                };
                for (std::size_t j = 0; j < visible.size(); ++j) {
                    const auto next_chunk = visible[j] / gemma4::attention_chunk;
                    if (next_chunk != chunk) { merge(); chunk = next_chunk; }
                    part = part + scores[j];
                    const float* v = value(visible[j]).data() + g * hd;
                    for (std::size_t d = 0; d < hd; ++d) partial[d] = partial[d] + scores[j] * v[d];
                }
                merge();
                for (std::size_t d = 0; d < hd; ++d) att[r * qw + h * hd + d] = numerator[d] / denominator;
            }
        }
        finite(att);
        linear(lw.o, att, block, output);
        for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = residual[i] + output[i];
        norm_rows(residual, lw.ffn_norm, x);
        linear(lw.gate, x, block, gate); linear(lw.up, x, block, up);
        if (diagnostics) {
            const auto range = std::minmax_element(gate.begin(), gate.end());
            diagnostics->gates.push_back({*range.first, *range.second,
                static_cast<std::size_t>(std::count_if(gate.begin(), gate.end(),
                    [](float value) { return value < -88.0F; }))});
        }
        for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = declared_silu(gate[i]) * up[i];
        finite(gate);
        linear(lw.down, gate, block, output);
        for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = residual[i] + output[i];
        finite(residual);
    }
    norm_rows(residual, w.final_norm, output);
    return output;
}

std::vector<float> embed_block(std::uint32_t anchor, LinearView table) {
    check_view(table);
    require(anchor < table.outputs && mask_id < table.outputs, "embedding token out of range");
    std::vector<float> result(std::size_t{block} * table.inputs);
    for (std::uint32_t r = 0; r < block; ++r)
        for (std::uint32_t i = 0; i < table.inputs; ++i)
            result[std::size_t{r} * table.inputs + i] = gemma4::embedding_value(coefficient(table, r == 0 ? anchor : mask_id, i));
    finite(result);
    return result;
}

std::array<std::uint32_t, max_drafts> proposals_from_hidden(const Weights& w,
    std::span<const float> normalized_block, LinearView head) {
    check_view(head);
    require(std::isfinite(w.logit_scale) && w.logit_scale > 0.0F && std::isfinite(w.logit_cap) && w.logit_cap >= 0.0F,
            "invalid draft head policy");
    require(head.inputs == w.shape.hidden && normalized_block.size() == std::size_t{block} * head.inputs,
            "target head shape mismatch");
    finite(normalized_block);
    std::array<std::uint32_t, max_drafts> result{};
    std::array<float, max_drafts> best;
    best.fill(-std::numeric_limits<float>::infinity());
    if (head.format == WeightFormat::exl3_mul1) {
        std::vector<float> logits(std::size_t(max_drafts) * head.outputs);
        exl3::linear(head, normalized_block.data() + head.inputs, max_drafts, logits.data());
        for (unsigned d = 0; d < max_drafts; ++d)
            for (unsigned token = 0; token < head.outputs; ++token) {
                float logit = logits[std::size_t(d) * head.outputs + token] * w.logit_scale;
                require(std::isfinite(logit), "nonfinite EXL3 draft logit");
                if (w.logit_cap > 0.0F) logit = gemma4::declared_tanh(logit / w.logit_cap) * w.logit_cap;
                if (gemma4::better(logit, token, best[d], result[d])) { best[d] = logit; result[d] = token; }
            }
        return result;
    }
    for (std::uint32_t token = 0; token < head.outputs; ++token) {
        std::vector<float> decoded(head.inputs);
        for (std::uint32_t i = 0; i < head.inputs; ++i) decoded[i] = coefficient(head, token, i);
        finite(decoded);
        for (std::uint32_t d = 0; d < max_drafts; ++d) {
            float logit = gemma4::chunk16_dot([&](std::uint32_t i) { return decoded[i]; },
                normalized_block.data() + std::size_t{d + 1} * head.inputs, head.inputs) * w.logit_scale;
            require(std::isfinite(logit), "nonfinite draft logit");
            if (w.logit_cap > 0.0F) {
                const float scaled = logit / w.logit_cap;
                require(std::isfinite(scaled), "nonfinite draft cap input");
                logit = gemma4::declared_tanh(scaled) * w.logit_cap;
            }
            if (gemma4::better(logit, token, best[d], result[d])) { best[d] = logit; result[d] = token; }
        }
    }
    return result;
}

CycleResult run_cycle(Target& target, KvTxn& transaction, std::uint32_t anchor, std::uint32_t k,
    const DraftCallback& draft, std::size_t budget, std::span<const std::uint32_t> eos, Acceptance* stats,
    std::uint64_t stream, std::uint64_t generation) {
    CycleResult result;
    result.next_anchor = anchor;
    if (budget == 0) return result;
    const std::uint64_t position = transaction.frontier();
    k = admit_window(k, position, budget, enabled_from_environment(), false);
    std::array<std::uint32_t, max_drafts> tokens{};
    if (k > 0) { require(static_cast<bool>(draft), "missing draft callback"); tokens = draft(); }
    std::vector<RowPlan> plan(k + 1);
    for (std::uint32_t r = 0; r <= k; ++r)
        plan[r] = RowPlan{stream, generation, position + r, r == 0 ? anchor : tokens[r - 1], &transaction, r, true};
    const ForwardResult forward = target.forward_rows(plan);
    require(forward.rows == plan.size() && forward.vocab > 0 &&
            forward.logits.size() == forward.rows * forward.vocab &&
            forward.taps.size() == forward.rows * forward.tap_width, "invalid target forward result");
    finite(forward.logits); finite(forward.taps);
    std::vector<std::uint32_t> greedy(plan.size());
    for (std::size_t r = 0; r < plan.size(); ++r) {
        float best = -std::numeric_limits<float>::infinity();
        for (std::uint32_t id = 0; id < forward.vocab; ++id) {
            const float x = forward.logits[r * forward.vocab + id];
            if (gemma4::better(x, id, best, greedy[r])) { best = x; greedy[r] = id; }
        }
    }
    result.verdict = verify_greedy(std::span<const std::uint32_t>(tokens).first(k), greedy, budget, eos, stats);
    result.committed_taps.assign(forward.taps.begin(), forward.taps.begin() + std::size_t{result.verdict.processed} * forward.tap_width);
    for (std::uint32_t r = 0; r < result.verdict.processed; ++r) result.committed_positions.push_back(position + r);
    result.next_anchor = result.verdict.emitted.back();
    transaction.commit(result.verdict.processed);
    return result;
}

} // namespace strix::gemma4::dflash
