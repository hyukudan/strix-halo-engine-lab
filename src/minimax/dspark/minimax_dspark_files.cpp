// MiniMax DSpark drafter: the draft checkpoint (safetensors) read into host memory and checked against the expected
// manifest, and row reads of the target's BF16 embedding.
#include "minimax_dspark.hpp"
#include "strix/minimax/model_assets.hpp"

#include <cstring>
#include <fstream>
#include "strix/minimax/asset_input.hpp"
#include <map>
#include <stdexcept>
#include <variant>

namespace strix::minimax::dspark {
namespace {

// ---- minimal JSON (safetensors headers and the index) ----
struct Json;
using JsonObject = std::map<std::string, Json>;
using JsonArray = std::vector<Json>;
struct Json {
    std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject> v;
    [[nodiscard]] const JsonObject& object() const { return std::get<JsonObject>(v); }
    [[nodiscard]] const JsonArray& array() const { return std::get<JsonArray>(v); }
    [[nodiscard]] const std::string& string() const { return std::get<std::string>(v); }
    [[nodiscard]] double number() const { return std::get<double>(v); }
};

struct Parser {
    const char *p, *end;
    void ws() {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }
    [[noreturn]] void fail() const { throw std::runtime_error("dspark: malformed JSON"); }
    std::string str() {
        if (*p++ != '"') fail();
        std::string s;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                ++p;
                if (p >= end) fail();
                switch (*p) {
                case 'n': s += '\n'; break;
                case 't': s += '\t'; break;
                case 'u': s += '?'; p += 4; break;  // names and dtypes are ASCII; escapes only in metadata
                default: s += *p;
                }
                ++p;
            } else {
                s += *p++;
            }
        }
        if (p >= end) fail();
        ++p;
        return s;
    }
    Json value() {
        ws();
        if (p >= end) fail();
        if (*p == '{') {
            ++p;
            JsonObject o;
            ws();
            if (*p == '}') { ++p; return {o}; }
            for (;;) {
                ws();
                std::string k = str();
                ws();
                if (*p++ != ':') fail();
                o.emplace(std::move(k), value());
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != '}') fail();
                return {o};
            }
        }
        if (*p == '[') {
            ++p;
            JsonArray a;
            ws();
            if (*p == ']') { ++p; return {a}; }
            for (;;) {
                a.push_back(value());
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != ']') fail();
                return {a};
            }
        }
        if (*p == '"') return {str()};
        if (std::strncmp(p, "true", 4) == 0) { p += 4; return {true}; }
        if (std::strncmp(p, "false", 5) == 0) { p += 5; return {false}; }
        if (std::strncmp(p, "null", 4) == 0) { p += 4; return {nullptr}; }
        char* stop = nullptr;
        const double d = std::strtod(p, &stop);
        if (stop == p) fail();
        p = stop;
        return {d};
    }
};
Json parse(const std::string& text) {
    Parser ps{text.data(), text.data() + text.size()};
    return ps.value();
}

struct TensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0, bytes = 0;  // absolute in the file
};
std::map<std::string, TensorInfo> read_header(const std::filesystem::path& file, LoaderSnapshot* snapshot = nullptr) {
    assets::Input f(file, snapshot);
    if (!f) throw std::runtime_error("dspark: cannot open " + file.string());
    std::uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    if (!f || n > (64U << 20U)) throw std::runtime_error("dspark: bad safetensors header in " + file.string());
    std::string head(n, '\0');
    f.read(head.data(), static_cast<std::streamsize>(n));
    std::map<std::string, TensorInfo> out;
    const Json parsed = parse(head);
    for (const auto& [name, v] : parsed.object()) {
        if (name == "__metadata__") continue;
        TensorInfo t;
        t.dtype = v.object().at("dtype").string();
        for (const auto& s : v.object().at("shape").array()) t.shape.push_back(static_cast<std::uint64_t>(s.number()));
        const auto& off = v.object().at("data_offsets").array();
        t.offset = 8 + n + static_cast<std::uint64_t>(off[0].number());
        t.bytes = static_cast<std::uint64_t>(off[1].number() - off[0].number());
        out.emplace(name, std::move(t));
    }
    return out;
}

// Expected manifest: name suffix (after checkpoint_prefix) -> dtype and shape.
struct Expect {
    std::string dtype;
    std::vector<std::uint64_t> shape;
};
std::map<std::string, Expect> manifest() {
    std::map<std::string, Expect> m;
    m["fc.weight"] = {"BF16", {hidden, tap_width}};
    m["hidden_norm.weight"] = {"BF16", {hidden}};
    m["final_norm.weight"] = {"BF16", {hidden}};
    m["markov_head.markov_w1.weight"] = {"BF16", {vocab, markov_rank}};
    m["markov_head.markov_w2.weight"] = {"BF16", {vocab, markov_rank}};
    m["confidence_head.proj.weight"] = {"BF16", {1, confidence_width}};
    m["confidence_head.proj.bias"] = {"BF16", {1}};
    const std::pair<const char*, std::array<std::uint64_t, 2>> linears[] = {
        {"self_attn.q_proj", {q_width, hidden}}, {"self_attn.k_proj", {kv_width, hidden}},
        {"self_attn.v_proj", {kv_width, hidden}}, {"self_attn.o_proj", {hidden, q_width}},
        {"mlp.gate_proj", {inter, hidden}},       {"mlp.up_proj", {inter, hidden}},
        {"mlp.down_proj", {hidden, inter}}};
    for (std::uint32_t L = 0; L < layers; ++L) {
        const std::string p = "layers." + std::to_string(L) + ".decoder_layer.";
        m[p + "input_layernorm.weight"] = {"BF16", {hidden}};
        m[p + "post_attention_layernorm.weight"] = {"BF16", {hidden}};
        m[p + "self_attn.q_norm.weight"] = {"BF16", {head_dim}};
        m[p + "self_attn.k_norm.weight"] = {"BF16", {head_dim}};
        for (const auto& [n, s] : linears) {
            m[p + n + ".weight"] = {"F8_E4M3", {s[0], s[1]}};
            m[p + n + ".weight_scale_inv"] = {"U8", {s[0], s[1] / 32U}};
        }
    }
    return m;
}

} // namespace

struct DraftFile::Blob {
    std::string name;
    std::vector<std::uint8_t> bytes;
};

DraftFile::~DraftFile() = default;

std::unique_ptr<DraftFile> DraftFile::open(const std::filesystem::path& dir, bool block_linears) {
    std::unique_ptr<DraftFile> f(new DraftFile());
    const auto want = manifest();
    std::map<std::string, const Blob*> found;
    std::size_t skipped = 0;
    for (const auto& path : assets::draft_files(dir)) {
        const auto header = read_header(path, &f->snapshot_);
        assets::Input in(path, &f->snapshot_);
        for (const auto& [name, t] : header) {
            if (name.rfind(checkpoint_prefix, 0) != 0) throw std::runtime_error("dspark: unexpected tensor " + name);
            const std::string key = name.substr(checkpoint_prefix.size());
            const auto it = want.find(key);
            if (it == want.end()) throw std::runtime_error("dspark: unexpected tensor " + name);
            if (it->second.dtype != t.dtype || it->second.shape != t.shape)
                throw std::runtime_error("dspark: dtype/shape mismatch for " + name);
            if (!block_linears && key.rfind("layers.", 0) == 0 && key.find("_proj.weight") != std::string::npos) {
                ++skipped;  // checked, not read: the caller supplies the block linears (DeviceDrafter::set_linear_hook)
                continue;
            }
            auto blob = std::make_unique<Blob>();
            blob->name = key;
            blob->bytes.resize(t.bytes);
            in.seekg(static_cast<std::streamoff>(t.offset));
            in.read(reinterpret_cast<char*>(blob->bytes.data()), static_cast<std::streamsize>(t.bytes));
            if (!in) throw std::runtime_error("dspark: short read of " + name);
            f->bytes_ += t.bytes;
            if (!found.emplace(key, blob.get()).second) throw std::runtime_error("dspark: duplicate tensor " + name);
            f->blobs_.push_back(std::move(blob));
        }
    }
    if (found.size() + skipped != want.size())
        throw std::runtime_error("dspark: " + std::to_string(found.size() + skipped) + " of " + std::to_string(want.size()) + " tensors in " + dir.string());
    const auto u16 = [&](const std::string& k) { return reinterpret_cast<const std::uint16_t*>(found.at(k)->bytes.data()); };
    const auto u8 = [&](const std::string& k) { return found.at(k)->bytes.data(); };
    const auto mx = [&](const std::string& k, std::uint32_t outputs, std::uint32_t width) {
        if (!block_linears) return Mxfp8Matrix{nullptr, nullptr, outputs, width};
        const std::uint8_t* s = u8(k + ".weight_scale_inv");
        for (std::size_t i = 0; i < static_cast<std::size_t>(outputs) * (width / 32U); ++i)
            if (s[i] == 255U) throw std::runtime_error("dspark: NaN E8M0 scale in " + k);
        return Mxfp8Matrix{u8(k + ".weight"), s, outputs, width};
    };
    Weights& w = f->weights_;
    w.fc = {u16("fc.weight"), hidden, tap_width};
    w.hidden_norm = u16("hidden_norm.weight");
    w.final_norm = u16("final_norm.weight");
    w.markov_w1 = u16("markov_head.markov_w1.weight");
    w.markov_w2 = u16("markov_head.markov_w2.weight");
    w.confidence_w = u16("confidence_head.proj.weight");
    w.confidence_b = bf16_value(*u16("confidence_head.proj.bias"));
    for (std::uint32_t L = 0; L < layers; ++L) {
        const std::string p = "layers." + std::to_string(L) + ".decoder_layer.";
        LayerWeights& y = w.layer[L];
        y.input_norm = u16(p + "input_layernorm.weight");
        y.post_norm = u16(p + "post_attention_layernorm.weight");
        y.q_norm = u16(p + "self_attn.q_norm.weight");
        y.k_norm = u16(p + "self_attn.k_norm.weight");
        y.q = mx(p + "self_attn.q_proj", q_width, hidden);
        y.k = mx(p + "self_attn.k_proj", kv_width, hidden);
        y.v = mx(p + "self_attn.v_proj", kv_width, hidden);
        y.o = mx(p + "self_attn.o_proj", hidden, q_width);
        y.gate = mx(p + "mlp.gate_proj", inter, hidden);
        y.up = mx(p + "mlp.up_proj", inter, hidden);
        y.down = mx(p + "mlp.down_proj", hidden, inter);
    }
    return f;
}

std::span<const std::uint8_t> DraftFile::tensor(const std::string& name) const {
    const std::string key = name.rfind(checkpoint_prefix, 0) == 0 ? name.substr(checkpoint_prefix.size()) : name;
    for (const auto& b : blobs_)
        if (b->name == key) return b->bytes;
    throw std::runtime_error("dspark: no tensor " + name);
}

std::vector<float> read_bf16_rows(const std::filesystem::path& dir, const std::string& name, std::span<const std::uint32_t> rows) {
    const auto file = assets::pack_weight_map(dir).at(name);
    const auto header = read_header(file);
    const TensorInfo& t = header.at(name);
    if (t.dtype != "BF16" || t.shape.size() != 2) throw std::runtime_error("dspark: " + name + " is not a BF16 matrix");
    const std::uint64_t width = t.shape[1];
    std::ifstream in(file, std::ios::binary);
    std::vector<float> out(rows.size() * width);
    std::vector<std::uint16_t> raw(width);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (rows[r] >= t.shape[0]) throw std::out_of_range("dspark: embedding row");
        in.seekg(static_cast<std::streamoff>(t.offset + static_cast<std::uint64_t>(rows[r]) * width * 2U));
        in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(width * 2U));
        if (!in) throw std::runtime_error("dspark: short read of " + name);
        for (std::uint64_t i = 0; i < width; ++i) out[r * width + i] = bf16_value(raw[i]);
    }
    return out;
}

} // namespace strix::minimax::dspark
