#include "strix/minimax/model_description.hpp"
#include "engine/mm_common.hpp"
#include "strix/serve/json.hpp"
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace strix::minimax::description {
namespace {
namespace c = engine::detail;
using Model = model_description::ModelDescription;
using Value = model_description::Value;
struct Check {
    std::vector<std::string> errors;
    void equal(const std::string& field, const std::string& expected, const std::string& actual) {
        if (expected != actual) errors.push_back(field + ": expected " + expected + ", actual " + actual);
    }
    void finish() const {
        if (errors.empty()) return;
        std::string message = "MiniMax model contract mismatches:";
        for (const auto& e : errors) message += "\n  " + e;
        throw std::runtime_error(message);
    }
};
std::string repr(const Value* v) {
    if (!v) return "<missing>";
    if (const auto* s = std::get_if<std::string>(&v->data)) return '"' + *s + '"';
    if (const auto* i = std::get_if<std::int64_t>(&v->data)) return std::to_string(*i);
    if (const auto* d = std::get_if<double>(&v->data)) { std::ostringstream s; s.precision(17); s << *d; return s.str() + " (float)"; }
    if (const auto* b = std::get_if<bool>(&v->data)) return *b ? "true" : "false";
    std::string out = "[";
    if (const auto* table = std::get_if<Value::Table>(&v->data)) {
        std::string result = "{";
        for (const auto& [key, item] : *table) {
            if (result.size() > 1) result += ',';
            result += key + ":" + repr(&item);
        }
        return result + '}';
    }
    for (const auto& item : std::get<Value::Array>(v->data)) {
        if (out.size() > 1) out += ',';
        out += repr(&item);
    }
    return out + ']';
}
std::string shape_text(const std::vector<std::uint64_t>& shape) {
    std::string s = "[";
    for (auto n : shape) { if (s.size() > 1) s += ','; s += std::to_string(n); }
    return s + ']';
}
void expected_shape(Check& check, const HeaderTensor& t, std::vector<std::uint64_t> expected) {
    check.equal(t.name + ".shape", shape_text(expected), shape_text(t.shape));
}
void plain(Check& check, const HeaderTensor& t, std::vector<std::uint64_t> shape, bool embed = false) {
    expected_shape(check, t, std::move(shape));
    if (embed ? t.dtype != "BF16" : (t.dtype != "BF16" && t.dtype != "F16" && t.dtype != "F32"))
        check.errors.push_back(t.name + ".dtype: expected " + (embed ? "BF16" : "BF16|F16|F32") + ", actual " + t.dtype);
}
// File layout from the compiled stage_in/stage_out adapter: input/output tiles
// are 16 wide, with a 16-word multiple encoding the independently packed bits.
bool linear(Check& check, const HeaderTensor& t, const std::string& suffix, std::uint64_t in, std::uint64_t out) {
    if (suffix == ".suh" || suffix == ".svh") {
        expected_shape(check, t, {suffix == ".suh" ? in : out});
        check.equal(t.name + ".dtype", "F16", t.dtype);
    } else if (suffix == ".trellis") {
        if (t.shape.size() != 3 || t.shape[0] != in / 16 || t.shape[1] != out / 16 ||
            t.shape[2] == 0 || t.shape[2] % 16 != 0)
            check.errors.push_back(t.name + ".shape: expected [" + std::to_string(in / 16) + ',' +
                                   std::to_string(out / 16) + ",positive multiple of 16], actual " + shape_text(t.shape));
        if (t.dtype != "I16" && t.dtype != "U16")
            check.errors.push_back(t.name + ".dtype: expected I16|U16, actual " + t.dtype);
    } else return false; // pack metadata for this linear is outside this dimension contract
    return true;
}
void tensor(Check& check, const HeaderTensor& t) {
    const auto& name = t.name;
    if (name == c::kP + "embed_tokens.weight") { plain(check, t, {c::kVocab, c::kHidden}, true); return; }
    if (name == c::kP + "norm.weight") { plain(check, t, {c::kHidden}); return; }
    const std::string head = "language_model.lm_head";
    if (name.starts_with(head)) { linear(check, t, name.substr(head.size()), c::kHidden, c::kVocab); return; }
    const auto prefix = c::kP + "layers.";
    if (!name.starts_with(prefix)) return;
    const auto rest = std::string_view(name).substr(prefix.size());
    const auto dot = rest.find('.');
    unsigned layer = 0;
    const auto r = std::from_chars(rest.data(), rest.data() + (dot == rest.npos ? rest.size() : dot), layer);
    if (dot == rest.npos || r.ec != std::errc{} || r.ptr != rest.data() + dot || layer >= c::kLayers) {
        check.errors.push_back(name + ".layer: expected integer in [0," + std::to_string(c::kLayers - 1) + "], actual " + std::string(rest.substr(0, dot)));
        return;
    }
    std::string role(rest.substr(dot + 1));
    if (role == "input_layernorm.weight" || role == "post_attention_layernorm.weight") { plain(check, t, {c::kHidden}); return; }
    const std::string attn = "self_attn.";
    if (role.starts_with(attn)) {
        role.erase(0, attn.size());
        if (role == "index_q_proj.weight") { plain(check, t, {c::kIdxQ, c::kHidden}); return; }
        if (role == "index_k_proj.weight") { plain(check, t, {128, c::kHidden}); return; }
        if (role == "q_norm.weight" || role == "k_norm.weight" || role == "index_q_norm.weight" || role == "index_k_norm.weight") {
            plain(check, t, {128}); return;
        }
        for (const auto& [proj, in, out] : std::vector<std::tuple<std::string, std::uint64_t, std::uint64_t>>{
                 {"q_proj", c::kHidden, c::kQ}, {"k_proj", c::kHidden, c::kKV},
                 {"v_proj", c::kHidden, c::kKV}, {"o_proj", c::kQ, c::kHidden}}) {
            if (role.starts_with(proj + '.')) { linear(check, t, role.substr(proj.size()), in, out); return; }
        }
        return;
    }
    std::uint64_t inter = c::kDense;
    bool expert = false;
    if (role.starts_with("mlp.")) {
        if (layer >= c::kFirstMoe) check.errors.push_back(name + ".role: expected dense layer < " + std::to_string(c::kFirstMoe) + ", actual " + std::to_string(layer));
        role.erase(0, 4);
    } else if (role.starts_with("block_sparse_moe.")) {
        if (layer < c::kFirstMoe) check.errors.push_back(name + ".role: expected MoE layer >= " + std::to_string(c::kFirstMoe) + ", actual " + std::to_string(layer));
        role.erase(0, 17);
        if (role == "gate.weight") { plain(check, t, {c::kExperts, c::kHidden}); return; }
        if (role == "e_score_correction_bias") { plain(check, t, {c::kExperts}); return; }
        inter = c::kExpertI;
        if (role.starts_with("shared_experts.")) role.erase(0, 15);
        else if (role.starts_with("experts.")) {
            role.erase(0, 8);
            const auto end = role.find('.');
            unsigned id = 0;
            const auto e = std::from_chars(role.data(), role.data() + (end == role.npos ? role.size() : end), id);
            if (end == role.npos || e.ec != std::errc{} || e.ptr != role.data() + end || id >= c::kExperts) {
                check.errors.push_back(name + ".expert: expected integer in [0," + std::to_string(c::kExperts - 1) + "], actual " + role.substr(0, end));
                return;
            }
            role.erase(0, end + 1);
            expert = true;
        } else return;
    } else return;
    for (const auto& proj : expert ? std::vector<std::string>{"w1", "w3", "w2"} : std::vector<std::string>{"gate_proj", "up_proj", "down_proj"}) {
        if (!role.starts_with(proj + '.')) continue;
        const bool down = proj == "down_proj" || proj == "w2";
        linear(check, t, role.substr(proj.size()), down ? inter : c::kHidden, down ? c::kHidden : inter);
        return;
    }
}
std::uint64_t uint(const serve::Json& value, const std::string& field) {
    std::uint64_t result = 0;
    if (value.is_number()) {
        const auto& s = value.number_source();
        const auto r = std::from_chars(s.data(), s.data() + s.size(), result);
        if (r.ec == std::errc{} && r.ptr == s.data() + s.size()) return result;
    }
    throw std::runtime_error(field + ": expected nonnegative uint64 integer, actual " + value.dump());
}
} // namespace
void validate(const Model& model) {
    Check check;
    std::set<std::string> fields;
    auto integer = [&](const char* key, std::uint64_t expected) {
        fields.insert(key);
        const auto* v = model.find(key);
        const auto* i = v ? std::get_if<std::int64_t>(&v->data) : nullptr;
        if (!i || *i < 0 || static_cast<std::uint64_t>(*i) != expected)
            check.errors.push_back(std::string(key) + ": expected integer " + std::to_string(expected) + ", actual " + repr(v));
    };
    auto string = [&](const char* key, const char* expected) {
        fields.insert(key);
        const auto* v = model.find(key);
        const auto* s = v ? std::get_if<std::string>(&v->data) : nullptr;
        if (!s || *s != expected) check.errors.push_back(std::string(key) + ": expected \"" + expected + "\", actual " + repr(v));
    };
    string("schema", "strix-model/v1");
    string("name", "minimax-m3.1-flash");
    string("graph.template", "minimax-m3.1-flash");
    string("attention.kind", "msa");
    integer("dims.hidden", c::kHidden); integer("dims.q", c::kQ); integer("dims.kv", c::kKV);
    integer("dims.index_q", c::kIdxQ); integer("dims.layers", c::kLayers); integer("dims.intermediate", c::kDense);
    integer("dims.head_dim", 128); integer("dims.n_heads", c::kQ / 128); integer("dims.n_kv_heads", c::kKV / 128);
    integer("dims.index_heads", c::kIdxQ / 128); integer("dims.vocab", c::kVocab); integer("dims.vocab_rank_pad", c::kVocabPad);
    integer("moe.first_layer", c::kFirstMoe); integer("moe.experts", c::kExperts); integer("moe.top_k", c::kTopE);
    integer("moe.intermediate", c::kExpertI);
    for (const auto& [key, expected] : std::vector<std::pair<std::string, float>>{
             {"math.eps", c::kEps}, {"math.alpha", c::kAlpha}, {"math.limit", c::kLimit}}) {
        fields.insert(key);
        const auto* v = model.find(key);
        const auto* d = v ? std::get_if<double>(&v->data) : nullptr;
        if (!d || static_cast<float>(*d) != expected) {
            std::ostringstream s; s.precision(9); s << expected;
            check.errors.push_back(key + ": expected float rounding to " + s.str() + ", actual " + repr(v));
        }
    }
    for (const auto& [key, expected] : std::vector<std::pair<std::string, std::vector<std::string>>>{
             {"graph.custom_subgraphs", {"msa-indexer", "dense-prefix", "sparse-moe"}},
             {"graph.state_effects", {"append-kv-index", "fixed-rank-order-exchange"}}}) {
        fields.insert(key);
        const auto* v = model.find(key);
        const auto* a = v ? std::get_if<Value::Array>(&v->data) : nullptr;
        bool same = a && a->size() == expected.size();
        if (same) for (std::size_t i = 0; i < a->size(); ++i) {
            const auto* s = std::get_if<std::string>(&(*a)[i].data);
            same &= s && *s == expected[i];
        }
        if (!same) {
            std::string list = "[";
            for (const auto& s : expected) { if (list.size() > 1) list += ','; list += '"' + s + '"'; }
            check.errors.push_back(key + ": expected " + list + "], actual " + repr(v));
        }
    }
    const auto unknown = [&](const auto& self, const Value::Table& table, const std::string& prefix) -> void {
        for (const auto& [key, v] : table) {
            if (prefix.empty() && model_description::is_launch_metadata(key)) continue;
            const auto path = prefix + key;
            // [moe.route] is bound by typed code from the generated constants (moe::route_policy_named), which rejects a
            // wrong word or width at compile time; it is not part of this contract.
            if (path == "moe.route" || path.starts_with("moe.route.")) continue;
            if (const auto* child = std::get_if<Value::Table>(&v.data); child && !child->empty()) self(self, *child, path + '.');
            else if (!fields.contains(path)) check.errors.push_back(path + ": expected supported adapter field, actual " + repr(&v));
        }
    };
    unknown(unknown, model.fields, "");
    check.finish();
}
Model load(const std::string& path) {
    auto model = model_description::load(path);
    validate(model);
    return model;
}
std::vector<HeaderTensor> header(std::string_view json, bool semantic) {
    const auto root = [&] {
        try { return serve::parse_json_unique(json); }
        catch (const std::runtime_error& e) { throw std::runtime_error(std::string("safetensors header: ") + e.what()); }
    }();
    if (!root.is_object()) throw std::runtime_error("safetensors header: expected object, actual " + root.dump());
    std::vector<HeaderTensor> out;
    Check check;
    for (const auto& [name, value] : root.as_object()) {
        if (name == "__metadata__") continue;
        try {
            const auto* dtype = value.find("dtype");
            const auto* shape = value.find("shape");
            const auto* offsets = value.find("data_offsets");
            if (!dtype || !dtype->is_string() || !shape || !shape->is_array() || !offsets || !offsets->is_array() || offsets->size() != 2)
                throw std::runtime_error(name + ": expected string dtype, array shape, two data_offsets; actual " + value.dump());
            HeaderTensor t{name, dtype->as_string(), {}, uint(offsets->as_array()[0], name + ".data_offsets[0]"),
                           uint(offsets->as_array()[1], name + ".data_offsets[1]")};
            std::uint64_t elements = 1;
            for (const auto& n : shape->as_array()) {
                const auto dim = uint(n, name + ".shape");
                if (dim && elements > std::numeric_limits<std::uint64_t>::max() / dim) throw std::runtime_error(name + ".shape: element count overflow");
                elements *= dim;
                t.shape.push_back(dim);
            }
            const std::map<std::string, std::uint64_t> widths{{"BOOL",1},{"U8",1},{"I8",1},{"I16",2},{"U16",2},
                {"BF16",2},{"F16",2},{"F32",4},{"I32",4},{"U32",4},{"I64",8},{"U64",8},{"F64",8}};
            const auto w = widths.find(t.dtype);
            if (w == widths.end()) throw std::runtime_error(name + ".dtype: expected supported safetensors dtype, actual " + t.dtype);
            if (elements > std::numeric_limits<std::uint64_t>::max() / w->second) throw std::runtime_error(name + ".shape: byte count overflow");
            const auto bytes = elements * w->second;
            if (t.end < t.begin || t.end - t.begin != bytes)
                check.errors.push_back(name + ".data_offsets: expected span " + std::to_string(bytes) + ", actual [" + std::to_string(t.begin) + ',' + std::to_string(t.end) + ']');
            if (semantic) tensor(check, t);
            out.push_back(std::move(t));
        } catch (const std::runtime_error& e) { check.errors.push_back(e.what()); }
    }
    // Sort references only: the returned tensor order and stored offsets stay
    // identical. Empty tensors own no bytes and may share an endpoint.
    std::vector<const HeaderTensor*> spans;
    for (const auto& t : out) if (t.begin < t.end) spans.push_back(&t);
    std::sort(spans.begin(), spans.end(), [](const auto* a, const auto* b) { return a->begin < b->begin; });
    for (std::size_t i = 1; i < spans.size(); ++i)
        if (spans[i]->begin < spans[i - 1]->end)
            check.errors.push_back(spans[i]->name + ".data_offsets: overlaps " + spans[i - 1]->name + ".data_offsets");
    check.finish();
    return out;
}
bool check_enabled() {
#ifdef _MSC_VER
    char* value = nullptr;
    std::size_t size = 0;
    const bool off = _dupenv_s(&value, &size, "STRIX_MM_MODEL_DESCRIPTION_CHECK") == 0 && value != nullptr &&
                     std::string_view(value) == "0";
    std::free(value);
    return !off;
#else
    const char* value = std::getenv("STRIX_MM_MODEL_DESCRIPTION_CHECK");
    return value == nullptr || std::string_view(value) != "0";
#endif
}
std::uint64_t header_payload_bytes(std::uint64_t header_bytes, std::uint64_t file_bytes) {
    if (file_bytes < 8 || header_bytes > file_bytes - 8)
        throw std::runtime_error("safetensors header length: exceeds file size");
    return file_bytes - 8 - header_bytes;
}
std::vector<HeaderTensor> PackHeaders::add(std::string_view json, std::uint64_t payload_bytes) {
    auto entries = header(json, semantic_);
    for (const auto& t : entries) {
        if (t.end > payload_bytes)
            throw std::runtime_error(t.name + ".data_offsets: expected within file payload, actual end " + std::to_string(t.end));
        if (names_.contains(t.name))
            throw std::runtime_error("duplicate tensor across safetensors shards: " + t.name);
    }
    // Refusal must not partially admit a shard.
    for (const auto& t : entries) names_.insert(t.name);
    return entries;
}
} // namespace strix::minimax::description
