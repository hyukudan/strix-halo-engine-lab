#include "description.hpp"

#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace strix::gemma4 {
namespace {
using model_description::Value;
using Fields = std::map<std::string, Value, std::less<>>;
// Diagnostics and the flat expected-field contract are adapter-owned. The
// shared reader retains the nested TOML tree and knows no model schema.
std::string describe(const Value& value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    if (const auto* integer = std::get_if<std::int64_t>(&value.data)) out << "integer " << *integer;
    else if (const auto* real = std::get_if<double>(&value.data)) out << "float " << *real;
    else if (const auto* boolean = std::get_if<bool>(&value.data)) out << "boolean " << (*boolean ? "true" : "false");
    else if (const auto* text = std::get_if<std::string>(&value.data)) out << "string " << std::quoted(*text);
    else if (const auto* array = std::get_if<Value::Array>(&value.data)) {
        out << '[';
        bool first = true;
        for (const auto& item : *array) { if (!first) out << ", "; first = false; out << describe(item); }
        out << ']';
    } else {
        out << "table {";
        bool first = true;
        for (const auto& [key, item] : std::get<Value::Table>(value.data)) {
            if (!first) out << ", ";
            first = false; out << key << " = " << describe(item);
        }
        out << '}';
    }
    return out.str();
}
void check_unknown(const Value::Table& table, const std::string& prefix,
                   const Fields& expected, std::string& errors) {
    for (const auto& [key, value] : table) {
        if (prefix.empty() && model_description::is_launch_metadata(key)) continue;
        const auto path = prefix + key;
        if (const auto* nested = std::get_if<Value::Table>(&value.data)) {
            const auto it = expected.lower_bound(path + ".");
            const bool known = it != expected.end() && it->first.starts_with(path + ".");
            if (!known && !expected.contains(path))
                errors += path + ": expected <absent>, actual " + describe(value) + "\n";
            else if (known && nested->empty())
                errors += path + ": expected nonempty table, actual table {}\n";
            check_unknown(*nested, path + ".", expected, errors);
        } else if (!expected.contains(path)) {
            errors += path + ": expected <absent>, actual " + describe(value) + "\n";
        }
    }
}
Fields contract() {
    Fields f;
    const auto integer = [&](const char* k, std::uint32_t v) { f.emplace(k, Value{std::int64_t(v)}); };
    const auto real = [&](const char* k, double v) { f.emplace(k, Value{v}); };
    const auto text = [&](const char* k, const char* v) { f.emplace(k, Value{std::string(v)}); };
    const auto boolean = [&](const char* k, bool v) { f.emplace(k, Value{v}); };
    text("schema", "strix-model/v1"); text("name", "nemotron-3.5-bolt-e4b");
    text("adapter", "gemma4-e4b-v1");
    text("source.gguf_architecture", "gemma4"); text("source.text_model_type", "gemma4_text");
    // Descriptive attribution only; existing output literals stay in their owners.
    text("contract.arithmetic", arithmetic_id);
    integer("dims.hidden", hidden); integer("dims.layers", layers); integer("dims.intermediate", inter);
    integer("dims.vocab", vocab); integer("dims.n_heads", heads); integer("dims.n_kv_heads", kv_heads);
    integer("dims.head_dim", 256); integer("dims.global_head_dim", 512);
    integer("dims.max_context", max_positions); integer("dims.ple", ple_dim);
    integer("dims.ple_vocab", vocab); integer("attention.group", group);
    integer("attention.window", window); integer("attention.chunk", attention_chunk);
    integer("attention.shared_kv_layers", 18);
    integer("attention.rope_dimension", 256); integer("attention.global_rope_dimension", 512);
    real("attention.rope_theta", 10000.0); real("attention.global_rope_theta", 1000000.0);
    real("attention.partial_rotary_factor", 1.0); real("attention.global_partial_rotary_factor", 0.25);
    real("attention.scale", attention_scale); real("attention.dropout", 0.0);
    text("attention.kind", "sliding+global"); text("attention.rope_type", "default");
    text("attention.global_rope_type", "proportional"); text("attention.kv_dtype", "F32");
    boolean("attention.qk_norm", true); boolean("attention.bias", false); boolean("attention.k_eq_v", false);
    // TOML has no null. The adapter spells the required JSON null as a tag,
    // never as a value to be fed to an existing loader.
    text("attention.bidirectional", "null");
    text("dense.activation", "gelu_pytorch_tanh"); boolean("dense.tie_word_embeddings", true);
    boolean("dense.moe", false); boolean("dense.double_wide", false);
    // Norm constant is widened exactly as in the GGUF check. JSON separately
    // requires the decimal double 1e-6, which is not that widened float.
    real("contract.norm_eps_f32", double(norm_eps)); real("contract.text_norm_eps", 1e-6);
    real("contract.logit_cap", logit_cap); real("extras.embedding_scale", embedding_scale);
    real("extras.ple_embedding_scale", ple_embedding_scale); real("extras.model_ple_scale", model_ple_scale);
    real("extras.ple_combine_scale", ple_combine_scale);
    Value::Array taps, pattern, dimensions, producers;
    for (const auto l : tap_layers) taps.push_back(Value{std::int64_t(l)});
    for (std::uint32_t l = 0; l < layers; ++l) {
        const bool global = l % 6 == 5;
        pattern.push_back(Value{std::string(global ? "full_attention" : "sliding_attention")});
        dimensions.push_back(Value{std::int64_t(global ? 512 : 256)});
        producers.push_back(Value{std::int64_t(l < 24 ? l : (global ? 23U : 22U))});
    }
    f.emplace("extras.tap_layers", Value{std::move(taps)});
    f.emplace("attention.layer_types", Value{std::move(pattern)});
    f.emplace("attention.head_dims", Value{std::move(dimensions)});
    f.emplace("attention.kv_producers", Value{std::move(producers)});
    return f;
}
void write_value(std::ostream& out, const Value& v) {
    if (const auto* integer = std::get_if<std::int64_t>(&v.data)) out << *integer;
    else if (const auto* real = std::get_if<double>(&v.data)) {
        std::ostringstream number; number.imbue(std::locale::classic()); number << std::setprecision(17) << *real;
        const auto s = number.str(); out << s;
        if (s.find_first_of(".eE") == s.npos) out << ".0";
    } else if (const auto* boolean = std::get_if<bool>(&v.data)) out << (*boolean ? "true" : "false");
    else if (const auto* text = std::get_if<std::string>(&v.data)) out << std::quoted(*text);
    else {
        out << '[';
        bool first = true;
        for (const auto& item : std::get<Value::Array>(v.data)) { if (!first) out << ", "; first = false; write_value(out, item); }
        out << ']';
    }
}
}
std::string export_model_description() {
    const auto fields = contract();
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "# Export-only Nemotron 3.5 Bolt E4B contract (gemma4 architecture); does not configure execution.\n";
    // Root fields must precede every table.
    for (const auto& [k, v] : fields) if (k.find('.') == k.npos) { out << k << " = "; write_value(out, v); out << '\n'; }
    std::string table;
    for (const auto& [k, v] : fields) {
        const auto dot = k.find('.');
        if (dot == k.npos) continue;
        if (table != k.substr(0, dot)) { table = k.substr(0, dot); out << "\n[" << table << "]\n"; }
        out << k.substr(dot + 1) << " = "; write_value(out, v); out << '\n';
    }
    return out.str();
}
DescriptionTemplate validate_model_description(const model_description::ModelDescription& description) {
    const auto expected = contract();
    std::string errors;
    for (const auto& [k, v] : expected) {
        const auto* actual = description.find(k);
        if (!actual || *actual != v)
            errors += k + ": expected " + describe(v) + ", actual " +
                      (actual ? describe(*actual) : "<missing>") + "\n";
    }
    check_unknown(description.fields, "", expected, errors);
    if (!errors.empty()) throw std::runtime_error("Gemma4 E4B description refused:\n" + errors);
    DescriptionTemplate result;
    // Only after strict equality, build the fixed typed schedule; no description
    // data enters the engine or chooses arithmetic/identity/dispatch.
    const auto& pattern = std::get<Value::Array>(description.find("attention.layer_types")->data);
    const auto& dimensions = std::get<Value::Array>(description.find("attention.head_dims")->data);
    const auto& producers = std::get<Value::Array>(description.find("attention.kv_producers")->data);
    for (std::size_t l = 0; l < pattern.size(); ++l)
        result.config.layer.push_back({std::get<std::string>(pattern[l].data) == "full_attention" ? AttentionKind::global : AttentionKind::sliding,
            std::uint32_t(std::get<std::int64_t>(dimensions[l].data)), std::uint32_t(std::get<std::int64_t>(producers[l].data))});
    for (const auto& v : std::get<Value::Array>(description.find("extras.tap_layers")->data)) result.taps.push_back(std::uint32_t(std::get<std::int64_t>(v.data)));
    result.attention_chunk_rows = attention_chunk;
    result.embedding_multiplier = embedding_scale; result.ple_embedding_multiplier = ple_embedding_scale;
    result.ple_model_multiplier = model_ple_scale; result.ple_combine_multiplier = ple_combine_scale;
    return result;
}
} // namespace strix::gemma4
