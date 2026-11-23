#include "strix/deepseek41_model_description.hpp"
#include "model_graph/deepseek41_adapter.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace strix::deepseek41 {
namespace {
using model_description::ModelDescription;
std::string literal(const std::string& text) {
    std::string out = "\"";
    for (const unsigned char ch : text) {
        if (ch == '"' || ch == '\\') { out += '\\'; out += static_cast<char>(ch); }
        else if (ch == '\b') out += "\\b";
        else if (ch == '\f') out += "\\f";
        else if (ch == '\n') out += "\\n";
        else if (ch == '\r') out += "\\r";
        else if (ch == '\t') out += "\\t";
        else if (ch < 32 || ch == 127) throw std::invalid_argument("deepseek41 description: unsupported string control byte");
        else out += static_cast<char>(ch);
    }
    return out + '"';
}
std::string literal(bool value) { return value ? "true" : "false"; }
template <typename T> requires std::is_integral_v<T>
std::string literal(T value) { return std::to_string(value); }
std::string literal(double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("deepseek41 description: nonfinite config value");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    auto text = out.str();
    if (text.find_first_of(".eE") == std::string::npos) text += ".0";
    return text;
}
template <typename T> std::string literal(const std::vector<T>& values) {
    std::string out = "[";
    for (const auto value : values) { if (out.size() > 1) out += ", "; out += literal(value); }
    return out + ']';
}
using model_description::Value;
// Rendering and recursive traversal are inspection-adapter policy, not reader API.
std::string value_text(const Value& value) {
    return std::visit([](const auto& item) -> std::string {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, Value::Array>) {
            std::string out = "[";
            for (const auto& child : item) { if (out.size() > 1) out += ", "; out += value_text(child); }
            return out + ']';
        } else if constexpr (std::is_same_v<T, Value::Table>) {
            std::string out = "{";
            for (const auto& [key, child] : item) {
                if (out.size() > 1) out += ", ";
                out += literal(key) + ": " + value_text(child);
            }
            return out + '}';
        } else return literal(item);
    }, value.data);
}
std::string type_name(const Value& value) {
    constexpr const char* names[] = {"integer", "float", "boolean", "string", "array", "table"};
    return names[value.data.index()];
}
using Entries = std::map<std::string, const Value*>;
void collect(const Value::Table& table, const std::string& prefix, Entries& entries) {
    for (const auto& [key, value] : table) {
        if (prefix.empty() && model_description::is_launch_metadata(key)) continue;
        const auto path = prefix.empty() ? key : prefix + '.' + key;
        entries.emplace(path, &value); // Include tables, even empty unknown tables.
        if (const auto* nested = std::get_if<Value::Table>(&value.data)) collect(*nested, path, entries);
    }
}
Entries entries(const ModelDescription& description) {
    Entries result;
    collect(description.fields, "", result);
    return result;
}
struct Field { std::string path, json_key, value; };
std::vector<Field> fields(const Config& c) {
    // Config membership follows parse_config_json; only the TOML mapping belongs here.
    return {
        {"dims.vocab", "vocab_size", literal(c.vocab_size)},
        {"dims.hidden", "hidden_size", literal(c.hidden_size)},
        {"dims.intermediate", "moe_intermediate_size", literal(c.moe_intermediate_size)},
        {"dims.layers", "num_hidden_layers", literal(c.num_hidden_layers)},
        {"dims.n_heads", "num_attention_heads", literal(c.num_attention_heads)},
        {"dims.n_kv_heads", "num_key_value_heads", literal(c.num_key_value_heads)},
        {"dims.head_dim", "head_dim", literal(c.head_dim)},
        {"attention.qk_rope_head_dim", "qk_rope_head_dim", literal(c.qk_rope_head_dim)},
        {"attention.q_lora_rank", "q_lora_rank", literal(c.q_lora_rank)},
        {"attention.o_lora_rank", "o_lora_rank", literal(c.o_lora_rank)},
        {"attention.o_groups", "o_groups", literal(c.o_groups)},
        {"moe.n_routed_experts", "n_routed_experts", literal(c.n_routed_experts)},
        {"moe.n_shared_experts", "n_shared_experts", literal(c.n_shared_experts)},
        {"moe.num_experts_per_tok", "num_experts_per_tok", literal(c.num_experts_per_tok)},
        {"dims.max_context", "max_position_embeddings", literal(c.max_position_embeddings)},
        {"attention.sliding_window", "sliding_window", literal(c.sliding_window)},
        {"attention.index_n_heads", "index_n_heads", literal(c.index_n_heads)},
        {"attention.index_head_dim", "index_head_dim", literal(c.index_head_dim)},
        {"attention.index_topk", "index_topk", literal(c.index_topk)},
        {"attention.candidate_source_layer_id", "candidate_source_layer_id", literal(c.candidate_source_layer_id)},
        {"attention.candidate_topk_blocks", "candidate_topk_blocks", literal(c.candidate_topk_blocks)},
        {"attention.candidate_block_size", "candidate_block_size", literal(c.candidate_block_size)},
        {"extras.hc_mult", "hc_mult", literal(c.hc_mult)},
        {"extras.hc_sinkhorn_iters", "hc_sinkhorn_iters", literal(c.hc_sinkhorn_iters)},
        {"extras.num_nextn_predict_layers", "num_nextn_predict_layers", literal(c.num_nextn_predict_layers)},
        {"attention.rope_theta", "rope_theta", literal(c.rope_theta)},
        {"attention.compress_rope_theta", "compress_rope_theta", literal(c.compress_rope_theta)},
        {"attention.rms_norm_eps", "rms_norm_eps", literal(c.rms_norm_eps)},
        {"extras.hc_eps", "hc_eps", literal(c.hc_eps)},
        {"moe.swiglu_limit", "swiglu_limit", literal(c.swiglu_limit)},
        {"moe.routed_scaling_factor", "routed_scaling_factor", literal(c.routed_scaling_factor)},
        {"moe.scoring_func", "scoring_func", literal(c.scoring_func)},
        {"moe.hidden_act", "hidden_act", literal(c.hidden_act)},
        {"moe.topk_method", "topk_method", literal(c.topk_method)},
        {"moe.norm_topk_prob", "norm_topk_prob", literal(c.norm_topk_prob)},
        {"attention.rope_scaling.type", "rope_scaling.rope_type", literal(c.rope_scaling.type)},
        {"attention.rope_scaling.factor", "rope_scaling.factor", literal(c.rope_scaling.factor)},
        {"attention.rope_scaling.beta_fast", "rope_scaling.beta_fast", literal(c.rope_scaling.beta_fast)},
        {"attention.rope_scaling.beta_slow", "rope_scaling.beta_slow", literal(c.rope_scaling.beta_slow)},
        {"attention.rope_scaling.original_max_position_embeddings", "rope_scaling.original_max_position_embeddings", literal(c.rope_scaling.original_max_position_embeddings)},
        {"attention.compress_ratios", "compress_ratios", literal(c.compress_ratios)},
        {"attention.kv_source_layer_ids", "kv_source_layer_ids", literal(c.kv_source_layer_ids)},
        {"attention.index_source_layer_ids", "index_source_layer_ids", literal(c.index_source_layer_ids)},
        {"extras.engram_layer_ids", "engram_layer_ids", literal(c.engram_layer_ids)},
        {"extras.engram_num_embeddings", "engram_num_embeddings", literal(c.engram_num_embeddings)},
        {"extras.engram_n_heads", "engram_n_heads", literal(c.engram_n_heads)},
        {"extras.engram_head_dim", "engram_head_dim", literal(c.engram_head_dim)},
        {"extras.engram_max_ngram_size", "engram_max_ngram_size", literal(c.engram_max_ngram_size)},
        {"extras.engram_vocab_size", "engram_vocab_size", literal(c.engram_vocab_size)},
        {"extras.engram_compressed_vocab_size", "engram_compressed_vocab_size", literal(c.engram_compressed_vocab_size)},
        {"extras.engram_pad_token_id", "engram_pad_token_id", literal(c.engram_pad_token_id)},
        {"extras.dspark_block_size", "dspark_block_size", literal(c.dspark_block_size)},
        {"extras.dspark_markov_rank", "dspark_markov_rank", literal(c.dspark_markov_rank)},
        {"extras.dspark_n_routed_experts", "dspark_n_routed_experts", literal(c.dspark_n_routed_experts)},
        {"extras.dspark_num_experts_per_tok", "dspark_num_experts_per_tok", literal(c.dspark_num_experts_per_tok)},
        {"extras.dspark_noise_token_id", "dspark_noise_token_id", literal(c.dspark_noise_token_id)},
        {"extras.dspark_target_layer_ids", "dspark_target_layer_ids", literal(c.dspark_target_layer_ids)},
        {"tokens.bos_token_id", "bos_token_id", literal(c.bos_token_id)},
        {"tokens.eos_token_id", "eos_token_id", literal(c.eos_token_id)},
        {"tokens.pad_token_id", "pad_token_id", literal(c.pad_token_id)},
        {"tokens.image_token_id", "image_token_id", literal(c.image_token_id)},
    };
}
std::string json_of(const std::vector<Field>& entries, const ModelDescription* description = nullptr) {
    std::string root, text, rope;
    for (const auto& field : entries) {
        auto key = field.json_key;
        const auto value = description ? value_text(*description->find(field.path)) : field.value;
        auto* group = &text;
        if (field.path.starts_with("tokens.")) group = &root;
        else if (key.starts_with("rope_scaling.")) { group = &rope; key.erase(0, 13); }
        if (!group->empty()) *group += ',';
        *group += literal(key) + ':' + value;
    }
    return "{" + root + ",\"text_config\":{" + text + ",\"rope_scaling\":{" + rope + "}}}";
}
}  // namespace

std::string model_description_config_json(const Config& config) { return json_of(fields(config)); }

model_graph::ModelGraph graph_from_model_description(
    const ModelDescription& description, const Config& expected) {
    return model_graph_from_config(config_from_model_description(description, expected));
}

std::string describe_model_toml(const Config& config) {
    // Reuse all existing semantic checks, including source-layer graph checks.
    const auto checked = parse_config_json(model_description_config_json(config));
    (void)layer_attributes(checked);
    std::map<std::string, std::map<std::string, std::string>> tables;
    for (const auto& field : fields(checked)) {
        const auto dot = field.path.rfind('.');
        tables[field.path.substr(0, dot)][field.path.substr(dot + 1)] = field.value;
    }
    tables["attention"]["kind"] = "\"dsa\"";
    std::string out = "# Generated by strix-model-describe from the existing DeepSeek Config parser.\n"
        "# Inspection template only; tensor mappings, state effects and schedules stay in the explicit adapter.\n"
        "schema = \"strix-model/v1\"\nname = \"deepseek-v4.1-flash\"\n";
    for (const auto& [table, entries] : tables) {
        out += "\n[" + table + "]\n";
        for (const auto& [key, value] : entries) out += key + " = " + value + '\n';
    }
    return out;
}

Config config_from_model_description(const ModelDescription& description, const Config& expected) {
    const auto reference = model_description::parse(describe_model_toml(expected));
    std::string errors;
    for (const auto& [key, wanted] : entries(reference)) {
        const auto* actual = description.find(key);
        // Containers require the right type; each descendant is checked separately
        // so every missing, mistyped or changed field has its own diagnostic.
        const bool table = std::holds_alternative<Value::Table>(wanted->data);
        if (!actual || (table ? !std::holds_alternative<Value::Table>(actual->data) : *actual != *wanted)) {
            errors += "\n  " + key + ": expected " + (table ? "<table>" : value_text(*wanted)) +
                " (" + type_name(*wanted) + "), actual " +
                (actual ? value_text(*actual) + " (" + type_name(*actual) + ")" : "<missing>");
        }
    }
    for (const auto& [key, actual] : entries(description)) {
        if (!reference.find(key)) errors += "\n  " + key + ": expected <absent>, actual " + value_text(*actual);
    }
    if (!errors.empty()) throw std::invalid_argument("deepseek41 model description contract mismatches:" + errors);
    auto config = parse_config_json(json_of(fields(expected), &description));
    (void)layer_attributes(config);
    return config;
}
}  // namespace strix::deepseek41
