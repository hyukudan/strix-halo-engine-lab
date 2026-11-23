#include "strix/qwen_model_description.hpp"

#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace strix::qwen_description {
namespace {
using model_description::ModelDescription;
using model_description::Value;
struct UIntField { const char* key; std::uint64_t ModelContract::*member; };
struct FloatField { const char* key; float ModelContract::*member; };
struct BoolField { const char* key; bool ModelContract::*member; };
constexpr UIntField uint_fields[] = {
    {"dims.vocab", &ModelContract::vocab_size},
    {"dims.hidden", &ModelContract::hidden_size},
    {"dims.max_context", &ModelContract::max_position_embeddings},
    {"dims.layers", &ModelContract::hidden_layers},
    {"dims.n_heads", &ModelContract::attention_heads},
    {"dims.n_kv_heads", &ModelContract::kv_heads},
    {"dims.head_dim", &ModelContract::head_dim},
    {"attention.full_interval", &ModelContract::full_attention_interval},
    {"attention.qsa_layers", &ModelContract::full_attention_layers},
    {"attention.gdn_layers", &ModelContract::gdn_layers},
    {"gdn.key_heads", &ModelContract::gdn_key_heads},
    {"gdn.key_head_dim", &ModelContract::gdn_key_head_dim},
    {"gdn.value_heads", &ModelContract::gdn_value_heads},
    {"gdn.value_head_dim", &ModelContract::gdn_value_head_dim},
    {"gdn.conv_kernel", &ModelContract::gdn_conv_kernel},
    {"indexer.n_heads", &ModelContract::indexer_n_heads},
    {"indexer.kv_heads", &ModelContract::indexer_kv_heads},
    {"indexer.head_dim", &ModelContract::indexer_head_dim},
    {"indexer.budget", &ModelContract::indexer_budget},
    {"indexer.compress_ratio", &ModelContract::indexer_compress_ratio},
    {"moe.intermediate", &ModelContract::moe_intermediate_size},
    {"moe.shared_intermediate", &ModelContract::shared_expert_intermediate_size},
    {"moe.experts", &ModelContract::num_experts},
    {"moe.top_k", &ModelContract::num_experts_per_token},
    {"ple.layer_1based", &ModelContract::ple_layer_1based},
    {"ple.embed_dim", &ModelContract::ple_embed_dim},
    {"ple.conv_kernel_size", &ModelContract::ple_conv_kernel_size},
    {"ple.ngram_size", &ModelContract::ngram_size},
    {"ple.heads_per_ngram", &ModelContract::heads_per_ngram},
    {"ple.ngram_vocab_size_base", &ModelContract::ngram_vocab_size_base},
    {"ple.ngram_vocab_divisor", &ModelContract::ngram_vocab_divisor},
    {"ple.ngram_hash_seed", &ModelContract::ngram_hash_seed},
    {"ple.eos_token_id", &ModelContract::ple_eos_token_id},
    {"hc.count", &ModelContract::hc_count},
    {"hc.lowrank", &ModelContract::hc_lowrank},
    {"mtp.full_attention_layers", &ModelContract::mtp_full_attention_layers},
    {"storage.bytes_bf16", &ModelContract::bytes_bf16},
    {"storage.bytes_fp32", &ModelContract::bytes_fp32},
};
constexpr FloatField float_fields[] = {
    {"gdn.qk_l2_epsilon", &ModelContract::gdn_qk_l2_epsilon},
    {"rope.partial", &ModelContract::partial_rotary_factor},
    {"rope.theta", &ModelContract::rope_theta},
    {"norm.rms_epsilon", &ModelContract::rms_norm_epsilon},
};
constexpr BoolField bool_fields[] = {
    {"gdn.output_gate_sigmoid", &ModelContract::gdn_output_gate_sigmoid},
    {"gdn.qk_l2norm", &ModelContract::gdn_qk_l2norm},
    {"gdn.conv_silu", &ModelContract::gdn_conv_silu},
    {"moe.norm_topk_probability", &ModelContract::norm_topk_probability},
    {"mtp.use_dedicated_embeddings", &ModelContract::mtp_use_dedicated_embeddings},
    {"mtp.hidden_state_source_layer_explicit", &ModelContract::mtp_hidden_state_source_layer_explicit},
};

std::string display(const Value* value) {
    if (!value) return "<missing>";
    if (auto p = std::get_if<std::int64_t>(&value->data)) return std::to_string(*p) + " (integer)";
    if (auto p = std::get_if<double>(&value->data)) {
        std::ostringstream out; out.precision(17); out << *p; return out.str() + " (float)";
    }
    if (auto p = std::get_if<bool>(&value->data)) return *p ? "true" : "false";
    if (auto p = std::get_if<std::string>(&value->data)) return "\"" + *p + "\" (string)";
    if (auto p = std::get_if<Value::Array>(&value->data)) {
        std::string out = "[";
        for (const auto& v : *p) { if (out.size() > 1) out += ", "; out += display(&v); }
        return out + "]";
    }
    return "<table>";
}
struct Checks {
    std::string errors;
    std::set<std::string> known;
    void mismatch(const std::string& key, const std::string& expected, const std::string& actual) {
        errors += "\n  " + key + ": expected " + expected + ", actual " + actual;
    }
    void finish(const char* prefix = "Qwen model description refused:") const {
        if (!errors.empty()) throw std::invalid_argument(std::string(prefix) + errors);
    }
    const Value* field(const ModelDescription& file, const std::string& key) {
        known.insert(key); return file.find(key);
    }
    void fixed(const ModelDescription& file, const std::string& key, const Value& expected) {
        const auto* actual = field(file, key);
        if (!actual || *actual != expected) mismatch(key, display(&expected), display(actual));
    }
    void unknown(const Value::Table& table, const std::string& prefix = "") {
        for (const auto& [key, value] : table) {
            if (prefix.empty() && model_description::is_launch_metadata(key)) continue;
            const auto path = prefix + key;
            if (known.contains(path)) continue;
            if (const auto* child = std::get_if<Value::Table>(&value.data)) {
                if (child->empty()) mismatch(path, "supported nonempty table", "empty/unknown table");
                else unknown(*child, path + ".");
            } else mismatch(path, "supported Qwen field", display(&value));
        }
    }
};
Value words(std::initializer_list<const char*> list) {
    Value::Array out;
    for (auto word : list) out.push_back(Value{std::string(word)});
    return Value{std::move(out)};
}

ModelContract read_contract(const ModelDescription& file, Checks& check, const ModelContract* baked) {
    ModelContract out;
    for (const auto& field : uint_fields) {
        const auto* value = check.field(file, field.key);
        const auto* p = value ? std::get_if<std::int64_t>(&value->data) : nullptr;
        const auto expected = baked ? std::to_string((*baked).*field.member) : "nonnegative integer";
        if (!p || *p < 0) check.mismatch(field.key, expected, display(value));
        else {
            out.*field.member = static_cast<std::uint64_t>(*p);
            if (baked && out.*field.member != (*baked).*field.member) check.mismatch(field.key, expected, display(value));
        }
    }
    for (const auto& field : float_fields) {
        const auto* value = check.field(file, field.key);
        const auto* d = value ? std::get_if<double>(&value->data) : nullptr;
        const auto* i = value ? std::get_if<std::int64_t>(&value->data) : nullptr;
        const double number = d ? *d : i ? static_cast<double>(*i) : 0;
        const Value expected{static_cast<double>(baked ? (*baked).*field.member : out.*field.member)};
        if ((!d && !i) || !std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max())
            check.mismatch(field.key, baked ? display(&expected) : "finite float32 number", display(value));
        else {
            out.*field.member = static_cast<float>(number);
            if (baked && out.*field.member != (*baked).*field.member) check.mismatch(field.key, display(&expected), display(value));
        }
    }
    for (const auto& field : bool_fields) {
        const auto* value = check.field(file, field.key);
        const auto* p = value ? std::get_if<bool>(&value->data) : nullptr;
        const std::string expected = baked ? ((*baked).*field.member ? "true" : "false") : "boolean";
        if (!p) check.mismatch(field.key, expected, display(value));
        else {
            out.*field.member = *p;
            if (baked && *p != (*baked).*field.member) check.mismatch(field.key, expected, display(value));
        }
    }
    check.fixed(file, "schema", Value{std::string("strix-model/v1")});
    check.fixed(file, "name", Value{std::string("qwen3.8-flash-next")});
    check.fixed(file, "source.format", Value{std::string("gguf")});
    check.fixed(file, "source.architecture", Value{std::string("qwen4exp")});
    check.fixed(file, "source.pack", Value{std::string("Q4_K_XL")});
    check.fixed(file, "attention.pattern", words({"gdn", "gdn", "gdn", "qsa"}));
    check.fixed(file, "rope.dimension", Value{std::int64_t{RotaryDescriptor::rotary_dim}});
    Value::Array section_values;
    for (auto s : RotaryDescriptor::sections) {
        Value item;
        item.data = static_cast<std::int64_t>(s);
        section_values.push_back(std::move(item));
    }
    Value sections;
    sections.data = std::move(section_values);
    check.fixed(file, "rope.sections", sections);
    check.fixed(file, "rope.original_max", Value{std::int64_t{RotaryDescriptor::original_max}});
    Value interleaved;
    interleaved.data = RotaryDescriptor::interleaved;
    check.fixed(file, "rope.interleaved", interleaved);
    const auto* kv = check.field(file, "storage.attention_kv");
    const auto* kv_text = kv ? std::get_if<std::string>(&kv->data) : nullptr;
    const auto expected_kv = baked ? baked->attention_kv_storage : AttentionKvStorage::bf16;
    if (!kv_text || (*kv_text != "bf16" && *kv_text != "int8_token"))
        check.mismatch("storage.attention_kv", "bf16 or int8_token", display(kv));
    else {
        out.attention_kv_storage = *kv_text == "bf16" ? AttentionKvStorage::bf16 : AttentionKvStorage::int8_token;
        if (baked && out.attention_kv_storage != expected_kv)
            check.mismatch("storage.attention_kv", expected_kv == AttentionKvStorage::bf16 ? "bf16" : "int8_token", *kv_text);
    }
    check.fixed(file, "formats.expert_gate_up", words({"q4_k", "q5_k"}));
    check.fixed(file, "formats.expert_down", words({"q5_1", "q8_0"}));
    check.fixed(file, "formats.dense_projection", Value{std::string("q8_0")});
    check.fixed(file, "formats.normalization", Value{std::string("f32")});
    check.fixed(file, "formats.index_projection", Value{std::string("bf16")});
    check.unknown(file.fields);
    if (check.errors.empty()) {
        const auto error = out.validate();
        if (!error.empty()) check.mismatch("contract", "valid Qwen dimensions/semantics", error);
        if (out.kv_heads == 0 || out.attention_heads == 0 || out.attention_heads % out.kv_heads != 0)
            check.mismatch("dims.n_heads", "positive multiple of dims.n_kv_heads", std::to_string(out.attention_heads));
        if (out.full_attention_interval != 4)
            check.mismatch("attention.full_interval", "4 (gdn,gdn,gdn,qsa template)", std::to_string(out.full_attention_interval));
        if (out.rotary_dim() != RotaryDescriptor::rotary_dim)
            check.mismatch("rope.dimension", "64 = dims.head_dim * rope.partial", std::to_string(out.rotary_dim()));
    }
    return out;
}

std::string metadata_value(const gguf::MetadataEntry& entry) {
    std::string value = "<unavailable>";
    if (entry.unsigned_value) value = std::to_string(*entry.unsigned_value);
    else if (entry.signed_value) value = std::to_string(*entry.signed_value);
    else if (entry.floating_value) { std::ostringstream out; out.precision(17); out << *entry.floating_value; value = out.str(); }
    else if (entry.string_value) value = "\"" + *entry.string_value + "\"";
    return value + " (GGUF type " + std::to_string(static_cast<std::uint32_t>(entry.type)) + ")";
}
bool integer_type(gguf::ValueType t) {
    return t == gguf::ValueType::uint8 || t == gguf::ValueType::uint16 || t == gguf::ValueType::uint32 ||
        t == gguf::ValueType::uint64 || t == gguf::ValueType::int8 || t == gguf::ValueType::int16 ||
        t == gguf::ValueType::int32 || t == gguf::ValueType::int64;
}
} // namespace

ModelContract contract_from_description(const ModelDescription& file) {
    Checks check; auto result = read_contract(file, check, nullptr); check.finish(); return result;
}
ModelContract contract_from_file(const std::string& path) { return contract_from_description(model_description::load(path)); }
void assert_matches(const ModelContract& baked, const ModelDescription& file) {
    Checks check; (void)read_contract(file, check, &baked); check.finish();
}
GraphTemplate graph_template(const ModelDescription& file) {
    const auto contract = contract_from_description(file);
    // This compiled adapter admits only the existing fixed template. No input
    // file may drive an unbounded graph allocation or substitute a new schedule.
    assert_matches(ModelContract::qwen38_flash_next(), file);
    return graph_template(contract);
}
GraphTemplate graph_template(const ModelContract& contract) {
    if (const auto error = contract.validate(); !error.empty()) throw std::invalid_argument(error);
    GraphTemplate out{contract, {}};
    for (std::uint64_t i = 0; i < out.contract.hidden_layers; ++i) {
        const bool qsa = (i + 1) % out.contract.full_attention_interval == 0;
        const bool ple = i + 1 == out.contract.ple_layer_1based;
        std::vector<StateEffect> effects = qsa
            ? std::vector{StateEffect::qsa_kv, StateEffect::qsa_index_pool}
            : std::vector{StateEffect::gdn_recurrence, StateEffect::gdn_convolution};
        if (ple) effects.push_back(StateEffect::ple_convolution);
        out.layers.push_back({i, qsa ? AttentionKind::qsa : AttentionKind::gdn, ple, std::move(effects)});
    }
    return out;
}

std::vector<std::string> assert_gguf_matches(const ModelContract& m, const gguf::FileInfo& file) {
    Checks check;
    std::vector<std::string> checked;
    const auto uint_key = [&](const std::string& key, std::uint64_t expected) {
        const auto* e = file.find_metadata(key);
        if (!e) return;
        checked.push_back(key);
        const bool valid = integer_type(e->type) &&
            ((e->unsigned_value && *e->unsigned_value == expected) ||
             (e->signed_value && *e->signed_value >= 0 && static_cast<std::uint64_t>(*e->signed_value) == expected));
        if (!valid) check.mismatch(key, std::to_string(expected) + " (integer)", metadata_value(*e));
    };
    if (const auto* e = file.find_metadata("general.architecture")) {
        checked.push_back(e->key);
        if (e->type != gguf::ValueType::string || e->string_value != "qwen4exp")
            check.mismatch(e->key, "qwen4exp (string)", metadata_value(*e));
    }
    const std::pair<const char*, std::uint64_t> dimensions[] = {
        {"vocab_size", m.vocab_size}, {"embedding_length", m.hidden_size},
        {"context_length", m.max_position_embeddings}, {"block_count", m.hidden_layers},
        {"attention.head_count", m.attention_heads}, {"attention.head_count_kv", m.kv_heads},
        {"attention.key_length", m.head_dim}, {"attention.value_length", m.head_dim},
        {"attention.full_attention_interval", m.full_attention_interval},
        {"rope.dimension_count", m.rotary_dim()}, {"expert_count", m.num_experts},
        {"expert_used_count", m.num_experts_per_token}, {"expert_feed_forward_length", m.moe_intermediate_size},
        {"expert_shared_feed_forward_length", m.shared_expert_intermediate_size},
        {"ssm.conv_kernel", m.gdn_conv_kernel}, {"ssm.state_size", m.gdn_key_head_dim},
        {"ssm.group_count", m.gdn_key_heads}, {"ssm.time_step_rank", m.gdn_value_heads},
        {"ssm.inner_size", m.gdn_value_heads * m.gdn_value_head_dim},
        {"indexer.head_count", m.indexer_n_heads}, {"indexer.head_count_kv", m.indexer_kv_heads},
        {"indexer.head_dim", m.indexer_head_dim}, {"indexer.top_k", m.indexer_budget},
        {"indexer.compress_ratio", m.indexer_compress_ratio}
    };
    for (const auto& [suffix, expected] : dimensions) uint_key(std::string("qwen4exp.") + suffix, expected);
    if (const auto* e = file.find_metadata("tokenizer.ggml.tokens")) {
        checked.push_back(e->key);
        if (e->type != gguf::ValueType::array || e->array_element_type != gguf::ValueType::string || e->array_length != m.vocab_size)
            check.mismatch(e->key, "string array length " + std::to_string(m.vocab_size),
                "type " + std::to_string(static_cast<std::uint32_t>(e->type)) + " element type " +
                (e->array_element_type ? std::to_string(static_cast<std::uint32_t>(*e->array_element_type)) : "missing") +
                " length " + (e->array_length ? std::to_string(*e->array_length) : "missing"));
    }
    if (const auto* e = file.find_metadata("qwen4exp.rope.freq_base")) {
        checked.push_back(e->key);
        if ((e->type != gguf::ValueType::float32 && e->type != gguf::ValueType::float64) ||
            !e->floating_value || *e->floating_value != static_cast<double>(m.rope_theta))
            check.mismatch(e->key, std::to_string(m.rope_theta) + " (float)", metadata_value(*e));
    }
    check.finish("Qwen GGUF metadata refused:");
    return checked;
}

} // namespace strix::qwen_description
