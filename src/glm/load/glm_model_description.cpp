#include "glm_model_description.hpp"

#include "../rtxpf/import/glm_dims.hpp"
#include "glm_rank_plan_graph.hpp"
#include "strix/serve/json.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace strix::glm::model {
namespace {
using Description = model_description::ModelDescription;
using Value = model_description::Value;
using Json = serve::Json;
using Errors = std::vector<std::string>;
struct Dimension { const char* field; unsigned value; };
constexpr Dimension dimensions[] = {
    {"dims.hidden", kHidden}, {"dims.layers", kLayers}, {"dims.vocab", kVocab},
    {"dims.intermediate", kDenseI},
    {"attention.kda.heads", kKdaHeads}, {"attention.kda.head_dim", kKdaDim},
    {"attention.kda.qkv", kKdaQkv}, {"attention.kda.conv_kernel", kConvK},
    {"attention.mla.q_lora", kQLora}, {"attention.mla.kv_lora", kKvLora},
    {"attention.mla.heads", kMlaHeads}, {"attention.mla.head_dim", kMlaDim},
    {"attention.mla.rope_dim", 0},
    {"attention.select.heads", kIdxHeads}, {"attention.select.head_dim", kIdxDim},
    {"attention.select.pool", kKpool}, {"attention.select.top_k", kIdxTopk},
    {"attention.select.pools", kIdxPools},
    {"moe.experts", kExperts}, {"moe.top_k", kTopE}, {"moe.intermediate", kExpertI},
    {"moe.first_layer", kFirstMoe}, {"moe.shared", 1},
    {"extras.hyper_connections.rank", kHc}, {"extras.hyper_connections.mix", kHcMix},
    {"extras.hyper_connections.iterations", kHcIters}, {"extras.mtp.layer", kMtpLayer},
    {"extras.mtp.heads", 1},
};
struct Scalar { const char* field; double value; };
constexpr Scalar scalars[] = {
    {"arithmetic.rms_eps", contract::rms_eps}, {"arithmetic.hc_eps", contract::hc_eps},
    {"arithmetic.l2_eps", contract::l2_eps}, {"arithmetic.ln_eps", contract::ln_eps},
    {"arithmetic.gate_lower_bound", contract::gate_lower_bound}, {"arithmetic.clamp", contract::clamp},
    {"arithmetic.route_scale", contract::route_scale}, {"arithmetic.route_den_eps", contract::route_den_eps},
    {"arithmetic.inv_sqrt128", contract::inv_sqrt128},
    {"arithmetic.inv_sqrt32", contract::inv_sqrt32}, {"arithmetic.mla_scale", contract::mla_scale},
};
std::string number(double n) { std::ostringstream s; s.precision(17); s << n; return s.str(); }
std::string actual(const Value* v) {
    if (!v) return "<missing>";
    if (const auto* p = std::get_if<std::int64_t>(&v->data)) return "integer " + std::to_string(*p);
    if (const auto* p = std::get_if<double>(&v->data)) return "float " + number(*p);
    if (const auto* p = std::get_if<bool>(&v->data)) return *p ? "boolean true" : "boolean false";
    if (const auto* p = std::get_if<std::string>(&v->data)) return "string \"" + *p + "\"";
    if (std::holds_alternative<Value::Array>(v->data)) return "array";
    return "table";
}
void mismatch(Errors& errors, const std::string& field, const std::string& expected, const std::string& got) {
    errors.push_back(field + ": expected " + expected + ", actual " + got);
}
bool known_field(const std::string& field) {
    bool known = field == "schema" || field == "name" || field == "graph.adapter" ||
        field == "attention.kind" || field == "graph.attention_pattern" || field == "graph.ffn_pattern";
    for (const auto& dimension : dimensions) known = known || field == dimension.field;
    for (const auto& scalar : scalars) known = known || field == scalar.field;
    return known;
}
void unknown_fields(Errors& errors, const Value::Table& table, const std::string& prefix = {}) {
    for (const auto& [key, value] : table) {
        const auto field = prefix.empty() ? key : prefix + "." + key;
        // [moe.route] is bound by typed code from the generated constants (moe::route_policy_named), which rejects a
        // wrong word or width at compile time; it is not part of this contract.
        if (field == "moe.route" || field.starts_with("moe.route.")) continue;
        if (const auto* child = std::get_if<Value::Table>(&value.data)) {
            // Empty declarations must not disappear during leaf traversal.
            if (child->empty())
                mismatch(errors, field, "nonempty supported GLM table", "empty table");
            else unknown_fields(errors, *child, field);
        } else if (!known_field(field)) {
            mismatch(errors, field, "supported GLM field", actual(&value));
        }
    }
}
void string_field(Errors& errors, const Description& d, const char* field, const char* expected) {
    const auto* v = d.find(field);
    if (!v || !std::holds_alternative<std::string>(v->data) || std::get<std::string>(v->data) != expected)
        mismatch(errors, field, std::string("string \"") + expected + "\"", actual(v));
}
void require_no_errors(const Errors& errors) {
    if (errors.empty()) return;
    std::string message = "GLM model contract mismatch";
    for (const auto& e : errors) message += "\n  " + e;
    throw std::runtime_error(message);
}
void json_integer(Errors& errors, const Json& object, const char* field, unsigned expected, bool required = true) {
    const auto* v = object.find(field);
    if (!v && !required) return;
    // Keep integer dimensions distinct from floats and booleans; as_int alone
    // accepts some numeric spellings that are inappropriate for dimensions.
    if (v && v->is_number() && v->number_source().find_first_not_of("0123456789") == std::string::npos) {
        try { if (v->as_int() == static_cast<std::int64_t>(expected)) return; } catch (const serve::JsonError&) {}
    }
    mismatch(errors, field, "integer " + std::to_string(expected), v ? v->dump() : "<missing>");
}
void json_scalar(Errors& errors, const Json& object, const char* field, double expected, bool required = true) {
    const auto* v = object.find(field);
    if (!v && !required) return;
    if (!v || !v->is_number() || v->as_double() != expected)
        mismatch(errors, field, "number " + number(expected), v ? v->dump() : "<missing>");
}
std::string read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("GLM model metadata: cannot open " + path.filename().string());
    const std::string out((std::istreambuf_iterator<char>(file)), {});
    if (file.bad()) throw std::runtime_error("GLM model metadata: cannot read " + path.filename().string());
    return out;
}
}  // namespace

std::vector<std::string> description_mismatches(const Description& d) {
    Errors errors;
    string_field(errors, d, "schema", "strix-model/v1");
    string_field(errors, d, "name", "glm-5.3-flash");
    string_field(errors, d, "graph.adapter", "glm-5.3-flash-v1");
    string_field(errors, d, "attention.kind", "kda+dsa");
    string_field(errors, d, "graph.attention_pattern", "kda,kda,kda,dsa");
    string_field(errors, d, "graph.ffn_pattern", "dense:0-2,moe:3-44");
    for (const auto& dimension : dimensions) {
        const auto* v = d.find(dimension.field);
        if (!v || !std::holds_alternative<std::int64_t>(v->data) ||
            std::get<std::int64_t>(v->data) != static_cast<std::int64_t>(dimension.value))
            mismatch(errors, dimension.field, "integer " + std::to_string(dimension.value), actual(v));
    }
    for (const auto& scalar : scalars) {
        const auto* v = d.find(scalar.field);
        if (!v || !std::holds_alternative<double>(v->data) || std::get<double>(v->data) != scalar.value)
            mismatch(errors, scalar.field, "float " + number(scalar.value), actual(v));
    }
    // Reject typos/unknown fields rather than silently accepting an unused
    // dimension or a request to change an unimplemented graph/state effect.
    unknown_fields(errors, d.fields);
    return errors;
}

std::vector<std::string> config_mismatches(std::string_view text, bool complete) {
    const auto root = serve::parse_json_unique(text);
    const auto* nested = root.find("text_config");
    const auto& cfg = nested ? *nested : root;
    Errors errors;
    if (!cfg.is_object()) { mismatch(errors, "config", "object", cfg.dump()); return errors; }
    // Same dimensional contract as the RTX source producer, mapped explicitly
    // to the HIP compiled dimensions rather than a second numeric authority.
    for (const auto& dimension : std::initializer_list<Dimension>{
        {"hidden_size", kHidden}, {"vocab_size", kVocab}, {"num_hidden_layers", kLayers},
        {"hc_mult", kHc}, {"q_lora_rank", kQLora}, {"kv_lora_rank", kKvLora},
        {"num_attention_heads", kMlaHeads}, {"qk_nope_head_dim", kMlaDim}, {"qk_rope_head_dim", 0},
        {"v_head_dim", kMlaDim}, {"index_n_heads", kIdxHeads}, {"index_head_dim", kIdxDim},
        {"index_kpool", kKpool}, {"index_topk", kIdxTopk}, {"n_routed_experts", kExperts},
        {"moe_intermediate_size", kExpertI}, {"intermediate_size", kDenseI}, {"num_nextn_predict_layers", 1}})
        json_integer(errors, cfg, dimension.field, dimension.value, complete);
    json_integer(errors, cfg, "num_experts_per_tok", kTopE, false);
    json_integer(errors, cfg, "first_k_dense_replace", kFirstMoe, false);
    json_scalar(errors, cfg, "routed_scaling_factor", contract::route_scale, false);
    json_scalar(errors, cfg, "rms_norm_eps", contract::rms_eps, false);
    const auto* linear = cfg.find("linear_attn_config");
    if (!linear && !complete) {
        // Admission of a legacy deployment: the keys that are present are checked, absent ones are not required.
    } else if (!linear || !linear->is_object())
        mismatch(errors, "linear_attn_config", "object", linear ? linear->dump() : "<missing>");
    else {
        Errors sub;
        json_integer(sub, *linear, "num_heads", kKdaHeads, complete);
        json_integer(sub, *linear, "head_dim", kKdaDim, complete);
        json_integer(sub, *linear, "short_conv_kernel_size", kConvK, complete);
        json_scalar(sub, *linear, "gate_lower_bound", contract::gate_lower_bound, complete);
        for (const auto& e : sub) errors.push_back("linear_attn_config." + e);
        Json expected = Json::array();
        for (unsigned l = 0; l < kLayers; ++l) if (l % 4 != 3) expected.push(Json(static_cast<int>(l)));
        const auto* layers = linear->find("kda_layers");
        if (!layers && !complete) {
        } else if (!layers || layers->dump() != expected.dump())
            mismatch(errors, "linear_attn_config.kda_layers", expected.dump(), layers ? layers->dump() : "<missing>");
    }
    for (const char* field : {"layer_types", "mlp_layer_types"}) {
        const auto* layers = cfg.find(field);
        if (!layers && !complete) continue;
        if (!layers || !layers->is_array()) {
            mismatch(errors, field, "array of " + std::to_string(kLayers) + " layer types", layers ? layers->dump() : "<missing>");
            continue;
        }
        if (layers->size() != kLayers)
            mismatch(errors, std::string(field) + ".size", std::to_string(kLayers), std::to_string(layers->size()));
        for (std::size_t l = 0; l < layers->size() && l < kLayers; ++l) {
            const char* expected = std::string_view(field) == "layer_types" ?
                (l % 4 != 3 ? "linear_attention" : "deepseek_sparse_attention") : (l < kFirstMoe ? "dense" : "sparse");
            const auto& v = layers->as_array()[l];
            if (!v.is_string() || v.as_string() != expected)
                mismatch(errors, std::string(field) + "[" + std::to_string(l) + "]", expected, v.dump());
        }
    }
    return errors;
}
void validate_description(const Description& d) { require_no_errors(description_mismatches(d)); }
void validate_config(std::string_view json) { require_no_errors(config_mismatches(json)); }
bool description_check_enabled() {
#ifdef _MSC_VER
    char* value = nullptr;
    std::size_t size = 0;
    const bool off = _dupenv_s(&value, &size, "STRIX_GLM_MODEL_DESCRIPTION_CHECK") == 0 && value != nullptr &&
                     std::string_view(value) == "0";
    std::free(value);
    return !off;
#else
    const char* value = std::getenv("STRIX_GLM_MODEL_DESCRIPTION_CHECK");
    return value == nullptr || std::string_view(value) != "0";
#endif
}
namespace {
void check_directory(const std::string& directory, bool complete) {
    const std::filesystem::path root(directory);
    const auto description = root / "glm-5.3-flash.toml";
    Errors errors;
    // status errors are failures too; an unreadable present description must not
    // be mistaken for an absent, optional deployment asset.
    if (std::filesystem::symlink_status(description).type() != std::filesystem::file_type::not_found)
        errors = description_mismatches(model_description::load(description.string()));
    const auto config = root / "config.json";
    // Admission keeps the legacy deployments loadable: a missing config.json and missing optional keys are not
    // refusals; every key that is present is still checked. The tool form requires the complete file.
    if (complete || std::filesystem::symlink_status(config).type() != std::filesystem::file_type::not_found)
        for (const auto& error : config_mismatches(read(config), complete)) errors.push_back(error);
    require_no_errors(errors);
}
}  // namespace
void validate_model_directory(const std::string& directory) {
    if (!description_check_enabled()) return;  // STRIX_GLM_MODEL_DESCRIPTION_CHECK=0: the previous admission
    check_directory(directory, false);
}
void validate_model_directory_complete(const std::string& directory) { check_directory(directory, true); }
namespace {
namespace mg = model_graph;
std::uint64_t dimension(const Description& d, const char* key) {
    // Called only after validate_description: type, presence and sign are checked.
    return static_cast<std::uint64_t>(std::get<std::int64_t>(d.find(key)->data));
}
mg::LayerDesc graph_layer(const Description& d, unsigned id, bool mtp) {
    mg::LayerDesc layer;
    layer.id = id;
    // The adapter has validated the complete declared cadence before applying it.
    layer.attention = !mtp && id % 4 != 3 ? mg::AttentionFamily::kda : mg::AttentionFamily::compressed_indexed;
    layer.norms = {{"rms", "arithmetic.rms_eps", mg::NormPlacement::pre_attention},
                   {"rms", "arithmetic.rms_eps", mg::NormPlacement::pre_mlp}};
    if (id < dimension(d, "moe.first_layer")) {
        layer.mlps = {{mg::MlpKind::dense, dimension(d, "dims.intermediate"), 0, 0}};
    } else {
        layer.mlps = {{mg::MlpKind::moe, dimension(d, "moe.intermediate"), dimension(d, "moe.experts"), dimension(d, "moe.top_k")},
                      {mg::MlpKind::shared_expert, dimension(d, "moe.intermediate"), dimension(d, "moe.shared"), 1}};
    }
    layer.tensors = {{"attention.projections", "source-linear", {"dims.hidden", "attention-width"}, ""},
                     {"mlp.projections", "source-linear", {"dims.hidden", "intermediate"}, "expert-partitionable"}};
    if (!mtp) {
        layer.hyper_connection = mg::HyperConnectionDesc{mg::HyperConnectionKind::mhc,
            dimension(d, "extras.hyper_connections.rank"), dimension(d, "extras.hyper_connections.iterations")};
        layer.tensors.push_back({"residual.streams", "fp32", {"extras.hyper_connections.rank", "dims.hidden"}, ""});
    } else {
        layer.tensors.push_back({"draft.embedding_hidden_projection", "source-linear", {"2*dims.hidden", "dims.hidden"}, ""});
        layer.norms.push_back({"rms", "arithmetic.rms_eps", mg::NormPlacement::input});
        layer.norms.push_back({"rms", "arithmetic.rms_eps", mg::NormPlacement::output});
    }
    if (layer.attention == mg::AttentionFamily::kda) {
        layer.states = {{{"attention.recurrent", "engine-state", {"attention.kda.heads", "attention.kda.head_dim", "attention.kda.head_dim"}, ""}, "sequence"},
                        {{"attention.convolution", "engine-state", {"3*attention.kda.qkv", "attention.kda.conv_kernel-1"}, ""}, "sequence"}};
    } else {
        layer.states = {{{"attention.latent_history", "engine-state", {"context", "attention.mla.kv_lora"}, ""}, "sequence"},
                        {{"attention.indexed_pool", "engine-state", {"context/attention.select.pool", "attention.select.head_dim"}, ""}, "sequence"},
                        {{"attention.position", "engine-state", {"1"}, ""}, "sequence"}};
    }
    return layer;
}
}  // namespace

model_graph::ModelGraph build_model_graph(const Description& d) {
    validate_description(d);
    mg::ModelGraph graph;
    graph.model = std::get<std::string>(d.find("name")->data);
    graph.semantic_version = std::get<std::string>(d.find("graph.adapter")->data);
    for (const char* field : {"schema", "attention.kind", "graph.attention_pattern", "graph.ffn_pattern"})
        graph.semantics[field] = std::get<std::string>(d.find(field)->data);
    for (const auto& entry : dimensions) graph.dimensions[entry.field] = dimension(d, entry.field);
    for (const auto& entry : scalars) {
        std::ostringstream value;
        value.imbue(std::locale::classic());
        value.precision(17);
        value << std::get<double>(d.find(entry.field)->data);
        graph.semantics[entry.field] = value.str();
    }
    graph.semantics["embedding_head"] = "separate weights; vocabulary partitionable; no deployment placement";
    graph.inputs = {{"token_ids", "u32", {"tokens"}, ""}};
    graph.outputs = {{"logits", "fp32", {"tokens", "dims.vocab"}, "vocabulary-partitionable"}};
    graph.tensors = {{"embedding", "source-weight", {"dims.vocab", "dims.hidden"}, "vocabulary-partitionable"},
                     {"head", "source-linear", {"dims.hidden", "dims.vocab"}, "vocabulary-partitionable"}};
    for (unsigned id = 0; id < dimension(d, "dims.layers"); ++id) {
        graph.layers.push_back(graph_layer(d, id, false));
        const auto endpoint = "layer." + std::to_string(id);
        graph.edges.push_back({id == 0 ? "embedding" : "layer." + std::to_string(id - 1) + ".output",
                               endpoint + ".input", mg::EdgeKind::data});
        graph.edges.push_back({endpoint + ".input", endpoint + ".output", mg::EdgeKind::hyper_connection});
    }
    graph.edges.push_back({"input.token_ids", "embedding", mg::EdgeKind::data});
    graph.edges.push_back({"layer." + std::to_string(graph.layers.back().id) + ".output", "head", mg::EdgeKind::data});
    graph.edges.push_back({"head", "output.logits", mg::EdgeKind::data});
    graph.attachments.push_back({mg::AttachmentKind::mtp, "mtp", "trunk.output",
        {graph_layer(d, static_cast<unsigned>(dimension(d, "extras.mtp.layer")), true)}});
    mg::validate(graph);
    return graph;
}

std::string print_model_graph(const std::string& directory) {
    // No opt-out: inspection must never describe an unvalidated source contract.
    check_directory(directory, false);
    const std::filesystem::path root(directory);
    // The description may live in the repository instead of next to the weights: STRIX_GLM_MODEL_DESCRIPTION names it.
    const char* described = std::getenv("STRIX_GLM_MODEL_DESCRIPTION");
    const auto description = described && *described ? std::filesystem::path(described) : root / "glm-5.3-flash.toml";
    const auto graph = build_model_graph(model_description::load(description.string()));
    const auto index = serve::parse_json_unique(read(root / "model.safetensors.index.json"));
    const auto* weights = index.find("weight_map");
    if (!weights || !weights->is_object()) throw std::runtime_error("GLM model graph: missing index weight_map");
    std::vector<std::string> names;
    for (const auto& [key, value] : weights->as_object()) {
        (void)value;
        names.push_back(key);
    }
    // MTP tensors ship in mtp.safetensors, outside the trunk index: add its tensor names from the file header.
    const auto mtp = root / "mtp.safetensors";
    if (std::filesystem::exists(mtp)) {
        std::ifstream in(mtp, std::ios::binary);
        unsigned char size_bytes[8] = {};
        if (!in.read(reinterpret_cast<char*>(size_bytes), 8)) throw std::runtime_error("GLM model graph: cannot read mtp.safetensors header size");
        std::uint64_t header_size = 0;
        for (int i = 7; i >= 0; --i) header_size = (header_size << 8U) | size_bytes[i];
        if (header_size == 0 || header_size > (std::uint64_t{64} << 20U)) throw std::runtime_error("GLM model graph: implausible mtp.safetensors header size");
        std::string header(static_cast<std::size_t>(header_size), '\0');
        if (!in.read(header.data(), static_cast<std::streamsize>(header_size))) throw std::runtime_error("GLM model graph: truncated mtp.safetensors header");
        const auto tensors = serve::parse_json_unique(header);
        if (!tensors.is_object()) throw std::runtime_error("GLM model graph: mtp.safetensors header is not an object");
        for (const auto& [key, value] : tensors.as_object()) {
            (void)value;
            if (key != "__metadata__") names.push_back(key);
        }
    }
    load::check_rank_plan_graph(graph, names);
    return mg::dump(graph);
}

}  // namespace strix::glm::model
