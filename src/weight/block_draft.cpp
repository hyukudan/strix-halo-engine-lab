#include "strix/weight/block_draft.hpp"
#include "strix/serve/json.hpp"
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::weight {
namespace {
const serve::Json& field(const serve::Json& j, const char* key) {
    const auto* p = j.find(key);
    if (!p) throw std::invalid_argument(std::string("block draft: missing ") + key);
    return *p;
}
std::uint32_t integer(const serve::Json& j, const char* key, bool zero = false) {
    const auto v = field(j, key).as_int();
    if (v < (zero ? 0 : 1) || v > 1048576) throw std::invalid_argument("block draft: invalid extent");
    return static_cast<std::uint32_t>(v);
}
double positive(const serve::Json& j, const char* key) {
    const auto v = field(j, key).as_double();
    if (!std::isfinite(v) || v <= 0) throw std::invalid_argument("block draft: invalid scalar");
    return v;
}
}

BlockDraftConfig parse_block_draft_config(std::string_view json) {
    const auto root = serve::parse_json_unique(json);
    const auto& d = field(root, "dflash_config");
    BlockDraftConfig c;
    c.hidden = integer(root, "hidden_size"); c.intermediate = integer(root, "intermediate_size");
    c.layers = integer(root, "num_hidden_layers"); c.heads = integer(root, "num_attention_heads");
    c.kv_heads = integer(root, "num_key_value_heads"); c.head_dim = integer(root, "head_dim");
    c.vocab = integer(root, "vocab_size"); c.window = integer(root, "sliding_window");
    c.block = integer(d, "block_size"); c.group = integer(d, "conv_group_size");
    c.taps = integer(d, "conv_kernel_size"); c.rank = integer(d, "selector_rank");
    c.top_k = integer(d, "selector_top_k"); c.mask = integer(d, "mask_token_id", true);
    c.epsilon = positive(root, "rms_norm_eps"); c.softcap = positive(d, "final_logit_softcapping");
    const auto& rope = field(root, "rope_parameters");
    c.theta = positive(rope, "rope_theta");
    const auto mode = field(d, "selector_softcap_mode").as_string();
    if (mode != "after_correction" && mode != "before_correction")
        throw std::invalid_argument("block draft: unknown selector softcap mode");
    c.post_softcap = mode == "after_correction";
    for (const auto* name : {"input_embedding_scale", "output_multiplier"}) {
        const auto* scale = d.find(name);
        if (!scale) scale = root.find(name);
        if (scale && scale->as_double() != 1.0)
            throw std::invalid_argument("block draft: non-unit embedding/head scale is unsupported");
    }
    if (const auto* dropout = root.find("attention_dropout"); dropout && dropout->as_double() != 0.0)
        throw std::invalid_argument("block draft: inference dropout must be zero");
    if (field(root, "attention_bias").as_bool() || field(root, "is_causal").as_bool() ||
        field(root, "hidden_act").as_string() != "silu" ||
        field(root, "dtype").as_string() != "bfloat16" ||
        field(rope, "rope_type").as_string() != "default")
        throw std::invalid_argument("block draft: unsupported arithmetic contract");
    const auto& types = field(root, "layer_types").as_array();
    if (types.size() != c.layers) throw std::invalid_argument("block draft: layer_types size");
    for (const auto& type : types)
        if (type.as_string() != "sliding_attention") throw std::invalid_argument("block draft: layer type");
    const auto target_layers = integer(root, "num_target_layers");
    for (const auto& id : field(d, "target_layer_ids").as_array()) {
        const auto n = id.as_int();
        if (n < 0 || n > target_layers || (!c.target_layers.empty() && n <= c.target_layers.back()))
            throw std::invalid_argument("block draft: capture ids must be ordered, unique and in range");
        c.target_layers.push_back(static_cast<std::uint32_t>(n));
    }
    if (c.target_layers.empty() || c.target_layers.size() > 64 || c.layers > 128 || c.block < 2 ||
        c.block > 32 || c.hidden % c.group || c.heads % c.kv_heads || c.head_dim % 2 ||
        c.taps != 2 || c.top_k > c.vocab || c.mask >= c.vocab)
        throw std::invalid_argument("block draft: inconsistent geometry");
    return c;
}

DraftShapes block_draft_shapes(const BlockDraftConfig& c) {
    const std::uint64_t h = c.hidden, q = std::uint64_t(c.heads) * c.head_dim;
    const std::uint64_t kv = std::uint64_t(c.kv_heads) * c.head_dim;
    DraftShapes s{{"fc.weight", {h, h * c.target_layers.size()}}, {"hidden_norm.weight", {h}},
        {"norm.weight", {h}}, {"candidate_selector.hidden_projection.weight", {c.rank, h}},
        {"candidate_selector.predecessor_codebook", {c.vocab, c.rank}},
        {"candidate_selector.successor_codebook", {c.vocab, c.rank}}};
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const auto p = "layers." + std::to_string(i) + ".";
        s[p + "input_layernorm.weight"] = {h}; s[p + "post_attention_layernorm.weight"] = {h};
        for (const auto* conv : {"attention_conv.", "mlp_conv."}) {
            s[p + conv + "base_kernel"] = {2, c.taps, h};
            s[p + conv + "kernel_projection.weight"] = {2 * c.taps * (h / c.group), h};
        }
        s[p + "self_attn.q_proj.weight"] = {q, h}; s[p + "self_attn.k_proj.weight"] = {kv, h};
        s[p + "self_attn.v_proj.weight"] = {kv, h}; s[p + "self_attn.o_proj.weight"] = {h, q};
        s[p + "self_attn.q_norm.weight"] = {c.head_dim}; s[p + "self_attn.k_norm.weight"] = {c.head_dim};
        s[p + "mlp.gate_proj.weight"] = {c.intermediate, h}; s[p + "mlp.up_proj.weight"] = {c.intermediate, h};
        s[p + "mlp.down_proj.weight"] = {h, c.intermediate};
    }
    return s;
}

Status index_block_draft(std::shared_ptr<const Source> source, std::string_view json, BlockDraftCheckpoint& out) {
    try {
        BlockDraftCheckpoint result;
        result.config = parse_block_draft_config(json);
        if (!source) return {Error::stale_source, "block draft: source absent"};
        // tensor_id otherwise asks an unnamed Source for its full-file digest.
        // Header admission must never read payload, even for an unnamed source.
        if (source->shard_name().empty()) source = namespace_source(std::move(source), "block-draft.header-only@1");
        auto status = index_safetensors(std::move(source), result.index);
        if (!status) return status;
        auto expected = block_draft_shapes(result.config);
        for (const auto& t : result.index.tensors) {
            const auto it = expected.find(t.name);
            if (it == expected.end() || it->second != t.extents || t.wire_dtype != "BF16")
                return {Error::malformed, "block draft: unexpected tensor, dtype or shape: " + t.name};
            result.payload_bytes += t.payload.range.bytes;
            expected.erase(it);
        }
        if (!expected.empty()) return {Error::malformed, "block draft: missing tensor " + expected.begin()->first};
        result.config_digest = digest_bytes({reinterpret_cast<const Byte*>(json.data()), json.size()});
        out = std::move(result);
        return {Error::ok, {}};
    } catch (const std::exception& e) { return {Error::malformed, e.what()}; }
}

Status digest_block_draft_payload(const BlockDraftCheckpoint& c, std::map<std::string, Digest256>& out) {
    std::map<std::string, Digest256> result;
    for (const auto& t : c.index.tensors) {
        Digest256 digest{};
        auto status = digest_slice(t.payload, digest);
        if (!status) return status;
        result.emplace(t.name, digest);
    }
    out = std::move(result);
    return {Error::ok, {}};
}

Status read_block_draft_tensor(const BlockDraftCheckpoint& c, std::string_view name, std::vector<float>& out) {
    for (const auto& t : c.index.tensors) if (t.name == name) {
        if (t.wire_dtype != "BF16" || t.payload.range.bytes % 2 ||
            t.payload.range.bytes > std::numeric_limits<std::size_t>::max())
            return {Error::malformed, "block draft: invalid BF16 payload"};
        std::vector<Byte> bytes(static_cast<std::size_t>(t.payload.range.bytes));
        const auto status = read_slice(t.payload, bytes);
        if (!status) return status;
        std::vector<float> values(bytes.size() / 2);
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto word = std::to_integer<std::uint32_t>(bytes[2*i]) |
                             (std::to_integer<std::uint32_t>(bytes[2*i+1]) << 8);
            values[i] = std::bit_cast<float>(word << 16);
            if (!std::isfinite(values[i])) return {Error::malformed, "block draft: nonfinite weight"};
        }
        out = std::move(values);
        return {Error::ok, {}};
    }
    return {Error::malformed, "block draft: tensor not found"};
}
} // namespace strix::weight
