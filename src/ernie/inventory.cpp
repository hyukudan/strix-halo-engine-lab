#include "strix/ernie/inventory.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <set>
#include <stdexcept>
#include <utility>

namespace strix::ernie {
namespace {
using enum Module;
using weight::Error;
using weight::Status;
namespace dim = data::dims;
namespace att = data::attention;
namespace moe = data::moe;
namespace extra = data::extras;
constexpr auto kFlat = extra::hc_mult * dim::hidden;
constexpr auto kMix = (2 + extra::hc_mult) * extra::hc_mult;

struct Matrix {
    std::string_view name;
    Module module;
    std::uint64_t rows;
    std::uint64_t columns;
};

struct Expected {
    Module module;
    std::string_view dtype;
    std::array<std::uint64_t, 2> shape;
};

using Schema = std::map<std::string, Expected, std::less<>>;

void tensor(Schema& schema, std::string name, Module module, std::string_view dtype,
            std::uint64_t rows, std::uint64_t columns = 0) {
    if (!schema.emplace(std::move(name), Expected{module, dtype, {rows, columns}}).second)
        throw std::logic_error("duplicate name in ERNIE tensor schema");
}

void fp8(Schema& schema, const std::string& prefix, const Matrix& matrix) {
    const auto name = prefix + std::string(matrix.name);
    tensor(schema, name, matrix.module, "F8_E4M3", matrix.rows, matrix.columns);
    constexpr auto block_rows = model::narrow<std::uint64_t>(data::weights::checkpoint::weight_block_size[0]);
    constexpr auto block_columns = model::narrow<std::uint64_t>(data::weights::checkpoint::weight_block_size[1]);
    tensor(schema, inverse_scale_name(name), matrix.module, "F32",
           (matrix.rows + block_rows - 1U) / block_rows, (matrix.columns + block_columns - 1U) / block_columns);
}

void add_attention(Schema& schema, const std::string& prefix, bool sparse, std::uint64_t compression) {
    constexpr auto mla_value = dim::n_heads * att::mla::v_head_dim;
    constexpr auto grouped_value = att::o_groups * att::o_lora_rank;
    const auto value_width = sparse ? std::uint64_t{mla_value} : std::uint64_t{grouped_value};
    tensor(schema, prefix + "vha_postmix_U", value_head_mix, "BF16", dim::n_heads, att::vha_rank);
    tensor(schema, prefix + "vha_postmix_V", value_head_mix, "BF16", dim::n_heads, att::vha_rank);
    fp8(schema, prefix, {"gate_proj.weight", attention_gate, value_width, att::q_lora_rank});
    fp8(schema, prefix, {"o_proj.weight", sparse ? mla : hca, dim::hidden, value_width});
    const auto core = prefix + "core_attention.";
    if (sparse) {
        constexpr Matrix projections[]{
            {"q_a_proj.weight", mla, att::mla::q_lora_rank, dim::hidden},
            {"q_b_proj.weight", mla, dim::n_heads * (att::mla::qk_nope_head_dim + att::mla::qk_rope_head_dim), att::mla::q_lora_rank},
            {"kv_a_proj_with_mqa.weight", mla, att::mla::kv_lora_rank + att::mla::qk_rope_head_dim, dim::hidden},
            {"kv_b_proj.weight", mla, att::mla::n_kv_heads * (att::mla::qk_nope_head_dim + att::mla::v_head_dim), att::mla::kv_lora_rank},
            {"core_attention.indexer.wq_b.weight", sparse_indexer, att::indexer::index_n_heads * att::indexer::index_head_dim, att::mla::q_lora_rank},
        };
        for (const auto& projection : projections) fp8(schema, prefix, projection);
        tensor(schema, prefix + "q_a_layernorm.weight", mla, "BF16", att::mla::q_lora_rank);
        tensor(schema, prefix + "kv_a_layernorm.weight", mla, "BF16", att::mla::kv_lora_rank);
        tensor(schema, core + "softmax_offset", mla, "BF16", dim::n_heads);
        const auto indexer = core + "indexer.";
        tensor(schema, indexer + "wk.weight", sparse_indexer, "BF16", att::indexer::index_head_dim, dim::hidden);
        tensor(schema, indexer + "weights_proj.weight", sparse_indexer, "BF16", att::indexer::index_n_heads, dim::hidden);
        tensor(schema, indexer + "k_norm.weight", sparse_indexer, "BF16", att::indexer::index_head_dim);
        tensor(schema, indexer + "k_norm.bias", sparse_indexer, "BF16", att::indexer::index_head_dim);
        tensor(schema, indexer + "flashindexer_bias", flash_indexer, "BF16", att::indexer::index_n_heads);
        tensor(schema, indexer + "linear_gate.weight", flash_indexer, "BF16", att::indexer::index_n_heads, dim::hidden);
    } else {
        constexpr Matrix projections[]{
            {"linear_q_down_proj.weight", hca, att::q_lora_rank, dim::hidden},
            {"linear_q_up_proj.weight", hca, dim::n_heads * dim::head_dim, att::q_lora_rank},
            {"linear_kv_proj.weight", hca, att::kv_lora_rank, dim::hidden},
            {"linear_o_group_proj", hca, grouped_value, (dim::n_heads / att::o_groups) * att::v_head_dim},
        };
        for (const auto& projection : projections) fp8(schema, prefix, projection);
        tensor(schema, prefix + "q_layernorm.weight", hca, "BF16", att::q_lora_rank);
        tensor(schema, prefix + "kv_layernorm.weight", hca, "BF16", att::kv_lora_rank);
        tensor(schema, core + "attn_sink", hca, "F32", dim::n_heads);
        if (compression == 0) return;
        const auto comp = core + "compressor.";
        tensor(schema, comp + "ape", compressor, "F32", compression, att::kv_lora_rank);
        tensor(schema, comp + "linear_wgate.weight", compressor, "BF16", att::kv_lora_rank, dim::hidden);
        tensor(schema, comp + "linear_wkv.weight", compressor, "BF16", att::kv_lora_rank, dim::hidden);
        tensor(schema, comp + "norm.weight", compressor, "BF16", att::kv_lora_rank);
    }
}

void add_moe(Schema& schema, const std::string& prefix) {
    tensor(schema, prefix + "e_score_correction_bias", router, "F32", kExpertCount);
    tensor(schema, prefix + "gate.routed_scaling_factor_param", router, "F32", kExpertCount);
    tensor(schema, prefix + "gate.weight", router, "BF16", kExpertCount, dim::hidden);
    tensor(schema, prefix + "gate.weight_1", router, "BF16", kExpertCount, dim::hidden);
    constexpr auto latent = moe::moe_latent_size, intermediate = moe::moe_intermediate_size;
    tensor(schema, prefix + "latent_norm.weight", latent_projection, "BF16", latent);
    constexpr Matrix projections[]{
        {"fc1_latent_proj.weight", latent_projection, latent, dim::hidden},
        {"fc2_latent_proj.weight", latent_projection, dim::hidden, latent},
        {"shared_experts.w1.weight", shared_expert, intermediate * moe::n_shared_experts, dim::hidden},
        {"shared_experts.w2.weight", shared_expert, dim::hidden, intermediate * moe::n_shared_experts},
        {"shared_experts.w3.weight", shared_expert, intermediate * moe::n_shared_experts, dim::hidden},
    };
    for (const auto& projection : projections) fp8(schema, prefix, projection);
    for (unsigned expert = 0; expert < kExpertCount; ++expert) {
        const auto base = prefix + "experts." + std::to_string(expert) + ".";
        for (const auto* proj : {"w1", "w2", "w3"}) {
            const auto name = base + proj + ".weight";
            const auto rows = std::string_view(proj) == "w2" ? latent : intermediate;
            const auto columns = std::string_view(proj) == "w2" ? intermediate : latent;
            tensor(schema, name, routed_expert, "I8", static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns / 2));
            tensor(schema, inverse_scale_name(name), routed_expert, "F8_E8M0", static_cast<std::uint64_t>(rows),
                   static_cast<std::uint64_t>(columns / data::weights::checkpoint::expert_block_columns));
        }
    }
}

void add_block(Schema& schema, const std::string& prefix, bool sparse, std::uint64_t compression) {
    tensor(schema, prefix + "input_layernorm.weight", layer_norm, "BF16", dim::hidden);
    tensor(schema, prefix + "post_attention_layernorm.weight", layer_norm, "BF16", dim::hidden);
    for (const auto* hc : {"self_attention_hyper_connection.", "mlp_hyper_connection."}) {
        const auto base = prefix + hc;
        for (const auto* scalar : {"alpha_pre", "alpha_post", "alpha_res"})
            tensor(schema, base + scalar, hyper_connection, "F32", 1);
        tensor(schema, base + "bias", hyper_connection, "F32", kMix);
        tensor(schema, base + "mapping_proj.weight", hyper_connection, "F32", kFlat, kMix);
    }
    add_attention(schema, prefix + "self_attn.", sparse, compression);
}

void add_contract(Schema& schema, const std::string& prefix) {
    tensor(schema, prefix + "hc_head_base", hc_contract, "F32", extra::hc_mult);
    tensor(schema, prefix + "hc_head_fn", hc_contract, "F32", kFlat, extra::hc_mult);
    tensor(schema, prefix + "hc_head_scale", hc_contract, "F32", 1);
}

const Schema& known_schema() {
    static const Schema schema = [] {
        Schema rows;
        tensor(rows, "model.embed_tokens.weight", embedding, "BF16", dim::vocab, dim::hidden);
        tensor(rows, "lm_head.weight", lm_head, "BF16", dim::vocab, dim::hidden);
        tensor(rows, "model.lm_head.multimax_ranges", lm_head, "BF16", extra::multimax_parameters);
        tensor(rows, "model.lm_head.multimax_ts", lm_head, "BF16", extra::multimax_parameters);
        tensor(rows, "model.norm.weight", final_norm, "BF16", dim::hidden);
        add_contract(rows, "model.mhc_contract.");
        for (unsigned layer = 0; layer < kLayerCount; ++layer) {
            const auto prefix = target_layer_prefix(layer);
            const bool sparse = layer_is_mla(layer);
            add_block(rows, prefix, sparse, sparse ? 0U : static_cast<std::uint64_t>(att::compress_ratios[layer]));
            if (!layer_has_moe(layer)) {
                constexpr Matrix dense[]{
                    {"mlp.gate_proj.weight", dense_mlp, dim::intermediate, dim::hidden},
                    {"mlp.up_proj.weight", dense_mlp, dim::intermediate, dim::hidden},
                    {"mlp.down_proj.weight", dense_mlp, dim::hidden, dim::intermediate},
                };
                for (const auto& projection : dense) fp8(rows, prefix, projection);
            } else {
                add_moe(rows, prefix + "block_sparse_moe.");
            }
        }
        const std::string draft = "model.mtp.";
        add_contract(rows, draft + "mhc_contract.");
        tensor(rows, draft + "norm.weight", final_norm, "BF16", dim::hidden);
        tensor(rows, draft + "feature_norm.weight", draft_input, "BF16", dim::hidden);
        fp8(rows, draft, {"feature_projection.weight", draft_input, extra::dspark_feature_projection[0], extra::dspark_feature_projection[1]});
        tensor(rows, draft + "markov_head.markov_w1.weight", draft_markov, "BF16", data::drafter::rank, dim::vocab);
        tensor(rows, draft + "markov_head.markov_w2.weight", draft_markov, "BF16", dim::vocab, data::drafter::rank);
        tensor(rows, draft + "confidence_head.projection.weight", draft_confidence, "BF16", 1, extra::dspark_confidence_input);
        for (unsigned layer = 0; layer < kDraftLayerCount; ++layer) {
            const auto prefix = draft + std::to_string(layer) + ".";
            add_block(rows, prefix, false, 0U);
            add_moe(rows, prefix + "block_sparse_moe.");
        }
        return rows;
    }();
    return schema;
}
std::optional<unsigned> tensor_layer(std::string_view name, bool drafter) {
    const std::string_view prefix = drafter ? "model.mtp." : "model.layers.";
    if (!name.starts_with(prefix)) return {};
    name.remove_prefix(prefix.size());
    const auto dot = name.find('.');
    if (dot == std::string_view::npos) return {};
    unsigned layer = 0;
    const auto stop = name.data() + dot;
    const auto [end, error] = std::from_chars(name.data(), stop, layer);
    if (error != std::errc{} || end != stop) return {};
    return layer;
}

Status check_manifest(const std::filesystem::path& path, Inventory& inventory) {
    auto missing = inventory.missing_shards;
    std::erase(missing, kFlashIndexerShard); // The sidecar is outside the main index.
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error == std::errc::no_such_file_or_directory && !missing.empty()) return {Error::ok, {}};
    if (error) return {Error::stale_source, "cannot read checkpoint manifest: " + error.message()};
    if (size > 64U * 1024U * 1024U) return {Error::malformed, "checkpoint manifest exceeds size bound"};
    std::ifstream file(path, std::ios::binary);
    std::string json(static_cast<std::size_t>(size), '\0');
    if (!file.read(json.data(), static_cast<std::streamsize>(json.size())) || file.peek() != EOF)
        return {Error::stale_source, "could not read the checkpoint manifest at its recorded size"};

    std::set<std::string, std::less<>> shard_names;
    std::vector<weight::SafetensorsManifestEntry> entries;
    entries.reserve(inventory.tensors.size());
    for (const auto& [name, tensor] : inventory.tensors) {
        const auto shard = tensor.payload.source->shard_name();
        if (shard != kFlashIndexerShard)
            entries.push_back({name, *shard_names.insert(shard).first, tensor.payload.range.bytes});
    }
    const auto status = weight::validate_safetensors_manifest(entries, json, missing);
    inventory.has_manifest = static_cast<bool>(status);
    return status;
}
} // namespace

std::string_view module_name(Module module) noexcept {
    switch (module) {
    case unclassified: return "unclassified";
    case embedding: return "embedding";
    case layer_norm: return "layer_norm";
    case hyper_connection: return "hyper_connection";
    case hca: return "hca";
    case compressor: return "compressor";
    case mla: return "mla";
    case attention_gate: return "attention_gate";
    case value_head_mix: return "value_head_mix";
    case sparse_indexer: return "sparse_indexer";
    case flash_indexer: return "flash_indexer";
    case dense_mlp: return "dense_mlp";
    case router: return "router";
    case latent_projection: return "latent_projection";
    case shared_expert: return "shared_expert";
    case routed_expert: return "routed_expert";
    case final_norm: return "final_norm";
    case hc_contract: return "hc_contract";
    case lm_head: return "lm_head";
    case draft_input: return "draft_input";
    case draft_markov: return "draft_markov";
    case draft_confidence: return "draft_confidence";
    }
    return "unclassified";
}

std::string main_shard_name(unsigned index) {
    if (index == 0 || index > kMainShardCount) throw std::out_of_range("ERNIE shard index");
    auto number = std::to_string(index);
    number.insert(0, 5U - number.size(), '0');
    auto count = std::to_string(kMainShardCount);
    count.insert(0, 5U - count.size(), '0');
    return "model-" + number + "-of-" + count + ".safetensors";
}

std::string target_layer_prefix(unsigned layer) {
    return "model.layers." + std::to_string(layer) + ".";
}

std::string inverse_scale_name(std::string_view name) {
    if (name.ends_with("linear_o_group_proj")) return std::string(name) + "_scale_inv";
    if (!name.ends_with(".weight")) throw std::invalid_argument("ERNIE: weight has no inverse-scale naming rule");
    return std::string(name.substr(0, name.size() - std::string_view(".weight").size())) + ".weight_scale_inv";
}

Status add_shard(const weight::TensorIndex& shard, Inventory& inventory) {
    if (shard.container != weight::Container::safetensors || !shard.owner)
        return {Error::malformed, "expected an owned safetensors header"};
    const auto shard_name = shard.owner->shard_name();
    const auto& schema = known_schema();
    decltype(inventory.tensors) pending;
    for (const auto& raw : shard.tensors) {
        if (inventory.tensors.contains(raw.name) || pending.contains(raw.name))
            return {Error::malformed, "duplicate ERNIE tensor " + raw.name};
        Module module = unclassified;
        if (const auto it = schema.find(raw.name); it != schema.end()) {
            const auto& expected = it->second;
            const auto rank = expected.shape[1] == 0 ? 1U : 2U;
            if (expected.dtype != raw.wire_dtype ||
                !std::ranges::equal(raw.extents, std::span(expected.shape).first(rank)))
                return {Error::malformed, "wire dtype or shape disagrees for " + raw.name};
            module = expected.module;
        }
        if (module != unclassified && (module == flash_indexer) != (shard_name == kFlashIndexerShard))
            return {Error::malformed, "unexpected FlashIndexer shard mapping for " + raw.name};
        const bool drafter = raw.name.starts_with("model.mtp.");
        pending.emplace(raw.name, Tensor{module, drafter, tensor_layer(raw.name, drafter),
                                         raw.wire_dtype, raw.extents, raw.payload});
    }
    inventory.tensors.merge(pending);
    return {Error::ok, {}};
}

std::size_t count_missing_known_tensors(const Inventory& inventory) {
    return static_cast<std::size_t>(std::ranges::count_if(known_schema(), [&](const auto& entry) {
        return !inventory.tensors.contains(entry.first);
    }));
}

std::vector<std::string> missing_known_tensor_names(const Inventory& inventory) {
    std::vector<std::string> names;
    for (const auto& entry : known_schema())
        if (!inventory.tensors.contains(entry.first)) names.push_back(entry.first);
    return names;
}

std::optional<unsigned> target_layer(std::string_view name) { return tensor_layer(name, false); }

Status inspect_weights(const std::filesystem::path& root, Inventory& out) {
    std::error_code error;
    if (!std::filesystem::is_directory(root, error))
        return {Error::stale_source, "ERNIE checkpoint directory is unavailable"};
    Inventory inventory;
    const auto inspect = [&](std::string name) -> Status {
        const auto path = root / name;
        std::error_code ec;
        if (!std::filesystem::exists(path, ec) && !ec) {
            inventory.missing_shards.push_back(std::move(name));
            return {Error::ok, {}};
        }
        if (ec) return {Error::stale_source, name + ": " + ec.message()};
        if (!std::filesystem::is_regular_file(path, ec))
            return {Error::malformed, name + ": expected a regular shard"};
        std::shared_ptr<const weight::Source> source;
        if (auto status = weight::open_source(path.string(), source, name); !status) return status;
        weight::TensorIndex shard;
        constexpr weight::SafetensorsIndexLimits limits{
            2U * 1024U * 1024U, 100000U, 5000U, 64U,
        };
        if (const auto status = weight::index_safetensors(std::move(source), shard, limits); !status)
            return {status.code, name + ": " + status.detail};
        return add_shard(shard, inventory);
    };
    for (unsigned index = 1; index <= kMainShardCount; ++index)
        if (auto status = inspect(main_shard_name(index)); !status) return status;
    if (auto status = inspect(std::string(kFlashIndexerShard)); !status) return status;
    if (auto status = check_manifest(root / "model.safetensors.index.json", inventory); !status)
        return status;
    inventory.missing_known = missing_known_tensor_names(inventory);
    inventory.missing_known_tensors = inventory.missing_known.size();
    out = std::move(inventory);
    return {Error::ok, {}};
}

} // namespace strix::ernie
