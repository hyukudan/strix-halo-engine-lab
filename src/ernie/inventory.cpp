#include "strix/ernie/inventory.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace strix::ernie {
namespace {
using enum Module;
using weight::Error;
using weight::Status;

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
    const auto stem = name.ends_with(".weight") ? name.substr(0, name.size() - 7U) : name;
    tensor(schema, stem + ".weight_scale_inv", matrix.module, "F32",
           (matrix.rows + 127U) / 128U, (matrix.columns + 127U) / 128U);
}

// The grouped output projection is the one FP8 matrix whose scale suffix uses
// an underscore after the tensor name itself.
void grouped_projection(Schema& schema, const std::string& prefix) {
    const auto name = prefix + "linear_o_group_proj";
    tensor(schema, name, hca, "F8_E4M3", 8192, 4096);
    tensor(schema, name + "_scale_inv", hca, "F32", 64, 32);
}

void add_attention(Schema& schema, const std::string& prefix, bool sparse) {
    tensor(schema, prefix + "vha_postmix_U", value_head_mix, "BF16", 64, 16);
    tensor(schema, prefix + "vha_postmix_V", value_head_mix, "BF16", 64, 16);
    fp8(schema, prefix, {"gate_proj.weight", attention_gate, sparse ? 16384U : 8192U, 1024});
    fp8(schema, prefix, {"o_proj.weight", sparse ? mla : hca, 4096, sparse ? 16384U : 8192U});
    const auto core = prefix + "core_attention.";
    if (sparse) {
        constexpr Matrix projections[]{
            {"q_a_proj.weight", mla, 1024, 4096}, {"q_b_proj.weight", mla, 16384, 1024},
            {"kv_a_proj_with_mqa.weight", mla, 576, 4096}, {"kv_b_proj.weight", mla, 28672, 512},
            {"core_attention.indexer.wq_b.weight", sparse_indexer, 8192, 1024},
        };
        for (const auto& projection : projections) fp8(schema, prefix, projection);
        tensor(schema, prefix + "q_a_layernorm.weight", mla, "BF16", 1024);
        tensor(schema, prefix + "kv_a_layernorm.weight", mla, "BF16", 512);
        tensor(schema, core + "softmax_offset", mla, "BF16", 64);
        const auto indexer = core + "indexer.";
        tensor(schema, indexer + "wk.weight", sparse_indexer, "BF16", 128, 4096);
        tensor(schema, indexer + "weights_proj.weight", sparse_indexer, "BF16", 64, 4096);
        tensor(schema, indexer + "k_norm.weight", sparse_indexer, "BF16", 128);
        tensor(schema, indexer + "k_norm.bias", sparse_indexer, "BF16", 128);
        tensor(schema, indexer + "flashindexer_bias", flash_indexer, "BF16", 64);
        tensor(schema, indexer + "linear_gate.weight", flash_indexer, "BF16", 64, 4096);
    } else {
        constexpr Matrix projections[]{
            {"linear_q_down_proj.weight", hca, 1024, 4096},
            {"linear_q_up_proj.weight", hca, 32768, 1024},
            {"linear_kv_proj.weight", hca, 512, 4096},
        };
        for (const auto& projection : projections) fp8(schema, prefix, projection);
        grouped_projection(schema, prefix);
        tensor(schema, prefix + "q_layernorm.weight", hca, "BF16", 1024);
        tensor(schema, prefix + "kv_layernorm.weight", hca, "BF16", 512);
        tensor(schema, core + "attn_sink", hca, "F32", 64);
        const auto comp = core + "compressor.";
        tensor(schema, comp + "ape", compressor, "F32", 128, 512);
        tensor(schema, comp + "linear_wgate.weight", compressor, "BF16", 512, 4096);
        tensor(schema, comp + "linear_wkv.weight", compressor, "BF16", 512, 4096);
        tensor(schema, comp + "norm.weight", compressor, "BF16", 512);
    }
}

void add_moe(Schema& schema, const std::string& prefix) {
    tensor(schema, prefix + "e_score_correction_bias", router, "F32", 512);
    tensor(schema, prefix + "gate.routed_scaling_factor_param", router, "F32", 512);
    tensor(schema, prefix + "gate.weight", router, "BF16", 512, 4096);
    tensor(schema, prefix + "gate.weight_1", router, "BF16", 512, 4096);
    tensor(schema, prefix + "latent_norm.weight", latent_projection, "BF16", 2048);
    constexpr Matrix projections[]{
        {"fc1_latent_proj.weight", latent_projection, 2048, 4096},
        {"fc2_latent_proj.weight", latent_projection, 4096, 2048},
        {"shared_experts.w1.weight", shared_expert, 2048, 4096},
        {"shared_experts.w2.weight", shared_expert, 4096, 2048},
        {"shared_experts.w3.weight", shared_expert, 2048, 4096},
    };
    for (const auto& projection : projections) fp8(schema, prefix, projection);
    for (unsigned expert = 0; expert < kExpertCount; ++expert) {
        const auto base = prefix + "experts." + std::to_string(expert) + ".";
        for (const auto* proj : {"w1", "w2", "w3"}) {
            const auto name = base + proj;
            tensor(schema, name + ".weight", routed_expert, "I8", 2048, 1024);
            tensor(schema, name + ".weight_scale_inv", routed_expert, "F8_E8M0", 2048, 64);
        }
    }
}

const Schema& known_schema() {
    static const Schema schema = [] {
        Schema rows;
        tensor(rows, "model.embed_tokens.weight", embedding, "BF16", 201216, 4096);
        for (unsigned layer = 0; layer < kLayerCount; ++layer) {
            const auto prefix = "model.layers." + std::to_string(layer) + ".";
            tensor(rows, prefix + "input_layernorm.weight", layer_norm, "BF16", 4096);
            tensor(rows, prefix + "post_attention_layernorm.weight", layer_norm, "BF16", 4096);
            for (const auto* hc : {"self_attention_hyper_connection.", "mlp_hyper_connection."}) {
                const auto base = prefix + hc;
                for (const auto* scalar : {"alpha_pre", "alpha_post", "alpha_res"})
                    tensor(rows, base + scalar, hyper_connection, "F32", 1);
                tensor(rows, base + "bias", hyper_connection, "F32", 24);
                tensor(rows, base + "mapping_proj.weight", hyper_connection, "F32", 16384, 24);
            }
            add_attention(rows, prefix + "self_attn.",
                          std::ranges::find(kMlaLayers, std::int64_t{layer}) != std::ranges::end(kMlaLayers));
            if (layer == 0) {
                constexpr Matrix dense[]{
                    {"mlp.gate_proj.weight", dense_mlp, 14336, 4096},
                    {"mlp.up_proj.weight", dense_mlp, 14336, 4096},
                    {"mlp.down_proj.weight", dense_mlp, 4096, 14336},
                };
                for (const auto& projection : dense) fp8(rows, prefix, projection);
            } else {
                add_moe(rows, prefix + "block_sparse_moe.");
            }
        }
        return rows;
    }();
    return schema;
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
    }
    return "unclassified";
}

std::string main_shard_name(unsigned index) {
    if (index == 0 || index > kMainShardCount) throw std::out_of_range("ERNIE shard index");
    auto number = std::to_string(index);
    number.insert(0, 5U - number.size(), '0');
    return "model-" + number + "-of-00128.safetensors";
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
        if ((module == flash_indexer) != (shard_name == kFlashIndexerShard))
            return {Error::malformed, "unexpected FlashIndexer shard mapping for " + raw.name};
        pending.emplace(raw.name, Tensor{module, raw.wire_dtype, raw.extents, raw.payload});
    }
    inventory.tensors.merge(pending);
    return {Error::ok, {}};
}

std::size_t count_missing_known_tensors(const Inventory& inventory) {
    return static_cast<std::size_t>(std::ranges::count_if(known_schema(), [&](const auto& entry) {
        return !inventory.tensors.contains(entry.first);
    }));
}

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
    inventory.missing_known_tensors = count_missing_known_tensors(inventory);
    out = std::move(inventory);
    return {Error::ok, {}};
}

} // namespace strix::ernie
