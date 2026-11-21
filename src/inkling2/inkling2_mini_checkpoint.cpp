#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_config.hpp"
#include "inkling2_mini_schedule.hpp"
#include "strix/serve/json.hpp"

#include <initializer_list>
#include <map>

namespace strix::inkling2 {
namespace {

constexpr unsigned kMxFp8First = 20;

using Extents = std::initializer_list<std::uint64_t>;
using weight::Error;

}  // namespace

std::vector<MiniTextTensorSpec> mini_text_tensor_schema() {
    std::vector<MiniTextTensorSpec> specs;
    const auto tensor = [&](const std::string& suffix, std::string_view dtype, Extents extents) {
        specs.push_back({"model.llm." + suffix, dtype, extents});
    };
    const auto bf16 = [&](const std::string& suffix, Extents extents) {
        tensor(suffix, "BF16", extents);
    };
    bf16("embed.weight", {kVocabulary, kHiddenWidth});
    bf16("embed_norm.weight", {kHiddenWidth});
    bf16("norm.weight", {kHiddenWidth});
    bf16("unembed.weight", {kVocabulary, kHiddenWidth});
    tensor("attn_res.output_query", "F32", {kHiddenWidth});
    tensor("middle_cycle_cross_pass_queries", "F32", {2U * MiniLayers::core_width, kHiddenWidth});
    tensor("middle_cycle_cross_pass_kv_queries", "F32", {2U * MiniLayers::core_width, kHiddenWidth});

    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
        const auto id = std::to_string(layer);
        const auto prefix = "layers." + id + ".";
        const auto layer_bf16 = [&](const std::string& suffix, Extents extents) {
            bf16(prefix + suffix, extents);
        };
        // The stored index names the preceding layer. The first layer uses the embedding
        // directly, and each full-attention layer has separate Q, K and V queries.
        if (layer + 1U < kPhysicalLayers) {
            const bool next_is_full = is_full_attention(layer + 1U);
            tensor("attn_res.attn_queries." + id, "F32",
                   next_is_full ? Extents{3, kHiddenWidth} : Extents{kHiddenWidth});
        }
        tensor("attn_res.mlp_queries." + id, "F32", {kHiddenWidth});

        layer_bf16("attn.wq_du.weight", {kQueryWidth, kHiddenWidth});
        layer_bf16("attn.wk_dv.weight", {kKvWidth, kHiddenWidth});
        layer_bf16("attn.wv_dv.weight", {kKvWidth, kHiddenWidth});
        layer_bf16("attn.wo_ud.weight", {kHiddenWidth, kQueryWidth});
        layer_bf16("attn.wr_du.weight", {kQueryHeads * kRelativeWidth, kHiddenWidth});
        layer_bf16("attn.rel_logits_proj.proj", {kRelativeWidth, kRelativeExtent});
        layer_bf16("attn.q_l2_clip.scale", {kQueryHeads, kHeadWidth});
        layer_bf16("attn.rel_l2_clip.scale", {kQueryHeads, kRelativeWidth});
        layer_bf16("attn.attn_sink_proj.weight", {kQueryHeads, kHiddenWidth});
        layer_bf16("attn.gate_attn.weight", {kQueryWidth, kHiddenWidth});
        layer_bf16("attn_sconv.weight", {kHiddenWidth, 1, kConvTaps});
        layer_bf16("mlp_norm.weight", {kHiddenWidth});
        layer_bf16("mlp_sconv.weight", {kHiddenWidth, 1, kConvTaps});
        layer_bf16("mlp.gate.weight", {kExperts, kHiddenWidth});
        tensor(prefix + "mlp.gate.bias", "F32", {kExperts});

        if (is_full_attention(layer)) {
            for (std::string_view path : {"q", "k", "v"})
                layer_bf16(std::string(path) + "_input_norm.weight", {kHiddenWidth});
            for (std::string_view path : {"k", "v"}) {
                const auto stem = "attn." + std::string(path) + "_sconv.";
                layer_bf16(stem + "static_w", {kConvTaps, kKvWidth});
                layer_bf16(stem + "weight_proj", {kConvTaps * kKvHeads, kHiddenWidth});
            }
        } else {
            layer_bf16("attn_norm.weight", {kHiddenWidth});
            layer_bf16("attn.k_sconv.weight", {kKvWidth, 1, kConvTaps});
            layer_bf16("attn.v_sconv.weight", {kKvWidth, 1, kConvTaps});
        }

        if (is_core(layer)) {
            for (std::string_view path : {"attn", "k", "v", "mlp"}) {
                const auto stem = "controllers." + std::string(path) + ".";
                layer_bf16(stem + "down_weight", {kControllerRank, 2 * kHiddenWidth});
                layer_bf16(stem + "gain_weight", {kHiddenWidth, kControllerRank});
                layer_bf16(stem + "forget_weight", {kHiddenWidth, kControllerRank});
            }
        }

        const auto expert_matrix = [&](std::string_view name, std::uint64_t rows, std::uint64_t columns) {
            const auto stem = prefix + "mlp.experts." + std::string(name);
            // NVFP4 stores two E2M1 values per byte and one E4M3 scale per 16 values.
            // Its outer multiplier is amax[expert] / denominator[0].
            if (layer < kMxFp8First) {
                tensor(stem, "U8", {kExperts, rows, columns / 2});
                tensor(stem + "_scale", "F8_E4M3", {kExperts, rows, columns / 16});
                tensor(stem + "_amax", "F32", {kExperts});
                tensor(stem + "_outer_scale_denominator", "F32", {1});
            } else {
                // MXFP8 stores E4M3 values and unsigned E8M0 scale bytes per 32 columns.
                tensor(stem, "F8_E4M3", {kExperts, rows, columns});
                tensor(stem + "_scale", "U8", {kExperts, rows, columns / 32});
            }
        };
        expert_matrix("w13_weight", 2U * kExpertWidth, kHiddenWidth);
        expert_matrix("w2_weight", kHiddenWidth, kExpertWidth);
    }
    return specs;
}

weight::Status validate_mini_text_checkpoint(const weight::TensorIndex& index, MiniCheckpointSummary& out) {
    if (index.container != weight::Container::safetensors || index.schema != "safetensors@1")
        return {Error::unknown_format, "expected the original Mini safetensors layout"};

    MiniCheckpointSummary result;
    std::map<std::string_view, const weight::RawTensor*> text;
    for (const auto& tensor : index.tensors) {
        if (tensor.name.starts_with("model.visual.")) {
            ++result.vision_tensors;
            continue;
        }
        if (!tensor.name.starts_with("model.llm.") || !text.emplace(tensor.name, &tensor).second)
            return {Error::malformed, "unexpected or duplicate tensor " + tensor.name};
    }

    for (const auto& spec : mini_text_tensor_schema()) {
        const auto found = text.find(spec.name);
        if (found == text.end()) return {Error::malformed, "missing tensor " + spec.name};
        const auto& actual = *found->second;
        if (actual.wire_dtype != spec.wire_dtype)
            return {Error::unknown_format, "unexpected dtype " + actual.wire_dtype + " for " + spec.name};
        if (actual.extents != spec.extents)
            return {Error::unsupported_shape, "unexpected shape for " + spec.name};
        result.text_bytes += actual.payload.range.bytes;
        ++result.text_tensors;
        text.erase(found);
    }
    if (!text.empty()) return {Error::malformed, "unexpected text tensor " + std::string(text.begin()->first)};
    out = result;
    return {Error::ok, {}};
}

weight::Status validate_mini_checkpoint_index(const weight::TensorIndex& index, std::string_view index_json) {
    try {
        const auto manifest = serve::parse_json_unique(index_json);
        const auto* metadata = manifest.find("metadata");
        const auto* total_size = metadata ? metadata->find("total_size") : nullptr;
        const auto* weights = manifest.find("weight_map");
        if (!total_size || !weights || weights->as_object().size() != index.tensors.size())
            return {Error::malformed, "Mini index does not describe every tensor in the shard"};

        std::map<std::string_view, std::string_view> shards;
        for (const auto& [tensor, shard] : weights->as_object())
            shards.emplace(tensor, shard.as_string());

        std::uint64_t payload_bytes = 0;
        for (const auto& tensor : index.tensors) {
            const auto found = shards.find(tensor.name);
            if (found == shards.end() || found->second != kMiniCheckpointShard)
                return {Error::malformed, "missing or incorrect shard mapping for " + tensor.name};
            payload_bytes += tensor.payload.range.bytes;
        }
        const auto declared_bytes = total_size->as_int();
        if (declared_bytes < 0 || static_cast<std::uint64_t>(declared_bytes) != payload_bytes)
            return {Error::malformed, "Mini index total_size disagrees with the shard payload"};
        return {Error::ok, {}};
    } catch (const serve::JsonError& error) {
        return {Error::malformed, error.what()};
    }
}

}  // namespace strix::inkling2
