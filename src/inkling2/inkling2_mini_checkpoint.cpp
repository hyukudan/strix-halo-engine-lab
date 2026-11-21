#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_expert_layout.hpp"
#include "inkling2_mini_schedule.hpp"

#include <initializer_list>
#include <utility>

namespace strix::inkling2 {
namespace {

using Extents = std::initializer_list<std::uint64_t>;
using weight::Error;

weight::TensorSchema text_schema(std::span<const MiniTextTensorSpec> specs) {
    return {weight::Container::safetensors, "safetensors@1", specs, "model.visual."};
}

}  // namespace

weight::microscaled::ExpertSlabNames mini_expert_slab_names(unsigned physical, MiniExpertProjection projection) {
    const auto stem = std::string(kMiniTextTensorPrefix) + "layers." + std::to_string(physical) + ".mlp.experts." +
                      (projection == MiniExpertProjection::gate_up ? "w13_weight" : "w2_weight");
    return {stem, stem + "_scale", stem + "_amax", stem + "_outer_scale_denominator"};
}

std::vector<MiniTextTensorSpec> mini_text_tensor_schema() {
    std::vector<MiniTextTensorSpec> specs;
    const auto tensor = [&](const std::string& suffix, std::string_view dtype, Extents extents,
                            std::optional<weight::Role> role = {}) {
        specs.push_back({std::string(kMiniTextTensorPrefix) + suffix, dtype, extents, true, role});
    };
    const auto bf16 = [&](const std::string& suffix, Extents extents, std::optional<weight::Role> role = {}) {
        tensor(suffix, "BF16", extents, role);
    };
    bf16("embed.weight", {kVocabulary, kHiddenWidth});
    bf16("embed_norm.weight", {kHiddenWidth});
    bf16("norm.weight", {kHiddenWidth});
    bf16("unembed.weight", {kVocabulary, kHiddenWidth}, weight::Role::head);
    tensor("attn_res.output_query", "F32", {kHiddenWidth});
    tensor("middle_cycle_cross_pass_queries", "F32", {2U * MiniLayers::core_width, kHiddenWidth});
    tensor("middle_cycle_cross_pass_kv_queries", "F32", {2U * MiniLayers::core_width, kHiddenWidth});

    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
        const auto id = std::to_string(layer);
        const auto prefix = "layers." + id + ".";
        const auto layer_bf16 = [&](const std::string& suffix, Extents extents, std::optional<weight::Role> role = {}) {
            bf16(prefix + suffix, extents, role);
        };
        // The stored index names the preceding layer. The first layer uses the embedding
        // directly, and each full-attention layer has separate Q, K and V queries.
        if (layer + 1U < kPhysicalLayers) {
            const bool next_is_full = is_full_attention(layer + 1U);
            tensor("attn_res.attn_queries." + id, "F32",
                   next_is_full ? Extents{3, kHiddenWidth} : Extents{kHiddenWidth});
        }
        tensor("attn_res.mlp_queries." + id, "F32", {kHiddenWidth});

        layer_bf16("attn.wq_du.weight", {kQueryWidth, kHiddenWidth}, weight::Role::attention_q);
        layer_bf16("attn.wk_dv.weight", {kKvWidth, kHiddenWidth}, weight::Role::attention_k);
        layer_bf16("attn.wv_dv.weight", {kKvWidth, kHiddenWidth}, weight::Role::attention_v);
        layer_bf16("attn.wo_ud.weight", {kHiddenWidth, kQueryWidth}, weight::Role::attention_o);
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

        const auto expert_matrix = [&](MiniExpertProjection projection) {
            const auto names = mini_expert_slab_names(layer, projection);
            const auto [columns, rows] = mini_expert_geometry(projection);
            const auto expert_tensor = [&](const std::string& name, std::string_view dtype, Extents extents,
                                           bool resident = true) {
                specs.push_back({name, dtype, extents, resident});
            };
            // NVFP4 stores two E2M1 values per byte and one E4M3 scale per 16 values.
            // Its outer multiplier is amax[expert] / denominator[0].
            using namespace weight::microscaled;
            if (mini_expert_format(layer) == MatrixFormat::nvfp4) {
                expert_tensor(names.codes, "U8", {kExperts, rows, columns / 2});
                expert_tensor(names.block_scales, "F8_E4M3", {kExperts, rows, columns / kNvfp4BlockColumns});
                expert_tensor(names.amax, "F32", {kExperts}, false);
                expert_tensor(names.denominator, "F32", {1}, false);
            } else {
                // MXFP8 stores E4M3 values and unsigned E8M0 scale bytes per 32 columns.
                expert_tensor(names.codes, "F8_E4M3", {kExperts, rows, columns});
                expert_tensor(names.block_scales, "U8", {kExperts, rows, columns / kMxfp8BlockColumns});
            }
        };
        expert_matrix(MiniExpertProjection::gate_up);
        expert_matrix(MiniExpertProjection::down);
    }
    return specs;
}

weight::Status validate_mini_text_checkpoint(const weight::TensorIndex& index, MiniCheckpointSummary& out) {
    const auto specs = mini_text_tensor_schema();
    weight::TensorSchemaSummary summary;
    if (const auto status = weight::validate_tensor_schema(index, text_schema(specs), summary); !status)
        return status;
    out = {summary.tensors, summary.ignored_tensors, summary.bytes};
    return {Error::ok, {}};
}

weight::Status validate_mini_checkpoint_index(const weight::TensorIndex& index, std::string_view index_json) {
    return weight::validate_safetensors_shard_manifest(index, index_json, kMiniCheckpointShard);
}

weight::Status MiniTextWeights::open(const std::string& path) {
    const auto specs = mini_text_tensor_schema();
    return tensors_.open_safetensors(path, text_schema(specs));
}

weight::Status MiniTextWeights::bind(weight::TensorIndex index) {
    const auto specs = mini_text_tensor_schema();
    return tensors_.bind(std::move(index), text_schema(specs));
}

weight::Status MiniTextWeights::expert_layer(unsigned physical, MiniExpertProjection projection,
                                            MiniExpertLayerView& out) const {
    if (physical >= kPhysicalLayers)
        return {Error::unsupported_shape, "Mini expert address outside checkpoint"};
    const auto geometry = mini_expert_geometry(projection);
    if (geometry.inputs == 0)
        return {Error::unsupported_role, "invalid Mini expert projection"};

    MiniExpertLayerView result{mini_expert_format(physical), mini_expert_row_order(projection),
                               kExperts, geometry.outputs, geometry.inputs, {}, {}, {}, {}};
    if (const auto status = result.bind_sources(tensors_, mini_expert_slab_names(physical, projection)); !status)
        return status;
    out = std::move(result);
    return {Error::ok, {}};
}

weight::Status MiniTextWeights::expert_matrix(unsigned physical, unsigned expert, MiniExpertProjection projection,
                                             weight::microscaled::MatrixView& out) const {
    if (expert >= kExperts) return {Error::unsupported_shape, "Mini expert address outside checkpoint"};
    MiniExpertLayerView slab;
    if (const auto status = expert_layer(physical, projection, slab); !status) return status;
    return slab.matrix(expert, out);
}

}  // namespace strix::inkling2
