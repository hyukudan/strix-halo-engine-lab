#pragma once

#include "strix/model_contract.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/q8_projection_organization.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix::q8_projection_organization {
struct WeightRequirement {
    std::uint32_t tensor_index;
    std::uint64_t bytes;
};
struct ManifestPlan {
    std::uint64_t manifest_identity = 0;
    std::vector<WeightRequirement> weights;
};

// Whole-model admission, before residency or the first state update. Keep the
// launcher's conservative 16-byte view contract; do not infer wide-load support
// for less-aligned files from their general GGUF validity.
inline ManifestPlan build_manifest_plan(const ModelContract& model, const ModelManifest& manifest,
                                       bool lm_prefetch, bool gdn_wave8) {
    ManifestPlan plan;
    if (!lm_prefetch && !gdn_wave8) return plan;
    if (!model.validate().empty() || !manifest.layout_identity ||
        manifest.tensors.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Q8 organization requires a valid model manifest");
    plan.manifest_identity = manifest.layout_identity;
    const auto add_weight = [&](std::size_t index, std::uint64_t rows) {
        validate_geometry(model.hidden_size, rows);
        const auto& tensor = manifest.tensors[index];
        const auto bytes = (model.hidden_size / 32U) * 34U * rows;
        if (tensor.ggml_type != 8U || tensor.dimensions.size() != 2U ||
            tensor.dimensions[0] != model.hidden_size || tensor.dimensions[1] != rows ||
            tensor.absolute_offset % 16U || tensor.byte_size < bytes ||
            tensor.byte_size > std::numeric_limits<std::uint64_t>::max() - tensor.absolute_offset)
            throw std::invalid_argument("Q8 organization weight type, shape, alignment or extent invalid");
        plan.weights.push_back({static_cast<std::uint32_t>(index), bytes});
    };
    if (lm_prefetch) {
        std::size_t heads = 0;
        for (std::size_t i = 0; i < manifest.tensors.size(); ++i)
            if (manifest.tensors[i].role == TensorRole::output_head) {
                ++heads;
                add_weight(i, model.vocab_size);
            }
        if (heads != 1U) throw std::invalid_argument("Q8 organization requires exactly one output head");
    }
    if (gdn_wave8) {
        // beta and gate are adjacent FP32 arrays. Every later projection target
        // must retain the same 16-byte alignment as the first workspace slice.
        if (!model.full_attention_interval || model.gdn_value_heads % 4U)
            throw std::invalid_argument("wave8 GDN workspace head count must align FP32 projection slices to16");
        if (model.gdn_key_heads > 248320U / model.gdn_key_head_dim ||
            model.gdn_value_heads > 248320U / model.gdn_value_head_dim)
            throw std::invalid_argument("wave8 GDN projection widths exceed admitted bounds");
        const auto key_width = model.gdn_key_heads * model.gdn_key_head_dim;
        const auto value_width = model.gdn_value_heads * model.gdn_value_head_dim;
        if (key_width > (248320U - value_width) / 2U || key_width % 4U || value_width % 4U)
            throw std::invalid_argument("wave8 GDN workspace projection widths are invalid or misaligned");
        validate_geometry(model.hidden_size, key_width * 2U + value_width);
        for (std::uint64_t layer = 0; layer < model.hidden_layers; ++layer) {
            if ((layer + 1U) % model.full_attention_interval == 0U) continue;
            std::array<unsigned, 4> counts{};
            for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
                const auto& tensor = manifest.tensors[i];
                if (tensor.layer != layer) continue;
                std::size_t family = counts.size();
                if (tensor.role == TensorRole::attention_projection) {
                    if (tensor.name.ends_with(".attn_qkv.weight")) family = 0;
                    else if (tensor.name.ends_with(".attn_gate.weight")) family = 1;
                } else if (tensor.role == TensorRole::recurrent_state) {
                    if (tensor.name.ends_with(".ssm_beta.weight")) family = 2;
                    else if (tensor.name.ends_with(".ssm_alpha.weight")) family = 3;
                }
                if (family == counts.size()) continue;
                ++counts[family];
                if (family < 2U || tensor.ggml_type == 8U)
                    add_weight(i, family == 0U ? key_width * 2U + value_width :
                               family == 1U ? value_width : model.gdn_value_heads);
            }
            for (const auto count : counts)
                if (count != 1U) throw std::invalid_argument("wave8 GDN input family is missing or ambiguous");
        }
    }
    return plan;
}

// Recheck actual registered/copied addresses after residency, before any token
// launch. File-offset alignment alone is not evidence about the device view.
inline void validate_resident_weights(const ManifestPlan& plan, const ModelRuntimeBinding& binding) {
    if (!plan.manifest_identity) return;
    if (binding.manifest_layout_identity != plan.manifest_identity)
        throw std::invalid_argument("Q8 organization resident manifest identity changed");
    for (const auto& required : plan.weights) {
        const auto* tensor = find_runtime_tensor_binding(binding, required.tensor_index);
        if (!tensor || !runtime_tensor_gpu_visible(tensor->storage) || !tensor->allocation_generation ||
            !tensor->address || tensor->address % 16U || tensor->bytes < required.bytes ||
            tensor->bytes > std::numeric_limits<std::uintptr_t>::max() - tensor->address)
            throw std::invalid_argument("Q8 organization resident weight alignment or extent invalid");
    }
}
} // namespace strix::q8_projection_organization
