#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/q8_projection_organization.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::q8_projection_organization {
struct WeightRequirement {
    std::uint32_t tensor_index;
    std::uint64_t bytes;
};
struct OutputWeight {
    std::uint32_t tensor_index;
    std::uint32_t layer;
    std::uint32_t width;
    std::uint32_t rows;
    std::uint64_t bytes;
    SerialOp consumer;
};
struct ManifestPlan {
    std::uint64_t manifest_identity = 0;
    std::vector<WeightRequirement> weights;
    std::vector<OutputWeight> outputs;
    bool gdn_output_wave8 = false;
    bool qsa_output_wave8 = false;
};

inline std::uint64_t output_checked_product(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::invalid_argument("Q8 output projection geometry product overflows");
    return a * b;
}
inline std::uint64_t output_checked_sum(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::invalid_argument("Q8 output projection geometry sum overflows");
    return a + b;
}
inline std::uint64_t gdn_output_core_offset_bytes(const ModelContract& model) {
    const auto key_width = output_checked_product(model.gdn_key_heads, model.gdn_key_head_dim);
    const auto value_width = output_checked_product(model.gdn_value_heads, model.gdn_value_head_dim);
    return output_checked_product(8U,
        output_checked_sum(output_checked_sum(key_width, value_width), model.gdn_value_heads));
}

// Whole-model admission, before residency or the first state update. Keep the
// launcher's conservative 16-byte view contract; do not infer wide-load support
// for less-aligned files from their general GGUF validity.
inline ManifestPlan build_manifest_plan(const ModelContract& model, const ModelManifest& manifest,
                                       bool lm_prefetch, bool gdn_wave8,
                                       bool gdn_output_wave8 = false, bool qsa_output_wave8 = false) {
    ManifestPlan plan;
    if (!lm_prefetch && !gdn_wave8 && !gdn_output_wave8 && !qsa_output_wave8) return plan;
    if (!model.validate().empty() || !manifest.layout_identity ||
        manifest.tensors.size() > std::numeric_limits<std::uint32_t>::max() ||
        model.hidden_layers > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Q8 organization requires a valid model manifest");
    plan.manifest_identity = manifest.layout_identity;
    const auto add_weight = [&](std::size_t index, std::uint64_t width, std::uint64_t rows) {
        validate_geometry(width, rows);
        const auto& tensor = manifest.tensors[index];
        const auto bytes = (width / 32U) * 34U * rows;
        if (tensor.ggml_type != 8U || tensor.dimensions.size() != 2U ||
            tensor.dimensions[0] != width || tensor.dimensions[1] != rows ||
            tensor.absolute_offset % 16U || tensor.byte_size < bytes ||
            tensor.byte_size > std::numeric_limits<std::uint64_t>::max() - tensor.absolute_offset)
            throw std::invalid_argument("Q8 organization weight type, shape, alignment or extent invalid");
        plan.weights.push_back({static_cast<std::uint32_t>(index), bytes});
        return bytes;
    };
    if (lm_prefetch) {
        std::size_t heads = 0;
        for (std::size_t i = 0; i < manifest.tensors.size(); ++i)
            if (manifest.tensors[i].role == TensorRole::output_head) {
                ++heads;
                add_weight(i, model.hidden_size, model.vocab_size);
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
                    if (tensor.name.find("attn_qkv") != std::string::npos) family = 0;
                    else if (tensor.name.find("attn_gate") != std::string::npos) family = 1;
                } else if (tensor.role == TensorRole::recurrent_state) {
                    // The launcher classifies convolution before beta/alpha.
                    if (tensor.name.find("ssm_conv1d") != std::string::npos) continue;
                    if (tensor.name.find("ssm_beta") != std::string::npos) family = 2;
                    else if (tensor.name.find("ssm_alpha") != std::string::npos) family = 3;
                }
                if (family == counts.size()) continue;
                ++counts[family];
                if (family < 2U || tensor.ggml_type == 8U)
                    add_weight(i, model.hidden_size, family == 0U ? key_width * 2U + value_width :
                               family == 1U ? value_width : model.gdn_value_heads);
            }
            for (const auto count : counts)
                if (count != 1U) throw std::invalid_argument("wave8 GDN input family is missing or ambiguous");
        }
    }
    if (gdn_output_wave8 || qsa_output_wave8) {
        if (model.bytes_fp32 != 4U || model.bytes_bf16 != 2U)
            throw std::invalid_argument("Q8 output projection requires FP32 and BF16 storage sizes");
        plan.gdn_output_wave8 = gdn_output_wave8;
        plan.qsa_output_wave8 = qsa_output_wave8;
        const auto add_outputs = [&](SerialOp consumer) {
            const bool gdn = consumer == SerialOp::gated_delta_net;
            const auto width = gdn
                ? output_checked_product(model.gdn_value_heads, model.gdn_value_head_dim)
                : output_checked_product(model.attention_heads, model.head_dim);
            validate_geometry(width, model.hidden_size);
            if (gdn && gdn_output_core_offset_bytes(model) % 16U)
                throw std::invalid_argument("wave8 GDN output core slice is not aligned to16");
            std::uint64_t layers_seen = 0;
            for (std::uint32_t layer = 0; layer < model.hidden_layers; ++layer) {
                const bool is_gdn = (static_cast<std::uint64_t>(layer) + 1U) % model.full_attention_interval != 0U;
                if (is_gdn != gdn) continue;
                std::size_t count = 0;
                for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
                    const auto& tensor = manifest.tensors[i];
                    if (tensor.layer != layer) continue;
                    const auto contains = [&](const char* text) { return tensor.name.find(text) != std::string::npos; };
                    if (gdn) {
                        if (tensor.role != TensorRole::recurrent_state || !contains("ssm_")) continue;
                        // Match the launcher's branch priority, including names
                        // containing more than one family substring.
                        if (contains("ssm_conv1d") || contains("ssm_beta") || contains("ssm_alpha") ||
                            contains("ssm_dt") || contains("ssm_norm") || !contains("ssm_out")) continue;
                    } else if (tensor.role != TensorRole::attention_projection || !contains("attn_") ||
                               (!contains("attn_output") && !contains("o_proj"))) continue;
                    ++count;
                    const auto bytes = add_weight(i, width, model.hidden_size);
                    plan.outputs.push_back({static_cast<std::uint32_t>(i), layer,
                        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(model.hidden_size),
                        bytes, consumer});
                }
                if (count != 1U) throw std::invalid_argument("wave8 output projection family is missing or ambiguous");
                ++layers_seen;
            }
            if (layers_seen != (gdn ? model.gdn_layers : model.full_attention_layers))
                throw std::invalid_argument("wave8 output projection layer count differs from model");
        };
        if (gdn_output_wave8) add_outputs(SerialOp::gated_delta_net);
        if (qsa_output_wave8) add_outputs(SerialOp::indexed_attention);
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
// Bind every selected output weight to its real step and actual scratch views
// before persistent state initialization. This is setup-only; launchers repeat
// the same constant-time dtype/view check before enqueuing their work.
inline void validate_output_steps(const ManifestPlan& plan, const ModelContract& model,
                                  const CompiledSerialExecution& compiled,
                                  const ExecutionTensorAccessPlan& tensor_access,
                                  const ExecutionMemoryBinding& memory,
                                  const ModelRuntimeBinding& binding) {
    if (!plan.gdn_output_wave8 && !plan.qsa_output_wave8) {
        if (!plan.outputs.empty()) throw std::invalid_argument("inactive output projection has planned weights");
        return;
    }
    if (!model.validate().empty() || model.bytes_fp32 != 4U || model.bytes_bf16 != 2U ||
        !plan.manifest_identity || tensor_access.manifest_layout_identity != plan.manifest_identity ||
        !tensor_access.plan_identity || compiled.logical_rows != 1U || compiled.request_concurrency != 1U ||
        compiled.device_graph_inputs || compiled.io.steps.size() != compiled.execution.steps.size())
        throw std::invalid_argument("wave8 output projection step admission identity or shape invalid");
    validate_resident_weights(plan, binding);
    validate_execution_memory_binding(compiled.execution_identity, compiled.arena, memory);
    const auto expected = (plan.gdn_output_wave8 ? model.gdn_layers : 0U) +
                          (plan.qsa_output_wave8 ? model.full_attention_layers : 0U);
    if (plan.outputs.size() != expected)
        throw std::invalid_argument("wave8 output projection plan coverage is incomplete");
    std::vector<unsigned> matches(plan.outputs.size(), 0U);
    const auto resolved = [&](ExecutionBufferUse use) {
        const auto result = resolve_execution_buffer(compiled.execution_identity, compiled.arena, memory, use);
        if (result.address < memory.arena_base || result.address - memory.arena_base > memory.arena_bytes ||
            result.bytes > memory.arena_bytes - (result.address - memory.arena_base) ||
            result.bytes > std::numeric_limits<std::uintptr_t>::max() - result.address)
            throw std::invalid_argument("wave8 output projection view exceeds its actual arena");
        return result;
    };
    for (std::size_t i = 0; i < compiled.execution.steps.size(); ++i) {
        const auto& step = compiled.execution.steps[i];
        const bool gdn = step.op == SerialOp::gated_delta_net;
        if (!(gdn ? plan.gdn_output_wave8 :
              step.op == SerialOp::indexed_attention && plan.qsa_output_wave8)) continue;
        if (!step.layer || *step.layer >= model.hidden_layers ||
            (((static_cast<std::uint64_t>(*step.layer) + 1U) % model.full_attention_interval != 0U) != gdn))
            throw std::invalid_argument("wave8 output projection step has an invalid model layer");
        std::size_t which = plan.outputs.size();
        for (std::size_t j = 0; j < plan.outputs.size(); ++j)
            if (plan.outputs[j].consumer == step.op && plan.outputs[j].layer == *step.layer) {
                if (which != plan.outputs.size())
                    throw std::invalid_argument("wave8 output projection weight is ambiguous for its step");
                which = j;
            }
        if (which == plan.outputs.size() || ++matches[which] != 1U)
            throw std::invalid_argument("wave8 output projection step coverage differs from its weights");
        const auto& output = plan.outputs[which];
        const auto width = gdn ? output_checked_product(model.gdn_value_heads, model.gdn_value_head_dim)
                               : output_checked_product(model.attention_heads, model.head_dim);
        validate_geometry(width, model.hidden_size);
        if (output.width != width || output.rows != model.hidden_size ||
            output.bytes != (width / 32U) * 34U * model.hidden_size)
            throw std::invalid_argument("wave8 output projection planned consumer geometry changed");
        const auto& io = compiled.io.steps[i];
        if (step.id != i || io.step_id != step.id || io.op != step.op ||
            io.reads.size() != 1U || io.writes.size() != 1U || io.temporaries.size() != 1U)
            throw std::invalid_argument("wave8 output projection IO differs from its step");
        const auto* access = find_execution_tensor_access(tensor_access, step.id, 0U);
        if (!access || access->op != step.op ||
            std::count_if(tensor_access.accesses.begin(), tensor_access.accesses.end(), [&](const auto& item) {
                return item.step_id == step.id && item.binding_slot == 0U;
            }) != 1 ||
            std::count(access->tensor_indices.begin(), access->tensor_indices.end(), output.tensor_index) != 1)
            throw std::invalid_argument("wave8 output projection weight differs from selected tensor access");
        (void)resolved(io.reads.front());
        const auto destination = resolved(io.writes.front());
        const auto temporary = resolved(io.temporaries.front());
        const auto offset = gdn ? gdn_output_core_offset_bytes(model)
                               : build_indexed_attention_workspace_layout(model).attention_output_offset;
        const auto source_bytes = width * 4U;
        if (offset > temporary.bytes || source_bytes > temporary.bytes - offset)
            throw std::invalid_argument("wave8 output projection source exceeds its workspace");
        const auto* weight = find_runtime_tensor_binding(binding, output.tensor_index);
        if (!weight || !runtime_tensor_gpu_visible(weight->storage) || !weight->allocation_generation)
            throw std::invalid_argument("wave8 output projection has no retained GPU-visible weight");
        validate_fp32_bf16_views(output.width, output.rows,
            {temporary.address + static_cast<std::uintptr_t>(offset), source_bytes},
            {weight->address, weight->bytes}, {destination.address, destination.bytes});
    }
    if (std::any_of(matches.begin(), matches.end(), [](unsigned count) { return count != 1U; }))
        throw std::invalid_argument("wave8 output projection weight has no unique compiled step");
}

} // namespace strix::q8_projection_organization
