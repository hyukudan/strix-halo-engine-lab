#include "strix/execution_tensor_binding.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

void hash_text(std::uint64_t& hash, std::string_view value) noexcept {
    for (const unsigned char byte : value) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= fnv_prime;
    }
    hash_u64(hash, value.size());
}

bool layer_matches(const ManifestTensor& tensor,
                   const std::optional<std::uint32_t>& layer) noexcept {
    if (!layer.has_value()) {
        return !tensor.layer.has_value();
    }
    return tensor.layer.has_value() && *tensor.layer == *layer;
}

bool name_contains(const ManifestTensor& tensor,
                   std::string_view needle) noexcept {
    return tensor.name.find(needle) != std::string::npos;
}

bool role_matches_step(const ManifestTensor& tensor,
                       const ExecutionStep& step) noexcept {
    switch (step.op) {
    case SerialOp::token_embedding:
        return tensor.role == TensorRole::embedding &&
               !tensor.layer.has_value();

    case SerialOp::ple:
        return tensor.role == TensorRole::ngram_table &&
               (layer_matches(tensor, step.layer) ||
                !tensor.layer.has_value());

    case SerialOp::attn_hyper_mix:
    case SerialOp::mlp_hyper_mix:
        return layer_matches(tensor, step.layer) &&
               (tensor.role == TensorRole::normalization ||
                tensor.role == TensorRole::dense_projection);

    case SerialOp::attn_inject:
    case SerialOp::mlp_inject:
        return false;

    case SerialOp::gated_delta_net:
        return layer_matches(tensor, step.layer) &&
               (tensor.role == TensorRole::recurrent_state ||
                tensor.role == TensorRole::attention_projection ||
                tensor.role == TensorRole::dense_projection);

    case SerialOp::indexed_attention:
        return layer_matches(tensor, step.layer) &&
               (tensor.role == TensorRole::attention_projection ||
                tensor.role == TensorRole::sparse_index ||
                tensor.role == TensorRole::normalization);

    case SerialOp::moe_route:
        return layer_matches(tensor, step.layer) &&
               tensor.role == TensorRole::moe_router;

    case SerialOp::moe_experts:
        return layer_matches(tensor, step.layer) &&
               tensor.role == TensorRole::moe_expert;

    case SerialOp::shared_expert:
        return layer_matches(tensor, step.layer) &&
               tensor.role == TensorRole::dense_projection &&
               (name_contains(tensor, "shared_expert") ||
                name_contains(tensor, "shared_experts"));

    case SerialOp::final_hyper_mix:
        return !tensor.layer.has_value() &&
               (tensor.role == TensorRole::normalization ||
                tensor.role == TensorRole::dense_projection);

    case SerialOp::lm_head:
        return tensor.role == TensorRole::output_head &&
               !tensor.layer.has_value();

    case SerialOp::position_prepare:
    case SerialOp::hyper_expand:
    case SerialOp::attn_inject:
    case SerialOp::mlp_inject:
    case SerialOp::moe_combine:
        return false;
    }
    return false;
}

bool requires_unique_tensor(SerialOp op) noexcept {
    return op == SerialOp::token_embedding ||
           op == SerialOp::lm_head;
}

bool requires_tensor_family(SerialOp op) noexcept {
    switch (op) {
    case SerialOp::token_embedding:
    case SerialOp::ple:
    case SerialOp::attn_hyper_mix:
    case SerialOp::gated_delta_net:
    case SerialOp::indexed_attention:
    case SerialOp::mlp_hyper_mix:
    case SerialOp::moe_route:
    case SerialOp::moe_experts:
    case SerialOp::shared_expert:
    case SerialOp::final_hyper_mix:
    case SerialOp::lm_head:
        return true;

    case SerialOp::position_prepare:
    case SerialOp::hyper_expand:
    case SerialOp::moe_combine:
        return false;
    }
    return false;
}

std::string requirement_description(SerialOp op) {
    switch (op) {
    case SerialOp::token_embedding:
        return "requires one global embedding tensor";
    case SerialOp::ple:
        return "requires PLE/ngram tensor family";
    case SerialOp::attn_hyper_mix:
        return "requires layer hyper-mix tensor family";
    case SerialOp::gated_delta_net:
        return "requires layer recurrent/projection tensor family";
    case SerialOp::indexed_attention:
        return "requires layer indexed-attention tensor family";
    case SerialOp::attn_inject:
        return "consumes precomputed attention injection buffer";
    case SerialOp::mlp_hyper_mix:
        return "requires layer MLP hyper-mix tensor family";
    case SerialOp::moe_route:
        return "requires layer MoE router tensor family";
    case SerialOp::moe_experts:
        return "requires layer routed-expert tensor family";
    case SerialOp::shared_expert:
        return "requires layer shared-expert tensor family";
    case SerialOp::mlp_inject:
        return "consumes precomputed MLP injection buffer";
    case SerialOp::final_hyper_mix:
        return "requires final hyper-mix tensor family";
    case SerialOp::lm_head:
        return "requires one global output-head tensor";
    case SerialOp::position_prepare:
    case SerialOp::hyper_expand:
    case SerialOp::moe_combine:
        return "host/runtime step has no direct model tensor requirement";
    }
    return "unknown tensor requirement";
}

void hash_binding(std::uint64_t& hash,
                  const StepTensorBinding& binding,
                  const ModelManifest& manifest) noexcept {
    hash_u64(hash, binding.step_id);
    hash_u64(hash, static_cast<std::uint64_t>(binding.op));
    hash_u64(hash,
             binding.layer.has_value()
                 ? static_cast<std::uint64_t>(*binding.layer) + 1ULL
                 : 0ULL);
    hash_u64(hash, static_cast<std::uint64_t>(binding.resolution));
    for (const auto index : binding.tensor_indices) {
        hash_u64(hash, index);
        if (index < manifest.tensors.size()) {
            hash_text(hash, manifest.tensors[index].name);
        }
    }
    hash_u64(hash, 0xffffffffffffffffULL);
}

}  // namespace

ExecutionTensorBindingAudit audit_execution_tensor_bindings(
    const ModelManifest& manifest,
    const SerialExecutionPlan& plan) {
    if (manifest.layout_identity == 0) {
        throw std::invalid_argument(
            "execution tensor binding requires a non-zero manifest identity");
    }

    ExecutionTensorBindingAudit audit;
    audit.manifest_layout_identity = manifest.layout_identity;
    audit.steps.reserve(plan.steps.size());

    for (const auto& step : plan.steps) {
        StepTensorBinding binding;
        binding.step_id = step.id;
        binding.op = step.op;
        binding.layer = step.layer;
        binding.reason = requirement_description(step.op);

        if (!requires_tensor_family(step.op)) {
            binding.resolution = TensorBindingResolution::not_required;
            ++audit.not_required_steps;
            audit.steps.push_back(std::move(binding));
            continue;
        }

        for (std::uint32_t index = 0;
             index < manifest.tensors.size();
             ++index) {
            if (role_matches_step(manifest.tensors[index], step)) {
                binding.tensor_indices.push_back(index);
            }
        }

        if (binding.tensor_indices.empty()) {
            binding.resolution = TensorBindingResolution::missing;
            ++audit.missing_steps;
        } else if (requires_unique_tensor(step.op) &&
                   binding.tensor_indices.size() != 1U) {
            binding.resolution = TensorBindingResolution::ambiguous;
            ++audit.ambiguous_steps;
        } else {
            binding.resolution = TensorBindingResolution::resolved;
            ++audit.resolved_steps;
        }

        audit.steps.push_back(std::move(binding));
    }

    std::uint64_t hash = fnv_offset;
    hash_u64(hash, manifest.layout_identity);
    hash_u64(hash, audit.steps.size());
    for (const auto& binding : audit.steps) {
        hash_binding(hash, binding, manifest);
    }
    audit.binding_identity = hash;
    if (audit.binding_identity == 0) {
        throw std::logic_error(
            "execution tensor binding identity must be non-zero");
    }

    return audit;
}

const StepTensorBinding* find_step_tensor_binding(
    const ExecutionTensorBindingAudit& audit,
    std::uint32_t step_id) noexcept {
    const auto found = std::find_if(
        audit.steps.begin(), audit.steps.end(),
        [step_id](const StepTensorBinding& binding) {
            return binding.step_id == step_id;
        });
    return found == audit.steps.end() ? nullptr : &*found;
}

}  // namespace strix
