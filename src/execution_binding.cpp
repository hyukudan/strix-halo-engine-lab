#include "strix/execution_binding.hpp"
#include "strix/qsa_input_prefetch.hpp"
#include "strix/shared_expert_prefetch.hpp"

#include <limits>
#include <algorithm>
#include "strix/lane_chain_hyper_output.hpp"
#include <stdexcept>
#include <utility>

namespace strix {

kernels::ShapeClass execution_shape_for(
    const ModelContract& model,
    SerialOp op,
    const ExecutionBindingRequest& request) {
    kernels::ShapeClass shape;
    shape.logical_rows = request.logical_rows;
    shape.request_concurrency = request.request_concurrency;
    shape.hidden_size =
        static_cast<std::uint32_t>(model.hidden_size);

    switch (op) {
    case SerialOp::gated_delta_net:
        shape.auxiliary_size =
            static_cast<std::uint32_t>(
                model.gdn_value_head_dim);
        break;
    case SerialOp::indexed_attention:
        shape.auxiliary_size =
            static_cast<std::uint32_t>(model.head_dim);
        break;
    case SerialOp::moe_route:
        shape.auxiliary_size =
            static_cast<std::uint32_t>(model.num_experts);
        break;
    case SerialOp::moe_experts:
    case SerialOp::shared_expert:
        shape.auxiliary_size =
            static_cast<std::uint32_t>(
                model.moe_intermediate_size);
        break;
    case SerialOp::lm_head:
        shape.auxiliary_size =
            static_cast<std::uint32_t>(model.vocab_size);
        break;
    case SerialOp::attn_hyper_mix:
    case SerialOp::mlp_hyper_mix:
    case SerialOp::final_hyper_mix:
    case SerialOp::hyper_expand:
    case SerialOp::attn_inject:
    case SerialOp::mlp_inject:
        shape.auxiliary_size =
            static_cast<std::uint32_t>(model.hc_count);
        break;
    case SerialOp::ple:
        shape.auxiliary_size = 16;
        break;
    default:
        shape.auxiliary_size = 0;
        break;
    }

    return shape;
}

namespace {

bool host_or_runtime_step(SerialOp op) noexcept {
    return op == SerialOp::position_prepare;
}

}  // namespace

std::optional<kernels::OperatorFamily>
operator_family_for(SerialOp op) noexcept {
    using kernels::OperatorFamily;
    switch (op) {
    case SerialOp::token_embedding:
        return OperatorFamily::embedding_lookup;
    case SerialOp::position_prepare:
        return std::nullopt;
    case SerialOp::hyper_expand:
        return OperatorFamily::hyper_expand;
    case SerialOp::ple:
        return OperatorFamily::sparse_lookup;
    case SerialOp::attn_hyper_mix:
    case SerialOp::mlp_hyper_mix:
    case SerialOp::final_hyper_mix:
        return OperatorFamily::hyper_mix;
    case SerialOp::gated_delta_net:
        return OperatorFamily::recurrent;
    case SerialOp::indexed_attention:
        return OperatorFamily::indexed_attention;
    case SerialOp::attn_inject:
    case SerialOp::mlp_inject:
        return OperatorFamily::elementwise_inject;
    case SerialOp::moe_route:
        return OperatorFamily::moe_route;
    case SerialOp::moe_experts:
        return OperatorFamily::moe_expert;
    case SerialOp::shared_expert:
        return OperatorFamily::shared_expert;
    case SerialOp::moe_combine:
        return OperatorFamily::moe_combine;
    case SerialOp::lm_head:
        return OperatorFamily::lm_head;
    }
    return std::nullopt;
}

ExecutionBindingPlan bind_serial_execution_plan(
    const ModelContract& model,
    const SerialExecutionPlan& plan,
    const kernels::KernelRegistry& registry,
    const ExecutionBindingRequest& request) {
    if (const auto error = model.validate(); !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.logical_rows == 0 ||
        request.request_concurrency == 0) {
        throw std::invalid_argument(
            "execution binding rows and concurrency must be non-zero");
    }

    constexpr auto maximum_shape =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max());
    if (model.hidden_size > maximum_shape ||
        model.gdn_value_head_dim > maximum_shape ||
        model.head_dim > maximum_shape ||
        model.num_experts > maximum_shape ||
        model.moe_intermediate_size > maximum_shape ||
        model.vocab_size > maximum_shape ||
        model.hc_count > maximum_shape) {
        throw std::overflow_error(
            "execution binding dimensions exceed registry shape fields");
    }

    qsa_input_prefetch::validate_layers(model,request.prefetch_q8_qsa_input,request.qsa_input_prefetch_layers);
    if(request.lane_chain_hyper_output && request.lane_chain_hyper_output_eligible_steps.empty())
        throw std::invalid_argument("lane-chain HC output has no eligible steps");
    if(!request.lane_chain_hyper_output && !request.lane_chain_hyper_output_eligible_steps.empty())
        throw std::invalid_argument("lane-chain HC output step list requires opt-in");
    for(const auto id:request.lane_chain_hyper_output_eligible_steps){
        const auto count=std::count(request.lane_chain_hyper_output_eligible_steps.begin(),request.lane_chain_hyper_output_eligible_steps.end(),id);
        if(count!=1 || std::none_of(plan.steps.begin(),plan.steps.end(),[&](const auto& step){return step.id==id && (step.op==SerialOp::attn_hyper_mix || step.op==SerialOp::mlp_hyper_mix || step.op==SerialOp::final_hyper_mix);}))
            throw std::invalid_argument("lane-chain HC output invalid eligible step list");
    }
    shared_expert_prefetch::validate_layers(model.hidden_layers,request.prefetch_q8_shared_expert,request.shared_expert_prefetch_layers);
    ExecutionBindingPlan output;
    if (request.offline_tuning && request.offline_tuning->step_keys.size() != plan.steps.size())
        throw std::invalid_argument("offline tuning requires one provenance key per compiled step");
    output.bindings.reserve(plan.steps.size());

    for (const auto& step : plan.steps) {
        ExecutionKernelBinding binding;
        binding.step_id = step.id;
        binding.op = step.op;

        if (host_or_runtime_step(step.op)) {
            binding.host_or_runtime_step = true;
            ++output.host_or_runtime_steps;
            output.bindings.push_back(std::move(binding));
            continue;
        }

        binding.family = operator_family_for(step.op);
        if (!binding.family.has_value()) {
            ++output.missing_steps;
            output.bindings.push_back(std::move(binding));
            continue;
        }

        kernels::KernelQuery query;
        query.family = *binding.family;
        query.phase = kernels::ExecutionPhase::decode;
        query.shape = execution_shape_for(model, step.op, request);
        query.require_graph_safe =
            request.require_graph_safe;
        query.require_row_invariant = true;
        query.require_deterministic = true;
        query.minimum_qualification =
            request.minimum_qualification;

        const auto* selected = registry.select(query);
        if(selected && step.op==SerialOp::indexed_attention){
            const bool wanted=qsa_input_prefetch::selected(request.prefetch_q8_qsa_input,request.qsa_input_prefetch_layers,step.layer);
            const bool actual=selected->id.ends_with(qsa_input_prefetch::kernel_suffix);
            if(wanted!=actual){std::string id=selected->id;if(actual)id.resize(id.size()-qsa_input_prefetch::kernel_suffix.size());else id+=qsa_input_prefetch::kernel_suffix;
                selected=registry.find(id);if(selected && !kernels::matches(*selected,query))selected=nullptr;}
        }
        if(*binding.family==kernels::OperatorFamily::hyper_mix){
            const bool want=request.lane_chain_hyper_output && std::find(request.lane_chain_hyper_output_eligible_steps.begin(),request.lane_chain_hyper_output_eligible_steps.end(),step.id)!=request.lane_chain_hyper_output_eligible_steps.end();
            if(selected && selected->id.ends_with(lane_chain_hyper_output_suffix)!=want){
                const auto other=want?selected->id+lane_chain_hyper_output_suffix:selected->id.substr(0,selected->id.size()-std::string_view(lane_chain_hyper_output_suffix).size());
                selected=registry.find(other);
                if(selected && !kernels::matches(*selected,query))selected=nullptr;
            }
        }
        if(selected && step.op==SerialOp::shared_expert){
            const bool wanted=shared_expert_prefetch::selected(request.prefetch_q8_shared_expert,request.shared_expert_prefetch_layers,step.layer);
            const bool actual=selected->id.ends_with(shared_expert_prefetch::kernel_suffix);
            if(wanted!=actual){std::string id=selected->id;if(actual)id.resize(id.size()-shared_expert_prefetch::kernel_suffix.size());else id+=shared_expert_prefetch::kernel_suffix;selected=registry.find(id);if(selected && !kernels::matches(*selected,query))selected=nullptr;}
        }
        if (request.offline_tuning && selected) {
            if (step.id >= request.offline_tuning->step_keys.size())
                throw std::invalid_argument("offline tuning step identity is outside its provenance keys");
            const auto tuned = registry.select_tuned(query,
                request.offline_tuning->step_keys[step.id], request.offline_tuning->records);
            binding.tuning_selection_reason = tuned.reason;
            if (tuned.reason == kernels::TuningSelectionReason::qualified_record) {
                // Offline costs cannot reverse a layer-scoped user choice or
                // use evidence for another resolved numerical contract.
                const auto* candidate = tuned.variant;
                bool compatible = candidate && selected->numerical_contract_identity != 0 &&
                    candidate->numerical_contract_identity == selected->numerical_contract_identity;
                if (candidate && step.op == SerialOp::indexed_attention)
                    compatible = compatible && candidate->id.ends_with(qsa_input_prefetch::kernel_suffix) ==
                        selected->id.ends_with(qsa_input_prefetch::kernel_suffix);
                if (candidate && *binding.family == kernels::OperatorFamily::hyper_mix)
                    compatible = compatible && candidate->id.ends_with(lane_chain_hyper_output_suffix) ==
                        selected->id.ends_with(lane_chain_hyper_output_suffix);
                if (candidate && step.op == SerialOp::shared_expert)
                    compatible = compatible && candidate->id.ends_with(shared_expert_prefetch::kernel_suffix) ==
                        selected->id.ends_with(shared_expert_prefetch::kernel_suffix);
                if (compatible) selected = candidate;
                else binding.tuning_selection_reason = kernels::TuningSelectionReason::selection_policy_mismatch;
            }
        }
        if (selected == nullptr) {
            ++output.missing_steps;
        } else {
            binding.kernel_id = selected->id;
            ++output.bound_steps;
        }

        output.bindings.push_back(std::move(binding));
    }

    return output;
}

}  // namespace strix
