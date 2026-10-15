#include "strix/speculative_verify_binding.hpp"

#include <stdexcept>
#include <utility>

namespace strix {

ExecutionBindingPlan
bind_speculative_verify_plan(
    const ModelContract& model,
    const SpeculativeVerifyPlan& plan,
    const kernels::KernelRegistry& registry,
    const SpeculativeVerifyBindingRequest& request) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (plan.candidate_depth == 0 ||
        plan.candidate_input_rows !=
            plan.candidate_depth ||
        plan.produced_logit_rows !=
            plan.candidate_depth ||
        request.request_concurrency == 0) {
        throw std::invalid_argument(
            "invalid speculative verify binding geometry");
    }

    ExecutionBindingRequest shape_request;
    shape_request.logical_rows =
        plan.candidate_depth;
    shape_request.request_concurrency =
        request.request_concurrency;
    shape_request.parallel_moe_router = request.parallel_moe_router;
    shape_request.ranked_moe_router_topk = request.ranked_moe_router_topk;
    shape_request.require_graph_safe =
        request.require_graph_safe;
    shape_request.minimum_qualification =
        request.minimum_qualification;

    ExecutionBindingPlan output;
    output.bindings.reserve(
        plan.steps.size());

    for (const auto& step :
         plan.steps) {
        ExecutionKernelBinding binding;
        binding.step_id = step.id;
        binding.op = step.op;

        if (step.op ==
            SerialOp::position_prepare) {
            binding.host_or_runtime_step = true;
            ++output.host_or_runtime_steps;
            output.bindings.push_back(
                std::move(binding));
            continue;
        }

        binding.family =
            operator_family_for(step.op);
        if (!binding.family.has_value()) {
            ++output.missing_steps;
            output.bindings.push_back(
                std::move(binding));
            continue;
        }

        kernels::KernelQuery query;
        query.family =
            *binding.family;
        query.phase =
            kernels::ExecutionPhase::
                speculative_verify;
        query.shape =
            execution_shape_for(
                model,
                step.op,
                shape_request);
        query.require_graph_safe =
            request.require_graph_safe;
        query.require_row_invariant = true;
        query.require_deterministic = true;
        query.minimum_qualification =
            request.minimum_qualification;

        const auto* selected =
            registry.select(query);
        if (selected == nullptr) {
            ++output.missing_steps;
        } else {
            binding.kernel_id =
                selected->id;
            ++output.bound_steps;
        }

        output.bindings.push_back(
            std::move(binding));
    }

    return output;
}

}  // namespace strix
