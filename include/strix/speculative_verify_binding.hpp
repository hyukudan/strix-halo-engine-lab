#pragma once

#include "strix/execution_binding.hpp"
#include "strix/kernel_registry.hpp"
#include "strix/model_contract.hpp"
#include "strix/speculative_verify_plan.hpp"

#include <cstdint>

namespace strix {

struct SpeculativeVerifyBindingRequest {
    std::uint32_t request_concurrency = 1;
    bool require_graph_safe = false;
    kernels::QualificationStatus minimum_qualification =
        kernels::QualificationStatus::oracle_matched;
    bool parallel_moe_router = false;
    bool ranked_moe_router_topk = false;
    bool wave_moe_router_topk = false;
    bool prefetch_q8_qsa_input = false;
    bool lane_chain_hyper_output = false;
    bool deferred_moe_status = false;
};

[[nodiscard]] ExecutionBindingPlan
bind_speculative_verify_plan(
    const ModelContract& model,
    const SpeculativeVerifyPlan& plan,
    const kernels::KernelRegistry& registry,
    const SpeculativeVerifyBindingRequest& request = {});

}  // namespace strix
