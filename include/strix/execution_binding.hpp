#pragma once

#include "strix/execution_plan.hpp"
#include "strix/kernel_registry.hpp"
#include "strix/model_contract.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strix {

struct ExecutionBindingRequest {
    std::uint32_t logical_rows = 1;
    std::uint32_t request_concurrency = 1;
    bool require_graph_safe = false;
    kernels::QualificationStatus minimum_qualification =
        kernels::QualificationStatus::oracle_matched;
    // Explicit C1 device-input graph topology; requires graph-safe selections.
    bool device_graph_inputs = false;
    bool prefetch_q8_lm_head = false;
    bool wave8_q8_lm_head = false;
    bool wave8_q8_gdn_input = false;
    bool wave8_q8_gdn_output = false;
    bool wave8_q8_qsa_output = false;
    bool parallel_moe_router = false;
    bool ranked_moe_router_topk = false;
    bool wave_moe_router_topk = false;
    bool prefetch_q8_qsa_input = false;
    std::vector<std::uint32_t> qsa_input_prefetch_layers;
    bool q4k_a8g32_gate_up = false;
    bool batched_moe_activation = false;
    bool grouped_exact_q51_down = false;
    bool lane_chain_hyper_output = false;
    bool prefetch_hc_projection_next_tile = false;
    std::vector<std::uint32_t> lane_chain_hyper_output_eligible_steps;
    bool deferred_moe_status = false;
    bool prefetch_q8_shared_expert = false;
    std::vector<std::uint32_t> shared_expert_prefetch_layers;
};

struct ExecutionKernelBinding {
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::optional<kernels::OperatorFamily> family;
    std::optional<std::string> kernel_id;
    bool host_or_runtime_step = false;
};

struct ExecutionBindingPlan {
    std::vector<ExecutionKernelBinding> bindings;
    std::uint32_t bound_steps = 0;
    std::uint32_t missing_steps = 0;
    std::uint32_t host_or_runtime_steps = 0;
};

[[nodiscard]] std::optional<kernels::OperatorFamily>
operator_family_for(SerialOp op) noexcept;

[[nodiscard]] kernels::ShapeClass execution_shape_for(
    const ModelContract& model,
    SerialOp op,
    const ExecutionBindingRequest& request);

[[nodiscard]] ExecutionBindingPlan bind_serial_execution_plan(
    const ModelContract& model,
    const SerialExecutionPlan& plan,
    const kernels::KernelRegistry& registry,
    const ExecutionBindingRequest& request);

}  // namespace strix
