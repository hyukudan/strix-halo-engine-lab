#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>

namespace strix {

struct ExecutionBufferRequest {
    std::uint32_t logical_rows = 1;
    std::uint32_t activation_element_bytes = 2;
    std::uint32_t control_element_bytes = 4;
    std::uint32_t index_element_bytes = 4;
    std::uint32_t logit_element_bytes = 4;
    std::uint32_t mlp_intermediate_element_bytes = 4;
    bool parallel_moe_router = false;
};

struct ExecutionBufferPlan {
    std::uint32_t logical_rows = 0;

    std::uint64_t expanded_hidden_bytes = 0;
    std::uint64_t hidden_bytes = 0;
    std::uint64_t injection_weight_bytes = 0;
    std::uint64_t route_index_bytes = 0;
    std::uint64_t route_weight_bytes = 0;
    std::uint64_t logits_bytes = 0;
    std::uint64_t mlp_intermediate_bytes = 0;
    std::uint64_t gdn_workspace_bytes = 0;
    std::uint64_t ple_workspace_bytes = 0;
    std::uint64_t indexed_attention_workspace_bytes = 0;
    std::uint64_t hyper_workspace_bytes = 0;

    std::uint32_t expanded_hidden_buffers = 2;
    std::uint32_t hidden_buffers = 3;

    std::uint64_t peak_internal_bytes = 0;
    std::uint64_t peak_with_materialized_logits_bytes = 0;
    std::uint64_t router_workspace_bytes = 0;
};

[[nodiscard]] ExecutionBufferPlan build_serial_execution_buffer_plan(
    const ModelContract& model,
    const ExecutionBufferRequest& request);

}  // namespace strix
