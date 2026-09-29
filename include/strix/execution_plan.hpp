#pragma once

#include "strix/model_contract.hpp"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace strix {

enum class SerialOp {
    token_embedding,
    position_prepare,
    hyper_expand,
    ple,
    attn_hyper_mix,
    gated_delta_net,
    indexed_attention,
    attn_inject,
    mlp_hyper_mix,
    moe_route,
    moe_experts,
    shared_expert,
    moe_combine,
    mlp_inject,
    final_hyper_mix,
    lm_head,
};

enum class PersistentState : std::uint32_t {
    none = 0,
    attention_kv = 1U << 0U,
    recurrent_matrix = 1U << 1U,
    convolution_history = 1U << 2U,
    sparse_index = 1U << 3U,
    ngram_history = 1U << 4U,
    position_history = 1U << 5U,
};

[[nodiscard]] constexpr PersistentState operator|(
    PersistentState left,
    PersistentState right) noexcept {
    return static_cast<PersistentState>(
        static_cast<std::uint32_t>(left) |
        static_cast<std::uint32_t>(right));
}

[[nodiscard]] constexpr bool has_state(
    PersistentState value,
    PersistentState flag) noexcept {
    return (static_cast<std::uint32_t>(value) &
            static_cast<std::uint32_t>(flag)) != 0;
}

struct ExecutionStep {
    std::uint32_t id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::optional<std::uint32_t> layer;
    std::vector<std::uint32_t> dependencies;
    PersistentState reads_state = PersistentState::none;
    PersistentState writes_state = PersistentState::none;
    bool graph_safe_candidate = true;
    bool dynamic_shape = false;
};

struct SerialExecutionPlan {
    std::vector<ExecutionStep> steps;
    std::uint32_t linear_attention_steps = 0;
    std::uint32_t indexed_attention_steps = 0;
    std::uint32_t ple_steps = 0;
    std::uint32_t moe_route_steps = 0;
    std::uint32_t moe_expert_steps = 0;
};

[[nodiscard]] std::string_view to_string(SerialOp op) noexcept;

[[nodiscard]] SerialExecutionPlan build_serial_decode_execution_plan(
    const ModelContract& model);

}  // namespace strix
