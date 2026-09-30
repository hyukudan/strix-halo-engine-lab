#pragma once

#include "strix/execution_plan.hpp"
#include "strix/model_contract.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace strix {

enum class SpeculativeVerifyMode {
    row_parallel,
    sequential_window,
    causal_window,
    dynamic_row_parallel,
    runtime_position_window,
};

struct SpeculativeVerifyStep {
    std::uint32_t id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::optional<std::uint32_t> layer;
    std::vector<std::uint32_t> dependencies;
    PersistentState reads_state = PersistentState::none;
    PersistentState writes_state = PersistentState::none;
    SpeculativeVerifyMode mode =
        SpeculativeVerifyMode::row_parallel;
    bool graph_safe_candidate = true;
    bool dynamic_shape = false;
};

struct SpeculativeVerifyPlan {
    std::uint32_t candidate_depth = 0;
    std::uint32_t candidate_input_rows = 0;
    std::uint32_t produced_logit_rows = 0;

    // target_tokens[0] is selected from logits that already exist at the base
    // committed state. Rows produced by this plan correspond to target_tokens
    // [1..candidate_depth].
    bool requires_base_target_token = true;

    std::vector<SpeculativeVerifyStep> steps;

    std::uint32_t row_parallel_steps = 0;
    std::uint32_t sequential_window_steps = 0;
    std::uint32_t causal_window_steps = 0;
    std::uint32_t dynamic_steps = 0;
};

[[nodiscard]] SpeculativeVerifyPlan
build_speculative_verify_plan(
    const ModelContract& model,
    std::uint32_t candidate_depth);

}  // namespace strix
