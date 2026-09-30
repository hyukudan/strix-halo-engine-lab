#include "strix/speculative_verify_plan.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

SpeculativeVerifyMode mode_for(
    SerialOp op) noexcept {
    switch (op) {
    case SerialOp::position_prepare:
        return SpeculativeVerifyMode::runtime_position_window;
    case SerialOp::ple:
    case SerialOp::gated_delta_net:
        return SpeculativeVerifyMode::sequential_window;
    case SerialOp::indexed_attention:
        return SpeculativeVerifyMode::causal_window;
    case SerialOp::moe_experts:
        return SpeculativeVerifyMode::dynamic_row_parallel;
    case SerialOp::token_embedding:
    case SerialOp::hyper_expand:
    case SerialOp::attn_hyper_mix:
    case SerialOp::attn_inject:
    case SerialOp::mlp_hyper_mix:
    case SerialOp::moe_route:
    case SerialOp::shared_expert:
    case SerialOp::moe_combine:
    case SerialOp::mlp_inject:
    case SerialOp::final_hyper_mix:
    case SerialOp::lm_head:
        return SpeculativeVerifyMode::row_parallel;
    }
    return SpeculativeVerifyMode::row_parallel;
}

}  // namespace

SpeculativeVerifyPlan
build_speculative_verify_plan(
    const ModelContract& model,
    std::uint32_t candidate_depth) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (candidate_depth == 0) {
        throw std::invalid_argument(
            "speculative verify plan requires positive candidate depth");
    }

    const auto serial =
        build_serial_decode_execution_plan(
            model);

    SpeculativeVerifyPlan verify;
    verify.candidate_depth =
        candidate_depth;
    verify.candidate_input_rows =
        candidate_depth;
    verify.produced_logit_rows =
        candidate_depth;
    verify.steps.reserve(
        serial.steps.size());

    for (const auto& source :
         serial.steps) {
        SpeculativeVerifyStep step;
        step.id = source.id;
        step.op = source.op;
        step.layer = source.layer;
        step.dependencies =
            source.dependencies;
        step.reads_state =
            source.reads_state;
        step.writes_state =
            source.writes_state;
        step.mode =
            mode_for(source.op);
        step.graph_safe_candidate =
            source.graph_safe_candidate;
        step.dynamic_shape =
            source.dynamic_shape;

        switch (step.mode) {
        case SpeculativeVerifyMode::row_parallel:
            ++verify.row_parallel_steps;
            break;
        case SpeculativeVerifyMode::runtime_position_window:
            // Host/runtime position preparation is deterministic but still a
            // distinct mode rather than a GPU row kernel.
            break;
        case SpeculativeVerifyMode::sequential_window:
            ++verify.sequential_window_steps;
            break;
        case SpeculativeVerifyMode::causal_window:
            ++verify.causal_window_steps;
            break;
        case SpeculativeVerifyMode::dynamic_row_parallel:
            ++verify.row_parallel_steps;
            ++verify.dynamic_steps;
            break;
        }

        if (step.dynamic_shape &&
            step.mode !=
                SpeculativeVerifyMode::dynamic_row_parallel) {
            ++verify.dynamic_steps;
        }

        verify.steps.push_back(
            std::move(step));
    }

    if (verify.steps.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(
            "speculative verify step count exceeds uint32");
    }

    return verify;
}

}  // namespace strix
