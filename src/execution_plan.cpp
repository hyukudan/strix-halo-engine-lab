#include "strix/execution_plan.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

std::uint32_t append_step(
    SerialExecutionPlan& plan,
    SerialOp op,
    std::optional<std::uint32_t> layer,
    std::vector<std::uint32_t> dependencies,
    PersistentState reads_state = PersistentState::none,
    PersistentState writes_state = PersistentState::none,
    bool graph_safe_candidate = true,
    bool dynamic_shape = false) {
    const auto id =
        static_cast<std::uint32_t>(plan.steps.size());
    plan.steps.push_back(
        ExecutionStep{
            id,
            op,
            layer,
            std::move(dependencies),
            reads_state,
            writes_state,
            graph_safe_candidate,
            dynamic_shape,
        });
    return id;
}

bool is_indexed_attention_layer(
    const ModelContract& model,
    std::uint32_t zero_based_layer) {
    const auto one_based =
        static_cast<std::uint64_t>(zero_based_layer) + 1ULL;
    return one_based % model.full_attention_interval == 0;
}

}  // namespace

std::string_view to_string(SerialOp op) noexcept {
    switch (op) {
    case SerialOp::token_embedding: return "token_embedding";
    case SerialOp::position_prepare: return "position_prepare";
    case SerialOp::hyper_expand: return "hyper_expand";
    case SerialOp::ple: return "ple";
    case SerialOp::attn_hyper_mix: return "attn_hyper_mix";
    case SerialOp::gated_delta_net: return "gated_delta_net";
    case SerialOp::indexed_attention: return "indexed_attention";
    case SerialOp::attn_inject: return "attn_inject";
    case SerialOp::mlp_hyper_mix: return "mlp_hyper_mix";
    case SerialOp::moe_route: return "moe_route";
    case SerialOp::moe_experts: return "moe_experts";
    case SerialOp::shared_expert: return "shared_expert";
    case SerialOp::moe_combine: return "moe_combine";
    case SerialOp::mlp_inject: return "mlp_inject";
    case SerialOp::final_hyper_mix: return "final_hyper_mix";
    case SerialOp::lm_head: return "lm_head";
    }
    return "unknown";
}

SerialExecutionPlan build_serial_decode_execution_plan(
    const ModelContract& model) {
    if (const auto error = model.validate(); !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (model.hidden_layers >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(
            "layer count does not fit execution-plan layer id");
    }

    SerialExecutionPlan plan;

    const auto embedding = append_step(
        plan, SerialOp::token_embedding, std::nullopt, {});
    const auto position = append_step(
        plan,
        SerialOp::position_prepare,
        std::nullopt,
        {},
        PersistentState::position_history,
        PersistentState::position_history,
        true,
        false);
    auto hidden = append_step(
        plan,
        SerialOp::hyper_expand,
        std::nullopt,
        {embedding});

    for (std::uint32_t layer = 0;
         layer < static_cast<std::uint32_t>(model.hidden_layers);
         ++layer) {
        const auto one_based =
            static_cast<std::uint64_t>(layer) + 1ULL;

        if (one_based == model.ple_layer_1based) {
            hidden = append_step(
                plan,
                SerialOp::ple,
                layer,
                {hidden},
                PersistentState::ngram_history |
                    PersistentState::convolution_history,
                PersistentState::ngram_history |
                    PersistentState::convolution_history,
                false,
                true);
            ++plan.ple_steps;
        }

        const auto attn_mix = append_step(
            plan,
            SerialOp::attn_hyper_mix,
            layer,
            {hidden});

        std::uint32_t core = 0;
        if (is_indexed_attention_layer(model, layer)) {
            core = append_step(
                plan,
                SerialOp::indexed_attention,
                layer,
                {attn_mix, position},
                PersistentState::attention_kv |
                    PersistentState::sparse_index |
                    PersistentState::position_history,
                PersistentState::attention_kv |
                    PersistentState::sparse_index,
                true,
                true);
            ++plan.indexed_attention_steps;
        } else {
            core = append_step(
                plan,
                SerialOp::gated_delta_net,
                layer,
                {attn_mix},
                PersistentState::recurrent_matrix |
                    PersistentState::convolution_history,
                PersistentState::recurrent_matrix |
                    PersistentState::convolution_history,
                true,
                false);
            ++plan.linear_attention_steps;
        }

        hidden = append_step(
            plan,
            SerialOp::attn_inject,
            layer,
            {attn_mix, core});

        const auto mlp_mix = append_step(
            plan,
            SerialOp::mlp_hyper_mix,
            layer,
            {hidden});

        const auto route = append_step(
            plan,
            SerialOp::moe_route,
            layer,
            {mlp_mix},
            PersistentState::none,
            PersistentState::none,
            true,
            false);
        ++plan.moe_route_steps;

        const auto shared = append_step(
            plan,
            SerialOp::shared_expert,
            layer,
            {mlp_mix},
            PersistentState::none,
            PersistentState::none,
            true,
            false);

        const auto experts = append_step(
            plan,
            SerialOp::moe_experts,
            layer,
            {mlp_mix, route},
            PersistentState::none,
            PersistentState::none,
            false,
            true);
        ++plan.moe_expert_steps;

        const auto combined = append_step(
            plan,
            SerialOp::moe_combine,
            layer,
            {experts, shared});

        hidden = append_step(
            plan,
            SerialOp::mlp_inject,
            layer,
            {mlp_mix, combined});
    }

    const auto final_mix = append_step(
        plan,
        SerialOp::final_hyper_mix,
        std::nullopt,
        {hidden});

    append_step(
        plan,
        SerialOp::lm_head,
        std::nullopt,
        {final_mix});

    return plan;
}

}  // namespace strix
