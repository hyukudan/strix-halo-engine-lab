#include "strix/execution_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

const strix::ExecutionStep* find_layer_op(
    const strix::SerialExecutionPlan& plan,
    std::uint32_t layer,
    strix::SerialOp op) {
    for (const auto& step : plan.steps) {
        if (step.layer.has_value() &&
            *step.layer == layer &&
            step.op == op) {
            return &step;
        }
    }
    return nullptr;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto plan =
        strix::build_serial_decode_execution_plan(model);

    expect(plan.linear_attention_steps == 36,
           "serial plan has thirty-six recurrent attention layers");
    expect(plan.indexed_attention_steps == 12,
           "serial plan has twelve indexed attention layers");
    expect(plan.ple_steps == 1,
           "serial plan has one PLE layer");
    expect(plan.moe_route_steps == 48 &&
               plan.moe_expert_steps == 48,
           "serial plan has routed MoE in every decoder layer");

    expect(plan.steps.front().op ==
               strix::SerialOp::token_embedding,
           "serial plan begins with token embedding");
    expect(plan.steps.back().op ==
               strix::SerialOp::lm_head,
           "serial plan ends with LM head");

    const auto* ple =
        find_layer_op(plan, 1, strix::SerialOp::ple);
    expect(ple != nullptr &&
               strix::has_state(
                   ple->reads_state,
                   strix::PersistentState::ngram_history) &&
               strix::has_state(
                   ple->writes_state,
                   strix::PersistentState::convolution_history) &&
               !ple->graph_safe_candidate,
           "PLE step owns n-gram/conv state and is not assumed graph-safe");

    expect(find_layer_op(
               plan, 0,
               strix::SerialOp::gated_delta_net) != nullptr,
           "layer zero is recurrent attention");
    const auto* indexed =
        find_layer_op(
            plan, 3,
            strix::SerialOp::indexed_attention);
    expect(indexed != nullptr &&
               strix::has_state(
                   indexed->reads_state,
                   strix::PersistentState::attention_kv) &&
               strix::has_state(
                   indexed->writes_state,
                   strix::PersistentState::sparse_index),
           "layer three is indexed attention with KV/index state");

    const auto* route =
        find_layer_op(plan, 0, strix::SerialOp::moe_route);
    const auto* shared =
        find_layer_op(plan, 0, strix::SerialOp::shared_expert);
    const auto* experts =
        find_layer_op(plan, 0, strix::SerialOp::moe_experts);
    const auto* combine =
        find_layer_op(plan, 0, strix::SerialOp::moe_combine);

    expect(route != nullptr && shared != nullptr &&
               route->dependencies == shared->dependencies,
           "router and shared expert can start from the same MLP input");
    expect(experts != nullptr &&
               std::find(
                   experts->dependencies.begin(),
                   experts->dependencies.end(),
                   route->id) != experts->dependencies.end(),
           "routed experts depend on router completion");
    expect(combine != nullptr &&
               std::find(
                   combine->dependencies.begin(),
                   combine->dependencies.end(),
                   experts->id) != combine->dependencies.end() &&
               std::find(
                   combine->dependencies.begin(),
                   combine->dependencies.end(),
                   shared->id) != combine->dependencies.end(),
           "MoE combine waits for routed and shared experts");

    for (std::size_t index = 0; index < plan.steps.size(); ++index) {
        expect(plan.steps[index].id == index,
               "execution step IDs match stable vector order");
        for (const auto dependency : plan.steps[index].dependencies) {
            expect(dependency < plan.steps[index].id,
                   "execution dependencies only point backward");
        }
    }

    if (failures == 0) {
        std::cout << "execution plan tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
