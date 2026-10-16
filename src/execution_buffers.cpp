#include "strix/execution_buffers.hpp"

#include "strix/indexed_attention_workspace.hpp"
#include "strix/moe_q4k_a8_runtime.hpp"

#include "strix/ple_workspace.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("execution buffer addition overflow");
    }
    return left + right;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("execution buffer multiplication overflow");
    }
    return left * right;
}

std::uint64_t product(
    std::uint64_t a,
    std::uint64_t b,
    std::uint64_t c) {
    return checked_mul(checked_mul(a, b), c);
}

}  // namespace

ExecutionBufferPlan build_serial_execution_buffer_plan(
    const ModelContract& model,
    const ExecutionBufferRequest& request) {
    if (const auto error = model.validate(); !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.logical_rows == 0 ||
        request.activation_element_bytes == 0 ||
        request.control_element_bytes == 0 ||
        request.index_element_bytes == 0 ||
        request.logit_element_bytes == 0 ||
        request.mlp_intermediate_element_bytes == 0) {
        throw std::invalid_argument(
            "execution buffer row and element sizes must be non-zero");
    }

    if(request.grouped_exact_q51_down && !request.q4k_a8g32_gate_up)
        throw std::invalid_argument("grouped exact Q5_1 down requires Q4K A8 slot workspace");
    if (request.q51_slot_parallel_down && !request.grouped_exact_q51_down)
        throw std::invalid_argument("Q5_1 slot-parallel down requires grouped Q5_1 workspace");
    if ((request.a8_down_q51 || request.a8_down_q80) &&
        (!request.a8_down_q51 || !request.q4k_a8g32_gate_up || !request.q51_slot_parallel_down ||
         request.logical_rows != 1 || request.activation_element_bytes != 2 ||
         model.hidden_size != 2560 || model.moe_intermediate_size != 640 ||
         model.num_experts != 512 || model.num_experts_per_token != 10))
        throw std::invalid_argument("A8 down workspace requires canonical one-row Q4K A8 topology");

    ExecutionBufferPlan plan;
    plan.logical_rows = request.logical_rows;

    plan.expanded_hidden_bytes =
        product(
            request.logical_rows,
            checked_mul(model.hidden_size, model.hc_count),
            request.activation_element_bytes);

    plan.hidden_bytes =
        product(
            request.logical_rows,
            model.hidden_size,
            request.activation_element_bytes);

    plan.injection_weight_bytes =
        product(
            request.logical_rows,
            model.hc_count,
            request.control_element_bytes);

    plan.route_index_bytes =
        product(
            request.logical_rows,
            model.num_experts_per_token,
            request.index_element_bytes);

    plan.route_weight_bytes =
        product(
            request.logical_rows,
            model.num_experts_per_token,
            request.control_element_bytes);

    plan.logits_bytes =
        product(
            request.logical_rows,
            model.vocab_size,
            request.logit_element_bytes);

    plan.mlp_intermediate_bytes =
        product(
            request.logical_rows,
            checked_mul(
                std::max(
                    model.moe_intermediate_size,
                    model.shared_expert_intermediate_size),
                2ULL),
            request.mlp_intermediate_element_bytes);
    // Routed expert outputs accumulate across all selected experts in FP32,
    // with a device validation word, before conversion to activation storage.
    const auto routed_workspace_bytes = checked_mul(request.logical_rows,
        checked_add(checked_mul(checked_add(checked_mul(model.moe_intermediate_size, 2ULL),
                                           model.hidden_size), model.bytes_fp32), 4ULL));
    plan.mlp_intermediate_bytes = std::max(plan.mlp_intermediate_bytes, routed_workspace_bytes);
    if (request.q4k_a8g32_gate_up) {
        if (request.logical_rows != 1 || request.control_element_bytes != 4 ||
            request.index_element_bytes != 4 || request.mlp_intermediate_element_bytes != 4 ||
            (request.activation_element_bytes != 2 && request.activation_element_bytes != 4))
            throw std::invalid_argument("Q4K A8 runtime requires C1 and typed FP32 workspace");
        const auto layout = moe_q4k_a8::build_runtime_workspace(model.hidden_size,
            model.moe_intermediate_size, model.num_experts, model.num_experts_per_token,
            request.q51_slot_parallel_down, request.a8_down_q51 || request.a8_down_q80);
        plan.mlp_intermediate_bytes = std::max(plan.mlp_intermediate_bytes, layout.total_bytes);
    }


    const auto gdn_key_width =
        checked_mul(
            model.gdn_key_heads,
            model.gdn_key_head_dim);
    const auto gdn_value_width =
        checked_mul(
            model.gdn_value_heads,
            model.gdn_value_head_dim);
    const auto gdn_conv_width =
        checked_add(
            checked_mul(
                gdn_key_width,
                2ULL),
            gdn_value_width);
    const auto gdn_workspace_elements =
        checked_add(
            checked_add(
                gdn_conv_width,
                gdn_value_width),
            checked_add(
                checked_mul(
                    model.gdn_value_heads,
                    2ULL),
                gdn_value_width));
    plan.gdn_workspace_bytes =
        product(
            request.logical_rows,
            gdn_workspace_elements,
            model.bytes_fp32);

    const auto ple_workspace =
        build_ple_workspace_layout(
            model);
    plan.ple_workspace_bytes =
        checked_mul(
            request.logical_rows,
            ple_workspace.total_bytes);

    const auto indexed_workspace =
        build_indexed_attention_workspace_layout(
            model, 256, request.qsa_hierarchical_selection, request.qsa_gqa_heads_per_block);
    plan.indexed_attention_workspace_bytes =
        checked_mul(
            request.logical_rows,
            indexed_workspace.total_bytes);
    plan.hyper_workspace_bytes = product(request.logical_rows,
        checked_add(checked_mul(model.hidden_size, model.hc_count), model.hc_lowrank), model.bytes_fp32);

    if (request.parallel_moe_router) {
        const auto raw = product(request.logical_rows, model.num_experts, sizeof(float));
        plan.router_workspace_bytes = checked_add(raw, 255ULL) & ~255ULL;
    }

    const auto attn_inject_peak =
        checked_add(
            checked_add(
                checked_mul(
                    plan.expanded_hidden_bytes,
                    plan.expanded_hidden_buffers),
                plan.hidden_bytes),
            plan.injection_weight_bytes);

    const auto moe_branch_peak =
        checked_add(
            checked_add(
                checked_add(
                    plan.expanded_hidden_bytes,
                    checked_mul(
                        plan.hidden_bytes,
                        plan.hidden_buffers)),
                plan.injection_weight_bytes),
            checked_add(
                checked_add(
                    plan.route_index_bytes,
                    plan.route_weight_bytes),
                plan.mlp_intermediate_bytes));

    const auto gdn_core_peak =
        checked_add(
            attn_inject_peak,
            plan.gdn_workspace_bytes);

    const auto ple_peak =
        checked_add(
            checked_mul(
                plan.expanded_hidden_bytes,
                2ULL),
            plan.ple_workspace_bytes);

    const auto indexed_attention_peak =
        checked_add(
            checked_add(
                checked_add(
                    plan.expanded_hidden_bytes,
                    checked_mul(
                        plan.hidden_bytes,
                        2ULL)),
                plan.injection_weight_bytes),
            plan.indexed_attention_workspace_bytes);

    plan.peak_internal_bytes =
        std::max({
            gdn_core_peak,
            moe_branch_peak,
            ple_peak,
            indexed_attention_peak,
            checked_add(attn_inject_peak, plan.hyper_workspace_bytes),
            checked_add(checked_add(checked_add(plan.expanded_hidden_bytes, plan.hidden_bytes),
                                   plan.injection_weight_bytes),
                        checked_add(checked_add(plan.route_index_bytes, plan.route_weight_bytes),
                                    plan.router_workspace_bytes)),
        });

    const auto logits_stage_peak =
        checked_add(plan.hidden_bytes, plan.logits_bytes);
    plan.peak_with_materialized_logits_bytes =
        std::max(plan.peak_internal_bytes, logits_stage_peak);

    return plan;
}

}  // namespace strix
