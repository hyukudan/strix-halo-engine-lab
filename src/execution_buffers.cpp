#include "strix/execution_buffers.hpp"

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
        request.logit_element_bytes == 0) {
        throw std::invalid_argument(
            "execution buffer row and element sizes must be non-zero");
    }

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
                plan.route_index_bytes,
                plan.route_weight_bytes));

    plan.peak_internal_bytes =
        std::max(attn_inject_peak, moe_branch_peak);

    const auto logits_stage_peak =
        checked_add(plan.hidden_bytes, plan.logits_bytes);
    plan.peak_with_materialized_logits_bytes =
        std::max(plan.peak_internal_bytes, logits_stage_peak);

    return plan;
}

}  // namespace strix
