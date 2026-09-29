#include "strix/execution_io.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace strix {
namespace {

using Buffer = ExecutionBufferRef;

constexpr Buffer hyper(std::uint32_t slot) {
    return Buffer{
        ExecutionBufferClass::expanded_hidden,
        slot};
}

constexpr Buffer hidden(std::uint32_t slot) {
    return Buffer{
        ExecutionBufferClass::hidden,
        slot};
}

constexpr Buffer injection() {
    return Buffer{
        ExecutionBufferClass::injection_weights,
        0};
}

constexpr Buffer route_indices() {
    return Buffer{
        ExecutionBufferClass::route_indices,
        0};
}

constexpr Buffer route_weights() {
    return Buffer{
        ExecutionBufferClass::route_weights,
        0};
}

constexpr Buffer logits() {
    return Buffer{
        ExecutionBufferClass::logits,
        0};
}

bool contains(
    std::span<const Buffer> values,
    Buffer needle) {
    return std::find(
               values.begin(),
               values.end(),
               needle) != values.end();
}

std::uint32_t slot_limit(
    const SerialExecutionIoPlan& plan,
    ExecutionBufferClass buffer_class) {
    switch (buffer_class) {
    case ExecutionBufferClass::expanded_hidden:
        return plan.expanded_hidden_slots;
    case ExecutionBufferClass::hidden:
        return plan.hidden_slots;
    case ExecutionBufferClass::injection_weights:
        return plan.injection_slots;
    case ExecutionBufferClass::route_indices:
        return plan.route_index_slots;
    case ExecutionBufferClass::route_weights:
        return plan.route_weight_slots;
    case ExecutionBufferClass::logits:
        return plan.logits_slots;
    }
    return 0;
}

std::size_t flat_index(
    ExecutionBufferRef buffer) {
    constexpr std::size_t stride = 8;
    return static_cast<std::size_t>(
               buffer.buffer_class) *
               stride +
           buffer.slot;
}

}  // namespace

SerialExecutionIoPlan build_serial_execution_io_plan(
    const SerialExecutionPlan& execution_plan) {
    SerialExecutionIoPlan output;
    output.expanded_hidden_slots = 2;
    output.hidden_slots = 3;
    output.injection_slots = 1;
    output.route_index_slots = 1;
    output.route_weight_slots = 1;
    output.logits_slots = 1;
    output.steps.reserve(
        execution_plan.steps.size());

    std::uint32_t current_hyper = 0;

    for (const auto& step : execution_plan.steps) {
        ExecutionStepIo io;
        io.step_id = step.id;
        io.op = step.op;

        switch (step.op) {
        case SerialOp::token_embedding:
            io.writes = {hidden(0)};
            break;

        case SerialOp::position_prepare:
            break;

        case SerialOp::hyper_expand:
            io.reads = {hidden(0)};
            io.writes = {hyper(current_hyper)};
            break;

        case SerialOp::ple: {
            const auto next = current_hyper ^ 1U;
            io.reads = {hyper(current_hyper)};
            io.writes = {hyper(next)};
            current_hyper = next;
            break;
        }

        case SerialOp::attn_hyper_mix:
            io.reads = {hyper(current_hyper)};
            io.writes = {
                hidden(0),
                injection(),
            };
            break;

        case SerialOp::gated_delta_net:
        case SerialOp::indexed_attention:
            io.reads = {hidden(0)};
            io.writes = {hidden(1)};
            break;

        case SerialOp::attn_inject: {
            const auto next = current_hyper ^ 1U;
            io.reads = {
                hyper(current_hyper),
                hidden(1),
                injection(),
            };
            io.writes = {hyper(next)};
            current_hyper = next;
            break;
        }

        case SerialOp::mlp_hyper_mix:
            io.reads = {hyper(current_hyper)};
            io.writes = {
                hidden(0),
                injection(),
            };
            break;

        case SerialOp::moe_route:
            io.reads = {hidden(0)};
            io.writes = {
                route_indices(),
                route_weights(),
            };
            break;

        case SerialOp::shared_expert:
            io.reads = {hidden(0)};
            io.writes = {hidden(1)};
            break;

        case SerialOp::moe_experts:
            io.reads = {
                hidden(0),
                route_indices(),
                route_weights(),
            };
            io.writes = {hidden(2)};
            break;

        case SerialOp::moe_combine:
            io.reads = {
                hidden(1),
                hidden(2),
            };
            // hidden(0) is dead after route/shared/routed expert consumers.
            io.writes = {hidden(0)};
            break;

        case SerialOp::mlp_inject: {
            const auto next = current_hyper ^ 1U;
            io.reads = {
                hyper(current_hyper),
                hidden(0),
                injection(),
            };
            io.writes = {hyper(next)};
            current_hyper = next;
            break;
        }

        case SerialOp::final_hyper_mix:
            io.reads = {hyper(current_hyper)};
            io.writes = {hidden(0)};
            break;

        case SerialOp::lm_head:
            io.reads = {hidden(0)};
            io.writes = {logits()};
            break;
        }

        output.steps.push_back(
            std::move(io));
    }

    validate_serial_execution_io_plan(
        execution_plan, output);
    return output;
}

void validate_serial_execution_io_plan(
    const SerialExecutionPlan& execution_plan,
    const SerialExecutionIoPlan& io_plan) {
    if (io_plan.steps.size() !=
        execution_plan.steps.size()) {
        throw std::invalid_argument(
            "execution IO step count does not match execution plan");
    }

    constexpr std::size_t maximum_buffers =
        6U * 8U;
    std::array<bool, maximum_buffers> produced{};

    for (std::size_t index = 0;
         index < execution_plan.steps.size();
         ++index) {
        const auto& step =
            execution_plan.steps[index];
        const auto& io =
            io_plan.steps[index];

        if (io.step_id != step.id ||
            io.op != step.op) {
            throw std::invalid_argument(
                "execution IO identity does not match execution step");
        }

        for (const auto buffer : io.reads) {
            if (buffer.slot >=
                slot_limit(
                    io_plan,
                    buffer.buffer_class)) {
                throw std::invalid_argument(
                    "execution IO read uses an invalid buffer slot");
            }
            if (!produced[flat_index(buffer)]) {
                throw std::invalid_argument(
                    "execution IO reads a buffer before it is produced");
            }
        }

        for (const auto buffer : io.writes) {
            if (buffer.slot >=
                slot_limit(
                    io_plan,
                    buffer.buffer_class)) {
                throw std::invalid_argument(
                    "execution IO write uses an invalid buffer slot");
            }
            if (contains(io.reads, buffer)) {
                throw std::invalid_argument(
                    "execution IO step aliases a read and write buffer");
            }
        }

        for (const auto buffer : io.writes) {
            produced[flat_index(buffer)] = true;
        }
    }
}

}  // namespace strix
