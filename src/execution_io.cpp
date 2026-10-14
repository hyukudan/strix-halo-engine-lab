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

constexpr Buffer mlp_intermediate() {
    return Buffer{
        ExecutionBufferClass::mlp_intermediate,
        0};
}

constexpr Buffer gdn_workspace() {
    return Buffer{
        ExecutionBufferClass::gdn_workspace,
        0};
}

constexpr Buffer ple_workspace() {
    return Buffer{
        ExecutionBufferClass::ple_workspace,
        0};
}

constexpr Buffer indexed_attention_workspace() {
    return Buffer{
        ExecutionBufferClass::indexed_attention_workspace,
        0};
}

constexpr Buffer hyper_workspace() {
    return Buffer{ExecutionBufferClass::hyper_workspace, 0};
}

bool contains_buffer(
    std::span<const ExecutionBufferUse> values,
    Buffer needle) {
    return std::find_if(
               values.begin(),
               values.end(),
               [&](const ExecutionBufferUse& use) {
                   return use.buffer == needle;
               }) != values.end();
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
    case ExecutionBufferClass::mlp_intermediate:
        return plan.mlp_intermediate_slots;
    case ExecutionBufferClass::gdn_workspace:
        return plan.gdn_workspace_slots;
    case ExecutionBufferClass::ple_workspace:
        return plan.ple_workspace_slots;
    case ExecutionBufferClass::indexed_attention_workspace:
        return plan.indexed_attention_workspace_slots;
    case ExecutionBufferClass::hyper_workspace:
        return plan.hyper_workspace_slots;
    case ExecutionBufferClass::router_workspace:
        return plan.router_workspace_slots;
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
    const SerialExecutionPlan& execution_plan, bool parallel_moe_router) {
    SerialExecutionIoPlan output;
    output.expanded_hidden_slots = 2;
    output.hidden_slots = 3;
    output.injection_slots = 1;
    output.route_index_slots = 1;
    output.route_weight_slots = 1;
    output.logits_slots = 1;
    output.mlp_intermediate_slots = 1;
    output.gdn_workspace_slots = 1;
    output.ple_workspace_slots = 1;
    output.indexed_attention_workspace_slots = 1;
    output.hyper_workspace_slots = 1;
    output.router_workspace_slots = parallel_moe_router ? 1U : 0U;
    output.steps.reserve(
        execution_plan.steps.size());

    constexpr std::size_t maximum_buffers =
        (static_cast<std::size_t>(ExecutionBufferClass::router_workspace) + 1U) * 8U;
    std::array<std::uint64_t, maximum_buffers> generations{};

    const auto read_use = [&](Buffer buffer) {
        return ExecutionBufferUse{
            buffer,
            generations[flat_index(buffer)],
        };
    };
    const auto write_use = [&](Buffer buffer) {
        auto& generation =
            generations[flat_index(buffer)];
        ++generation;
        return ExecutionBufferUse{
            buffer,
            generation,
        };
    };
    const auto temporary_use = [&](Buffer buffer) {
        auto& generation =
            generations[flat_index(buffer)];
        ++generation;
        return ExecutionBufferUse{
            buffer,
            generation,
        };
    };

    std::uint32_t current_hyper = 0;

    for (const auto& step : execution_plan.steps) {
        ExecutionStepIo io;
        io.step_id = step.id;
        io.op = step.op;

        switch (step.op) {
        case SerialOp::token_embedding:
            io.writes = {write_use(hidden(0))};
            break;

        case SerialOp::position_prepare:
            break;

        case SerialOp::hyper_expand:
            io.reads = {read_use(hidden(0))};
            io.writes = {write_use(hyper(current_hyper))};
            break;

        case SerialOp::ple: {
            const auto next = current_hyper ^ 1U;
            io.reads = {read_use(hyper(current_hyper))};
            io.writes = {write_use(hyper(next))};
            io.temporaries = {
                temporary_use(
                    ple_workspace()),
            };
            current_hyper = next;
            break;
        }

        case SerialOp::attn_hyper_mix:
            io.reads = {read_use(hyper(current_hyper))};
            io.writes = {
                write_use(hidden(0)),
                write_use(injection()),
            };
            io.temporaries = {temporary_use(hyper_workspace())};
            break;

        case SerialOp::gated_delta_net:
            io.reads = {read_use(hidden(0))};
            io.writes = {write_use(hidden(1))};
            io.temporaries = {
                temporary_use(
                    gdn_workspace()),
            };
            break;

        case SerialOp::indexed_attention:
            io.reads = {read_use(hidden(0))};
            io.writes = {write_use(hidden(1))};
            io.temporaries = {
                temporary_use(
                    indexed_attention_workspace()),
            };
            break;

        case SerialOp::attn_inject: {
            const auto next = current_hyper ^ 1U;
            io.reads = {
                read_use(hyper(current_hyper)),
                read_use(hidden(1)),
                read_use(injection()),
            };
            io.writes = {write_use(hyper(next))};
            current_hyper = next;
            break;
        }

        case SerialOp::mlp_hyper_mix:
            io.reads = {read_use(hyper(current_hyper))};
            io.writes = {
                write_use(hidden(0)),
                write_use(injection()),
            };
            io.temporaries = {temporary_use(hyper_workspace())};
            break;

        case SerialOp::moe_route:
            io.reads = {read_use(hidden(0))};
            io.writes = {
                write_use(route_indices()),
                write_use(route_weights()),
            };
            if (parallel_moe_router)
                io.temporaries = {temporary_use(Buffer{ExecutionBufferClass::router_workspace, 0})};
            break;

        case SerialOp::shared_expert:
            io.reads = {read_use(hidden(0))};
            io.writes = {write_use(hidden(1))};
            io.temporaries = {
                temporary_use(
                    mlp_intermediate()),
            };
            break;

        case SerialOp::moe_experts:
            io.reads = {
                read_use(hidden(0)),
                read_use(route_indices()),
                read_use(route_weights()),
            };
            io.writes = {write_use(hidden(2))};
            io.temporaries = {
                temporary_use(
                    mlp_intermediate()),
            };
            break;

        case SerialOp::moe_combine:
            io.reads = {
                read_use(hidden(1)),
                read_use(hidden(2)),
            };
            io.writes = {write_use(hidden(0))};
            break;

        case SerialOp::mlp_inject: {
            const auto next = current_hyper ^ 1U;
            io.reads = {
                read_use(hyper(current_hyper)),
                read_use(hidden(0)),
                read_use(injection()),
            };
            io.writes = {write_use(hyper(next))};
            current_hyper = next;
            break;
        }

        case SerialOp::final_hyper_mix:
            io.reads = {read_use(hyper(current_hyper))};
            io.writes = {write_use(hidden(0))};
            io.temporaries = {temporary_use(hyper_workspace())};
            break;

        case SerialOp::lm_head:
            io.reads = {read_use(hidden(0))};
            io.writes = {write_use(logits())};
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
        (static_cast<std::size_t>(ExecutionBufferClass::router_workspace) + 1U) * 8U;
    std::array<bool, maximum_buffers> produced{};
    std::array<std::uint64_t, maximum_buffers> generation{};

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

        const auto router_temporaries = std::count_if(io.temporaries.begin(), io.temporaries.end(),
            [](const ExecutionBufferUse& use) { return use.buffer.buffer_class == ExecutionBufferClass::router_workspace; });
        if (io_plan.router_workspace_slots > 1U ||
            (step.op == SerialOp::moe_route && io.temporaries.size() != io_plan.router_workspace_slots) ||
            (step.op == SerialOp::moe_route &&
             router_temporaries != static_cast<std::ptrdiff_t>(io_plan.router_workspace_slots)) ||
            (step.op != SerialOp::moe_route && router_temporaries != 0))
            throw std::invalid_argument("router workspace IO does not match admitted topology");

        for (const auto& use : io.reads) {
            const auto buffer = use.buffer;
            if (buffer.slot >=
                slot_limit(
                    io_plan,
                    buffer.buffer_class)) {
                throw std::invalid_argument(
                    "execution IO read uses an invalid buffer slot");
            }
            const auto flat = flat_index(buffer);
            if (!produced[flat]) {
                throw std::invalid_argument(
                    "execution IO reads a buffer before it is produced");
            }
            if (use.generation != generation[flat]) {
                throw std::invalid_argument(
                    "execution IO read references a stale buffer generation");
            }
        }

        for (const auto& use : io.writes) {
            const auto buffer = use.buffer;
            if (buffer.slot >=
                slot_limit(
                    io_plan,
                    buffer.buffer_class)) {
                throw std::invalid_argument(
                    "execution IO write uses an invalid buffer slot");
            }
            if (contains_buffer(io.reads, buffer)) {
                throw std::invalid_argument(
                    "execution IO step aliases a read and write buffer");
            }

            const auto flat = flat_index(buffer);
            if (use.generation != generation[flat] + 1ULL) {
                throw std::invalid_argument(
                    "execution IO write generation is not the next slot generation");
            }
        }

        for (const auto& use : io.temporaries) {
            const auto buffer = use.buffer;
            if (buffer.slot >=
                slot_limit(
                    io_plan,
                    buffer.buffer_class)) {
                throw std::invalid_argument(
                    "execution IO temporary uses an invalid buffer slot");
            }
            if (contains_buffer(io.reads, buffer) ||
                contains_buffer(io.writes, buffer)) {
                throw std::invalid_argument(
                    "execution IO temporary aliases a semantic read/write buffer");
            }
            const auto flat = flat_index(buffer);
            if (use.generation != generation[flat] + 1ULL) {
                throw std::invalid_argument(
                    "execution IO temporary generation is not the next slot generation");
            }
        }

        for (const auto& use : io.writes) {
            const auto flat = flat_index(use.buffer);
            generation[flat] = use.generation;
            produced[flat] = true;
        }
        for (const auto& use : io.temporaries) {
            const auto flat = flat_index(use.buffer);
            generation[flat] = use.generation;
        }
    }
}

}  // namespace strix
