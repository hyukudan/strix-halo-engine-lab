#include "strix/execution_runtime.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix {
namespace {

constexpr std::array<PersistentState, 6> state_flags{
    PersistentState::attention_kv,
    PersistentState::recurrent_matrix,
    PersistentState::convolution_history,
    PersistentState::sparse_index,
    PersistentState::ngram_history,
    PersistentState::position_history,
};

std::uint32_t state_bits(PersistentState value) noexcept {
    return static_cast<std::uint32_t>(value);
}

bool subset(
    PersistentState available,
    PersistentState required) noexcept {
    const auto have = state_bits(available);
    const auto need = state_bits(required);
    return (have & need) == need;
}

const ExecutionKernelBinding& binding_for(
    const ExecutionBindingPlan& bindings,
    std::uint32_t step_id) {
    if (step_id >= bindings.bindings.size()) {
        throw std::invalid_argument(
            "execution binding is missing a step id");
    }
    return bindings.bindings[step_id];
}


bool has_component(
    const SerialSessionStateLayout& layout,
    SessionStateComponent component) noexcept {
    return find_session_state_allocation(
               layout, component) != nullptr;
}

void require_component(
    const SerialSessionStateLayout& layout,
    SessionStateComponent component,
    const char* message) {
    if (!has_component(layout, component)) {
        throw std::invalid_argument(message);
    }
}

void validate_state_component_for_step(
    const SerialSessionStateLayout& layout,
    const ExecutionStep& step) {
    const auto required =
        step.reads_state | step.writes_state;

    if (has_state(required, PersistentState::attention_kv)) {
        require_component(
            layout,
            SessionStateComponent::attention_kv,
            "execution step requires attention KV storage");
    }
    if (has_state(required, PersistentState::recurrent_matrix)) {
        require_component(
            layout,
            SessionStateComponent::recurrent_matrix,
            "execution step requires recurrent matrix storage");
    }
    if (has_state(required, PersistentState::convolution_history)) {
        if (step.op == SerialOp::ple) {
            require_component(
                layout,
                SessionStateComponent::ple_convolution,
                "PLE step requires PLE convolution storage");
        } else if (step.op == SerialOp::gated_delta_net) {
            require_component(
                layout,
                SessionStateComponent::gdn_convolution,
                "GDN step requires GDN convolution storage");
        } else {
            throw std::invalid_argument(
                "convolution-history state is attached to an unsupported operator");
        }
    }
    if (has_state(required, PersistentState::sparse_index)) {
        require_component(
            layout,
            SessionStateComponent::sparse_index,
            "execution step requires sparse-index storage");
    }
    if (has_state(required, PersistentState::ngram_history)) {
        require_component(
            layout,
            SessionStateComponent::ngram_history,
            "execution step requires n-gram history storage");
    }
    if (has_state(required, PersistentState::position_history)) {
        require_component(
            layout,
            SessionStateComponent::position_history,
            "execution step requires position-history storage");
    }
}

void validate_step_identity(
    const ExecutionStep& step,
    const ExecutionKernelBinding& binding) {
    if (binding.step_id != step.id ||
        binding.op != step.op) {
        throw std::invalid_argument(
            "execution binding identity does not match the serial plan");
    }

    if (binding.host_or_runtime_step) {
        if (binding.family.has_value() ||
            binding.kernel_id.has_value()) {
            throw std::invalid_argument(
                "runtime-managed execution step must not bind a kernel family");
        }
        return;
    }

    if (!binding.family.has_value() ||
        !binding.kernel_id.has_value() ||
        binding.kernel_id->empty()) {
        throw std::invalid_argument(
            "kernel execution step is not fully bound");
    }
}

}  // namespace

bool PersistentStateGenerations::contains(
    PersistentState required) const noexcept {
    return subset(valid, required);
}

void PersistentStateGenerations::publish_write(
    PersistentState written) {
    if (written == PersistentState::none) {
        return;
    }

    valid = valid | written;
    for (std::size_t index = 0;
         index < state_flags.size();
         ++index) {
        if (!has_state(written, state_flags[index])) {
            continue;
        }
        if (generations[index] ==
            std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "persistent-state generation overflow");
        }
        ++generations[index];
    }
}

void validate_eager_execution_preflight(
    const SerialExecutionPlan& plan,
    const ExecutionBindingPlan& bindings,
    const SerialExecutionIoPlan& io_plan,
    const ExecutionArenaPlan& arena_plan,
    const EagerExecutionRequest& request,
    const SerialSessionStateLayout* session_state_layout,
    const SessionStateMemoryBinding* session_state_memory,
    const ExecutionRuntimeResources* resources) {
    if (request.logical_rows == 0 ||
        request.request_concurrency == 0) {
        throw std::invalid_argument(
            "eager execution rows and concurrency must be non-zero");
    }
    if (!request.input_tokens.empty() &&
        request.input_tokens.size() !=
            request.logical_rows) {
        throw std::invalid_argument(
            "input token span must match logical row count");
    }
    if (request.logical_rows > 1 &&
        request.input_tokens.empty()) {
        throw std::invalid_argument(
            "multi-row eager execution requires one input token per logical row");
    }

    if (bindings.bindings.size() != plan.steps.size()) {
        throw std::invalid_argument(
            "binding count does not match serial execution plan");
    }
    validate_serial_execution_io_plan(plan, io_plan);
    if (arena_plan.total_bytes == 0) {
        throw std::invalid_argument(
            "eager execution requires a non-empty arena plan");
    }
    for (const auto& io : io_plan.steps) {
        for (const auto use : io.reads) {
            if (find_arena_allocation(arena_plan, use) == nullptr) {
                throw std::invalid_argument(
                    "eager execution arena is missing a read generation");
            }
        }
        for (const auto use : io.writes) {
            if (find_arena_allocation(arena_plan, use) == nullptr) {
                throw std::invalid_argument(
                    "eager execution arena is missing a write generation");
            }
        }
        for (const auto use : io.temporaries) {
            if (find_arena_allocation(arena_plan, use) == nullptr) {
                throw std::invalid_argument(
                    "eager execution arena is missing a temporary generation");
            }
        }
    }
    if (bindings.missing_steps != 0) {
        throw std::invalid_argument(
            "eager execution requires a fully bound plan");
    }

    if ((session_state_layout == nullptr) !=
        (session_state_memory == nullptr)) {
        throw std::invalid_argument(
            "session state layout and memory must be supplied together");
    }
    if (resources != nullptr) {
        if (resources->session_state_layout != session_state_layout ||
            resources->session_state_memory != session_state_memory) {
            throw std::invalid_argument(
                "runtime resource state binding does not match eager arguments");
        }
    }
    if (session_state_layout != nullptr) {
        validate_session_state_memory_binding(
            *session_state_layout,
            *session_state_memory);
        if (request.position >=
            session_state_layout->context_tokens) {
            throw std::out_of_range(
                "execution position exceeds allocated session-state capacity");
        }
    }

    std::array<std::uint64_t, 6> pending_writes{};

    std::uint32_t actual_bound = 0;
    std::uint32_t actual_runtime = 0;

    for (std::size_t index = 0;
         index < plan.steps.size();
         ++index) {
        const auto& step = plan.steps[index];
        if (step.id != index) {
            throw std::invalid_argument(
                "serial execution step ids must be dense and ordered");
        }

        const auto& binding =
            binding_for(bindings, step.id);
        validate_step_identity(step, binding);

        for (const auto dependency : step.dependencies) {
            if (dependency >= step.id ||
                dependency >= plan.steps.size()) {
                throw std::invalid_argument(
                    "serial execution dependency is not backward-only");
            }
        }

        const auto required_objects =
            step.reads_state | step.writes_state;
        if (session_state_layout != nullptr) {
            validate_state_component_for_step(
                *session_state_layout, step);
        }
        if (!subset(request.state.valid, required_objects)) {
            throw std::invalid_argument(
                "persistent state object required by an execution step is unavailable");
        }

        if (binding.host_or_runtime_step) {
            ++actual_runtime;
        } else {
            ++actual_bound;
        }

        for (std::size_t state_index = 0;
             state_index < state_flags.size();
             ++state_index) {
            if (has_state(step.writes_state, state_flags[state_index])) {
                ++pending_writes[state_index];
            }
        }
    }

    if (actual_bound != bindings.bound_steps ||
        actual_runtime != bindings.host_or_runtime_steps ||
        actual_bound + actual_runtime != plan.steps.size()) {
        throw std::invalid_argument(
            "execution binding counters do not match binding contents");
    }

    for (std::size_t state_index = 0;
         state_index < state_flags.size();
         ++state_index) {
        const auto writes = pending_writes[state_index];
        const auto generation =
            request.state.generations[state_index];
        if (writes >
            std::numeric_limits<std::uint64_t>::max() - generation) {
            throw std::overflow_error(
                "persistent-state generation would overflow during execution");
        }
    }
}

EagerExecutionResult execute_serial_eager(
    const SerialExecutionPlan& plan,
    const ExecutionBindingPlan& bindings,
    const SerialExecutionIoPlan& io_plan,
    const ExecutionArenaPlan& arena_plan,
    const EagerExecutionRequest& request,
    EagerExecutionDispatcher& dispatcher,
    const ExecutionMemoryBinding* memory,
    const SerialSessionStateLayout* session_state_layout,
    const SessionStateMemoryBinding* session_state_memory,
    const ExecutionRuntimeResources* resources) {
    validate_eager_execution_preflight(
        plan, bindings, io_plan, arena_plan, request,
        session_state_layout, session_state_memory, resources);

    EagerExecutionResult result;
    result.state = request.state;
    result.completion_order.reserve(plan.steps.size());

    std::vector<bool> completed(
        plan.steps.size(), false);

    for (const auto& step : plan.steps) {
        for (const auto dependency : step.dependencies) {
            if (!completed[dependency]) {
                throw std::logic_error(
                    "execution dependency was not completed");
            }
        }

        if (!result.state.contains(step.reads_state)) {
            throw std::logic_error(
                "persistent state disappeared after preflight");
        }

        const auto& binding =
            binding_for(bindings, step.id);
        const auto& io =
            io_plan.steps[step.id];

        ExecutionDispatchContext context;
        context.step_id = step.id;
        context.op = step.op;
        context.layer = step.layer;
        context.reads = io.reads;
        context.writes = io.writes;
        context.temporaries =
            io.temporaries;
        context.arena = &arena_plan;
        context.memory = memory;
        context.session_state_layout = session_state_layout;
        context.session_state_memory = session_state_memory;
        context.resources = resources;
        context.reads_state = step.reads_state;
        context.writes_state = step.writes_state;
        context.logical_rows = request.logical_rows;
        context.request_concurrency =
            request.request_concurrency;
        context.input_token_id =
            request.input_token_id;
        context.input_tokens =
            request.input_tokens;
        context.position =
            request.position;

        if (binding.host_or_runtime_step) {
            dispatcher.dispatch_runtime_step(context);
            ++result.runtime_steps;
        } else {
            context.kernel_id =
                std::string_view(*binding.kernel_id);
            dispatcher.dispatch_kernel(context);
            ++result.kernel_steps;
        }

        // Publish persistent-state generations only after the dispatcher
        // returns successfully. A failed step cannot make partial state
        // visible to subsequent steps.
        result.state.publish_write(
            step.writes_state);

        completed[step.id] = true;
        result.completion_order.push_back(step.id);
        ++result.completed_steps;
    }

    return result;
}

}  // namespace strix
