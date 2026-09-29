#include "strix/execution_runtime.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

struct RecordingDispatcher final
    : strix::EagerExecutionDispatcher {
    std::vector<std::uint32_t> kernel_steps;
    std::vector<std::uint32_t> runtime_steps;
    std::optional<std::uint32_t> fail_step;

    void dispatch_kernel(
        const strix::ExecutionDispatchContext& context) override {
        if (fail_step.has_value() &&
            *fail_step == context.step_id) {
            throw std::runtime_error(
                "injected kernel failure");
        }
        expect(context.kernel_id.has_value() &&
                   !context.kernel_id->empty(),
               "kernel dispatch carries a kernel id");
        expect(context.arena != nullptr,
               "dispatch context exposes its arena plan");
        if (context.op == strix::SerialOp::lm_head) {
            expect(context.reads.size() == 1 &&
                       context.writes.size() == 1 &&
                       context.reads[0].generation > 0 &&
                       context.writes[0].generation == 1,
                   "kernel dispatch receives versioned IO bindings");
            if (context.arena != nullptr) {
                const auto* input =
                    strix::find_arena_allocation(
                        *context.arena,
                        context.reads[0]);
                const auto* output =
                    strix::find_arena_allocation(
                        *context.arena,
                        context.writes[0]);
                expect(input != nullptr &&
                           output != nullptr &&
                           input->offset != output->offset,
                       "LM-head dispatch resolves non-aliased arena buffers");
            }
        }
        kernel_steps.push_back(context.step_id);
    }

    void dispatch_runtime_step(
        const strix::ExecutionDispatchContext& context) override {
        if (fail_step.has_value() &&
            *fail_step == context.step_id) {
            throw std::runtime_error(
                "injected runtime-step failure");
        }
        expect(!context.kernel_id.has_value(),
               "runtime dispatch does not carry a kernel id");
        runtime_steps.push_back(context.step_id);
    }
};

strix::ExecutionBindingPlan fully_bound(
    const strix::SerialExecutionPlan& plan) {
    strix::ExecutionBindingPlan bindings;
    bindings.bindings.reserve(plan.steps.size());

    for (const auto& step : plan.steps) {
        strix::ExecutionKernelBinding binding;
        binding.step_id = step.id;
        binding.op = step.op;

        if (step.op == strix::SerialOp::position_prepare) {
            binding.host_or_runtime_step = true;
            ++bindings.host_or_runtime_steps;
        } else {
            const auto family =
                strix::operator_family_for(step.op);
            if (!family.has_value()) {
                throw std::runtime_error(
                    "test plan contains an unmapped operator");
            }
            binding.family = *family;
            binding.kernel_id =
                std::string("test.") +
                std::string(strix::to_string(step.op));
            ++bindings.bound_steps;
        }
        bindings.bindings.push_back(std::move(binding));
    }
    return bindings;
}

strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv |
           PersistentState::recurrent_matrix |
           PersistentState::convolution_history |
           PersistentState::sparse_index |
           PersistentState::ngram_history |
           PersistentState::position_history;
}

std::size_t generation_index(
    strix::PersistentState state) {
    using strix::PersistentState;
    switch (state) {
    case PersistentState::attention_kv: return 0;
    case PersistentState::recurrent_matrix: return 1;
    case PersistentState::convolution_history: return 2;
    case PersistentState::sparse_index: return 3;
    case PersistentState::ngram_history: return 4;
    case PersistentState::position_history: return 5;
    default:
        throw std::invalid_argument(
            "state flag does not have one generation slot");
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto plan =
        strix::build_serial_decode_execution_plan(model);
    const auto io =
        strix::build_serial_execution_io_plan(plan);
    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model, buffer_request, plan, io, 256);
    auto bindings = fully_bound(plan);

    strix::EagerExecutionRequest request;
    request.logical_rows = 1;
    request.request_concurrency = 1;
    request.state.valid = all_state();

    RecordingDispatcher dispatcher;
    const auto result =
        strix::execute_serial_eager(
            plan, bindings, io, arena, request, dispatcher);

    expect(result.completed_steps == plan.steps.size(),
           "eager executor completes every serial-plan step");
    expect(result.kernel_steps == bindings.bound_steps,
           "eager executor dispatches every bound kernel");
    expect(result.runtime_steps ==
               bindings.host_or_runtime_steps,
           "eager executor dispatches runtime-managed steps");
    expect(result.completion_order.size() == plan.steps.size() &&
               result.completion_order.front() == 0 &&
               result.completion_order.back() ==
                   plan.steps.back().id,
           "eager executor preserves deterministic DAG order");
    expect(dispatcher.runtime_steps.size() == 1 &&
               plan.steps[dispatcher.runtime_steps[0]].op ==
                   strix::SerialOp::position_prepare,
           "position preparation uses runtime dispatcher");

    const auto position_index =
        generation_index(
            strix::PersistentState::position_history);
    expect(result.state.generations[position_index] == 1,
           "position state generation increments once per decode execution");

    const auto recurrent_index =
        generation_index(
            strix::PersistentState::recurrent_matrix);
    expect(result.state.generations[recurrent_index] == 36,
           "recurrent state generation increments once per recurrent layer");

    const auto kv_index =
        generation_index(
            strix::PersistentState::attention_kv);
    expect(result.state.generations[kv_index] == 12,
           "KV generation increments once per indexed-attention layer");

    auto incomplete = bindings;
    incomplete.bindings[0].kernel_id.reset();
    incomplete.missing_steps = 1;
    bool preflight_rejected = false;
    try {
        RecordingDispatcher unused;
        static_cast<void>(
            strix::execute_serial_eager(
                plan, incomplete, io, arena, request, unused));
    } catch (const std::invalid_argument&) {
        preflight_rejected = true;
    }
    expect(preflight_rejected,
           "missing binding is rejected before execution");

    auto unavailable_state = request;
    unavailable_state.state.valid =
        strix::PersistentState::none;
    bool state_rejected = false;
    try {
        RecordingDispatcher unused;
        static_cast<void>(
            strix::execute_serial_eager(
                plan, bindings, io, arena, unavailable_state, unused));
    } catch (const std::invalid_argument&) {
        state_rejected = true;
    }
    expect(state_rejected,
           "missing persistent-state objects fail preflight");

    RecordingDispatcher failing;
    const auto recurrent_step =
        static_cast<std::uint32_t>(
            std::distance(
                plan.steps.begin(),
                std::find_if(
                    plan.steps.begin(),
                    plan.steps.end(),
                    [](const auto& step) {
                        return step.op ==
                            strix::SerialOp::gated_delta_net;
                    })));
    failing.fail_step = recurrent_step;

    bool injected_failure = false;
    try {
        static_cast<void>(
            strix::execute_serial_eager(
                plan, bindings, io, arena, request, failing));
    } catch (const std::runtime_error&) {
        injected_failure = true;
    }
    expect(injected_failure,
           "dispatcher failure propagates immediately");
    expect(!failing.kernel_steps.empty() &&
               failing.kernel_steps.back() != recurrent_step,
           "failed kernel is not recorded as completed by the dispatcher");


    auto generation_overflow = request;
    generation_overflow.state.generations[
        generation_index(strix::PersistentState::recurrent_matrix)] =
        std::numeric_limits<std::uint64_t>::max() - 35ULL;
    bool generation_rejected = false;
    RecordingDispatcher overflow_dispatcher;
    try {
        static_cast<void>(
            strix::execute_serial_eager(
                plan,
                bindings,
                io,
                arena,
                generation_overflow,
                overflow_dispatcher));
    } catch (const std::overflow_error&) {
        generation_rejected = true;
    }
    expect(generation_rejected,
           "generation overflow fails before dispatch");
    expect(overflow_dispatcher.kernel_steps.empty() &&
               overflow_dispatcher.runtime_steps.empty(),
           "generation-overflow preflight touches no dispatcher");

    auto bad_runtime_binding = bindings;
    const auto runtime_step_id =
        dispatcher.runtime_steps.front();
    bad_runtime_binding.bindings[runtime_step_id].family =
        strix::kernels::OperatorFamily::partial_rope;
    bool runtime_binding_rejected = false;
    try {
        RecordingDispatcher unused;
        static_cast<void>(
            strix::execute_serial_eager(
                plan,
                bad_runtime_binding,
                io,
                arena,
                request,
                unused));
    } catch (const std::invalid_argument&) {
        runtime_binding_rejected = true;
    }
    expect(runtime_binding_rejected,
           "runtime-managed step rejects an attached kernel family");

    auto bad_dependency_plan = plan;
    bad_dependency_plan.steps[1].dependencies.push_back(1);
    bool dependency_rejected = false;
    try {
        RecordingDispatcher unused;
        static_cast<void>(
            strix::execute_serial_eager(
                bad_dependency_plan,
                bindings,
                io,
                arena,
                request,
                unused));
    } catch (const std::invalid_argument&) {
        dependency_rejected = true;
    }
    expect(dependency_rejected,
           "non-backward dependency fails preflight");

    if (failures == 0) {
        std::cout << "execution runtime tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
