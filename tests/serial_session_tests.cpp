#include "strix/serial_session.hpp"

#include <set>
#include <string>
#include <utility>

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

strix::kernels::KernelVariant make_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::decode;
    variant.shape = shape;
    variant.wave =
        strix::kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision = "bf16";
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision = "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        strix::kernels::QualificationStatus::oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "test";
    variant.output_layout = "test";
    variant.compiler_identity = "test";
    return variant;
}

strix::kernels::KernelRegistry make_registry(
    const strix::ModelContract& model,
    const strix::ExecutionBindingRequest& request) {
    strix::kernels::KernelRegistry registry;
    const auto execution =
        strix::build_serial_decode_execution_plan(model);

    std::set<std::string> added;
    for (const auto& step : execution.steps) {
        const auto family =
            strix::operator_family_for(step.op);
        if (!family.has_value()) {
            continue;
        }
        const auto shape =
            strix::execution_shape_for(
                model, step.op, request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(shape.auxiliary_size);
        if (!added.insert(key).second) {
            continue;
        }
        registry.add(
            make_variant(
                "session." + key,
                *family,
                shape));
    }
    return registry;
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

struct SessionDispatcher final
    : strix::EagerExecutionDispatcher {
    std::uint32_t expected_token = 0;
    std::uint64_t expected_position = 0;
    std::uint32_t calls = 0;
    bool fail_on_first_recurrent = false;
    bool failed = false;

    void verify_scalars(
        const strix::ExecutionDispatchContext& context) {
        expect(context.input_token_id == expected_token,
               "dispatch receives session input token");
        expect(context.position == expected_position,
               "dispatch receives session position");
    }

    void dispatch_kernel(
        const strix::ExecutionDispatchContext& context) override {
        verify_scalars(context);
        if (fail_on_first_recurrent &&
            !failed &&
            context.op ==
                strix::SerialOp::gated_delta_net) {
            failed = true;
            throw std::runtime_error(
                "injected recurrent failure");
        }
        ++calls;
    }

    void dispatch_runtime_step(
        const strix::ExecutionDispatchContext& context) override {
        verify_scalars(context);
        ++calls;
    }
};

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::ExecutionBindingRequest binding_request;
    binding_request.logical_rows = 1;
    binding_request.request_concurrency = 1;
    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;

    const auto registry =
        make_registry(model, binding_request);
    const auto compiled =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            buffer_request,
            256);

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        compiled.execution_identity;
    memory.arena_base =
        static_cast<std::uintptr_t>(0x30000000ULL);
    memory.arena_bytes =
        compiled.arena.total_bytes;
    memory.allocation_generation = 1;

    strix::SerialSessionContext session;
    session.execution_identity =
        compiled.execution_identity;
    session.state.valid = all_state();

    SessionDispatcher first;
    first.expected_token = 123;
    first.expected_position = 0;
    const auto first_result =
        strix::execute_serial_token(
            compiled,
            session,
            memory,
            123,
            first);

    expect(first_result.completed_steps ==
               compiled.execution.steps.size(),
           "first serial token executes the full DAG");
    expect(session.next_position == 1 &&
               session.committed_tokens == 1,
           "successful serial token commits position and token count");

    SessionDispatcher second;
    second.expected_token = 456;
    second.expected_position = 1;
    static_cast<void>(
        strix::execute_serial_token(
            compiled,
            session,
            memory,
            456,
            second));
    expect(session.next_position == 2 &&
               session.committed_tokens == 2,
           "second serial token advances from committed session state");

    const auto before_failure = session;
    SessionDispatcher failing;
    failing.expected_token = 789;
    failing.expected_position = 2;
    failing.fail_on_first_recurrent = true;

    bool failure_observed = false;
    try {
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                session,
                memory,
                789,
                failing));
    } catch (const std::runtime_error&) {
        failure_observed = true;
    }
    expect(failure_observed,
           "injected serial kernel failure propagates");
    expect(session.next_position ==
               before_failure.next_position &&
               session.committed_tokens ==
                   before_failure.committed_tokens &&
               session.state.generations ==
                   before_failure.state.generations &&
               session.state.valid ==
                   before_failure.state.valid,
           "failed serial token does not publish committed metadata");
    expect(session.health ==
               strix::SerialSessionHealth::poisoned,
           "mid-execution failure poisons the session until recovery");

    bool poisoned_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                session,
                memory,
                1,
                unused));
    } catch (const std::logic_error&) {
        poisoned_rejected = true;
    }
    expect(poisoned_rejected,
           "poisoned session cannot continue decoding");


    auto missing_state = before_failure;
    missing_state.state.valid =
        strix::PersistentState::none;
    bool missing_state_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                missing_state,
                memory,
                1,
                unused));
    } catch (const std::invalid_argument&) {
        missing_state_rejected = true;
    }
    expect(missing_state_rejected &&
               missing_state.health ==
                   strix::SerialSessionHealth::ready,
           "preflight state failure does not poison session");

    auto wrong_identity = before_failure;
    ++wrong_identity.execution_identity;
    bool identity_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                wrong_identity,
                memory,
                1,
                unused));
    } catch (const std::invalid_argument&) {
        identity_rejected = true;
    }
    expect(identity_rejected,
           "session compiled-plan identity mismatch is rejected");

    bool token_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                before_failure,
                memory,
                static_cast<std::uint32_t>(
                    compiled.vocab_size),
                unused));
    } catch (const std::invalid_argument&) {
        token_rejected = true;
    }
    expect(token_rejected,
           "out-of-vocabulary input token is rejected before dispatch");

    auto full_context = before_failure;
    full_context.next_position =
        compiled.max_position_embeddings;
    bool context_rejected = false;
    try {
        SessionDispatcher unused;
        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                full_context,
                memory,
                1,
                unused));
    } catch (const std::out_of_range&) {
        context_rejected = true;
    }
    expect(context_rejected,
           "maximum context is enforced before dispatch");

    if (failures == 0) {
        std::cout << "serial session tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
