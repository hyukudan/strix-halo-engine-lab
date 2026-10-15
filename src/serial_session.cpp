#include "strix/serial_session.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {


EagerExecutionResult execute_serial_token(
    const CompiledSerialExecution& compiled,
    SerialSessionContext& session,
    const ExecutionRuntimeResources& resources,
    std::uint32_t input_token_id,
    EagerExecutionDispatcher& dispatcher,
    const std::function<void()>& before_commit) {
    validate_execution_runtime_resources(
        compiled.execution_identity,
        compiled.arena,
        resources,
        session.next_position);

    const auto identities =
        build_runtime_execution_identities(
            compiled.execution_identity,
            compiled.arena,
            resources);

    if (session.runtime_compatibility_identity == 0 ||
        session.runtime_compatibility_identity !=
            identities.compatibility_identity) {
        throw std::invalid_argument(
            "serial session runtime compatibility identity does not match resources");
    }
    if (session.state_layout_identity == 0 ||
        session.state_layout_identity !=
            resources.session_state_layout->layout_identity) {
        throw std::invalid_argument(
            "serial session state-layout identity does not match resources");
    }
    if (session.health != SerialSessionHealth::ready) {
        throw std::logic_error(
            "serial session is poisoned and requires recovery");
    }
    if (compiled.logical_rows != 1 ||
        compiled.request_concurrency != 1) {
        throw std::invalid_argument(
            "serial token execution requires a compiled C1 one-row plan");
    }
    if (compiled.execution_identity == 0 ||
        session.execution_identity !=
            compiled.execution_identity) {
        throw std::invalid_argument(
            "serial session execution identity does not match compiled plan");
    }
    if (compiled.vocab_size == 0 ||
        static_cast<std::uint64_t>(input_token_id) >=
            compiled.vocab_size) {
        throw std::invalid_argument(
            "serial input token is outside compiled vocabulary");
    }
    if (session.next_position >=
            compiled.max_position_embeddings ||
        session.next_position >=
            resources.session_state_layout->context_tokens) {
        throw std::out_of_range(
            "serial session reached allocated context capacity");
    }
    if (session.committed_tokens ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "serial committed-token counter overflow");
    }

    auto next_lineage = prepare_serial_row_lineage(session.arithmetic_lineage,
        session.next_position, compiled.execution_identity, compiled.kernel_plan_identity);

    EagerExecutionRequest request;
    request.logical_rows = 1;
    request.request_concurrency = 1;
    request.input_token_id = input_token_id;
    request.position = session.next_position;
    request.state = session.state;

    validate_eager_execution_preflight(
        compiled.execution,
        compiled.bindings,
        compiled.io,
        compiled.arena,
        request,
        resources.session_state_layout,
        resources.session_state_memory);

    if (resources.ple_lookup != nullptr &&
        resources.ple_hash_contract != nullptr &&
        resources.ple_prefetch != nullptr) {
        const auto found =
            std::find_if(
                compiled.execution.steps.begin(),
                compiled.execution.steps.end(),
                [](const ExecutionStep& step) {
                    return step.op == SerialOp::ple;
                });
        if (found == compiled.execution.steps.end()) {
            throw std::logic_error(
                "PLE prefetch resources supplied to plan without PLE step");
        }
        start_ple_lookup_prefetch(
            *resources.ple_hash_contract,
            found->id,
            input_token_id,
            session.next_position,
            *resources.model_manifest,
            *resources.model_weights,
            *resources.tensor_access,
            *resources.session_state_layout,
            *resources.session_state_memory,
            *resources.ple_lookup,
            *resources.ple_prefetch);
    }

    EagerExecutionResult result;
    try {
        result =
            execute_compiled_serial_eager(
                compiled,
                resources,
                request,
                dispatcher);
        if (resources.ple_prefetch != nullptr &&
            resources.ple_prefetch->active) {
            cancel_ple_lookup_prefetch(
                *resources.ple_lookup,
                *resources.ple_prefetch);
            throw std::logic_error(
                "serial execution completed without consuming PLE prefetch");
        }
        // Completion/status and transactional host history precede session
        // generations, lineage, position and committed-token publication.
        if (before_commit) before_commit();
    } catch (...) {
        if (resources.ple_lookup != nullptr &&
            resources.ple_prefetch != nullptr) {
            cancel_ple_lookup_prefetch(
                *resources.ple_lookup,
                *resources.ple_prefetch);
        }
        session.health =
            SerialSessionHealth::poisoned;
        session.arithmetic_lineage.tainted = true;
        throw;
    }

    session.state = result.state;
    session.arithmetic_lineage = std::move(next_lineage);
    ++session.next_position;
    ++session.committed_tokens;
    return result;
}

EagerExecutionResult execute_serial_token(
    const CompiledSerialExecution& compiled,
    SerialSessionContext& session,
    const ExecutionMemoryBinding& memory,
    const SerialSessionStateLayout& session_state_layout,
    const SessionStateMemoryBinding& session_state_memory,
    std::uint32_t input_token_id,
    EagerExecutionDispatcher& dispatcher,
    const std::function<void()>& before_commit) {
    if (session.runtime_compatibility_identity != 0) {
        throw std::invalid_argument(
            "model-bound serial session requires complete runtime resources");
    }

    validate_execution_memory_binding(
        compiled.execution_identity,
        compiled.arena,
        memory);
    validate_session_state_memory_binding(
        session_state_layout,
        session_state_memory);

    if (session.state_layout_identity == 0 ||
        session.state_layout_identity !=
            session_state_layout.layout_identity) {
        throw std::invalid_argument(
            "serial session state-layout identity does not match allocated state");
    }

    if (session.health != SerialSessionHealth::ready) {
        throw std::logic_error(
            "serial session is poisoned and requires recovery");
    }
    if (compiled.logical_rows != 1 ||
        compiled.request_concurrency != 1) {
        throw std::invalid_argument(
            "serial token execution requires a compiled C1 one-row plan");
    }
    if (compiled.execution_identity == 0 ||
        session.execution_identity !=
            compiled.execution_identity) {
        throw std::invalid_argument(
            "serial session execution identity does not match compiled plan");
    }
    if (compiled.vocab_size == 0 ||
        static_cast<std::uint64_t>(input_token_id) >=
            compiled.vocab_size) {
        throw std::invalid_argument(
            "serial input token is outside compiled vocabulary");
    }
    if (compiled.max_position_embeddings == 0 ||
        session.next_position >=
            compiled.max_position_embeddings ||
        session.next_position >=
            session_state_layout.context_tokens) {
        throw std::out_of_range(
            "serial session reached allocated context capacity");
    }
    if (session.committed_tokens ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "serial committed-token counter overflow");
    }

    auto next_lineage = prepare_serial_row_lineage(session.arithmetic_lineage,
        session.next_position, compiled.execution_identity, compiled.kernel_plan_identity);

    EagerExecutionRequest request;
    request.logical_rows = 1;
    request.request_concurrency = 1;
    request.input_token_id = input_token_id;
    request.position = session.next_position;
    request.state = session.state;

    // Reject every deterministic plan/state error before dispatch. These
    // failures do not mutate persistent device state and therefore do not
    // poison the session.
    validate_eager_execution_preflight(
        compiled.execution,
        compiled.bindings,
        compiled.io,
        compiled.arena,
        request,
        &session_state_layout,
        &session_state_memory);

    // execute_compiled_serial_eager works on a copy of the state-generation
    // metadata. Once dispatch begins, however, persistent device state may be
    // modified in place; any subsequent failure therefore poisons the session.
    EagerExecutionResult result;
    try {
        result =
            execute_compiled_serial_eager(
                compiled,
                memory,
                session_state_layout,
                session_state_memory,
                request,
                dispatcher);
        if (before_commit) before_commit();
    } catch (...) {
        // Preflight inside execute_compiled_serial_eager runs before the first
        // dispatcher call, but once dispatch begins persistent device state may
        // have been modified in place. Without an explicit rollback snapshot we
        // cannot safely reuse the session after an execution failure.
        session.health =
            SerialSessionHealth::poisoned;
        session.arithmetic_lineage.tainted = true;
        throw;
    }

    session.state = result.state;
    session.arithmetic_lineage = std::move(next_lineage);
    ++session.next_position;
    ++session.committed_tokens;

    return result;
}

}  // namespace strix
