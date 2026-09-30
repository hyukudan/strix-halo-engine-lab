#include "strix/speculative_verify_runtime.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    const char* message) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(message);
    }
    return left + right;
}

}  // namespace

SpeculativeVerifyExecutionResult
execute_compiled_speculative_verify_eager(
    const CompiledSpeculativeVerifyExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    std::span<const std::uint32_t> candidate_tokens,
    EagerExecutionDispatcher& dispatcher) {
    if (compiled.execution_identity == 0 ||
        compiled.candidate_depth == 0 ||
        compiled.request_concurrency != 1) {
        throw std::invalid_argument(
            "speculative verify execution is not initialized");
    }
    if (candidate_tokens.size() !=
        compiled.candidate_depth) {
        throw std::invalid_argument(
            "candidate token count does not match compiled verify depth");
    }
    if (session.health !=
        SerialSessionHealth::ready) {
        throw std::logic_error(
            "speculative verify requires ready serial session");
    }
    if (session.state_layout_identity == 0 ||
        resources.session_state_layout == nullptr ||
        session.state_layout_identity !=
            resources.session_state_layout->layout_identity) {
        throw std::invalid_argument(
            "speculative verify state layout does not match serial session");
    }

    validate_execution_runtime_resources(
        compiled.execution_identity,
        compiled.arena,
        resources,
        session.next_position);

    const auto depth =
        static_cast<std::uint64_t>(
            compiled.candidate_depth);
    const auto post_position =
        checked_add(
            session.next_position,
            depth,
            "speculative verify position overflow");
    const auto post_committed =
        checked_add(
            session.committed_tokens,
            depth,
            "speculative verify committed-token overflow");

    if (post_position >
        resources.session_state_layout->context_tokens) {
        throw std::out_of_range(
            "speculative verify exceeds allocated session-state capacity");
    }

    EagerExecutionRequest request;
    request.logical_rows =
        compiled.candidate_depth;
    request.request_concurrency = 1;
    request.input_token_id =
        candidate_tokens.front();
    request.input_tokens =
        candidate_tokens;
    request.position =
        session.next_position;
    request.state =
        session.state;

    // All deterministic shape/resource/state validation happens before the
    // first dispatch. After dispatch begins, a failing verify may have mutated
    // target persistent state and therefore poisons the session until restored.
    validate_eager_execution_preflight(
        compiled.topology,
        compiled.bindings,
        compiled.io,
        compiled.arena,
        request,
        resources.session_state_layout,
        resources.session_state_memory,
        &resources);

    SpeculativeVerifyExecutionResult result;
    result.base_position =
        session.next_position;
    result.base_committed_tokens =
        session.committed_tokens;
    result.post_position =
        post_position;
    result.post_committed_tokens =
        post_committed;

    try {
        result.execution =
            execute_serial_eager(
                compiled.topology,
                compiled.bindings,
                compiled.io,
                compiled.arena,
                request,
                dispatcher,
                resources.scratch,
                resources.session_state_layout,
                resources.session_state_memory,
                &resources,
                compiled.execution_identity);
    } catch (...) {
        session.health =
            SerialSessionHealth::poisoned;
        throw;
    }

    session.state =
        result.execution.state;
    session.next_position =
        post_position;
    session.committed_tokens =
        post_committed;

    return result;
}

}  // namespace strix
