#include "strix/serial_session.hpp"

#include <limits>
#include <stdexcept>

namespace strix {

EagerExecutionResult execute_serial_token(
    const CompiledSerialExecution& compiled,
    SerialSessionContext& session,
    std::uint32_t input_token_id,
    EagerExecutionDispatcher& dispatcher) {
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
            compiled.max_position_embeddings) {
        throw std::out_of_range(
            "serial session reached compiled maximum context");
    }
    if (session.committed_tokens ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "serial committed-token counter overflow");
    }

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
        request);

    // execute_compiled_serial_eager works on a copy of the state-generation
    // metadata. Once dispatch begins, however, persistent device state may be
    // modified in place; any subsequent failure therefore poisons the session.
    EagerExecutionResult result;
    try {
        result =
            execute_compiled_serial_eager(
                compiled,
                request,
                dispatcher);
    } catch (...) {
        // Preflight inside execute_compiled_serial_eager runs before the first
        // dispatcher call, but once dispatch begins persistent device state may
        // have been modified in place. Without an explicit rollback snapshot we
        // cannot safely reuse the session after an execution failure.
        session.health =
            SerialSessionHealth::poisoned;
        throw;
    }

    session.state = result.state;
    ++session.next_position;
    ++session.committed_tokens;

    return result;
}

}  // namespace strix
