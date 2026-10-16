#include "strix/prepared_serial_session.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
// Preserves the existing serial session's preflight, PLE, completion hook and
// poison/commit boundaries. Full admission runs once, before PLE starts; the
// private runner dispatches that same admitted request. Device completion
// remains the caller's responsibility.
PreparedEagerExecutionResult execute_prepared_serial_token(
    const PreparedEagerExecution& prepared,
    SerialSessionContext& session,
    const ExecutionRuntimeResources& resources,
    std::uint32_t input_token_id,
    EagerExecutionDispatcher& dispatcher,
    const std::function<void()>& before_commit) {
    const auto& compiled = prepared.compiled();
    const EagerExecutionRequest request = [&] {
        EagerExecutionRequest value;
        value.logical_rows = 1;
        value.request_concurrency = 1;
        value.input_token_id = input_token_id;
        value.position = session.next_position;
        value.state = session.state;
        return value;
    }();
    prepared.validate_dynamic(resources, request);

    if (session.runtime_compatibility_identity == 0 ||
        session.runtime_compatibility_identity !=
            prepared.compatibility_identity()) {
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

    if (resources.ple_lookup != nullptr &&
        resources.ple_hash_contract != nullptr &&
        resources.ple_prefetch != nullptr) {
        const auto ple_step = prepared.ple_step_id();
        if (!ple_step) {
            throw std::logic_error(
                "PLE prefetch resources supplied to plan without PLE step");
        }
        start_ple_lookup_prefetch(
            *resources.ple_hash_contract,
            *ple_step,
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

    PreparedEagerExecutionResult result;
    try {
        result =
            prepared.run_admitted(resources, request, dispatcher);
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

} // namespace strix
