#include "strix/speculation_sampling_executor.hpp"

#include <exception>
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

struct SessionMetadata {
    std::uint64_t execution_identity = 0;
    std::uint64_t state_layout_identity = 0;
    std::uint64_t runtime_compatibility_identity = 0;
    std::uint64_t next_position = 0;
    std::uint64_t committed_tokens = 0;
    SerialSessionHealth health = SerialSessionHealth::ready;
    PersistentStateGenerations state;
};

SessionMetadata snapshot_metadata(
    const SerialSessionContext& session) {
    return {
        session.execution_identity,
        session.state_layout_identity,
        session.runtime_compatibility_identity,
        session.next_position,
        session.committed_tokens,
        session.health,
        session.state,
    };
}

bool same_identity(
    const SerialSessionContext& session,
    const SessionMetadata& base) noexcept {
    return session.execution_identity ==
               base.execution_identity &&
           session.state_layout_identity ==
               base.state_layout_identity &&
           session.runtime_compatibility_identity ==
               base.runtime_compatibility_identity;
}

void validate_session_binding(
    const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources,
    const SerialSessionContext& session) {
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
    if (session.execution_identity !=
            compiled.execution_identity ||
        session.state_layout_identity !=
            resources.session_state_layout->layout_identity ||
        session.runtime_compatibility_identity !=
            identities.compatibility_identity) {
        throw std::invalid_argument(
            "sampled speculation session identity does not match runtime resources");
    }
    if (session.health !=
        SerialSessionHealth::ready) {
        throw std::logic_error(
            "sampled speculation requires a ready serial session");
    }
}

void restore_base(
    SerialSessionContext& session,
    const ModelContract& model,
    const ExecutionRuntimeResources& resources,
    const SerialSessionSnapshotPlan& snapshot,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& transfer) {
    const auto restore =
        build_serial_session_restore_plan(
            model,
            *resources.session_state_layout,
            session.runtime_compatibility_identity,
            snapshot);
    restore_serial_session_snapshot(
        session,
        restore,
        *resources.session_state_layout,
        *resources.session_state_memory,
        snapshot_memory,
        transfer);
}

}  // namespace

SampledSpeculationExecutionResult
execute_sampled_speculation_cycle(
    const ModelContract& model,
    const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    const SampledSpeculationExecutionRequest& request,
    RuntimeMemoryAllocator& snapshot_allocator,
    AllocationGenerationSource& allocation_generations,
    SessionSnapshotTransferBackend& snapshot_transfer,
    SampledSpeculationExecutionBackend& backend) {
    validate_session_binding(
        compiled,
        resources,
        session);
    if (request.token_budget == 0) {
        throw std::invalid_argument(
            "sampled speculation token budget must be non-zero");
    }
    if (request.draft_probabilities.size() !=
        request.candidate_tokens.size()) {
        throw std::invalid_argument(
            "sampled speculation requires one draft distribution per candidate");
    }

    const auto base =
        snapshot_metadata(
            session);

    SampledSpeculationCycleRequest prepare_request;
    prepare_request.compiled_execution_identity =
        compiled.execution_identity;
    prepare_request.runtime_compatibility_identity =
        session.runtime_compatibility_identity;
    prepare_request.base_position =
        session.next_position;
    prepare_request.committed_tokens =
        session.committed_tokens;
    prepare_request.state =
        session.state;
    prepare_request.candidate_tokens =
        request.candidate_tokens;
    prepare_request.token_budget =
        request.token_budget;
    prepare_request.eos_tokens =
        request.eos_tokens;
    prepare_request.snapshot_alignment =
        request.snapshot_alignment;

    const auto preparation =
        prepare_sampled_speculation_cycle(
            model,
            *resources.session_state_layout,
            prepare_request);

    SessionSnapshotMemoryOwner snapshot_owner;
    bool snapshot_captured = false;
    if (!preparation.candidate_tokens.empty()) {
        snapshot_owner =
            SessionSnapshotMemoryOwner::allocate(
                preparation.pre_window_snapshot,
                snapshot_allocator,
                allocation_generations);
        capture_serial_session_snapshot(
            preparation.pre_window_snapshot,
            *resources.session_state_layout,
            *resources.session_state_memory,
            snapshot_owner.binding(),
            snapshot_transfer);
        snapshot_captured = true;
    }

    std::vector<std::vector<double>>
        target_probabilities;
    try {
        target_probabilities =
            backend.verify_target_distributions(
                session,
                resources,
                preparation.candidate_tokens);
    } catch (...) {
        const auto original =
            std::current_exception();
        if (snapshot_captured) {
            try {
                restore_base(
                    session,
                    model,
                    resources,
                    preparation.pre_window_snapshot,
                    snapshot_owner.binding(),
                    snapshot_transfer);
            } catch (...) {
                session.health =
                    SerialSessionHealth::poisoned;
                throw;
            }
        } else if (!same_identity(session, base) ||
                   session.next_position !=
                       base.next_position ||
                   session.committed_tokens !=
                       base.committed_tokens ||
                   session.state.generations !=
                       base.state.generations ||
                   session.state.valid !=
                       base.state.valid) {
            session.health =
                SerialSessionHealth::poisoned;
        }
        std::rethrow_exception(original);
    }

    if (!same_identity(session, base) ||
        session.health !=
            SerialSessionHealth::ready) {
        if (snapshot_captured) {
            restore_base(
                session,
                model,
                resources,
                preparation.pre_window_snapshot,
                snapshot_owner.binding(),
                snapshot_transfer);
        } else {
            session.health =
                SerialSessionHealth::poisoned;
        }
        throw std::logic_error(
            "sampled target verification altered session identity or health");
    }

    const auto depth =
        static_cast<std::uint64_t>(
            preparation.candidate_tokens.size());
    const auto expected_position =
        checked_add(
            base.next_position,
            depth,
            "sampled verifier position overflow");
    const auto expected_committed =
        checked_add(
            base.committed_tokens,
            depth,
            "sampled verifier committed-token overflow");

    if (snapshot_captured) {
        if (session.next_position !=
                expected_position ||
            session.committed_tokens !=
                expected_committed) {
            restore_base(
                session,
                model,
                resources,
                preparation.pre_window_snapshot,
                snapshot_owner.binding(),
                snapshot_transfer);
            throw std::logic_error(
                "sampled verifier did not advance through every candidate");
        }
    } else if (session.next_position !=
                   base.next_position ||
               session.committed_tokens !=
                   base.committed_tokens ||
               session.state.generations !=
                   base.state.generations ||
               session.state.valid !=
                   base.state.valid) {
        session.health =
            SerialSessionHealth::poisoned;
        throw std::logic_error(
            "depth-zero sampled verifier mutated serial state");
    }

    SampledSpeculationFinalizeRequest finalize;
    finalize.target_probabilities =
        target_probabilities;
    finalize.draft_probabilities =
        request.draft_probabilities;
    finalize.rng =
        request.rng;
    finalize.token_budget =
        request.token_budget;
    finalize.eos_tokens =
        request.eos_tokens;

    SampledSpeculationExecutionResult result;
    try {
        result.plan =
            finalize_sampled_speculation_cycle(
                model,
                *resources.session_state_layout,
                preparation,
                finalize);
    } catch (...) {
        const auto original =
            std::current_exception();
        if (snapshot_captured) {
            try {
                restore_base(
                    session,
                    model,
                    resources,
                    preparation.pre_window_snapshot,
                    snapshot_owner.binding(),
                    snapshot_transfer);
            } catch (...) {
                session.health =
                    SerialSessionHealth::poisoned;
                throw;
            }
        }
        std::rethrow_exception(original);
    }

    result.snapshot_captured =
        snapshot_captured;

    if (result.plan.restore_required) {
        restore_base(
            session,
            model,
            resources,
            preparation.pre_window_snapshot,
            snapshot_owner.binding(),
            snapshot_transfer);
        result.restored_pre_window = true;

        for (const auto token :
             result.plan.replay_tokens) {
            try {
                backend.replay_target_token(
                    session,
                    resources,
                    token);
            } catch (...) {
                const auto original =
                    std::current_exception();
                try {
                    restore_base(
                        session,
                        model,
                        resources,
                        preparation.pre_window_snapshot,
                        snapshot_owner.binding(),
                        snapshot_transfer);
                } catch (...) {
                    session.health =
                        SerialSessionHealth::poisoned;
                    throw;
                }
                std::rethrow_exception(original);
            }
            ++result.replayed_tokens;
        }
    }

    if (session.next_position !=
            result.plan.post_cycle_state_position ||
        session.committed_tokens !=
            result.plan.post_cycle_committed_tokens ||
        session.health !=
            SerialSessionHealth::ready) {
        if (snapshot_captured) {
            try {
                restore_base(
                    session,
                    model,
                    resources,
                    preparation.pre_window_snapshot,
                    snapshot_owner.binding(),
                    snapshot_transfer);
            } catch (...) {
                session.health =
                    SerialSessionHealth::poisoned;
                throw;
            }
        } else {
            session.health =
                SerialSessionHealth::poisoned;
        }
        throw std::logic_error(
            "sampled speculation finished with inconsistent target state");
    }

    return result;
}

}  // namespace strix
