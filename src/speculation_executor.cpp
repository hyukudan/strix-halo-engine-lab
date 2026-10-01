#include "strix/speculation_executor.hpp"

#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {

std::uint64_t checked_session_add(
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
    SerialSessionHealth health =
        SerialSessionHealth::ready;
    PersistentStateGenerations state;
};

SessionMetadata snapshot_metadata(
    const SerialSessionContext& session) {
    return SessionMetadata{
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
            "speculation session identity does not match compiled runtime resources");
    }
    if (session.health !=
        SerialSessionHealth::ready) {
        throw std::logic_error(
            "speculation requires a ready serial session");
    }
}

SerialSessionRestorePlan build_restore(
    const ModelContract& model,
    const SerialSessionStateLayout& state_layout,
    const SerialSessionSnapshotPlan& snapshot,
    std::uint64_t runtime_compatibility_identity) {
    return build_serial_session_restore_plan(
        model,
        state_layout,
        runtime_compatibility_identity,
        snapshot);
}

void restore_base(
    SerialSessionContext& session,
    const ModelContract& model,
    const ExecutionRuntimeResources& resources,
    const SerialSessionSnapshotPlan& snapshot,
    const SessionSnapshotMemoryBinding& snapshot_memory,
    SessionSnapshotTransferBackend& transfer) {
    const auto restore =
        build_restore(
            model,
            *resources.session_state_layout,
            snapshot,
            session.runtime_compatibility_identity);

    restore_serial_session_snapshot(
        session,
        restore,
        *resources.session_state_layout,
        *resources.session_state_memory,
        snapshot_memory,
        transfer);
}

}  // namespace

GreedySpeculationExecutionResult
execute_greedy_speculation_cycle(
    const ModelContract& model,
    const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    const GreedySpeculationExecutionRequest& request,
    RuntimeMemoryAllocator& snapshot_allocator,
    AllocationGenerationSource& allocation_generations,
    SessionSnapshotTransferBackend& snapshot_transfer,
    GreedySpeculationExecutionBackend& backend) {
    validate_session_binding(
        compiled,
        resources,
        session);

    if (request.token_budget == 0) {
        throw std::invalid_argument(
            "speculation execution token budget must be non-zero");
    }

    const auto base =
        snapshot_metadata(session);

    GreedySpeculationCycleRequest prepare_request;
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
        prepare_greedy_speculation_cycle(
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

    std::vector<std::uint32_t> target_tokens;
    try {
        target_tokens =
            backend.verify_target_candidates(
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
        } else {
            // Depth zero has no recovery snapshot. A verifier failure is only
            // safe if it did not alter visible session metadata.
            if (!same_identity(session, base) ||
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
            "target verification altered session identity or health");
    }

    const auto depth =
        static_cast<std::uint64_t>(
            preparation.candidate_tokens.size());
    const auto expected_verify_position =
        checked_session_add(
            base.next_position,
            depth,
            "speculative verifier position overflow");
    const auto expected_verify_committed =
        checked_session_add(
            base.committed_tokens,
            depth,
            "speculative verifier committed-token overflow");

    if (snapshot_captured) {
        if (session.next_position !=
                expected_verify_position ||
            session.committed_tokens !=
                expected_verify_committed) {
            restore_base(
                session,
                model,
                resources,
                preparation.pre_window_snapshot,
                snapshot_owner.binding(),
                snapshot_transfer);
            throw std::logic_error(
                "target verifier did not leave state advanced through every candidate");
        }
    } else {
        if (session.next_position !=
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
                "depth-zero target verifier mutated serial session state");
        }
    }

    GreedySpeculationFinalizeRequest finalize_request;
    finalize_request.target_tokens =
        target_tokens;
    finalize_request.token_budget =
        request.token_budget;
    finalize_request.eos_tokens =
        request.eos_tokens;

    GreedySpeculationExecutionResult result;
    try {
        result.plan =
            finalize_greedy_speculation_cycle(
                model,
                *resources.session_state_layout,
                preparation,
                finalize_request);
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
            "speculation cycle finished with inconsistent committed target state");
    }

    return result;
}

GreedyDraftExecutionResult
execute_greedy_speculation_with_draft(
    const ModelContract& model,
    const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    const GreedyDraftExecutionRequest& request,
    RuntimeMemoryAllocator& snapshot_allocator,
    AllocationGenerationSource& allocation_generations,
    SessionSnapshotTransferBackend& snapshot_transfer,
    DraftProposalStateContext& draft_state,
    GreedyDraftProposalBackend& draft_backend,
    GreedySpeculationExecutionBackend& target_backend) {
    GreedyDraftExecutionResult result;

    validate_draft_proposal_state(
        draft_state,
        session);

    bool proposal_started = false;
    if (request.max_candidate_depth != 0) {
        begin_draft_proposal(
            draft_state,
            session);
        proposal_started = true;
        try {
            result.proposed_tokens =
                draft_backend.propose_candidates(
                    session,
                    draft_state,
                    request.max_candidate_depth);
            if (result.proposed_tokens.size() >
                static_cast<std::size_t>(
                    request.max_candidate_depth)) {
                throw std::invalid_argument(
                    "draft backend returned more candidates than requested");
            }
            finish_draft_proposal(
                draft_state);
        } catch (...) {
            poison_draft_proposal_state(
                draft_state);
            throw;
        }
    }

    GreedySpeculationExecutionRequest target_request;
    target_request.candidate_tokens =
        result.proposed_tokens;
    target_request.token_budget =
        request.token_budget;
    target_request.eos_tokens =
        request.eos_tokens;
    target_request.snapshot_alignment =
        request.snapshot_alignment;

    try {
        result.target_cycle =
            execute_greedy_speculation_cycle(
                model,
                compiled,
                resources,
                session,
                target_request,
                snapshot_allocator,
                allocation_generations,
                snapshot_transfer,
                target_backend);
    } catch (...) {
        if (proposal_started) {
            poison_draft_proposal_state(
                draft_state);
        }
        throw;
    }

    if (proposal_started) {
        try {
            draft_backend.reconcile_after_target(
                draft_state,
                session,
                result.target_cycle.plan.emitted_tokens);
            reconcile_draft_proposal_state(
                draft_state,
                session);
        } catch (...) {
            poison_draft_proposal_state(
                draft_state);
            throw;
        }
    }

    return result;
}

}  // namespace strix
