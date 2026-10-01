#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/draft_proposal_state.hpp"
#include "strix/execution_resources.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_snapshot_memory.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include "strix/speculation_cycle.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

class GreedyDraftProposalBackend {
public:
    virtual ~GreedyDraftProposalBackend() = default;

    // Lifecycle metadata is runtime-owned and read-only to the drafter.
    // Concrete proposal caches/state remain backend-owned. The target session
    // is read-only here so proposal generation cannot mutate committed state.
    [[nodiscard]] virtual std::vector<std::uint32_t>
    propose_candidates(
        const SerialSessionContext& target_session,
        const DraftProposalStateContext& proposal_state,
        std::uint32_t max_candidate_depth) = 0;

    // Reconcile concrete drafter caches/state to the committed target state
    // after target verification/replay has completed successfully.
    virtual void reconcile_after_target(
        const DraftProposalStateContext& proposal_state,
        const SerialSessionContext& target_session,
        std::span<const std::uint32_t> emitted_tokens) = 0;
};

class GreedySpeculationExecutionBackend {
public:
    virtual ~GreedySpeculationExecutionBackend() = default;

    // Evaluate the target across every candidate input token and return
    // candidate_length + 1 greedy target selections. For positive depth, the
    // backend MUST leave target persistent state advanced through all candidate
    // inputs, but MUST NOT process the final bonus token into target state.
    [[nodiscard]] virtual std::vector<std::uint32_t>
    verify_target_candidates(
        SerialSessionContext& session,
        const ExecutionRuntimeResources& resources,
        std::span<const std::uint32_t> candidate_tokens) = 0;

    // Replay one accepted target input token through the exact serial target
    // path after restoring the pre-window snapshot.
    virtual void replay_target_token(
        SerialSessionContext& session,
        const ExecutionRuntimeResources& resources,
        std::uint32_t token) = 0;
};

struct GreedySpeculationExecutionRequest {
    std::span<const std::uint32_t> candidate_tokens;
    std::uint32_t token_budget = 1;
    std::span<const std::uint32_t> eos_tokens;
    std::uint64_t snapshot_alignment = 64;
};

struct GreedySpeculationExecutionResult {
    GreedySpeculationCyclePlan plan;
    std::uint32_t replayed_tokens = 0;
    bool snapshot_captured = false;
    bool restored_pre_window = false;
};

struct GreedyDraftExecutionRequest {
    std::uint32_t max_candidate_depth = 0;
    std::uint32_t token_budget = 1;
    std::span<const std::uint32_t> eos_tokens;
    std::uint64_t snapshot_alignment = 64;
};

struct GreedyDraftExecutionResult {
    std::uint32_t proposal_depth_limit = 0;
    std::vector<std::uint32_t> proposed_tokens;
    GreedySpeculationExecutionResult target_cycle;
};

[[nodiscard]] GreedySpeculationExecutionResult
execute_greedy_speculation_cycle(
    const ModelContract& model,
    const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    const GreedySpeculationExecutionRequest& request,
    RuntimeMemoryAllocator& snapshot_allocator,
    AllocationGenerationSource& allocation_generations,
    SessionSnapshotTransferBackend& snapshot_transfer,
    GreedySpeculationExecutionBackend& backend);

[[nodiscard]] GreedyDraftExecutionResult
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
    GreedySpeculationExecutionBackend& target_backend);

}  // namespace strix
