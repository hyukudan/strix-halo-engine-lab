#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_resources.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_snapshot_memory.hpp"
#include "strix/session_snapshot_transfer.hpp"
#include "strix/speculation_sampling_cycle.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

class SampledSpeculationExecutionBackend {
public:
    virtual ~SampledSpeculationExecutionBackend() = default;

    // Return one target distribution per candidate position plus one final
    // bonus distribution. For positive depth, target state must be left
    // advanced through every candidate input; depth zero must not mutate it.
    [[nodiscard]] virtual std::vector<std::vector<double>>
    verify_target_distributions(
        SerialSessionContext& session,
        const ExecutionRuntimeResources& resources,
        std::span<const std::uint32_t> candidate_tokens) = 0;

    virtual void replay_target_token(
        SerialSessionContext& session,
        const ExecutionRuntimeResources& resources,
        std::uint32_t token) = 0;
};

struct SampledSpeculationExecutionRequest {
    std::span<const std::uint32_t> candidate_tokens;
    std::span<const std::vector<double>> draft_probabilities;
    SpeculationRngIdentity rng;
    std::uint32_t token_budget = 1;
    std::span<const std::uint32_t> eos_tokens;
    std::uint64_t snapshot_alignment = 64;
};

struct SampledSpeculationExecutionResult {
    SampledSpeculationCyclePlan plan;
    std::uint32_t replayed_tokens = 0;
    bool snapshot_captured = false;
    bool restored_pre_window = false;
};

[[nodiscard]] SampledSpeculationExecutionResult
execute_sampled_speculation_cycle(
    const ModelContract& model,
    const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    const SampledSpeculationExecutionRequest& request,
    RuntimeMemoryAllocator& snapshot_allocator,
    AllocationGenerationSource& allocation_generations,
    SessionSnapshotTransferBackend& snapshot_transfer,
    SampledSpeculationExecutionBackend& backend);

}  // namespace strix
