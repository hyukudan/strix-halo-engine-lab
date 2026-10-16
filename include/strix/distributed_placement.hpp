#pragma once

#include "strix/compiled_execution.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/model_manifest.hpp"
#include "strix/transport.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct DistributedStepPlacement {
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::uint32_t node = 0;
    bool remote = false;
    bool placement_ambiguous = false;
};

struct DistributedExecutionBoundary {
    std::uint64_t boundary_id = 0;
    std::uint32_t source_step = 0;
    std::uint32_t destination_step = 0;
    std::uint32_t source_node = 0;
    std::uint32_t destination_node = 0;
    std::uint64_t payload_bytes = 0;
    std::vector<ExecutionBufferUse> payloads;
};

struct DistributedPlacementPlan {
    std::uint32_t local_node = 0;
    std::uint64_t compiled_execution_identity = 0;
    std::uint64_t manifest_layout_identity = 0;
    std::uint64_t tensor_access_identity = 0;

    std::vector<DistributedStepPlacement> steps;
    std::vector<DistributedExecutionBoundary> boundaries;
    std::vector<std::uint32_t> remote_nodes;

    std::uint64_t boundary_payload_bytes_per_token = 0;
    bool has_remote_work = false;
    bool executable = false;
    std::uint64_t plan_identity = 0;
};

[[nodiscard]] DistributedPlacementPlan build_distributed_placement_plan(
    const CompiledSerialExecution& compiled,
    const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& tensor_access,
    std::uint32_t local_node = 0);

// Explicit compute placement, independent of immutable local weight copies.
// The initial native stage component closes nodes to0/1 and requires one node
// per dense serial step. Boundaries and identity are derived, never supplied.
[[nodiscard]] DistributedPlacementPlan build_distributed_placement_plan(
    const CompiledSerialExecution& compiled, const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& tensor_access,
    std::span<const std::uint32_t> compute_nodes, std::uint32_t local_node = 0);

enum class DistributedExecutionMode : std::uint8_t {
    local_only,
    distributed,
    local_fallback,
    unavailable,
};

struct DistributedFallbackPolicy {
    bool local_model_copy_available = false;
};

struct DistributedExecutionDecision {
    DistributedExecutionMode mode =
        DistributedExecutionMode::unavailable;
    std::vector<std::uint32_t> missing_remote_nodes;
};

[[nodiscard]] DistributedExecutionDecision
select_distributed_execution_mode(
    const DistributedPlacementPlan& plan,
    std::span<const std::uint32_t> available_remote_nodes,
    DistributedFallbackPolicy fallback = {});

[[nodiscard]] transport::TransferDescriptor
make_distributed_boundary_descriptor(
    const DistributedPlacementPlan& plan,
    const DistributedExecutionBoundary& boundary,
    std::uint64_t connection_epoch,
    std::uint64_t request_id,
    std::uint64_t generation,
    std::uint64_t sequence,
    std::uint32_t dtype_id);

}  // namespace strix
