#pragma once
#include "strix/hip_distributed_sequence.hpp"
#include "strix/model_content_hash.hpp"

namespace strix::qualification {
// Caller loads/pins the actual model once using the existing mapping/residency
// APIs. Reference has independent seven-state/scratch/D1/PLE owners; weights
// may share immutable residency. No bootstrap, synthetic capability or skip.
struct DistributedSequenceReference {
    SerialSessionContext* session = nullptr;
    ExecutionRuntimeResources resources;
    HipLauncherCatalogConfig config;
    HipLaunchContext scope;
    std::shared_ptr<const void> actual_owner;
    std::uint64_t maximum_logit_payload_bytes = 0;
};
struct DistributedSequenceQualificationResult {
    ModelContentIdentity model_content, input_u32le_sha256;
    std::uint32_t local_node = 0, known_inputs = 0;
    bool native_sequence_completed = false;
    bool terminal_rows_bitwise_equal = false; // terminal peer1 only
    bool locally_owned_state_equal = false;
    bool streaming_inputs = false;
    bool layer_boundary_position_dependencies = false;
    std::uint32_t data_boundaries = 0, logical_position_dependencies = 0;
    bool stage_owned_resources = false, state_partitioned = false;
    std::uint64_t stage_resource_identity = 0, state_mapping_identity = 0;
    std::uint64_t retained_state_bytes = 0;
    std::uint32_t compared_state_ranges = 0;
    // Explicit opt-in; returned pins retain the actual stage/participant.
    std::optional<HipDistributedOwnerInventory> owner_inventory;
    // This callable case is not general link, memory-sharding, distributed
    // generation, crash-recovery or model-quality qualification.
};
[[nodiscard]] DistributedSequenceQualificationResult qualify_hip_distributed_known_sequence(
    HipDistributedSequenceSetup actual_setup, const DistributedSequenceReference& full_serial_reference,
    std::span<const std::uint32_t> known_inputs, const ModelContentIdentity& pinned_model_content,
    const SerialSessionContext& actual_ordinary_anchor);
// Exercises the retained one-input transaction with a literal input oracle.
// One reusable actual row and one reusable reference row; this is not the
// autoregressive cursor/wire qualification surface. Requires explicit streaming
// setup and a fresh participant/independent serial reference on both peers.
[[nodiscard]] DistributedSequenceQualificationResult qualify_hip_distributed_streaming_inputs(
    HipDistributedSequenceSetup actual_setup, const DistributedSequenceReference& full_serial_reference,
    std::span<const std::uint32_t> known_inputs, const ModelContentIdentity& pinned_model_content,
    const SerialSessionContext& actual_ordinary_anchor);
// Requires an actual fresh setup compiled/bound at the complete-layer cut.
// Counts ALL canonical zero dependencies, while native staging transports only
// the one data payload. Supports the explicit known or streaming input mode.
[[nodiscard]] DistributedSequenceQualificationResult qualify_hip_distributed_layer_boundary_inputs(
    HipDistributedSequenceSetup actual_setup, const DistributedSequenceReference& full_serial_reference,
    std::span<const std::uint32_t> known_inputs, const ModelContentIdentity& pinned_model_content,
    const SerialSessionContext& actual_ordinary_anchor);
// Genuine typed stage owner/participant, no ordinary-session surrogate. The
// independent reference remains a complete serial session with separate state.
[[nodiscard]] DistributedSequenceQualificationResult qualify_hip_distributed_stage_inputs(
    HipDistributedSequenceSetup actual_setup, const DistributedSequenceReference& full_serial_reference,
    std::span<const std::uint32_t> known_inputs, const ModelContentIdentity& pinned_model_content,
    bool include_owner_inventory = false);
// Qualification-only payload comparison shared with the greedy fixture. Uses
// global logical offsets from the retained mapping, never component-prefix or
// full-flat comparison of absent state. Setup/reference retain all real owners.
void compare_hip_distributed_owned_state(
    const HipDistributedSequenceSetup&, const DistributedSequenceReference&);
} // namespace strix::qualification
