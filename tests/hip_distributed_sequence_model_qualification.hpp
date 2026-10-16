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
    // This callable case is not general link, memory-sharding, distributed
    // generation, crash-recovery or model-quality qualification.
};
[[nodiscard]] DistributedSequenceQualificationResult qualify_hip_distributed_known_sequence(
    HipDistributedSequenceSetup actual_setup, const DistributedSequenceReference& full_serial_reference,
    std::span<const std::uint32_t> known_inputs, const ModelContentIdentity& pinned_model_content,
    const SerialSessionContext& actual_ordinary_anchor);
} // namespace strix::qualification
