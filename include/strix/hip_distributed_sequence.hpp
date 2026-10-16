#pragma once
#include "strix/hip_distributed_transaction.hpp"
#include "strix/direct_gpu_boundary_staging.hpp"
#include "strix/distributed_control_receiver.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <memory>
#include <span>
#include <vector>

namespace strix {
// Actual externally commissioned native owners, not a capability declaration.
// Their aggregate pin covers driver, pools, lease registry, GPU consumers,
// dedicated kind4 transport progress and the live session's borrowed metadata.
struct HipDistributedSequenceTransport {
    std::shared_ptr<transport::DirectTransportNativeLinkSession> link;
    std::shared_ptr<transport::DirectGpuStreamGpu> gpu;
    std::shared_ptr<transport::DirectGpuBoundaryReceiveIo> receive;
    std::shared_ptr<transport::DualLinkRuntimeScheduler> scheduler;
    std::shared_ptr<const void> actual_owner;
    std::shared_ptr<DistributedControlByteStream> step_control, token_control;
};
struct HipDistributedSequenceSetup {
    std::shared_ptr<HipDistributedTokenParticipant> participant;
    CompiledSerialExecution serial;
    DistributedPlacementPlan canonical_plan;
    ModelContract model;
    ExecutionRuntimeResources resources;
    HipLaunchContext scope;
    std::shared_ptr<const void> actual_model_owner;
    HipDistributedSequenceTransport transport;
    std::uint32_t local_node = 0;
    std::uint32_t maximum_inputs = 16;
    // Bounds full FP32 terminal-row payload only, not mapping/snapshot/pool RSS.
    std::uint64_t maximum_logit_payload_bytes = 0;
};
struct HipDistributedKnownInputResult {
    DistributedTokenCommit commit;
    std::uint32_t input_token = 0;
    // Present only on terminal peer1 AFTER both ACKs and actual retirement.
    std::vector<float> terminal_logits;
    std::uint32_t terminal_firstmax = 0;
};
struct HipDistributedSequenceResult {
    std::uint32_t local_node = 0;
    bool coordinator_confirmation = false;
    std::vector<HipDistributedKnownInputResult> inputs;
};
// One bounded literal known-input sequence, no sampling/greedy generation or
// reconnect/retry. Exactly one 0->1 cut boundary, one qualified native link.
// Both peers call execute with the same input sequence and negotiated binding.
// The ordinary serial anchor stays quarantined after a distributed commit.
// Serialized caller access only. Destroying after an uncertain external DMA
// outcome must close the actual native lease; inability to prove release is
// fail-stop, never normal owner unwinding or a GPU-only completion assertion.
class HipDistributedKnownSequence final {
public:
    explicit HipDistributedKnownSequence(HipDistributedSequenceSetup);
    ~HipDistributedKnownSequence();
    HipDistributedKnownSequence(const HipDistributedKnownSequence&) = delete;
    HipDistributedKnownSequence& operator=(const HipDistributedKnownSequence&) = delete;
    [[nodiscard]] HipDistributedSequenceResult execute(std::span<const std::uint32_t> known_inputs);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
