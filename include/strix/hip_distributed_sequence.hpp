#pragma once
#include "strix/hip_distributed_transaction.hpp"
#include "strix/direct_gpu_boundary_staging.hpp"
#include "strix/distributed_control_receiver.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <cstdint>
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
    // Explicit independent mode. Legacy execute(span) remains single-use 2..16.
    // Streaming admits this entire positive target-count limit at construction;
    // it retains ONE terminal row, not one row per possible target.
    bool streaming = false;
    std::uint64_t max_target_inputs = 0;
    // Explicit complete-layer cut: one mlp_inject(L)->attn_hyper_mix(L+1)
    // data edge plus <=64 canonical position_prepare->suffix-QSA dependencies.
    // Logical position proof does not replace actual data retirement.
    bool layer_boundary_position_dependencies = false;
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
struct HipDistributedInputCompletion {
    DistributedTokenCommit commit;
    std::uint32_t input_token = 0;
    // Present only on terminal_peer, computed from the actual finite full row.
    std::uint32_t terminal_firstmax = 0, terminal_firstmax_bits = 0;
    bool terminal_peer = false;
};
// Bounded literal known-input sequence or explicit retained one-input mode.
// No sampling/emission policy or reconnect/retry. Exactly one 0->1 cut boundary,
// one qualified native link. Both peers execute the same negotiated inputs.
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
    // Streaming mode only. No EOS, output-budget or emission policy is applied.
    // The first input must match the negotiated initial binding. Every return
    // follows BOTH committed ACKs and a real local native retirement proof.
    [[nodiscard]] HipDistributedInputCompletion execute_input(std::uint32_t input_token);
    // Terminal peer only, streaming mode only, after a successful execute_input.
    // Borrowed until the next admitted call or destruction; serialized access.
    // A failed transaction invalidates publication. No full-row wire transport.
    // Caller copies its small reply before submitting the next input.
    [[nodiscard]] std::span<const float> last_terminal_logits() const noexcept;
    [[nodiscard]] std::uint64_t completed_inputs() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix
