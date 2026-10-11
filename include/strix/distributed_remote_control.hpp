#pragma once
#include "strix/distributed_native_backend.hpp"
#include <cstdint>
namespace strix {
struct DistributedRemoteStepCommand {
    std::uint32_t protocol_version = 1;
    std::uint64_t placement_plan_identity = 0;
    std::uint64_t request_id = 0;
    std::uint64_t generation = 0;
    std::uint64_t sequence = 0;
    std::uint32_t step_id = 0;
    SerialOp op = SerialOp::token_embedding;
    std::uint32_t target_node = 0;
};
struct DistributedRemoteStepAck {
    std::uint32_t protocol_version = 1;
    std::uint64_t placement_plan_identity = 0;
    std::uint64_t request_id = 0;
    std::uint64_t generation = 0;
    std::uint64_t sequence = 0;
    std::uint32_t step_id = 0;
    std::uint32_t target_node = 0;
    bool success = false;
};
class DistributedControlWireBackend {
public:
    virtual ~DistributedControlWireBackend() = default;
    [[nodiscard]] virtual DistributedRemoteStepAck execute_remote_step(
        const DistributedRemoteStepCommand& command) = 0;
    virtual void publish_boundary_transfer(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) = 0;
    [[nodiscard]] virtual DistributedBoundaryTransfer await_boundary_transfer(
        const DistributedExecutionBoundary& boundary) = 0;
};
class SequencedDistributedNativeControlPlane final
    : public DistributedNativeControlPlane {
public:
    SequencedDistributedNativeControlPlane(
        const DistributedPlacementPlan& placement,
        DistributedRuntimeRequest request,
        DistributedControlWireBackend& wire,
        std::uint32_t protocol_version = 1);
    void execute_remote_step(const DistributedStepPlacement& step) override;
    void publish_boundary_transfer(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) override;
    [[nodiscard]] DistributedBoundaryTransfer await_boundary_transfer(
        const DistributedExecutionBoundary& boundary) override;
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] std::uint64_t next_sequence() const noexcept { return next_sequence_; }
private:
    [[nodiscard]] bool step_in_plan(const DistributedStepPlacement& step) const noexcept;
    [[nodiscard]] bool boundary_in_plan(const DistributedExecutionBoundary& boundary) const noexcept;
    void require_healthy() const;
    const DistributedPlacementPlan* placement_ = nullptr;
    DistributedRuntimeRequest request_{};
    DistributedControlWireBackend* wire_ = nullptr;
    std::uint32_t protocol_version_ = 1;
    std::uint64_t next_sequence_ = 1;
    std::uint64_t next_boundary_send_sequence_ = 1;
    std::uint64_t next_boundary_receive_sequence_ = 1;
    bool failed_ = false;
};
}  // namespace strix
