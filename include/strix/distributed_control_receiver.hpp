#pragma once

#include "strix/distributed_control_server.hpp"
#include "strix/distributed_native_backend.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace strix {

class DistributedControlReceiver final
    : public DistributedControlServerHandler {
public:
    DistributedControlReceiver(
        const DistributedPlacementPlan& placement,
        DistributedRuntimeRequest request,
        std::uint32_t receiver_node,
        DistributedNativeLocalExecutor& executor,
        std::uint32_t protocol_version = 1);

    [[nodiscard]] bool execute_remote_step(
        const DistributedRemoteStepCommand& command) override;

    void receive_boundary_transfer(
        const DistributedBoundaryTransfer& transfer) override;

    [[nodiscard]] bool has_boundary(
        std::uint64_t boundary_id) const noexcept;

    [[nodiscard]] DistributedBoundaryTransfer
    take_boundary(
        std::uint64_t boundary_id);

    [[nodiscard]] bool failed() const noexcept {
        return failed_;
    }

    [[nodiscard]] std::uint64_t next_command_sequence() const noexcept {
        return next_command_sequence_;
    }

private:
    [[nodiscard]] const DistributedStepPlacement*
    find_step(
        std::uint32_t step_id) const noexcept;

    [[nodiscard]] const DistributedExecutionBoundary*
    find_boundary(
        std::uint64_t boundary_id) const noexcept;

    void require_healthy() const;

    const DistributedPlacementPlan* placement_ = nullptr;
    DistributedRuntimeRequest request_{};
    std::uint32_t receiver_node_ = 0;
    DistributedNativeLocalExecutor* executor_ = nullptr;
    std::uint32_t protocol_version_ = 1;
    std::uint64_t next_command_sequence_ = 1;
    std::uint64_t next_boundary_sequence_ = 1;
    bool failed_ = false;

    std::unordered_map<
        std::uint64_t,
        DistributedBoundaryTransfer>
        pending_boundaries_;
};

}  // namespace strix
