#pragma once

#include "strix/distributed_remote_control.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix {

enum class DistributedControlFrameKind : std::uint16_t {
    remote_step_command = 1,
    remote_step_ack = 2,
    boundary_transfer = 3,
};

struct DistributedControlWireLimits {
    std::uint32_t protocol_version = 1;
    std::uint64_t maximum_frame_bytes = 64ULL * 1024ULL;
    std::uint32_t maximum_slices = 64;
};

class DistributedControlByteStream {
public:
    virtual ~DistributedControlByteStream() = default;
    virtual void send(
        std::span<const std::byte> frame) = 0;
    [[nodiscard]] virtual std::vector<std::byte>
    receive() = 0;
};

[[nodiscard]] std::vector<std::byte>
encode_distributed_remote_step_command(
    const DistributedRemoteStepCommand& command,
    DistributedControlWireLimits limits = {});

[[nodiscard]] DistributedRemoteStepCommand
decode_distributed_remote_step_command(
    std::span<const std::byte> frame,
    DistributedControlWireLimits limits = {});

[[nodiscard]] std::vector<std::byte>
encode_distributed_remote_step_ack(
    const DistributedRemoteStepAck& ack,
    DistributedControlWireLimits limits = {});

[[nodiscard]] DistributedRemoteStepAck
decode_distributed_remote_step_ack(
    std::span<const std::byte> frame,
    DistributedControlWireLimits limits = {});

[[nodiscard]] std::vector<std::byte>
encode_distributed_boundary_transfer(
    const DistributedBoundaryTransfer& transfer,
    DistributedControlWireLimits limits = {});

[[nodiscard]] DistributedBoundaryTransfer
decode_distributed_boundary_transfer(
    std::span<const std::byte> frame,
    DistributedControlWireLimits limits = {});

class FramedDistributedControlWire final
    : public DistributedControlWireBackend {
public:
    explicit FramedDistributedControlWire(
        DistributedControlByteStream& stream,
        DistributedControlWireLimits limits = {});

    [[nodiscard]] DistributedRemoteStepAck
    execute_remote_step(
        const DistributedRemoteStepCommand& command) override;

    void publish_boundary_transfer(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) override;

    [[nodiscard]] DistributedBoundaryTransfer
    await_boundary_transfer(
        const DistributedExecutionBoundary& boundary) override;

private:
    DistributedControlByteStream* stream_ = nullptr;
    DistributedControlWireLimits limits_{};
};

}  // namespace strix
