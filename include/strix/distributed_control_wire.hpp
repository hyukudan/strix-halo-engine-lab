#pragma once

#include "strix/distributed_remote_control.hpp"
#include "strix/distributed_token_transaction.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix {

enum class DistributedControlFrameKind : std::uint16_t {
    remote_step_command = 1,
    remote_step_ack = 2,
    boundary_transfer = 3,
    direct_gpu_stream = 4,
    token_transaction = 5,
};

// Explicit dedicated-channel extension: v2 outer framing is unchanged; this
// payload has its own v1 contract. Legacy peers reject unsupported kind4.
// Controller completion, GPU release and consumer credit remain distinct.
enum class DistributedDirectGpuControlKind : std::uint32_t {
    producer_released = 1, receive_ready = 2, received = 3, consumed = 4,
};
struct DistributedDirectGpuControlFrame {
    std::uint32_t contract_version = 1;
    DistributedDirectGpuControlKind kind = DistributedDirectGpuControlKind::producer_released;
    std::string physical_link_id, local_endpoint_identity, peer_endpoint_identity;
    transport::TransferDescriptor descriptor;
    transport::GpuCommitRecord commit;
};

struct DistributedControlWireLimits {
    // Wire v2 adds explicit physical-link slice identities. Descriptor and
    // command protocol versions remain independent. Wire v1 is unsupported;
    // peers must reconnect/reconcile using v2 rather than silently downgrade.
    std::uint32_t protocol_version = 2;
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

[[nodiscard]] DistributedControlFrameKind
inspect_distributed_control_frame_kind(
    std::span<const std::byte> frame,
    DistributedControlWireLimits limits = {});

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

[[nodiscard]] std::vector<std::byte> encode_distributed_direct_gpu_control(
    const DistributedDirectGpuControlFrame&, DistributedControlWireLimits limits = {});
[[nodiscard]] DistributedDirectGpuControlFrame decode_distributed_direct_gpu_control(
    std::span<const std::byte>, DistributedControlWireLimits limits = {});

// Dedicated transaction channel or explicit caller demultiplexing. Existing
// remote-step server handlers do not silently consume or acknowledge kind5.
[[nodiscard]] std::vector<std::byte> encode_distributed_token_message(
    const DistributedTokenMessage&,DistributedControlWireLimits limits = {});
[[nodiscard]] DistributedTokenMessage decode_distributed_token_message(
    std::span<const std::byte>,DistributedControlWireLimits limits = {});
class FramedDistributedTokenWire final {
public:
    explicit FramedDistributedTokenWire(DistributedControlByteStream& stream,DistributedControlWireLimits limits = {})
        :stream_(&stream),limits_(limits) {}
    void send(const DistributedTokenMessage& message);
    [[nodiscard]] DistributedTokenMessage receive();
private:
    DistributedControlByteStream* stream_;
    DistributedControlWireLimits limits_;
};

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
