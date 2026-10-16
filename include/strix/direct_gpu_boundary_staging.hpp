#pragma once

#include "strix/direct_gpu_stream_gpu.hpp"
#include "strix/distributed_native_backend.hpp"
#include "strix/execution_memory.hpp"

namespace strix::transport {
// Trusted descriptor-keyed receive seam implemented by the concrete driver.
// These methods resolve an existing prepared operation, never create GPU proof
// from a CPU consumer callback. The provider retains native import/GPU owners.
class DirectGpuBoundaryReceiveIo {
public:
    virtual ~DirectGpuBoundaryReceiveIo() = default;
    virtual void require_current() = 0;
    [[nodiscard]] virtual TransportBufferView acquired_receive_view(const TransferDescriptor&) = 0;
    virtual void consume_received_segments(const TransferDescriptor&, std::span<const TransportBufferView>) = 0;
    [[nodiscard]] virtual DirectDriverVisibilityProof complete_receive_consumer(const TransferDescriptor&) = 0;
};
} // namespace strix::transport

namespace strix {
// Retains the actual model arena allocations and excludes replacement/writes
// outside the serialized executor. GPU helper ownership must independently
// cover all these buffers. Address metadata does not qualify a model or link.
class DirectGpuBoundaryMemoryOwner {
public:
    virtual ~DirectGpuBoundaryMemoryOwner() = default;
    [[nodiscard]] virtual std::vector<ExecutionBufferAddress> current_buffers() const = 0;
};

// Concrete C1 bridge to NativeDistributedBackend. Packs/unpacks ordered real
// execution buffers via GPU copies; supports one physical slice and one exact
// request. No CPU payload path, striping, automatic recovery or qualification.
class DirectGpuBoundaryStaging final : public DistributedBoundaryStagingProvider {
public:
    DirectGpuBoundaryStaging(DistributedPlacementPlan, DistributedRuntimeRequest,
        transport::DirectTransportLease, std::shared_ptr<DirectGpuBoundaryMemoryOwner>,
        std::shared_ptr<transport::DirectGpuStreamGpu>,
        std::shared_ptr<transport::DirectGpuBoundaryReceiveIo>);
    // Explicit two-peer endpoint with the SAME canonical wire-plan identity.
    // The legacy overload continues using placement.local_node unchanged.
    // A nonzero selector is admitted ONLY with the explicit position policy:
    // one selected data edge plus at most64 canonical logical position edges.
    // The complete canonical placement and its identity are retained unchanged.
    DirectGpuBoundaryStaging(DistributedPlacementPlan, DistributedRuntimeRequest,
        transport::DirectTransportLease, std::shared_ptr<DirectGpuBoundaryMemoryOwner>,
        std::shared_ptr<transport::DirectGpuStreamGpu>,
        std::shared_ptr<transport::DirectGpuBoundaryReceiveIo>, std::uint32_t endpoint_node,
        std::uint64_t payload_boundary_id = 0,
        NativeDistributedPositionDependencyPolicy = NativeDistributedPositionDependencyPolicy::reject);
    ~DirectGpuBoundaryStaging() override;
    [[nodiscard]] bool requires_full_send_retirement() const noexcept override { return true; }
    [[nodiscard]] transport::TransportBufferView stage_outbound(
        const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) override;
    [[nodiscard]] transport::TransportBufferView stage_inbound(
        const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) override;
    void consume_inbound(const DistributedExecutionBoundary&, const transport::TransportBufferView&) override;
    [[nodiscard]] std::unique_ptr<DistributedInboundCompletion> finish_inbound(
        const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) override;
    [[nodiscard]] bool failed() const noexcept;
private:
    DirectGpuBoundaryStaging(DistributedPlacementPlan, DistributedRuntimeRequest,
        transport::DirectTransportLease, std::shared_ptr<DirectGpuBoundaryMemoryOwner>,
        std::shared_ptr<transport::DirectGpuStreamGpu>,
        std::shared_ptr<transport::DirectGpuBoundaryReceiveIo>, std::uint32_t endpoint_node, bool explicit_endpoint,
        std::uint64_t payload_boundary_id, NativeDistributedPositionDependencyPolicy);
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
} // namespace strix
