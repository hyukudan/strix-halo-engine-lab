#pragma once

#include "strix/direct_transport_gpu_export.hpp"

namespace strix::transport {
// Controller and GPU proofs are separate. Implementations retain every queued
// producer/consumer and terminal-copy source/destination through actual GPU
// completion, including callback failure. STOP is not a GPU completion proof.
class DirectGpuStreamGpu {
public:
    virtual ~DirectGpuStreamGpu() = default;
    [[nodiscard]] virtual const DirectGpuExportBinding& exported_binding() const noexcept = 0;
    virtual void require_current() = 0;
    [[nodiscard]] virtual std::shared_ptr<DirectGpuExportImportLease> import_lease() const noexcept = 0;
    virtual void bind_import(const DirectDriverImportBinding&) = 0;
    virtual void stage_outbound_segments(const TransferDescriptor&, std::span<const TransportBufferView>, const TransportBufferView& tx) = 0;
    virtual void prepare(const DirectDriverOperation&) = 0;
    [[nodiscard]] virtual DirectDriverVisibilityProof publish_terminal(
        const DirectDriverOperation&, const TransportBufferView& terminal, const GpuCommitRecord&) = 0;
    [[nodiscard]] virtual DirectDriverCommitRead read_terminal(
        const DirectDriverOperation&, const TransportBufferView& terminal) = 0;
    [[nodiscard]] virtual TransportBufferView acquired_receive_view(const DirectDriverOperation&) = 0;
    virtual void consume_received_segments(const DirectDriverOperation&, std::span<const TransportBufferView>) = 0;
    [[nodiscard]] virtual DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation&) = 0;
    virtual void retire(const DirectDriverOperation&) = 0;
    virtual void request_stop() noexcept = 0;
    [[nodiscard]] virtual DirectDriverQuiescenceStatus poll_quiescence() noexcept = 0;
    [[nodiscard]] virtual DirectPoolReleaseStatus close() noexcept = 0;
};
} // namespace strix::transport
