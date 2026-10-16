#pragma once

#include "strix/direct_gpu_stream_gpu.hpp"
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

namespace strix::transport {
// Commissioning adapter, not a qualification provider. The existing Linux
// provider invokes read_terminal only after validating all controller frames.
// The hook runs BEFORE the delegate's acquire copy and therefore observes the
// original DMA destination, rather than a potentially repaired/copied payload.
class ObservedCommissioningGpu final : public DirectGpuStreamGpu {
public:
    using Observer = std::function<void(const DirectDriverOperation&)>;
    ObservedCommissioningGpu(std::shared_ptr<DirectGpuStreamGpu> gpu, Observer observer)
        : gpu_(std::move(gpu)), observer_(std::move(observer)) {
        if (!gpu_ || !observer_) throw std::invalid_argument("missing GPU observation dependency");
    }
    const DirectGpuExportBinding& exported_binding() const noexcept override { return gpu_->exported_binding(); }
    void require_current() override { gpu_->require_current(); }
    std::shared_ptr<DirectGpuExportImportLease> import_lease() const noexcept override { return gpu_->import_lease(); }
    void bind_import(const DirectDriverImportBinding& b) override { gpu_->bind_import(b); }
    void stage_outbound_segments(const TransferDescriptor& d, std::span<const TransportBufferView> s, const TransportBufferView& t) override { gpu_->stage_outbound_segments(d, s, t); }
    void prepare(const DirectDriverOperation& op) override { gpu_->prepare(op); }
    DirectDriverVisibilityProof publish_terminal(const DirectDriverOperation& op, const TransportBufferView& t, const GpuCommitRecord& r) override { return gpu_->publish_terminal(op, t, r); }
    DirectDriverCommitRead read_terminal(const DirectDriverOperation& op, const TransportBufferView& t) override {
        try {
            if (!op.receive_path) throw std::invalid_argument("raw observation requires an RX operation");
            gpu_->require_current(); observer_(op); gpu_->require_current();
            return gpu_->read_terminal(op, t);
        } catch (...) {
            // The owning provider performs normal stop/drain/retention. Never
            // substitute an ACK, a successful acquire or a canceled GPU event.
            gpu_->request_stop();
            throw;
        }
    }
    TransportBufferView acquired_receive_view(const DirectDriverOperation& op) override { return gpu_->acquired_receive_view(op); }
    void consume_received_segments(const DirectDriverOperation& op, std::span<const TransportBufferView> v) override { gpu_->consume_received_segments(op, v); }
    DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation& op) override { return gpu_->consumer_completion(op); }
    void retire(const DirectDriverOperation& op) override { gpu_->retire(op); }
    void request_stop() noexcept override { gpu_->request_stop(); }
    DirectDriverQuiescenceStatus poll_quiescence() noexcept override { return gpu_->poll_quiescence(); }
    DirectPoolReleaseStatus close() noexcept override { return gpu_->close(); }
private:
    std::shared_ptr<DirectGpuStreamGpu> gpu_;
    Observer observer_;
};
} // namespace strix::transport
