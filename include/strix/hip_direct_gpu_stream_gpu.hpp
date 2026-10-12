#pragma once

#include "strix/direct_gpu_stream_gpu.hpp"
#include "strix/hip_direct_transport_gpu_export.hpp"

namespace strix::transport {
struct HipDirectGpuStreamGpuPolicy {
    bool experimental_opt_in = false;
    std::uint64_t maximum_completion_polls = 4096;
    std::uint64_t maximum_execution_retained_bytes = 0;
};
struct HipDirectGpuStreamExecutionOwner {
    // Trusted complete owner for every input read and destination written by
    // work on this stream. Native implementation accepts resident GPU buffers;
    // caller must not release/rebind them while this owner is retained.
    std::shared_ptr<void> lifetime;
    std::vector<TransportBufferView> buffers;
    // Explicit independently owned payload-acquire destination. Must be one
    // complete registered buffer; model consumption uses acquired_receive_view.
    TransportBufferView receive_destination;
    std::function<std::vector<TransportBufferView>()> current_buffers;
};
struct HipDirectGpuStreamGpuCalls {
    HipDirectGpuExportCalls observations;
    std::function<hipError_t(hipStream_t*, unsigned)> create_stream;
    std::function<hipError_t(hipStream_t)> query_stream, destroy_stream;
    std::function<hipError_t(hipStream_t, hipDevice_t*)> stream_device;
    std::function<hipError_t(hipStream_t, unsigned long long*)> stream_id;
    std::function<hipError_t(hipStream_t, hipStreamCaptureStatus*)> stream_capture;
    std::function<hipError_t(void**, std::size_t, unsigned)> allocate_host;
    std::function<hipError_t(void*)> free_host;
    std::function<hipError_t(hipEvent_t*, unsigned)> create_event;
    std::function<hipError_t(hipEvent_t, hipStream_t)> record_event;
    std::function<hipError_t(hipEvent_t)> query_event, destroy_event;
    std::function<hipError_t(void*, const void*, std::size_t, hipMemcpyKind, hipStream_t)> copy;
    [[nodiscard]] static HipDirectGpuStreamGpuCalls native();
};

// Fixed explicitly owned stream, two pinned 32-byte terminal metadata slots,
// C1 per direction and at most six events. All staging/consumption must enqueue
// on stream(), with the complete execution owner retained here. Default HIP
// system-fence events are used; their completion alone does not qualify the
// actual external-DMA/cache path. Unknown completion/cleanup retains resources.
class HipDirectGpuStreamGpu final : public DirectGpuStreamGpu {
public:
    HipDirectGpuStreamGpu(std::shared_ptr<DirectGpuExportImportLease>, HipDirectGpuStreamExecutionOwner,
        HipDirectGpuStreamGpuPolicy, HipDirectGpuStreamGpuCalls = HipDirectGpuStreamGpuCalls::native());
    ~HipDirectGpuStreamGpu() override;
    [[nodiscard]] hipStream_t stream() const noexcept;
    [[nodiscard]] const DirectGpuExportBinding& exported_binding() const noexcept override;
    void require_current() override;
    [[nodiscard]] std::shared_ptr<DirectGpuExportImportLease> import_lease() const noexcept override;
    void bind_import(const DirectDriverImportBinding&) override;
    void stage_outbound_segments(const TransferDescriptor&, std::span<const TransportBufferView>, const TransportBufferView&) override;
    void prepare(const DirectDriverOperation&) override;
    [[nodiscard]] DirectDriverVisibilityProof publish_terminal(const DirectDriverOperation&, const TransportBufferView&, const GpuCommitRecord&) override;
    [[nodiscard]] DirectDriverCommitRead read_terminal(const DirectDriverOperation&, const TransportBufferView&) override;
    [[nodiscard]] TransportBufferView acquired_receive_view(const DirectDriverOperation&) override;
    void consume_received_segments(const DirectDriverOperation&, std::span<const TransportBufferView>) override;
    [[nodiscard]] DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation&) override;
    void retire(const DirectDriverOperation&) override;
    void request_stop() noexcept override;
    [[nodiscard]] DirectDriverQuiescenceStatus poll_quiescence() noexcept override;
    [[nodiscard]] DirectPoolReleaseStatus close() noexcept override;
    [[nodiscard]] bool failed() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::transport
