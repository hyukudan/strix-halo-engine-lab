#pragma once
#include "direct_gpu_stream_commissioning.hpp"
#include <hip/hip_runtime.h>
namespace strix::transport {
struct HipDirectGpuStreamGpuPolicy { bool experimental_opt_in=false;std::uint64_t maximum_completion_polls=0,maximum_execution_retained_bytes=0;std::chrono::milliseconds completion_timeout{10000},completion_pause{1}; };
struct HipDirectGpuStreamExecutionOwner { std::shared_ptr<void> lifetime;std::vector<TransportBufferView> buffers;TransportBufferView receive_destination;std::function<std::vector<TransportBufferView>()> current_buffers; };
struct HipDirectGpuStreamGpuCalls { std::function<hipError_t(hipEvent_t*,unsigned)> create_event;static HipDirectGpuStreamGpuCalls native(); };
class HipDirectGpuStreamGpu final : public DirectGpuStreamGpu {
public:
 HipDirectGpuStreamGpu(std::shared_ptr<DirectGpuExportImportLease>,HipDirectGpuStreamExecutionOwner,HipDirectGpuStreamGpuPolicy,HipDirectGpuStreamGpuCalls);
 ~HipDirectGpuStreamGpu() override;
 hipStream_t stream() const noexcept;
 const DirectGpuExportBinding& exported_binding() const noexcept override;
 void require_current() override;
 std::shared_ptr<DirectGpuExportImportLease> import_lease() const noexcept override;
 void bind_import(const DirectDriverImportBinding&) override;
 void stage_outbound_segments(const TransferDescriptor&,std::span<const TransportBufferView>,const TransportBufferView&) override;
 void prepare(const DirectDriverOperation&) override;
 DirectDriverVisibilityProof publish_terminal(const DirectDriverOperation&,const TransportBufferView&,const GpuCommitRecord&) override;
 DirectDriverCommitRead read_terminal(const DirectDriverOperation&,const TransportBufferView&) override;
 TransportBufferView acquired_receive_view(const DirectDriverOperation&) override;
 void consume_received_segments(const DirectDriverOperation&,std::span<const TransportBufferView>) override;
 DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation&) override;
 void retire(const DirectDriverOperation&) override;
 void request_stop() noexcept override;
 DirectDriverQuiescenceStatus poll_quiescence() noexcept override;
 DirectPoolReleaseStatus close() noexcept override;
};
}
