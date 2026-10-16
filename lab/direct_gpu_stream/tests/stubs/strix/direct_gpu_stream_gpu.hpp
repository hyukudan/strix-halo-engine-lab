#pragma once
// Type/API double matching the inspected engine interface, not its native path.
#include <cstdint>
#include <memory>
#include <span>
#include <array>
#include <string>
#include <vector>
namespace strix::transport {
enum class TransportMemoryDomain { gpu_device, host };
enum class TransportBufferAccess { read_only, write_only, read_write };
struct TransportBufferView { std::uintptr_t address=0;std::uint64_t bytes=0,allocation_generation=0;TransportMemoryDomain domain=TransportMemoryDomain::gpu_device;TransportBufferAccess access=TransportBufferAccess::read_write; };
struct DirectTransportLinkPoolBinding { TransportBufferView tx,rx; };
struct DirectGpuExportDevice { int index=0;std::array<std::uint8_t,16> uuid{};int runtime_version=0,driver_version=0;bool operator==(const DirectGpuExportDevice&) const=default; };
struct DirectGpuExportBinding { DirectTransportLinkPoolBinding pool; };
struct DirectGpuExportImportLease {};
struct DirectDriverImportBinding {};
struct TransferDescriptor { std::uint32_t protocol_version=0;std::uint64_t connection_epoch=0,model_plan_id=0,request_id=0,generation=0,boundary_id=0,sequence=0,shape_fingerprint=0;std::uint32_t dtype_id=0;std::uint64_t byte_length=0; };
struct DirectDriverOperation { TransferDescriptor descriptor;TransportBufferView buffer;bool receive_path=true; };
struct DirectDriverVisibilityProof {};
struct DirectDriverCommitRead {};
struct GpuCommitRecord {};
enum class DirectDriverQuiescenceStatus { quiescent, pending, failed_retained };
enum class DirectPoolReleaseStatus { released, pending, failed_retained };
class DirectGpuStreamGpu {
public:
    virtual ~DirectGpuStreamGpu()=default;
    virtual const DirectGpuExportBinding& exported_binding() const noexcept=0;
    virtual void require_current()=0;
    virtual std::shared_ptr<DirectGpuExportImportLease> import_lease() const noexcept=0;
    virtual void bind_import(const DirectDriverImportBinding&)=0;
    virtual void stage_outbound_segments(const TransferDescriptor&,std::span<const TransportBufferView>,const TransportBufferView&)=0;
    virtual void prepare(const DirectDriverOperation&)=0;
    virtual DirectDriverVisibilityProof publish_terminal(const DirectDriverOperation&,const TransportBufferView&,const GpuCommitRecord&)=0;
    virtual DirectDriverCommitRead read_terminal(const DirectDriverOperation&,const TransportBufferView&)=0;
    virtual TransportBufferView acquired_receive_view(const DirectDriverOperation&)=0;
    virtual void consume_received_segments(const DirectDriverOperation&,std::span<const TransportBufferView>)=0;
    virtual DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation&)=0;
    virtual void retire(const DirectDriverOperation&)=0;
    virtual void request_stop() noexcept=0;
    virtual DirectDriverQuiescenceStatus poll_quiescence() noexcept=0;
    virtual DirectPoolReleaseStatus close() noexcept=0;
};
}
