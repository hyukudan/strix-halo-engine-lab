#pragma once
#include "direct_gpu_stream_commissioning.hpp"
namespace strix { struct AllocationGenerationSource { std::uint64_t next(); }; }
namespace strix::transport {
struct HipDirectGpuPoolPolicy { bool experimental_opt_in;std::uint64_t maximum_bytes,alignment; };
class HipDirectGpuPoolAllocator {public:HipDirectGpuPoolAllocator(DirectGpuExportDevice,HipDirectGpuPoolPolicy);DirectPoolReleaseStatus drain();};
class HipDirectGpuExportIo {public: DirectGpuExportDevice inspect_device();};
struct DirectFrameGeometry { std::uint64_t frame_bytes,ring_frames;bool frame_mode; };
struct MultiLinkTransportPoolPlan {};
MultiLinkTransportPoolPlan plan_multi_link_transport_pools(std::span<const DirectFrameGeometry>);
class DirectGpuExportOwner {public:static std::shared_ptr<DirectGpuExportOwner> allocate_and_export(std::shared_ptr<HipDirectGpuExportIo>,std::shared_ptr<HipDirectGpuPoolAllocator>,MultiLinkTransportPoolPlan,strix::AllocationGenerationSource&,DirectGpuExportDevice);std::shared_ptr<DirectGpuExportImportLease> retain_for_import();DirectPoolReleaseStatus release();};
}
