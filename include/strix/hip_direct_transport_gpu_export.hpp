#pragma once

#include "strix/direct_transport_gpu_export.hpp"

#include <hip/hip_runtime_api.h>
#include <functional>

namespace strix::transport {

// Injectable exact local HIP/POSIX call boundary. Native defaults invoke the
// runtime/syscalls; fixtures may exercise assembly and ownership without GPU.
struct HipDirectGpuExportCalls {
    std::function<hipError_t(int*)> get_device, runtime_version, driver_version;
    std::function<hipError_t(hipUUID*, hipDevice_t)> device_uuid;
    std::function<hipError_t(hipPointerAttribute_t*, const void*)> pointer_attributes;
    std::function<hipError_t(hipDeviceptr_t*, std::size_t*, hipDeviceptr_t)> address_range;
    std::function<hipError_t(int*, hipDeviceptr_t, std::size_t)> dma_buf_range_export;
    std::function<int(int)> get_fd_flags, close_fd;
    std::function<int(int, int)> set_fd_flags;
    std::function<DirectGpuExportFdIdentity(int)> fd_identity;
    [[nodiscard]] static HipDirectGpuExportCalls native();
};

// Linux HIP DMA-BUF export prerequisite only. Calls the documented HIP range
// handle API, never a transport ioctl or opaque-FD HIP import. Failures remain
// failures; exported buffers do not qualify the missing driver import/events.
class HipDirectGpuExportIo final : public DirectGpuExportIo {
public:
    explicit HipDirectGpuExportIo(HipDirectGpuExportCalls calls = HipDirectGpuExportCalls::native());
    [[nodiscard]] DirectGpuExportDevice inspect_device() override;
    [[nodiscard]] DirectGpuExportAllocation inspect_allocation(const TransportBufferView&) override;
    void export_dma_buf(const DirectGpuExportAllocation&, int& owned_descriptor) override;
    [[nodiscard]] DirectGpuExportFdIdentity inspect_descriptor(int) override;
    [[nodiscard]] bool close_descriptor(int) noexcept override;
private:
    HipDirectGpuExportCalls calls_;
};

} // namespace strix::transport
