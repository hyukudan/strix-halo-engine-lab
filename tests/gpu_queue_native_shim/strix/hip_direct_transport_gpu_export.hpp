#pragma once
// Minimal exporter observations used by native_memory.hpp behavioral tests.
// This is NOT a real DMA-BUF exporter or a native build of the engine headers.
#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <memory>
namespace strix::transport {
enum class TransportMemoryDomain {gpu_device};
enum class TransportBufferAccess {read_write};
struct TransportBufferView {std::uintptr_t address=0;std::uint64_t bytes=0,allocation_generation=0;TransportMemoryDomain domain{};TransportBufferAccess access{};};
struct DirectGpuExportDevice {int index=0;std::array<std::uint8_t,16> uuid{};int runtime_version=1,driver_version=1;bool operator==(const DirectGpuExportDevice&)const=default;};
struct DirectGpuExportAllocation {std::uintptr_t base=0;std::uint64_t bytes=0;int device=0;bool device_memory=false,managed=false;bool operator==(const DirectGpuExportAllocation&)const=default;};
struct DirectGpuExportBinding {DirectGpuExportDevice device;std::array<DirectGpuExportAllocation,2> allocations;};
struct AllocationGenerationSource {std::uint64_t value=0;std::uint64_t next(){return ++value;}};
struct DirectGpuExportImportLease {DirectGpuExportBinding value;const DirectGpuExportBinding& binding()const{return value;}void require_current(){}};
struct HipDirectGpuExportIo {
    DirectGpuExportDevice inspect_device(){DirectGpuExportDevice d;d.index=fake_hip::device;return d;}
    DirectGpuExportAllocation inspect_allocation(const TransportBufferView& view){
        auto it=fake_hip::lookup(view.address);if(it==fake_hip::allocations.end())throw std::runtime_error("unknown simulated allocation");
        return {it->first,it->second.size,it->second.device,!it->second.host,it->second.managed};
    }
};
}
