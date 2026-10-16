#pragma once
#include "strix/hip_direct_transport_gpu_export.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace strix::transport::queue_lab {
inline void check(hipError_t e,const char* what){if(e!=hipSuccess)throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));}
struct Terminal {
    std::uint64_t epoch=0,sequence=0,tag=0,bytes=0;
    bool operator==(const Terminal&)const=default;
};
static_assert(sizeof(Terminal)==32);
static_assert(std::endian::native==std::endian::little,"queue terminal profile requires little endian Strix Halo peers");
struct QueueGpuMemory {
    struct Slot {
        std::array<TransportBufferView,3> views{};
        std::array<DirectGpuExportAllocation,3> extents{};
        hipStream_t stream=nullptr;hipEvent_t event=nullptr;unsigned long long stream_id=0;
        Terminal* terminal=nullptr;
    };
    struct Impl {
        std::shared_ptr<DirectGpuExportImportLease> lease;
        DirectGpuExportDevice device;
        std::vector<Slot> slots;
        std::chrono::milliseconds timeout{10000};
        bool closed=false,cleanup_unknown=false,allocation_attempted=false;
    };
    std::unique_ptr<Impl> state=std::make_unique<Impl>();
    explicit QueueGpuMemory(std::shared_ptr<DirectGpuExportImportLease> lease){
        if(!lease)throw std::invalid_argument("queue GPU requires a retained export/import lease");
        state->device=lease->binding().device;state->lease=std::move(lease);
    }
    void current(){
        auto& i=*state;if(i.closed)throw std::logic_error("queue GPU owner is closed");i.lease->require_current();
        HipDirectGpuExportIo inspect;if(inspect.inspect_device()!=i.device)throw std::runtime_error("queue GPU UUID/runtime/device changed");
        for(const auto& slot:i.slots){
            for(std::size_t n=0;n<slot.views.size();++n)if(slot.views[n].address&&slot.extents[n].base&&inspect.inspect_allocation(slot.views[n])!=slot.extents[n])
                throw std::runtime_error("queue GPU execution allocation changed");
            if(slot.stream){hipDevice_t d=-1;unsigned long long id=0;hipStreamCaptureStatus capture{};
                check(hipStreamGetDevice(slot.stream,&d),"queue stream device");check(hipStreamGetId(slot.stream,&id),"queue stream ID");
                check(hipStreamIsCapturing(slot.stream,&capture),"queue stream capture");
                if(d!=i.device.index||!id||(slot.stream_id&&id!=slot.stream_id)||capture!=hipStreamCaptureStatusNone)
                    throw std::runtime_error("queue GPU stream identity/capture changed");
            }
        }
    }
    void allocate(unsigned depth,std::uint64_t bytes,AllocationGenerationSource& generations,std::chrono::milliseconds timeout){
        auto& i=*state;
        if(i.allocation_attempted||i.closed||depth<2||depth>8||!bytes||bytes>16ULL*1024*1024||
            bytes>64ULL*1024*1024/(3U*depth)||timeout.count()<=0||timeout>std::chrono::minutes{1})
            throw std::invalid_argument("queue GPU allocation requires one bounded setup");
        i.allocation_attempted=true;i.timeout=timeout;i.slots.resize(depth);std::uint64_t retained=0;current();
        for(auto& slot:i.slots){
            for(std::size_t n=0;n<slot.views.size();++n){
                void* pointer=nullptr;check(hipMalloc(&pointer,static_cast<std::size_t>(bytes)),"queue execution allocate");
                slot.views[n]={reinterpret_cast<std::uintptr_t>(pointer),bytes,generations.next(),TransportMemoryDomain::gpu_device,TransportBufferAccess::read_write};
                if(!pointer)throw std::runtime_error("queue execution allocation is missing");
                HipDirectGpuExportIo inspect;slot.extents[n]=inspect.inspect_allocation(slot.views[n]);
                const auto& extent=slot.extents[n];
                if(!extent.device_memory||extent.managed||extent.device!=i.device.index||!extent.base||!extent.bytes||extent.bytes>64ULL*1024*1024-retained)
                    throw std::runtime_error("queue execution actual allocation exceeds admission");
                if(extent.bytes>std::numeric_limits<std::uintptr_t>::max()-extent.base||slot.views[n].address<extent.base||
                    slot.views[n].address-extent.base>extent.bytes||bytes>extent.bytes-(slot.views[n].address-extent.base))
                    throw std::runtime_error("queue execution view exceeds the actual GPU allocation");
                for(const auto& imported:i.lease->binding().allocations)
                    if(imported.bytes>std::numeric_limits<std::uintptr_t>::max()-imported.base||
                        (extent.base<imported.base+imported.bytes&&imported.base<extent.base+extent.bytes))
                        throw std::runtime_error("queue execution allocation aliases an imported pool");
                for(const auto& other_slot:i.slots)for(const auto& other:other_slot.extents)
                    if(&other!=&extent&&other.base&&extent.base<other.base+other.bytes&&other.base<extent.base+extent.bytes)
                        throw std::runtime_error("queue execution allocation aliases another slot");
                retained+=extent.bytes;current();
            }
            check(hipStreamCreateWithFlags(&slot.stream,hipStreamNonBlocking),"queue owned stream");
            check(hipStreamGetId(slot.stream,&slot.stream_id),"queue owned stream identity");
            check(hipEventCreateWithFlags(&slot.event,hipEventDisableTiming | hipEventReleaseToSystem),"queue system-fence event");
            check(hipHostMalloc(reinterpret_cast<void**>(&slot.terminal),sizeof(Terminal),hipHostMallocDefault),"queue pinned terminal");
            if(!slot.stream||!slot.stream_id||!slot.event||!slot.terminal)throw std::runtime_error("queue GPU stream/event/terminal allocation is missing");
            current();
        }
    }
    Slot& slot(unsigned index){return state->slots.at(index);}
    void wait(unsigned index){
        auto& s=slot(index);current();const auto start=std::chrono::steady_clock::now();
        check(hipEventRecord(s.event,s.stream),"queue system-fence event record");
        const auto duration=std::chrono::duration_cast<std::chrono::steady_clock::duration>(state->timeout);
        if(start>std::chrono::steady_clock::time_point::max()-duration)throw std::overflow_error("queue GPU deadline overflow");
        const auto deadline=start+duration;auto last=start;
        for(unsigned n=0;n<1000000;++n){
            const auto now=std::chrono::steady_clock::now();if(now<last||now>=deadline)throw std::runtime_error("queue GPU completion deadline");last=now;
            current();const auto result=hipEventQuery(s.event);current();
            const auto after=std::chrono::steady_clock::now();if(after<last||after>=deadline)throw std::runtime_error("queue GPU late completion");last=after;
            if(result==hipSuccess)return;
            if(result!=hipErrorNotReady)check(result,"queue GPU event");
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        throw std::runtime_error("queue GPU completion poll budget");
    }
    bool quiescent() noexcept {
        if(state->closed)return true;
        try{current();for(const auto& s:state->slots)if(s.stream){const auto result=hipStreamQuery(s.stream);if(result==hipErrorNotReady)return false;check(result,"queue GPU stream quiescence");}current();return true;}
        catch(...){return false;}
    }
    bool close() noexcept {
        auto& i=*state;if(i.closed)return true;if(i.cleanup_unknown||!quiescent())return false;
        try{
            for(auto& s:i.slots){
                if(s.event){i.cleanup_unknown=true;check(hipEventDestroy(s.event),"queue event destroy");s.event=nullptr;i.cleanup_unknown=false;current();}
                if(s.stream){i.cleanup_unknown=true;check(hipStreamDestroy(s.stream),"queue stream destroy");s.stream=nullptr;i.cleanup_unknown=false;current();}
                if(s.terminal){i.cleanup_unknown=true;check(hipHostFree(s.terminal),"queue terminal free");s.terminal=nullptr;i.cleanup_unknown=false;current();}
                for(auto& view:s.views)if(view.address){i.cleanup_unknown=true;check(hipFree(reinterpret_cast<void*>(view.address)),"queue execution free");view={};i.cleanup_unknown=false;current();}
            }
            i.lease.reset();i.closed=true;return true;
        }catch(...){return false;}
    }
    ~QueueGpuMemory(){if(state&&!close())static_cast<void>(state.release());}
};
}
