// Experimental GPU-to-USB4-to-GPU commissioning, never a production backend.
#include "run_options.hpp"
#include "../direct_gpu_stream/bounded_tcp_control.hpp"
#include "strix/direct_gpu_stream_commissioning.hpp"
#include "strix/hip_direct_gpu_pool_allocator.hpp"
#include "strix/hip_direct_gpu_stream_gpu.hpp"
#include "strix/hip_gpu_visibility_probe.hpp"
#include "strix/model_content_hash.hpp"
#include <array>
#include <iostream>
#include <memory>
#include <thread>

namespace {
using namespace strix;
using namespace strix::transport;
using namespace strix::transport::visibility_lab;
void check(hipError_t e,const char* message){if(e!=hipSuccess)throw std::runtime_error(std::string(message)+": "+hipGetErrorString(e));}
std::string uuid(const std::array<std::uint8_t,16>& value){constexpr char hex[]="0123456789abcdef";std::string out;for(auto b:value){out+=hex[b>>4];out+=hex[b&15];}return out;}
struct Execution {
    DirectGpuExportDevice identity;
    std::array<void*,3> allocations{};
    std::vector<TransportBufferView> views;
    bool quiescent=true,release_unknown=false;
    void current(){HipDirectGpuExportIo probe;if(probe.inspect_device()!=identity)throw std::runtime_error("execution GPU identity changed");}
    void allocate(std::uint64_t bytes,AllocationGenerationSource& generations){
        current();
        for(auto& p:allocations){check(hipMalloc(&p,static_cast<std::size_t>(bytes)),"visibility execution allocate");
            if(!p)throw std::runtime_error("missing allocation");
            views.push_back({reinterpret_cast<std::uintptr_t>(p),bytes,generations.next(),TransportMemoryDomain::gpu_device,TransportBufferAccess::read_write});current();}
    }
    bool release() noexcept{
        if(!quiescent||release_unknown)return false;
        try{current();for(auto& p:allocations)if(p){release_unknown=true;check(hipFree(p),"execution free");p=nullptr;release_unknown=false;current();}return true;}catch(...){return false;}
    }
    ~Execution(){static_cast<void>(release());}
};
struct Observation {
    bool producer_completed=false,raw_verified=false,copied_verified=false,negative_detected=false;
    bool raw_previously_read=false;
    GpuVisibilityResult raw{},copied{};
};
std::uint64_t shape(const RunOptions& o,std::uint64_t seq){
    const auto text=o.campaign+"\n"+o.run+"\ngpu-byte-check-v2\n"+std::to_string(o.epoch)+"\n"+
        std::to_string(seq)+"\n"+std::to_string(o.seed)+"\n"+std::to_string(o.bytes);
    const auto hash=sha256_bytes({reinterpret_cast<const std::byte*>(text.data()),text.size()});
    std::uint64_t value=0;for(unsigned i=0;i<8;++i)value|=static_cast<std::uint64_t>(hash.bytes[i])<<(8*i);return value?value:1;
}
void result_json(const GpuVisibilityResult& r){
    std::cout<<"{\"epoch\":"<<r.epoch<<",\"sequence\":"<<r.sequence<<",\"nonce\":"<<r.nonce
        <<",\"checked_bytes\":"<<r.checked_bytes<<",\"mismatches\":"<<r.mismatches
        <<",\"first_mismatch\":"<<r.first_mismatch<<",\"observed_digest\":"<<r.observed_digest
        <<",\"consumed_digest\":"<<r.consumed_digest<<"}";
}
int run(const RunOptions& o){
    auto driver=LinuxDirectGpuStreamCalls::native();const auto build=driver.driver_build_identity();
    if(!build||*build!=o.build||driver.kernel_release()!=o.kernel)throw std::runtime_error("expected driver/kernel missing before opening DMA");
    check(hipSetDevice(o.gpu_device),"GPU selection");
    auto exporter=std::make_shared<HipDirectGpuExportIo>();const auto identity=exporter->inspect_device();
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,identity.index),"GPU architecture");
    std::string architecture=properties.gcnArchName;if(auto pos=architecture.find(':');pos!=std::string::npos)architecture.resize(pos);
    if(uuid(identity.uuid)!=o.gpu_uuid||identity.runtime_version!=o.runtime_version||identity.driver_version!=o.driver_version||architecture!="gfx1151")
        throw std::runtime_error("expected Strix Halo GPU/runtime/driver differs");
    AllocationGenerationSource generations;
    auto allocator=std::make_shared<HipDirectGpuPoolAllocator>(identity,HipDirectGpuPoolPolicy{true,128ULL*1024*1024,4096});
    const auto geometry=plan_multi_link_transport_pools(std::array{DirectFrameGeometry{4096,o.ring_frames,true}});
    auto exported=DirectGpuExportOwner::allocate_and_export(exporter,allocator,geometry,generations,identity);
    auto lease=exported->retain_for_import();
    auto execution=std::make_shared<Execution>();execution->identity=identity;execution->allocate(o.bytes,generations);
    HipDirectGpuStreamExecutionOwner owner{execution,execution->views,execution->views[1],[execution]{execution->current();return execution->views;}};
    auto gpu=std::make_shared<HipDirectGpuStreamGpu>(lease,owner,
        HipDirectGpuStreamGpuPolicy{true,100000,64ULL*1024*1024,o.timeout,std::chrono::milliseconds{1}});
    // Owns only the result storage. Execution/stream leases stay alive through
    // uncertain GPU completion; query failure never authorizes freeing them.
    auto probe=std::make_shared<HipGpuVisibilityProbe>(execution,[gpu]{return gpu->stream();},
        [execution]{execution->current();},HipGpuVisibilityProbePolicy{o.bytes,o.timeout});
    auto control=std::make_shared<BoundedCommissioningTcpControl>(CommissioningTcpPolicy{o.listener,o.port,2048,100000,o.timeout});
    DirectDriverBinding discovery;discovery.interface.physical_link_id=o.link;discovery.interface.connection_epoch=o.epoch;
    auto commissioning=std::make_unique<LinuxDirectGpuStreamCommissioning>(inspect_direct_stream_device(o.device),discovery,gpu,control,
        LinuxDirectGpuStreamPolicy{true,o.bytes,128ULL*1024*1024,4096});
    const auto observed=commissioning->observe();const auto& binding=observed.binding;
    if(observed.native.query_status!=DirectNativeQueryStatus::unavailable||!observed.backend.maximum_payload_bytes||
        binding.interface.local_endpoint_identity!=o.local||binding.interface.peer_endpoint_identity!=o.peer||
        binding.interface.kernel_release!=o.kernel||binding.interface.driver_identity!="thunderbolt_stream:gnu-build-id:"+o.build||
        binding.interface.abi_identity!="sgu-1.0-5354475055303031")throw std::runtime_error("exact commissioning endpoint changed");
    commissioning->bind_exact_observed(binding);
    std::vector<DirectGpuCommissioningConfig> configs;std::vector<DirectGpuCommissioningPayload> payloads;
    auto observations=std::make_shared<std::vector<Observation>>(o.iterations);
    auto raw_read_count=std::make_shared<unsigned>(0);
    for(unsigned index=0;index<o.iterations;++index){
        const bool rx=o.duplex&&index%2?!o.receiver:o.receiver;
        const std::uint64_t seq=static_cast<std::uint64_t>(index)+1;
        TransferDescriptor d{1,o.epoch,1,o.request,1,1,seq,shape(o,seq),1,o.bytes};
        configs.push_back({o.campaign,o.run+".s"+std::to_string(index),binding,d,o.seed,rx,o.timeout,100000});
        auto source=execution->views[0],destination=execution->views[2];
        const GpuVisibilityPattern pattern{o.epoch,seq,o.seed};
        payloads.push_back({{source},{destination},
            [execution,probe,source,pattern,o,observations,index]{
                execution->quiescent=false;
                std::this_thread::sleep_for(o.producer_delay);
                probe->produce(std::array{GpuVisibilityRange{source.address,source.bytes}},pattern);
                (*observations)[index].producer_completed=true;
            },
            [execution,gpu,probe,destination,pattern,o,observations,index,raw_read_count]{
                execution->quiescent=false;auto& out=(*observations)[index];
                // Called by the existing session only after its RX callbacks,
                // acquire/unpack/consumer event, and BEFORE credit/repost.
                gpu->require_current();std::this_thread::sleep_for(o.consumer_delay);
                const auto raw=gpu->exported_binding().pool.rx;
                if(raw.bytes<4096||o.bytes>raw.bytes-4096)throw std::runtime_error("raw receive extent changed");
                out.raw_previously_read=*raw_read_count!=0;
                out.raw=probe->verify(std::array{GpuVisibilityRange{raw.address,o.bytes}},pattern);
                out.raw_verified=valid_gpu_visibility_result(out.raw,pattern,out.raw.nonce,o.bytes);
                ++*raw_read_count;
                out.copied=probe->verify(std::array{GpuVisibilityRange{destination.address,destination.bytes}},pattern);
                out.copied_verified=valid_gpu_visibility_result(out.copied,pattern,out.copied.nonce,o.bytes);
                if(!out.raw_verified||!out.copied_verified)return false;
                // This destination is separately allocated and has no model
                // consumer in commissioning. NEVER mutate an imported RX slot.
                out.negative_detected=probe->negative_control(std::array{GpuVisibilityRange{destination.address,destination.bytes}},pattern,o.bytes-1);
                return out.negative_detected;
            }});
    }
    DirectGpuCommissioningSessionReport report;
    const auto started = std::chrono::steady_clock::now();
    if(o.iterations==1){auto one=commissioning->run_once(configs[0],payloads[0]);report.requested_transfers=1;report.completed_count=one.completed_count;report.cleanup=one.cleanup;report.error=one.error;report.transfers.push_back(std::move(one));}
    else report=o.duplex?commissioning->run_duplex_session(configs,payloads):commissioning->run_session(configs,payloads);
    const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    // Verify probe quiescence before the session's execution-owner cleanup.
    const bool probe_closed=probe->close();
    if(!probe_closed){report.cleanup=DirectPoolReleaseStatus::failed_retained;report.error+=" GPU verifier retained";}
    if(report.cleanup==DirectPoolReleaseStatus::released){execution->quiescent=true;if(!execution->release()){report.cleanup=DirectPoolReleaseStatus::failed_retained;report.error+=" execution retained";}}
    commissioning.reset();payloads.clear();probe.reset();owner={};gpu.reset();lease.reset();
    if(report.cleanup==DirectPoolReleaseStatus::released){
        const auto e=exported->release(),a=allocator->drain();
        if(e!=DirectPoolReleaseStatus::released||a!=DirectPoolReleaseStatus::released){report.cleanup=DirectPoolReleaseStatus::failed_retained;report.error+=" export/allocator retained";}
    }
    const auto b=[](bool v){return v?"true":"false";};
    std::cout<<"{\"schema\":\"strix-gpu-visibility-v1\",\"test_mode\":\"native_gpu\",\"production_qualified\":false"
        <<",\"campaign\":"<<quote(o.campaign)<<",\"run\":"<<quote(o.run)<<",\"local\":"<<quote(o.local)<<",\"peer\":"<<quote(o.peer)
        <<",\"gpu_uuid\":"<<quote(o.gpu_uuid)<<",\"architecture\":"<<quote(architecture)<<",\"runtime_version\":"<<identity.runtime_version<<",\"gpu_driver_version\":"<<identity.driver_version
        <<",\"kernel\":"<<quote(o.kernel)<<",\"driver_build\":"<<quote(o.build)<<",\"epoch\":"<<o.epoch<<",\"seed\":"<<o.seed<<",\"bytes\":"<<o.bytes
        <<",\"elapsed_ns\":"<<elapsed_ns<<",\"concurrency\":"<<(o.duplex?2:1)<<",\"operations_per_direction\":1,\"iterations\":"<<o.iterations<<",\"completed\":"<<report.completed_count
        <<",\"producer_delay_ms\":"<<o.producer_delay.count()<<",\"consumer_delay_ms\":"<<o.consumer_delay.count()
        <<",\"cleanup\":"<<quote(report.cleanup==DirectPoolReleaseStatus::released?"released":"retained")<<",\"transfers\":[";
    for(unsigned i=0;i<o.iterations;++i){
        if(i) std::cout<<',';
        const auto& t=(*observations)[i];const bool rx=configs[i].receiver;
        const bool completed=i<report.transfers.size()&&report.transfers[i].completed_count==1;
        const auto transfer=i<report.transfers.size()?report.transfers[i]:DirectGpuCommissioningReport{};
        std::cout<<"{\"sequence\":"<<i+1<<",\"receiver\":"<<b(rx)<<",\"completed\":"<<b(completed)
            <<",\"controller_complete\":"<<b(transfer.controller_frames_complete)<<",\"consumer_credit\":"<<b(transfer.consumer_credit)
            <<",\"transport_ack\":"<<b(transfer.transport_ack)<<",\"receive_reposted\":"<<b(transfer.receive_reposted)
            <<",\"producer_completed\":"<<b(t.producer_completed)<<",\"raw_gpu_verified\":"<<b(t.raw_verified)
            <<",\"copied_gpu_verified\":"<<b(t.copied_verified)<<",\"negative_control_detected\":"<<b(t.negative_detected)
            <<",\"raw_previously_read\":"<<b(t.raw_previously_read)<<",\"raw\":";result_json(t.raw);
        std::cout<<",\"copied\":";result_json(t.copied);std::cout<<'}';
    }
    std::cout<<"],\"error\":"<<quote(report.error.substr(0,4096))<<"}\n";
    return report.completed_count==o.iterations&&report.cleanup==DirectPoolReleaseStatus::released&&report.error.empty()?0:1;
}
}
int main(int argc,char** argv){
    try{const auto options=strix::transport::visibility_lab::parse(argc,argv);
        if(options.check_only){std::cout<<"{\"schema\":\"strix-gpu-visibility-config-v1\",\"executed\":false,\"production_qualified\":false}\n";return 0;}
        return run(options);
    }catch(const std::exception& e){std::cerr<<"GPU visibility commissioning failed: "<<e.what()<<'\n';return 1;}
}
