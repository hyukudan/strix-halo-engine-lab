#include "run_options.hpp"
#include "native_memory.hpp"
#include "../direct_gpu_stream/bounded_tcp_control.hpp"
#include "strix/gpu_queue_session.hpp"
#include "strix/hip_direct_gpu_pool_allocator.hpp"
#include "strix/hip_gpu_visibility_probe.hpp"
#include "strix/linux_direct_transport_driver.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>

namespace {
using namespace strix;
using namespace strix::transport;
using visibility_lab::quote;
using queue_lab::Options;
using queue_lab::QueueGpuMemory;
using queue_lab::Terminal;
void check(hipError_t e,const char* what){if(e!=hipSuccess)throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));}
std::array<std::uint8_t,16> bytes_uuid(std::string_view text){
    if(!visibility_lab::hex_id(text,32,32))throw std::invalid_argument("invalid UUID");
    std::array<std::uint8_t,16> out{};
    const auto nibble=[](char c){return c<='9'?c-'0':c-'a'+10;};
    for(std::size_t n=0;n<out.size();++n)out[n]=static_cast<std::uint8_t>(nibble(text[n*2])*16+nibble(text[n*2+1]));
    return out;
}
std::string uuid(const std::array<std::uint8_t,16>& id){constexpr char hex[]="0123456789abcdef";std::string out;for(auto b:id){out+=hex[b>>4];out+=hex[b&15];}return out;}
std::uint64_t endpoint_salt(std::string_view text){
    const auto id=bytes_uuid(text);std::uint64_t low=0,high=0;
    for(unsigned n=0;n<8;++n){low|=static_cast<std::uint64_t>(id[n])<<(8*n);high|=static_cast<std::uint64_t>(id[n+8])<<(8*n);}
    return gpu_visibility_mix(low^gpu_visibility_mix(high));
}
struct Receipt {
    GpuQueueOperation operation;
    GpuVisibilityResult raw{},copied{};
    bool producer=false,terminal=false,verified=false,negative=false;
};
void result_json(const GpuVisibilityResult& r){
    std::cout<<"{\"epoch\":"<<r.epoch<<",\"sequence\":"<<r.sequence<<",\"nonce\":"<<r.nonce
        <<",\"checked_bytes\":"<<r.checked_bytes<<",\"mismatches\":"<<r.mismatches<<",\"first_mismatch\":"<<r.first_mismatch
        <<",\"observed_digest\":"<<r.observed_digest<<",\"consumed_digest\":"<<r.consumed_digest<<'}';
}
int run(const Options& options){
    const auto& o=options.base;
    if(const char* value=std::getenv("HSA_OVERRIDE_GFX_VERSION");value&&*value)throw std::invalid_argument("queue runner rejects architecture override");
    const auto driver=LinuxDirectGpuStreamCalls::native();
    const auto preflight=[driver,o]{const auto build=driver.driver_build_identity();if(!build||*build!=o.build||driver.kernel_release()!=o.kernel)throw std::runtime_error("provisioned module/kernel identity changed");};
    preflight();check(hipSetDevice(o.gpu_device),"queue explicit GPU");
    auto exporter=std::make_shared<HipDirectGpuExportIo>();const auto device=exporter->inspect_device();
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,device.index),"queue GPU architecture");
    std::string architecture=properties.gcnArchName;if(auto n=architecture.find(':');n!=std::string::npos)architecture.resize(n);
    if(device.uuid!=bytes_uuid(o.gpu_uuid)||device.index!=o.gpu_device||device.runtime_version!=o.runtime_version||device.driver_version!=o.driver_version||architecture!="gfx1151")
        throw std::runtime_error("queue GPU UUID/runtime/architecture differs from exact expectation");
    // The operator must maintain exclusive ConfigFS mode/route ownership across
    // preflight/open/close. A build ID alone cannot prove gpu_direct selection.
    LinuxGpuQueue queue(open_linux_gpu_queue(o.device,preflight),GpuQueueExpected{bytes_uuid(o.local),bytes_uuid(o.peer),o.ring_frames,options.depth});
    AllocationGenerationSource generations;
    auto allocator=std::make_shared<HipDirectGpuPoolAllocator>(device,HipDirectGpuPoolPolicy{true,128ULL*1024*1024,4096,2ULL*1024*1024});
    const auto geometry=plan_multi_link_transport_pools(std::array{DirectFrameGeometry{4096,o.ring_frames,true}});
    auto exported=DirectGpuExportOwner::allocate_and_export(exporter,allocator,geometry,generations,device);auto lease=exported->retain_for_import();
    const auto binding=lease->binding();
    auto memory=std::make_shared<QueueGpuMemory>(lease);memory->allocate(options.depth,o.bytes,generations,o.timeout);
    std::vector<std::shared_ptr<HipGpuVisibilityProbe>> probes;
    for(unsigned slot=0;slot<options.depth;++slot)probes.push_back(std::make_shared<HipGpuVisibilityProbe>(memory,[memory,slot]{return memory->slot(slot).stream;},[memory]{memory->current();},HipGpuVisibilityProbePolicy{o.bytes,o.timeout}));
    GpuQueuePool pool;pool.lifetime=memory;pool.require_current=[memory]{memory->current();};pool.gpu_quiescent=[memory]{return memory->quiescent();};
    for(unsigned direction=0;direction<2;++direction){auto& b=direction?pool.rx:pool.tx;const auto& view=direction?binding.pool.rx:binding.pool.tx;
        b.fd=binding.descriptors[direction];b.full_bytes=binding.allocations[direction].bytes;b.window_offset=binding.view_offsets[direction];b.window_bytes=view.bytes;
        b.allocation_generation=view.allocation_generation;b.fd_device=binding.descriptor_identities[direction].device;b.fd_inode=binding.descriptor_identities[direction].inode;
    }
    auto canonical=o.campaign+"\n"+o.run+"\n"+o.link+"\n"+std::to_string(o.request)+"\n"+std::to_string(o.seed)+"\n"+
        std::to_string(o.bytes)+"\n"+std::to_string(options.depth)+"\n"+std::to_string(options.windows)+"\n"+std::to_string(options.recovery_every)+"\nqueue-gpu-v1";
    if(options.controller_reset) canonical+="\ncontroller-reset-v1\n"+std::to_string(options.include_network);
    const auto hash=sha256_bytes({reinterpret_cast<const std::byte*>(canonical.data()),canonical.size()});
    const auto tag=gpu_visibility_mix(o.seed^o.request^0x715545554531ULL);
    auto control=std::make_shared<BoundedCommissioningTcpControl>(CommissioningTcpPolicy{o.listener,o.port,2048,100000,o.timeout});
    queue.import_pool(pool,o.epoch,binding.pool_binding_identity);queue.arm();
    GpuQueueSessionPolicy policy;std::copy(hash.bytes.begin(),hash.bytes.end(),policy.campaign.begin());policy.payload_bytes=o.bytes;
    policy.windows=options.windows;policy.directions=options.full_duplex?3:o.receiver?2:1;policy.terminal_bytes=sizeof(Terminal);
    policy.recovery_every_windows=options.recovery_every;policy.window_timeout=o.timeout;
    policy.recovery_flags=options.controller_reset?gpu_queue_reset_controller:0U;
    if(options.include_network)policy.recovery_flags|=gpu_queue_reset_network;
    policy.independent_control=options.independent_control;
    std::vector<Receipt> receipts;receipts.reserve(static_cast<std::size_t>(options.depth)*options.windows*(options.full_duplex?2U:1U));
    const auto terminal_for=[&](const GpuQueueOperation& op){return Terminal{op.epoch,op.sequence,tag^gpu_visibility_mix(op.slot)^endpoint_salt(op.direction?o.peer:o.local),op.payload_bytes};};
    GpuQueueSessionGpu gpu{
        [&](const GpuQueueOperation& op){
            Receipt receipt;receipt.operation=op;auto& s=memory->slot(op.slot);memory->current();
            std::this_thread::sleep_for(o.producer_delay);
            // Domain-separate the payload by SENDER endpoint: swapping opposite
            // direction data must not pass full-duplex verification.
            const auto sender_salt=endpoint_salt(o.local);
            const GpuVisibilityPattern pattern{op.epoch,op.sequence,o.seed^gpu_visibility_mix(sender_salt)};
            probes[op.slot]->produce(std::array{GpuVisibilityRange{s.views[0].address,op.payload_bytes}},pattern);
            check(hipMemcpyAsync(reinterpret_cast<void*>(binding.pool.tx.address+op.pool_offset),reinterpret_cast<const void*>(s.views[0].address),
                static_cast<std::size_t>(op.payload_bytes),hipMemcpyDeviceToDevice,s.stream),"queue TX pack");
            *s.terminal=terminal_for(op);
            check(hipMemcpyAsync(reinterpret_cast<void*>(binding.pool.tx.address+queue.terminal_offset(op.slot)),s.terminal,sizeof(Terminal),hipMemcpyHostToDevice,s.stream),"queue TX terminal publish");
            memory->wait(op.slot);receipt.producer=true;receipts.push_back(receipt);
        },
        [&](const GpuQueueOperation& op){
            Receipt receipt;receipt.operation=op;auto& s=memory->slot(op.slot);memory->current();std::this_thread::sleep_for(o.consumer_delay);
            const auto raw=binding.pool.rx.address+op.pool_offset;
            const auto sender_salt=endpoint_salt(o.peer);
            const GpuVisibilityPattern pattern{op.epoch,op.sequence,o.seed^gpu_visibility_mix(sender_salt)};
            receipt.raw=probes[op.slot]->verify(std::array{GpuVisibilityRange{raw,op.payload_bytes}},pattern);
            if(!valid_gpu_visibility_result(receipt.raw,pattern,receipt.raw.nonce,op.payload_bytes)) {
                receipts.push_back(receipt);return false;
            }
            check(hipMemcpyAsync(reinterpret_cast<void*>(s.views[1].address),reinterpret_cast<const void*>(raw),static_cast<std::size_t>(op.payload_bytes),hipMemcpyDeviceToDevice,s.stream),"queue RX acquire");
            check(hipMemcpyAsync(reinterpret_cast<void*>(s.views[2].address),reinterpret_cast<const void*>(s.views[1].address),static_cast<std::size_t>(op.payload_bytes),hipMemcpyDeviceToDevice,s.stream),"queue RX unpack");
            check(hipMemcpyAsync(s.terminal,reinterpret_cast<const void*>(binding.pool.rx.address+queue.terminal_offset(op.slot)),sizeof(Terminal),hipMemcpyDeviceToHost,s.stream),"queue RX terminal acquire");
            memory->wait(op.slot);receipt.terminal=*s.terminal==terminal_for(op);
            receipt.copied=probes[op.slot]->verify(std::array{GpuVisibilityRange{s.views[2].address,op.payload_bytes}},pattern);
            receipt.verified=receipt.terminal&&valid_gpu_visibility_result(receipt.raw,pattern,receipt.raw.nonce,op.payload_bytes)&&
                valid_gpu_visibility_result(receipt.copied,pattern,receipt.copied.nonce,op.payload_bytes);
            if(receipt.verified)receipt.negative=probes[op.slot]->negative_control(std::array{GpuVisibilityRange{s.views[2].address,op.payload_bytes}},pattern,op.payload_bytes-1);
            receipts.push_back(receipt);return receipt.verified&&receipt.negative;
        }};
    const auto started=std::chrono::steady_clock::now();
    auto report=run_gpu_queue_session(queue,pool,policy,{o.listener,[control](auto bytes){control->send(bytes);},[control]{return control->receive();}},gpu);
    const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
    gpu={};
    if(report.cleanup_released){
        for(auto& probe:probes)if(!probe->close())report.cleanup_released=false;
        if(report.cleanup_released&&!memory->close())report.cleanup_released=false;
    }
    probes.clear();pool={};memory.reset();lease.reset();
    if(report.cleanup_released){if(exported->release()!=DirectPoolReleaseStatus::released||allocator->drain()!=DirectPoolReleaseStatus::released)report.cleanup_released=false;}
    if(!report.cleanup_released&&report.error.empty())report.error="queue native GPU/export cleanup retained";
    std::array<std::uint8_t,16> instance{};std::copy_n(queue.observation().fence.device_instance,16,instance.begin());
    const auto b=[](bool v){return v?"true":"false";};
    std::cout<<"{\"schema\":\"strix-gpu-queue-visibility-v1\",\"production_qualified\":false,\"test_mode\":\"native_gpu\",\"campaign\":"<<quote(o.campaign)
        <<",\"run\":"<<quote(o.run)<<",\"contract_sha256\":"<<quote(model_content_identity_hex(hash))<<",\"local\":"<<quote(o.local)<<",\"peer\":"<<quote(o.peer)
        <<",\"gpu_uuid\":"<<quote(uuid(device.uuid))<<",\"architecture\":"<<quote(architecture)<<",\"kernel\":"<<quote(o.kernel)<<",\"driver_build\":"<<quote(o.build)
        <<",\"runtime_version\":"<<device.runtime_version<<",\"gpu_driver_version\":"<<device.driver_version<<",\"device_instance\":"<<quote(uuid(instance))
        <<",\"physical_link\":"<<quote(o.link)<<",\"request_id\":"<<o.request<<",\"seed\":"<<o.seed<<",\"ring_frames\":"<<o.ring_frames<<",\"recovery_every\":"<<options.recovery_every
        <<",\"epoch\":"<<o.epoch<<",\"final_epoch\":"<<report.final_epoch<<",\"bytes\":"<<o.bytes<<",\"depth\":"<<options.depth<<",\"windows\":"<<options.windows
        <<",\"directions\":"<<policy.directions<<",\"completed_windows\":"<<report.completed_windows<<",\"completed_tx\":"<<report.completed_tx<<",\"completed_rx\":"<<report.completed_rx
        <<",\"controller_reset_requested\":"<<b(options.controller_reset)<<",\"network_recovery_requested\":"<<b(options.include_network)<<",\"controller_resets\":"<<report.controller_resets<<",\"recoveries\":"<<report.recoveries<<",\"peak_outstanding_tx\":"<<report.peak_inflight_tx<<",\"peak_outstanding_rx\":"<<report.peak_inflight_rx<<",\"elapsed_ns\":"<<elapsed
        <<",\"cleanup\":"<<quote(report.cleanup_released?"released":"retained")<<",\"transfers\":[";
    for(std::size_t n=0;n<receipts.size();++n){if(n)std::cout<<',';const auto& r=receipts[n];const auto& op=r.operation;
        std::cout<<"{\"epoch\":"<<op.epoch<<",\"import_generation\":"<<op.import_generation<<",\"sequence\":"<<op.sequence<<",\"slot\":"<<op.slot<<",\"direction\":"<<op.direction
            <<",\"handle\":"<<op.handle<<",\"producer_complete\":"<<b(r.producer)<<",\"terminal_verified\":"<<b(r.terminal)<<",\"payload_verified\":"<<b(r.verified)<<",\"negative_control\":"<<b(r.negative)<<",\"raw\":";
        result_json(r.raw);std::cout<<",\"copied\":";result_json(r.copied);std::cout<<'}';
    }
    std::cout<<"],\"error\":"<<quote(report.error.substr(0,4096))<<"}\n";
    return report.error.empty()&&report.cleanup_released&&report.completed_windows==options.windows?0:1;
}
}
int main(int argc,char** argv){
    if(argc==2&&std::string_view(argv[1])=="--help"){
        std::cout<<"Experimental native GPU queue runner: normal gpu_visibility identity arguments, plus\n"
            "--configured-gpu-direct --queue-depth 2..8 --windows 1..128 [--full-duplex] [--recover-every N].\n"
            "Optional --controller-reset with --recover-every N: privileged local group reset after peer stop agreement.\n"
            "--include-network --independent-control permits USB4NET outage; control must use another physical path.\n"
            "Use --check-config for no HIP/driver/socket work. --iterations/--duplex-pairs are rejected.\n"
            "Authenticated SSH loopback tunnel and exclusive provisioned ConfigFS ownership are required.\n";return 0;
    }
    try{const auto options=queue_lab::parse(argc,argv);
        if(options.base.check_only){std::cout<<"{\"schema\":\"strix-gpu-queue-config-v1\",\"executed\":false,\"production_qualified\":false,\"depth\":"<<options.depth<<",\"windows\":"<<options.windows<<"}\n";return 0;}
        return run(options);
    }catch(const std::exception& error){std::cout<<"{\"schema\":\"strix-gpu-queue-error-v1\",\"production_qualified\":false,\"error\":"<<quote(error.what())<<"}\n";return 1;}
}
