#include "strix/hip_q51_grouped_down_probe.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "q51_grouped_down_oracle.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace old = strix::q51_grouped_down;
namespace candidate = strix::q51_slot_parallel;
using Guard=std::shared_ptr<const strix::HipSetupQuiescenceGuard>;
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
void check(hipError_t error,const char* message) {
    if(error!=hipSuccess) throw std::runtime_error(std::string(message)+": "+hipGetErrorString(error));
}
struct Buffer {
    Guard guard; void* owner=nullptr; std::uint64_t bytes=0;
    Buffer(Guard g,std::uint64_t n):guard(std::move(g)),bytes(n) {
        guard->run([&] { check(hipMalloc(&owner,static_cast<std::size_t>(bytes+512U)),"allocate guarded slot view");
            check(hipMemset(owner,0xa5,static_cast<std::size_t>(bytes+512U)),"initialize slot owner guards"); });
    }
    ~Buffer() { if(owner) { guard->recover_or_fail_stop(); if(hipFree(owner)!=hipSuccess) std::_Exit(3); } }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    void* data() const { return static_cast<std::byte*>(owner)+256U; }
    old::DeviceBytes view() const { return {reinterpret_cast<std::uintptr_t>(data()),bytes}; }
    void upload(const void* source) { guard->run([&] {
        check(hipMemcpy(data(),source,static_cast<std::size_t>(bytes),hipMemcpyHostToDevice),"upload slot view"); }); }
    void poison(hipStream_t stream) { check(hipMemsetAsync(data(),0xa5,static_cast<std::size_t>(bytes),stream),"poison slot destination"); }
    std::vector<std::uint32_t> read() const {
        std::vector<std::uint32_t> result(static_cast<std::size_t>(bytes/4U));
        guard->run([&] { check(hipMemcpy(result.data(),data(),static_cast<std::size_t>(bytes),hipMemcpyDeviceToHost),"read slot view"); });
        return result;
    }
    void unchanged() const { const auto values=read(); require(std::all_of(values.begin(),values.end(),
        [](auto value) { return value==0xa5a5a5a5U; }),"rejected slot call changed destination"); }
    void guards() const {
        std::array<std::byte,256> before{},after{};
        guard->run([&] { check(hipMemcpy(before.data(),owner,256,hipMemcpyDeviceToHost),"read slot leading guard");
            check(hipMemcpy(after.data(),static_cast<std::byte*>(data())+bytes,256,hipMemcpyDeviceToHost),"read slot trailing guard"); });
        const auto intact=[](auto value) { return value==std::byte{0xa5}; };
        require(std::all_of(before.begin(),before.end(),intact)&&std::all_of(after.begin(),after.end(),intact),"slot owner guard changed");
    }
};
struct Stream {
    Guard guard; hipStream_t value=nullptr;
    explicit Stream(Guard g):guard(std::move(g)) { check(hipStreamCreateWithFlags(&value,hipStreamNonBlocking),"create slot stream"); }
    ~Stream() { guard->recover_or_fail_stop(); if(value&&hipStreamDestroy(value)!=hipSuccess) std::_Exit(3); }
    Stream(const Stream&)=delete; Stream& operator=(const Stream&)=delete;
};
void half_at(std::span<std::byte> row,unsigned block,unsigned offset,std::uint16_t bits) {
    row[block*24U+offset]=std::byte(bits&255U); row[block*24U+offset+1U]=std::byte(bits>>8U);
}
void run(Guard guard,old::Shape shape,unsigned special=0,bool duplicate=false) {
    const auto n=old::required_bytes(shape);
    const auto raw_bytes=candidate::required_rawdots_bytes(shape);
    std::vector<float> input(static_cast<std::size_t>(n.intermediates/4U),std::bit_cast<float>(0x7fc45678U));
    std::vector<std::byte> weights(static_cast<std::size_t>(n.weights));
    std::vector<std::uint32_t> routes(shape.top_k);
    std::vector<float> mix(shape.top_k);
    for(unsigned slot=0;slot<shape.top_k;++slot) {
        routes[slot]=duplicate?0U:(slot*3U+1U)%shape.experts;
        mix[slot]=slot%2U?-0.75F:0.5F;
        if(special==1U) mix[slot]=slot%2U?-0.0F:0.0F;
        if(special==2U) mix[slot]=-1.0F;
        if(special==5U) mix[slot]=0.0F;
        if(special==7U) mix[slot]=slot%2U?-0.0F:0.0F;
        for(unsigned feature=0;feature<shape.intermediate;++feature) {
            auto value=static_cast<float>(static_cast<int>((feature*7U+slot*11U)%257U)-128)/32.0F;
            if(special==1U) value=-0.0F;
            if(special==2U) value=0.0F;
            if(special==3U) value=std::bit_cast<float>((feature%127U)+1U);
            if(special==4U||special==5U) value=std::numeric_limits<float>::max();
            if(special==6U||special==8U) value=std::bit_cast<float>(feature%2U?0x7fc54321U:0x7fc12345U);
            if(special==7U) value=std::bit_cast<float>(0x80000001U);
            input[static_cast<std::size_t>(slot)*2U*shape.intermediate+feature]=value;
        }
    }
    const auto row_bytes=static_cast<std::size_t>(shape.intermediate/32U)*24U;
    for(unsigned expert=0;expert<shape.experts;++expert) for(unsigned out=0;out<shape.hidden;++out) {
        auto row=std::span<std::byte>(weights).subspan((static_cast<std::size_t>(expert)*shape.hidden+out)*row_bytes,row_bytes);
        for(unsigned block=0;block<shape.intermediate/32U;++block) {
            // Code zero with nonzero m exercises the offset term independently.
            half_at(row,block,0,special==8U?0x7e11U:0x3c00U);
            half_at(row,block,2,special==8U?0x7e22U:special==4U||special==5U?0x7bffU:0x3800U);
        }
    }
    Buffer activations(guard,n.intermediates),packed(guard,n.weights),ids(guard,n.routes),route_weights(guard,n.route_weights),
        status(guard,n.status),accumulation(guard,n.accumulation),raw(guard,raw_bytes);
    activations.upload(input.data()); packed.upload(weights.data()); ids.upload(routes.data()); route_weights.upload(mix.data());
    const old::Request reference{true,old::numerical_identity,shape};
    const old::Bindings old_bindings{activations.view(),packed.view(),ids.view(),route_weights.view(),status.view(),accumulation.view()};
    const candidate::Request request{true,candidate::numerical_identity,shape};
    const candidate::Bindings bindings{activations.view(),packed.view(),ids.view(),route_weights.view(),status.view(),accumulation.view(),raw.view()};
    Stream stream(guard);
    const auto execute=[&](bool use_candidate,unsigned preset=0U) {
        guard->run([&] {
            accumulation.poison(stream.value); raw.poison(stream.value);
            check(hipMemcpyAsync(status.data(),&preset,4,hipMemcpyHostToDevice,stream.value),"set slot sticky status");
            if(use_candidate) candidate::enqueue_slot_parallel_probe(request,bindings,stream.value);
            else old::enqueue_grouped_probe(reference,old_bindings,stream.value);
        });
        return accumulation.read();
    };
    const auto expected=execute(false);
    for(unsigned repeat=0;repeat<3U;++repeat) {
        require(execute(true)==expected,"slot-parallel complete accumulator differs bitwise from actual grouped kernel");
        require(status.read().front()==0U,"valid slot probe changed sticky status");
        if(special<=3U||special==7U) {
            const auto actual=raw.read();
            for(unsigned slot=0;slot<shape.top_k;++slot) for(unsigned out=0;out<shape.hidden;++out) {
                const auto row=(static_cast<std::size_t>(routes[slot])*shape.hidden+out)*row_bytes;
                const auto dot=q51_down_test::legacy(std::span<const float>(input).subspan(
                    static_cast<std::size_t>(slot)*2U*shape.intermediate,shape.intermediate),
                    std::span<const std::byte>(weights).subspan(row,row_bytes));
                require(actual[static_cast<std::size_t>(slot)*shape.hidden+out]==std::bit_cast<std::uint32_t>(dot),
                    "raw dot differs from original virtual256 independent oracle");
            }
        }
    }
    const auto rejected=[&](unsigned preset,unsigned expected_status) {
        static_cast<void>(execute(true,preset));
        require(status.read().front()==expected_status,"slot route/weight validator changed sticky error semantics");
        accumulation.unchanged(); raw.unchanged();
    };
    rejected(0x40U,0x40U);
    for(unsigned slot:std::array<unsigned,3>{0U,shape.top_k/2U,shape.top_k-1U}) {
        auto bad=routes; bad[slot]=shape.experts; ids.upload(bad.data()); rejected(0,1); rejected(0x40U,0x41U); ids.upload(routes.data());
    }
    for(unsigned bits:{0x7fc00000U,0x7f800000U,0xff800000U}) {
        auto bad=mix; bad.back()=std::bit_cast<float>(bits); route_weights.upload(bad.data()); rejected(0,1); rejected(0x40U,0x41U); route_weights.upload(mix.data());
    }
    // A source admission error during capture must enqueue no validator or dots.
    guard->run([&] { accumulation.poison(stream.value); raw.poison(stream.value); check(hipMemsetAsync(status.data(),0,4,stream.value),"reset capture sentinel"); });
    check(hipStreamBeginCapture(stream.value,hipStreamCaptureModeThreadLocal),"begin rejected slot capture");
    bool rejected_capture=false;
    try { candidate::enqueue_slot_parallel_probe(request,bindings,stream.value); }
    catch(const std::invalid_argument& error) { rejected_capture=std::string(error.what()).find("eager execution only")!=std::string::npos; }
    hipGraph_t graph=nullptr;
    check(hipStreamEndCapture(stream.value,&graph),"end rejected slot capture");
    std::size_t nodes=0; check(hipGraphGetNodes(graph,nullptr,&nodes),"inspect rejected slot graph");
    check(hipGraphDestroy(graph),"destroy empty slot graph");
    require(rejected_capture&&nodes==0U,"slot probe capture admission did not reject before first enqueue");
    accumulation.unchanged(); raw.unchanged(); require(status.read().front()==0U,"capture rejection changed status");
    for(const auto* buffer:{&activations,&packed,&ids,&route_weights,&status,&accumulation,&raw}) buffer->guards();
    std::cout<<"slot probe H="<<shape.hidden<<" I="<<shape.intermediate<<" K="<<shape.top_k<<" special="<<special<<" duplicate="<<duplicate<<'\n';
}
} // namespace
int main() try {
    require(std::fegetround()==FE_TONEAREST,"raw-dot CPU oracle requires RNE");
    volatile float tiny=std::numeric_limits<float>::denorm_min(),one=1.0F;
    require(std::bit_cast<std::uint32_t>(static_cast<float>(tiny*one))==1U,"raw-dot CPU oracle requires gradual denormals");
    const auto guard=std::make_shared<strix::HipSetupQuiescenceGuard>();
    require(guard->architecture().starts_with("gfx1151"),"slot probe requires actual gfx1151");
    for(auto shape:std::array<old::Shape,4>{{{1,32,1,1},{2557,32,10,10},{2560,640,10,10},{67,640,10,10}}}) run(guard,shape);
    run(guard,{67,640,10,10},0,true);
    for(unsigned special=1;special<=8U;++special) run(guard,{9,288,3,3},special);
    std::cout<<"Slot-parallel Q5_1 eager probe complete; no full-model or performance qualification\n";
    return 0;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
