#include "strix/hip_lane_chain_hyper_output_tests.hpp"
#include "strix/hip_launch_context.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
void check(hipError_t e,const char* text){if(e!=hipSuccess)throw std::runtime_error(std::string(text)+": "+hipGetErrorString(e));}
void expect(bool value,const char* text){if(!value)throw std::runtime_error(text);}
struct Owner{void* p=nullptr;explicit Owner(std::size_t n){check(hipMalloc(&p,n),"allocate HC fixture owner");}~Owner(){if(p)static_cast<void>(hipFree(p));}Owner(const Owner&)=delete;};
std::vector<std::byte> read(const Owner& x,std::size_t n){std::vector<std::byte> b(n);check(hipMemcpy(b.data(),x.p,n,hipMemcpyDeviceToHost),"read HC receipt");return b;}
void run(unsigned hidden,unsigned streams,unsigned rank,unsigned rows,bool bf16,unsigned adversary){
    strix::HipHyperMixLauncherConfig c;c.hidden_size=hidden;c.hc_count=streams;c.lowrank=rank;c.parallel=true;c.staged_projections=true;c.fused_injection=true;c.typed_projection=true;c.prefetch_rows=true;c.prefetch_q8_output=true;c.activation=bf16?strix::HipActivationStorage::bf16:strix::HipActivationStorage::fp32;
    const auto width=hidden*streams;std::vector<float> normalized(static_cast<std::size_t>(rows)*width),low(static_cast<std::size_t>(rows)*rank);
    for(std::size_t i=0;i<normalized.size();++i)normalized[i]=static_cast<float>(static_cast<int>(i%47U)-23)*0.125F;
    for(std::size_t i=0;i<low.size();++i)low[i]=static_cast<float>(static_cast<int>(i%31U)-15)*0.03125F;
    std::vector<std::uint8_t> weights(static_cast<std::size_t>(width)*(rank/32U)*34U);
    constexpr std::array<std::uint16_t,8> scales={0x0000,0x8000,0x0001,0x8001,0x3c00,0xbc00,0x7bff,0xfbff};
    for(std::size_t i=0;i<weights.size()/34U;++i){const auto scale=adversary?scales[i%scales.size()]:static_cast<std::uint16_t>(0x2400U+(i%128U));weights[i*34U]=static_cast<std::uint8_t>(scale&255U);weights[i*34U+1U]=static_cast<std::uint8_t>(scale>>8U);for(unsigned k=0;k<32;++k)weights[i*34U+2U+k]=static_cast<std::uint8_t>((i*17U+k*59U)&255U);}
    if(adversary==2){constexpr std::array<std::uint32_t,8> bits={0U,0x80000000U,1U,0x80000001U,0x00800000U,0x80800000U,0x7f7fffffU,0xff7fffffU};for(std::size_t i=0;i<low.size();++i)low[i]=std::bit_cast<float>(bits[i%bits.size()]);}
    Owner n(normalized.size()*4U),l(low.size()*4U),w(weights.size());const auto bytes=static_cast<std::size_t>(rows)*hidden*(bf16?2U:4U);Owner base(bytes+32U),candidate(bytes+32U);
    check(hipMemcpy(n.p,normalized.data(),normalized.size()*4U,hipMemcpyHostToDevice),"upload normalized");check(hipMemcpy(l.p,low.data(),low.size()*4U,hipMemcpyHostToDevice),"upload low");check(hipMemcpy(w.p,weights.data(),weights.size(),hipMemcpyHostToDevice),"upload original Q8");
    check(hipMemset(base.p,0xa5,bytes+32U),"poison base");check(hipMemset(candidate.p,0x5a,bytes+32U),"poison candidate");
    strix::enqueue_lane_chain_hyper_output_test(c,static_cast<float*>(n.p),static_cast<float*>(l.p),w.p,base.p,rows,false,nullptr);
    strix::enqueue_lane_chain_hyper_output_test(c,static_cast<float*>(n.p),static_cast<float*>(l.p),w.p,candidate.p,rows,true,nullptr);check(hipDeviceSynchronize(),"drain eager pair");
    const auto expected=read(base,bytes+32U);
    const auto verify=[&]{const auto actual=read(candidate,bytes+32U);expect(std::equal(actual.begin(),actual.begin()+static_cast<std::ptrdiff_t>(bytes),expected.begin()),"lane-chain HC output differs from original same-input kernel bits");expect(std::all_of(actual.begin()+static_cast<std::ptrdiff_t>(bytes),actual.end(),[](std::byte x){return x==std::byte{0x5a};}),"lane-chain output guard overwritten");};verify();
    expect(std::all_of(expected.begin()+static_cast<std::ptrdiff_t>(bytes),expected.end(),[](std::byte x){return x==std::byte{0xa5};}),"original output guard overwritten");
    hipStream_t stream=nullptr;hipGraph_t graph=nullptr;hipGraphExec_t executable=nullptr;check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"create HC stream");
    strix::HipLaunchContext scope;scope.stream=stream;
    {strix::HipLaunchScope active(scope);check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"capture output kernel");try{strix::enqueue_lane_chain_hyper_output_test(c,static_cast<float*>(n.p),static_cast<float*>(l.p),w.p,candidate.p,rows,true,stream);}catch(...){static_cast<void>(hipStreamEndCapture(stream,&graph));if(graph)static_cast<void>(hipGraphDestroy(graph));static_cast<void>(hipStreamDestroy(stream));throw;}check(hipStreamEndCapture(stream,&graph),"end output capture");}
    check(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0),"instantiate HC output");
    for(unsigned replay=0;replay<3;++replay){check(hipMemsetAsync(candidate.p,0x5a,bytes+32U,stream),"poison replay");check(hipGraphLaunch(executable,stream),"replay HC output");check(hipStreamSynchronize(stream),"drain HC replay");verify();}
    check(hipGraphExecDestroy(executable),"destroy HC executable");check(hipGraphDestroy(graph),"destroy HC graph");check(hipStreamDestroy(stream),"destroy HC stream");
    const auto before=read(candidate,bytes+32U);bool rejected=false;
    try{strix::enqueue_lane_chain_hyper_output_test(c,static_cast<float*>(n.p),static_cast<float*>(l.p),static_cast<char*>(w.p)+2,candidate.p,rows,true,nullptr);}catch(const std::invalid_argument&){rejected=true;}
    expect(rejected && read(candidate,bytes+32U)==before,"misaligned C1 owner must reject before enqueue");
    std::cout<<"HC_LANE_CHAIN H="<<hidden<<" C="<<streams<<" R="<<rank<<" rows="<<rows<<" bf16="<<bf16<<" adversary="<<adversary<<" exact eager+graph3\n";
}
}
int main()try{
    for(bool bf16:{false,true})for(unsigned adversary=0;adversary<3;++adversary){run(32,2,32,2,bf16,adversary);run(64,3,96,2,bf16,adversary);run(2560,4,320,1,bf16,adversary);run(32,8,224,2,bf16,adversary);}
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
