#include "strix/hip_q51_grouped_down_probe.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "q51_grouped_down_oracle.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>
namespace {
using namespace strix::q51_grouped_down;
int failures=0;
void expect(bool b,const char* m){if(!b){++failures;std::cerr<<"FAIL: "<<m<<'\n';}}
void check(hipError_t e,const char* m){if(e!=hipSuccess)throw std::runtime_error(std::string(m)+": "+hipGetErrorString(e));}
using Guard=std::shared_ptr<const strix::HipSetupQuiescenceGuard>;
struct Buffer {
    Guard guard;void* pointer=nullptr;std::uint64_t bytes=0;
    Buffer(Guard g,std::uint64_t n):guard(std::move(g)),bytes(n){
        guard->run([&]{check(hipMalloc(&pointer,static_cast<std::size_t>(bytes+64)),"allocate guarded Q5_1 view");
            check(hipMemset(static_cast<std::byte*>(pointer)+bytes,0xff,64),"poison readonly/canary tail");});
    }
    ~Buffer(){if(pointer){guard->recover_or_fail_stop();if(hipFree(pointer)!=hipSuccess)std::_Exit(3);}}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    DeviceBytes view()const{return{reinterpret_cast<std::uintptr_t>(pointer),bytes};}
    void upload(const void* src){guard->run([&]{check(hipMemcpy(pointer,src,static_cast<std::size_t>(bytes),hipMemcpyHostToDevice),"upload Q5_1 view");});}
    void tail()const{
        std::array<std::byte,64> after{};
        guard->run([&]{check(hipMemcpy(after.data(),static_cast<const std::byte*>(pointer)+bytes,64,hipMemcpyDeviceToHost),"read Q5_1 canary");});
        expect(std::all_of(after.begin(),after.end(),[](auto b){return b==std::byte{0xff};}),"Q5_1 owner tail stays intact");
    }
};
struct Capture {
    Guard guard;hipStream_t stream=nullptr;hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
    Capture(Guard g,const Request& r,const Bindings& b,bool original):guard(std::move(g)){
        guard->run([&]{
            check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"create Q5_1 stream");
            check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"begin Q5_1 capture");
            if(original)enqueue_original_receipt(r,b,stream);else enqueue_grouped_probe(r,b,stream);
            check(hipStreamEndCapture(stream,&graph),"end Q5_1 capture");
            check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate Q5_1 graph");
        });
    }
    ~Capture(){
        guard->recover_or_fail_stop();
        if(exec&&hipGraphExecDestroy(exec)!=hipSuccess) std::_Exit(3);
        if(graph&&hipGraphDestroy(graph)!=hipSuccess) std::_Exit(3);
        if(stream&&hipStreamDestroy(stream)!=hipSuccess) std::_Exit(3);
    }
    Capture(const Capture&)=delete;Capture& operator=(const Capture&)=delete;
    std::vector<hipKernelNodeParams> kernels(){
        std::size_t count=0;check(hipGraphGetNodes(graph,nullptr,&count),"count Q5_1 graph nodes");
        std::vector<hipGraphNode_t> nodes(count);check(hipGraphGetNodes(graph,nodes.data(),&count),"read Q5_1 graph nodes");
        std::vector<hipKernelNodeParams> out;
        for(auto node:nodes){hipGraphNodeType type;check(hipGraphNodeGetType(node,&type),"inspect Q5_1 node type");
            if(type==hipGraphNodeTypeKernel){hipKernelNodeParams param{};check(hipGraphKernelNodeGetParams(node,&param),"inspect Q5_1 kernel");out.push_back(param);}}
        return out;
    }
};
void half_at(std::span<std::byte> row,unsigned block,unsigned offset,std::uint16_t h){row[block*24U+offset]=std::byte(h&255U);row[block*24U+offset+1U]=std::byte(h>>8U);}
void code_at(std::span<std::byte> row,unsigned f,unsigned code){
    const auto base=f/32U*24U,lane=f%32U;auto bits=std::to_integer<unsigned>(row[base+8U+lane%16U]);
    const auto shift=lane/16U*4U;bits=(bits&~(15U<<shift))|((code&15U)<<shift);row[base+8U+lane%16U]=std::byte(bits);
    if(code&16U)row[base+4U+lane/8U]|=std::byte(1U<<(lane%8U));
}
void run(Guard guard,Shape shape,bool signed_zero=false,bool duplicate=false){
    Request r{true,numerical_identity,shape};const auto n=required_bytes(shape);
    std::vector<float> input(static_cast<std::size_t>(n.intermediates/4),std::bit_cast<float>(0x7fffffffU));
    std::vector<std::byte> weights(static_cast<std::size_t>(n.weights));
    std::vector<unsigned> routes(shape.top_k);std::vector<float> mix(shape.top_k);
    for(unsigned slot=0;slot<shape.top_k;++slot){
        routes[slot]=duplicate ? shape.experts-1U : (slot==0 ? shape.experts-1U : (slot*7U)%shape.experts);
        mix[slot]=signed_zero ? 0.0F : std::ldexp(slot%2 ? -0.75F : 0.5F,-static_cast<int>(slot%4U));
        for(unsigned f=0;f<shape.intermediate;++f){
            const auto hash=(f+slot*31U)*2654435761U;
            input[static_cast<std::size_t>(slot)*2U*shape.intermediate+f]=signed_zero ? -1.0F :
                std::ldexp(static_cast<float>(static_cast<int>(hash%257U)-128)/127.0F,static_cast<int>(hash%25U)-12);
        }
    }
    const auto row_bytes=static_cast<std::size_t>(shape.intermediate/32U)*24U;
    constexpr std::array<std::uint16_t,12> ds{0x0001,0x8001,0x3c01,0xbc01,0x001f,0x801f,0x7bff,0xfbff,0,0x8000,0x3555,0xb555};
    constexpr std::array<std::uint16_t,8> ms{0,0x8000,1,0x8001,0x3c00,0xbc00,0x7bff,0xfbff};
    for(const auto expert:std::set<unsigned>(routes.begin(),routes.end()))for(unsigned out=0;out<shape.hidden;++out){
        const auto offset=(static_cast<std::size_t>(expert)*shape.hidden+out)*row_bytes;
        auto row=std::span<std::byte>(weights).subspan(offset,row_bytes);
        for(unsigned block=0;block<shape.intermediate/32U;++block){
            half_at(row,block,0,signed_zero ? 0x3c00U : ds[(block+out+expert)%ds.size()]);
            half_at(row,block,2,signed_zero ? 0U : ms[(block*3U+out+expert)%ms.size()]);
        }
        for(unsigned f=0;f<shape.intermediate;++f)code_at(row,f,signed_zero ? 1U : (f*11U+out*3U+expert)%32U);
    }
    std::vector<float> expected(shape.hidden);
    for(unsigned out=0;out<shape.hidden;++out){
        float current=0.0F;
        for(unsigned slot=0;slot<shape.top_k;++slot){
            const auto offset=(static_cast<std::size_t>(routes[slot])*shape.hidden+out)*row_bytes;
            const auto dot=q51_down_test::legacy(std::span<const float>(input).subspan(static_cast<std::size_t>(slot)*2U*shape.intermediate,shape.intermediate),
                std::span<const std::byte>(weights).subspan(offset,row_bytes));
            current=std::fma(dot,mix[slot],current);
        }
        if(!std::isfinite(current)) throw std::runtime_error("normal Q5_1 oracle escaped finite scope");
        expected[out]=current;
    }
    Buffer gu(guard,n.intermediates),w(guard,n.weights),ids(guard,n.routes),mw(guard,n.route_weights),status(guard,4),accum(guard,n.accumulation);
    gu.upload(input.data());w.upload(weights.data());ids.upload(routes.data());mw.upload(mix.data());
    Bindings b{gu.view(),w.view(),ids.view(),mw.view(),status.view(),accum.view()};
    const auto verify=[&](const std::vector<unsigned>& actual){
        for(unsigned i=0;i<shape.hidden;++i)expect(actual[i]==std::bit_cast<unsigned>(expected[i]),"Q5_1 complete FP32 accumulator equals independent original256/fmaf oracle");};
    std::vector<unsigned> original(shape.hidden),candidate(shape.hidden);unsigned local=0;
    guard->run([&]{check(hipMemset(accum.pointer,0,static_cast<std::size_t>(n.accumulation)),"clear original accumulator");
        check(hipMemset(status.pointer,0,4),"clear original status");enqueue_original_receipt(r,b);
        check(hipMemcpyAsync(original.data(),accum.pointer,static_cast<std::size_t>(n.accumulation),hipMemcpyDeviceToHost,nullptr),"read original receipt");});
    verify(original);
    guard->run([&]{check(hipMemset(accum.pointer,0xff,static_cast<std::size_t>(n.accumulation)),"poison grouped accumulator");
        check(hipMemset(status.pointer,0,4),"clear grouped status");enqueue_grouped_probe(r,b);
        check(hipMemcpyAsync(candidate.data(),accum.pointer,static_cast<std::size_t>(n.accumulation),hipMemcpyDeviceToHost,nullptr),"read grouped receipt");
        check(hipMemcpyAsync(&local,status.pointer,4,hipMemcpyDeviceToHost,nullptr),"read grouped status");});
    expect(local==0 && original==candidate,"Q5_1 eager grouped equals actual original bits");verify(candidate);
    Capture baseline(guard,r,b,true),grouped(guard,r,b,false);
    const auto old_nodes=baseline.kernels(),new_nodes=grouped.kernels();
    expect(old_nodes.size()==shape.top_k+1U && new_nodes.size()==2U,"Q5_1 capture has K original downs versus one grouped down plus validator");
    unsigned new_down_count=0;
    for(const auto& node:new_nodes){
        const bool known=std::any_of(old_nodes.begin(),old_nodes.end(),[&](const auto& old){return old.func==node.func;});
        if(!known){++new_down_count;expect(node.gridDim.x==(shape.hidden+7U)/8U && node.gridDim.y==1U &&
            node.blockDim.x==256U && node.blockDim.y==1U && node.sharedMemBytes==0U,"actual grouped function uses8-output256-thread geometry without dynamic LDS");}}
    expect(new_down_count==1U,"captured grouped function differs from original/validator");
    for(unsigned replay=0;replay<3;++replay){
        guard->run([&]{check(hipMemsetAsync(accum.pointer,0xff,static_cast<std::size_t>(n.accumulation),grouped.stream),"stream poison before every grouped replay");
            check(hipMemsetAsync(status.pointer,0,4,grouped.stream),"stream reset grouped status");check(hipGraphLaunch(grouped.exec,grouped.stream),"replay grouped Q5_1");
            check(hipMemcpyAsync(candidate.data(),accum.pointer,static_cast<std::size_t>(n.accumulation),hipMemcpyDeviceToHost,grouped.stream),"read grouped graph");
            check(hipMemcpyAsync(&local,status.pointer,4,hipMemcpyDeviceToHost,grouped.stream),"read graph local status");});
        expect(local==0 && candidate==original,"poisoned static replay equals original bits");verify(candidate);
    }
    const auto rejected=[&](unsigned preset,unsigned expected_status){
        guard->run([&]{check(hipMemsetAsync(accum.pointer,0xa5,static_cast<std::size_t>(n.accumulation),grouped.stream),"poison rejected graph destination");
            check(hipMemcpyAsync(status.pointer,&preset,4,hipMemcpyHostToDevice,grouped.stream),"preset sticky Q5_1 status");
            check(hipGraphLaunch(grouped.exec,grouped.stream),"replay rejected Q5_1 graph");
            check(hipMemcpyAsync(candidate.data(),accum.pointer,static_cast<std::size_t>(n.accumulation),hipMemcpyDeviceToHost,grouped.stream),"read rejected Q5_1 destination");
            check(hipMemcpyAsync(&local,status.pointer,4,hipMemcpyDeviceToHost,grouped.stream),"read rejected Q5_1 status");});
        expect(local==expected_status && std::all_of(candidate.begin(),candidate.end(),[](auto v){return v==0xa5a5a5a5U;}),"invalid/sticky Q5_1 graph leaves destination unchanged");
    };
    rejected(0x40U,0x40U);
    for(const auto slot:std::set<unsigned>{0U,shape.top_k/2U,shape.top_k-1U}){
        auto bad=routes;bad[slot]=shape.experts;ids.upload(bad.data());rejected(0,1);ids.upload(routes.data());
    }
    for(const auto bits:{0x7fc00000U,0x7f800000U,0xff800000U}){
        auto bad=mix;bad[0]=std::bit_cast<float>(bits);mw.upload(bad.data());rejected(0,1);mw.upload(mix.data());
    }
    for(const auto* owner:{&gu,&w,&ids,&mw,&status,&accum})owner->tail();
    std::cout<<"Q5_1 grouped probe H="<<shape.hidden<<" I="<<shape.intermediate<<" E="<<shape.experts<<" K="<<shape.top_k
        <<" compared="<<shape.hidden*5U<<" signedzero="<<signed_zero<<" duplicate="<<duplicate<<'\n';
}
}
int main()try{
    if(std::fegetround()!=FE_TONEAREST) throw std::runtime_error("CPU oracle requires RNE");
    volatile float tiny=std::numeric_limits<float>::denorm_min(),one=1.0F;
    if(std::bit_cast<unsigned>(static_cast<float>(tiny*one))!=1U)
        throw std::runtime_error("CPU oracle requires gradual subnormals");
    const auto guard=std::make_shared<strix::HipSetupQuiescenceGuard>();
    for(const auto shape:std::array<strix::q51_grouped_down::Shape,7>{{{1,32,1,1},{7,64,3,2},{8,256,3,3},{9,288,4,3},{67,512,10,10},{67,640,10,10},{2560,640,512,10}}})run(guard,shape);
    run(guard,{9,288,3,3},true);run(guard,{67,640,10,10},false,true);
    if(failures==0)std::cout<<"Q5_1 grouped standalone device probe PASS; no native integration/model/performance qualification\n";
    return failures?1:0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
