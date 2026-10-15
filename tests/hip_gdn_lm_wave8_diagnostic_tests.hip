#include "strix/hip_gdn_lm_wave8_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace strix::gdn_lm_wave8_diagnostic;
void check(hipError_t e,const char* operation) {
    if(e!=hipSuccess)throw std::runtime_error(std::string(operation)+": "+hipGetErrorString(e));
}
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
struct Owner {
    strix::HipSetupQuiescenceGuard quiescence;
    hipStream_t stream{};
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    std::vector<hipGraphExec_t> executions;
    Owner(){quiescence.run([&]{check(hipStreamCreate(&stream),"create owned stream");});}
    ~Owner(){
        quiescence.recover_or_fail_stop();
        const auto cleanup=[](hipError_t e){if(e!=hipSuccess)std::_Exit(3);};
        for(auto e:executions)cleanup(hipGraphExecDestroy(e));
        for(auto g:graphs)cleanup(hipGraphDestroy(g));
        for(auto p:allocations)cleanup(hipFree(p));
        cleanup(hipStreamDestroy(stream));
    }
    Span allocate(std::uint64_t bytes){
        void* p=nullptr;check(hipMalloc(&p,static_cast<std::size_t>(bytes+128U)),"guarded allocate");
        try{allocations.push_back(p);}catch(...){quiescence.recover_or_fail_stop();(void)hipFree(p);throw;}
        check(hipMemset(p,0xa5,static_cast<std::size_t>(bytes+128U)),"poison allocation");
        return {reinterpret_cast<std::uintptr_t>(p)+64U,bytes};
    }
    void finish(){check(hipStreamSynchronize(stream),"owned stream completion");quiescence.require_original();}
};
template<class T> void upload(Span span,const std::vector<T>& values){
    require(span.bytes==values.size()*sizeof(T),"upload extent mismatch");
    check(hipMemcpy(reinterpret_cast<void*>(span.address),values.data(),static_cast<std::size_t>(span.bytes),hipMemcpyHostToDevice),"upload");
}
std::vector<unsigned char> download(Span span){
    std::vector<unsigned char> result(static_cast<std::size_t>(span.bytes));
    check(hipMemcpy(result.data(),reinterpret_cast<const void*>(span.address),result.size(),hipMemcpyDeviceToHost),"download");
    return result;
}
void guards(Span span){
    std::array<unsigned char,64> observed{};
    for(auto address:{span.address-64U,span.address+span.bytes}){
        check(hipMemcpy(observed.data(),reinterpret_cast<const void*>(address),observed.size(),hipMemcpyDeviceToHost),"guard download");
        require(std::all_of(observed.begin(),observed.end(),[](auto b){return b==0xa5;}),"canary changed");
    }
}
float half(std::uint16_t value){
    const unsigned exponent=(value>>10U)&31U,mantissa=value&1023U;
    require(exponent!=31U,"finite host oracle received nonfinite half");
    const float magnitude=exponent==0?std::ldexp(static_cast<float>(mantissa),-24):
        std::ldexp(1.0F+static_cast<float>(mantissa)/1024.0F,static_cast<int>(exponent)-15);
    return value&0x8000U?-magnitude:magnitude;
}
std::vector<unsigned char> oracle(Shape shape,const std::vector<std::uint16_t>& input,
                                 const std::vector<unsigned char>& weight,Kind kind){
    std::vector<unsigned char> result(static_cast<std::size_t>(shape.outputs)*4U);
    for(unsigned row=0;row<shape.outputs;++row){
        std::array<float,256> leaf{};
        for(unsigned i=0;i<256;++i)for(unsigned feature=i;feature<shape.width;feature+=256U){
            const auto offset=(static_cast<std::uint64_t>(row)*(shape.width/32U)+feature/32U)*34U;
            const auto scale=static_cast<std::uint16_t>(weight[offset]|static_cast<unsigned>(weight[offset+1U])<<8U);
            const auto code=std::bit_cast<signed char>(weight[offset+2U+feature%32U]);
            volatile float coefficient=half(scale)*static_cast<float>(code);
            const float x=std::bit_cast<float>(static_cast<std::uint32_t>(input[feature])<<16U);
            leaf[i]=std::fma(coefficient,x,leaf[i]);
        }
        for(unsigned distance=128;distance;distance/=2U)
            for(unsigned i=0;i<distance;++i){volatile float sum=leaf[i]+leaf[i+distance];leaf[i]=sum;}
        auto bits=std::bit_cast<std::uint32_t>(leaf[0]);
        if(kind==Kind::gdn_bf16)bits=(bits+0x7fffU+((bits>>16U)&1U))&0xffff0000U;
        std::memcpy(result.data()+static_cast<std::size_t>(row)*4U,&bits,4);
    }
    return result;
}
void same(const std::vector<unsigned char>& actual,const std::vector<unsigned char>& expected,
          Shape shape,Kind kind,Arm arm,unsigned flavor,const char* phase){
    if(actual==expected)return;
    std::size_t first=0;while(first<actual.size()&&actual[first]==expected[first])++first;
    std::cerr<<"wave8 mismatch width="<<shape.width<<" outputs="<<shape.outputs
        <<" kind="<<static_cast<int>(kind)<<" arm="<<static_cast<int>(arm)
        <<" flavor="<<flavor<<" phase="<<phase<<" first_byte="<<first<<'\n';
    throw std::runtime_error("diagnostic output bits differ");
}
std::pair<unsigned,unsigned> nan_pair(unsigned index){
    for(unsigned distance:{128U,64U,32U,16U,8U})for(unsigned j=0;j<8;++j){
        if(!index)return {j,j+distance};
        --index;
    }
    for(unsigned distance:{4U,2U,1U})for(unsigned j=0;j<distance;++j){
        if(!index)return {j,j+distance};
        --index;
    }
    throw std::runtime_error("NaN site index outside47");
}
void fill(Shape shape,unsigned flavor,std::vector<std::uint16_t>& input,std::vector<unsigned char>& weight){
    for(unsigned feature=0;feature<shape.width;++feature){
        if(flavor==0)input[feature]=static_cast<std::uint16_t>(0x3e00U+((feature*37U)&0x1ffU));
        else if(flavor==1){constexpr std::uint16_t b[]{0,0x8000U,1,0x8001U,0x0080U,0x8080U};input[feature]=b[feature%6U];}
        else if(flavor==2)input[feature]=feature&1U?0xbf80U:0x3f80U;
        else if(flavor==3)input[feature]=feature&1U?0xff7fU:0x7f7fU;
        else if(flavor==4){constexpr std::uint16_t b[]{0x7f80U,0xff80U,0x7fc1U,0xffd5U,0,0x3f80U,0x7f81U,0xff81U};input[feature]=b[feature%8U];}
        else input[feature]=0;
    }
    if(flavor>=5){const auto pair=nan_pair(flavor-5U);input[pair.first]=0x7fc1U;input[pair.second]=0xffd5U;}
    for(unsigned row=0;row<shape.outputs;++row)for(unsigned group=0;group<shape.width/32U;++group){
        const auto offset=(static_cast<std::uint64_t>(row)*(shape.width/32U)+group)*34U;
        std::uint16_t scale=0x3c00U;
        if(flavor==0)scale=static_cast<std::uint16_t>(0x3000U+((row*53U+group*17U)&0x7ffU));
        if(flavor==1){constexpr std::uint16_t h[]{0,0x8000U,1,0x8001U,0x0400U};scale=h[(row+group)%5U];}
        if(flavor==4){constexpr std::uint16_t h[]{0x7c00U,0xfc00U,0x7e01U,0xfe55U,0x3c00U,0x7c01U};scale=h[(row+group)%6U];}
        weight[offset]=static_cast<unsigned char>(scale);weight[offset+1U]=static_cast<unsigned char>(scale>>8U);
        for(unsigned code=0;code<32;++code){
            const int value=flavor>=5?1:static_cast<int>((row*19U+group*13U+code*7U)%256U)-128;
            weight[offset+2U+code]=std::bit_cast<unsigned char>(static_cast<signed char>(value));
        }
    }
}
void graph_identity(hipGraph_t graph,Shape shape,Kind kind,Arm arm){
    std::size_t count=0;check(hipGraphGetNodes(graph,nullptr,&count),"graph node count");
    require(count==1,"diagnostic graph must have exactly one node");
    hipGraphNode_t node{};check(hipGraphGetNodes(graph,&node,&count),"graph node");
    hipGraphNodeType type{};check(hipGraphNodeGetType(node,&type),"graph node type");
    require(type==hipGraphNodeTypeKernel,"diagnostic graph node is not kernel");
    hipKernelNodeParams parameters{};check(hipGraphKernelNodeGetParams(node,&parameters),"graph node parameters");
    const unsigned waves=arm==Arm::wave8?8U:1U;
    require(parameters.func==kernel_symbol(kind,arm),"graph kernel identity mismatch");
    require(parameters.gridDim.x==(shape.outputs+waves-1U)/waves&&parameters.gridDim.y==1&&parameters.gridDim.z==1&&
            parameters.blockDim.x==32U*waves&&parameters.blockDim.y==1&&parameters.blockDim.z==1&&parameters.sharedMemBytes==0,
            "graph kernel geometry mismatch");
}
void family(Shape shape,Kind kind,unsigned flavor,bool host_oracle){
    const std::uint64_t weights_bytes=static_cast<std::uint64_t>(shape.width/32U)*34U*shape.outputs;
    const auto device_bytes=weights_bytes+static_cast<std::uint64_t>(shape.width)*2U+static_cast<std::uint64_t>(shape.outputs)*4U+384U;
    require(device_bytes<=1024ULL*1024ULL*1024ULL,"fixture exceeds1GiB device budget");
    std::size_t free=0,total=0;check(hipMemGetInfo(&free,&total),"memory admission");
    require(free>=device_bytes+512ULL*1024ULL*1024ULL,"insufficient admitted fixture memory");
    Owner owner;
    const auto input=owner.allocate(static_cast<std::uint64_t>(shape.width)*2U),
        weights=owner.allocate(weights_bytes),output=owner.allocate(static_cast<std::uint64_t>(shape.outputs)*4U);
    std::vector<std::uint16_t> hidden(shape.width);
    std::vector<unsigned char> packed(static_cast<std::size_t>(weights_bytes));fill(shape,flavor,hidden,packed);
    upload(input,hidden);upload(weights,packed);owner.finish();
    const auto original_input=download(input),original_weights=download(weights);
    enqueue(shape,input,weights,output,kind,Arm::reference,owner.stream);owner.finish();
    const auto expected=download(output);
    if(host_oracle){
        same(expected,oracle(shape,hidden,packed,kind),shape,kind,Arm::reference,flavor,"finite CPU oracle");
        auto changed=packed;changed[1]^=0x40U; // finite block scale x65536 survives BF16 rounding
        require(oracle(shape,hidden,changed,kind)!=expected,"fixture does not detect corrupted coefficient");
        auto changed_input=hidden;changed_input[0]=0x4000U;
        require(oracle(shape,changed_input,packed,kind)!=expected,"fixture does not detect wrong input leaf");
    }
    for(auto arm:{Arm::reference,Arm::prefetch1,Arm::wave8}){
        check(hipMemset(reinterpret_cast<void*>(output.address),0xcd,static_cast<std::size_t>(output.bytes)),"poison output");
        enqueue(shape,input,weights,output,kind,arm,owner.stream);owner.finish();
        same(download(output),expected,shape,kind,arm,flavor,"eager");
        hipGraph_t graph{};check(hipStreamBeginCapture(owner.stream,hipStreamCaptureModeGlobal),"begin capture");
        try{enqueue(shape,input,weights,output,kind,arm,owner.stream);}
        catch(...){(void)hipStreamEndCapture(owner.stream,&graph);if(graph)owner.graphs.push_back(graph);throw;}
        check(hipStreamEndCapture(owner.stream,&graph),"end capture");owner.graphs.push_back(graph);
        graph_identity(graph,shape,kind,arm);
        hipGraphExec_t execution{};check(hipGraphInstantiate(&execution,graph,nullptr,nullptr,0),"instantiate graph");owner.executions.push_back(execution);
        for(unsigned replay=0;replay<3;++replay){
            check(hipMemset(reinterpret_cast<void*>(output.address),static_cast<int>(0x31U+replay),static_cast<std::size_t>(output.bytes)),"poison graph output");
            check(hipGraphLaunch(execution,owner.stream),"graph replay");owner.finish();
            same(download(output),expected,shape,kind,arm,flavor,"graph");
        }
    }
    require(download(input)==original_input&&download(weights)==original_weights,"read-only buffer changed");
    for(auto span:{input,weights,output})guards(span);
}
void invalid_preflight(){
    Owner owner;const Shape good{32,9};
    const auto input=owner.allocate(64),weights=owner.allocate(34U*9U),output=owner.allocate(36);
    const auto before=download(output);
    for(auto valid_kind:{Kind::gdn_bf16,Kind::lm_bf16})
      for(auto valid_arm:{Arm::reference,Arm::prefetch1,Arm::wave8})
       for(unsigned mode=0;mode<12;++mode){
        auto shape=good;auto x=input,w=weights,y=output;auto kind=valid_kind;auto arm=valid_arm;
        if(mode==0)shape.width=33;
        if(mode==1)--x.bytes;
        if(mode==2)++w.bytes;
        if(mode==3)x.address=0;
        if(mode==4)++w.address;
        if(mode==5)y.address=x.address;
        if(mode==6)kind=static_cast<Kind>(99);
        if(mode==7)arm=static_cast<Arm>(99);
        if(mode==8)shape.outputs=0;
        if(mode==9)y.bytes=0;
        if(mode==10)y.address=w.address;
        if(mode==11)++y.bytes;
        hipGraph_t graph{};check(hipStreamBeginCapture(owner.stream,hipStreamCaptureModeGlobal),"begin rejection capture");
        bool rejected=false;
        try{enqueue(shape,x,w,y,kind,arm,owner.stream);}catch(const std::invalid_argument&){rejected=true;}
        catch(...){(void)hipStreamEndCapture(owner.stream,&graph);if(graph)owner.graphs.push_back(graph);throw;}
        check(hipStreamEndCapture(owner.stream,&graph),"end rejection capture");owner.graphs.push_back(graph);
        std::size_t nodes=0;check(hipGraphGetNodes(graph,nullptr,&nodes),"rejection graph nodes");
        require(rejected&&nodes==0,"invalid preflight enqueued work or wrong rejection");owner.finish();
        require(download(output)==before,"invalid preflight changed output");
    }
    for(auto span:{input,weights,output})guards(span);
}
}
int main()try{
    invalid_preflight();unsigned families=0;
    for(auto kind:{Kind::gdn_bf16,Kind::lm_bf16}){
        require(kernel_symbol(kind,Arm::reference)!=kernel_symbol(kind,Arm::wave8),"wave8 symbol aliases incumbent");
        if(kind==Kind::lm_bf16)require(kernel_symbol(kind,Arm::reference)!=kernel_symbol(kind,Arm::prefetch1),"LM prefetch control aliases incumbent");
        for(auto width:{32U,96U,256U,288U,640U,2560U})
            for(auto outputs:{1U,7U,9U,19U})
                for(unsigned flavor=0;flavor<5;++flavor){family({width,outputs},kind,flavor,flavor==0);++families;}
        for(unsigned site=0;site<47;++site){family({256,9},kind,5U+site,false);++families;}
    }
    for(auto outputs:{10240U,6144U}){family({2560,outputs},Kind::gdn_bf16,0,false);++families;}
    family({2560,248320},Kind::lm_bf16,0,false);++families;
    std::cout<<"wave8 native diagnostic PASS families="<<families<<" NaN_sites_per_kind=47 graph_replays_per_arm=3 canonical_shapes=3 timing=false\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
