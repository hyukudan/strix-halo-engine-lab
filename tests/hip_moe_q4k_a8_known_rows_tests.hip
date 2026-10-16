#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/moe_plan.hpp"
#include "q4k_a8_oracle.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix;
namespace a8=moe_q4k_a8;
namespace candidate=moe_q4k_a8::known_rows;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
void check(hipError_t e,const char* why){if(e!=hipSuccess)throw std::runtime_error(std::string(why)+": "+hipGetErrorString(e));}
struct Owner {
    HipSetupQuiescenceGuard guard;hipStream_t stream=nullptr;std::vector<void*> allocations;
    Owner(){guard.run([&]{check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"create A8 CSR stream");});}
    ~Owner(){guard.recover_or_fail_stop();for(auto* p:allocations)if(hipFree(p)!=hipSuccess)std::_Exit(3);
        if(hipStreamDestroy(stream)!=hipSuccess)std::_Exit(3);}
    a8::DeviceBytes allocate(std::size_t bytes){
        void* p=nullptr;check(hipMalloc(&p,bytes+512U),"allocate guarded A8 CSR owner");
        try{allocations.push_back(p);}catch(...){guard.recover_or_fail_stop();(void)hipFree(p);throw;}
        check(hipMemset(p,0xa5,bytes+512U),"poison guarded A8 CSR owner");return{reinterpret_cast<std::uintptr_t>(p)+256U,bytes};
    }
    void drain(){check(hipStreamSynchronize(stream),"drain A8 CSR stream");guard.require_original();}
};
void upload(a8::DeviceBytes v,const void* p,std::size_t bytes){require(bytes==v.bytes,"A8 CSR upload extent differs");
    check(hipMemcpy(reinterpret_cast<void*>(v.address),p,bytes,hipMemcpyHostToDevice),"upload A8 CSR view");}
std::vector<std::byte> read(a8::DeviceBytes v){std::vector<std::byte> out(static_cast<std::size_t>(v.bytes));
    check(hipMemcpy(out.data(),reinterpret_cast<const void*>(v.address),out.size(),hipMemcpyDeviceToHost),"read A8 CSR view");return out;}
void poison(a8::DeviceBytes v){check(hipMemset(reinterpret_cast<void*>(v.address),0xa5,static_cast<std::size_t>(v.bytes)),"poison A8 CSR view");}
void guards(a8::DeviceBytes v){for(auto p:{v.address-256U,v.address+v.bytes}){const auto b=read({p,256U});
    require(std::all_of(b.begin(),b.end(),[](auto x){return x==std::byte{0xa5};}),"A8 CSR owner guard changed");}}
std::vector<std::byte> weights(unsigned hidden,unsigned outputs,unsigned experts,unsigned salt,unsigned special){
    std::vector<std::byte> out(static_cast<std::size_t>(hidden/256U)*outputs*experts*144U);
    for(std::size_t b=0;b<out.size()/144U;++b){auto* p=out.data()+b*144U;
        const auto header=[&](unsigned offset,unsigned bits){p[offset]=std::byte(bits&255U);p[offset+1U]=std::byte(bits>>8U);};
        header(0U,special?special:0x2000U+static_cast<unsigned>((b+salt)%3U)*0x400U);header(2U,0x1400U);
        for(unsigned i=4U;i<144U;++i)p[i]=std::byte((b*37U+i*23U+salt*41U)%256U);
    }return out;
}
struct Fixture {
    std::shared_ptr<Owner> owner=std::make_shared<Owner>();
    unsigned count,H,I,E=16U,K;
    a8::Request request;
    std::vector<a8::Bindings> bindings;
    q51_known_rows::Workspace workspace;
    a8::DeviceBytes gate,up,csr;
    std::vector<std::byte> host_gate,host_up,csr_seed;
    std::vector<unsigned> routes;
    std::vector<std::vector<float>> values;
    Fixture(unsigned rows,unsigned hidden,unsigned outputs,unsigned top_k,bool bf16,unsigned pattern,unsigned special=0U):
        count(rows),H(hidden),I(outputs),K(top_k){
        require(owner->guard.architecture()=="gfx1151","A8 CSR fixture requires actual gfx1151");
        request={true,a8::numerical_identity,bf16?a8::InputKind::bf16:a8::InputKind::fp32,{H,I,E,K,1U}};
        host_gate=weights(H,I,E,3U,special==1U?0x7e11U:0U);host_up=weights(H,I,E,7U,special==2U?0x7c00U:0U);
        gate=owner->allocate(host_gate.size());up=owner->allocate(host_up.size());upload(gate,host_gate.data(),host_gate.size());upload(up,host_up.data(),host_up.size());
        csr=owner->allocate(q51_known_rows::required_scratch_bytes(count,E,K));workspace={{91U,csr.address,csr.bytes,97U},owner};
        bindings.resize(count);values.resize(count);routes.resize(count*K);
        for(unsigned r=0;r<count;++r){auto& b=bindings[r];values[r].resize(H);std::vector<std::uint16_t> bf(H);
            for(unsigned f=0;f<H;++f){float value=static_cast<float>(static_cast<int>((f*31U+r*43U)%251U)-125)/64.0F;
                auto bits=std::bit_cast<unsigned>(value);bits+=0x7fffU+((bits>>16U)&1U);bf[f]=static_cast<std::uint16_t>(bits>>16U);
                values[r][f]=bf16?std::bit_cast<float>(static_cast<unsigned>(bf[f])<<16U):value;}
            std::vector<unsigned> ids(K);for(unsigned s=0;s<K;++s){ids[s]=pattern==0U?s:(pattern==1U?(s+r*3U)%E:(s/3U+r)%4U);routes[r*K+s]=ids[s];}
            b.input=owner->allocate(H*(bf16?2U:4U));b.gate=gate;b.up=up;b.routes=owner->allocate(K*4U);
            b.codes=owner->allocate(H);b.deltas=owner->allocate(H/32U*4U);b.sums=owner->allocate(H/32U*4U);
            b.status=owner->allocate(4U);b.output=owner->allocate(static_cast<std::size_t>(K)*2U*I*4U);
            upload(b.input,bf16?static_cast<const void*>(bf.data()):values[r].data(),static_cast<std::size_t>(b.input.bytes));upload(b.routes,ids.data(),ids.size()*4U);
        }
        // Existing planner in assignment-row form admits repeated original
        // row/expert slots while preserving its stable expert CSR definition.
        const auto plan=build_moe_route_plan({count*K,1U,E,0U,routes});const auto l=q51_known_rows::scratch_layout(count,E,K);
        csr_seed.assign(static_cast<std::size_t>(csr.bytes),std::byte{0xa5});
        const auto word=[&](std::uint64_t offset,unsigned value){std::memcpy(csr_seed.data()+offset,&value,4U);};
        for(unsigned n=0;n<32U;++n)word(n*4U,0U);word(4U,plan.distinct_experts);word(12U,count*K);word(24U,count);word(28U,1U);
        for(unsigned e=0;e<=E;++e)word(l.offsets+e*4U,static_cast<unsigned>(plan.expert_offsets[e]));
        for(unsigned t=0;t<plan.touched_experts.size();++t)word(l.touched+t*4U,plan.touched_experts[t]);
        for(unsigned a=0;a<count*K;++a){word(l.rows+a*4U,plan.assignment_rows[a]/K);word(l.slots+a*4U,plan.assignment_rows[a]%K);}
        unsigned batches=0U;for(unsigned e=0;e<E;++e)for(auto begin=plan.expert_offsets[e];begin<plan.expert_offsets[e+1U];begin+=count){
            word(l.batch_experts+batches*4U,e);word(l.batch_begins+batches*4U,static_cast<unsigned>(begin));++batches;}
        word(l.batch_begins+batches*4U,count*K);word(8U,batches);
    }
    using Snapshot=std::vector<std::array<std::vector<std::byte>,5>>;
    void reset(unsigned status=0U){upload(csr,csr_seed.data(),csr_seed.size());for(const auto& b:bindings){
        for(const auto v:{b.codes,b.deltas,b.sums,b.output})poison(v);upload(b.status,&status,4U);}}
    Snapshot snapshot() const{Snapshot out(count);for(unsigned r=0;r<count;++r){const auto& b=bindings[r];
        const std::array v{b.codes,b.deltas,b.sums,b.status,b.output};for(unsigned i=0;i<5U;++i)out[r][i]=read(v[i]);}return out;}
    Snapshot run(bool use_candidate){reset();
        if(use_candidate){for(const auto& b:bindings)a8::enqueue_quantize_only(request,b,owner->stream);
            candidate::enqueue(request,bindings,workspace,owner->stream);}
        else for(const auto& b:bindings)a8::enqueue_probe(request,b,owner->stream);
        owner->drain();require(read(csr)==csr_seed,"valid A8 projection changed CSR metadata");return snapshot();}
    void oracle(const Snapshot& actual,bool special) const{
        for(unsigned r=0;r<count;++r){const auto quant=q4k_a8_oracle::quantize(values[r]);
            require(actual[r][0].size()==quant.codes.size(),"A8 oracle code extent differs");
            require(std::memcmp(actual[r][0].data(),quant.codes.data(),quant.codes.size())==0,"A8 CSR codes differ from independent quantizer");
            require(std::memcmp(actual[r][1].data(),quant.deltas.data(),quant.deltas.size()*4U)==0&&
                    std::memcmp(actual[r][2].data(),quant.sums.data(),quant.sums.size()*4U)==0,"A8 CSR group metadata differs from independent quantizer");
            require(std::all_of(actual[r][3].begin(),actual[r][3].end(),[](auto x){return x==std::byte{0};}),"A8 CSR finite input status changed");
            if(!special)for(unsigned slot=0;slot<K;++slot)for(unsigned which=0;which<2U;++which)for(unsigned out=0;out<I;++out){
                const auto bytes=static_cast<std::size_t>(H/256U)*144U;
                const auto& packed=which?host_up:host_gate;const auto offset=(static_cast<std::size_t>(routes[r*K+slot])*I+out)*bytes;
                const auto expected=q4k_a8_oracle::row(std::span<const std::byte>(packed).subspan(offset,bytes),quant);
                unsigned bits=0U;std::memcpy(&bits,actual[r][4].data()+((slot*2U+which)*I+out)*4U,4U);
                require(bits==std::bit_cast<unsigned>(expected),"A8 CSR every gate/up output differs from independent affine pair-tree oracle");
            }
        }
    }
    void malformed(unsigned fault){reset();const auto l=q51_known_rows::scratch_layout(count,E,K);auto bad=csr_seed;
        const auto word=[&](std::uint64_t offset,unsigned value){std::memcpy(bad.data()+offset,&value,4U);};
        if(fault==0U)word(8U,count*K+1U);if(fault==1U)word(l.rows,count);if(fault==2U)word(l.slots,K);
        if(fault==3U)word(l.batch_begins+4U,count*K+1U);if(fault==4U)word(l.batch_experts,E);
        if(fault==5U){unsigned row=0,slot=0;std::memcpy(&row,bad.data()+l.rows,4U);std::memcpy(&slot,bad.data()+l.slots,4U);word(l.rows+4U,row);word(l.slots+4U,slot);}
        upload(csr,bad.data(),bad.size());for(const auto& b:bindings)a8::enqueue_quantize_only(request,b,owner->stream);
        candidate::enqueue(request,bindings,workspace,owner->stream);owner->drain();
        for(const auto& b:bindings){const auto output=read(b.output);require(std::all_of(output.begin(),output.end(),[](auto x){return x==std::byte{0xa5};}),"malformed CSR published projection output");
            unsigned status=0U;const auto bytes=read(b.status);std::memcpy(&status,bytes.data(),4U);require(status==1U,"malformed CSR did not fail closed for all rows");}
    }
    void host_rejections(){reset();auto bad=bindings;bad.back().output=bad.front().output;
        const auto before=snapshot();const auto metadata=read(csr);bool rejected=false;
        try{candidate::enqueue(request,bad,workspace,owner->stream);}catch(const std::invalid_argument& e){rejected=std::string(e.what())=="A8 CSR gate/up mutable row owners alias";}
        require(rejected&&snapshot()==before&&read(csr)==metadata,"A8 CSR cross-row alias rejection changed payload");
        check(hipStreamBeginCapture(owner->stream,hipStreamCaptureModeThreadLocal),"begin A8 CSR capture rejection");rejected=false;
        std::exception_ptr unexpected;
        try{candidate::enqueue(request,bindings,workspace,owner->stream);}catch(const std::invalid_argument& e){rejected=std::string(e.what())=="A8 CSR gate/up admits explicit eager streams only";}
        catch(...){unexpected=std::current_exception();}
        hipGraph_t graph=nullptr;check(hipStreamEndCapture(owner->stream,&graph),"end A8 CSR capture rejection");std::size_t nodes=0U;
        check(hipGraphGetNodes(graph,nullptr,&nodes),"count A8 CSR rejected capture nodes");check(hipGraphDestroy(graph),"destroy A8 CSR empty graph");
        if(unexpected)std::rethrow_exception(unexpected);
        require(rejected&&nodes==0U&&snapshot()==before&&read(csr)==metadata,"A8 CSR capture rejection enqueued or mutated payload");
    }
    void sticky(){reset(0x40U);const auto before=snapshot();
        for(const auto& b:bindings)a8::enqueue_quantize_only(request,b,owner->stream);
        candidate::enqueue(request,bindings,workspace,owner->stream);owner->drain();
        require(snapshot()==before,"A8 CSR sticky failure changed quantized image/status/output");
        const auto actual=read(csr);auto expected=csr_seed;const unsigned status=0x40U;std::memcpy(expected.data(),&status,4U);
        require(actual==expected,"A8 CSR sticky failure changed unrelated CSR metadata");
    }
    void retained_checks()const{guards(gate);guards(up);guards(csr);require(read(gate)==host_gate&&read(up)==host_up,"A8 CSR changed packed weights");
        for(const auto& b:bindings)for(const auto v:{b.input,b.routes,b.codes,b.deltas,b.sums,b.status,b.output})guards(v);}
};
void run(unsigned rows,unsigned hidden,unsigned outputs,unsigned top_k,bool bf16,unsigned pattern,unsigned special=0U){
    Fixture f(rows,hidden,outputs,top_k,bf16,pattern,special);const auto expected=f.run(false);f.oracle(expected,special!=0U);
    if(!special){require(expected.front()[4]!=expected.back()[4],"A8 CSR fixture does not distinguish source rows");
        bool distinct=false;for(unsigned slot=0;slot<top_k;++slot)for(unsigned out=0;out<outputs;++out){
            unsigned gate=0U,up=0U;std::memcpy(&gate,expected.front()[4].data()+(slot*2U*outputs+out)*4U,4U);
            std::memcpy(&up,expected.front()[4].data()+((slot*2U+1U)*outputs+out)*4U,4U);distinct|=gate!=up;}
        require(distinct,"A8 CSR fixture does not distinguish gate/up coefficients");}
    for(unsigned repeat=0;repeat<3U;++repeat){const auto actual=f.run(true);f.oracle(actual,special!=0U);
        for(unsigned r=0;r<rows;++r){for(unsigned v=0;v<4U;++v)require(actual[r][v]==expected[r][v],"A8 CSR complete quantization image/status differs from incumbent");
            if(!special)require(actual[r][4]==expected[r][4],"A8 CSR complete gate/up output differs bitwise from incumbent");
            else for(std::size_t i=0;i<actual[r][4].size();i+=4U){unsigned a=0,e=0;std::memcpy(&a,actual[r][4].data()+i,4U);std::memcpy(&e,expected[r][4].data()+i,4U);
                const bool an=(a&0x7fffffffU)>0x7f800000U,en=(e&0x7fffffffU)>0x7f800000U;
                require((an&&en)||a==e,"A8 CSR nonfinite class or unaffected projection branch differs");}
        }}
    if(!special){for(unsigned fault=0;fault<6U;++fault)f.malformed(fault);f.host_rejections();f.sticky();}
    f.retained_checks();
}
} // namespace
int main(int argc,char** argv)try{
    const unsigned rows=argc>1?static_cast<unsigned>(std::stoul(argv[1])):2U;
    require(rows==2U||rows==4U||rows==8U,"A8 CSR fixture rows must be 2/4/8");
    require(candidate::staged_activation_bytes(rows)==static_cast<std::uint64_t>(rows)*3200U,"A8 CSR source staging footprint differs");
    for(bool bf16:{false,true})for(unsigned pattern=0;pattern<3U;++pattern){run(rows,256U,32U,2U,bf16,pattern);run(rows,512U,64U,10U,bf16,pattern);}
    run(rows,2560U,640U,10U,true,1U);
    run(rows,256U,32U,2U,true,1U,1U);run(rows,256U,32U,2U,true,1U,2U);
    std::cout<<"A8 CSR known rows fixture completed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
