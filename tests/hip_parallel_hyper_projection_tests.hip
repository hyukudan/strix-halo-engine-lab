#include "strix/hip_lane_chain_hyper_projection_tests.hpp"
#include "strix/hc_parallel_projection_oracle.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_content_hash.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

namespace {
namespace api = strix::hc_projection_diagnostic;
namespace oracle = strix::hc_parallel_projection_oracle;
using api::Shape; using api::Span; using api::SplitKVariant;
void require(bool yes,const std::string& text){if(!yes)throw std::runtime_error(text);}
void check(hipError_t status,const char* text){if(status!=hipSuccess)throw std::runtime_error(std::string(text)+": "+hipGetErrorString(status));}
std::string digest(std::span<const std::byte> bytes){return strix::model_content_identity_hex(strix::sha256_bytes(bytes));}
template<class T> std::span<const std::byte> bytes(const std::vector<T>& v){return std::as_bytes(std::span(v));}
struct Owner{
    void* allocation=nullptr;void* p=nullptr;std::size_t n=0;
    explicit Owner(std::size_t size):n(size){if(n){check(hipMalloc(&allocation,n+64U),"allocate guarded owner");p=static_cast<std::byte*>(allocation)+32U;check(hipMemset(allocation,0xa5,n+64U),"guard owner");}}
    ~Owner(){if(allocation)static_cast<void>(hipFree(allocation));} Owner(const Owner&)=delete;
    Span span()const{return {reinterpret_cast<std::uintptr_t>(p),n};}
    std::vector<std::byte> read()const{std::vector<std::byte> v(n);if(n)check(hipMemcpy(v.data(),p,n,hipMemcpyDeviceToHost),"read owner");return v;}
    void copy(std::span<const std::byte> v){require(v.size()<=n,"copy exceeds owner");if(!v.empty())check(hipMemcpy(p,v.data(),v.size(),hipMemcpyHostToDevice),"copy owner");}
    void bounds()const{if(n)for(auto edge:{static_cast<const std::byte*>(p)-32U,static_cast<const std::byte*>(p)+n}){std::array<unsigned char,32> v{};check(hipMemcpy(v.data(),edge,32,hipMemcpyDeviceToHost),"read guard");require(std::all_of(v.begin(),v.end(),[](auto x){return x==0xa5;}),"outer owner guard changed");}}
};
struct Stream{hipStream_t value=nullptr;Stream(){check(hipStreamCreateWithFlags(&value,hipStreamNonBlocking),"create stream");}~Stream(){if(value){static_cast<void>(hipStreamSynchronize(value));static_cast<void>(hipStreamDestroy(value));}}};
struct Graph{hipGraph_t graph=nullptr;hipGraphExec_t executable=nullptr;~Graph(){if(executable)static_cast<void>(hipGraphExecDestroy(executable));if(graph)static_cast<void>(hipGraphDestroy(graph));}};
struct Events{hipEvent_t start=nullptr,stop=nullptr;Events(){check(hipEventCreate(&start),"create event");check(hipEventCreate(&stop),"create event");}~Events(){if(start)static_cast<void>(hipEventDestroy(start));if(stop)static_cast<void>(hipEventDestroy(stop));}};
std::vector<float> floats(const Owner& owner,std::size_t count){auto raw=owner.read();require(count*4U<=raw.size(),"float read exceeds owner");std::vector<float> v(count);std::memcpy(v.data(),raw.data(),count*4U);return v;}
bool same(std::span<const float> a,std::span<const float> b){return a.size()==b.size()&&std::equal(std::as_bytes(a).begin(),std::as_bytes(a).end(),std::as_bytes(b).begin());}
constexpr std::array<SplitKVariant,4> variants{SplitKVariant::p1_r4,SplitKVariant::p1_r2,SplitKVariant::p2_r4,SplitKVariant::p1_r4_fp64};
double gamma(unsigned depth,double u){return depth*u/(1.0-depth*u);}
constexpr double u=0x1p-24;
struct NumericalSummary{double max_dot_bound_fraction=0,max_candidate_bound_fraction=0,max_epilogue_c_fraction=0,max_serial_absolute=0,max_candidate_absolute=0;std::uint64_t candidate_serial_bit_differences=0,finite_outputs=0,nonfinite_inputs_outputs=0;};

struct Fixture{
    Shape s;std::vector<float> input,inject;std::vector<std::byte> weights;
    Owner normalized,down,injection_weights,base_low,base_inj,low,inj,partial;
    std::array<std::string,3> readonly_hashes;
    Fixture(Shape shape,std::vector<float> n,std::vector<std::byte> w,std::vector<float> i):s(shape),input(std::move(n)),inject(std::move(i)),weights(std::move(w)),normalized(input.size()*4U),down(weights.size()),injection_weights(inject.size()*4U),base_low(static_cast<std::size_t>(s.rows)*s.rank*4U+32U),base_inj(inject.empty()?0U:static_cast<std::size_t>(s.rows)*s.streams*4U+32U),low(base_low.n),inj(base_inj.n),partial(static_cast<std::size_t>(s.rows)*16U*(s.rank+(inject.empty()?0U:s.streams))*4U+32U){upload();}
    void upload(){normalized.copy(bytes(input));down.copy(weights);injection_weights.copy(bytes(inject));readonly_hashes={digest(bytes(input)),digest(weights),digest(bytes(inject))};}
    void poison(bool candidate,hipStream_t stream=nullptr){for(auto owner:{candidate?&low:&base_low,candidate?&inj:&base_inj,candidate?&partial:nullptr})if(owner&&owner->n)check(hipMemsetAsync(owner->p,candidate?0x5a:0xa5,owner->n,stream),"poison output");}
    void reference(hipStream_t stream=nullptr){api::enqueue(s,normalized.span(),down.span(),injection_weights.span(),base_low.span(),base_inj.span(),0,1,true,stream);}
    void launch(SplitKVariant v,hipStream_t stream=nullptr){api::enqueue_splitk(s,normalized.span(),down.span(),injection_weights.span(),low.span(),inj.span(),v==SplitKVariant::p2_r4?partial.span():Span{},v,stream);}
    std::vector<float> semantic(bool candidate)const{auto v=floats(candidate?low:base_low,static_cast<std::size_t>(s.rows)*s.rank);if(!inject.empty()){auto tail=floats(candidate?inj:base_inj,static_cast<std::size_t>(s.rows)*s.streams);v.insert(v.end(),tail.begin(),tail.end());}return v;}
    void guards(bool candidate)const{for(auto owner:{candidate?&low:&base_low,candidate?&inj:&base_inj,candidate?&partial:nullptr})if(owner&&owner->n){auto v=owner->read();auto pattern=candidate?std::byte{0x5a}:std::byte{0xa5};require(std::all_of(v.end()-32,v.end(),[&](auto x){return x==pattern;}),"semantic guard changed");owner->bounds();}}
    void readonly()const{require(digest(normalized.read())==readonly_hashes[0]&&digest(down.read())==readonly_hashes[1]&&digest(injection_weights.read())==readonly_hashes[2],"readonly input changed");for(auto owner:{&normalized,&down,&injection_weights,&low,&inj,&base_low,&base_inj,&partial})owner->bounds();}
    oracle::Result expected()const{return oracle::evaluate({s.width,s.streams,s.rank,s.rows},input,weights,inject);}
    std::vector<oracle::Output> all_expected()const{auto r=expected();r.down.insert(r.down.end(),r.injection.begin(),r.injection.end());return r.down;}
    void capture(Graph& graph,SplitKVariant variant,hipStream_t stream){check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"begin capture");try{launch(variant,stream);}catch(...){static_cast<void>(hipStreamEndCapture(stream,&graph.graph));throw;}check(hipStreamEndCapture(stream,&graph.graph),"end capture");check(hipGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0),"instantiate graph");
        std::size_t count=0;check(hipGraphGetNodes(graph.graph,nullptr,&count),"count graph nodes");const bool p2=variant==SplitKVariant::p2_r4;require(count==(p2?2U:1U),"wrong captured kernel count");std::vector<hipGraphNode_t> nodes(count);check(hipGraphGetNodes(graph.graph,nodes.data(),&count),"read graph nodes");bool partial_seen=false,final_seen=false;hipGraphNode_t partial_node=nullptr,final_node=nullptr;
        for(auto node:nodes){hipKernelNodeParams p{};check(hipGraphKernelNodeGetParams(node,&p),"read kernel geometry");if(p2){if(p.blockDim.x==256U&&p.gridDim.x==(s.rank+31U)/32U+(inject.empty()?0U:1U)&&p.gridDim.y==16U&&p.gridDim.z==s.rows){partial_seen=true;partial_node=node;}else if(p.blockDim.x==128U&&p.gridDim.x==(s.rank+(inject.empty()?0U:s.streams)+127U)/128U&&p.gridDim.y==s.rows&&p.gridDim.z==1U){final_seen=true;final_node=node;}else require(false,"wrong P2 geometry");}else{const unsigned r=variant==SplitKVariant::p1_r2?2U:4U;require(p.blockDim.x==512U&&p.gridDim.x==(s.rank+r-1U)/r+(inject.empty()?0U:(s.streams+r-1U)/r)&&p.gridDim.y==s.rows&&p.gridDim.z==1U,"wrong P1 geometry");}require(p.blockDim.y==1U&&p.blockDim.z==1U,"wrong block dimensions");}
        if(p2){require(partial_seen&&final_seen,"missing P2 graph phase");std::size_t edges=0;check(hipGraphGetEdges(graph.graph,nullptr,nullptr,&edges),"count graph edges");require(edges==1U,"P2 missing ordered dependency");hipGraphNode_t from=nullptr,to=nullptr;check(hipGraphGetEdges(graph.graph,&from,&to,&edges),"read P2 dependency edge");require(edges==1U&&from==partial_node&&to==final_node,"P2 graph dependency must be partial to final");}
    }
};

void check_class(float actual,double expected,bool inject){
    // Native down epilogue is z*sigmoid(z), so -inf gives NaN, not SiLU's limit.
    if(!inject&&expected==-std::numeric_limits<double>::infinity())require(std::isnan(actual),"down -inf epilogue class");
    else if(std::isnan(expected))require(std::isnan(actual),"NaN did not propagate");
    else if(std::isinf(expected))require(std::isinf(actual)&&std::signbit(actual)==std::signbit(expected),"infinity sign/class differs");
    else require(std::isfinite(actual),"finite oracle produced nonfinite output");
}

void validate(Fixture& f,const std::vector<float>& candidate,const std::vector<float>& baseline,SplitKVariant variant,NumericalSummary& summary){
    const auto expected=f.all_expected();require(candidate.size()==expected.size()&&(baseline.empty()||baseline.size()==expected.size()),"semantic output count");const auto low_count=static_cast<std::size_t>(f.s.rows)*f.s.rank;
    const auto partial_values=variant==SplitKVariant::p2_r4?floats(f.partial,static_cast<std::size_t>(f.s.rows)*16U*(f.s.rank+(f.inject.empty()?0U:f.s.streams))):std::vector<float>{};
    for(std::size_t index=0;index<expected.size();++index){const auto& e=expected[index];const bool injection=index>=low_count;const double actual=candidate[index];
        if(!std::isfinite(e.dot)||!std::isfinite(e.sum_abs)){++summary.nonfinite_inputs_outputs;if(!injection&&e.scaled==-std::numeric_limits<double>::infinity())require(std::isnan(actual),"negative infinite down must be NaN");else{check_class(candidate[index],e.postactivation,injection);if(injection&&std::isinf(e.scaled))require(actual==(e.scaled>0?2.0:0.0),"infinite injection sigmoid endpoint");}continue;}
        require(std::isfinite(actual)&&(baseline.empty()||std::isfinite(baseline[index])),"finite synthetic case overflowed");++summary.finite_outputs;
        const double bdot=(gamma(40,u)+gamma(10240,0x1p-53))*e.sum_abs+2.0*10240.0*0x1p-126;
        const double dz=bdot/f.s.streams+u*std::abs(e.scaled);
        const double bound=(injection?0.5:1.1)*dz+16.0*u*std::abs(e.postactivation)+0x1p-125;
        const double error=std::abs(actual-e.postactivation);if(bound>0)summary.max_candidate_bound_fraction=std::max(summary.max_candidate_bound_fraction,error/bound);
        require(error<=bound,"N2 fixed gamma40/postactivation bound failed at output "+std::to_string(index));
        const double base_bdot=(gamma(10240,u)+gamma(10240,0x1p-53))*e.sum_abs+2.0*10240.0*0x1p-126;
        const double base_dz=base_bdot/f.s.streams+u*std::abs(e.scaled);
        if(!baseline.empty())require(std::abs(static_cast<double>(baseline[index])-e.postactivation)<=(injection?0.5:1.1)*base_dz+16*u*std::abs(e.postactivation)+0x1p-125,"N4 actual baseline postactivation bound failed");
        if(variant==SplitKVariant::p1_r4_fp64){const float zf=static_cast<float>(e.dot)/static_cast<float>(f.s.streams);const double z=zf;const double ideal=injection?2.0*oracle::stable_sigmoid(z):oracle::stable_silu(z);const double cbound=16*u*std::abs(ideal)+0x1p-125;const double ce=std::abs(actual-ideal);if(cbound>0)summary.max_epilogue_c_fraction=std::max(summary.max_epilogue_c_fraction,ce/cbound);require(ce<=cbound,"N3 native epilogue16u bound failed");}
        if(variant==SplitKVariant::p2_r4){const auto role_count=f.s.rank+(f.inject.empty()?0U:f.s.streams);const auto row=injection?(index-low_count)/f.s.streams:index/f.s.rank;const auto out=injection?f.s.rank+(index-low_count)%f.s.streams:index%f.s.rank;float sum=0;for(unsigned k=0;k<16;++k)sum=sum+partial_values[(row*16U+k)*role_count+out];const double raw_error=std::abs(static_cast<double>(sum)-e.dot);summary.max_dot_bound_fraction=std::max(summary.max_dot_bound_fraction,raw_error/bdot);require(raw_error<=bdot,"N1 P2 actual partial fixedFP32sum bound failed");}
        if(!baseline.empty()){
            summary.max_serial_absolute=std::max(summary.max_serial_absolute,std::abs(static_cast<double>(baseline[index])-e.postactivation));
            summary.candidate_serial_bit_differences+=std::bit_cast<unsigned>(candidate[index])!=std::bit_cast<unsigned>(baseline[index]);
        }
        summary.max_candidate_absolute=std::max(summary.max_candidate_absolute,error);
    }
    f.guards(true);
}

void negative_cases(Fixture& f){
    f.poison(true);check(hipDeviceSynchronize(),"drain poison");const auto low=f.low.read(),inj=f.inj.read(),part=f.partial.read();
    const auto test=[&](Shape s,Span n,Span d,Span i,Span l,Span o,Span p,SplitKVariant v){
        Stream stream;Graph graph;bool rejected=false;
        check(hipStreamBeginCapture(stream.value,hipStreamCaptureModeThreadLocal),"begin rejected-call capture");
        try{api::enqueue_splitk(s,n,d,i,l,o,p,v,stream.value);}catch(const std::invalid_argument&){rejected=true;}
        catch(...){static_cast<void>(hipStreamEndCapture(stream.value,&graph.graph));throw;}
        check(hipStreamEndCapture(stream.value,&graph.graph),"end rejected-call capture");
        std::size_t nodes=0;check(hipGraphGetNodes(graph.graph,nullptr,&nodes),"count rejected-call nodes");
        require(rejected&&nodes==0U,"invalid splitK call accepted or queued a node");
        // Synchronous output reads are deliberately outside thread-local capture.
        require(f.low.read()==low&&f.inj.read()==inj&&f.partial.read()==part,"rejected call wrote output");
    };
    const auto call=[&](Span n,Span d,Span i,Span l,Span o,Span p,SplitKVariant v){test(f.s,n,d,i,l,o,p,v);};
    auto n=f.normalized.span();n.address+=4;call(n,f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),{},variants[0]);n=f.normalized.span();n.bytes-=1;call(n,f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),{},variants[0]);
    auto d=f.down.span();d.address+=2;call(f.normalized.span(),d,f.injection_weights.span(),f.low.span(),f.inj.span(),{},variants[0]);
    call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.normalized.span(),f.inj.span(),{},variants[0]);
    call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),f.partial.span(),variants[0]);
    call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),{},variants[2]);
    auto p=f.partial.span();p.address+=4;call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),p,variants[2]);
    p=f.partial.span();p.bytes=static_cast<std::uint64_t>(f.s.rows)*16U*(f.s.rank+(f.inject.empty()?0U:f.s.streams))*4U-1U;call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),p,variants[2]);
    call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),f.low.span(),variants[2]);
    call(f.normalized.span(),f.down.span(),{},f.low.span(),f.inj.span(),{},variants[0]);
    call(f.normalized.span(),f.down.span(),{0,4},f.low.span(),{}, {},variants[0]);
    call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),{},static_cast<SplitKVariant>(99));
    auto i=f.injection_weights.span();i.address+=4;call(f.normalized.span(),f.down.span(),i,f.low.span(),f.inj.span(),{},variants[0]);
    auto l=f.low.span();l.address+=1;call(f.normalized.span(),f.down.span(),f.injection_weights.span(),l,f.inj.span(),{},variants[0]);
    call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.low.span(),{},variants[0]);
    call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),f.normalized.span(),variants[2]);
    auto injection_output=f.inj.span();injection_output.address+=1;call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),injection_output,{},variants[0]);
    for(auto variant:{SplitKVariant::p1_r2,SplitKVariant::p1_r4_fp64})call(f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),f.partial.span(),variant);
    auto rows_outside=f.s;rows_outside.rows=65536U;test(rows_outside,f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),{},variants[0]);
    for(auto s:std::array<Shape,5>{{{10208,4,4,1},{10240,1,4,1},{10240,9,4,1},{10240,4,513,1},{10240,4,4,0}}})test(s,f.normalized.span(),f.down.span(),f.injection_weights.span(),f.low.span(),f.inj.span(),{},variants[0]);
    f.readonly();
}

std::unique_ptr<Fixture> synthetic(Shape s,unsigned flavor,bool with_injection=true){
    std::vector<float> input(static_cast<std::size_t>(s.rows)*s.width),inject(with_injection?static_cast<std::size_t>(s.streams)*s.width:0U);std::vector<std::byte> down(static_cast<std::size_t>(s.rank)*(s.width/32U)*34U);
    for(std::size_t k=0;k<input.size();++k)input[k]=std::bit_cast<float>(0x3e000001U+static_cast<unsigned>((k*7919U)%0x007ffffeU))*(k%3?1.0F:-1.0F);
    for(std::size_t k=0;k<inject.size();++k)inject[k]=static_cast<float>(static_cast<int>(k%23U)-11)*0.0001220703125F;
    for(std::size_t b=0;b<down.size()/34U;++b){const unsigned scale=0x2401U+static_cast<unsigned>(b%255U);down[b*34U]=static_cast<std::byte>(scale&255U);down[b*34U+1U]=static_cast<std::byte>(scale>>8U);for(unsigned k=0;k<32;++k)down[b*34U+2U+k]=static_cast<std::byte>((b*17U+k*59U)&255U);}
    if(flavor==1||flavor==2){std::fill(input.begin(),input.end(),flavor==1?0.0F:-0.0F);}
    if(flavor==3){for(std::size_t k=0;k<input.size();++k)input[k]=std::bit_cast<float>(static_cast<unsigned>(k%1023U+1U)|(k%2?0x80000000U:0U));}
    if(flavor==4){std::fill(input.begin(),input.end(),0.0F);for(unsigned row=0;row<s.rows;++row){auto base=static_cast<std::size_t>(row)*s.width;input[base]=16777216.0F;input[base+1U]=1;input[base+2U]=-16777216.0F;}for(std::size_t b=0;b<down.size()/34U;++b){down[b*34U]=std::byte{0};down[b*34U+1U]=std::byte{0x3c};for(unsigned k=0;k<32;++k)down[b*34U+2U+k]=std::byte{1};}std::fill(inject.begin(),inject.end(),1.0F);}
    if(flavor==5)input[0]=std::bit_cast<float>(0x7fc52000U);
    if(flavor==6||flavor==7){input[0]=flavor==6?std::numeric_limits<float>::infinity():-std::numeric_limits<float>::infinity();for(unsigned out=0;out<s.rank;++out)down[static_cast<std::size_t>(out)*(s.width/32U)*34U+2U]=std::byte{1};for(unsigned out=0;out<s.streams&&with_injection;++out)inject[static_cast<std::size_t>(out)*s.width]=1.0F;}
    if(flavor==8||flavor==9){for(unsigned out=0;out<s.rank;++out){const auto at=static_cast<std::size_t>(out)*(s.width/32U)*34U;down[at]=flavor==8?std::byte{0x11}:std::byte{0};down[at+1U]=flavor==8?std::byte{0x7e}:std::byte{0x7c};down[at+2U]=std::byte{0};}}
    return std::make_unique<Fixture>(s,std::move(input),std::move(down),std::move(inject));
}

void native_case(Fixture& f,NumericalSummary& summary,bool negatives){
    std::vector<float> baseline;const bool baseline_admitted=f.s.width%f.s.streams==0U;
    if(baseline_admitted){f.poison(false);f.reference();check(hipDeviceSynchronize(),"drain baseline");baseline=f.semantic(false);f.guards(false);
    for(auto owner:{&f.base_low,&f.base_inj}){if(owner->n)check(hipMemset(owner->p,0x3c,owner->n-32U),"second semantic poison");}
    f.reference();check(hipDeviceSynchronize(),"drain write witness");require(same(baseline,f.semantic(false)),"reference missing semantic writes");}
    std::array<std::vector<float>,4> own;
    for(unsigned mode=0;mode<4;++mode){f.poison(true);f.launch(variants[mode]);check(hipDeviceSynchronize(),"drain parallel candidate");own[mode]=f.semantic(true);validate(f,own[mode],baseline,variants[mode],summary);}
    if(std::all_of(f.input.begin(),f.input.end(),[](float x){return x==0.0F;})){const auto low_count=static_cast<std::size_t>(f.s.rows)*f.s.rank;for(const auto& mode:own)for(std::size_t k=0;k<mode.size();++k)require(k<low_count?std::bit_cast<unsigned>(mode[k])==0U:mode[k]==1.0F,"zero dot fixed output bits");}
    const auto expected=f.all_expected();const bool all_finite=std::all_of(expected.begin(),expected.end(),[](const auto& e){return std::isfinite(e.dot)&&std::isfinite(e.sum_abs);});
    if(all_finite){require(same(own[0],own[2])&&same(own[1],own[2]),"N5 P1R4/P1R2/P2 finite bit mismatch");
        for(unsigned mode=0;mode<2;++mode)for(unsigned repeat=0;repeat<100;++repeat){f.poison(true);f.launch(variants[mode]);check(hipDeviceSynchronize(),"drain deterministic repeat");require(same(own[mode],f.semantic(true)),"N5 100selfrepeat failed");}
        // Raw ascending FP32 CPU chain sanity, explicitly not a device raw-dot tap.
        for(unsigned row=0;row<f.s.rows;++row)for(unsigned out=0;out<f.s.rank;++out){float sum=0;for(unsigned k=0;k<f.s.width;++k)sum=std::fma(f.input[static_cast<std::size_t>(row)*f.s.width+k],oracle::q8_coefficient(f.weights,f.s.width,out,k),sum);const auto& e=expected[static_cast<std::size_t>(row)*f.s.rank+out];require(std::abs(static_cast<double>(sum)-e.dot)<=(gamma(10240,u)+gamma(10240,0x1p-53))*e.sum_abs+2*10240*0x1p-126,"N4 CPU serial raw sanity failed");}
    }
    Stream stream;
    for(unsigned mode=0;mode<4;++mode){Graph graph;f.capture(graph,variants[mode],stream.value);for(unsigned replay=0;replay<3;++replay){f.poison(true,stream.value);check(hipGraphLaunch(graph.executable,stream.value),"replay graph");check(hipStreamSynchronize(stream.value),"drain graph");require(same(own[mode],f.semantic(true)),"N5 eager/graph selfbits differ");f.guards(true);}}
    // Update the same normalized allocation after capture, then compare fresh eager
    // and replay outputs. Device graph arguments must observe current owner bytes.
    if(all_finite){Graph graph;f.capture(graph,variants[0],stream.value);const float saved=f.input.back();for(unsigned replay=0;replay<3;++replay){f.input.back()=saved+static_cast<float>(replay+1U)*0.125F;f.upload();if(baseline_admitted){f.poison(false);f.reference();}f.poison(true);f.launch(variants[0]);check(hipDeviceSynchronize(),"drain updated eager");auto actual=f.semantic(true);validate(f,actual,baseline_admitted?f.semantic(false):std::vector<float>{},variants[0],summary);f.poison(true,stream.value);check(hipGraphLaunch(graph.executable,stream.value),"replay updated graph");check(hipStreamSynchronize(stream.value),"drain updated replay");require(same(actual,f.semantic(true)),"graph retained stale input");f.guards(true);}f.input.back()=saved;f.upload();}
    f.readonly();if(negatives)negative_cases(f);
}

void native(){
    int device=0;check(hipGetDevice(&device),"read device");hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,device),"read architecture");require(properties.warpSize==32&&std::string(properties.gcnArchName).starts_with("gfx1151"),"expected actual gfx1151 wave32 before allocating fixtures");
    NumericalSummary summary;unsigned cases=0;
    for(auto shape:std::array<Shape,5>{{{10240,4,320,1},{10240,3,318,1},{10240,2,316,2},{10240,8,4,43},{10240,3,1,2}}}){auto f=synthetic(shape,0);native_case(*f,summary,cases==0);++cases;}
    for(unsigned flavor=1;flavor<=9;++flavor){auto f=synthetic({10240,3,4,1},flavor);native_case(*f,summary,false);++cases;}
    auto no_inject=synthetic({10240,2,4,2},0,false);native_case(*no_inject,summary,false);++cases;
    std::cout<<"HC splitK diagnostic native cases="<<cases<<" variants4 P1/P2 finitebits+100selfrepeat+eagergraph3 PASS; finite_outputs="<<summary.finite_outputs<<" classified_nonfinite_dot_outputs="<<summary.nonfinite_inputs_outputs<<" max_N2_fraction="<<summary.max_candidate_bound_fraction<<" max_N3_fraction="<<summary.max_epilogue_c_fraction<<" max_serial_abs="<<summary.max_serial_absolute<<" max_candidate_abs="<<summary.max_candidate_absolute<<" serial_bit_differences="<<summary.candidate_serial_bit_differences<<"; N4 raw CPU sanity only/device post bound; no model or M97 qualification\n";
}

std::string quote(const std::string& s){std::string v="\"";for(unsigned char c:s){require(c>=32U,"control character in JSON string");if(c=='"'||c=='\\')v+='\\';v+=static_cast<char>(c);}return v+'"';}
template<class T> void array(std::ostream& out,const std::vector<T>& a){out<<'[';for(std::size_t j=0;j<a.size();++j)out<<(j?",":"")<<a[j];out<<']';}
double median(std::vector<double> v){require(!v.empty(),"empty timing array");std::sort(v.begin(),v.end());return (v[(v.size()-1U)/2U]+v[v.size()/2U])*0.5;}
std::vector<std::byte> file_bytes(const std::filesystem::path& path){std::ifstream in(path,std::ios::binary);require(in.good(),"open binary "+path.string());std::vector<char> raw((std::istreambuf_iterator<char>(in)),{});require(in.eof()||in.good(),"binary read failed");std::vector<std::byte> result(raw.size());std::memcpy(result.data(),raw.data(),raw.size());return result;}
struct TensorBytes{std::vector<std::byte> data;std::uint64_t offset=0;};
TensorBytes tensor(const std::filesystem::path& path,const strix::gguf::FileInfo& info,const std::string& name,unsigned type,std::vector<std::uint64_t> dimensions){
    const strix::gguf::TensorInfo* found=nullptr;for(const auto& t:info.tensors)if(t.name==name){require(found==nullptr,"duplicate GGUF tensor");found=&t;}require(found&&found->ggml_type==type&&found->dimensions==dimensions,"GGUF tensor type/shape mismatch");
    TensorBytes value;value.offset=found->absolute_offset;value.data.resize(static_cast<std::size_t>(found->byte_size));std::ifstream f(path,std::ios::binary);require(f.good(),"open model tensor");f.seekg(static_cast<std::streamoff>(value.offset));f.read(reinterpret_cast<char*>(value.data.data()),static_cast<std::streamsize>(value.data.size()));require(f.good(),"truncated tensor window");return value;
}
struct Library{std::string path,sha;std::uint64_t size;bool operator==(const Library&)const=default;};
std::vector<Library> libraries(){std::ifstream maps("/proc/self/maps");require(maps.good(),"read live mappings");std::string line;std::vector<std::string> paths;while(std::getline(maps,line)){const auto at=line.find('/');if(at==std::string::npos)continue;const auto path=line.substr(at);if(path.find("libamdhip64.so")!=std::string::npos||path.find("libhsa-runtime64.so")!=std::string::npos){require(path.find(" (deleted)")==std::string::npos,"deleted runtime mapping");const auto canonical=std::filesystem::canonical(path).string();require(canonical.starts_with("/opt/rocm/core-10.1/"),"wrong Core10 live runtime");if(std::find(paths.begin(),paths.end(),canonical)==paths.end())paths.push_back(canonical);}}std::sort(paths.begin(),paths.end());require(paths.size()==2U,"expected live HIP/HSA mappings");std::vector<Library> result;for(const auto& path:paths){auto b=file_bytes(path);result.push_back({path,digest(b),b.size()});}return result;}
struct Options{std::filesystem::path model,normalized,output;std::string down="blk.0.hc_attn_down.weight",inject="blk.0.hc_attn_inject.weight";};
Options parse(int argc,char** argv){Options o;for(int at=2;at<argc;at+=2){require(at+1<argc,"missing option value");const std::string k=argv[at],v=argv[at+1];if(k=="--model")o.model=v;else if(k=="--normalized")o.normalized=v;else if(k=="--output")o.output=v;else if(k=="--down-tensor")o.down=v;else if(k=="--inject-tensor")o.inject=v;else if(k=="--width")require(v=="10240","benchmark width fixed10240");else if(k=="--streams")require(v=="4","benchmark streams fixed4");else if(k=="--rank")require(v=="320","benchmark rank fixed320");else if(k=="--pairs")require(v=="200","benchmark pairs fixed200");else if(k=="--iterations")require(v=="16","benchmark iterations fixed16");else if(k=="--cold-copies")require(v=="64","benchmark rotation fixed64");else if(k=="--warmup")require(v=="5","benchmark warmup fixed5");else throw std::invalid_argument("unknown benchmark option "+k);}require(!o.model.empty()&&!o.normalized.empty()&&!o.output.empty()&&!std::filesystem::exists(o.output),"need model/normalized/new output");return o;}
void capture_reference(Fixture& f,Graph& graph,hipStream_t stream){check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"begin reference capture");try{f.reference(stream);}catch(...){static_cast<void>(hipStreamEndCapture(stream,&graph.graph));throw;}check(hipStreamEndCapture(stream,&graph.graph),"end reference capture");check(hipGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0),"instantiate reference graph");std::size_t n=0;check(hipGraphGetNodes(graph.graph,nullptr,&n),"reference node count");require(n==1U,"reference graph node count changed");}

void bench(const Options& options){
    int device=0;check(hipGetDevice(&device),"read benchmark device");hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,device),"read benchmark device properties");require(properties.warpSize==32&&std::string(properties.gcnArchName).starts_with("gfx1151"),"expected actual gfx1151 wave32");
    const auto initial_libraries=libraries();const auto exe=std::filesystem::canonical("/proc/self/exe");const auto exe_sha=digest(file_bytes(exe));const auto info=strix::gguf::read_file(options.model);const auto down=tensor(options.model,info,options.down,8,{10240,320});const auto inject=tensor(options.model,info,options.inject,0,{10240,4});const auto normalized=file_bytes(options.normalized);require(normalized.size()==40960U&&inject.data.size()==163840U&&down.data.size()==3481600U,"canonical actual payload extents differ");std::vector<float> input(10240),injection(40960);std::memcpy(input.data(),normalized.data(),normalized.size());std::memcpy(injection.data(),inject.data.data(),inject.data.size());require(std::all_of(input.begin(),input.end(),[](float x){return std::isfinite(x);}),"captured input nonfinite");
    std::vector<std::unique_ptr<Fixture>> owners;for(unsigned copy=0;copy<64;++copy)owners.push_back(std::make_unique<Fixture>(Shape{10240,4,320,1},input,down.data,injection));
    NumericalSummary prequalification;native_case(*owners.front(),prequalification,false);
    require(!std::filesystem::exists(options.output)&&!std::filesystem::is_symlink(options.output),"benchmark output already exists or is symlink");
    Stream stream;Events events;std::ofstream out(options.output,std::ios::out);require(out.good(),"open benchmark receipt");out<<std::setprecision(17)<<"{\"eager_timing_includes_candidate_device_queries\":true,\"host_query_asymmetry\":\"enqueue_splitk eager performs hipGetDevice and hipGetDeviceProperties each call; reference eager has no equivalent queries. Graph replay excludes those enqueue queries after capture.\",\"numerical_identity\":\"hc-down-splitk16x640-r40-fp32-v1\",\"prequalification_N1_max_fraction\":"<<prequalification.max_dot_bound_fraction<<",\"prequalification_N2_max_fraction\":"<<prequalification.max_candidate_bound_fraction<<",\"prequalification_N3_max_fraction\":"<<prequalification.max_epilogue_c_fraction<<",\"prequalification_eager_graph_selfrepeat_passed\":true,\"schema\":\"private-parallel-hc-partial-micro-v1\",\"scope\":\"One realinput repeated, 3FP32 variants x eager/graph x hot/rot64; NOT actual97-call graph and NOT M gate or model qualification\",\"architecture\":"<<quote(properties.gcnArchName)<<",\"model\":"<<quote(options.model.string())<<",\"normalized_path\":"<<quote(options.normalized.string())<<",\"normalized_sha256\":"<<quote(digest(normalized))<<",\"executable_sha256\":"<<quote(exe_sha)<<",\"down_tensor\":"<<quote(options.down)<<",\"down_offset\":"<<down.offset<<",\"down_sha256\":"<<quote(digest(down.data))<<",\"inject_tensor\":"<<quote(options.inject)<<",\"inject_offset\":"<<inject.offset<<",\"inject_sha256\":"<<quote(digest(inject.data))<<",\"arm_schedule\":\"global-pair-even-cycle-copy-balanced-v1\",\"measured_pairs\":200,\"calls_per_sample\":16,\"warmup_pairs\":5,\"samples\":[";
    bool first=true;
    for(unsigned mode=0;mode<3;++mode)for(bool graph_mode:{false,true})for(unsigned copies:{1U,64U}){
        std::vector<Fixture*> pool;for(unsigned j=0;j<copies;++j)pool.push_back(owners[j].get());std::vector<std::unique_ptr<Graph>> baseline_graph,candidate_graph;std::vector<std::vector<float>> baseline_expected,candidate_expected;NumericalSummary numerical;
        for(auto* f:pool){f->poison(false,stream.value);f->reference(stream.value);f->poison(true,stream.value);f->launch(variants[mode],stream.value);check(hipStreamSynchronize(stream.value),"drain copy prequalification");baseline_expected.push_back(f->semantic(false));candidate_expected.push_back(f->semantic(true));if(candidate_expected.size()==1U)validate(*f,candidate_expected.back(),baseline_expected.back(),variants[mode],numerical);require(std::all_of(candidate_expected.back().begin(),candidate_expected.back().end(),[](float x){return std::isfinite(x);}),"real candidate nonfinite");if(pool.size()>1U){require(same(baseline_expected.back(),baseline_expected.front())&&same(candidate_expected.back(),candidate_expected.front()),"physical allocation changes arithmetic");}if(graph_mode){baseline_graph.push_back(std::make_unique<Graph>());candidate_graph.push_back(std::make_unique<Graph>());capture_reference(*f,*baseline_graph.back(),stream.value);f->capture(*candidate_graph.back(),variants[mode],stream.value);}}
        // Both poisons exercise both actual arms/graph executables on every copy.
        for(unsigned pass=0;pass<2;++pass)for(unsigned j=0;j<copies;++j)for(bool candidate:{false,true}){auto* f=pool[j];f->poison(candidate,stream.value);auto& owner=candidate?f->low:f->base_low;check(hipMemsetAsync(owner.p,pass?0x3c:0x69,owner.n-32U,stream.value),"alternate priming poison");if(graph_mode)check(hipGraphLaunch(candidate?candidate_graph[j]->executable:baseline_graph[j]->executable,stream.value),"prime graph");else if(candidate)f->launch(variants[mode],stream.value);else f->reference(stream.value);check(hipStreamSynchronize(stream.value),"drain priming");require(same(f->semantic(candidate),candidate?candidate_expected[j]:baseline_expected[j]),"priming write witness failed");f->guards(candidate);}
        unsigned cursor=0;std::vector<unsigned> base_visits(copies),candidate_visits(copies);
        const auto sample=[&](bool candidate,bool measured){for(auto* f:pool)f->poison(candidate,stream.value);std::vector<bool> touched(copies);check(hipEventRecord(events.start,stream.value),"start timed sample");for(unsigned call=0;call<16;++call){const auto j=cursor;cursor=(cursor+1U)%copies;touched[j]=true;if(measured)++(candidate?candidate_visits:base_visits)[j];if(graph_mode)check(hipGraphLaunch(candidate?candidate_graph[j]->executable:baseline_graph[j]->executable,stream.value),"timed graph");else if(candidate)pool[j]->launch(variants[mode],stream.value);else pool[j]->reference(stream.value);}check(hipEventRecord(events.stop,stream.value),"stop sample");check(hipEventSynchronize(events.stop),"drain sample");float ms=0;check(hipEventElapsedTime(&ms,events.start,events.stop),"elapsed event time");require(std::isfinite(ms)&&ms>0,"invalid event interval");for(unsigned j=0;j<copies;++j)if(touched[j]){require(same(pool[j]->semantic(candidate),candidate?candidate_expected[j]:baseline_expected[j]),"timed sample differs from own qualified output");pool[j]->guards(candidate);}return static_cast<double>(ms)*1000.0/16.0;};
        // Same accepted 64x16 balanced schedule as the independent CPU contract:
        // change which arm starts on each 2-pair copy cycle, including warmup.
        const auto base_first=[&](unsigned pair){const unsigned cycle=copies==64U?pair/2U:0U;return (pair+cycle)%2U==0U;};
        unsigned global_pair=0;for(;global_pair<5;++global_pair){const bool order=base_first(global_pair);static_cast<void>(sample(!order,false));static_cast<void>(sample(order,false));}
        std::vector<double> base,candidate,ratios;for(unsigned pair=0;pair<200;++pair,++global_pair){double b=0,c=0;if(base_first(global_pair)){b=sample(false,true);c=sample(true,true);}else{c=sample(true,true);b=sample(false,true);}base.push_back(b);candidate.push_back(c);ratios.push_back(b/c);}
        const unsigned visits=copies==1U?3200U:50U;require(std::all_of(base_visits.begin(),base_visits.end(),[&](auto v){return v==visits;})&&base_visits==candidate_visits,"physical-copy arm imbalance");for(auto* f:pool)f->readonly();
        out<<(first?"":",")<<"{\"variant\":"<<mode<<",\"mode_name\":"<<quote(mode==0?"p1_r4":mode==1?"p1_r2":"p2_r4")<<",\"graph\":"<<(graph_mode?"true":"false")<<",\"copies\":"<<copies<<",\"copy_arm_balance_max_delta\":0,\"base_visits_per_copy\":";array(out,base_visits);out<<",\"candidate_visits_per_copy\":";array(out,candidate_visits);out<<",\"base_us\":";array(out,base);out<<",\"candidate_us\":";array(out,candidate);out<<",\"paired_speedup\":";array(out,ratios);out<<",\"base_median_us\":"<<median(base)<<",\"candidate_median_us\":"<<median(candidate)<<",\"paired_speedup_median\":"<<median(ratios)<<",\"max_N2_fraction\":"<<numerical.max_candidate_bound_fraction<<",\"serial_bit_differences\":"<<numerical.candidate_serial_bit_differences<<",\"sample_final_outputs_self_exact\":true}";first=false;out.flush();require(out.good(),"benchmark receipt write failed");
    }
    require(libraries()==initial_libraries&&digest(file_bytes(exe))==exe_sha&&digest(file_bytes(options.normalized))==digest(normalized),"runtime/executable/input changed during benchmark");for(auto* f:{owners.front().get()})f->readonly();out<<"],\"loaded_core10_libraries\":[";for(std::size_t j=0;j<initial_libraries.size();++j){const auto& lib=initial_libraries[j];out<<(j?",":"")<<"{\"path\":"<<quote(lib.path)<<",\"sha256\":"<<quote(lib.sha)<<",\"bytes\":"<<lib.size<<'}';}out<<"],\"readonly_inputs_exact\":true,\"owner_guards_intact\":true,\"finished\":true,\"M97_gate_passed\":false}\n";require(out.good(),"final benchmark receipt write failed");
}
} // namespace

int main(int argc,char** argv){try{if(argc>1){require(std::string(argv[1])=="--bench","expected no arguments or --bench");bench(parse(argc,argv));}else native();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
