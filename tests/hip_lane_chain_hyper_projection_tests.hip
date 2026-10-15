#include "strix/hip_lane_chain_hyper_projection_tests.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_content_hash.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using strix::hc_projection_diagnostic::Shape;using strix::hc_projection_diagnostic::Span;
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
void check(hipError_t e,const char* text){if(e!=hipSuccess)throw std::runtime_error(std::string(text)+": "+hipGetErrorString(e));}
struct Owner{void* p=nullptr;std::size_t n=0;explicit Owner(std::size_t bytes):n(bytes){if(n)check(hipMalloc(&p,n),"allocate projection owner");}~Owner(){if(p)static_cast<void>(hipFree(p));}Owner(const Owner&)=delete;Span span()const{return {reinterpret_cast<std::uintptr_t>(p),n};}};
struct Stream{hipStream_t value=nullptr;Stream(){check(hipStreamCreateWithFlags(&value,hipStreamNonBlocking),"create projection stream");}~Stream(){if(value){static_cast<void>(hipStreamSynchronize(value));static_cast<void>(hipStreamDestroy(value));}}};
struct Graph{hipGraph_t graph=nullptr;hipGraphExec_t executable=nullptr;~Graph(){if(executable)static_cast<void>(hipGraphExecDestroy(executable));if(graph)static_cast<void>(hipGraphDestroy(graph));}};
struct Events{hipEvent_t start=nullptr,stop=nullptr;Events(){check(hipEventCreate(&start),"create start event");check(hipEventCreate(&stop),"create stop event");}~Events(){if(start)static_cast<void>(hipEventDestroy(start));if(stop)static_cast<void>(hipEventDestroy(stop));}};
std::vector<std::byte> read(const Owner& owner){std::vector<std::byte> b(owner.n);if(owner.n)check(hipMemcpy(b.data(),owner.p,owner.n,hipMemcpyDeviceToHost),"read projection output");return b;}
std::string digest(const std::vector<std::byte>& bytes){return strix::model_content_identity_hex(strix::sha256_bytes(bytes));}
std::string quoted(const std::string& value){std::string result="\"";for(char raw:value){const auto c=static_cast<unsigned char>(raw);if(c=='"'||c=='\\'){result+='\\';result+=static_cast<char>(c);}else{require(c>=32,"control character in receipt string");result+=static_cast<char>(c);}}return result+'"';}
struct Fixture{
    Shape shape;bool injection;Owner normalized,down,inject,base_low,base_inject,test_low,test_inject;std::vector<std::byte> expected;
    Fixture(Shape s,const std::vector<std::byte>& n,const std::vector<std::byte>& w,const std::vector<std::byte>& i):shape(s),injection(!i.empty()),normalized(n.size()),down(w.size()),inject(i.size()),base_low(static_cast<std::size_t>(s.rows)*s.rank*4U+32U),base_inject(injection?static_cast<std::size_t>(s.rows)*s.streams*4U+32U:0U),test_low(base_low.n),test_inject(base_inject.n){
        check(hipMemcpy(normalized.p,n.data(),n.size(),hipMemcpyHostToDevice),"copy normalized FP32");check(hipMemcpy(down.p,w.data(),w.size(),hipMemcpyHostToDevice),"copy original Q8 down");if(injection)check(hipMemcpy(inject.p,i.data(),i.size(),hipMemcpyHostToDevice),"copy original F32 injection");
        poison(false,nullptr);launch(false,0,1,true,nullptr);check(hipDeviceSynchronize(),"drain original projection");expected=semantic(false);guards(false);
    }
    void launch(bool candidate,unsigned opw,unsigned prefetch,bool rows8,hipStream_t stream){strix::hc_projection_diagnostic::enqueue(shape,normalized.span(),down.span(),inject.span(),candidate?test_low.span():base_low.span(),candidate?test_inject.span():base_inject.span(),opw,prefetch,rows8,stream);}
    void poison(bool candidate,hipStream_t stream){auto& low=candidate?test_low:base_low;auto& inj=candidate?test_inject:base_inject;const int pattern=candidate?0x5a:0xa5;check(hipMemsetAsync(low.p,pattern,low.n,stream),"poison low owner");if(injection)check(hipMemsetAsync(inj.p,pattern,inj.n,stream),"poison injection owner");}
    std::vector<std::byte> semantic(bool candidate){auto b=read(candidate?test_low:base_low);b.resize(b.size()-32U);if(injection){auto i=read(candidate?test_inject:base_inject);i.resize(i.size()-32U);b.insert(b.end(),i.begin(),i.end());}return b;}
    void guards(bool candidate){for(const auto* owner:{candidate?&test_low:&base_low,candidate?&test_inject:&base_inject})if(owner->n){const auto b=read(*owner);const auto pattern=candidate?std::byte{0x5a}:std::byte{0xa5};require(std::all_of(b.end()-32,b.end(),[&](std::byte x){return x==pattern;}),"projection output guard overwritten");}}
    void verify(bool candidate){require(semantic(candidate)==expected,"projection changed original complete FP32 output bits");guards(candidate);}
    void capture(Graph& graph,bool candidate,unsigned opw,unsigned p,hipStream_t stream){check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"begin projection capture");try{launch(candidate,opw,p,true,stream);}catch(...){static_cast<void>(hipStreamEndCapture(stream,&graph.graph));throw;}check(hipStreamEndCapture(stream,&graph.graph),"end projection capture");check(hipGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0),"instantiate projection graph");}
};
std::vector<std::byte> bytes(const std::vector<float>& values){const auto view=std::as_bytes(std::span(values));return {view.begin(),view.end()};}

std::uint32_t hash_word(std::uint32_t x){x^=x>>16U;x*=0x7feb352dU;x^=x>>15U;x*=0x846ca68bU;return x^(x>>16U);}
float half_value(unsigned bits){const auto magnitude=bits&0x7fffU;const auto exponent=(magnitude>>10U)&31U;const auto fraction=magnitude&1023U;const auto value=exponent?std::ldexp(1.0F+static_cast<float>(fraction)/1024.0F,static_cast<int>(exponent)-15):std::ldexp(static_cast<float>(fraction),-24);return bits&0x8000U?-value:value;}
bool fma_sensitive(Shape s,const std::vector<float>& n,const std::vector<std::byte>& weights){
    for(unsigned row=0;row<s.rank;++row){float fused=0,unfused=0;
        for(unsigned k=0;k<s.width;++k){const auto block=(static_cast<std::size_t>(row)*(s.width/32U)+k/32U)*34U;const auto bits=std::to_integer<unsigned>(weights[block])|(std::to_integer<unsigned>(weights[block+1U])<<8U);const auto byte=std::to_integer<unsigned>(weights[block+2U+k%32U]);const auto code=byte<128U?static_cast<int>(byte):static_cast<int>(byte)-256;const float coefficient=half_value(bits)*static_cast<float>(code);const float input=n[k];fused=std::fmaf(input,coefficient,fused);volatile float product=input*coefficient;unfused=unfused+product;}
        const float fused_scaled=fused/static_cast<float>(s.streams),unfused_scaled=unfused/static_cast<float>(s.streams);
        const float fused_output=fused_scaled*(1.0F/(1.0F+std::exp(-fused_scaled))),unfused_output=unfused_scaled*(1.0F/(1.0F+std::exp(-unfused_scaled)));
        if(std::isfinite(fused_output) && std::isfinite(unfused_output) && std::bit_cast<std::uint32_t>(fused_output)!=std::bit_cast<std::uint32_t>(unfused_output))return true;
    }return false;
}
void negative_cases(Fixture& f){
    f.poison(true,nullptr);check(hipDeviceSynchronize(),"drain negative-case poison");
    auto s=f.shape;const auto n=f.normalized.span(),d=f.down.span(),i=f.inject.span(),l=f.test_low.span(),o=f.test_inject.span();
    const auto before_low=read(f.test_low),before_inject=read(f.test_inject);
    const auto rejects=[&](Shape shape,Span norm,Span down,Span inject,Span low,Span output,unsigned opw,unsigned p,const char* message){bool rejected=false;try{strix::hc_projection_diagnostic::enqueue(shape,norm,down,inject,low,output,opw,p,true,nullptr);}catch(const std::invalid_argument& error){rejected=std::string(error.what())==message;}require(rejected,"diagnostic negative rejected with wrong type/message or enqueued");require(read(f.test_low)==before_low && read(f.test_inject)==before_inject,"diagnostic negative wrote poisoned owners");};
    constexpr auto geometry="HC projection diagnostic unsupported shape/variant";constexpr auto pair="HC projection diagnostic injection owner pair differs";constexpr auto owner="HC projection diagnostic owner extent/alignment differs";constexpr auto overlap="HC projection diagnostic owners overlap";
    for(unsigned value:{0U,10496U,257U}){auto x=s;x.width=value;rejects(x,n,d,i,l,o,8,1,geometry);}
    for(unsigned value:{1U,9U}){auto x=s;x.streams=value;rejects(x,n,d,i,l,o,8,1,geometry);}
    {auto x=s;x.width=256;x.streams=3;rejects(x,n,d,i,l,o,8,1,geometry);}
    for(unsigned value:{0U,513U}){auto x=s;x.rank=value;rejects(x,n,d,i,l,o,8,1,geometry);}
    for(unsigned value:{0U,65536U}){auto x=s;x.rows=value;rejects(x,n,d,i,l,o,8,1,geometry);}
    for(unsigned value:{4U,64U})rejects(s,n,d,i,l,o,value,1,geometry);
    for(unsigned value:{0U,4U})rejects(s,n,d,i,l,o,8,value,geometry);
    if(f.injection){rejects(s,n,d,{},l,o,8,1,pair);rejects(s,n,d,i,l,{},8,1,pair);auto bad=i;bad.address+=4;rejects(s,n,d,bad,l,o,8,1,owner);}
    rejects(s,n,d,{0,1},l,{},8,1,pair);rejects(s,n,d,{},l,{0,1},8,1,pair);
    for(unsigned index=0;index<(f.injection?5U:3U);++index){auto nn=n,dd=d,ii=i,ll=l,oo=o;if(index==0)nn.bytes=static_cast<std::uint64_t>(s.rows)*s.width*4U-1U;if(index==1)dd.bytes=static_cast<std::uint64_t>(s.rank)*(s.width/32U)*34U-1U;if(index==2)ll.bytes=static_cast<std::uint64_t>(s.rows)*s.rank*4U-1U;if(index==3)ii.bytes=static_cast<std::uint64_t>(s.streams)*s.width*4U-1U;if(index==4)oo.bytes=static_cast<std::uint64_t>(s.rows)*s.streams*4U-1U;rejects(s,nn,dd,ii,ll,oo,8,1,owner);}
    auto low=l;low.address=n.address;rejects(s,n,d,i,low,o,8,1,overlap);
    if(f.injection){low=l;low.address=o.address;rejects(s,n,d,i,low,o,8,1,overlap);auto output=o;output.address=i.address;rejects(s,n,d,i,l,output,8,1,overlap);}
}

void synthetic(Shape s,unsigned adversary,bool injection){
    std::vector<float> n(static_cast<std::size_t>(s.rows)*s.width),i(injection?static_cast<std::size_t>(s.streams)*s.width:0U);
    for(std::size_t j=0;j<n.size();++j)n[j]=static_cast<float>(static_cast<int>(j%47U)-23)*0.03125F;
    for(std::size_t j=0;j<i.size();++j)i[j]=static_cast<float>(static_cast<int>(j%31U)-15)*0.0078125F;
    std::vector<std::byte> w(static_cast<std::size_t>(s.rank)*(s.width/32U)*34U);constexpr std::array<unsigned,8> scales={0U,0x8000U,1U,0x8001U,0x3c00U,0xbc00U,0x7bffU,0xfbffU};
    for(std::size_t b=0;b<w.size()/34U;++b){const auto scale=adversary?scales[b%8U]:0x2400U+static_cast<unsigned>(b%128U);w[b*34U]=static_cast<std::byte>(scale&255U);w[b*34U+1U]=static_cast<std::byte>(scale>>8U);for(unsigned k=0;k<32;++k)w[b*34U+2U+k]=static_cast<std::byte>((b*17U+k*59U)&255U);}
    if(adversary==3){constexpr std::array<unsigned,8> patterns={0U,0x80000000U,1U,0x80000001U,0x00800000U,0x80800000U,0x7f7fffffU,0xff7fffffU};for(std::size_t j=0;j<n.size();++j)n[j]=std::bit_cast<float>(patterns[j%8U]);}

    if(adversary==2){
        constexpr std::array<unsigned,8> patterns={0U,0x80000000U,1U,0x80000001U,0x00800000U,0x80800000U,0x00800001U,0x000003ffU};for(std::size_t j=0;j<n.size();++j)n[j]=std::bit_cast<float>(patterns[j%8U]);
        for(std::size_t b=0;b<w.size()/34U;++b){w[b*34U]=std::byte{0};w[b*34U+1U]=std::byte{0x3c};for(unsigned k=0;k<32;++k)w[b*34U+2U+k]=std::byte{1};}
    }
    if(adversary==4){
        for(std::size_t j=0;j<n.size();++j){const auto h=hash_word(static_cast<unsigned>(j)+17U);n[j]=std::bit_cast<float>((h&0x807fffffU)|((121U+(h%13U))<<23U));}
        for(std::size_t b=0;b<w.size()/34U;++b){const auto h=hash_word(static_cast<unsigned>(b)+71U);const auto bits=((5U+h%5U)<<10U)|(h&1023U)|(h&0x8000U);w[b*34U]=static_cast<std::byte>(bits&255U);w[b*34U+1U]=static_cast<std::byte>(bits>>8U);}
        // Make both final-coefficient mutation controls observable, avoiding
        // saturation or cancellation in the final output rows.
        if(s.rank>1){for(unsigned row=0;row<s.rows;++row)n[static_cast<std::size_t>(row)*s.width+s.width-1U]=std::bit_cast<float>(0x3f8f1234U);const auto first=static_cast<std::size_t>(s.rank-1U)*(s.width/32U)*34U;for(std::size_t j=first;j<w.size();++j)w[j]=std::byte{0};w[w.size()-33U]=std::byte{0x3c};w.back()=std::byte{4};}
        if(injection){const auto first=static_cast<std::size_t>(s.streams-1U)*s.width;std::fill(i.begin()+static_cast<std::ptrdiff_t>(first),i.end(),0.0F);i.back()=0.125F;}
        require(fma_sensitive(s,n,w),"full-mantissa fixture does not discriminate FMA from rounded mul+add");
    }
    Fixture f(s,bytes(n),w,bytes(i));f.poison(false,nullptr);f.launch(false,0,1,false,nullptr);check(hipDeviceSynchronize(),"drain plain reference");f.verify(false);

    if(adversary==2){bool nonzero=false;for(std::size_t j=0;j<static_cast<std::size_t>(s.rows)*s.rank*4U;j+=4U){float value=0;std::memcpy(&value,f.expected.data()+j,4);require(std::isfinite(value),"isolated subnormal fixture overflowed");nonzero|=value!=0.0F;}require(nonzero,"isolated subnormal outputs all zero");}
    if(adversary==4 && s.rank>1){auto changed=w;changed.back()=std::byte{3};Fixture mutation(s,bytes(n),changed,bytes(i));require(mutation.expected!=f.expected,"last code in last Q8 group/row mutation is not observable");}
    if(adversary==4 && injection){auto changed=i;changed.back()=0.25F;Fixture mutation(s,bytes(n),w,bytes(changed));require(mutation.expected!=f.expected,"last injection coefficient mutation is not observable");}
    negative_cases(f);
    Stream stream;
    for(unsigned opw:{8U,16U,32U})for(unsigned prefetch:{1U,2U}){
        f.poison(true,nullptr);f.launch(true,opw,prefetch,true,nullptr);check(hipDeviceSynchronize(),"drain eager candidate");f.verify(true);
        Graph graph;f.capture(graph,true,opw,prefetch,stream.value);for(unsigned replay=0;replay<3;++replay){f.poison(true,stream.value);check(hipGraphLaunch(graph.executable,stream.value),"replay projection");check(hipStreamSynchronize(stream.value),"drain projection replay");f.verify(true);}
        const auto before=f.semantic(true),low_before=read(f.test_low),inject_before=read(f.test_inject);auto bad=f.down.span();bad.address+=2U;bool rejected=false;try{strix::hc_projection_diagnostic::enqueue(s,f.normalized.span(),bad,f.inject.span(),f.test_low.span(),f.test_inject.span(),opw,prefetch,true,nullptr);}catch(const std::invalid_argument&){rejected=true;}require(rejected && f.semantic(true)==before && read(f.test_low)==low_before && read(f.test_inject)==inject_before,"misaligned diagnostic owner did not fail before enqueue");
    }
}
struct Library{std::string path,sha;std::uint64_t bytes=0;bool operator==(const Library&)const=default;};
std::vector<Library> libraries(){
    std::ifstream maps("/proc/self/maps");require(maps.good(),"cannot read live benchmark mappings");std::string text((std::istreambuf_iterator<char>(maps)),std::istreambuf_iterator<char>());
    const auto core=std::filesystem::canonical("/opt/rocm/core-10.1").string()+"/";std::vector<Library> result;
    for(const auto* name:{"libamdhip64.so","libhsa-runtime64.so"}){
        std::set<std::string> paths;std::size_t first=0;
        while(first<text.size()){const auto last=text.find('\n',first);const auto line=text.substr(first,last==std::string::npos?text.size()-first:last-first);if(line.find(name)!=std::string::npos){require(line.find("(deleted)")==std::string::npos,"deleted benchmark runtime mapping");const auto slash=line.find('/');require(slash!=std::string::npos,"mapped benchmark library path absent");paths.insert(line.substr(slash));}if(last==std::string::npos)break;first=last+1U;}
        require(paths.size()==1U,"missing/ambiguous benchmark runtime library");const auto path=std::filesystem::canonical(*paths.begin());require(path.string().starts_with(core),"non-Core10 benchmark runtime mapping");const auto size=std::filesystem::file_size(path);const auto hash=strix::model_content_identity_hex(strix::sha256_file_content(path));require(size==std::filesystem::file_size(path),"runtime library changed during hashing");result.push_back({path.string(),hash,size});
    }
    return result;
}
struct TensorBytes{std::vector<std::byte> data;std::uint64_t offset=0;};
TensorBytes tensor(const std::filesystem::path& path,const strix::gguf::FileInfo& info,const std::string& name,unsigned type,const std::vector<std::uint64_t>& dims){
    const strix::gguf::TensorInfo* found=nullptr;for(const auto& t:info.tensors)if(t.name==name){require(!found,"duplicated benchmark tensor");found=&t;}
    require(found && found->ggml_type==type && found->dimensions==dims,"benchmark tensor type/shape differs");require(found->absolute_offset<=static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()),"benchmark tensor offset overflow");require(found->byte_size<=std::numeric_limits<std::size_t>::max(),"benchmark tensor too large");
    TensorBytes result;result.offset=found->absolute_offset;result.data.resize(static_cast<std::size_t>(found->byte_size));std::ifstream f(path,std::ios::binary);f.seekg(static_cast<std::streamoff>(result.offset));f.read(reinterpret_cast<char*>(result.data.data()),static_cast<std::streamsize>(result.data.size()));require(f.good(),"benchmark tensor truncated");return result;
}
std::vector<std::byte> file_bytes(const std::filesystem::path& path,std::size_t count){require(std::filesystem::file_size(path)==count,"normalized FP32 capture extent differs");std::vector<std::byte> b(count);std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(count));require(f.good(),"normalized capture truncated");return b;}
double median(std::vector<double> values){std::sort(values.begin(),values.end());return values.size()%2U?values[values.size()/2U]:(values[values.size()/2U-1U]+values[values.size()/2U])*0.5;}
void array(std::ostream& out,const std::vector<double>& values){out<<'[';for(std::size_t j=0;j<values.size();++j)out<<(j?",":"")<<values[j];out<<']';}
struct Options{std::filesystem::path model,normalized,output;std::string down,inject;unsigned width=10240,streams=4,rank=320,warmup=5,pairs=200,iterations=16,cold_copies=1;};
Options parse(int argc,char** argv){Options a;std::vector<std::string> seen;for(int k=2;k<argc;k+=2){require(k+1<argc,"benchmark option needs value");const std::string key=argv[k],value=argv[k+1];require(std::find(seen.begin(),seen.end(),key)==seen.end(),"duplicate benchmark option");seen.push_back(key);if(key=="--model")a.model=value;else if(key=="--normalized")a.normalized=value;else if(key=="--output")a.output=value;else if(key=="--down-tensor")a.down=value;else if(key=="--inject-tensor")a.inject=value;else{std::size_t consumed=0;const auto number=std::stoul(value,&consumed);require(consumed==value.size() && !value.empty() && value.front()!='-',"invalid benchmark integer");require(number<=std::numeric_limits<unsigned>::max(),"benchmark integer overflow");const auto v=static_cast<unsigned>(number);if(key=="--width")a.width=v;else if(key=="--streams")a.streams=v;else if(key=="--rank")a.rank=v;else if(key=="--warmup")a.warmup=v;else if(key=="--pairs")a.pairs=v;else if(key=="--iterations")a.iterations=v;else if(key=="--cold-copies")a.cold_copies=v;else throw std::invalid_argument("unknown benchmark option");}}
    require(a.width && a.width<=10240U && a.width%256U==0 && a.streams>=2U && a.streams<=8U && a.width%a.streams==0 && a.rank && a.rank<=512U,"unsupported benchmark shape");
    require(!a.model.empty()&&!a.normalized.empty()&&!a.output.empty()&&!a.down.empty()&&!a.inject.empty(),"need --model --normalized --down-tensor --inject-tensor --output");require(a.warmup>=5 && a.warmup<=100 && a.pairs>=200 && a.pairs<=10000 && a.iterations>=1 && a.iterations<=256,"benchmark count outside qualification bounds");require(a.cold_copies>=1 && a.cold_copies<=128,"cold copy count outside 1..128");require(!std::filesystem::exists(a.output),"benchmark output already exists");return a;
}
void bench(const Options& a){
    require(std::endian::native==std::endian::little,"FP32/Q8 capture requires little-endian host");if(const auto* override_arch=std::getenv("HSA_OVERRIDE_GFX_VERSION");override_arch && *override_arch)throw std::invalid_argument("architecture override forbidden");
    const auto info=strix::gguf::read_file(a.model);const auto down=tensor(a.model,info,a.down,8,{a.width,a.rank});TensorBytes inject;if(!a.inject.empty())inject=tensor(a.model,info,a.inject,0,{a.width,a.streams});
    const auto normalized=file_bytes(a.normalized,static_cast<std::size_t>(a.width)*4U);for(std::size_t j=0;j<normalized.size();j+=4U){float x=0;std::memcpy(&x,normalized.data()+j,4);require(std::isfinite(x),"real normalized capture contains nonfinite input");}
    int device=0;check(hipGetDevice(&device),"read benchmark device");hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,device),"read benchmark architecture");require(std::string(properties.gcnArchName).starts_with("gfx1151"),"expected actual gfx1151");
    int runtime_version=0,driver_version=0;check(hipRuntimeGetVersion(&runtime_version),"read runtime version");check(hipDriverGetVersion(&driver_version),"read driver version");
    const auto loaded=libraries();const auto executable_sha=strix::model_content_identity_hex(strix::sha256_file_content(std::filesystem::canonical("/proc/self/exe")));
    const auto owner_bytes=normalized.size()+down.data.size()+inject.data.size()+2U*(static_cast<std::size_t>(a.rank)+a.streams)*4U+128U;
    require(owner_bytes<=512U*1024U*1024U/(static_cast<std::size_t>(a.cold_copies)+1U),"rotating projection owners exceed 512 MiB budget");
    Fixture f({a.width,a.streams,a.rank,1},normalized,down.data,inject.data);for(std::size_t j=0;j<f.expected.size();j+=4U){float x=0;std::memcpy(&x,f.expected.data()+j,4);require(std::isfinite(x),"real projection reference output is nonfinite");}
    std::ofstream out(a.output);require(out.good(),"cannot create benchmark output");out<<std::setprecision(17)<<"{\"schema\":\"private-exact-hc-projection-microbench-v1\",\"device\":"<<device<<",\"architecture\":"<<quoted(properties.gcnArchName)<<",\"runtime_version\":"<<runtime_version<<",\"driver_version\":"<<driver_version<<",\"width\":"<<a.width<<",\"streams\":"<<a.streams<<",\"rank\":"<<a.rank<<",\"normalized_path\":"<<quoted(a.normalized.string())<<",\"normalized_sha256\":"<<quoted(digest(normalized))<<",\"model\":"<<quoted(a.model.string())<<",\"down_tensor\":"<<quoted(a.down)<<",\"down_offset\":"<<down.offset<<",\"down_sha256\":"<<quoted(digest(down.data))<<",\"inject_tensor\":"<<quoted(a.inject)<<",\"inject_offset\":"<<inject.offset<<",\"inject_sha256\":"<<quoted(digest(inject.data))<<",\"warmup_pairs\":"<<a.warmup<<",\"measured_pairs\":"<<a.pairs<<",\"calls_per_sample\":"<<a.iterations<<",\"executable_sha256\":"<<quoted(executable_sha)<<",\"weight_residency\":\"private copied allocations; original payload bytes, no repack\"";
    out<<",\"loaded_core10_libraries\":[";
    for(std::size_t j=0;j<loaded.size();++j)out<<(j?",":"")<<"{\"path\":"<<quoted(loaded[j].path)<<",\"sha256\":"<<quoted(loaded[j].sha)<<",\"bytes\":"<<loaded[j].bytes<<"}";
    out<<"],\"samples\":[";
    std::vector<std::unique_ptr<Fixture>> rotated;
    if(a.cold_copies>1)for(unsigned copy=0;copy<a.cold_copies;++copy){rotated.push_back(std::make_unique<Fixture>(Shape{a.width,a.streams,a.rank,1},normalized,down.data,inject.data));require(rotated.back()->expected==f.expected,"rotating copy reference differs");}
    Stream stream;Events events;bool first=true;
    for(unsigned copies: a.cold_copies>1?std::vector<unsigned>{1U,a.cold_copies}:std::vector<unsigned>{1U}){
    for(unsigned opw:{8U,16U,32U})for(unsigned prefetch:{1U,2U})for(bool captured:{false,true}){
        std::vector<Fixture*> pool;if(copies==1)pool.push_back(&f);else for(auto& copy:rotated)pool.push_back(copy.get());
        std::vector<std::unique_ptr<Graph>> base_graphs,candidate_graphs;
        if(captured)for(auto* fixture:pool){base_graphs.push_back(std::make_unique<Graph>());candidate_graphs.push_back(std::make_unique<Graph>());fixture->capture(*base_graphs.back(),false,0,1,stream.value);fixture->capture(*candidate_graphs.back(),true,opw,prefetch,stream.value);}
        unsigned base_cursor=0,candidate_cursor=0;
        const auto sample=[&](bool candidate){
            for(auto* fixture:pool)fixture->poison(candidate,stream.value);
            auto& cursor=candidate?candidate_cursor:base_cursor;std::vector<bool> touched(pool.size(),false);
            check(hipEventRecord(events.start,stream.value),"record start");
            for(unsigned call=0;call<a.iterations;++call){const auto index=cursor;cursor=(cursor+1U)%copies;touched[index]=true;auto* fixture=pool[index];if(captured)check(hipGraphLaunch(candidate?candidate_graphs[index]->executable:base_graphs[index]->executable,stream.value),"launch timed graph");else fixture->launch(candidate,candidate?opw:0U,candidate?prefetch:1U,true,stream.value);}
            check(hipEventRecord(events.stop,stream.value),"record stop");check(hipEventSynchronize(events.stop),"drain timed events");float ms=0;check(hipEventElapsedTime(&ms,events.start,events.stop),"read event timing");require(std::isfinite(ms)&&ms>0,"invalid event interval");
            for(std::size_t index=0;index<pool.size();++index)if(touched[index])pool[index]->verify(candidate);
            return static_cast<double>(ms)*1000.0/a.iterations;
        };
        for(unsigned pair=0;pair<a.warmup;++pair){static_cast<void>(sample(pair%2U!=0));static_cast<void>(sample(pair%2U==0));}
        std::vector<double> base,candidate,ratios;
        for(unsigned pair=0;pair<a.pairs;++pair){double b=0,c=0;if(pair%2U==0){b=sample(false);c=sample(true);}else{c=sample(true);b=sample(false);}base.push_back(b);candidate.push_back(c);ratios.push_back(b/c);}
        out<<(first?"":",")<<"{\"cache_mode\":"<<quoted(copies==1?"hot-single-copy":"rotating-private-copies")<<",\"copies\":"<<copies<<",\"down_working_set_bytes\":"<<static_cast<std::uint64_t>(down.data.size())*copies<<",\"opw\":"<<opw<<",\"prefetch\":"<<prefetch<<",\"graph\":"<<(captured?"true":"false")<<",\"base_us\":";array(out,base);out<<",\"candidate_us\":";array(out,candidate);out<<",\"paired_speedup\":";array(out,ratios);out<<",\"base_median_us\":"<<median(base)<<",\"candidate_median_us\":"<<median(candidate)<<",\"paired_speedup_median\":"<<median(ratios)<<",\"every_timed_output_exact\":true}";first=false;out.flush();require(out.good(),"microbench receipt write failed");
    }
    }
    require(libraries()==loaded,"loaded Core10 library identities changed during benchmark");
    out<<"],\"finished\":true,\"scope\":\"Rotating copies do not establish measured LLC capacity or guarantee eviction; inspect down_working_set_bytes. Single projection only. No full-model integration or throughput promise. P4 deliberately not instantiated: 272 packed dwords alone exceed 256 VGPR. Actual register/scratch and scalar input loads require ISA audit before execution.\"}\n";require(out.good(),"final microbench receipt write failed");
}
}
int main(int argc,char** argv)try{
    if(argc>1){require(std::string(argv[1])=="--bench","expected no args or --bench options");bench(parse(argc,argv));return 0;}
    for(bool injection:{false,true})for(unsigned adversary=0;adversary<5;++adversary){synthetic({256,2,1,2},adversary,injection);synthetic({768,3,33,2},adversary,injection);synthetic({10240,4,320,1},adversary,injection);synthetic({512,8,511,2},adversary,injection);}
    std::cout<<"HC exact projection prototype cases 40 variants 6 eager+graph3 and plain/rows8 reference PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
