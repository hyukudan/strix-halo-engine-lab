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
void synthetic(Shape s,unsigned adversary,bool injection){
    std::vector<float> n(static_cast<std::size_t>(s.rows)*s.width),i(injection?static_cast<std::size_t>(s.streams)*s.width:0U);
    for(std::size_t j=0;j<n.size();++j)n[j]=static_cast<float>(static_cast<int>(j%47U)-23)*0.03125F;
    for(std::size_t j=0;j<i.size();++j)i[j]=static_cast<float>(static_cast<int>(j%31U)-15)*0.0078125F;
    std::vector<std::byte> w(static_cast<std::size_t>(s.rank)*(s.width/32U)*34U);constexpr std::array<unsigned,8> scales={0U,0x8000U,1U,0x8001U,0x3c00U,0xbc00U,0x7bffU,0xfbffU};
    for(std::size_t b=0;b<w.size()/34U;++b){const auto scale=adversary?scales[b%8U]:0x2400U+static_cast<unsigned>(b%128U);w[b*34U]=static_cast<std::byte>(scale&255U);w[b*34U+1U]=static_cast<std::byte>(scale>>8U);for(unsigned k=0;k<32;++k)w[b*34U+2U+k]=static_cast<std::byte>((b*17U+k*59U)&255U);}
    if(adversary==2){constexpr std::array<unsigned,8> patterns={0U,0x80000000U,1U,0x80000001U,0x00800000U,0x80800000U,0x7f7fffffU,0xff7fffffU};for(std::size_t j=0;j<n.size();++j)n[j]=std::bit_cast<float>(patterns[j%8U]);}
    Fixture f(s,bytes(n),w,bytes(i));f.poison(false,nullptr);f.launch(false,0,1,false,nullptr);check(hipDeviceSynchronize(),"drain plain reference");f.verify(false);
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
struct Options{std::filesystem::path model,normalized,output;std::string down,inject;unsigned width=10240,streams=4,rank=320,warmup=5,pairs=200,iterations=16;};
Options parse(int argc,char** argv){Options a;std::vector<std::string> seen;for(int k=2;k<argc;k+=2){require(k+1<argc,"benchmark option needs value");const std::string key=argv[k],value=argv[k+1];require(std::find(seen.begin(),seen.end(),key)==seen.end(),"duplicate benchmark option");seen.push_back(key);if(key=="--model")a.model=value;else if(key=="--normalized")a.normalized=value;else if(key=="--output")a.output=value;else if(key=="--down-tensor")a.down=value;else if(key=="--inject-tensor")a.inject=value;else{std::size_t consumed=0;const auto number=std::stoul(value,&consumed);require(consumed==value.size() && !value.empty() && value.front()!='-',"invalid benchmark integer");require(number<=std::numeric_limits<unsigned>::max(),"benchmark integer overflow");const auto v=static_cast<unsigned>(number);if(key=="--width")a.width=v;else if(key=="--streams")a.streams=v;else if(key=="--rank")a.rank=v;else if(key=="--warmup")a.warmup=v;else if(key=="--pairs")a.pairs=v;else if(key=="--iterations")a.iterations=v;else throw std::invalid_argument("unknown benchmark option");}}
    require(a.width && a.width<=10240U && a.width%256U==0 && a.streams>=2U && a.streams<=8U && a.width%a.streams==0 && a.rank && a.rank<=512U,"unsupported benchmark shape");
    require(!a.model.empty()&&!a.normalized.empty()&&!a.output.empty()&&!a.down.empty(),"need --model --normalized --down-tensor --output");require(a.warmup>=5 && a.warmup<=100 && a.pairs>=200 && a.pairs<=10000 && a.iterations>=1 && a.iterations<=256,"benchmark count outside qualification bounds");require(!std::filesystem::exists(a.output),"benchmark output already exists");return a;
}
void bench(const Options& a){
    require(std::endian::native==std::endian::little,"FP32/Q8 capture requires little-endian host");if(const auto* override_arch=std::getenv("HSA_OVERRIDE_GFX_VERSION");override_arch && *override_arch)throw std::invalid_argument("architecture override forbidden");
    const auto info=strix::gguf::read_file(a.model);const auto down=tensor(a.model,info,a.down,8,{a.width,a.rank});TensorBytes inject;if(!a.inject.empty())inject=tensor(a.model,info,a.inject,0,{a.width,a.streams});
    const auto normalized=file_bytes(a.normalized,static_cast<std::size_t>(a.width)*4U);for(std::size_t j=0;j<normalized.size();j+=4U){float x=0;std::memcpy(&x,normalized.data()+j,4);require(std::isfinite(x),"real normalized capture contains nonfinite input");}
    Fixture f({a.width,a.streams,a.rank,1},normalized,down.data,inject.data);for(std::size_t j=0;j<f.expected.size();j+=4U){float x=0;std::memcpy(&x,f.expected.data()+j,4);require(std::isfinite(x),"real projection reference output is nonfinite");}
    int device=0;check(hipGetDevice(&device),"read benchmark device");hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,device),"read benchmark architecture");require(std::string(properties.gcnArchName).starts_with("gfx1151"),"expected actual gfx1151");
    int runtime_version=0,driver_version=0;check(hipRuntimeGetVersion(&runtime_version),"read runtime version");check(hipDriverGetVersion(&driver_version),"read driver version");
    const auto loaded=libraries();const auto executable_sha=strix::model_content_identity_hex(strix::sha256_file_content(std::filesystem::canonical("/proc/self/exe")));
    std::ofstream out(a.output);require(out.good(),"cannot create benchmark output");out<<std::setprecision(17)<<"{\"schema\":\"private-exact-hc-projection-microbench-v1\",\"device\":"<<device<<",\"architecture\":"<<quoted(properties.gcnArchName)<<",\"runtime_version\":"<<runtime_version<<",\"driver_version\":"<<driver_version<<",\"width\":"<<a.width<<",\"streams\":"<<a.streams<<",\"rank\":"<<a.rank<<",\"normalized_path\":"<<quoted(a.normalized.string())<<",\"normalized_sha256\":"<<quoted(digest(normalized))<<",\"model\":"<<quoted(a.model.string())<<",\"down_tensor\":"<<quoted(a.down)<<",\"down_offset\":"<<down.offset<<",\"down_sha256\":"<<quoted(digest(down.data))<<",\"inject_tensor\":"<<quoted(a.inject)<<",\"inject_offset\":"<<inject.offset<<",\"inject_sha256\":"<<quoted(digest(inject.data))<<",\"warmup_pairs\":"<<a.warmup<<",\"measured_pairs\":"<<a.pairs<<",\"calls_per_sample\":"<<a.iterations<<",\"executable_sha256\":"<<quoted(executable_sha)<<",\"weight_residency\":\"private copied allocations; original payload bytes, no repack\"";
    out<<",\"loaded_core10_libraries\":[";
    for(std::size_t j=0;j<loaded.size();++j)out<<(j?",":"")<<"{\"path\":"<<quoted(loaded[j].path)<<",\"sha256\":"<<quoted(loaded[j].sha)<<",\"bytes\":"<<loaded[j].bytes<<"}";
    out<<"],\"samples\":[";
    Stream stream;Events events;bool first=true;
    for(unsigned opw:{8U,16U,32U})for(unsigned prefetch:{1U,2U})for(bool captured:{false,true}){
        Graph base_graph,candidate_graph;if(captured){f.capture(base_graph,false,0,1,stream.value);f.capture(candidate_graph,true,opw,prefetch,stream.value);}
        const auto sample=[&](bool candidate){f.poison(candidate,stream.value);check(hipEventRecord(events.start,stream.value),"record start");for(unsigned call=0;call<a.iterations;++call){if(captured)check(hipGraphLaunch(candidate?candidate_graph.executable:base_graph.executable,stream.value),"launch timed graph");else f.launch(candidate,candidate?opw:0U,candidate?prefetch:1U,true,stream.value);}check(hipEventRecord(events.stop,stream.value),"record stop");check(hipEventSynchronize(events.stop),"drain timed events");float ms=0;check(hipEventElapsedTime(&ms,events.start,events.stop),"read event timing");require(std::isfinite(ms)&&ms>0,"invalid event interval");f.verify(candidate);return static_cast<double>(ms)*1000.0/a.iterations;};
        for(unsigned pair=0;pair<a.warmup;++pair){static_cast<void>(sample(pair%2U!=0));static_cast<void>(sample(pair%2U==0));}
        std::vector<double> base,candidate,ratios;
        for(unsigned pair=0;pair<a.pairs;++pair){double b=0,c=0;if(pair%2U==0){b=sample(false);c=sample(true);}else{c=sample(true);b=sample(false);}base.push_back(b);candidate.push_back(c);ratios.push_back(b/c);}
        out<<(first?"":",")<<"{\"opw\":"<<opw<<",\"prefetch\":"<<prefetch<<",\"graph\":"<<(captured?"true":"false")<<",\"base_us\":";array(out,base);out<<",\"candidate_us\":";array(out,candidate);out<<",\"paired_speedup\":";array(out,ratios);out<<",\"base_median_us\":"<<median(base)<<",\"candidate_median_us\":"<<median(candidate)<<",\"paired_speedup_median\":"<<median(ratios)<<",\"every_timed_output_exact\":true}";first=false;out.flush();require(out.good(),"microbench receipt write failed");
    }
    require(libraries()==loaded,"loaded Core10 library identities changed during benchmark");
    out<<"],\"finished\":true,\"scope\":\"Single projection only. No full-model integration or throughput promise. P4 deliberately not instantiated:272 packed dwords alone exceed256 VGPR. Actual register/scratch and scalar input loads require ISA audit before execution.\"}\n";require(out.good(),"final microbench receipt write failed");
}
}
int main(int argc,char** argv)try{
    if(argc>1){require(std::string(argv[1])=="--bench","expected no args or --bench options");bench(parse(argc,argv));return 0;}
    for(bool injection:{false,true})for(unsigned adversary=0;adversary<3;++adversary){synthetic({256,2,1,2},adversary,injection);synthetic({768,3,33,2},adversary,injection);synthetic({10240,4,320,1},adversary,injection);synthetic({512,8,511,2},adversary,injection);}
    std::cout<<"HC exact projection prototype cases24 variants6 eager+graph3 and plain/rows8 reference PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
