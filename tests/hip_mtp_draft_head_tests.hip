// MTP draft head fixture: subset rows are bit-identical to the full Q8 head,
// device argmax follows the full selector order, plus a per-draft micro-benchmark
// (--bench) of the full head + host selector against the sync-free subset chain.
#include "strix/hip_mtp_draft_head.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using strix::HipMtpDraftHead;using strix::HipMtpDraftHeadConfig;
void require(bool x,const std::string& m){if(!x)throw std::runtime_error(m);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
template<class T> struct Device {
    T* p=nullptr;std::size_t n=0;
    explicit Device(std::size_t count):n(count){check(hipMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));}
    ~Device(){if(p)static_cast<void>(hipFree(p));}
    std::vector<T> read() const {std::vector<T> v(n);check(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));return v;}
};
struct Stream {hipStream_t s=nullptr;Stream(){check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));}~Stream(){static_cast<void>(hipStreamDestroy(s));}};
// Deterministic Q8_0 rows: scale in [2^-9,2^-6], codes in [-127,127].
__global__ void fill_q8(std::uint8_t* head,std::uint64_t blocks,std::uint32_t seed) {
    for(std::uint64_t b=blockIdx.x*static_cast<std::uint64_t>(blockDim.x)+threadIdx.x;b<blocks;b+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        auto x=static_cast<std::uint32_t>(b*2654435761ULL)^seed;
        auto next=[&]{x^=x<<13U;x^=x>>17U;x^=x<<5U;return x;};
        auto* block=head+b*34U;
        *reinterpret_cast<__half*>(block)=__float2half(ldexpf(1.0F+static_cast<float>(next()%1024U)/1024.0F,-9-static_cast<int>(next()%4U)));
        for(unsigned i=0;i<32;++i) block[2+i]=static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(next()%255U)-127));
    }
}
std::uint16_t bf16(float f){const auto b=std::bit_cast<std::uint32_t>(f);return static_cast<std::uint16_t>((b+0x7fffU+((b>>16U)&1U))>>16U);}
std::vector<std::uint16_t> hidden_values(std::uint32_t h,std::uint32_t seed) {
    std::mt19937 g(seed);std::normal_distribution<float> n(0.0F,1.0F);std::vector<std::uint16_t> v(h);
    for(auto& x:v)x=bf16(n(g));return v;
}
std::uint32_t key(float v){auto b=std::bit_cast<std::uint32_t>(v);if((b&0x7fffffffU)==0)b=0;return (b&0x80000000U)?~b:(b^0x80000000U);}
// Host reference: finite first-max with lowest real token id on ties.
std::uint32_t reference(const std::vector<float>& logits,const std::vector<std::uint32_t>& ids) {
    std::uint32_t best=UINT32_MAX;std::uint32_t best_key=0;
    for(std::size_t i=0;i<logits.size();++i) {
        const auto id=ids.empty()?static_cast<std::uint32_t>(i):ids[i];
        const auto k=key(logits[i]);
        if(best==UINT32_MAX || k>best_key || (k==best_key && id<best)) {best=id;best_key=k;}
    }
    return best;
}
struct Fixture {
    std::uint32_t h,v;Stream stream;Device<std::uint8_t> head;
    Fixture(std::uint32_t hidden,std::uint32_t vocab,std::uint32_t seed):h(hidden),v(vocab),head(static_cast<std::size_t>(vocab)*hidden/32U*34U) {
        hipLaunchKernelGGL(fill_q8,dim3(1024),dim3(256),0,stream.s,head.p,static_cast<std::uint64_t>(vocab)*hidden/32U,seed);
        check(hipGetLastError());check(hipStreamSynchronize(stream.s));
    }
    void copy_row(std::uint32_t to,std::uint32_t from) {
        const auto bytes=h/32U*34U;check(hipMemcpy(head.p+static_cast<std::size_t>(to)*bytes,head.p+static_cast<std::size_t>(from)*bytes,bytes,hipMemcpyDeviceToDevice));
    }
    std::vector<float> full_logits(const std::uint16_t* hidden) {
        Device<float> out(v);strix::launch_hip_q8_head_rows(hidden,head.p,h,v,out.p,stream.s);check(hipStreamSynchronize(stream.s));return out.read();
    }
};
int passed=0;
void case_named(const char* name,bool ok){require(ok,std::string("FAIL ")+name);++passed;std::cout<<"ok "<<name<<"\n";}
void exactness(std::uint32_t h,std::uint32_t v,std::uint32_t base_count,std::uint32_t cap,std::uint32_t depth,std::uint32_t seed) {
    Fixture f(h,v,seed);
    std::mt19937 g(seed);std::vector<std::uint32_t> base(base_count);
    for(auto& x:base)x=g()%v;base.push_back(base.front());                              // duplicate id
    HipMtpDraftHeadConfig full_config{h,v,depth,true,{},0};
    HipMtpDraftHeadConfig subset_config{h,v,depth,false,base,cap};
    HipMtpDraftHead full(f.head.p,full_config,f.stream.s),subset(f.head.p,subset_config,f.stream.s);
    std::vector<std::uint32_t> context(cap+7U);for(auto& x:context)x=g()%v;
    subset.observe(context);check(hipStreamSynchronize(f.stream.s));
    const auto stats=subset.stats();
    Device<std::uint32_t> ids_probe(1);
    Device<std::uint16_t> hidden(static_cast<std::size_t>(h)*depth);
    full.begin_chain();subset.begin_chain();
    std::vector<std::vector<float>> references;
    for(std::uint32_t k=0;k<depth;++k) {
        const auto x=hidden_values(h,seed+k);check(hipMemcpy(hidden.p+k*h,x.data(),h*2U,hipMemcpyHostToDevice));
        references.push_back(f.full_logits(hidden.p+k*h));
        full.launch(hidden.p+k*h,k,f.stream.s);subset.launch(hidden.p+k*h,k,f.stream.s);
    }
    check(hipStreamSynchronize(f.stream.s));
    const auto full_tokens=full.read(depth),subset_tokens=subset.read(depth);
    // Compact logits of the last step against the full head rows.
    std::vector<std::uint32_t> ids(stats.rows);std::vector<float> compact(stats.rows),full_last(v);
    check(hipMemcpy(ids.data(),subset.ids(),stats.rows*4U,hipMemcpyDeviceToHost));
    check(hipMemcpy(compact.data(),subset.logits(),stats.rows*4U,hipMemcpyDeviceToHost));
    check(hipMemcpy(full_last.data(),full.logits(),v*4U,hipMemcpyDeviceToHost));
    bool bits=true;for(std::uint32_t j=0;j<stats.rows;++j)bits=bits&&std::bit_cast<std::uint32_t>(compact[j])==std::bit_cast<std::uint32_t>(references.back()[ids[j]]);
    bool full_bits=std::memcmp(full_last.data(),references.back().data(),v*4U)==0;
    bool tokens=true;
    for(std::uint32_t k=0;k<depth;++k) {
        std::vector<float> sub(stats.rows);for(std::uint32_t j=0;j<stats.rows;++j)sub[j]=references[k][ids[j]];
        tokens=tokens && full_tokens[k]==reference(references[k],{}) && subset_tokens[k]==reference(sub,ids);
        std::uint32_t slot=0;check(hipMemcpy(&slot,subset.device_token(k),4,hipMemcpyDeviceToHost));tokens=tokens&&slot==subset_tokens[k];
    }
    std::vector<std::uint32_t> unique(base);std::sort(unique.begin(),unique.end());unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
    char name[160];std::snprintf(name,sizeof(name),"h%u v%u base%u cap%u depth%u",h,v,base_count,cap,depth);
    case_named((std::string(name)+" subset rows bit-exact").c_str(),bits);
    case_named((std::string(name)+" full head bit-exact").c_str(),full_bits);
    case_named((std::string(name)+" argmax/slots").c_str(),tokens);
    case_named((std::string(name)+" base dedup + context cap").c_str(),stats.base_rows==unique.size() && stats.context_rows<=cap &&
        stats.rows==stats.base_rows+stats.context_rows && (stats.context_rows==cap || stats.context_dropped==0));
}
void ties_and_nonfinite() {
    constexpr std::uint32_t h=256,v=4096;Fixture f(h,v,99);
    Device<std::uint16_t> hidden(h);const auto x=hidden_values(h,5);check(hipMemcpy(hidden.p,x.data(),h*2U,hipMemcpyHostToDevice));
    auto logits=f.full_logits(hidden.p);const auto top=reference(logits,{});
    // Duplicate the winning row to lower and higher ids: lowest id must win, in either order of rows.
    f.copy_row(17,top);f.copy_row(4000,top);
    {
        HipMtpDraftHead full(f.head.p,{h,v,1,true,{},0},f.stream.s);full.begin_chain();full.launch(hidden.p,0,f.stream.s);
        check(hipStreamSynchronize(f.stream.s));case_named("full tie picks lowest id",full.read(1)[0]==std::min(17U,top));
        HipMtpDraftHead sub(f.head.p,{h,v,1,false,{4000,top,17,1},0},f.stream.s);sub.begin_chain();sub.launch(hidden.p,0,f.stream.s);
        check(hipStreamSynchronize(f.stream.s));case_named("subset tie picks lowest real id",sub.read(1)[0]==std::min(17U,top));
        HipMtpDraftHead high(f.head.p,{h,v,1,false,{4000,2},0},f.stream.s);high.begin_chain();high.launch(hidden.p,0,f.stream.s);
        check(hipStreamSynchronize(f.stream.s));case_named("subset maps compact index to id",high.read(1)[0]==4000U);
    }
    // Non-finite scale in a subset row poisons the step; rows outside the subset are not read.
    const __half inf=__float2half(INFINITY);check(hipMemcpy(f.head.p+33U*(h/32U*34U),&inf,2,hipMemcpyHostToDevice));
    HipMtpDraftHead clean(f.head.p,{h,v,2,false,{1,2,3},0},f.stream.s),bad(f.head.p,{h,v,2,false,{1,33,3},0},f.stream.s);
    clean.begin_chain();bad.begin_chain();clean.launch(hidden.p,0,f.stream.s);bad.launch(hidden.p,0,f.stream.s);
    check(hipStreamSynchronize(f.stream.s));
    bool threw=false;try{(void)bad.read(1);}catch(const std::exception&){threw=true;}
    case_named("non-finite subset row rejected",threw && clean.read(1).size()==1U);
    // Status of later steps does not affect an earlier prefix read.
    bad.begin_chain();clean.launch(hidden.p,0,f.stream.s);bad.launch(hidden.p,1,f.stream.s);check(hipStreamSynchronize(f.stream.s));
    threw=false;try{(void)bad.read(2);}catch(const std::exception&){threw=true;}
    case_named("non-finite step scoped to its prefix",threw && bad.read(1).size()==1U);
    // Token file roundtrip and rejection.
    const std::string path="/tmp/strix-draft-vocab-test.u32";
    {std::ofstream out(path,std::ios::binary);const std::uint32_t ids[]={0,5,4095};out.write(reinterpret_cast<const char*>(ids),sizeof(ids));}
    const auto ids=HipMtpDraftHead::read_token_file(path,v);
    threw=false;try{(void)HipMtpDraftHead::read_token_file(path,4095);}catch(const std::exception&){threw=true;}
    case_named("token file roundtrip/reject",ids==std::vector<std::uint32_t>{0,5,4095} && threw);
    std::remove(path.c_str());
}
// Q4_0 subset: GPU logits against host Q4_0 requantization of the same Q8 rows
// (nibble ties may differ by one code), plus top-1 agreement with the Q8 head.
void q4_subset() {
    constexpr std::uint32_t h=2560,v=248320,count=2000;Fixture f(h,v,21);
    std::mt19937 g(4);std::vector<std::uint32_t> base(count);for(auto& x:base)x=g()%v;
    std::sort(base.begin(),base.end());base.erase(std::unique(base.begin(),base.end()),base.end());std::shuffle(base.begin(),base.end(),g);
    HipMtpDraftHeadConfig q8_config{h,v,1,false,base,0},q4_config{h,v,1,false,base,0,true};
    HipMtpDraftHead q8(f.head.p,q8_config,f.stream.s),q4(f.head.p,q4_config,f.stream.s);
    const auto rb=h/32U*34U;Device<std::uint16_t> hidden(h);
    double worst=0;std::uint32_t agree=0;constexpr std::uint32_t trials=16;
    for(std::uint32_t t=0;t<trials;++t) {
        const auto x=hidden_values(h,100+t);check(hipMemcpy(hidden.p,x.data(),h*2U,hipMemcpyHostToDevice));
        q8.begin_chain();q4.begin_chain();q8.launch(hidden.p,0,f.stream.s);q4.launch(hidden.p,0,f.stream.s);check(hipStreamSynchronize(f.stream.s));
        agree+=q8.read(1)[0]==q4.read(1)[0];
        if(t) continue;
        std::vector<float> logits(base.size());check(hipMemcpy(logits.data(),q4.logits(),base.size()*4U,hipMemcpyDeviceToHost));
        std::vector<std::uint8_t> row(rb);
        for(std::size_t j=0;j<base.size();++j) {
            check(hipMemcpy(row.data(),f.head.p+static_cast<std::size_t>(base[j])*rb,rb,hipMemcpyDeviceToHost));
            double sum=0,magnitude=0;
            for(std::uint32_t b=0;b<h/32U;++b) {
                __half hs;std::memcpy(&hs,row.data()+b*34U,2);const float scale=__half2float(hs);
                float values[32],max=0,amax=0;
                for(unsigned i=0;i<32;++i){values[i]=scale*static_cast<float>(static_cast<std::int8_t>(row[b*34U+2U+i]));if(std::fabs(values[i])>amax){amax=std::fabs(values[i]);max=values[i];}}
                const float d=max/-8.0F,inverse=d!=0.0F?1.0F/d:0.0F;const float dh=__half2float(__float2half(d));
                for(unsigned i=0;i<32;++i){
                    const int q=static_cast<int>(std::fmin(15.0F,values[i]*inverse+8.5F))-8;
                    const double xv=std::bit_cast<float>(static_cast<std::uint32_t>(x[b*32U+i])<<16U);
                    sum+=static_cast<double>(dh)*q*xv;magnitude+=std::fabs(static_cast<double>(values[i])*xv);
                }
            }
            worst=std::max(worst,std::fabs(sum-logits[j])/std::max(magnitude,1e-30));
        }
    }
    std::printf("q4 subset: max |gpu-host|/sum|w x| %.3g, top-1 agreement with Q8 %u/%u\n",worst,agree,trials);
    case_named("Q4 subset logits match host Q4_0 requantization",worst<2e-3);
}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
void bench(std::uint32_t base_rows,std::uint32_t context_rows) {
    constexpr std::uint32_t h=2560,v=248320,depth=3;constexpr int runs=30;
    Fixture f(h,v,1234);
    Device<std::uint16_t> hidden(h);const auto x=hidden_values(h,3);check(hipMemcpy(hidden.p,x.data(),h*2U,hipMemcpyHostToDevice));
    Device<float> logits(v);
    auto guard=std::make_shared<strix::HipSetupQuiescenceGuard>();auto owner=std::make_shared<int>(0);
    strix::HipGreedyOutput selector(v,{1U,reinterpret_cast<std::uintptr_t>(logits.p),v*4ULL,1U},owner,guard);
    std::uint64_t epoch=0;
    hipEvent_t a,b;check(hipEventCreate(&a));check(hipEventCreate(&b));
    auto gpu=[&](auto&& body){std::vector<double> t;for(int r=0;r<runs+3;++r){check(hipEventRecord(a,f.stream.s));body();check(hipEventRecord(b,f.stream.s));
        check(hipEventSynchronize(b));float ms=0;check(hipEventElapsedTime(&ms,a,b));if(r>=3)t.push_back(ms);}return median(t);};
    auto wall=[&](auto&& body){std::vector<double> t;for(int r=0;r<runs+3;++r){check(hipStreamSynchronize(f.stream.s));
        const auto s=std::chrono::steady_clock::now();body();t.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count());}
        t.erase(t.begin(),t.begin()+3);return median(t);};
    std::vector<std::uint32_t> base(base_rows),context(context_rows);
    for(std::uint32_t i=0;i<base_rows;++i)base[i]=i;                               // merge-rank order prefix
    std::mt19937 g(7);for(auto& c:context)c=base_rows+g()%(v-base_rows);
    HipMtpDraftHead full(f.head.p,{h,v,depth,true,{},0},f.stream.s);
    HipMtpDraftHead subset(f.head.p,{h,v,depth,false,base,context_rows},f.stream.s);
    subset.observe(context);check(hipStreamSynchronize(f.stream.s));
    const auto rows=subset.stats().rows;
    const double head_full=gpu([&]{strix::launch_hip_q8_head_rows(hidden.p,f.head.p,h,v,logits.p,f.stream.s);});
    const double draft_full=gpu([&]{full.launch(hidden.p,0,f.stream.s);});
    const double draft_subset=gpu([&]{subset.launch(hidden.p,0,f.stream.s);});
    HipMtpDraftHeadConfig q4_config{h,v,depth,false,base,context_rows,true};HipMtpDraftHead q4(f.head.p,q4_config,f.stream.s);
    q4.observe(context);check(hipStreamSynchronize(f.stream.s));
    const double draft_q4=gpu([&]{q4.launch(hidden.p,0,f.stream.s);});
    // Before: per draft, head on the MTP stream, completion, then the host selector (own stream + sync).
    const double before=wall([&]{for(std::uint32_t k=0;k<depth;++k){
        strix::launch_hip_q8_head_rows(hidden.p,f.head.p,h,v,logits.p,f.stream.s);check(hipStreamSynchronize(f.stream.s));
        (void)selector.select({{},reinterpret_cast<std::uintptr_t>(logits.p),v*4ULL,1U},v,k,++epoch);}})/depth;
    const double chain_full=wall([&]{full.begin_chain();for(std::uint32_t k=0;k<depth;++k)full.launch(hidden.p,k,f.stream.s);
        check(hipStreamSynchronize(f.stream.s));(void)full.read(depth);})/depth;
    const double chain_subset=wall([&]{subset.begin_chain();for(std::uint32_t k=0;k<depth;++k)subset.launch(hidden.p,k,f.stream.s);
        check(hipStreamSynchronize(f.stream.s));(void)subset.read(depth);})/depth;
    std::printf("bench h=%u v=%u subset_rows=%u depth=%u median of %d (ms per draft)\n",h,v,rows,depth,runs);
    std::printf("  gpu  full Q8 head kernel             %.4f\n",head_full);
    std::printf("  gpu  draft head full + device argmax %.4f\n",draft_full);
    std::printf("  gpu  draft head subset + argmax      %.4f\n",draft_subset);
    std::printf("  gpu  draft head Q4 subset + argmax   %.4f\n",draft_q4);
    std::printf("  wall before: full head + sync + host selector  %.4f\n",before);
    std::printf("  wall chain full head, one sync               %.4f\n",chain_full);
    std::printf("  wall chain subset head, one sync             %.4f\n",chain_subset);
    check(hipEventDestroy(a));check(hipEventDestroy(b));
}
}
int main(int argc,char** argv) {
    try {
        int devices=0;if(hipGetDeviceCount(&devices)!=hipSuccess || !devices){std::cout<<"skip: no HIP device\n";return 77;}
        if(argc>1 && std::string(argv[1])=="--bench") {
            for(const auto [b,c]:{std::pair{32768U,4096U},std::pair{16384U,4096U},std::pair{65536U,4096U}})bench(b,c);return 0;
        }
        exactness(2560,248320,32768,4096,3,11);
        exactness(2560,248320,1000,0,8,12);
        exactness(64,5000,3,40,2,13);
        exactness(32,300,250,8,4,14);
        ties_and_nonfinite();
        q4_subset();
        std::cout<<"passed "<<passed<<"\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
