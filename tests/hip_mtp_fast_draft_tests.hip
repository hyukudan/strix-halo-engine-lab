// Draft-only MTP kernels: the 2-launch Q8 MoE and the fused input projection
// against the exact launch sequences and a double host reference at the
// canonical sidecar geometry, plus a per-draft benchmark (--bench).
#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/hip_mtp_input_fusion.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool x,const std::string& m){if(!x)throw std::runtime_error(m);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
template<class T> struct Device {
    T* p=nullptr;std::size_t n=0;
    explicit Device(std::size_t count):n(count){check(hipMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));check(hipMemset(p,0,count*sizeof(T)));}
    ~Device(){if(p)static_cast<void>(hipFree(p));}
    Device(const Device&)=delete;
    std::vector<T> read() const {std::vector<T> v(n);check(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));return v;}
    void write(const std::vector<T>& v){require(v.size()==n,"upload extent");check(hipMemcpy(p,v.data(),n*sizeof(T),hipMemcpyHostToDevice));}
    std::uintptr_t address() const {return reinterpret_cast<std::uintptr_t>(p);}
};
constexpr std::uint32_t H=2560,I=640,E=512,HC=4;
__global__ void fill_q8(std::uint8_t* q,std::uint64_t blocks,std::uint32_t seed,int exponent) {
    for(std::uint64_t b=blockIdx.x*static_cast<std::uint64_t>(blockDim.x)+threadIdx.x;b<blocks;b+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        auto x=static_cast<std::uint32_t>(b*2654435761ULL)^seed;
        auto next=[&]{x^=x<<13U;x^=x>>17U;x^=x<<5U;return x;};
        auto* block=q+b*34U;
        *reinterpret_cast<__half*>(block)=__float2half(ldexpf(1.0F+static_cast<float>(next()%1024U)/1024.0F,exponent-static_cast<int>(next()%2U)));
        for(unsigned i=0;i<32;++i) block[2+i]=static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(next()%255U)-127));
    }
}
__global__ void combine(const std::uint16_t* a,const std::uint16_t* b,std::uint16_t* out,std::uint32_t n) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const float v=__uint_as_float(static_cast<unsigned>(a[i])<<16U)+__uint_as_float(static_cast<unsigned>(b[i])<<16U);
    const auto bits=__float_as_uint(v);out[i]=static_cast<std::uint16_t>((bits+0x7fffU+((bits>>16U)&1U))>>16U);
}
std::uint16_t bf16(float f){const auto b=std::bit_cast<std::uint32_t>(f);return static_cast<std::uint16_t>((b+0x7fffU+((b>>16U)&1U))>>16U);}
float value(std::uint16_t b){return std::bit_cast<float>(static_cast<std::uint32_t>(b)<<16U);}
std::vector<std::uint8_t> host(const Device<std::uint8_t>& d,std::uint64_t offset,std::uint64_t bytes){
    std::vector<std::uint8_t> v(bytes);check(hipMemcpy(v.data(),d.p+offset,bytes,hipMemcpyDeviceToHost));return v;
}
double q8(const std::uint8_t* row,std::uint32_t f){
    __half h;std::memcpy(&h,row+(f/32)*34,2);return static_cast<double>(__half2float(h))*static_cast<std::int8_t>(row[(f/32)*34+2+f%32]);
}
double dot(const std::uint8_t* row,const std::vector<double>& x){double s=0;for(std::uint32_t f=0;f<x.size();++f)s+=q8(row,f)*x[f];return s;}
std::vector<double> ffn(const std::uint8_t* g,const std::uint8_t* u,const std::vector<double>& x){
    std::vector<double> a(I);const auto rb=H/32*34;
    for(std::uint32_t j=0;j<I;++j){const double gs=dot(g+j*rb,x),us=dot(u+j*rb,x);a[j]=gs/(1+std::exp(-gs))*us;}
    return a;
}
struct Moe {
    Device<std::uint8_t> gate{static_cast<std::size_t>(E)*I*H/32*34},up{gate.n},down{static_cast<std::size_t>(E)*H*I/32*34};
    Device<std::uint8_t> sgate{static_cast<std::size_t>(I)*H/32*34},sup{sgate.n},sdown{static_cast<std::size_t>(H)*I/32*34};
    Device<float> scalar{H},rw{10},temp{3840},sws{1280},ws{strix::hip_mtp_fast_moe_workspace_floats};
    Device<std::uint32_t> routes{10},status{2};
    Device<std::uint16_t> input{H},shared_out{H},routed_out{H},exact{H},fast{H};
    std::vector<std::uint16_t> x;std::vector<std::uint32_t> r;std::vector<float> w,sc;
    hipStream_t s=nullptr;
    explicit Moe(std::uint32_t seed){
        check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
        int k=0;for(auto* d:{&gate,&up,&down,&sgate,&sup,&sdown}){
            hipLaunchKernelGGL(fill_q8,dim3(2048),dim3(256),0,s,d->p,static_cast<std::uint64_t>(d->n/34),seed+static_cast<std::uint32_t>(k),k%3==2?-10:-9);++k;}
        check(hipStreamSynchronize(s));
        std::mt19937 g(seed);std::normal_distribution<float> n(0,1);
        x.resize(H);for(auto& v:x)v=bf16(n(g));input.write(x);
        sc.resize(H);for(auto& v:sc)v=0.02F*n(g);scalar.write(sc);
        std::vector<std::uint32_t> all(E);for(std::uint32_t i=0;i<E;++i)all[i]=i;std::shuffle(all.begin(),all.end(),g);
        r.assign(all.begin(),all.begin()+10);routes.write(r);
        w.resize(10);float t=0;for(auto& v:w){v=std::exp(n(g));t+=v;}for(auto& v:w)v/=t;rw.write(w);
    }
    ~Moe(){static_cast<void>(hipStreamDestroy(s));}
    void run_exact(){
        strix::enqueue_hip_mtp_shared_expert_q8(input.p,sgate.p,sup.p,sdown.p,scalar.p,sws.p,shared_out.p,s);
        strix::enqueue_hip_mtp_routed_experts_q8(input.p,routes.p,rw.p,gate.p,up.p,down.p,temp.p,status.p,routed_out.p,s);
        hipLaunchKernelGGL(combine,dim3(H/256),dim3(256),0,s,shared_out.p,routed_out.p,exact.p,H);check(hipGetLastError());
    }
    strix::HipMtpFastMoeViews views(std::uint16_t* out){
        return {input.p,routes.p,rw.p,gate.p,up.p,down.p,sgate.p,sup.p,sdown.p,scalar.p,ws.p,out,status.p};
    }
    void run_fast(std::uint16_t* out){strix::launch_hip_mtp_fast_moe(views(out),s);}
    std::vector<double> reference(){
        std::vector<double> xd(H);for(std::uint32_t i=0;i<H;++i)xd[i]=value(x[i]);
        const auto gb=static_cast<std::uint64_t>(I)*H/32*34,db=static_cast<std::uint64_t>(H)*I/32*34;
        std::vector<double> out(H,0.0);
        auto add=[&](const std::vector<std::uint8_t>& g,const std::vector<std::uint8_t>& u,const std::vector<std::uint8_t>& d,double weight){
            const auto a=ffn(g.data(),u.data(),xd);for(std::uint32_t row=0;row<H;++row)out[row]+=weight*dot(d.data()+row*(I/32*34),a);
        };
        for(std::uint32_t k=0;k<10;++k)add(host(gate,r[k]*gb,gb),host(up,r[k]*gb,gb),host(down,r[k]*db,db),w[k]);
        double sg=0;for(std::uint32_t i=0;i<H;++i)sg+=static_cast<double>(sc[i])*xd[i];
        add(host(sgate,0,gb),host(sup,0,gb),host(sdown,0,db),1/(1+std::exp(-sg)));
        return out;
    }
};
int passed=0;
void ok(const char* name,bool v){require(v,std::string("FAIL ")+name);++passed;std::cout<<"ok "<<name<<"\n";}
// Error in units of the BF16 half-ulp of the reference magnitude scale.
double max_error(const std::vector<std::uint16_t>& got,const std::vector<double>& ref,double& scale){
    scale=0;for(auto v:ref)scale=std::max(scale,std::abs(v));double e=0;
    for(std::size_t i=0;i<ref.size();++i)e=std::max(e,std::abs(static_cast<double>(value(got[i]))-ref[i]));return e;
}
void moe_cases(){
    for(const std::uint32_t seed:{1U,2U}){
        Moe m(seed);m.run_exact();m.run_fast(m.fast.p);check(hipStreamSynchronize(m.s));
        const auto ref=m.reference();double scale=0;
        const double exact=max_error(m.exact.read(),ref,scale),fast=max_error(m.fast.read(),ref,scale);
        std::printf("moe seed %u: |ref|max %.4g exact err %.3g fast err %.3g\n",seed,scale,exact,fast);
        ok("fast MoE within exact-path error budget",fast<=std::max(2.0*exact,scale*std::ldexp(1.0,-8)));
        std::uint32_t st[2];check(hipMemcpy(st,m.status.p,8,hipMemcpyDeviceToHost));ok("valid routes keep status clear",!st[0]);
        // In-place output (input aliases output) as the runtime uses it.
        check(hipMemcpy(m.routed_out.p,m.input.p,H*2,hipMemcpyDeviceToDevice));
        strix::launch_hip_mtp_fast_moe({m.routed_out.p,m.routes.p,m.rw.p,m.gate.p,m.up.p,m.down.p,m.sgate.p,m.sup.p,m.sdown.p,m.scalar.p,m.ws.p,m.routed_out.p,m.status.p},m.s);
        check(hipStreamSynchronize(m.s));ok("in-place output equals separate output",m.routed_out.read()==m.fast.read());
        // Invalid route: status set, output untouched.
        auto bad=m.r;bad[3]=E;m.routes.write(bad);check(hipMemset(m.fast.p,0x5a,H*2));
        m.run_fast(m.fast.p);check(hipStreamSynchronize(m.s));
        check(hipMemcpy(st,m.status.p,8,hipMemcpyDeviceToHost));
        const auto untouched=m.fast.read();
        ok("invalid route sets status and skips output",(st[0]&1U) && std::all_of(untouched.begin(),untouched.end(),[](std::uint16_t v){return v==0x5a5aU;}));
        check(hipMemset(m.status.p,0,8));m.routes.write(m.r);
        auto nan_w=m.w;nan_w[0]=NAN;m.rw.write(nan_w);m.run_fast(m.fast.p);check(hipStreamSynchronize(m.s));
        check(hipMemcpy(st,m.status.p,8,hipMemcpyDeviceToHost));ok("non-finite route weight sets status",(st[0]&1U)!=0);
    }
}
struct Fusion {
    Device<std::uint16_t> embedding{H},wide{H*HC},out_exact{H*HC},out_fast{H*HC};
    Device<float> enorm{H},hnorm{H*HC},normalized{H*(HC+1)},projected{H*(HC+1)};
    Device<std::uint8_t> packed{static_cast<std::size_t>(H)*(2*H/32*34)};
    hipStream_t s=nullptr;
    Fusion(){
        check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
        hipLaunchKernelGGL(fill_q8,dim3(2048),dim3(256),0,s,packed.p,static_cast<std::uint64_t>(packed.n/34),77U,-9);check(hipStreamSynchronize(s));
        std::mt19937 g(5);std::normal_distribution<float> n(0,1);
        std::vector<std::uint16_t> e(H),w(H*HC);for(auto& v:e)v=bf16(n(g));for(auto& v:w)v=bf16(n(g));embedding.write(e);wide.write(w);
        std::vector<float> a(H),b(H*HC);for(auto& v:a)v=1+0.1F*n(g);for(auto& v:b)v=1+0.1F*n(g);enorm.write(a);hnorm.write(b);
    }
    ~Fusion(){static_cast<void>(hipStreamDestroy(s));}
    strix::HipMtpInputFusionRequest request(Device<std::uint16_t>& out,strix::HipMtpInputRounding projection){
        auto v=[](auto& d){return strix::HipMtpInputFusionView{d.address(),d.n*sizeof(*d.p),1U};};
        strix::HipMtpInputFusionRequest r;r.hidden_size=H;r.hc_count=HC;r.epsilon=1e-6F;
        r.embedding_storage=r.hidden_storage=r.output_storage=strix::HipMtpInputStorage::bf16;
        r.normalized_rounding=strix::HipMtpInputRounding::fp32;r.projection_rounding=projection;
        r.norm_weight_encoding=strix::MtpNormWeightEncoding::effective_scale;r.norm_encoding_admitted=true;
        r.arithmetic=strix::MtpInputFusionArithmetic::split_projections_then_add;
        r.embedding=v(embedding);r.wide_hidden=v(wide);r.embedding_norm=v(enorm);r.hidden_norm=v(hnorm);r.packed_projection=v(packed);
        r.projection_row_stride_bytes=2*H/32*34;r.normalized_scratch=v(normalized);r.projection_scratch=v(projected);r.output=v(out);return r;
    }
};
void fusion_cases(){
    Fusion f;
    for(const auto rounding:{strix::HipMtpInputRounding::fp32,strix::HipMtpInputRounding::bf16_rne}){
        strix::launch_hip_mtp_input_fusion(f.request(f.out_exact,rounding),f.s);
        strix::launch_hip_mtp_input_fusion_draft(f.request(f.out_fast,rounding),f.s);check(hipStreamSynchronize(f.s));
        const auto a=f.out_exact.read(),b=f.out_fast.read();
        // Different summation order only. Each output adds two separately rounded
        // projections, so a one-ulp operand change can be many ulps of a cancelling
        // sum: bound the absolute difference by two BF16 ulps of the output scale.
        double scale=0,worst=0;std::uint32_t differ=0;
        for(std::size_t i=0;i<a.size();++i){scale=std::max(scale,std::abs(static_cast<double>(value(a[i]))));
            const double d=std::abs(static_cast<double>(value(a[i]))-value(b[i]));worst=std::max(worst,d);differ+=d!=0;}
        std::printf("input fusion rounding %u: %u/%zu differ, max |diff| %.3g (output scale %.3g)\n",static_cast<unsigned>(rounding),differ,a.size(),worst,scale);
        ok("draft input fusion within 2 bf16 ulp of output scale",worst<=scale*std::ldexp(1.0,-7));
    }
}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
template<class F> double time_gpu(hipStream_t s,F&& body){
    hipEvent_t a,b;check(hipEventCreate(&a));check(hipEventCreate(&b));std::vector<double> t;
    for(int r=0;r<33;++r){check(hipEventRecord(a,s));body();check(hipEventRecord(b,s));check(hipEventSynchronize(b));
        float ms=0;check(hipEventElapsedTime(&ms,a,b));if(r>=3)t.push_back(ms);}
    check(hipEventDestroy(a));check(hipEventDestroy(b));return median(t);
}
void bench(){
    Moe m(9);
    // Rotate expert sets so successive runs do not hit the same 52 MB in the 32 MB cache.
    std::vector<std::vector<std::uint32_t>> sets;std::mt19937 g(3);
    for(int i=0;i<8;++i){std::vector<std::uint32_t> all(E);for(std::uint32_t j=0;j<E;++j)all[j]=j;std::shuffle(all.begin(),all.end(),g);sets.emplace_back(all.begin(),all.begin()+10);}
    int k=0;auto next=[&]{check(hipMemcpyAsync(m.routes.p,sets[static_cast<std::size_t>(k++%8)].data(),40,hipMemcpyHostToDevice,m.s));};
    const double exact=time_gpu(m.s,[&]{next();m.run_exact();}),fast=time_gpu(m.s,[&]{next();m.run_fast(m.fast.p);});
    Fusion f;
    const double fe=time_gpu(f.s,[&]{strix::launch_hip_mtp_input_fusion(f.request(f.out_exact,strix::HipMtpInputRounding::bf16_rne),f.s);});
    const double ff=time_gpu(f.s,[&]{strix::launch_hip_mtp_input_fusion_draft(f.request(f.out_fast,strix::HipMtpInputRounding::bf16_rne),f.s);});
    std::printf("bench per draft, GPU median of 30 (ms)\n");
    std::printf("  MoE exact (shared 5 + routed 42 + combine 1 launches)  %.4f\n",exact);
    std::printf("  MoE fast  (2 launches)                                 %.4f\n",fast);
    std::printf("  input fusion exact (3 launches)                        %.4f\n",fe);
    std::printf("  input fusion draft (2 launches)                        %.4f\n",ff);
}
}
int main(int argc,char** argv){
    try{
        int devices=0;if(hipGetDeviceCount(&devices)!=hipSuccess||!devices){std::cout<<"skip: no HIP device\n";return 77;}
        if(argc>1&&std::string(argv[1])=="--bench"){bench();return 0;}
        fusion_cases();moe_cases();std::cout<<"passed "<<passed<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
