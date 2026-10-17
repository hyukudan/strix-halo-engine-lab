// Draft-only MTP kernels: the 2-launch Q8 MoE and the fused input projection
// against the exact launch sequences and a double host reference at the
// canonical sidecar geometry, plus a per-draft benchmark (--bench).
#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/hip_mtp_input_fusion.hpp"
#include "strix/hip_mtp_attention.hpp"
#include "strix/hip_ple_launcher.hpp"
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
// Attention-shaped Q8 GEMVs: draft wave-per-row against the exact projection and
// a double reference (BF16 in -> F32 out), and F32 in -> BF16 out against the reference.
struct Gemv {
    std::uint32_t width,rows;Device<std::uint8_t> w;Device<std::uint16_t> x;Device<float> xf,exact,fast;Device<std::uint16_t> fast_bf16;hipStream_t s=nullptr;
    Gemv(std::uint32_t in,std::uint32_t out,std::uint32_t seed):width(in),rows(out),w(static_cast<std::size_t>(out)*in/32*34),x(in),xf(in),exact(out),fast(out),fast_bf16(out){
        check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
        hipLaunchKernelGGL(fill_q8,dim3(2048),dim3(256),0,s,w.p,static_cast<std::uint64_t>(w.n/34),seed,-9);check(hipStreamSynchronize(s));
        std::mt19937 g(seed);std::normal_distribution<float> n(0,1);std::vector<std::uint16_t> v(in);std::vector<float> f(in);
        for(std::uint32_t i=0;i<in;++i){v[i]=bf16(n(g));f[i]=value(v[i]);}x.write(v);xf.write(f);
    }
    ~Gemv(){static_cast<void>(hipStreamDestroy(s));}
};
void gemv_cases(){
    for(const auto [in,out]:{std::pair{H,12288U},std::pair{H,512U},std::pair{6144U,H}}){
        Gemv m(in,out,in+out);
        strix::enqueue_hip_mtp_attention_q8_linear(m.x.p,m.w.p,in,out,m.exact.p,m.s);
        strix::launch_hip_mtp_draft_q8_gemv(m.x.p,true,m.w.p,in,out,m.fast.p,false,m.s);
        strix::launch_hip_mtp_draft_q8_gemv(m.xf.p,false,m.w.p,in,out,m.fast_bf16.p,true,m.s);check(hipStreamSynchronize(m.s));
        const auto e=m.exact.read(),f=m.fast.read();const auto b=m.fast_bf16.read();
        const auto weights=host(m.w,0,m.w.n);std::vector<std::uint16_t> xs(in);check(hipMemcpy(xs.data(),m.x.p,in*2,hipMemcpyDeviceToHost));
        std::vector<double> xd(in);for(std::uint32_t i=0;i<in;++i)xd[i]=value(xs[i]);
        double scale=0,ee=0,fe=0,be=0;
        for(std::uint32_t r=0;r<out;++r){const double ref=dot(weights.data()+static_cast<std::size_t>(r)*(in/32*34),xd);scale=std::max(scale,std::abs(ref));
            ee=std::max(ee,std::abs(e[r]-ref));fe=std::max(fe,std::abs(f[r]-ref));be=std::max(be,std::abs(value(b[r])-ref));}
        std::printf("gemv %ux%u: |ref|max %.4g exact err %.3g draft err %.3g draft->bf16 err %.3g\n",out,in,scale,ee,fe,be);
        ok("draft Q8 GEMV F32 within exact error budget",fe<=std::max(2*ee,scale*1e-6));
        ok("draft Q8 GEMV BF16 output within half a BF16 ulp of scale",be<=scale*std::ldexp(1.0,-8)+fe);
    }
}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
template<class F> double time_gpu(hipStream_t s,F&& body){
    hipEvent_t a,b;check(hipEventCreate(&a));check(hipEventCreate(&b));std::vector<double> t;
    for(int r=0;r<33;++r){check(hipEventRecord(a,s));body();check(hipEventRecord(b,s));check(hipEventSynchronize(b));
        float ms=0;check(hipEventElapsedTime(&ms,a,b));if(r>=3)t.push_back(ms);}
    check(hipEventDestroy(a));check(hipEventDestroy(b));return median(t);
}
// MTP HC mix: exact staged path vs draft split-K/lane-chain path vs a double reference.
struct Hyper {
    static constexpr std::uint32_t W=H*HC,R=320;
    Device<std::uint16_t> input{W},exact{H},draft{H};Device<float> norm{W},ws{W+R},inj_exact{HC},inj_draft{HC},inject_f32{HC*W};
    Device<std::uint8_t> down{static_cast<std::size_t>(R)*W/32*34},inject{static_cast<std::size_t>(HC)*W/32*34},up{static_cast<std::size_t>(W)*R/32*34};
    hipStream_t s=nullptr;std::vector<std::uint16_t> x;std::vector<float> g;
    explicit Hyper(std::uint32_t seed){
        check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
        int k=0;for(auto* d:{&down,&inject,&up}){hipLaunchKernelGGL(fill_q8,dim3(512),dim3(256),0,s,d->p,static_cast<std::uint64_t>(d->n/34),seed+static_cast<std::uint32_t>(k),k==1?-12:-8);++k;}
        strix::launch_hip_mtp_dequantize_q8(inject.p,HC*W,inject_f32.p,s);check(hipStreamSynchronize(s));
        std::mt19937 r(seed);std::normal_distribution<float> n(0,1);x.resize(W);g.resize(W);
        for(auto& v:x)v=bf16(3*n(r));for(auto& v:g)v=1+0.2F*n(r);input.write(x);norm.write(g);
    }
    ~Hyper(){static_cast<void>(hipStreamDestroy(s));}
    void run(bool use_draft,bool with_inject,std::uint16_t* out,float* injection){
        strix::enqueue_hip_mtp_hyper_mix_q8(input.p,norm.p,down.p,with_inject?(use_draft?static_cast<const void*>(inject_f32.p):inject.p):nullptr,up.p,ws.p,
            with_inject?injection:nullptr,out,1e-6F,use_draft,s);
    }
    std::pair<std::vector<double>,std::vector<double>> reference(){
        const auto d=host(down,0,down.n),i=host(inject,0,inject.n),u=host(up,0,up.n);
        std::vector<double> nx(W);
        for(std::uint32_t t=0;t<HC;++t){double ss=0;for(std::uint32_t f=0;f<H;++f){const double v=value(x[t*H+f]);ss+=v*v;}
            const double inv=1/std::sqrt(ss/H+1e-6);for(std::uint32_t f=0;f<H;++f)nx[t*H+f]=value(x[t*H+f])*inv*g[t*H+f];}
        auto sig=[](double v){return 1/(1+std::exp(-v));};
        std::vector<double> low(R),inj(HC),out(H,0);
        for(std::uint32_t o=0;o<R;++o){const double v=dot(d.data()+static_cast<std::size_t>(o)*(W/32*34),nx)/HC;low[o]=v*sig(v);}
        for(std::uint32_t o=0;o<HC;++o)inj[o]=2*sig(dot(i.data()+static_cast<std::size_t>(o)*(W/32*34),nx)/HC);
        for(std::uint32_t f=0;f<H;++f){for(std::uint32_t t=0;t<HC;++t){const double gate=dot(u.data()+static_cast<std::size_t>(t*H+f)*(R/32*34),low);out[f]+=nx[t*H+f]*sig(gate);}out[f]/=HC;}
        return {out,inj};
    }
};
void hyper_cases(){
    for(const bool with_inject:{true,false}){
        Hyper m(with_inject?31U:32U);
        m.run(false,with_inject,m.exact.p,m.inj_exact.p);m.run(true,with_inject,m.draft.p,m.inj_draft.p);check(hipStreamSynchronize(m.s));
        const auto [ref,inj]=m.reference();double scale=0;
        const double ee=max_error(m.exact.read(),ref,scale),de=max_error(m.draft.read(),ref,scale);
        double ie=0,id=0;if(with_inject){const auto a=m.inj_exact.read(),b=m.inj_draft.read();
            for(std::uint32_t t=0;t<HC;++t){ie=std::max(ie,std::abs(a[t]-inj[t]));id=std::max(id,std::abs(b[t]-inj[t]));}}
        std::printf("hc %s: |ref|max %.4g exact err %.3g draft err %.3g injection err exact %.3g draft %.3g\n",with_inject?"attn/mlp":"head",scale,ee,de,ie,id);
        ok("draft HC output within exact-path error budget",de<=std::max(2*ee,scale*std::ldexp(1.0,-8)));
        ok("draft HC injection within 1e-5",id<=std::max(2*ie,1e-5));
    }
}
// PLE Q8 projection (target path): wave8 kernel must be bit-identical to the generic kernel.
void ple_cases(){
    hipStream_t s=nullptr;check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
    for(const auto [in,out]:{std::pair{2560U,10240U},std::pair{2560U,2560U},std::pair{96U,77U},std::pair{32U,9U},std::pair{288U,300U}}){
        Device<std::uint8_t> w(static_cast<std::size_t>(out)*in/32*34);Device<std::uint16_t> x(in);Device<float> a(out),b(out);
        hipLaunchKernelGGL(fill_q8,dim3(512),dim3(256),0,s,w.p,static_cast<std::uint64_t>(w.n/34),in*7U+out,-9);check(hipStreamSynchronize(s));
        std::mt19937 r(in+out);std::normal_distribution<float> n(0,1);std::vector<std::uint16_t> v(in);
        for(std::uint32_t i=0;i<in;++i)v[i]=i%17==3?static_cast<std::uint16_t>(0x0001U):bf16(n(r)*std::ldexp(1.0F,static_cast<int>(r()%40)-20));x.write(v);
        strix::testing::enqueue_ple_q8_projection(false,x.p,w.p,in,out,a.p,s);strix::testing::enqueue_ple_q8_projection(true,x.p,w.p,in,out,b.p,s);
        check(hipStreamSynchronize(s));
        const auto ra=a.read(),rb=b.read();
        char name[96];std::snprintf(name,sizeof(name),"PLE wave8 Q8 projection bit-exact %ux%u",out,in);
        ok(name,std::memcmp(ra.data(),rb.data(),ra.size()*4)==0);
    }
    check(hipStreamDestroy(s));
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
    Gemv q(H,12288U,1),kv(H,512U,2),o(6144U,H,3);
    const double qe=time_gpu(q.s,[&]{strix::enqueue_hip_mtp_attention_q8_linear(q.x.p,q.w.p,H,12288U,q.exact.p,q.s);
        strix::enqueue_hip_mtp_attention_q8_linear(kv.x.p,kv.w.p,H,512U,kv.exact.p,q.s);strix::enqueue_hip_mtp_attention_q8_linear(kv.x.p,kv.w.p,H,512U,kv.exact.p,q.s);
        strix::enqueue_hip_mtp_attention_q8_linear(o.x.p,o.w.p,6144U,H,o.exact.p,q.s);});
    const double qf=time_gpu(q.s,[&]{strix::launch_hip_mtp_draft_q8_gemv(q.x.p,true,q.w.p,H,12288U,q.fast.p,false,q.s);
        strix::launch_hip_mtp_draft_q8_gemv(kv.x.p,true,kv.w.p,H,512U,kv.fast.p,false,q.s);strix::launch_hip_mtp_draft_q8_gemv(kv.x.p,true,kv.w.p,H,512U,kv.fast.p,false,q.s);
        strix::launch_hip_mtp_draft_q8_gemv(o.xf.p,false,o.w.p,6144U,H,o.fast_bf16.p,true,q.s);});
    Hyper hc(5);
    const double he=time_gpu(hc.s,[&]{hc.run(false,true,hc.exact.p,hc.inj_exact.p);}),hd=time_gpu(hc.s,[&]{hc.run(true,true,hc.draft.p,hc.inj_draft.p);});
    Device<std::uint8_t> pk(static_cast<std::size_t>(10240)*2560/32*34),pv(static_cast<std::size_t>(2560)*2560/32*34);Device<std::uint16_t> px(2560);Device<float> po(10240);
    hipLaunchKernelGGL(fill_q8,dim3(1024),dim3(256),0,hc.s,pk.p,static_cast<std::uint64_t>(pk.n/34),1U,-9);
    hipLaunchKernelGGL(fill_q8,dim3(1024),dim3(256),0,hc.s,pv.p,static_cast<std::uint64_t>(pv.n/34),2U,-9);
    const auto ple=[&](bool wave){strix::testing::enqueue_ple_q8_projection(wave,px.p,pk.p,2560,10240,po.p,hc.s);
        strix::testing::enqueue_ple_q8_projection(wave,px.p,pv.p,2560,2560,po.p,hc.s);};
    const double pg=time_gpu(hc.s,[&]{ple(false);}),pw=time_gpu(hc.s,[&]{ple(true);});
    Device<std::uint16_t> iq(static_cast<std::size_t>(640)*2560);Device<float> io(640);
    const double ix=time_gpu(hc.s,[&]{strix::enqueue_hip_mtp_attention_bf16_linear(px.p,iq.p,2560,512,io.p,hc.s);
        strix::enqueue_hip_mtp_attention_bf16_linear(px.p,iq.p+512*2560,2560,128,io.p+512,hc.s);});
    std::printf("bench per draft, GPU median of 30 (ms)\n");
    std::printf("  indexer BF16 q+k projections (exact, per draft)        %.4f\n",ix);
    std::printf("  HC mix exact staged (per mix, x3 per draft)            %.4f\n",he);
    std::printf("  HC mix draft split-K + lane-chain (per mix)            %.4f\n",hd);
    std::printf("  PLE key+value Q8 generic (target, per row)             %.4f\n",pg);
    std::printf("  PLE key+value Q8 wave8 bit-exact (target, per row)     %.4f\n",pw);
    std::printf("  attention Q8 q/k/v/out exact block-per-row kernel        %.4f\n",qe);
    std::printf("  attention Q8 q/k/v/out draft wave-per-row GEMV           %.4f\n",qf);
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
        ple_cases();hyper_cases();fusion_cases();gemv_cases();moe_cases();std::cout<<"passed "<<passed<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
