#include "strix/hip_ds41_attention.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix/hip_ds41_dense_fp16.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace a=strix::ds41::attn;
namespace g=strix::ds41::attn::hip;
namespace {
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what.c_str()); }
}
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template <class T> struct Dev {
    T* p = nullptr; std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) { check(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T))); check(hipMemset(p, 0, std::max<std::size_t>(count, 1) * sizeof(T))); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { put(h); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    void put(const std::vector<T>& h) { check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> h(n); check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
};
std::mt19937 rng(4141);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
template <class T> bool same(const std::vector<T>& x, const std::vector<T>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}
std::vector<a::RopeRow> ropes(a::RopeKind k, const std::vector<std::uint64_t>& pos, unsigned ratio = 0) {
    std::vector<a::RopeRow> r;
    for (const auto p : pos) r.push_back(a::rope_row(k, ratio == 0 ? p : p / ratio * ratio));
    return r;
}
std::vector<float> random_f(std::size_t n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = uni(lo, hi); return v; }

bool measure=true;
template<class F> float timing(F f){
    if(!measure){f();check(hipDeviceSynchronize());return 0;}
    f();check(hipDeviceSynchronize());hipEvent_t b,e;check(hipEventCreate(&b));check(hipEventCreate(&e));
    std::vector<float> times;
    for(int j=0;j<3;++j){check(hipEventRecord(b));f();check(hipEventRecord(e));check(hipEventSynchronize(e));float ms;check(hipEventElapsedTime(&ms,b,e));times.push_back(ms);}
    check(hipEventDestroy(b));check(hipEventDestroy(e));std::sort(times.begin(),times.end());return times[1];
}
float relative(const std::vector<float>& ref,const std::vector<float>& got){
    std::size_t bad=0;for(float g:got)bad+=!std::isfinite(g);
    if(bad){std::fprintf(stderr,"nonfinite operator outputs: %zu of %zu\n",bad,got.size());return std::numeric_limits<float>::infinity();}
    double num=0,den=0;for(std::size_t i=0;i<ref.size();++i){num+=double(got[i]-ref[i])*(got[i]-ref[i]);den+=double(ref[i])*ref[i];}return static_cast<float>(std::sqrt(num/std::max(den,1e-30)));
}
void attention_smoke(std::uint64_t last,unsigned rows,unsigned nc){
    // compressed keys: 4096, or every position of a long context (selected indices up to the last position, ratio 1)
    const unsigned cap = a::kWindow + a::kMaxRows, ncache = last >= 65536U ? static_cast<unsigned>(last + 1U) : 4096U;
    std::vector<std::uint8_t> ring(cap * a::kWindowEntryBytes), kv(ncache * a::kKvEntryBytes);
    for (std::size_t s = 0; s < cap; ++s) {
        for (unsigned d = 0; d < 512; ++d) { std::uint8_t c; do c = static_cast<std::uint8_t>(rng()); while ((c & 0x7FU) == 0x7FU); ring[s * 528 + d] = c; }
        for (unsigned b = 0; b < 16; ++b) ring[s * 528 + 512 + b] = static_cast<std::uint8_t>(118 + rng() % 4);
    }
    for (std::size_t t = 0; t < ncache; ++t) {
        for (unsigned i = 0; i < 256; ++i) kv[t * 288 + i] = static_cast<std::uint8_t>(rng());
        for (unsigned g2 = 0; g2 < 32; ++g2) kv[t * 288 + 256 + g2] = static_cast<std::uint8_t>(0x30 + rng() % 16);
    }
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = last - (rows - 1) + r;
    const auto q = random_f(std::size_t{rows} * 64 * 512, -0.5F, 0.5F), sink = random_f(64, -2, 2);
    std::vector<std::uint32_t> idx(std::size_t{rows} * a::kTopK), cnt(rows, nc);
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> pick(ncache);
        for (std::uint32_t i = 0; i < ncache; ++i) pick[i] = i;
        std::shuffle(pick.begin(), pick.end(), rng);
        std::sort(pick.begin(), pick.begin() + nc);
        std::copy(pick.begin(), pick.begin() + nc, &idx[std::size_t{r} * a::kTopK]);
    }
    std::vector<float> o(std::size_t{rows} * 64 * 512);

    Dev<std::uint8_t> dring(ring), dkv(kv);
    Dev<float> dq(q), dsink(sink), dpart(std::size_t{rows} * 64 * a::kMaxChunks * 514), dout(o.size());
    Dev<std::uint64_t> dpos(pos);
    Dev<std::uint32_t> didx(idx), dcnt(cnt);
    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos));
    const g::AttentionView av{rows, cap, a::kTopK, dq.p, dsink.p, drope.p, dpos.p, dring.p, dkv.p, didx.p, nc != 0 ? dcnt.p : nullptr, dpart.p, dout.p};
    float old_ms=timing([&]{g::enqueue_attention(av,nullptr);});auto ref=dout.get();
    float new_ms=timing([&]{g::enqueue_attention_bf16_prefill(av,nullptr);});auto got=dout.get();
    float l2=relative(ref,got);expect(l2<.002f,"BF16 attention relative L2");
    float v2_ms=timing([&]{g::enqueue_attention_bf16_prefill(av,nullptr,2);});auto got2=dout.get();
    float l2v2=relative(ref,got2);expect(l2v2<.002f,"BF16 attention v2 relative L2");
    std::printf("attention pos=%llu rows=%u compressed=%u exact_ms=%.6f bf16_ms=%.6f rel_l2=%.9g bf16v2_ms=%.6f v2_rel_l2=%.9g\n",static_cast<unsigned long long>(last),rows,nc,old_ms,new_ms,l2,v2_ms,l2v2);std::fflush(stdout);
}
void dense_smoke(unsigned rows,unsigned width,unsigned count,unsigned row_width=0,unsigned offset=0,bool accumulate=false){
    namespace d=strix::ds41;
    unsigned full=row_width?row_width:width;
    std::vector<unsigned char> weight(std::size_t(count)*full/32*34);
    for(std::size_t b=0;b<weight.size()/34;++b){auto h=static_cast<_Float16>(uni(.001f,.01f));auto bits=__builtin_bit_cast(unsigned short,h);weight[b*34]=static_cast<unsigned char>(bits);weight[b*34+1]=static_cast<unsigned char>(bits>>8);for(unsigned j=0;j<32;++j)weight[b*34+2+j]=static_cast<unsigned char>(rng()%255-127);}
    auto x=random_f(std::size_t(rows)*(width+32),-.5f,.5f);
    Dev<unsigned char> dw(weight);Dev<float> dx(x),out(std::size_t(rows)*(count+16));
    Dev<signed char> codes(x.size());Dev<float> deltas(x.size()/32);Dev<int> sums(x.size()/32);
    d::enqueue_quantize_a8(dx.p,width+32,rows,codes.p,deltas.p,sums.p,nullptr);
    d::Q8A8DensePrefillView v;v.weight=dw.p;v.width=width;v.count=count;v.rows=rows;v.x_stride=width+32;v.out_stride=count+16;
    v.row_width=row_width;v.k_offset=offset;v.accumulate=accumulate;v.codes=codes.p;v.deltas=deltas.p;v.out=out.p;
    // Timing calls start from the same zero output when accumulating K halves.
    auto clear=[&]{check(hipMemset(out.p,0,out.n*4));};
    clear();d::enqueue_q8_0_a8_dense_prefill(v,nullptr);auto ref=out.get();
    d::DenseFp16Cache cache;clear();cache.enqueue(v,dx.p,nullptr);auto got=out.get();
    float l2=relative(ref,got);expect(l2<.02f,"FP16 dense relative L2 vs A8");
    float old_ms=timing([&]{d::enqueue_q8_0_a8_dense_prefill(v,nullptr);});
    float new_ms=timing([&]{cache.enqueue(v,dx.p,nullptr);});
    // A second cache object must replay the disk solution, independently of weight addresses.
    d::DenseFp16Cache replay;clear();replay.enqueue(v,dx.p,nullptr);auto twice=out.get();expect(same(got,twice),"FP16 cached solution replay bits");
    // keep_a8: the FP16 operands are the exact path's A8 activations (code * delta), so only FP16 rounding and the
    // accumulation order differ from the exact kernel.
    clear();cache.enqueue(v,dx.p,nullptr,true);auto a8=out.get();float l2a8=relative(ref,a8);expect(l2a8<.002f,"FP16 keep-A8 relative L2 vs A8");
    float a8_ms=timing([&]{cache.enqueue(v,dx.p,nullptr,true);});
    std::printf("dense rows=%u K=%u N=%u row_width=%u offset=%u accumulate=%d exact_ms=%.6f fp16_ms=%.6f rel_l2=%.9g a8_ms=%.6f a8_rel_l2=%.9g\n",rows,width,count,row_width,offset,accumulate,old_ms,new_ms,l2,a8_ms,l2a8);std::fflush(stdout);
}
}
// Tall exact form: bitwise equal to the staged wide form, timed per tile shape.
void tall_smoke(unsigned rows,unsigned width,unsigned count,unsigned row_width=0,unsigned offset=0,bool accumulate=false){
    namespace d=strix::ds41;
    unsigned full=row_width?row_width:width;
    std::vector<unsigned char> weight(std::size_t(count)*full/32*34);
    for(std::size_t b=0;b<weight.size()/34;++b){auto h=static_cast<_Float16>(uni(.001f,.01f));auto bits=__builtin_bit_cast(unsigned short,h);weight[b*34]=static_cast<unsigned char>(bits);weight[b*34+1]=static_cast<unsigned char>(bits>>8);for(unsigned j=0;j<32;++j)weight[b*34+2+j]=static_cast<unsigned char>(rng()%255-127);}
    auto x=random_f(std::size_t(rows)*(width+32),-.5f,.5f);
    Dev<unsigned char> dw(weight);Dev<float> dx(x),out(std::size_t(rows)*(count+16));
    Dev<signed char> codes(x.size());Dev<float> deltas(x.size()/32);Dev<int> sums(x.size()/32);
    d::enqueue_quantize_a8(dx.p,width+32,rows,codes.p,deltas.p,sums.p,nullptr);
    d::Q8A8DensePrefillView v;v.weight=dw.p;v.width=width;v.count=count;v.rows=rows;v.x_stride=width+32;v.out_stride=count+16;
    v.row_width=row_width;v.k_offset=offset;v.accumulate=accumulate;v.codes=codes.p;v.deltas=deltas.p;v.out=out.p;
    // every bitwise run starts from the same nonzero prior output (so accumulating calls test out + y, not 0 + y)
    const auto prior=random_f(out.n,-4.f,4.f);
    auto clear=[&]{out.put(prior);};
    clear();d::enqueue_q8_0_a8_dense_prefill(v,nullptr);auto ref=out.get();
    float base=timing([&]{d::enqueue_q8_0_a8_dense_prefill(v,nullptr);});
    std::printf("tall rows=%u K=%u N=%u wide_ms=%.6f",rows,width,count,base);
    // TALL_REVERSE=1: the forms in reverse order (order / thermal bias check between passes)
    std::vector<unsigned> forms{22U,41U,21U,14U,12U,114U,112U,214U,212U,314U,312U,414U,412U,514U,614U,612U,714U,814U,812U,914U,422U,822U,1514U,1522U,1614U,1714U,1722U,1814U,1914U,1922U,2014U,2114U,2122U,2214U,1U,97U,96U};
    if(const char* r=std::getenv("TALL_REVERSE");r&&std::string(r)=="1")std::reverse(forms.begin(),forms.end());
    for(unsigned form:forms){
        v.tall=form;clear();d::enqueue_q8_0_a8_dense_prefill(v,nullptr);auto got=out.get();
        expect(same(ref,got),"tall form "+std::to_string(form)+" bitwise");
        float ms=timing([&]{d::enqueue_q8_0_a8_dense_prefill(v,nullptr);});
        std::printf(" t%u_ms=%.6f%s",form,ms,same(ref,got)?"":"(DIFF)");
    }
    std::printf("\n");std::fflush(stdout);v.tall=0;
}
// FP16 output form (STRIX_DS41_PREFILL_FP16_FAST): exact wide / tall auto / FP16 v1 / FP16 output with the conversion
// pass / FP16 output from a pre-converted activation (the norm's FP16, i.e. the fused conversion) / q_a || kv joint.
void fast_smoke(unsigned rows,unsigned width,unsigned count,unsigned count2=0,unsigned row_width=0,unsigned offset=0){
    namespace d=strix::ds41;
    unsigned full=row_width?row_width:width;
    auto make=[&](unsigned n){std::vector<unsigned char> w(std::size_t(n)*full/32*34);
        for(std::size_t b=0;b<w.size()/34;++b){auto h=static_cast<_Float16>(uni(.001f,.01f));auto bits=__builtin_bit_cast(unsigned short,h);w[b*34]=static_cast<unsigned char>(bits);w[b*34+1]=static_cast<unsigned char>(bits>>8);for(unsigned j=0;j<32;++j)w[b*34+2+j]=static_cast<unsigned char>(rng()%255-127);}
        return w;};
    auto weight=make(count),weight2=make(count2?count2:1);
    auto x=random_f(std::size_t(rows)*width,-2.f,2.f);
    std::vector<unsigned short> x16(x.size());for(std::size_t i=0;i<x.size();++i)x16[i]=__builtin_bit_cast(unsigned short,static_cast<_Float16>(x[i]));
    Dev<unsigned char> dw(weight),dw2(weight2);Dev<float> dx(x),out(std::size_t(rows)*count),out2(std::size_t(rows)*(count2?count2:1));
    Dev<unsigned short> dx16(x16);
    Dev<signed char> codes(x.size());Dev<float> deltas(x.size()/32);Dev<int> sums(x.size()/32);
    d::enqueue_quantize_a8(dx.p,width,rows,codes.p,deltas.p,sums.p,nullptr);
    d::Q8A8DensePrefillView v;v.weight=dw.p;v.width=width;v.count=count;v.rows=rows;v.x_stride=width;v.out_stride=count;
    v.row_width=row_width;v.k_offset=offset;v.codes=codes.p;v.deltas=deltas.p;v.out=out.p;
    auto v2=v;v2.weight=dw2.p;v2.count=count2;v2.out=out2.p;v2.out_stride=count2;
    d::enqueue_q8_0_a8_dense_prefill(v,nullptr);auto ref=out.get();
    float wide_ms=timing([&]{d::enqueue_q8_0_a8_dense_prefill(v,nullptr);});
    v.tall=1;float tall_ms=timing([&]{d::enqueue_q8_0_a8_dense_prefill(v,nullptr);});auto tall=out.get();v.tall=0;
    expect(same(ref,tall),"tall auto bitwise");
    d::DenseFp16Cache cache;
    float v1_ms=timing([&]{cache.enqueue(v,dx.p,nullptr);});auto g1=out.get();
    float conv_ms=timing([&]{cache.enqueue_fast(&v,1,rows,nullptr,0,dx.p,nullptr);});auto gc=out.get();
    float fused_ms=timing([&]{cache.enqueue_fast(&v,1,rows,dx16.p,width,nullptr,nullptr);});auto gf=out.get();
    float l1=relative(ref,g1),lc=relative(ref,gc),lf=relative(ref,gf);
    expect(lc<.02f&&lf<.02f,"FP16 output relative L2 vs A8");
    std::printf("fast rows=%u K=%u N=%u wide_ms=%.4f tall_ms=%.4f v1_ms=%.4f out16_conv_ms=%.4f out16_fused_ms=%.4f l2_v1=%.3g l2_conv=%.3g l2_fused=%.3g",
                rows,width,count,wide_ms,tall_ms,v1_ms,conv_ms,fused_ms,l1,lc,lf);
    if(count2){
        d::enqueue_q8_0_a8_dense_prefill(v2,nullptr);auto ref2=out2.get();
        v2.tall=1;float tall2_ms=timing([&]{d::enqueue_q8_0_a8_dense_prefill(v2,nullptr);});v2.tall=0;
        float sep_ms=timing([&]{cache.enqueue_fast(&v,1,rows,dx16.p,width,nullptr,nullptr);cache.enqueue_fast(&v2,1,rows,dx16.p,width,nullptr,nullptr);});
        d::Q8A8DensePrefillView both[2]{v,v2};
        float joint_ms=timing([&]{cache.enqueue_fast(both,2,rows,dx16.p,width,nullptr,nullptr);});
        auto j1=out.get(),j2=out2.get();
        // the joint GEMM (N = 1792) may pick another solution than q_a alone: same accuracy, not the same bits
        float lj=relative(ref2,j2);expect(lj<.02f,"joint second view relative L2");
        std::printf(" | N2=%u tall_pair_ms=%.4f out16_separate_ms=%.4f joint_ms=%.4f l2_joint2=%.3g joint_q_a_bits=%s",count2,tall_ms+tall2_ms,sep_ms,joint_ms,lj,same(j1,gf)?"same":"DIFF");
    }
    expect(cache.take_nonfinite(nullptr)==0,"FP16 output nonfinite");
    std::printf("\n");std::fflush(stdout);
}
// attn_output_a as the engine runs it: `groups` o-groups of (width -> count) over one row of groups * width values;
// exact = per-group dense Q8_0 x A8 calls, FP16 = per-group GEMMs (as before) and one strided batch.
void grouped_smoke(unsigned rows,unsigned groups,unsigned width,unsigned count){
    namespace d=strix::ds41;
    std::vector<unsigned char> weight(std::size_t(groups)*count*width/32*34);
    for(std::size_t b=0;b<weight.size()/34;++b){auto h=static_cast<_Float16>(uni(.001f,.01f));auto bits=__builtin_bit_cast(unsigned short,h);weight[b*34]=static_cast<unsigned char>(bits);weight[b*34+1]=static_cast<unsigned char>(bits>>8);for(unsigned j=0;j<32;++j)weight[b*34+2+j]=static_cast<unsigned char>(rng()%255-127);}
    const unsigned QL=groups*width,OLL=groups*count;
    auto x=random_f(std::size_t(rows)*QL,-.5f,.5f);
    Dev<unsigned char> dw(weight);Dev<float> dx(x),out(std::size_t(rows)*OLL);
    Dev<signed char> codes(x.size());Dev<float> deltas(x.size()/32);Dev<int> sums(x.size()/32);
    d::enqueue_quantize_a8(dx.p,QL,rows,codes.p,deltas.p,sums.p,nullptr);
    // exact reference: the per-group dense Q8_0 x A8 kernel (the decode's bits, as the engine's grouped launch)
    auto exact=[&]{for(unsigned g=0;g<groups;++g){d::Q8A8DensePrefillView q;q.weight=dw.p+std::size_t(g)*count*(width/32*34);q.width=width;q.count=count;q.rows=rows;q.x_stride=QL;
        q.codes=codes.p+g*width;q.deltas=deltas.p+g*width/32;q.out=out.p+g*count;q.out_stride=OLL;d::enqueue_q8_0_a8_dense_prefill(q,nullptr);}};
    exact();auto ref=out.get();
    float exact_ms=timing(exact);
    d::DenseFp16Cache cache;
    auto per_group=[&]{for(unsigned g=0;g<groups;++g){d::Q8A8DensePrefillView q;q.weight=dw.p+std::size_t(g)*count*(width/32*34);q.width=width;q.count=count;q.rows=rows;q.x_stride=QL;q.out=out.p+g*count;q.out_stride=OLL;cache.enqueue_fast(&q,1,rows,nullptr,0,dx.p+g*width,nullptr);}};
    float pg_ms=timing(per_group);auto gp=out.get();
    d::Q8A8DensePrefillView q;q.weight=dw.p;q.width=width;q.count=count;q.rows=rows;q.x_stride=QL;q.out=out.p;q.out_stride=OLL;
    float b_ms=timing([&]{cache.enqueue_fast(&q,1,rows,nullptr,0,dx.p,nullptr,groups);});auto gb=out.get();
    float lp=relative(ref,gp),lb=relative(ref,gb);expect(lp<.02f&&lb<.02f,"grouped FP16 relative L2");
    std::printf("grouped rows=%u groups=%u K=%u N=%u exact_ms=%.4f per_group_ms=%.4f batched_ms=%.4f l2_per_group=%.3g l2_batched=%.3g\n",rows,groups,width,count,exact_ms,pg_ms,b_ms,lp,lb);std::fflush(stdout);
}
// --conv: f16_to_float_select against f16_to_float on the device for all 65536 halves (the Lean forms' scale conversion)
__global__ void conv_kernel(unsigned* bad){
    const unsigned h=blockIdx.x*256U+threadIdx.x;
    const float a=strix::ds41::f16_to_float(static_cast<std::uint16_t>(h)),b=strix::ds41::f16_to_float_select(static_cast<std::uint16_t>(h));
    if(__float_as_uint(a)!=__float_as_uint(b))atomicAdd(bad,1U);
    // the hardware conversion (Lean 3 forms): must match on every non-NaN half; NaN halves are counted apart
    const float c=static_cast<float>(__builtin_bit_cast(_Float16,static_cast<unsigned short>(h)));
    if(__float_as_uint(a)!=__float_as_uint(c)){if(a!=a)atomicAdd(bad+2,1U);else atomicAdd(bad+1,1U);}
}
int main(int argc,char** argv)try{
    if(argc==2 && std::string(argv[1])=="--conv"){
        Dev<unsigned> bad(3);hipLaunchKernelGGL(conv_kernel,dim3(256),dim3(256),0,nullptr,bad.p);check(hipDeviceSynchronize());
        const auto b=bad.get();
        std::printf("conv halves=65536 select_differing=%u hw_differing_nonnan=%u hw_differing_nan=%u\n",b[0],b[1],b[2]);
        return b[0]||b[1]?1:0;  // NaN halves may differ only in payload (Lean 3 excludes NaN scales)
    }
    if(argc>=2 && std::string(argv[1])=="--fast"){  // production shapes, row split 8K / 2K and TP
        hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));std::printf("arch=%s device=%s\n",prop.gcnArchName,prop.name);
        for(unsigned r:{8192U,4096U,2048U,1024U})fast_smoke(r,5120,1280,512);
        fast_smoke(1024,1280,16384);fast_smoke(1024,1280,8192);fast_smoke(512,1280,8192);
        fast_smoke(1024,4096,1024);fast_smoke(1024,8192,5120);fast_smoke(1024,4096,5120,0,8192,4096);fast_smoke(512,8192,5120);
        fast_smoke(64,8192,5120);fast_smoke(17,128,37,64);
        grouped_smoke(1024,8,4096,1024);grouped_smoke(1024,4,4096,1024);grouped_smoke(512,8,4096,1024);
        std::printf("failures=%d\n",failures);return failures?1:0;
    }
    if(argc>2 && std::string(argv[1])=="--tall"){  // --tall R,K,N[,row_width,offset,accumulate] ...
        for(int i=2;i<argc;++i){unsigned f[6]{};int k=0;std::string s=argv[i];std::size_t at=0;
            while(k<6&&at<=s.size()){auto c=s.find(',',at);f[k++]=static_cast<unsigned>(std::stoul(s.substr(at,c-at)));if(c==std::string::npos)break;at=c+1;}
            tall_smoke(f[0],f[1],f[2],f[3],f[4],f[5]!=0);}
        std::printf("failures=%d\n",failures);return failures?1:0;
    }
    if(argc==2 && std::string(argv[1])=="--tall"){
        tall_smoke(2048,5120,1280);tall_smoke(2048,5120,512);tall_smoke(8192,5120,1280);tall_smoke(8192,5120,512);
        tall_smoke(1024,1280,16384);tall_smoke(1024,4096,1024);tall_smoke(1024,4096,5120,8192,4096);tall_smoke(64,4096,5120,8192,4096,true);
        tall_smoke(17,128,37,256,128);tall_smoke(1000,4096,1000);
        std::printf("failures=%d\n",failures);return failures?1:0;
    }
    if(argc==2 && std::string(argv[1])=="--attention-checks"){measure=false;attention_smoke(2047,8,a::kTopK);attention_smoke(8191,8,a::kTopK);return failures?1:0;}
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));std::printf("arch=%s device=%s\n",prop.gcnArchName,prop.name);
    dense_smoke(2048,5120,1280);dense_smoke(2048,5120,512);
    dense_smoke(8192,5120,1280);dense_smoke(8192,5120,512);
    dense_smoke(1024,1280,16384);dense_smoke(1024,4096,1024);dense_smoke(1024,4096,5120);
    dense_smoke(64,4096,5120,8192,4096,true);dense_smoke(17,128,37,256,128);
    attention_smoke(5,1,0);attention_smoke(20,2,37);attention_smoke(2047,8,a::kTopK);attention_smoke(8191,8,a::kTopK);
    std::printf("failures=%d\n",failures);return failures?1:0;
}catch(const std::exception& e){std::fprintf(stderr,"approx-prefill: %s\n",e.what());return 2;}
