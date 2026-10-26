// Multi-session MTP draft batch (STRIX_MTP_DRAFT_BATCH): every batched draft launcher against the one-token
// launcher run on each row alone, bit for bit, for 1..8 rows in permuted orders (synthetic canonical-sidecar
// weights: Q8_0 with registered Q4 copies where production reads them).
#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/hip_mtp_input_fusion.hpp"
#include "strix/hip_mtp_draft_head.hpp"
#include "strix/hip_mtp_attention.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool x,const std::string& m){if(!x)throw std::runtime_error(m);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
int failures=0;
void ok(const std::string& what,bool pass){std::printf("%s %s\n",pass?"PASS":"FAIL",what.c_str());if(!pass)++failures;}
template<class T> struct Device {
    T* p=nullptr;std::size_t n=0;
    explicit Device(std::size_t count):n(count){check(hipMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));check(hipMemset(p,0,count*sizeof(T)));check(hipDeviceSynchronize());}
    ~Device(){if(p)static_cast<void>(hipFree(p));}
    Device(const Device&)=delete;
    std::vector<T> read() const {std::vector<T> v(n);check(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));return v;}
    void write(const std::vector<T>& v){require(v.size()==n,"upload extent");check(hipMemcpy(p,v.data(),n*sizeof(T),hipMemcpyHostToDevice));}
    void poison(){check(hipMemset(p,0x7f,n*sizeof(T)));check(hipDeviceSynchronize());}
};
// Diagnostics: the first differing element of up to 60 failed comparisons (where = the case being compared).
std::string where;int reports=0;
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b){
    if(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0) return true;
    if(reports++<60) {
        std::size_t first=SIZE_MAX,count=0;
        for(std::size_t i=0;i<std::min(a.size(),b.size());++i) if(std::memcmp(&a[i],&b[i],sizeof(T))) {if(first==SIZE_MAX) first=i;++count;}
        std::printf("  diff %s: sizes %zu/%zu, %zu elements differ, first %zu",where.c_str(),a.size(),b.size(),count,first);
        if(first!=SIZE_MAX) {
            std::uint64_t x=0,y=0;std::memcpy(&x,&a[first],sizeof(T));std::memcpy(&y,&b[first],sizeof(T));
            std::printf(" batch=%llx ref=%llx",static_cast<unsigned long long>(x),static_cast<unsigned long long>(y));
            if constexpr(sizeof(T)==4) {float fa,fb;std::memcpy(&fa,&a[first],4);std::memcpy(&fb,&b[first],4);std::printf(" (%.9g vs %.9g)",fa,fb);}
        }
        std::printf("\n");
    }
    return false;
}
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
std::uint16_t bf16(float f){const auto b=std::bit_cast<std::uint32_t>(f);return static_cast<std::uint16_t>((b+0x7fffU+((b>>16U)&1U))>>16U);}
std::vector<std::uint16_t> random_bf16(std::size_t n,std::mt19937& g,float scale=1.0F){
    std::normal_distribution<float> d(0,scale);std::vector<std::uint16_t> v(n);for(auto& x:v)x=bf16(d(g));return v;
}
std::vector<float> random_f32(std::size_t n,std::mt19937& g,float mean,float scale){
    std::normal_distribution<float> d(mean,scale);std::vector<float> v(n);for(auto& x:v)x=d(g);return v;
}
struct Stream {hipStream_t s=nullptr;Stream(){check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));}~Stream(){static_cast<void>(hipStreamDestroy(s));}};
void q8_random(Device<std::uint8_t>& d,std::uint32_t seed,int exponent,hipStream_t s){
    hipLaunchKernelGGL(fill_q8,dim3(2048),dim3(256),0,s,d.p,static_cast<std::uint64_t>(d.n/34),seed,exponent);check(hipStreamSynchronize(s));
}
std::vector<std::uint32_t> order(std::uint32_t n,std::mt19937& g){std::vector<std::uint32_t> o(n);std::iota(o.begin(),o.end(),0U);std::shuffle(o.begin(),o.end(),g);return o;}

// GEMV: three segments (q/k/v shape) BF16 in -> F32 out, and one F32 in -> BF16 out (o shape), Q8 and Q4 weights.
void gemv_cases(bool q4){
    Stream st;std::mt19937 g(q4?11U:7U);
    const std::uint32_t qrows=4096,kvrows=512,orows=H,owidth=4096;
    // Weights live to the end of the process: the Q4 copy registry is keyed by the Q8 address.
    auto& wq=*new Device<std::uint8_t>(static_cast<std::size_t>(qrows)*H/32*34);auto& wk=*new Device<std::uint8_t>(static_cast<std::size_t>(kvrows)*H/32*34);
    auto& wv=*new Device<std::uint8_t>(wk.n);auto& wo=*new Device<std::uint8_t>(static_cast<std::size_t>(orows)*owidth/32*34);
    q8_random(wq,1,-9,st.s);q8_random(wk,2,-9,st.s);q8_random(wv,3,-9,st.s);q8_random(wo,4,-10,st.s);
    if(q4) for(auto* w:{&wq,&wk,&wv,&wo}) strix::register_hip_mtp_draft_q4(w->p,w->n/34*32,st.s);
    check(hipStreamSynchronize(st.s));
    constexpr std::uint32_t N=strix::hip_mtp_draft_batch_rows;
    std::vector<std::unique_ptr<Device<std::uint16_t>>> x;std::vector<std::unique_ptr<Device<float>>> xf,q,k,v,qs,ks,vs;
    std::vector<std::unique_ptr<Device<std::uint16_t>>> o,os;
    for(std::uint32_t t=0;t<N;++t){
        x.push_back(std::make_unique<Device<std::uint16_t>>(H));x.back()->write(random_bf16(H,g));
        xf.push_back(std::make_unique<Device<float>>(owidth));xf.back()->write(random_f32(owidth,g,0,1));
        for(auto* set:{&q,&qs}) set->push_back(std::make_unique<Device<float>>(qrows));
        for(auto* set:{&k,&ks,&v,&vs}) set->push_back(std::make_unique<Device<float>>(kvrows));
        for(auto* set:{&o,&os}) set->push_back(std::make_unique<Device<std::uint16_t>>(orows));
        strix::launch_hip_mtp_draft_q8_gemv(x[t]->p,true,wq.p,H,qrows,qs[t]->p,false,st.s);
        strix::launch_hip_mtp_draft_q8_gemv(x[t]->p,true,wk.p,H,kvrows,ks[t]->p,false,st.s);
        strix::launch_hip_mtp_draft_q8_gemv(x[t]->p,true,wv.p,H,kvrows,vs[t]->p,false,st.s);
        strix::launch_hip_mtp_draft_q8_gemv(xf[t]->p,false,wo.p,owidth,orows,os[t]->p,true,st.s);
    }
    check(hipStreamSynchronize(st.s));
    bool all=true;
    for(std::uint32_t count=1;count<=N;++count) for(int rep=0;rep<2;++rep) {
        const auto perm=order(N,g);
        strix::HipMtpDraftGemvRows qkv{},out{};
        qkv.weights[0]=wq.p;qkv.weights[1]=wk.p;qkv.weights[2]=wv.p;qkv.first[1]=qrows;qkv.first[2]=qrows+kvrows;qkv.first[3]=qrows+2*kvrows;qkv.segments=3;
        out.weights[0]=wo.p;out.first[1]=orows;out.segments=1;
        for(std::uint32_t r=0;r<count;++r){
            const auto t=perm[r];q[t]->poison();k[t]->poison();v[t]->poison();o[t]->poison();
            qkv.input[r]=x[t]->p;qkv.output[0][r]=q[t]->p;qkv.output[1][r]=k[t]->p;qkv.output[2][r]=v[t]->p;
            out.input[r]=xf[t]->p;out.output[0][r]=o[t]->p;
        }
        strix::launch_hip_mtp_draft_q8_gemv_rows(qkv,count,true,H,false,st.s);
        strix::launch_hip_mtp_draft_q8_gemv_rows(out,count,false,owidth,true,st.s);
        check(hipStreamSynchronize(st.s));
        for(std::uint32_t r=0;r<count;++r){const auto t=perm[r];
            where="gemv "+std::string(q4?"q4":"q8")+" count="+std::to_string(count)+" row="+std::to_string(r)+" (q,k,v,o)";
            all=all && same(q[t]->read(),qs[t]->read()) && same(k[t]->read(),ks[t]->read()) && same(v[t]->read(),vs[t]->read()) && same(o[t]->read(),os[t]->read());}
    }
    ok(std::string("batched draft GEMV == one-token GEMV per row (")+(q4?"Q4 copies":"Q8")+", 1..8 rows, permuted)",all);
}

// BF16 projection (index query 512 and index key 128 rows of the canonical sidecar): BF16 in -> F32 out.
__global__ void fill_bf16(std::uint16_t* w,std::uint64_t n,std::uint32_t seed) {
    for(std::uint64_t i=blockIdx.x*static_cast<std::uint64_t>(blockDim.x)+threadIdx.x;i<n;i+=static_cast<std::uint64_t>(gridDim.x)*blockDim.x) {
        auto x=static_cast<std::uint32_t>(i*2654435761ULL)^seed;x^=x<<13U;x^=x>>17U;x^=x<<5U;
        const float v=ldexpf(static_cast<float>(static_cast<int>(x%2001U)-1000)/1000.0F,-4);
        w[i]=static_cast<std::uint16_t>(__float_as_uint(v)>>16U);
    }
}
void bf16_linear_cases(){
    Stream st;std::mt19937 g(31);bool all=true;
    for(const std::uint32_t rows:{512U,128U}){
        Device<std::uint16_t> w(static_cast<std::size_t>(rows)*H);
        hipLaunchKernelGGL(fill_bf16,dim3(1024),dim3(256),0,st.s,w.p,static_cast<std::uint64_t>(w.n),rows);check(hipStreamSynchronize(st.s));
        constexpr std::uint32_t N=strix::hip_mtp_draft_batch_rows;
        std::vector<std::unique_ptr<Device<std::uint16_t>>> x;std::vector<std::unique_ptr<Device<float>>> out,ref;
        for(std::uint32_t t=0;t<N;++t){
            x.push_back(std::make_unique<Device<std::uint16_t>>(H));x.back()->write(random_bf16(H,g));
            out.push_back(std::make_unique<Device<float>>(rows));ref.push_back(std::make_unique<Device<float>>(rows));
            strix::enqueue_hip_mtp_attention_bf16_linear(x[t]->p,w.p,H,rows,ref[t]->p,st.s);
        }
        check(hipStreamSynchronize(st.s));
        for(std::uint32_t count=1;count<=N;++count){
            const auto perm=order(N,g);std::vector<const std::uint16_t*> in;std::vector<float*> o;
            for(std::uint32_t r=0;r<count;++r){out[perm[r]]->poison();in.push_back(x[perm[r]]->p);o.push_back(out[perm[r]]->p);}
            strix::enqueue_hip_mtp_attention_bf16_linear_rows(in,w.p,H,rows,o,st.s);check(hipStreamSynchronize(st.s));
            for(std::uint32_t r=0;r<count;++r){where="bf16 rows="+std::to_string(rows)+" count="+std::to_string(count)+" row="+std::to_string(r);
                all=all && same(out[perm[r]]->read(),ref[perm[r]]->read());}
        }
    }
    ok("batched BF16 index projection == one-token BF16 projection per row (512 and 128 rows, 1..8 rows, permuted)",all);
}
// Fast MoE: each row its own input, routes, weights, workspace, status and output; shared Q4 expert copies.
void moe_cases(){
    Stream st;std::mt19937 g(5);
    auto& gate=*new Device<std::uint8_t>(static_cast<std::size_t>(E)*I*H/32*34);auto& up=*new Device<std::uint8_t>(gate.n);
    auto& down=*new Device<std::uint8_t>(static_cast<std::size_t>(E)*H*I/32*34);
    auto& sgate=*new Device<std::uint8_t>(static_cast<std::size_t>(I)*H/32*34);auto& sup=*new Device<std::uint8_t>(sgate.n);
    auto& sdown=*new Device<std::uint8_t>(static_cast<std::size_t>(H)*I/32*34);
    Device<float> scalar(H);scalar.write(random_f32(H,g,0,0.02F));
    int k=0;for(auto* d:{&gate,&up,&down,&sgate,&sup,&sdown}){q8_random(*d,20U+static_cast<std::uint32_t>(k),k%3==2?-10:-9,st.s);++k;}
    for(auto* d:{&gate,&up,&down,&sgate,&sup,&sdown}) strix::register_hip_mtp_draft_q4(d->p,d->n/34*32,st.s);
    constexpr std::uint32_t N=strix::hip_mtp_draft_batch_rows;
    struct Row {Device<std::uint16_t> input{H},out{H},ref{H};Device<std::uint32_t> routes{10},status{2};Device<float> rw{10},ws{strix::hip_mtp_fast_moe_workspace_floats};};
    std::vector<std::unique_ptr<Row>> rows;
    auto views=[&](Row& r,std::uint16_t* out){
        return strix::HipMtpFastMoeViews{r.input.p,r.routes.p,r.rw.p,gate.p,up.p,down.p,sgate.p,sup.p,sdown.p,scalar.p,r.ws.p,out,r.status.p};
    };
    for(std::uint32_t t=0;t<N;++t){
        rows.push_back(std::make_unique<Row>());auto& r=*rows.back();
        r.input.write(random_bf16(H,g));
        std::vector<std::uint32_t> all(E);std::iota(all.begin(),all.end(),0U);std::shuffle(all.begin(),all.end(),g);
        // Rows 0 and 1 share some experts (union overlap is legal and must not matter).
        if(t==1) {auto r0=rows[0]->routes.read();std::copy(r0.begin(),r0.begin()+5,all.begin());}
        r.routes.write(std::vector<std::uint32_t>(all.begin(),all.begin()+10));
        auto w=random_f32(10,g,0,1);float sum=0;for(auto& x:w){x=std::exp(x);sum+=x;}for(auto& x:w)x/=sum;r.rw.write(w);
        strix::launch_hip_mtp_fast_moe(views(r,r.ref.p),st.s);
    }
    check(hipStreamSynchronize(st.s));
    bool all=true;
    for(std::uint32_t count=1;count<=N;++count){
        const auto perm=order(N,g);std::vector<strix::HipMtpFastMoeViews> v;
        for(std::uint32_t r=0;r<count;++r){rows[perm[r]]->out.poison();v.push_back(views(*rows[perm[r]],rows[perm[r]]->out.p));}
        strix::launch_hip_mtp_fast_moe_rows(v,st.s);check(hipStreamSynchronize(st.s));
        for(std::uint32_t r=0;r<count;++r) all=all && same(rows[perm[r]]->out.read(),rows[perm[r]]->ref.read());
    }
    ok("batched fast MoE == one-token fast MoE per row (1..8 rows, permuted)",all);
    // Row-local fault: row 0 routes an invalid expert. Alone, its status word gets bit 0 and the down launch leaves its
    // output untouched; in the batch exactly that happens to row 0 while every other row stays exact.
    {
        auto bad=rows[0]->routes.read();const auto good=bad;bad[0]=E;rows[0]->routes.write(bad);
        rows[0]->status.write({0U,0U});rows[0]->out.poison();
        strix::launch_hip_mtp_fast_moe(views(*rows[0],rows[0]->out.p),st.s);check(hipStreamSynchronize(st.s));
        const auto alone_out=rows[0]->out.read();const auto alone_status=rows[0]->status.read();
        rows[0]->status.write({0U,0U});
        std::vector<strix::HipMtpFastMoeViews> v;
        for(std::uint32_t r=0;r<N;++r){rows[r]->out.poison();v.push_back(views(*rows[r],rows[r]->out.p));}
        strix::launch_hip_mtp_fast_moe_rows(v,st.s);check(hipStreamSynchronize(st.s));
        bool iso=same(rows[0]->out.read(),alone_out) && same(rows[0]->status.read(),alone_status) && (alone_status[0]&1U);
        for(std::uint32_t r=1;r<N;++r) iso=iso && same(rows[r]->out.read(),rows[r]->ref.read()) && rows[r]->status.read()[0]==0U;
        ok("batched fast MoE row-local fault: only the faulty row flags and skips, the others stay exact",iso);
        rows[0]->routes.write(good);rows[0]->status.write({0U,0U});
    }
}

// Input fusion: each row its own embedding/wide input/scratch/output; shared norms and Q4 projection copy.
void fusion_cases(){
    Stream st;std::mt19937 g(9);
    auto& packed=*new Device<std::uint8_t>(static_cast<std::size_t>(H)*2*H/32*34);q8_random(packed,40,-9,st.s);
    strix::register_hip_mtp_draft_q4(packed.p,packed.n/34*32,st.s);check(hipStreamSynchronize(st.s));
    Device<float> enorm(H),hnorm(H*HC);enorm.write(random_f32(H,g,1,0.1F));hnorm.write(random_f32(H*HC,g,1,0.1F));
    const auto scratch=strix::hip_mtp_input_fusion_scratch_bytes(H,HC)/4;
    struct Row {Device<std::uint16_t> embedding{H},wide{H*HC},out{H*HC},ref{H*HC};Device<float> normalized,projected;
        explicit Row(std::uint64_t n):normalized(n),projected(n){}};
    constexpr std::uint32_t N=strix::hip_mtp_draft_batch_rows;
    std::vector<std::unique_ptr<Row>> rows;
    auto v=[](auto& d){return strix::HipMtpInputFusionView{reinterpret_cast<std::uintptr_t>(d.p),d.n*sizeof(*d.p),1};};
    auto request=[&](Row& r,Device<std::uint16_t>& out,strix::HipMtpInputRounding projection){
        strix::HipMtpInputFusionRequest q;q.hidden_size=H;q.hc_count=HC;q.epsilon=1e-6F;
        q.embedding_storage=q.hidden_storage=q.output_storage=strix::HipMtpInputStorage::bf16;
        q.normalized_rounding=strix::HipMtpInputRounding::fp32;q.projection_rounding=projection;
        q.norm_weight_encoding=strix::MtpNormWeightEncoding::effective_scale;q.norm_encoding_admitted=true;
        q.arithmetic=strix::MtpInputFusionArithmetic::split_projections_then_add;
        q.embedding=v(r.embedding);q.wide_hidden=v(r.wide);q.embedding_norm=v(enorm);q.hidden_norm=v(hnorm);q.packed_projection=v(packed);
        q.projection_row_stride_bytes=2*H/32*34;q.normalized_scratch=v(r.normalized);q.projection_scratch=v(r.projected);q.output=v(out);return q;
    };
    for(std::uint32_t t=0;t<N;++t){rows.push_back(std::make_unique<Row>(scratch));rows.back()->embedding.write(random_bf16(H,g));rows.back()->wide.write(random_bf16(H*HC,g,3));}
    bool all=true;
    for(const auto rounding:{strix::HipMtpInputRounding::fp32,strix::HipMtpInputRounding::bf16_rne}){
        for(std::uint32_t t=0;t<N;++t) strix::launch_hip_mtp_input_fusion_draft(request(*rows[t],rows[t]->ref,rounding),st.s);
        check(hipStreamSynchronize(st.s));
        std::vector<std::vector<float>> norm_ref;for(auto& r:rows) norm_ref.push_back(r->normalized.read());
        for(std::uint32_t count=1;count<=N;++count){
            const auto perm=order(N,g);std::vector<strix::HipMtpInputFusionRequest> q;
            for(std::uint32_t r=0;r<count;++r){rows[perm[r]]->out.poison();rows[perm[r]]->normalized.poison();q.push_back(request(*rows[perm[r]],rows[perm[r]]->out,rounding));}
            strix::launch_hip_mtp_input_fusion_draft_rows(q,st.s);check(hipStreamSynchronize(st.s));
            for(std::uint32_t r=0;r<count;++r){where="fusion count="+std::to_string(count)+" row="+std::to_string(r)+" (out,normalized)";
                all=all && same(rows[perm[r]]->out.read(),rows[perm[r]]->ref.read()) && same(rows[perm[r]]->normalized.read(),norm_ref[perm[r]]);}
        }
    }
    ok("batched draft input fusion == one-token draft fusion per row (both roundings, 1..8 rows, permuted)",all);
}

// HC mix: draft split-K/lane-chain mix with and without injection.
void hyper_cases(){
    Stream st;std::mt19937 g(13);
    constexpr std::uint32_t W=H*HC,R=320;
    Device<std::uint8_t> down(static_cast<std::size_t>(R)*W/32*34),inject(static_cast<std::size_t>(HC)*W/32*34),up(static_cast<std::size_t>(W)*R/32*34);
    q8_random(down,50,-8,st.s);q8_random(inject,51,-12,st.s);q8_random(up,52,-8,st.s);
    Device<float> norm(W),inject_f32(HC*W);norm.write(random_f32(W,g,1,0.2F));
    strix::launch_hip_mtp_dequantize_q8(inject.p,HC*W,inject_f32.p,st.s);check(hipStreamSynchronize(st.s));
    struct Row {Device<std::uint16_t> input{W},out{H},ref{H};Device<float> ws{W+R},inj{HC},inj_ref{HC};};
    constexpr std::uint32_t N=strix::hip_mtp_draft_batch_rows;
    std::vector<std::unique_ptr<Row>> rows;
    for(std::uint32_t t=0;t<N;++t){rows.push_back(std::make_unique<Row>());rows.back()->input.write(random_bf16(W,g,3));}
    bool all=true;
    for(const bool with_inject:{true,false}){
        for(auto& r:rows) strix::enqueue_hip_mtp_hyper_mix_q8(r->input.p,norm.p,down.p,with_inject?static_cast<const void*>(inject_f32.p):nullptr,up.p,
            r->ws.p,with_inject?r->inj_ref.p:nullptr,r->ref.p,1e-6F,true,st.s);
        check(hipStreamSynchronize(st.s));
        // The F32 workspace (normalised streams and low-rank activations) is compared too, not only the BF16 output.
        std::vector<std::vector<float>> ws_ref;for(auto& r:rows) ws_ref.push_back(r->ws.read());
        for(std::uint32_t count=1;count<=N;++count){
            const auto perm=order(N,g);strix::HipMtpHyperDraftRows v{};
            for(std::uint32_t r=0;r<count;++r){auto& row=*rows[perm[r]];row.out.poison();row.inj.poison();row.ws.poison();
                v.input[r]=row.input.p;v.output[r]=row.out.p;v.workspace[r]=row.ws.p;v.injection[r]=with_inject?row.inj.p:nullptr;}
            strix::enqueue_hip_mtp_hyper_mix_q8_rows(v,count,norm.p,down.p,with_inject?inject_f32.p:nullptr,up.p,1e-6F,st.s);
            check(hipStreamSynchronize(st.s));
            for(std::uint32_t r=0;r<count;++r){auto& row=*rows[perm[r]];
                where="hc inject="+std::to_string(with_inject)+" count="+std::to_string(count)+" row="+std::to_string(r)+" (out,ws,inj)";
                all=all && same(row.out.read(),row.ref.read()) && same(row.ws.read(),ws_ref[perm[r]]) &&
                    (!with_inject || same(row.inj.read(),row.inj_ref.read()));}
        }
    }
    ok("batched draft HC mix == one-token draft HC mix per row (output, injection, F32 workspace; with/without injection, 1..8 rows, permuted)",all);
}

// Draft head: Q4 subset heads with one base set and different context rows (shared prefix + own tails), every
// head's logits, selected token, probability and runner-up probability.
void head_cases(bool runner_up){
    Stream st;std::mt19937 g(runner_up?17U:19U);
    constexpr std::uint32_t V=12000,N=strix::hip_mtp_draft_batch_rows,depth=8;
    Device<std::uint8_t> head(static_cast<std::size_t>(V)*H/32*34);q8_random(head,60,-9,st.s);
    std::vector<std::uint32_t> ids(V);std::iota(ids.begin(),ids.end(),0U);std::shuffle(ids.begin(),ids.end(),g);
    std::vector<std::uint32_t> base(ids.begin(),ids.begin()+6000);
    std::vector<std::unique_ptr<Stream>> streams;std::vector<std::unique_ptr<strix::HipMtpDraftHead>> heads;
    for(std::uint32_t t=0;t<N;++t){
        streams.push_back(std::make_unique<Stream>());
        strix::HipMtpDraftHeadConfig c;c.hidden_size=H;c.vocab_size=V;c.max_depth=depth;c.full=false;c.base_tokens=base;c.context_capacity=1024;
        c.q4=true;c.runner_up=runner_up;
        heads.push_back(std::make_unique<strix::HipMtpDraftHead>(head.p,c,streams.back()->s));
        // Context rows: heads 0 and 1 observe the same tokens (longer shared prefix), the others their own.
        std::vector<std::uint32_t> seen;std::uniform_int_distribution<std::uint32_t> pick(0,V-1);
        std::mt19937 gc(t<2U?99U:100U+t);std::uniform_int_distribution<std::uint32_t> pc(0,V-1);
        for(int i=0;i<300;++i) seen.push_back(pc(gc));
        heads.back()->observe(seen);check(hipStreamSynchronize(streams.back()->s));
    }
    std::vector<strix::HipMtpDraftHead*> all_heads;for(auto& h:heads) all_heads.push_back(h.get());
    const auto shared_all=strix::HipMtpDraftHead::shared_rows(all_heads);
    ok(std::string("draft head shared prefix covers the base rows")+(runner_up?" (runner-up)":""),shared_all!=UINT32_MAX && shared_all>=6000U);
    std::vector<std::unique_ptr<Device<std::uint16_t>>> hidden;
    for(std::uint32_t t=0;t<N;++t){hidden.push_back(std::make_unique<Device<std::uint16_t>>(H));}
    bool all=true;
    for(std::uint32_t count=1;count<=N;++count){
        const auto perm=order(N,g);
        std::vector<strix::HipMtpDraftHead*> hs;std::vector<const std::uint16_t*> xs;
        for(std::uint32_t r=0;r<count;++r){hs.push_back(heads[perm[r]].get());xs.push_back(hidden[perm[r]]->p);}
        const auto shared=strix::HipMtpDraftHead::shared_rows(hs);
        // Reference: each head alone on its own stream, steps 0..2 with fresh hidden rows.
        std::vector<std::vector<std::vector<float>>> logits(count);std::vector<std::vector<std::uint32_t>> tokens(count);
        std::vector<std::vector<float>> prob(count),second(count);
        std::vector<std::vector<std::vector<std::uint16_t>>> hid(count);
        for(std::uint32_t r=0;r<count;++r){
            hs[r]->begin_chain();
            for(std::uint32_t step=0;step<3;++step){
                hid[r].push_back(random_bf16(H,g,2));hidden[perm[r]]->write(hid[r].back());
                hs[r]->launch(hidden[perm[r]]->p,step,streams[perm[r]]->s);check(hipStreamSynchronize(streams[perm[r]]->s));
                prob[r].push_back(hs[r]->probability(step));second[r].push_back(hs[r]->runner_up_probability(step));
                std::vector<float> l(hs[r]->stats().rows);check(hipMemcpy(l.data(),hs[r]->logits(),l.size()*4,hipMemcpyDeviceToHost));
                logits[r].push_back(l);
            }
            tokens[r]=hs[r]->read(3);
        }
        for(std::uint32_t r=0;r<count;++r){hs[r]->begin_chain();check(hipMemset(const_cast<float*>(hs[r]->logits()),0x7f,hs[r]->stats().rows*4));}
        for(std::uint32_t step=0;step<3;++step){
            for(std::uint32_t r=0;r<count;++r) hidden[perm[r]]->write(hid[r][step]);
            strix::HipMtpDraftHead::launch_rows(hs,xs,step,shared,st.s);
            for(std::uint32_t r=0;r<count;++r){hs[r]->wait_step(step,st.s);
                all=all && std::bit_cast<std::uint32_t>(hs[r]->probability(step))==std::bit_cast<std::uint32_t>(prob[r][step]) &&
                    std::bit_cast<std::uint32_t>(hs[r]->runner_up_probability(step))==std::bit_cast<std::uint32_t>(second[r][step]);}
            check(hipStreamSynchronize(st.s));
            for(std::uint32_t r=0;r<count;++r){
                std::vector<float> l(hs[r]->stats().rows);check(hipMemcpy(l.data(),hs[r]->logits(),l.size()*4,hipMemcpyDeviceToHost));
                all=all && same(l,logits[r][step]);
            }
        }
        for(std::uint32_t r=0;r<count;++r){
            all=all && hs[r]->read(3)==tokens[r];
            std::vector<std::uint32_t> slot(3);check(hipMemcpy(slot.data(),hs[r]->device_token(0),12,hipMemcpyDeviceToHost));all=all && slot==tokens[r];
        }
    }
    ok(std::string("batched draft head == one-token head per row (logits per step, slots, tokens, probability")+(runner_up?", runner-up":"")+", 1..8 rows)",all);
    // Row-local fault: head 0 gets a NaN hidden row. Alone its step fails (read() throws, probability 0); in the batch
    // exactly that happens to head 0 while every other head stays exact.
    {
        auto nan_row=random_bf16(H,g,2);nan_row[7]=0x7fc0U;
        std::vector<std::vector<std::uint16_t>> x(N);for(auto& v:x) v=random_bf16(H,g,2);x[0]=nan_row;
        std::vector<float> p_alone(N),q_alone(N);std::vector<int> threw_alone(N,0);std::vector<std::uint32_t> tok_alone(N,0);
        for(std::uint32_t r=0;r<N;++r){heads[r]->begin_chain();hidden[r]->write(x[r]);heads[r]->launch(hidden[r]->p,0,streams[r]->s);
            check(hipStreamSynchronize(streams[r]->s));p_alone[r]=heads[r]->probability(0);q_alone[r]=heads[r]->runner_up_probability(0);
            try{tok_alone[r]=heads[r]->read(1)[0];}catch(const std::exception&){threw_alone[r]=1;}}
        std::vector<const std::uint16_t*> xs;for(std::uint32_t r=0;r<N;++r){heads[r]->begin_chain();xs.push_back(hidden[r]->p);}
        strix::HipMtpDraftHead::launch_rows(all_heads,xs,0,shared_all,st.s);check(hipStreamSynchronize(st.s));
        bool iso=threw_alone[0]==1;
        for(std::uint32_t r=0;r<N;++r){
            int threw=0;std::uint32_t tok=0;try{tok=heads[r]->read(1)[0];}catch(const std::exception&){threw=1;}
            iso=iso && threw==threw_alone[r] && (threw || tok==tok_alone[r]) &&
                std::bit_cast<std::uint32_t>(heads[r]->probability(0))==std::bit_cast<std::uint32_t>(p_alone[r]) &&
                std::bit_cast<std::uint32_t>(heads[r]->runner_up_probability(0))==std::bit_cast<std::uint32_t>(q_alone[r]);
        }
        ok("batched draft head row-local fault: only the NaN head fails, the others stay exact",iso);
    }
}
} // namespace
int main(){
    try{
        gemv_cases(false);gemv_cases(true);bf16_linear_cases();moe_cases();fusion_cases();hyper_cases();head_cases(false);head_cases(true);
    }catch(const std::exception& e){std::printf("FAIL exception: %s\n",e.what());return 1;}
    std::printf("%s (%d failures)\n",failures?"FAILED":"ALL PASS",failures);
    return failures?1:0;
}
