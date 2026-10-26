#include "strix/hip_ds41_dense_fp16.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <unistd.h>

namespace strix::ds41 {
namespace {
void ck(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
void lt(hipblasStatus_t e) { if (e != HIPBLAS_STATUS_SUCCESS) throw std::runtime_error("dense FP16 hipBLASLt status " + std::to_string(e)); }
template<class T> struct Buffer {
    T* p = nullptr; std::size_t n = 0;
    ~Buffer() { if (p) (void)hipFree(p); }
    void reserve(std::size_t count) { if (count <= n) return; if (p) ck(hipFree(p)); p=nullptr; n=0; ck(hipMalloc(&p,count*sizeof(T))); n=count; }
};
// lo != nullptr (weight split): lo = (f - w) * 2^12 in FP16, so w + lo / 2^12 holds f to ~22 bits (d * code needs up to 18)
constexpr float split_scale = 4096.0f;
__global__ void reconstruct(Q8A8DensePrefillView v, __half* w, unsigned* bad, __half* lo = nullptr) {
    const std::size_t i=blockIdx.x*256ULL+threadIdx.x;
    if(i>=std::size_t(v.count)*v.width)return;
    unsigned o=static_cast<unsigned>(i/v.width),k=static_cast<unsigned>(i%v.width);
    auto* b=v.weight+std::size_t(o)*((v.row_width?v.row_width:v.width)/32)*34+(v.k_offset+k)/32*34;
    float f=__half2float(*reinterpret_cast<const __half*>(b))*float(reinterpret_cast<const signed char*>(b+2)[k%32]);
    if(!isfinite(f)||fabsf(f)>65504.0f)atomicAdd(bad,1U);
    w[i]=__float2half_rn(f);
    if(lo)lo[i]=__float2half_rn((f-__half2float(w[i]))*split_scale);
}
// A power-of-two row scale keeps large finite activations within FP16 range.
// The same scale is undone after the FP32 GEMM, including accumulated K parts.
// keep_a8: the operand is the exact path's A8 activation, code * delta (codes/deltas of the view), instead of x.
__device__ __forceinline__ float act_value(const float* x,const signed char* codes,const float* deltas,unsigned stride,unsigned r,unsigned i) {
    return codes?float(codes[std::size_t(r)*stride+i])*deltas[std::size_t(r)*(stride/32)+i/32]:x[std::size_t(r)*stride+i];
}
// top: the row maximum is scaled into [2^top, 2^(top+1)); 10 for FP32 output, 0 for the FP16 output form, whose dot
// products (K up to 8192) must stay below 65504.
__global__ void activation(const float* x,const signed char* codes,const float* deltas,unsigned k,unsigned stride,__half* y,float* scales,int top=10) {
    unsigned r=blockIdx.x,t=threadIdx.x;
    float mx=0;
    for(unsigned i=t;i<k;i+=256)mx=fmaxf(mx,fabsf(act_value(x,codes,deltas,stride,r,i)));
    __shared__ float red[256];red[t]=mx;__syncthreads();
    for(unsigned d=128;d;d/=2){if(t<d)red[t]=fmaxf(red[t],red[t+d]);__syncthreads();}
    int e=red[0]>0?max(-120,min(120,top-ilogbf(red[0]))):0;
    float sc=ldexpf(1.0f,e);
    if(t==0)scales[r]=ldexpf(1.0f,-e);
    for(unsigned i=t;i<k;i+=256)y[std::size_t(r)*k+i]=__float2half_rn(act_value(x,codes,deltas,stride,r,i)*sc);
}
__global__ void finish(Q8A8DensePrefillView v,const float* y,const float* scales) {
    std::size_t i=blockIdx.x*256ULL+threadIdx.x;
    if(i>=std::size_t(v.rows)*v.count)return;
    unsigned r=static_cast<unsigned>(i/v.count),o=static_cast<unsigned>(i%v.count);
    float f=y[i]*scales[r];auto* dst=v.out+std::size_t(r)*v.out_stride+o;
    *dst=v.accumulate?*dst+f:f;
}
// FP16 output form: the views' column ranges of the joint GEMM, widened (x row scale) into their outputs.
constexpr unsigned max_views=4;
struct FastViews {
    unsigned n=0,total=0;
    unsigned first[max_views],count[max_views],skip[max_views],stride[max_views];
    float* out[max_views];
    bool accumulate[max_views];
};
__global__ void finish_fast(FastViews f,const __half* y,const float* scales,unsigned rows,unsigned* bad,const __half* ylo=nullptr) {
    std::size_t i=blockIdx.x*256ULL+threadIdx.x;
    if(i>=std::size_t(rows)*f.total)return;
    unsigned r=static_cast<unsigned>(i/f.total),o=static_cast<unsigned>(i%f.total),j=0;
    while(j+1<f.n&&o>=f.first[j+1])++j;
    if(r<f.skip[j])return;
    float v=__half2float(y[i]);if(ylo)v+=__half2float(ylo[i])*(1.0f/split_scale);if(scales)v*=scales[r];
    if(!isfinite(v))atomicAdd(bad,1U);
    float* dst=f.out[j]+std::size_t(r-f.skip[j])*f.stride[j]+(o-f.first[j]);
    *dst=f.accumulate[j]?*dst+v:v;
}
struct Plan {
    hipblasLtMatmulDesc_t op=nullptr;
    hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    hipblasLtMatmulAlgo_t algo{};
    ~Plan(){if(op)hipblasLtMatmulDescDestroy(op);if(a)hipblasLtMatrixLayoutDestroy(a);if(b)hipblasLtMatrixLayoutDestroy(b);if(c)hipblasLtMatrixLayoutDestroy(c);}
};
}
struct DenseFp16Cache::Impl {
    hipblasLtHandle_t handle=nullptr;
    int version=0;std::string arch,revision;
    using WKey=std::tuple<const void*,unsigned,unsigned,unsigned,unsigned>;
    std::map<WKey,Buffer<__half>> weights;
    std::map<std::tuple<unsigned,unsigned,unsigned>,Plan> plans;
    Buffer<__half> input;Buffer<float> output,scales;Buffer<unsigned> bad;Buffer<char> work;
    std::map<std::vector<WKey>,Buffer<__half>> joint,joint_lo;
    Buffer<__half> half_lo;
    std::map<std::tuple<unsigned,unsigned,unsigned,unsigned>,Plan> fast_plans;
    Buffer<__half> half_out;Buffer<unsigned> fast_bad;
    static constexpr std::size_t workspace=32ULL<<20;
    Impl(){lt(hipblasLtCreate(&handle));lt(hipblasLtGetVersion(handle,&version));char rev[256]{};lt(hipblasLtGetGitRevision(handle,rev));revision=rev;int dev;ck(hipGetDevice(&dev));hipDeviceProp_t p{};ck(hipGetDeviceProperties(&p,dev));arch=p.gcnArchName;work.reserve(workspace);bad.reserve(1);}
    ~Impl(){if(handle)hipblasLtDestroy(handle);}
    void run(const Q8A8DensePrefillView& v,const float* x,bool keep_a8,hipStream_t stream){
        auto& w=weights[{v.weight,v.width,v.count,v.row_width,v.k_offset}];
        if(!w.p){w.reserve(std::size_t(v.width)*v.count);ck(hipMemsetAsync(bad.p,0,4,stream));
            reconstruct<<<static_cast<unsigned>((w.n+255)/256),256,0,stream>>>(v,w.p,bad.p);unsigned invalid=0;
            ck(hipMemcpyAsync(&invalid,bad.p,4,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
            if(invalid)throw std::runtime_error("dense FP16 weight range failure");}
        input.reserve(std::size_t(v.rows)*v.width);output.reserve(std::size_t(v.rows)*v.count);scales.reserve(v.rows);
        activation<<<v.rows,256,0,stream>>>(x,keep_a8?v.codes:nullptr,v.deltas,v.width,v.x_stride,input.p,scales.p);
        auto key=std::make_tuple(v.rows,v.count,v.width);auto [it,fresh]=plans.try_emplace(key);auto& p=it->second;
        const float alpha=1,beta=0;
        auto matmul=[&](hipblasLtMatmulAlgo_t& algo){lt(hipblasLtMatmul(handle,p.op,&alpha,w.p,p.a,input.p,p.b,&beta,output.p,p.c,output.p,p.c,&algo,work.p,workspace,stream));};
        if(fresh){
            lt(hipblasLtMatmulDescCreate(&p.op,HIPBLAS_COMPUTE_32F,HIP_R_32F));
            hipblasOperation_t trans=HIPBLAS_OP_T;
            lt(hipblasLtMatmulDescSetAttribute(p.op,HIPBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
            // Y[N,R] = transpose(W[K,N]) * X[K,R], all column major.
            lt(hipblasLtMatrixLayoutCreate(&p.a,HIP_R_16F,v.width,v.count,v.width));
            lt(hipblasLtMatrixLayoutCreate(&p.b,HIP_R_16F,v.width,v.rows,v.width));
            lt(hipblasLtMatrixLayoutCreate(&p.c,HIP_R_32F,v.count,v.rows,v.count));
            std::string id="v1-"+arch+"-"+std::to_string(version)+"-"+revision+"-"+std::to_string(v.rows)+"-"+std::to_string(v.count)+"-"+std::to_string(v.width);
            const char* dir=std::getenv("STRIX_DS41_FP16_CACHE");
            std::filesystem::path path=dir?std::filesystem::path(dir)/(id+".pick"):std::filesystem::path{};
            bool cached=false;int index=-1;std::size_t bytes=0;
            if(!path.empty()){
                std::ifstream f(path);std::string read_id,extra;
                if(f>>read_id>>index && read_id==id && !(f>>extra)){
                    std::vector<int> ids{index};std::vector<hipblasLtMatmulHeuristicResult_t> results;
                    if(hipblaslt_ext::getAlgosFromIndex(handle,ids,results)==HIPBLAS_STATUS_SUCCESS && results.size()==1){
                        p.algo=results[0].algo;
                        cached=hipblaslt_ext::matmulIsAlgoSupported(handle,p.op,&alpha,p.a,p.b,&beta,p.c,p.c,p.algo,bytes)==HIPBLAS_STATUS_SUCCESS && bytes<=workspace;}
                }
            }
            if(!cached){
                const char* readonly=std::getenv("STRIX_DS41_FP16_CACHE_READONLY");
                if(readonly && std::string(readonly)=="1")throw std::runtime_error("dense FP16 missing/invalid frozen solution "+id);

                hipblasLtMatmulPreference_t pref=nullptr;lt(hipblasLtMatmulPreferenceCreate(&pref));
                auto cap=workspace;lt(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&cap,sizeof(cap)));
                hipblasLtMatmulHeuristicResult_t results[16];int count=0;
                lt(hipblasLtMatmulAlgoGetHeuristic(handle,p.op,p.a,p.b,p.c,p.c,pref,16,results,&count));hipblasLtMatmulPreferenceDestroy(pref);
                if(!count)throw std::runtime_error("dense FP16 no supported solution");
                p.algo=results[0].algo;
                const char* tune=std::getenv("STRIX_DS41_FP16_TUNE");
                if(tune && std::string(tune)=="1"){
                    hipEvent_t start,end;ck(hipEventCreate(&start));ck(hipEventCreate(&end));float best=1e30f;
                    for(int j=0;j<count;++j){if(results[j].state!=HIPBLAS_STATUS_SUCCESS||results[j].workspaceSize>workspace)continue;
                        auto candidate=results[j].algo;matmul(candidate);ck(hipEventRecord(start,stream));
                        for(int rep=0;rep<2;++rep)matmul(candidate);
                        ck(hipEventRecord(end,stream));ck(hipEventSynchronize(end));float ms;ck(hipEventElapsedTime(&ms,start,end));
                        if(ms<best){best=ms;p.algo=candidate;}}
                    ck(hipEventDestroy(start));ck(hipEventDestroy(end));
                }
                index=hipblaslt_ext::getIndexFromAlgo(p.algo);
                if(!path.empty()){
                    std::filesystem::create_directories(path.parent_path());auto tmp=path.string()+"."+std::to_string(getpid())+".tmp";
                    {std::ofstream f(tmp);f<<id<<' '<<index<<'\n';if(!f)throw std::runtime_error("dense FP16 cache write");}
                    std::filesystem::rename(tmp,path);
                }
            }
            std::fprintf(stderr,"dense_fp16 solution %s index=%d cached=%d\n",id.c_str(),index,cached);
        }
        matmul(p.algo);finish<<<static_cast<unsigned>((std::size_t(v.rows)*v.count+255)/256),256,0,stream>>>(v,output.p,scales.p);ck(hipGetLastError());
    }
    // Solution of the FP16 output GEMM for (rows, N, K): the pick file (STRIX_DS41_FP16_CACHE) of the power-of-two row
    // bucket if it supports these rows, else hipBLASLt's heuristic list (64), timed when STRIX_DS41_FP16_TUNE=1.
    // ldc > n: one group's outputs inside a row of ldc (the o-groups, one GEMM per group: faster than hipBLASLt's
    // strided batch on gfx1151, 4.0 vs 4.8 ms for 8 groups of 1024 x 4096 x 1024)
    void pick_fast(Plan& p,unsigned rows,unsigned n,unsigned k,unsigned ldb,unsigned ldc,const __half* w,hipStream_t stream){
        const float alpha=1,beta=0;
        lt(hipblasLtMatmulDescCreate(&p.op,HIPBLAS_COMPUTE_32F,HIP_R_32F));
        hipblasOperation_t trans=HIPBLAS_OP_T;
        lt(hipblasLtMatmulDescSetAttribute(p.op,HIPBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
        lt(hipblasLtMatrixLayoutCreate(&p.a,HIP_R_16F,k,n,k));
        lt(hipblasLtMatrixLayoutCreate(&p.b,HIP_R_16F,k,rows,ldb));
        lt(hipblasLtMatrixLayoutCreate(&p.c,HIP_R_16F,n,rows,ldc));
        unsigned bucket=1;while(bucket<rows)bucket*=2;
        std::string id="v3h-"+arch+"-"+std::to_string(version)+"-"+revision+"-"+std::to_string(bucket)+"-"+std::to_string(n)+"-"+std::to_string(k)+"-"+std::to_string(ldb)+(ldc!=n?"-c"+std::to_string(ldc):"");
        const char* dir=std::getenv("STRIX_DS41_FP16_CACHE");
        std::filesystem::path path=dir?std::filesystem::path(dir)/(id+".pick"):std::filesystem::path{};
        bool cached=false;int index=-1;std::size_t bytes=0;
        if(!path.empty()){
            std::ifstream f(path);std::string read_id,extra;
            if(f>>read_id>>index && read_id==id && !(f>>extra)){
                std::vector<int> ids{index};std::vector<hipblasLtMatmulHeuristicResult_t> results;
                if(hipblaslt_ext::getAlgosFromIndex(handle,ids,results)==HIPBLAS_STATUS_SUCCESS && results.size()==1){
                    p.algo=results[0].algo;
                    cached=hipblaslt_ext::matmulIsAlgoSupported(handle,p.op,&alpha,p.a,p.b,&beta,p.c,p.c,p.algo,bytes)==HIPBLAS_STATUS_SUCCESS && bytes<=workspace;}
            }
        }
        if(!cached){
            const char* readonly=std::getenv("STRIX_DS41_FP16_CACHE_READONLY");
            if(readonly && std::string(readonly)=="1")throw std::runtime_error("dense FP16 missing/invalid frozen solution "+id);
            hipblasLtMatmulPreference_t pref=nullptr;lt(hipblasLtMatmulPreferenceCreate(&pref));
            auto cap=workspace;lt(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&cap,sizeof(cap)));
            std::vector<hipblasLtMatmulHeuristicResult_t> results(64);int count=0;
            lt(hipblasLtMatmulAlgoGetHeuristic(handle,p.op,p.a,p.b,p.c,p.c,pref,64,results.data(),&count));hipblasLtMatmulPreferenceDestroy(pref);
            if(!count)throw std::runtime_error("dense FP16 output: no supported solution");
            p.algo=results[0].algo;
            const char* tune=std::getenv("STRIX_DS41_FP16_TUNE");
            if(tune && std::string(tune)=="1"){
                tune_in.reserve(std::size_t(rows)*ldb);tune_out.reserve(std::size_t(rows)*ldc);
                ck(hipMemsetAsync(tune_in.p,0,tune_in.n*2,stream));
                auto matmul=[&](hipblasLtMatmulAlgo_t& algo){lt(hipblasLtMatmul(handle,p.op,&alpha,w,p.a,tune_in.p,p.b,&beta,tune_out.p,p.c,tune_out.p,p.c,&algo,work.p,workspace,stream));};
                hipEvent_t start,end;ck(hipEventCreate(&start));ck(hipEventCreate(&end));float best=1e30f;
                for(int j=0;j<count;++j){if(results[std::size_t(j)].state!=HIPBLAS_STATUS_SUCCESS||results[std::size_t(j)].workspaceSize>workspace)continue;
                    auto candidate=results[std::size_t(j)].algo;matmul(candidate);ck(hipEventRecord(start,stream));
                    for(int rep=0;rep<3;++rep)matmul(candidate);
                    ck(hipEventRecord(end,stream));ck(hipEventSynchronize(end));float ms;ck(hipEventElapsedTime(&ms,start,end));
                    if(ms<best){best=ms;p.algo=candidate;}}
                ck(hipEventDestroy(start));ck(hipEventDestroy(end));
                std::fprintf(stderr,"dense_fp16 tuned %s candidates=%d best_ms=%.4f\n",id.c_str(),count,best/3);
            }
            index=hipblaslt_ext::getIndexFromAlgo(p.algo);
            if(!path.empty()){
                std::filesystem::create_directories(path.parent_path());auto tmp=path.string()+"."+std::to_string(getpid())+".tmp";
                {std::ofstream f(tmp);f<<id<<' '<<index<<'\n';if(!f)throw std::runtime_error("dense FP16 cache write");}
                std::filesystem::rename(tmp,path);
            }
        }
        std::fprintf(stderr,"dense_fp16 solution %s rows=%u index=%d cached=%d\n",id.c_str(),rows,index,cached);
    }
    Buffer<__half> tune_in,tune_out;
    void run_fast(const Q8A8DensePrefillView* views,unsigned nv,unsigned rows,const __half* x16,unsigned x16_stride,const float* x,
                  unsigned groups,bool split,hipStream_t stream){
        const unsigned k=views[0].width;
        if(groups>1&&nv!=1)throw std::invalid_argument("dense FP16 grouped form: one view");
        FastViews f;f.n=nv;
        std::vector<WKey> key;
        for(unsigned j=0;j<nv;++j){
            const auto& v=views[j];
            if(v.width!=k||v.rows>rows||!v.rows)throw std::invalid_argument("dense FP16 output views");
            f.first[j]=f.total;f.count[j]=v.count*groups;f.skip[j]=rows-v.rows;f.stride[j]=v.out_stride;f.out[j]=v.out;f.accumulate[j]=v.accumulate;
            f.total+=v.count*groups;key.emplace_back(v.weight,v.width,v.count*groups,v.row_width,v.k_offset);
        }
        auto& w=joint[key];
        auto& wl=joint_lo[key];
        if(split&&!wl.p){wl.reserve(std::size_t(k)*f.total);w.n=0;if(w.p)ck(hipFree(w.p));w.p=nullptr;}  // rebuild both halves
        if(!w.p){w.reserve(std::size_t(k)*f.total);ck(hipMemsetAsync(bad.p,0,4,stream));
            for(unsigned j=0;j<nv;++j){
                auto all=views[j];all.count*=groups;  // the groups' weight rows are consecutive
                reconstruct<<<static_cast<unsigned>((std::size_t(k)*all.count+255)/256),256,0,stream>>>(all,w.p+std::size_t(f.first[j])*k,bad.p,
                                                                                                        wl.p?wl.p+std::size_t(f.first[j])*k:nullptr);}
            unsigned invalid=0;ck(hipMemcpyAsync(&invalid,bad.p,4,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
            if(invalid)throw std::runtime_error("dense FP16 weight range failure");}
        if(!fast_bad.p){fast_bad.reserve(1);ck(hipMemsetAsync(fast_bad.p,0,4,stream));}
        const float* sc=nullptr;
        if(!x16){
            input.reserve(std::size_t(rows)*k*groups);scales.reserve(rows);
            activation<<<rows,256,0,stream>>>(x,nullptr,nullptr,k*groups,views[0].x_stride,input.p,scales.p,0);
            x16=input.p;x16_stride=k*groups;sc=scales.p;
        }
        auto [it,fresh]=fast_plans.try_emplace(std::make_tuple(rows,f.total,k,x16_stride*1000U+groups));auto& p=it->second;
        if(fresh)pick_fast(p,rows,f.total/groups,k,x16_stride,f.total,w.p,stream);
        half_out.reserve(std::size_t(rows)*f.total);
        const float alpha=1,beta=0;
        const std::size_t n=f.total/groups;
        for(unsigned g=0;g<groups;++g)
            lt(hipblasLtMatmul(handle,p.op,&alpha,w.p+g*n*k,p.a,x16+std::size_t(g)*k,p.b,&beta,half_out.p+g*n,p.c,half_out.p+g*n,p.c,&p.algo,work.p,workspace,stream));
        if(split){  // the low half: same solution, its own FP16 output
            half_lo.reserve(std::size_t(rows)*f.total);
            for(unsigned g=0;g<groups;++g)
                lt(hipblasLtMatmul(handle,p.op,&alpha,wl.p+g*n*k,p.a,x16+std::size_t(g)*k,p.b,&beta,half_lo.p+g*n,p.c,half_lo.p+g*n,p.c,&p.algo,work.p,workspace,stream));
        }
        finish_fast<<<static_cast<unsigned>((std::size_t(rows)*f.total+255)/256),256,0,stream>>>(f,half_out.p,sc,rows,fast_bad.p,split?half_lo.p:nullptr);
        ck(hipGetLastError());
    }
};
DenseFp16Cache::DenseFp16Cache()=default;
DenseFp16Cache::~DenseFp16Cache()=default;
void DenseFp16Cache::enqueue(const Q8A8DensePrefillView& v,const float* x,void* stream,bool keep_a8){
    if(keep_a8&&(!v.codes||!v.deltas||v.x_stride%32))throw std::invalid_argument("dense FP16 A8 view");
    if(!x||!v.weight||!v.out||!v.rows||!v.count||!v.width||v.width%32||v.x_stride<v.width||v.out_stride<v.count||
       (v.row_width&&(v.k_offset%32||v.k_offset+v.width>v.row_width)))throw std::invalid_argument("dense FP16 view");
    if(!impl_)impl_=std::make_unique<Impl>();impl_->run(v,x,keep_a8,static_cast<hipStream_t>(stream));
}
void DenseFp16Cache::enqueue_fast(const Q8A8DensePrefillView* views,unsigned count,std::uint32_t rows,const std::uint16_t* x16,
                                  std::uint32_t x16_stride,const float* x,void* stream,unsigned groups,bool split){
    if(!groups||(groups>1&&(count!=1||views[0].x_stride<views[0].width*groups||views[0].out_stride<views[0].count*groups||views[0].row_width)))
        throw std::invalid_argument("dense FP16 grouped form");
    if(!views||!count||count>max_views||!rows||(!x16&&!x)||(x16&&x16_stride<views[0].width))throw std::invalid_argument("dense FP16 output form");
    for(unsigned j=0;j<count;++j){const auto& v=views[j];
        if(!v.weight||!v.out||!v.count||!v.width||v.width%32||v.x_stride<v.width||v.out_stride<v.count||
           (v.row_width&&(v.k_offset%32||v.k_offset+v.width>v.row_width)))throw std::invalid_argument("dense FP16 view");}
    if(!impl_)impl_=std::make_unique<Impl>();
    impl_->run_fast(views,count,rows,reinterpret_cast<const __half*>(x16),x16_stride,x,groups,split,static_cast<hipStream_t>(stream));
}
unsigned DenseFp16Cache::take_nonfinite(void* stream){
    if(!impl_||!impl_->fast_bad.p)return 0;
    auto s=static_cast<hipStream_t>(stream);unsigned n=0;
    ck(hipMemcpyAsync(&n,impl_->fast_bad.p,4,hipMemcpyDeviceToHost,s));ck(hipMemsetAsync(impl_->fast_bad.p,0,4,s));ck(hipStreamSynchronize(s));
    return n;
}
}
