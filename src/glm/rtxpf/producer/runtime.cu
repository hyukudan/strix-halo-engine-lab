// CUDA port of project-owned node mHC kernels; retain ascending chains and FP32 nonlinear arithmetic.
#include "../kda.cuh"
#include "../mtp.cuh"
#include "runtime.hpp"
#include "policy.hpp"
#include "state_slices.hpp"
#include "strix/serve/json.hpp"
#include "strix/model_content_hash.hpp"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fstream>
#include <map>
#include <future>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <chrono>
#include "shards.hpp"
#include "schedule.hpp"
#include "read_ahead.hpp"
#include "verified_io.hpp"

namespace strix::glm::rtxpf::producer {
namespace {
constexpr unsigned kHc=4,kHidden=4096,kHcMix=24,kHcIters=20;
constexpr float kEps=1.e-5f,kHcEps=1.e-6f;
using strix::serve::Json;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
__device__ float bf16_to_float(U16 v){return __uint_as_float(U32(v)<<16);}
__device__ float inv_sqrt(float x){return 1.f/sqrtf(x);}
__device__ float sigmoid(float x){return 1.f/(1.f+declared_exp(-x));}
__device__ U32 ordered(float v){U32 bits=__float_as_uint(v);if((bits&0x7fffffffu)==0)bits=0;return bits&0x80000000u?~bits:bits^0x80000000u;}
__device__ void flag_if(unsigned* f,float x){if(f&&!isfinite(x))atomicOr(f,1u);}
template<unsigned L> __device__ float lds_tree(float* s,float a){unsigned l=threadIdx.x;s[l]=a;__syncthreads();for(unsigned h=L/2;h;h/=2){if(l<h)s[l]+=s[l+h];__syncthreads();}float v=s[0];__syncthreads();return v;}
__global__ __launch_bounds__(256) void hc_pre_kernel(const float* X, const float* fn, const float* base,
                                                     const float* scale, float* F, float* mix, float* xc) {
    constexpr unsigned n = kHc * kHidden;
    __shared__ float s[256];
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float* Fr = F + static_cast<std::size_t>(r) * n;
    float a = 0.0F;
    for (unsigned i = l; i < n; i += 256) a = __fmaf_rn(Xr[i], Xr[i], a);
    const float ss = lds_tree<256>(s, a);
    const float rr = inv_sqrt(ss / static_cast<float>(n) + kEps);
    for (unsigned i = l; i < n; i += 256) Fr[i] = Xr[i] * rr;
    __syncthreads();
    if (l < kHcMix) {  // 24 chains over the normalized streams (each lane reads the F values it did not write: the
                       // __syncthreads above orders global writes within the block)
        float acc = 0.0F;
        const float* row = fn + static_cast<std::size_t>(l) * n;
        for (unsigned i = 0; i < n; ++i) acc = __fmaf_rn(Fr[i], row[i], acc);
        z[l] = acc;
    }
    __syncthreads();
    if (l == 0) {  // R6 on lane 0 (host hc_mix order)
        for (unsigned j = 0; j < kHc; ++j) m[j] = sigmoid(z[j] * scale[0] + base[j]) + kHcEps;
        for (unsigned j = 0; j < kHc; ++j) m[kHc + j] = 2.0F * sigmoid(z[kHc + j] * scale[1] + base[kHc + j]);
        float* c = m + 2 * kHc;
        for (unsigned k = 0; k < kHc * kHc; ++k) c[k] = z[2 * kHc + k] * scale[2] + base[2 * kHc + k];
        for (unsigned j = 0; j < kHc; ++j) {
            float mx = c[j * kHc];
            for (unsigned k = 1; k < kHc; ++k) mx = c[j * kHc + k] > mx ? c[j * kHc + k] : mx;
            float e[kHc];
            for (unsigned k = 0; k < kHc; ++k) e[k] = declared_exp(c[j * kHc + k] - mx);
            const float sum = ((e[0] + e[1]) + e[2]) + e[3];
            for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = e[k] / sum + kHcEps;
        }
        for (unsigned it = 0; it < kHcIters; ++it) {
            if (it > 0) {  // rows
                float sr[kHc];
                for (unsigned j = 0; j < kHc; ++j)
                    sr[j] = (((c[j * kHc + 0] + c[j * kHc + 1]) + c[j * kHc + 2]) + c[j * kHc + 3]) + kHcEps;
                for (unsigned j = 0; j < kHc; ++j)
                    for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sr[j];
            }
            float sc[kHc];  // columns
            for (unsigned k = 0; k < kHc; ++k)
                sc[k] = (((c[0 * kHc + k] + c[1 * kHc + k]) + c[2 * kHc + k]) + c[3 * kHc + k]) + kHcEps;
            for (unsigned j = 0; j < kHc; ++j)
                for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sc[k];
        }
        for (unsigned k = 0; k < kHcMix; ++k) mix[static_cast<std::size_t>(r) * kHcMix + k] = m[k];
    }
    __syncthreads();
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
}

__global__ void hc_expand_kernel(const float* X, const float* mix, const float* y, float* Xo, unsigned* flag, unsigned flag_stride) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    if (flag) flag += static_cast<std::size_t>(r) * flag_stride;  // stride 0: one flag for the call; 1: per-row flags (spec)
    const float* m = mix + static_cast<std::size_t>(r) * kHcMix;
    const float* c = m + 2 * kHc;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    const float yd = y[static_cast<std::size_t>(r) * kHidden + d];
    for (unsigned i = 0; i < kHc; ++i) {
        const float mx = ((c[0 * kHc + i] * Xr[d] + c[1 * kHc + i] * Xr[kHidden + d]) + c[2 * kHc + i] * Xr[2 * kHidden + d]) +
                         c[3 * kHc + i] * Xr[3 * kHidden + d];
        const float v = m[kHc + i] * yd + mx;
        flag_if(flag, v);
        Xo[static_cast<std::size_t>(r) * kHc * kHidden + i * kHidden + d] = v;
    }
}

__global__ void hc_final_kernel(const float* X, float* h) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    h[static_cast<std::size_t>(r) * kHidden + d] = (((Xr[d] + Xr[kHidden + d]) + Xr[2 * kHidden + d]) + Xr[3 * kHidden + d]) * 0.25F;
}


__global__ void norm(const float* x,const U16* w,float* y,unsigned rows){
    __shared__ float s[256];unsigned r=blockIdx.x,l=threadIdx.x;float a=0;
    for(unsigned d=l;d<4096;d+=256){float v=x[std::size_t(r)*4096+d];a=__fmaf_rn(v,v,a);}
    float rr=inv_sqrt(lds_tree<256>(s,a)/4096+1.e-5f);
    for(unsigned d=l;d<4096;d+=256)y[std::size_t(r)*4096+d]=(x[std::size_t(r)*4096+d]*rr)*bf16_to_float(w[d]);
}
__global__ void embed(const U16* e,const U32* ids,float* x,unsigned rows,unsigned streams){
    auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j>=std::size_t(rows)*streams*4096)return;
    unsigned r=j/(streams*4096),d=j%4096;x[j]=bf16_to_float(e[std::size_t(ids[r])*4096+d]);
}
__global__ void add(const float* a,const float* b,float* y,std::size_t n,unsigned* flag){auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j<n){float v=a[j]+b[j];flag_if(flag,v);y[j]=v;}}
__global__ void router(const float* x,const U16* w,const float* bias,U32* ids,float* rates,unsigned* flag){
    __shared__ float sg[288];__shared__ U32 keys[288];unsigned r=blockIdx.x,e=threadIdx.x;
    if(e<288){float v=0;for(unsigned k=0;k<4096;++k)v=__fmaf_rn(x[std::size_t(r)*4096+k],hf(w[std::size_t(e)*4096+k]),v);flag_if(flag,v);sg[e]=sigmoid(v);keys[e]=ordered(sg[e]+bias[e]);}__syncthreads();if(e)return;
    bool used[288]{};unsigned pick[8];for(unsigned j=0;j<8;++j){unsigned best=288;for(unsigned k=0;k<288;++k)if(!used[k]&&(best==288||keys[k]>keys[best]))best=k;used[best]=true;pick[j]=best;}
    for(unsigned j=1;j<8;++j)for(unsigned k=j;k&&pick[k-1]>pick[k];--k){unsigned t=pick[k-1];pick[k-1]=pick[k];pick[k]=t;}
    float den=0;for(unsigned j=0;j<8;++j)den+=sg[pick[j]];den+=1.e-20f;
    for(unsigned j=0;j<8;++j){ids[std::size_t(r)*8+j]=pick[j];rates[std::size_t(r)*8+j]=sg[pick[j]]/den*2.5f;}
}
__global__ void check_finite(const float* x,std::size_t n,unsigned* flag){auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j<n)flag_if(flag,x[j]);}
struct Buffer {
    void* p=nullptr;std::size_t bytes=0;bool pinned=false;
    Buffer()=default;Buffer(const Buffer&)=delete;
    void reset(){if(p){if(pinned){cudaHostUnregister(p);VirtualFree(p,0,MEM_RELEASE);}else cudaFree(p);}p=nullptr;bytes=0;}
    ~Buffer(){reset();}
    bool allocate(std::size_t n,bool host=false,bool optional=false){need(!p,"double allocation");pinned=host;bytes=n;if(!n)return true;
        cudaError_t e;
        if(host){p=VirtualAlloc(nullptr,io_round(n),MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);e=p?cudaHostRegister(p,io_round(n),cudaHostRegisterDefault):cudaErrorMemoryAllocation;}
        else e=cudaMalloc(&p,n);
        if(e!=cudaSuccess){cudaGetLastError();if(host&&p)VirtualFree(p,0,MEM_RELEASE);p=nullptr;bytes=0;if(!optional)ck(e);return false;}return true;}
    template<class T>T* as()const{return static_cast<T*>(p);}
};
struct Stream {cudaStream_t s{};Stream(){ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s)cudaStreamDestroy(s);}};
struct Event {cudaEvent_t e{};Event(){ck(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));}~Event(){if(e)cudaEventDestroy(e);}};
std::uint64_t integer(const Json& j){auto n=j.as_int();need(n>=0,"negative cache integer");return std::uint64_t(n);}
const Json& at(const Json& j,const std::string& key){auto* v=j.find(key);need(v!=nullptr,"missing manifest field");return *v;}
std::string str(const Json& j,const std::string& key){return at(j,key).as_string();}
std::pair<std::uint64_t,std::uint64_t> host_free(){MEMORYSTATUSEX m{};m.dwLength=sizeof m;need(GlobalMemoryStatusEx(&m)!=0,"host memory status");return {m.ullAvailPhys,m.ullAvailPageFile};}
struct Hc {const float *fn,*base,*scale;};
struct Layer {
    KdaWeights k{};DsaWeights d{};MtpWeights m{};Hc attn{},ffn{};
    const U16 *input=nullptr,*post=nullptr,*router=nullptr;
    const float* bias=nullptr;
    Expert shared{},dense{};Buffer descriptors,host_descriptors,shared_descriptor;Event ready;
};
struct Blob {std::string name,sha;std::size_t bytes;Buffer device,host,file_cache;bool cache_ready=false;std::size_t source_bytes=0;CoalescedPlan plan;std::unique_ptr<VerifiedFile> file;std::unique_ptr<TierReader> reader;};
struct Ring {Buffer device;Event retired;bool used=false;};
struct FileRing {double read_ms=0;Buffer host;Event loaded;};
// Two templates per layer: complete row tile and exact (unpadded) tail.
// Rebinding recaptures host launch parameters, then updates the existing graph
// executable. This bounds graph memory independently of prefix length.
struct GraphTemplate {
    cudaGraphExec_t executable=nullptr;std::string binding;double binding_ms=0;
    ~GraphTemplate(){if(executable)cudaGraphExecDestroy(executable);}
    template<class F> void launch(cudaStream_t stream,const std::string& key,F enqueue){
        binding_ms=0;auto begin=std::chrono::steady_clock::now();
        if(!executable||binding!=key){
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            cudaGraph_t graph=nullptr;
            try{enqueue();ck(cudaStreamEndCapture(stream,&graph));}
            catch(...){cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
            try{
                bool instantiate=!executable;
                if(executable){cudaGraphExecUpdateResultInfo info{};auto status=cudaGraphExecUpdate(executable,graph,&info);instantiate=status!=cudaSuccess||info.result!=cudaGraphExecUpdateSuccess;if(instantiate){cudaGetLastError();ck(cudaGraphExecDestroy(executable));executable=nullptr;}}
                if(instantiate){ck(cudaGraphInstantiate(&executable,graph,0));std::size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total));need(total-free<=99860000000ULL,"graph allocation exceeds 99.86 GB device cap");}
                binding=key;cudaGraphDestroy(graph);graph=nullptr;binding_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            }catch(...){if(graph)cudaGraphDestroy(graph);if(executable)cudaGraphExecDestroy(executable);executable=nullptr;binding.clear();throw;}
        }
        ck(cudaGraphLaunch(executable,stream));
    }
};

}
struct Runtime::Impl {
    std::filesystem::path cache;std::unique_ptr<Shards> shards;bool graphs;GraphTemplate graph[46][2];std::map<std::string,const Json*> tensor_index;Json manifest;std::string model_hash;unsigned capacity,tile;
    Stream compute,h2d,decode;CutlassExecution exec{};std::vector<cudaEvent_t> exec_events;
    std::map<std::string,std::unique_ptr<Blob>> blobs;std::vector<std::unique_ptr<Layer>> layers;
    Ring ring[2];FileRing file_ring[2];std::unique_ptr<LayerReadAhead> read_ahead;unsigned host_floor_gib=8;int file_cache_mode=1;std::uint64_t cache_bytes=0;DriveStats request_drive_before;unsigned device_count=0,host_count=0,file_count=0;bool staged[46]{};bool serial_stage=false,legacy_index=false,require_file_tier=false,both_file_rings=false;unsigned file_ring_mask=0;std::uint64_t file_ring_bytes[2]{};
    Buffer X,X2,F,mix,collapsed,normalized,part,routed,shared,hidden,embedding,tokens,ids,rates,workspace,flag;
    Buffer state,conv,latent,key,gate,pooled;std::size_t ws_bytes=0;Json profile;
    std::chrono::steady_clock::time_point request_start;Json releases=Json::array(),last_timings=Json::object();double staging_ms=0,file_read_ms=0,source_load_ms=0,startup_file_verify_ms=0,graph_binding_ms=0;std::uint64_t file_bytes=0,source_load_bytes=0,startup_file_verify_bytes=0;
    Impl(const std::string& path,unsigned cap,unsigned rows,bool from_shards,bool use_graphs,unsigned floor_gib,const VerifiedManifest* verified,unsigned max_device,unsigned max_pinned,bool require_file,const std::string& weight_dirs,unsigned drive_readers,int cache_mode):cache(path),graphs(use_graphs),capacity(cap),tile(effective_row_tile(cap,rows)){
        rows=tile;host_floor_gib=floor_gib;file_cache_mode=cache_mode;need(cache_mode>=0&&cache_mode<=2,"file cache mode");
        need(max_device<=43&&max_pinned<=43,"layer limits must be 0..43");
        require_file_tier=require_file||caps_require_file_tier(cap,max_device,max_pinned);both_file_rings=require_file;
        need(valid_schedule(cap,rows),"capacity 4-aligned <=131072; tile 32-aligned <=min(capacity,32768)");
        // Retain deny-write/delete handles before verification and all reads.
        // Persistent hits require the complete filesystem identity, not size.
        auto source_begin=std::chrono::steady_clock::now();
        if(from_shards){shards=std::make_unique<Shards>(path);manifest=shards->manifest;}else {need(verified!=nullptr,"cache requires held verified manifest");manifest=verified->parsed;}
        need(str(manifest,"format")=="glm-native-cache-v1"&&str(manifest,"layout")=="k16-n16-msb-mul1"&&str(manifest,"decoder")=="mul1-8tiles-v2"&&str(manifest,"tile")=="cutlass-c1-16experts-v1"&&str(manifest,"transform")=="h128-rn16-v1","cache version mismatch");
        model_hash=str(manifest,"model_sha256");need(model_hash.size()==64,"model hash");
        auto key_contract=Json::object();
        for(auto name:{"decoder","format","layout","model_sha256","tile","transform"})key_contract.set(name,at(manifest,name));
        auto key_text=key_contract.dump();need(strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(key_text))))==str(manifest,"cache_key"),"cache key mismatch");
        auto source_text=at(manifest,"sources").dump();need(strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(source_text))))==model_hash,"source manifest identity mismatch");
        for(const auto& [name,j]:at(manifest,"blobs").as_object()){
            need(std::filesystem::path(name).filename()==name,"cache path traversal");auto b=std::make_unique<Blob>();b->name=name;b->sha=str(j,"sha256");b->bytes=integer(at(j,"bytes"));
            if(!shards)b->file=std::make_unique<VerifiedFile>(cache/name,b->bytes);blobs.emplace(name,std::move(b));
        }
        need(blobs.size()==44&&blobs.contains("base.bin"),"full-model cache requires base +43 expert layers");
        std::atomic<unsigned> hash_hits=shards?shards->verified_hash_hits:0;
        if(!shards){std::vector<Blob*> jobs;for(auto& [name,b]:blobs)jobs.push_back(b.get());auto hash_cache=default_hash_cache();
            parallel_hash_files(jobs.size(),[&](std::size_t j){auto& b=*jobs[j];if(b.file->verify(b.sha,hash_cache))++hash_hits;});}
        if(shards)shards->enable_direct(weight_dirs,drive_readers);
        else for(auto& [name,b]:blobs){b->reader=std::make_unique<TierReader>(*b->file,weight_dirs,drive_readers);b->plan=CoalescedPlan({{0,0,b->bytes}},b->bytes);}
        for(auto& [name,b]:blobs)b->source_bytes=shards?shards->direct_bytes(name):b->plan.bytes;
        auto resident_before=drive_stats();
        auto source_inventory_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-source_begin).count();
        for(const auto& [name,t]:at(manifest,"tensors").as_object()){
            tensor_index.emplace(name,&t);auto bi=blobs.find(str(t,"blob"));need(bi!=blobs.end(),"tensor blob");auto off=integer(at(t,"offset")),n=integer(at(t,"bytes"));need(off%256==0&&off<=bi->second->bytes&&n<=bi->second->bytes-off,"tensor bounds/alignment");
        }
        int count=0;ck(cudaGetDeviceCount(&count));need(count==1,"one visible device required");cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
        need(prop.major==12&&prop.minor==0&&std::string(prop.name).find("PRO")!=std::string::npos,"requires sm_120 PRO");
        ws_bytes=std::max({workspace_bytes_cutlass(rows,288,4096,2048),workspace_bytes_cutlass(rows,1,4096,12288),kda_workspace_bytes(rows,4096,64),dsa_workspace_bytes(rows,cap),mtp_workspace_bytes(rows,cap)});
        need(ws_bytes>0,"workspace shape");
        std::size_t largest=0;for(auto& [name,b]:blobs)if(name!="base.bin")largest=std::max(largest,b->bytes);
        std::size_t initial_free,initial_total;ck(cudaMemGetInfo(&initial_free,&initial_total));
        const auto fixed=activation_bytes(cap,tile)+64*128*128*4+3*64*128*3*4+
            std::uint64_t(cap)*1664+ws_bytes+blobs.at("base.bin")->bytes+2*largest+(4ULL<<20);
        auto existing=initial_total-initial_free;
        need(admit(fixed,initial_free,2ULL<<30)&&existing+fixed+(2ULL<<30)<=99860000000ULL,"fixed allocations exceed free VRAM/99.86 GB cap");
        std::size_t largest_source=0;for(auto& [name,b]:blobs)if(name!="base.bin")largest_source=std::max(largest_source,b->source_bytes);
        auto [physical,commit]=host_free();need(2*io_round(largest_source)+Shards::direct_batch_bytes<=host_budget(physical,commit,floor_gib),"insufficient physical RAM/commit for bounded pinned rings");
        // Only residual streams and final hidden span the prefix; attention,
        // HC, FFN dispatch and MTP scratch use the declared row tile.
        auto full=std::size_t(cap)*4096*4,row_bytes=std::size_t(tile)*4096*4;
        X.allocate(full*4);X2.allocate(row_bytes*4);F.allocate(row_bytes*4);mix.allocate(std::size_t(tile)*24*4);
        collapsed.allocate(row_bytes);normalized.allocate(row_bytes);part.allocate(row_bytes);routed.allocate(row_bytes);shared.allocate(row_bytes);hidden.allocate(full);embedding.allocate(row_bytes);
        tokens.allocate(std::size_t(cap+1)*4);ids.allocate(std::size_t(tile)*8*4);rates.allocate(std::size_t(tile)*8*4);flag.allocate(4);
        state.allocate(64*128*128*4);conv.allocate(3*64*128*3*4);latent.allocate(std::size_t(cap)*1024);key.allocate(std::size_t(cap)*256);gate.allocate(std::size_t(cap)*256);pooled.allocate(std::size_t(cap/4)*512);
        workspace.allocate(ws_bytes);
        auto& base=*blobs.at("base.bin");base.device.allocate(base.bytes);
        // Cold nonexpert loads use bounded 16-deep direct pinned batches.
        Buffer staging;staging.allocate(Shards::direct_batch_bytes,true);
        // On a startup scan/submit failure, drain H2D before staging unwinds.
        struct UploadDrain {cudaStream_t stream;~UploadDrain(){cudaStreamSynchronize(stream);}} startup_uploads{h2d.s};
        ck(cudaMemsetAsync(base.device.p,0,base.bytes,h2d.s));
        if(shards){auto begin=std::chrono::steady_clock::now();shards->stream_direct("base.bin",staging.p,[&](std::uint64_t off,const void* p,std::size_t n){ck(cudaMemcpyAsync(base.device.as<char>()+off,p,n,cudaMemcpyHostToDevice,h2d.s));},[&]{ck(cudaStreamSynchronize(h2d.s));});source_load_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();source_load_bytes+=base.bytes;}
        else {BlobFiniteScan scan(blob_finite_ranges(manifest,"base.bin"),base.bytes);
            for(std::size_t first=0;first<base.plan.reads.size();first+=16){std::vector<ss_mfile_io> ios;
            auto end=std::min(first+16,base.plan.reads.size()),offset=std::size_t(base.plan.reads[first].source);
            for(auto i=first;i<end;++i){auto r=base.plan.reads[i];ios.push_back({staging.as<char>()+r.source-offset,r.source,r.bytes});}
            auto last=base.plan.reads[end-1];auto bytes=std::min<std::size_t>(base.bytes,last.source+last.bytes)-offset;
            // Scan each held batch before it can reach the device.
            base.reader->read(ios);scan.append(offset,staging.p,bytes);ck(cudaMemcpyAsync(base.device.as<char>()+offset,staging.p,bytes,cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));}
            scan.finish();}
        staging.reset();
        for(auto& r:ring)r.device.allocate(largest);
        for(auto& r:file_ring)r.host.allocate(largest_source,true);
        auto target=targets(cap);std::size_t free,total;ck(cudaMemGetInfo(&free,&total));
        std::uint64_t used=total-free,cap_bytes=99860000000ULL;
        auto device_budget=std::min<std::uint64_t>(free,used<cap_bytes?cap_bytes-used:0);
        for(unsigned l=3;l<=45;++l){auto& b=*blobs.at(expert_name(l));bool admitted=false;
            if(device_count<std::min(target.device,max_device)&&admit(b.bytes,device_budget,2ULL<<30)&&b.device.allocate(b.bytes,false,true)){
                load(b,file_ring[0].host.p);upload(b,b.device.p,file_ring[0].host.p);ck(cudaStreamSynchronize(h2d.s));device_budget-=b.bytes;++device_count;admitted=true;
            }
            if(!admitted){auto [physical,commit]=host_free();if(host_count<max_pinned&&io_round(b.source_bytes)<=host_budget(physical,commit,floor_gib)&&b.host.allocate(b.source_bytes,true,true)){load(b,b.host.p);++host_count;admitted=true;}}
            if(!admitted){++file_count;file_ring_mask|=1u<<(l%2);
                // All file-tier floating weights are checked ONCE at startup
                // through the same immutable authenticated source handles.
                void* raw=file_ring[0].host.p;
                if(cache_allocate(b))raw=b.file_cache.p;
                load(b,raw,false);b.cache_ready=b.file_cache.p!=nullptr;
            }
        }
        reset_read_ahead(false);
        need(!require_file_tier||file_tier_covered(file_count,1,file_ring_mask,both_file_rings),"requested file tier requires admitted file layers and both ring parities");
        exec.decode_stream=decode.s;ck(prepare_cutlass(exec));
        auto ev=[&](){cudaEvent_t e{};ck(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));exec_events.push_back(e);return e;};
        exec.dispatch=ev();for(unsigned i=0;i<18;++i){exec.ready[i]=ev();exec.retired[i]=ev();}
        ck(prepare_kda());ck(prepare_dsa());
        for(unsigned l=0;l<46;++l){auto w=std::make_unique<Layer>();auto p=layer_name(l);w->input=u16(p+"input_layernorm.weight");w->post=u16(p+"post_attention_layernorm.weight");
            if(l<45){auto hc=[&](const char* site){auto n=p+site;return Hc{f32(n+"fn"),f32(n+"base"),f32(n+"scale")};};w->attn=hc("hc_attn_");w->ffn=hc("hc_ffn_");}
            auto a=p+"self_attn.";
            if(l<45&&l%4!=3)w->k={4096,64,matrix(a+"qkv_proj",4096,24576),matrix(a+"o_proj",8192,4096),u16(a+"b_proj.weight"),u16(a+"f_a_proj.weight"),u16(a+"f_b_proj.weight"),u16(a+"g_a_proj.weight"),u16(a+"g_b_proj.weight"),u16(a+"conv1d.weight"),u16(a+"o_norm.weight"),f32(a+"A_log"),f32(a+"dt_bias")};
            else w->d={matrix(a+"q_a_proj",4096,1536),matrix(a+"q_b_proj",1536,16384),matrix(a+"kv_a_proj_with_mqa",4096,512),matrix(a+"o_proj",16384,4096),matrix(a+"indexer.wq_b",1536,4096),u16(a+"q_a_layernorm.weight"),u16(a+"kv_a_layernorm.weight"),u16(a+"kv_b_proj.weight"),u16(a+"indexer.wk.weight"),u16(a+"indexer.index_kpool_compress_gate"),u16(a+"indexer.weights_proj.weight"),u16(a+"indexer.k_norm.weight"),u16(a+"indexer.k_norm.bias"),f32(a+"indexer.index_kpool_compress_ape")};
            if(l<3)w->dense=expert(p+"mlp.",12288);else {w->router=u16(p+"mlp.gate.weight");w->bias=f32(p+"mlp.gate.e_score_correction_bias");w->shared=expert(p+"mlp.shared_experts.",2048);w->descriptors.allocate(288*sizeof(Expert));}
            w->shared_descriptor.allocate(sizeof(Expert));auto descriptor=l<3?w->dense:w->shared;
            ck(cudaMemcpyAsync(w->shared_descriptor.p,&descriptor,sizeof descriptor,cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));
            if(l==45){w->m.layer=45;w->m.eh=matrix(p+"eh_proj",8192,4096);w->m.attention=w->d;w->m.enorm=u16(p+"enorm.weight");w->m.hnorm=u16(p+"hnorm.weight");w->m.input_norm=w->input;w->m.post_norm=w->post;w->m.head_norm=u16(p+"shared_head.norm.weight");w->m.router=w->router;w->m.router_bias=w->bias;w->m.experts=w->descriptors.as<Expert>();w->m.shared=w->shared_descriptor.as<Expert>();}
            if(l>=3){auto& b=*blobs.at(expert_name(l));const void* base=b.device.p?b.device.p:ring[l%2].device.p;
                w->host_descriptors.allocate(288*sizeof(Expert),true);auto* desc=w->host_descriptors.as<Expert>();
                for(unsigned e=0;e<288;++e)desc[e]=expert(p+"mlp.experts."+std::to_string(e)+".",2048,base);
                // Every layer has a fixed resident address or fixed ring parity.
                // These descriptors are immutable for the lifetime of the service.
                ck(cudaMemcpyAsync(w->descriptors.p,desc,288*sizeof(Expert),cudaMemcpyHostToDevice,h2d.s));}
            layers.push_back(std::move(w));
        }
        profile=Json::object();profile.set("startup_drive_io",drive_receipt(drive_stats(),resident_before));profile.set("startup_file_verify_ms",startup_file_verify_ms);profile.set("startup_file_verify_bytes",startup_file_verify_bytes);profile.set("integrity_binding","held FileId with locked canonical namespace");profile.set("drive_readers",int(drive_readers));profile.set("file_cache_mode",file_cache_mode==0?"off":file_cache_mode==1?"auto":"on");profile.set("file_cache_bytes",cache_bytes);profile.set("finite_verification","startup over held immutable bytes");profile.set("finite_verification_basis",shards?"source tensor dtypes per shard slice":"cache manifest tensor offsets/dtypes per blob");profile.set("max_device_layers",int(max_device));profile.set("max_pinned_layers",int(max_pinned));profile.set("require_file_tier",require_file_tier);profile.set("require_both_file_rings",both_file_rings);profile.set("file_ring_mask",int(file_ring_mask));profile.set("weight_source",shards?"shards":"cache");profile.set("source_inventory_hash_ms",source_inventory_ms);profile.set("source_load_ms",source_load_ms);profile.set("source_load_bytes",source_load_bytes);profile.set("graph_templates",graphs);profile.set("graph_template_limit",92);profile.set("workspace_rows",int(tile));
        profile.set("verified_hash_hits",int(hash_hits.load()));profile.set("hash_workers",int(std::max(1u,std::thread::hardware_concurrency())));profile.set("host_floor_gib",int(floor_gib));profile.set("file_io_chunk_bytes",std::uint64_t(CoalescedPlan::target));profile.set("file_io_min_bytes",std::uint64_t(CoalescedPlan::minimum));profile.set("file_io_max_bytes",std::uint64_t(CoalescedPlan::maximum));profile.set("cold_staging_bytes",std::uint64_t(Shards::direct_batch_bytes));profile.set("file_io_inflight",int(drive_readers));profile.set("file_io_inflight_basis","per-volume workers, each issues concurrent 4 MiB subreads");
        std::uint64_t request_bytes=0;for(auto& [name,b]:blobs)if(name!="base.bin"&&!b->device.p&&!b->host.p)request_bytes+=b->bytes;profile.set("file_tier_bytes_per_request",request_bytes);profile.set("estimated_file_tier_host_ms",source_load_bytes?double(request_bytes)*source_load_ms/double(source_load_bytes):0.0);profile.set("cost_estimate_basis","resident load throughput; excludes overlapped GPU compute; cold storage may differ");profile.set("device_layers",int(device_count));profile.set("pinned_layers",int(host_count));profile.set("file_layers",int(file_count));profile.set("compressed_rings",2);profile.set("decoded_buffers",2);profile.set("workspace_bytes",std::uint64_t(ws_bytes));profile.set("read_ahead_layers",2);profile.set("h2d_streams",1);
        ck(cudaMemGetInfo(&free,&total));profile.set("device_free_bytes",std::uint64_t(free));profile.set("device_used_bytes",std::uint64_t(total-free));
        auto [remaining_physical,remaining_commit]=host_free();profile.set("host_available_physical_bytes",remaining_physical);profile.set("host_available_commit_bytes",remaining_commit);
        ck(cudaStreamSynchronize(h2d.s));
    }
    ~Impl(){if(read_ahead)try{read_ahead->drain();}catch(...){}cudaStreamSynchronize(compute.s);cudaStreamSynchronize(h2d.s);cudaStreamSynchronize(decode.s);for(auto e:exec_events)cudaEventDestroy(e);}
    static std::string layer_name(unsigned l){return "model.language_model.layers."+std::to_string(l)+".";}
    static std::string expert_name(unsigned l){return std::string("experts-")+(l<10?"0":"")+std::to_string(l)+".bin";}
    const Json& tensor(const std::string& name)const{auto it=tensor_index.find(name);need(it!=tensor_index.end(),"missing model tensor");return *it->second;}
    void* pointer(const std::string& name,const void* expert_base=nullptr)const{
        const auto& t=tensor(name);auto& b=*blobs.at(str(t,"blob"));const void* base=b.device.p?b.device.p:expert_base;need(base!=nullptr,"nonresident tensor without ring");return const_cast<char*>(static_cast<const char*>(base))+integer(at(t,"offset"));
    }
    const U16* u16(const std::string& n)const{auto dt=str(tensor(n),"dtype");need(dt=="F16"||dt=="BF16","expected 16-bit tensor");return static_cast<U16*>(pointer(n));}
    const float* f32(const std::string& n)const{need(str(tensor(n),"dtype")=="F32","expected FP32 tensor");return static_cast<float*>(pointer(n));}
    Matrix matrix(const std::string& n,unsigned in,unsigned out,const void* ring_base=nullptr)const{
        auto& t=tensor(n+".trellis");auto& sh=at(t,"shape").as_array();need(str(t,"dtype")=="I16"&&sh.size()==3&&integer(sh[0])==in/16&&integer(sh[1])==out/16,"matrix geometry");unsigned rate=unsigned(integer(sh[2])/16);need(integer(sh[2])==rate*16&&rate>=3&&rate<=6,"matrix rate");
        return {static_cast<U16*>(pointer(n+".trellis",ring_base)),static_cast<U16*>(pointer(n+".suh",ring_base)),static_cast<U16*>(pointer(n+".svh",ring_base)),in,out,rate};
    }
    Expert expert(const std::string& n,unsigned inter,const void* base=nullptr)const{return {matrix(n+"gate_proj",4096,inter,base),matrix(n+"up_proj",4096,inter,base),matrix(n+"down_proj",inter,4096,base)};}
    DriveStats drive_stats()const {if(shards)return shards->drive_stats();DriveStats stats;for(auto& [name,b]:blobs)if(b->reader)b->reader->counters(stats);return stats;}
    bool cache_floor_ok()const {auto [physical,commit]=host_free();return host_budget(physical,commit,host_floor_gib)>=(1ULL<<30);}
    bool cache_admits(std::size_t bytes)const {
        auto current=host_free();if(!admit(io_round(bytes),host_budget(current.first,current.second,host_floor_gib),1ULL<<30))return false;
        auto a=host_free();Sleep(50);auto b=host_free();Sleep(50);auto c=host_free();
        return file_cache_admits(io_round(bytes),{a.first,b.first,c.first},{a.second,b.second,c.second},host_floor_gib);
    }
    bool cache_allocate(Blob& b){
        if(!file_cache_mode||require_file_tier||b.device.p||b.host.p||b.file_cache.p||!cache_admits(b.source_bytes))return false;
        if(!b.file_cache.allocate(b.source_bytes,true,true))return false;
        if(!cache_floor_ok()){b.file_cache.reset();return false;}
        cache_bytes+=io_round(b.source_bytes);return true;
    }
    void raw_read(Blob& b,void* out){if(shards)shards->read_direct(b.name,out);else {
        std::vector<ss_mfile_io> ios;for(auto r:b.plan.reads)ios.push_back({static_cast<char*>(out)+r.buffer,r.source,r.bytes});b.reader->read(ios);}}
    void upload(Blob& b,void* device,const void* raw){
        if(shards){ck(cudaMemsetAsync(device,0,b.bytes,h2d.s));shards->emit_direct(b.name,raw,[&](std::uint64_t off,const void* p,std::size_t n){ck(cudaMemcpyAsync(static_cast<char*>(device)+off,p,n,cudaMemcpyHostToDevice,h2d.s));});}
        else ck(cudaMemcpyAsync(device,raw,b.bytes,cudaMemcpyHostToDevice,h2d.s));
    }
    void load(Blob& b,void* out,bool resident=true){auto begin=std::chrono::steady_clock::now();raw_read(b,out);
        if(shards)shards->emit_direct(b.name,out,[](std::uint64_t,const void*,std::size_t){},true);
        else {BlobFiniteScan scan(blob_finite_ranges(manifest,b.name),b.bytes);scan.append(0,out,b.bytes);scan.finish();}
        auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();if(resident){source_load_ms+=ms;source_load_bytes+=b.bytes;}else{startup_file_verify_ms+=ms;startup_file_verify_bytes+=b.bytes;}}
    void read_file(unsigned slot,unsigned layer){auto& b=*blobs.at(expert_name(layer));auto& r=file_ring[slot];auto begin=std::chrono::steady_clock::now();raw_read(b,b.file_cache.p?b.file_cache.p:r.host.p);r.read_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
    void reset_read_ahead(bool start){
        if(read_ahead)read_ahead->drain();std::vector<unsigned> files;
        for(unsigned l=3;l<=45;++l){auto& b=*blobs.at(expert_name(l));if(!b.device.p&&!b.host.p&&!b.cache_ready)files.push_back(l);}
        read_ahead=std::make_unique<LayerReadAhead>(std::move(files));if(start)read_ahead->reset([this](unsigned s,unsigned l){read_file(s,l);});
    }
    void stage(unsigned l,bool acquire=true){if(l<3||l>45)return;
        if(staged[l]){if(acquire)ck(cudaStreamWaitEvent(compute.s,layers[l]->ready.e,0));return;}
        auto& b=*blobs.at(expert_name(l));auto& r=ring[l%2];
        if(!b.device.p){if(r.used)ck(cudaStreamWaitEvent(h2d.s,r.retired.e,0));
            const void* host=b.host.p?b.host.p:(b.cache_ready?b.file_cache.p:nullptr);unsigned slot=0;bool file=!host;
            if(file){slot=read_ahead->acquire(l);host=b.file_cache.p?b.file_cache.p:file_ring[slot].host.p;b.cache_ready=b.file_cache.p!=nullptr;file_read_ms+=file_ring[slot].read_ms;file_bytes+=b.bytes;file_ring_bytes[l%2]+=b.bytes;}
            upload(b,r.device.p,host);r.used=true;
            if(file){ck(cudaEventRecord(file_ring[slot].loaded.e,h2d.s));read_ahead->release(l,[this](unsigned s,unsigned layer){read_file(s,layer);},[this](unsigned s){ck(cudaEventSynchronize(file_ring[s].loaded.e));});}
        }
        ck(cudaEventRecord(layers[l]->ready.e,h2d.s));staged[l]=true;
        if(serial_stage)ck(cudaEventSynchronize(layers[l]->ready.e));
        if(acquire)ck(cudaStreamWaitEvent(compute.s,layers[l]->ready.e,0));
    }
    DsaState dstate(){return {latent.as<U16>(),key.as<U16>(),gate.as<U16>(),pooled.as<float>(),capacity};}
    void zero_state(){for(Buffer* b:{&state,&conv,&latent,&key,&gate,&pooled})ck(cudaMemsetAsync(b->p,0,b->bytes,compute.s));}
    void pre(const Hc& hc,const U16* weight,const float* x,float* normalized_out,unsigned n){
        hc_pre_kernel<<<n,256,0,compute.s>>>(x,hc.fn,hc.base,hc.scale,F.as<float>(),mix.as<float>(),collapsed.as<float>());
        norm<<<n,256,0,compute.s>>>(collapsed.as<float>(),weight,normalized_out,n);
    }
    void post(const float* x,const float* y,float* out,unsigned n){hc_expand_kernel<<<dim3(16,n),256,0,compute.s>>>(x,mix.as<float>(),y,out,flag.as<unsigned>(),0);}
    void export_state(unsigned l,unsigned P,const StateSink& sink){
        ck(cudaStreamSynchronize(compute.s));unsigned bad=0;ck(cudaMemcpy(&bad,flag.p,4,cudaMemcpyDeviceToHost));need(!bad,"nonfinite prefix; abort handoff");
        auto release=Json::object();release.set("layer",int(l));release.set("state_ready_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-request_start).count());releases.push(release);
        auto download=[&](const void* p,std::size_t bytes){std::vector<std::byte> out(bytes);ck(cudaMemcpy(out.data(),p,bytes,cudaMemcpyDeviceToHost));return out;};
        if(l<45&&l%4!=3){
            auto s=download(state.p,state.bytes),c=download(conv.p,conv.bytes);
            for(unsigned rank=0;rank<2;++rank){sink(rank,l,0,kda_tp_state(s,rank));
                // Node serialization is q/k/v-major. Select 32 heads separately
                // inside each part, then concatenate (never interleave heads).
                auto sliced=kda_tp_conv(c,rank);
                sink(rank,l,1,sliced);
            }
        }else{
            for(auto pair:{std::pair<Buffer*,unsigned>{&latent,2},{&key,3},{&gate,4},{&pooled,5}}){auto bytes=pair.second==2?std::size_t(P)*1024:pair.second==5?std::size_t(P/4)*512:std::size_t(P)*256;if(!bytes)continue;auto out=download(pair.first->p,bytes);for(unsigned rank=0;rank<2;++rank)sink(rank,l,pair.second,out);}
        }
    }
    DsaIndex index(const std::string& id){if(id=="glm-pf-rtx-native-v1"||id==dsa_identity)return DsaIndex::fp32;for(auto v:{DsaIndex::mma,DsaIndex::mma_proj_fp32_score,DsaIndex::fp32_proj_mma_score,DsaIndex::tf32x3})if(id==dsa_index_identity(v))return v;throw std::runtime_error("unsupported producer identity");}
    void run(std::span<const U32> input,const std::string& identity,const StateSink& sink){
        request_start=std::chrono::steady_clock::now();releases=Json::array();staging_ms=0;file_read_ms=0;file_bytes=0;file_ring_bytes[0]=file_ring_bytes[1]=0;graph_binding_ms=0;
        if(read_ahead)read_ahead->drain();ck(cudaStreamSynchronize(h2d.s));
        // Memory pressure can revoke optional caches between requests. Retained
        // pinned residency is separately budgeted; explicit file gates bypass cache.
        if(cache_bytes&&!cache_floor_ok()){for(auto& [name,b]:blobs){b->file_cache.reset();b->cache_ready=false;}cache_bytes=0;}
        // RAM released by other workloads can become cache space later. Newly
        // admitted slots are filled by the ordinary read-ahead, directly, and
        // become hits only after acquire has observed successful completion.
        for(auto& [name,b]:blobs)if(name!="base.bin")cache_allocate(*b);
        profile.set("file_cache_bytes",cache_bytes);
        request_drive_before=drive_stats();reset_read_ahead(true);
        std::fill(std::begin(staged),std::end(staged),false);
        need(input.size()>=2&&input.size()<=capacity,"prompt exceeds resident capacity");for(auto t:input)need(t<154880,"token outside vocabulary");unsigned P=unsigned(input.size()-1);auto mode=index(identity);
        ck(cudaMemsetAsync(flag.p,0,4,compute.s));ck(cudaMemcpyAsync(tokens.p,input.data(),input.size_bytes(),cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));
        embed<<<unsigned((std::size_t(P)*4*4096+255)/256),256,0,compute.s>>>(u16("model.language_model.embed_tokens.weight"),tokens.as<U32>(),X.as<float>(),P,4);
        // Entire prefix for each layer. No state from a later layer becomes
        // visible before the preceding layer's last prefix row is complete.
        for(unsigned l=0;l<45;++l){auto& w=*layers[l];zero_state();auto stage_start=std::chrono::steady_clock::now();stage(l);staging_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-stage_start).count();
            for(unsigned first=0;first<P;first+=tile){auto chunk=row_tile(first,P,tile);unsigned n=chunk.rows;auto off=std::size_t(first)*4096;
                auto sequence=[&]{
                    pre(w.attn,w.input,X.as<float>()+off*4,normalized.as<float>(),n);
                    if(l%4!=3){KdaCall v{n,w.k,normalized.as<float>(),part.as<float>(),state.as<float>(),conv.as<float>()};ck(enqueue_kda(v,workspace.p,ws_bytes,compute.s));}
                    else {DsaCall v{};v.rows=n;v.first=first;v.w=w.d;v.state=dstate();v.x=normalized.as<float>();v.y=part.as<float>();v.nonfinite=flag.as<unsigned>();v.index=mode;v.legacy_fp32_index=legacy_index;ck(enqueue_dsa(v,workspace.p,ws_bytes,compute.s));}
                    post(X.as<float>()+off*4,part.as<float>(),X2.as<float>(),n);
                    pre(w.ffn,w.post,X2.as<float>(),normalized.as<float>(),n);
                    if(l<3){Call v{n,1,4096,12288,normalized.as<float>(),nullptr,nullptr,w.shared_descriptor.as<Expert>(),part.as<float>()};ck(enqueue_cutlass(v,workspace.p,ws_bytes,compute.s,exec));}
                    else {router<<<n,320,0,compute.s>>>(normalized.as<float>(),w.router,w.bias,ids.as<U32>(),rates.as<float>(),flag.as<unsigned>());
                        Call v{n,288,4096,2048,normalized.as<float>(),ids.as<U32>(),rates.as<float>(),w.descriptors.as<Expert>(),routed.as<float>()};ck(enqueue_cutlass(v,workspace.p,ws_bytes,compute.s,exec));
                        Call sh{n,1,4096,2048,normalized.as<float>(),nullptr,nullptr,w.shared_descriptor.as<Expert>(),shared.as<float>()};ck(enqueue_cutlass(sh,workspace.p,ws_bytes,compute.s,exec));
                        add<<<unsigned((std::size_t(n)*4096+255)/256),256,0,compute.s>>>(routed.as<float>(),shared.as<float>(),part.as<float>(),std::size_t(n)*4096,flag.as<unsigned>());
                    }
                    post(X2.as<float>(),part.as<float>(),X.as<float>()+off*4,n);
                };
                auto binding=std::to_string(first)+":"+std::to_string(n)+":"+identity+":"+std::to_string(legacy_index);
                if(graphs){auto& g=graph[l][chunk.bucket];g.launch(compute.s,binding,sequence);graph_binding_ms+=g.binding_ms;}else sequence();
            }
            // H2D for N+1 runs while N's queued kernels execute. Acquisition
            // occurs only at N+1; exporting N never waits for N+1's ready event.
            if(!serial_stage){auto begin=std::chrono::steady_clock::now();stage(l+1,false);staging_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
            check_finite<<<unsigned((state.bytes/4+255)/256),256,0,compute.s>>>(state.as<float>(),state.bytes/4,flag.as<unsigned>());
            export_state(l,P,sink);
            if(l>=3)ck(cudaEventRecord(ring[l%2].retired.e,compute.s));
        }
        for(unsigned first=0;first<P;first+=tile){auto chunk=row_tile(first,P,tile);unsigned n=chunk.rows;auto off=std::size_t(first)*4096;hc_final_kernel<<<dim3(16,n),256,0,compute.s>>>(X.as<float>()+off*4,part.as<float>());norm<<<n,256,0,compute.s>>>(part.as<float>(),u16("model.language_model.norm.weight"),hidden.as<float>()+off,n);}
        zero_state();auto stage_start=std::chrono::steady_clock::now();stage(45);staging_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-stage_start).count();auto& w=*layers[45];
        // Each exact row tile pairs hidden[t] with input[t+1]. Absolute DSA
        // positions and complete pools survive between calls; no padded tokens.
        for(unsigned first=0;first<P;first+=tile){auto chunk=row_tile(first,P,tile);unsigned n=chunk.rows;auto off=std::size_t(first)*4096;
            auto sequence=[&]{embed<<<unsigned((std::size_t(n)*4096+255)/256),256,0,compute.s>>>(u16("model.language_model.embed_tokens.weight"),tokens.as<U32>()+first+1,embedding.as<float>(),n,1);
                MtpCall v{n,first,w.m,dstate(),embedding.as<float>(),hidden.as<float>()+off,part.as<float>(),flag.as<unsigned>(),mode};v.legacy_fp32_index=legacy_index;ck(enqueue_mtp(v,workspace.p,ws_bytes,compute.s,exec));};
            auto binding=std::to_string(first)+":"+std::to_string(n)+":"+identity+":"+std::to_string(legacy_index);if(graphs){auto& g=graph[45][chunk.bucket];g.launch(compute.s,binding,sequence);graph_binding_ms+=g.binding_ms;}else sequence();
        }
        ck(cudaEventRecord(ring[1].retired.e,compute.s));export_state(45,P,sink);ck(cudaStreamSynchronize(compute.s));ck(cudaGetLastError());
        unsigned read_mask=(file_ring_bytes[0]?1u:0u)|(file_ring_bytes[1]?2u:0u);
        need(!require_file_tier||file_tier_covered(file_count,file_bytes,read_mask,both_file_rings),"requested file tier performed no reads or missed a ring");
        last_timings=Json::object();last_timings.set("drive_io",drive_receipt(drive_stats(),request_drive_before));last_timings.set("file_cache_bytes",cache_bytes);last_timings.set("file_ring0_bytes",file_ring_bytes[0]);last_timings.set("file_ring1_bytes",file_ring_bytes[1]);last_timings.set("layer_state_releases",releases);last_timings.set("trunk_staging_host_ms",staging_ms);last_timings.set("graph_binding_host_ms",graph_binding_ms);last_timings.set("file_tier_host_read_ms",file_read_ms);last_timings.set("file_tier_bytes",file_bytes);last_timings.set("runtime_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-request_start).count());
    }
};
Runtime::Runtime(const std::string& path,unsigned cap,unsigned rows,bool source,bool graphs,unsigned floor_gib,bool legacy,bool serial,const VerifiedManifest* verified,unsigned max_device,unsigned max_pinned,bool require_file,const std::string& weight_dirs,unsigned drive_readers,int cache_mode):p_(std::make_unique<Impl>(path,cap,rows,source,graphs,floor_gib,verified,max_device,max_pinned,require_file,weight_dirs,drive_readers,cache_mode)){p_->legacy_index=legacy;p_->serial_stage=serial;p_->profile.set("legacy_fp32_index",legacy);p_->profile.set("serial_stage",serial);}
Runtime::~Runtime()=default;
std::string Runtime::model_sha256()const{return p_->model_hash;}
std::string Runtime::timings()const{return p_->last_timings.dump();}
std::string Runtime::self_check(std::span<const U32> input,const std::string& identity,unsigned reference_tile){
    need(valid_schedule(p_->capacity,reference_tile)&&reference_tile<=p_->tile,"self-check baseline must fit allocated tile");
    struct Snapshot{unsigned rank,layer,component;std::vector<std::byte> bytes;};std::vector<Snapshot> old;
    const unsigned candidate_tile=p_->tile;const bool candidate_serial=p_->serial_stage,candidate_legacy=p_->legacy_index;auto report=Json::object();report.set("residency",p_->profile);
    strix::Sha256Fast old_hash[2],new_hash[2];
    try{
        p_->tile=reference_tile;p_->serial_stage=true;p_->legacy_index=true;
        prefill(input,identity,[&](unsigned rank,unsigned layer,unsigned component,std::span<const std::byte> bytes){old_hash[rank].update(bytes);old.push_back({rank,layer,component,{bytes.begin(),bytes.end()}});});
        report.set("old_timings",p_->last_timings);
        std::vector<std::byte> old_hidden(std::size_t(input.size()-1)*4096*4);ck(cudaMemcpy(old_hidden.data(),p_->hidden.p,old_hidden.size(),cudaMemcpyDeviceToHost));
        p_->tile=candidate_tile;p_->serial_stage=candidate_serial;p_->legacy_index=candidate_legacy;
        std::size_t next=0;std::uint64_t different=0,compared=0,memcmp_failures=0;
        auto compare=[&](std::span<const std::byte> a,std::span<const std::byte> b){need(a.size()==b.size(),"self-check state extent");memcmp_failures+=std::memcmp(a.data(),b.data(),a.size())!=0;
            for(std::size_t j=0;j<a.size();++j)different+=a[j]!=b[j];compared+=a.size();};
        prefill(input,identity,[&](unsigned rank,unsigned layer,unsigned component,std::span<const std::byte> bytes){new_hash[rank].update(bytes);need(next<old.size(),"extra self-check component");auto& ref=old[next++];need(ref.rank==rank&&ref.layer==layer&&ref.component==component,"self-check state order");compare(ref.bytes,bytes);});
        need(next==old.size(),"missing self-check component");
        std::vector<std::byte> new_hidden(old_hidden.size());ck(cudaMemcpy(new_hidden.data(),p_->hidden.p,new_hidden.size(),cudaMemcpyDeviceToHost));compare(old_hidden,new_hidden);
        for(unsigned rank=0;rank<2;++rank){report.set("old_rank"+std::to_string(rank)+"_sha256",strix::model_content_identity_hex(old_hash[rank].finalize()));report.set("new_rank"+std::to_string(rank)+"_sha256",strix::model_content_identity_hex(new_hash[rank].finalize()));}
        report.set("digest_layout","rank-local state payloads in release order, no wire headers");report.set("tokens",std::uint64_t(input.size()));
        report.set("new_timings",p_->last_timings);report.set("old_row_tile",int(reference_tile));report.set("new_row_tile",int(candidate_tile));report.set("compared_bytes",compared);report.set("differing_bytes",different);report.set("memcmp_failures",memcmp_failures);report.set("verdict",different?"MISMATCH":"EXACT");report.set("qualification","DRAFT");
        return report.dump();
    }catch(...){p_->tile=candidate_tile;p_->serial_stage=candidate_serial;p_->legacy_index=candidate_legacy;throw;}
}
std::string Runtime::health()const{auto j=Json::object();j.set("status","ready");j.set("identity","glm-pf-rtx-native-v1");j.set("index_identity",dsa_identity);j.set("qualification","DRAFT");j.set("capacity",int(p_->capacity));j.set("row_tile",int(p_->tile));j.set("model_sha256",p_->model_hash);j.set("cache_key",str(p_->manifest,"cache_key"));j.set("residency",p_->profile);return j.dump();}
void Runtime::prefill(std::span<const U32> t,const std::string& id,const StateSink& sink){try{p_->run(t,id,sink);}catch(...){
    // A failed request can leave work on either stream. Drain all producers
    // before any persistent ring/event/workspace is reused by the next request.
    cudaStreamSynchronize(p_->compute.s);cudaStreamSynchronize(p_->decode.s);cudaStreamSynchronize(p_->h2d.s);
    if(p_->read_ahead)try{p_->read_ahead->drain();}catch(...){}
    for(auto& r:p_->ring)cudaEventRecord(r.retired.e,p_->compute.s);
    cudaStreamSynchronize(p_->compute.s);throw;
}}
}
