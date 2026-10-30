// Research only. Including the implementation gives this executable the actual
// production kernels and decoders without exporting another dispatch interface.
#include "../src/hip_moe_q4k_a8_known_rows.hip"
#include "moe_int8_wmma_fixture.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_content_hash.hpp"
#if __has_include("strix/build_info.hpp")
#include "strix/build_info.hpp"
#endif
#include <chrono>
#include <cfenv>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <sys/utsname.h>
#include <unistd.h>

namespace probe {
namespace kr=strix::moe_q4k_a8::known_rows;
namespace a8=strix::moe_q4k_a8;
using namespace i8fixture;
using kr::at;
inline void gpu(hipError_t e,const char* why) { if(e!=hipSuccess)throw std::runtime_error(std::string(why)+": "+hipGetErrorString(e)); }
struct Buffer {
    void* p=nullptr;std::size_t stride=0,size=0;
    Buffer()=default;
    Buffer(std::size_t rows,std::size_t width):stride((width+255)/256*256),size(rows*stride){gpu(hipMalloc(&p,size),"allocation");}
    ~Buffer(){if(p)(void)hipFree(p);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    Buffer(Buffer&& b) noexcept {std::swap(p,b.p);std::swap(stride,b.stride);std::swap(size,b.size);}
    Buffer& operator=(Buffer&& b) noexcept {std::swap(p,b.p);std::swap(stride,b.stride);std::swap(size,b.size);return *this;}
    template<class T> kr::StridedRows<T> rows()const{return {static_cast<T*>(p),stride};}
    template<class T> T get(std::size_t row,std::size_t column)const {
        T value;gpu(hipMemcpy(&value,static_cast<const unsigned char*>(p)+row*stride+column*sizeof(T),sizeof(T),hipMemcpyDeviceToHost),"sample");return value;
    }
    void upload(const void* src,std::size_t bytes)const{require(bytes<=size,"upload overflow");gpu(hipMemcpy(p,src,bytes,hipMemcpyHostToDevice),"upload");}
};
struct Event {
    hipEvent_t e{};Event(){gpu(hipEventCreate(&e),"event");}~Event(){(void)hipEventDestroy(e);}
    Event(const Event&)=delete;Event& operator=(const Event&)=delete;
    void record(hipStream_t s){gpu(hipEventRecord(e,s),"record");}
};
using Clock=std::chrono::steady_clock;
struct Budget {
    Clock::time_point start=Clock::now();double seconds;
    explicit Budget(double s):seconds(s){}
    double left()const{return seconds-std::chrono::duration<double>(Clock::now()-start).count();}
    void check(double reserve=3)const{require(left()>reserve,"400-second turn budget exhausted (inconclusive)");}
};
struct Tensor {
    unsigned type=0;std::vector<unsigned char> host;Buffer device;
    Tensor(const strix::gguf::TensorInfo& t,const std::string& path):type(t.ggml_type),host(t.byte_size),device(1,t.byte_size) {
        std::ifstream f(path,std::ios::binary);f.seekg(static_cast<std::streamoff>(t.absolute_offset));
        f.read(reinterpret_cast<char*>(host.data()),static_cast<std::streamsize>(host.size()));require(bool(f),"short packed tensor read");
        device.upload(host.data(),host.size());
    }
    const unsigned char* row(unsigned expert,unsigned out,unsigned width,unsigned outputs)const {
        return host.data()+(std::size_t(expert)*outputs+out)*row_bytes(type,width);
    }
    const unsigned char* ptr()const{return static_cast<const unsigned char*>(device.p);}
};
struct Layer {
    Tensor gate,up,down;
    static const strix::gguf::TensorInfo& find(const strix::gguf::FileInfo& info,unsigned layer,const char* suffix) {
        const auto name="blk."+std::to_string(layer)+"."+suffix+".weight";
        for(const auto& t:info.tensors)if(t.name==name)return t;
        throw std::runtime_error("required packed layer tensor missing");
    }
    Layer(const strix::gguf::FileInfo& info,unsigned layer,const std::string& path):
        gate(validate(find(info,layer,"ffn_gate_exps"),true),path),
        up(validate(find(info,layer,"ffn_up_exps"),true),path),
        down(validate(find(info,layer,"ffn_down_exps"),false),path){}
    static const strix::gguf::TensorInfo& validate(const strix::gguf::TensorInfo& t,bool gu) {
        require(t.dimensions==std::vector<std::uint64_t>{gu?H:I,gu?I:H,E},"GGUF layer shape differs from 2560/640/512");
        require(gu?t.ggml_type==12:(t.ggml_type==7||t.ggml_type==8),"need Q4_K gate/up and Q5_1 or Q8_0 down");
        require(t.byte_size==std::size_t(E)*(gu?I:H)*row_bytes(t.ggml_type,gu?H:I),"packed tensor byte size mismatch");return t;
    }
};
// Same CSR layout, explicitly owned at research T16384. Production's host
// admission and its static operand pools remain capped at T8192.
strix::q51_known_rows::ScratchLayout research_layout(unsigned rows) {
    require(rows>=16&&rows<=16384&&rows%8==0,"research row bound");
    const auto align=[](std::uint64_t x){return (x+255)/256*256;};
    strix::q51_known_rows::ScratchLayout s;s.offsets=256;s.touched=s.offsets+align(4ULL*(E+1));s.rows=s.touched+align(4ULL*E);
    s.slots=s.rows+align(4ULL*rows*K);s.batch_experts=s.slots+align(4ULL*rows*K);s.batch_begins=s.batch_experts+align(4ULL*rows*K);
    s.bytes=s.batch_begins+align(4ULL*(rows*K+1));return s;
}
kr::PrefillRows shift(kr::PrefillRows v,unsigned offset) {
    const auto move=[&](auto& a){a.base=reinterpret_cast<decltype(a.base)>(reinterpret_cast<std::uintptr_t>(a.base)+offset*a.stride);};
    move(v.input);move(v.routes);move(v.route_weights);move(v.codes);move(v.deltas);move(v.sums);move(v.status);
    move(v.output);move(v.clear);move(v.down_codes);move(v.down_deltas);move(v.down_sums);move(v.rawdots);return v;
}
struct Fixture {
    unsigned T;bool balanced;Layer& layer;hipStream_t stream{};a8::Request request{};kr::PrefillRows v{};
    Buffer input,routes,rweights,codes,deltas,sums,status,output,clear,dcodes,ddeltas,dsums,raw,csr,tap,a16,image,arrive,csrscratch;
    std::vector<float> x,rw;std::vector<unsigned> rt;
    Fixture(Layer& l,unsigned rows,bool bal):T(rows),balanced(bal),layer(l),
        input(T,2*H),routes(T,4*K),rweights(T,4*K),codes(T,H),deltas(T,4*(H/32)),sums(T,4*(H/32)),status(T,4),
        output(T,4*K*2*I),clear(T,4*H),dcodes(T,K*I),ddeltas(T,4*K*(I/32)),dsums(T,4*K*(I/32)),raw(T,4*K*H),
        csr(1,research_layout(T).bytes),tap(T,4*K*3*I),a16(T,2*H),image(1,std::size_t(T)*K*I*2),arrive(1,4*(T*K+E)),
        csrscratch(1,4ULL*(kr::csr_mw_head+((T*K+kr::csr_mw_span-1)/kr::csr_mw_span)*512)),
        x(std::size_t(T)*H),rw(std::size_t(T)*K),rt(std::size_t(T)*K) {
        gpu(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"stream");
        request.enabled=true;request.input=a8::InputKind::bf16;request.shape={H,I,E,K,T};request.gate_type=request.up_type=12;
        std::vector<std::uint16_t> bf(std::size_t(T)*input.stride/2);
        std::vector<unsigned> routing(std::size_t(T)*routes.stride/4);
        std::vector<float> weights(std::size_t(T)*rweights.stride/4);
        std::array<double,E> cdf{};double total=0;
        for(unsigned e=0;e<E;++e){total+=std::pow(double(e+1),-1.5);cdf[e]=total;}
        for(double& c:cdf)c/=total;
        for(unsigned r=0;r<T;++r) {
            for(unsigned k=0;k<H;++k) {
                const auto z=mix(std::uint64_t(r)*H+k+17);
                float value=float(int(z%20001)-10000)/16384.0F;
                if(k%257==0)value*=8;
                if(r==0)value=0;
                if(r==1&&k<32)value=std::ldexp(float(int(k)-16),-120);
                const auto bits=static_cast<std::uint16_t>(std::bit_cast<unsigned>(value)>>16U);
                bf[std::size_t(r)*input.stride/2+k]=bits;x[std::size_t(r)*H+k]=std::bit_cast<float>(unsigned(bits)<<16U);
            }
            double sum=0;
            for(unsigned s=0;s<K;++s) {
                auto z=mix(std::uint64_t(r)*K+s+987);unsigned e;
                for(;;) {
                    e=bal?(r*K+s)%E:unsigned(std::lower_bound(cdf.begin(),cdf.end(),double(z>>11U)*0x1p-53)-cdf.begin());
                    bool seen=false;for(unsigned j=0;j<s;++j)seen|=rt[std::size_t(r)*K+j]==e;
                    if(!seen)break;z=mix(z);
                }
                rt[std::size_t(r)*K+s]=e;routing[std::size_t(r)*routes.stride/4+s]=e;
                rw[std::size_t(r)*K+s]=float(1+(z%1000));sum+=rw[std::size_t(r)*K+s];
            }
            for(unsigned s=0;s<K;++s){rw[std::size_t(r)*K+s]=float(rw[std::size_t(r)*K+s]/sum);weights[std::size_t(r)*rweights.stride/4+s]=rw[std::size_t(r)*K+s];}
        }
        input.upload(bf.data(),input.size);routes.upload(routing.data(),routes.size);rweights.upload(weights.data(),rweights.size);
        v.input=input.rows<const void>();v.routes=routes.rows<const unsigned>();v.route_weights=rweights.rows<const float>();
        v.codes=codes.rows<signed char>();v.deltas=deltas.rows<float>();v.sums=sums.rows<int>();v.status=status.rows<unsigned>();
        v.output=output.rows<float>();v.clear=clear.rows<float>();v.down_codes=dcodes.rows<signed char>();
        v.down_deltas=ddeltas.rows<float>();v.down_sums=dsums.rows<int>();v.rawdots=raw.rows<float>();
        const auto layout=research_layout(T);
        const auto p=[&](std::size_t at){return reinterpret_cast<unsigned*>(static_cast<unsigned char*>(csr.p)+at);};
        v.csr={p(layout.header),p(layout.offsets),p(layout.touched),p(layout.rows),p(layout.slots),p(layout.batch_experts),p(layout.batch_begins)};
    }
    ~Fixture(){if(stream)(void)hipStreamDestroy(stream);}
    unsigned batches(unsigned m)const{return std::min(T*K,E+(T*K+m-1)/m);}
};

// FP32 ordered merge, including the engine's admission and unsafe BF16 boundary.
__global__ void merge(kr::PrefillRows v,unsigned rows) {
    const unsigned r=blockIdx.y,k=blockIdx.x*256+threadIdx.x;if(r>=rows||k>=H)return;
    float y=0;bool valid=v.csr.summary[0]==0&&*at(v.status,r)==0;
    for(unsigned s=0;s<K;++s)valid=valid&&at(v.routes,r)[s]<E;
    if(valid)for(unsigned s=0;s<K;++s) {
        const float w=at(v.route_weights,r)[s],d=at(v.rawdots,r)[s*H+k];
        asm volatile("v_fmac_f32 %0, %1, %2":"+v"(y):"s"(w),"v"(d));
    }
    at(v.clear,r)[k]=y;
    if((__float_as_uint(y)&0x7fffffffU)>=0x7f7f8000U){atomicOr(at(v.status,r),8U);atomicOr(v.csr.summary,8U);}
}
__global__ void finite_scan(kr::PrefillRows v,unsigned rows,unsigned* bad) {
    for(std::size_t j=blockIdx.x*std::size_t(blockDim.x)+threadIdx.x;j<std::size_t(rows)*K*H;j+=std::size_t(gridDim.x)*blockDim.x)
        if(!isfinite(at(v.rawdots,unsigned(j/(K*H)))[j%(K*H)]))atomicOr(bad,1U);
    for(std::size_t j=blockIdx.x*std::size_t(blockDim.x)+threadIdx.x;j<std::size_t(rows)*H;j+=std::size_t(gridDim.x)*blockDim.x)
        if(!isfinite(at(v.clear,unsigned(j/H))[j%H]))atomicOr(bad,1U);
}
void status_ok(Fixture& f) {
    gpu(hipStreamSynchronize(f.stream),"chain completion");unsigned summary=0;
    gpu(hipMemcpy(&summary,f.v.csr.summary,4,hipMemcpyDeviceToHost),"summary");require(summary==0,"routed chain rejected/nonfinite");
    Buffer bad(1,4);gpu(hipMemset(bad.p,0,4),"finite flag");
    hipLaunchKernelGGL(finite_scan,dim3(256),dim3(256),0,f.stream,f.v,f.T,static_cast<unsigned*>(bad.p));
    gpu(hipStreamSynchronize(f.stream),"finite scan");require(bad.get<unsigned>(0,0)==0,"nonfinite output");
}
void csr_stage(Fixture& f,unsigned cap) {
    const unsigned blocks=(f.T*K+kr::csr_mw_span-1)/kr::csr_mw_span;auto* scratch=static_cast<unsigned*>(f.csrscratch.p);
    gpu(hipMemsetAsync(scratch,0,kr::csr_mw_head*sizeof(unsigned),f.stream),"CSR clear");
    hipLaunchKernelGGL(kr::csr_mw_count,dim3(blocks),dim3(1024),0,f.stream,f.v,f.T,E,K,scratch);
    hipLaunchKernelGGL(kr::csr_mw_scan,dim3(1),dim3(1024),0,f.stream,f.v,f.T,E,K,cap,blocks,scratch);
    hipLaunchKernelGGL(kr::csr_mw_scatter,dim3(blocks),dim3(1024),0,f.stream,f.v,f.T,K,scratch);
}

typedef int V4 __attribute__((ext_vector_type(4)));
typedef int V8 __attribute__((ext_vector_type(8)));
template<unsigned M,unsigned N> struct Stage {
    alignas(16) signed char a[M][64],b[N][64];
    float ad[M][2];int sums[M][2];float bd[N][2],bm[N][2];
    // Eight successive 16-byte row loads cover all 32 banks. With a plain
    // 64-byte pitch those same loads repeatedly hit only two bank quartets.
    __device__ signed char* ap(unsigned r,unsigned k){return a[r]+16*((k/16)^((r/2)%4))+k%16;}
    __device__ signed char* bp(unsigned r,unsigned k){return b[r]+16*((k/16)^((r/2)%4))+k%16;}
};
__device__ __forceinline__ V4 frag(const signed char* p) {return *reinterpret_cast<const V4*>(p);}
__device__ __forceinline__ V8 dot32(V4 a0,V4 b0,V4 a1,V4 b1) {
    V8 z{};z=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a0,true,b0,z,false);
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a1,true,b1,z,false);
}
// Original packed readers: GSQ's Q4_K G32 reader and the routed Q5_1/Q8_0
// block/operand readers, not a requantized or persistently expanded model.
template<unsigned Type>
__device__ __forceinline__ void load_group(const unsigned char* row,unsigned group,signed char* dest,float& d,float& m) {
    if constexpr(Type==12) {
        const auto x=strix::gsq_a8::dense_group<12>(row,group);d=x.s0;m=-x.s1;
#pragma unroll
        for(unsigned j=0;j<8;++j)reinterpret_cast<int*>(dest)[j]=x.w[j];
    } else {
        kr::wmma_v4i lo,hi;kr::down_operands<Type==7>(kr::down_block<Type==7>(row,group),group,lo,hi,d,m);
        *reinterpret_cast<V4*>(dest)=lo;*reinterpret_cast<V4*>(dest+16)=hi;
    }
}
template<unsigned M,unsigned N,bool Gate,unsigned Type,bool Capture>
__global__ __launch_bounds__(N) void candidate(const unsigned char* gate,const unsigned char* up,kr::PrefillRows v,kr::StridedRows<float> tap) {
    constexpr unsigned Width=Gate?H:I,Out=Gate?I:H,Cols=Gate?N/2:N;
    const auto& csr=v.csr;const unsigned batch=blockIdx.y;
    if((Gate?(csr.summary[0]&~8U):csr.summary[0])||batch>=csr.summary[2])return;
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1]-begin,expert=csr.experts[batch];
    __shared__ union Storage {Stage<M,N> st;float h[16][N/2];} storage;
    __shared__ unsigned rr[M],ss[M];
    const unsigned t=threadIdx.x,lane=t%32,c=lane%16,parity=lane/16,wave=t/32;
    if(t<M){rr[t]=t<count?csr.rows[begin+t]:0;ss[t]=t<count?csr.slots[begin+t]:0;}
    __syncthreads();
    float acc[M/16][2][8]={};
    constexpr unsigned AChunks=(2*M+N-1)/N;
    struct Regs {V4 a[AChunks][2],b[2][2];float ad[AChunks],bd[2],bm[2];int sums[AChunks];};
    Regs reg;
    const auto fetch=[&](unsigned step) __attribute__((always_inline)) {
#pragma unroll
        for(unsigned part=0;part<AChunks;++part) {
            const unsigned j=t+part*N;if(j>=M*2)continue;
            const unsigned r=j/2,g=j%2;const bool live=r<count;
            const auto* code=Gate?at(v.codes,rr[r]):at(v.down_codes,rr[r])+ss[r]*Width;
            const auto* delta=Gate?at(v.deltas,rr[r]):at(v.down_deltas,rr[r])+ss[r]*(Width/32);
            const auto* sum=Gate?at(v.sums,rr[r]):at(v.down_sums,rr[r])+ss[r]*(Width/32);
            reg.a[part][0]=live?frag(code+64*step+32*g):V4{};reg.a[part][1]=live?frag(code+64*step+32*g+16):V4{};
            reg.ad[part]=live?delta[2*step+g]:0;reg.sums[part]=live?sum[2*step+g]:0;
        }
#pragma unroll
        for(unsigned part=0;part<2;++part) {
            const unsigned j=t+part*N;
            const unsigned col=j/2,g=j%2,out=blockIdx.x*Cols+(col%Cols);
            if(out<Out) {
                const auto* weights=Gate&&col>=Cols?up:gate;
                const auto* row=weights+(std::size_t(expert)*Out+out)*(Width/(Type==12?256:32))*(Type==12?144:Type==7?24:34);
                load_group<Type>(row,2*step+g,reinterpret_cast<signed char*>(reg.b[part]),reg.bd[part],reg.bm[part]);
            } else {
                reg.b[part][0]=reg.b[part][1]=V4{};reg.bd[part]=reg.bm[part]=0;
            }
        }
    };
    fetch(0);
    for(unsigned step=0;step<Width/64;++step) {
        auto& st=storage.st;
#pragma unroll
        for(unsigned part=0;part<AChunks;++part) {
            const unsigned j=t+part*N;if(j>=M*2)continue;const unsigned r=j/2,g=j%2;
            *reinterpret_cast<V4*>(st.ap(r,32*g))=reg.a[part][0];*reinterpret_cast<V4*>(st.ap(r,32*g+16))=reg.a[part][1];
            st.ad[r][g]=reg.ad[part];st.sums[r][g]=reg.sums[part];
        }
#pragma unroll
        for(unsigned part=0;part<2;++part) {
            const unsigned j=t+part*N,col=j/2,g=j%2;
            *reinterpret_cast<V4*>(st.bp(col,32*g))=reg.b[part][0];*reinterpret_cast<V4*>(st.bp(col,32*g+16))=reg.b[part][1];
            st.bd[col][g]=reg.bd[part];st.bm[col][g]=reg.bm[part];
        }
        __syncthreads();
        // One LDS stage, with next K64 operands prefetched into registers while
        // this stage's independently scaled K32 products consume LDS.
        if(step+1<Width/64)fetch(step+1);
#pragma unroll
        for(unsigned g=0;g<2;++g)
#pragma unroll
        for(unsigned branch=0;branch<2;++branch) {
            const unsigned col=wave*16+c+branch*(N/2);
            const V4 b0=frag(st.bp(col,32*g)),b1=frag(st.bp(col,32*g+16));
            const float d=st.bd[col][g],m=st.bm[col][g];
#pragma unroll
            for(unsigned mi=0;mi<M/16;++mi) {
                if(16*mi>=count)continue;
                const V8 z=dot32(frag(st.ap(mi*16+c,32*g)),b0,frag(st.ap(mi*16+c,32*g+16)),b1);
#pragma unroll
                for(unsigned e=0;e<8;++e) {
                    const unsigned r=mi*16+2*e+parity;
                    acc[mi][branch][e]=__builtin_fmaf(st.ad[r][g],__builtin_fmaf(d,float(z[e]),m*float(st.sums[r][g])),acc[mi][branch][e]);
                }
            }
        }
        __syncthreads();
    }
    if constexpr(!Gate) {
#pragma unroll
        for(unsigned branch=0;branch<2;++branch)
#pragma unroll
        for(unsigned mi=0;mi<M/16;++mi) {
            const unsigned out=blockIdx.x*N+wave*16+c+branch*(N/2);
#pragma unroll
            for(unsigned e=0;e<8;++e) {
                const unsigned a=16*mi+2*e+parity;
                if(a<count&&out<H) {
                    const float y=acc[mi][branch][e];at(v.rawdots,rr[a])[ss[a]*H+out]=y;
                    if(!isfinite(y)){atomicOr(at(v.status,rr[a]),8U);atomicOr(csr.summary,8U);}
                }
            }
        }
    } else {
        // Reuse the stage after K, sixteen assignments at a time. Complete G32
        // groups belong to this CTA, so no last-CTA conversion/counter is needed.
#pragma unroll
        for(unsigned mi=0;mi<M/16;++mi) {
            if(16*mi>=count)continue; // CTA-uniform; do not quantize padded row tiles
#pragma unroll
            for(unsigned e=0;e<8;++e) {
                const unsigned a=16*mi+2*e+parity,col=wave*16+c,out=blockIdx.x*Cols+col;
                const float g=acc[mi][0][e],u=acc[mi][1][e],h=(g/(1.0F+expf(-g)))*u;
                storage.h[2*e+parity][col]=h;
                if(a<count&&out<I) {
                    if constexpr(Capture) {float* p=at(tap,rr[a])+ss[a]*3*I;p[out]=g;p[I+out]=u;p[2*I+out]=h;}
                    if(!isfinite(g)||!isfinite(u)||!isfinite(h)){atomicOr(at(v.status,rr[a]),8U);atomicOr(csr.summary,8U);}
                }
            }
            __syncthreads();
            for(unsigned j=t;j<16*Cols;j+=N) {
                const unsigned r=j/Cols,col=j%Cols,a=mi*16+r,out=blockIdx.x*Cols+col;
                const float hv=storage.h[r][col];
                // Existing routed down-input quantizer: tiny threshold, RNE
                // division, clipping and exact sums from project_prefill_wmma.
                float x=fabsf(hv);for(int stride=16;stride;stride/=2)x=fmaxf(x,__shfl_xor(x,stride,32));
                const bool tiny=x>0.0F&&x<0x1p-119F;
                const float delta=x==0.0F||tiny?1.0F:x/127.0F;
                const float rounded=tiny?0.0F:rintf(hv/delta);
                const int code=isfinite(rounded)?int(fminf(127.0F,fmaxf(-127.0F,rounded))):0;
                int sum=code;for(int stride=16;stride;stride/=2)sum+=__shfl_xor(sum,stride,32);
                if(a<count&&out<I) {
                    at(v.down_codes,rr[a])[ss[a]*I+out]=static_cast<signed char>(code);
                    if(lane==0){at(v.down_deltas,rr[a])[ss[a]*(I/32)+out/32]=delta;at(v.down_sums,rr[a])[ss[a]*(I/32)+out/32]=sum;}
                }
            }
            __syncthreads();
        }
    }
}

// Untimed capture of the production gate/up body; measured launches below use
// project_prefill_f16x_da16_mg directly, with no capture stores or test branch.
template<unsigned Tune>
__global__ __launch_bounds__((kr::F16TileX<Tune,0>::Threads)) void capture_f16(const unsigned char* gate,const unsigned char* up,
    kr::PrefillRows view,unsigned hidden,unsigned outputs,const uint4* a16,kr::StridedRows<float> tap) {
    using namespace kr;
    constexpr bool Q5K=false;constexpr unsigned Core=0;
#define STRIX_F16X_BATCH_TILE const unsigned batch=blockIdx.y,tile=blockIdx.x;
#define STRIX_F16X_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
#define STRIX_F16X_HDR_REUSE true
#define STRIX_F16X_TEST_CAPTURE if(live) { float* p=at(tap,r)+slot*3U*outputs;p[out]=g;p[outputs+out]=u;p[2U*outputs+out]=hv; }
#include "../src/hip_moe_f16x_project_body.inc"
#undef STRIX_F16X_TEST_CAPTURE
#undef STRIX_F16X_HDR_REUSE
#undef STRIX_F16X_LDS
#undef STRIX_F16X_BATCH_TILE
}
struct Times {std::array<double,5> stage{};double chain=0;};
// Events at stage boundaries, no synchronization or host CSR readback in chain.
template<class F> Times timed(Fixture& f,F&& launch) {
    std::array<Event,6> e;e[0].record(f.stream);launch(e);gpu(hipGetLastError(),"chain launch");
    gpu(hipEventSynchronize(e[5].e),"chain timing");Times t;
    unsigned status=0;gpu(hipMemcpy(&status,f.v.csr.summary,sizeof(status),hipMemcpyDeviceToHost),"timed status");
    require(status==0,"timed chain rejected/nonfinite");
    for(unsigned i=0;i<5;++i){float ms=0;gpu(hipEventElapsedTime(&ms,e[i].e,e[i+1].e),"stage time");t.stage[i]=ms;}
    float ms=0;gpu(hipEventElapsedTime(&ms,e[0].e,e[5].e),"chain time");t.chain=ms;return t;
}
unsigned incumbent_tune(Fixture& f){return kr::prefill_effective_tune(f.request,f.T,f.layer.down.type,53);}
template<unsigned Tune> Times incumbent(Fixture& f,bool capture=false) {
    return timed(f,[&](auto& e){
        hipLaunchKernelGGL(kr::f16_prefill_prologue_v2,dim3(f.T),dim3(256),0,f.stream,f.v,H,K,E,0,static_cast<uint4*>(f.a16.p),1U);e[1].record(f.stream);
        constexpr unsigned Cap=(Tune&2)?128:64;csr_stage(f,Cap);e[2].record(f.stream);
        using Tile=kr::F16TileX<Tune,0>;const unsigned batches=f.batches(Cap);
        auto* arrive=static_cast<unsigned*>(f.arrive.p);auto* image=static_cast<uint4*>(f.image.p);
        gpu(hipMemsetAsync(arrive,0,batches*sizeof(unsigned),f.stream),"DA16 counters");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kr::project_prefill_f16x_da16_mg<false,Tune>),dim3(I/(Tile::Columns/2),batches),dim3(Tile::Threads),0,f.stream,
            f.layer.gate.ptr(),f.layer.up.ptr(),f.v,H,I,static_cast<const uint4*>(f.a16.p),arrive,image);e[3].record(f.stream);
        if(f.layer.down.type==7) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kr::down_prefill_f16x_da16<true,Tune>),dim3(H/kr::F16Tile<Tune>::Columns,batches),dim3(Tile::Threads),0,f.stream,f.layer.down.ptr(),f.v,H,I,image);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kr::down_prefill_f16x_da16<false,Tune>),dim3(H/kr::F16Tile<Tune>::Columns,batches),dim3(Tile::Threads),0,f.stream,f.layer.down.ptr(),f.v,H,I,image);
        }
        e[4].record(f.stream);
        hipLaunchKernelGGL(merge,dim3(H/256,f.T),dim3(256),0,f.stream,f.v,f.T);e[5].record(f.stream);
        if(capture) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(capture_f16<Tune>),dim3(I/(Tile::Columns/2),batches),dim3(Tile::Threads),0,f.stream,
                f.layer.gate.ptr(),f.layer.up.ptr(),f.v,H,I,static_cast<const uint4*>(f.a16.p),f.tap.rows<float>());
            gpu(hipStreamSynchronize(f.stream),"incumbent capture");
        }
    });
}
Times run_inc(Fixture& f,bool capture=false) {
    return incumbent_tune(f)==82?incumbent<82>(f,capture):incumbent<80>(f,capture);
}
template<unsigned M,unsigned N,bool Capture=false> Times run_candidate(Fixture& f) {
    return timed(f,[&](auto& e){
        for(unsigned first=0;first<f.T;first+=strix::q51_known_rows::prefill_max_rows)
            kr::enqueue_prefill_prologue(f.request,shift(f.v,first),std::min(f.T-first,strix::q51_known_rows::prefill_max_rows),0,f.stream);
        e[1].record(f.stream);
        csr_stage(f,M);e[2].record(f.stream);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate<M,N,true,12,Capture>),dim3((I+N/2-1)/(N/2),f.batches(M)),dim3(N),0,f.stream,
            f.layer.gate.ptr(),f.layer.up.ptr(),f.v,f.tap.rows<float>());e[3].record(f.stream);
        if(f.layer.down.type==7) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate<M,N,false,7,false>),dim3((H+N-1)/N,f.batches(M)),dim3(N),0,f.stream,f.layer.down.ptr(),nullptr,f.v,f.tap.rows<float>());
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(candidate<M,N,false,8,false>),dim3((H+N-1)/N,f.batches(M)),dim3(N),0,f.stream,f.layer.down.ptr(),nullptr,f.v,f.tap.rows<float>());
        }
        e[4].record(f.stream);hipLaunchKernelGGL(merge,dim3(H/256,f.T),dim3(256),0,f.stream,f.v,f.T);e[5].record(f.stream);
    });
}
// Exact WMMA layout + K32 dots, signedness, scale/min unpacking and sums.
template<unsigned Type>
__global__ void integer_fixture(const unsigned char* packed,unsigned width,int* dots,int* codes,float* scales,int* sums) {
    const unsigned group=blockIdx.x,lane=threadIdx.x,c=lane%16,half_lane=lane/16;
    alignas(16) signed char decoded[32];float d,m;
    constexpr unsigned BlockElements=Type==12?256:32,BlockBytes=Type==12?144:Type==7?24:34;
    load_group<Type>(packed+std::size_t(c)*(width/BlockElements)*BlockBytes,group,decoded,d,m);
    V4 a0{},a1{};
    for(unsigned k=0;k<32;++k) {
        const int value=int((c*31+k*17+group*7)%255)-127;
        if(k<16)reinterpret_cast<signed char*>(&a0)[k]=static_cast<signed char>(value);
        else reinterpret_cast<signed char*>(&a1)[k-16]=static_cast<signed char>(value);
        if(lane<16)codes[(std::size_t(group)*16+c)*32+k]=decoded[k];
    }
    if(lane<16) {scales[(std::size_t(group)*16+c)*2]=d;scales[(std::size_t(group)*16+c)*2+1]=m;}
    const V8 z=dot32(a0,frag(decoded),a1,frag(decoded+16));
    for(unsigned e=0;e<8;++e) {
        const unsigned row=2*e+half_lane;dots[(std::size_t(group)*16+row)*16+c]=z[e];
        if(c==0){int sum=0;for(unsigned k=0;k<32;++k)sum+=int((row*31+k*17+group*7)%255)-127;sums[std::size_t(group)*16+row]=sum;}
    }
}
void exact_fixtures(Budget& budget) {
    for(unsigned type:{12U,7U,8U}) {
        budget.check();const unsigned blocks=16*256,width=256*elements(type),groups=width/32;
        const auto weights=packed(type,blocks);Buffer w(1,weights.size()),dots(1,std::size_t(groups)*256*4),codes(1,std::size_t(groups)*16*32*4),
            scales(1,std::size_t(groups)*16*2*4),sums(1,std::size_t(groups)*16*4);w.upload(weights.data(),weights.size());
        const auto launch=[&]<unsigned Type>(){hipLaunchKernelGGL(integer_fixture<Type>,dim3(groups),dim3(32),0,nullptr,
            static_cast<const unsigned char*>(w.p),width,static_cast<int*>(dots.p),static_cast<int*>(codes.p),static_cast<float*>(scales.p),static_cast<int*>(sums.p));};
        if(type==12)launch.template operator()<12>();else if(type==7)launch.template operator()<7>();else launch.template operator()<8>();
        gpu(hipDeviceSynchronize(),"exact integer fixture");
        std::vector<int> dd(std::size_t(groups)*256),qq(std::size_t(groups)*16*32),ss(std::size_t(groups)*16);
        std::vector<float> sm(std::size_t(groups)*16*2);
        gpu(hipMemcpy(dd.data(),dots.p,dd.size()*4,hipMemcpyDeviceToHost),"dots");gpu(hipMemcpy(qq.data(),codes.p,qq.size()*4,hipMemcpyDeviceToHost),"decoded codes");
        gpu(hipMemcpy(sm.data(),scales.p,sm.size()*4,hipMemcpyDeviceToHost),"decoded scales");gpu(hipMemcpy(ss.data(),sums.p,ss.size()*4,hipMemcpyDeviceToHost),"sums");
        for(unsigned g=0;g<groups;++g)for(unsigned c=0;c<16;++c) {
            const auto* row=weights.data()+std::size_t(c)*row_bytes(type,width);const auto coef=coefficient(row,32*g,type);
            require(sm[(std::size_t(g)*16+c)*2]==float(coef.d)&&sm[(std::size_t(g)*16+c)*2+1]==float(coef.m),"device scale/min decode mismatch");
            for(unsigned k=0;k<32;++k)require(qq[(std::size_t(g)*16+c)*32+k]==coefficient(row,32*g+k,type).q,"device packed code mismatch");
            for(unsigned r=0;r<16;++r) {
                int dot=0,sum=0;for(unsigned k=0;k<32;++k){const int a=int((r*31+k*17+g*7)%255)-127;sum+=a;dot+=a*coefficient(row,32*g+k,type).q;}
                require(dot==dd[(std::size_t(g)*16+r)*16+c]&&sum==ss[std::size_t(g)*16+r],"integer K32 dot or sum mismatch");
            }
        }
        std::printf("{\"integer_fixture\":\"PASS\",\"type\":%u,\"K32_dots\":%u}\n",type,groups*256);
    }
}
struct Sample {
    unsigned row;std::vector<float> tap,raw,merged,down_delta;std::vector<signed char> down_codes;std::vector<int> down_sums;
    std::vector<signed char> input_codes;std::vector<float> input_delta;std::vector<int> input_sums;
};
std::vector<Sample> snapshot(Fixture& f) {
    gpu(hipStreamSynchronize(f.stream),"accuracy sample");std::vector<Sample> out;
    for(unsigned r:{0U,f.T/2,f.T-1}) {
        Sample s{r,std::vector<float>(K*3*I),std::vector<float>(K*H),std::vector<float>(H),std::vector<float>(K*(I/32)),
            std::vector<signed char>(K*I),std::vector<int>(K*(I/32)),std::vector<signed char>(H),std::vector<float>(H/32),std::vector<int>(H/32)};
        const auto get=[&](auto& host,const Buffer& b) {gpu(hipMemcpy(host.data(),static_cast<const unsigned char*>(b.p)+r*b.stride,
            host.size()*sizeof(host[0]),hipMemcpyDeviceToHost),"sample row");};
        get(s.tap,f.tap);get(s.raw,f.raw);get(s.merged,f.clear);get(s.down_delta,f.ddeltas);get(s.down_codes,f.dcodes);get(s.down_sums,f.dsums);
        get(s.input_codes,f.codes);get(s.input_delta,f.deltas);get(s.input_sums,f.sums);out.push_back(std::move(s));
    }
    return out;
}
struct Oracle {
    unsigned row;std::vector<double> gate,up,h,qgate,qup,qh,raw,qraw,merged,qmerged;
};
std::vector<Oracle> oracles(Fixture& f,Budget& budget) {
    std::vector<Oracle> all;
    for(unsigned row:{0U,f.T/2,f.T-1}) {
        budget.check(5);Oracle o;o.row=row;
        for(auto* p:{&o.gate,&o.up,&o.h,&o.qgate,&o.qup,&o.qh})p->resize(K*I);
        for(auto* p:{&o.raw,&o.qraw})p->resize(K*H);
        o.merged.resize(H);o.qmerged.resize(H);
        std::vector<float> x(f.x.data()+std::size_t(row)*H,f.x.data()+std::size_t(row+1)*H);const auto qx=image(x,false);
        for(unsigned s=0;s<K;++s) {
            const unsigned expert=f.rt[std::size_t(row)*K+s];
            for(unsigned i=0;i<I;++i) {
                const auto* g=f.layer.gate.row(expert,i,H,I);const auto* u=f.layer.up.row(expert,i,H,I);const unsigned j=s*I+i;
                o.gate[j]=dot(x,g,12);o.up[j]=dot(x,u,12);o.h[j]=silu(o.gate[j],o.up[j]);
                o.qgate[j]=qdot(qx,g,12);o.qup[j]=qdot(qx,u,12);o.qh[j]=silu(o.qgate[j],o.qup[j]);
            }
            std::vector<float> qh(I);for(unsigned i=0;i<I;++i)qh[i]=float(o.qh[s*I+i]);const auto qhi=image(qh,true);
            // Eight output channels per sample row. Complete I and all ten slots
            // are evaluated, including the route-weighted mathematical merge.
            for(unsigned sample=0;sample<8;++sample) {
                const unsigned k=sample==0?0:sample==7?H-1:unsigned(mix(sample*719+row)%H);
                const auto* w=f.layer.down.row(expert,k,I,H);double y=0;
                for(unsigned i=0;i<I;++i)y+=o.h[s*I+i]*weight(w,i,f.layer.down.type);
                o.raw[s*H+k]=y;o.qraw[s*H+k]=qdot(qhi,w,f.layer.down.type);
                o.merged[k]+=double(f.rw[std::size_t(row)*K+s])*y;o.qmerged[k]+=double(f.rw[std::size_t(row)*K+s])*o.qraw[s*H+k];
            }
        }
        all.push_back(std::move(o));
    }
    return all;
}
void accuracy(Fixture& f,const std::vector<Oracle>& oracle,const std::vector<Sample>& snap,bool cand,unsigned m,unsigned n) {
    std::array<std::array<Metrics,5>,3> metric;
    for(unsigned r=0;r<oracle.size();++r) {
        const auto& o=oracle[r];const auto& s=snap[r];
        if(cand)for(unsigned g=0;g<H/32;++g) {
            const auto q=quantize(f.x.data()+std::size_t(o.row)*H+32*g,false);
            require(q.status==0&&q.delta==s.input_delta[g]&&q.sum==s.input_sums[g],"input quantizer scale/sum mismatch");
            for(unsigned k=0;k<32;++k)require(q.q[k]==s.input_codes[32*g+k],"input quantizer code mismatch");
        }
        for(unsigned slot=0;slot<K;++slot) {
            std::vector<Group> actual(I/32);
            if(cand)for(unsigned g=0;g<I/32;++g) {
                const auto q=quantize(s.tap.data()+slot*3*I+2*I+32*g,true);
                auto& a=actual[g];a.delta=s.down_delta[slot*(I/32)+g];a.sum=s.down_sums[slot*(I/32)+g];
                require(q.delta==a.delta&&q.sum==a.sum,"down scale/sum mismatch");
                for(unsigned k=0;k<32;++k){a.q[k]=s.down_codes[slot*I+g*32+k];require(a.q[k]==q.q[k],"down quantizer code mismatch");}
            }
            for(unsigned k=0;k<I;++k) {
                const unsigned j=slot*I+k;const double observed[3]{s.tap[slot*3*I+k],s.tap[slot*3*I+I+k],s.tap[slot*3*I+2*I+k]};
                const double unquant[3]{o.gate[j],o.up[j],o.h[j]},quant[3]{o.qgate[j],o.qup[j],o.qh[j]};
                for(unsigned stage=0;stage<3;++stage) {
                    metric[0][stage].add(observed[stage],unquant[stage]);metric[1][stage].add(observed[stage],quant[stage]);
                }
            }
            const unsigned expert=f.rt[std::size_t(o.row)*K+slot];
            for(unsigned sample=0;sample<8;++sample) {
                const unsigned k=sample==0?0:sample==7?H-1:unsigned(mix(sample*719+o.row)%H);
                const double y=s.raw[slot*H+k];metric[0][3].add(y,o.raw[slot*H+k]);metric[1][3].add(y,o.qraw[slot*H+k]);
                if(cand)metric[2][3].add(y,qdot(actual,f.layer.down.row(expert,k,I,H),f.layer.down.type));
            }
        }
        for(unsigned sample=0;sample<8;++sample) {
            const unsigned k=sample==0?0:sample==7?H-1:unsigned(mix(sample*719+o.row)%H);
            metric[0][4].add(s.merged[k],o.merged[k]);metric[1][4].add(s.merged[k],o.qmerged[k]);
            double local=0;for(unsigned slot=0;slot<K;++slot)local+=double(f.rw[std::size_t(o.row)*K+slot])*s.raw[slot*H+k];
            metric[2][4].add(s.merged[k],local);
        }
    }
    const char* stages[]{"gate","up","swiglu","raw_down","merged"};const char* refs[]{"unquantized-fp64","quantized-chain-fp64","quantized-local-fp64"};
    for(unsigned ref=0;ref<3;++ref)for(unsigned stage=0;stage<5;++stage)if(!metric[ref][stage].errors.empty())
        metric[ref][stage].print(cand?"candidate":"incumbent",refs[ref],stages[stage],f.T,m,n);
    if(cand)for(unsigned stage:{0U,1U})require(std::sqrt(metric[1][stage].e2/std::max(1e-24,metric[1][stage].r2))<2e-5,"candidate gate/up arithmetic mismatch");
    if(cand)require(std::sqrt(metric[2][3].e2/std::max(1e-24,metric[2][3].r2))<2e-5,"candidate down arithmetic mismatch");
}

double median(std::vector<double> x) {require(!x.empty(),"no timing samples");std::sort(x.begin(),x.end());const auto i=x.size()/2;return x.size()%2?x[i]:(x[i-1]+x[i])/2;}
double spread(const std::vector<double>& x) {return *std::max_element(x.begin(),x.end())-*std::min_element(x.begin(),x.end());}
std::vector<double> chains(const std::vector<Times>& x){std::vector<double> r;for(const auto& t:x)r.push_back(t.chain);return r;}
void stage_summary(const char* arm,const std::vector<Times>& times,unsigned t,unsigned m,unsigned n) {
    const char* names[]{"quantize_or_f16_prologue","csr","gate_up_and_down_image","down","merge"};
    for(unsigned stage=0;stage<5;++stage) {
        std::vector<double> x;for(const auto& time:times)x.push_back(time.stage[stage]);
        std::printf("{\"timing\":\"%s\",\"stage\":\"%s\",\"T\":%u,\"M\":%u,\"N\":%u,\"median_ms\":%.6f,\"range_ms\":%.6f}\n",arm,names[stage],t,m,n,median(x),spread(x));
    }
}
template<unsigned M,unsigned N> void resources() {
    const auto show=[&](const char* name,auto kernel) {
        hipFuncAttributes a{};gpu(hipFuncGetAttributes(&a,reinterpret_cast<const void*>(kernel)),"kernel resources");
        std::printf("{\"resources\":\"%s\",\"M\":%u,\"N\":%u,\"registers\":%d,\"lds_bytes\":%zu,\"private_bytes\":%zu,\"threads\":%u}\n",name,M,N,a.numRegs,a.sharedSizeBytes,a.localSizeBytes,N);
    };
    show("gate_up",candidate<M,N,true,12,false>);show("down_q51",candidate<M,N,false,7,false>);show("down_q80",candidate<M,N,false,8,false>);
}
// Byte equality with the complete production entry point is checked before any
// direct-kernel timings. The timed recipe therefore cannot silently drift.
void production_witness(Fixture& f) {
    kr::set_prefill_core2(0);kr::set_prefill_route(14);kr::set_prefill_da16(1);kr::set_prefill_hdr(1);kr::set_prefill_small(0);
    for(unsigned first=0;first<f.T;first+=strix::q51_known_rows::prefill_max_rows) {
        const auto v=shift(f.v,first);const unsigned rows=std::min(f.T-first,strix::q51_known_rows::prefill_max_rows);
        kr::enqueue_prefill(f.request,v,rows,H,f.layer.gate.ptr(),f.layer.up.ptr(),f.layer.down.type,f.layer.down.ptr(),f.stream,53);
        require(kr::prefill_da16_ran()&&!kr::prefill_fused_merge()&&kr::prefill_clear_skipped(),"production DA16 route missing");
        hipLaunchKernelGGL(merge,dim3(H/256,rows),dim3(256),0,f.stream,v,rows);
    }
    status_ok(f);
    // Full raw and merged bytes, not just the accuracy sample. Retain only these
    // two host images briefly; no receipt or capture is written to disk.
    std::vector<unsigned char> raw(f.raw.size),merged(f.clear.size);
    gpu(hipMemcpy(raw.data(),f.raw.p,raw.size(),hipMemcpyDeviceToHost),"production raw witness");
    gpu(hipMemcpy(merged.data(),f.clear.p,merged.size(),hipMemcpyDeviceToHost),"production merge witness");
    run_inc(f);status_ok(f);
    std::vector<unsigned char> chunk(4*1024*1024);
    const auto equal=[&](const Buffer& b,const std::vector<unsigned char>& host) {
        for(std::size_t at=0;at<b.size;at+=chunk.size()) {
            const auto size=std::min(chunk.size(),b.size-at);gpu(hipMemcpy(chunk.data(),static_cast<const unsigned char*>(b.p)+at,size,hipMemcpyDeviceToHost),"witness compare");
            require(std::memcmp(chunk.data(),host.data()+at,size)==0,"direct incumbent differs from production");
        }
    };
    equal(f.raw,raw);equal(f.clear,merged);
    std::printf("{\"production_witness\":\"byte-equal\",\"T\":%u,\"api_chunk_rows\":%u,\"tune\":%u,\"da16\":true,\"route_bf16\":false}\n",f.T,std::min(f.T,strix::q51_known_rows::prefill_max_rows),incumbent_tune(f));
}
void rejection_fixtures(Fixture& f) {
    const auto original=static_cast<std::uint16_t>(std::bit_cast<unsigned>(f.x[0])>>16U);
    for(std::uint16_t bad:{std::uint16_t(0x7fc0),std::uint16_t(0x7f80),std::uint16_t(0xff80)}) {
        gpu(hipMemcpy(f.input.p,&bad,2,hipMemcpyHostToDevice),"nonfinite injection");
        kr::enqueue_prefill_prologue(f.request,f.v,f.T,H,f.stream);gpu(hipStreamSynchronize(f.stream),"rejection fixture");
        require((f.status.get<unsigned>(0,0)&a8::nonfinite_activation)!=0,"nonfinite input admitted");
    }
    gpu(hipMemcpy(f.input.p,&original,2,hipMemcpyHostToDevice),"restore input");
    const unsigned invalid=E;gpu(hipMemcpy(f.routes.p,&invalid,4,hipMemcpyHostToDevice),"bad route injection");
    kr::enqueue_prefill_prologue(f.request,f.v,f.T,H,f.stream);csr_stage(f,64);gpu(hipStreamSynchronize(f.stream),"route rejection fixture");
    require(f.csr.get<unsigned>(0,0)!=0,"invalid CSR route admitted");
    gpu(hipMemcpy(f.routes.p,f.rt.data(),4,hipMemcpyHostToDevice),"restore route");
    std::puts("{\"rejection_fixtures\":\"PASS\"}");
}
void integer_control(Fixture& f,Budget& budget) {
    budget.check(5);require(kr::prefill_kernel_form(f.request,f.layer.down.type,1)==kr::PrefillKernelForm::iu8_wmma,"old integer control did not select WMMA");
    std::vector<double> samples;
    for(unsigned round=0;round<4;++round) {
        budget.check(2);Event begin,end;begin.record(f.stream);
        // At T16384 the diagnostic uses two admitted T8192 calls. It is labelled
        // explicitly and contributes no speed credit to a larger-tile candidate.
        for(unsigned first=0;first<f.T;first+=strix::q51_known_rows::prefill_max_rows) {
            const auto v=shift(f.v,first);const unsigned rows=std::min(f.T-first,strix::q51_known_rows::prefill_max_rows);
            kr::enqueue_prefill(f.request,v,rows,H,f.layer.gate.ptr(),f.layer.up.ptr(),f.layer.down.type,f.layer.down.ptr(),f.stream,1);
            hipLaunchKernelGGL(merge,dim3(H/256,rows),dim3(256),0,f.stream,v,rows);
        }
        end.record(f.stream);gpu(hipEventSynchronize(end.e),"old integer control");float ms=0;gpu(hipEventElapsedTime(&ms,begin.e,end.e),"control time");
        if(round)samples.push_back(ms);status_ok(f);
    }
    std::printf("{\"diagnostic\":\"existing-M16-iu8-wmma\",\"T\":%u,\"api_chunk_rows\":%u,\"median_ms\":%.6f,\"range_ms\":%.6f}\n",f.T,std::min(f.T,strix::q51_known_rows::prefill_max_rows),median(samples),spread(samples));
}
template<unsigned M,unsigned N>
void variant(Fixture& f,const std::vector<Oracle>& oracle,Budget& budget,unsigned rounds) {
    budget.check(8);const auto cold=run_candidate<M,N>(f);run_candidate<M,N,true>(f);status_ok(f);
    accuracy(f,oracle,snapshot(f),true,M,N);
    run_candidate<M,N>(f);run_inc(f); // warm both operand images and all scratch readers
    std::vector<Times> a,b;std::vector<double> noises,aa0,aa1;
    // A/A control is counterbalanced too: A0,A1,A1,A0 followed by A,B,B,A
    // (reversed on odd rounds). Noise includes the full A/A range, conservatively.
    for(unsigned round=0;round<rounds;++round) {
        const double estimate=std::max(cold.chain,20.0)*10/1000+2;
        if(budget.left()<estimate+4)break;
        std::array<double,4> aa{};for(double& x:aa)x=run_inc(f).chain;
        aa0.push_back((aa[0]+aa[3])/2);aa1.push_back((aa[1]+aa[2])/2);
        const double base=median(std::vector<double>(aa.begin(),aa.end()));
        noises.push_back(100*(*std::max_element(aa.begin(),aa.end())-*std::min_element(aa.begin(),aa.end()))/base);
        for(unsigned j=0;j<4;++j) {
            const bool candidate_arm=((j==1||j==2)^(round%2!=0));
            if(candidate_arm)b.push_back(run_candidate<M,N>(f));else a.push_back(run_inc(f));
        }
    }
    const bool complete=aa0.size()>=3; // insufficient counterbalanced rounds never continue
    if(a.empty()||b.empty())throw std::runtime_error("budget exhausted before A/B (inconclusive)");
    status_ok(f);const auto ca=chains(a),cb=chains(b);const double am=median(ca),bm=median(cb),gain=100*(am-bm)/am;
    const double noise=std::max(*std::max_element(noises.begin(),noises.end()),100*std::abs(median(aa0)-median(aa1))/median(aa0));
    const bool kill=!complete||gain<5||gain<=noise;
    stage_summary("incumbent",a,f.T,M,N);stage_summary("candidate",b,f.T,M,N);
    std::printf("{\"summary\":\"int8-wmma-routed-research-v1\",\"T\":%u,\"routing\":\"%s\",\"M\":%u,\"N\":%u,\"down_type\":%u,\"rounds\":%zu,\"chain_ms_incumbent\":%.6f,\"chain_ms_candidate\":%.6f,\"gain_pct\":%.6f,\"aa_noise_pct\":%.6f,\"kill\":%s,\"complete\":%s,\"chain_range_ms_incumbent\":%.6f,\"chain_range_ms_candidate\":%.6f,\"cold_chain_ms_candidate\":%.6f,\"quality_qualified\":false}\n",
        f.T,f.balanced?"balanced-tail":"zipf-1.5-unique",M,N,f.layer.down.type,aa0.size(),am,bm,gain,noise,kill?"true":"false",complete?"true":"false",spread(ca),spread(cb),cold.chain);
    std::fflush(stdout);
}
void identity(unsigned layer) {
#if __has_include("strix/build_info.hpp")
    std::printf("%.*s\n",int(strix::build_info().size()),strix::build_info().data());
#endif
    const auto binary=strix::model_content_identity_hex(strix::sha256_file_content("/proc/self/exe",strix::FileHashPolicy::full));
    std::printf("{\"binary_sha256\":\"%s\"}\n",binary.c_str());
    hipDeviceProp_t p{};gpu(hipGetDeviceProperties(&p,0),"device identity");require(std::string(p.gcnArchName).starts_with("gfx1151"),"requires gfx1151");
    int runtime=0,driver=0;gpu(hipRuntimeGetVersion(&runtime),"runtime version");gpu(hipDriverGetVersion(&driver),"driver version");
    struct utsname os{};require(uname(&os)==0,"running kernel identity");
    std::ifstream perf("/sys/module/amdgpu/parameters/iommu_perfopt");std::string state="unavailable";if(perf)perf>>state;
    std::printf("{\"identity\":\"int8-wmma-routed-research-v1\",\"arch\":\"%s\",\"hip_runtime\":%d,\"hip_driver\":%d,\"compiler\":\"%s\",\"kernel\":\"%s\",\"iommu_perfopt\":\"%s\",\"weights\":\"original-packed-gguf-layer\",\"layer\":%u,\"hidden\":2560,\"intermediate\":640,\"experts\":512,\"top_k\":10,\"activation\":\"deterministic-bf16\",\"accumulation\":\"relaxed-fp32-k32-fma\",\"qualification\":\"pending-until-qualified\"}\n",
        p.gcnArchName,runtime,driver,__clang_version__,os.release,state.c_str(),layer);
}
std::string read_value(const std::filesystem::path& path,bool active=false) {
    std::ifstream f(path);std::string line;
    while(std::getline(f,line))if(!active||line.find('*')!=std::string::npos)return line;
    return "unavailable";
}
std::string escaped(const std::string& s) {
    std::string result;for(char c:s){if(c=='\\'||c=='"')result+='\\';if(static_cast<unsigned char>(c)>=32)result+=c;}return result;
}
void telemetry(const char* phase) {
    char pci[32]{};gpu(hipDeviceGetPCIBusId(pci,sizeof(pci),0),"telemetry device");
    const std::filesystem::path device=std::filesystem::path("/sys/bus/pci/devices")/pci;
    std::string power="unavailable",temp="unavailable";std::error_code ec;
    for(const auto& entry:std::filesystem::directory_iterator(device/"hwmon",ec)) {
        power=read_value(entry.path()/"power1_average");temp=read_value(entry.path()/"temp1_input");break;
    }
    std::printf("{\"telemetry\":\"%s\",\"sclk\":\"%s\",\"mclk\":\"%s\",\"power_microwatts\":\"%s\",\"temperature_millicelsius\":\"%s\"}\n",phase,
        escaped(read_value(device/"pp_dpm_sclk",true)).c_str(),escaped(read_value(device/"pp_dpm_mclk",true)).c_str(),escaped(power).c_str(),escaped(temp).c_str());
}
} // namespace probe

int main(int argc,char** argv) {
    using namespace probe;
    try {
        if(argc<2) {std::fprintf(stderr,"usage: strix-hip-moe-int8-wmma-probe MODEL.gguf [--layer N] [--seconds 390] [--rounds 5]\n");return 2;}
        unsigned layer=0,rounds=5;double seconds=390;
        for(int i=2;i<argc;++i) {
            require(i+1<argc,"missing argument value");const std::string key=argv[i++];
            if(key=="--layer")layer=unsigned(std::stoul(argv[i]));else if(key=="--seconds")seconds=std::stod(argv[i]);
            else if(key=="--rounds")rounds=unsigned(std::stoul(argv[i]));else throw std::runtime_error("unknown option");
        }
        require(seconds>0&&seconds<=390&&rounds>=3&&rounds<=9,"seconds must be <=390; rounds must be 3..9");
        require(std::fegetround()==FE_TONEAREST,"requires round to nearest even");alarm(400);Budget budget(seconds);identity(layer);telemetry("start");
        resources<64,128>();resources<64,256>();resources<128,128>();resources<128,256>();exact_fixtures(budget);
        const auto info=strix::gguf::read_file(argv[1]);budget.check();Layer weights(info,layer,argv[1]);
        for(const auto* tensor:{&weights.gate,&weights.up,&weights.down}) {
            strix::Sha256Fast hash;hash.update(std::as_bytes(std::span(tensor->host)));
            const auto digest=strix::model_content_identity_hex(hash.finalize());
            std::printf("{\"packed_tensor_sha256\":\"%s\",\"type\":%u,\"bytes\":%zu}\n",digest.c_str(),tensor->type,tensor->host.size());budget.check();
        }
        for(unsigned t:{256U,8192U,16384U}) {
            budget.check(10);Fixture f(weights,t,t==256);if(t==256)rejection_fixtures(f);
            const auto cold_inc=run_inc(f);std::printf("{\"cold_incumbent_ms\":%.6f,\"T\":%u}\n",cold_inc.chain,t);production_witness(f);integer_control(f,budget);
            const auto ref=oracles(f,budget);run_inc(f,true);status_ok(f);accuracy(f,ref,snapshot(f),false,0,0);
            variant<64,128>(f,ref,budget,rounds);variant<64,256>(f,ref,budget,rounds);
            variant<128,128>(f,ref,budget,rounds);variant<128,256>(f,ref,budget,rounds);
            telemetry("shape-end");
        }
        gpu(hipDeviceSynchronize(),"final sync");budget.check(0);std::printf("{\"turn_complete\":true,\"elapsed_seconds\":%.3f,\"qualified\":false}\n",seconds-budget.left());return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"probe: %s\n",e.what());std::puts("{\"turn_complete\":false,\"kill\":true,\"qualified\":false}");return 1;
    }
}
