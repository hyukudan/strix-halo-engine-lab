#include "strix/hip_mtp_fast_draft.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix {
namespace {
constexpr unsigned hidden=2560,inter=640,experts=512,slots=11,routed=10,waves=8;
template<bool Q4> constexpr unsigned block_bytes=Q4?18U:34U;
template<bool Q4> constexpr unsigned gate_row_bytes=hidden/32U*block_bytes<Q4>;
template<bool Q4> constexpr unsigned down_row_bytes=inter/32U*block_bytes<Q4>;
void check(hipError_t e,const char* m) {if(e!=hipSuccess) throw std::runtime_error(std::string(m)+": "+hipGetErrorString(e));}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for(unsigned d=16;d;d>>=1U) v+=__shfl_xor(v,static_cast<int>(d),32);
    return v;
}
__device__ __forceinline__ std::uint16_t to_bf16(float x) {
    const unsigned bits=__float_as_uint(x);
    if((bits&0x7f800000U)==0x7f800000U && (bits&0x007fffffU)) return static_cast<std::uint16_t>((bits>>16U)|0x0040U);
    return static_cast<std::uint16_t>((bits+0x7fffU+((bits>>16U)&1U))>>16U);
}
// Eight consecutive Q8_0 codes at feature (multiple of 8) of one row, dotted with x[0..7].
__device__ __forceinline__ float q8_dot8(const unsigned char* row,unsigned feature,const float* x,float sum) {
    const auto* block=row+(feature/32U)*34U;
    const float scale=__half2float(*reinterpret_cast<const __half*>(block));
    const auto* codes=reinterpret_cast<const signed char*>(block+2U+feature%32U);
    float local=0.0F;
#pragma unroll
    for(unsigned i=0;i<8;++i) local=fmaf(static_cast<float>(codes[i]),x[i],local);
    return fmaf(scale,local,sum);
}
// Q4_0 block: half scale, element j in the low nibble of byte j (j<16) or the high nibble of byte j-16.
__device__ __forceinline__ float q4_dot8(const unsigned char* row,unsigned feature,const float* x,float sum) {
    const auto* block=row+(feature/32U)*18U;
    const float scale=__half2float(*reinterpret_cast<const __half*>(block));
    const unsigned within=feature%32U,shift=within>=16U?4U:0U;
    const auto* codes=block+2U+(within&15U);
    float local=0.0F;
#pragma unroll
    for(unsigned i=0;i<8;++i) local=fmaf(static_cast<float>(static_cast<int>((codes[i]>>shift)&15U)-8),x[i],local);
    return fmaf(scale,local,sum);
}
template<bool Q4>
__device__ __forceinline__ float dot8(const unsigned char* row,unsigned feature,const float* x,float sum) {
    if constexpr(Q4) return q4_dot8(row,feature,x,sum); else return q8_dot8(row,feature,x,sum);
}
template<bool Q4>
__global__ __launch_bounds__(256) void mtp_fast_gate_up(HipMtpFastMoeViews v) {
    const unsigned slot=blockIdx.y,wave=threadIdx.x/32U,lane=threadIdx.x%32U,row=blockIdx.x*waves+wave;
    if(!slot && !blockIdx.x && threadIdx.x<routed &&
       (v.routes[threadIdx.x]>=experts || !isfinite(v.route_weights[threadIdx.x]))) atomicOr(v.status,1U);
    if(*v.status) return;
    float x[8*hidden/256U];
#pragma unroll
    for(unsigned c=0;c<hidden/256U;++c) {
        const uint4 bits=*reinterpret_cast<const uint4*>(v.input+c*256U+8U*lane);
        const unsigned w[4]={bits.x,bits.y,bits.z,bits.w};
#pragma unroll
        for(unsigned i=0;i<4;++i) {x[8*c+2*i]=__uint_as_float(w[i]<<16U);x[8*c+2*i+1]=__uint_as_float(w[i]&0xffff0000U);}
    }
    if(row>=inter) {
        if(slot!=routed || row!=inter) return;
        float s=0.0F;
#pragma unroll
        for(unsigned c=0;c<hidden/256U;++c)
#pragma unroll
            for(unsigned i=0;i<8;++i) s=fmaf(v.shared_scalar[c*256U+8U*lane+i],x[8*c+i],s);
        s=wave_sum(s);
        if(!lane) v.workspace[slots*inter]=1.0F/(1.0F+expf(-s));
        return;
    }
    const unsigned char *g,*u;
    if(slot<routed) {
        const auto e=v.routes[slot];
        if(e>=experts) {if(!lane) v.workspace[slot*inter+row]=0.0F;return;}
        const auto offset=(static_cast<std::uint64_t>(e)*inter+row)*gate_row_bytes<Q4>;
        g=static_cast<const unsigned char*>(v.gate)+offset;u=static_cast<const unsigned char*>(v.up)+offset;
    } else {
        g=static_cast<const unsigned char*>(v.shared_gate)+row*gate_row_bytes<Q4>;u=static_cast<const unsigned char*>(v.shared_up)+row*gate_row_bytes<Q4>;
    }
    float gs=0.0F,us=0.0F;
#pragma unroll
    for(unsigned c=0;c<hidden/256U;++c) {gs=dot8<Q4>(g,c*256U+8U*lane,x+8*c,gs);us=dot8<Q4>(u,c*256U+8U*lane,x+8*c,us);}
    gs=wave_sum(gs);us=wave_sum(us);
    if(!lane) v.workspace[slot*inter+row]=(gs/(1.0F+expf(-gs)))*us;
}
template<bool Q4>
__global__ __launch_bounds__(256) void mtp_fast_down(HipMtpFastMoeViews v) {
    __shared__ float act[slots*inter];
    __shared__ float weight[slots];
    __shared__ const unsigned char* rows[slots];
    if(*v.status) return;
    for(unsigned i=threadIdx.x;i<slots*inter;i+=256U) act[i]=v.workspace[i];
    if(threadIdx.x<slots) {
        const unsigned s=threadIdx.x;
        weight[s]=s<routed?v.route_weights[s]:v.workspace[slots*inter];
        rows[s]=s<routed?static_cast<const unsigned char*>(v.down)+static_cast<std::uint64_t>(v.routes[s])*hidden*down_row_bytes<Q4>:
            static_cast<const unsigned char*>(v.shared_down);
    }
    __syncthreads();
    const unsigned wave=threadIdx.x/32U,lane=threadIdx.x%32U,row=blockIdx.x*waves+wave;
    float sum=0.0F;
    for(unsigned s=0;s<slots;++s) {
        const auto* r=rows[s]+static_cast<std::uint64_t>(row)*down_row_bytes<Q4>;
        float p=0.0F;
        for(unsigned group=lane;group<inter/8U;group+=32U) p=dot8<Q4>(r,8U*group,act+s*inter+8U*group,p);
        sum=fmaf(weight[s],p,sum);
    }
    sum=wave_sum(sum);
    if(!lane) v.output[row]=to_bf16(sum);
}
template<bool InBf16,bool OutBf16,bool Q4=false>
__global__ __launch_bounds__(256) void mtp_draft_q8_gemv(const void* input,const unsigned char* q8,std::uint32_t width,
    std::uint32_t rows,void* output) {
    const unsigned lane=threadIdx.x%32U,row=blockIdx.x*waves+threadIdx.x/32U;
    if(row>=rows) return;
    const auto* r=q8+static_cast<std::uint64_t>(row)*(width/32U)*block_bytes<Q4>;
    float sum=0.0F,x[8];
    for(unsigned base=8U*lane;base<width;base+=256U) {
        if constexpr(InBf16) {
            const uint4 bits=*reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(input)+base);
            const unsigned w[4]={bits.x,bits.y,bits.z,bits.w};
#pragma unroll
            for(unsigned i=0;i<4;++i) {x[2*i]=__uint_as_float(w[i]<<16U);x[2*i+1]=__uint_as_float(w[i]&0xffff0000U);}
        } else {
            const auto* f=static_cast<const float*>(input)+base;
#pragma unroll
            for(unsigned i=0;i<8;++i) x[i]=f[i];
        }
        sum=dot8<Q4>(r,base,x,sum);
    }
    sum=wave_sum(sum);
    if(lane) return;
    if constexpr(OutBf16) static_cast<std::uint16_t*>(output)[row]=to_bf16(sum);
    else static_cast<float*>(output)[row]=sum;
}
// Q8_0 -> Q4_0 per 32-block, the draft head's rule: d=signed max/-8, q=min(15, v/d+8.5).
__global__ void mtp_q8_to_q4(const unsigned char* q8,std::uint64_t blocks,unsigned char* q4) {
    const auto b=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(b>=blocks) return;
    const auto* source=q8+b*34U;auto* target=q4+b*18U;
    const float scale=__half2float(*reinterpret_cast<const __half*>(source));
    float values[32],max=0.0F,amax=0.0F;
    for(unsigned i=0;i<32;++i) {
        values[i]=scale*static_cast<float>(static_cast<signed char>(source[2+i]));
        if(fabsf(values[i])>amax) {amax=fabsf(values[i]);max=values[i];}
    }
    const float d=max/-8.0F,inverse=d!=0.0F?1.0F/d:0.0F;
    *reinterpret_cast<__half*>(target)=__float2half(d);
    for(unsigned j=0;j<16;++j) {
        const auto lo=static_cast<unsigned>(fminf(15.0F,values[j]*inverse+8.5F));
        const auto hi=static_cast<unsigned>(fminf(15.0F,values[j+16]*inverse+8.5F));
        target[2+j]=static_cast<unsigned char>(lo|(hi<<4U));
    }
}
struct Q4Copies {std::mutex lock;std::vector<std::pair<const void*,void*>> copies;};
Q4Copies& q4_copies() {static Q4Copies c;return c;}
__global__ void mtp_dequantize_q8(const unsigned char* q8,std::uint64_t elements,float* output) {
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=elements) return;
    const auto* block=q8+(i/32U)*34U;
    output[i]=__half2float(*reinterpret_cast<const __half*>(block))*static_cast<float>(static_cast<signed char>(block[2U+i%32U]));
}
} // namespace
void launch_hip_mtp_dequantize_q8(const void* q8,std::uint64_t elements,float* output,void* stream) {
    if(!q8 || !output || !stream || !elements || elements%32U) throw std::invalid_argument("MTP Q8 dequantize views are invalid");
    hipLaunchKernelGGL(mtp_dequantize_q8,dim3(static_cast<unsigned>((elements+255U)/256U)),dim3(256),0,static_cast<hipStream_t>(stream),
        static_cast<const unsigned char*>(q8),elements,output);
    check(hipGetLastError(),"MTP Q8 dequantize launch");
}
void register_hip_mtp_draft_q4(const void* q8,std::uint64_t elements,void* stream) {
    if(!q8 || !elements || elements%32U || !stream) throw std::invalid_argument("MTP draft Q4 copy views are invalid");
    auto& c=q4_copies();const std::lock_guard guard(c.lock);
    for(const auto& known:c.copies) if(known.first==q8) return;
    const auto blocks=elements/32U;void* q4=nullptr;
    check(hipMalloc(&q4,blocks*18U),"MTP draft Q4 copy");
    hipLaunchKernelGGL(mtp_q8_to_q4,dim3(static_cast<unsigned>((blocks+255U)/256U)),dim3(256),0,static_cast<hipStream_t>(stream),
        static_cast<const unsigned char*>(q8),blocks,static_cast<unsigned char*>(q4));
    check(hipGetLastError(),"MTP draft Q4 conversion launch");
    c.copies.emplace_back(q8,q4);
}
const void* hip_mtp_draft_q4(const void* q8) noexcept {
    auto& c=q4_copies();const std::lock_guard guard(c.lock);
    for(const auto& known:c.copies) if(known.first==q8) return known.second;
    return nullptr;
}
void launch_hip_mtp_draft_q8_gemv(const void* input,bool input_bf16,const void* q8,std::uint32_t width,
    std::uint32_t rows,void* output,bool output_bf16,void* stream,bool any_order) {
    if(!input || !q8 || !output || !stream || !rows || !width || width%32U ||
       reinterpret_cast<std::uintptr_t>(input)%(input_bf16?16U:4U) || reinterpret_cast<std::uintptr_t>(q8)%2U)
        throw std::invalid_argument("MTP draft Q8 GEMV views are invalid");
    const auto s=static_cast<hipStream_t>(stream);const dim3 grid((rows+waves-1U)/waves);
    const auto start=[&](auto kernel,const unsigned char* w) {
        if(any_order) hipExtLaunchKernelGGL(kernel,grid,dim3(256),0,s,nullptr,nullptr,hipExtAnyOrderLaunch,input,w,width,rows,output);
        else hipLaunchKernelGGL(kernel,grid,dim3(256),0,s,input,w,width,rows,output);
    };
    if(const auto* q4=static_cast<const unsigned char*>(hip_mtp_draft_q4(q8))) {
        if(input_bf16 && output_bf16) start(mtp_draft_q8_gemv<true,true,true>,q4);
        else if(input_bf16) start(mtp_draft_q8_gemv<true,false,true>,q4);
        else if(output_bf16) start(mtp_draft_q8_gemv<false,true,true>,q4);
        else start(mtp_draft_q8_gemv<false,false,true>,q4);
        check(hipGetLastError(),"MTP draft Q4 GEMV launch");
        return;
    }
    const auto* w=static_cast<const unsigned char*>(q8);
    if(input_bf16 && output_bf16) start(mtp_draft_q8_gemv<true,true>,w);
    else if(input_bf16) start(mtp_draft_q8_gemv<true,false>,w);
    else if(output_bf16) start(mtp_draft_q8_gemv<false,true>,w);
    else start(mtp_draft_q8_gemv<false,false>,w);
    check(hipGetLastError(),"MTP draft Q8 GEMV launch");
}
void launch_hip_mtp_fast_moe(const HipMtpFastMoeViews& v,void* stream) {
    if(!v.input || !v.routes || !v.route_weights || !v.gate || !v.up || !v.down || !v.shared_gate || !v.shared_up ||
       !v.shared_down || !v.shared_scalar || !v.workspace || !v.output || !v.status || !stream ||
       reinterpret_cast<std::uintptr_t>(v.input)%16U)
        throw std::invalid_argument("MTP fast MoE views are incomplete or misaligned");
    const auto s=static_cast<hipStream_t>(stream);
    auto q4=v;
    q4.gate=hip_mtp_draft_q4(v.gate);q4.up=hip_mtp_draft_q4(v.up);q4.down=hip_mtp_draft_q4(v.down);
    q4.shared_gate=hip_mtp_draft_q4(v.shared_gate);q4.shared_up=hip_mtp_draft_q4(v.shared_up);q4.shared_down=hip_mtp_draft_q4(v.shared_down);
    if(q4.gate && q4.up && q4.down && q4.shared_gate && q4.shared_up && q4.shared_down) {
        hipLaunchKernelGGL(mtp_fast_gate_up<true>,dim3((inter+waves)/waves,slots),dim3(256),0,s,q4);
        check(hipGetLastError(),"MTP fast Q4 gate/up launch");
        hipLaunchKernelGGL(mtp_fast_down<true>,dim3(hidden/waves),dim3(256),0,s,q4);
        check(hipGetLastError(),"MTP fast Q4 down launch");
        return;
    }
    hipLaunchKernelGGL(mtp_fast_gate_up<false>,dim3((inter+waves)/waves,slots),dim3(256),0,s,v);
    check(hipGetLastError(),"MTP fast gate/up launch");
    hipLaunchKernelGGL(mtp_fast_down<false>,dim3(hidden/waves),dim3(256),0,s,v);
    check(hipGetLastError(),"MTP fast down launch");
}
} // namespace strix
