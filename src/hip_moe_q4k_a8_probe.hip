#include "strix/hip_moe_q4k_a8_probe.hpp"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "Q4K A8G32 requires nonfinite checks and strict floating-point semantics"
#endif
#include <hip/hip_fp16.h>
#include <cmath>
#include <stdexcept>
#include <string>
#pragma clang fp contract(off)
#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__gfx1151__)
#error "Experimental Q4K A8 probe admits gfx1151 only"
#endif
#if !__has_builtin(__builtin_amdgcn_sudot4)
#error "No native signed-A unsigned-B dot4 support"
#endif
#endif
namespace strix::moe_q4k_a8 {
namespace {
// Explicit target option -fhip-fp32-correctly-rounded-divide-sqrt is mandatory.
// The available compiler lowers /; the declared OCML rte entry point did not link.
// Actual IR/ISA, denormal modes and independent device RN oracles remain required.
__device__ float divide_rne(float a, float b) { return a/b; }
__device__ int dot(int a, unsigned b, int acc) {
    return __builtin_amdgcn_sudot4(true,a,false,static_cast<int>(b),acc,false);
}
__device__ unsigned factor(uint4 header, unsigned group, bool minimum) {
    const unsigned shift = 8U*(group&3U);
    const unsigned a=(header.y>>shift)&255U,b=(header.z>>shift)&255U,c=(header.w>>shift)&255U;
    return group<4U ? (minimum?b:a)&63U : minimum ? (c>>4U)|((b>>6U)<<4U) : (c&15U)|((a>>6U)<<4U);
}
__device__ float half_float(unsigned bits) { return __half2float(__ushort_as_half(static_cast<unsigned short>(bits))); }
__global__ void validate_routes(const unsigned* routes, unsigned top_k, unsigned experts, unsigned* status) {
    if (threadIdx.x<top_k && routes[threadIdx.x]>=experts) atomicOr(status,invalid_route);
}
template<bool Bf16>
__global__ void quantize(const void* input, unsigned hidden, signed char* codes, float* deltas,
    int* sums, unsigned* status) {
    __shared__ __align__(16) signed char local_codes[max_hidden];
    __shared__ __align__(16) float local_deltas[max_hidden/32U];
    __shared__ __align__(16) int local_sums[max_hidden/32U];
    __shared__ unsigned bad, initial_status;
    if(threadIdx.x==0) {initial_status=*status;bad=0;}
    __syncthreads();
    if(initial_status!=0) return;
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32U;
    for(unsigned group=wave;group<hidden/32U;group+=8U) {
        const unsigned i=group*32U+lane;
        float x;
        if constexpr(Bf16) x=__uint_as_float(static_cast<unsigned>(static_cast<const unsigned short*>(input)[i])<<16U);
        else x=static_cast<const float*>(input)[i];
        if(!isfinite(x)) {atomicOr(&bad,nonfinite_activation);x=0.0F;}
        float a=fabsf(x);
        for(int stride=16;stride!=0;stride/=2) a=fmaxf(a,__shfl_xor(a,stride,32));
        float delta=a==0.0F?1.0F:divide_rne(a,127.0F);
        if(!(delta>0.0F) || !isfinite(delta)) {atomicOr(&bad,invalid_delta);delta=1.0F;}
        const float rounded=rintf(divide_rne(x,delta));
        const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
        local_codes[i]=static_cast<signed char>(code);
        int sum=code;
        for(int stride=16;stride!=0;stride/=2) sum+=__shfl_xor(sum,stride,32);
        if(lane==0) {local_deltas[group]=delta;local_sums[group]=sum;}
    }
    __syncthreads();
    if(bad!=0) {if(threadIdx.x==0) atomicOr(status,bad);return;}
    for(unsigned i=threadIdx.x;i<hidden;i+=256U) codes[i]=local_codes[i];
    for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U) {deltas[i]=local_deltas[i];sums[i]=local_sums[i];}
}
__global__ void project(const unsigned char* gate,const unsigned char* up,const unsigned* routes,
    const signed char* codes,const float* deltas,const int* sums,const unsigned* status,
    float* output,unsigned hidden,unsigned rows) {
    __shared__ __align__(16) signed char local_codes[max_hidden];
    __shared__ __align__(16) float local_deltas[max_hidden/32U];
    __shared__ __align__(16) int local_sums[max_hidden/32U];
    // Prior kernels are on the same stream; this kernel never changes status.
    if(*status!=0) return;
    for(unsigned i=threadIdx.x;i<hidden;i+=256U) local_codes[i]=codes[i];
    for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U) {local_deltas[i]=deltas[i];local_sums[i]=sums[i];}
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U;
    const unsigned row=blockIdx.x*32U+(threadIdx.x/32U)*4U+lane/8U;
    const unsigned which=blockIdx.y,slot=blockIdx.z,expert=routes[slot];
    const unsigned blocks=hidden/256U;
    const auto* weights=(which==0?gate:up)+(static_cast<unsigned long long>(expert)*rows+row)*blocks*144U;
    float acc=0.0F;
    #pragma unroll 1
    for(unsigned block=0;block<blocks;++block) {
        const auto* packed=weights+block*144U;
        const uint4 hd=*reinterpret_cast<const uint4*>(packed);
        const uint4 q=*reinterpret_cast<const uint4*>(packed+16U+c*16U);
        const int4 a0=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+h*16U);
        const int4 a1=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+32U+h*16U);
        constexpr unsigned mask=0x0f0f0f0fU;
        int r0=dot(a0.x,q.x&mask,0);r0=dot(a0.y,q.y&mask,r0);r0=dot(a0.z,q.z&mask,r0);r0=dot(a0.w,q.w&mask,r0);
        int r1=dot(a1.x,(q.x>>4U)&mask,0);r1=dot(a1.y,(q.y>>4U)&mask,r1);r1=dot(a1.z,(q.z>>4U)&mask,r1);r1=dot(a1.w,(q.w>>4U)&mask,r1);
        r0+=__shfl_xor(r0,1,32);r1+=__shfl_xor(r1,1,32);
        const unsigned g0=2U*p,g1=g0+1U;
        const int i0=static_cast<int>(factor(hd,g0,false))*r0,i1=static_cast<int>(factor(hd,g1,false))*r1;
        const int j0=static_cast<int>(factor(hd,g0,true))*local_sums[block*8U+g0];
        const int j1=static_cast<int>(factor(hd,g1,true))*local_sums[block*8U+g1];
        const float d=half_float(hd.x&65535U),dm=half_float(hd.x>>16U);
        const float u0=d*static_cast<float>(i0),v0=dm*static_cast<float>(j0),t0=u0-v0;
        const float u1=d*static_cast<float>(i1),v1=dm*static_cast<float>(j1),t1=u1-v1;
        const float c0=local_deltas[block*8U+g0]*t0,c1=local_deltas[block*8U+g1]*t1;
        const float paired=c0+c1;
        acc=acc+paired;
    }
    // Only c0 is observable. c0 receives exactly ((p0+p1)+(p2+p3)).
    acc=acc+__shfl_xor(acc,2,32);
    acc=acc+__shfl_xor(acc,4,32);
    if(c==0) output[(slot*2U+which)*rows+row]=acc;
}
// Two independent projection DAGs consume the same staged activation values.
// Keep all affine corrections, operand order and rounding boundaries literal.
__device__ __forceinline__ float paired_shared(const unsigned char* packed,int4 a0,int4 a1,
    int s0,int s1,float e0,float e1,unsigned c,unsigned p) {
    const uint4 hd=*reinterpret_cast<const uint4*>(packed);
    const uint4 q=*reinterpret_cast<const uint4*>(packed+16U+c*16U);
    constexpr unsigned mask=0x0f0f0f0fU;
    int r0=dot(a0.x,q.x&mask,0);r0=dot(a0.y,q.y&mask,r0);r0=dot(a0.z,q.z&mask,r0);r0=dot(a0.w,q.w&mask,r0);
    int r1=dot(a1.x,(q.x>>4U)&mask,0);r1=dot(a1.y,(q.y>>4U)&mask,r1);r1=dot(a1.z,(q.z>>4U)&mask,r1);r1=dot(a1.w,(q.w>>4U)&mask,r1);
    r0+=__shfl_xor(r0,1,32);r1+=__shfl_xor(r1,1,32);
    const unsigned g0=2U*p,g1=g0+1U;
    const int i0=static_cast<int>(factor(hd,g0,false))*r0,i1=static_cast<int>(factor(hd,g1,false))*r1;
    const int j0=static_cast<int>(factor(hd,g0,true))*s0;
    const int j1=static_cast<int>(factor(hd,g1,true))*s1;
    const float d=half_float(hd.x&65535U),dm=half_float(hd.x>>16U);
    const float u0=d*static_cast<float>(i0),v0=dm*static_cast<float>(j0),t0=u0-v0;
    const float u1=d*static_cast<float>(i1),v1=dm*static_cast<float>(j1),t1=u1-v1;
    const float c0=e0*t0,c1=e1*t1;
    return c0+c1;
}
__global__ void project_shared_gate_up(const unsigned char* gate,const unsigned char* up,const unsigned* routes,
    const signed char* codes,const float* deltas,const int* sums,const unsigned* status,
    float* output,unsigned hidden,unsigned rows) {
    __shared__ __align__(16) signed char local_codes[max_hidden];
    __shared__ __align__(16) float local_deltas[max_hidden/32U];
    __shared__ __align__(16) int local_sums[max_hidden/32U];
    if(*status!=0) return;
    for(unsigned i=threadIdx.x;i<hidden;i+=256U) local_codes[i]=codes[i];
    for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U) {local_deltas[i]=deltas[i];local_sums[i]=sums[i];}
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U;
    const unsigned row=blockIdx.x*32U+(threadIdx.x/32U)*4U+lane/8U;
    const unsigned slot=blockIdx.z,expert=routes[slot],blocks=hidden/256U;
    const auto offset=(static_cast<unsigned long long>(expert)*rows+row)*blocks*144U;
    const auto* gate_weights=gate+offset;const auto* up_weights=up+offset;
    const unsigned g0=2U*p,g1=g0+1U;
    float acc_gate=0.0F,acc_up=0.0F;
    #pragma unroll 1
    for(unsigned block=0;block<blocks;++block) {
        const int4 a0=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+h*16U);
        const int4 a1=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+32U+h*16U);
        const int s0=local_sums[block*8U+g0],s1=local_sums[block*8U+g1];
        const float e0=local_deltas[block*8U+g0],e1=local_deltas[block*8U+g1];
        const float paired_gate=paired_shared(gate_weights+block*144U,a0,a1,s0,s1,e0,e1,c,p);
        acc_gate=acc_gate+paired_gate;
        const float paired_up=paired_shared(up_weights+block*144U,a0,a1,s0,s1,e0,e1,c,p);
        acc_up=acc_up+paired_up;
    }
    acc_gate=acc_gate+__shfl_xor(acc_gate,2,32);acc_gate=acc_gate+__shfl_xor(acc_gate,4,32);
    acc_up=acc_up+__shfl_xor(acc_up,2,32);acc_up=acc_up+__shfl_xor(acc_up,4,32);
    if(c==0) {output[(slot*2U+0U)*rows+row]=acc_gate;output[(slot*2U+1U)*rows+row]=acc_up;}
}
__global__ void division_receipt(const unsigned* a,const unsigned* b,unsigned* output,unsigned count) {
    const unsigned i=blockIdx.x*256U+threadIdx.x;
    if(i<count) output[i]=__float_as_uint(divide_rne(__uint_as_float(a[i]),__uint_as_float(b[i])));
}
__global__ void half_conversion(unsigned* output) {
    const unsigned i=blockIdx.x*256U+threadIdx.x;
    output[i]=__float_as_uint(half_float(i));
}
void check(hipError_t error,const char* operation) {
    if(error!=hipSuccess) throw std::runtime_error(std::string(operation)+": "+hipGetErrorString(error));
}
}
void enqueue_probe(const Request& request,const Bindings& b,hipStream_t stream) {
    validate(request,b);
    check(hipPeekAtLastError(),"Q4K A8 pre-enqueue stale error");
    auto* status=reinterpret_cast<unsigned*>(b.status.address);
    const auto* routes=reinterpret_cast<const unsigned*>(b.routes.address);
    validate_routes<<<1,32,0,stream>>>(routes,request.shape.top_k,request.shape.experts,status);
    check(hipGetLastError(),"Q4K A8 route enqueue");
    const void* input=reinterpret_cast<const void*>(b.input.address);
    auto* codes=reinterpret_cast<signed char*>(b.codes.address);auto* deltas=reinterpret_cast<float*>(b.deltas.address);
    auto* sums=reinterpret_cast<int*>(b.sums.address);
    if(request.input==InputKind::bf16) quantize<true><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    else quantize<false><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    check(hipGetLastError(),"Q4K A8 quantizer enqueue");
    project<<<dim3(request.shape.rows/32U,2,request.shape.top_k),256,0,stream>>>(
        reinterpret_cast<const unsigned char*>(b.gate.address),reinterpret_cast<const unsigned char*>(b.up.address),
        routes,codes,deltas,sums,status,reinterpret_cast<float*>(b.output.address),request.shape.hidden,request.shape.rows);
    check(hipGetLastError(),"Q4K A8 projection enqueue");
}
void enqueue_probe_shared_gate_up(const Request& request,const Bindings& b,hipStream_t stream) {
    validate(request,b);
    check(hipPeekAtLastError(),"Q4K A8 pre-enqueue stale error");
    auto* status=reinterpret_cast<unsigned*>(b.status.address);
    const auto* routes=reinterpret_cast<const unsigned*>(b.routes.address);
    validate_routes<<<1,32,0,stream>>>(routes,request.shape.top_k,request.shape.experts,status);
    check(hipGetLastError(),"Q4K A8 route enqueue");
    const void* input=reinterpret_cast<const void*>(b.input.address);
    auto* codes=reinterpret_cast<signed char*>(b.codes.address);auto* deltas=reinterpret_cast<float*>(b.deltas.address);
    auto* sums=reinterpret_cast<int*>(b.sums.address);
    if(request.input==InputKind::bf16) quantize<true><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    else quantize<false><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    check(hipGetLastError(),"Q4K A8 quantizer enqueue");
    project_shared_gate_up<<<dim3(request.shape.rows/32U,1,request.shape.top_k),256,0,stream>>>(
        reinterpret_cast<const unsigned char*>(b.gate.address),reinterpret_cast<const unsigned char*>(b.up.address),
        routes,codes,deltas,sums,status,reinterpret_cast<float*>(b.output.address),request.shape.hidden,request.shape.rows);
    check(hipGetLastError(),"Q4K A8 shared gate/up projection enqueue");
}
void enqueue_division_probe(const std::uint32_t* a,const std::uint32_t* b,std::uint32_t* output,unsigned count,hipStream_t stream) {
    if(!a || !b || !output || count==0) throw std::invalid_argument("invalid division receipt bindings");
    division_receipt<<<(count+255U)/256U,256,0,stream>>>(a,b,output,count);check(hipGetLastError(),"Q4K A8 division receipt enqueue");
}
void enqueue_half_conversion_probe(std::uint32_t* output,hipStream_t stream) {
    if(output==nullptr) throw std::invalid_argument("null half conversion destination");
    half_conversion<<<256,256,0,stream>>>(output);check(hipGetLastError(),"Q4K A8 half conversion enqueue");
}
}
