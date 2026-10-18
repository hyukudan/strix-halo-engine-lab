#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "A8 CSR gate/up requires strict floating-point semantics"
#endif
#include <hip/hip_fp16.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#pragma clang fp contract(off)
#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__gfx1151__) || !__has_builtin(__builtin_amdgcn_sudot4)
#error "A8 CSR gate/up requires gfx1151 signed-A unsigned-B dot4"
#endif
#endif

namespace strix::moe_q4k_a8::known_rows {
namespace {
struct DeviceRows {
    const signed char* codes[8]{}; const float* deltas[8]{}; const int* sums[8]{};
    const unsigned* routes[8]{}; unsigned* status[8]{}; float* output[8]{};
    signed char* down_codes[8]{}; float* down_deltas[8]{}; int* down_sums[8]{}; float* rawdots[8]{};
};
struct DeviceCsr {
    unsigned* summary; unsigned* offsets; unsigned* touched;
    unsigned* rows; unsigned* slots; unsigned* experts; unsigned* begins;
};
void check(hipError_t error,const char* operation) {
    if(error!=hipSuccess)throw std::runtime_error(std::string(operation)+": "+hipGetErrorString(error));
}
__device__ int dot(int a,unsigned b,int acc) {
    return __builtin_amdgcn_sudot4(true,a,false,static_cast<int>(b),acc,false);
}
__device__ unsigned factor(uint4 header,unsigned group,bool minimum) {
    const unsigned shift=8U*(group&3U);
    const unsigned a=(header.y>>shift)&255U,b=(header.z>>shift)&255U,c=(header.w>>shift)&255U;
    return group<4U?(minimum?b:a)&63U:minimum?(c>>4U)|((b>>6U)<<4U):(c&15U)|((a>>6U)<<4U);
}
struct Coefficient {
    uint4 low,high;
    unsigned scale0,scale1,minimum0,minimum1;
    float d,dm;
};
__device__ Coefficient coefficient(uint4 hd,uint4 q,unsigned p) {
    constexpr unsigned mask=0x0f0f0f0fU;
    return {{q.x&mask,q.y&mask,q.z&mask,q.w&mask},
        {(q.x>>4U)&mask,(q.y>>4U)&mask,(q.z>>4U)&mask,(q.w>>4U)&mask},
        factor(hd,2U*p,false),factor(hd,2U*p+1U,false),factor(hd,2U*p,true),factor(hd,2U*p+1U,true),
        __half2float(__ushort_as_half(static_cast<unsigned short>(hd.x&65535U))),
        __half2float(__ushort_as_half(static_cast<unsigned short>(hd.x>>16U)))};
}
__device__ Coefficient coefficient(const unsigned char* packed,unsigned c,unsigned p) {
    return coefficient(*reinterpret_cast<const uint4*>(packed),*reinterpret_cast<const uint4*>(packed+16U+c*16U),p);
}
__device__ __forceinline__ float paired(Coefficient q,int4 a0,int4 a1,int s0,int s1,float e0,float e1) {
    int r0=dot(a0.x,q.low.x,0);r0=dot(a0.y,q.low.y,r0);r0=dot(a0.z,q.low.z,r0);r0=dot(a0.w,q.low.w,r0);
    int r1=dot(a1.x,q.high.x,0);r1=dot(a1.y,q.high.y,r1);r1=dot(a1.z,q.high.z,r1);r1=dot(a1.w,q.high.w,r1);
    r0+=__shfl_xor(r0,1,32);r1+=__shfl_xor(r1,1,32);
    const int i0=static_cast<int>(q.scale0)*r0,i1=static_cast<int>(q.scale1)*r1;
    const int j0=static_cast<int>(q.minimum0)*s0,j1=static_cast<int>(q.minimum1)*s1;
    const float u0=q.d*static_cast<float>(i0),v0=q.dm*static_cast<float>(j0),t0=u0-v0;
    const float u1=q.d*static_cast<float>(i1),v1=q.dm*static_cast<float>(j1),t1=u1-v1;
    const float c0=e0*t0,c1=e1*t1;
    return c0+c1;
}

// The public component does not trust caller-provided device CSR contents.
// Admit the complete <=80 assignment permutation before any projection CTA.
__global__ void admit_csr(DeviceRows view,DeviceCsr csr,unsigned rows,unsigned experts,unsigned top_k) {
    if(threadIdx.x) return;
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if(failure){csr.summary[0]=failure;return;}
    const unsigned count=rows*top_k,batches=csr.summary[2],touched=csr.summary[1];
    bool bad=csr.summary[3]!=count||csr.summary[6]!=rows||csr.summary[7]!=1U||
        !batches||batches>count||!touched||touched>experts||touched>count;
    if(!bad) {
        bad=csr.offsets[0]!=0U||csr.offsets[experts]!=count||csr.begins[0]!=0U||csr.begins[batches]!=count;
        for(unsigned e=0;e<experts;++e)bad|=csr.offsets[e]>csr.offsets[e+1U]||csr.offsets[e+1U]>count;
        for(unsigned t=0;t<touched;++t) {
            const auto e=csr.touched[t];
            bad|=e>=experts||(t&&e<=csr.touched[t-1U]);
            if(e<experts)bad|=csr.offsets[e]==csr.offsets[e+1U];
        }
    }
    bool seen[80]={};
    if(!bad) for(unsigned b=0;b<batches;++b) {
        const auto begin=csr.begins[b],end=csr.begins[b+1U],e=csr.experts[b];
        if(e>=experts||begin>=end||end>count||end-begin>rows||(b&&e<csr.experts[b-1U])){bad=true;break;}
        if(begin<csr.offsets[e]||end>csr.offsets[e+1U]){bad=true;break;}
        for(unsigned a=begin;a<end;++a) {
            const auto r=csr.rows[a],slot=csr.slots[a];
            if(r>=rows||slot>=top_k){bad=true;break;}
            const auto key=r*top_k+slot;
            if(seen[key]||view.routes[r][slot]!=e){bad=true;break;}
            seen[key]=true;
        }
        if(bad)break;
    }
    if(!bad)for(unsigned a=0;a<count;++a)bad|=!seen[a];
    if(bad) {
        atomicOr(csr.summary,invalid_route);
        for(unsigned r=0;r<rows;++r)atomicOr(view.status[r],invalid_route);
    }
}

template<unsigned Rows>
__global__ void project_csr(const unsigned char* gate,const unsigned char* up,DeviceRows view,DeviceCsr csr,
    unsigned hidden,unsigned outputs) {
    if(csr.summary[0]||blockIdx.y>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.y],end=csr.begins[blockIdx.y+1U],expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) signed char local_codes[Rows][max_hidden];
    __shared__ __align__(16) float local_deltas[Rows][max_hidden/32U];
    __shared__ __align__(16) int local_sums[Rows][max_hidden/32U];
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(begin+a<end) {
        const auto r=csr.rows[begin+a];
        for(unsigned i=threadIdx.x;i<hidden;i+=256U)local_codes[a][i]=view.codes[r][i];
        for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U){local_deltas[a][i]=view.deltas[r][i];local_sums[a][i]=view.sums[r][i];}
    }
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U;
    const unsigned out=blockIdx.x*32U+(threadIdx.x/32U)*4U+lane/8U,blocks=hidden/256U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*blocks*144U;
    const unsigned g0=2U*p,g1=g0+1U;
    float acc_gate[Rows]={},acc_up[Rows]={};
#pragma unroll 1
    for(unsigned block=0;block<blocks;++block) {
        // Packed bytes, factors and half headers are decoded once per tile.
        const auto gate_coefficient=coefficient(gate+offset+block*144U,c,p);
        const auto up_coefficient=coefficient(up+offset+block*144U,c,p);
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(begin+a<end) {
            const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+block*256U+p*64U+h*16U);
            const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+block*256U+p*64U+32U+h*16U);
            const int s0=local_sums[a][block*8U+g0],s1=local_sums[a][block*8U+g1];
            const float e0=local_deltas[a][block*8U+g0],e1=local_deltas[a][block*8U+g1];
            const float gate_pair=paired(gate_coefficient,a0,a1,s0,s1,e0,e1);
            acc_gate[a]=acc_gate[a]+gate_pair;
            const float up_pair=paired(up_coefficient,a0,a1,s0,s1,e0,e1);
            acc_up[a]=acc_up[a]+up_pair;
        }
    }
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(begin+a<end) {
        acc_gate[a]=acc_gate[a]+__shfl_xor(acc_gate[a],2,32);acc_gate[a]=acc_gate[a]+__shfl_xor(acc_gate[a],4,32);
        acc_up[a]=acc_up[a]+__shfl_xor(acc_up[a],2,32);acc_up[a]=acc_up[a]+__shfl_xor(acc_up[a],4,32);
        if(c==0U) {
            const auto r=csr.rows[begin+a],slot=csr.slots[begin+a];
            view.output[r][(slot*2U)*outputs+out]=acc_gate[a];view.output[r][(slot*2U+1U)*outputs+out]=acc_up[a];
        }
    }
}
__device__ bool nonfinite(float x){return (__float_as_uint(x)&0x7fffffffU)>=0x7f800000U;}
// Serial Q5_K A8G32 pair (project_q5k): unsigned five-bit codes from qs/qh,
// then the project/project_csr affine and pair order.
__device__ __forceinline__ float paired_q5k(uint4 hd,uint4 q,uint4 qh,unsigned p,int4 a0,int4 a1,int s0,int s1,float e0,float e1) {
    constexpr unsigned mask=0x0f0f0f0fU;
    const auto g0=2U*p,g1=g0+1U;
    const uint4 q0{(q.x&mask)|(((qh.x>>g0)&0x01010101U)<<4U),
        (q.y&mask)|(((qh.y>>g0)&0x01010101U)<<4U),
        (q.z&mask)|(((qh.z>>g0)&0x01010101U)<<4U),
        (q.w&mask)|(((qh.w>>g0)&0x01010101U)<<4U)};
    const uint4 q1{((q.x>>4U)&mask)|(((qh.x>>g1)&0x01010101U)<<4U),
        ((q.y>>4U)&mask)|(((qh.y>>g1)&0x01010101U)<<4U),
        ((q.z>>4U)&mask)|(((qh.z>>g1)&0x01010101U)<<4U),
        ((q.w>>4U)&mask)|(((qh.w>>g1)&0x01010101U)<<4U)};
    int r0=dot(a0.x,q0.x,0);r0=dot(a0.y,q0.y,r0);r0=dot(a0.z,q0.z,r0);r0=dot(a0.w,q0.w,r0);
    int r1=dot(a1.x,q1.x,0);r1=dot(a1.y,q1.y,r1);r1=dot(a1.z,q1.z,r1);r1=dot(a1.w,q1.w,r1);
    r0+=__shfl_xor(r0,1,32);r1+=__shfl_xor(r1,1,32);
    const int i0=static_cast<int>(factor(hd,g0,false))*r0,i1=static_cast<int>(factor(hd,g1,false))*r1;
    const int j0=static_cast<int>(factor(hd,g0,true))*s0;
    const int j1=static_cast<int>(factor(hd,g1,true))*s1;
    const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(hd.x&65535U)));
    const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(hd.x>>16U)));
    const float u0=d*static_cast<float>(i0),v0=dm*static_cast<float>(j0),t0=u0-v0;
    const float u1=d*static_cast<float>(i1),v1=dm*static_cast<float>(j1),t1=u1-v1;
    const float c0=e0*t0,c1=e1*t1;
    return c0+c1;
}
__device__ float divide_rne(float a,float b){return a/b;}
// project_csr arithmetic (Q4_K, or the serial Q5_K pair) with every K block of
// a Q4_K matrix loaded before its chains (each wave fetches four complete
// packed rows at once). Batch rows are staged G at a time; a batch wider than
// G re-reads its weight slice from cache per group. Epilogue per
// (row,slot,output): the float profile's finite scan/SiLU/scan, or (A8Down)
// the serial A8 down activation quantizer on the block's 32-output group.
// Chunk: Q4_K K blocks held in registers at once (fewer lets more waves stay resident;
// each accumulator still adds its blocks in ascending order).
template<unsigned G,bool Q5K,bool A8Down,unsigned Chunk=max_hidden/256U>
__global__ __launch_bounds__(256) void project_window(const unsigned char* gate,const unsigned char* up,DeviceRows view,DeviceCsr csr,
    unsigned rows,unsigned hidden,unsigned outputs) {
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if(failure||blockIdx.y>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.y],count=csr.begins[blockIdx.y+1U]-begin,expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) signed char local_codes[G][max_hidden];
    __shared__ __align__(16) float local_deltas[G][max_hidden/32U];
    __shared__ __align__(16) int local_sums[G][max_hidden/32U];
    __shared__ float tile[2][G][32];
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U,wave=threadIdx.x/32U;
    const unsigned out=blockIdx.x*32U+wave*4U+lane/8U,blocks=hidden/256U;
    constexpr unsigned block_bytes=Q5K?176U:144U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*blocks*block_bytes;
    const unsigned g0=2U*p,g1=g0+1U;
    constexpr unsigned max_blocks=max_hidden/256U;
    for(unsigned group=0;group<count;group+=G) {
        if(group)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
            const auto r=csr.rows[begin+group+a];
            for(unsigned i=16U*threadIdx.x;i<hidden;i+=4096U)
                *reinterpret_cast<int4*>(local_codes[a]+i)=*reinterpret_cast<const int4*>(view.codes[r]+i);
            for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U){local_deltas[a][i]=view.deltas[r][i];local_sums[a][i]=view.sums[r][i];}
        }
        __syncthreads();
        float acc[2][G]={};
#pragma unroll
        for(unsigned m=0;m<2U;++m) {
            const unsigned char* w=(m==0U?gate:up)+offset;
            if constexpr(Q5K) {
#pragma unroll 1
                for(unsigned b=0;b<blocks;++b) {
                    const auto* packed=w+b*176U;
                    const uint4 hd=*reinterpret_cast<const uint4*>(packed);
                    const uint4 q=*reinterpret_cast<const uint4*>(packed+48U+c*16U);
                    const uint4 qh=*reinterpret_cast<const uint4*>(packed+16U+h*16U);
#pragma unroll
                    for(unsigned a=0;a<G;++a)if(group+a<count) {
                        const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+h*16U);
                        const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+32U+h*16U);
                        const float pair=paired_q5k(hd,q,qh,p,a0,a1,local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],
                            local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                        acc[m][a]=acc[m][a]+pair;
                    }
                }
            } else {
#pragma unroll
                for(unsigned first=0;first<max_blocks;first+=Chunk) {
                uint4 hd[Chunk],q[Chunk];
#pragma unroll
                for(unsigned i=0;i<Chunk;++i)if(first+i<blocks) {
                    const unsigned b=first+i;
                    hd[i]=*reinterpret_cast<const uint4*>(w+b*144U);q[i]=*reinterpret_cast<const uint4*>(w+b*144U+16U+c*16U);
                }
#pragma unroll
                for(unsigned i=0;i<Chunk;++i)if(first+i<blocks) {
                    const unsigned b=first+i;
                    const auto k=coefficient(hd[i],q[i],p);
#pragma unroll
                    for(unsigned a=0;a<G;++a)if(group+a<count) {
                        const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+h*16U);
                        const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+32U+h*16U);
                        const float pair=paired(k,a0,a1,local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],
                            local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                        acc[m][a]=acc[m][a]+pair;
                    }
                }
                }
            }
        }
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
#pragma unroll
            for(unsigned m=0;m<2U;++m){acc[m][a]=acc[m][a]+__shfl_xor(acc[m][a],2,32);acc[m][a]=acc[m][a]+__shfl_xor(acc[m][a],4,32);}
            if constexpr(A8Down) {
                if(c==0U){tile[0][a][out%32U]=acc[0][a];tile[1][a][out%32U]=acc[1][a];}
            } else if(c==0U) {
                const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
                const float g=acc[0][a],u=acc[1][a],silu=(g/(1.0F+expf(-g)))*u;
                view.output[r][(slot*2U)*outputs+out]=silu;view.output[r][(slot*2U+1U)*outputs+out]=u;
                if(nonfinite(g)||nonfinite(u)||nonfinite(silu))atomicOr(view.status[r],8U);
            }
        }
        if constexpr(A8Down) {
            // Serial down_activation_quantize for group blockIdx.x of each staged
            // (row,slot): same lane = feature, collectives and fail-closed writes.
            // One wave per staged (row, slot): rows are independent, each keeps the serial quantizer.
            __syncthreads();
            {
                for(unsigned a=wave;a<G;a+=8U)if(group+a<count) {
                    const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
                    const unsigned feature=blockIdx.x*32U+lane,at=slot*2U*outputs+feature;
                    const float g=tile[0][a][lane],u=tile[1][a][lane];
                    const float hv=(g/(1.0F+expf(-g)))*u;
                    view.output[r][at+outputs]=u;
                    const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
                    if(bad!=0){view.output[r][at]=g;if(lane==0)atomicOr(view.status[r],8U);continue;}
                    float x=fabsf(hv);
                    for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,__shfl_xor(x,stride,32));
                    const bool tiny=x>0.0F && x<0x1p-119F;
                    const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
                    const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
                    const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
                    int sum=code;
                    for(int stride=16;stride!=0;stride/=2) sum+=__shfl_xor(sum,stride,32);
                    view.output[r][at]=hv;
                    view.down_codes[r][slot*outputs+feature]=static_cast<signed char>(code);
                    if(lane==0){view.down_deltas[r][slot*(outputs/32U)+blockIdx.x]=delta;view.down_sums[r][slot*(outputs/32U)+blockIdx.x]=sum;}
                }
            }
        }
    }
}
__device__ unsigned down_u16(const unsigned char* p) {
    return static_cast<unsigned>(p[0])|(static_cast<unsigned>(p[1])<<8U);
}
__device__ unsigned down_u32(const unsigned char* p) {return down_u16(p)|(down_u16(p+2U)<<16U);}
__device__ unsigned down_q51_word(const unsigned char* block,unsigned first) {
    const unsigned high=down_u32(block+4U);
    unsigned result=0;
#pragma unroll
    for(unsigned j=0;j<4U;++j) {
        const unsigned i=first+j;
        const unsigned packed=block[8U+i%16U];
        const unsigned low=(i<16U?packed:packed>>4U)&15U;
        const unsigned q=low|(((high>>i)&1U)<<4U);
        result|=q<<(8U*j);
    }
    return result;
}
// Serial down_rawdots (Q5_1 s8xu5 / Q8_0 s8xs8 A8G32) per CSR batch: each
// lane decodes its block words once and runs every staged row's chain with
// the serial lane/group/tree order. Writes the incumbent raw slot dots. A
// failing block (0,0) folds all row statuses into the summary.
template<unsigned Rows,bool Q51>
__global__ __launch_bounds__(256) void down_window(const unsigned char* weights,DeviceRows view,DeviceCsr csr,
    unsigned rows,unsigned hidden,unsigned width) {
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if(failure){if(blockIdx.x==0U&&blockIdx.y==0U&&threadIdx.x==0U)csr.summary[0]=failure;return;}
    if(blockIdx.y>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.y],count=csr.begins[blockIdx.y+1U]-begin,expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) signed char codes[Rows][max_rows];
    __shared__ float deltas[Rows][max_rows/32U];
    __shared__ int sums[Rows][max_rows/32U];
    const unsigned groups=width/32U;
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(a<count) {
        const auto r=csr.rows[begin+a],slot=csr.slots[begin+a];
        for(unsigned i=16U*threadIdx.x;i<width;i+=4096U)
            *reinterpret_cast<int4*>(codes[a]+i)=*reinterpret_cast<const int4*>(view.down_codes[r]+slot*width+i);
        for(unsigned i=threadIdx.x;i<groups;i+=256U){deltas[a][i]=view.down_deltas[r][slot*groups+i];sums[a][i]=view.down_sums[r][slot*groups+i];}
    }
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=blockIdx.x*64U+(threadIdx.x/32U)*8U+lane/4U;
    constexpr unsigned block_bytes=Q51?24U:34U;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*groups*block_bytes;
    float acc[Rows]={};
#pragma unroll 1
    for(unsigned group=owner;group<groups;group+=4U) {
        const auto* packed=row_weights+group*block_bytes;
        unsigned words[8];
#pragma unroll
        for(unsigned word=0;word<8U;++word)words[word]=Q51?down_q51_word(packed,word*4U):down_u32(packed+2U+word*4U);
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(down_u16(packed))));
        const float m=Q51?__half2float(__ushort_as_half(static_cast<unsigned short>(down_u16(packed+2U)))):0.0F;
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(a<count) {
            const auto* image=reinterpret_cast<const int*>(codes[a]+group*32U);
            int integer=0;
#pragma unroll
            for(unsigned word=0;word<8U;++word) {
                if constexpr(Q51) integer=dot(image[word],words[word],integer);
                else integer=__builtin_amdgcn_sudot4(true,image[word],true,static_cast<int>(words[word]),integer,false);
            }
            const float first=d*static_cast<float>(integer);
            float combined=first;
            if constexpr(Q51) {
                const float correction=m*static_cast<float>(sums[a][group]);
                combined=first+correction;
            }
            const float contribution=deltas[a][group]*combined;
            acc[a]=acc[a]+contribution;
        }
    }
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(a<count) {
        acc[a]=acc[a]+__shfl_xor(acc[a],1,32);
        acc[a]=acc[a]+__shfl_xor(acc[a],2,32);
        if(owner==0U)view.rawdots[csr.rows[begin+a]][csr.slots[begin+a]*hidden+row]=acc[a];
    }
}
template<class T> __device__ __forceinline__ T* at(StridedRows<T> v,unsigned r) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(v.base)+static_cast<unsigned long long>(r)*v.stride);
}
// Prefill CSR over T*top_k assignments, one 1024-thread CTA: counting sort by
// expert; assignment a=r*top_k+s goes to offsets[e]+#{a'<a with expert e}, so
// each expert's rows (then slots) ascend. Batches are runs of at most `cap`
// assignments of one expert. Runs after the prologue and folds every row
// status into summary[0], so the projection kernels read one word.
__global__ __launch_bounds__(1024) void build_prefill_csr(PrefillRows v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap) {
    const auto& csr=v.csr;
    const unsigned t=threadIdx.x,count=rows*top_k;
    __shared__ unsigned short expert_of[q51_known_rows::prefill_max_rows*10U];
    __shared__ unsigned n_of[512],scan[3][512],hist[16],failure,status,duplicates,maximum;
    if(t==0){failure=0U;status=0U;duplicates=0U;maximum=0U;}
    if(t<16U)hist[t]=0U;
    if(t<512U)n_of[t]=0U;
    __syncthreads();
    for(unsigned a=t;a<count;a+=1024U) {
        const unsigned r=a/top_k,s=a%top_k,e=at(v.routes,r)[s];
        if(e>=experts||!isfinite(at(v.route_weights,r)[s]))atomicOr(&failure,invalid_route);
        expert_of[a]=static_cast<unsigned short>(e<experts?e:0U);
    }
    for(unsigned r=t;r<rows;r+=1024U){const unsigned s=*at(v.status,r);if(s)atomicOr(&status,s);}
    __syncthreads();
    if(failure){if(t<32U)csr.summary[t]=t==0U?failure|status:t==3U?count:t==6U?rows:t==7U?1U:0U;return;}
    for(unsigned a=t;a<count;a+=1024U)atomicAdd(&n_of[expert_of[a]],1U);
    __syncthreads();
    const unsigned n=t<512U?n_of[t]:0U,nb=(n+cap-1U)/cap;
    if(t<512U){scan[0][t]=n;scan[1][t]=n!=0U;scan[2][t]=nb;}
    __syncthreads();
    for(unsigned d=1U;d<512U;d<<=1U) {
        unsigned x[3]={};
        if(t>=d&&t<512U)for(unsigned k=0;k<3U;++k)x[k]=scan[k][t-d];
        __syncthreads();
        if(t>=d&&t<512U)for(unsigned k=0;k<3U;++k)scan[k][t]+=x[k];
        __syncthreads();
    }
    if(t<experts) {
        const unsigned off=scan[0][t]-n,touched=scan[1][t]-(n!=0U),batch=scan[2][t]-nb;
        csr.offsets[t]=off;
        if(t==experts-1U)csr.offsets[experts]=off+n;
        if(n) {
            csr.touched[touched]=t;atomicMax(&maximum,n);
            for(unsigned b=0;b<nb;++b){csr.experts[batch+b]=t;csr.begins[batch+b]=off+b*cap;atomicAdd(&hist[8U+min(8U,min(cap,n-b*cap))-1U],1U);}
            unsigned k=0U,unique=0U,last=UINT32_MAX;
            for(unsigned a=0;a<count&&k<n;++a)if(expert_of[a]==t) {
                const unsigned r=a/top_k;
                csr.rows[off+k]=r;csr.slots[off+k]=a%top_k;++k;
                unique+=r!=last;last=r;
            }
            atomicAdd(&hist[min(8U,unique)-1U],1U);atomicAdd(&duplicates,n-unique);
        }
    }
    __syncthreads();
    if(t==0U)csr.begins[scan[2][511]]=count;
    if(t<32U)csr.summary[t]=t==0U?status:t==1U?scan[1][511]:t==2U?scan[2][511]:t==3U?count:t==4U?duplicates:t==5U?maximum:
        t==6U?rows:t==7U?1U:t>=8U&&t<24U?hist[t-8U]:0U;
}
// build_prefill_csr with the same outputs (every array and summary word) in parallel: wave w
// owns the assignments [w*S, (w+1)*S) and counts its experts (two 16-bit counters per word);
// exclusive prefixes over the waves plus the expert offsets give each wave's first position per
// expert; then each wave walks its range in order, 32 assignments at a time, ranking lanes of one
// expert by lane order, so each expert's assignments ascend exactly as in the serial scan.
__global__ __launch_bounds__(1024) void build_prefill_csr_waves(PrefillRows v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap) {
    constexpr unsigned Waves=32U;
    const auto& csr=v.csr;
    const unsigned t=threadIdx.x,lane=t&31U,wave=t/32U,count=rows*top_k,span=(count+Waves-1U)/Waves;
    const unsigned first=wave*span,last=min(count,first+span);
    __shared__ unsigned position[Waves][256],n_of[512],scan[3][512],hist[16],failure,status,duplicates,maximum;
    if(t==0){failure=0U;status=0U;duplicates=0U;maximum=0U;}
    if(t<16U)hist[t]=0U;
    for(unsigned k=t;k<Waves*256U;k+=1024U)position[k/256U][k%256U]=0U;
    __syncthreads();
    const auto expert_at=[&](unsigned a){return at(v.routes,a/top_k)[a%top_k];};
    for(unsigned a=first+lane;a<last;a+=32U) {
        const unsigned e=expert_at(a);
        if(e>=experts||!isfinite(at(v.route_weights,a/top_k)[a%top_k]))atomicOr(&failure,invalid_route);
        else atomicAdd(&position[wave][e/2U],1U<<(16U*(e&1U)));
    }
    for(unsigned r=t;r<rows;r+=1024U){const unsigned s=*at(v.status,r);if(s)atomicOr(&status,s);}
    __syncthreads();
    if(failure){if(t<32U)csr.summary[t]=t==0U?failure|status:t==3U?count:t==6U?rows:t==7U?1U:0U;return;}
    if(t<256U) {
        unsigned run=0U;
        for(unsigned w=0;w<Waves;++w){const unsigned c=position[w][t];position[w][t]=run;run+=c;}
        n_of[2U*t]=run&65535U;n_of[2U*t+1U]=run>>16U;
    }
    __syncthreads();
    const unsigned n=t<512U?n_of[t]:0U,nb=(n+cap-1U)/cap;
    if(t<512U){scan[0][t]=n;scan[1][t]=n!=0U;scan[2][t]=nb;}
    __syncthreads();
    for(unsigned d=1U;d<512U;d<<=1U) {
        unsigned x[3]={};
        if(t>=d&&t<512U)for(unsigned k=0;k<3U;++k)x[k]=scan[k][t-d];
        __syncthreads();
        if(t>=d&&t<512U)for(unsigned k=0;k<3U;++k)scan[k][t]+=x[k];
        __syncthreads();
    }
    if(t<256U) {
        const unsigned base=(scan[0][2U*t]-n_of[2U*t])|(scan[0][2U*t+1U]-n_of[2U*t+1U])<<16U;
        for(unsigned w=0;w<Waves;++w)position[w][t]+=base;
    }
    __syncthreads();
    for(unsigned step=first;step<last;step+=32U) {
        const unsigned a=step+lane;
        const bool valid=a<last;
        const unsigned e=valid?expert_at(a):0U;
        unsigned active=static_cast<unsigned>(__ballot(valid)),peers=0U;
        while(active) {
            const unsigned leader_e=static_cast<unsigned>(__shfl(static_cast<int>(e),__ffs(active)-1,32));
            const unsigned same=static_cast<unsigned>(__ballot(valid&&e==leader_e))&active;
            if(valid&&e==leader_e)peers=same;
            active&=~same;
        }
        if(valid) {
            const unsigned slot=(position[wave][e/2U]>>(16U*(e&1U)))&65535U;
            const unsigned at_pos=slot+static_cast<unsigned>(__popc(peers&((1U<<lane)-1U)));
            csr.rows[at_pos]=a/top_k;csr.slots[at_pos]=a%top_k;
            if((peers&((1U<<lane)-1U))==0U)atomicAdd(&position[wave][e/2U],static_cast<unsigned>(__popc(peers))<<(16U*(e&1U)));
        }
    }
    __syncthreads();
    if(t<experts) {
        const unsigned off=scan[0][t]-n,touched=scan[1][t]-(n!=0U),batch=scan[2][t]-nb;
        csr.offsets[t]=off;
        if(t==experts-1U)csr.offsets[experts]=off+n;
        if(n) {
            csr.touched[touched]=t;atomicMax(&maximum,n);
            for(unsigned b=0;b<nb;++b){csr.experts[batch+b]=t;csr.begins[batch+b]=off+b*cap;atomicAdd(&hist[8U+min(8U,min(cap,n-b*cap))-1U],1U);}
            unsigned unique=0U,previous=UINT32_MAX;
            for(unsigned k=0;k<n;++k){const unsigned r=csr.rows[off+k];unique+=r!=previous;previous=r;}
            atomicAdd(&hist[min(8U,unique)-1U],1U);atomicAdd(&duplicates,n-unique);
        }
    }
    __syncthreads();
    if(t==0U)csr.begins[scan[2][511]]=count;
    if(t<32U)csr.summary[t]=t==0U?status:t==1U?scan[1][511]:t==2U?scan[2][511]:t==3U?count:t==4U?duplicates:t==5U?maximum:
        t==6U?rows:t==7U?1U:t>=8U&&t<24U?hist[t-8U]:0U;
}
// project_window over strided prefill rows: same per-(row,slot) lane arithmetic,
// staging, chains and A8 down quantizer epilogue. Grid x = CSR batch (runs of one
// expert are adjacent, so their weight slices hit L2), y = 32-output tile.
template<unsigned G,bool Q5K,unsigned Chunk=max_hidden/256U>
__global__ __launch_bounds__(256) void project_prefill(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs) {
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.x>=csr.summary[2])return;
    const unsigned tile_index=blockIdx.y;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ __align__(16) signed char local_codes[G][max_hidden];
    __shared__ __align__(16) float local_deltas[G][max_hidden/32U];
    __shared__ __align__(16) int local_sums[G][max_hidden/32U];
    __shared__ float tile[2][G][32];
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U,wave=threadIdx.x/32U;
    const unsigned out=tile_index*32U+wave*4U+lane/8U,blocks=hidden/256U;
    constexpr unsigned block_bytes=Q5K?176U:144U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*blocks*block_bytes;
    const unsigned g0=2U*p,g1=g0+1U;
    constexpr unsigned max_blocks=max_hidden/256U;
    for(unsigned group=0;group<count;group+=G) {
        if(group)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
            const auto r=csr.rows[begin+group+a];
            const auto* codes=at(view.codes,r);const auto* deltas=at(view.deltas,r);const auto* sums=at(view.sums,r);
            for(unsigned i=16U*threadIdx.x;i<hidden;i+=4096U)
                *reinterpret_cast<int4*>(local_codes[a]+i)=*reinterpret_cast<const int4*>(codes+i);
            for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U){local_deltas[a][i]=deltas[i];local_sums[a][i]=sums[i];}
        }
        __syncthreads();
        float acc[2][G]={};
#pragma unroll
        for(unsigned m=0;m<2U;++m) {
            const unsigned char* w=(m==0U?gate:up)+offset;
            if constexpr(Q5K) {
#pragma unroll 1
                for(unsigned b=0;b<blocks;++b) {
                    const auto* packed=w+b*176U;
                    const uint4 hd=*reinterpret_cast<const uint4*>(packed);
                    const uint4 q=*reinterpret_cast<const uint4*>(packed+48U+c*16U);
                    const uint4 qh=*reinterpret_cast<const uint4*>(packed+16U+h*16U);
#pragma unroll
                    for(unsigned a=0;a<G;++a)if(group+a<count) {
                        const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+h*16U);
                        const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+32U+h*16U);
                        const float pair=paired_q5k(hd,q,qh,p,a0,a1,local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],
                            local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                        acc[m][a]=acc[m][a]+pair;
                    }
                }
            } else {
#pragma unroll
                for(unsigned first=0;first<max_blocks;first+=Chunk) {
                uint4 hd[Chunk],q[Chunk];
#pragma unroll
                for(unsigned i=0;i<Chunk;++i)if(first+i<blocks) {
                    const unsigned b=first+i;
                    hd[i]=*reinterpret_cast<const uint4*>(w+b*144U);q[i]=*reinterpret_cast<const uint4*>(w+b*144U+16U+c*16U);
                }
#pragma unroll
                for(unsigned i=0;i<Chunk;++i)if(first+i<blocks) {
                    const unsigned b=first+i;
                    const auto k=coefficient(hd[i],q[i],p);
#pragma unroll
                    for(unsigned a=0;a<G;++a)if(group+a<count) {
                        const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+h*16U);
                        const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+32U+h*16U);
                        const float pair=paired(k,a0,a1,local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],
                            local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                        acc[m][a]=acc[m][a]+pair;
                    }
                }
                }
            }
        }
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
#pragma unroll
            for(unsigned m=0;m<2U;++m){acc[m][a]=acc[m][a]+__shfl_xor(acc[m][a],2,32);acc[m][a]=acc[m][a]+__shfl_xor(acc[m][a],4,32);}
            if(c==0U){tile[0][a][out%32U]=acc[0][a];tile[1][a][out%32U]=acc[1][a];}
        }
        // Serial down_activation_quantize for group tile_index of each staged (row, slot).
        __syncthreads();
        for(unsigned a=wave;a<G;a+=8U)if(group+a<count) {
            const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
            float* output=at(view.output,r);
            const unsigned feature=tile_index*32U+lane,at_slot=slot*2U*outputs+feature;
            const float g=tile[0][a][lane],u=tile[1][a][lane];
            const float hv=(g/(1.0F+expf(-g)))*u;
            output[at_slot+outputs]=u;
            const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
            if(bad!=0){output[at_slot]=g;if(lane==0){atomicOr(at(view.status,r),8U);atomicOr(csr.summary,8U);}continue;}
            float x=fabsf(hv);
            for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,__shfl_xor(x,stride,32));
            const bool tiny=x>0.0F && x<0x1p-119F;
            const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
            const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
            const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
            int sum=code;
            for(int stride=16;stride!=0;stride/=2) sum+=__shfl_xor(sum,stride,32);
            output[at_slot]=hv;
            at(view.down_codes,r)[slot*outputs+feature]=static_cast<signed char>(code);
            if(lane==0){at(view.down_deltas,r)[slot*(outputs/32U)+tile_index]=delta;at(view.down_sums,r)[slot*(outputs/32U)+tile_index]=sum;}
        }
    }
}
// down_window over strided prefill rows, Rows staged rows at a time: same lane
// chains, block decode and reduction per (row, slot). Grid x = batch, y = 64-row tile.
template<unsigned Rows,bool Q51>
__global__ __launch_bounds__(256) void down_prefill(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.x>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ __align__(16) signed char codes[Rows][max_rows];
    __shared__ float deltas[Rows][max_rows/32U];
    __shared__ int sums[Rows][max_rows/32U];
    const unsigned groups=width/32U;
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=blockIdx.y*64U+(threadIdx.x/32U)*8U+lane/4U;
    constexpr unsigned block_bytes=Q51?24U:34U;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*groups*block_bytes;
    for(unsigned first=0;first<count;first+=Rows) {
        if(first)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(first+a<count) {
            const auto r=csr.rows[begin+first+a],slot=csr.slots[begin+first+a];
            const auto* down_codes=at(view.down_codes,r)+slot*width;
            const auto* down_deltas=at(view.down_deltas,r)+slot*groups;const auto* down_sums=at(view.down_sums,r)+slot*groups;
            for(unsigned i=16U*threadIdx.x;i<width;i+=4096U)
                *reinterpret_cast<int4*>(codes[a]+i)=*reinterpret_cast<const int4*>(down_codes+i);
            for(unsigned i=threadIdx.x;i<groups;i+=256U){deltas[a][i]=down_deltas[i];sums[a][i]=down_sums[i];}
        }
        __syncthreads();
        float acc[Rows]={};
#pragma unroll 1
        for(unsigned group=owner;group<groups;group+=4U) {
            const auto* packed=row_weights+group*block_bytes;
            unsigned words[8];
#pragma unroll
            for(unsigned word=0;word<8U;++word)words[word]=Q51?down_q51_word(packed,word*4U):down_u32(packed+2U+word*4U);
            const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(down_u16(packed))));
            const float m=Q51?__half2float(__ushort_as_half(static_cast<unsigned short>(down_u16(packed+2U)))):0.0F;
#pragma unroll
            for(unsigned a=0;a<Rows;++a)if(first+a<count) {
                const auto* image=reinterpret_cast<const int*>(codes[a]+group*32U);
                int integer=0;
#pragma unroll
                for(unsigned word=0;word<8U;++word) {
                    if constexpr(Q51) integer=dot(image[word],words[word],integer);
                    else integer=__builtin_amdgcn_sudot4(true,image[word],true,static_cast<int>(words[word]),integer,false);
                }
                const float first_term=d*static_cast<float>(integer);
                float combined=first_term;
                if constexpr(Q51) {
                    const float correction=m*static_cast<float>(sums[a][group]);
                    combined=first_term+correction;
                }
                const float contribution=deltas[a][group]*combined;
                acc[a]=acc[a]+contribution;
            }
        }
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(first+a<count) {
            acc[a]=acc[a]+__shfl_xor(acc[a],1,32);
            acc[a]=acc[a]+__shfl_xor(acc[a],2,32);
            if(owner==0U)at(view.rawdots,csr.rows[begin+first+a])[csr.slots[begin+first+a]*hidden+row]=acc[a];
        }
    }
}
// iu8 WMMA forms of project_prefill/down_prefill, bit-identical to them. A CSR batch holds at
// most 16 assignments of one expert: the M = 16 rows of v_wmma_i32_16x16x16_iu8 (rows past the
// batch are zero codes, their D elements are never stored and skip the epilogue in pairs). Per
// 32-value group, two WMMAs (K = 16 + 16) on an accumulator of 0x4b400000 give the exact int32
// group dot r of every (row, output) as the bits of 1.5*2^23 + r (|r| < 2^22), so one exact
// FP32 subtraction yields float(r); the epilogue then runs the incumbent per-lane expressions
// on (row, output) elements: gate/up u = d*float(sc*r) with float(sc*r) = float(sc)*float(r)
// (exact, |sc*r| < 2^24), v = dm*float(m*s) likewise, (e0*(u0-v0) + e1*(u1-v1)) added to the
// accumulator of pair p in ascending K blocks, and the lanes' tree (acc0+acc1)+(acc2+acc3);
// down d*float(r) (+ m*float(sum)), times the delta, added to the accumulator of group%4 in
// ascending groups, same tree. Q4_K/Q5_K nibbles (plus the Q5 high bit) and Q5_1 u5 codes are
// unsigned 0..31, Q8_0 signed: all exact as signed int8 operands. Layouts (wave32): A lane
// i%16 = row i, B lane i%16 = output i, D element e of lane l = (row 2e + l/16, output l%16);
// the upper half wave repeats A/B. The epilogue quantizer and the stores are project_prefill's.
typedef int wmma_v4i __attribute__((ext_vector_type(4)));
typedef int wmma_v8i __attribute__((ext_vector_type(8)));
struct __attribute__((aligned(4))) Dword4 { unsigned x,y,z,w; };
constexpr unsigned wmma_batch=16U;
constexpr int wmma_magic=0x4b400000;
__device__ __forceinline__ wmma_v4i operand(uint4 v) {
    return wmma_v4i{static_cast<int>(v.x),static_cast<int>(v.y),static_cast<int>(v.z),static_cast<int>(v.w)};
}
__device__ __forceinline__ wmma_v8i group_dot(wmma_v4i a0,wmma_v4i b0,wmma_v4i a1,wmma_v4i b1) {
    const wmma_v8i magic{wmma_magic,wmma_magic,wmma_magic,wmma_magic,wmma_magic,wmma_magic,wmma_magic,wmma_magic};
    const wmma_v8i c=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a0,true,b0,magic,false);
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a1,true,b1,c,false);
}
__device__ __forceinline__ float group_float(int bits){return __int_as_float(bits)-0x1.8p23F;}
// Row r of a batch at slot (r&1)*8 + r/2, so D element e of half h reads slot 8h + e.
__device__ __forceinline__ unsigned wmma_row_slot(unsigned r){return (r&1U)<<3U|r>>1U;}
template<unsigned Groups> struct WmmaRowsStage {
    wmma_v4i a[2U*Groups][wmma_batch]; // 16-value plane k of every row: codes [16k, 16k+16)
    float delta[Groups][wmma_batch];
    float sum[Groups][wmma_batch];  // float(int sum), exact
};
// Batch rows (row, slot) and their codes/deltas/sums into a stage; rows >= count are zero.
template<unsigned Groups,class Codes,class Deltas,class Sums>
__device__ __forceinline__ void wmma_stage_rows(WmmaRowsStage<Groups>& s,unsigned count,unsigned width,unsigned threads,
    const Codes& codes,const Deltas& deltas,const Sums& sums) {
    const unsigned planes=width/16U,groups=width/32U;
    for(unsigned item=threadIdx.x;item<wmma_batch*planes;item+=threads) {
        const unsigned a=item/planes,k=item%planes;
        wmma_v4i v{0,0,0,0};
        if(a<count)v=*reinterpret_cast<const wmma_v4i*>(codes(a)+16U*k);
        s.a[k][a]=v;
    }
    for(unsigned item=threadIdx.x;item<wmma_batch*groups;item+=threads) {
        const unsigned a=item/groups,g=item%groups,slot=wmma_row_slot(a);
        float e=0.0F,f=0.0F;
        if(a<count){e=deltas(a)[g];f=static_cast<float>(sums(a)[g]);}
        s.delta[g][slot]=e;s.sum[g][slot]=f;
    }
}
// x[e] = p[e], e < V (two 16-byte reads for V = 8).
template<unsigned V>
__device__ __forceinline__ void row_values(const float* p,float (&x)[8]) {
    const float4 lo=reinterpret_cast<const float4*>(p)[0];
    x[0]=lo.x;x[1]=lo.y;
    if constexpr(V>2U){x[2]=lo.z;x[3]=lo.w;}
    if constexpr(V>4U){const float4 hi=reinterpret_cast<const float4*>(p)[1];x[4]=hi.x;x[5]=hi.y;x[6]=hi.z;x[7]=hi.w;}
}
// The (row, output) elements e < V of one wave: four chain accumulators, folded at the end as the
// lanes' xor-2/xor-4 (gate/up) or xor-1/xor-2 (down) tree: (acc0+acc1)+(acc2+acc3).
// FP32 value of an fp16 scale (exact) behind an empty asm, so its products stay v_mul_f32 (the
// compiler would otherwise fold the conversion into every product as v_fma_mix_f32 with a -0
// addend: the same value, but not dual-issuable).
__device__ __forceinline__ float scale_f32(unsigned short bits) {
    float x=__half2float(__ushort_as_half(bits));
    __asm__("" : "+v"(x));
    return x;
}
template<unsigned V> struct Chains {
    float acc[4][8];
    __device__ Chains(){
#pragma unroll
        for(unsigned c=0;c<4U;++c)
#pragma unroll
            for(unsigned e=0;e<8U;++e)acc[c][e]=0.0F;
    }
    __device__ void fold(float (&out)[8]) const {
#pragma unroll
        for(unsigned e=0;e<V;++e){const float s01=acc[0][e]+acc[1][e],s23=acc[2][e]+acc[3][e];out[e]=s01+s23;}
    }
};
// One matrix (gate or up) of a wave: K blocks ascending, the four pairs p of a block in order,
// two block buffers in turn (the next block loads while one computes). Elements e < V only
// (rows 2e, 2e+1 of the batch); the WMMAs always cover all 16 rows.
template<bool Q5K,unsigned V,unsigned Groups>
__device__ __forceinline__ void gate_up_chains(const unsigned char* w,const WmmaRowsStage<Groups>& s,unsigned blocks,unsigned i,unsigned h,
    float (&out)[8]) {
    constexpr unsigned block_bytes=Q5K?176U:144U,qs_at=Q5K?48U:16U,mask=0x0f0f0f0fU;
    struct Block { uint4 hd,q[8],qh[2]; };
    const auto load=[&](unsigned b,Block& x) {
        const auto* packed=w+b*block_bytes;
        x.hd=*reinterpret_cast<const uint4*>(packed);
#pragma unroll
        for(unsigned k=0;k<8U;++k)x.q[k]=*reinterpret_cast<const uint4*>(packed+qs_at+16U*k);
        if constexpr(Q5K){x.qh[0]=*reinterpret_cast<const uint4*>(packed+16U);x.qh[1]=*reinterpret_cast<const uint4*>(packed+32U);}
    };
    Chains<V> chains;
    const auto step=[&](unsigned b,const Block& cur) {
        const float d=scale_f32(static_cast<unsigned short>(cur.hd.x&65535U));
        const float dm=scale_f32(static_cast<unsigned short>(cur.hd.x>>16U));
#pragma unroll
        for(unsigned p=0;p<4U;++p) {
            const unsigned g0=2U*p,g1=g0+1U;
            uint4 w0[2],w1[2];
#pragma unroll
            for(unsigned k=0;k<2U;++k) {
                const uint4 q=cur.q[2U*p+k];
                w0[k]={q.x&mask,q.y&mask,q.z&mask,q.w&mask};
                w1[k]={(q.x>>4U)&mask,(q.y>>4U)&mask,(q.z>>4U)&mask,(q.w>>4U)&mask};
                if constexpr(Q5K) {
                    const uint4 qh=cur.qh[k];
                    w0[k].x|=((qh.x>>g0)&0x01010101U)<<4U;w0[k].y|=((qh.y>>g0)&0x01010101U)<<4U;
                    w0[k].z|=((qh.z>>g0)&0x01010101U)<<4U;w0[k].w|=((qh.w>>g0)&0x01010101U)<<4U;
                    w1[k].x|=((qh.x>>g1)&0x01010101U)<<4U;w1[k].y|=((qh.y>>g1)&0x01010101U)<<4U;
                    w1[k].z|=((qh.z>>g1)&0x01010101U)<<4U;w1[k].w|=((qh.w>>g1)&0x01010101U)<<4U;
                }
            }
            const unsigned plane=16U*b+4U*p;
            const wmma_v8i r0=group_dot(s.a[plane][i],operand(w0[0]),s.a[plane+1U][i],operand(w0[1]));
            const wmma_v8i r1=group_dot(s.a[plane+2U][i],operand(w1[0]),s.a[plane+3U][i],operand(w1[1]));
            float e0[8],e1[8],s0[8],s1[8];
            row_values<V>(&s.delta[8U*b+g0][8U*h],e0);row_values<V>(&s.delta[8U*b+g1][8U*h],e1);
            row_values<V>(&s.sum[8U*b+g0][8U*h],s0);row_values<V>(&s.sum[8U*b+g1][8U*h],s1);
            const float sc0=static_cast<float>(factor(cur.hd,g0,false)),sc1=static_cast<float>(factor(cur.hd,g1,false));
            const float mn0=static_cast<float>(factor(cur.hd,g0,true)),mn1=static_cast<float>(factor(cur.hd,g1,true));
#pragma unroll
            for(unsigned e=0;e<V;++e) {
                const float i0=sc0*group_float(r0[e]),i1=sc1*group_float(r1[e]);
                const float j0=mn0*s0[e],j1=mn1*s1[e];
                const float u0=d*i0,v0=dm*j0,t0=u0-v0;
                const float u1=d*i1,v1=dm*j1,t1=u1-v1;
                const float c0=e0[e]*t0,c1=e1[e]*t1;
                const float pair=c0+c1;
                chains.acc[p][e]=chains.acc[p][e]+pair;
            }
        }
    };
    // Two block buffers in turn: block b + 1 loads while block b computes.
    Block x[2];
    load(0U,x[0]);
#pragma unroll 1
    for(unsigned b=0;b<blocks;b+=2U) {
        if(b+1U<blocks)load(b+1U,x[1]);
        step(b,x[0]);
        if(b+1U<blocks) {
            if(b+2U<blocks)load(b+2U,x[0]);
            step(b+1U,x[1]);
        }
    }
    chains.fold(out);
}
// Gate/up of one CSR batch (<= 16 assignments) for Waves/2 16-output tiles: wave w runs matrix
// w%2 (gate, up) of tile w/2. Grid x = batch, y = output tile.
template<bool Q5K,unsigned Waves>
__global__ __launch_bounds__(32U*Waves) void project_prefill_wmma(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs) {
    constexpr unsigned Outs=8U*Waves,Groups=max_hidden/32U;
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.x>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ union { WmmaRowsStage<Groups> s; float tile[2][wmma_batch][Outs]; } lds;
    __shared__ unsigned row_of[wmma_batch],slot_of[wmma_batch];
    if(threadIdx.x<wmma_batch&&threadIdx.x<count){row_of[threadIdx.x]=csr.rows[begin+threadIdx.x];slot_of[threadIdx.x]=csr.slots[begin+threadIdx.x];}
    __syncthreads();
    wmma_stage_rows<Groups>(lds.s,count,hidden,32U*Waves,[&](unsigned a){return at(view.codes,row_of[a]);},
        [&](unsigned a){return at(view.deltas,row_of[a]);},[&](unsigned a){return at(view.sums,row_of[a]);});
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32U,i=lane&15U,h=lane>>4U,m=wave&1U,t=wave>>1U;
    const unsigned tile_base=blockIdx.y*Outs,out=tile_base+16U*t+i,blocks=hidden/256U,valid=(count+1U)/2U;
    constexpr unsigned block_bytes=Q5K?176U:144U;
    const auto* w=(m==0U?gate:up)+(static_cast<unsigned long long>(expert)*outputs+out)*blocks*block_bytes;
    float result[8]={};
    if(valid<=2U)gate_up_chains<Q5K,2U>(w,lds.s,blocks,i,h,result);
    else if(valid<=4U)gate_up_chains<Q5K,4U>(w,lds.s,blocks,i,h,result);
    else gate_up_chains<Q5K,8U>(w,lds.s,blocks,i,h,result);
    __syncthreads(); // every wave is done with the staged rows: the tile reuses them
#pragma unroll
    for(unsigned e=0;e<8U;++e)if(2U*e+h<count)lds.tile[m][2U*e+h][16U*t+i]=result[e];
    __syncthreads();
    // Serial down_activation_quantize (project_prefill's epilogue) for each staged (row, slot)
    // and 32-output group of the CTA, one wave per pair.
    constexpr unsigned Quads=Outs/32U;
    for(unsigned job=wave;job<count*Quads;job+=Waves) {
        const unsigned a=job/Quads,quad=job%Quads,tile_index=tile_base/32U+quad;
        const auto r=row_of[a],slot=slot_of[a];
        float* output=at(view.output,r);
        const unsigned feature=tile_index*32U+lane,at_slot=slot*2U*outputs+feature;
        const float g=lds.tile[0][a][32U*quad+lane],u=lds.tile[1][a][32U*quad+lane];
        const float hv=(g/(1.0F+expf(-g)))*u;
        output[at_slot+outputs]=u;
        const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
        if(bad!=0){output[at_slot]=g;if(lane==0){atomicOr(at(view.status,r),8U);atomicOr(csr.summary,8U);}continue;}
        float x=fabsf(hv);
        for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,__shfl_xor(x,stride,32));
        const bool tiny=x>0.0F && x<0x1p-119F;
        const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
        const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
        const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
        int sum=code;
        for(int stride=16;stride!=0;stride/=2) sum+=__shfl_xor(sum,stride,32);
        output[at_slot]=hv;
        at(view.down_codes,r)[slot*outputs+feature]=static_cast<signed char>(code);
        if(lane==0){at(view.down_deltas,r)[slot*(outputs/32U)+tile_index]=delta;at(view.down_sums,r)[slot*(outputs/32U)+tile_index]=sum;}
    }
}
// Raw bytes of one down block for this lane's output (Q5_1: d|m, qh, qs; Q8_0: nine dwords from
// the block's dword-aligned start: rows hold an even block count, so an even block starts at its
// scale and an odd one two bytes before it).
template<bool Q51> struct DownBlock;
template<> struct DownBlock<true> { uint2 head; Dword4 qs; };
template<> struct DownBlock<false> { Dword4 lo,hi; unsigned last; };
template<bool Q51> __device__ __forceinline__ DownBlock<Q51> down_block(const unsigned char* row_weights,unsigned group) {
    if constexpr(Q51) {
        const auto* block=row_weights+group*24U;
        return {*reinterpret_cast<const uint2*>(block),*reinterpret_cast<const Dword4*>(block+8U)};
    } else {
        const auto* p=row_weights+((34U*group)&~3U);
        return {reinterpret_cast<const Dword4*>(p)[0],reinterpret_cast<const Dword4*>(p)[1],reinterpret_cast<const unsigned*>(p)[8]};
    }
}
// WMMA B operands (K halves) and scales of a down block; Q5_1 value k = nibble | qh bit k << 4.
template<bool Q51>
__device__ __forceinline__ void down_operands(const DownBlock<Q51>& x,unsigned group,wmma_v4i& b0,wmma_v4i& b1,float& d,float& m) {
    if constexpr(Q51) {
        const unsigned q[4]{x.qs.x,x.qs.y,x.qs.z,x.qs.w};
        unsigned lo[4],hi[4];
#pragma unroll
        for(unsigned k=0;k<4U;++k) {
            const unsigned l=(x.head.y>>(4U*k))&15U,u=(x.head.y>>(16U+4U*k))&15U;
            lo[k]=(q[k]&0x0f0f0f0fU)|((l*0x00204081U)&0x01010101U)<<4U;
            hi[k]=((q[k]>>4U)&0x0f0f0f0fU)|((u*0x00204081U)&0x01010101U)<<4U;
        }
        b0=wmma_v4i{static_cast<int>(lo[0]),static_cast<int>(lo[1]),static_cast<int>(lo[2]),static_cast<int>(lo[3])};
        b1=wmma_v4i{static_cast<int>(hi[0]),static_cast<int>(hi[1]),static_cast<int>(hi[2]),static_cast<int>(hi[3])};
        d=scale_f32(static_cast<unsigned short>(x.head.x&65535U));
        m=scale_f32(static_cast<unsigned short>(x.head.x>>16U));
    } else {
        const unsigned odd=group&1U,select=odd?0x07060504U:0x05040302U;
        const unsigned q[9]{x.lo.x,x.lo.y,x.lo.z,x.lo.w,x.hi.x,x.hi.y,x.hi.z,x.hi.w,x.last};
        unsigned c[8];
#pragma unroll
        for(unsigned k=0;k<8U;++k)c[k]=__builtin_amdgcn_perm(q[k+1U],q[k],select);
        b0=wmma_v4i{static_cast<int>(c[0]),static_cast<int>(c[1]),static_cast<int>(c[2]),static_cast<int>(c[3])};
        b1=wmma_v4i{static_cast<int>(c[4]),static_cast<int>(c[5]),static_cast<int>(c[6]),static_cast<int>(c[7])};
        d=scale_f32(static_cast<unsigned short>((q[0]>>(odd?16U:0U))&65535U));
        m=0.0F;
    }
}
// Down chains of one wave for width = max_rows: groups ascending in phases of four (group
// 4k + j adds to chain j), two phase buffers in turn: the next phase loads while one computes.
template<bool Q51,unsigned V>
__device__ __forceinline__ void down_chains(const unsigned char* row_weights,const WmmaRowsStage<max_rows/32U>& s,unsigned i,unsigned h,
    float (&out)[8]) {
    constexpr unsigned Phases=max_rows/128U;
    struct Phase { DownBlock<Q51> block[4]; };
    const auto load=[&](unsigned phase,Phase& x) {
#pragma unroll
        for(unsigned j=0;j<4U;++j)x.block[j]=down_block<Q51>(row_weights,4U*phase+j);
    };
    Chains<V> chains;
    const auto step=[&](unsigned phase,const Phase& cur) {
#pragma unroll
        for(unsigned j=0;j<4U;++j) {
            const unsigned group=4U*phase+j;
            wmma_v4i b0,b1;float d,m;
            down_operands<Q51>(cur.block[j],j,b0,b1,d,m);
            const wmma_v8i r=group_dot(s.a[2U*group][i],b0,s.a[2U*group+1U][i],b1);
            float deltas[8],sums[8];
            row_values<V>(&s.delta[group][8U*h],deltas);
            if constexpr(Q51)row_values<V>(&s.sum[group][8U*h],sums);
#pragma unroll
            for(unsigned e=0;e<V;++e) {
                const float first_term=d*group_float(r[e]);
                float combined=first_term;
                if constexpr(Q51) {
                    const float correction=m*sums[e];
                    combined=first_term+correction;
                }
                const float contribution=deltas[e]*combined;
                chains.acc[j][e]=chains.acc[j][e]+contribution;
            }
        }
    };
    Phase x[2];
    load(0U,x[0]);
#pragma unroll 1
    for(unsigned phase=0;phase<Phases;phase+=2U) {
        if(phase+1U<Phases)load(phase+1U,x[1]);
        step(phase,x[0]);
        if(phase+1U<Phases) {
            if(phase+2U<Phases)load(phase+2U,x[0]);
            step(phase+1U,x[1]);
        }
    }
    chains.fold(out);
}
// Down of one CSR batch (<= 16 assignments) for Waves 16-row tiles of the hidden output, width =
// max_rows. Grid x = batch, y = row tile.
template<bool Q51,unsigned Waves>
__global__ __launch_bounds__(32U*Waves) void down_prefill_wmma(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    constexpr unsigned Outs=16U*Waves,Groups=max_rows/32U;
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.x>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ WmmaRowsStage<Groups> s;
    __shared__ unsigned row_of[wmma_batch],slot_of[wmma_batch];
    if(threadIdx.x<wmma_batch&&threadIdx.x<count){row_of[threadIdx.x]=csr.rows[begin+threadIdx.x];slot_of[threadIdx.x]=csr.slots[begin+threadIdx.x];}
    __syncthreads();
    const unsigned groups=width/32U;
    wmma_stage_rows<Groups>(s,count,width,32U*Waves,[&](unsigned a){return at(view.down_codes,row_of[a])+slot_of[a]*width;},
        [&](unsigned a){return at(view.down_deltas,row_of[a])+slot_of[a]*groups;},
        [&](unsigned a){return at(view.down_sums,row_of[a])+slot_of[a]*groups;});
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32U,i=lane&15U,h=lane>>4U,valid=(count+1U)/2U;
    constexpr unsigned block_bytes=Q51?24U:34U;
    const unsigned row=blockIdx.y*Outs+16U*wave+i;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*groups*block_bytes;
    float result[8]={};
    if(valid<=2U)down_chains<Q51,2U>(row_weights,s,i,h,result);
    else if(valid<=4U)down_chains<Q51,4U>(row_weights,s,i,h,result);
    else down_chains<Q51,8U>(row_weights,s,i,h,result);
#pragma unroll
    for(unsigned e=0;e<8U;++e)if(2U*e+h<count) {
        const unsigned a=2U*e+h;
        at(view.rawdots,row_of[a])[slot_of[a]*hidden+row]=result[e];
    }
}
// Fast prefill profile (fast-prefill-f16wmma-v1, form 3): not the A8 contract. Weights are
// dequantized inside the kernels to FP16 (the FP32 value d*sc*q - dm*m, Q5_1 d*q + m, Q8_0 d*q,
// rounded once to nearest), activations are FP16, and v_wmma_f32_16x16x16_f16 accumulates over
// the whole K in FP32, with no per-group epilogue. Activations carry a power-of-two scale per
// row that puts the row's largest magnitude in [2^13, 2^14): exact for every value that stays
// normal (29 binades below the maximum) and undone exactly on the FP32 sums. Input x: one scale
// per prompt row (f16_row_scales); h = silu(g)*u: gate/up writes it in FP32 and the largest |h|
// of each (row, slot), and down scales and rounds it while staging. A CSR batch holds at most 64
// assignments of one expert (M tile 64; waves whose 16 rows are all past the batch skip their
// WMMAs). Each (row, output) sums its K steps in the same order whatever batch it lands in, so
// results do not depend on the batch composition. A non-finite g, u or h fails the row closed
// (status 8), as the A8 quantizer does; down writes the raw slot dots of the unchanged merge.
// Scratch: the A8 views the profile does not use: sums[r][0] = the x scale exponent,
// down_sums[r][slot] = bits of max |h|, intermediates[r][slot*2*outputs + o] = h.
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
constexpr unsigned f16_batch=64U,f16_pitch=72U; // M tile rows; LDS row pitch in halves (K step 64 + 8)
__device__ __forceinline__ unsigned f16_pair(float a,float b) {
    const _Float16 x=static_cast<_Float16>(a),y=static_cast<_Float16>(b);
    return static_cast<unsigned>(__builtin_bit_cast(unsigned short,x))|static_cast<unsigned>(__builtin_bit_cast(unsigned short,y))<<16U;
}
// Exponent s with peak * 2^s in [2^13, 2^14) (0 for a zero or non-finite max), clamped so 2^s and
// 2^-s are normal FP32 values.
__device__ __forceinline__ int f16_exponent(float peak) {
    if(!(peak>0.0F)||!isfinite(peak))return 0;
    int e;(void)frexpf(peak,&e); // peak in [2^(e-1), 2^e)
    return min(120,max(-120,14-e));
}
// 8 BF16 (one uint4) times scale as 8 FP16.
__device__ __forceinline__ uint4 f16_from_bf16(uint4 v,float scale) {
    const auto lo=[&](unsigned w){return __uint_as_float(w<<16U)*scale;};
    const auto hi=[&](unsigned w){return __uint_as_float(w&0xffff0000U)*scale;};
    return {f16_pair(lo(v.x),hi(v.x)),f16_pair(lo(v.y),hi(v.y)),f16_pair(lo(v.z),hi(v.z)),f16_pair(lo(v.w),hi(v.w))};
}
// 8 FP32 (the bits in two uint4) times scale as 8 FP16.
__device__ __forceinline__ uint4 f16_from_f32(uint4 a,uint4 b,float scale) {
    const auto x=[&](unsigned w){return __uint_as_float(w)*scale;};
    return {f16_pair(x(a.x),x(a.y)),f16_pair(x(a.z),x(a.w)),f16_pair(x(b.x),x(b.y)),f16_pair(x(b.z),x(b.w))};
}
// The four unsigned bytes of w as a*byte + b (one FP32 rounding), rounded to two FP16 pairs.
__device__ __forceinline__ uint2 f16_affine_u8(unsigned w,float a,float b) {
    const float v0=fmaf(a,static_cast<float>(w&255U),b),v1=fmaf(a,static_cast<float>((w>>8U)&255U),b);
    const float v2=fmaf(a,static_cast<float>((w>>16U)&255U),b),v3=fmaf(a,static_cast<float>(w>>24U),b);
    return {f16_pair(v0,v1),f16_pair(v2,v3)};
}
// The four signed bytes of w times a, as two FP16 pairs.
__device__ __forceinline__ uint2 f16_scaled_s8(unsigned w,float a) {
    const int s=static_cast<int>(w);
    const float v0=a*static_cast<float>((s<<24)>>24),v1=a*static_cast<float>((s<<16)>>24);
    const float v2=a*static_cast<float>((s<<8)>>24),v3=a*static_cast<float>(s>>24);
    return {f16_pair(v0,v1),f16_pair(v2,v3)};
}
// One CTA per prompt row: the x scale exponent of the row and zeroed max |h| words. Rows that
// already failed (prologue status) are skipped by every later kernel through the summary.
__global__ __launch_bounds__(256) void f16_row_scales(PrefillRows view,unsigned hidden,unsigned top_k) {
    const unsigned r=blockIdx.x,t=threadIdx.x;
    const auto* x=reinterpret_cast<const unsigned short*>(at(view.input,r));
    float m=0.0F;
    for(unsigned k=t;k<hidden;k+=256U)m=fmaxf(m,fabsf(__uint_as_float(static_cast<unsigned>(x[k])<<16U)));
    for(int stride=16;stride!=0;stride/=2)m=fmaxf(m,__shfl_xor(m,stride,32));
    __shared__ float wave_max[8];
    if((t&31U)==0U)wave_max[t/32U]=m;
    __syncthreads();
    if(t==0U) {
        for(unsigned w=1;w<8U;++w)m=fmaxf(m,wave_max[w]);
        at(view.sums,r)[0]=f16_exponent(m);
    }
    if(t<top_k)at(view.down_sums,r)[t]=0;
}
// LDS-only barrier: the staged tiles are complete (lgkmcnt) but the next step's global loads stay
// in flight (__syncthreads would also wait for them, as its workgroup fence does).
__device__ __forceinline__ void f16_barrier(){__asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");}
// 16-byte global (not flat) load: flat loads also count in lgkmcnt and would end at f16_barrier.
typedef unsigned f16_u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 f16_load(const void* p) {
    const f16_u32x4 v=*reinterpret_cast<const __attribute__((address_space(1))) f16_u32x4*>(reinterpret_cast<std::uintptr_t>(p));
    return uint4{v.x,v.y,v.z,v.w};
}
// One K64 step of a CTA: s.a holds the batch rows (64 x K64), s.b 256 B columns (x K64). A wave
// owns two B columns tiles (cols[0], cols[1]) and every 16-row tile below mb: the B fragments are
// read once per K16 plane and reused by up to four row tiles. K16 planes ascend.
struct F16Stage { _Float16 a[f16_batch][f16_pitch]; _Float16 b[256][f16_pitch]; };
__device__ __forceinline__ void f16_step(const F16Stage& s,unsigned mb,const unsigned (&cols)[2],unsigned i,f32x8 (&acc)[4][2]) {
#pragma unroll
    for(unsigned kk=0;kk<4U;++kk) {
        const f16x16 b0=*reinterpret_cast<const f16x16*>(&s.b[cols[0]+i][16U*kk]);
        const f16x16 b1=*reinterpret_cast<const f16x16*>(&s.b[cols[1]+i][16U*kk]);
#pragma unroll
        for(unsigned mi=0;mi<4U;++mi)if(mi<mb) {
            const f16x16 a=*reinterpret_cast<const f16x16*>(&s.a[16U*mi+i][16U*kk]);
            acc[mi][0]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b0,acc[mi][0]);
            acc[mi][1]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b1,acc[mi][1]);
        }
    }
}
// Batch rows of a CTA into LDS (row, slot, scale exponent).
struct F16Rows { unsigned row[f16_batch],slot[f16_batch]; int exponent[f16_batch]; };
// Gate/up of one CSR batch (<= 64 assignments) for 128 outputs of both matrices. Grid x = output
// tile (the tiles of a batch run together and share its rows in L2), y = batch. B columns 0..127
// are gate outputs, 128..255 the same up outputs; wave w owns gate and up columns 16w.. of every
// row, so one lane holds g and u of the same (row, output) for the fused SwiGLU.
template<bool Q5K>
__global__ __launch_bounds__(256) void project_prefill_f16(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs) {
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.y>=csr.summary[2])return;
    const unsigned begin=csr.begins[blockIdx.y],count=csr.begins[blockIdx.y+1U]-begin,expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) F16Stage s;
    __shared__ F16Rows rows;
    const unsigned t=threadIdx.x;
    if(t<f16_batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=live?csr.slots[begin+t]:0U;rows.exponent[t]=live?at(view.sums,r)[0]:0;
    }
    __syncthreads();
    constexpr unsigned block_bytes=Q5K?176U:144U,qs_at=Q5K?48U:16U,mask=0x0f0f0f0fU;
    // Staging roles: A row t/4, K values 16(t%4)..; B column t: the 32 qs bytes of the step's pair.
    const unsigned a_row=t/4U,a_part=t%4U;
    const bool a_live=a_row<count;
    const float a_scale=ldexpf(1.0F,rows.exponent[a_row]);
    const auto* a_src=reinterpret_cast<const unsigned char*>(at(view.input,rows.row[a_row]))+32U*a_part;
    const unsigned blocks=hidden/256U,steps=hidden/64U;
    const unsigned char* b_src=(t<128U?gate:up)+
        (static_cast<unsigned long long>(expert)*outputs+blockIdx.x*128U+(t&127U))*blocks*block_bytes;
    struct Regs { uint4 a0{},a1{},h{},q0{},q1{},qh0{},qh1{}; };
    const auto fetch=[&](unsigned step,Regs& r) {
        if(a_live){r.a0=f16_load(a_src+128U*step);r.a1=f16_load(a_src+128U*step+16U);}
        const unsigned char* blk=b_src+(step/4U)*block_bytes;
        r.h=f16_load(blk);
        r.q0=f16_load(blk+qs_at+32U*(step%4U));r.q1=f16_load(blk+qs_at+32U*(step%4U)+16U);
        if constexpr(Q5K){r.qh0=f16_load(blk+16U);r.qh1=f16_load(blk+32U);}
    };
    const auto stage=[&](unsigned step,const Regs& r) {
        const uint4 rh=r.h,rq0=r.q0,rq1=r.q1,rqh0=r.qh0,rqh1=r.qh1;
        uint4* a_dst=reinterpret_cast<uint4*>(&s.a[a_row][16U*a_part]);
        a_dst[0]=f16_from_bf16(r.a0,a_scale);a_dst[1]=f16_from_bf16(r.a1,a_scale);
        const unsigned p=step%4U,g0=2U*p,g1=g0+1U;
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x&65535U)));
        const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x>>16U)));
        const float a0=d*static_cast<float>(factor(rh,g0,false)),b0=-(dm*static_cast<float>(factor(rh,g0,true)));
        const float a1=d*static_cast<float>(factor(rh,g1,false)),b1=-(dm*static_cast<float>(factor(rh,g1,true)));
        const unsigned q[8]{rq0.x,rq0.y,rq0.z,rq0.w,rq1.x,rq1.y,rq1.z,rq1.w};
        uint4* b_lo=reinterpret_cast<uint4*>(&s.b[t][0]);
        uint4* b_hi=reinterpret_cast<uint4*>(&s.b[t][32]);
        const unsigned qh[8]{rqh0.x,rqh0.y,rqh0.z,rqh0.w,rqh1.x,rqh1.y,rqh1.z,rqh1.w};
#pragma unroll
        for(unsigned k=0;k<8U;k+=2U) {
            unsigned l0=q[k]&mask,u0=(q[k]>>4U)&mask,l1=q[k+1U]&mask,u1=(q[k+1U]>>4U)&mask;
            if constexpr(Q5K) {
                l0|=((qh[k]>>g0)&0x01010101U)<<4U;u0|=((qh[k]>>g1)&0x01010101U)<<4U;
                l1|=((qh[k+1U]>>g0)&0x01010101U)<<4U;u1|=((qh[k+1U]>>g1)&0x01010101U)<<4U;
            }
            const uint2 x0=f16_affine_u8(l0,a0,b0),x1=f16_affine_u8(l1,a0,b0),y0=f16_affine_u8(u0,a1,b1),y1=f16_affine_u8(u1,a1,b1);
            b_lo[k/2U]={x0.x,x0.y,x1.x,x1.y};b_hi[k/2U]={y0.x,y0.y,y1.x,y1.y};
        }
    };
    const unsigned lane=t&31U,wave=t/32U,i=lane&15U,h=lane>>4U,mb=(count+15U)/16U;
    const unsigned cols[2]{16U*wave,128U+16U*wave};
    f32x8 acc[4][2]={};
    Regs r0,r1;
    fetch(0U,r0);fetch(1U,r1); // steps = hidden/64 is even (hidden % 256 == 0)
#pragma unroll 1
    for(unsigned step=0;step<steps;step+=2U) {
        if(step)f16_barrier();
        stage(step,r0);
        f16_barrier();
        if(step+2U<steps)fetch(step+2U,r0);
        f16_step(s,mb,cols,i,acc);
        f16_barrier();
        stage(step+1U,r1);
        f16_barrier();
        if(step+3U<steps)fetch(step+3U,r1);
        f16_step(s,mb,cols,i,acc);
    }
    const unsigned out=blockIdx.x*128U+16U*wave+i;
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=16U*mi+2U*e+h;
        const bool live=a<count;
        const unsigned r=rows.row[live?a:0U],slot=rows.slot[live?a:0U];
        const float unscale=ldexpf(1.0F,-rows.exponent[live?a:0U]);
        const float g=acc[mi][0][e]*unscale,u=acc[mi][1][e]*unscale;
        const float hv=(g/(1.0F+expf(-g)))*u;
        const bool bad=!isfinite(g)||!isfinite(u)||!isfinite(hv);
        if(live)at(view.output,r)[slot*2U*outputs+out]=hv;
        // Largest |h| of the wave's 16 outputs of this row (the lanes of one half hold one row).
        float m=fabsf(hv);
        for(int stride=8;stride!=0;stride/=2)m=fmaxf(m,__shfl_xor(m,stride,32));
        if(live&&i==0U)atomicMax(reinterpret_cast<unsigned*>(at(view.down_sums,r))+slot,__float_as_uint(m));
        if(live&&bad){atomicOr(at(view.status,r),8U);atomicOr(csr.summary,8U);}
    }
}
// The 32 FP16 weights of a Q8_0 down block (down_operands' byte selection, written without arrays).
__device__ __forceinline__ void f16_q80_block(const DownBlock<false>& x,unsigned group,uint4 (&out)[4]) {
    const unsigned odd=group&1U,select=odd?0x07060504U:0x05040302U;
    const float d=__half2float(__ushort_as_half(static_cast<unsigned short>((x.lo.x>>(odd?16U:0U))&65535U)));
    const auto pair=[&](unsigned hi0,unsigned lo0,unsigned hi1,unsigned lo1) {
        const uint2 a=f16_scaled_s8(__builtin_amdgcn_perm(hi0,lo0,select),d),b=f16_scaled_s8(__builtin_amdgcn_perm(hi1,lo1,select),d);
        return uint4{a.x,a.y,b.x,b.y};
    };
    out[0]=pair(x.lo.y,x.lo.x,x.lo.z,x.lo.y);out[1]=pair(x.lo.w,x.lo.z,x.hi.x,x.lo.w);
    out[2]=pair(x.hi.y,x.hi.x,x.hi.z,x.hi.y);out[3]=pair(x.hi.w,x.hi.z,x.last,x.hi.w);
}
// Down of one CSR batch (<= 64 assignments) for 256 hidden outputs, K = width (FP32 h of each
// (row, slot), scaled and rounded to FP16 while staging). Grid x = output tile, y = batch. B
// column t decodes K blocks 2*step, 2*step + 1 of its output row; wave w owns columns 32w..+32.
template<bool Q51>
__global__ __launch_bounds__(256) void down_prefill_f16(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.y>=csr.summary[2])return;
    const unsigned begin=csr.begins[blockIdx.y],count=csr.begins[blockIdx.y+1U]-begin,expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) F16Stage s;
    __shared__ F16Rows rows;
    const unsigned t=threadIdx.x;
    if(t<f16_batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U,slot=live?csr.slots[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=slot;
        rows.exponent[t]=live?f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,r)[slot]))):0;
    }
    __syncthreads();
    constexpr unsigned block_bytes=Q51?24U:34U;
    const unsigned a_row=t/4U,a_part=t%4U,groups=width/32U,steps=width/64U;
    const bool a_live=a_row<count;
    const float a_scale=ldexpf(1.0F,rows.exponent[a_row]);
    const auto* a_src=at(view.output,rows.row[a_row])+2ULL*rows.slot[a_row]*width+16U*a_part;
    const auto* b_src=weights+(static_cast<unsigned long long>(expert)*hidden+blockIdx.x*256U+t)*groups*block_bytes;
    struct Regs { uint4 a0{},a1{},a2{},a3{}; DownBlock<Q51> b0,b1; };
    const auto fetch=[&](unsigned step,Regs& r) {
        if(a_live){r.a0=f16_load(a_src+64U*step);r.a1=f16_load(a_src+64U*step+4U);r.a2=f16_load(a_src+64U*step+8U);r.a3=f16_load(a_src+64U*step+12U);}
        r.b0=down_block<Q51>(b_src,2U*step);r.b1=down_block<Q51>(b_src,2U*step+1U);
    };
    const auto stage=[&](unsigned step,const Regs& r) {
        const uint4 ra[4]{r.a0,r.a1,r.a2,r.a3};const DownBlock<Q51>& rb0=r.b0;const DownBlock<Q51>& rb1=r.b1;
        uint4* a_dst=reinterpret_cast<uint4*>(&s.a[a_row][16U*a_part]);
        a_dst[0]=f16_from_f32(ra[0],ra[1],a_scale);a_dst[1]=f16_from_f32(ra[2],ra[3],a_scale);
        uint4* b_dst=reinterpret_cast<uint4*>(&s.b[t][0]);
#pragma unroll
        for(unsigned half=0;half<2U;++half) {
            if constexpr(Q51) {
                wmma_v4i b0,b1;float d,m;
                down_operands<true>(half?rb1:rb0,2U*step+half,b0,b1,d,m);
                const unsigned c[8]{static_cast<unsigned>(b0.x),static_cast<unsigned>(b0.y),static_cast<unsigned>(b0.z),static_cast<unsigned>(b0.w),
                    static_cast<unsigned>(b1.x),static_cast<unsigned>(b1.y),static_cast<unsigned>(b1.z),static_cast<unsigned>(b1.w)};
#pragma unroll
                for(unsigned k=0;k<8U;k+=2U) {
                    const uint2 x=f16_affine_u8(c[k],d,m),y=f16_affine_u8(c[k+1U],d,m);
                    b_dst[4U*half+k/2U]={x.x,x.y,y.x,y.y};
                }
            } else {
                uint4 w[4];
                f16_q80_block(half?rb1:rb0,2U*step+half,w);
#pragma unroll
                for(unsigned k=0;k<4U;++k)b_dst[4U*half+k]=w[k];
            }
        }
    };
    const unsigned lane=t&31U,wave=t/32U,i=lane&15U,h=lane>>4U,mb=(count+15U)/16U;
    const unsigned cols[2]{32U*wave,32U*wave+16U};
    f32x8 acc[4][2]={};
    Regs r0,r1;
    fetch(0U,r0);fetch(1U,r1); // steps = width/64 is even (width % 128 == 0)
#pragma unroll 1
    for(unsigned step=0;step<steps;step+=2U) {
        if(step)f16_barrier();
        stage(step,r0);
        f16_barrier();
        if(step+2U<steps)fetch(step+2U,r0);
        f16_step(s,mb,cols,i,acc);
        f16_barrier();
        stage(step+1U,r1);
        f16_barrier();
        if(step+3U<steps)fetch(step+3U,r1);
        f16_step(s,mb,cols,i,acc);
    }
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=16U*mi+2U*e+h;
        if(a>=count)continue;
        const float unscale=ldexpf(1.0F,-rows.exponent[a]);
        float* out=at(view.rawdots,rows.row[a])+rows.slot[a]*hidden+blockIdx.x*256U+32U*wave+i;
        out[0]=acc[mi][0][e]*unscale;out[16]=acc[mi][1][e]*unscale;
    }
}
bool overlap(DeviceBytes a,DeviceBytes b){return a.address<b.address+b.bytes&&b.address<a.address+a.bytes;}
void range(DeviceBytes v){if(!v.address||!v.bytes||v.bytes>UINTPTR_MAX-v.address)throw std::invalid_argument("A8 CSR gate/up owner range is invalid");}
struct Prepared {DeviceRows rows;DeviceCsr csr;};
Prepared prepare(const Request& request,std::span<const Bindings> rows,const q51_known_rows::Workspace& workspace,hipStream_t stream) {
    if(rows.size()!=2U&&rows.size()!=4U&&rows.size()!=8U)throw std::invalid_argument("A8 CSR gate/up requires R2/R4/R8");
    if(staged_activation_bytes(static_cast<unsigned>(rows.size()))>65536U)
        throw std::invalid_argument("A8 CSR gate/up staging exceeds bounded LDS budget");
    const auto layout=q51_known_rows::scratch_layout(static_cast<unsigned>(rows.size()),request.shape.experts,request.shape.top_k);
    const DeviceBytes csr{workspace.memory.arena_base,workspace.memory.arena_bytes};range(csr);
    if(!workspace.owner||!workspace.memory.execution_identity||!workspace.memory.allocation_generation||
        csr.address%256U||csr.bytes!=layout.bytes)throw std::invalid_argument("A8 CSR gate/up retained CSR owner differs");
    if(!stream||stream==hipStreamLegacy||stream==hipStreamPerThread)throw std::invalid_argument("A8 CSR gate/up admits explicit eager streams only");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;check(hipStreamIsCapturing(stream,&capture),"A8 CSR gate/up capture query");
    if(capture!=hipStreamCaptureStatusNone)throw std::invalid_argument("A8 CSR gate/up admits explicit eager streams only");
    Prepared out;std::array<DeviceBytes,40> writes{};std::array<DeviceBytes,32> reads{};
    for(unsigned r=0;r<rows.size();++r) {
        const auto& b=rows[r];validate(request,b);
        if(b.gate.address!=rows.front().gate.address||b.gate.bytes!=rows.front().gate.bytes||
           b.up.address!=rows.front().up.address||b.up.bytes!=rows.front().up.bytes)
            throw std::invalid_argument("A8 CSR gate/up weight owners differ");
        const std::array mutable_views{b.codes,b.deltas,b.sums,b.status,b.output};
        const std::array readonly_views{b.input,b.gate,b.up,b.routes};
        for(unsigned v=0;v<5U;++v)writes[5U*r+v]=mutable_views[v];
        for(unsigned v=0;v<4U;++v)reads[4U*r+v]=readonly_views[v];
        out.rows.codes[r]=reinterpret_cast<const signed char*>(b.codes.address);out.rows.deltas[r]=reinterpret_cast<const float*>(b.deltas.address);
        out.rows.sums[r]=reinterpret_cast<const int*>(b.sums.address);out.rows.routes[r]=reinterpret_cast<const unsigned*>(b.routes.address);
        out.rows.status[r]=reinterpret_cast<unsigned*>(b.status.address);out.rows.output[r]=reinterpret_cast<float*>(b.output.address);
    }
    for(unsigned w=0;w<5U*rows.size();++w) {
        range(writes[w]);if(overlap(csr,writes[w]))throw std::invalid_argument("A8 CSR gate/up CSR aliases row storage");
        for(unsigned previous=0;previous<w;++previous)if(overlap(writes[w],writes[previous]))throw std::invalid_argument("A8 CSR gate/up mutable row owners alias");
        for(unsigned r=0;r<4U*rows.size();++r){range(reads[r]);if(overlap(writes[w],reads[r]))throw std::invalid_argument("A8 CSR gate/up writable/read-only owners alias");}
    }
    for(unsigned r=0;r<4U*rows.size();++r)if(overlap(csr,reads[r]))throw std::invalid_argument("A8 CSR gate/up CSR aliases read-only owner");
    const auto ptr=[&](std::uint64_t offset){return reinterpret_cast<unsigned*>(csr.address+offset);};
    out.csr={ptr(layout.header),ptr(layout.offsets),ptr(layout.touched),ptr(layout.rows),ptr(layout.slots),ptr(layout.batch_experts),ptr(layout.batch_begins)};
    return out;
}
} // namespace
void enqueue_window(const Request& request,std::span<const Bindings> rows,const q51_known_rows::Workspace& workspace,
    const WindowRows& window,hipStream_t stream) {
    auto p=prepare(request,rows,workspace,stream);
    const DeviceBytes csr{workspace.memory.arena_base,workspace.memory.arena_bytes};
    const bool a8=window.down_type!=0U,q5k=request.gate_type==13U;
    if((a8&&window.down_type!=7U&&window.down_type!=8U)||(q5k&&!a8))
        throw std::invalid_argument("A8 window down/gate format combination is invalid");
    const auto count=static_cast<unsigned>(rows.size()),groups=request.shape.rows/32U;
    PrologueRows v;
    for(unsigned r=0;r<count;++r) {
        const auto& b=rows[r];const auto weights=window.route_weights[r],clear=window.clear[r];
        range(weights);range(clear);
        if(weights.bytes<4ULL*request.shape.top_k||clear.bytes!=window.clear[0].bytes||clear.bytes%4U||clear.address%4U||
           clear.bytes/4U>UINT32_MAX||overlap(clear,csr)||overlap(weights,csr))
            throw std::invalid_argument("A8 window row extent is invalid");
        for(unsigned q=0;q<count;++q)for(const auto view:{rows[q].codes,rows[q].deltas,rows[q].sums,rows[q].status,
            rows[q].output,rows[q].input,rows[q].routes,window.route_weights[q]})if(overlap(clear,view))
            throw std::invalid_argument("A8 window cleared region aliases row storage");
        if(a8) {
            const std::array down{window.down_codes[r],window.down_deltas[r],window.down_sums[r],window.rawdots[r]};
            const std::array<std::uint64_t,4> need{static_cast<std::uint64_t>(request.shape.top_k)*request.shape.rows,
                4ULL*request.shape.top_k*groups,4ULL*request.shape.top_k*groups,4ULL*request.shape.top_k*request.shape.hidden};
            for(unsigned i=0;i<4U;++i) {
                range(down[i]);
                if(down[i].bytes<need[i]||down[i].address%16U||overlap(down[i],csr)||overlap(down[i],window.down_weights))
                    throw std::invalid_argument("A8 window down row view is invalid");
                for(unsigned q=0;q<count;++q)for(const auto view:{rows[q].codes,rows[q].deltas,rows[q].sums,rows[q].status,
                    rows[q].output,rows[q].input,rows[q].routes,rows[q].gate,rows[q].up,window.clear[q],window.route_weights[q]})
                    if(overlap(down[i],view))throw std::invalid_argument("A8 window down row view aliases row storage");
                for(unsigned q=0;q<=r;++q)for(unsigned j=0;j<4U;++j)if((q<r||j<i)&&overlap(down[i],
                    std::array{window.down_codes[q],window.down_deltas[q],window.down_sums[q],window.rawdots[q]}[j]))
                    throw std::invalid_argument("A8 window down row views alias");
            }
            p.rows.down_codes[r]=reinterpret_cast<signed char*>(window.down_codes[r].address);
            p.rows.down_deltas[r]=reinterpret_cast<float*>(window.down_deltas[r].address);
            p.rows.down_sums[r]=reinterpret_cast<int*>(window.down_sums[r].address);
            p.rows.rawdots[r]=reinterpret_cast<float*>(window.rawdots[r].address);
        }
        v.input[r]=reinterpret_cast<const void*>(b.input.address);v.routes[r]=p.rows.routes[r];
        v.route_weights[r]=reinterpret_cast<const float*>(weights.address);v.codes[r]=reinterpret_cast<signed char*>(b.codes.address);
        v.deltas[r]=reinterpret_cast<float*>(b.deltas.address);v.sums[r]=reinterpret_cast<int*>(b.sums.address);
        v.status[r]=p.rows.status[r];v.clear[r]=reinterpret_cast<float*>(clear.address);
    }
    if(a8) {
        range(window.down_weights);
        const std::uint64_t block_bytes=window.down_type==7U?24U:34U;
        if(window.down_weights.bytes!=static_cast<std::uint64_t>(request.shape.experts)*request.shape.hidden*groups*block_bytes||
           request.shape.rows>max_rows||request.shape.rows%128U||request.shape.hidden%64U||overlap(window.down_weights,csr))
            throw std::invalid_argument("A8 window down weights/geometry are invalid");
    }
    v.csr={p.csr.summary,p.csr.offsets,p.csr.touched,p.csr.rows,p.csr.slots,p.csr.experts,p.csr.begins};
    enqueue_prologue(request,v,count,static_cast<unsigned>(window.clear[0].bytes/4U),stream);
    const auto* gate=reinterpret_cast<const unsigned char*>(rows.front().gate.address);
    const auto* up=reinterpret_cast<const unsigned char*>(rows.front().up.address);
    const dim3 project_grid(request.shape.rows/32U,count*request.shape.top_k);
#define STRIX_A8_WINDOW_PROJECT(G,Q5,A8) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window<G,Q5,A8>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr, \
        count,request.shape.hidden,request.shape.rows)
    // STRIX_MOE_GATEUP_CHUNK=5: half the Q4_K blocks in registers at a time (same sums).
    static const bool half_chunk=[]{const char* v=std::getenv("STRIX_MOE_GATEUP_CHUNK");return v && v[0]=='5';}();
    const auto chunked=[&]<unsigned G>(){
        hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window<G,false,true,5U>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr,
            count,request.shape.hidden,request.shape.rows);
    };
    if(!a8){if(count==2U){STRIX_A8_WINDOW_PROJECT(2U,false,false);}else{STRIX_A8_WINDOW_PROJECT(4U,false,false);}}
    else if(!q5k && half_chunk){if(count==2U)chunked.template operator()<2U>();else chunked.template operator()<4U>();}
    else if(!q5k){if(count==2U){STRIX_A8_WINDOW_PROJECT(2U,false,true);}else{STRIX_A8_WINDOW_PROJECT(4U,false,true);}}
    else {if(count==2U){STRIX_A8_WINDOW_PROJECT(2U,true,true);}else{STRIX_A8_WINDOW_PROJECT(4U,true,true);}}
#undef STRIX_A8_WINDOW_PROJECT
    check(hipGetLastError(),"A8 window shared gate/up projection");
    if(!a8)return;
    const auto* down=reinterpret_cast<const unsigned char*>(window.down_weights.address);
    const dim3 down_grid(request.shape.hidden/64U,count*request.shape.top_k);
#define STRIX_A8_WINDOW_DOWN(R,Q51) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(down_window<R,Q51>),down_grid,dim3(256),0,stream,down,p.rows,p.csr, \
        count,request.shape.hidden,request.shape.rows)
    if(window.down_type==7U){if(count==2U){STRIX_A8_WINDOW_DOWN(2U,true);}else if(count==4U){STRIX_A8_WINDOW_DOWN(4U,true);}else{STRIX_A8_WINDOW_DOWN(8U,true);}}
    else {if(count==2U){STRIX_A8_WINDOW_DOWN(2U,false);}else if(count==4U){STRIX_A8_WINDOW_DOWN(4U,false);}else{STRIX_A8_WINDOW_DOWN(8U,false);}}
#undef STRIX_A8_WINDOW_DOWN
    check(hipGetLastError(),"A8 window shared integer down");
}
void enqueue_prefill(const Request& request,const PrefillRows& rows,unsigned count,unsigned clear_words,
    const unsigned char* gate,const unsigned char* up,unsigned down_type,const unsigned char* down_weights,hipStream_t stream,int form) {
    const bool q5k=request.gate_type==13U;
    if((down_type!=7U&&down_type!=8U)||(request.gate_type!=12U&&!q5k)||!gate||!up||!down_weights||
       count<16U||count%8U||count>q51_known_rows::prefill_max_rows||request.shape.top_k>10U||request.shape.experts>512U||
       request.shape.hidden>max_hidden||request.shape.hidden%256U||request.shape.rows>max_rows||request.shape.rows%128U||
       request.shape.hidden%64U)
        throw std::invalid_argument("A8 prefill geometry/format is invalid");
    // STRIX_PREFILL_MOE_BATCH: assignments of one expert per CTA (default 8).
    static const unsigned dot4_cap=[]{const char* v=std::getenv("STRIX_PREFILL_MOE_BATCH");const int n=v?std::atoi(v):8;
        return n>=1&&n<=64?static_cast<unsigned>(n):8U;}();
    // STRIX_PREFILL_MOE_WMMA=0/1: dot4 or iu8 WMMA kernels (bit-identical results; default WMMA).
    static const int wmma_env=[]{const char* v=std::getenv("STRIX_PREFILL_MOE_WMMA");return v?std::atoi(v):-1;}();
    // Whole CTA tiles; the WMMA down is unrolled for width = max_rows.
    const bool wmma_shape=request.shape.rows==max_rows&&request.shape.hidden%256U==0U;
    const bool f16=form==3;
    if(f16&&(request.shape.rows%128U||request.shape.hidden%256U))throw std::invalid_argument("A8 prefill FP16 WMMA geometry is invalid");
    const bool wmma=f16||(wmma_shape&&(form>=0?form>=1:wmma_env>=0?wmma_env==1:true));
    const unsigned cap=f16?f16_batch:wmma?wmma_batch:dot4_cap;
    enqueue_prefill_prologue(request,rows,count,clear_words,stream);
    const unsigned assignments=count*request.shape.top_k;
    // The WMMA form builds the (identical) CSR with the parallel builder; form 2 keeps the serial one.
    if(wmma&&form!=2)hipLaunchKernelGGL(build_prefill_csr_waves,dim3(1),dim3(1024),0,stream,rows,count,request.shape.experts,request.shape.top_k,cap);
    else hipLaunchKernelGGL(build_prefill_csr,dim3(1),dim3(1024),0,stream,rows,count,request.shape.experts,request.shape.top_k,cap);
    check(hipGetLastError(),"A8 prefill CSR");
    // STRIX_PREFILL_MOE_CSR_DUMP=1 (diagnostic, synchronizes): rows-per-expert histogram of each call.
    static const bool csr_dump=[]{const char* v=std::getenv("STRIX_PREFILL_MOE_CSR_DUMP");return v&&v[0]=='1';}();
    if(csr_dump) {
        std::vector<unsigned> offsets(request.shape.experts+1U);
        check(hipMemcpyAsync(offsets.data(),rows.csr.offsets,4U*offsets.size(),hipMemcpyDeviceToHost,stream),"CSR dump");
        check(hipStreamSynchronize(stream),"CSR dump");
        std::array<unsigned,8> bucket{};unsigned largest=0;
        for(unsigned e=0;e<request.shape.experts;++e) {
            const unsigned n=offsets[e+1U]-offsets[e];largest=std::max(largest,n);
            bucket[n==0U?0U:n<=16U?1U:n<=32U?2U:n<=64U?3U:n<=128U?4U:n<=256U?5U:n<=512U?6U:7U]++;
        }
        std::fprintf(stderr,"moe-csr T=%u experts n=0:%u 1-16:%u 17-32:%u 33-64:%u 65-128:%u 129-256:%u 257-512:%u >512:%u max %u\n",
            count,bucket[0],bucket[1],bucket[2],bucket[3],bucket[4],bucket[5],bucket[6],bucket[7],largest);
    }
    const unsigned batches=std::min(assignments,request.shape.experts+(assignments+cap-1U)/cap);
    if(f16) {
        hipLaunchKernelGGL(f16_row_scales,dim3(count),dim3(256),0,stream,rows,request.shape.hidden,request.shape.top_k);
        check(hipGetLastError(),"FP16 prefill row scales");
        const dim3 project_grid(request.shape.rows/128U,batches);
        if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16<true>),project_grid,dim3(256),0,stream,
            gate,up,rows,request.shape.hidden,request.shape.rows);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16<false>),project_grid,dim3(256),0,stream,
            gate,up,rows,request.shape.hidden,request.shape.rows);
        check(hipGetLastError(),"FP16 prefill WMMA gate/up projection");
        const dim3 down_grid(request.shape.hidden/256U,batches);
        if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16<true>),down_grid,dim3(256),0,
            stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16<false>),down_grid,dim3(256),0,
            stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        check(hipGetLastError(),"FP16 prefill WMMA down");
        return;
    }
    if(wmma) {
        // 16 waves per CTA (measured best against 8; down also against 4).
        constexpr unsigned project_waves=16U,down_waves=16U;
        const dim3 project_grid(batches,request.shape.rows/(8U*project_waves));
        if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_wmma<true,project_waves>),project_grid,dim3(32U*project_waves),0,stream,
            gate,up,rows,request.shape.hidden,request.shape.rows);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_wmma<false,project_waves>),project_grid,dim3(32U*project_waves),0,stream,
            gate,up,rows,request.shape.hidden,request.shape.rows);
        check(hipGetLastError(),"A8 prefill WMMA gate/up projection");
        const dim3 down_grid(batches,request.shape.hidden/(16U*down_waves));
        if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_wmma<true,down_waves>),down_grid,dim3(32U*down_waves),0,
            stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_wmma<false,down_waves>),down_grid,dim3(32U*down_waves),0,
            stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        check(hipGetLastError(),"A8 prefill WMMA integer down");
        return;
    }
    const dim3 project_grid(batches,request.shape.rows/32U);
    if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill<4U,true>),project_grid,dim3(256),0,stream,gate,up,rows,
        request.shape.hidden,request.shape.rows);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill<4U,false>),project_grid,dim3(256),0,stream,gate,up,rows,
        request.shape.hidden,request.shape.rows);
    check(hipGetLastError(),"A8 prefill gate/up projection");
    const dim3 down_grid(batches,request.shape.hidden/64U);
    if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill<8U,true>),down_grid,dim3(256),0,stream,down_weights,rows,
        request.shape.hidden,request.shape.rows);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill<8U,false>),down_grid,dim3(256),0,stream,down_weights,rows,
        request.shape.hidden,request.shape.rows);
    check(hipGetLastError(),"A8 prefill integer down");
}
void preflight(const Request& request,std::span<const Bindings> rows,const q51_known_rows::Workspace& workspace,hipStream_t stream) {
    (void)prepare(request,rows,workspace,stream);
}
void enqueue(const Request& request,std::span<const Bindings> rows,const q51_known_rows::Workspace& workspace,hipStream_t stream) {
    const auto p=prepare(request,rows,workspace,stream);check(hipPeekAtLastError(),"A8 CSR gate/up stale launch error");
    admit_csr<<<1,1,0,stream>>>(p.rows,p.csr,static_cast<unsigned>(rows.size()),request.shape.experts,request.shape.top_k);
    check(hipGetLastError(),"A8 CSR gate/up device CSR admission");
#define STRIX_A8_CSR_PROJECT(R) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(project_csr<R>),dim3(request.shape.rows/32U,static_cast<unsigned>(rows.size())*request.shape.top_k), \
        dim3(256),0,stream,reinterpret_cast<const unsigned char*>(rows.front().gate.address), \
        reinterpret_cast<const unsigned char*>(rows.front().up.address),p.rows,p.csr,request.shape.hidden,request.shape.rows)
    if(rows.size()==2U){STRIX_A8_CSR_PROJECT(2);}
    else if(rows.size()==4U){STRIX_A8_CSR_PROJECT(4);}
    else {STRIX_A8_CSR_PROJECT(8);}
#undef STRIX_A8_CSR_PROJECT
    check(hipGetLastError(),"A8 CSR shared gate/up projection");
}
} // namespace strix::moe_q4k_a8::known_rows
