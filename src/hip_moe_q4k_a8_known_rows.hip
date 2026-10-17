#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "A8 CSR gate/up requires strict floating-point semantics"
#endif
#include <hip/hip_fp16.h>
#include <algorithm>
#include <array>
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
    const unsigned char* gate,const unsigned char* up,unsigned down_type,const unsigned char* down_weights,hipStream_t stream) {
    const bool q5k=request.gate_type==13U;
    if((down_type!=7U&&down_type!=8U)||(request.gate_type!=12U&&!q5k)||!gate||!up||!down_weights||
       count<16U||count%8U||count>q51_known_rows::prefill_max_rows||request.shape.top_k>10U||request.shape.experts>512U||
       request.shape.hidden>max_hidden||request.shape.hidden%256U||request.shape.rows>max_rows||request.shape.rows%128U||
       request.shape.hidden%64U)
        throw std::invalid_argument("A8 prefill geometry/format is invalid");
    // STRIX_PREFILL_MOE_BATCH: assignments of one expert per CTA (default 8).
    static const unsigned cap=[]{const char* v=std::getenv("STRIX_PREFILL_MOE_BATCH");const int n=v?std::atoi(v):8;
        return n>=1&&n<=64?static_cast<unsigned>(n):8U;}();
    enqueue_prefill_prologue(request,rows,count,clear_words,stream);
    const unsigned assignments=count*request.shape.top_k;
    hipLaunchKernelGGL(build_prefill_csr,dim3(1),dim3(1024),0,stream,rows,count,request.shape.experts,request.shape.top_k,cap);
    check(hipGetLastError(),"A8 prefill CSR");
    const unsigned batches=std::min(assignments,request.shape.experts+(assignments+cap-1U)/cap);
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
