// qpf2k/moe G3 core (STRIX_QPF2K_MOE bit 10, prefill only): the grouped-expert form of the dense FP16 tiled core
// (q8_rows::tiled form 3: 64 x 64 per wave, two LDS stages of 32 K values with one LDS-only barrier each, padded /
// XOR-permuted LDS rows) for the routed-expert gate/up and down. Included by hip_moe_q4k_a8_known_rows.hip inside its
// anonymous namespace after down_prefill_f16x (it uses factor, f16_affine_u8(_fast), f16_affine_exact, f16_q51_exact,
// f16_scaled_s8, f16_from_f32, f16_exponent, scale_f32, f16_load, f16_wmma).
//
// Same numerics as the FP16 prefill forms, bit for bit. Every (assignment, output) value is one v_wmma_f32_16x16x16_f16
// chain over the K16 planes 0, 1, ... ascending from +0, with A = the row-scaled FP16 image (gate/up: the prologue's a16
// copy; down: f16_from_f32 of h with the (row, slot) scale) and B = the FP16 dequantized weights computed by the same
// functions as the base (Q4_K/Q5_K: f16_affine_u8 of a = d*sc, b = -dmin*m, or f16_affine_u8_fast where the wave's
// blocks pass f16_affine_exact - equal bits by the bit-6 self-test; Q5_1: the same with f16_q51_exact; Q8_0:
// f16_scaled_s8). The epilogues are the base ones per element (SwiGLU, |h| maxima over the same 16-output groups, raw
// dots). Only the decomposition changes:
//   - a CTA owns BM assignments of ONE expert (BM 256: up to two 128- or four 64-assignment CSR batches, the batch at
//     the start of each BM run leads, the others exit) x 256 B columns (gate/up: 128 outputs of gate and the same of up;
//     down: 256 hidden outputs), BM / 16 waves of 64 rows x 64 columns (16 accumulators; every A and B fragment reused
//     4x); a wave whose rows are all past the expert's count skips its WMMAs (per row tile: a scalar branch);
//   - one stage = 32 K values (one Q4_K/Q5_K sub-block, one Q5_1/Q8_0 block); two LDS buffers, the next stage's global
//     loads in registers while the current stage is multiplied, s_waitcnt lgkmcnt(0) + s_barrier only (no global fence);
//   - LDS rows of 32 halves whose 16-byte chunks are XOR-permuted by (row / 2) % 4 (BM 256: two stages in 64 KiB).
// A rows past the expert's count stage the first assignment's row (finite, unused: WMMA rows are independent and those
// accumulators are never stored).
// Probe (bench2 attribution only, wrong outputs, never selectable from the environment): 1 = B codes stored without
// dequantization, 2 = no WMMA (cheap accumulate), 3 = no global loads in the K loop.

template<unsigned BM> struct G3Shape {
    static constexpr unsigned Threads=2U*BM,Waves=Threads/32U,WavesN=4U,WavesM=Waves/WavesN;
    static constexpr unsigned BN=256U,Pitch=32U,StageH=(BM+BN)*Pitch;
    static constexpr unsigned TPC=Threads/BN,VPT=32U/TPC,CW=VPT/16U; // threads per B column, values and 16-byte code chunks per thread
    static_assert(WavesM*64U==BM&&TPC*BN==Threads&&(VPT==16U||VPT==32U));
};
__device__ __forceinline__ unsigned g3_at(unsigned row,unsigned half){return row*32U+((((half>>3U)^((row>>1U)&3U)))<<3U)+(half&7U);}
__device__ __forceinline__ unsigned g3_load4(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) unsigned*>(reinterpret_cast<std::uintptr_t>(p));
}
__device__ __forceinline__ uint2 g3_load8(const void* p) {
    typedef unsigned u32x2_t __attribute__((ext_vector_type(2)));
    const u32x2_t v=*reinterpret_cast<const __attribute__((address_space(1))) u32x2_t*>(reinterpret_cast<std::uintptr_t>(p));
    return uint2{v.x,v.y};
}
// 16-byte global load from a dword-aligned address (Q5_1 qs at block + 8, Q8_0 code dwords): the vector type promises 4 bytes only.
__device__ __forceinline__ uint4 g3_load16a4(const void* p) {
    typedef unsigned u32x4a4_t __attribute__((ext_vector_type(4),aligned(4)));
    const u32x4a4_t v=*reinterpret_cast<const __attribute__((address_space(1))) u32x4a4_t*>(reinterpret_cast<std::uintptr_t>(p));
    return uint4{v.x,v.y,v.z,v.w};
}
__device__ __forceinline__ void g3_barrier(){__asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");}
// 32-byte fragment (16 halves) of LDS row `row`, K halves 16 kt .. + 15 (two permuted 16-byte chunks).
__device__ __forceinline__ f16_u32x8 g3_frag(const _Float16* base,unsigned row,unsigned kt) {
    const uint4 lo=*reinterpret_cast<const uint4*>(base+g3_at(row,16U*kt)),hi=*reinterpret_cast<const uint4*>(base+g3_at(row,16U*kt+8U));
    return f16_u32x8{lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
}
// The CTA's rows: the expert's assignments [begin, begin + count) when this batch leads a BM run, else run = false.
// CSR batches of an expert are consecutive cap-assignment runs from its offset.
struct G3Run { unsigned begin,count,expert; bool run; };
__device__ __forceinline__ G3Run g3_run(const CsrRows& csr,unsigned batch,unsigned bm,unsigned cap) {
    const unsigned begin=csr.begins[batch],expert=csr.experts[batch];
    const unsigned first=csr.offsets[expert],end=csr.offsets[expert+1U];
    const bool lead=((begin-first)/cap)%(bm/cap)==0U;
    return {begin,min(end-begin,bm),expert,lead};
}
// Accumulators of one wave: acc[t][u] = rows 16 t.., columns 16 u.. of the wave's 64 x 64 tile. Per K16 plane: the four B
// fragments, then per live row tile (t < smb, a scalar branch: smb is wave-uniform) its A fragment and four WMMAs (u ascending).
// (A switch over the tile count per plane made the register allocator move the accumulators at every merge and spill.)
template<unsigned Probe>
__device__ __forceinline__ void g3_compute(const _Float16* as,const _Float16* bs,unsigned arow0,const unsigned (&brow)[4],unsigned smb,f32x8 (&acc)[4][4]) {
#pragma unroll
    for(unsigned kt=0;kt<2U;++kt) {
        f16_u32x8 b[4];
#pragma unroll
        for(unsigned u=0;u<4U;++u)b[u]=g3_frag(bs,brow[u],kt);
#pragma unroll
        for(unsigned t=0;t<4U;++t)if(t<smb) {
            const f16_u32x8 a=g3_frag(as,arow0+16U*t,kt);
#pragma unroll
            for(unsigned u=0;u<4U;++u) {
                if constexpr(Probe==2U)acc[t][u][0]+=__uint_as_float(a[0]^b[u][1]);
                else acc[t][u]=f16_wmma(a,b[u],acc[t][u]);
            }
        }
    }
}
// The K loop: stage s in LDS buffer s % 2; the loads of stage s + 1 are in registers while stage s is multiplied, then
// stored into the other buffer; one LDS-only barrier per stage. stages is even.
template<class Regs,class Fetch,class Store,class Compute>
__device__ __forceinline__ void g3_pipeline(unsigned stages,const Fetch& fetch,const Store& store,const Compute& compute) {
    Regs r{};
    fetch(0U,r);
    store(0U,r,0U);
    g3_barrier();
#pragma unroll 1
    for(unsigned s=0;s<stages;s+=2U) {
        fetch(s+1U,r);
        compute(0U);
        store(s+1U,r,1U);
        g3_barrier();
        const bool more=s+2U<stages;
        if(more)fetch(s+2U,r);
        compute(1U);
        if(more)store(s+2U,r,0U);
        g3_barrier();
    }
}

// Core bits 5/6 (depth D = 2/3): the same stage order, LDS buffers and barriers with D stages of global loads in flight in a
// register ring (t4: with one stage in flight G3 is latency-bound - no-WMMA probe 17.7 of 18.3 ms; the base keeps two K64
// steps in flight at twice the waves). Stage x is in ring slot x % D and LDS buffer x % 2; every main-loop trip issues the same
// loads (past the end the last stage is re-read, unused), so the wait counts stay exact.
template<unsigned D,class Regs,class Fetch,class Store,class Compute>
__device__ __forceinline__ void g3_pipeline_d(unsigned stages,const Fetch& fetch,const Store& store,const Compute& compute) {
    if constexpr(D==1U) { g3_pipeline<Regs>(stages,fetch,store,compute); return; }
    else {
        constexpr unsigned U=(D%2U==0U)?D:2U*D;
        Regs r[D]{};
#pragma unroll
        for(unsigned j=0;j<D;++j)fetch(min(j,stages-1U),r[j]);
        store(0U,r[0],0U);
        g3_barrier();
        fetch(min(D,stages-1U),r[0]);
        const unsigned whole=stages-stages%U;
#pragma unroll 1
        for(unsigned s=0;s<whole;s+=U) {
#pragma unroll
            for(unsigned j=0;j<U;++j) {
                const unsigned cur=s+j;
                compute(j%2U);
                if(cur+1U<stages)store(cur+1U,r[(j+1U)%D],(j+1U)%2U);
                g3_barrier();
                fetch(min(cur+1U+D,stages-1U),r[(j+1U)%D]);
            }
        }
#pragma unroll
        for(unsigned j=0;j<U;++j) {
            const unsigned cur=whole+j;
            if(cur<stages) {
                compute(j%2U);
                if(cur+1U<stages)store(cur+1U,r[(j+1U)%D],(j+1U)%2U);
                g3_barrier();
            }
        }
    }
}

// Gate/up: grid x = 128-output tile, y = CSR batch. hidden % 256 == 0, outputs % 128 == 0.
template<bool Q5K,unsigned BM,unsigned Probe=0U,unsigned D=1U>
__global__ __launch_bounds__(G3Shape<BM>::Threads) void g3_project(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs,const uint4* a16,unsigned cap) {
    using S=G3Shape<BM>;
    constexpr unsigned BN=S::BN,Outs=BN/2U,VPT=S::VPT,CW=S::CW,TPC=S::TPC;
    __shared__ __align__(16) _Float16 lds[2U*S::StageH];
    const unsigned batch=blockIdx.y,tile=blockIdx.x;
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||batch>=csr.summary[2])return;
    const G3Run run=g3_run(csr,batch,BM,cap);
    if(!run.run)return;
    const unsigned begin=run.begin,count=run.count,expert=run.expert;
    const unsigned t=threadIdx.x,lane=t&31U,i=lane&15U,h=lane>>4U;
    const unsigned wave=static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(t/32U)));
    const unsigned wm=wave/S::WavesN,wn=wave%S::WavesN;
    constexpr unsigned block_bytes=Q5K?176U:144U,qs_at=Q5K?48U:16U,mask=0x0f0f0f0fU;
    const unsigned stages=hidden/32U;
    // A: thread t stages halves 8 (t % 4) .. + 7 of rows t / 4 + BM / 2 j (j = 0, 1) from the FP16 image.
    const unsigned a_part=t%4U;
    const unsigned char* a_src[2];
#pragma unroll
    for(unsigned j=0;j<2U;++j) {
        const unsigned a=t/4U+(BM/2U)*j;
        const unsigned r=csr.rows[begin+(a<count?a:0U)];
        a_src[j]=reinterpret_cast<const unsigned char*>(a16+static_cast<std::size_t>(r)*(hidden/8U))+16U*a_part;
    }
    // B: column t / TPC (0..127 gate outputs, 128..255 the same up outputs), values VPT (t % TPC) .. of the stage's sub-block.
    const unsigned b_col=t/TPC,b_part=t%TPC;
    const unsigned blocks=hidden/256U;
    const unsigned char* b_src=(b_col<Outs?gate:up)+
        (static_cast<unsigned long long>(expert)*outputs+tile*Outs+(b_col%Outs))*blocks*block_bytes;
    struct Regs { uint4 a[2],h,q[CW],qh[CW]; };
    const auto fetch=[&](unsigned s,Regs& r) {
        if constexpr(Probe==3U){(void)s;(void)r;return;}
        else {
#pragma unroll
            for(unsigned j=0;j<2U;++j)r.a[j]=f16_load(a_src[j]+64U*s);
            const unsigned char* blk=b_src+(s/8U)*block_bytes;
            const unsigned sub=s%8U;
            r.h=f16_load(blk);
#pragma unroll
            for(unsigned c=0;c<CW;++c) {
                r.q[c]=f16_load(blk+qs_at+32U*(sub/2U)+VPT*b_part+16U*c);
                if constexpr(Q5K)r.qh[c]=f16_load(blk+16U+VPT*b_part+16U*c);
            }
        }
    };
    bool fast=false; // the wave's superblock passes f16_affine_exact (set at its first sub-block, as the base per superblock)
    const auto store=[&](unsigned s,const Regs& r,unsigned buf) {
        _Float16* as=lds+buf*S::StageH;
        _Float16* bs=as+BM*S::Pitch;
#pragma unroll
        for(unsigned j=0;j<2U;++j)*reinterpret_cast<uint4*>(as+g3_at(t/4U+(BM/2U)*j,8U*a_part))=r.a[j];
        const unsigned sub=s%8U;
        const uint4 rh=r.h;
        if constexpr(Probe==1U) {
#pragma unroll
            for(unsigned c=0;c<CW;++c) {
                *reinterpret_cast<uint4*>(bs+g3_at(b_col,VPT*b_part+16U*c))=r.q[c];
                *reinterpret_cast<uint4*>(bs+g3_at(b_col,VPT*b_part+16U*c+8U))=uint4{r.q[c].x^rh.x,r.q[c].y,r.q[c].z,r.q[c].w};
            }
            return;
        }
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x&65535U)));
        const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x>>16U)));
        const float a0=d*static_cast<float>(factor(rh,sub,false)),b0=-(dm*static_cast<float>(factor(rh,sub,true)));
        if(sub==0U)fast=__all(f16_affine_exact(rh.x&65535U,63U*(Q5K?31U:15U),rh.x>>16U,63U));
        const bool hi_nibble=(sub&1U)!=0U;
#pragma unroll
        for(unsigned c=0;c<CW;++c) {
            const unsigned q[4]{r.q[c].x,r.q[c].y,r.q[c].z,r.q[c].w};
            unsigned w[4];
#pragma unroll
            for(unsigned k=0;k<4U;++k) {
                w[k]=hi_nibble?(q[k]>>4U)&mask:q[k]&mask;
                if constexpr(Q5K) {
                    const unsigned qh[4]{r.qh[c].x,r.qh[c].y,r.qh[c].z,r.qh[c].w};
                    w[k]|=((qh[k]>>sub)&0x01010101U)<<4U;
                }
            }
            uint2 x[4];
            if(fast) {
#pragma unroll
                for(unsigned k=0;k<4U;++k)x[k]=f16_affine_u8_fast(w[k],a0*16777216.0F,b0);
            } else {
#pragma unroll
                for(unsigned k=0;k<4U;++k)x[k]=f16_affine_u8(w[k],a0,b0);
            }
            *reinterpret_cast<uint4*>(bs+g3_at(b_col,VPT*b_part+16U*c))=uint4{x[0].x,x[0].y,x[1].x,x[1].y};
            *reinterpret_cast<uint4*>(bs+g3_at(b_col,VPT*b_part+16U*c+8U))=uint4{x[2].x,x[2].y,x[3].x,x[3].y};
        }
    };
    const unsigned arow0=64U*wm+i;
    const unsigned brow[4]{32U*wn+i,32U*wn+16U+i,Outs+32U*wn+i,Outs+32U*wn+16U+i};
    const unsigned rbase=64U*wm,mb=count>rbase?min(4U,(count-rbase+15U)/16U):0U;
    const unsigned smb=static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(mb)));
    f32x8 acc[4][4]={};
    g3_pipeline_d<D,Regs>(stages,fetch,store,[&](unsigned buf) {
        const _Float16* as=lds+buf*S::StageH;
        g3_compute<Probe>(as,as+BM*S::Pitch,arow0,brow,smb,acc);
    });
    // Rows table (row, slot, scale exponent) in the free stage memory.
    unsigned* tab=reinterpret_cast<unsigned*>(lds);
    if(t<BM) {
        const bool live=t<count;
        const unsigned r=live?csr.rows[begin+t]:0U;
        tab[t]=r;tab[BM+t]=live?csr.slots[begin+t]:0U;tab[2U*BM+t]=static_cast<unsigned>(live?at(view.sums,r)[0]:0);
    }
    __syncthreads();
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<smb)
#pragma unroll
    for(unsigned u=0;u<2U;++u) {
        const unsigned out=tile*Outs+32U*wn+16U*u+i;
#pragma unroll
        for(unsigned e=0;e<8U;++e) {
            const unsigned a=rbase+16U*mi+2U*e+h;
            const bool live=a<count;
            const unsigned ai=live?a:0U;
            const unsigned r=tab[ai],slot=tab[BM+ai];
            const float unscale=ldexpf(1.0F,-static_cast<int>(tab[2U*BM+ai]));
            const float g=acc[mi][u][e]*unscale,uu=acc[mi][u+2U][e]*unscale;
            const float hv=(g/(1.0F+expf(-g)))*uu;
            const bool bad=!isfinite(g)||!isfinite(uu)||!isfinite(hv);
            if(live)at(view.output,r)[slot*2U*outputs+out]=hv;
            float m=fabsf(hv);
            for(int stride=8;stride!=0;stride/=2)m=fmaxf(m,__shfl_xor(m,stride,32));
            if(live&&i==0U)atomicMax(reinterpret_cast<unsigned*>(at(view.down_sums,r))+slot,__float_as_uint(m));
            if(live&&bad){atomicOr(at(view.status,r),projection_nonfinite);atomicOr(csr.summary,projection_nonfinite);}
        }
    }
}

// Down: grid x = 256-output tile, y = CSR batch. K = width (h of each (row, slot) in FP32, scaled and rounded while
// staging), width % 64 == 0, hidden % 256 == 0.
template<bool Q51,unsigned BM,unsigned Probe=0U,unsigned D=1U>
__global__ __launch_bounds__(G3Shape<BM>::Threads) void g3_down(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width,unsigned cap) {
    using S=G3Shape<BM>;
    constexpr unsigned BN=S::BN,VPT=S::VPT,TPC=S::TPC;
    __shared__ __align__(16) _Float16 lds[2U*S::StageH];
    const unsigned batch=blockIdx.y,tile=blockIdx.x;
    const auto& csr=view.csr;
    if(csr.summary[0]||batch>=csr.summary[2])return;
    const G3Run run=g3_run(csr,batch,BM,cap);
    if(!run.run)return;
    const unsigned begin=run.begin,count=run.count,expert=run.expert;
    const unsigned t=threadIdx.x,lane=t&31U,i=lane&15U,h=lane>>4U;
    const unsigned wave=static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(t/32U)));
    const unsigned wm=wave/S::WavesN,wn=wave%S::WavesN;
    constexpr unsigned block_bytes=Q51?24U:34U;
    const unsigned stages=width/32U,groups=width/32U;
    // A: thread t stages K values 8 (t % 4) .. + 7 (two 16-byte FP32 loads -> one 16-byte FP16 word) of rows t / 4 + BM / 2 j.
    const unsigned a_part=t%4U;
    const float* a_src[2];float a_scale[2];
#pragma unroll
    for(unsigned j=0;j<2U;++j) {
        const unsigned a=t/4U+(BM/2U)*j,ai=a<count?a:0U;
        const unsigned r=csr.rows[begin+ai],slot=csr.slots[begin+ai];
        a_scale[j]=ldexpf(1.0F,f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,r)[slot]))));
        a_src[j]=at(view.output,r)+2ULL*slot*width+8U*a_part;
    }
    const unsigned b_col=t/TPC,b_part=t%TPC;
    const unsigned char* b_src=weights+(static_cast<unsigned long long>(expert)*hidden+tile*BN+b_col)*groups*block_bytes;
    // Q5_1: head (d | m, qh) + 16 qs bytes; Q8_0: the block's first dword (scale) + the VPT / 4 + 1 code dwords from the
    // block's dword-aligned start (an even block starts at its scale, an odd one two bytes before it).
    constexpr unsigned QD=Q51?4U:VPT/4U+1U;
    struct Regs { uint4 a[2][2]; uint2 head; unsigned q[QD]; };
    const auto fetch=[&](unsigned s,Regs& r) {
        if constexpr(Probe==3U){(void)s;(void)r;return;}
        else {
#pragma unroll
            for(unsigned j=0;j<2U;++j){r.a[j][0]=f16_load(a_src[j]+32U*s);r.a[j][1]=f16_load(a_src[j]+32U*s+4U);}
            if constexpr(Q51) {
                const unsigned char* blk=b_src+24U*s;
                r.head=g3_load8(blk);
                const uint4 q=g3_load16a4(blk+8U);
                r.q[0]=q.x;r.q[1]=q.y;r.q[2]=q.z;r.q[3]=q.w;
            } else {
                const unsigned char* p=b_src+((34U*s)&~3U);
                r.head.x=g3_load4(p);
                if constexpr(VPT==16U) {
                    const uint4 q=g3_load16a4(p+16U*b_part);
                    r.q[0]=q.x;r.q[1]=q.y;r.q[2]=q.z;r.q[3]=q.w;r.q[4]=g3_load4(p+16U*b_part+16U);
                } else {
                    const uint4 q0=g3_load16a4(p),q1=g3_load16a4(p+16U);
                    r.q[0]=q0.x;r.q[1]=q0.y;r.q[2]=q0.z;r.q[3]=q0.w;r.q[4]=q1.x;r.q[5]=q1.y;r.q[6]=q1.z;r.q[7]=q1.w;r.q[8]=g3_load4(p+32U);
                }
            }
        }
    };
    const auto store=[&](unsigned s,const Regs& r,unsigned buf) {
        _Float16* as=lds+buf*S::StageH;
        _Float16* bs=as+BM*S::Pitch;
#pragma unroll
        for(unsigned j=0;j<2U;++j)*reinterpret_cast<uint4*>(as+g3_at(t/4U+(BM/2U)*j,8U*a_part))=f16_from_f32(r.a[j][0],r.a[j][1],a_scale[j]);
        if constexpr(Probe==1U) {
#pragma unroll
            for(unsigned c=0;c<VPT/8U;++c)*reinterpret_cast<uint4*>(bs+g3_at(b_col,VPT*b_part+8U*c))=uint4{r.q[c%QD],r.head.x,r.q[(c+1U)%QD],r.head.y};
            return;
        }
        if constexpr(Q51) {
            // Values 16 p .. 16 p + 15 (p = 0: low nibbles, 1: high nibbles of the 16 qs bytes) with qh bits 16 p ..
            // (down_operands' lo / hi words); both halves when one thread owns the column.
            const float d=scale_f32(static_cast<unsigned short>(r.head.x&65535U)),m=scale_f32(static_cast<unsigned short>(r.head.x>>16U));
            const bool fast=__all(f16_q51_exact(r.head.x));
#pragma unroll
            for(unsigned pp=0;pp<VPT/16U;++pp) {
                const unsigned p=VPT==16U?b_part:pp;
                unsigned w[4];
#pragma unroll
                for(unsigned k=0;k<4U;++k) {
                    const unsigned bits=(r.head.y>>(16U*p+4U*k))&15U;
                    w[k]=(p?(r.q[k]>>4U)&0x0f0f0f0fU:r.q[k]&0x0f0f0f0fU)|((bits*0x00204081U)&0x01010101U)<<4U;
                }
                uint2 x[4];
                if(fast) {
#pragma unroll
                    for(unsigned k=0;k<4U;++k)x[k]=f16_affine_u8_fast(w[k],d*16777216.0F,m);
                } else {
#pragma unroll
                    for(unsigned k=0;k<4U;++k)x[k]=f16_affine_u8(w[k],d,m);
                }
                *reinterpret_cast<uint4*>(bs+g3_at(b_col,16U*p))=uint4{x[0].x,x[0].y,x[1].x,x[1].y};
                *reinterpret_cast<uint4*>(bs+g3_at(b_col,16U*p+8U))=uint4{x[2].x,x[2].y,x[3].x,x[3].y};
            }
        } else {
            // f16_q80_block: codes perm(q[k + 1], q[k], select) for the code dwords k, scale from the first dword.
            const unsigned odd=s&1U,select=odd?0x07060504U:0x05040302U;
            const float d=__half2float(__ushort_as_half(static_cast<unsigned short>((r.head.x>>(odd?16U:0U))&65535U)));
#pragma unroll
            for(unsigned c=0;c<VPT/8U;++c) {
                const uint2 x=f16_scaled_s8(__builtin_amdgcn_perm(r.q[2U*c+1U],r.q[2U*c],select),d);
                const uint2 y=f16_scaled_s8(__builtin_amdgcn_perm(r.q[2U*c+2U],r.q[2U*c+1U],select),d);
                *reinterpret_cast<uint4*>(bs+g3_at(b_col,VPT*b_part+8U*c))=uint4{x.x,x.y,y.x,y.y};
            }
        }
    };
    const unsigned arow0=64U*wm+i;
    const unsigned brow[4]{64U*wn+i,64U*wn+16U+i,64U*wn+32U+i,64U*wn+48U+i};
    const unsigned rbase=64U*wm,mb=count>rbase?min(4U,(count-rbase+15U)/16U):0U;
    const unsigned smb=static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(mb)));
    f32x8 acc[4][4]={};
    g3_pipeline_d<D,Regs>(stages,fetch,store,[&](unsigned buf) {
        const _Float16* as=lds+buf*S::StageH;
        g3_compute<Probe>(as,as+BM*S::Pitch,arow0,brow,smb,acc);
    });
    unsigned* tab=reinterpret_cast<unsigned*>(lds);
    if(t<BM) {
        const bool live=t<count;
        const unsigned r=live?csr.rows[begin+t]:0U,slot=live?csr.slots[begin+t]:0U;
        tab[t]=r;tab[BM+t]=slot;
        tab[2U*BM+t]=static_cast<unsigned>(live?f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,r)[slot]))):0);
    }
    __syncthreads();
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<smb)
#pragma unroll
    for(unsigned u=0;u<4U;++u)
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=rbase+16U*mi+2U*e+h;
        if(a>=count)continue;
        const float unscale=ldexpf(1.0F,-static_cast<int>(tab[2U*BM+a]));
        at(view.rawdots,tab[a])[tab[BM+a]*hidden+tile*BN+64U*wn+16U*u+i]=acc[mi][u][e]*unscale;
    }
}
