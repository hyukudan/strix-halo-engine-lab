__device__ __forceinline__ f16x16 direct_b_decode(uint4 q,unsigned half,float a,float b,bool fast_block) {
    const unsigned words[2]{half?q.z:q.x,half?q.w:q.y};
    f16x16 local;
#pragma unroll
    for(unsigned w=0;w<2U;++w) {
        const unsigned x=words[w];
        const unsigned even=x&0x0f0f0f0fU,odd=(x>>4U)&0x0f0f0f0fU;
        uint4 decoded;
        if(fast_block) {
            // Put adjacent low/high nibbles directly into FP16 subnormal pairs.
            // Four permutations per word instead of interleaving bytes first.
            const float a2=a*16777216.0F;
            decoded={QwenKDecodeOps::f16_mix_pair(a2,__builtin_amdgcn_perm(odd,even,0x0c040c00U),b),
                QwenKDecodeOps::f16_mix_pair(a2,__builtin_amdgcn_perm(odd,even,0x0c050c01U),b),
                QwenKDecodeOps::f16_mix_pair(a2,__builtin_amdgcn_perm(odd,even,0x0c060c02U),b),
                QwenKDecodeOps::f16_mix_pair(a2,__builtin_amdgcn_perm(odd,even,0x0c070c03U),b)};
        } else {
            const unsigned c0=__builtin_amdgcn_perm(odd,even,0x05010400U);
            const unsigned c1=__builtin_amdgcn_perm(odd,even,0x07030602U);
            const uint2 lo=QwenKDecodeOps::f16_affine_u8(c0,a,b),hi=QwenKDecodeOps::f16_affine_u8(c1,a,b);
            decoded={lo.x,lo.y,hi.x,hi.y};
        }
        __builtin_memcpy(reinterpret_cast<char*>(&local)+16U*w,&decoded,16U);
    }
    return local;
}
__device__ __forceinline__ void direct_b_exchange(f16x16 local,unsigned h,f16x16& b0,f16x16& b1) {
    f16x16 peer;
    typedef unsigned U8 __attribute__((ext_vector_type(8)));
    const U8 lw=__builtin_bit_cast(U8,local);
    U8 pw;
#pragma unroll
    for(unsigned w=0;w<8U;++w)
        // Identity selectors in the opposite 16-lane row: lane XOR 16.
        pw[w]=STRIX_HIP_EXCHANGE16(0U,lw[w],false,false);
    peer=__builtin_bit_cast(f16x16,pw);
    b0=h?peer:local;b1=h?local:peer;
}
__device__ __forceinline__ void direct_b_decode_pair(uint4 q,unsigned h,float a,float b,bool fast_block,
    f16x16& even,f16x16& odd) {
    // The two replicas decode adjacent K16 planes of one K32 code word.
    // Exchange FP16 bits so both WMMA replicas see each plane unchanged.
    direct_b_exchange(direct_b_decode(q,h,a,b,fast_block),h,even,odd);
}
template<unsigned Distance>
__device__ __forceinline__ float direct_b_max_xor(float m) {
    // gfx1151 row_xmask has exactly the shuffle's lane XOR mapping for D < 16.
    const int peer=__builtin_amdgcn_mov_dpp(__float_as_int(m),static_cast<int>(0x160U|Distance),0xf,0xf,false);
    return fmaxf(m,__int_as_float(peer));
}

template<unsigned Tune> struct DirectBTile {
    static constexpr unsigned Batch=F16Tile<Tune>::Batch,Outputs=128U,Groups=Outputs/16U;
    static constexpr bool Split=Tune==82U;
    static constexpr unsigned Matrices=Split?1U:2U;
    static constexpr unsigned Rows=Batch/16U,Threads=32U*Groups*(Split?2U:1U),AChunks=Batch*8U/Threads;
};
// Each wave reuses a decoded B fragment across the entire assignment batch.
// Batch-128 uses separate gate/up waves to halve the accumulator registers and
// share adjacent-plane decode across replicas. Their epilogue rendezvous reuses
// the A storage.
template<unsigned Tune,bool DA16>
__device__ __forceinline__ void project_prefill_direct_b_body(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs,const uint4* a16,unsigned* arrive,uint4* image) {
    using T=DirectBTile<Tune>;
    static_assert(T::Threads==(T::Split?512U:256U));
    constexpr unsigned Batch=T::Batch,Outputs=T::Outputs;
    const unsigned batch=blockIdx.y,tile=blockIdx.x;
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||batch>=csr.summary[2])return;
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,expert=csr.experts[batch];
    struct AStage {
        _Float16 a[Batch*64U];
        __device__ uint4* a16(unsigned row,unsigned half){return reinterpret_cast<uint4*>(a+f16_lds<8U>(row,half));}
    };
    union Stage {
        AStage input;
        float up[32U*Outputs];
    };
    static_assert(!T::Split||sizeof(Stage)==sizeof(AStage));
    // Do not enlarge the batch-64 staging allocation for the split epilogue.
    __shared__ __align__(16) typename std::conditional<T::Split,Stage,AStage>::type s;
    __shared__ F16Rows<Batch> rows;
    __shared__ unsigned last;
    __shared__ float scale_of[DA16?Batch:1U];
    const unsigned t=threadIdx.x,lane=t&31U,i=lane&15U,h=lane>>4U;
    const unsigned cg=(t/32U)%T::Groups;
    const bool up_wave=T::Split&&t/32U>=T::Groups;
    if(t<Batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=live?csr.slots[begin+t]:0U;rows.exponent[t]=live?at(view.sums,r)[0]:0;
    }
    __syncthreads();
    const auto row_of=[&](unsigned a)->unsigned{return rows.row[a];};
    const auto slot_of=[&](unsigned a)->unsigned{return rows.slot[a];};
    const auto exp_of=[&](unsigned a)->int{return rows.exponent[a];};
    const unsigned mb=min(T::Rows,(count+15U)/16U);
    const unsigned blocks=hidden/256U,steps=hidden/64U;
    const unsigned out=tile*Outputs+16U*cg+i;
    const auto* b_src=((T::Split?up_wave:h!=0U)?up:gate)+((static_cast<unsigned long long>(expert)*(outputs/16U)+out/16U)*blocks)*2304U;
    // Two compressed K64 register slots, with one K64 of lookahead.
    // Paired waves load different matrices in their two lane halves. Split waves
    // load one matrix and decode different adjacent planes in the two halves.
    struct Regs {uint4 head{},q0{},q1{};};
    const auto fetch=[&](unsigned step,Regs& r,const Regs& previous) {
        const auto* unit=b_src+(step/4U)*2304U;
        r.head=step%4U==0U?f16_load(unit+16U*i):previous.head;
        r.q0=f16_load(unit+256U+((2U*(step%4U))*16U+i)*16U);
        r.q1=f16_load(unit+256U+((2U*(step%4U)+1U)*16U+i)*16U);
    };
    bool fast_block=false;
    f32x8 acc[T::Rows][T::Matrices]={};
    Regs r0,r1;
    const auto a_stage=[&]()->AStage& {
        if constexpr(T::Split)return s.input;
        else return s;
    };
    const auto consume=[&](unsigned step,const Regs& r) {
        if(step)f16_barrier();
#pragma unroll
        for(unsigned j=0;j<T::AChunks;++j) {
            const unsigned c=t+T::Threads*j,ar=c/8U,ah=8U*(c%8U);
            uint4 v{};
            if(ar<count) {
                if constexpr((Tune&16U)!=0U)v=f16_load(a16+static_cast<std::size_t>(row_of(ar))*(hidden/8U)+8U*step+ah/8U);
                else v=f16_from_bf16(f16_load(reinterpret_cast<const unsigned char*>(at(view.input,row_of(ar)))+128U*step+2U*ah),ldexpf(1.0F,exp_of(ar)));
            }
            *a_stage().a16(ar,ah)=v;
        }
        const uint4 head=r.head;
        if(step%4U==0U) {
            if constexpr((Tune&64U)!=0U)fast_block=__all(QwenKDecodeOps::f16_affine_exact(head.x&65535U,63U*15U,head.x>>16U,63U));
        }
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(head.x&65535U)));
        const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(head.x>>16U)));
        f16_barrier();
        // A's global-to-LDS staging is complete before issuing the lookahead,
        // so its VMEM waits cannot drain the next compressed B slot.
        if(step+1U<steps)fetch(step+1U,r1,r);
        const auto multiply=[&](unsigned kk,f16x16 b0,f16x16 b1) {
            const unsigned key=i%8U*8U,lo=(16U*kk)^key,hi=(16U*kk+8U)^key;
#pragma unroll
            for(unsigned mi=0;mi<T::Rows;++mi)if(mi<mb) {
                // Keep each A operand beside its WMMA calls. Hoisting all eight
                // row operands across decode/fallback needlessly extends them.
                const f16x16 av=f16_pair(a_stage().a+(16U*mi+i)*64U,lo,hi);
                acc[mi][0]=STRIX_HIP_WMMA_F16(av,b0,acc[mi][0]);
                if constexpr(T::Split) {
                    __asm__ volatile("" :: "v"(acc[mi][0]) : "memory");
                } else {
                    acc[mi][1]=STRIX_HIP_WMMA_F16(av,b1,acc[mi][1]);
                    __asm__ volatile("" :: "v"(acc[mi][0]),"v"(acc[mi][1]) : "memory");
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        };
#pragma unroll
        for(unsigned kk=0;kk<4U;kk+=(T::Split?2U:1U)) {
            const unsigned group=2U*(step%4U)+kk/2U;
            const float a=d*static_cast<float>(factor(head,group,false)),b=-(dm*static_cast<float>(factor(head,group,true)));
            f16x16 b0,b1;
            if constexpr(T::Split) {
                direct_b_decode_pair(kk<2U?r.q0:r.q1,h,a,b,fast_block,b0,b1);
                // Preserve every output's ascending K16 accumulator chain.
                multiply(kk,b0,b0);
                multiply(kk+1U,b1,b1);
            } else {
                direct_b_exchange(direct_b_decode(kk<2U?r.q0:r.q1,kk&1U,a,b,fast_block),h,b0,b1);
                multiply(kk,b0,b1);
            }
        }
    };
    fetch(0U,r0,r1);
#pragma unroll 1
    for(unsigned step=0;step<steps;++step) {
        consume(step,r0);
        r0=r1;
    }
    const auto finish=[&](unsigned mi) {
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=16U*mi+2U*e+h;
        const bool live=a<count;
        const unsigned r=row_of(live?a:0U),slot=slot_of(live?a:0U);
        const float unscale=ldexpf(1.0F,-exp_of(live?a:0U));
        const float g=acc[mi][0][e]*unscale;
        float u;
        if constexpr(T::Split)u=s.up[((mi%2U)*16U+2U*e+h)*Outputs+16U*cg+i]*unscale;
        else u=acc[mi][1][e]*unscale;
        const float hv=(g/(1.0F+expf(-g)))*u;
        const bool bad=!isfinite(g)||!isfinite(u)||!isfinite(hv);
        if(live) {
            at(view.output,r)[slot*2U*outputs+out]=hv;
        }
        float m=fabsf(hv);
        m=direct_b_max_xor<8U>(m);m=direct_b_max_xor<4U>(m);
        m=direct_b_max_xor<2U>(m);m=direct_b_max_xor<1U>(m);
        if(live&&i==0U)atomicMax(reinterpret_cast<unsigned*>(at(view.down_sums,r))+slot,__float_as_uint(m));
        if(live&&bad){atomicOr(at(view.status,r),projection_nonfinite);atomicOr(csr.summary,projection_nonfinite);}
    }
    };
    if constexpr(T::Split) {
        // Two row16 tiles fit exactly in A's existing 16 KiB allocation. Every
        // barrier is CTA-uniform, including partial batches. The first waits for
        // final A reads; later barriers protect gate reads before the next write.
#pragma unroll
        for(unsigned pair=0;pair<T::Rows/2U;++pair)if(2U*pair<mb) {
            f16_barrier();
            if(up_wave) {
#pragma unroll
                for(unsigned j=0;j<2U;++j)if(2U*pair+j<mb)
#pragma unroll
                for(unsigned e=0;e<8U;++e)
                    s.up[(16U*j+2U*e+h)*Outputs+16U*cg+i]=acc[2U*pair+j][0][e];
            }
            f16_barrier();
            if(!up_wave) {
#pragma unroll
                for(unsigned j=0;j<2U;++j)if(2U*pair+j<mb)finish(2U*pair+j);
            }
        }
    } else {
#pragma unroll
        for(unsigned mi=0;mi<T::Rows;++mi)if(mi<mb)finish(mi);
    }
    if constexpr(DA16) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent"); // this CTA's h stores and max |h| atomics before its arrival
    __syncthreads();
    if(t==0U)last=__hip_atomic_fetch_add(arrive+batch,1U,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT)==gridDim.x-1U?1U:0U;
    __syncthreads();
    if(!last)return; // CTA-uniform (LDS)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent"); // every tile's h and the final max |h| words
    // rows.row / rows.slot: the body's table of this batch (written before its K loop, never overwritten).
    const unsigned per=outputs/8U,total=count*per;
    // The down's a_scale of each (row, slot): ldexpf(1, f16_exponent(final max |h|)) (its rows.exponent, then ldexpf).
    if(t<count)scale_of[t]=ldexpf(1.0F,f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,rows.row[t])[rows.slot[t]]))));
    __syncthreads();
    // Two words per thread and trip, both loads issued before either store.
    for(unsigned item=t;item<total;item+=2U*blockDim.x) {
        const unsigned j=item+blockDim.x;
        const bool two=j<total;
        const unsigned a0=item/per,c0=item%per,a1=two?j/per:a0,c1=two?j%per:c0;
        const float* h0=at(view.output,rows.row[a0])+2ULL*rows.slot[a0]*outputs+8U*c0;
        const float* h1=at(view.output,rows.row[a1])+2ULL*rows.slot[a1]*outputs+8U*c1;
        const uint4 lo0=*reinterpret_cast<const uint4*>(h0),hi0=*reinterpret_cast<const uint4*>(h0+4);
        uint4 lo1{},hi1{};
        if(two){lo1=*reinterpret_cast<const uint4*>(h1);hi1=*reinterpret_cast<const uint4*>(h1+4);}
        image[static_cast<std::size_t>(begin+a0)*per+c0]=f16_from_f32(lo0,hi0,scale_of[a0]);
        if(two)image[static_cast<std::size_t>(begin+a1)*per+c1]=f16_from_f32(lo1,hi1,scale_of[a1]);
    }
    }
}

// Concrete entries keep host registration independent of generic launchers.
#define STRIX_DIRECT_B_ARGS const unsigned char* gate,const unsigned char* up,PrefillRows view,unsigned hidden,unsigned outputs,const uint4* a16,unsigned* arrive,uint4* image
#define STRIX_DIRECT_B_VALUES gate,up,view,hidden,outputs,a16,arrive,image
#define STRIX_DIRECT_B_ENTRY(Name,Tune,DA16) \
__global__ __launch_bounds__(DirectBTile<Tune>::Threads,2) void Name(STRIX_DIRECT_B_ARGS) { \
    project_prefill_direct_b_body<Tune,DA16>(STRIX_DIRECT_B_VALUES); \
}
STRIX_DIRECT_B_ENTRY(project_prefill_direct_b_80,80U,false)
STRIX_DIRECT_B_ENTRY(project_prefill_direct_b_80_da16,80U,true)
STRIX_DIRECT_B_ENTRY(project_prefill_direct_b_82,82U,false)
STRIX_DIRECT_B_ENTRY(project_prefill_direct_b_82_da16,82U,true)
STRIX_DIRECT_B_ENTRY(project_prefill_direct_b_88,88U,false)
STRIX_DIRECT_B_ENTRY(project_prefill_direct_b_88_da16,88U,true)
#undef STRIX_DIRECT_B_ENTRY
template<unsigned Tune,bool DA16> constexpr auto direct_b_kernel() {
    static_assert(Tune==80U||Tune==82U||Tune==88U);
    if constexpr(Tune==80U) {if constexpr(DA16)return project_prefill_direct_b_80_da16;else return project_prefill_direct_b_80;}
    else if constexpr(Tune==82U) {if constexpr(DA16)return project_prefill_direct_b_82_da16;else return project_prefill_direct_b_82;}
    else {if constexpr(DA16)return project_prefill_direct_b_88_da16;else return project_prefill_direct_b_88;}
}
template<unsigned Tune,bool DA16>
void launch_prefill_direct_b(dim3 grid,hipStream_t stream,STRIX_DIRECT_B_ARGS) {
#define STRIX_DIRECT_B_LAUNCH(Name) STRIX_HIP_LAUNCH_GGL(Rdna3Layout, Name,grid,dim3(DirectBTile<Tune>::Threads),0,stream,STRIX_DIRECT_B_VALUES)
    static_assert(Tune==80U||Tune==82U||Tune==88U);
    if constexpr(Tune==80U) {if constexpr(DA16)STRIX_DIRECT_B_LAUNCH(project_prefill_direct_b_80_da16);else STRIX_DIRECT_B_LAUNCH(project_prefill_direct_b_80);}
    else if constexpr(Tune==82U) {if constexpr(DA16)STRIX_DIRECT_B_LAUNCH(project_prefill_direct_b_82_da16);else STRIX_DIRECT_B_LAUNCH(project_prefill_direct_b_82);}
    else {if constexpr(DA16)STRIX_DIRECT_B_LAUNCH(project_prefill_direct_b_88_da16);else STRIX_DIRECT_B_LAUNCH(project_prefill_direct_b_88);}
#undef STRIX_DIRECT_B_LAUNCH
}
#undef STRIX_DIRECT_B_VALUES
#undef STRIX_DIRECT_B_ARGS
