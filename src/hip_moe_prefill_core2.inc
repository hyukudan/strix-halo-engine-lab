// qpf2k/moe: routed-expert prefill core2 helpers, included by hip_moe_q4k_a8_known_rows.hip inside its anonymous
// namespace (after F16Stage / f16_pair). Prefill only; decode/window code does not use it.
// qpf2k core2 (STRIX_QPF2K_MOE, tunes 80/82 only): the plane step of f16_step_rows for a wave-uniform tile count MB
// (a template, chosen once per wave), straight-line: B0, B1 and the MB A fragments of a K16 plane are loaded into
// distinct registers before its 2*MB WMMAs, which run in the same (mi, column) order on the same operands and
// accumulators as f16_step_rows (the base compiles each `mi < mb` tile under its own exec mask and serialises the four
// A fragments through one register set: one full LDS round trip per two WMMAs). Core bit 1: plane kk+1's B
// fragments are loaded before plane kk's WMMAs. Operands, K16 order and accumulators are unchanged: same bits.
// A fragment as 8 dwords (two 16-byte LDS loads); bit-cast to 16 halves only as the WMMA operand, so the true16
// code generator does not split every loaded dword into halves (v_lshrrev_b32 + v_mov_b16 per dword).
typedef unsigned f16_u32x8 __attribute__((ext_vector_type(8),aligned(16)));
// Contiguous (unswizzled, Tune bit 3 clear: hi == lo + 8): one 32-byte vector load typed as 8 dwords.
template<bool Contiguous> __device__ __forceinline__ f16_u32x8 f16_frag(const _Float16* row,unsigned lo_half,unsigned hi_half) {
    if constexpr(Contiguous) { (void)hi_half; return *reinterpret_cast<const f16_u32x8*>(row+lo_half); }
    else {
        const uint4 lo=*reinterpret_cast<const uint4*>(row+lo_half),hi=*reinterpret_cast<const uint4*>(row+hi_half);
        return f16_u32x8{lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
    }
}
__device__ __forceinline__ f32x8 f16_wmma(f16_u32x8 a,f16_u32x8 b,f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(f16x16,a),__builtin_bit_cast(f16x16,b),c);
}
template<unsigned MB,unsigned Core,unsigned Batch,unsigned Columns,unsigned Tune>
__device__ __forceinline__ void f16_step_rows_u(const F16Stage<Batch,Columns,Tune>& s,unsigned rbase,const unsigned (&cols)[2],unsigned i,f32x8 (&acc)[4][2]) {
    if constexpr(MB!=0U) {
        constexpr unsigned P=F16Stage<Batch,Columns,Tune>::pitch;
        constexpr bool C=(Tune&8U)==0U;
        const unsigned key=(Tune&8U)?i%8U*8U:0U;
        const _Float16* const b0p=s.b+(cols[0]+i)*P;
        const _Float16* const b1p=s.b+(cols[1]+i)*P;
        const _Float16* const ap=s.a+(rbase+i)*P;
        f16_u32x8 b0=f16_frag<C>(b0p,key,8U^key),b1=f16_frag<C>(b1p,key,8U^key);
#pragma unroll
        for(unsigned kk=0;kk<4U;++kk) {
            const unsigned lo=(16U*kk)^key,hi=(16U*kk+8U)^key,nlo=(16U*kk+16U)^key,nhi=(16U*kk+24U)^key;
            if constexpr((Core&2U)!=0U) {
                // Whole plane in flight: every A fragment and the next plane's B before the plane's WMMAs.
                f16_u32x8 a[MB];
#pragma unroll
                for(unsigned mi=0;mi<MB;++mi)a[mi]=f16_frag<C>(ap+16U*mi*P,lo,hi);
                const f16_u32x8 n0=kk<3U?f16_frag<C>(b0p,nlo,nhi):b0,n1=kk<3U?f16_frag<C>(b1p,nlo,nhi):b1;
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for(unsigned mi=0;mi<MB;++mi) {
                    acc[mi][0]=f16_wmma(a[mi],b0,acc[mi][0]);
                    acc[mi][1]=f16_wmma(a[mi],b1,acc[mi][1]);
                }
                b0=n0;b1=n1;
            } else {
                // Pairwise pipeline: the next A fragment (or, after the last tile, the next plane's B) is in flight
                // during each pair of WMMAs; the schedule barriers keep the loads ahead of the pair.
                f16_u32x8 cur=f16_frag<C>(ap,lo,hi);
#pragma unroll
                for(unsigned mi=0;mi<MB;++mi) {
                    f16_u32x8 next=cur,n0=b0,n1=b1;
                    if(mi+1U<MB)next=f16_frag<C>(ap+16U*(mi+1U)*P,lo,hi);
                    else if(kk<3U){n0=f16_frag<C>(b0p,nlo,nhi);n1=f16_frag<C>(b1p,nlo,nhi);}
                    __builtin_amdgcn_sched_barrier(0);
                    acc[mi][0]=f16_wmma(cur,b0,acc[mi][0]);
                    acc[mi][1]=f16_wmma(cur,b1,acc[mi][1]);
                    __builtin_amdgcn_sched_barrier(0);
                    cur=next;
                    if(mi+1U==MB){b0=n0;b1=n1;}
                }
            }
        }
    }
}
// Core bit 7 (single copy): ONE plane-step body for every tile count (no per-MB pipeline copies: their code growth is the
// t1b suspect). All four A fragments are loaded unconditionally into distinct registers (rows past the batch are staged
// zeros, valid LDS addresses), so the LDS wait counts are path-independent; each WMMA pair runs under the scalar
// (wave-uniform) condition mi < mb, in the base (mi, column) order; a wave with mb == 0 skips the plane step.
template<unsigned Core,unsigned Batch,unsigned Columns,unsigned Tune>
__device__ __forceinline__ void f16_step_rows_s(const F16Stage<Batch,Columns,Tune>& s,unsigned rbase,unsigned mb,const unsigned (&cols)[2],unsigned i,f32x8 (&acc)[4][2]) {
    if(mb==0U)return;
    constexpr unsigned P=F16Stage<Batch,Columns,Tune>::pitch;
    constexpr bool C=(Tune&8U)==0U;
    const unsigned key=(Tune&8U)?i%8U*8U:0U;
    const _Float16* const b0p=s.b+(cols[0]+i)*P;
    const _Float16* const b1p=s.b+(cols[1]+i)*P;
    const _Float16* const ap=s.a+(rbase+i)*P;
    f16_u32x8 b0=f16_frag<C>(b0p,key,8U^key),b1=f16_frag<C>(b1p,key,8U^key);
#pragma unroll
    for(unsigned kk=0;kk<4U;++kk) {
        const unsigned lo=(16U*kk)^key,hi=(16U*kk+8U)^key,nlo=(16U*kk+16U)^key,nhi=(16U*kk+24U)^key;
        f16_u32x8 a[4];
#pragma unroll
        for(unsigned mi=0;mi<4U;++mi)a[mi]=f16_frag<C>(ap+16U*mi*P,lo,hi);
        const f16_u32x8 n0=kk<3U?f16_frag<C>(b0p,nlo,nhi):b0,n1=kk<3U?f16_frag<C>(b1p,nlo,nhi):b1;
#pragma unroll
        for(unsigned mi=0;mi<4U;++mi)if(mi<mb) {
            acc[mi][0]=f16_wmma(a[mi],b0,acc[mi][0]);
            acc[mi][1]=f16_wmma(a[mi],b1,acc[mi][1]);
        }
        b0=n0;b1=n1;
    }
}
// Calls body.template operator()<MB>() for the wave-uniform mb (0..4).
template<class Body> __device__ __forceinline__ void f16_by_mb(unsigned mb,const Body& body) {
    switch(static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(mb)))) { // mb is wave-uniform: scalar branch
    case 4U: body.template operator()<4U>(); break;
    case 3U: body.template operator()<3U>(); break;
    case 2U: body.template operator()<2U>(); break;
    case 1U: body.template operator()<1U>(); break;
    default: body.template operator()<0U>(); break;
    }
}
// Core bit 2: epilogue stores through global (not flat) addresses (flat stores also count in lgkmcnt).
__device__ __forceinline__ void f16_store_global(float* p,float v) {
    *reinterpret_cast<__attribute__((address_space(1))) float*>(reinterpret_cast<std::uintptr_t>(p))=v;
}
// STRIX_QPF2K_MOE (default 0 = the base kernels): core2 bits for the production tunes 80/82 (bit 0 required; bit 1 whole
// plane in flight, bit 2 global epilogue stores, bit 3 128-assignment batches at every T, bits 4-5 prefetch depth: 17/33/49 = 3/4/5 steps, bit 6 256-row gate/up CTAs (tune 82: 65/81/97), bit 7 single-copy plane step (129, 145 = with depth 3), bit 8 with bit 6: 256-row CTAs for paired batches + 128-row CTAs for lone ones (321, 449 = with bit 7), bit 9 down + ordered merge fused by last arrival (513, 641 = with bit 7)); other values select 0.
// 257 (bit 8 alone): qpf2k P1, gate/up and down in one ordered-claim launch (hip_moe_prefill_p1.inc).
// Test hook set_prefill_core2 (-1 = environment).
inline int& prefill_core2_override(){static int v=-1;return v;}
inline bool& prefill_fused_flag(){static thread_local bool v=false;return v;}
// Bit 3 (128-assignment batches below T 4916) is rejected: it changes the CSR summary words (batch count and batch-size
// histogram, t1 gate), although every output tensor is identical.
// qpf2k G3 core (hip_moe_prefill_g3.inc): bit 10 (1024) with bit 0; bit 1 BM 128 (else 256); bit 2 G3 gate/up only (base
// down); bit 4 G3 down only (base gate/up); bit 5 global prefetch depth 2 (else 1; not with probes); bits 11-12 attribution probe (tests only: never taken from the environment).
constexpr bool g3_core_valid(int n){return n>=1025&&n<8192&&(n&1)&&(n&1024)&&(n&~(1|2|4|16|32|1024|2048|4096))==0&&!((n&4)&&(n&16))&&!((n&32)&&(n&6144));}
constexpr bool prefill_core2_valid(int n){return g3_core_valid(n)||n==257||(n>=1&&n<=1023&&(n&1)&&(n<=7||n==17||n==33||n==49||n==65||n==81||n==97||n==129||n==145||n==321||n==449||n==513||n==641));}
inline unsigned prefill_core2() {
    static const unsigned env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE");const int n=v?std::atoi(v):0;
        return prefill_core2_valid(n)&&n<2048?static_cast<unsigned>(n):0U;}();
    const int o=prefill_core2_override();
    return o<0?env:prefill_core2_valid(o)?static_cast<unsigned>(o):0U;
}
// Minimum waves per SIMD for the launch bounds: core2 at tune 82 (16-wave CTAs, 56.8 KB LDS: two CTAs per WGP) keeps
// 8 waves (<= 192 VGPRs); everything else 1 (the HIP default, so the base kernels compile as before).
constexpr unsigned prefill_core2_min_waves(unsigned tune,unsigned core){return core!=0U&&tune==82U?8U:1U;}
// Core bits 4-5: global prefetch depth 2 + ((core >> 4) & 3) K steps (the base pipeline keeps 2 register stages in flight).
// Only when the loads are issued changes; staging order, barriers and the plane steps are those of f16_pipeline.
constexpr unsigned prefill_core2_depth(unsigned core){return 2U+((core>>4U)&3U);}
template<unsigned D,class Regs,class Fetch,class Stage,class Step>
__device__ __forceinline__ void f16_pipeline_d(unsigned steps,const Fetch& fetch,const Stage& stage,const Step& step_fn) {
    // Every iteration issues exactly one fetch per register stage (past the end it re-reads the last step, unused), so the
    // number of loads in flight is the same on every path and the wait-count pass waits only for the oldest stage (a
    // guarded fetch made it drain every load once per loop trip). steps >= D (gate/up 40, down 10).
    Regs r[D];
#pragma unroll
    for(unsigned j=0;j<D;++j)fetch(j,r[j]);
#pragma unroll 1
    for(unsigned step=0;step<steps;step+=D) {
#pragma unroll
        for(unsigned j=0;j<D;++j) {
            const unsigned s=step+j;
            const bool live=s<steps;
            if(live) {
                if(s)f16_barrier();
                stage(s,r[j],0U);
                f16_barrier();
            }
            fetch(min(s+D,steps-1U),r[j]);
            if(live)step_fn(0U);
        }
    }
}
// ---- Core bit 9 (512): down + ordered merge, fused by last arrival (deterministic) ----
// Down writes every raw slot dot exactly as before. After its stores a CTA publishes (fence, then one counter increment per
// (row, column tile) it covered); the CTA whose increment completes a (row, tile) - top_k slots - re-reads the row's top_k
// raw dots of that tile (agent-scope loads after an acquire fence) and runs the unchanged merge chain (current = 0; for slot
// 0..top_k-1: v_fmac_f32 current, weight(SGPR), dot) into the accumulation row (PrefillRows::clear). The merge's guards,
// non-finite flag and the zeroed accumulation of a failed row are applied by merge_check (enqueue_prefill_merge_check), which
// replaces the merge launch. Bits of every merged value equal the merge kernel's: same dots, same weights, same instruction,
// same slot order; which CTA runs it does not matter.
// Fused-merge arguments (a kernel argument of the down; FuseNone for every other core). count: rows x tiles counters + one
// word (index rows*tiles) set when any fused merged value is non-finite - then merge check recomputes the merge exactly as
// the merge kernel does (its header/status race included), so failure outcomes keep the serial path's semantics. The counter
// pool is one per process, like the FP16 A image pool: prefill calls are serialized (same assumption as f16_prefill_a16).
struct FuseArgs { unsigned* count; unsigned top_k, tiles, rows; };
struct FuseNone {};
template<bool F> struct f16_fuse_sel { using type=FuseArgs; };
template<> struct f16_fuse_sel<false> { using type=FuseNone; };
template<unsigned Core> using FuseArgT=typename f16_fuse_sel<(Core&512U)!=0U>::type;
inline unsigned* fuse_counter_pool() {
    static unsigned* const pool=[]{void* p=nullptr;
        check(hipMalloc(&p,(static_cast<std::size_t>(q51_known_rows::prefill_max_rows)*16U+64U)*sizeof(unsigned)),"fused merge counters");
        return static_cast<unsigned*>(p);}();
    return pool;
}
__device__ __forceinline__ float fuse_load(const float* p) {
    return __hip_atomic_load(p,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
}
template<unsigned Batch,unsigned Columns,unsigned Threads,class RowOf>
__device__ __forceinline__ void fuse_merge_tail(const FuseArgs& fa,const PrefillRows& view,unsigned count,unsigned tile,unsigned hidden,const RowOf& row_of) {
    __shared__ unsigned fuse_rows[Batch];
    __shared__ unsigned fuse_n;
    const unsigned t=threadIdx.x;
    __threadfence();          // release this CTA's raw dot stores
    if(t==0U)fuse_n=0U;
    __syncthreads();
    if(t<count) {
        const unsigned r=row_of(t);
        const unsigned old=__hip_atomic_fetch_add(fa.count+static_cast<std::size_t>(r)*fa.tiles+tile,1U,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT);
        if(old==fa.top_k-1U)fuse_rows[atomicAdd(&fuse_n,1U)]=r;
    }
    __syncthreads();
    const unsigned n=fuse_n;
    if(n==0U)return;
    __threadfence();          // acquire the other slots' stores
    constexpr unsigned RowsPerPass=Threads/Columns;
    const unsigned col=tile*Columns+t%Columns;
    bool bad=false;
    for(unsigned k=t/Columns;k<n;k+=RowsPerPass) {
        const unsigned r=fuse_rows[k];
        const float* raw=at(view.rawdots,r);
        const float* w=at(view.route_weights,r);
        float dots[10];
#pragma unroll
        for(unsigned slot=0;slot<10U;++slot)if(slot<fa.top_k)dots[slot]=fuse_load(raw+static_cast<std::size_t>(slot)*hidden+col);
        float current=0.0F;
#pragma unroll
        for(unsigned slot=0;slot<10U;++slot)if(slot<fa.top_k) {
            const float weight=__builtin_bit_cast(float,__builtin_amdgcn_readfirstlane(__builtin_bit_cast(int,w[slot])));
            asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
        }
        at(view.clear,r)[col]=current;
        bad|=(__float_as_uint(current)&0x7fffffffU)>=0x7f7f8000U;
    }
    if(bad)atomicOr(fa.count+static_cast<std::size_t>(fa.rows)*fa.tiles,1U);
}
