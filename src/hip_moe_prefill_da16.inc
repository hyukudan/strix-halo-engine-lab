// qpf2k/moe-mi DA16 (STRIX_QPF2K_MOE_DA16=1; prefill only, tunes 80/82 with STRIX_QPF2K_MOE unset/0; default 0 = the base
// kernels). Included by hip_moe_q4k_a8_known_rows.hip inside its anonymous namespace after hip_moe_prefill_p1.inc.
//
// The down stages A = h (FP32, written by gate/up) scaled by 2^exponent(max |h| of the (row, slot)) and rounded to FP16
// (f16_from_f32) while staging, in every one of the D = hidden / Columns tiles of a batch: each h word is read from L2/DRAM
// D times as FP32 and converted D times (t4 attribution: the down without its A loads is 2.6 ms faster at T 8192).
// DA16: the LAST gate/up CTA of a batch to finish (per-batch arrival counter; release/acquire at agent scope) converts the
// batch's h once into an FP16 image in CSR order (image[(begin + a) * width + k] = the halves the down would stage, from
// the same f16_from_f32 on the same FP32 words with the same scale), and the down stages A as a copy of the image (half the
// bytes, no conversion). Every WMMA operand in LDS is the base's halves, so raw dots, h, max |h|, status and summary are the
// base's bytes. Failure paths: a guard-skipped batch (route/prologue failure, batch >= count) neither counts nor converts;
// the down keeps the base guard (summary[0] != 0 or batch >= count: return), so the image is read only when the base down
// would have run. The image and the counters are private scratch (not a contract view).
// MoE header reuse: STRIX_QPF2K_MOE_HDR=1 launches project_prefill_f16x_da16_mg (same bits; unset 0, decoder injects 1).
inline int& prefill_hdr_override(){static int v=-1;return v;}
inline bool prefill_hdr_reuse(bool gsq) {
    static const bool env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_HDR");return v&&v[0]=='1'&&v[1]=='\0';}();
    const int o=prefill_hdr_override();
    return !qpf2k::suppressed(qpf2k::Item::moe_hdr,gsq)&&(o<0?env:o!=0);
}
// MoE XOR rows (STRIX_QPF2K_MOE_XOR64): the 64-assignment DA16 tune (80) stages its LDS rows XOR-permuted (tune 88,
// F16Stage pitch 64 without padding: 40 KB, three 8-wave CTAs per WGP instead of two) from 3072 prompt rows; operands, WMMAs and
// the K order of every output are tune 80's, so the bytes are identical. Below 3072 rows the third CTA does not pay (measured).
// Same bits; unset 0, decoder injects 1.
inline int& prefill_xor64_override(){static int v=-1;return v;}
inline constexpr unsigned prefill_xor64_min_rows=3072U;
inline bool prefill_xor64(bool gsq) {
    static const bool env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_XOR64");return v&&v[0]=='1'&&v[1]=='\0';}();
    const int o=prefill_xor64_override();
    return !qpf2k::suppressed(qpf2k::Item::moe_xor64,gsq)&&(o<0?env:o!=0);
}
inline int& prefill_da16_override(){static int v=-1;return v;}
inline bool prefill_da16() {
    if(prefill_moe_gsq_off()&1U)return false;
    static const bool env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_DA16");return v&&v[0]=='1'&&v[1]=='\0';}();
    const int o=prefill_da16_override();
    return o<0?env:o==1;
}
// FP16 image of every (row, slot) h in CSR order: prefill_max_rows x max_top_k assignments x max_rows (intermediate) halves.
HipPrefillPool& da16_image_pool(){static HipPrefillPool pool;return pool;}
inline uint4* da16_image(bool setup = false) {
    auto& pool = da16_image_pool();
    return static_cast<uint4*>(pool.get(static_cast<std::size_t>(prefill_pool_rows())*max_top_k*max_rows*2U,
        "DA16 image allocation",setup));
}
// Test hook state: the last enqueue_prefill of this thread took the DA16 path.
inline bool& prefill_da16_flag(){static thread_local bool v=false;return v;}
// Per-batch arrival counters (cleared by one hipMemsetAsync per call); sized as the P1 pool for the largest batch bound.
inline unsigned da16_counter_words(){return static_cast<unsigned>(prefill_pool_rows()*10U/64U+1024U);}
inline unsigned* da16_counters(bool setup = false) {
    static HipPrefillPool pool;
    return static_cast<unsigned*>(pool.get(static_cast<std::size_t>(da16_counter_words())*sizeof(unsigned),
        "DA16 counters",setup,true));
}
// Gate/up: the base body (project_f16x_body, the same source the base kernel includes, Core 0) for (blockIdx.y, blockIdx.x),
// then the arrival count; the last arriving tile of the batch writes the batch's FP16 image.
// Reuse (MoE header reuse, STRIX_QPF2K_MOE_HDR): the body skips reloading the superblock header / Q5_K qh words that the
// register stage already holds (hip_moe_f16x_project_body.inc); the base kernel instantiates Reuse = false.
template<bool Q5K,unsigned Tune,bool Reuse>
__device__ __forceinline__ void project_prefill_f16x_da16_entry(const unsigned char* gate,const unsigned char* up,
    const PrefillRows& view,unsigned hidden,unsigned outputs,const uint4* a16,unsigned* arrive,uint4* image) {
    using L=F16LdsOf<F16TileX<Tune,0U>>;
    __shared__ __align__(16) typename L::Stage sb[1];
    __shared__ typename L::Rows rows;
    __shared__ unsigned last;
    __shared__ float scale_of[F16TileX<Tune,0U>::Batch];
    const unsigned batch=blockIdx.y,t=threadIdx.x;
    project_f16x_body<Q5K,Tune,0U,Reuse>(gate,up,view,hidden,outputs,a16,batch,blockIdx.x,sb,rows);
    const auto& csr=view.csr;
    // The body's own guard (bit 8 masked: it may be set by other CTAs meanwhile, the rest is fixed before the launch).
    if((csr.summary[0]&~projection_nonfinite)||batch>=csr.summary[2])return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent"); // this CTA's h stores and max |h| atomics before its arrival
    __syncthreads();
    if(t==0U)last=__hip_atomic_fetch_add(arrive+batch,1U,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT)==gridDim.x-1U?1U:0U;
    __syncthreads();
    if(!last)return; // CTA-uniform (LDS)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent"); // every tile's h and the final max |h| words
    // rows.row / rows.slot: the body's table of this batch (written before its K loop, never overwritten).
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,per=outputs/8U,total=count*per;
    // The down's a_scale of each (row, slot): ldexpf(1, f16_exponent(final max |h|)) (its rows.exponent, then ldexpf).
    if(t<count)scale_of[t]=ldexpf(1.0F,f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,rows.row[t])[rows.slot[t]]))));
    __syncthreads();
    // Two words per thread and trip, both loads issued before either store.
    for(unsigned i=t;i<total;i+=2U*blockDim.x) {
        const unsigned j=i+blockDim.x;
        const bool two=j<total;
        const unsigned a0=i/per,c0=i%per,a1=two?j/per:a0,c1=two?j%per:c0;
        const float* h0=at(view.output,rows.row[a0])+2ULL*rows.slot[a0]*outputs+8U*c0;
        const float* h1=at(view.output,rows.row[a1])+2ULL*rows.slot[a1]*outputs+8U*c1;
        const uint4 lo0=*reinterpret_cast<const uint4*>(h0),hi0=*reinterpret_cast<const uint4*>(h0+4);
        uint4 lo1{},hi1{};
        if(two){lo1=*reinterpret_cast<const uint4*>(h1);hi1=*reinterpret_cast<const uint4*>(h1+4);}
        image[static_cast<std::size_t>(begin+a0)*per+c0]=f16_from_f32(lo0,hi0,scale_of[a0]);
        if(two)image[static_cast<std::size_t>(begin+a1)*per+c1]=f16_from_f32(lo1,hi1,scale_of[a1]);
    }
}
template<bool Q5K,unsigned Tune>
__global__ __launch_bounds__((F16TileX<Tune,0U>::Threads)) void project_prefill_f16x_da16(const unsigned char* gate,const unsigned char* up,
    PrefillRows view,unsigned hidden,unsigned outputs,const uint4* a16,unsigned* arrive,uint4* image) {
    project_prefill_f16x_da16_entry<Q5K,Tune,false>(gate,up,view,hidden,outputs,a16,arrive,image);
}
template<bool Q5K,unsigned Tune>
__global__ __launch_bounds__((F16TileX<Tune,0U>::Threads)) void project_prefill_f16x_da16_mg(const unsigned char* gate,const unsigned char* up,
    PrefillRows view,unsigned hidden,unsigned outputs,const uint4* a16,unsigned* arrive,uint4* image) {
    project_prefill_f16x_da16_entry<Q5K,Tune,true>(gate,up,view,hidden,outputs,a16,arrive,image);
}
// Down: down_prefill_f16x (Core 0, no probe bits) with A staged from the image. B, the K chain and the epilogue are the base's.
template<bool Q51,unsigned Tune>
__global__ __launch_bounds__(F16Tile<Tune>::Threads) void down_prefill_f16x_da16(const unsigned char* weights,PrefillRows view,
    unsigned hidden,unsigned width,const uint4* image) {
#define STRIX_DA16_BATCH_TILE const unsigned batch=blockIdx.y,tile=blockIdx.x;
#define STRIX_DA16_GUARD if(csr.summary[0]||batch>=csr.summary[2])return;
#define STRIX_DA16_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
#include "hip_moe_da16_down_body.inc"
#undef STRIX_DA16_BATCH_TILE
#undef STRIX_DA16_GUARD
#undef STRIX_DA16_LDS
}

// Certificate entry only; base entry and the K pipeline remain unchanged.
template<bool Q51,unsigned Tune>
__global__ __launch_bounds__(F16Tile<Tune>::Threads) void down_prefill_f16x_da16_certificate(const unsigned char* weights,PrefillRows view,
    unsigned hidden,unsigned width,const uint4* image) {
    constexpr bool RouteBF16=false;
#define STRIX_DA16_BATCH_TILE const unsigned batch=blockIdx.y,tile=blockIdx.x;
#define STRIX_DA16_GUARD if(csr.summary[0]||batch>=csr.summary[2])return;
#define STRIX_DA16_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
#define STRIX_DA16_CERTIFICATE
#include "hip_moe_da16_down_body.inc"
#undef STRIX_DA16_CERTIFICATE
#undef STRIX_DA16_BATCH_TILE
#undef STRIX_DA16_GUARD
#undef STRIX_DA16_LDS
}

// Candidate-only entry; the incumbent certificate kernel retains its name and specialization.
template<bool Q51,unsigned Tune>
__global__ __launch_bounds__(F16Tile<Tune>::Threads) void down_prefill_f16x_da16_certificate_bf16(const unsigned char* weights,PrefillRows view,
    unsigned hidden,unsigned width,const uint4* image) {
    constexpr bool RouteBF16=true;
#define STRIX_DA16_BATCH_TILE const unsigned batch=blockIdx.y,tile=blockIdx.x;
#define STRIX_DA16_GUARD if(csr.summary[0]||batch>=csr.summary[2])return;
#define STRIX_DA16_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
#define STRIX_DA16_CERTIFICATE
#include "hip_moe_da16_down_body.inc"
#undef STRIX_DA16_CERTIFICATE
#undef STRIX_DA16_BATCH_TILE
#undef STRIX_DA16_GUARD
#undef STRIX_DA16_LDS
}
