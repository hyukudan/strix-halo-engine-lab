// qpf2k/moe-mi: routed prefill prologue and CSR in fewer serial steps (STRIX_QPF2K_MOE_ROUTE, default 0 = the base kernels).
// Included by hip_moe_q4k_a8_known_rows.hip inside its anonymous namespace after f16_prefill_prologue and build_prefill_csr_waves.
// Every word they write is the base kernels' word (same values, same set of written words, failure paths included); only
// the order of independent work and the number of serial memory round trips change.
//
// Bit 1: build_prefill_csr_waves_v2. The base ends with one thread per expert walking that expert's whole CSR row list from
// global memory (`unique` = distinct rows of the expert): a serial chain as long as the largest expert (thousands of dependent
// L2 round trips at a skewed T 8192). The rows of one prompt row are adjacent in its expert's list (assignments ascend by
// a = r * top_k + s), so the distinct rows of expert e = n_e - #{(r, s): an earlier slot s' < s of row r also routes to e}.
// The first phase counts those repeats per expert in LDS (no atomic at all without repeated routes); unique, duplicates and
// the histograms are then the base's values. Everything else is build_prefill_csr_waves token for token.
//
// Bit 2: f16_prefill_prologue_v2. The base runs clear -> barrier -> route checks + a 2-byte strided max pass -> barrier ->
// thread 0 exponent -> barrier -> a second read of the row to convert it. v2 issues the row's 16-byte loads first and keeps
// them in registers, clears with 16-byte stores (when the clear view is 16-byte aligned, else the base loop), checks routes
// with one wave ballot, and every thread folds the 8 wave maxima itself: one barrier, one read of the row. The max is the
// same value (fmaxf over the same set of |x|: order-free, NaNs ignored the same way, start 0), so the exponent, the FP16
// image (f16_from_bf16 on the same words), sums, status and the zeroed max |h| words are the base's bytes.
// Bit 8: the prologue does not zero the accumulation view (84 MB per call at T 8192); the caller's merge writes those zeros
// on its skip paths instead (known_moe_prefill_merge_kernel_z: header[0] or the row status set, or an invalid route), where
// the base merge returns without a store. A successful row is overwritten by the merge in both paths, so the view holds the
// same bytes after the merge in every case (only when clear_words == hidden, i.e. exactly the words the merge covers).
inline bool& prefill_clear_skipped_flag(){static thread_local bool v=false;return v;}
inline int& prefill_route_override(){static int v=-1;return v;}
inline unsigned prefill_route() {
    static const unsigned env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_ROUTE");const int n=v?std::atoi(v):0;
        return n>=0&&n<=15?static_cast<unsigned>(n):0U;}();
    const int o=prefill_route_override();
    return o<0?env:(o>=0&&o<=15?static_cast<unsigned>(o):0U);
}
__global__ __launch_bounds__(1024) void build_prefill_csr_waves_v2(PrefillRows v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap) {
    constexpr unsigned Waves=32U;
    const auto& csr=v.csr;
    const unsigned t=threadIdx.x,lane=t&31U,wave=t/32U,count=rows*top_k,span=(count+Waves-1U)/Waves;
    const unsigned first=wave*span,last=min(count,first+span);
    __shared__ unsigned position[Waves][256],n_of[512],repeat_of[512],scan[3][512],hist[16],failure,status,duplicates,maximum;
    if(t==0){failure=0U;status=0U;duplicates=0U;maximum=0U;}
    if(t<16U)hist[t]=0U;
    if(t<512U)repeat_of[t]=0U;
    for(unsigned k=t;k<Waves*256U;k+=1024U)position[k/256U][k%256U]=0U;
    __syncthreads();
    const auto expert_at=[&](unsigned a){return at(v.routes,a/top_k)[a%top_k];};
    for(unsigned a=first+lane;a<last;a+=32U) {
        const unsigned e=expert_at(a);
        if(e>=experts||!isfinite(at(v.route_weights,a/top_k)[a%top_k]))atomicOr(&failure,invalid_route);
        else {
            atomicAdd(&position[wave][e/2U],1U<<(16U*(e&1U)));
            // A repeat: an earlier slot of the same prompt row routes to the same expert (not a new distinct row of e).
            const unsigned* const route=at(v.routes,a/top_k);
            bool repeat=false;
            for(unsigned s=0;s<a%top_k;++s)repeat|=route[s]==e;
            if(repeat)atomicAdd(&repeat_of[e],1U);
        }
    }
    for(unsigned r=t;r<rows;r+=1024U){const unsigned s=*at(v.status,r);if(s)atomicOr(&status,s);}
    __syncthreads();
    if(failure){if(t<32U)csr.summary[t]=t==0U?failure|status:t==3U?count:t==6U?rows:t==7U?1U:0U;return;}
    if(t<256U) {
        unsigned lo=0U,hi=0U;
        for(unsigned w=0;w<Waves;++w){const unsigned c=position[w][t];position[w][t]=lo|hi<<16U;lo+=c&65535U;hi+=c>>16U;}
        if(lo>65535U||hi>65535U)atomicOr(&failure,invalid_route);
        n_of[2U*t]=lo;n_of[2U*t+1U]=hi;
    }
    __syncthreads();
    if(failure){if(t<32U)csr.summary[t]=t==0U?failure|status:t==3U?count:t==6U?rows:t==7U?1U:0U;return;}
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
            const unsigned slot=scan[0][e]-n_of[e]+((position[wave][e/2U]>>(16U*(e&1U)))&65535U);
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
            const unsigned unique=n-repeat_of[t];
            atomicAdd(&hist[min(8U,unique)-1U],1U);atomicAdd(&duplicates,n-unique);
        }
    }
    __syncthreads();
    if(t==0U)csr.begins[scan[2][511]]=count;
    if(t<32U)csr.summary[t]=t==0U?status:t==1U?scan[1][511]:t==2U?scan[2][511]:t==3U?count:t==4U?duplicates:t==5U?maximum:
        t==6U?rows:t==7U?1U:t>=8U&&t<24U?hist[t-8U]:0U;
}
// Chunks = 16-byte words of a row per thread (max_hidden / 8 / 256, rounded up). Requires: hidden % 8 == 0, 16-byte aligned
// input rows, top_k <= 32 (host checks; else the base prologue runs). clear16: the clear view is 16-byte aligned and
// clear_words % 4 == 0 (else the base clear loop, same words).
constexpr unsigned route_prologue_chunks=(max_hidden/8U+255U)/256U;
__global__ __launch_bounds__(256) void f16_prefill_prologue_v2(PrefillRows v,unsigned hidden,unsigned top_k,unsigned experts,
    unsigned clear_words,uint4* a16,unsigned clear16) {
    const unsigned r=blockIdx.x,t=threadIdx.x,n16=hidden/8U;
    __shared__ float wave_max[8];
    const auto* x16=reinterpret_cast<const uint4*>(at(v.input,r));
    uint4 x[route_prologue_chunks];
#pragma unroll
    for(unsigned j=0;j<route_prologue_chunks;++j){const unsigned k=t+256U*j;x[j]=k<n16?x16[k]:uint4{0U,0U,0U,0U};}
    float* clear=at(v.clear,r);
    if(clear16){uint4* c4=reinterpret_cast<uint4*>(clear);for(unsigned i=t;i<clear_words/4U;i+=256U)c4[i]=uint4{0U,0U,0U,0U};}
    else for(unsigned i=t;i<clear_words;i+=256U)clear[i]=0.0F;
    unsigned route_bits=0U;
    if(t<32U) {
        const bool bad=t<top_k&&(at(v.routes,r)[t]>=experts||!isfinite(at(v.route_weights,r)[t]));
        route_bits=__ballot(bad)?invalid_route:0U;
    }
    if(t<top_k)at(v.down_sums,r)[t]=0;
    float m=0.0F;
    const auto fold=[&](unsigned w){m=fmaxf(m,fabsf(__uint_as_float(w<<16U)));m=fmaxf(m,fabsf(__uint_as_float(w&0xffff0000U)));};
#pragma unroll
    for(unsigned j=0;j<route_prologue_chunks;++j){fold(x[j].x);fold(x[j].y);fold(x[j].z);fold(x[j].w);}
    for(int stride=16;stride!=0;stride/=2)m=fmaxf(m,__shfl_xor(m,stride,32));
    if((t&31U)==0U)wave_max[t/32U]=m;
    __syncthreads();
    m=wave_max[0];
    for(unsigned w=1;w<8U;++w)m=fmaxf(m,wave_max[w]);
    const int row_exponent=f16_exponent(m);
    if(t==0U) {
        at(v.sums,r)[0]=row_exponent;
        *at(v.status,r)=route_bits;
    }
    if(!a16)return;
    const float scale=ldexpf(1.0F,row_exponent);
    uint4* out=a16+static_cast<std::size_t>(r)*(hidden/8U);
#pragma unroll
    for(unsigned j=0;j<route_prologue_chunks;++j){const unsigned k=t+256U*j;if(k<n16)out[k]=f16_from_bf16(x[j],scale);}
}
// Bit 4 (supersedes bit 1): the CSR in three multi-CTA kernels instead of one 1024-thread CTA (r1: the one-CTA builder runs
// 0.32-0.57 ms per call against a data floor of a few us; one WGP of 20 works). Same words as build_prefill_csr_waves:
//  count  : B CTAs, CTA k owns the assignments [k*S, (k+1)*S) (S = 32 waves x 64): per-expert counts -> blockcount[k][e],
//           route validity -> fail word, row statuses -> status word, per-row repeats -> repeat[e] (global, rare atomics);
//  scan   : one CTA: n_e = sum_k blockcount[k][e] and blockcount[k][e] := exclusive prefix over k; the 65535 check; then the
//           base builder's scans, offsets, touched, batches, histograms (unique = n - repeat) and summary words, token for
//           token; on a failure only the base's failure summary (and the fail word, so scatter writes nothing);
//  scatter: B CTAs: position of assignment a = offsets[e] + blockcount[k][e] + #{a' < a in CTA k with expert e} (per-wave
//           counts, exclusive over the CTA's waves, ballot ranks over lanes as the base) -> rows/slots.
// Each expert's assignments ascend by a exactly as in the base (CTA order, then wave order, then lane order).
constexpr unsigned csr_mw_span=32U*64U; // assignments per CTA
inline unsigned csr_mw_blocks_max(){return (q51_known_rows::prefill_max_rows*10U+csr_mw_span-1U)/csr_mw_span;}
// Scratch: [0] fail, [1] status, [2..514) repeat[512], then blockcount[blocks_max][512]. The first 514 words are cleared per call.
constexpr unsigned csr_mw_head=2U+512U;
inline unsigned* csr_mw_scratch() {
    static unsigned* const pool=[]{void* p=nullptr;
        check(hipMalloc(&p,(static_cast<std::size_t>(csr_mw_head)+static_cast<std::size_t>(csr_mw_blocks_max())*512U)*sizeof(unsigned)),"CSR scratch");
        return static_cast<unsigned*>(p);}();
    return pool;
}
__global__ __launch_bounds__(1024) void csr_mw_count(PrefillRows v,unsigned rows,unsigned experts,unsigned top_k,unsigned* scratch) {
    const unsigned t=threadIdx.x,count=rows*top_k,k=blockIdx.x,first=k*csr_mw_span,last=min(count,first+csr_mw_span);
    __shared__ unsigned n_of[512],failure,status;
    if(t<512U)n_of[t]=0U;
    if(t==0U){failure=0U;status=0U;}
    __syncthreads();
    for(unsigned a=first+t;a<last;a+=1024U) {
        const unsigned r=a/top_k,s=a%top_k,e=at(v.routes,r)[s];
        if(e>=experts||!isfinite(at(v.route_weights,r)[s]))atomicOr(&failure,invalid_route);
        else {
            atomicAdd(&n_of[e],1U);
            const unsigned* const route=at(v.routes,r);
            bool repeat=false;
            for(unsigned q=0;q<s;++q)repeat|=route[q]==e;
            if(repeat)atomicAdd(scratch+2U+e,1U);
        }
    }
    const unsigned per=(rows+gridDim.x-1U)/gridDim.x;
    for(unsigned r=k*per+t;r<min(rows,(k+1U)*per);r+=1024U){const unsigned s=*at(v.status,r);if(s)atomicOr(&status,s);}
    __syncthreads();
    if(t==0U){if(failure)atomicOr(scratch,failure);if(status)atomicOr(scratch+1U,status);}
    if(t<512U)scratch[csr_mw_head+k*512U+t]=n_of[t];
}
__global__ __launch_bounds__(1024) void csr_mw_scan(PrefillRows v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap,unsigned blocks,unsigned* scratch) {
    const auto& csr=v.csr;
    const unsigned t=threadIdx.x,count=rows*top_k;
    __shared__ unsigned n_of[512],scan[3][512],hist[16],failure,status,duplicates,maximum;
    if(t==0){failure=scratch[0];status=scratch[1];duplicates=0U;maximum=0U;}
    if(t<16U)hist[t]=0U;
    __syncthreads();
    if(failure){if(t<32U)csr.summary[t]=t==0U?failure|status:t==3U?count:t==6U?rows:t==7U?1U:0U;return;}
    if(t<512U) {
        unsigned n=0U;
        for(unsigned k=0;k<blocks;++k){unsigned* const c=scratch+csr_mw_head+k*512U+t;const unsigned x=*c;*c=n;n+=x;}
        if(n>65535U)atomicOr(&failure,invalid_route);
        n_of[t]=n;
    }
    __syncthreads();
    if(failure){if(t==0U)atomicOr(scratch,failure);if(t<32U)csr.summary[t]=t==0U?failure|status:t==3U?count:t==6U?rows:t==7U?1U:0U;return;}
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
            const unsigned unique=n-scratch[2U+t];
            atomicAdd(&hist[min(8U,unique)-1U],1U);atomicAdd(&duplicates,n-unique);
        }
    }
    __syncthreads();
    if(t==0U)csr.begins[scan[2][511]]=count;
    if(t<32U)csr.summary[t]=t==0U?status:t==1U?scan[1][511]:t==2U?scan[2][511]:t==3U?count:t==4U?duplicates:t==5U?maximum:
        t==6U?rows:t==7U?1U:t>=8U&&t<24U?hist[t-8U]:0U;
}
__global__ __launch_bounds__(1024) void csr_mw_scatter(PrefillRows v,unsigned rows,unsigned top_k,const unsigned* scratch) {
    constexpr unsigned Waves=32U;
    const auto& csr=v.csr;
    if(*reinterpret_cast<const volatile unsigned*>(scratch))return; // a failed CSR writes no rows / slots (as the base)
    const unsigned t=threadIdx.x,lane=t&31U,wave=t/32U,count=rows*top_k,k=blockIdx.x;
    const unsigned first=min(count,k*csr_mw_span+wave*64U),last=min(count,first+64U);
    __shared__ unsigned position[Waves][256];
    for(unsigned x=t;x<Waves*256U;x+=1024U)position[x/256U][x%256U]=0U;
    __syncthreads();
    const auto expert_at=[&](unsigned a){return at(v.routes,a/top_k)[a%top_k];};
    for(unsigned a=first+lane;a<last;a+=32U)atomicAdd(&position[wave][expert_at(a)/2U],1U<<(16U*(expert_at(a)&1U)));
    __syncthreads();
    if(t<256U) {
        unsigned lo=0U,hi=0U;
        for(unsigned w=0;w<Waves;++w){const unsigned c=position[w][t];position[w][t]=lo|hi<<16U;lo+=c&65535U;hi+=c>>16U;}
    }
    __syncthreads();
    const unsigned* const base=scratch+csr_mw_head+k*512U;
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
            const unsigned slot=csr.offsets[e]+base[e]+((position[wave][e/2U]>>(16U*(e&1U)))&65535U);
            const unsigned at_pos=slot+static_cast<unsigned>(__popc(peers&((1U<<lane)-1U)));
            csr.rows[at_pos]=a/top_k;csr.slots[at_pos]=a%top_k;
            if((peers&((1U<<lane)-1U))==0U)atomicAdd(&position[wave][e/2U],static_cast<unsigned>(__popc(peers))<<(16U*(e&1U)));
        }
    }
}
