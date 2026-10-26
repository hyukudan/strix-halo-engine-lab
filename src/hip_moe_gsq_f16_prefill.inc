// GSQ routed experts in the fast prefill profile (fast-prefill-f16wmma-v1). Included by
// hip_moe_q4k_a8_known_rows.hip after hip_moe_gsq_weights.inc (the weight decoders), inside its anonymous namespace.
//
// The GSQ gate/up and down weights enter the FP16 WMMA tiles of project_prefill_f16x / down_prefill_f16x under
// the profile's rule: each weight value is formed in FP32 as the GSQ group decomposition defines it and rounded
// once to FP16 inside the kernel. Everything else is those kernels': CSR batches (64, or 128 with tune bit 1),
// the A staging (BF16 input scaled by 2^s and rounded to FP16, or the prologue's FP16 image with bit 4; h scaled
// per (row, slot) and rounded while the down kernel stages it), K steps of 64 ascending with K16 planes ascending,
// the WMMAs per output, FP32 accumulation over the whole K, the SwiGLU / max |h| / fail-closed epilogue and the
// raw slot dots. So, on weights both formats can hold, a GSQ kernel writes the bytes of the production kernel
// (tests/hip_moe_wmma_prefill_tests.hip "gsq": Q2_0 down vs the same values as Q8_0 blocks, Q2_0 gate/up vs Q4_K
// blocks with sc = m = 1 and dmin = d > 0).
//
// Q2_0 (type 42): 18-byte blocks of 64 weights (FP16 d, 16 code bytes; weight k at bits 2(k%4) of byte k/4),
// value d * float(q - 1), one block per K64 step. Every value is an FP16 value (q - 1 in {-1, 0, 1, 2}): the
// rounding is exact except 2d beyond the FP16 range (infinity; g, u or the raw dot is then non-finite and the
// row fails closed). Rows hold an even block count (K % 128 == 0), so an even block starts at a dword and an odd
// one two bytes after a dword: a thread loads the five dwords from the block's dword-aligned start.
// Conversion: the four values of a block (q = 0..3) are formed once per thread and step; their low and high bytes
// make two dword tables, and one v_perm with selector bytes (c0, c0 + 4, c1, c1 + 4) returns the FP16 pair of codes
// c0, c1 (~2 VALU per weight, no FP32 work per weight).
//
// Tunes: bits 1 (batch-major) and 4 (A16 image) as production; diagnostic bits 256 (no WMMA), 512 (raw words
// staged), 1024 (no global loads). The layout-only bits 0, 2, 3 and the dequantization bit 6 have no GSQ form and
// map to the default layout (results are the same bytes either way).
// Gate/up of one CSR batch for Columns/2 outputs of both matrices (project_prefill_f16x with GSQ weights).
template<std::uint32_t G,unsigned Tune>
__global__ __launch_bounds__(F16Tile<Tune>::Threads) void project_prefill_f16g(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs,const uint4* a16) {
    using T=F16Tile<Tune>;
    constexpr unsigned Batch=T::Batch,Columns=T::Columns,Outputs=Columns/2U,TPC=T::TPC,AChunks=T::AChunks;
    using B=GsqB<G,TPC>;
    const unsigned batch=blockIdx.y,tile=blockIdx.x;
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||batch>=csr.summary[2])return;
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,expert=csr.experts[batch];
    using Stage=F16Stage<Batch,Columns,T::LTune>;
    __shared__ __align__(16) Stage sb[1];
    __shared__ F16Rows<Batch> rows;
    __shared__ unsigned tables[B::table_bytes?B::table_bytes/4U:1U];
    const unsigned t=threadIdx.x;
    B::load_tables(tables,t,T::Threads);
    if(t<Batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=live?csr.slots[begin+t]:0U;rows.exponent[t]=live?at(view.sums,r)[0]:0;
    }
    __syncthreads();
    const unsigned steps=hidden/64U;
    const unsigned char* a_src[AChunks];float a_scale[AChunks];bool a_live[AChunks];unsigned a_row[AChunks],a_half[AChunks];
#pragma unroll
    for(unsigned j=0;j<AChunks;++j) {
        const unsigned c=t+T::Threads*j;
        a_row[j]=c/8U;a_half[j]=8U*(c%8U);a_live[j]=a_row[j]<count;
        a_scale[j]=ldexpf(1.0F,rows.exponent[a_row[j]]);
        a_src[j]=(Tune&16U)?reinterpret_cast<const unsigned char*>(a16+static_cast<std::size_t>(rows.row[a_row[j]])*(hidden/8U))+2U*a_half[j]:
            reinterpret_cast<const unsigned char*>(at(view.input,rows.row[a_row[j]]))+2U*a_half[j];
    }
    // B: column t % Columns (gate outputs, then the same up outputs), halves 64/TPC*part.. of each step's block.
    const unsigned b_col=t%Columns,b_part=t/Columns;
    constexpr unsigned BChunks=8U/TPC;
    const unsigned char* b_row=(b_col<Outputs?gate:up)+
        (static_cast<unsigned long long>(expert)*outputs+tile*Outputs+(b_col%Outputs))*gsq_a8::row_bytes(G,hidden);
    struct Regs { uint4 a[AChunks]{}; typename B::Raw b{}; };
    struct Cvt { uint4 a[AChunks],b[BChunks]; };
    const auto fetch=[&](unsigned step,Regs& r) {
        if constexpr((Tune&1024U)!=0U)return;
#pragma unroll
        for(unsigned j=0;j<AChunks;++j)if(a_live[j])r.a[j]=f16_load(a_src[j]+128U*step);
        B::fetch(b_row,step,b_part,r.b);
    };
    const auto stage=[&](unsigned step,const Regs& r,unsigned buf) {
        Stage& st=sb[buf];
        if constexpr((Tune&512U)!=0U) {
#pragma unroll
            for(unsigned j=0;j<AChunks;++j)*st.a16(a_row[j],a_half[j])=r.a[j];
#pragma unroll
            for(unsigned q=0;q<BChunks;++q)*st.b16(b_col,(64U/TPC)*b_part+8U*q)=B::raw4(r.b);
            return;
        }
        Cvt cv;
#pragma unroll
        for(unsigned j=0;j<AChunks;++j)cv.a[j]=(Tune&16U)?r.a[j]:f16_from_bf16(r.a[j],a_scale[j]);
        B::convert(r.b,step,b_part,tables,cv.b);
#pragma unroll
        for(unsigned j=0;j<AChunks;++j)*st.a16(a_row[j],a_half[j])=cv.a[j];
#pragma unroll
        for(unsigned q=0;q<BChunks;++q)*st.b16(b_col,(64U/TPC)*b_part+8U*q)=cv.b[q];
    };
    const unsigned lane=t&31U,wave=t/32U,i=lane&15U,h=lane>>4U,rh=wave/T::Groups,cg=wave%T::Groups;
    const unsigned rbase=64U*rh,mb=count>rbase?min(4U,(count-rbase+15U)/16U):0U;
    const unsigned cols[2]{16U*cg,Outputs+16U*cg};
    f32x8 acc[4][2]={};
    f16_pipeline<Tune,Regs>(steps,fetch,stage,[&](unsigned){if constexpr((Tune&256U)==0U)f16_step_rows(sb[0],rbase,mb,cols,i,acc);});
    const unsigned out=tile*Outputs+16U*cg+i;
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=rbase+16U*mi+2U*e+h;
        const bool live=a<count;
        const unsigned r=rows.row[live?a:0U],slot=rows.slot[live?a:0U];
        const float unscale=ldexpf(1.0F,-rows.exponent[live?a:0U]);
        const float g=acc[mi][0][e]*unscale,u=acc[mi][1][e]*unscale;
        const float hv=(g/(1.0F+expf(-g)))*u;
        const bool bad=!isfinite(g)||!isfinite(u)||!isfinite(hv);
        if(live)at(view.output,r)[slot*2U*outputs+out]=hv;
        float m=fabsf(hv);
        for(int stride=8;stride!=0;stride/=2)m=fmaxf(m,__shfl_xor(m,stride,32));
        if(live&&i==0U)atomicMax(reinterpret_cast<unsigned*>(at(view.down_sums,r))+slot,__float_as_uint(m));
        if(live&&bad){atomicOr(at(view.status,r),projection_nonfinite);atomicOr(csr.summary,projection_nonfinite);}
    }
}
// Down of one CSR batch for Columns hidden outputs, K = width (down_prefill_f16x with GSQ weights).
template<std::uint32_t D,unsigned Tune>
__global__ __launch_bounds__(F16Tile<Tune>::Threads) void down_prefill_f16g(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    using T=F16Tile<Tune>;
    constexpr unsigned Batch=T::Batch,Columns=T::Columns,TPC=T::TPC,AParts=Batch*4U/T::Threads;
    using B=GsqB<D,TPC>;
    const unsigned batch=blockIdx.y,tile=blockIdx.x;
    const auto& csr=view.csr;
    if(csr.summary[0]||batch>=csr.summary[2])return;
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,expert=csr.experts[batch];
    using Stage=F16Stage<Batch,Columns,T::LTune>;
    __shared__ __align__(16) Stage sb[1];
    __shared__ F16Rows<Batch> rows;
    __shared__ unsigned tables[B::table_bytes?B::table_bytes/4U:1U];
    const unsigned t=threadIdx.x;
    B::load_tables(tables,t,T::Threads);
    if(t<Batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U,slot=live?csr.slots[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=slot;
        rows.exponent[t]=live?f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,r)[slot]))):0;
    }
    __syncthreads();
    const unsigned steps=width/64U;
    const float* a_src[AParts];float a_scale[AParts];bool a_live[AParts];unsigned a_row[AParts],a_part[AParts];
#pragma unroll
    for(unsigned j=0;j<AParts;++j) {
        const unsigned c=t+T::Threads*j;
        a_row[j]=c/4U;a_part[j]=c%4U;a_live[j]=a_row[j]<count;
        a_scale[j]=ldexpf(1.0F,rows.exponent[a_row[j]]);
        a_src[j]=at(view.output,rows.row[a_row[j]])+2ULL*rows.slot[a_row[j]]*width+16U*a_part[j];
    }
    const unsigned b_col=t%Columns,b_part=t/Columns;
    constexpr unsigned BChunks=8U/TPC;
    const unsigned char* b_row=weights+(static_cast<unsigned long long>(expert)*hidden+tile*Columns+b_col)*gsq_a8::row_bytes(D,width);
    struct Regs { uint4 a[AParts][4]{}; typename B::Raw b{}; };
    struct Cvt { uint4 a[AParts][2],b[BChunks]; };
    const auto fetch=[&](unsigned step,Regs& r) {
        if constexpr((Tune&1024U)!=0U)return;
#pragma unroll
        for(unsigned j=0;j<AParts;++j)if(a_live[j])
#pragma unroll
            for(unsigned q=0;q<4U;++q)r.a[j][q]=f16_load(a_src[j]+64U*step+4U*q);
        B::fetch(b_row,step,b_part,r.b);
    };
    const auto stage=[&](unsigned step,const Regs& r,unsigned buf) {
        Stage& st=sb[buf];
        if constexpr((Tune&512U)!=0U) {
#pragma unroll
            for(unsigned j=0;j<AParts;++j){*st.a16(a_row[j],16U*a_part[j])=r.a[j][0];*st.a16(a_row[j],16U*a_part[j]+8U)=r.a[j][2];}
#pragma unroll
            for(unsigned q=0;q<BChunks;++q)*st.b16(b_col,(64U/TPC)*b_part+8U*q)=B::raw4(r.b);
            return;
        }
        Cvt cv;
#pragma unroll
        for(unsigned j=0;j<AParts;++j) {
            cv.a[j][0]=f16_from_f32(r.a[j][0],r.a[j][1],a_scale[j]);
            cv.a[j][1]=f16_from_f32(r.a[j][2],r.a[j][3],a_scale[j]);
        }
        B::convert(r.b,step,b_part,tables,cv.b);
#pragma unroll
        for(unsigned j=0;j<AParts;++j){*st.a16(a_row[j],16U*a_part[j])=cv.a[j][0];*st.a16(a_row[j],16U*a_part[j]+8U)=cv.a[j][1];}
#pragma unroll
        for(unsigned q=0;q<BChunks;++q)*st.b16(b_col,(64U/TPC)*b_part+8U*q)=cv.b[q];
    };
    const unsigned lane=t&31U,wave=t/32U,i=lane&15U,h=lane>>4U,rh=wave/T::Groups,cg=wave%T::Groups;
    const unsigned rbase=64U*rh,mb=count>rbase?min(4U,(count-rbase+15U)/16U):0U;
    const unsigned cols[2]{32U*cg,32U*cg+16U};
    f32x8 acc[4][2]={};
    f16_pipeline<Tune,Regs>(steps,fetch,stage,[&](unsigned){if constexpr((Tune&256U)==0U)f16_step_rows(sb[0],rbase,mb,cols,i,acc);});
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=rbase+16U*mi+2U*e+h;
        if(a>=count)continue;
        const float unscale=ldexpf(1.0F,-rows.exponent[a]);
        float* out=at(view.rawdots,rows.row[a])+rows.slot[a]*hidden+tile*Columns+32U*cg+i;
        out[0]=acc[mi][0][e]*unscale;out[16]=acc[mi][1][e]*unscale;
    }
}
