// K-quant FP16 down operands of the f16x down body (moe_core/hip_moe_f16x_down_body.inc with STRIX_F16X_DOWN_KQ, entry
// down_prefill_f16x_kq in hip_moe_q4k_a8_known_rows.hip; Q4_K 12 / Q6_K 14). Included
// inside that file's anonymous namespace after factor, f16_load, f16_scaled_s8 and the Qwen K decoders.
// A row is width/256 superblocks. The 64-K step s covers sub-blocks 2(s%4) and 2(s%4)+1 of superblock s/4; half h (32
// weights, K 32h..32h+31 of the step) is sub-block 2(s%4)+h. Each FP16 weight is ONE rounding of an FP32 value:
// - Q4_K: the gate/up decode, fma(d*sc, q, -(dmin*m)) (its exact short form under tune bit 6 gives the same words);
// - Q6_K: d*sc*(q-32), an exact FP32 product (d 11 bits, |sc| <= 127 7 bits or -128, |q-32| <= 31 5 bits or 32: powers of two
//   add no bits), so the only rounding is the FP16 one.
template<unsigned Type> struct F16xKQuantDown;
// The B register stage of the body: one raw per half (per-step reads), or one step-pair read (KQ::pair).
template<class KQ,unsigned Halves,bool P=KQ::pair> struct F16xKQuantRegs { using type=typename KQ::Raw[Halves]; };
template<class KQ,unsigned Halves> struct F16xKQuantRegs<KQ,Halves,true> { using type=typename KQ::template Pair<Halves>; };
template<> struct F16xKQuantDown<12U> {
    static constexpr unsigned block_bytes=144U;
    static constexpr bool pair=false;  // per-step reads (load / decode per half)
    struct Raw { uint4 head,lo,hi; };
    // Superblocks are 16-B aligned (144 B, rows of whole superblocks).
    __device__ __forceinline__ static Raw load(const unsigned char* row,unsigned step,unsigned) {
        const unsigned char* blk=row+(step/4U)*block_bytes;
        return {f16_load(blk),f16_load(blk+16U+32U*(step%4U)),f16_load(blk+32U+32U*(step%4U))};
    }
    template<unsigned Tune>
    __device__ __forceinline__ static void decode(const Raw& r,unsigned step,unsigned half,uint4 (&out)[4]) {
        const unsigned g=2U*(step%4U)+half,shift=4U*half,mask=0x0f0f0f0fU;
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(r.head.x&65535U)));
        const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(r.head.x>>16U)));
        const float a=d*static_cast<float>(factor(r.head,g,false)),b=-(dm*static_cast<float>(factor(r.head,g,true)));
        const bool fast=(Tune&64U)!=0U&&__all(QwenKDecode::f16_affine_exact(r.head.x&65535U,63U*15U,r.head.x>>16U,63U));
        const unsigned q[8]{r.lo.x,r.lo.y,r.lo.z,r.lo.w,r.hi.x,r.hi.y,r.hi.z,r.hi.w};
#pragma unroll
        for(unsigned w=0;w<4U;++w) {
            const unsigned l0=(q[2U*w]>>shift)&mask,l1=(q[2U*w+1U]>>shift)&mask;
            const uint2 x0=fast?QwenKDecode::f16_affine_u8_fast(l0,a*16777216.0F,b):QwenKScalarDecode::f16_affine_u8(l0,a,b);
            const uint2 x1=fast?QwenKDecode::f16_affine_u8_fast(l1,a*16777216.0F,b):QwenKScalarDecode::f16_affine_u8(l1,a,b);
            out[w]={x0.x,x0.y,x1.x,x1.y};
        }
    }
};
// Q6_K (210-B blocks: ql 128, qh 64, int8 scales 16, d at 208) is 2-B aligned: words are read dword-aligned and realigned
// with alignbit (the Q8_0 down discipline); d is one 16-bit load inside the block. Offsets and the in-block rule:
// moe_core/kquant_block_reads.hpp (host-tested). Half h of step s, n = (s%4)/2, j = 2(s%2)+h: ql bytes 64n+32h (nibble 4(s%2)),
// qh bytes 128+32n (bits 2j), scales 8n+2j (l < 16) and 8n+2j+1; code byte (q << 2) ^ 0x80 = 4(q-32) as int8, times d*sc/4.
// Step pairs (pair = true): steps 2m and 2m+1 read the same ql / qh bytes and the scale bytes 8n..8n+7, so the even step's
// register stage loads them once (load_pair) and decode_pair yields the FP16 words of both steps (same bytes, same
// expression as one step's decode); the body keeps the odd step's words in registers for its stage.
template<> struct F16xKQuantDown<14U> {
    static constexpr unsigned block_bytes=210U;
    static constexpr bool pair=true;
    template<unsigned Halves> struct Pair { unsigned ql[8U*Halves+1U],qh[9],sc[3],d,odd; };
    // Halves consecutive halves from `first` of the pair (step even).
    template<unsigned Halves>
    __device__ __forceinline__ static Pair<Halves> load_pair(const unsigned char* row,unsigned step,unsigned first) {
        const unsigned char* blk=row+(step/4U)*block_bytes;
        namespace q6=::strix::moe_core::q6k_reads;
        const unsigned odd=static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(blk))&2U;
        const unsigned char* base=blk-odd;  // dword-aligned (blocks start on even bytes)
        const auto* ql=reinterpret_cast<const unsigned*>(base+q6::ql_from_base(step,first));
        const auto* qh=reinterpret_cast<const unsigned*>(base+q6::qh_from_base(step));
        Pair<Halves> r;
#pragma unroll
        for(unsigned q=0;q<2U*Halves;++q) {
            const Dword4 v=reinterpret_cast<const Dword4*>(ql)[q];
            r.ql[4U*q]=v.x;r.ql[4U*q+1U]=v.y;r.ql[4U*q+2U]=v.z;r.ql[4U*q+3U]=v.w;
        }
        r.ql[8U*Halves]=ql[8U*Halves];
#pragma unroll
        for(unsigned q=0;q<2U;++q) {
            const Dword4 v=reinterpret_cast<const Dword4*>(qh)[q];
            r.qh[4U*q]=v.x;r.qh[4U*q+1U]=v.y;r.qh[4U*q+2U]=v.z;r.qh[4U*q+3U]=v.w;
        }
        r.qh[8]=qh[8];
        const auto* sc=reinterpret_cast<const unsigned*>(base+q6::sc_pair_from_base(step));
        r.sc[0]=sc[0];r.sc[1]=sc[1];
        r.sc[2]=*reinterpret_cast<const unsigned*>(base+q6::sc_tail_from_base(odd,step));
        r.d=*reinterpret_cast<const unsigned short*>(blk+q6::d_from_block);  // in-block: the last block needs no slack
        r.odd=odd;
        return r;
    }
    template<unsigned Tune,unsigned Halves>
    __device__ __forceinline__ static void decode_pair(const Pair<Halves>& r,unsigned first,uint4 (&even)[Halves][4],
        uint4 (&odd)[Halves][4]) {
        const unsigned sh=8U*r.odd;
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(r.d)));
        const unsigned sc[2]{__builtin_amdgcn_alignbit(r.sc[1],r.sc[0],sh),__builtin_amdgcn_alignbit(r.sc[2],r.sc[1],sh)};
        unsigned hi[8];
#pragma unroll
        for(unsigned w=0;w<8U;++w)hi[w]=__builtin_amdgcn_alignbit(r.qh[w+1U],r.qh[w],sh);
#pragma unroll
        for(unsigned k=0;k<Halves;++k) {
            unsigned lo[8];
#pragma unroll
            for(unsigned w=0;w<8U;++w)lo[w]=__builtin_amdgcn_alignbit(r.ql[8U*k+w+1U],r.ql[8U*k+w],sh);
            const unsigned half=first+k;
#pragma unroll
            for(unsigned p=0;p<2U;++p) {  // step parity: nibble 4p, bit pair j = 2p + half, scale bytes 2j and 2j + 1
                const unsigned j=2U*p+half,nib=4U*p,s=sc[p]>>(16U*(half&1U));
                const float a0=d*static_cast<float>(static_cast<signed char>(s&255U))*0.25F;
                const float a1=d*static_cast<float>(static_cast<signed char>((s>>8U)&255U))*0.25F;
                uint2 v[8];
#pragma unroll
                for(unsigned w=0;w<8U;++w) {
                    const unsigned code=((((lo[w]>>nib)&0x0f0f0f0fU)|(((hi[w]>>(2U*j))&0x03030303U)<<4U))<<2U)^0x80808080U;
                    v[w]=f16_scaled_s8(code,w<4U?a0:a1);
                }
                uint4 (&out)[Halves][4]=p?odd:even;
#pragma unroll
                for(unsigned q=0;q<4U;++q)out[k][q]={v[2U*q].x,v[2U*q].y,v[2U*q+1U].x,v[2U*q+1U].y};
            }
        }
    }
};
