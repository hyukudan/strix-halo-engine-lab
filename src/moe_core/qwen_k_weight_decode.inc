struct QwenKDecodeOps : QwenKScalarDecode {
struct F16Int { unsigned man; int exp; bool finite; };
static __device__ __forceinline__ F16Int f16_int(unsigned h) {
    const unsigned e=(h>>10U)&31U,f=h&1023U;
    return {e?(f|1024U):f,static_cast<int>(e?e:1U)-25,e!=31U};
}
// Every a*q + b with |a| <= |x|*wx, |b| <= |y|*wy (a, b integer multiples of x, y; q <= qmax folded into wx) is an
// exact FP32 value: |x|*wx + |y|*wy < 2^(min lsb + 24), in integers relative to the lower lsb.
static __device__ __forceinline__ bool f16_affine_exact(unsigned hx,unsigned wx,unsigned hy,unsigned wy) {
    const F16Int x=f16_int(hx),y=f16_int(hy);
    if(!x.finite||!y.finite)return false;
    // Odd mantissa times 2^lsb; relative to the lower lsb both shifts are >= 0.
    const int cx=x.man?__builtin_ctz(x.man):0,cy=y.man?__builtin_ctz(y.man):0;
    const int lx=x.man?x.exp+cx:1000,ly=y.man?y.exp+cy:1000;
    const int lo=min(lx,ly);
    if(lo==1000)return true;
    const int sx=lx-lo,sy=ly-lo;
    if((x.man&&sx>24)||(y.man&&sy>24))return false;
    const unsigned long long t=(x.man?(static_cast<unsigned long long>(x.man>>cx)*wx)<<sx:0ULL)+
        (y.man?(static_cast<unsigned long long>(y.man>>cy)*wy)<<sy:0ULL);
    return t<(1ULL<<24U);
}
// Q5_1 block (head word: d low, m high, both FP16): every d*q + m, q <= 31, is an exact FP32 value when both are normal
// and max(ed + 5, em) - min(ed, em) <= 12 (biased exponent fields): |d*q + m| < 2^(max(ed + 5, em) - 13) and every term
// is a multiple of 2^(min(ed, em) - 25). Exponent fields only (a few instructions per block); every Q5_1 block of the
// production Qwen down passes.
static __device__ __forceinline__ bool f16_q51_exact(unsigned head) {
    const int ed=static_cast<int>((head>>10U)&31U),em=static_cast<int>((head>>26U)&31U);
    return ed>=1&&ed<=30&&em>=1&&em<=30&&max(ed+5,em)-min(ed,em)<=12;
}
// round_f16(a*q + b) for the four unsigned bytes q of w as two FP16 pairs (bytes 0,1 and 2,3), a2 = a * 2^24.
static __device__ __forceinline__ unsigned f16_mix_pair(float a2,unsigned q2,float b) {
    unsigned r;
    STRIX_HIP_RDNA3_ASM(__asm__("v_fma_mixlo_f16 %0, %1, %2, %3 op_sel_hi:[0,1,0]\n\t"
            "v_fma_mixhi_f16 %0, %1, %2, %3 op_sel:[0,1,0] op_sel_hi:[0,1,0]" : "=&v"(r) : "v"(a2),"v"(q2),"v"(b)));
    return r;
}
static __device__ __forceinline__ uint2 f16_affine_u8_fast(unsigned w,float a2,float b) {
    return {f16_mix_pair(a2,__builtin_amdgcn_perm(0U,w,0x0c010c00U),b),f16_mix_pair(a2,__builtin_amdgcn_perm(0U,w,0x0c030c02U),b)};
}
};
template<bool Q5> using QwenKWeightDecode = strix::moe_core::GroupedWeightDecode<
    Q5 ? strix::moe_core::GroupedWeightFormat::q5_k : strix::moe_core::GroupedWeightFormat::q4_k,QwenKDecodeOps>;
using QwenKCore = strix::moe_core::GroupedPrefillCore<QwenKWeightDecode<false>>;
using QwenKDecode = QwenKCore::WeightDecode;
