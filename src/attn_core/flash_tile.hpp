#pragma once
// Shared WMMA tile + online-softmax helpers for the attention cores in src/attn_core (gfx11 wave32,
// v_wmma_f32_16x16x16_{bf16,f16}). One implementation of: operand rounding (RNE), 32-B fragment loads, the WMMA call,
// the accumulator row-slot permutation, and the per-block online softmax step (8 lanes per row).
// Fragment layout (RDNA3): A lane l holds row l % 16, k 0..15; B lane l holds column l % 16, k 0..15 (both half-waves
// hold the same data); accumulator element e of lane l is (row 2 e + l / 16, column l % 16).
// Users: sparse_flash_wmma.hpp (GLM MLA/DSA, MiniMax MSA policies). Intended to also carry dense_flash_prefill.hpp
// (attn-flash) and the index_score_wmma.hpp primitives (idx-wmma) when those land, so there is one softmax and one
// tile implementation.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::attn_core::tile {

typedef unsigned short Frag __attribute__((ext_vector_type(16)));
typedef float Acc __attribute__((ext_vector_type(8)));
typedef unsigned U4 __attribute__((ext_vector_type(4)));
typedef unsigned U2 __attribute__((ext_vector_type(2)));

enum class Operand : unsigned { bf16 = 1, f16 = 2 };

__device__ __forceinline__ std::uint16_t to_bf16(float x) {  // round to nearest even, NaN stays NaN
    const std::uint32_t u = __builtin_bit_cast(std::uint32_t, x);
    if ((u & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ std::uint16_t to_f16(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }
template <Operand O> __device__ __forceinline__ std::uint16_t to_operand(float x) {
    if constexpr (O == Operand::bf16) return to_bf16(x);
    else return to_f16(x);
}
template <Operand O> __device__ __forceinline__ float operand_value(std::uint16_t b) {
    if constexpr (O == Operand::bf16) return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U);
    else return static_cast<float>(__builtin_bit_cast(_Float16, b));
}
template <Operand O> __device__ __forceinline__ unsigned pack2(float a, float b) {
    return static_cast<unsigned>(to_operand<O>(a)) | (static_cast<unsigned>(to_operand<O>(b)) << 16U);
}
__device__ __forceinline__ Frag frag32(const void* p) {  // 32 B (16 operand elements), 16-B aligned (LDS or global)
    const U4* v = static_cast<const U4*>(p);
    const U4 lo = v[0], hi = v[1];
    Frag f;
    __builtin_memcpy(&f, &lo, 16U);
    __builtin_memcpy(reinterpret_cast<char*>(&f) + 16U, &hi, 16U);
    return f;
}
// 16 FP32 values -> one A/B fragment of operand elements.
template <Operand O> __device__ __forceinline__ Frag frag_from_f32(const float* p) {
    const float4* q = reinterpret_cast<const float4*>(p);
    const float4 v0 = q[0], v1 = q[1], v2 = q[2], v3 = q[3];
    const U4 lo = U4{pack2<O>(v0.x, v0.y), pack2<O>(v0.z, v0.w), pack2<O>(v1.x, v1.y), pack2<O>(v1.z, v1.w)};
    const U4 hi = U4{pack2<O>(v2.x, v2.y), pack2<O>(v2.z, v2.w), pack2<O>(v3.x, v3.y), pack2<O>(v3.z, v3.w)};
    Frag f;
    __builtin_memcpy(&f, &lo, 16U);
    __builtin_memcpy(reinterpret_cast<char*>(&f) + 16U, &hi, 16U);
    return f;
}
template <Operand O> __device__ __forceinline__ Acc mma(Frag a, Frag b, Acc c) {
#if defined(__gfx1151__) || defined(__gfx1150__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    if constexpr (O == Operand::bf16) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    } else {
        typedef _Float16 Half16 __attribute__((ext_vector_type(16)));
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(Half16, a), __builtin_bit_cast(Half16, b), c);
    }
#else
    (void)a;
    (void)b;
    return c;
#endif
}
__device__ __forceinline__ Acc acc_zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
// Per-row scalars (alpha, 1/l) of a 16-row M tile mt as a lane reads them: element e of lane l is row 2 e + l / 16,
// stored at [mt * 16 + (l / 16) * 8 + e] so one lane reads 8 contiguous floats.
__host__ __device__ constexpr unsigned row_slot(unsigned row) { return (row >> 4U) * 16U + (row & 1U) * 8U + ((row & 15U) >> 1U); }
__device__ __forceinline__ void row_scalars(const float* slots, unsigned mt, unsigned lane, float (&v)[8]) {
    const float4 a = *reinterpret_cast<const float4*>(slots + mt * 16U + (lane >> 4U) * 8U);
    const float4 b = *reinterpret_cast<const float4*>(slots + mt * 16U + (lane >> 4U) * 8U + 4U);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

// Online softmax step of one row over one key block, the row's keys spread over 8 consecutive lanes (K each).
// s: scores already x scale*log2(e), masked keys = -inf (at least one finite key in the block across the 8 lanes).
// Updates the running max m (log2 units) and sum l (of the ROUNDED p, so P/l normalises what PV accumulates), writes
// the operand-rounded p, returns alpha = exp2(m_old - m_new) for the accumulator rescale (0 on the first block).
template <Operand O, unsigned K> __device__ __forceinline__ float online_step(const float (&s)[K], float& m, float& l, std::uint16_t (&p)[K]) {
    float bm = -__builtin_huge_valf();
#pragma unroll
    for (unsigned i = 0; i < K; ++i) bm = s[i] > bm ? s[i] : bm;
#pragma unroll
    for (unsigned x = 1; x < 8; x <<= 1U) {
        const float o = __shfl_xor(bm, static_cast<int>(x));
        bm = o > bm ? o : bm;
    }
    const float mn = bm > m ? bm : m;
    const float alpha = __builtin_amdgcn_exp2f(m - mn);
    float ps = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < K; ++i) {
        p[i] = to_operand<O>(__builtin_amdgcn_exp2f(s[i] - mn));
        ps += operand_value<O>(p[i]);
    }
#pragma unroll
    for (unsigned x = 1; x < 8; x <<= 1U) ps += __shfl_xor(ps, static_cast<int>(x));
    l = l * alpha + ps;
    m = mn;
    return alpha;
}

}  // namespace strix::attn_core::tile
