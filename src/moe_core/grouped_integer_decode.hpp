#pragma once
#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "grouped_policy.hpp"

namespace strix::moe_core {

// Integer group operands retain their engine's accumulation and rounding contract.
// Traits supply storage types and format identities; no half-fragment conversion occurs.
template <class Traits>
struct GroupedIntegerDecodeOps {
    using WeightFormat = typename Traits::WeightFormat;
    using v4i = typename Traits::Int4;
    using v8i = typename Traits::Int8;
    using Operands = typename Traits::OperandGroup;
    template <WeightFormat Format> using Raw = typename Traits::template RawGroup<Format>;
    static constexpr int dot_magic = Traits::magic_bits;
    static constexpr auto iq2_xxs_block_bytes = Traits::iq2_xxs_block_bytes;
    static constexpr auto q2_k_block_bytes = Traits::q2_k_block_bytes;
    static constexpr auto mxfp4_block_bytes = Traits::mxfp4_block_bytes;
    static constexpr auto q8_0_block_bytes = Traits::q8_0_block_bytes;

static __device__ __forceinline__ float half_bits(unsigned short bits) { return __half2float(__ushort_as_half(bits)); }
static __device__ __forceinline__ unsigned short load_u16(const unsigned char* p) {
    unsigned short v;
    __builtin_memcpy(&v, p, 2);
    return v;
}
static __device__ __forceinline__ unsigned long long load_u64(const unsigned char* p) {
    unsigned long long v;
    __builtin_memcpy(&v, p, 8);
    return v;
}
static __device__ __forceinline__ float e8m0_half(unsigned code) {
    if (code == 255U) return __uint_as_float(0x7fc00000U);
    return __uint_as_float(code < 2U ? 0x00200000U << code : (code - 1U) << 23U);
}
static __device__ __forceinline__ unsigned byte_masks(unsigned bits4) { return ((bits4 * 0x00204081U) & 0x01010101U) * 0xffU; }
static __device__ __forceinline__ unsigned sign_byte(unsigned index) {
    index &= 127U;
    return index | ((static_cast<unsigned>(__builtin_popcount(index)) & 1U) << 7U);
}
static __device__ __forceinline__ int apply_signs(unsigned magnitudes, unsigned mask) {
    return static_cast<int>((magnitudes ^ mask) + (mask & 0x01010101U));
}
static __device__ __forceinline__ int e2m1_bytes(unsigned nibbles) {
    const unsigned index = nibbles & 0x07070707U, negative = ((nibbles >> 3U) & 0x01010101U) * 0xffU;
    const unsigned positive_values = __builtin_amdgcn_perm(0x0c080604U, 0x03020100U, index);
    const unsigned negative_values = __builtin_amdgcn_perm(0xf4f8fafcU, 0xfdfeff00U, index);
    return static_cast<int>((positive_values & ~negative) | (negative_values & negative));
}


static __device__ __forceinline__ float magic_float(int bits) { return __int_as_float(bits) - 0x1.8p23F; }
// Exact int32 dot of 32 values for every (row, output) of a 16 x 16 tile, as magic-offset bits.
static __device__ __forceinline__ v8i group_dot(v4i a0, v4i b0, v4i a1, v4i b1) {
    const v8i magic{dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic, dot_magic};
    const v8i c = STRIX_HIP_WMMA_IU8(true, a0, true, b0, magic, false);
    return STRIX_HIP_WMMA_IU8(true, a1, true, b1, c, false);
}

// Raw bytes of group g of one weight row, loaded ahead of their decoding.


template <WeightFormat Format>
static __device__ __forceinline__ Raw<Format> load_raw(const unsigned char* row, unsigned g) {
    Raw<Format> raw;
    if constexpr (Format == WeightFormat::iq2_xxs) {
        const auto* block = row + g / 8U * iq2_xxs_block_bytes;
        raw.d = load_u16(block);
        raw.aux = load_u64(block + 2U + 8U * (g % 8U));
    } else if constexpr (Format == WeightFormat::q2_k) {
        const auto* block = row + g / 8U * q2_k_block_bytes;
        const unsigned group = g % 8U, half = group / 4U;
        __builtin_memcpy(raw.q, block + 16U + 32U * half, 32);
        raw.scales = load_u16(block + 8U * half + 2U * (group % 4U));
        __builtin_memcpy(&raw.dm, block + 80U, 4);
        raw.shift = 2U * (group % 4U);
    } else if constexpr (Format == WeightFormat::q8_0) {
        const auto* block = row + g * q8_0_block_bytes;
        raw.d = load_u16(block);
        __builtin_memcpy(raw.q, block + 2, 32);
    } else {
        const auto* block = row + g * mxfp4_block_bytes;
        raw.e = block[0];
        __builtin_memcpy(raw.q, block + 1, 16);
    }
    return raw;
}
// B operands of one group (codes 0-15, 16-31) and its scale terms. For Q2_K the codes are
// pre-multiplied by (scale_h & 15) and m0/m1 are the half minimum factors (scale_h >> 4).

static __device__ __forceinline__ Operands decode(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const unsigned signs = sign_byte(aux1 >> (7U * l));
        w[2U * l] = apply_signs(static_cast<unsigned>(g), byte_masks(signs & 15U));
        w[2U * l + 1U] = apply_signs(static_cast<unsigned>(g >> 32U), byte_masks(signs >> 4U));
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}
static __device__ __forceinline__ Operands decode(const Raw<WeightFormat::mxfp4>& raw, const unsigned long long*) {
    Operands out{};
    out.scale = e8m0_half(raw.e);
    out.b0 = v4i{e2m1_bytes(raw.q[0] & 0x0f0f0f0fU), e2m1_bytes(raw.q[1] & 0x0f0f0f0fU), e2m1_bytes(raw.q[2] & 0x0f0f0f0fU),
                 e2m1_bytes(raw.q[3] & 0x0f0f0f0fU)};
    out.b1 = v4i{e2m1_bytes((raw.q[0] >> 4U) & 0x0f0f0f0fU), e2m1_bytes((raw.q[1] >> 4U) & 0x0f0f0f0fU),
                 e2m1_bytes((raw.q[2] >> 4U) & 0x0f0f0f0fU), e2m1_bytes((raw.q[3] >> 4U) & 0x0f0f0f0fU)};
    return out;
}
static __device__ __forceinline__ Operands decode(const Raw<WeightFormat::q8_0>& raw, const unsigned long long*) {
    Operands out{};
    out.scale = half_bits(raw.d);
    out.b0 = v4i{static_cast<int>(raw.q[0]), static_cast<int>(raw.q[1]), static_cast<int>(raw.q[2]), static_cast<int>(raw.q[3])};
    out.b1 = v4i{static_cast<int>(raw.q[4]), static_cast<int>(raw.q[5]), static_cast<int>(raw.q[6]), static_cast<int>(raw.q[7])};
    return out;
}
static __device__ __forceinline__ Operands decode(const Raw<WeightFormat::q2_k>& raw, const unsigned long long*) {
    Operands out{};
    const unsigned sc0 = raw.scales & 0xffU, sc1 = raw.scales >> 8U;
    const unsigned f0 = sc0 & 15U, f1 = sc1 & 15U;
    // Codes <= 3 times factors <= 15 stay below 256 in every byte: no carries between bytes.
    out.b0 = v4i{static_cast<int>(((raw.q[0] >> raw.shift) & 0x03030303U) * f0), static_cast<int>(((raw.q[1] >> raw.shift) & 0x03030303U) * f0),
                 static_cast<int>(((raw.q[2] >> raw.shift) & 0x03030303U) * f0), static_cast<int>(((raw.q[3] >> raw.shift) & 0x03030303U) * f0)};
    out.b1 = v4i{static_cast<int>(((raw.q[4] >> raw.shift) & 0x03030303U) * f1), static_cast<int>(((raw.q[5] >> raw.shift) & 0x03030303U) * f1),
                 static_cast<int>(((raw.q[6] >> raw.shift) & 0x03030303U) * f1), static_cast<int>(((raw.q[7] >> raw.shift) & 0x03030303U) * f1)};
    out.m0 = static_cast<int>(sc0 >> 4U);
    out.m1 = static_cast<int>(sc1 >> 4U);
    out.d = half_bits(static_cast<unsigned short>(raw.dm & 0xffffU));
    out.dmin = half_bits(static_cast<unsigned short>(raw.dm >> 16U));
    out.dm_bits = raw.dm;
    out.sc_bits = raw.scales;
    return out;
}

// Lean forms of the IQ2_XXS and Q2_K decoders: the same integers with full-rate instructions instead of the
// quarter-rate v_mul_lo_u32. IQ2_XXS: the byte masks m * 255 (bytes of m are 0 or 1) come from one v_perm_b32. Q2_K: every
// byte of codes * factor is at most 3 * 15 = 45, so the packed product equals a byte lookup of the codes (0..3) in
// the table {0, f, 2f, 3f} (v_perm_b32 selector bytes 0-3 pick bytes of the second operand).
static __device__ __forceinline__ unsigned byte_masks_lean(unsigned bits4) {
    // v_perm_b32 selector byte 0x0c gives 0x00 and 0x0d gives 0xff: bytes m_j + 12 select the mask byte directly.
    const unsigned m = (bits4 * 0x00204081U) & 0x01010101U;
    return __builtin_amdgcn_perm(0U, 0U, m + 0x0c0c0c0cU);
}
static __device__ __forceinline__ Operands decode_lean(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const unsigned signs = sign_byte(aux1 >> (7U * l));
        w[2U * l] = apply_signs(static_cast<unsigned>(g), byte_masks_lean(signs & 15U));
        w[2U * l + 1U] = apply_signs(static_cast<unsigned>(g >> 32U), byte_masks_lean(signs >> 4U));
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}
static __device__ __forceinline__ Operands decode_lean(const Raw<WeightFormat::q2_k>& raw, const unsigned long long*) {
    Operands out{};
    const unsigned sc0 = raw.scales & 0xffU, sc1 = raw.scales >> 8U;
    const unsigned t0 = ((sc0 & 15U) * 0x00030201U) << 8U, t1 = ((sc1 & 15U) * 0x00030201U) << 8U;  // v_mul_u32_u24 (operands < 2^24)
    const auto mul = [&](unsigned q, unsigned t) {
        return static_cast<int>(__builtin_amdgcn_perm(t, t, (q >> raw.shift) & 0x03030303U));
    };
    out.b0 = v4i{mul(raw.q[0], t0), mul(raw.q[1], t0), mul(raw.q[2], t0), mul(raw.q[3], t0)};
    out.b1 = v4i{mul(raw.q[4], t1), mul(raw.q[5], t1), mul(raw.q[6], t1), mul(raw.q[7], t1)};
    out.m0 = static_cast<int>(sc0 >> 4U);
    out.m1 = static_cast<int>(sc1 >> 4U);
    out.d = half_bits(static_cast<unsigned short>(raw.dm & 0xffffU));
    out.dmin = half_bits(static_cast<unsigned short>(raw.dm >> 16U));
    out.dm_bits = raw.dm;
    out.sc_bits = raw.scales;
    return out;
}
// IQ2_XXS with a sign table (form 2): entry i (the 7-bit sign index of a grid lookup) holds the masks of
// byte_masks(sign_byte(i) & 15) and of (sign_byte(i) >> 4), each with its mask & 0x01010101, so apply_signs is one
// v_xad per word: the same integers as decode().
struct SignEntry { unsigned lo, lo_inc, hi, hi_inc; };
static __device__ __forceinline__ void load_signs(SignEntry* signs) {
    for (unsigned t = threadIdx.x; t < 128U; t += blockDim.x) {
        const unsigned b = sign_byte(t), lo = byte_masks(b & 15U), hi = byte_masks(b >> 4U);
        signs[t] = SignEntry{lo, lo & 0x01010101U, hi, hi & 0x01010101U};
    }
}
static __device__ __forceinline__ Operands decode_signs(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid,
                                                 const SignEntry* signs) {
    Operands out{};
    const float d = half_bits(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
    int w[8];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const SignEntry e = signs[(aux1 >> (7U * l)) & 127U];
        w[2U * l] = static_cast<int>((static_cast<unsigned>(g) ^ e.lo) + e.lo_inc);
        w[2U * l + 1U] = static_cast<int>((static_cast<unsigned>(g >> 32U) ^ e.hi) + e.hi_inc);
    }
    out.b0 = v4i{w[0], w[1], w[2], w[3]};
    out.b1 = v4i{w[4], w[5], w[6], w[7]};
    return out;
}
// Prefill decoder forms: 0 = decode(), 1 = decode_lean(), 2 = decode_lean() with the IQ2_XXS sign table.
template <int Form, WeightFormat Format>
static __device__ __forceinline__ Operands decode_form(const Raw<Format>& raw, const unsigned long long* grid, const SignEntry* signs) {
    if constexpr (Form == 2 && Format == WeightFormat::iq2_xxs) return decode_signs(raw, grid, signs);
    else if constexpr (Form >= 1 && (Format == WeightFormat::iq2_xxs || Format == WeightFormat::q2_k)) return decode_lean(raw, grid);
    else return decode(raw, grid);
}

// ---- Staged activations ----
//
// The 32 lane classes of the declared product are visited in bit-reversed order i -> l = rev5(i),
// eight per stage; a stage holds, for every batch row, the codes of the classes' groups
// l + 32 j (j < J), their deltas and (Q2_K) the sums of each 16-code half. Group slot q = u * J + j
// for the class of stage position u. Within a 16-row tile, row r sits at (r & 1) * 8 + r / 2 in the
// delta / sum arrays, so D element e of half-wave h (row 2e + h) reads position 8h + e.
};

// The grouped-core format traits describe integer operands, unlike the Qwen
// GsqGroup half-fragment interface. Reuse the shared format/policy contracts.
template <class Traits>
using GroupedIq2XxsWeightDecode = GroupedWeightDecode<
    GroupedWeightFormat::iq2_xxs, GroupedIntegerDecodeOps<Traits>>;
template <class Traits>
using GroupedQ2KWeightDecode = GroupedWeightDecode<
    GroupedWeightFormat::q2_k, GroupedIntegerDecodeOps<Traits>>;

} // namespace strix::moe_core
