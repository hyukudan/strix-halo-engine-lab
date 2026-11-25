#pragma once
// Low-bit resident rows (strix/weight/host_lowbit.hpp q4_g64 / q6_g64) as FP16 matrix operands of the f16 tile GEMM
// family. Both consumers, the 128-row projection tile (LowBitF16Staging below) and the small-M schedule
// (f16_tile_gemm_small.hpp), compute ONE arithmetic, so a row's bits do not depend on the route: the integer codes are
// the operand as is (exact FP16 integers, integer_pairs), each K64 group is one WMMA chain from +0, and the group's FP16
// scale d is applied once to the FP32 accumulator, acc = fma(group sum, float(d), acc), groups in ascending order.
// Decode of 16 consecutive elements (two 4-bit plane words, one 2-bit plane word for q6): bytes of biased codes
// (code + 8 / code + 32) are built with v_perm_b32, then v_perm_b32 puts 0x64 above each byte, i.e. the FP16 integer
// 1024 + biased code; the packed subtraction of 1032 / 1056 is exact.

#include "strix/weight/host_lowbit.hpp"

#include <cmath>
#include <cstdint>

#if defined(__HIPCC__)
#include "hip/weight/arch/gfx1151_sources.hpp"
#endif

namespace strix::core::lowbit_f16 {

#if defined(__HIPCC__)
#define STRIX_LOWBIT_GROUP_INLINE __device__ __forceinline__
#else
#define STRIX_LOWBIT_GROUP_INLINE inline
#endif

// One WMMA output fragment for a scale group. A schedule controls operand staging and
// supplies the K16 units in order, then applies the scale once per lane to the running sum.
template <class Accumulator, class Mma>
struct GroupAccumulator {
    Accumulator sum{};

    template <class Fragment>
    STRIX_LOWBIT_GROUP_INLINE void add(Fragment a, Fragment b) {
        sum = Mma{}(a, b, sum);
    }

    template <class Scale>
    STRIX_LOWBIT_GROUP_INLINE void apply(Scale scale, Accumulator& acc) const {
#if defined(__HIPCC__)
#pragma unroll
#endif
        for (unsigned i = 0; i < 8U; ++i) {
            float partial = sum[i];
            float multiplier;
            if constexpr (requires { scale.partial_multiplier; }) {
                partial *= scale.partial_multiplier;
                multiplier = scale.multiplier;
            } else {
                multiplier = scale;
            }
#if defined(__HIPCC__)
            acc[i] = __builtin_fmaf(partial, multiplier, acc[i]);
#else
            acc[i] = std::fma(partial, multiplier, acc[i]);
#endif
        }
    }
};

#undef STRIX_LOWBIT_GROUP_INLINE

// FP16 integer bias of a biased code: 1024 + 2^(bits - 1).
template <class Storage> inline constexpr float kBias = Storage::high_bytes == 0U ? 1032.0F : 1056.0F;

#if defined(__HIPCC__)
using Half2 = weight::hip::gfx1151::Half2;
using WordsG = const std::uint32_t __attribute__((address_space(1), aligned(4)));
typedef std::uint32_t Words4 __attribute__((ext_vector_type(4), aligned(4)));
using Words4G = const Words4 __attribute__((address_space(1)));

// Bytes of four biased codes, element e of the result in byte e, for elements 4 q .. 4 q + 3 (q = 0..3) of the 16-element
// run whose 4-bit plane words are lo0 (elements 0..7) and lo1 (8..15) and whose 2-bit plane word is hi (q6 only).
template <class Storage>
__device__ __forceinline__ void biased_bytes(std::uint32_t lo0, std::uint32_t lo1, std::uint32_t hi, std::uint32_t (&z)[4]) {
    constexpr bool kQ6 = Storage::high_bytes != 0U;
#pragma unroll
    for (unsigned p = 0; p < 2U; ++p) {
        // q4: XOR 8 per nibble turns the signed code into code + 8.
        const std::uint32_t word = (p == 0U ? lo0 : lo1) ^ (kQ6 ? 0U : 0x88888888U);
        const std::uint32_t even = word & 0x0F0F0F0FU, odd = (word >> 4U) & 0x0F0F0F0FU;
        z[2U * p] = __builtin_amdgcn_perm(odd, even, 0x05010400U);       // elements 8p + 0..3
        z[2U * p + 1U] = __builtin_amdgcn_perm(odd, even, 0x07030602U);  // elements 8p + 4..7
    }
    if constexpr (kQ6) {
#pragma unroll
        for (unsigned q = 0; q < 4U; ++q) {
            // 2-bit fields of elements 4q..4q+3 (hi bits 8q..8q+7) to bits 0-1 of bytes 0..3, then above the nibble;
            // XOR 32 turns the signed 6-bit code into code + 32.
            const std::uint32_t t = (hi >> (8U * q)) & 0xFFU;
            const std::uint32_t a = (t | (t << 12U)) & 0x000F000FU;
            const std::uint32_t b = (a | (a << 6U)) & 0x03030303U;
            z[q] = (z[q] | (b << 4U)) ^ 0x20202020U;
        }
    }
}

// Eight FP16 pairs (elements 2 i, 2 i + 1 in word i) of the exact integer codes of a 16-element run.
template <class Storage>
__device__ __forceinline__ void integer_pairs(std::uint32_t lo0, std::uint32_t lo1, std::uint32_t hi, std::uint32_t (&words)[8]) {
    std::uint32_t z[4];
    biased_bytes<Storage>(lo0, lo1, hi, z);
    const auto bias = static_cast<_Float16>(kBias<Storage>);
#pragma unroll
    for (unsigned q = 0; q < 4U; ++q) {
        const auto l = __builtin_bit_cast(Half2, __builtin_amdgcn_perm(0x64646464U, z[q], 0x04010400U)) - Half2{bias, bias};
        const auto h = __builtin_bit_cast(Half2, __builtin_amdgcn_perm(0x64646464U, z[q], 0x04030402U)) - Half2{bias, bias};
        words[2U * q] = __builtin_bit_cast(std::uint32_t, l);
        words[2U * q + 1U] = __builtin_bit_cast(std::uint32_t, h);
    }
}

// The FP32 value of the FP16 scale in bits 0..15 (upper bits ignored).
__device__ __forceinline__ float scale_value(std::uint32_t bits) {
    return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(bits & 0xFFFFU)));
}

// Staging trait of the 128-row projection tile (interface of q8_f16_staging::Q8F16Staging, plus the group scale): a
// thread stages the 16 elements [k0 + 16 half, + 16) of one weight row as integer codes (a K32 stage is half of one
// 64-group, so one scale per thread) and the group's scale bits into the row's LDS pad word kScaleWord, which the tile
// core reads at the end of each group (f16_tile_gemm_core.inc: kGroupK).
template <class Storage>
struct LowBitF16Staging {
    static constexpr unsigned kGroupK = Storage::group;
    static constexpr unsigned kScaleWord = 16U;  // dwords 0..15 hold the stage's 32 FP16 values, 16..19 are pad
    struct Raw {
        std::uint32_t lo0, lo1, hi, scale, half;
    };

    __device__ __forceinline__ static Raw load(const std::uint8_t* row, unsigned k0, unsigned half) {
        // A null column keeps the thread's half: publish lets half 0 alone store the scale word (dword 16 of the LDS row).
        if (!row) {
            Raw raw{};
            raw.half = half;
            return raw;
        }
        const unsigned k = k0 + 16U * half, j = k % Storage::elements;
        const auto* block = row + std::size_t{k / Storage::elements} * Storage::bytes;
        Raw raw{};
        raw.lo0 = *(WordsG*)(block + j / 2U);
        raw.lo1 = *(WordsG*)(block + j / 2U + 4U);
        if constexpr (Storage::high_bytes != 0U) raw.hi = *(WordsG*)(block + Storage::low_bytes + j / 4U);
        // The two scales of a superblock are one aligned word: pick the group's half.
        raw.scale = *(WordsG*)(block + Storage::scale_offset) >> (16U * (j / Storage::group));
        raw.half = half;
        return raw;
    }

    // dst = the row's dword 8 half. Both halves carry the same group scale; half 0 alone stores it (one writer of the
    // LDS word; the tile's barrier before the WMMA stage orders it for every reader).
    __device__ __forceinline__ static void publish(const Raw& raw, unsigned, std::uint32_t* dst) {
        using U4 = weight::hip::gfx1151::U4;
        std::uint32_t words[8];
        integer_pairs<Storage>(raw.lo0, raw.lo1, raw.hi, words);
        *reinterpret_cast<U4*>(dst) = U4{words[0], words[1], words[2], words[3]};
        *reinterpret_cast<U4*>(dst + 4U) = U4{words[4], words[5], words[6], words[7]};
        if (raw.half == 0U) dst[kScaleWord] = raw.scale & 0xFFFFU;
    }

    // The scale of the current group of the column whose LDS row starts at `row`.
    __device__ __forceinline__ static float scale(const std::uint32_t* row) { return scale_value(row[kScaleWord]); }
};
#endif

}  // namespace strix::core::lowbit_f16
