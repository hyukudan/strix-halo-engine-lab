#pragma once
// Device decoders of the candidate layouts (core/strixpack_layout.hpp), written as traits of the existing kernels:
//   Operand<F>   operand trait of the small-M schedule (f16_tile_gemm_small.hpp run): codes as exact FP16 integers
//   Staging<F>   staging trait of the 128-row projection tile (f16_tile_gemm_core.inc q8_core), group-scaled;
//                StagingFolded<F> folds the scale into the FP16 operand (the Q8_0 staging's class)
//   Chunk<F>     16-weight chunk of the FP32 rows sweep (gemv_rows_direct.inc)
//   A8Block<F>   32 code bytes of one block for an integer dot against int8 activations (sudot4)
// Bench instruments (tests/hip_strixpack_bench.hip): nothing here is bound by an engine.
#include "core/strixpack_layout.hpp"

#if defined(__HIPCC__)
#include "core/lowbit_f16_staging.hpp"

namespace strix::core::strixpack {

using lowbit_f16::Half2;
using lowbit_f16::Words4;
using lowbit_f16::Words4G;
using lowbit_f16::WordsG;

#define STRIX_SP_INLINE __device__ __forceinline__

STRIX_SP_INLINE std::uint32_t perm(std::uint32_t s0, std::uint32_t s1, std::uint32_t select) {
    return __builtin_amdgcn_perm(s0, s1, select);
}
// Nibbles of a dword as bytes: lo = low nibbles of bytes 0..3, hi = high nibbles.
STRIX_SP_INLINE void split4(std::uint32_t w, std::uint32_t& lo, std::uint32_t& hi) {
    lo = w & 0x0F0F0F0FU;
    hi = (w >> 4U) & 0x0F0F0F0FU;
}
// The sixteen 2-bit fields of a dword as bytes: x[a] byte b = field 4 b + {0, 2, 1, 3}[a].
STRIX_SP_INLINE void split2(std::uint32_t w, std::uint32_t* x) {
    const std::uint32_t e = w & 0x33333333U, o = (w >> 2U) & 0x33333333U;
    x[0] = e & 0x0F0F0F0FU;
    x[1] = (e >> 4U) & 0x0F0F0F0FU;
    x[2] = o & 0x0F0F0F0FU;
    x[3] = (o >> 4U) & 0x0F0F0F0FU;
}
// Bits 0..3 as the low bits of bytes 0..3.
STRIX_SP_INLINE std::uint32_t bit_bytes(std::uint32_t nibble) { return ((nibble & 15U) * 0x00204081U) & 0x01010101U; }

// The raw words of one 16-element run: c0 (and c1 for 4-bit codes), hi = the run's 16 high bits (p3).
struct Run {
    std::uint32_t c0, c1, hi;
};
template <Kind F>
STRIX_SP_INLINE Run load_run(const std::uint8_t* anchor, unsigned k) {
    const unsigned r = (k % kSuper) / 16U;
    const std::uint8_t* block = anchor + std::size_t{k / kSuper} * code_bytes(F);
    Run run{};
    if constexpr (F == Kind::p4 || F == Kind::p4b) {
        run.c0 = *(WordsG*)(block + 8U * r);
        run.c1 = *(WordsG*)(block + 8U * r + 4U);
    } else {
        run.c0 = *(WordsG*)(block + 4U * r);
        if constexpr (F == Kind::p3) run.hi = (*(WordsG*)(block + 32U + 4U * (r / 2U)) >> (16U * (r % 2U))) & 0xFFFFU;
    }
    return run;
}
template <Kind F>
STRIX_SP_INLINE std::uint32_t load_scale(const std::uint8_t* anchor, unsigned k) {
    const std::uint32_t w = *(WordsG*)(anchor - 4U * (std::size_t{k / kSuper} + 1U));
    const unsigned e = k % kSuper;
    if constexpr (group_of(F) == 64U) return (w >> (16U * (e / 64U))) & 0xFFFFU;
    else return (w >> (8U * (e / 32U))) & 0xFFU;
}
template <Kind F>
STRIX_SP_INLINE float scale_of(std::uint32_t bits) {
    if constexpr (group_of(F) == 64U) return lowbit_f16::scale_value(bits);
    else return __builtin_bit_cast(float, byte_scale_bits(bits & 0xFFU));
}

// Raw code bytes of a run: z[a] byte b = element 4 a + b.
template <Kind F>
STRIX_SP_INLINE void run_bytes(const Run& run, std::uint32_t* z) {
    if constexpr (F == Kind::p4 || F == Kind::p4b) {
        split4(run.c0, z[0], z[1]);
        split4(run.c1, z[2], z[3]);
    } else {
        split2(run.c0, z);
        if constexpr (F == Kind::p3) {
#pragma unroll
            for (unsigned a = 0; a < 4U; ++a) z[a] |= bit_bytes(run.hi >> (4U * a)) << 2U;
        }
    }
}
// High bytes of the FP16 levels of raw codes 0..3 (lo) and 4..7 (hi): every level has a zero low byte.
template <Kind F> inline constexpr std::uint32_t kLevelLo = F == Kind::p2 ? 0x423CBCC2U : F == Kind::p2t ? 0x443CBCC4U : 0xBCC2C5C7U;
template <Kind F> inline constexpr std::uint32_t kLevelHi = F == Kind::p3 ? 0x4745423CU : 0U;

// Eight FP16 pairs (elements 2 i, 2 i + 1 in word i) of the exact integer levels of a run.
template <Kind F>
STRIX_SP_INLINE void pairs16(const Run& run, std::uint32_t (&words)[8]) {
    std::uint32_t z[4];
    run_bytes<F>(run, z);
#pragma unroll
    for (unsigned a = 0; a < 4U; ++a) {
        if constexpr (F == Kind::p4 || F == Kind::p4b) {
            // 0x64 above a byte u is the FP16 integer 1024 + u; the packed subtraction of 1032 is exact.
            const auto bias = static_cast<_Float16>(1032.0F);
            const auto l = __builtin_bit_cast(Half2, perm(0x64646464U, z[a], 0x04010400U)) - Half2{bias, bias};
            const auto h = __builtin_bit_cast(Half2, perm(0x64646464U, z[a], 0x04030402U)) - Half2{bias, bias};
            words[2U * a] = __builtin_bit_cast(std::uint32_t, l);
            words[2U * a + 1U] = __builtin_bit_cast(std::uint32_t, h);
        } else {
            // One table permute gives the four high bytes; two permutes place them above zero low bytes.
            const std::uint32_t v = perm(kLevelHi<F>, kLevelLo<F>, z[a]);
            words[2U * a] = perm(0U, v, 0x01040004U);
            words[2U * a + 1U] = perm(0U, v, 0x03040204U);
        }
    }
}

template <Kind F>
struct Operand {
    static constexpr unsigned kGroup = group_of(F), kUnits = kGroup / 16U, kSuperK = kSuper;
    static constexpr unsigned kVectors = code_bytes(F) / 16U;
    using Row = const std::uint8_t*;
    struct Words {
        Words4 c[kVectors];
        std::uint32_t scale;
    };
    STRIX_SP_INLINE static Words load(Row row, unsigned index) {
        Words w{};
        const std::uint8_t* block = row + std::size_t{index} * code_bytes(F);
#pragma unroll
        for (unsigned i = 0; i < kVectors; ++i) w.c[i] = *(Words4G*)(block + 16U * i);
        w.scale = *(WordsG*)(row - 4U * (std::size_t{index} + 1U));
        return w;
    }
    STRIX_SP_INLINE static void unit(const Words& w, unsigned gi, unsigned u, std::uint32_t (&words)[8]) {
        const unsigned r = kUnits * gi + u;
        Run run{};
        if constexpr (F == Kind::p4 || F == Kind::p4b) {
            run.c0 = w.c[(2U * r) / 4U][(2U * r) % 4U];
            run.c1 = w.c[(2U * r + 1U) / 4U][(2U * r + 1U) % 4U];
        } else {
            run.c0 = w.c[r / 4U][r % 4U];
            if constexpr (F == Kind::p3) run.hi = (w.c[2U][r / 2U] >> (16U * (r % 2U))) & 0xFFFFU;
        }
        pairs16<F>(run, words);
    }
    STRIX_SP_INLINE static float scale(const Words& w, unsigned gi) {
        if constexpr (kGroup == 64U) return scale_of<F>(w.scale >> (16U * gi));
        else return scale_of<F>(w.scale >> (8U * gi));
    }
};

template <Kind F>
struct Staging {
    static constexpr unsigned kGroupK = group_of(F);
    static constexpr unsigned kScaleWord = 16U;
    struct Raw {
        Run run;
        std::uint32_t scale, half;
    };
    STRIX_SP_INLINE static Raw load(const std::uint8_t* row, unsigned k0, unsigned half) {
        Raw raw{};
        raw.half = half;
        if (!row) return raw;
        raw.run = load_run<F>(row, k0 + 16U * half);
        raw.scale = load_scale<F>(row, k0 + 16U * half);
        return raw;
    }
    STRIX_SP_INLINE static void publish(const Raw& raw, unsigned, std::uint32_t* dst) {
        using U4 = weight::hip::gfx1151::U4;
        std::uint32_t words[8];
        pairs16<F>(raw.run, words);
        *reinterpret_cast<U4*>(dst) = U4{words[0], words[1], words[2], words[3]};
        *reinterpret_cast<U4*>(dst + 4U) = U4{words[4], words[5], words[6], words[7]};
        if (raw.half == 0U) dst[kScaleWord] = raw.scale;
    }
    STRIX_SP_INLINE static float scale(const std::uint32_t* row) { return scale_of<F>(row[kScaleWord]); }
};

// Staging trait with the scale folded into the operand (the arithmetic class of the Q8_0 staging): the FP16 operand
// is fp16(level) * fp16(scale), one packed FP16 product per pair, and the tile core runs its plain WMMA chain.
template <Kind F>
struct StagingFolded {
    using Raw = typename Staging<F>::Raw;
    STRIX_SP_INLINE static Raw load(const std::uint8_t* row, unsigned k0, unsigned half) { return Staging<F>::load(row, k0, half); }
    STRIX_SP_INLINE static void publish(const Raw& raw, unsigned, std::uint32_t* dst) {
        using U4 = weight::hip::gfx1151::U4;
        std::uint32_t words[8];
        pairs16<F>(raw.run, words);
        const auto d = static_cast<_Float16>(scale_of<F>(raw.scale));
        const Half2 d2{d, d};
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) words[i] = __builtin_bit_cast(std::uint32_t, __builtin_bit_cast(Half2, words[i]) * d2);
        *reinterpret_cast<U4*>(dst) = U4{words[0], words[1], words[2], words[3]};
        *reinterpret_cast<U4*>(dst + 4U) = U4{words[4], words[5], words[6], words[7]};
    }
};

template <Kind F>
struct Chunk {
    Run run;
    float d;
    __device__ Chunk() = default;
    __device__ __forceinline__ Chunk(const std::uint8_t* row, unsigned i)
        : run(load_run<F>(row, i)), d(scale_of<F>(load_scale<F>(row, i))) {}
    __device__ __forceinline__ float at(unsigned e) const {
        if constexpr (F == Kind::p4 || F == Kind::p4b) {
            const unsigned i = e % 8U;
            const unsigned u = ((e < 8U ? run.c0 : run.c1) >> (8U * (i % 4U) + 4U * (i / 4U))) & 15U;
            return d * static_cast<float>(static_cast<int>(u) - 8);
        } else {
            unsigned u = (run.c0 >> field2_shift(e)) & 3U;
            if constexpr (F == Kind::p3) {
                u |= ((run.hi >> e) & 1U) << 2U;
                return d * static_cast<float>(2 * static_cast<int>(u) - 7);
            } else if constexpr (F == Kind::p2) {
                return d * static_cast<float>(2 * static_cast<int>(u) - 3);
            } else {
                return d * static_cast<float>(level_of(F, u));
            }
        }
    }
};

// The 32 elements of block b (elements 32 b .. + 32) as code bytes, q[i] byte j = element 4 i + j, and the block's
// scale. Linear kinds keep the raw unsigned code: level = kMul * u - kOff, so a dot against activations a is
// kMul * dot(a, u) - kOff * sum(a); q8 and p2t give signed levels.
template <Kind F>
struct A8Block {
    static constexpr bool kSigned = F == Kind::q8 || F == Kind::p2t;
    static constexpr int kMul = F == Kind::p3 || F == Kind::p2 ? 2 : 1;
    static constexpr int kOff = F == Kind::p4 || F == Kind::p4b ? 8 : F == Kind::p3 ? 7 : F == Kind::p2 ? 3 : 0;
    std::uint32_t q[8];
    float d;
    STRIX_SP_INLINE void load(const std::uint8_t* row, unsigned b) {
        const unsigned t = b % 4U;
        const std::uint8_t* block = row + std::size_t{b / 4U} * code_bytes(F);
        if constexpr (F == Kind::q8) {
            const Words4 v0 = *(Words4G*)(block + 32U * t), v1 = *(Words4G*)(block + 32U * t + 16U);
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) {
                q[i] = v0[i];
                q[4U + i] = v1[i];
            }
        } else if constexpr (F == Kind::p4 || F == Kind::p4b) {
            const Words4 v = *(Words4G*)(block + 16U * t);
#pragma unroll
            for (unsigned p = 0; p < 4U; ++p) split4(v[p], q[2U * p], q[2U * p + 1U]);
        } else {
            split2(*(WordsG*)(block + 8U * t), q);
            split2(*(WordsG*)(block + 8U * t + 4U), q + 4U);
            if constexpr (F == Kind::p3) {
                const std::uint32_t hi = *(WordsG*)(block + 32U + 4U * t);
#pragma unroll
                for (unsigned i = 0; i < 8U; ++i) q[i] |= bit_bytes(hi >> (4U * i)) << 2U;
            }
            if constexpr (F == Kind::p2t) {
#pragma unroll
                for (unsigned i = 0; i < 8U; ++i) q[i] = perm(0U, 0x0401FFFCU, q[i]);
            }
        }
        d = scale_of<F>(load_scale<F>(row, 32U * b));
    }
};

#undef STRIX_SP_INLINE

}  // namespace strix::core::strixpack
#endif
