#pragma once

// Shared HC lane chain: the per-lane serial dot of a quantised HC weight row (the low-rank projection and the
// output gate of a multi-stream hyper-connection). One lane owns one weight row (one output feature / one rank
// index) and walks it block by block; every launcher form (window, deferred, row tiles, row split, register
// resident, persistent, wide) repeats the same chain and differs only in where the codes and the inputs live.
//
// Declared arithmetic of one 32-code block, for each row t of the chain (t < Rows):
//   coefficient(k) = scale * float(code(k))            k = 0 .. 31 ascending
//   acc(t)        += x(t, k) * coefficient(k)          in k order, rows independent, carried across blocks
// scale is the block's FP16 scale widened to FP32, code a signed 8-bit value. The coefficient is exact in FP32
// (an 11-bit significand times |code| <= 128), so the product order x * coefficient versus coefficient * x is
// immaterial; the accumulation is written as the original sites write it and the backend contracts it exactly as
// it did there (HIP: one fused multiply-add per step under the TU's contraction mode).
//
// Policies (template parameters; every form is the same chain):
//   Unrolled  the k loop (and the row loop) carry #pragma unroll, as the register-tile sites write them; false keeps
//             the plain loops of the scalar sites so their schedule is unchanged;
//   Rows      rows advancing together on one decoded coefficient (Count rows of a row tile; 1 = one chain);
//   code(k)   the k-th signed code of the block (Q8_0 byte, staged word, register word ...);
//   x(t, k)   the k-th input of row t for this block; acc(t) the float accumulator of row t.
// The eight-code form takes code words of eight codes (byte l = code 8g + l), as the Q6P32 copies store them, and
// reads its inputs as x(t, g, l) = input 8g + l of row t (the sites form that index from g and l themselves).
// Tile byte geometry is backend-neutral and lives in core/hc_geometry.hpp (LaneChainTile).

#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define STRIX_HC_LANE_HD __host__ __device__ __forceinline__
// Accessor lambdas passed to the chain are inlined with the chain itself (before the optimiser sees the loop), so a
// bound site compiles from the same IR as the inline loop it replaces.
#define STRIX_HC_LANE_ACCESSOR __attribute__((always_inline))
#define STRIX_HC_LANE_UNROLL _Pragma("unroll")
#else
#define STRIX_HC_LANE_ACCESSOR
#define STRIX_HC_LANE_UNROLL
#define STRIX_HC_LANE_HD inline
#endif

namespace strix::core::hc {

STRIX_HC_LANE_HD float lane_chain_coefficient(float scale, signed char code) {
    return scale * static_cast<float>(code);
}

template <bool Unrolled, std::uint32_t Rows, class Code, class X, class Acc>
STRIX_HC_LANE_HD void lane_chain_block(float scale, Code code, X x, Acc acc) {
    static_assert(Rows != 0U);
    if constexpr (Unrolled) {
STRIX_HC_LANE_UNROLL
        for (std::uint32_t k = 0; k < 32U; ++k) {
            const float coefficient = lane_chain_coefficient(scale, code(k));
STRIX_HC_LANE_UNROLL
            for (std::uint32_t t = 0; t < Rows; ++t) acc(t) += x(t, k) * coefficient;
        }
    } else if constexpr (Rows == 1U) {
        for (std::uint32_t k = 0; k < 32U; ++k) {
            const float coefficient = lane_chain_coefficient(scale, code(k));
            acc(0U) += x(0U, k) * coefficient;
        }
    } else {
        for (std::uint32_t k = 0; k < 32U; ++k) {
            const float coefficient = lane_chain_coefficient(scale, code(k));
            for (std::uint32_t t = 0; t < Rows; ++t) acc(t) += x(t, k) * coefficient;
        }
    }
}

// Eight-code words: words(g) holds codes 8g .. 8g + 7 of the block, byte l = code 8g + l (g = 0 .. 3); x(t, g, l).
template <bool Unrolled, std::uint32_t Rows, class Words, class X, class Acc>
STRIX_HC_LANE_HD void lane_chain_block8(float scale, Words words, X x, Acc acc) {
    static_assert(Rows != 0U);
    static_assert(Unrolled || Rows == 1U, "the plain eight-code chain is the one-row form");
    if constexpr (Unrolled) {
STRIX_HC_LANE_UNROLL
        for (std::uint32_t g = 0; g < 4U; ++g) {
            const std::uint64_t codes = words(g);
STRIX_HC_LANE_UNROLL
            for (std::uint32_t l = 0; l < 8U; ++l) {
                const float coefficient = lane_chain_coefficient(scale, static_cast<signed char>(codes >> (8U * l)));
STRIX_HC_LANE_UNROLL
                for (std::uint32_t t = 0; t < Rows; ++t) acc(t) += x(t, g, l) * coefficient;
            }
        }
    } else {
        for (std::uint32_t g = 0; g < 4U; ++g) {
            const std::uint64_t codes = words(g);
            for (std::uint32_t l = 0; l < 8U; ++l) {
                const float coefficient = lane_chain_coefficient(scale, static_cast<signed char>(codes >> (8U * l)));
                acc(0U) += x(0U, g, l) * coefficient;
            }
        }
    }
}

// Little-endian FP16 scale bits of a Q8_0 block (bytes 0 and 1), as every staged-byte site assembles them.
STRIX_HC_LANE_HD unsigned short lane_chain_q8_scale_bits(const unsigned char* block) {
    return static_cast<unsigned short>(static_cast<unsigned>(block[0]) | (static_cast<unsigned>(block[1]) << 8U));
}

#if defined(__HIPCC__)
__device__ __forceinline__ float lane_chain_q8_scale(const unsigned char* block) {
    return __half2float(__ushort_as_half(lane_chain_q8_scale_bits(block)));
}

// One Q8_0 block (34 bytes: FP16 scale, 32 codes) read from staged bytes.
template <bool Unrolled, std::uint32_t Rows, class X, class Acc>
__device__ __forceinline__ void lane_chain_q8_block(const unsigned char* block, X x, Acc acc) {
    const float scale = lane_chain_q8_scale(block);
    lane_chain_block<Unrolled, Rows>(scale, [block](std::uint32_t k) STRIX_HC_LANE_ACCESSOR { return static_cast<signed char>(block[2U + k]); }, x, acc);
}
#endif

}  // namespace strix::core::hc

#undef STRIX_HC_LANE_HD
#undef STRIX_HC_LANE_UNROLL
