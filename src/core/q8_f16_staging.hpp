#pragma once

#include "strix/weight/host_q8.hpp"

#if defined(__HIPCC__)
#include "hip/weight/arch/gfx1151_sources.hpp"
#endif

namespace strix::core::q8_f16_staging {

inline constexpr unsigned kHalfCodes = weight::Q8Storage::elements / 2U;
#if defined(__HIPCC__)
__host__ __device__
#endif
constexpr unsigned code_offset(unsigned half) { return weight::Q8Storage::scale_bytes + half * kHalfCodes; }

// Four little-endian code words become eight pairs of FP16 operands in K order.
// The device binds the same packed decode as the Q8_0 x A8 tall core.
template <class CodeWords, class HalfWords, class DecodePair>
#if defined(__HIPCC__)
__device__ __forceinline__
#endif
void decode(const CodeWords& codes, HalfWords& words, const DecodePair& pair) {
#if defined(__HIPCC__)
#pragma unroll
#endif
    for (unsigned i = 0; i < 4U; ++i) {
        const std::uint32_t biased = codes[i] ^ 0x80808080U;
        words[2U * i] = pair(biased, 0U);
        words[2U * i + 1U] = pair(biased, 1U);
    }
}

#if defined(__HIPCC__)
struct Q8F16Staging {
    using U4 = weight::hip::gfx1151::U4;
    struct Raw {
        U4 codes;
        unsigned scale;
    };

    __device__ __forceinline__ static Raw load(const std::uint8_t* row, unsigned k0, unsigned half) {
        if (!row) return {};
        const auto* block = row + std::size_t{k0 / weight::Q8Storage::elements} * weight::Q8Storage::bytes;
        // Adjacent lanes load disjoint halves of the codes, and only the even lane loads d.
        return {*(weight::hip::gfx1151::GlobalBytes16*)(block + code_offset(half)),
                half == 0U ? *reinterpret_cast<const std::uint16_t*>(block) : 0U};
    }

    __device__ __forceinline__ static void publish(const Raw& raw, unsigned lane, std::uint32_t* dst) {
        namespace arch = weight::hip::gfx1151;
        // Share the scale after the prefetch, with both lanes participating for padded columns too.
        const auto bits = __shfl(raw.scale, static_cast<int>(lane & ~1U), 32);
        const auto scale = __builtin_bit_cast(arch::Half2, bits | (bits << 16U));
        std::uint32_t words[8];
        decode(raw.codes, words, [&](std::uint32_t biased, unsigned high) {
            return __builtin_bit_cast(std::uint32_t, arch::w16_pair(biased, high, scale));
        });
        *reinterpret_cast<U4*>(dst) = U4{words[0], words[1], words[2], words[3]};
        *reinterpret_cast<U4*>(dst + 4U) = U4{words[4], words[5], words[6], words[7]};
    }
};
#endif

}  // namespace strix::core::q8_f16_staging
