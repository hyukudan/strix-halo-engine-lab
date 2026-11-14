#pragma once
// Test-only scalar shim: compiles the actual GGUF decoder on a CPU. It does
// not emulate a GPU wave, validate WMMA lane layout or establish device equality.
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#define __builtin_memcpy std::memcpy
#define __device__
#define __forceinline__ inline
inline struct HostDim { unsigned x = 0; } threadIdx, blockDim{32};
#if defined(_MSC_VER)
inline unsigned __builtin_popcount(unsigned v) { return std::popcount(v); }
#endif
inline float __uint_as_float(unsigned v) { return std::bit_cast<float>(v); }
inline float __int_as_float(int v) { return std::bit_cast<float>(v); }
inline unsigned __builtin_amdgcn_perm(unsigned a, unsigned b, unsigned selectors) {
    unsigned out = 0;
    for (unsigned j = 0; j < 4; ++j) {
        unsigned s = (selectors >> (8 * j)) & 255U;
        unsigned byte = s < 4 ? (b >> (8 * s)) & 255U :
                        (s < 8 ? (a >> (8 * (s - 4))) & 255U : (s == 13 ? 255U : 0U));
        out |= byte << (8 * j);
    }
    return out;
}
template<class V4, class V8>
inline V8 __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(bool, V4 a, bool, V4 b, V8 c, bool) {
    int dot = 0;
    for (unsigned j = 0; j < 16; ++j) {
        const int av = int(std::int8_t(unsigned(a[j / 4]) >> (8 * (j % 4))));
        const int bv = int(std::int8_t(unsigned(b[j / 4]) >> (8 * (j % 4))));
        dot += av * bv;
    }
    for (auto& x : c) x += dot;
    return c;
}
inline float __shfl_xor(float v, int, int) { return v; }
inline int __shfl_xor(int v, int, int) { return v; }

template<class V2, class V8>
inline V8 __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(bool, V2 a, bool, V2 b, V8 c, bool) {
    int dot = 0;
    for (unsigned j = 0; j < 16; ++j) {
        const unsigned aq = (unsigned(a[j / 8]) >> (4 * (j % 8))) & 15U;
        const unsigned bq = (unsigned(b[j / 8]) >> (4 * (j % 8))) & 15U;
        dot += (int(aq & 7U) - int(aq & 8U)) * (int(bq & 7U) - int(bq & 8U));
    }
    for (auto& x : c) x += dot;
    return c;
}
