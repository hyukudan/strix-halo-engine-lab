#pragma once

#include "core/strixpack_cpu_impl.hpp"

#include <immintrin.h>

namespace strix::weight::sp::cpu_detail {
namespace {

template <class T>
T unaligned(const std::uint8_t* data) {
    T word;
    std::memcpy(&word, data, sizeof word);
    return word;
}

#if defined(STRIX_SP_WIDE)
using Vec = __m512;
constexpr unsigned lanes = 16;
template <class S>
Vec values(const std::uint8_t* block, unsigned j, Vec table, float scale) {
    constexpr unsigned low_bits = S::bits == 4U ? 4U : 2U;
    const auto lo = unaligned<int>(block + j * low_bits / 8U);
    auto low = _mm512_set1_epi32(lo);
    if constexpr (low_bits == 4U)
        low = _mm512_mask_set1_epi32(low, 0xff00, unaligned<int>(block + j / 2U + 4U));
    const auto lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const auto index_shift = _mm512_and_si512(lane, _mm512_set1_epi32(low_bits == 4U ? 7 : 15));
    const auto shift = _mm512_slli_epi32(index_shift, low_bits == 4U ? 2 : 1);
    auto code = _mm512_and_si512(_mm512_maskz_srlv_epi32(0xffff, low, shift),
                                 _mm512_set1_epi32((1 << low_bits) - 1));
    if constexpr (S::bits == 3U) {
        const auto sign = _mm512_set1_epi32(unaligned<std::uint16_t>(block + 32U + j / 8U));
        code = _mm512_or_si512(code, _mm512_and_si512(
            _mm512_slli_epi32(_mm512_maskz_srlv_epi32(0xffff, sign, lane), 2), _mm512_set1_epi32(4)));
    }
    return _mm512_mul_ps(_mm512_maskz_permutexvar_ps(0xffff, code, table), _mm512_set1_ps(scale));
}
Vec load(const float* p) { return _mm512_loadu_ps(p); }
void store(float* p, Vec v) { _mm512_storeu_ps(p, v); }
template <unsigned Half>
__m256 half(Vec v) { return _mm512_maskz_extractf32x8_ps(0xff, v, Half); }
#else
using Vec = __m256;
constexpr unsigned lanes = 8;
template <class S>
Vec values(const std::uint8_t* block, unsigned j, Vec table, float scale) {
    constexpr unsigned low_bits = S::bits == 4U ? 4U : 2U;
    int low;
    if constexpr (low_bits == 4U) low = unaligned<int>(block + j / 2U);
    else low = unaligned<std::uint16_t>(block + j / 4U);
    const auto lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const auto shift = _mm256_slli_epi32(lane, low_bits == 4U ? 2 : 1);
    auto code = _mm256_srlv_epi32(_mm256_set1_epi32(low), shift);
    if constexpr (S::bits == 3U) {
        const auto sign = _mm256_set1_epi32(block[32U + j / 8U]);
        code = _mm256_or_si256(_mm256_and_si256(code, _mm256_set1_epi32(3)),
            _mm256_and_si256(_mm256_slli_epi32(_mm256_srlv_epi32(sign, lane), 2), _mm256_set1_epi32(4)));
    }
    auto mag = _mm256_permutevar8x32_ps(table, code);
    if constexpr (S::bits == 4U)
        mag = _mm256_xor_ps(mag, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(code, _mm256_set1_epi32(8)), 28)));
    return _mm256_mul_ps(mag, _mm256_set1_ps(scale));
}
Vec load(const float* p) { return _mm256_loadu_ps(p); }
void store(float* p, Vec v) { _mm256_storeu_ps(p, v); }
template <unsigned Half>
__m256 half(Vec v) {
    static_assert(Half == 0);
    return v;
}
#endif

template <class S>
Vec make_table(const Header& h) {
    float lut[lanes];
    for (unsigned i = 0; i < lanes; ++i) lut[i] = level<S>(h, i % (2U * S::levels));
    return load(lut);
}

template <class S>
void decode_rows(const Header& h, const std::uint8_t* block, std::size_t rows, std::size_t k, float* out) {
    const auto table = make_table<S>(h);
    for (std::size_t i = 0; i < rows * (k / S::elements); ++i, block += S::bytes)
        for (unsigned g = 0; g < S::groups; ++g) {
            const auto scale = scale_value(S::scale_byte(block, g), h.base);
            for (unsigned j = g * S::group; j < (g + 1U) * S::group; j += lanes, out += lanes)
                store(out, values<S>(block, j, table, scale));
        }
}

// Eight rows of decoded weights -> eight vectors across rows, consumed in
// ascending K. FMA lanes own independent outputs, so no reduction is needed.
template <unsigned Half>
__m256 accumulate(const Vec (&v)[8], const float* x, __m256 sum) {
    const auto a0 = _mm256_unpacklo_ps(half<Half>(v[0]), half<Half>(v[1]));
    const auto a1 = _mm256_unpackhi_ps(half<Half>(v[0]), half<Half>(v[1]));
    const auto a2 = _mm256_unpacklo_ps(half<Half>(v[2]), half<Half>(v[3]));
    const auto a3 = _mm256_unpackhi_ps(half<Half>(v[2]), half<Half>(v[3]));
    const auto a4 = _mm256_unpacklo_ps(half<Half>(v[4]), half<Half>(v[5]));
    const auto a5 = _mm256_unpackhi_ps(half<Half>(v[4]), half<Half>(v[5]));
    const auto a6 = _mm256_unpacklo_ps(half<Half>(v[6]), half<Half>(v[7]));
    const auto a7 = _mm256_unpackhi_ps(half<Half>(v[6]), half<Half>(v[7]));
    const __m256 lo[] = {_mm256_shuffle_ps(a0, a2, 0x44), _mm256_shuffle_ps(a0, a2, 0xee),
                         _mm256_shuffle_ps(a1, a3, 0x44), _mm256_shuffle_ps(a1, a3, 0xee)};
    const __m256 hi[] = {_mm256_shuffle_ps(a4, a6, 0x44), _mm256_shuffle_ps(a4, a6, 0xee),
                         _mm256_shuffle_ps(a5, a7, 0x44), _mm256_shuffle_ps(a5, a7, 0xee)};
    for (unsigned j = 0; j < 4; ++j)
        sum = _mm256_fmadd_ps(_mm256_set1_ps(x[j]), _mm256_permute2f128_ps(lo[j], hi[j], 0x20), sum);
    for (unsigned j = 0; j < 4; ++j)
        sum = _mm256_fmadd_ps(_mm256_set1_ps(x[j + 4]), _mm256_permute2f128_ps(lo[j], hi[j], 0x31), sum);
    return sum;
}

template <class S>
void project_rows(const Header& h, const std::uint8_t* bytes, std::size_t rows, std::size_t k, const float* x, float* y) {
    constexpr std::size_t tile = 8;
    const std::size_t pitch = S::row_bytes(k);
    const auto table = make_table<S>(h);
    const auto full = rows / tile * tile;
    for (std::size_t row = 0; row < full; row += tile, bytes += tile * pitch) {
        auto sum = _mm256_setzero_ps();
        const auto* block = bytes;
        for (std::size_t col = 0; col < k; col += S::elements, block += S::bytes)
            for (unsigned g = 0; g < S::groups; ++g) {
                float scale[tile];
                for (unsigned r = 0; r < tile; ++r)
                    scale[r] = scale_value(S::scale_byte(block + r * pitch, g), h.base);
                for (unsigned j = g * S::group; j < (g + 1U) * S::group; j += lanes) {
                    Vec v[tile];
                    for (unsigned r = 0; r < tile; ++r) v[r] = values<S>(block + r * pitch, j, table, scale[r]);
                    sum = accumulate<0>(v, x + col + j, sum);
                    if constexpr (lanes == 16) sum = accumulate<1>(v, x + col + j + 8, sum);
                }
            }
        _mm256_storeu_ps(y + row, sum);
    }
    if (full < rows) cpu_project(S::bits, CpuIsa::scalar)(h, bytes, rows - full, k, x, y + full);
}

}  // namespace

#if defined(STRIX_SP_WIDE)
const Operations avx512[]
#else
const Operations avx2[]
#endif
    {{decode_rows<SP2Storage>, project_rows<SP2Storage>},
     {decode_rows<SP3Storage>, project_rows<SP3Storage>},
     {decode_rows<SP4Storage>, project_rows<SP4Storage>}};

}  // namespace strix::weight::sp::cpu_detail
