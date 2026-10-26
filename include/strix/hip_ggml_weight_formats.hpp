#pragma once

// Device decode of one weight element for the GSQ-RCO GGUF formats (Q2_0, Q4_0, Q5_0, Q3_K, Q6_K,
// IQ2_S, IQ3_XXS, IQ3_S; Q4_K/Q5_K/Q8_0/IQ4_NL/IQ4_XS/Q5_1 stay in each launcher). Every value is the
// same float operations, in the same order, as strix::ggml_dequant (= llama.cpp's dequantize_row_*),
// written with __fmul_rn / __fadd_rn so no contraction can change a bit. Row-major ggml blocks:
// element (row, feature) lives in block row * (row_width / block) + feature / block.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::hip_ggml {

inline constexpr std::uint32_t type_q4_0 = 2, type_q5_0 = 6, type_q8_0 = 8, type_q3_k = 11, type_q4_k = 12,
                               type_q5_k = 13, type_q6_k = 14, type_iq3_xxs = 18, type_iq4_nl = 20, type_iq3_s = 21,
                               type_iq2_s = 22, type_iq4_xs = 23, type_q2_0 = 42;

// Kinds a launcher can add on top of its own (F32/F16/BF16/Q8_0/IQ4_XS everywhere): X(kind, ggml type).
// Launchers that already decode one of these leave it out of their list.
#define STRIX_GGML_KINDS_K(X) X(q4_k, 12) X(q5_k, 13)
#define STRIX_GGML_KINDS_GSQ(X) X(q2_0, 42) X(q4_0, 2) X(q5_0, 6) X(q3_k, 11) X(q6_k, 14)     X(iq2_s, 22) X(iq3_xxs, 18) X(iq3_s, 21)

struct Geometry { std::uint32_t elements, bytes; };

__host__ __device__ constexpr Geometry geometry(std::uint32_t type) {
    return type == type_q2_0 ? Geometry{64, 18} : type == type_q4_0 ? Geometry{32, 18}
         : type == type_q5_0 ? Geometry{32, 22} : type == type_q3_k ? Geometry{256, 110}
         : type == type_q6_k ? Geometry{256, 210} : type == type_iq2_s ? Geometry{256, 82}
         : type == type_iq3_xxs ? Geometry{256, 98} : type == type_iq3_s ? Geometry{256, 110}
         : type == type_q4_k ? Geometry{256, 144} : type == type_q5_k ? Geometry{256, 176}
         : type == type_iq4_nl ? Geometry{32, 18} : type == type_iq4_xs ? Geometry{256, 136}
         : type == type_q8_0 ? Geometry{32, 34} : Geometry{0, 0};
}

__host__ constexpr bool supported(std::uint32_t type) { return geometry(type).elements != 0; }

// Grid tables (llama.cpp format data, see the .inc headers).
__device__ __constant__ inline std::uint64_t iq2s_grid[1024] = {
#include "strix/qwen_iq2_s_grid.inc"
};
__device__ __constant__ inline std::uint32_t iq3xxs_grid[256] = {
#include "strix/qwen_iq3_xxs_grid.inc"
};
__device__ __constant__ inline std::uint32_t iq3s_grid[512] = {
#include "strix/qwen_iq3_s_grid.inc"
};

__device__ inline float half_bits(const std::uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | (p[1] << 8))));
}
__device__ inline std::uint32_t le32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
__device__ inline std::uint8_t grid_byte(std::uint64_t entry, std::uint32_t j) {
    return static_cast<std::uint8_t>(entry >> (8U * j));
}
__device__ inline float signed_mag(float scaled, bool negative) {  // (dl * grid) * (+-1.f): exact sign flip
    return __fmul_rn(scaled, negative ? -1.0F : 1.0F);
}

// Element e (0 <= e < block elements) of one block.
template <std::uint32_t Type>
__device__ float block_element(const std::uint8_t* x, std::uint32_t e);

template <>
__device__ inline float block_element<type_q2_0>(const std::uint8_t* x, std::uint32_t e) {
    const auto q = static_cast<int>((x[2 + e / 4U] >> ((e % 4U) * 2U)) & 3U);
    return __fmul_rn(static_cast<float>(q - 1), half_bits(x));
}
template <>
__device__ inline float block_element<type_q4_0>(const std::uint8_t* x, std::uint32_t e) {
    const auto byte = x[2 + e % 16U];
    const int v = static_cast<int>(e < 16U ? (byte & 0x0FU) : (byte >> 4U)) - 8;
    return __fmul_rn(static_cast<float>(v), half_bits(x));
}
template <>
__device__ inline float block_element<type_q5_0>(const std::uint8_t* x, std::uint32_t e) {
    const auto byte = x[6 + e % 16U];
    const auto high = (le32(x + 2) >> e) & 1U;
    const int v = static_cast<int>((e < 16U ? (byte & 0x0FU) : (byte >> 4U)) | (high << 4U)) - 16;
    return __fmul_rn(static_cast<float>(v), half_bits(x));
}
// block_q3_K: hmask[32], qs[64], scales[12], d
template <>
__device__ inline float block_element<type_q3_k>(const std::uint8_t* x, std::uint32_t e) {
    const std::uint32_t n = e / 128U, j = (e % 128U) / 32U, sub = (e % 32U) / 16U, l = e % 16U;
    const std::uint32_t is = n * 8U + j * 2U + sub;
    const std::uint8_t* scales = x + 96;
    const std::uint32_t low = is < 8U ? (scales[is] & 0x0FU) : (scales[is - 8U] >> 4U);
    const std::uint32_t high = (scales[8U + is % 4U] >> (2U * (is / 4U))) & 3U;
    const int scale = static_cast<int>(low | (high << 4U)) - 32;
    const float dl = __fmul_rn(half_bits(x + 108), static_cast<float>(scale));
    const auto q = (x[32U + n * 32U + sub * 16U + l] >> (2U * j)) & 3U;
    const bool set = (x[sub * 16U + l] & (1U << (n * 4U + j))) != 0U;
    return __fmul_rn(dl, static_cast<float>(static_cast<int>(q) - (set ? 0 : 4)));
}
// block_q6_K: ql[128], qh[64], scales[16] (int8), d
template <>
__device__ inline float block_element<type_q6_k>(const std::uint8_t* x, std::uint32_t e) {
    const std::uint32_t n = e / 128U, r = e % 128U, quarter = r / 32U, l = r % 32U;
    const auto low = (x[n * 64U + l + (quarter & 1U) * 32U] >> (4U * (quarter >> 1U))) & 0x0FU;
    const auto high = (x[128U + n * 32U + l] >> (2U * quarter)) & 3U;
    const int q = static_cast<int>(low | (high << 4U)) - 32;
    const auto sc = static_cast<std::int8_t>(x[192U + n * 8U + l / 16U + 2U * quarter]);
    return __fmul_rn(__fmul_rn(half_bits(x + 208), static_cast<float>(sc)), static_cast<float>(q));
}
// block_iq2_s: d, qs[64] (32 grid low bytes + 32 sign bytes), qh[8], scales[8]
template <>
__device__ inline float block_element<type_iq2_s>(const std::uint8_t* x, std::uint32_t e) {
    const std::uint32_t ib32 = e / 32U, l = (e % 32U) / 8U, j = e % 8U;
    const std::uint32_t s = l < 2U ? (x[74U + ib32] & 0x0FU) : (x[74U + ib32] >> 4U);
    const float dl = __fmul_rn(__fmul_rn(half_bits(x), __fadd_rn(0.5F, static_cast<float>(s))), 0.25F);
    const std::uint32_t index = x[2U + ib32 * 4U + l] | ((static_cast<std::uint32_t>(x[66U + ib32]) << (8U - 2U * l)) & 0x300U);
    const bool negative = (x[34U + ib32 * 4U + l] & (1U << j)) != 0U;
    return signed_mag(__fmul_rn(dl, static_cast<float>(grid_byte(iq2s_grid[index], j))), negative);
}
// block_iq3_xxs: d, qs[64] grid indices, then 8 x u32 scales_and_signs
template <>
__device__ inline float block_element<type_iq3_xxs>(const std::uint8_t* x, std::uint32_t e) {
    const std::uint32_t ib32 = e / 32U, l = (e % 32U) / 8U, j8 = e % 8U;
    const std::uint32_t aux = le32(x + 66U + 4U * ib32);
    const float db = __fmul_rn(__fmul_rn(half_bits(x), __fadd_rn(0.5F, static_cast<float>(aux >> 28U))), 0.5F);
    const std::uint32_t s7 = (aux >> (7U * l)) & 127U;
    const std::uint32_t signs = s7 | ((__popc(s7) & 1U) << 7U);
    const std::uint32_t index = x[2U + ib32 * 8U + 2U * l + (j8 >= 4U ? 1U : 0U)];
    const auto g = static_cast<std::uint8_t>(iq3xxs_grid[index] >> (8U * (j8 % 4U)));
    return signed_mag(__fmul_rn(db, static_cast<float>(g)), (signs & (1U << j8)) != 0U);
}
// block_iq3_s: d, qs[64], qh[8], signs[32], scales[4]
template <>
__device__ inline float block_element<type_iq3_s>(const std::uint8_t* x, std::uint32_t e) {
    const std::uint32_t ib = e / 32U, l = (e % 32U) / 8U, j8 = e % 8U;  // ib = 2 * pair + half
    const std::uint32_t s = (ib % 2U) == 0U ? (x[106U + ib / 2U] & 0x0FU) : (x[106U + ib / 2U] >> 4U);
    const float db = __fmul_rn(half_bits(x), static_cast<float>(1 + 2 * static_cast<int>(s)));
    const std::uint32_t qh = x[66U + ib];
    const std::uint32_t second = j8 >= 4U ? 1U : 0U;
    const std::uint32_t index = x[2U + ib * 8U + 2U * l + second] | ((qh << (8U - 2U * l - second)) & 256U);
    const auto g = static_cast<std::uint8_t>(iq3s_grid[index] >> (8U * (j8 % 4U)));
    return signed_mag(__fmul_rn(db, static_cast<float>(g)), (x[74U + ib * 4U + l] & (1U << j8)) != 0U);
}

// block_q4_K / block_q5_K: d, dmin, scales[12], [qh[32]], qs[128]; (d * sc) * q - dmin * m
__device__ inline void k_scale_min(const std::uint8_t* q, std::uint32_t j, std::uint32_t& d, std::uint32_t& m) {
    if (j < 4U) {
        d = q[j] & 63U; m = q[j + 4U] & 63U;
    } else {
        d = (q[j + 4U] & 0xFU) | ((static_cast<std::uint32_t>(q[j - 4U]) >> 6U) << 4U);
        m = (static_cast<std::uint32_t>(q[j + 4U]) >> 4U) | ((static_cast<std::uint32_t>(q[j]) >> 6U) << 4U);
    }
}
template <bool Q5>
__device__ inline float k_element(const std::uint8_t* x, std::uint32_t e) {
    const std::uint32_t group = e / 32U, l = e % 32U;
    const std::uint8_t* qs = x + (Q5 ? 48U : 16U);
    const auto byte = qs[(group / 2U) * 32U + l];
    std::uint32_t code = group % 2U == 0U ? (byte & 15U) : (byte >> 4U);
    if constexpr (Q5) code += (x[16U + l] & (1U << group)) != 0U ? 16U : 0U;
    std::uint32_t sc = 0, m = 0;
    k_scale_min(x + 4, group, sc, m);
    const float d1 = __fmul_rn(half_bits(x), static_cast<float>(sc));
    const float m1 = __fmul_rn(half_bits(x + 2), static_cast<float>(m));
    return __fsub_rn(__fmul_rn(d1, static_cast<float>(code)), m1);
}
template <>
__device__ inline float block_element<type_q4_k>(const std::uint8_t* x, std::uint32_t e) { return k_element<false>(x, e); }
template <>
__device__ inline float block_element<type_q5_k>(const std::uint8_t* x, std::uint32_t e) { return k_element<true>(x, e); }

__device__ __constant__ inline std::int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
template <>
__device__ inline float block_element<type_iq4_nl>(const std::uint8_t* x, std::uint32_t e) {
    const auto byte = x[2U + e % 16U];
    return __fmul_rn(half_bits(x), static_cast<float>(kvalues_iq4nl[e < 16U ? (byte & 0xFU) : (byte >> 4U)]));
}
// block_iq4_xs: d, scales_h (u16), scales_l[4], qs[128]
template <>
__device__ inline float block_element<type_iq4_xs>(const std::uint8_t* x, std::uint32_t e) {
    const std::uint32_t ib = e / 32U, w = e % 32U;
    const std::uint32_t scales_h = x[2] | (static_cast<std::uint32_t>(x[3]) << 8U);
    const int ls = static_cast<int>(((x[4U + ib / 2U] >> (4U * (ib % 2U))) & 0xFU) | (((scales_h >> (2U * ib)) & 3U) << 4U));
    const float dl = __fmul_rn(half_bits(x), static_cast<float>(ls - 32));
    const auto byte = x[8U + ib * 16U + w % 16U];
    return __fmul_rn(dl, static_cast<float>(kvalues_iq4nl[w < 16U ? (byte & 0xFU) : (byte >> 4U)]));
}
template <>
__device__ inline float block_element<type_q8_0>(const std::uint8_t* x, std::uint32_t e) {
    return __fmul_rn(static_cast<float>(static_cast<std::int8_t>(x[2U + e])), half_bits(x));
}

// Element (row, feature) of a row-major tensor of `row_width` columns.
template <std::uint32_t Type>
__device__ inline float element(const void* weights, std::uint32_t row_width, std::uint64_t row, std::uint32_t feature) {
    constexpr Geometry g = geometry(Type);
    const auto* base = static_cast<const std::uint8_t*>(weights);
    const std::uint64_t block = row * (row_width / g.elements) + feature / g.elements;
    return block_element<Type>(base + block * g.bytes, feature % g.elements);
}

}  // namespace strix::hip_ggml
