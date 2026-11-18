#pragma once
// Contract dense-iu4a4-rot-v1: the rotated W4A4 class of the shared dense tall GEMM on integer WMMA (IU4 x IU4).
// y[m][n] = sum_k x[m][k] w[n][k] is computed as (R x)[m] . (W R)[n] with R = H128 / sqrt(128) (a4_tiles::fwht128:
// symmetric, orthogonal, block diagonal over 128 inputs), both operands quantised to int4 codes in [-7, 7]:
//   Activations, per row m, groups of kGroupA = 32 rotated inputs:
//     amax_g = max |y| over the group; amax_m = max_g amax_g; base_m = amax_m / (7 * kUMax) (FP32);
//     u_g = 1 for amax_g == 0, else min(kUMax, max(1, ceil(amax_g / (7 * base_m)))) (an INTEGER relative scale);
//     step_g = base_m * u_g; code = a4_code_f(y, step_g) (rint, clamp +-7, NaN -> 0).
//     A zero row has base 0 (codes 0, u 1). A non-finite row maximum gives base = that value (the output is NaN).
//   Weights, once at load, from the IU4 g64 pack (core/iu4.hpp layout, dense_linear_iu4.hpp:150): per column n and
//     block of kBlock = 128 inputs: dequantise (q * fp16 scale, exact in FP32), fwht128, step = a4_step_f(amax),
//     code = a4_code_f. Output: the same tile layout as the pack (codes) + FP32 steps [N][K/128].
//   Dot (bit-defined; the kernel performs exactly these operations):
//     C = cumulative integer dot over k (never reset; |C| <= K * 49 < 2^23 for K <= kMaxInputs);
//     per block b: t = lead_b * C_before_b + sum_{g in b} e_g * C_after_g, where e_g = u_g - u_{g+1} (u past the
//     block's last group = 0) and lead_b = -u_{first group of b}. By Abel summation t = sum_{g in b} u_g * P_g exactly
//     (P_g = the group's integer dot), so the per-A-group scale costs one integer multiply-add per output element and
//     no conversion. |t| <= 4 * 32 * 49 * kUMax < 2^24, so float(t) is exact;
//     acc = fmaf(float(t), wstep[n][b], acc) in ascending b; y = acc * base_m.
// Why this form (wave4 sd_i4 STATE (4)): with FP group scales every A group pays convert + scale product + FMA per
// output element (measured 65 TOPS of 109 at a g32); here an A group pays 1 VALU op per element and a W block 2.
// Model-agnostic. Home: shared (backend core; host + device rule, no model names).
#include "a4_tiles.hpp"
#include "iu4_onfly.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define STRIX_ROT_HD __host__ __device__
#else
#define STRIX_ROT_HD
#endif

namespace strix::core::iu4a4_rot {
inline constexpr const char* kIdentity = "dense-iu4a4-rot-v1";
inline constexpr unsigned kBlock = 128;   // Hadamard block = weight group
inline constexpr unsigned kGroupA = 32;   // activation group
inline constexpr unsigned kGroupsPerBlock = kBlock / kGroupA;
inline constexpr int kUMax = 255;
inline constexpr unsigned kMaxInputs = 131072;  // keeps |C| < 2^23 (a signed 24-bit multiply operand)
static_assert(std::int64_t{kMaxInputs} * 49 < (std::int64_t{1} << 23), "cumulative C must fit a signed 24-bit operand");
static_assert(std::int64_t{kBlock} * 49 * kUMax < (std::int64_t{1} << 24), "a block's t must convert to FP32 exactly");

// Activation side, one row of `in` inputs (in % kBlock == 0, in <= kMaxInputs). `y` holds the row and is rotated in
// place; codes[in], u[in / kGroupA]. Returns base_m.
STRIX_ROT_HD inline float quantise_row(float* y, unsigned in, std::int8_t* codes, std::uint8_t* u) {
    float amax_m = 0.0f, bad = 0.0f;
    bool finite = true;
    for (unsigned b = 0; b < in; b += kBlock) a4_tiles::fwht128(y + b);
    for (unsigned k = 0; k < in; ++k) {
        const float a = a4_tiles::fabs_(y[k]);
        if (!(a <= 3.402823466e38f)) {  // NaN or Inf: the first one becomes the base
            if (finite) bad = y[k];
            finite = false;
        } else if (a > amax_m) {
            amax_m = a;
        }
    }
    const float base = finite ? amax_m / (7.0f * static_cast<float>(kUMax)) : bad;
    for (unsigned g = 0; g < in / kGroupA; ++g) {
        const float* v = y + g * kGroupA;
        float amax = 0.0f;
        for (unsigned i = 0; i < kGroupA; ++i) amax = a4_tiles::fabs_(v[i]) > amax ? a4_tiles::fabs_(v[i]) : amax;
        int ug = 1;
        if (finite && base > 0.0f && amax > 0.0f) {
            const float q = amax / (7.0f * base);
            ug = q >= static_cast<float>(kUMax) ? kUMax : static_cast<int>(::ceilf(q));
            ug = ug < 1 ? 1 : ug;
        }
        u[g] = static_cast<std::uint8_t>(ug);
        const float step = base * static_cast<float>(ug);
        for (unsigned i = 0; i < kGroupA; ++i)
            codes[g * kGroupA + i] = static_cast<std::int8_t>(finite && base > 0.0f ? integer_math::a4_code_f(v[i], step) : 0);
    }
    return base;
}

// Abel coefficients of one block of row m: e[0..3] and lead (see the header).
struct BlockCoef {
    int e[kGroupsPerBlock];
    int lead;
};
STRIX_ROT_HD inline BlockCoef block_coef(const std::uint8_t* u_block) {
    BlockCoef c{};
    for (unsigned g = 0; g < kGroupsPerBlock; ++g)
        c.e[g] = int(u_block[g]) - (g + 1U < kGroupsPerBlock ? int(u_block[g + 1U]) : 0);
    c.lead = -int(u_block[0]);
    return c;
}

// Byte of input k, column n in the IU4 tile layout (N columns, N % 16 == 0); low nibble = even k.
STRIX_ROT_HD inline std::size_t pack_byte(unsigned k, unsigned n, unsigned outputs) {
    return ((std::size_t(k / 16U) * (outputs / 16U) + n / 16U) * 16U + n % 16U) * 8U + (k % 16U) / 2U;
}
STRIX_ROT_HD inline int pack_code(const std::uint8_t* packed, unsigned k, unsigned n, unsigned outputs) {
    const unsigned nib = (packed[pack_byte(k, n, outputs)] >> ((k & 1U) * 4U)) & 15U;
    return nib < 8U ? int(nib) : int(nib) - 16;
}
STRIX_ROT_HD inline void put_code(std::uint8_t* packed, unsigned k, unsigned n, unsigned outputs, int code) {
    std::uint8_t& b = packed[pack_byte(k, n, outputs)];
    const unsigned sh = (k & 1U) * 4U;
    b = static_cast<std::uint8_t>((b & ~(15U << sh)) | ((static_cast<unsigned>(code) & 15U) << sh));
}

// Weight side, one column n of one block b, from the g64 pack (codes + fp16 scales [N][K/64] as raw bits) into the
// rotated pack (same tile layout) + FP32 step wstep[n * (K / kBlock) + b]. Load-time; host or one device thread.
STRIX_ROT_HD inline void requantise_block(const std::uint8_t* src, const std::uint16_t* src_scales, unsigned in, unsigned outputs,
                                         unsigned n, unsigned b, std::uint8_t* dst, float* wstep) {
    float v[kBlock];
    for (unsigned i = 0; i < kBlock; ++i) {
        const unsigned k = b * kBlock + i;
        const float s = onfly::f16_bits_to_f32(src_scales[std::size_t(n) * (in / 64U) + k / 64U]);
        v[i] = static_cast<float>(pack_code(src, k, n, outputs)) * s;
    }
    a4_tiles::fwht128(v);
    float amax = 0.0f;
    for (unsigned i = 0; i < kBlock; ++i) amax = a4_tiles::fabs_(v[i]) > amax ? a4_tiles::fabs_(v[i]) : amax;
    const float step = integer_math::a4_step_f(amax);
    for (unsigned i = 0; i < kBlock; ++i) put_code(dst, b * kBlock + i, n, outputs, integer_math::a4_code_f(v[i], step));
    wstep[std::size_t(n) * (in / kBlock) + b] = step;
}

// The bit-defined dot of one output (row codes/u/base, rotated column n of the requantised pack).
STRIX_ROT_HD inline float dot(const std::int8_t* acodes, const std::uint8_t* u, float base, const std::uint8_t* wpack,
                             const float* wstep, unsigned in, unsigned outputs, unsigned n) {
    std::int32_t c = 0;
    float acc = 0.0f;
    for (unsigned b = 0; b < in / kBlock; ++b) {
        const BlockCoef bc = block_coef(u + b * kGroupsPerBlock);
        std::int32_t t = bc.lead * c;
        for (unsigned g = 0; g < kGroupsPerBlock; ++g) {
            for (unsigned i = 0; i < kGroupA; ++i) {
                const unsigned k = b * kBlock + g * kGroupA + i;
                c += int(acodes[k]) * pack_code(wpack, k, n, outputs);
            }
            t += bc.e[g] * c;
        }
        acc = ::fmaf(static_cast<float>(t), wstep[std::size_t(n) * (in / kBlock) + b], acc);
    }
    return acc * base;
}
}  // namespace strix::core::iu4a4_rot
#undef STRIX_ROT_HD
