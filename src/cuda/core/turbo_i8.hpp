// Shared contract of the TURBO integer tensor-core GSQ expert route (arithmetic id gsq-iq-i8mma-turbo-cuda-v1), host and
// device. TURBO class: selected by the TURBO profile (or the explicit expert arithmetic) only; never part of EXACT or FAST.
//
// Operands (all integers are exact):
//   activations  the shared A8 quantization the int8 routes already use: gate/up input = prologue codes c_x (|c| <= 127)
//                with one FP32 delta e_x per (row, 32-value group); down input h = the A8 down quantizer of the SwiGLU
//                output (project_activation_tile: codes c_h, delta e_h per (row, slot, 32 features)).
//   weights      GSQ integer codes (GsqDecode<T>::word, |c| <= 127) and FP32 group scales s_w (GsqDecode<T>::scales; the
//                two-scale type has one scale per 16-value half).
// Per (row, output) and 32-value group g, in ascending g from acc = +0:
//   one-scale : dot = sum_32 c_x * c_w           (one m16n8k32 s8 IMMA, exact int32)
//               acc = fma(float(dot), e_x * s_w, acc)
//   two-scale : dot_a / dot_b over k 0..15 / 16..31 (two m16n8k16 s8 IMMA)
//               acc = fma(float(dot_b), e_x * s_wb, fma(float(dot_a), e_x * s_wa, acc))
// float(dot) is exact (|dot| <= 32 * 127 * 127 < 2^22), every product e_x * s_w is one FP32 rounding (__fmul_rn), every
// accumulation one FMA rounding (__fmaf_rn). The integer sums are order-free, so the whole route is bit-specified: the
// GPU result must equal this file's reference bit for bit (no hardware freedom, unlike the BF16 FAST contract).
// The gate/up output pair then goes through the shared A8 SwiGLU + down quantizer unchanged (Binding::activation_tile).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define STRIX_TURBO_I8_HD __host__ __device__ __forceinline__
#else
#define STRIX_TURBO_I8_HD inline
#endif

namespace strix::cuda::core::turbo_i8 {

inline constexpr const char* arithmetic_id = "gsq-iq-i8mma-turbo-cuda-v1";

// Exact FP32 of an int32 dot through the IMMA accumulator: C starts at the bits of 1.5 * 2^23, so D's bits are those of
// 1.5 * 2^23 + dot for |dot| < 2^22, and one FP32 subtraction recovers dot exactly (+0 for dot = 0).
inline constexpr std::uint32_t kMagicBits = 0x4B400000U;
inline constexpr float kMagic = 12582912.0F;
inline constexpr int kMaxDot = 32 * 127 * 127;
static_assert(kMaxDot < (1 << 22), "an exact group dot stays inside the magic window");
STRIX_TURBO_I8_HD float bits_f32(std::uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }
STRIX_TURBO_I8_HD float dot_float(std::uint32_t d_bits) { return bits_f32(d_bits) - kMagic; }

// Fragment geometry, g = lane / 4, q = lane % 4 (PTX ISA, mma.sync.aligned.*.row.col.s32.s8.s8.s32).
// m16n8k32: A reg r (0..3) = 4 int8 of row g + 8 (r & 1), physical k 4 q + 16 (r >> 1) + e;
//           B reg r (0..1) = 4 int8 of column g, physical k 4 q + 16 r + e;
// m16n8k16: A reg r (0..1) = row g + 8 r, physical k 4 q + e; B reg 0 = column g, physical k 4 q + e.
// C element i (0..3): row g + 8 (i >> 1), column 2 q + (i & 1) (both shapes).
STRIX_TURBO_I8_HD unsigned a_row(unsigned lane, unsigned r) { return lane / 4U + 8U * (r & 1U); }
STRIX_TURBO_I8_HD unsigned k32_a_phys(unsigned lane, unsigned r, unsigned e) { return 4U * (lane % 4U) + 16U * (r >> 1U) + e; }
STRIX_TURBO_I8_HD unsigned k32_b_phys(unsigned lane, unsigned r, unsigned e) { return 4U * (lane % 4U) + 16U * r + e; }
STRIX_TURBO_I8_HD unsigned k16_phys(unsigned lane, unsigned e) { return 4U * (lane % 4U) + e; }
STRIX_TURBO_I8_HD unsigned b_col(unsigned lane) { return lane / 4U; }
STRIX_TURBO_I8_HD unsigned c_row(unsigned lane, unsigned i) { return lane / 4U + 8U * (i >> 1U); }
STRIX_TURBO_I8_HD unsigned c_col(unsigned lane, unsigned i) { return 2U * (lane % 4U) + (i & 1U); }

// One-scale (k32) placement: physical slot p of the group carries logical k32_logical(p). Lane q's four A/B slots
// {4q..4q+3, 16+4q..16+4q+3} carry the eight consecutive logical values 8 q .. 8 q + 7, so A is ONE 8-byte load per row
// (low word -> reg 0/1, high word -> reg 2/3) and B is the packed code words 2 q (reg 0) and 2 q + 1 (reg 1). A and B use
// the same map, so the integer dot is the logical dot (a permutation of an exact sum).
STRIX_TURBO_I8_HD unsigned k32_logical(unsigned p) { return 8U * ((p & 15U) >> 2U) + 4U * (p >> 4U) + (p & 3U); }
STRIX_TURBO_I8_HD unsigned k32_word(unsigned lane, unsigned r) { return 2U * (lane % 4U) + r; }
STRIX_TURBO_I8_HD unsigned k32_a_byte(unsigned lane) { return 8U * (lane % 4U); }  // byte offset of the 8-byte row load
// Two-scale (k16 per half h): identity placement; logical k = 16 h + 4 q + e, B word q + 4 h, A 4-byte load at 16 h + 4 q.
STRIX_TURBO_I8_HD unsigned k16_word(unsigned lane, unsigned h) { return lane % 4U + 4U * h; }
STRIX_TURBO_I8_HD unsigned k16_a_byte(unsigned lane, unsigned h) { return 16U * h + 4U * (lane % 4U); }
STRIX_TURBO_I8_HD void a_regs(std::uint32_t r0x, std::uint32_t r0y, std::uint32_t r1x, std::uint32_t r1y, std::uint32_t (&a)[4]) {
    a[0] = r0x; a[1] = r1x; a[2] = r0y; a[3] = r1y;
}
// The lane whose B column is the C column of element i of this lane (its weight row owns the scale).
STRIX_TURBO_I8_HD unsigned scale_lane(unsigned lane, unsigned i) { return 8U * (lane % 4U) + 4U * (i & 1U); }

// The contract's FP32 step for one group (device: __fmul_rn / __fmaf_rn under --fmad=false; host: the same roundings).
STRIX_TURBO_I8_HD float mul_rn(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fmul_rn(a, b);
#else
    volatile float p = a * b;  // one rounding; volatile keeps the host compiler from contracting it into the FMA below
    return p;
#endif
}
STRIX_TURBO_I8_HD float fma_rn(float a, float b, float c) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);  // one rounding (C99 fmaf semantics)
#endif
}
STRIX_TURBO_I8_HD float step(float acc, float dot, float e, float s) { return fma_rn(dot, mul_rn(e, s), acc); }

// Block geometry (kernels and oracle): 8 warps; a CSR tile has <= 64 rows = 4 m16 blocks; activations are staged 128
// values (4 groups) per stage; row pitch 160 B puts the 8-byte A loads of rows g .. g + 3 on distinct bank octets.
inline constexpr unsigned kThreads = 256U, kTileRows = 64U, kProjectOutputs = 64U, kDownOutputs = 128U, kStageK = 128U;
inline constexpr unsigned kPitch = kStageK + 32U;
STRIX_TURBO_I8_HD unsigned m_blocks(unsigned count) { return (count + 15U) >> 4U; }
// project warp w owns outputs 8 w .. 8 w + 7 of both matrices; down warp w owns 16 w .. in two n8 blocks.
STRIX_TURBO_I8_HD unsigned project_output(unsigned block_y, unsigned warp, unsigned col) { return block_y * kProjectOutputs + 8U * warp + col; }
STRIX_TURBO_I8_HD unsigned down_output(unsigned block_y, unsigned warp, unsigned n, unsigned col) {
    return block_y * kDownOutputs + 16U * warp + 8U * n + col;
}

}  // namespace strix::cuda::core::turbo_i8
