#pragma once
// Hi/lo operand split of the FP16 tall GEMM (f16_dense_tall::core with a ColsHiLo policy): arithmetic shared by the
// device core and its host checks. No SDK dependency.
//
// Operand: hi = fp16(x) (RNE), lo = fp16((x - float(hi)) * scale) (RNE), scale a power of two chosen by the adapter.
// Product: per output two WMMA chains over the same weight fragments, h on hi and l on lo, each from +0 over the k
//   units in ascending order. Result, every operation a separately rounded FP32 operation (never an FMA):
//   unblocked (block_units == 0): y = h + (l * unscale)
//   blocked (block_units = B k units): both chains restart at +0 after every B units and after the last unit;
//     at each such point part = h + (l * unscale) and s = s + part, with s starting at +0; y = s.
#if defined(__HIPCC__)
#define STRIX_HILO_HD __host__ __device__
#else
#define STRIX_HILO_HD
#endif

namespace strix::core::f16_hilo {

// part = h + (l * unscale). The scaled product is materialized before the add so the compiler cannot contract the
// two operations into one FMA (the result would round once instead of twice).
STRIX_HILO_HD inline float merge(float h, float l, float unscale) {
#if defined(__HIP_DEVICE_COMPILE__)
    float scaled = l * unscale;
    asm volatile("" : "+v"(scaled));
    return h + scaled;
#else
    volatile float scaled = l * unscale;
    return h + scaled;
#endif
}

// The low operand before its FP16 rounding, given the high operand already widened back to FP32.
STRIX_HILO_HD inline float low_operand(float x, float hi, float scale) { return (x - hi) * scale; }

// Whether the chains merge (and, when blocked, restart) after k unit `unit` (0-based) of `units`.
constexpr bool merge_after(unsigned unit, unsigned units, unsigned block_units) {
    return unit + 1U >= units || (block_units != 0U && (unit + 1U) % block_units == 0U);
}

}  // namespace strix::core::f16_hilo

#undef STRIX_HILO_HD
