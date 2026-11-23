// Bit-exact emulation of the gfx1151 (ROCm OCML / hardware) single-precision functions used by the Qwen prefill.
//
// The hardware approximations v_exp_f32, v_rsq_f32 and v_rcp_f32 are faithful (measured exhaustively: never more than one
// ulp from the exact value), so wherever one is not the correctly rounded result it is the other neighbour of the exact
// value. One bit per input selects it. The bits come from an exhaustive gfx1151 sweep (qwen-rtx-prefill probe p2):
//   - v_exp_f32 differs from the rounding of the fp64 reference only for |x| in [2^-25, 1): bitmap over those inputs;
//   - v_rsq_f32(x * 4^k) == v_rsq_f32(x) * 2^-k and v_rcp_f32(x * 2^k) == v_rcp_f32(x) * 2^-k for every normal input with
//     a normal result (0 violations in ~2.1e9 inputs each): bitmaps over [1, 4) and [1, 2) only;
//   - "hard" inputs, whose fp64 reference lies within 2^-48 (relative) of a float midpoint, are listed with their hardware
//     result (rsq/rcp canonicalised to [1, 4) / [1, 2) by the invariance).
// The reference is the same expression on both sides: (float) of the fp64 function (exp2, 1/sqrt, 1/x).
// The OCML sequences around the hardware instructions (expf, rsqrtf) are transliterated from the gfx1151 ISA of the engine
// build flags (-fno-fast-math, correctly rounded division/sqrt, IEEE denormals): every operation is one IEEE operation with
// the same rounding, written with explicit __fmul_rn/__fadd_rn/__fmaf_rn so that nvcc cannot contract or reorder.
// Domains: amd_v_exp: |x| < 1 (all expf reductions); amd_v_rsq: positive normal x; amd_v_rcp: normal x with |x| < 2^126.
// NaN payloads are not reproduced (any NaN in the prefill is a failure anyway).
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_GFX1151_HD __host__ __device__ __forceinline__
#else
#define STRIX_GFX1151_HD inline
#endif

// The original qualified name of Tables is retained for incumbent CUDA
// kernel symbols. Backend-neutral callers use strix::numeric::gfx1151 below.
namespace strix::cuda::amdmath {

struct Tables {
    const std::uint32_t* exp_bits;   // [2 signs][25 * 2^23 bits]: |x| patterns [102<<23, 127<<23), positive then negative
    const std::uint32_t* rsq_bits;   // [2^24 >> 5] over patterns [127<<23, 129<<23)
    const std::uint32_t* rcp_bits;   // [2^23 >> 5] over patterns [127<<23, 128<<23)
    const std::uint32_t* exp_hard_in; const float* exp_hard_out; int exp_hard_n;   // sorted by input
    const std::uint32_t* rsq_hard_in; const float* rsq_hard_out; int rsq_hard_n;   // canonical [1, 4), sorted
    const std::uint32_t* rcp_hard_in; const float* rcp_hard_out; int rcp_hard_n;   // canonical [1, 2) positive, sorted
    const void* exp2_tab;             // optional CUDA float2 pairs for 2^(n/64), n = -64..64
};

namespace detail {

#if defined(__CUDA_ARCH__) && !defined(__HIPCC__)
STRIX_GFX1151_HD std::uint32_t bits(float x) { return __float_as_uint(x); }
STRIX_GFX1151_HD float from_bits(std::uint32_t x) { return __uint_as_float(x); }
template <class T> STRIX_GFX1151_HD T read(const T* p) { return __ldg(p); }
STRIX_GFX1151_HD float mul(float a, float b) { return __fmul_rn(a, b); }
STRIX_GFX1151_HD float add(float a, float b) { return __fadd_rn(a, b); }
STRIX_GFX1151_HD float sub(float a, float b) { return __fsub_rn(a, b); }
STRIX_GFX1151_HD float fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
#else
STRIX_GFX1151_HD std::uint32_t bits(float x) { return __builtin_bit_cast(std::uint32_t, x); }
STRIX_GFX1151_HD float from_bits(std::uint32_t x) { return __builtin_bit_cast(float, x); }
template <class T> STRIX_GFX1151_HD T read(const T* p) { return *p; }
STRIX_GFX1151_HD float mul(float a, float b) {
    // Materialize the single-rounded product before any following add or subtract.
    volatile float product = a * b;
    return product;
}
STRIX_GFX1151_HD float add(float a, float b) { return a + b; }
STRIX_GFX1151_HD float sub(float a, float b) { return a - b; }
STRIX_GFX1151_HD float fma(float a, float b, float c) { return fmaf(a, b, c); }
#endif

}  // namespace detail

STRIX_GFX1151_HD bool table_bit(const std::uint32_t* t, std::uint32_t index) {
    return (detail::read(t + (index >> 5)) >> (index & 31U)) & 1U;
}

STRIX_GFX1151_HD int find_hard(const std::uint32_t* in, int n, std::uint32_t key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const std::uint32_t v = detail::read(in + mid);
        if (v == key) return mid;
        if (v < key) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

// The neighbour of f on the side of the exact value r (r != f).
STRIX_GFX1151_HD float other_neighbour(float f, double r) {
    const std::uint32_t b = detail::bits(f);
    const bool up = r > static_cast<double>(f);
    // magnitude grows when moving away from zero
    const bool away = (b >> 31) ? !up : up;
    return detail::from_bits(away ? b + 1U : b - 1U);
}

// v_exp_f32 for |x| < 1. The correctly rounded value comes from a float double-float evaluation (error < 2^-37
// relative: 2^(n/64) table, degree-5 series on |r| <= 2^-7); only when the exact value lies within 2^-35 of a rounding
// boundary (about 1 input in 2^10) is the fp64 reference used, so the result is the same as the pure fp64 form.
STRIX_GFX1151_HD float amd_v_exp_slow(const Tables& t, float x) {
    const std::uint32_t b = detail::bits(x), mag = b & 0x7fffffffU;
    const double r = exp2(static_cast<double>(x));
    const float f = static_cast<float>(r);
    if (mag < (102U << 23)) return f;  // |x| < 2^-25: the hardware equals the rounding (probe p2)
    if (t.exp_hard_n) {
        const int h = find_hard(t.exp_hard_in, t.exp_hard_n, b);
        if (h >= 0) return detail::read(t.exp_hard_out + h);
    }
    const std::uint32_t index = (b >> 31) * (25U << 23) + (mag - (102U << 23));  // positive binades, then negative
    return table_bit(t.exp_bits, index) ? other_neighbour(f, r) : f;
}

STRIX_GFX1151_HD float amd_v_exp(const Tables& t, float x) {
#if defined(__CUDA_ARCH__) && !defined(__HIPCC__)
    if (!t.exp2_tab) return amd_v_exp_slow(t, x);
    const std::uint32_t b = __float_as_uint(x), mag = b & 0x7fffffffU;
    const float n = rintf(__fmul_rn(x, 64.0F));
    const float r = __fsub_rn(x, __fmul_rn(n, 0.015625F));  // exact
    const float2 tab = __ldg(static_cast<const float2*>(t.exp2_tab) + static_cast<int>(n) + 64);
    const float lh = __uint_as_float(0x3f317218U), ll = __uint_as_float(0xb102e308U);  // ln 2 = lh + ll
    const float qh = __fmul_rn(r, lh);
    float ql = __fmaf_rn(r, lh, -qh);
    ql = __fmaf_rn(r, ll, ql);
    float p = __fmaf_rn(qh, 1.0F / 120.0F, 1.0F / 24.0F);
    p = __fmaf_rn(qh, p, 1.0F / 6.0F);
    p = __fmaf_rn(qh, p, 0.5F);
    const float tail = __fmul_rn(__fmul_rn(qh, qh), p);
    const float sh = __fadd_rn(1.0F, qh);
    float sl = __fsub_rn(qh, __fsub_rn(sh, 1.0F));
    sl = __fadd_rn(sl, __fadd_rn(ql, tail));
    const float s1 = __fadd_rn(sh, sl);
    const float s2 = __fsub_rn(sl, __fsub_rn(s1, sh));
    const float ph = __fmul_rn(tab.x, s1);
    float pl = __fmaf_rn(tab.x, s1, -ph);
    pl = __fmaf_rn(tab.x, s2, pl);
    pl = __fmaf_rn(tab.y, s1, pl);
    const float f = __fadd_rn(ph, pl);
    const float e = __fsub_rn(pl, __fsub_rn(f, ph));  // (ph + pl) - f, |e| <= half an ulp of f
    // half an ulp of f on the side of e (below a power of two the ulp halves)
    const std::uint32_t fb = __float_as_uint(f);
    const float half_up = __uint_as_float(((fb >> 23) - 24U) << 23);
    const float half = (e < 0.0F && (fb & 0x007fffffU) == 0U) ? __fmul_rn(half_up, 0.5F) : half_up;
    const float bound = __uint_as_float(((fb >> 23) - 35U) << 23);  // 2^-35 relative
    if (__fsub_rn(half, fabsf(e)) <= bound) return amd_v_exp_slow(t, x);  // near a rounding boundary
    if (mag < (102U << 23)) return f;
    const std::uint32_t index = (b >> 31) * (25U << 23) + (mag - (102U << 23));
    if (!table_bit(t.exp_bits, index)) return f;
    if (fabsf(e) <= bound) return amd_v_exp_slow(t, x);  // the side of the exact value is not certain
    return __uint_as_float(((e > 0.0F) == ((fb >> 31) == 0U)) ? fb + 1U : fb - 1U);
#else
    return amd_v_exp_slow(t, x);
#endif
}

// v_rsq_f32 for a positive normal input.
STRIX_GFX1151_HD float amd_v_rsq(const Tables& t, float x) {
    const std::uint32_t b = detail::bits(x);
    const int e = static_cast<int>(b >> 23) - 127;
    const int k = e >= 0 ? e / 2 : -((-e + 1) / 2);  // floor(e / 2)
    const std::uint32_t cb = (b & 0x007fffffU) | (static_cast<std::uint32_t>(e - 2 * k + 127) << 23);  // [1, 4)
    const float c = detail::from_bits(cb);
    float y;
    const int h = t.rsq_hard_n ? find_hard(t.rsq_hard_in, t.rsq_hard_n, cb) : -1;
    if (h >= 0) y = detail::read(t.rsq_hard_out + h);
    else {
#if defined(__CUDA_ARCH__) && !defined(__HIPCC__)
        // Outside the hard list the fp64 reference rounds to the correctly rounded value, which __frsqrt_rn gives.
        const float f = __frsqrt_rn(c);
        y = f;
        if (table_bit(t.rsq_bits, cb - (127U << 23))) {
            // side of the exact value: sign(1 - c*f^2), with f^2 and c*f^2 split exactly
            const float sh = __fmul_rn(f, f), sl = __fmaf_rn(f, f, -sh);
            const float p = __fmul_rn(c, sh), q = __fmaf_rn(c, sh, -p);
            const float u = __fmul_rn(c, sl), v = __fmaf_rn(c, sl, -u);
            const float d = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(1.0F, p), q), u), v);
            if (fabsf(d) < 0x1p-44F) {
                const double r = 1.0 / sqrt(static_cast<double>(c));
                y = other_neighbour(static_cast<float>(r), r);
            } else {
                y = __uint_as_float(d > 0.0F ? __float_as_uint(f) + 1U : __float_as_uint(f) - 1U);
            }
        }
#else
        const double r = 1.0 / sqrt(static_cast<double>(c));
        const float f = static_cast<float>(r);
        y = table_bit(t.rsq_bits, cb - (127U << 23)) ? other_neighbour(f, r) : f;
#endif
    }
    // exact power-of-two scaling (the result stays normal for normal inputs)
    return detail::from_bits(static_cast<std::uint32_t>(static_cast<int>(detail::bits(y)) - (k << 23)));
}

// v_rcp_f32 for a normal input with |x| < 2^126.
STRIX_GFX1151_HD float amd_v_rcp(const Tables& t, float x) {
    const std::uint32_t b = detail::bits(x);
    const int e = static_cast<int>((b >> 23) & 255U) - 127;
    const std::uint32_t cb = (b & 0x007fffffU) | (127U << 23);  // positive, [1, 2)
    const float c = detail::from_bits(cb);
    float y;
    const int h = t.rcp_hard_n ? find_hard(t.rcp_hard_in, t.rcp_hard_n, cb) : -1;
    if (h >= 0) y = detail::read(t.rcp_hard_out + h);
    else {
#if defined(__CUDA_ARCH__) && !defined(__HIPCC__)
        // Outside the hard list the fp64 reference rounds to the correctly rounded value, which __frcp_rn gives;
        // the side of the exact value is the sign of the exact residual 1 - c*f (one fma).
        const float f = __frcp_rn(c);
        y = f;
        if (table_bit(t.rcp_bits, cb - (127U << 23))) {
            const float d = __fmaf_rn(-c, f, 1.0F);
            y = __uint_as_float(d > 0.0F ? __float_as_uint(f) + 1U : __float_as_uint(f) - 1U);
        }
#else
        const double r = 1.0 / static_cast<double>(c);
        const float f = static_cast<float>(r);
        y = table_bit(t.rcp_bits, cb - (127U << 23)) ? other_neighbour(f, r) : f;
#endif
    }
    const std::uint32_t yb = static_cast<std::uint32_t>(static_cast<int>(detail::bits(y)) - (e << 23));
    return detail::from_bits(yb | (b & 0x80000000U));
}

// v_ldexp_f32 (exact scaling, IEEE rounding of a subnormal result, overflow to inf).
STRIX_GFX1151_HD float amd_ldexp(float v, int n) { return scalbnf(v, n); }

// expf: OCML / LLVM AMDGPU lowering at -fno-fast-math (ISA of eval_f, probe p1):
//   ph = x * log2e; e = rndne(ph); pl = fma(x, log2e, -ph); pl = fma(x, log2e_lo, pl); a = (ph - e) + pl;
//   y = ldexp(v_exp(a), (int)e); x < -103.97208 -> 0; x > 88.72284 -> +inf.
STRIX_GFX1151_HD float amd_expf(const Tables& t, float x) {
    const float kLog2e = detail::from_bits(0x3fb8aa3bU), kLog2eLo = detail::from_bits(0x32a5705fU);
    const float ph = detail::mul(x, kLog2e);
    const float e = rintf(ph);
    float pl = detail::fma(x, kLog2e, -ph);
    pl = detail::fma(x, kLog2eLo, pl);
    const float a = detail::add(detail::sub(ph, e), pl);
    float y;
    if (fabsf(a) < 1.0F) y = amd_ldexp(amd_v_exp(t, a), static_cast<int>(e));
    else y = a != a ? a : 0.0F;  // only reached for infinite/NaN x (then the selects below decide) [checked by the tests]
    // v_cmp_ngt_f32 vcc, -103.97208, x : vcc = !(K > x)  -> keep y, else 0
    if (detail::from_bits(0xc2ce8ed0U) > x) y = 0.0F;
    // v_cmp_nlt_f32 vcc, 88.72284, x : vcc = !(K < x)  -> keep y, else +inf
    if (detail::from_bits(0x42b17218U) < x) y = detail::from_bits(0x7f800000U);
    return y;
}

// rsqrtf: subnormal inputs scaled by 2^24 before v_rsq_f32 and the result by 2^12.
STRIX_GFX1151_HD float amd_rsqrtf(const Tables& t, float x) {
    const bool small = detail::from_bits(0x00800000U) > x;  // v_cmp_gt_f32 vcc, 0x800000, x
    const float s = small ? detail::mul(x, detail::from_bits(0x4b800000U)) : x;
    float y;
    const std::uint32_t sb = detail::bits(s);
    if (sb == 0U) y = detail::from_bits(0x7f800000U);
    else if (sb == 0x80000000U) y = detail::from_bits(0xff800000U);
    else if (sb >= 0x7f800000U) y = sb == 0x7f800000U ? 0.0F : detail::from_bits(0x7fc00000U);
    else y = amd_v_rsq(t, s);
    return small ? detail::mul(y, detail::from_bits(0x45800000U)) : y;
}

}  // namespace strix::cuda::amdmath

namespace strix::numeric::gfx1151 {
using cuda::amdmath::Tables;
using cuda::amdmath::amd_v_exp;
using cuda::amdmath::amd_v_rsq;
using cuda::amdmath::amd_v_rcp;
using cuda::amdmath::amd_expf;
using cuda::amdmath::amd_rsqrtf;
}  // namespace strix::numeric::gfx1151

#undef STRIX_GFX1151_HD
