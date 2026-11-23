#pragma once

#include "strix/numeric/math_gfx1151.hpp"

namespace strix::cuda::amdmath {

// amd_expf for N independent inputs: y[i] has the bits of amd_expf(t, x[i]) (the same IEEE operations; amd_v_exp's branches
// only choose among values both forms compute). The difference is the memory schedule: amd_v_exp consumes its exp2_tab
// gather and its exp_bits gather (a random word of a ~52 MB bit table, an L2 round trip) through branches before the next
// call starts, so a loop of N calls pays 2N dependent round trips; here all N exp2_tab gathers, then all N exp_bits
// gathers, are issued before the first is consumed (2 round trips per batch). Out-of-domain lanes (|a| >= 1, NaN, tiny)
// gather a fixed in-bounds word and ignore it. The rare slow cases call amd_v_exp_slow exactly as amd_v_exp does.
template <unsigned N>
__device__ __forceinline__ void amd_expf_batch(const Tables& t, const float (&x)[N], float (&y)[N]) {
    if (!t.exp2_tab) {
#pragma unroll
        for (unsigned i = 0; i < N; ++i) y[i] = amd_expf(t, x[i]);
        return;
    }
    const float kLog2e = __uint_as_float(0x3fb8aa3bU), kLog2eLo = __uint_as_float(0x32a5705fU);
    float a[N], ex[N], n[N];
    float2 tab[N];
#pragma unroll
    for (unsigned i = 0; i < N; ++i) {  // amd_expf's reduction, then amd_v_exp's table index; gathers 1 (exp2_tab)
        const float ph = __fmul_rn(x[i], kLog2e);
        ex[i] = rintf(ph);
        float pl = __fmaf_rn(x[i], kLog2e, -ph);
        pl = __fmaf_rn(x[i], kLog2eLo, pl);
        a[i] = __fadd_rn(__fsub_rn(ph, ex[i]), pl);
        n[i] = fabsf(a[i]) < 1.0F ? rintf(__fmul_rn(a[i], 64.0F)) : 0.0F;
        tab[i] = __ldg(static_cast<const float2*>(t.exp2_tab) + static_cast<int>(n[i]) + 64);
    }
    float f[N], err[N], bound[N];
    bool edge[N], tiny[N];
    std::uint32_t word[N], index[N];
#pragma unroll
    for (unsigned i = 0; i < N; ++i) {  // amd_v_exp's fast-path arithmetic (same operations, same order); gathers 2 (exp_bits)
        const std::uint32_t b = __float_as_uint(a[i]), mag = b & 0x7fffffffU;
        const float r = __fsub_rn(a[i], __fmul_rn(n[i], 0.015625F));
        const float lh = __uint_as_float(0x3f317218U), ll = __uint_as_float(0xb102e308U);
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
        const float ph = __fmul_rn(tab[i].x, s1);
        float pl = __fmaf_rn(tab[i].x, s1, -ph);
        pl = __fmaf_rn(tab[i].x, s2, pl);
        pl = __fmaf_rn(tab[i].y, s1, pl);
        f[i] = __fadd_rn(ph, pl);
        err[i] = __fsub_rn(pl, __fsub_rn(f[i], ph));
        const std::uint32_t fb = __float_as_uint(f[i]);
        const float half_up = __uint_as_float(((fb >> 23) - 24U) << 23);
        const float half = (err[i] < 0.0F && (fb & 0x007fffffU) == 0U) ? __fmul_rn(half_up, 0.5F) : half_up;
        bound[i] = __uint_as_float(((fb >> 23) - 35U) << 23);
        edge[i] = __fsub_rn(half, fabsf(err[i])) <= bound[i];
        tiny[i] = mag < (102U << 23);
        index[i] = (tiny[i] || !(fabsf(a[i]) < 1.0F)) ? 0U : (b >> 31) * (25U << 23) + (mag - (102U << 23));
        word[i] = __ldg(t.exp_bits + (index[i] >> 5));
    }
#pragma unroll
    for (unsigned i = 0; i < N; ++i) {  // amd_v_exp's decisions (same order), then amd_expf's scaling and range selects
        float v;
        if (fabsf(a[i]) < 1.0F) {
            const bool bit = (word[i] >> (index[i] & 31U)) & 1U;
            v = f[i];
            if (edge[i] || (!tiny[i] && bit && fabsf(err[i]) <= bound[i])) v = amd_v_exp_slow(t, a[i]);
            else if (!tiny[i] && bit) {
                const std::uint32_t fb = __float_as_uint(f[i]);
                v = __uint_as_float(((err[i] > 0.0F) == ((fb >> 31) == 0U)) ? fb + 1U : fb - 1U);
            }
            v = amd_ldexp(v, static_cast<int>(ex[i]));
        } else v = a[i] != a[i] ? a[i] : 0.0F;
        if (__uint_as_float(0xc2ce8ed0U) > x[i]) v = 0.0F;
        if (__uint_as_float(0x42b17218U) < x[i]) v = __uint_as_float(0x7f800000U);
        y[i] = v;
    }
}

// log1pf: the OCML double-float algorithm as compiled for gfx1151 (ISA of eval_f, probe p1). Every line is one IEEE
// instruction; VOPD pairs (v_dual_*) read both sources before either writes, kept with temporaries. The single hardware
// approximation is v_rcp_f32 of a value in [1.66, 2.34].
__device__ __forceinline__ float amd_log1pf(const Tables& t, float x) {
    auto add = [](float a, float b) { return __fadd_rn(a, b); };
    auto sub = [](float a, float b) { return __fsub_rn(a, b); };
    auto mul = [](float a, float b) { return __fmul_rn(a, b); };
    auto fma = [](float a, float b, float c) { return __fmaf_rn(a, b, c); };
    const float v1 = x;
    float v0 = add(1.0F, v1);
    int e;
    (void)frexp(static_cast<double>(v0), &e);                 // v_frexp_exp_i32_f64 of cvt_f64(v0)
    int ef;
    float v3 = frexpf(v0, &ef);                               // v_frexp_mant_f32
    if (!(v0 == v0) || v0 == 0.0F || fabsf(v0) == __uint_as_float(0x7f800000U)) { e = 0; v3 = v0; }
    const bool lt = __uint_as_float(0x3f2aaaabU) > v3;
    v3 = add(-1.0F, v0);
    float v5 = sub(v3, v0);
    v3 = sub(v1, v3);
    int v2i = e - (lt ? 1 : 0);
    const int v4i = -v2i;
    const float v2 = static_cast<float>(v2i);
    v0 = scalbnf(v0, v4i);
    v5 = add(1.0F, v5);
    float v6 = add(1.0F, v0);
    v3 = add(v3, v5);
    v3 = scalbnf(v3, v4i);
    float v4 = add(-1.0F, v0);
    v5 = add(-1.0F, v6);
    float v7 = add(1.0F, v4);
    v5 = sub(v0, v5);
    v0 = sub(v0, v7);
    v0 = add(v3, v0);
    v5 = add(v3, v5);
    float v8 = add(v4, v0);
    v7 = add(v6, v5);
    v4 = sub(v8, v4);
    v3 = amd_v_rcp(t, v7);
    v6 = sub(v7, v6);
    v5 = sub(v5, v6);
    v0 = sub(v0, v4);
    const float v9 = mul(v8, v3);
    float v10 = mul(v7, v9);
    v6 = fma(v9, v7, -v10);
    v6 = fma(v9, v5, v6);
    float v11 = add(v10, v6);
    float v12 = sub(v8, v11);
    v8 = sub(v8, v12);
    v4 = sub(v11, v10);
    v8 = sub(v8, v11);
    v4 = sub(v4, v6);
    v0 = add(v0, v8);
    v0 = add(v4, v0);
    v4 = add(v12, v0);
    v6 = mul(v3, v4);
    v11 = sub(v12, v4);
    v8 = mul(v7, v6);
    v0 = add(v0, v11);
    v7 = fma(v6, v7, -v8);
    v7 = fma(v6, v5, v7);
    v5 = add(v8, v7);
    v10 = sub(v4, v5);
    v4 = sub(v4, v10);
    v4 = sub(v4, v5);
    v0 = add(v0, v4);
    v4 = add(v9, v6);
    v8 = sub(v5, v8);
    v5 = sub(v8, v7);
    { const float t0 = add(v5, v0); v5 = sub(v4, v9); v0 = t0; }
    v0 = add(v10, v0);
    v5 = sub(v6, v5);
    v0 = mul(v3, v0);
    v0 = add(v5, v0);
    v3 = add(v4, v0);
    v5 = mul(v3, v3);
    v6 = fma(__uint_as_float(0x3e9b6dacU), v5, __uint_as_float(0x3ecc95a3U));
    v7 = mul(v3, v5);
    v5 = fma(v5, v6, __uint_as_float(0x3f2aaadaU));
    v6 = scalbnf(v3, 1);
    v3 = sub(v3, v4);
    v5 = mul(v7, v5);
    v7 = mul(__uint_as_float(0x3f317218U), v2);
    v0 = sub(v0, v3);
    v4 = add(v6, v5);
    v0 = scalbnf(v0, 1);
    v3 = sub(v4, v6);
    v6 = fma(__uint_as_float(0x3f317218U), v2, -v7);
    v3 = sub(v5, v3);
    v0 = add(v0, v3);
    float w2 = fma(v2, __uint_as_float(0xb102e308U), v6);
    v5 = add(v4, v0);
    v3 = add(v7, w2);
    v4 = sub(v5, v4);
    v7 = sub(v3, v7);
    v0 = sub(v0, v4);
    w2 = sub(w2, v7);
    v6 = add(v3, v5);
    v8 = sub(v6, v3);
    v4 = sub(v5, v8);
    v5 = add(w2, v0);
    const float w9 = sub(v6, v8);
    v3 = sub(v3, w9);
    { const float t0 = add(v4, v3); v4 = sub(v5, w2); v3 = t0; }
    v3 = add(v5, v3);
    v5 = sub(v5, v4);
    v0 = sub(v0, v4);
    v7 = add(v6, v3);
    w2 = sub(w2, v5);
    v4 = sub(v7, v6);
    v0 = add(v0, w2);
    w2 = sub(v3, v4);
    v0 = add(v0, w2);
    v0 = add(v7, v0);
    if (!(v1 != __uint_as_float(0x7f800000U))) v0 = __uint_as_float(0x7f800000U);
    if (-1.0F > v1) v0 = __uint_as_float(0x7fc00000U);
    if (!(v1 != -1.0F)) v0 = __uint_as_float(0xff800000U);
    if (__uint_as_float(0x33800000U) > fabsf(v1)) v0 = v1;
    return v0;
}

// Correctly rounded IEEE operations (the AMD division and sqrtf are correctly rounded at the engine's flags).
__device__ __forceinline__ float amd_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float amd_sqrtf(float x) { return __fsqrt_rn(x); }

}  // namespace strix::cuda::amdmath
