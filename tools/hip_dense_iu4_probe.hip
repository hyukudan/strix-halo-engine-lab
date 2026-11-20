// Probe (measurement only, not linked into any engine): where the shared FP16 dense tall core's time goes for the
// dense-iu4a16-f16-v1 policy (src/core/f16_dense_iu4.hpp) on one prefill window shape, and the gfx1151 WMMA rates.
// Each arm runs the production core<16> with ONE operand path removed or replaced (values change, instruction stream
// otherwise the same), so the time difference to the production arm is that path's exposed cost:
//   prod16 / prod8   production enqueue, 16-wave CTAs (256 rows) / 8-wave CTAs (128 rows, two CTAs per WGP)
//   alias            weight loads read the k units 0..3 of their column (L2-resident; removes the DRAM weight stream)
//   nodec            stagers copy the raw IU4 words to LDS (removes the IU4 -> FP16 decode VALU)
//   alias+nodec      both
//   x16              A fragments from the shared x16 tiles (FP16, column-fast; X16Cols), GEMM only; x16cvt = the tile pass
//   x16+alias+nodec  x16 with both
//   noA / noA+alias+nodec   A fragment computed in registers (no A loads, no cvt): the WMMA + LDS + barrier floor
// Attention prep (attn_core/dense_decode.hpp, one 512-row window at pos0 7680, 24 q + 4 kv heads of 256, rot_dim 64):
//   prep_inline (incumbent), prep_rot0 (no RoPE: the trig share), prep_table (rope_table_kernel + prep<Table>; outputs
//   compared bit for bit with prep_inline).
// x16prod+cvt: the production x16 entry (enqueue_x16_tiles + iu4a16::enqueue_x16).
// Rates: 4 independent accumulator chains per wave, 16 waves per WGP, f16 / iu8 / iu4 16x16x16 (ops = 2 * 16^3 each).
// Usage: hip-dense-iu4-probe [rows=512]   prints one line per (shape, arm): us/call and TFLOPS at the timed clock.
//        hip-dense-iu4-probe 512 ldsfloor [micro|core] [rand]   operand-path ceilings of the tall core (see ldsfloor).
// Data class matters on gfx1151: WMMA power depends on the operand bits, and with full-mantissa FP16 data the clock
// settles ~15-20 % below its constant-data level (sampled 2.85 -> 2.3-2.4 GHz). Compare rates on the same data class.
#include "core/f16_dense_iu4.hpp"
#include "attn_core/dense_decode.hpp"
#include "core/dense_linear_iu4.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace iu = strix::core::f16_dense_tall::iu4a16;
namespace tall = strix::core::f16_dense_tall;

#define CK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

template <unsigned W, bool Alias, bool NoDec> struct PCols : iu::Cols<W> {
    using B = iu::Cols<W>;
    using Raw = typename B::Raw;
    __device__ __forceinline__ Raw stage_load(unsigned t, unsigned c, unsigned u) const { return B::stage_load(t, c, Alias ? (u & 3U) : u); }
    __device__ __forceinline__ void stage_decode(const Raw& raw, tall::U4* nb) const {
        if constexpr (NoDec) {
            Raw r = raw;
#pragma unroll
            for (unsigned q = 0; q < 8U; ++q) asm volatile("" : "+v"(r.w[q]));
#pragma unroll
            for (unsigned q = 0; q < 8U; ++q) nb[q] = tall::U4{r.w[q], r.w[q] ^ r.s, r.w[q], r.w[q]};
        } else {
            B::stage_decode(raw, nb);
        }
    }
};
template <class Base> struct NoACols : Base {
    __device__ __forceinline__ tall::Half16 fragment(unsigned row, unsigned unit) const {
        const _Float16 h = static_cast<_Float16>(static_cast<float>((row ^ unit) & 7U) * 0.125F);
        tall::Half16 v;
#pragma unroll
        for (unsigned i = 0; i < 16U; ++i) v[i] = h;
        return v;
    }
};

template <class C> __global__ __launch_bounds__(512) void pk(C c, const float* x, unsigned in, unsigned rows, iu::Store st) {
    __shared__ tall::Storage<16> lds;
    tall::core<16>(x, in, rows, c, lds, st);
}

template <int Kind> __global__ __launch_bounds__(256) void rate(int iters, float* out) {
    const unsigned t = threadIdx.x;
    if constexpr (Kind == 0) {
        tall::Half16 a, b;
        for (unsigned i = 0; i < 16U; ++i) a[i] = static_cast<_Float16>(float((t + i) & 3U)), b[i] = static_cast<_Float16>(float((t ^ i) & 3U));
        tall::Float8 c0 = {}, c1 = {}, c2 = {}, c3 = {};
        for (int i = 0; i < iters; ++i) {
            c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c0);
            c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c1);
            c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c2);
            c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c3);
        }
        const float s = c0[0] + c1[1] + c2[2] + c3[3];
        if (s == 1234.5F) out[t] = s;
    } else {
        using I8 = int __attribute__((ext_vector_type(8)));
        using I4 = int __attribute__((ext_vector_type(4)));
        using I2 = int __attribute__((ext_vector_type(2)));
        I8 c0 = {}, c1 = {}, c2 = {}, c3 = {};
        if constexpr (Kind == 1) {
            const I4 a = {int(t), int(t * 3U), int(t * 5U), int(t * 7U)}, b = {int(t ^ 1U), int(t ^ 2U), int(t ^ 3U), int(t ^ 4U)};
            for (int i = 0; i < iters; ++i) {
                c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c0, false);
                c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c1, false);
                c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c2, false);
                c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c3, false);
            }
        } else {
            const I2 a = {int(t), int(t * 3U)}, b = {int(t ^ 1U), int(t ^ 2U)};
            for (int i = 0; i < iters; ++i) {
                c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c0, false);
                c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c1, false);
                c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c2, false);
                c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c3, false);
            }
        }
        const int s = c0[0] + c1[1] + c2[2] + c3[3];
        if (s == 123457) out[t] = float(s);
    }
}

// FAST4 rescale arms (hip_dense5 STATE item 4; measurement only): per 16x16 tile and K64 weight group, the iu4 WMMAs of
// a W4A4 class plus its per-element rescale, 4 tiles per wave, chains loop-carried (nothing hoistable). Kind 3: a4 g64,
// C carries the magic 0x4B400000 (float(dot) = bits - 12582912 by v_sub), s = sa[r] * sw, acc = fma(f, s, acc);
// Kind 4: a4 g32 (two K32 chains): t = f0 * sa0; t = fma(f1, sa1, t); acc = fma(t, sw, acc); Kind 5 = Kind 4 with
// v_cvt_f32_i32 instead of the magic subtraction. Ops counted as WMMA work (16 x 16 x 16 x 2 per instruction).
template <int Kind> __global__ __launch_bounds__(256) void rescale_rate(int iters, float* out) {
    using I8 = int __attribute__((ext_vector_type(8)));
    using I2 = int __attribute__((ext_vector_type(2)));
    using F8 = float __attribute__((ext_vector_type(8)));
    const unsigned t = threadIdx.x;
    const I2 a = {int(t), int(t * 3U)}, b = {int(t ^ 1U), int(t ^ 2U)};
    constexpr float kM = 12582912.0F;
    F8 acc[4], sa0, sa1;
    I8 c0[4], c1[4];
    for (unsigned e = 0; e < 8U; ++e) {
        sa0[e] = 1.0F + float((t + e) & 7U) * 0x1p-10F;
        sa1[e] = 1.0F + float((t ^ e) & 7U) * 0x1p-11F;
    }
    for (unsigned q = 0; q < 4U; ++q)
        for (unsigned e = 0; e < 8U; ++e) acc[q][e] = 0.0F, c0[q][e] = 0x4B400000, c1[q][e] = 0x4B400000;
    float sw = 1.0F + float(t & 15U) * 0x1p-12F;
    auto f32 = [&](int v) { return Kind == 5 ? float(v) : __builtin_bit_cast(float, v) - kM; };
    for (int i = 0; i < iters; ++i) {
        sw = sw * 0.99999994F;  // loop-variant weight scale
#pragma unroll
        for (unsigned q = 0; q < 4U; ++q) {
            if constexpr (Kind == 3) {
                c0[q] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c0[q], false);
                c0[q] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c0[q], false);
                c0[q] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c0[q], false);
                c0[q] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c0[q], false);
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) acc[q][e] = __builtin_fmaf(f32(c0[q][e]), sa0[e] * sw, acc[q][e]);
            } else {
                c0[q] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c0[q], false);
                c0[q] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c0[q], false);
                c1[q] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c1[q], false);
                c1[q] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c1[q], false);
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) {
                    float v = f32(c0[q][e]) * sa0[e];
                    v = __builtin_fmaf(f32(c1[q][e]), sa1[e], v);
                    acc[q][e] = __builtin_fmaf(v, sw, acc[q][e]);
                }
            }
        }
    }
    const float s = acc[0][0] + acc[1][1] + acc[2][2] + acc[3][3];
    if (s == 1234.5F) out[t] = s;
}

// sd_i4 arms (contract dense-iu4a4-rot-v1, core/iu4a4_rot.hpp; measurement only): 8 column tiles per wave (the
// register budget of the planned trait), per 128-K block 4 A groups of 32 (Kind 6) or 2 of 64 (Kind 7). C is cumulative
// (never reset); t = lead * C + sum_g e_g * C with 24-bit multiplies (Abel form: no convert per A group), then
// acc = fma(float(t), wstep, acc) once per block. e / lead (8 rows per lane) and wstep (one per tile) are loaded from a
// cache-resident table per group / block, as the production loop would. Values are not meaningful (timing only).
template <int Kind> __global__ __launch_bounds__(256) void rot_rate(int iters, const int* __restrict__ coef, const float* __restrict__ ws, float* out) {
    using I8 = int __attribute__((ext_vector_type(8)));
    using I2 = int __attribute__((ext_vector_type(2)));
    using F8 = float __attribute__((ext_vector_type(8)));
    constexpr unsigned kT = 8U, kG = Kind == 6 ? 4U : 2U, kU = 8U / kG;  // tiles, A groups per block, K16 units per group
    const unsigned t = threadIdx.x, lane = t & 31U;
    const I2 a = {int(t), int(t * 3U)}, b = {int(t ^ 1U), int(t ^ 2U)};
    I8 c[kT], s[kT];
    F8 acc[kT];
    for (unsigned q = 0; q < kT; ++q)
        for (unsigned e = 0; e < 8U; ++e) c[q][e] = 0, s[q][e] = 0, acc[q][e] = 0.0F;
    for (int i = 0; i < iters; ++i) {
        const I8* cf = reinterpret_cast<const I8*>(coef) + (unsigned(i) & 63U) * (kG + 1U) * 32U + lane;
        const I8 lead = cf[kG * 32U];
#pragma unroll
        for (unsigned q = 0; q < kT; ++q)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) s[q][e] = __mul24(lead[e], c[q][e]);
#pragma unroll
        for (unsigned g = 0; g < kG; ++g) {
            const I8 eg = cf[g * 32U];
#pragma unroll
            for (unsigned k = 0; k < kU; ++k)
#pragma unroll
                for (unsigned q = 0; q < kT; ++q) c[q] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c[q], false);
#pragma unroll
            for (unsigned q = 0; q < kT; ++q)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) s[q][e] += __mul24(eg[e], c[q][e]);
        }
        const float* w = ws + (unsigned(i) & 63U) * kT * 32U + lane;
#pragma unroll
        for (unsigned q = 0; q < kT; ++q) {
            const float sw = w[q * 32U];
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[q][e] = __builtin_fmaf(float(s[q][e]), sw, acc[q][e]);
        }
    }
    float r = 0.0F;
    for (unsigned q = 0; q < kT; ++q) r += acc[q][q];
    if (r == 1234.5F) out[t] = r;
}

template <class F> static double time_us(F&& launch, int reps = 20);
__device__ __forceinline__ float act_silu(float g) { return g / (1.0F + expf(-g)); }
struct ActSilu {
    __device__ __forceinline__ static float apply(float g) { return act_silu(g); }
};
__global__ void silu_mul(const float* g, const float* u, float* out, std::size_t n) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        out[i] = act_silu(g[i]) * u[i];
}

static void prep_arms(unsigned rows) {
    namespace dd = strix::attn_core::dense_decode;
    dd::Params p{rows, 24U, 4U, 512U, 64U, 7680U, 1e-6F, 0.0625F, 1e7, true};
    const std::size_t nq = std::size_t(rows) * 24U * 512U, nk = std::size_t(rows) * 4U * 256U, ncache = std::size_t(7680U + rows) * 4U * 256U;
    float *q, *k, *vv, *gain, *qo, *qo2, *tab;
    std::uint16_t *kc, *vc, *kc2;
    CK(hipMalloc(&q, nq * 4U)); CK(hipMalloc(&k, nk * 4U)); CK(hipMalloc(&vv, nk * 4U)); CK(hipMalloc(&gain, 1024U));
    CK(hipMalloc(&qo, std::size_t(rows) * 24U * 256U * 4U)); CK(hipMalloc(&qo2, std::size_t(rows) * 24U * 256U * 4U));
    CK(hipMalloc(&tab, std::size_t(rows) * 64U * 4U));
    CK(hipMalloc(&kc, ncache * 2U)); CK(hipMalloc(&vc, ncache * 2U)); CK(hipMalloc(&kc2, ncache * 2U));
    {
        std::mt19937 g(11);
        std::uniform_real_distribution<float> d(-3.0F, 3.0F);
        std::vector<float> h(nq);
        for (auto& e : h) e = d(g);
        CK(hipMemcpy(q, h.data(), nq * 4U, hipMemcpyHostToDevice));
        CK(hipMemcpy(k, h.data(), nk * 4U, hipMemcpyHostToDevice));
        CK(hipMemcpy(vv, h.data() + nk, nk * 4U, hipMemcpyHostToDevice));
        CK(hipMemcpy(gain, h.data() + 2U * nk, 1024U, hipMemcpyHostToDevice));
    }
    const dim3 grid(rows, 24U + 8U);
    const double inl = time_us([&] { dd::prep_kernel<256><<<grid, 256>>>(p, q, k, vv, gain, gain, qo, kc, vc); });
    dd::Params p0 = p;
    p0.rot_dim = 0U;
    const double rot0 = time_us([&] { dd::prep_kernel<256><<<grid, 256>>>(p0, q, k, vv, gain, gain, qo2, kc2, vc); });
    const double tbl = time_us([&] {
        dd::rope_table_kernel<<<(rows * 32U + 255U) / 256U, 256>>>(p, tab);
        dd::prep_kernel<256, true><<<grid, 256>>>(p, q, k, vv, gain, gain, qo2, kc2, vc, tab);
    });
    CK(hipDeviceSynchronize());
    std::vector<float> a(std::size_t(rows) * 24U * 256U), b(a.size());
    std::vector<std::uint16_t> ka(ncache), kb(ncache);
    CK(hipMemcpy(a.data(), qo, a.size() * 4U, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), qo2, b.size() * 4U, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ka.data(), kc, ncache * 2U, hipMemcpyDeviceToHost)); CK(hipMemcpy(kb.data(), kc2, ncache * 2U, hipMemcpyDeviceToHost));
    const bool same = std::memcmp(a.data(), b.data(), a.size() * 4U) == 0 && std::memcmp(ka.data(), kb.data(), ncache * 2U) == 0;
    std::printf("prep rows %u: inline %.1f us, rot0 %.1f us, table %.1f us; table vs inline q/k %s\n", rows, inl, rot0, tbl,
                same ? "bitwise" : "DIFFER");
    CK(hipFree(q)); CK(hipFree(k)); CK(hipFree(vv)); CK(hipFree(gain)); CK(hipFree(qo)); CK(hipFree(qo2)); CK(hipFree(tab));
    CK(hipFree(kc)); CK(hipFree(vc)); CK(hipFree(kc2));
}

struct Bufs {
    std::uint8_t* w;
    std::uint16_t* s;
    float *x, *y, *g, *u;
    _Float16* xt;
};

template <class F> static double time_us(F&& launch, int reps) {
    for (int i = 0; i < 3; ++i) launch();
    CK(hipDeviceSynchronize());
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    CK(hipEventRecord(a, nullptr));
    for (int i = 0; i < reps; ++i) launch();
    CK(hipEventRecord(b, nullptr));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    CK(hipGetLastError());
    return 1000.0 * ms / reps;
}

template <class C> static double run_arm(const C& c, const iu::View& v) {
    const unsigned rb = (v.rows + 255U) / 256U, cb = v.count / 256U;
    const dim3 g = tall::kColumnFast<C> ? dim3(cb, rb) : dim3(rb, cb);
    return time_us([&] { hipLaunchKernelGGL((pk<C>), g, dim3(512), 0, nullptr, c, v.x, v.in, v.rows, iu::Store{v.count, 0U, 0U}); });
}

// WMMA rate loops and the FAST4 rescale arms (argv[2] = "rates": only these).
static int rates() {
    float* o = nullptr;
    CK(hipMalloc(&o, 4096));
    const int iters = 4096;
    const double ops = 640.0 * 256.0 / 32.0 * 4.0 * iters * 8192.0;  // blocks * waves * chains * iters * ops per WMMA
    std::printf("rate f16 %.1f TOPS\n", ops / time_us([&] { hipLaunchKernelGGL(rate<0>, dim3(640), dim3(256), 0, nullptr, iters, o); }, 5) / 1e6);
    std::printf("rate iu8 %.1f TOPS\n", ops / time_us([&] { hipLaunchKernelGGL(rate<1>, dim3(640), dim3(256), 0, nullptr, iters, o); }, 5) / 1e6);
    std::printf("rate iu4 %.1f TOPS\n", ops / time_us([&] { hipLaunchKernelGGL(rate<2>, dim3(640), dim3(256), 0, nullptr, iters, o); }, 5) / 1e6);
    const double ops16 = ops * 4.0;  // 4 tiles x 4 WMMAs per iteration (the rate loops: 4 x 1)
    std::printf("rate a4g64-magic %.1f TOPS\n", ops16 / time_us([&] { hipLaunchKernelGGL(rescale_rate<3>, dim3(640), dim3(256), 0, nullptr, iters, o); }, 5) / 1e6);
    std::printf("rate a4g32-magic %.1f TOPS\n", ops16 / time_us([&] { hipLaunchKernelGGL(rescale_rate<4>, dim3(640), dim3(256), 0, nullptr, iters, o); }, 5) / 1e6);
    std::printf("rate a4g32-cvt %.1f TOPS\n", ops16 / time_us([&] { hipLaunchKernelGGL(rescale_rate<5>, dim3(640), dim3(256), 0, nullptr, iters, o); }, 5) / 1e6);
    // sd_i4: 64 WMMAs per iteration at iters / 4 = the same work as ops16.
    int* coef = nullptr;
    float* wsb = nullptr;
    CK(hipMalloc(&coef, 64U * 5U * 32U * 8U * sizeof(int)));
    CK(hipMalloc(&wsb, 64U * 8U * 32U * sizeof(float)));
    CK(hipMemset(coef, 0, 64U * 5U * 32U * 8U * sizeof(int)));
    CK(hipMemset(wsb, 0, 64U * 8U * 32U * sizeof(float)));
    std::printf("rate a4i-g32-w128-abel %.1f TOPS\n", ops16 / time_us([&] { hipLaunchKernelGGL(rot_rate<6>, dim3(640), dim3(256), 0, nullptr, iters / 4, coef, wsb, o); }, 5) / 1e6);
    std::printf("rate a4i-g64-w128-abel %.1f TOPS\n", ops16 / time_us([&] { hipLaunchKernelGGL(rot_rate<7>, dim3(640), dim3(256), 0, nullptr, iters / 4, coef, wsb, o); }, 5) / 1e6);
    return 0;
}
// ldsfloor mode (diagnostic, values meaningless): the tall core's WMMA loop with its operands read from a FIXED LDS
// tile (no global loads, no decode), so each arm's rate is the ceiling of one fragment structure on this chip.
// W waves per CTA arranged WR (row groups) x W / WR (column groups); a wave owns R row tiles x C column tiles. Per
// k unit it reads C B fragments (2 x ds_load_b128 each, ring 2 ahead, like core<>) and, with ALds, R A fragments
// (loaded one unit ahead); else A is a register constant. Stage 1 = the incumbent stage (every 4 units: barrier, 256
// stagers store 8 x b128, barrier); Stage 2 = one barrier + stores (double-buffer form). Unique: lanes 16-31 read
// other rows than lanes 0-15 (does the LDS merge the repeated half-wave addresses?).
namespace ldsfloor {
constexpr unsigned kPitch = 20;  // dwords per LDS row: 2 k units (32 halves) + 4 pad, conflict-free b128 like 36
// Compile-time loop: every index is a constant when the fragments are named, so no register array is promoted to one
// wide vector (a Half16 b[3] ring indexed in a #pragma unroll loop became <48 x half> + 15 re-pack VALU per WMMA).
template <unsigned I, unsigned N, class F> __device__ __forceinline__ void static_for(F&& f) {
    if constexpr (I < N) {
        f(std::integral_constant<unsigned, I>{});
        static_for<I + 1U, N>(f);
    }
}
template <unsigned W, unsigned R, unsigned C, unsigned WR, bool ALds, int Stage, bool Unique>
__global__ __launch_bounds__(32 * W) void kern(unsigned units, float* out, unsigned rnd) {
    constexpr unsigned Rows = 16U * R * WR, Cols = 16U * C * (W / WR);
    __shared__ alignas(16) std::uint32_t la[Rows][kPitch];
    __shared__ alignas(16) std::uint32_t lb[Cols][kPitch];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, c = lane & 15U;
    // rnd: FP16 values uniform in [-1, 1] (full mantissa toggling, like real activations / weights); else constants.
    const auto rh = [](unsigned i) __attribute__((always_inline)) {
        unsigned z = i * 2654435761U + 0x9E3779B9U;
        z ^= z >> 15U, z *= 2246822519U, z ^= z >> 13U;
        return static_cast<std::uint32_t>(__builtin_bit_cast(std::uint16_t, static_cast<_Float16>(float(int(z % 2001U) - 1000) / 1000.0F)));
    };
    const auto rw = [&](unsigned i) __attribute__((always_inline)) { return rh(2U * i) | (rh(2U * i + 1U) << 16U); };
    for (unsigned i = tid; i < Rows * kPitch; i += 32U * W) (&la[0][0])[i] = rnd ? rw(i) : 0x1C001C00U ^ (i & 0x00030003U);
    for (unsigned i = tid; i < Cols * kPitch; i += 32U * W) (&lb[0][0])[i] = rnd ? rw(i + 77777U) : 0x1C001C00U ^ (i & 0x00050005U);
    __syncthreads();
    const unsigned rb = (wave % WR) * 16U * R, cb = (wave / WR) * 16U * C;
    const unsigned bl = Unique ? lane : c;
    const auto frag = [](const std::uint32_t* p) __attribute__((always_inline)) { return tall::detail::staged_fragment(p); };
    tall::Half16 a[R], an[R];
    tall::Float8 acc[R][C] = {};
    static_for<0U, R>([&](auto r) __attribute__((always_inline)) {
        if constexpr (ALds) a[r] = frag(&la[rb + 16U * r + c][0]);
        else for (unsigned i = 0; i < 16U; ++i) a[r][i] = rnd ? __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(rh(tid * 64U + 16U * r + i)))
                                                           : static_cast<_Float16>(float((tid + i + r) & 3U));
    });
    tall::U4 keep[8];
    for (unsigned i = 0; i < 8U; ++i) keep[i] = *reinterpret_cast<const tall::U4*>(&lb[tid % Cols][4U * (i % 5U)]);
    for (unsigned u = 0; u < units; ++u) {
        const unsigned kk = u & 1U;
        if constexpr (Stage != 0) {
            if (u % 4U == 0U) {
                __syncthreads();
                if (tid < Cols) {
#pragma unroll
                    for (unsigned i = 0; i < 8U; ++i) *reinterpret_cast<tall::U4*>(&lb[tid][4U * (i & 3U)]) = keep[i];
                }
                if constexpr (Stage == 1) __syncthreads();
            }
        }
        asm volatile("" ::: "memory");
        if constexpr (ALds) static_for<0U, R>([&](auto r) __attribute__((always_inline)) { an[r] = frag(&la[rb + 16U * r + c][8U * (kk ^ 1U)]); });
        tall::detail::sched_fence();
        tall::Half16 r0, r1, r2;  // B ring, two fragments ahead
        const auto slot = [&](auto k) __attribute__((always_inline)) -> tall::Half16& {
            if constexpr (k % 3U == 0U) return r0;
            else if constexpr (k % 3U == 1U) return r1;
            else return r2;
        };
        const auto bfrag = [&](unsigned t) __attribute__((always_inline)) { return frag(&lb[(cb + 16U * t + bl) % Cols][8U * kk]); };
        r0 = bfrag(0U);
        if constexpr (C > 1U) r1 = bfrag(1U);
        tall::detail::sched_fence();
        static_for<0U, C>([&](auto t) __attribute__((always_inline)) {
            if constexpr (t + 2U < C) slot(std::integral_constant<unsigned, t + 2U>{}) = bfrag(t + 2U);
            static_for<0U, R>([&](auto r) __attribute__((always_inline)) {
                acc[r][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[r], slot(t), acc[r][t]);
            });
            tall::detail::sched_fence();
        });
        if constexpr (ALds) static_for<0U, R>([&](auto r) __attribute__((always_inline)) { a[r] = an[r]; });
    }
    float s = 0.0F;
    static_for<0U, R>([&](auto r) __attribute__((always_inline)) {
        static_for<0U, C>([&](auto t) __attribute__((always_inline)) { s += acc[r][t][(r + t) & 7U]; });
    });
    if (s == 1234.5F) out[tid] = s;
}
// The production core with its DRAM / decode paths removed: stage_load returns register values (no global loads),
// decode is a copy, A fragments read one 256-row x 64-k chunk of x16 tiles (L2-resident). Barriers, stores, the B ring
// and the A loads keep the production instruction stream: the incumbent structure's ceiling at a real shape.
template <unsigned W> struct FloorCols : iu::Cols<W> {
    using Raw = typename iu::Cols<W>::Raw;
    __device__ __forceinline__ Raw stage_load(unsigned t, unsigned c, unsigned u) const {
        Raw r;
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q) r.w[q] = 0x1C001C00U ^ ((t * 16U + c) & 3U) ^ (u & 1U) ^ q;
        r.s = u;
        return r;
    }
    __device__ __forceinline__ void stage_decode(const Raw& raw, tall::U4* nb) const {
        Raw r = raw;
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q) asm volatile("" : "+v"(r.w[q]));
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q) nb[q] = tall::U4{r.w[q], r.w[q] ^ r.s, r.w[q], r.w[q]};
    }
};
template <class Base> struct SmallA : Base {
    __device__ __forceinline__ tall::Half16 fragment(unsigned row, unsigned unit) const { return Base::fragment(row & 255U, unit & 3U); }
};
// Epilogue removed: the sums are only compared (keeps every WMMA live), no output bytes are written.
struct NullStore {
    struct Tile {
        float* y;
        unsigned c;
    };
    __device__ __forceinline__ Tile open(unsigned, const tall::ColumnTile& ct, unsigned c) const { return {ct.y, c}; }
    __device__ __forceinline__ void put(const Tile& t, unsigned, float v) const {
        if (v == 1234.5F) t.y[t.c] = v;
    }
    __device__ __forceinline__ void close() const {}
    template <class F> __device__ __forceinline__ void run(F&& f) { f(*this); }
};
template <class C> __global__ __launch_bounds__(512) void pkn(C c, const float* x, unsigned in, unsigned rows) {
    __shared__ tall::Storage<16> lds;
    tall::core<16>(x, in, rows, c, lds, NullStore{});
}
template <class C> static double run_nostore(const C& c, const iu::View& v) {
    const dim3 g(v.count / 256U, (v.rows + 255U) / 256U);  // column-fast (x16 policies)
    return time_us([&] { hipLaunchKernelGGL((pkn<C>), g, dim3(512), 0, nullptr, c, v.x, v.in, v.rows); });
}
template <unsigned W, unsigned R, unsigned C, unsigned WR, bool ALds, int Stage, bool Unique>
void arm(const char* name, float* o, unsigned rnd) {
    constexpr unsigned units = 256U, waves = 20U * 16U * 4U;  // 4 rounds of 16 waves on each of the 20 WGPs
    const unsigned grid = waves / W;
    const double ops = double(grid) * W * R * C * units * 8192.0;
    const auto go = [&] { hipLaunchKernelGGL((kern<W, R, C, WR, ALds, Stage, Unique>), dim3(grid), dim3(32 * W), 0, nullptr, units, o, rnd); };
    for (int i = 0; i < 100; ++i) go();  // ~80 ms: the clock settles at its power-limited level before timing
    const double us = time_us(go, 150);
    std::printf("ldsfloor %s %-28s W%-2u %ux%-2u A%s stage%d %8.1f us %6.2f TFLOPS  lds/WMMA %.3f KB\n", rnd ? "rand " : "const", name, W, R, C,
                ALds ? "lds" : "reg", Stage, us, ops / us / 1e6, (double(C) + (ALds ? double(R) : 0.0)) / double(R * C));
}
// argv[3] "core": only the real-shape decomposition; argv[3] or argv[4] "rand": random weights / x tiles (else
// constant memset data: the WMMA power and therefore the sustained clock depend on the data).
static void fill_rand(void* p, std::size_t bytes, std::uint32_t seed, bool half_pm1) {
    std::vector<std::uint16_t> h(bytes / 2U);
    std::uint32_t z = seed;
    for (auto& e : h) {
        z = z * 1664525U + 1013904223U;
        e = half_pm1 ? __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(float(int(z >> 9) % 2001 - 1000) / 1000.0F))
                     : static_cast<std::uint16_t>(z >> 16);
    }
    CK(hipMemcpy(p, h.data(), bytes, hipMemcpyHostToDevice));
}
static int run(int argc, char** argv) {
    bool core_only = false, rnd = false, micro = false;
    for (int i = 3; i < argc; ++i)
        core_only |= std::strcmp(argv[i], "core") == 0, rnd |= std::strcmp(argv[i], "rand") == 0, micro |= std::strcmp(argv[i], "micro") == 0;
    float* o = nullptr;
    CK(hipMalloc(&o, 4096));
    for (int round = 0; round < (core_only ? 0 : 2); ++round) {
        arm<16, 1, 16, 16, false, 0, false>("incumbent-1x16", o, rnd ? 1U : 0U);
        arm<16, 1, 16, 16, false, 1, false>("incumbent-1x16+stage2bar", o, rnd ? 1U : 0U);
        arm<16, 1, 16, 16, false, 2, false>("incumbent-1x16+stage1bar", o, rnd ? 1U : 0U);
        arm<16, 1, 16, 16, false, 0, true>("incumbent-1x16-unique", o, rnd ? 1U : 0U);
        arm<8, 1, 16, 8, false, 0, false>("incumbent-1x16-w8", o, rnd ? 1U : 0U);
        arm<16, 1, 16, 16, true, 0, false>("A-lds-1x16", o, rnd ? 1U : 0U);
        arm<16, 2, 8, 8, true, 0, false>("A-lds-2x8-w16", o, rnd ? 1U : 0U);
        arm<8, 2, 8, 4, true, 0, false>("A-lds-2x8-w8", o, rnd ? 1U : 0U);
        arm<4, 2, 8, 2, true, 0, false>("A-lds-2x8-w4", o, rnd ? 1U : 0U);
        arm<8, 4, 4, 2, true, 0, false>("A-lds-4x4-w8", o, rnd ? 1U : 0U);
        arm<8, 2, 8, 4, true, 2, false>("A-lds-2x8-w8+stage1bar", o, rnd ? 1U : 0U);
        arm<8, 4, 4, 2, true, 2, false>("A-lds-4x4-w8+stage1bar", o, rnd ? 1U : 0U);
    }
    // Real-shape decomposition (GEMM only, x16 form, 16 waves): floor (no DRAM, no decode, A one hot chunk), + real A
    // tiles, + real weight loads (copy, no decode), + decode (= production x16 GEMM). Two rounds, reversed order in round 1.
    const unsigned shapes[][2] = {{5120U, 17408U}, {17408U, 5120U}, {5120U, 10240U}};
    for (const unsigned rows : {512U, 4096U}) {
        if (micro) break;
        for (const auto& sh : shapes) {
            iu::View v{};
            v.in = sh[0], v.n = sh[1], v.count = sh[1], v.rows = rows, v.out_stride = sh[1];
            std::uint8_t* w = nullptr;
            std::uint16_t* sc = nullptr;
            _Float16* xt = nullptr;
            CK(hipMalloc(&v.out, std::size_t(rows) * sh[1] * 4U));
            CK(hipMalloc(&w, std::size_t(sh[0]) * sh[1] / 2U));
            CK(hipMalloc(&sc, std::size_t(sh[0]) / 64U * sh[1] * 2U));
            CK(hipMalloc(&xt, std::size_t(rows) * sh[0] * 2U));
            if (rnd) {
                fill_rand(w, std::size_t(sh[0]) * sh[1] / 2U, 7U, false);
                const std::vector<std::uint16_t> hs(std::size_t(sh[0]) / 64U * sh[1], std::uint16_t(0x2c00));  // 1/16
                CK(hipMemcpy(sc, hs.data(), hs.size() * 2U, hipMemcpyHostToDevice));
                fill_rand(xt, std::size_t(rows) * sh[0] * 2U, 11U, true);
            } else {
                CK(hipMemset(w, 0x5A, std::size_t(sh[0]) * sh[1] / 2U));
                CK(hipMemset(sc, 0x20, std::size_t(sh[0]) / 64U * sh[1] * 2U));
                CK(hipMemset(xt, 0x21, std::size_t(rows) * sh[0] * 2U));
            }
            v.weight = w, v.scale = sc;
            const double gf = 2.0 * rows * v.in * v.n;
            double t[4] = {};
            for (int round = 0; round < 2; ++round) {
                for (int k = 0; k < 4; ++k) {
                    const int arm_k = round ? 3 - k : k;
                    double us = 0.0;
                    if (arm_k == 0) us = run_arm(SmallA<tall::X16Cols<FloorCols<16>>>{{{{v}}, xt, rows}}, v);
                    if (arm_k == 1) us = run_nostore(SmallA<tall::X16Cols<FloorCols<16>>>{{{{v}}, xt, rows}}, v);
                    if (arm_k == 2) us = run_nostore(tall::X16Cols<PCols<16, false, false>>{{{v}}, xt, rows}, v);
                    if (arm_k == 3) us = run_arm(tall::X16Cols<PCols<16, false, false>>{{{v}}, xt, rows}, v);
                    t[arm_k] += us / 2.0;
                }
            }
            const char* names[4] = {"floor", "floor-nostore", "prod-nostore", "prod"};
            for (int k = 0; k < 4; ++k)
                std::printf("ldsfloor core %s in %5u n %5u rows %5u %-14s %8.1f us %6.2f TFLOPS\n", rnd ? "rand " : "const", v.in, v.n, rows, names[k], t[k], gf / t[k] / 1e6);
            CK(hipFree(v.out));
            CK(hipFree(w));
            CK(hipFree(sc));
            CK(hipFree(xt));
        }
    }
    return 0;
}
}  // namespace ldsfloor
// cmp mode: our production IU4 g64 W4A16 entries at the given shapes, the "ours" column of a GEMM-library comparison.
//   hip-dense-iu4-probe <rows> cmp in:n [in:n ...]
//   hip-dense-iu4-probe <rows> tile in:n [in:n ...]   (tall-core forms x CTA waves, ABBA, output bits vs direct 16 waves)
// rows <= 16: the small-M core (dense_linear_iu4 prepare + run: the decode / verify entry; prepare timed alone, run
// alone); rows > 16: prod16 / prod8 (iu4a16::enqueue) and x16prod+cvt (enqueue_x16_tiles + enqueue_x16).
// Weights are random codes with scale 1/16 (timing only). wMB = IU4 codes + FP16 scales read once per call.
static int cmp(unsigned rows, int argc, char** argv, bool tile = false) {
    namespace sm = strix::core::dense_iu4;
    std::vector<std::pair<unsigned, unsigned>> shapes;
    unsigned max_in = 0U, max_n = 0U;
    std::size_t max_w = 0U;
    for (int i = 3; i < argc; ++i) {
        unsigned in = 0U, n = 0U;
        if (std::sscanf(argv[i], "%u:%u", &in, &n) != 2 || in == 0U || n == 0U) {
            std::printf("bad shape '%s' (want in:n)\n", argv[i]);
            return 2;
        }
        shapes.emplace_back(in, n);
        max_in = std::max(max_in, in);
        max_n = std::max(max_n, n);
        max_w = std::max(max_w, sm::tensor_bytes(n, in));
    }
    if (shapes.empty() || rows == 0U) {
        std::printf("usage: hip-dense-iu4-probe <rows> cmp in:n [in:n ...]\n");
        return 2;
    }
    const std::size_t xrows = std::max<std::size_t>(rows, sm::wide_xh_rows(rows));
    std::uint8_t* w = nullptr;
    float *x = nullptr, *y = nullptr;
    _Float16 *xt = nullptr, *xh = nullptr;
    CK(hipMalloc(&w, max_w));
    CK(hipMalloc(&x, std::size_t(rows) * max_in * 4U));
    CK(hipMalloc(&y, std::size_t(rows) * max_n * 4U));
    CK(hipMalloc(&xt, std::size_t(rows) * max_in * 2U));
    CK(hipMalloc(&xh, xrows * max_in * 2U));
    {
        std::mt19937 g(7U);
        std::vector<std::uint8_t> hw(max_w);
        for (auto& e : hw) e = std::uint8_t(g());
        CK(hipMemcpy(w, hw.data(), max_w, hipMemcpyHostToDevice));
        std::vector<float> hx(std::size_t(rows) * max_in);
        std::uniform_real_distribution<float> d(-1.0F, 1.0F);
        for (auto& e : hx) e = d(g);
        CK(hipMemcpy(x, hx.data(), hx.size() * 4U, hipMemcpyHostToDevice));
        CK(hipMemset(xh, 0, xrows * max_in * 2U));
    }
    for (const auto& [in, n] : shapes) {
        // Scales 1/16 (0x2c00) right after this shape's packed codes (the pack layout).
        const std::size_t packed = sm::packed_bytes(n, in), sbytes = std::size_t(n) * (in / 64U) * 2U;
        {
            std::vector<std::uint16_t> hs(sbytes / 2U, 0x2c00U);
            CK(hipMemcpy(w + packed, hs.data(), sbytes, hipMemcpyHostToDevice));
        }
        const auto* scale = reinterpret_cast<const std::uint16_t*>(w + packed);
        const double gf = 2.0 * rows * double(in) * n / 1e3;
        const double wmb = double(packed + sbytes) / 1e6;
        auto pr = [&](const char* arm, double us) {
            std::printf("cmp in %5u n %5u rows %5u %-12s %10.1f us %6.2f TFLOPS wMB %.1f %.1f GB/s\n", in, n, rows, arm, us,
                gf / us / 1e3, wmb, wmb / us * 1e3);
        };
        if (rows <= sm::kMaxRows) {
            if (!sm::supported(n, in)) {
                std::printf("cmp in %5u n %5u rows %5u small-M unsupported shape\n", in, n, rows);
                continue;
            }
            pr("sm-prepare", time_us([&] { sm::prepare(x, xh, in, rows, nullptr); }));
            pr("sm-run", time_us([&] { sm::run(w, in, n, xh, y, rows, nullptr); }));
            continue;
        }
        if (in % 128U != 0U || n % 16U != 0U) {
            std::printf("cmp in %5u n %5u rows %5u tall unsupported shape\n", in, n, rows);
            continue;
        }
        iu::View v{w, scale, x, y, in, n, n, rows, n, 0U, 0U};
        if (!tile) {
            pr("prod16", time_us([&] { iu::enqueue(v, nullptr, 16U); }));
            pr("prod8", time_us([&] { iu::enqueue(v, nullptr, 8U); }));
            pr("x16prod+cvt", time_us([&] {
                tall::enqueue_x16_tiles(x, xt, rows, in, nullptr);
                iu::enqueue_x16(v, xt, nullptr);
            }));
            continue;
        }
        // tile mode: the tall core's forms (direct / x16 GEMM, the x16 tile pass timed alone) x CTA waves (16 / 8), in
        // ABBA order (round 1 reversed), every arm's output bits vs the direct form with 16 waves.
        const std::size_t ny = std::size_t(rows) * n;
        std::vector<float> ref(ny), got(ny);
        iu::enqueue(v, nullptr, 16U);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(ref.data(), y, ny * 4U, hipMemcpyDeviceToHost));
        tall::enqueue_x16_tiles(x, xt, rows, in, nullptr);
        const int reps = rows >= 2048U ? 8 : 20;
        pr("x16cvt", time_us([&] { tall::enqueue_x16_tiles(x, xt, rows, in, nullptr); }, reps));
        const unsigned arms[4][2] = {{0U, 16U}, {0U, 8U}, {1U, 16U}, {1U, 8U}};
        for (int round = 0; round < 2; ++round)
            for (unsigned a = 0; a < 4U; ++a) {
                const unsigned x16 = arms[round ? 3U - a : a][0], wv = arms[round ? 3U - a : a][1];
                const double us = time_us([&] {
                    if (x16) iu::enqueue_x16(v, xt, nullptr, wv);
                    else iu::enqueue(v, nullptr, wv);
                }, reps);
                CK(hipMemcpy(got.data(), y, ny * 4U, hipMemcpyDeviceToHost));
                const bool same = std::memcmp(got.data(), ref.data(), ny * 4U) == 0;
                char arm[32];
                std::snprintf(arm, sizeof arm, "%s-w%u", x16 ? "x16" : "dir", wv);
                std::printf("tile%d %s ", round, same ? "bits=" : "BITS!");
                pr(arm, us);
            }
    }
    CK(hipFree(w)); CK(hipFree(x)); CK(hipFree(y)); CK(hipFree(xt)); CK(hipFree(xh));
    return 0;
}
int main(int argc, char** argv) {
    if (argc > 2 && argv[2][0] == 'r') return rates();
    if (argc > 2 && std::strcmp(argv[2], "ldsfloor") == 0) return ldsfloor::run(argc, argv);
    if (argc > 2 && std::strcmp(argv[2], "cmp") == 0) return cmp(unsigned(std::atoi(argv[1])), argc, argv);
    if (argc > 2 && std::strcmp(argv[2], "tile") == 0) return cmp(unsigned(std::atoi(argv[1])), argc, argv, true);
    const unsigned rows = argc > 1 ? unsigned(std::atoi(argv[1])) : 512U;
    const unsigned shapes[][2] = {{5120U, 17408U}, {17408U, 5120U}, {5120U, 10240U}};
    Bufs b{};
    const std::size_t wmax = std::size_t(17408U) * 5120U / 2U, smax = std::size_t(17408U) * 5120U / 64U;
    CK(hipMalloc(&b.w, wmax));
    CK(hipMalloc(&b.s, smax * 2U));
    CK(hipMalloc(&b.x, std::size_t(rows) * 17408U * 4U));
    CK(hipMalloc(&b.y, std::size_t(rows) * 17408U * 4U));
    CK(hipMalloc(&b.xt, std::size_t(rows) * 17408U * 2U));
    CK(hipMalloc(&b.g, std::size_t(rows) * 17408U * 4U));
    CK(hipMalloc(&b.u, std::size_t(rows) * 17408U * 4U));
    CK(hipMemset(b.g, 0, std::size_t(rows) * 17408U * 4U));
    {
        std::mt19937 g(7);
        std::vector<std::uint8_t> hw(wmax);
        for (auto& e : hw) e = std::uint8_t(g());
        std::vector<std::uint16_t> hs(smax, 0x2c00U);  // scale 1/16
        std::vector<float> hx(std::size_t(rows) * 17408U);
        std::uniform_real_distribution<float> d(-1.0F, 1.0F);
        for (auto& e : hx) e = d(g);
        CK(hipMemcpy(b.w, hw.data(), wmax, hipMemcpyHostToDevice));
        CK(hipMemcpy(b.s, hs.data(), smax * 2U, hipMemcpyHostToDevice));
        CK(hipMemcpy(b.x, hx.data(), hx.size() * 4U, hipMemcpyHostToDevice));
    }
    for (const auto& sh : shapes) {
        const unsigned in = sh[0], n = sh[1];
        iu::View v{b.w, b.s, b.x, b.y, in, n, n, rows, n, 0U, 0U};
        const double gf = 2.0 * rows * double(in) * n / 1e3;  // flop per us -> /1e6 = TFLOPS
        auto pr = [&](const char* arm, double us) { std::printf("in %5u n %5u rows %u %-16s %9.1f us %6.2f TFLOPS\n", in, n, rows, arm, us, gf / us / 1e3); };
        pr("prod16", time_us([&] { iu::enqueue(v, nullptr, 16U); }));
        pr("prod8", time_us([&] { iu::enqueue(v, nullptr, 8U); }));
        pr("alias", run_arm(PCols<16, true, false>{{v}}, v));
        pr("nodec", run_arm(PCols<16, false, true>{{v}}, v));
        pr("alias+nodec", run_arm(PCols<16, true, true>{{v}}, v));
        pr("x16cvt", time_us([&] { tall::enqueue_x16_tiles(b.x, b.xt, rows, in, nullptr); }));
        pr("x16", run_arm(tall::X16Cols<PCols<16, false, false>>{{{v}}, b.xt, rows}, v));
        pr("x16+alias+nodec", run_arm(tall::X16Cols<PCols<16, true, true>>{{{v}}, b.xt, rows}, v));
        pr("noA", run_arm(NoACols<PCols<16, false, false>>{{{v}}}, v));
        pr("noA+alias+nodec", run_arm(NoACols<PCols<16, true, true>>{{{v}}}, v));
        pr("x16prod+cvt", time_us([&] {
            tall::enqueue_x16_tiles(b.x, b.xt, rows, in, nullptr);
            iu::enqueue_x16(v, b.xt, nullptr);
        }));
    }
    prep_arms(rows);
    return rates();
}
