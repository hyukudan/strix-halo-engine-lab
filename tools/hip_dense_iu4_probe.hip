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
#include "core/f16_dense_iu4.hpp"
#include "attn_core/dense_decode.hpp"
#include "core/dense_linear_iu4.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
    return 0;
}
// cmp mode: our production IU4 g64 W4A16 entries at the given shapes, the "ours" column of a GEMM-library comparison.
//   hip-dense-iu4-probe <rows> cmp in:n [in:n ...]
// rows <= 16: the small-M core (dense_linear_iu4 prepare + run: the decode / verify entry; prepare timed alone, run
// alone); rows > 16: prod16 / prod8 (iu4a16::enqueue) and x16prod+cvt (enqueue_x16_tiles + enqueue_x16).
// Weights are random codes with scale 1/16 (timing only). wMB = IU4 codes + FP16 scales read once per call.
static int cmp(unsigned rows, int argc, char** argv) {
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
        pr("prod16", time_us([&] { iu::enqueue(v, nullptr, 16U); }));
        pr("prod8", time_us([&] { iu::enqueue(v, nullptr, 8U); }));
        pr("x16prod+cvt", time_us([&] {
            tall::enqueue_x16_tiles(x, xt, rows, in, nullptr);
            iu::enqueue_x16(v, xt, nullptr);
        }));
    }
    CK(hipFree(w)); CK(hipFree(x)); CK(hipFree(y)); CK(hipFree(xt)); CK(hipFree(xh));
    return 0;
}
int main(int argc, char** argv) {
    if (argc > 2 && argv[2][0] == 'r') return rates();
    if (argc > 2 && std::strcmp(argv[2], "cmp") == 0) return cmp(unsigned(std::atoi(argv[1])), argc, argv);
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
