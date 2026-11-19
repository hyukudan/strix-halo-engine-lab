// cuda_core device test: every kernel vs a float64 CPU reference (tolerance) and bitwise row-count / repeat invariance.
// Needs one CUDA device (run with CUDA_VISIBLE_DEVICES=0). Exit 0 = pass.
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <random>
#include <limits>
#include <vector>

#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/attention_row_walk.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/gdn.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/rope.hpp"
#include "strix/cuda_core/nvfp4.hpp"
#include "../../src/cuda_core/common.cuh"

namespace cc = strix::cuda_core;
static int g_fail = 0;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr; cudaMalloc(&p, h.size() * sizeof(T)); cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); return p;
}
template <class T> T* devz(std::size_t n) { T* p = nullptr; cudaMalloc(&p, n * sizeof(T)); cudaMemset(p, 0, n * sizeof(T)); return p; }
template <class T> std::vector<T> get(const T* p, std::size_t n) {
    std::vector<T> h(n); cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return h;
}
static bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
static double maxrel(const std::vector<float>& a, const std::vector<double>& ref) {
    if (a.size() != ref.size()) return std::numeric_limits<double>::infinity();
    for (float v : a) if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
    for (double v : ref) if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
    double worst = 0, scale = 1e-30;
    for (double r : ref) scale = std::max(scale, std::fabs(r));
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - ref[i]) / scale);
    return worst;
}
static std::uint32_t* dpos(std::uint32_t v) { std::uint32_t* p = nullptr; cudaMalloc((void**)&p, 4); cudaMemcpy(p, &v, 4, cudaMemcpyHostToDevice); return p; }
static std::uint16_t bf16_bits(float f) {
    std::uint32_t u; std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<std::uint16_t>(u >> 16);
}
static float bf16_val(std::uint16_t b) { std::uint32_t u = static_cast<std::uint32_t>(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static std::uint16_t half_bits(float f) {  // exact for the values used (d = k/1024)
    __half h = __float2half(f); std::uint16_t b; std::memcpy(&b, &h, 2); return b;
}
static float half_val(std::uint16_t b) { __half h; std::memcpy(&h, &b, 2); return __half2float(h); }

static std::mt19937 rng(1234);
static float rnd(float s) { return std::uniform_real_distribution<float>(-s, s)(rng); }

static int test_dense_arith(cc::DenseArith arith, std::uint32_t in) {
    cc::set_dense_arith(arith);
    const bool mma = arith != cc::DenseArith::fp32;
    const std::uint32_t out = mma ? 80 : 70, R = 32;
    std::vector<std::uint16_t> wb(out * in);
    std::vector<double> wd(out * in);
    for (std::size_t i = 0; i < wb.size(); ++i) { wb[i] = bf16_bits(rnd(0.1f)); wd[i] = bf16_val(wb[i]); }
    std::vector<unsigned char> wq(static_cast<std::size_t>(out) * (in / 32) * 34);
    std::vector<double> wqd(out * in);
    for (std::uint32_t o = 0; o < out; ++o)
        for (std::uint32_t b = 0; b < in / 32; ++b) {
            unsigned char* blk = &wq[(static_cast<std::size_t>(o) * (in / 32) + b) * 34];
            const std::uint16_t hb = half_bits(static_cast<float>(1 + (rng() % 50)) / 1024.0f);
            blk[0] = hb & 0xFF; blk[1] = hb >> 8;
            for (int j = 0; j < 32; ++j) { const int q = static_cast<int>(rng() % 255) - 127; blk[2 + j] = static_cast<unsigned char>(static_cast<signed char>(q)); wqd[o * in + b * 32 + j] = half_val(hb) * q; }
        }
    std::vector<float> wf(out * in);
    for (auto& v : wf) v = rnd(0.1f);
    std::vector<float> x(R * in), res(R * out);
    for (auto& v : x) v = rnd(1.0f);
    for (auto& v : res) v = rnd(1.0f);
    auto* dx = dev(x); auto* dres = dev(res);
    auto* dwb = dev(wb); auto* dwq = dev(wq); auto* dwf = dev(wf);
    auto* dy = devz<float>(R * out); auto* dy1 = devz<float>(R * out);
    struct Case { const char* name; cc::DenseWeight w; const std::vector<double>* ref; std::vector<double> reff; };
    std::vector<double> wfd(wf.begin(), wf.end());
    Case cases[3] = {{"bf16", {dwb, cc::WeightFormat::bf16, out, in}, &wd, {}}, {"q8_0", {dwq, cc::WeightFormat::q8_0, out, in}, &wqd, {}},
                     {"f32", {dwf, cc::WeightFormat::f32, out, in}, &wfd, {}}};
    for (auto& cs : cases) {
        for (std::uint32_t rows : {1u, 2u, 3u, 5u, 8u, 11u, 16u, 17u, 24u, 32u}) {
            if (!mma && rows > 16) continue;
            CK(cc::dense_rows(cs.w, dx, dy, rows, nullptr, nullptr));
            auto y = get(dy, rows * out);
            std::vector<double> ref(rows * out);
            for (std::uint32_t r = 0; r < rows; ++r)
                for (std::uint32_t o = 0; o < out; ++o) { double s = 0; for (std::uint32_t k = 0; k < in; ++k) s += (*cs.ref)[o * in + k] * x[r * in + k]; ref[r * out + o] = s; }
            CHECK(maxrel(y, ref) < (arith == cc::DenseArith::mma1 && cs.w.format != cc::WeightFormat::f32 ? 1e-2 : 1e-5), "%s rows %u relerr %g", cs.name, rows, maxrel(y, ref));
            // row invariance: each row alone must equal the batched row, bitwise
            for (std::uint32_t r = 0; r < rows; ++r) {
                CK(cc::dense_rows(cs.w, dx + r * in, dy1 + r * out, 1, nullptr, nullptr));
            }
            auto y1 = get(dy1, rows * out);
            CHECK(bits_equal(y, y1), "%s rows %u not bitwise equal to rows=1", cs.name, rows);
        }
        // residual: y = res + (W x) bitwise
        CK(cc::dense_rows(cs.w, dx, dy, 7, nullptr, nullptr));
        CK(cc::dense_rows(cs.w, dx, dy1, 7, dres, nullptr));
        auto a = get(dy, 7 * out), b = get(dy1, 7 * out);
        bool ok = true;
        for (std::size_t i = 0; i < a.size(); ++i) { const float e = res[i] + a[i]; ok &= std::memcmp(&e, &b[i], 4) == 0; }
        CHECK(ok, "%s residual not bitwise res + dot", cs.name);
    }
    // fused gate|up: silu(gate)*up vs two dense calls + host silu (tolerance), fused == fused rows=1
    CK(cc::dense_gate_up_silu_rows(cases[0].w, cases[0].w, dx, dy, 9, nullptr));
    CK(cc::dense_rows(cases[0].w, dx, dy1, 9, nullptr, nullptr));
    auto f = get(dy, 9 * out), g = get(dy1, 9 * out);
    std::vector<double> ref(9 * out);
    for (std::size_t i = 0; i < ref.size(); ++i) ref[i] = g[i] / (1.0 + std::exp(-(double)g[i])) * g[i];
    CHECK(maxrel(f, ref) < 1e-5, "fused relerr %g", maxrel(f, ref));
    for (std::uint32_t r = 0; r < 9; ++r) CK(cc::dense_gate_up_silu_rows(cases[0].w, cases[0].w, dx + r * in, dy1 + r * out, 1, nullptr));
    CHECK(bits_equal(f, get(dy1, 9 * out)), "fused rows invariance");
    cc::set_dense_arith(cc::DenseArith::mma3);
    return 0;
}
int test_dense() {
    if (int e = test_dense_arith(cc::DenseArith::fp32, 1056)) return e;
    if (int e = test_dense_arith(cc::DenseArith::mma3, 1056)) return e;
    if (int e = test_dense_arith(cc::DenseArith::mma3, 1280)) return e;  // q8_0 staged path (in % 256 == 0)
    return test_dense_arith(cc::DenseArith::mma1, 1280);
}

// norm_kernel loads gain / z together with x (one batch before any store) instead of inside the store loop. Invariant:
// rms == host emulation of the documented order (thread t sums x[t + 256 i] ascending, xor butterfly 16..1, warps 0..7,
// inv = 1 / sqrt(ss / n + eps), y = (x * inv) * gain) bit for bit; gated swish == rms * silu(z) bit for bit; gated in
// place (y == z) == out of place. n 5120 (NREG 20, the verify rms) and n 128 x 48 groups (the GDN gated norm).
static float f_mul(float a, float b) { volatile float r = a * b; return r; }
static float f_add(float a, float b) { volatile float r = a + b; return r; }
static float host_rms_inv(const float* x, std::uint32_t n, float eps) {
    float v[256];
    for (std::uint32_t t = 0; t < 256; ++t) {
        float p = 0.0f;
        for (std::uint32_t idx = t; idx < n; idx += 256) p = f_add(p, f_mul(x[idx], x[idx]));
        v[t] = p;
    }
    for (std::uint32_t w = 0; w < 8; ++w)
        for (std::uint32_t off = 16; off > 0; off >>= 1) {
            float nv[32];
            for (std::uint32_t l = 0; l < 32; ++l) nv[l] = f_add(v[w * 32 + l], v[w * 32 + (l ^ off)]);
            for (std::uint32_t l = 0; l < 32; ++l) v[w * 32 + l] = nv[l];
        }
    float s = v[0];
    for (std::uint32_t w = 1; w < 8; ++w) s = f_add(s, v[w * 32]);
    volatile float q = s / static_cast<float>(n);
    q = q + eps;
    volatile float r = std::sqrt(static_cast<float>(q));
    volatile float inv = 1.0f / r;
    return inv;
}
__global__ void silu_mul_ref(float* y, const float* z, std::uint32_t count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = __fmul_rn(y[i], __fdiv_rn(z[i], __fadd_rn(1.0f, expf(-z[i]))));
}
int test_norm_batched_loads() {
    for (std::uint32_t n : {5120u, 128u}) {
        const std::uint32_t groups = n == 128 ? 48 : 1;
        for (std::uint32_t rows : {1u, 8u, 16u}) {
            const std::size_t cnt = static_cast<std::size_t>(rows) * groups * n;
            std::vector<float> x(cnt), gain(n), z(cnt);
            for (auto& e : x) e = rnd(2.0f);
            for (auto& e : gain) e = rnd(1.0f);
            for (auto& e : z) e = rnd(3.0f);
            auto *dx = dev(x), *dg = dev(gain), *dz = dev(z), *dy = devz<float>(cnt), *dy2 = devz<float>(cnt);
            const cc::GroupedView v{rows, groups, n, 0, 0};
            CK(cc::rmsnorm(dx, v, dg, dy, v, 1e-6f, nullptr));
            const auto y = get(dy, cnt);
            std::vector<float> ref(cnt);
            for (std::size_t b = 0; b < static_cast<std::size_t>(rows) * groups; ++b) {
                const float inv = host_rms_inv(x.data() + b * n, n, 1e-6f);
                for (std::uint32_t i = 0; i < n; ++i) ref[b * n + i] = f_mul(f_mul(x[b * n + i], inv), gain[i]);
            }
            CHECK(bits_equal(y, ref), "rmsnorm n %u rows %u != host order emulation", n, rows);
            CK(cc::gated_rmsnorm(dx, v, dg, dz, v, cc::GateKind::swish, dy2, v, 1e-6f, nullptr));
            silu_mul_ref<<<static_cast<unsigned>((cnt + 255) / 256), 256>>>(dy, dz, static_cast<std::uint32_t>(cnt));
            CK(cudaGetLastError());
            const auto gated = get(dy2, cnt);
            CHECK(bits_equal(gated, get(dy, cnt)), "gated n %u rows %u != rms * silu(z)", n, rows);
            CK(cc::gated_rmsnorm(dx, v, dg, dz, v, cc::GateKind::swish, dz, v, 1e-6f, nullptr));  // y aliases z
            CHECK(bits_equal(get(dz, cnt), gated), "in-place gated n %u rows %u", n, rows);
            for (float* p : {dx, dg, dz, dy, dy2}) cudaFree(p);
        }
    }
    return 0;
}

int test_norms_rope() {
    const std::uint32_t rows = 5, groups = 3, n = 300;
    std::vector<float> x(rows * groups * n), gain(n), z(x.size());
    for (auto& v : x) v = rnd(2.0f);
    for (auto& v : gain) v = rnd(1.0f);
    for (auto& v : z) v = rnd(2.0f);
    auto *dx = dev(x), *dg = dev(gain), *dz = dev(z), *dy = devz<float>(x.size());
    const cc::GroupedView v{rows, groups, n, 0, 0};
    CK(cc::rmsnorm(dx, v, dg, dy, v, 1e-6f, nullptr));
    auto y = get(dy, x.size());
    std::vector<double> ref(x.size()), ref2(x.size()), ref3(x.size());
    for (std::uint32_t r = 0; r < rows * groups; ++r) {
        double ss = 0; for (std::uint32_t i = 0; i < n; ++i) ss += (double)x[r * n + i] * x[r * n + i];
        const double inv = 1.0 / std::sqrt(ss / n + 1e-6), inv2 = 1.0 / std::sqrt(ss + 1e-6);
        for (std::uint32_t i = 0; i < n; ++i) {
            ref[r * n + i] = x[r * n + i] * inv * gain[i];
            ref2[r * n + i] = x[r * n + i] * inv * gain[i] * (z[r * n + i] / (1.0 + std::exp(-(double)z[r * n + i])));
            ref3[r * n + i] = x[r * n + i] * inv2 * 0.25;
        }
    }
    CHECK(maxrel(y, ref) < 1e-6, "rmsnorm %g", maxrel(y, ref));
    CK(cc::gated_rmsnorm(dx, v, dg, dz, v, cc::GateKind::swish, dy, v, 1e-6f, nullptr));
    CHECK(maxrel(get(dy, x.size()), ref2) < 1e-6, "gated %g", maxrel(get(dy, x.size()), ref2));
    CK(cc::l2norm(dx, v, dy, v, 1e-6f, 0.25f, nullptr));
    CHECK(maxrel(get(dy, x.size()), ref3) < 1e-6, "l2 %g", maxrel(get(dy, x.size()), ref3));
    // in place == out of place
    CK(cc::rmsnorm(dx, v, dg, dx, v, 1e-6f, nullptr));
    CHECK(bits_equal(get(dx, x.size()), y), "in-place rmsnorm");
    // rope
    const std::uint32_t heads = 4, hd = 256, rd = 64, rr = 3, pos0 = 10;
    std::vector<float> ct, st; cc::rope_table_host(32, rd, 1e7, ct, st);
    std::vector<float> q(rr * heads * hd); for (auto& e : q) e = rnd(1.0f);
    auto *dq = dev(q), *dc = dev(ct), *ds = dev(st);
    const cc::GroupedView qv{rr, heads, hd, 0, 0};
    CK(cc::rope_apply(dq, qv, rd, dc, ds, dpos(pos0), nullptr));
    auto qo = get(dq, q.size());
    std::vector<double> qr(q.size());
    for (std::uint32_t r = 0; r < rr; ++r) for (std::uint32_t h = 0; h < heads; ++h) for (std::uint32_t i = 0; i < hd; ++i) {
        const std::size_t o = (r * heads + h) * hd; double val = q[o + i];
        if (i < rd) {
            const std::uint32_t j = i % 32; const double ang = (double)(pos0 + r) * std::pow(1e7, -2.0 * j / rd);
            const double x1 = q[o + j], x2 = q[o + j + 32];
            val = i < 32 ? x1 * std::cos(ang) - x2 * std::sin(ang) : x2 * std::cos(ang) + x1 * std::sin(ang);
        }
        qr[o + i] = val;
    }
    CHECK(maxrel(qo, qr) < 1e-6, "rope %g", maxrel(qo, qr));
    return 0;
}

int test_attention() {
    cc::AttentionDims d; d.heads = 6; d.kv_heads = 2; d.head_dim = 256; d.scale = 1.0f / 16.0f; d.max_positions = 64;
    const std::uint32_t pos0 = 37, R = 4, total = pos0 + R;
    std::vector<float> kf(static_cast<std::size_t>(total) * 2 * 256), vf(kf.size());
    for (auto& e : kf) e = rnd(1.0f);
    for (auto& e : vf) e = rnd(1.0f);
    auto *dk = dev(kf), *dv = dev(vf);
    auto *kc = devz<__nv_bfloat16>(2 * 64 * 256), *vc = devz<__nv_bfloat16>(2 * 64 * 256);
    // history stored in one call, then window
    CK(cc::kv_store(d, dk, dv, kc, vc, dpos(0), 16, nullptr));
    for (std::uint32_t p = 16; p < pos0; p += 16) CK(cc::kv_store(d, dk + p * 512, dv + p * 512, kc, vc, dpos(p), std::min(16u, pos0 - p), nullptr));
    CK(cc::kv_store(d, dk + pos0 * 512, dv + pos0 * 512, kc, vc, dpos(pos0), R, nullptr));
    std::vector<float> q(R * 6 * 256), gt(q.size());
    for (auto& e : q) e = rnd(2.0f);
    for (auto& e : gt) e = rnd(2.0f);
    auto *dq = dev(q), *dgt = dev(gt), *dout = devz<float>(q.size()), *dsc = devz<float>(cc::attention_scratch_floats(d, R)), *dout1 = devz<float>(q.size());
    const cc::GroupedView qv{R, 6, 256, 0, 0};
    CK(cc::attention_decode(d, dq, qv, dgt, qv, kc, vc, dpos(pos0), R, dout, dsc, nullptr));
    auto o = get(dout, q.size());
    // reference with bf16-rounded K/V
    std::vector<double> ref(q.size());
    for (std::uint32_t r = 0; r < R; ++r) for (std::uint32_t h = 0; h < 6; ++h) {
        const std::uint32_t kvh = h / 3, len = pos0 + r + 1;
        std::vector<double> s(len); double m = -1e300;
        for (std::uint32_t p = 0; p < len; ++p) {
            double dot = 0; for (int i = 0; i < 256; ++i) dot += (double)q[(r * 6 + h) * 256 + i] * bf16_val(bf16_bits(kf[(p * 2 + kvh) * 256 + i]));
            s[p] = dot / 16.0; m = std::max(m, s[p]);
        }
        double den = 0; for (auto& e : s) { e = std::exp(e - m); den += e; }
        for (int i = 0; i < 256; ++i) {
            double acc = 0; for (std::uint32_t p = 0; p < len; ++p) acc += s[p] * bf16_val(bf16_bits(vf[(p * 2 + kvh) * 256 + i]));
            const double g = gt[(r * 6 + h) * 256 + i];
            ref[(r * 6 + h) * 256 + i] = acc / den / (1.0 + std::exp(-g));
        }
    }
    CHECK(maxrel(o, ref) < 1e-5, "attention relerr %g", maxrel(o, ref));
    for (std::uint32_t r = 0; r < R; ++r)
        CK(cc::attention_decode(d, dq + r * 1536, cc::GroupedView{1, 6, 256, 0, 0}, dgt + r * 1536, cc::GroupedView{1, 6, 256, 0, 0}, kc, vc, dpos(pos0 + r), 1, dout1 + r * 1536, dsc, nullptr));
    CHECK(bits_equal(o, get(dout1, q.size())), "attention rows invariance");
    return 0;
}

// Frozen copy of the decode attention phases A and C as they were before the load batching (one K load per position per
// warp, V unroll 4), test-only: the library kernels must write bit-identical scratch (scores, chunk max, chunk denominators,
// chunk value partials); the unchanged combine then gives identical outputs.
namespace attn_v0 {
using namespace strix::cuda_core;
constexpr std::uint32_t CH = 256;
__global__ void __launch_bounds__(256) scores(cc::AttentionDims d, const float* q, cc::GroupedView qv, const __nv_bfloat16* kc,
                                              const std::uint32_t* pos0p, float* sscores, float* scmax) {
    const std::uint32_t chunk = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
    const std::uint32_t len = *pos0p + r + 1, base = chunk * CH;
    if (base >= len) return;
    const std::uint32_t G = d.heads / d.kv_heads, lane = threadIdx.x & 31, w = threadIdx.x >> 5, mc = cc::attn_max_chunks(d);
    float qr[8][8];
    for (int g = 0; g < 8; ++g)
        if (g < static_cast<int>(G))
            for (int j = 0; j < 8; ++j) qr[g][j] = q[static_cast<std::size_t>(r) * qv.rs() + static_cast<std::size_t>(kvh * G + g) * qv.gs() + lane * 8 + j];
    float mx[8];
    for (int g = 0; g < 8; ++g) mx[g] = __int_as_float(0xff800000);
    const std::uint32_t end = min(len, base + CH);
    const __nv_bfloat16* kb = kc + static_cast<std::size_t>(kvh) * d.max_positions * 256 + lane * 8;
    for (std::uint32_t p = base + w; p < end; p += 8) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(kb + static_cast<std::size_t>(p) * 256));
        const float kf[8] = {bf16_lo(u.x), bf16_hi(u.x), bf16_lo(u.y), bf16_hi(u.y), bf16_lo(u.z), bf16_hi(u.z), bf16_lo(u.w), bf16_hi(u.w)};
        for (int g = 0; g < 8; ++g) {
            if (g < static_cast<int>(G)) {
                float dot = 0.0f;
                for (int j = 0; j < 8; ++j) dot = __fadd_rn(dot, __fmul_rn(qr[g][j], kf[j]));
                dot = warp_sum_xor(dot);
                const float s = __fmul_rn(dot, d.scale);
                mx[g] = fmaxf(mx[g], s);
                if (lane == 0) sscores[(static_cast<std::size_t>(r) * d.heads + kvh * G + g) * d.max_positions + p] = s;
            }
        }
    }
    __shared__ float wm[8][8];
    for (int g = 0; g < 8; ++g) wm[w][g] = mx[g];
    __syncthreads();
    if (threadIdx.x < G) {
        float m = wm[0][threadIdx.x];
        for (int i = 1; i < 8; ++i) m = fmaxf(m, wm[i][threadIdx.x]);
        scmax[(static_cast<std::size_t>(r) * d.heads + kvh * G + threadIdx.x) * mc + chunk] = m;
    }
}
__global__ void __launch_bounds__(256) values(cc::AttentionDims d, const __nv_bfloat16* vc, const std::uint32_t* pos0p,
                                              const float* sscores, const float* scmax, float* sdpart, float* spart) {
    const std::uint32_t chunk = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
    const std::uint32_t len = *pos0p + r + 1, base = chunk * CH;
    if (base >= len) return;
    const std::uint32_t G = d.heads / d.kv_heads, t = threadIdx.x, lane = t & 31, w = t >> 5;
    const std::uint32_t mc = cc::attn_max_chunks(d), nch = (len + CH - 1) / CH;
    __shared__ float mg[8];
    __shared__ float es[8][CH];
    __shared__ float red[8];
    if (w < G) {
        const float* cm = scmax + (static_cast<std::size_t>(r) * d.heads + kvh * G + w) * mc;
        float m = __int_as_float(0xff800000);
        for (std::uint32_t c = lane; c < nch; c += 32) m = fmaxf(m, cm[c]);
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (lane == 0) mg[w] = m;
    }
    __syncthreads();
    const std::uint32_t p = base + t;
    for (std::uint32_t g = 0; g < G; ++g) {
        float e = 0.0f;
        if (p < len) e = cc_expf(__fsub_rn(sscores[(static_cast<std::size_t>(r) * d.heads + kvh * G + g) * d.max_positions + p], mg[g]));
        es[g][t] = e;
        const float ds = block_sum256(e, red);
        if (t == 0) sdpart[(static_cast<std::size_t>(r) * d.heads + kvh * G + g) * mc + chunk] = ds;
    }
    __syncthreads();
    float acc[8];
    for (int g = 0; g < 8; ++g) acc[g] = 0.0f;
    const std::uint32_t n = (len - base) < CH ? (len - base) : CH;
    const __nv_bfloat16* vp = vc + (static_cast<std::size_t>(kvh) * d.max_positions + base) * 256 + t;
#pragma unroll 4
    for (std::uint32_t i = 0; i < n; ++i) {
        const float vv = __bfloat162float(vp[static_cast<std::size_t>(i) * 256]);
        for (int g = 0; g < 8; ++g)
            if (g < static_cast<int>(G)) acc[g] = __fadd_rn(acc[g], __fmul_rn(es[g][i], vv));
    }
    for (int g = 0; g < 8; ++g)
        if (g < static_cast<int>(G)) spart[((static_cast<std::size_t>(r) * d.heads + kvh * G + g) * mc + chunk) * 256 + t] = acc[g];
}
}  // namespace attn_v0

// Frozen copy of origin/main 1490c00f8's decode attention (scratch layout, phases A / C and the combine, one row per
// block), test-only: the --attn-bench "main" arm times the kernels this branch replaces, launched as main launches them.
namespace attn_main {
using namespace strix::cuda_core;
constexpr std::uint32_t CH = kAttnChunk;  // 256 positions per chunk
constexpr int D = 256;
// Loads in flight per warp (scores: K rows of positions p, p+8, ...) and per thread (values: V elements, double-buffered).
// Decode attention at short context is bound by DRAM latency, not bandwidth; these only change when loads are issued.
constexpr int kScoreBatch = 8;
constexpr int kValueBatch = 32;
struct Scratch {
    float* scores;  // [rows][heads][max_positions]
    float* cmax;    // [rows][heads][maxchunks]
    float* dpart;   // [rows][heads][maxchunks]
    float* part;    // [rows][heads][maxchunks][D]
};
__host__ __device__ inline Scratch carve(float* base, const AttentionDims& d, std::uint32_t rows) {
    const std::size_t rh = static_cast<std::size_t>(rows) * d.heads, mc = attn_max_chunks(d);
    Scratch s;
    s.scores = base;
    s.cmax = s.scores + rh * d.max_positions;
    s.dpart = s.cmax + rh * mc;
    s.part = s.dpart + rh * mc;
    return s;
}

// Phase A: block (chunk, kv head, row); warp w handles positions base + w + 8 i for all G query heads of the group
// (the K row is read once). dot: lane owns 8 consecutive dims (sequential), xor butterfly 16..1; score = fl(dot) * scale.
__global__ void __launch_bounds__(256) attn_scores_kernel(AttentionDims d, const float* q, GroupedView qv, const __nv_bfloat16* kc,
                                                          const std::uint32_t* pos0p, std::uint32_t rows, Scratch sc, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t chunk = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
    const std::uint32_t len = (rm.pos ? rm.pos[r] : *pos0p + r) + 1, base = chunk * CH;
    if (base >= len) return;
    if (rm.slot) kc += rm.slot[r] * rm.slot_stride;
    const std::uint32_t G = d.heads / d.kv_heads, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const std::uint32_t mc = attn_max_chunks(d);
    float qr[8][8];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g < static_cast<int>(G)) {
            const float* qp = q + static_cast<std::size_t>(r) * qv.rs() + static_cast<std::size_t>(kvh * G + g) * qv.gs() + lane * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) qr[g][j] = qp[j];
        }
    }
    float mx[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) mx[g] = __int_as_float(0xff800000);
    const std::uint32_t end = min(len, base + CH);
    const __nv_bfloat16* kb = kc + static_cast<std::size_t>(kvh) * d.max_positions * D + lane * 8;
    // Loads of kScoreBatch positions (p, p+8, ...) are issued together and the next batch is issued before the current one
    // is computed, so a warp waits for memory about once per batch instead of once per position. The batch is staged in
    // shared memory (each lane reads back only its own slots) so the position loop stays rolled: code size as before.
    // Per-position arithmetic and the ascending position order are unchanged (bitwise).
    __shared__ uint4 kst[kScoreBatch][8][32];
    uint4 u[kScoreBatch];
#pragma unroll
    for (int b = 0; b < kScoreBatch; ++b) {
        const std::uint32_t p = base + w + 8u * static_cast<std::uint32_t>(b);
        u[b] = p < end ? __ldg(reinterpret_cast<const uint4*>(kb + static_cast<std::size_t>(p) * D)) : make_uint4(0, 0, 0, 0);
    }
    for (std::uint32_t p0 = base + w; p0 < end; p0 += 8u * kScoreBatch) {
#pragma unroll
        for (int b = 0; b < kScoreBatch; ++b) kst[b][w][lane] = u[b];
        const std::uint32_t p1 = p0 + 8u * kScoreBatch;
#pragma unroll
        for (int b = 0; b < kScoreBatch; ++b) {
            const std::uint32_t p = p1 + 8u * static_cast<std::uint32_t>(b);
            u[b] = p < end ? __ldg(reinterpret_cast<const uint4*>(kb + static_cast<std::size_t>(p) * D)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll 1
        for (std::uint32_t b = 0; b < static_cast<std::uint32_t>(kScoreBatch); ++b) {
            const std::uint32_t p = p0 + 8u * b;
            if (p >= end) break;  // warp-uniform
            const uint4 k4 = kst[b][w][lane];
            const float kf[8] = {bf16_lo(k4.x), bf16_hi(k4.x), bf16_lo(k4.y), bf16_hi(k4.y),
                                 bf16_lo(k4.z), bf16_hi(k4.z), bf16_lo(k4.w), bf16_hi(k4.w)};
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                if (g < static_cast<int>(G)) {
                    float dot = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) dot = __fadd_rn(dot, __fmul_rn(qr[g][j], kf[j]));
                    dot = warp_sum_xor(dot);
                    const float s = __fmul_rn(dot, d.scale);
                    mx[g] = fmaxf(mx[g], s);
                    if (lane == 0) sc.scores[(static_cast<std::size_t>(r) * d.heads + kvh * G + g) * d.max_positions + p] = s;
                }
            }
        }
    }
    __shared__ float wm[8][8];
#pragma unroll
    for (int g = 0; g < 8; ++g) wm[w][g] = mx[g];
    __syncthreads();
    if (threadIdx.x < G) {
        float m = wm[0][threadIdx.x];
        for (int i = 1; i < 8; ++i) m = fmaxf(m, wm[i][threadIdx.x]);
        sc.cmax[(static_cast<std::size_t>(r) * d.heads + kvh * G + threadIdx.x) * mc + chunk] = m;
    }
}

// Phase C: block (chunk, kv head, row); thread t = output dim and (for the weights) position base + t.
// e_p = expf(s_p - m) with the exact global max m; chunk denominator = block tree of the e (thread t holds position base + t);
// chunk value partial = sum over the chunk positions ascending of e_p * v[p][t] (separate rounded mul, add).
__global__ void __launch_bounds__(256) attn_values_kernel(AttentionDims d, const __nv_bfloat16* vc, const std::uint32_t* pos0p,
                                                          std::uint32_t rows, Scratch sc, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t chunk = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
    const std::uint32_t len = (rm.pos ? rm.pos[r] : *pos0p + r) + 1, base = chunk * CH;
    if (base >= len) return;
    if (rm.slot) vc += rm.slot[r] * rm.slot_stride;
    const std::uint32_t G = d.heads / d.kv_heads, t = threadIdx.x, lane = t & 31, w = t >> 5;
    const std::uint32_t mc = attn_max_chunks(d), nch = (len + CH - 1) / CH;
    __shared__ float mg[8];
    __shared__ float es[8][CH];
    __shared__ float red[8];
    if (w < G) {
        const float* cm = sc.cmax + (static_cast<std::size_t>(r) * d.heads + kvh * G + w) * mc;
        float m = __int_as_float(0xff800000);
        for (std::uint32_t c = lane; c < nch; c += 32) m = fmaxf(m, cm[c]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (lane == 0) mg[w] = m;
    }
    __syncthreads();
    const std::uint32_t p = base + t;
    // The G scores of this thread are loaded together (one memory wait, not one per head); the expf and the block tree of
    // each head are unchanged and run in the same head order.
    float sv[8];
#pragma unroll
    for (int g = 0; g < 8; ++g)
        sv[g] = (g < static_cast<int>(G) && p < len)
                    ? sc.scores[(static_cast<std::size_t>(r) * d.heads + kvh * G + static_cast<std::uint32_t>(g)) * d.max_positions + p]
                    : 0.0f;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g >= static_cast<int>(G)) break;  // block-uniform
        float e = 0.0f;
        if (p < len) e = cc_expf(__fsub_rn(sv[g], mg[g]));
        es[g][t] = e;
        const float ds = block_sum256(e, red);
        if (t == 0) sc.dpart[(static_cast<std::size_t>(r) * d.heads + kvh * G + static_cast<std::uint32_t>(g)) * mc + chunk] = ds;
    }
    __syncthreads();
    float acc[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) acc[g] = 0.0f;
    const std::uint32_t n = (len - base) < CH ? (len - base) : CH;
    const __nv_bfloat16* vp = vc + (static_cast<std::size_t>(kvh) * d.max_positions + base) * D + t;
    // V in batches of kValueBatch positions: the next batch is issued before the current one is accumulated, so a thread
    // waits for memory about once per batch. The batch is staged in shared memory (each thread reads back only its own
    // column) so the accumulation loop stays rolled. Accumulation stays ascending in i, rounded mul then add (bitwise).
    __shared__ __nv_bfloat16 vst[kValueBatch][CH];
    const __nv_bfloat16 vzero = __ushort_as_bfloat16(static_cast<unsigned short>(0));
    __nv_bfloat16 u[kValueBatch];
#pragma unroll
    for (int b = 0; b < kValueBatch; ++b) u[b] = static_cast<std::uint32_t>(b) < n ? vp[static_cast<std::size_t>(b) * D] : vzero;
    for (std::uint32_t i0 = 0; i0 < n; i0 += kValueBatch) {
#pragma unroll
        for (int b = 0; b < kValueBatch; ++b) vst[b][t] = u[b];
        const std::uint32_t i1 = i0 + kValueBatch;
#pragma unroll
        for (int b = 0; b < kValueBatch; ++b) {
            const std::uint32_t i = i1 + static_cast<std::uint32_t>(b);
            u[b] = i < n ? vp[static_cast<std::size_t>(i) * D] : vzero;
        }
        const std::uint32_t nb = min(n - i0, static_cast<std::uint32_t>(kValueBatch));
#pragma unroll 4
        for (std::uint32_t b = 0; b < nb; ++b) {
            const float vv = __bfloat162float(vst[b][t]);
            const std::uint32_t i = i0 + b;
#pragma unroll
            for (int g = 0; g < 8; ++g)
                if (g < static_cast<int>(G)) acc[g] = __fadd_rn(acc[g], __fmul_rn(es[g][i], vv));
        }
    }
#pragma unroll
    for (int g = 0; g < 8; ++g)
        if (g < static_cast<int>(G)) sc.part[((static_cast<std::size_t>(r) * d.heads + kvh * G + g) * mc + chunk) * D + t] = acc[g];
}

// Phase D: block (head, row), thread = dim. out = (sum over chunks ascending of the partials) / (sum of chunk denominators
// ascending), times sigmoid(gate).
__global__ void __launch_bounds__(256) attn_combine_kernel(AttentionDims d, const float* gate, GroupedView gv, const std::uint32_t* pos0p,
                                                           float* out, Scratch sc, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const std::uint32_t len = (rm.pos ? rm.pos[r] : *pos0p + r) + 1, nch = (len + CH - 1) / CH, mc = attn_max_chunks(d);
    const std::size_t rh = static_cast<std::size_t>(r) * d.heads + h;
    float denom = 0.0f, acc = 0.0f;
    for (std::uint32_t c = 0; c < nch; ++c) {
        denom = __fadd_rn(denom, sc.dpart[rh * mc + c]);
        acc = __fadd_rn(acc, sc.part[(rh * mc + c) * D + t]);
    }
    float o = __fdiv_rn(acc, denom);
    if (gate != nullptr) o = __fmul_rn(o, cc_sigmoid(gate[static_cast<std::size_t>(r) * gv.rs() + static_cast<std::size_t>(h) * gv.gs() + t]));
    out[rh * D + t] = o;
}
// main's attention_decode launch sequence (shape checks dropped: the bench shape is valid).
cudaError_t decode(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                   const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, const std::uint32_t* pos0, std::uint32_t rows,
                   float* out, float* scratch) {
    const Scratch sc = carve(scratch, d, rows);
    const std::uint32_t mc = attn_max_chunks(d);
    const RowMap rm{};
    if (auto e = launch_pdl(attn_scores_kernel, dim3(mc, d.kv_heads, rows), dim3(256), nullptr, d, q, qv, k_cache, pos0, rows, sc, rm);
        e != cudaSuccess)
        return e;
    if (auto e = launch_pdl(attn_values_kernel, dim3(mc, d.kv_heads, rows), dim3(256), nullptr, d, v_cache, pos0, rows, sc, rm); e != cudaSuccess)
        return e;
    return launch_pdl(attn_combine_kernel, dim3(d.heads, rows), dim3(256), nullptr, d, gate, gv, pos0, out, sc, rm);
}
}  // namespace attn_main

// Multi-chunk decode attention (4 chunks, G = 6 as the 27B dense model; G = 8 and 4: the other compiled group widths): positions
// crossing the load batches and the 256-position chunks, rows 1 / 8 / 16. The library scratch must equal the frozen v0 schedule bit for bit.
int test_attention_schedule(std::uint32_t heads = 24, std::uint32_t kv_heads = 4) {
    cc::AttentionDims d; d.heads = heads; d.kv_heads = kv_heads; d.head_dim = 256; d.scale = 1.0f / 16.0f; d.max_positions = 1024;
    const std::uint32_t P = d.max_positions, KV = d.kv_heads * 256u, RMAX = 16, mc = cc::attn_max_chunks(d);
    std::vector<float> kf(static_cast<std::size_t>(P) * KV), vf(kf.size());
    for (auto& e : kf) e = rnd(1.0f);
    for (auto& e : vf) e = rnd(1.0f);
    auto *dk = dev(kf), *dv = dev(vf);
    auto *kc = devz<__nv_bfloat16>(static_cast<std::size_t>(P) * KV), *vc = devz<__nv_bfloat16>(static_cast<std::size_t>(P) * KV);
    for (std::uint32_t p = 0; p < P; p += 16) CK(cc::kv_store(d, dk + static_cast<std::size_t>(p) * KV, dv + static_cast<std::size_t>(p) * KV, kc, vc, dpos(p), 16, nullptr));
    std::vector<float> q(static_cast<std::size_t>(RMAX) * d.heads * 256), gt(q.size());
    for (auto& e : q) e = rnd(2.0f);
    for (auto& e : gt) e = rnd(2.0f);
    const std::size_t nsc = cc::attention_scratch_floats(d, RMAX);
    auto *dq = dev(q), *dgt = dev(gt), *dout = devz<float>(q.size()), *sa = devz<float>(nsc), *sb = devz<float>(nsc);
    const struct { std::uint32_t pos0, rows; } cases[] = {{0, 1},   {6, 1},    {7, 8},   {56, 9},   {63, 1},    {248, 16}, {255, 1},  {250, 12},
                                                           {256, 8}, {300, 1}, {500, 16}, {511, 8},   {700, 16}, {1000, 8}, {1023, 1}, {1018, 4},
                                                           {1012, 12}};
    // Row walks (attention_row_walk.hpp): every mode must give the v0 scratch bit for bit; `all` forces the longest walks
    // at this short length (the automatic plan keeps one row per block here).
    const cc::DecodeRowShare modes[] = {cc::DecodeRowShare::off, cc::DecodeRowShare::automatic, cc::DecodeRowShare::all};
    for (const auto& cs : cases) {
        const std::uint32_t R = cs.rows;
        const cc::GroupedView qv{R, d.heads, 256, 0, 0};
        auto* p0 = dpos(cs.pos0);
        // same carve as the library: scores [R][heads][P] | cmax [R][heads][mc] | dpart [R][heads][mc] | part [R][heads][mc][256]
        const std::size_t rh = static_cast<std::size_t>(R) * d.heads;
        const std::size_t used = rh * P + 2 * rh * mc + rh * mc * 256;
        CK(cudaMemset(sb, 0, nsc * 4));
        float *bs = sb, *bm = bs + rh * P, *bd = bm + rh * mc, *bp = bd + rh * mc;
        attn_v0::scores<<<dim3(mc, d.kv_heads, R), 256>>>(d, dq, qv, kc, p0, bs, bm);
        attn_v0::values<<<dim3(mc, d.kv_heads, R), 256>>>(d, vc, p0, bs, bm, bd, bp);
        CK(cudaDeviceSynchronize());
        const auto ref = get(sb, used);
        for (const auto mode : modes) {
            cc::set_decode_row_share(mode);
            CK(cudaMemset(sa, 0, nsc * 4));
            CK(cc::attention_decode(d, dq, qv, dgt, qv, kc, vc, p0, R, dout, sa, nullptr));
            CK(cudaDeviceSynchronize());
            CHECK(bits_equal(get(sa, used), ref), "attention schedule G %u pos0 %u rows %u mode %d: scratch differs from v0", heads / kv_heads, cs.pos0, R,
                  static_cast<int>(mode));
        }
        cudaFree(p0);
    }
    // Two cache slots, explicit positions (RowMap): walks never cross a slot; walks of `all` == one row per block (off).
    {
        const std::uint32_t R = 12, split = 7;
        const std::size_t one = static_cast<std::size_t>(P) * KV;
        auto *kc2 = devz<__nv_bfloat16>(2 * one), *vc2 = devz<__nv_bfloat16>(2 * one);
        // slot 0 = the cache above, slot 1 = the same with K and V swapped
        CK(cudaMemcpy(kc2, kc, one * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(vc2, vc, one * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(kc2 + one, vc, one * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(vc2 + one, kc, one * 2, cudaMemcpyDeviceToDevice));
        std::vector<std::uint32_t> pos(R), slot(R);
        for (std::uint32_t r = 0; r < R; ++r) {
            pos[r] = r < split ? 500 + r : 760 + (r - split);
            slot[r] = r < split ? 0u : 1u;
        }
        auto *dp = dev(pos), *ds = dev(slot);
        const cc::RowMap rm{dp, ds, one};
        const cc::GroupedView qv{R, d.heads, 256, 0, 0};
        const std::size_t rh = static_cast<std::size_t>(R) * d.heads, used = rh * P + 2 * rh * mc + rh * mc * 256;
        std::vector<float> ref;
        for (const auto mode : modes) {
            cc::set_decode_row_share(mode);
            CK(cudaMemset(sa, 0, nsc * 4));
            CK(cc::attention_decode(d, dq, qv, dgt, qv, kc2, vc2, nullptr, R, dout, sa, nullptr, rm));
            CK(cudaDeviceSynchronize());
            auto got = get(sa, used);
            if (mode == cc::DecodeRowShare::off) ref = std::move(got);
            else CHECK(bits_equal(got, ref), "attention slots mode %d: scratch differs from one row per block", static_cast<int>(mode));
        }
        cudaFree(kc2); cudaFree(vc2); cudaFree(dp); cudaFree(ds);
    }
    cc::set_decode_row_share(cc::DecodeRowShare::off);
    cudaFree(dk); cudaFree(dv); cudaFree(kc); cudaFree(vc); cudaFree(dq); cudaFree(dgt); cudaFree(dout); cudaFree(sa); cudaFree(sb);
    return 0;
}

// Long-context row walks on synthetic BF16 K/V (no model): lengths 1K / 33K / 131K, rows 1 / 4 / 12 / 16, windows crossing
// a 256-position chunk boundary. Library scratch in every row-walk mode == the frozen v0 schedule, compared on the device.
__global__ void fill_bf16_hash(__nv_bfloat16* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        p[i] = __float2bfloat16_rn((static_cast<float>(h & 0xffffu) - 32768.0f) / 32768.0f);
    }
}
__global__ void count_diff(const float* a, const float* b, std::size_t n, unsigned long long* out) {
    unsigned long long c = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        c += __float_as_uint(a[i]) != __float_as_uint(b[i]) ? 1ull : 0ull;
    if (c) atomicAdd(out, c);
}
static const std::uint32_t kLongP = 131328;  // 513 chunks
struct LongAttn {
    cc::AttentionDims d;
    __nv_bfloat16 *kc = nullptr, *vc = nullptr;
    float *q = nullptr, *gate = nullptr, *out = nullptr, *sa = nullptr, *sb = nullptr;
    std::size_t nsc = 0;
    bool init(std::uint32_t rmax) {
        d.heads = 24; d.kv_heads = 4; d.head_dim = 256; d.scale = 1.0f / 16.0f; d.max_positions = kLongP;
        const std::size_t kv = static_cast<std::size_t>(kLongP) * d.kv_heads * 256;
        kc = devz<__nv_bfloat16>(kv); vc = devz<__nv_bfloat16>(kv);
        fill_bf16_hash<<<1024, 256>>>(kc, kv, 0x1234u);
        fill_bf16_hash<<<1024, 256>>>(vc, kv, 0xabcdu);
        std::vector<float> hq(static_cast<std::size_t>(rmax) * d.heads * 256);
        for (auto& e : hq) e = rnd(2.0f);
        q = dev(hq);
        for (auto& e : hq) e = rnd(2.0f);
        gate = dev(hq);
        out = devz<float>(hq.size());
        nsc = cc::attention_scratch_floats(d, rmax);
        sa = devz<float>(nsc); sb = devz<float>(nsc);
        return cudaDeviceSynchronize() == cudaSuccess && kc && vc && sa && sb;
    }
    ~LongAttn() { cudaFree(kc); cudaFree(vc); cudaFree(q); cudaFree(gate); cudaFree(out); cudaFree(sa); cudaFree(sb); }
};

int test_attention_row_share() {
    LongAttn a;
    if (!a.init(16)) { CHECK(false, "row share: allocation failed"); return 2; }
    const cc::AttentionDims& d = a.d;
    const std::uint32_t mc = cc::attn_max_chunks(d);
    unsigned long long* dc = devz<unsigned long long>(1);
    const struct { std::uint32_t pos0, rows; } cases[] = {{1020, 1},  {1020, 4},  {1018, 12},  {1015, 16},  {33020, 1},   {33022, 4},
                                                           {33020, 12}, {33010, 16}, {131070, 1}, {131070, 4}, {131064, 12}, {131064, 16},
                                                           // walk tails of every width: rows 2 / 3 / 5 / 7 / 13
                                                           {33022, 2},  {33021, 3},  {33019, 5},  {33017, 7},  {33011, 13},  {131069, 5},
                                                           {131063, 13}};
    const cc::DecodeRowShare modes[] = {cc::DecodeRowShare::off, cc::DecodeRowShare::automatic, cc::DecodeRowShare::all};
    for (const auto& cs : cases) {
        const std::uint32_t R = cs.rows;
        const cc::GroupedView qv{R, d.heads, 256, 0, 0};
        auto* p0 = dpos(cs.pos0);
        const std::size_t rh = static_cast<std::size_t>(R) * d.heads, used = rh * kLongP + 2 * rh * mc + rh * mc * 256;
        CK(cudaMemset(a.sb, 0, a.nsc * 4));
        float *bs = a.sb, *bm = bs + rh * kLongP, *bd = bm + rh * mc, *bp = bd + rh * mc;
        attn_v0::scores<<<dim3(mc, d.kv_heads, R), 256>>>(d, a.q, qv, a.kc, p0, bs, bm);
        attn_v0::values<<<dim3(mc, d.kv_heads, R), 256>>>(d, a.vc, p0, bs, bm, bd, bp);
        for (const auto mode : modes) {
            cc::set_decode_row_share(mode);
            CK(cudaMemset(a.sa, 0, a.nsc * 4));
            CK(cudaMemset(dc, 0, 8));
            CK(cc::attention_decode(d, a.q, qv, a.gate, qv, a.kc, a.vc, p0, R, a.out, a.sa, nullptr));
            count_diff<<<1024, 256>>>(a.sa, a.sb, used, dc);
            CK(cudaDeviceSynchronize());
            const auto nd = get(dc, 1)[0];
            CHECK(nd == 0, "row share len %u rows %u mode %d: %llu scratch floats differ from v0", cs.pos0 + R, R, static_cast<int>(mode), nd);
        }
        cudaFree(p0);
    }
    cc::set_decode_row_share(cc::DecodeRowShare::off);
    cudaFree(dc);
    std::printf("attention row share: %zu cases x 3 modes compared with v0\n", std::size(cases));
    return 0;
}

// e4m3 shadow of LongAttn's caches (kv_shadow_e4m3 layout), zeroed once like the engine's.
struct Shadow8 {
    cc::KvShadowE4m3 s;
    std::uint8_t* base = nullptr;
    std::size_t bytes = 0;
    bool init(const cc::AttentionDims& d) {
        bytes = cc::kv_shadow_e4m3_bytes(d);
        base = devz<std::uint8_t>(bytes);
        const std::size_t p8 = cc::attn8_capacity(d), kvd = static_cast<std::size_t>(d.kv_heads) * p8 * d.head_dim;
        s.k8 = base;
        s.v8t = base + kvd;
        s.ks = reinterpret_cast<float*>(base + 2 * kvd);
        s.vs = s.ks + static_cast<std::size_t>(d.kv_heads) * p8;
        return base != nullptr;
    }
    ~Shadow8() { cudaFree(base); }
};

// FAST8 decode attention (attention_decode8): (1) R verify rows give, per row, the same bits as a one-row call at that row's
// position, even when the shadow keys past the row hold different (stale) data; (2) kv_shadow_e4m3_rows writes the same bytes
// as kv_shadow_e4m3; (3) close to the exact BF16 attention_decode (relative L2, loose bound: catches indexing errors only).
int test_attention_decode8() {
    LongAttn a;
    Shadow8 sh, sh2;
    if (!a.init(32) || !sh.init(a.d) || !sh2.init(a.d)) { CHECK(false, "decode8: allocation failed"); return 2; }
    const cc::AttentionDims& d = a.d;
    const std::size_t HD = static_cast<std::size_t>(d.heads) * 256;
    float* sc = devz<float>(cc::attention_decode8_scratch_floats(d, 32));
    float *outR = devz<float>(32 * HD), *out1 = devz<float>(HD);
    unsigned long long* dc = devz<unsigned long long>(1);
    const struct { std::uint32_t pos0, rows; } cases[] = {{100, 1}, {2040, 16}, {4090, 12}, {6130, 32}, {33020, 12}, {131060, 12}};
    double worst_rel = 0.0;
    for (const auto& cs : cases) {
        const std::uint32_t R = cs.rows, len = cs.pos0 + R;
        const cc::GroupedView qv{R, d.heads, 256, 0, 0};
        auto* p0 = dpos(cs.pos0);
        CK(cc::kv_shadow_e4m3(d, a.kc, a.vc, sh.s, 0, len, nullptr));
        CK(cc::attention_decode8(d, a.q, qv, a.gate, qv, sh.s, p0, R, outR, sc, nullptr));
        // (3) vs exact BF16 attention
        CK(cc::attention_decode(d, a.q, qv, a.gate, qv, a.kc, a.vc, p0, R, a.out, a.sa, nullptr));
        CK(cudaDeviceSynchronize());
        const auto e = get(a.out, R * HD), f = get(outR, R * HD);
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < e.size(); ++i) {
            num += (static_cast<double>(f[i]) - e[i]) * (static_cast<double>(f[i]) - e[i]);
            den += static_cast<double>(e[i]) * e[i];
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        worst_rel = std::max(worst_rel, rel);
        CHECK(rel < 0.25, "decode8 len %u rows %u: relative L2 vs exact %.4f", len, R, rel);
        // (1) row r alone, with the keys past it re-quantised from the swapped caches (stale data a serial step would see)
        for (std::uint32_t r = 0; r < R; ++r) {
            const std::uint32_t pr = cs.pos0 + r;
            if (pr + 1 < len) CK(cc::kv_shadow_e4m3(d, a.vc, a.kc, sh.s, pr + 1, len - pr - 1, nullptr));
            auto* pp = dpos(pr);
            const cc::GroupedView q1{1, d.heads, 256, 0, 0};
            CK(cc::attention_decode8(d, a.q + r * HD, q1, a.gate + r * HD, q1, sh.s, pp, 1, out1, sc, nullptr));
            CK(cudaMemset(dc, 0, 8));
            count_diff<<<64, 256>>>(out1, outR + r * HD, HD, dc);
            CK(cudaDeviceSynchronize());
            const auto nd = get(dc, 1)[0];
            CHECK(nd == 0, "decode8 len %u rows %u row %u: %llu floats differ from the one-row call", len, R, r, nd);
            if (pr + 1 < len) CK(cc::kv_shadow_e4m3(d, a.kc, a.vc, sh.s, pr + 1, len - pr - 1, nullptr));
            cudaFree(pp);
        }
        cudaFree(p0);
    }
    // (2) device-position quantiser == host-range quantiser (both shadows start zeroed)
    {
        CK(cudaMemset(sh.base, 0, sh.bytes));
        auto* p0 = dpos(4090);
        CK(cc::kv_shadow_e4m3(d, a.kc, a.vc, sh.s, 4090, 12, nullptr));
        CK(cc::kv_shadow_e4m3_rows(d, a.kc, a.vc, sh2.s, p0, 12, nullptr));
        CK(cudaMemset(dc, 0, 8));
        count_diff<<<1024, 256>>>(reinterpret_cast<const float*>(sh.base), reinterpret_cast<const float*>(sh2.base), sh.bytes / 4, dc);
        CK(cudaDeviceSynchronize());
        const auto nd = get(dc, 1)[0];
        CHECK(nd == 0, "decode8: kv_shadow_e4m3_rows differs from kv_shadow_e4m3 in %llu words", nd);
        cudaFree(p0);
    }
    cudaFree(sc); cudaFree(outR); cudaFree(out1); cudaFree(dc);
    std::printf("attention decode8: %zu cases, rows == one-row calls bitwise, worst relative L2 vs exact %.4f\n", std::size(cases), worst_rel);
    return 0;
}

// Profiler target (--attn-ncu): exactly one attention_decode call per arm (off, then automatic) at 131,072 positions x
// 12 rows on synthetic K/V, so a profiler filtered on the attention kernels sees 3 launches per arm.
static int attn_ncu() {
    LongAttn a;
    if (!a.init(16)) { std::printf("attn-ncu: allocation failed\n"); return 2; }
    const std::uint32_t len = 131072, R = 12;
    const cc::GroupedView qv{R, a.d.heads, 256, 0, 0};
    auto* p0 = dpos(len - R);
    for (const auto mode : {cc::DecodeRowShare::off, cc::DecodeRowShare::automatic}) {
        cc::set_decode_row_share(mode);
        CK(cc::attention_decode(a.d, a.q, qv, a.gate, qv, a.kc, a.vc, p0, R, a.out, a.sa, nullptr));
        CK(cudaDeviceSynchronize());
    }
    cudaFree(p0);
    cc::set_decode_row_share(cc::DecodeRowShare::off);
    std::printf("attn-ncu: done (len %u rows %u, arms off then automatic)\n", len, R);
    return 0;
}

// No-model kernel bench (--attn-bench): ms per attention_decode call (one layer: 24 q / 4 kv heads, head_dim 256) on
// synthetic K/V; arms main (origin/main's kernels, frozen copy attn_main), off (this branch's kernels, one row per
// block, chunk fastest) and auto (shared plan), alternated, median of 5 timed repeats of 10 calls after one warm-up
// repeat. off/main = the kernel rewrite alone, auto/off = the row share alone, auto/main = the branch against main.
static int attn_bench() {
    LongAttn a;
    if (!a.init(16)) { std::printf("attn-bench: allocation failed\n"); return 2; }
    int dev = 0, sms = 0, clk = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    std::printf("attn-bench device sms %d max_clock_mhz %d P %u\n", sms, clk / 1000, kLongP);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const std::uint32_t lens[] = {1024, 8192, 32768, 131072}, rowsv[] = {1, 2, 3, 4, 6, 8, 12, 16};
    const cc::DecodeRowShare arms[] = {cc::DecodeRowShare::off, cc::DecodeRowShare::off, cc::DecodeRowShare::automatic};
    const char* names[] = {"main", "off", "auto"};
    constexpr int kArms = 3;
    for (const auto len : lens) {
        for (const auto R : rowsv) {
            const cc::GroupedView qv{R, a.d.heads, 256, 0, 0};
            auto* p0 = dpos(len - R);  // the last row attends `len` positions
            std::vector<float> ms[kArms];
            for (int rep = 0; rep < 6; ++rep) {
                for (int arm = 0; arm < kArms; ++arm) {
                    cc::set_decode_row_share(arms[arm]);
                    const auto call = [&] {
                        return arm == 0 ? attn_main::decode(a.d, a.q, qv, a.gate, qv, a.kc, a.vc, p0, R, a.out, a.sa)
                                        : cc::attention_decode(a.d, a.q, qv, a.gate, qv, a.kc, a.vc, p0, R, a.out, a.sa, nullptr);
                    };
                    CK(call());
                    CK(cudaEventRecord(e0));
                    for (int it = 0; it < 10; ++it) CK(call());
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    float t = 0.0f;
                    cudaEventElapsedTime(&t, e0, e1);
                    if (rep > 0) ms[arm].push_back(t / 10.0f);  // rep 0 = warm-up
                }
            }
            double med[kArms];
            for (int arm = 0; arm < kArms; ++arm) {
                std::sort(ms[arm].begin(), ms[arm].end());
                med[arm] = ms[arm][ms[arm].size() / 2];
                std::printf("attn-bench len %u rows %u arm %s ms %.4f (min %.4f max %.4f)\n", len, R, names[arm], med[arm], ms[arm].front(), ms[arm].back());
            }
            std::printf("attn-bench len %u rows %u auto/off %.3f off/main %.3f auto/main %.3f\n", len, R, med[2] / med[1],
                        med[1] / med[0], med[2] / med[0]);
            cudaFree(p0);
        }
    }
    cc::set_decode_row_share(cc::DecodeRowShare::off);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    std::printf("attn-bench: done\n");
    return 0;
}

// No-model kernel bench (--attn8-bench): ms per call, one layer (24 q / 4 kv heads, head_dim 256), synthetic K/V. Arms: exact
// (attention_decode, automatic row share) and kv8 (attention_decode8 over the e4m3 shadow; shadow filled once, not timed),
// alternated, median of 5 timed repeats of 10 calls after one warm-up repeat.
static int attn8_bench() {
    LongAttn a;
    Shadow8 sh;
    if (!a.init(16) || !sh.init(a.d)) { std::printf("attn8-bench: allocation failed\n"); return 2; }
    float* sc = devz<float>(cc::attention_decode8_scratch_floats(a.d, 16));
    CK(cc::kv_shadow_e4m3(a.d, a.kc, a.vc, sh.s, 0, kLongP, nullptr));
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    std::printf("attn8-bench device sms %d P %u chunk %u\n", sms, kLongP, cc::kAttn8Chunk);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const std::uint32_t lens[] = {1024, 8192, 28672, 65536, 131072}, rowsv[] = {1, 4, 8, 12};
    const char* names[] = {"exact", "kv8"};
    cc::set_decode_row_share(cc::DecodeRowShare::automatic);
    for (const auto len : lens) {
        for (const auto R : rowsv) {
            const cc::GroupedView qv{R, a.d.heads, 256, 0, 0};
            auto* p0 = dpos(len - R);
            // the shared per-layer entry: exact = BF16 row walk; kv8 = step-row shadow store + e4m3 walk (the engine's real cost)
            auto call = [&](int arm) -> int {
                CK(cc::attention_decode_kv(a.d, a.q, qv, a.gate, qv, a.kc, a.vc, arm == 0 ? nullptr : &sh.s, p0, R, a.out,
                                           arm == 0 ? a.sa : sc, nullptr));
                return 0;
            };
            std::vector<float> ms[2];
            for (int rep = 0; rep < 6; ++rep) {
                for (int arm = 0; arm < 2; ++arm) {
                    if (call(arm) != 0) return 1;
                    CK(cudaEventRecord(e0));
                    for (int it = 0; it < 10; ++it)
                        if (call(arm) != 0) return 1;
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    float t = 0.0f;
                    cudaEventElapsedTime(&t, e0, e1);
                    if (rep > 0) ms[arm].push_back(t / 10.0f);
                }
            }
            double med[2];
            for (int arm = 0; arm < 2; ++arm) {
                std::sort(ms[arm].begin(), ms[arm].end());
                med[arm] = ms[arm][ms[arm].size() / 2];
                std::printf("attn8-bench len %u rows %u arm %s ms %.4f (min %.4f max %.4f)\n", len, R, names[arm], med[arm], ms[arm].front(), ms[arm].back());
            }
            std::printf("attn8-bench len %u rows %u kv8/exact %.3f\n", len, R, med[1] / med[0]);
            cudaFree(p0);
        }
    }
    cc::set_decode_row_share(cc::DecodeRowShare::off);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(sc);
    std::printf("attn8-bench: done\n");
    return 0;
}

int test_gdn() {
    cc::GdnDims d; d.key_heads = 4; d.value_heads = 12; d.key_dim = 128; d.value_dim = 128;
    const std::uint32_t C = d.conv_channels(), R = 6, hv = d.value_heads, K = 128;
    std::vector<float> convw(C * 4), a(hv), dt(hv), ng(128), qkv(R * C), z(R * hv * 128), ar(R * hv), br(R * hv);
    for (auto& e : convw) e = rnd(0.5f);
    for (auto& e : a) e = -std::exp(rnd(1.0f));
    for (auto& e : dt) e = rnd(1.0f);
    for (auto& e : ng) e = 0.5f + rnd(0.3f);
    for (auto& e : qkv) e = rnd(1.0f);
    for (auto& e : z) e = rnd(1.0f);
    for (auto& e : ar) e = rnd(1.0f);
    for (auto& e : br) e = rnd(1.0f);
    // two initial states: zero conv state, random delta state
    std::vector<float> s0(static_cast<std::size_t>(hv) * K * K), cs0(C * 3);
    for (auto& e : s0) e = rnd(0.2f);
    for (auto& e : cs0) e = rnd(0.5f);
    auto *dcw = dev(convw), *da = dev(a), *ddt = dev(dt), *dng = dev(ng), *dz = dev(z), *dar = dev(ar), *dbr = dev(br);
    cc::GdnLayer L{dcw, da, ddt, dng};
    auto run = [&](std::uint32_t chunk, std::vector<float>& outv, std::vector<float>& sout) {
        std::vector<float> two(2 * s0.size(), 0.0f);
        std::copy(s0.begin(), s0.end(), two.begin());
        auto *draw = dev(qkv), *dqc = devz<float>(R * C), *dout = devz<float>(R * hv * 128), *dst = dev(two), *dcs = dev(cs0);
        auto* dpar = devz<std::uint32_t>(1);
        for (std::uint32_t r = 0; r < R; r += chunk) {
            const std::uint32_t n = std::min(chunk, R - r);
            cc::GdnWindow w{draw + r * C, dqc + r * C, dar + r * hv, dbr + r * hv};
            if (cc::gdn_window(d, L, {dcs, dst}, w, dz + r * hv * 128, dout + r * hv * 128, n, 1e-6f, dpar, nullptr) != cudaSuccess) std::printf("gdn launch fail\n");
            cc::GdnCommitLayer cl{L, {dcs, dst}, w};
            if (cc::gdn_commit(d, &cl, 1, n, n, dpar, nullptr) != cudaSuccess) std::printf("gdn commit fail\n");
        }
        std::uint32_t par = 0;
        cudaMemcpy(&par, dpar, 4, cudaMemcpyDeviceToHost);
        outv = get(dout, R * hv * 128);
        auto all = get(dst, two.size());
        sout.assign(all.begin() + par * s0.size(), all.begin() + (par + 1) * s0.size());
        cudaFree(draw); cudaFree(dqc); cudaFree(dout); cudaFree(dst); cudaFree(dcs); cudaFree(dpar);
    };
    std::vector<float> o6, s6, o1, s1, o3, s3;
    run(6, o6, s6); run(1, o1, s1); run(3, o3, s3);
    CHECK(bits_equal(o6, o1) && bits_equal(s6, s1), "gdn rows=6 vs rows=1 not bitwise");
    CHECK(bits_equal(o6, o3) && bits_equal(s6, s3), "gdn rows=6 vs rows=3 not bitwise");
    {  // prefill entry (one call over all rows, delta lands on the shadow side, then flip): bitwise the window path
        std::vector<float> two(2 * s0.size(), 0.0f);
        std::copy(s0.begin(), s0.end(), two.begin());
        auto *draw = dev(qkv), *dqc = devz<float>(R * C), *dout = devz<float>(R * hv * 128), *dst = dev(two), *dcs = dev(cs0);
        auto* dpar = devz<std::uint32_t>(1);
        cc::GdnWindow w{draw, dqc, dar, dbr};
        CK(cc::gdn_prefill_layer(d, L, {dcs, dst}, w, dz, dout, R, 1e-6f, dpar, nullptr));
        CK(cc::gdn_prefill_flip(dpar, nullptr));
        std::uint32_t par = 0;
        CK(cudaMemcpy(&par, dpar, 4, cudaMemcpyDeviceToHost));
        const auto po = get(dout, R * hv * 128);
        const auto all = get(dst, two.size());
        const std::vector<float> ps(all.begin() + par * s0.size(), all.begin() + (par + 1) * s0.size());
        cudaFree(draw); cudaFree(dqc); cudaFree(dout); cudaFree(dst); cudaFree(dcs); cudaFree(dpar);
        CHECK(bits_equal(po, o6) && bits_equal(ps, s6), "gdn prefill entry vs windows not bitwise");
    }
    {  // build-to-build identity: compare this line between two binaries (same fixture seed) for schedule-only kernel changes
        auto fnv = [](const std::vector<float>& v) {
            std::uint64_t h = 1469598103934665603ull;
            for (float f : v) { std::uint32_t u; std::memcpy(&u, &f, 4); for (int i = 0; i < 4; ++i) { h ^= (u >> (8 * i)) & 0xffu; h *= 1099511628211ull; } }
            return static_cast<unsigned long long>(h);
        };
        std::printf("gdn digest out %016llx state %016llx\n", fnv(o6), fnv(s6));
    }
    // float64 reference
    std::vector<double> cs(cs0.begin(), cs0.end()), S(s0.begin(), s0.end()), ref(R * hv * 128);
    auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
    for (std::uint32_t r = 0; r < R; ++r) {
        std::vector<double> mix(C);
        for (std::uint32_t c = 0; c < C; ++c) {
            const double in3[4] = {cs[c * 3], cs[c * 3 + 1], cs[c * 3 + 2], qkv[r * C + c]};
            double s = 0; for (int j = 0; j < 4; ++j) s += convw[c * 4 + j] * in3[j];
            mix[c] = silu(s); cs[c * 3] = in3[1]; cs[c * 3 + 1] = in3[2]; cs[c * 3 + 2] = in3[3];
        }
        std::vector<double> qn(4 * K), kn(4 * K);
        for (int h = 0; h < 4; ++h) {
            double sq = 0, sk = 0; for (int i = 0; i < K; ++i) { sq += mix[h * K + i] * mix[h * K + i]; sk += mix[512 + h * K + i] * mix[512 + h * K + i]; }
            for (int i = 0; i < K; ++i) { qn[h * K + i] = mix[h * K + i] / std::sqrt(sq + 1e-6) / std::sqrt(128.0); kn[h * K + i] = mix[512 + h * K + i] / std::sqrt(sk + 1e-6); }
        }
        for (std::uint32_t h = 0; h < hv; ++h) {
            const double beta = 1.0 / (1.0 + std::exp(-(double)br[r * hv + h]));
            const double xin = ar[r * hv + h] + dt[h], sp = xin > 20 ? xin : std::log1p(std::exp(xin));
            const double decay = std::exp(a[h] * sp);
            const int kh = h % 4;
            std::vector<double> od(K);
            for (int j = 0; j < K; ++j) {
                double kv = 0;
                for (int k = 0; k < K; ++k) { S[(h * K + k) * K + j] *= decay; kv += S[(h * K + k) * K + j] * kn[kh * K + k]; }
                const double dl = (mix[1024 + h * K + j] - kv) * beta;
                double acc = 0;
                for (int k = 0; k < K; ++k) { S[(h * K + k) * K + j] += kn[kh * K + k] * dl; acc += S[(h * K + k) * K + j] * qn[kh * K + k]; }
                od[j] = acc;
            }
            double ss = 0; for (double e : od) ss += e * e;
            const double inv = 1.0 / std::sqrt(ss / K + 1e-6);
            for (int j = 0; j < K; ++j) ref[(r * hv + h) * K + j] = od[j] * inv * ng[j] * silu(z[(r * hv + h) * K + j]);
        }
    }
    CHECK(maxrel(o6, ref) < 1e-4, "gdn relerr %g", maxrel(o6, ref));
    std::vector<double> sref = S;
    CHECK(maxrel(s6, sref) < 1e-4, "gdn state relerr %g", maxrel(s6, sref));
    return 0;
}

__global__ void nvfp4_test_epilogue(const float* gate, const float* up, float* y, std::uint32_t out) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out) y[i] = __fmul_rn(cc::cc_silu(gate[i]), up[i]);
}

// GDN conv rows across the row-batch boundary (R = 20 > 16 rows per batch): windows of 20 / 1 / 7 rows and the prefill entry
// give bitwise the same conv output and delta output. Uses only the public API, so the same file builds against the previous
// serial conv loop: the digest line must be equal between that binary and this one (schedule-only change).
int test_conv_row_batch() {
    cc::GdnDims d; d.key_heads = 2; d.value_heads = 6; d.key_dim = 128; d.value_dim = 128;
    const std::uint32_t C = d.conv_channels(), R = 20, hv = d.value_heads, K = 128;
    std::vector<float> convw(C * 4), a(hv), dt(hv), ng(128), qkv(R * C), z(R * hv * 128), ar(R * hv), br(R * hv);
    for (auto& e : convw) e = rnd(0.5f);
    for (auto& e : a) e = -std::exp(rnd(1.0f));
    for (auto& e : dt) e = rnd(1.0f);
    for (auto& e : ng) e = 0.5f + rnd(0.3f);
    for (auto& e : qkv) e = rnd(1.0f);
    for (auto& e : z) e = rnd(1.0f);
    for (auto& e : ar) e = rnd(1.0f);
    for (auto& e : br) e = rnd(1.0f);
    std::vector<float> s0(static_cast<std::size_t>(hv) * K * K), cs0(C * 3);
    for (auto& e : s0) e = rnd(0.2f);
    for (auto& e : cs0) e = rnd(0.5f);
    auto *dcw = dev(convw), *da = dev(a), *ddt = dev(dt), *dng = dev(ng), *dz = dev(z), *dar = dev(ar), *dbr = dev(br);
    cc::GdnLayer L{dcw, da, ddt, dng};
    // chunk == 0: prefill entry over all R rows. Returns 0, or 2 on a CUDA error (CK).
    auto run = [&](std::uint32_t chunk, std::vector<float>& qcv, std::vector<float>& outv) -> int {
        std::vector<float> two(2 * s0.size(), 0.0f);
        std::copy(s0.begin(), s0.end(), two.begin());
        auto *draw = dev(qkv), *dqc = devz<float>(R * C), *dout = devz<float>(R * hv * 128), *dst = dev(two), *dcs = dev(cs0);
        auto* dpar = devz<std::uint32_t>(1);
        if (chunk == 0) {
            cc::GdnWindow w{draw, dqc, dar, dbr};
            CK(cc::gdn_prefill_layer(d, L, {dcs, dst}, w, dz, dout, R, 1e-6f, dpar, nullptr));
        } else {
            for (std::uint32_t r = 0; r < R; r += chunk) {
                const std::uint32_t n = std::min(chunk, R - r);
                cc::GdnWindow w{draw + r * C, dqc + r * C, dar + r * hv, dbr + r * hv};
                CK(cc::gdn_window(d, L, {dcs, dst}, w, dz + r * hv * 128, dout + r * hv * 128, n, 1e-6f, dpar, nullptr));
                cc::GdnCommitLayer cl{L, {dcs, dst}, w};
                CK(cc::gdn_commit(d, &cl, 1, n, n, dpar, nullptr));
            }
        }
        CK(cudaDeviceSynchronize());
        qcv = get(dqc, R * C);
        outv = get(dout, R * hv * 128);
        cudaFree(draw); cudaFree(dqc); cudaFree(dout); cudaFree(dst); cudaFree(dcs); cudaFree(dpar);
        return 0;
    };
    std::vector<float> q20, o20, q1, o1, q7, o7, qp, op;
    if (run(20, q20, o20) | run(1, q1, o1) | run(7, q7, o7) | run(0, qp, op)) return 2;
    CHECK(bits_equal(q20, q1) && bits_equal(o20, o1), "conv rows=20 vs rows=1 not bitwise");
    CHECK(bits_equal(q20, q7) && bits_equal(o20, o7), "conv rows=20 vs rows=7 not bitwise");
    CHECK(bits_equal(q20, qp) && bits_equal(o20, op), "conv prefill vs windows not bitwise");
    auto fnv = [](const std::vector<float>& v) {
        std::uint64_t h = 1469598103934665603ull;
        for (float f : v) { std::uint32_t u; std::memcpy(&u, &f, 4); for (int i = 0; i < 4; ++i) { h ^= (u >> (8 * i)) & 0xffu; h *= 1099511628211ull; } }
        return static_cast<unsigned long long>(h);
    };
    std::printf("conv digest qc %016llx out %016llx\n", fnv(q20), fnv(o20));
    cudaFree(dcw); cudaFree(da); cudaFree(ddt); cudaFree(dng); cudaFree(dz); cudaFree(dar); cudaFree(dbr);
    {  // kernel probe at the Qwen 27B dense GDN shape (16 key / 48 value heads): 40 windows of 8 rows, then 40 of 16 rows, no
       // commit (state unchanged), for an nsys kernel trace of conv_pre_kernel; nothing is checked here.
        cc::GdnDims P; P.key_heads = 16; P.value_heads = 48; P.key_dim = 128; P.value_dim = 128;
        const std::uint32_t PC = P.conv_channels(), PR = 16, ph = P.value_heads;
        std::vector<float> pw(PC * 4), pa(ph), pdt(ph), png(128), praw(PR * PC), pz(PR * ph * 128), par_(PR * ph), pbr(PR * ph);
        std::vector<float> pst(2 * static_cast<std::size_t>(ph) * K * K), pcs(PC * 3);
        for (auto* v : {&pw, &pdt, &praw, &pz, &par_, &pbr, &pst, &pcs}) for (auto& e : *v) e = rnd(0.5f);
        for (auto& e : pa) e = -std::exp(rnd(1.0f));
        for (auto& e : png) e = 0.5f + rnd(0.3f);
        auto *dw = dev(pw), *dpa = dev(pa), *dpdt = dev(pdt), *dpng = dev(png), *dpraw = dev(praw), *dpz = dev(pz);
        auto *dpar_ = dev(par_), *dpbr = dev(pbr), *dpst = dev(pst), *dpcs = dev(pcs);
        auto *dpqc = devz<float>(PR * PC), *dpout = devz<float>(PR * ph * 128);
        auto* dpp = devz<std::uint32_t>(1);
        cc::GdnLayer PL{dw, dpa, dpdt, dpng};
        cc::GdnWindow pwin{dpraw, dpqc, dpar_, dpbr};
        // synchronised per window: no early-launched predecessor, so a conv_pre_kernel duration is its own execution
        // (under PDL a block waiting in griddepcontrol.wait would be counted as kernel time)
        for (std::uint32_t rows : {8u, 16u})
            for (int it = 0; it < 40; ++it) {
                CK(cc::gdn_window(P, PL, {dpcs, dpst}, pwin, dpz, dpout, rows, 1e-6f, dpp, nullptr));
                CK(cudaDeviceSynchronize());
            }
        for (void* p : {static_cast<void*>(dw), static_cast<void*>(dpa), static_cast<void*>(dpdt), static_cast<void*>(dpng),
                        static_cast<void*>(dpraw), static_cast<void*>(dpz), static_cast<void*>(dpar_), static_cast<void*>(dpbr),
                        static_cast<void*>(dpst), static_cast<void*>(dpcs), static_cast<void*>(dpqc), static_cast<void*>(dpout),
                        static_cast<void*>(dpp)})
            cudaFree(p);
        std::printf("conv probe: 40 windows x rows 8, 16 at 16/48 heads done\n");
    }
    return 0;
}

static int test_nvfp4_fused() {
    constexpr std::uint32_t in = 256;
    cc::nvfp4_set_split_target(0);
    for (const std::uint32_t out : {248320u, 16u}) {
        CHECK(cc::nvfp4_split_k(out, in) == (out == 248320u ? 1u : 2u), "NVFP4 split coverage out=%u", out);
        // Exact packed weights: gate = 1, up = 2, all local scales = 1 (e4m3 0x38).
        std::vector<unsigned char> gate(cc::nvfp4_bytes(out, in), 0x22), up(gate.size(), 0x44);
        for (std::size_t base = 0; base < gate.size(); base += cc::kNvfp4RecordBytes)
            for (std::size_t i = 1024; i < cc::kNvfp4RecordBytes; ++i) gate[base + i] = up[base + i] = 0x38;
        auto* dg = dev(gate);
        auto* du = dev(up);
        std::vector<float> x(in, 0.0f);
        // One split retains the small term after cancellation; two splits lose it.
        x[0] = 32768.0f; x[128] = -32768.0f; x[144] = 1.0f / 1024.0f;
        if (out == 16u) x[128] = -16384.0f;  // Nonzero split-K epilogue coverage.
        auto* dx = dev(x);
        auto* g = devz<float>(out);
        auto* u = devz<float>(out);
        auto* separate = devz<float>(out);
        auto* fused = devz<float>(out);
        const cc::DenseWeight gw{dg, cc::WeightFormat::nvfp4, out, in, 0.5f};
        const cc::DenseWeight uw{du, cc::WeightFormat::nvfp4, out, in, 0.25f};
        CK(cc::nvfp4_dense_rows(gw, dx, g, 1, nullptr, nullptr));
        CK(cc::nvfp4_dense_rows(uw, dx, u, 1, nullptr, nullptr));
        nvfp4_test_epilogue<<<(out + 255) / 256, 256>>>(g, u, separate, out);
        CK(cudaGetLastError());
        CK(cc::nvfp4_gate_up_silu_rows(gw, uw, dx, fused, 1, nullptr));
        CK(cudaDeviceSynchronize());
        CHECK(bits_equal(get(fused, out), get(separate, out)), "NVFP4 fused vs separate bits out=%u", out);
        CK(cudaFree(dg)); CK(cudaFree(du)); CK(cudaFree(dx));
        CK(cudaFree(g)); CK(cudaFree(u)); CK(cudaFree(separate)); CK(cudaFree(fused));
    }
    cc::nvfp4_release_workspace(nullptr);  // the test runs on the default stream
    return 0;
}

int main(int argc, char** argv) {
    if (cudaSetDevice(0) != cudaSuccess) return 2;
    if (argc > 1 && std::strcmp(argv[1], "--attn-bench") == 0) return attn_bench();
    if (argc > 1 && std::strcmp(argv[1], "--attn-ncu") == 0) return attn_ncu();
    if (argc > 1 && std::strcmp(argv[1], "--attn8-bench") == 0) return attn8_bench();
    if (argc > 1 && std::strcmp(argv[1], "--attn8") == 0) {  // FAST e4m3 decode attention only (rows == one-row bits), bounded ticket
        const int r = test_attention_decode8();
        std::printf(g_fail || r ? "cuda_core_test --attn8: FAILED (%d checks)\n" : "cuda_core_test --attn8: PASS\n", g_fail);
        return (g_fail || r) ? 1 : 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--attn-rows") == 0) {  // the row-walk tests only (short + long), for a bounded ticket
        const int r = test_attention_schedule() | test_attention_schedule(16, 2) | test_attention_schedule(8, 2) | test_attention_row_share();
        std::printf(g_fail || r ? "cuda_core_test --attn-rows: FAILED (%d checks)\n" : "cuda_core_test --attn-rows: PASS\n", g_fail);
        return (g_fail || r) ? 1 : 0;
    }
    int rc = test_dense() | test_norms_rope() | test_norm_batched_loads() | test_attention() | test_attention_schedule() | test_attention_schedule(16, 2) | test_attention_schedule(8, 2) |
             test_attention_row_share() | test_attention_decode8() | test_gdn() | test_conv_row_batch() | test_nvfp4_fused();
    // repeat determinism is covered by the identical-bits checks (each compares two separate launches)
    std::printf(g_fail || rc ? "cuda_core_test: FAILED (%d checks)\n" : "cuda_core_test: PASS\n", g_fail);
    return (g_fail || rc) ? 1 : 0;
}
