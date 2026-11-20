// Shared policy dense-iu4a16-f16-v1 (src/core/f16_dense_iu4.hpp):
//  1. weight decode: every FP16 scale (65,536) x every code (16) through Cols::stage from memory (a check on register
//     words is not equivalent: the compiler folds their known bits), bit for bit vs host_w16 (any NaN matches any NaN);
//  2. a GEMM with a partial row CTA, a remainder column launch and a partial last column tile: error vs a double sum
//     of the contract's x16 * w16 products (tolerance 2^-10 of sum |terms|: FP32 WMMA accumulation), columns past
//     count untouched; 16-wave CTAs give the same bits as 8-wave CTAs;
//  3. the x16 form (enqueue_x16_tiles + enqueue_x16, column-fast) gives the direct form's bits, 8 and 16 waves;
//  4. x16 tiles of a producer (enqueue_x16_tiles_map2) == the tile pass over the producer's stored FP32 values;
//  5. A4 emulation tiles (a4_tiles::Emu, g32 / g64 / Hadamard-128 g32) of both producers == a4_tiles::reference;
//  6. the GLU pair sweep (enqueue_glu_x16) writes the same x16 tiles of h as two x16 GEMMs + map2, bit for bit.
#include "core/a4_tiles.hpp"
#include "core/f16_dense_iu4.hpp"
#include "core/iu4.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace iu = strix::core::f16_dense_tall::iu4a16;
namespace tall = strix::core::f16_dense_tall;

#define CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

__global__ void decode_kernel(iu::View v, std::uint16_t* out) {
    const unsigned o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= v.count) return;
    tall::U4 nb[8];
    iu::Cols<8>{v}.stage(o / 16U, o % 16U, 0U, nb);
    for (unsigned q = 0; q < 8U; ++q)
        for (unsigned p = 0; p < 4U; ++p) {
            out[std::size_t(o) * 64U + 8U * q + 2U * p] = std::uint16_t(nb[q][p] & 0xffffU);
            out[std::size_t(o) * 64U + 8U * q + 2U * p + 1U] = std::uint16_t(nb[q][p] >> 16U);
        }
}

// 4. x16 tiles of a producer: x16_tiles_map2_kernel<Mul> == x16_tiles_kernel over the stored FP32 products, bit for bit.
template <class Q> struct Tag { using type = Q; };
struct Mul {
    __device__ __forceinline__ static float apply(float a, float b) { return a / (1.0F + expf(-a)) * b; }
};
__global__ void mul_kernel(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Mul::apply(a[i], b[i]);
}

static bool is_nan16(std::uint16_t b) { return (b & 0x7c00U) == 0x7c00U && (b & 0x3ffU); }

int main() {
    int failures = 0;
    {  // 1. decode: column o has scale o, codes (k % 16) - 8 at k = 0..63
        const unsigned N = 65536U, K = 64U;
        std::vector<std::uint8_t> w(std::size_t(N) * K / 2U + std::size_t(N) * 2U, 0U);
        for (unsigned o = 0; o < N; ++o)
            for (unsigned k = 0; k < K; ++k) {
                auto& byte = w[strix::core::iu4_weight_byte(int(k), int(o), int(N))];
                strix::core::iu4_put(byte, int(k % 2U), int(k % 16U) - 8);
            }
        auto* sc = reinterpret_cast<std::uint16_t*>(w.data() + std::size_t(N) * K / 2U);
        for (unsigned o = 0; o < N; ++o) sc[o] = std::uint16_t(o);
        std::uint8_t* dw = nullptr; std::uint16_t* dout = nullptr;
        CHECK(hipMalloc(&dw, w.size())); CHECK(hipMalloc(&dout, std::size_t(N) * K * 2U));
        CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        iu::View v{};
        v.weight = dw; v.scale = reinterpret_cast<const std::uint16_t*>(dw + std::size_t(N) * K / 2U);
        v.in = K; v.n = N; v.count = N; v.rows = 1U; v.out_stride = N;
        hipLaunchKernelGGL(decode_kernel, dim3(N / 256U), dim3(256), 0, 0, v, dout);
        CHECK(hipGetLastError());
        std::vector<std::uint16_t> got(std::size_t(N) * K);
        CHECK(hipMemcpy(got.data(), dout, got.size() * 2U, hipMemcpyDeviceToHost));
        unsigned bad = 0;
        for (unsigned o = 0; o < N; ++o)
            for (unsigned k = 0; k < K; ++k) {
                const std::uint16_t want = __builtin_bit_cast(std::uint16_t, iu::host_w16(int(k % 16U) - 8, std::uint16_t(o)));
                const std::uint16_t g = got[std::size_t(o) * K + k];
                if (g != want && !(is_nan16(g) && is_nan16(want)) && bad++ < 5)
                    std::printf("decode scale 0x%04x code %d: got 0x%04x want 0x%04x\n", o, int(k % 16U) - 8, g, want);
            }
        std::printf("decode: %u mismatches of %u\n", bad, N * K);
        failures += bad != 0U;
        CHECK(hipFree(dw)); CHECK(hipFree(dout));
    }
    {  // 2. GEMM
        const unsigned R = 200U, K = 256U, N = 80U, C = 72U;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> ux(-2.0F, 2.0F), us(0.001F, 0.02F);
        std::uniform_int_distribution<int> uc(-8, 7);
        std::vector<float> x(std::size_t(R) * K);
        for (auto& f : x) f = ux(rng);
        std::vector<int> code(std::size_t(N) * K);
        for (auto& c : code) c = uc(rng);
        std::vector<std::uint8_t> w(std::size_t(N) * K / 2U + std::size_t(N) * (K / 64U) * 2U, 0U);
        for (unsigned o = 0; o < N; ++o)
            for (unsigned k = 0; k < K; ++k)
                strix::core::iu4_put(w[strix::core::iu4_weight_byte(int(k), int(o), int(N))], int(k % 2U), code[std::size_t(o) * K + k]);
        auto* sc = reinterpret_cast<std::uint16_t*>(w.data() + std::size_t(N) * K / 2U);
        for (unsigned i = 0; i < N * (K / 64U); ++i) sc[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(us(rng)));
        std::uint8_t* dw = nullptr; float *dx = nullptr, *dy = nullptr;
        CHECK(hipMalloc(&dw, w.size())); CHECK(hipMalloc(&dx, x.size() * 4U)); CHECK(hipMalloc(&dy, std::size_t(R) * N * 4U));
        CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice));
        iu::View v{};
        v.weight = dw; v.scale = reinterpret_cast<const std::uint16_t*>(dw + std::size_t(N) * K / 2U);
        v.x = dx; v.out = dy; v.in = K; v.n = N; v.count = C; v.rows = R; v.out_stride = N;
        if (!iu::supported(v)) { std::printf("GEMM view not supported\n"); return 1; }
        std::vector<float> y8(std::size_t(R) * N), y16(std::size_t(R) * N);
        for (unsigned waves : {8U, 16U}) {
            std::vector<float> sentinel(std::size_t(R) * N, -12345.0F);
            CHECK(hipMemcpy(dy, sentinel.data(), sentinel.size() * 4U, hipMemcpyHostToDevice));
            iu::enqueue(v, 0, waves);
            CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy((waves == 8U ? y8 : y16).data(), dy, y8.size() * 4U, hipMemcpyDeviceToHost));
        }
        double worst = 0.0;
        unsigned untouched_bad = 0;
        for (unsigned r = 0; r < R; ++r)
            for (unsigned o = 0; o < N; ++o) {
                const float got = y8[std::size_t(r) * N + o];
                if (o >= C) { untouched_bad += got != -12345.0F; continue; }
                double sum = 0.0, mag = 0.0;
                for (unsigned k = 0; k < K; ++k) {
                    const double t = double(static_cast<float>(static_cast<_Float16>(x[std::size_t(r) * K + k]))) *
                        double(static_cast<float>(iu::host_w16(code[std::size_t(o) * K + k], sc[o * (K / 64U) + k / 64U])));
                    sum += t, mag += std::fabs(t);
                }
                worst = std::max(worst, std::fabs(double(got) - sum) / (mag > 0.0 ? mag : 1.0));
            }
        const bool same = std::memcmp(y8.data(), y16.data(), y8.size() * 4U) == 0;
        std::printf("gemm: worst |err|/sum|terms| %.3g, columns past count written %u, 8 vs 16 waves %s\n", worst, untouched_bad,
                    same ? "bitwise" : "DIFFER");
        failures += !(worst < 1.0 / 1024.0) + (untouched_bad != 0U) + !same;
        {  // 3. x16 form (tiles of x, column-fast) vs the direct form, 8 and 16 waves
            _Float16* dxt = nullptr;
            float* dz = nullptr;
            CHECK(hipMalloc(&dxt, tall::x16_bytes(R, K))); CHECK(hipMalloc(&dz, std::size_t(R) * N * 4U));
            tall::enqueue_x16_tiles(dx, dxt, R, K, 0);
            std::vector<float> got(std::size_t(R) * N);
            unsigned bad = 0;
            for (unsigned waves : {8U, 16U}) {
                std::vector<float> sentinel(std::size_t(R) * N, -12345.0F);
                CHECK(hipMemcpy(dz, sentinel.data(), sentinel.size() * 4U, hipMemcpyHostToDevice));
                iu::View vz = v;
                vz.out = dz;
                iu::enqueue_x16(vz, dxt, 0, waves);
                CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
                CHECK(hipMemcpy(got.data(), dz, got.size() * 4U, hipMemcpyDeviceToHost));
                bad += std::memcmp(got.data(), y16.data(), got.size() * 4U) != 0;
            }
            std::printf("x16 form: 8 / 16 waves vs direct %s\n", bad ? "DIFFER" : "bitwise (columns past count untouched)");
            failures += bad != 0U;
            CHECK(hipFree(dxt)); CHECK(hipFree(dz));
        }
        CHECK(hipFree(dw)); CHECK(hipFree(dx)); CHECK(hipFree(dy));
    }
    {  // 4. producer tiles vs stored products + tile pass
        const unsigned R = 130U, K = 384U;
        const std::size_t n = std::size_t(R) * K;
        std::mt19937 rng(9);
        std::uniform_real_distribution<float> ua(-8.0F, 8.0F);
        std::vector<float> a(n), b(n);
        for (auto& f : a) f = ua(rng);
        for (auto& f : b) f = ua(rng) * 1.0e-3F;
        float *da = nullptr, *db = nullptr, *dp = nullptr;
        _Float16 *t1 = nullptr, *t2 = nullptr;
        CHECK(hipMalloc(&da, n * 4U)); CHECK(hipMalloc(&db, n * 4U)); CHECK(hipMalloc(&dp, n * 4U));
        CHECK(hipMalloc(&t1, n * 2U)); CHECK(hipMalloc(&t2, n * 2U));
        CHECK(hipMemcpy(da, a.data(), n * 4U, hipMemcpyHostToDevice)); CHECK(hipMemcpy(db, b.data(), n * 4U, hipMemcpyHostToDevice));
        mul_kernel<<<unsigned((n + 255U) / 256U), 256>>>(da, db, dp, n);
        tall::enqueue_x16_tiles(dp, t1, R, K, 0);
        tall::enqueue_x16_tiles_map2<Mul>(da, db, t2, R, K, 0);
        CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
        std::vector<std::uint16_t> h1(n), h2(n);
        CHECK(hipMemcpy(h1.data(), t1, n * 2U, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(h2.data(), t2, n * 2U, hipMemcpyDeviceToHost));
        const bool same = std::memcmp(h1.data(), h2.data(), n * 2U) == 0;
        std::printf("producer tiles: map2 vs stored products + tile pass %s\n", same ? "bitwise" : "DIFFER");
        failures += !same;
        CHECK(hipFree(da)); CHECK(hipFree(db)); CHECK(hipFree(dp)); CHECK(hipFree(t1)); CHECK(hipFree(t2));
    }
    {  // 5. A4 emulation tiles (core/a4_tiles.hpp Emu<G, Rotate>) == the host reference, bit for bit, both producers
        const unsigned R = 37U, K = 384U;
        const std::size_t n = std::size_t(R) * K;
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        std::vector<float> x(n), ones(n, 1.0F);
        for (std::size_t i = 0; i < n; ++i) x[i] = nd(rng) * (i % 97U == 0U ? 40.0F : 1.0F);  // sparse outliers
        for (unsigned k = 0; k < 64U; ++k) x[std::size_t(3) * K + k] = 0.0F;                  // an all-zero group
        float *dx = nullptr, *d1 = nullptr, *dp = nullptr;
        _Float16 *t1 = nullptr, *t2 = nullptr;
        CHECK(hipMalloc(&dx, n * 4U)); CHECK(hipMalloc(&d1, n * 4U)); CHECK(hipMalloc(&dp, n * 4U));
        CHECK(hipMalloc(&t1, n * 2U)); CHECK(hipMalloc(&t2, n * 2U));
        CHECK(hipMemcpy(dx, x.data(), n * 4U, hipMemcpyHostToDevice)); CHECK(hipMemcpy(d1, ones.data(), n * 4U, hipMemcpyHostToDevice));
        // map2 quantises the producer's values Mul(x, 1) = SiLU(x): its reference input is the stored products (section 4
        // proves the producer's FP32 values equal them bit for bit).
        mul_kernel<<<unsigned((n + 255U) / 256U), 256>>>(dx, d1, dp, n);
        std::vector<float> prod(n);
        CHECK(hipMemcpy(prod.data(), dp, n * 4U, hipMemcpyDeviceToHost));
        auto arm = [&](auto tag, const char* name, unsigned group, bool rotate) {
            using Q = typename decltype(tag)::type;
            tall::enqueue_x16_tiles<Q>(dx, t1, R, K, 0);
            tall::enqueue_x16_tiles_map2<Mul, Q>(dx, d1, t2, R, K, 0);
            CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
            std::vector<std::uint16_t> h1(n), h2(n);
            CHECK(hipMemcpy(h1.data(), t1, n * 2U, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(h2.data(), t2, n * 2U, hipMemcpyDeviceToHost));
            unsigned bad1 = 0U, bad2 = 0U;
            std::vector<float> row(K), row2(K);
            for (unsigned r = 0; r < R; ++r) {
                std::memcpy(row.data(), x.data() + std::size_t(r) * K, K * 4U);
                std::memcpy(row2.data(), prod.data() + std::size_t(r) * K, K * 4U);
                strix::core::a4_tiles::reference(row.data(), K, group, rotate);
                strix::core::a4_tiles::reference(row2.data(), K, group, rotate);
                for (unsigned k = 0; k < K; ++k) {
                    const std::size_t at = (std::size_t(k / 64U) * R + r) * 64U + k % 64U;
                    bad1 += h1[at] != __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(row[k]));
                    bad2 += h2[at] != __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(row2[k]));
                }
            }
            std::printf("a4 emu %s: tiles vs host reference: %u / %u of %zu differ (tile pass / map2)\n", name, bad1, bad2, n);
            failures += (bad1 + bad2) != 0U;
            return 0;  // CHECK returns 1 on a HIP error
        };
        namespace a4 = strix::core::a4_tiles;
        arm(Tag<a4::Emu<32U, false>>{}, "g32", 32U, false);
        arm(Tag<a4::Emu<64U, false>>{}, "g64", 64U, false);
        arm(Tag<a4::Emu<32U, true>>{}, "h128g32", 32U, true);
        arm(Tag<a4::Emu<64U, true>>{}, "h128g64", 64U, true);
        CHECK(hipFree(dx)); CHECK(hipFree(d1)); CHECK(hipFree(dp)); CHECK(hipFree(t1)); CHECK(hipFree(t2));
    }
    {  // 6. GLU pair sweep (enqueue_glu_x16) == enqueue_x16(first) + enqueue_x16(second) + map2, bit for bit, 8 / 16 waves
        const unsigned R = 300U, K = 384U, N = 256U;  // partial row CTA; two 128-output pair column blocks
        std::mt19937 rng(11);
        std::uniform_int_distribution<int> uc(-8, 7);
        std::uniform_real_distribution<float> us(-0.05F, 0.05F), ux(-2.0F, 2.0F);
        const auto pack = [&]() {
            std::vector<std::uint8_t> w(std::size_t(N) * K / 2U + std::size_t(N) * (K / 64U) * 2U, 0U);
            for (unsigned o = 0; o < N; ++o)
                for (unsigned k = 0; k < K; ++k) strix::core::iu4_put(w[strix::core::iu4_weight_byte(int(k), int(o), int(N))], int(k % 2U), uc(rng));
            auto* sc = reinterpret_cast<std::uint16_t*>(w.data() + std::size_t(N) * K / 2U);
            for (unsigned i = 0; i < N * (K / 64U); ++i) sc[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(us(rng)));
            return w;
        };
        const std::vector<std::uint8_t> wg = pack(), wu = pack();
        std::vector<float> x(std::size_t(R) * K);
        for (auto& v : x) v = ux(rng);
        std::uint8_t *dg = nullptr, *du = nullptr;
        float *dx = nullptr, *yg = nullptr, *yu = nullptr;
        _Float16 *dxt = nullptr, *href = nullptr, *hpair = nullptr;
        const std::size_t ht = tall::x16_bytes(R, N);
        CHECK(hipMalloc(&dg, wg.size())); CHECK(hipMalloc(&du, wu.size())); CHECK(hipMalloc(&dx, x.size() * 4U));
        CHECK(hipMalloc(&yg, std::size_t(R) * N * 4U)); CHECK(hipMalloc(&yu, std::size_t(R) * N * 4U));
        CHECK(hipMalloc(&dxt, tall::x16_bytes(R, K))); CHECK(hipMalloc(&href, ht)); CHECK(hipMalloc(&hpair, ht));
        CHECK(hipMemcpy(dg, wg.data(), wg.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(du, wu.data(), wu.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice));
        const auto view = [&](const std::uint8_t* w, float* y) {
            iu::View v{};
            v.weight = w; v.scale = reinterpret_cast<const std::uint16_t*>(w + std::size_t(N) * K / 2U);
            v.x = dx; v.out = y; v.in = K; v.n = N; v.count = N; v.rows = R; v.out_stride = N;
            return v;
        };
        iu::GluView gv{view(dg, nullptr), du, reinterpret_cast<const std::uint16_t*>(du + std::size_t(N) * K / 2U), hpair};
        if (!iu::glu_pair_supported(gv)) { std::printf("GLU pair view not supported\n"); return 1; }
        tall::enqueue_x16_tiles(dx, dxt, R, K, 0);
        unsigned bad = 0;
        for (unsigned waves : {8U, 16U}) {
            CHECK(hipMemset(href, 0x7f, ht)); CHECK(hipMemset(hpair, 0x55, ht));
            iu::enqueue_x16(view(dg, yg), dxt, 0, waves);
            iu::enqueue_x16(view(du, yu), dxt, 0, waves);
            tall::enqueue_x16_tiles_map2<Mul>(yg, yu, href, R, N, 0);
            iu::enqueue_glu_x16<Mul>(gv, dxt, 0, waves);
            CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
            std::vector<std::uint16_t> a(ht / 2U), b(ht / 2U);
            CHECK(hipMemcpy(a.data(), href, ht, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(b.data(), hpair, ht, hipMemcpyDeviceToHost));
            bad += std::memcmp(a.data(), b.data(), ht) != 0;
        }
        std::printf("glu pair: h tiles vs x16(first) + x16(second) + map2, 8 / 16 waves %s\n", bad ? "DIFFER" : "bitwise");
        failures += bad != 0U;
        for (void* p : {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(dx), static_cast<void*>(yg), static_cast<void*>(yu),
                        static_cast<void*>(dxt), static_cast<void*>(href), static_cast<void*>(hpair)})
            CHECK(hipFree(p));
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
