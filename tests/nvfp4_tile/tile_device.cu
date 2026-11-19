// Device fixture + operator bench of the NVFP4 tile GEMM (sm_120; run only through an operator ticket on device 0).
//   tile_device            : quantiser bytes == host reference (bitwise), GEMM store / residual / silu_mul / glu within the FP32
//                            accumulation bound of the double reference, repeated launches bitwise identical.
//   tile_device --bench R  : model shapes of the 27B dense layer at R activation rows, median of 20 timed launches per shape
//                            (lsu and bulk alternating launch by launch); --bench R lsu: lsu alone, main's timing loop.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strix/cuda/core/nvfp4_tile_cuda.hpp"
#include "strix/cuda_core/gdn.hpp"
#include "strix/cuda_core/norms.hpp"
#include "tile_ref.hpp"

namespace {
namespace T = strix::cuda::core::nvfp4::tile;
using namespace tile_ref;
#define CU(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)
template <class V>
typename V::value_type* up(const V& h) {
    typename V::value_type* d = nullptr;
    CU(cudaMalloc(&d, std::max<std::size_t>(1, h.size() * sizeof(typename V::value_type))));
    if (!h.empty()) CU(cudaMemcpy(d, h.data(), h.size() * sizeof(typename V::value_type), cudaMemcpyHostToDevice));
    return d;
}
template <class E>
std::vector<E> down(const E* d, std::size_t n) {
    std::vector<E> h(n);
    CU(cudaMemcpy(h.data(), d, n * sizeof(E), cudaMemcpyDeviceToHost));
    return h;
}
int g_fail = 0;

// Feed::bulk must reproduce the lsu output `y` bit for bit (same stage bytes, same MMA order per accumulator). `reset`: the
// residual form's starting y (it runs in place); otherwise y is poisoned with NaN first so an unwritten element cannot match.
bool bulk_bitwise(T::GemmArgs g, const float* reset, const std::vector<float>& y) {
    if (reset) CU(cudaMemcpy(g.y, reset, y.size() * 4, cudaMemcpyDeviceToDevice));
    else CU(cudaMemset(g.y, 0xff, y.size() * 4));
    g.feed = T::Feed::bulk;
    CU(T::gemm_enqueue(g, nullptr));
    const auto b = down(g.y, y.size());
    return std::memcmp(b.data(), y.data(), y.size() * 4) == 0;
}

// one plain GEMM case: epi store / residual / silu_mul; returns max |err| / bound
void check_gemm(const char* name, std::uint32_t out, std::uint32_t in, std::uint32_t rows, T::Epi epi, std::uint32_t seed) {
    const Weight w = random_weight(out, in, seed);
    const auto x = random_x(rows, in, seed + 5);
    const Act ha = quantize(x, rows, in);
    std::vector<float> aux(static_cast<std::size_t>(rows) * out);
    for (std::size_t i = 0; i < aux.size(); ++i) aux[i] = 0.001f * static_cast<float>(static_cast<int>(i % 2001) - 1000);
    std::uint8_t* drec = up(w.rec);
    float* dx = up(x);
    float* daux = up(aux);
    std::uint8_t *dw = nullptr, *dxt = nullptr;
    float *drs = nullptr, *dy = nullptr;
    CU(cudaMalloc(&dw, T::w_tile_bytes(out, in)));
    CU(cudaMalloc(&dxt, T::x_tile_bytes(ha.rows_pad, in)));
    CU(cudaMalloc(&drs, ha.rows_pad * sizeof(float)));
    CU(cudaMalloc(&dy, aux.size() * sizeof(float)));
    CU(T::repack_enqueue(drec, out, in, dw, nullptr));
    CU(T::quantize_enqueue(dx, rows, in, dxt, drs, nullptr));
    const auto xt = down(dxt, ha.tile.size());
    const auto rs = down(drs, ha.rows_pad);
    const bool qsame = xt == ha.tile && std::memcmp(rs.data(), ha.rs.data(), rs.size() * 4) == 0;
    T::GemmArgs g;  // reference arm: Feed::lsu (also the default; bulk_bitwise re-runs it with bulk)
    g.feed = T::Feed::lsu;
    g.w = dw;
    g.x = dxt;
    g.rs = drs;
    g.aux = daux;
    g.y = dy;
    g.gs = w.gs;
    g.out = out;
    g.in = in;
    g.rows = rows;
    g.rows_pad = ha.rows_pad;
    g.epi = epi;
    if (epi == T::Epi::residual) {  // aliasing form (aux == y), as the engine's residual GEMMs
        CU(cudaMemcpy(dy, daux, aux.size() * 4, cudaMemcpyDeviceToDevice));
        g.aux = dy;
    }
    CU(T::gemm_enqueue(g, nullptr));
    const auto y = down(dy, aux.size());
    const bool bsame = bulk_bitwise(g, epi == T::Epi::residual ? daux : nullptr, y);
    double worst = 0;
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < out; ++o) {
            double ref = 0, mag = 0;
            for (std::uint32_t k = 0; k < in; ++k) {
                const double p = w_value(w, o, k) * x_value(ha, r, k);
                ref += p;
                mag += std::fabs(p);
            }
            const double sc = static_cast<double>(w.gs) * ha.rs[r];
            ref *= sc;
            mag = mag * sc * 3e-5 + 1e-30;  // FP32 tensor accumulation bound (~500 ulp of sum|p|); layout errors are O(1)
            const double a = aux[static_cast<std::size_t>(r) * out + o];
            if (epi == T::Epi::residual) ref += a, mag += std::fabs(a) * 1.2e-7;
            if (epi == T::Epi::silu_mul) ref *= silu(a), mag *= std::fabs(silu(a)) + 1e-30;
            worst = std::max(worst, std::fabs(y[static_cast<std::size_t>(r) * out + o] - ref) / mag);
        }
    // determinism: a second launch is bitwise identical (re-run store over the same inputs)
    bool det = true;
    if (epi == T::Epi::store) {
        CU(T::gemm_enqueue(g, nullptr));
        det = down(dy, aux.size()) == y;
    }
    const bool ok = qsame && det && bsame && worst <= 1.0;
    std::printf("TILE CHECK %-10s out=%u in=%u rows=%u quant-bitwise=%d deterministic=%d bulk-bitwise=%d err/bound=%.3f %s\n", name,
                out, in, rows, qsame ? 1 : 0, det ? 1 : 0, bsame ? 1 : 0, worst, ok ? "PASS" : "FAIL");
    g_fail += ok ? 0 : 1;
    for (void* p : {static_cast<void*>(drec), static_cast<void*>(dx), static_cast<void*>(daux), static_cast<void*>(dw),
                    static_cast<void*>(dxt), static_cast<void*>(drs), static_cast<void*>(dy)})
        CU(cudaFree(p));
}

void check_glu(std::uint32_t ch, std::uint32_t in, std::uint32_t rows, std::uint32_t seed) {
    const Weight wg = random_weight(ch, in, seed), wu = random_weight(ch, in, seed + 1);
    const auto x = random_x(rows, in, seed + 3);
    const Act ha = quantize(x, rows, in);
    std::uint8_t *dg = up(wg.rec), *du = up(wu.rec), *tg = nullptr, *tu = nullptr, *dxt = nullptr;
    float *dx = up(x), *drs = nullptr, *dy = nullptr;
    CU(cudaMalloc(&tg, T::w_tile_bytes(ch, in)));
    CU(cudaMalloc(&tu, T::w_tile_bytes(ch, in)));
    CU(cudaMalloc(&dxt, T::x_tile_bytes(ha.rows_pad, in)));
    CU(cudaMalloc(&drs, ha.rows_pad * 4));
    CU(cudaMalloc(&dy, static_cast<std::size_t>(rows) * ch * 4));
    CU(T::repack_enqueue(dg, ch, in, tg, nullptr));
    CU(T::repack_enqueue(du, ch, in, tu, nullptr));
    CU(T::quantize_enqueue(dx, rows, in, dxt, drs, nullptr));
    T::GemmArgs g;  // reference arm: Feed::lsu
    g.feed = T::Feed::lsu;
    g.w = tg;
    g.w2 = tu;
    g.x = dxt;
    g.rs = drs;
    g.y = dy;
    g.gs = wg.gs;
    g.gs2 = wu.gs;
    g.out = ch;
    g.in = in;
    g.rows = rows;
    g.rows_pad = ha.rows_pad;
    g.epi = T::Epi::glu;
    CU(T::gemm_enqueue(g, nullptr));
    const auto y = down(dy, static_cast<std::size_t>(rows) * ch);
    const bool bsame = bulk_bitwise(g, nullptr, y);
    double worst = 0;
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < ch; ++c) {
            double gr = 0, gm = 0, ur = 0, um = 0;
            for (std::uint32_t k = 0; k < in; ++k) {
                const double xv = x_value(ha, r, k), pg = w_value(wg, c, k) * xv, pu = w_value(wu, c, k) * xv;
                gr += pg, gm += std::fabs(pg), ur += pu, um += std::fabs(pu);
            }
            const double sg = static_cast<double>(wg.gs) * ha.rs[r], su = static_cast<double>(wu.gs) * ha.rs[r];
            const double ref = silu(gr * sg) * ur * su;
            const double bound = (gm * sg * 3e-5 * 1.2 * std::fabs(ur * su) + std::fabs(silu(gr * sg)) * um * su * 3e-5) + 1e-30;
            worst = std::max(worst, std::fabs(y[static_cast<std::size_t>(r) * ch + c] - ref) / bound);
        }
    const bool ok = bsame && worst <= 1.0;
    std::printf("TILE CHECK %-10s ch=%u in=%u rows=%u bulk-bitwise=%d err/bound=%.3f %s\n", "glu", ch, in, rows, bsame ? 1 : 0, worst,
                ok ? "PASS" : "FAIL");
    g_fail += ok ? 0 : 1;
    for (void* p : {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(tg), static_cast<void*>(tu),
                    static_cast<void*>(dxt), static_cast<void*>(dx), static_cast<void*>(drs), static_cast<void*>(dy)})
        CU(cudaFree(p));
}

// RMSNorm folded into the quantiser: norm_quantize_enqueue(x, gain) must give the X tile + rs of cuda_core rmsnorm(x, gain) followed
// by quantize_enqueue, byte for byte (also with a null gain and padding rows); repeated launches bitwise identical.
void check_norm_quant(std::uint32_t rows, std::uint32_t in, bool with_gain, std::uint32_t seed) {
    namespace cc = strix::cuda_core;
    const std::uint32_t rp = T::rows_padded(rows);
    auto hx = random_x(rows, in, seed);
    for (std::size_t i = 0; i < hx.size(); ++i) hx[i] *= static_cast<float>(1 + (i * 2654435761u >> 7) % 61) * 0.37f;  // spread magnitudes
    std::vector<float> hg(in);
    for (std::uint32_t i = 0; i < in; ++i) hg[i] = 0.25f + static_cast<float>((i * 40503u + seed) % 997) / 400.f;
    float* dx = up(hx);
    float* dg = up(hg);
    float *dxn = nullptr, *drs_a = nullptr, *drs_b = nullptr;
    std::uint8_t *dxt_a = nullptr, *dxt_b = nullptr;
    const std::size_t tb = T::x_tile_bytes(rp, in);
    CU(cudaMalloc(&dxn, hx.size() * sizeof(float)));
    CU(cudaMalloc(&dxt_a, tb));
    CU(cudaMalloc(&dxt_b, tb));
    CU(cudaMalloc(&drs_a, rp * sizeof(float)));
    CU(cudaMalloc(&drs_b, rp * sizeof(float)));
    const float eps = 1e-6f;
    const cc::GroupedView v{rows, 1, in, in, in};
    CU(cc::rmsnorm(dx, v, with_gain ? dg : nullptr, dxn, v, eps, nullptr));
    CU(T::quantize_enqueue(dxn, rows, in, dxt_a, drs_a, nullptr));
    CU(cudaMemset(dxt_b, 0xa5, tb));
    CU(T::norm_quantize_enqueue(dx, with_gain ? dg : nullptr, eps, rows, in, dxt_b, drs_b, nullptr));
    const auto ta = down(dxt_a, tb), tb1 = down(dxt_b, tb);
    const auto ra = down(drs_a, rp), rb1 = down(drs_b, rp);
    CU(T::norm_quantize_enqueue(dx, with_gain ? dg : nullptr, eps, rows, in, dxt_b, drs_b, nullptr));
    const auto tb2 = down(dxt_b, tb);
    const auto rb2 = down(drs_b, rp);
    const bool same = ta == tb1 && std::memcmp(ra.data(), rb1.data(), rp * 4) == 0;
    const bool det = tb1 == tb2 && std::memcmp(rb1.data(), rb2.data(), rp * 4) == 0;
    const bool ok = same && det;
    if (!ok) ++g_fail;
    std::printf("TILE CHECK %-10s in=%u rows=%u gain=%d bitwise-vs-rmsnorm+quantize=%d deterministic=%d %s\n", "norm_quant", in, rows,
                with_gain ? 1 : 0, same ? 1 : 0, det ? 1 : 0, ok ? "PASS" : "FAIL");
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dg), static_cast<void*>(dxn), static_cast<void*>(dxt_a),
                    static_cast<void*>(dxt_b), static_cast<void*>(drs_a), static_cast<void*>(drs_b)})
        CU(cudaFree(p));
}

// GDN gated RMSNorm folded into the quantiser: gated_norm_quantize_enqueue(x, z, gain) must give the X tile + rs of cuda_core
// gdn_chunk_norm (in place on a copy of x) followed by quantize_enqueue, byte for byte; repeated launches bitwise identical.
void check_gated_norm_quant(std::uint32_t rows, std::uint32_t in, std::uint32_t seed) {
    namespace cc = strix::cuda_core;
    const std::uint32_t rp = T::rows_padded(rows), heads = in / T::kGatedNormHead;
    auto hx = random_x(rows, in, seed);
    for (std::size_t i = 0; i < hx.size(); ++i) hx[i] *= static_cast<float>(1 + (i * 2654435761u >> 7) % 61) * 0.37f;
    auto hz = random_x(rows, in, seed + 7);
    for (std::size_t i = 0; i < hz.size(); ++i) hz[i] *= 3.0f;
    std::vector<float> hg(T::kGatedNormHead);
    for (std::uint32_t i = 0; i < T::kGatedNormHead; ++i) hg[i] = 0.25f + static_cast<float>((i * 40503u + seed) % 997) / 400.f;
    float* dx = up(hx);
    float* dref = up(hx);
    float* dz = up(hz);
    float* dg = up(hg);
    float *drs_a = nullptr, *drs_b = nullptr;
    std::uint8_t *dxt_a = nullptr, *dxt_b = nullptr;
    const std::size_t tb = T::x_tile_bytes(rp, in);
    CU(cudaMalloc(&dxt_a, tb));
    CU(cudaMalloc(&dxt_b, tb));
    CU(cudaMalloc(&drs_a, rp * sizeof(float)));
    CU(cudaMalloc(&drs_b, rp * sizeof(float)));
    const float eps = 1e-6f;
    const cc::GdnDims d{16, 128, heads, 128, 4};
    CU(cc::gdn_chunk_norm(d, dg, dz, dref, rows, eps, nullptr));
    CU(T::quantize_enqueue(dref, rows, in, dxt_a, drs_a, nullptr));
    CU(cudaMemset(dxt_b, 0xa5, tb));
    CU(T::gated_norm_quantize_enqueue(dx, dz, dg, eps, rows, in, dxt_b, drs_b, nullptr));
    const auto ta = down(dxt_a, tb), tb1 = down(dxt_b, tb);
    const auto ra = down(drs_a, rp), rb1 = down(drs_b, rp);
    CU(T::gated_norm_quantize_enqueue(dx, dz, dg, eps, rows, in, dxt_b, drs_b, nullptr));
    const auto tb2 = down(dxt_b, tb);
    const auto rb2 = down(drs_b, rp);
    const bool same = ta == tb1 && std::memcmp(ra.data(), rb1.data(), rp * 4) == 0;
    const bool det = tb1 == tb2 && std::memcmp(rb1.data(), rb2.data(), rp * 4) == 0;
    const bool ok = same && det;
    if (!ok) ++g_fail;
    std::printf("TILE CHECK %-10s in=%u rows=%u heads=%u bitwise-vs-gdn_chunk_norm+quantize=%d deterministic=%d %s\n", "gnorm_quant", in,
                rows, heads, same ? 1 : 0, det ? 1 : 0, ok ? "PASS" : "FAIL");
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dref), static_cast<void*>(dz), static_cast<void*>(dg),
                    static_cast<void*>(dxt_a), static_cast<void*>(dxt_b), static_cast<void*>(drs_a), static_cast<void*>(drs_b)})
        CU(cudaFree(p));
}

// timing of rmsnorm + quantize (two passes, f32 rows through memory) vs the folded producer: median of 20 after 3 warm-ups
void bench_norm_quant(std::uint32_t rows, std::uint32_t in, std::uint32_t gdn_in) {
    namespace cc = strix::cuda_core;
    const std::uint32_t rp = T::rows_padded(rows);
    float *x = nullptr, *xn = nullptr, *g = nullptr, *rs = nullptr;
    std::uint8_t* xt = nullptr;
    CU(cudaMalloc(&x, static_cast<std::size_t>(rows) * in * 4));
    CU(cudaMalloc(&xn, static_cast<std::size_t>(rows) * in * 4));
    CU(cudaMalloc(&g, in * 4));
    CU(cudaMalloc(&xt, T::x_tile_bytes(rp, in)));
    CU(cudaMalloc(&rs, rp * 4));
    CU(cudaMemset(x, 0x3c, static_cast<std::size_t>(rows) * in * 4));
    CU(cudaMemset(g, 0x3f, in * 4));
    const cc::GroupedView v{rows, 1, in, in, in};
    cudaEvent_t a, b;
    CU(cudaEventCreate(&a));
    CU(cudaEventCreate(&b));
    auto timed = [&](auto&& f) {
        for (int i = 0; i < 3; ++i) f();
        std::vector<float> ms;
        for (int i = 0; i < 20; ++i) {
            CU(cudaEventRecord(a));
            f();
            CU(cudaEventRecord(b));
            CU(cudaEventSynchronize(b));
            float t = 0;
            CU(cudaEventElapsedTime(&t, a, b));
            ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };
    const float sep = timed([&] {
        CU(cc::rmsnorm(x, v, g, xn, v, 1e-6f, nullptr));
        CU(T::quantize_enqueue(xn, rows, in, xt, rs, nullptr));
    });
    const float fused = timed([&] { CU(T::norm_quantize_enqueue(x, g, 1e-6f, rows, in, xt, rs, nullptr)); });
    std::printf("TILE BENCH norm_quant in=%u rows=%u separate_ms=%.4f fused_ms=%.4f saved_ms=%.4f\n", in, rows, sep, fused, sep - fused);
    if (gdn_in != 0) {  // GDN gated RMSNorm (gdn_chunk_norm in place, then quantize) vs gated_norm_quantize_enqueue; z = x buffer
        float *gx = nullptr, *gz = nullptr;
        CU(cudaMalloc(&gx, static_cast<std::size_t>(rows) * gdn_in * 4));
        CU(cudaMalloc(&gz, static_cast<std::size_t>(rows) * gdn_in * 4));
        CU(cudaMemset(gx, 0x3c, static_cast<std::size_t>(rows) * gdn_in * 4));
        CU(cudaMemset(gz, 0x3c, static_cast<std::size_t>(rows) * gdn_in * 4));
        const cc::GdnDims d{16, 128, gdn_in / T::kGatedNormHead, 128, 4};
        const float gsep = timed([&] {
            CU(cc::gdn_chunk_norm(d, g, gz, gx, rows, 1e-6f, nullptr));
            CU(T::quantize_enqueue(gx, rows, gdn_in, xt, rs, nullptr));
        });
        const float gfused = timed([&] { CU(T::gated_norm_quantize_enqueue(gx, gz, g, 1e-6f, rows, gdn_in, xt, rs, nullptr)); });
        std::printf("TILE BENCH gnorm_quant in=%u rows=%u separate_ms=%.4f fused_ms=%.4f saved_ms=%.4f\n", gdn_in, rows, gsep, gfused,
                    gsep - gfused);
        CU(cudaFree(gx));
        CU(cudaFree(gz));
    }
    for (void* p : {static_cast<void*>(x), static_cast<void*>(xn), static_cast<void*>(g), static_cast<void*>(xt), static_cast<void*>(rs)})
        CU(cudaFree(p));
    CU(cudaEventDestroy(a));
    CU(cudaEventDestroy(b));
}

// --bench R lsu: time Feed::lsu alone with main's loop (3 warm-ups, 20 timed launches, each one preceded by the same kernel).
// The default --bench alternates lsu and bulk launch by launch (fair bulk/lsu ratio), so there every timed lsu launch follows
// a bulk launch; that sequence is not main's, and the lsu-vs-main comparison must use this mode in both binaries' order.
bool lsu_only = false;
// nvfp4-tile-h16-v1 (BF16 FFN hidden rows), two bitwise invariants on the device:
//   (1) glu with y16 stores exactly bf16_rn (host IEEE transcription) of what glu with y stores, and repeats bitwise;
//   (2) quantize_enqueue over the BF16 rows gives the X tile + rs of quantize_enqueue over their exact FP32 widening, byte for byte.
// ch % 128 == 0 so the hidden rows are a valid quantiser width; rows not a multiple of 8 exercise the padding rows.
void check_glu_h16(std::uint32_t ch, std::uint32_t in, std::uint32_t rows, std::uint32_t seed) {
    const Weight wg = random_weight(ch, in, seed), wu = random_weight(ch, in, seed + 1);
    const auto x = random_x(rows, in, seed + 3);
    const std::uint32_t rp = T::rows_padded(rows);
    const std::size_t n = static_cast<std::size_t>(rows) * ch;
    std::uint8_t *dg = up(wg.rec), *du = up(wu.rec), *tg = nullptr, *tu = nullptr, *dxt = nullptr, *qa = nullptr, *qb = nullptr;
    float *dx = up(x), *drs = nullptr, *dy = nullptr, *ra = nullptr, *rb = nullptr;
    std::uint16_t* dy16 = nullptr;
    CU(cudaMalloc(&tg, T::w_tile_bytes(ch, in)));
    CU(cudaMalloc(&tu, T::w_tile_bytes(ch, in)));
    CU(cudaMalloc(&dxt, T::x_tile_bytes(rp, in)));
    CU(cudaMalloc(&drs, rp * 4));
    CU(cudaMalloc(&dy, n * 4));
    CU(cudaMalloc(&dy16, n * 2));
    CU(cudaMalloc(&qa, T::x_tile_bytes(rp, ch)));
    CU(cudaMalloc(&qb, T::x_tile_bytes(rp, ch)));
    CU(cudaMalloc(&ra, rp * 4));
    CU(cudaMalloc(&rb, rp * 4));
    CU(T::repack_enqueue(dg, ch, in, tg, nullptr));
    CU(T::repack_enqueue(du, ch, in, tu, nullptr));
    CU(T::quantize_enqueue(dx, rows, in, dxt, drs, nullptr));
    T::GemmArgs g;
    g.w = tg;
    g.w2 = tu;
    g.x = dxt;
    g.rs = drs;
    g.y = dy;
    g.gs = wg.gs;
    g.gs2 = wu.gs;
    g.out = ch;
    g.in = in;
    g.rows = rows;
    g.rows_pad = rp;
    g.epi = T::Epi::glu;
    CU(T::gemm_enqueue(g, nullptr));
    g.y = nullptr;
    g.y16 = dy16;
    CU(T::gemm_enqueue(g, nullptr));
    const auto y = down(dy, n);
    const auto h1 = down(dy16, n);
    CU(T::gemm_enqueue(g, nullptr));
    const auto h2 = down(dy16, n);
    std::size_t store_bad = 0;
    for (std::size_t i = 0; i < n; ++i) store_bad += h1[i] != bf16_rn_bits(y[i]) ? 1u : 0u;
    const bool det = h1 == h2;
    std::vector<float> wide(n);
    for (std::size_t i = 0; i < n; ++i) wide[i] = bf16_widen(h1[i]);
    float* dwide = up(wide);
    CU(T::quantize_enqueue(dwide, rows, ch, qa, ra, nullptr));
    CU(T::quantize_enqueue(static_cast<const std::uint16_t*>(dy16), rows, ch, qb, rb, nullptr));
    const bool qsame = down(qa, T::x_tile_bytes(rp, ch)) == down(qb, T::x_tile_bytes(rp, ch)) && down(ra, rp) == down(rb, rp);
    const bool ok = store_bad == 0 && det && qsame;
    std::printf("H16 CHECK glu ch=%u in=%u rows=%u store==bf16_rn(f32) mismatches=%zu deterministic=%d quant(bf16)==quant(widen)=%d %s\n",
                ch, in, rows, store_bad, det ? 1 : 0, qsame ? 1 : 0, ok ? "PASS" : "FAIL");
    g_fail += ok ? 0 : 1;
    for (void* p : {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(tg), static_cast<void*>(tu),
                    static_cast<void*>(dxt), static_cast<void*>(dx), static_cast<void*>(drs), static_cast<void*>(dy),
                    static_cast<void*>(dy16), static_cast<void*>(qa), static_cast<void*>(qb), static_cast<void*>(ra),
                    static_cast<void*>(rb), static_cast<void*>(dwide)})
        CU(cudaFree(p));
}

// H16 operator bench at the FFN shape (gate+up 2 x 17408 over 5120 inputs, down operand 17408 wide): glu FP32 vs BF16 store and
// the down quantiser over FP32 vs BF16 rows, median of 20 launches each after 3 warm-ups.
void bench_h16(std::uint32_t rows) {
    const std::uint32_t ch = 17408, in = 5120, rp = T::rows_padded(rows);
    const std::size_t n = static_cast<std::size_t>(rows) * ch;
    std::uint8_t *w = nullptr, *w2 = nullptr, *xt = nullptr, *ht = nullptr;
    float *rs = nullptr, *x = nullptr, *y = nullptr, *hrs = nullptr;
    std::uint16_t* y16 = nullptr;
    CU(cudaMalloc(&w, T::w_tile_bytes(ch, in)));
    CU(cudaMalloc(&w2, T::w_tile_bytes(ch, in)));
    CU(cudaMalloc(&xt, T::x_tile_bytes(rp, in)));
    CU(cudaMalloc(&rs, rp * 4));
    CU(cudaMalloc(&x, static_cast<std::size_t>(rows) * in * 4));
    CU(cudaMalloc(&y, n * 4));
    CU(cudaMalloc(&y16, n * 2));
    CU(cudaMalloc(&ht, T::x_tile_bytes(rp, ch)));
    CU(cudaMalloc(&hrs, rp * 4));
    CU(cudaMemset(w, 0x35, T::w_tile_bytes(ch, in)));
    CU(cudaMemset(w2, 0x35, T::w_tile_bytes(ch, in)));
    CU(cudaMemset(x, 0x3c, static_cast<std::size_t>(rows) * in * 4));
    CU(T::quantize_enqueue(x, rows, in, xt, rs, nullptr));
    T::GemmArgs g;
    g.w = w;
    g.w2 = w2;
    g.x = xt;
    g.rs = rs;
    g.out = ch;
    g.in = in;
    g.rows = rows;
    g.rows_pad = rp;
    g.epi = T::Epi::glu;
    cudaEvent_t a, b;
    CU(cudaEventCreate(&a));
    CU(cudaEventCreate(&b));
    const auto median = [&](const auto& fn) {
        for (int i = 0; i < 3; ++i) fn();
        std::vector<float> ms;
        for (int i = 0; i < 20; ++i) {
            CU(cudaEventRecord(a));
            fn();
            CU(cudaEventRecord(b));
            CU(cudaEventSynchronize(b));
            float t = 0;
            CU(cudaEventElapsedTime(&t, a, b));
            ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        return static_cast<double>(ms[ms.size() / 2]);
    };
    g.y = y;
    g.y16 = nullptr;
    const double glu32 = median([&] { CU(T::gemm_enqueue(g, nullptr)); });
    const double q32 = median([&] { CU(T::quantize_enqueue(static_cast<const float*>(y), rows, ch, ht, hrs, nullptr)); });
    g.y = nullptr;
    g.y16 = y16;
    const double glu16 = median([&] { CU(T::gemm_enqueue(g, nullptr)); });
    const double q16 = median([&] { CU(T::quantize_enqueue(static_cast<const std::uint16_t*>(y16), rows, ch, ht, hrs, nullptr)); });
    const double saved = (glu32 - glu16) + (q32 - q16);
    std::printf("H16 BENCH rows=%u glu_f32_ms=%.4f glu_h16_ms=%.4f quant_f32_ms=%.4f quant_h16_ms=%.4f saved_ms=%.4f\n", rows, glu32, glu16,
                q32, q16, saved);
    std::printf("H16 OP GATE %s (saved %.4f ms per %u-row FFN; predicted 0.10-0.14, floor 0.05)\n", saved >= 0.05 ? "PASS" : "FAIL", saved,
                rows);
    for (void* p : {static_cast<void*>(w), static_cast<void*>(w2), static_cast<void*>(xt), static_cast<void*>(rs), static_cast<void*>(x),
                    static_cast<void*>(y), static_cast<void*>(y16), static_cast<void*>(ht), static_cast<void*>(hrs)})
        CU(cudaFree(p));
    CU(cudaEventDestroy(a));
    CU(cudaEventDestroy(b));
}

// timing of one shape with random tiles (no reference): median of 20 launches after 3 warm-ups
void bench(const char* name, std::uint32_t out, std::uint32_t in, std::uint32_t rows, bool glu, bool residual = false) {
    const std::uint32_t rp = T::rows_padded(rows);
    std::uint8_t *w = nullptr, *w2 = nullptr, *xt = nullptr;
    float *rs = nullptr, *y = nullptr, *x = nullptr;
    CU(cudaMalloc(&w, T::w_tile_bytes(out, in)));
    CU(cudaMalloc(&w2, glu ? T::w_tile_bytes(out, in) : 16));
    CU(cudaMalloc(&xt, T::x_tile_bytes(rp, in)));
    CU(cudaMalloc(&rs, rp * 4));
    CU(cudaMalloc(&x, static_cast<std::size_t>(rows) * in * 4));
    CU(cudaMalloc(&y, static_cast<std::size_t>(rows) * out * 4));
    CU(cudaMemset(w, 0x35, T::w_tile_bytes(out, in)));
    if (glu) CU(cudaMemset(w2, 0x35, T::w_tile_bytes(out, in)));
    CU(cudaMemset(x, 0x3c, static_cast<std::size_t>(rows) * in * 4));
    CU(T::quantize_enqueue(x, rows, in, xt, rs, nullptr));
    T::GemmArgs g;  // lsu arm (the default feed, explicit for ABAB benchmark)
    g.feed = T::Feed::lsu;
    g.w = w;
    g.w2 = glu ? w2 : nullptr;
    g.x = xt;
    g.rs = rs;
    g.y = y;
    g.out = out;
    g.in = in;
    g.rows = rows;
    g.rows_pad = rp;
    g.epi = glu ? T::Epi::glu : residual ? T::Epi::residual : T::Epi::store;
    if (residual) {  // in place (aux == y), as the engine's down / o / ssm_out GEMMs
        CU(cudaMemset(y, 0, static_cast<std::size_t>(rows) * out * 4));
        g.aux = y;
    }
    cudaEvent_t a, b;
    CU(cudaEventCreate(&a));
    CU(cudaEventCreate(&b));
    T::GemmArgs gb = g;  // Feed::bulk arm, timed launch by launch alternating with the lsu arm (strict ABAB in one process)
    gb.feed = T::Feed::bulk;
    for (int i = 0; i < 3; ++i) {
        CU(T::gemm_enqueue(g, nullptr));
        if (!lsu_only) CU(T::gemm_enqueue(gb, nullptr));
    }
    std::vector<float> ms, msb;
    const auto timed = [&](const T::GemmArgs& ga, std::vector<float>& out_ms) {
        CU(cudaEventRecord(a));
        CU(T::gemm_enqueue(ga, nullptr));
        CU(cudaEventRecord(b));
        CU(cudaEventSynchronize(b));
        float t = 0;
        CU(cudaEventElapsedTime(&t, a, b));
        out_ms.push_back(t);
    };
    for (int i = 0; i < 20; ++i) {
        timed(g, ms);
        if (!lsu_only) timed(gb, msb);
    }
    float qms = 0;
    CU(cudaEventRecord(a));
    CU(T::quantize_enqueue(x, rows, in, xt, rs, nullptr));
    CU(cudaEventRecord(b));
    CU(cudaEventSynchronize(b));
    CU(cudaEventElapsedTime(&qms, a, b));
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2], macs = static_cast<double>(out) * (glu ? 2 : 1) * in * rows;
    std::printf("TILE BENCH %-6s out=%u in=%u rows=%u ms=%.4f TFLOPS=%.1f quant_ms=%.4f\n", name, out * (glu ? 2 : 1), in, rows, med,
                2.0 * macs / (med * 1e-3) / 1e12, qms);
    std::sort(msb.begin(), msb.end());
    const double medb = lsu_only ? 0.0 : msb[msb.size() / 2];
    if (!lsu_only) std::printf("TILE BENCH %-6s feed=bulk out=%u in=%u rows=%u ms=%.4f TFLOPS=%.1f bulk/lsu=%.4f\n", name, out * (glu ? 2 : 1), in,
                rows, medb, 2.0 * macs / (medb * 1e-3) / 1e12, medb / med);
    for (void* p : {static_cast<void*>(w), static_cast<void*>(w2), static_cast<void*>(xt), static_cast<void*>(rs), static_cast<void*>(x),
                    static_cast<void*>(y)})
        CU(cudaFree(p));
    CU(cudaEventDestroy(a));
    CU(cudaEventDestroy(b));
}
}  // namespace

int main(int argc, char** argv) {
    if (!T::supported()) {
        std::printf("nvfp4_tile_device: SKIP (not sm_120)\n");
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "--bench") {
        const auto rows = static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
        lsu_only = argc >= 4 && std::string(argv[3]) == "lsu";
        bench("glu", 17408, 5120, rows, true);   // FFN gate+up fused
        bench("down", 5120, 17408, rows, false);  // FFN down
        bench("qkv", 10240, 5120, rows, false);   // GDN qkv
        bench("o", 5120, 6144, rows, false);      // GDN out / attention o
        bench("kv", 1024, 5120, rows, false);     // attention k (and v)
        bench("down_r", 5120, 17408, rows, false, true);  // FFN down, residual in place (engine form)
        bench("o_r", 5120, 6144, rows, false, true);      // GDN out / attention o, residual in place (engine form)
        bench_norm_quant(rows, 5120, 6144);               // attn / post RMSNorm (hidden rows) + GDN gated RMSNorm (ssm_out input)
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "--bench-h16") {  // the H16 op ticket: correctness first, then the FFN-shape bench
        check_glu_h16(256, 1024, 300, 141);
        check_glu_h16(128, 128, 20, 151);
        check_glu_h16(17408, 5120, 64, 161);  // model shape (64 rows: host reference not needed, the invariants are bitwise)
        if (g_fail) {
            std::printf("nvfp4_tile_device: FAIL (H16 checks; bench refused)\n");
            return 1;
        }
        bench_h16(static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)));
        return 0;
    }
    check_gemm("store", 48, 256, 40, T::Epi::store, 11);
    check_gemm("store", 256, 384, 300, T::Epi::store, 21);
    check_gemm("residual", 128, 512, 257, T::Epi::residual, 41);
    check_gemm("silu_mul", 64, 256, 33, T::Epi::silu_mul, 51);
    check_gemm("store", 1024, 5120, 24, T::Epi::store, 61);  // model K
    check_glu(80, 128, 20, 31);
    check_glu(192, 1024, 300, 71);
    check_gemm("residual", 400, 768, 520, T::Epi::residual, 81);  // bulk feed: partial weight tile (!ok records), 3 x 2 tiles, KS 6
    check_glu(1040, 640, 300, 91);                                // bulk feed: 17 x 64-channel tiles, last partial (!ok w2 slots)
    check_glu_h16(256, 1024, 300, 141);
    check_glu_h16(128, 128, 20, 151);
    check_norm_quant(300, 5120, true, 81);   // model width, padding rows 300..511
    check_norm_quant(257, 384, false, 91);   // null gain, NREG 2 -> bucket 8
    check_norm_quant(64, 8192, true, 101);   // widest fused row (NREG 32)
    check_gated_norm_quant(300, 6144, 111);  // model width (48 heads), padding rows 300..511
    check_gated_norm_quant(257, 384, 121);   // 3 heads: half-filled 256-thread step, NREG 2 -> bucket 8
    check_gated_norm_quant(64, 8192, 131);   // widest fused row (64 heads, NREG 32)
    std::printf("nvfp4_tile_device: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
