// Device fixture + operator bench of the NVFP4 tile GEMM (sm_120; run only through an operator ticket on device 0).
//   tile_device            : quantiser bytes == host reference (bitwise), GEMM store / residual / silu_mul / glu within the FP32
//                            accumulation bound of the double reference, repeated launches bitwise identical.
//   tile_device --bench R  : model shapes of the 27B dense layer at R activation rows, median of 20 timed launches per shape.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strix/cuda/core/nvfp4_tile_cuda.hpp"
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
    T::GemmArgs g;
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
    const bool ok = qsame && det && worst <= 1.0;
    std::printf("TILE CHECK %-10s out=%u in=%u rows=%u quant-bitwise=%d deterministic=%d err/bound=%.3f %s\n", name, out, in, rows,
                qsame ? 1 : 0, det ? 1 : 0, worst, ok ? "PASS" : "FAIL");
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
    g.rows_pad = ha.rows_pad;
    g.epi = T::Epi::glu;
    CU(T::gemm_enqueue(g, nullptr));
    const auto y = down(dy, static_cast<std::size_t>(rows) * ch);
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
    const bool ok = worst <= 1.0;
    std::printf("TILE CHECK %-10s ch=%u in=%u rows=%u err/bound=%.3f %s\n", "glu", ch, in, rows, worst, ok ? "PASS" : "FAIL");
    g_fail += ok ? 0 : 1;
    for (void* p : {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(tg), static_cast<void*>(tu),
                    static_cast<void*>(dxt), static_cast<void*>(dx), static_cast<void*>(drs), static_cast<void*>(dy)})
        CU(cudaFree(p));
}

// timing of one shape with random tiles (no reference): median of 20 launches after 3 warm-ups
void bench(const char* name, std::uint32_t out, std::uint32_t in, std::uint32_t rows, bool glu) {
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
    T::GemmArgs g;
    g.w = w;
    g.w2 = glu ? w2 : nullptr;
    g.x = xt;
    g.rs = rs;
    g.y = y;
    g.out = out;
    g.in = in;
    g.rows = rows;
    g.rows_pad = rp;
    g.epi = glu ? T::Epi::glu : T::Epi::store;
    cudaEvent_t a, b;
    CU(cudaEventCreate(&a));
    CU(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) CU(T::gemm_enqueue(g, nullptr));
    std::vector<float> ms;
    for (int i = 0; i < 20; ++i) {
        CU(cudaEventRecord(a));
        CU(T::gemm_enqueue(g, nullptr));
        CU(cudaEventRecord(b));
        CU(cudaEventSynchronize(b));
        float t = 0;
        CU(cudaEventElapsedTime(&t, a, b));
        ms.push_back(t);
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
        bench("glu", 17408, 5120, rows, true);   // FFN gate+up fused
        bench("down", 5120, 17408, rows, false);  // FFN down
        bench("qkv", 10240, 5120, rows, false);   // GDN qkv
        bench("o", 5120, 6144, rows, false);      // GDN out / attention o
        bench("kv", 1024, 5120, rows, false);     // attention k (and v)
        return 0;
    }
    check_gemm("store", 48, 256, 40, T::Epi::store, 11);
    check_gemm("store", 256, 384, 300, T::Epi::store, 21);
    check_gemm("residual", 128, 512, 257, T::Epi::residual, 41);
    check_gemm("silu_mul", 64, 256, 33, T::Epi::silu_mul, 51);
    check_gemm("store", 1024, 5120, 24, T::Epi::store, 61);  // model K
    check_glu(80, 128, 20, 31);
    check_glu(192, 1024, 300, 71);
    std::printf("nvfp4_tile_device: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
