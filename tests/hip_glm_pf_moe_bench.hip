// GLM prefill MoE v1 bench (per rank): E = 288 experts, K4 EXL3, rank slice I_local = 1024 (gate/up 4096 -> 1024,
// down 1024 -> 4096), top-8 uniform random routes, chunk C rows. Per expert: gather rows (ascending token order) ->
// strix_gemm_pinned gate, up -> R15 swiglu -> strix_gemm_pinned down -> scatter into the route slot -> R16 combine.
// Synthetic weights (random trellis bits, +-1 suh/svh scales): timing only, no correctness claim beyond determinism
// (two passes bitwise equal). Reports ms per chunk and effective TOPS (25.17 MFLOP per routed row-expert).
// usage: hip_glm_pf_moe_bench [streams]
#include "glm/kernels/glm_tier_a.hpp"
#include "strix_trellis.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace strix::glm;
namespace D = strix::glm::dev;

namespace {

void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
void sok(strix_status s, const char* w) {
    if (s != STRIX_SUCCESS) {
        std::fprintf(stderr, "%s: %s\n", w, strix_status_string(s));
        std::exit(3);
    }
}

constexpr unsigned kE = 288, kTop = 8, kI = 1024, kH = 4096, kBits = 4;

strix_weights make_w(strix_context c, strix_stream* st, unsigned N, unsigned K, std::mt19937& g) {
    std::vector<std::uint16_t> tr(static_cast<std::size_t>(K / 16) * (N / 16) * 16 * kBits);
    for (auto& v : tr) v = static_cast<std::uint16_t>(g());
    std::vector<std::uint16_t> suh(K), svh(N);
    const _Float16 one = 1.0F, mone = -1.0F;
    for (auto& v : suh) std::memcpy(&v, (g() & 1) ? &one : &mone, 2);
    const _Float16 sc = 0.02F;
    for (auto& v : svh) std::memcpy(&v, &sc, 2);
    strix_exl3_parts_desc p{};
    p.struct_size = sizeof(p);
    p.abi_version = STRIX_TRELLIS_ABI_VERSION;
    p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    p.bits_x2 = 2 * kBits;
    p.rows = N;
    p.columns = K;
    p.codebook = STRIX_CODEBOOK_MUL1;
    p.marker = STRIX_EXL3_MARKER_MUL1;
    p.trellis = {tr.data(), tr.size() * 2};
    p.suh = {suh.data(), suh.size() * 2};
    p.svh = {svh.data(), svh.size() * 2};
    strix_weights w{};
    sok(strix_weights_prepare_parts(c, &p, st, &w), "prepare");
    return w;
}

strix_matmul_desc desc(strix_weights w, unsigned N, unsigned K, const float* x, float* y, unsigned rows, void* ws, std::uint64_t wsb) {
    strix_matmul_desc d;
    strix_matmul_desc_init(&d);
    d.weights = w;
    d.input = {sizeof(strix_tensor_desc), STRIX_TRELLIS_ABI_VERSION, const_cast<float*>(x), static_cast<std::uint64_t>(rows) * K * 4, STRIX_DTYPE_F32, 2,
               {rows, K}, {K, 1}, {}};
    d.output = {sizeof(strix_tensor_desc), STRIX_TRELLIS_ABI_VERSION, y, static_cast<std::uint64_t>(rows) * N * 4, STRIX_DTYPE_F32, 2, {rows, N}, {N, 1}, {}};
    d.workspace = ws;
    d.workspace_bytes = wsb;
    return d;
}

__global__ void gather_kernel(const float* x, const std::uint32_t* rows, float* xg, unsigned K) {
    const unsigned r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i < K) xg[static_cast<std::size_t>(r) * K + i] = x[static_cast<std::size_t>(rows[r]) * K + i];
}
// eo[row][slot] = y_e[r] for the group's (row, slot) pairs.
__global__ void scatter_kernel(const float* y, const std::uint32_t* dst, float* eo) {
    const unsigned r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i < kH) eo[static_cast<std::size_t>(dst[r]) * kH + i] = y[static_cast<std::size_t>(r) * kH + i];
}

}  // namespace

int main(int argc, char** argv) {
    const unsigned nstreams = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 4;
    strix_context c{};
    strix_context_desc cd{};
    cd.struct_size = sizeof(cd);
    cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    sok(strix_context_create(&cd, &c), "ctx");
    if (nstreams != 1 && nstreams != 4) {
        std::fprintf(stderr, "streams must be 1 or 4\n");
        return 2;
    }
    int status = 0;
    std::vector<hipStream_t> hs(nstreams);
    std::vector<strix_stream> st(nstreams);
    for (unsigned i = 0; i < nstreams; ++i) {
        hok(hipStreamCreate(&hs[i]), "stream");
        st[i] = {sizeof(strix_stream), STRIX_TRELLIS_ABI_VERSION, hs[i], {}};
    }
    std::mt19937 g(7);
    std::vector<strix_weights> wg(kE), wu(kE), wd(kE);
    for (unsigned e = 0; e < kE; ++e) {
        wg[e] = make_w(c, &st[0], kI, kH, g);
        wu[e] = make_w(c, &st[0], kI, kH, g);
        wd[e] = make_w(c, &st[0], kH, kI, g);
    }
    // pinned splits: the library's own choice at M = 1024 for each shape (read from its workspace size)
    float *qx = nullptr, *qy = nullptr;  // real buffers for the workspace queries
    hok(hipMalloc(&qx, std::size_t{1024} * kH * 4), "m");
    hok(hipMalloc(&qy, std::size_t{1024} * kH * 4), "m");
    auto lib_splits = [&](strix_weights w, unsigned N, unsigned K) {
        std::uint64_t ws = 0;
        auto d = desc(w, N, K, qx, qy, 1024, nullptr, 0);
        sok(strix_matmul_workspace_bytes(c, &d, &ws), "ws");
        return static_cast<unsigned>((ws - ((static_cast<std::uint64_t>(1024) * K * 2 + 255) & ~std::uint64_t{255})) / (std::uint64_t{1024} * N * 4));
    };
    const unsigned s_gu = lib_splits(wg[0], kI, kH), s_d = lib_splits(wd[0], kH, kI);
    hok(hipFree(qx), "f");
    hok(hipFree(qy), "f");
    std::printf("pinned splits: gate/up %u, down %u; streams %u\n", s_gu, s_d, nstreams);
    for (unsigned C : {512U, 1024U, 2048U, 4096U}) {
        // routes: 8 distinct experts per row, uniform
        std::vector<std::uint32_t> ids(static_cast<std::size_t>(C) * kTop);
        std::vector<unsigned> perm(kE);
        std::iota(perm.begin(), perm.end(), 0U);
        for (unsigned r = 0; r < C; ++r) {
            for (unsigned k = 0; k < kTop; ++k) std::swap(perm[k], perm[k + g() % (kE - k)]);
            std::vector<std::uint32_t> pick(perm.begin(), perm.begin() + kTop);
            std::sort(pick.begin(), pick.end());
            std::copy(pick.begin(), pick.end(), ids.begin() + static_cast<std::size_t>(r) * kTop);
        }
        // groups: rows ascending per expert; dst = row * 8 + slot
        std::vector<std::vector<std::uint32_t>> rows(kE), dst(kE);
        for (unsigned r = 0; r < C; ++r)
            for (unsigned k = 0; k < kTop; ++k) {
                const unsigned e = ids[static_cast<std::size_t>(r) * kTop + k];
                rows[e].push_back(r);
                dst[e].push_back(r * kTop + k);
            }
        unsigned gmax = 0;
        for (auto& v : rows) gmax = std::max<unsigned>(gmax, static_cast<unsigned>(v.size()));
        std::vector<std::uint32_t> flat_rows, flat_dst;
        std::vector<unsigned> off(kE + 1, 0);
        for (unsigned e = 0; e < kE; ++e) {
            off[e + 1] = off[e] + static_cast<unsigned>(rows[e].size());
            flat_rows.insert(flat_rows.end(), rows[e].begin(), rows[e].end());
            flat_dst.insert(flat_dst.end(), dst[e].begin(), dst[e].end());
        }
        std::uint32_t *drows = nullptr, *ddst = nullptr;
        hok(hipMalloc(&drows, flat_rows.size() * 4), "m");
        hok(hipMalloc(&ddst, flat_dst.size() * 4), "m");
        hok(hipMemcpy(drows, flat_rows.data(), flat_rows.size() * 4, hipMemcpyHostToDevice), "c");
        hok(hipMemcpy(ddst, flat_dst.data(), flat_dst.size() * 4, hipMemcpyHostToDevice), "c");
        std::normal_distribution<float> nd(0.0F, 1.0F);
        std::vector<float> xh(static_cast<std::size_t>(C) * kH), wh(static_cast<std::size_t>(C) * kTop);
        for (auto& v : xh) v = nd(g);
        for (auto& v : wh) v = 0.125F;
        float *x, *w, *eo, *y, *sh, *xg, *gg, *uu, *aa, *yd;
        const std::size_t R8 = static_cast<std::size_t>(C) * kTop;
        hok(hipMalloc(&x, xh.size() * 4), "m");
        hok(hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice), "c");
        hok(hipMalloc(&w, wh.size() * 4), "m");
        hok(hipMemcpy(w, wh.data(), wh.size() * 4, hipMemcpyHostToDevice), "c");
        hok(hipMalloc(&eo, R8 * kH * 4), "m");
        hok(hipMalloc(&y, static_cast<std::size_t>(C) * kH * 4), "m");
        hok(hipMalloc(&sh, static_cast<std::size_t>(C) * kH * 4), "m");
        hok(hipMemset(sh, 0, static_cast<std::size_t>(C) * kH * 4), "z");
        hok(hipMalloc(&xg, R8 * kH * 4), "m");  // grouped rows, all experts back to back
        hok(hipMalloc(&gg, R8 * kI * 4), "m");
        hok(hipMalloc(&uu, R8 * kI * 4), "m");
        hok(hipMalloc(&aa, R8 * kI * 4), "m");
        hok(hipMalloc(&yd, R8 * kH * 4), "m");
        // one workspace per stream, sized for the largest group
        std::vector<void*> ws(nstreams);
        std::uint64_t wsb = 0;
        {
            std::uint64_t a = 0, b = 0;
            auto d1 = desc(wg[0], kI, kH, xg, gg, gmax, nullptr, 0);
            auto d2 = desc(wd[0], kH, kI, aa, yd, gmax, nullptr, 0);
            sok(strix_gemm_pinned_workspace_bytes(c, &d1, s_gu, &a), "ws");
            sok(strix_gemm_pinned_workspace_bytes(c, &d2, s_d, &b), "ws");
            wsb = std::max(a, b);
        }
        for (auto& p : ws) hok(hipMalloc(&p, wsb), "ws");
        hipEvent_t ev0, ev1;
        hok(hipEventCreate(&ev0), "e");
        hok(hipEventCreate(&ev1), "e");
        std::vector<hipEvent_t> done(nstreams);
        for (auto& e : done) hok(hipEventCreate(&e), "e");
        auto run = [&]() {
            hok(hipEventRecord(ev0, hs[0]), "e");
            for (unsigned i = 1; i < nstreams; ++i) hok(hipStreamWaitEvent(hs[i], ev0, 0), "w");
            for (unsigned e = 0; e < kE; ++e) {
                const unsigned M = off[e + 1] - off[e];
                if (!M) continue;
                const unsigned si = e % nstreams;
                float *xe = xg + static_cast<std::size_t>(off[e]) * kH, *ge = gg + static_cast<std::size_t>(off[e]) * kI;
                float *ue = uu + static_cast<std::size_t>(off[e]) * kI, *ae = aa + static_cast<std::size_t>(off[e]) * kI;
                float* ye = yd + static_cast<std::size_t>(off[e]) * kH;
                gather_kernel<<<dim3(kH / 256, M), 256, 0, hs[si]>>>(x, drows + off[e], xe, kH);
                auto dg = desc(wg[e], kI, kH, xe, ge, M, ws[si], wsb);
                sok(strix_gemm_pinned(c, &dg, s_gu, &st[si]), "gate");
                auto du = desc(wu[e], kI, kH, xe, ue, M, ws[si], wsb);
                sok(strix_gemm_pinned(c, &du, s_gu, &st[si]), "up");
                D::swiglu(ge, ue, ae, static_cast<std::size_t>(M) * kI, hs[si]);
                auto dd = desc(wd[e], kH, kI, ae, ye, M, ws[si], wsb);
                sok(strix_gemm_pinned(c, &dd, s_d, &st[si]), "down");
                scatter_kernel<<<dim3(kH / 256, M), 256, 0, hs[si]>>>(ye, ddst + off[e], eo);
            }
            for (unsigned i = 1; i < nstreams; ++i) {
                hok(hipEventRecord(done[i], hs[i]), "e");
                hok(hipStreamWaitEvent(hs[0], done[i], 0), "w");
            }
            D::moe_combine(w, eo, sh, y, C, hs[0]);
            hok(hipEventRecord(ev1, hs[0]), "e");
            hok(hipEventSynchronize(ev1), "s");
            float ms = 0;
            hok(hipEventElapsedTime(&ms, ev0, ev1), "t");
            return ms;
        };
        run();
        std::vector<float> first(static_cast<std::size_t>(C) * kH);
        hok(hipMemcpy(first.data(), y, first.size() * 4, hipMemcpyDeviceToHost), "d");
        float best = 1e30F, sum = 0;
        for (int rep = 0; rep < 3; ++rep) {
            const float ms = run();
            best = std::min(best, ms);
            sum += ms;
        }
        std::vector<float> again(first.size());
        hok(hipMemcpy(again.data(), y, again.size() * 4, hipMemcpyDeviceToHost), "d");
        bool det = std::memcmp(first.data(), again.data(), first.size() * 4) == 0;
        for (float v : again) det = det && std::isfinite(v);
        if (!det) status = 1;
        const double flop = static_cast<double>(C) * kTop * 2.0 * (2.0 * kH * kI + static_cast<double>(kI) * kH);
        std::printf("TIME moe C=%u (mean group %.1f, max %u): best %.3f ms, mean %.3f ms, %.1f TOPS (best); deterministic+finite %s\n", C,
                    static_cast<double>(C) * kTop / kE, gmax, best, sum / 3, flop / best / 1e9, det ? "yes" : "NO");
        for (void* p : ws) hok(hipFree(p), "f");
        for (void* p : {static_cast<void*>(drows), static_cast<void*>(ddst), static_cast<void*>(x), static_cast<void*>(w), static_cast<void*>(eo),
                        static_cast<void*>(y), static_cast<void*>(sh), static_cast<void*>(xg), static_cast<void*>(gg), static_cast<void*>(uu),
                        static_cast<void*>(aa), static_cast<void*>(yd)})
            hok(hipFree(p), "f");
    }
    return status;
}
