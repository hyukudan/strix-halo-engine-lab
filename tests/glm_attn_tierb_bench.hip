// Tier B bench, strix_gemv at the GLM-5.3-Flash attention EXL3 shapes (per TP2 rank, K6 mul1), R = 1..8,
// cold (rotated weight copies, working set >> the 32 MB MALL). Synthetic trellis bytes (any bytes are a valid trellis;
// the kernel's work does not depend on the values), F16 suh / svh near 1. Timing only; no exactness claim here.
// Also: the KDA front at R = 1 with the f_a / g_a / b_proj chains serial vs on a second stream (overlap probe).
// usage: glm_attn_tierb_bench
#include "glm/attn/glm_attn_fast.hpp"

#include <strix_trellis.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

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
        std::exit(4);
    }
}
constexpr double kCeilGBs = 246.7;

struct Shape {
    const char* name;
    unsigned N, I, per_layer, layers;  // per_layer: gemv calls of this shape per layer; layers: count in the model
};

strix_tensor_desc tensor(float* p, unsigned rows, unsigned cols) {
    strix_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.abi_version = STRIX_TRELLIS_ABI_VERSION;
    t.data = p;
    t.buffer_bytes = static_cast<std::uint64_t>(rows) * cols * 4;
    t.dtype = STRIX_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.strides[0] = cols;
    t.strides[1] = 1;
    return t;
}

strix_weights make_k6(strix_context ctx, const strix_stream* st, unsigned N, unsigned I, std::mt19937& g) {
    const std::uint64_t tb = static_cast<std::uint64_t>(N) * I * 6 / 8;
    std::vector<std::uint8_t> trellis(tb);
    std::vector<std::uint16_t> suh(I), svh(N);
    for (auto& b : trellis) b = static_cast<std::uint8_t>(g());
    for (auto& v : suh) v = static_cast<std::uint16_t>(0x3c00U | (g() & 0x8000U));
    for (auto& v : svh) v = static_cast<std::uint16_t>(0x2000U + (g() & 0xffU));
    strix_exl3_parts_desc pd{};
    pd.struct_size = sizeof(pd);
    pd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    pd.transport_version = 1;
    pd.bits_x2 = 12;
    pd.rows = N;
    pd.columns = I;
    pd.codebook = STRIX_CODEBOOK_MUL1;
    pd.marker = STRIX_EXL3_MARKER_MUL1;
    pd.trellis = {trellis.data(), tb};
    pd.suh = {suh.data(), suh.size() * 2};
    pd.svh = {svh.data(), svh.size() * 2};
    strix_weights w{};
    sok(strix_weights_prepare_parts(ctx, &pd, st, &w), "prepare");
    return w;
}

// KDA layer front at R = 1: the three q/k/v gemv (bandwidth-bound) vs the f_a|g_a|b_proj R1 chains (latency-bound),
// serial on one stream vs the chains on a second stream joined by events. Same kernels, same bits either way.
void overlap_probe(strix_context ctx, hipStream_t s1, hipStream_t s2, std::mt19937& g) {
    namespace F = strix::glm::fast;
    constexpr unsigned L = 12, LD = 34, N = 4096, I = 4096;  // 12 x 3 gemv copies (453 MB), 34 dense copies (80 MB)
    strix_stream ss1{sizeof(strix_stream), STRIX_TRELLIS_ABI_VERSION, s1, {}};
    std::vector<strix_weights> w(3 * L);
    for (auto& ww : w) ww = make_k6(ctx, &ss1, N, I, g);
    std::vector<std::uint16_t*> dw(LD);
    std::vector<std::uint16_t> host(static_cast<std::size_t>(288) * I);
    for (auto& p : dw) {
        for (auto& v : host) v = static_cast<std::uint16_t>(0x2000U + (g() & 0x0fffU));
        hok(hipMalloc(&p, host.size() * 2), "m");
        hok(hipMemcpy(p, host.data(), host.size() * 2, hipMemcpyHostToDevice), "c");
    }
    float *x = nullptr, *y = nullptr, *yd = nullptr;
    hok(hipMalloc(&x, I * 4), "m");
    hok(hipMalloc(&y, 3 * N * 4), "m");
    hok(hipMalloc(&yd, 288 * 4), "m");
    hok(hipMemset(x, 0, I * 4), "m");
    std::vector<strix_matmul_desc> d(3 * L);
    std::uint64_t ws = 0;
    for (unsigned c = 0; c < 3 * L; ++c) {
        strix_matmul_desc_init(&d[c]);
        d[c].weights = w[c];
        d[c].input = tensor(x, 1, I);
        d[c].output = tensor(y + (c % 3) * N, 1, N);
        std::uint64_t b = 0;
        sok(strix_matmul_workspace_bytes(ctx, &d[c], &b), "ws");
        ws = std::max(ws, b);
    }
    void* wsp = nullptr;
    if (ws) hok(hipMalloc(&wsp, ws), "m");
    for (auto& dd : d) {
        dd.workspace = wsp;
        dd.workspace_bytes = ws;
    }
    auto dense = [&](unsigned l, hipStream_t s) {
        const F::DenseJob j3[3] = {{dw[l], x, yd, 128, I, I, 128}, {dw[l] + 128 * I, x, yd + 128, 128, I, I, 128},
                                   {dw[l] + 256 * I, x, yd + 256, 32, I, I, 32}};
        F::dense_f16(j3, 3, 1, s);
    };
    auto gemv3 = [&](unsigned l) {
        for (unsigned q = 0; q < 3; ++q) sok(strix_gemv(ctx, &d[3 * l + q], &ss1), "gemv");
    };
    hipEvent_t a, b, e1, e2;
    for (hipEvent_t* e : {&a, &b}) hok(hipEventCreate(e), "e");
    for (hipEvent_t* e : {&e1, &e2}) hok(hipEventCreateWithFlags(e, hipEventDisableTiming), "e");
    auto run = [&](int mode) {  // 0 gemv3, 1 dense, 2 serial both, 3 overlapped
        for (unsigned l = 0; l < L; ++l) {  // warm-up pass
            if (mode != 1) gemv3(l);
            if (mode != 0) dense(l % LD, s1);
        }
        hok(hipDeviceSynchronize(), "s");
        const unsigned iters = 8 * L;
        hok(hipEventRecord(a, s1), "r");
        for (unsigned it = 0; it < iters; ++it) {
            const unsigned l = it % L, ld = it % LD;
            if (mode == 3) {
                hok(hipEventRecord(e1, s1), "r");
                hok(hipStreamWaitEvent(s2, e1, 0), "w");
                dense(ld, s2);
                gemv3(l);
                hok(hipEventRecord(e2, s2), "r");
                hok(hipStreamWaitEvent(s1, e2, 0), "w");
            } else {
                if (mode != 1) gemv3(l);
                if (mode != 0) dense(ld, s1);
            }
        }
        hok(hipEventRecord(b, s1), "r");
        hok(hipEventSynchronize(b), "s");
        float ms = 0;
        hok(hipEventElapsedTime(&ms, a, b), "t");
        return ms * 1000.0 / iters;
    };
    const double t_g = run(0), t_d = run(1);
    double t_s[3], t_o[3];
    for (unsigned k = 0; k < 3; ++k) {  // serial / overlapped alternated x3 (a scheduling probe, not a model measurement)
        t_s[k] = run(2);
        t_o[k] = run(3);
    }
    std::printf("OVERLAP probe, KDA front R=1: qkv gemv x3 %.2f us, f_a|g_a|b_proj chains %.2f us; serial %.2f/%.2f/%.2f us, two streams %.2f/%.2f/%.2f us\n",
                t_g, t_d, t_s[0], t_s[1], t_s[2], t_o[0], t_o[1], t_o[2]);
    std::fflush(stdout);
    for (auto& ww : w) sok(strix_weights_destroy(ww), "destroy");
    for (auto& p : dw) hok(hipFree(p), "f");
    for (float* p : {x, y, yd}) hok(hipFree(p), "f");
    if (wsp) hok(hipFree(wsp), "f");
}

}  // namespace

int main() {
    strix_context ctx{};
    strix_context_desc cd{};
    cd.struct_size = sizeof(cd);
    cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    sok(strix_context_create(&cd, &ctx), "context");
    const std::vector<Shape> shapes = {
        {"kda.q|k|v (each)", 4096, 4096, 3, 34}, {"kda.o_proj", 4096, 4096, 1, 34}, {"dsa.q_a", 1536, 4096, 1, 11},
        {"dsa.kv_a", 512, 4096, 1, 11},          {"dsa.q_b", 8192, 1536, 1, 11},    {"dsa.wq_b", 4096, 1536, 1, 11},
        {"dsa.o_proj", 4096, 8192, 1, 11},
    };
    std::mt19937 g(7);
    hipStream_t s1, s2;
    hok(hipStreamCreate(&s1), "stream");
    hok(hipStreamCreate(&s2), "stream");
    strix_stream ss1{sizeof(strix_stream), STRIX_TRELLIS_ABI_VERSION, s1, {}};
    double total_ms[9] = {};
    for (const Shape& sh : shapes) {
        const std::uint64_t tb = static_cast<std::uint64_t>(sh.N) * sh.I * 6 / 8;  // K6 trellis bytes
        const unsigned copies = static_cast<unsigned>(std::max<std::uint64_t>(4, ((512ULL << 20) + tb - 1) / tb));  // >= 512 MiB rotated
        std::vector<std::uint8_t> trellis(tb);
        std::vector<std::uint16_t> suh(sh.I), svh(sh.N);
        std::vector<strix_weights> w(copies);
        for (unsigned c = 0; c < copies; ++c) {
            for (auto& b : trellis) b = static_cast<std::uint8_t>(g());
            for (auto& v : suh) v = static_cast<std::uint16_t>(0x3c00U | (g() & 0x8000U));  // +-1
            for (auto& v : svh) v = static_cast<std::uint16_t>(0x2000U + (g() & 0xffU));    // ~2^-7
            strix_exl3_parts_desc pd{};
            pd.struct_size = sizeof(pd);
            pd.abi_version = STRIX_TRELLIS_ABI_VERSION;
            pd.transport_version = 1;
            pd.bits_x2 = 12;
            pd.rows = sh.N;
            pd.columns = sh.I;
            pd.codebook = STRIX_CODEBOOK_MUL1;
            pd.marker = STRIX_EXL3_MARKER_MUL1;
            pd.trellis = {trellis.data(), tb};
            pd.suh = {suh.data(), suh.size() * 2};
            pd.svh = {svh.data(), svh.size() * 2};
            sok(strix_weights_prepare_parts(ctx, &pd, &ss1, &w[c]), "prepare");
        }
        float *x = nullptr, *y = nullptr;
        hok(hipMalloc(&x, 8ULL * sh.I * 4), "m");
        hok(hipMalloc(&y, 8ULL * sh.N * 4), "m");
        hok(hipMemset(x, 0, 8ULL * sh.I * 4), "m");
        for (unsigned R : {1U, 2U, 4U, 8U}) {
            std::vector<strix_matmul_desc> d(copies);
            std::uint64_t ws = 0;
            for (unsigned c = 0; c < copies; ++c) {
                strix_matmul_desc_init(&d[c]);
                d[c].weights = w[c];
                d[c].input = tensor(x, R, sh.I);
                d[c].output = tensor(y, R, sh.N);
                std::uint64_t b = 0;
                sok(strix_matmul_workspace_bytes(ctx, &d[c], &b), "ws");
                ws = std::max(ws, b);
            }
            void* wsp = nullptr;
            if (ws) hok(hipMalloc(&wsp, ws), "m");
            for (auto& dd : d) {
                dd.workspace = wsp;
                dd.workspace_bytes = ws;
            }
            hipEvent_t a, b;
            hok(hipEventCreate(&a), "e");
            hok(hipEventCreate(&b), "e");
            for (unsigned c = 0; c < copies; ++c) sok(strix_gemv(ctx, &d[c], &ss1), "gemv warm");
            hok(hipStreamSynchronize(s1), "s");
            const unsigned iters = copies * 4;
            hok(hipEventRecord(a, s1), "r");
            for (unsigned it = 0; it < iters; ++it) sok(strix_gemv(ctx, &d[it % copies], &ss1), "gemv");
            hok(hipEventRecord(b, s1), "r");
            hok(hipEventSynchronize(b), "s");
            float ms = 0;
            hok(hipEventElapsedTime(&ms, a, b), "t");
            const double us = ms * 1000.0 / iters;
            const double bytes = static_cast<double>(tb) + (sh.I + sh.N) * 2.0;
            const double gbs = bytes / (us * 1e3);
            std::printf("TIERB %-18s N=%5u I=%5u R=%u  %8.2f us  %6.1f GB/s  %5.1f %% of ceiling (floor %.2f us)  x%u/layer x%u layers\n",
                        sh.name, sh.N, sh.I, R, us, gbs, 100.0 * gbs / kCeilGBs, bytes / (kCeilGBs * 1e3), sh.per_layer, sh.layers);
            std::fflush(stdout);
            total_ms[R] += us * sh.per_layer * sh.layers / 1000.0;
            hok(hipEventDestroy(a), "e");
            hok(hipEventDestroy(b), "e");
            if (wsp) hok(hipFree(wsp), "f");
        }
        hok(hipFree(x), "f");
        hok(hipFree(y), "f");
        for (auto& ww : w) sok(strix_weights_destroy(ww), "destroy");
    }
    for (unsigned R : {1U, 2U, 4U, 8U})
        std::printf("TIERB attention EXL3 sum per rank, R=%u rows: %.3f ms per R-row step (%.3f ms per row)\n", R, total_ms[R], total_ms[R] / R);
    overlap_probe(ctx, s1, s2, g);
    sok(strix_context_destroy(ctx), "ctx destroy");
    return 0;
}
