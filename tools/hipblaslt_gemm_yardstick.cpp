// Measurement only (not linked into any engine): the vendor GEMM (hipBLASLt) at our dense shapes on gfx1151, as the yardstick
// for our IU4 W4A16 core (tools/hip_dense_iu4_probe.hip cmp). D[rows][n] = X[rows][in] * W[n][in]^T (TN, column-major
// m = n, n = rows, k = in), compute F32. Arms per shape: f16 (A/B FP16, D FP32 then D FP16), bf16 (D FP32), and an
// int4-A probe (A HIP_R_4I, B FP16) that only asks the heuristic whether any solution exists.
// For each arm: the heuristic's top 8 algos are timed (3 warm + 10 timed, events) and the best is printed.
// Usage: hipblaslt-gemm-yardstick <rows> in:n [in:n ...]
// Build (host C++, needs hipBLASLt): g++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -I$ROCM/include tools/hipblaslt_gemm_yardstick.cpp
//   -o hipblaslt-gemm-yardstick -L$ROCM/lib -lhipblaslt -lamdhip64 -Wl,-rpath,$ROCM/lib
// Env: HB_ARMS=<bit mask of the arms, default 15>, HB_NAMES=1 prints the winning kernel name.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-version.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(1); } } while (0)
#define LT(x) do { const hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { std::printf("hipblaslt status %d at %d\n", int(s_), __LINE__); std::exit(1); } } while (0)
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Arm { const char* name; hipDataType a, b, d; };
int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: hipblaslt-gemm-yardstick <rows> in:n ...\n"); return 2; }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const unsigned rows = unsigned(std::atoi(argv[1]));
    const unsigned mask = std::getenv("HB_ARMS") ? unsigned(std::atoi(std::getenv("HB_ARMS"))) : 15U;  // bit i = arms[i]
    std::printf("hipblaslt %d.%d.%d rows %u\n", HIPBLASLT_VERSION_MAJOR, HIPBLASLT_VERSION_MINOR, HIPBLASLT_VERSION_PATCH, rows);
    std::size_t max_w = 0, max_x = 0, max_d = 0;
    std::vector<std::pair<unsigned, unsigned>> shapes;
    for (int i = 2; i < argc; ++i) {
        unsigned in = 0, n = 0;
        if (std::sscanf(argv[i], "%u:%u", &in, &n) != 2) return 2;
        shapes.emplace_back(in, n);
        max_w = std::max(max_w, std::size_t(in) * n * 2);
        max_x = std::max(max_x, std::size_t(in) * rows * 2);
        max_d = std::max(max_d, std::size_t(n) * rows * 4);
    }
    const double t0 = now_ms();
    hipblasLtHandle_t h;
    LT(hipblasLtCreate(&h));
    std::printf("hipblasLtCreate %.1f ms\n", now_ms() - t0);
    void *w, *x, *d, *ws;
    const std::size_t ws_bytes = std::size_t(64) << 20;
    CK(hipMalloc(&w, max_w)); CK(hipMalloc(&x, max_x)); CK(hipMalloc(&d, max_d)); CK(hipMalloc(&ws, ws_bytes));
    CK(hipMemset(w, 0x11, max_w)); CK(hipMemset(x, 0x22, max_x));  // small finite FP16 / BF16 values
    hipEvent_t ea, eb;
    CK(hipEventCreate(&ea)); CK(hipEventCreate(&eb));
    const Arm arms[] = {{"f16-d32", HIP_R_16F, HIP_R_16F, HIP_R_32F}, {"f16-d16", HIP_R_16F, HIP_R_16F, HIP_R_16F},
                        {"bf16-d32", HIP_R_16BF, HIP_R_16BF, HIP_R_32F}, {"i4a-f16b", HIP_R_4I, HIP_R_16F, HIP_R_16F}};
    bool first = true;
    for (const auto& [in, n] : shapes) {
        const double gf = 2.0 * rows * double(in) * n;
        for (unsigned ai = 0; ai < 4U; ++ai) {
            if (!(mask & (1U << ai))) continue;
            const Arm& arm = arms[ai];
            const double c0 = now_ms();
            hipblasLtMatmulDesc_t md;
            LT(hipblasLtMatmulDescCreate(&md, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
            LT(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
            LT(hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
            hipblasLtMatrixLayout_t la, lb, lc;
            LT(hipblasLtMatrixLayoutCreate(&la, arm.a, in, n, in));    // stored k x m (op T)
            LT(hipblasLtMatrixLayoutCreate(&lb, arm.b, in, rows, in)); // k x n
            LT(hipblasLtMatrixLayoutCreate(&lc, arm.d, n, rows, n));   // m x n
            hipblasLtMatmulPreference_t pref;
            LT(hipblasLtMatmulPreferenceCreate(&pref));
            LT(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
            hipblasLtMatmulHeuristicResult_t res[8];
            int got = 0;
            const hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(h, md, la, lb, lc, lc, pref, 8, res, &got);
            const double heur_ms = now_ms() - c0;
            if (hs != HIPBLAS_STATUS_SUCCESS || got == 0) {
                std::printf("hb in %5u n %5u rows %5u %-9s NO SOLUTION (status %d, algos %d)\n", in, n, rows, arm.name, int(hs), got);
            } else {
                const float alpha = 1.0F, beta = 0.0F;
                double best = 1e30;
                int best_i = -1;
                double first_call_ms = 0;
                for (int i = 0; i < got; ++i) {
                    if (res[i].workspaceSize > ws_bytes) continue;
                    const double f0 = now_ms();
                    if (hipblasLtMatmul(h, md, &alpha, w, la, x, lb, &beta, d, lc, d, lc, &res[i].algo, ws, ws_bytes, nullptr) !=
                        HIPBLAS_STATUS_SUCCESS) continue;
                    CK(hipDeviceSynchronize());
                    if (i == 0) first_call_ms = now_ms() - f0;
                    for (int r = 0; r < 2; ++r)
                        LT(hipblasLtMatmul(h, md, &alpha, w, la, x, lb, &beta, d, lc, d, lc, &res[i].algo, ws, ws_bytes, nullptr));
                    CK(hipEventRecord(ea, nullptr));
                    for (int r = 0; r < 10; ++r)
                        LT(hipblasLtMatmul(h, md, &alpha, w, la, x, lb, &beta, d, lc, d, lc, &res[i].algo, ws, ws_bytes, nullptr));
                    CK(hipEventRecord(eb, nullptr));
                    CK(hipEventSynchronize(eb));
                    float ms = 0;
                    CK(hipEventElapsedTime(&ms, ea, eb));
                    const double us = 1000.0 * ms / 10.0;
                    if (us < best) { best = us; best_i = i; }
                }
                std::printf("hb in %5u n %5u rows %5u %-9s %10.1f us %6.2f TFLOPS algos %d best #%d heur %.1f ms first-call %.1f ms%s\n",
                            in, n, rows, arm.name, best, gf / best / 1e6, got, best_i, heur_ms, first_call_ms,
                            first ? " (process first)" : "");
                first = false;
                if (std::getenv("HB_NAMES")) std::printf("  kernel %s\n", hipblaslt_ext::getKernelNameFromAlgo(h, res[best_i].algo).c_str());
            }
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc);
            hipblasLtMatmulDescDestroy(md);
        }
    }
    hipblasLtDestroy(h);
    return 0;
}
