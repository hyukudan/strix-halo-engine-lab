// RTX PRO driver of the FP32 prefill identity tests (src/minimax/rtxpf/fp32_identity_tests.inc).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "minimax/rtxpf/fp32_identity.hpp"

static void cu_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
#define GPU_MALLOC(p, n) cu_check(cudaMalloc(reinterpret_cast<void**>(p), (n)), "malloc")
#define GPU_FREE(p) cu_check(cudaFree(p), "free")
#define GPU_H2D(d, h, n) cu_check(cudaMemcpy((d), (h), (n), cudaMemcpyHostToDevice), "h2d")
#define GPU_D2H(h, d, n) cu_check(cudaMemcpy((h), (d), (n), cudaMemcpyDeviceToHost), "d2h")
#define GPU_SYNC() cu_check(cudaDeviceSynchronize(), "sync")
#include "minimax/rtxpf/fp32_identity_tests.inc"

int main() {
    namespace R = strix::minimax::rtxpf::ref;
    R::Gen g{7};
    long long bad = 0;
    for (int i = 0; i < 2000000; ++i) {
        const float f = g.value(4, 30);
        bad += R::float_to_half(f) != __half_as_ushort(__float2half_rn(f));
    }
    std::printf("host float_to_half vs __float2half_rn: %lld mismatches%s\n", bad, bad ? "  FAIL" : "");
    return fp32_identity_tests::run_all() | (bad ? 1 : 0);
}
