// gfx1151 driver of the FP32 prefill identity tests (src/minimax/rtxpf/fp32_identity_tests.inc).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "minimax/rtxpf/fp32_identity.hpp"

static void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
#define GPU_MALLOC(p, n) hip_check(hipMalloc(reinterpret_cast<void**>(p), (n)), "malloc")
#define GPU_FREE(p) hip_check(hipFree(p), "free")
#define GPU_H2D(d, h, n) hip_check(hipMemcpy((d), (h), (n), hipMemcpyHostToDevice), "h2d")
#define GPU_D2H(h, d, n) hip_check(hipMemcpy((h), (d), (n), hipMemcpyDeviceToHost), "d2h")
#define GPU_SYNC() hip_check(hipDeviceSynchronize(), "sync")
#include "minimax/rtxpf/fp32_identity_tests.inc"

int main() {
    // host RNE half helper vs the compiler's _Float16 conversion
    namespace R = strix::minimax::rtxpf::ref;
    R::Gen g{7};
    long long bad = 0;
    for (int i = 0; i < 2000000; ++i) {
        const float f = g.value(4, 30);
        bad += R::float_to_half(f) != __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(f));
    }
    std::printf("host float_to_half vs _Float16: %lld mismatches%s\n", bad, bad ? "  FAIL" : "");
    return fp32_identity_tests::run_all() | (bad ? 1 : 0);
}
