#pragma once
#include "strix/hip_hyper_mix_launcher.hpp"
#include <hip/hip_runtime_api.h>
#include <cstdint>
namespace strix::testing {
// Test-only seam for the prefill wide HC kernels (projection + output of one HC step over `rows`
// rows; hidden 2560, 4 streams, rank 320). Row r of each view sits at base + r * stride bytes:
// normalized = 10240 normalized FP32 values followed by the 320 low values the projection writes;
// injection = 4 FP32 values (when `inject` is set); output = hidden values in the config's
// activation storage. down: Q8_0 [rank][10240]; inject: F32 [4][10240]; up: Q8_0 [4 * hidden][rank].
// The caller owns every extent through completion and drains the stream.
struct WideHyperRowsTestViews {
    float* normalized = nullptr;
    std::uint64_t normalized_stride = 0;
    float* injection = nullptr;
    std::uint64_t injection_stride = 0;
    void* output = nullptr;
    std::uint64_t output_stride = 0;
    std::uint32_t rows = 0;
    const void* down = nullptr;
    const void* inject = nullptr;
    const void* up = nullptr;
    // Optional FP16 copy of the normalized rows (row r at normalized16 + r * 10240 halves) for projection kernel 2.
    const void* normalized16 = nullptr;
    std::uint32_t normalized16_pitch = 0;  // halves; 0 = 10240
    bool normalized16_only = false;        // the output's stream mix reads the FP16 rows too
};
void enqueue_wide_hyper_rows_test(const HipHyperMixLauncherConfig& config, const WideHyperRowsTestViews& views,
    unsigned projection_kernel, unsigned output_kernel, hipStream_t stream);
}
