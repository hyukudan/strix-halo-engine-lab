// Device fixture of the shared EXACT chunk-partial combine (src/attn_core/chunk_exact_combine.hpp): the real
// combine_lanes<S> (wave of 32 lanes, readlane broadcasts, partial layout pt[cc * (2 + kDim) + ...]) against the
// independent host rule (chunk_exact_combine_rule.hpp), output bits compared. Random partials, every cnt 1 .. 32, NaN
// poison past cnt, exponent spreads that underflow e. Empty rows (cnt == 0) keep an output canary bit for bit and never
// invoke the output callback. The declared exponential is the correctly rounded 1 / (1 - x) (exact on host and device).
// Prints one PASS / FAIL line; exit 0 only on PASS (exit 77: no device, ctest skip; exit 2: runtime error).
#include "attn_core/chunk_exact_combine.hpp"
#include "attn_core/chunk_exact_combine_rule.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace cec = strix::attn_core::chunk_exact;

struct Spec {
    static constexpr unsigned kDim = 128, kMaxChunks = 32;
    static __host__ __device__ float exp(float x) { return 1.0F / (1.0F - x); }  // x <= 0: in (0, 1], 0 at -inf
};

constexpr unsigned kDim = Spec::kDim, kMaxChunks = Spec::kMaxChunks, kPart = 2U + kDim;
constexpr float kCanary = 1234.5F;

void hok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

__global__ __launch_bounds__(32) void combine_kernel(const float* pt, const unsigned* cnts, float* out, unsigned* called) {
    const unsigned row = blockIdx.x;
    cec::combine_lanes<Spec>(pt + static_cast<std::size_t>(row) * kMaxChunks * kPart, cnts[row], threadIdx.x, [&] {
        called[row] = 1U;
        return out + static_cast<std::size_t>(row) * kDim;
    });
}

}  // namespace

int main() {
    int devices = 0;
    hok(hipGetDeviceCount(&devices), "hipGetDeviceCount");
    if (devices < 1) {
        std::fprintf(stderr, "hip_chunk_exact_combine_tests: no device\n");
        return 77;
    }
    constexpr unsigned kTrials = 100, kRows = kTrials * (kMaxChunks + 1U);  // cnt 0 .. 32 per trial
    std::mt19937 rng(20261123U);
    std::vector<float> pt(static_cast<std::size_t>(kRows) * kMaxChunks * kPart);
    std::vector<unsigned> cnts(kRows);
    std::vector<float> want(static_cast<std::size_t>(kRows) * kDim, kCanary);
    unsigned row = 0;
    for (unsigned trial = 0; trial < kTrials; ++trial) {
        const float spread = trial % 4 == 0 ? 120.0F : (trial % 4 == 1 ? 8.0F : 1.0F);
        std::uniform_real_distribution<float> mdist(-spread, spread), ldist(1.0F, 128.0F), adist(-64.0F, 64.0F);
        for (unsigned cnt = 0; cnt <= kMaxChunks; ++cnt, ++row) {
            cnts[row] = cnt;
            float* base = pt.data() + static_cast<std::size_t>(row) * kMaxChunks * kPart;
            for (unsigned cc = 0; cc < kMaxChunks; ++cc) {
                float* p = base + cc * kPart;
                if (cc >= cnt) {
                    for (unsigned j = 0; j < kPart; ++j) p[j] = NAN;
                    continue;
                }
                p[0] = mdist(rng);
                p[1] = ldist(rng);
                for (unsigned d = 0; d < kDim; ++d) p[2 + d] = adist(rng);
            }
            if (cnt != 0) cec::combine_rule(base, cnt, kDim, Spec::exp, want.data() + static_cast<std::size_t>(row) * kDim);
        }
    }
    float *d_pt = nullptr, *d_out = nullptr;
    unsigned *d_cnt = nullptr, *d_called = nullptr;
    hok(hipMalloc(&d_pt, pt.size() * 4), "malloc pt");
    hok(hipMalloc(&d_out, want.size() * 4), "malloc out");
    hok(hipMalloc(&d_cnt, cnts.size() * 4), "malloc cnt");
    hok(hipMalloc(&d_called, cnts.size() * 4), "malloc called");
    std::vector<float> canary(want.size(), kCanary);
    hok(hipMemcpy(d_pt, pt.data(), pt.size() * 4, hipMemcpyHostToDevice), "copy pt");
    hok(hipMemcpy(d_out, canary.data(), canary.size() * 4, hipMemcpyHostToDevice), "copy out");
    hok(hipMemcpy(d_cnt, cnts.data(), cnts.size() * 4, hipMemcpyHostToDevice), "copy cnt");
    hok(hipMemset(d_called, 0, cnts.size() * 4), "memset called");
    combine_kernel<<<kRows, 32>>>(d_pt, d_cnt, d_out, d_called);
    hok(hipGetLastError(), "launch");
    hok(hipDeviceSynchronize(), "sync");
    std::vector<float> got(want.size());
    std::vector<unsigned> called(cnts.size());
    hok(hipMemcpy(got.data(), d_out, got.size() * 4, hipMemcpyDeviceToHost), "read out");
    hok(hipMemcpy(called.data(), d_called, called.size() * 4, hipMemcpyDeviceToHost), "read called");
    unsigned bad = 0, empty_bad = 0;
    for (unsigned r = 0; r < kRows; ++r) {
        const bool empty = cnts[r] == 0;
        const std::size_t at = static_cast<std::size_t>(r) * kDim;
        const bool bits = std::memcmp(got.data() + at, want.data() + at, kDim * 4) == 0;
        const bool callback_ok = called[r] == (empty ? 0U : 1U);
        if (!bits || !callback_ok) {
            ++bad;
            if (empty) ++empty_bad;
            if (bad <= 5)
                std::fprintf(stderr, "row %u cnt %u: bits %s, callback %s\n", r, cnts[r], bits ? "equal" : "DIFFER",
                             callback_ok ? "ok" : "WRONG");
        }
    }
    hok(hipFree(d_pt), "free");
    hok(hipFree(d_out), "free");
    hok(hipFree(d_cnt), "free");
    hok(hipFree(d_called), "free");
    std::printf("hip_chunk_exact_combine_tests: %u rows (incl. %u empty), %u mismatches (%u empty): %s\n", kRows, kTrials, bad,
                empty_bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
