// FAST ple_a8 (qwen-cuda-ple-a8-v1), no model needed: the PLE key/value projections (Q8_0 weights, BF16 gathered rows,
// width 2560 -> 10240 + 2560 outputs) through the exact FP32 lane chain (qwen_plemtp::launch_ple_project, two launches)
// and through the shared dense A8 projection (qwen_dense::quantize_project, ONE site with two segments). Checks the A8
// outputs against the exact ones (relative L2, overall and worst row) and times both forms on GPU index 0.
// Usage: qwen_ple_a8_bench [--rows N]   (exit 1 on a bound failure; the timing lines are the bench receipt)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "strix/cuda/qwen_dense.hpp"
#include "strix/cuda/qwen_plemtp.hpp"

namespace pm = strix::cuda::qwen_plemtp;
namespace dn = strix::cuda::qwen_dense;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));                \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)

namespace {

constexpr std::uint32_t kWidth = pm::kPleWidth, kKey = pm::kExpanded, kValue = pm::kHidden, kBlocks = kWidth / 32U;

std::uint16_t half_bits(float v) {
    const __half h = __float2half_rn(v);
    std::uint16_t b = 0;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}
std::uint16_t bf16_bits(float v) {
    std::uint32_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16);
}
std::vector<std::uint8_t> q8_weights(std::uint32_t outputs, std::mt19937_64& rng) {
    std::vector<std::uint8_t> w(static_cast<std::size_t>(outputs) * kBlocks * 34U);
    std::uniform_real_distribution<float> scale(0.0005F, 0.004F);
    for (std::size_t b = 0; b < static_cast<std::size_t>(outputs) * kBlocks; ++b) {
        const std::uint16_t h = half_bits(scale(rng));
        w[b * 34U] = static_cast<std::uint8_t>(h & 255U);
        w[b * 34U + 1U] = static_cast<std::uint8_t>(h >> 8U);
        for (std::size_t j = 0; j < 32U; ++j)
            w[b * 34U + 2U + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(rng() % 255U) - 127));
    }
    return w;
}
template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    CK(cudaMalloc(&d, v.size() * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
std::vector<float> download(const float* d, std::size_t n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

struct Buffers {
    const std::uint16_t* rows = nullptr;
    const std::uint8_t* wk = nullptr;
    const std::uint8_t* wv = nullptr;
    float *key = nullptr, *value = nullptr;
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    std::uint32_t cap = 0;
};

void run_exact(const Buffers& b, std::uint32_t rows) {
    pm::launch_ple_project(b.rows, kWidth, rows, b.wk, kWidth, kKey, b.key, kKey, nullptr);
    pm::launch_ple_project(b.rows, kWidth, rows, b.wv, kWidth, kValue, b.value, kValue, nullptr);
}
void run_a8(const Buffers& b, std::uint32_t rows) {
    dn::Site s;
    s.store = dn::Store::f32;
    s.width = kWidth;
    s.segments = 2;
    s.segment[0] = {b.wk, kKey, b.key, kKey};
    s.segment[1] = {b.wv, kValue, b.value, kValue};
    CK(dn::quantize_project(nullptr, s, rows, dn::Input::bf16, b.rows, kWidth, b.codes, b.deltas, b.cap));
}

// relative L2 of a against ref, overall and worst row
void compare(const std::vector<float>& ref, const std::vector<float>& a, std::uint32_t rows, std::uint32_t cols, double& rel,
             double& worst) {
    double dd = 0, rr = 0;
    worst = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        double d1 = 0, r1 = 0;
        for (std::uint32_t c = 0; c < cols; ++c) {
            const double u = ref[static_cast<std::size_t>(r) * cols + c], v = a[static_cast<std::size_t>(r) * cols + c];
            d1 += (v - u) * (v - u);
            r1 += u * u;
        }
        dd += d1;
        rr += r1;
        worst = std::max(worst, std::sqrt(d1 / std::max(r1, 1e-30)));
    }
    rel = std::sqrt(dd / std::max(rr, 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
    std::uint32_t max_rows = 8192;
    for (int i = 1; i + 1 < argc; i += 2)
        if (std::string(argv[i]) == "--rows") max_rows = static_cast<std::uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
    CK(cudaSetDevice(0));
    std::mt19937_64 rng(20261119);
    const auto wk = q8_weights(kKey, rng), wv = q8_weights(kValue, rng);
    // gathered PLE rows: 16 heads x 160 BF16 values per row (IQ4_NL table rows); a gaussian with per-feature gains and outliers
    std::vector<std::uint16_t> x(static_cast<std::size_t>(max_rows) * kWidth);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (std::size_t i = 0; i < x.size(); ++i) {
        float v = nd(rng) * (1.0F + 0.02F * static_cast<float>(i % 97U));
        if ((rng() & 0x3ffU) == 0U) v *= 16.0F;
        x[i] = bf16_bits(v);
    }
    Buffers b;
    b.rows = upload(x);
    b.wk = upload(wk);
    b.wv = upload(wv);
    b.cap = max_rows;
    CK(cudaMalloc(&b.key, static_cast<std::size_t>(max_rows) * kKey * sizeof(float)));
    CK(cudaMalloc(&b.value, static_cast<std::size_t>(max_rows) * kValue * sizeof(float)));
    CK(cudaMalloc(&b.codes, static_cast<std::size_t>(max_rows) * kWidth));
    CK(cudaMalloc(&b.deltas, static_cast<std::size_t>(kBlocks) * max_rows * sizeof(float)));

    int failed = 0;
    for (std::uint32_t rows : {1U, 17U, 129U, 1000U, max_rows}) {
        if (rows > max_rows) continue;
        run_exact(b, rows);
        CK(cudaDeviceSynchronize());
        const auto ek = download(b.key, static_cast<std::size_t>(rows) * kKey), ev = download(b.value, static_cast<std::size_t>(rows) * kValue);
        run_a8(b, rows);
        CK(cudaDeviceSynchronize());
        const auto ak = download(b.key, static_cast<std::size_t>(rows) * kKey), av = download(b.value, static_cast<std::size_t>(rows) * kValue);
        double rk = 0, wk_ = 0, rv = 0, wv_ = 0;
        compare(ek, ak, rows, kKey, rk, wk_);
        compare(ev, av, rows, kValue, rv, wv_);
        bool finite = true;
        for (const float f : ak) finite = finite && std::isfinite(f);
        for (const float f : av) finite = finite && std::isfinite(f);
        // per-32 int8 activation codes: element error <= delta/2 = amax/254 -> dot-product relative error ~0.3 %
        const bool ok = finite && rk < 1e-2 && rv < 1e-2 && wk_ < 3e-2 && wv_ < 3e-2;
        failed += ok ? 0 : 1;
        std::printf("ple_a8 vs exact rows %u: key relL2 %.3e (worst row %.3e) value relL2 %.3e (worst row %.3e) %s\n", rows, rk, wk_,
                    rv, wv_, ok ? "PASS" : "FAIL");
    }
    const double tops = 2.0 * kWidth * (kKey + kValue) * 1e-12;  // per row
    for (std::uint32_t rows : {max_rows, std::min(max_rows, 2048U)}) {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        float ms[2] = {0, 0};
        for (int form = 0; form < 2; ++form) {
            const auto call = [&] { form ? run_a8(b, rows) : run_exact(b, rows); };
            for (int i = 0; i < 3; ++i) call();
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) call();
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms[form], e0, e1));
            ms[form] /= 20.0F;
        }
        std::printf("ple_a8 bench rows %u: exact %.3f ms (%.1f TF/s), a8 %.3f ms (%.1f TOPS), a8/exact %.3f\n", rows, ms[0],
                    tops * rows / (ms[0] * 1e-3), ms[1], tops * rows / (ms[1] * 1e-3), ms[1] / ms[0]);
        std::fflush(stdout);
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
    }
    std::printf("qwen_ple_a8_bench: %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
