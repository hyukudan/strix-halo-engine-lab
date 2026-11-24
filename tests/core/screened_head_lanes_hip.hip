// EXACT screened greedy LM head, lanes form (core/hip_screened_head.hpp enqueue_screened_greedy_lanes / ResidentScreen::
// enqueue_lanes), device test. For every lane count, bit width and activation width, lane l's token and status equal
// (a) the one-row screened head on that row alone and (b) the full head (dense_linear tight rows + Bf16Softcap +
// first-max) on that row. Lanes mix random rows, a peaked row, an exact tie of the best row and a non-finite activation
// (fallback); a head of identical rows sends every lane over the candidate capacity. Rows are strided (padded pitch).
// The bench (informational) times lanes x the one-row head against one lanes enqueue at the Inkling 2 Mini head shape.
#include "core/hip_screened_head.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {

namespace hs = strix::core::hip_screened_head;
namespace sh = strix::core::screened_head;
namespace dl = strix::core::dense_linear;
using strix::HipGreedyRowsReply;

int failures = 0;

void check(hipError_t status, const char* what) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(status));
    std::exit(2);
}

__global__ void softcap_kernel(float* y, unsigned n, float cap) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = strix::core::row_transforms::bf16_softcap(y[i], cap);
}

template <class T>
T* device_alloc(std::size_t count) {
    void* p = nullptr;
    check(hipMalloc(&p, count * sizeof(T)), "alloc");
    return static_cast<T*>(p);
}

std::uint16_t to_bf16(float v) { return static_cast<std::uint16_t>(strix::weight::scalar_float::bf16_encode(v)); }

struct Replies {
    HipGreedyRowsReply* host = nullptr;
    HipGreedyRowsReply* device = nullptr;
    explicit Replies(unsigned n) {
        void* mapped = nullptr;
        check(hipHostMalloc(&mapped, n * sizeof(HipGreedyRowsReply), hipHostMallocMapped | hipHostMallocCoherent), "reply");
        void* d = nullptr;
        check(hipHostGetDevicePointer(&d, mapped, 0), "map reply");
        host = static_cast<HipGreedyRowsReply*>(mapped);
        device = static_cast<HipGreedyRowsReply*>(d);
    }
    ~Replies() { (void)hipHostFree(host); }
    Replies(const Replies&) = delete;
    Replies& operator=(const Replies&) = delete;
};

// identical: every head row equal (every lane over capacity). Otherwise random rows; lane 0 gets a peaked row, lane 1 an
// exact tie of its best row, the last lane of three or more a NaN activation.
void run(unsigned width, unsigned bits, unsigned rows, bool identical, unsigned lanes, std::mt19937& rng) {
    constexpr float cap = 30.0F;
    const unsigned stride = width + 8U;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(std::size_t{lanes} * stride, 0.0F);
    for (unsigned l = 0; l < lanes; ++l)
        for (unsigned k = 0; k < width; ++k) x[std::size_t{l} * stride + k] = normal(rng);
    std::vector<std::uint16_t> w(std::size_t{rows} * width);
    if (identical) {
        std::vector<std::uint16_t> row(width);
        for (auto& v : row) v = to_bf16(0.02F * normal(rng));
        for (unsigned r = 0; r < rows; ++r) std::copy_n(row.data(), width, w.data() + std::size_t{r} * width);
    } else {
        for (auto& v : w) v = to_bf16(0.02F * normal(rng));
        for (unsigned l = 0; l < std::min(lanes, 2U); ++l) {
            const float boost = l == 0U ? 0.2F : 0.05F;
            const std::size_t target = std::size_t{7U + 11U * l} * width;
            for (unsigned k = 0; k < width; ++k)
                w[target + k] = to_bf16((x[std::size_t{l} * stride + k] > 0.0F ? boost : -boost) + 0.01F * normal(rng));
            if (l == 1U) std::copy_n(w.data() + target, width, w.data() + std::size_t{rows - 3U} * width);  // tie
        }
        if (lanes >= 3U) x[std::size_t{lanes - 1U} * stride + width / 2U] = std::numeric_limits<float>::quiet_NaN();
    }
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dx = device_alloc<float>(x.size());
    auto* dy = device_alloc<float>(rows);
    check(hipMemcpy(dw, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "upload w");
    check(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "upload x");
    const dl::View view{reinterpret_cast<const std::uint8_t*>(dw), width, rows, std::size_t{width} * 2U};
    std::vector<sh::Pick> full(lanes);
    std::vector<float> y(rows);
    for (unsigned l = 0; l < lanes; ++l) {
        check(dl::enqueue_bf16_tight_rows(view, dx + std::size_t{l} * stride, dy, 1U, nullptr), "full gemv");
        softcap_kernel<<<(rows + 255U) / 256U, 256>>>(dy, rows, cap);
        check(hipMemcpy(y.data(), dy, rows * 4U, hipMemcpyDeviceToHost), "copy y");
        full[l] = sh::pick_reference(y);
    }
    hs::ResidentScreen screen(dw, rows, width, cap, nullptr, bits, lanes);
    if (!screen.usable() || screen.max_lanes() != lanes) {
        std::printf("FAIL screened-head-lanes-hip K=%u bits=%u lanes=%u: screen not usable\n", width, bits, lanes);
        ++failures;
    } else {
        Replies one(lanes), many(lanes);
        for (unsigned l = 0; l < lanes; ++l)
            check(screen.enqueue(dx + std::size_t{l} * stride, one.device + l, nullptr), "one-row");
        check(hipDeviceSynchronize(), "sync one-row");
        const std::vector<HipGreedyRowsReply> single(one.host, one.host + lanes);
        for (unsigned repeat = 0; repeat < 2U; ++repeat) {  // the second launch reuses the scratch state
            check(screen.enqueue_lanes(dx, lanes, stride, many.device, nullptr), "lanes");
            check(hipDeviceSynchronize(), "sync lanes");
            for (unsigned l = 0; l < lanes; ++l) {
                const auto got = many.host[l];
                const bool ok = got.token == single[l].token && got.status == single[l].status &&
                                got.token == full[l].token && got.status == full[l].status;
                if (!ok) ++failures;
                if (!ok || repeat == 0U)
                    std::printf("%s screened-head-lanes-hip K=%u bits=%u rows=%u%s lanes=%u lane=%u: full {%u,%u} "
                                "one-row {%u,%u} lanes {%u,%u}\n",
                                ok ? "PASS" : "FAIL", width, bits, rows, identical ? " identical" : "", lanes, l,
                                full[l].token, full[l].status, single[l].token, single[l].status, got.token, got.status);
            }
        }
        // Contract: zero lanes, more lanes than the scratch holds, or an unaligned stride are rejected.
        const bool rejects = screen.enqueue_lanes(dx, 0U, stride, many.device, nullptr) == hipErrorInvalidValue &&
                             screen.enqueue_lanes(dx, lanes + 1U, stride, many.device, nullptr) == hipErrorInvalidValue &&
                             (lanes == 1U ||
                              screen.enqueue_lanes(dx, lanes, stride + 2U, many.device, nullptr) == hipErrorInvalidValue);
        if (!rejects) ++failures;
        if (!rejects) std::printf("FAIL screened-head-lanes-hip lanes=%u: invalid launch accepted\n", lanes);
    }
    check(hipFree(dw), "free");
    check(hipFree(dx), "free");
    check(hipFree(dy), "free");
}

// Informational timing at the Inkling 2 Mini head shape (200,058 x 2,304 BF16), one dominant row per lane so every lane's
// candidate set is small: `lanes` one-row screened heads against one lanes enqueue. Also checks the tokens once more.
void bench(std::mt19937& rng) {
    constexpr unsigned rows = 200058U, width = 2304U, iterations = 20U, max_lanes = 8U;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(std::size_t{max_lanes} * width);
    for (auto& v : x) v = normal(rng);
    std::vector<std::uint16_t> w(std::size_t{rows} * width);
    constexpr unsigned distinct = 1024U;
    for (std::size_t i = 0; i < std::size_t{distinct} * width; ++i) w[i] = to_bf16(0.02F * normal(rng));
    for (unsigned r = distinct; r < rows; ++r)
        std::copy_n(w.data() + std::size_t{r % distinct} * width, width, w.data() + std::size_t{r} * width);
    for (unsigned l = 0; l < max_lanes; ++l)
        for (unsigned k = 0; k < width; ++k)
            w[std::size_t{777U + 1000U * l} * width + k] = to_bf16(x[std::size_t{l} * width + k] > 0.0F ? 0.01F : -0.01F);
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dx = device_alloc<float>(x.size());
    check(hipMemcpy(dw, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "upload w");
    check(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "upload x");
    w = {};
    hs::ResidentScreen screen(dw, rows, width, 30.0F, nullptr, 8U, max_lanes);
    Replies one(max_lanes), many(max_lanes);
    hipEvent_t start{}, stop{};
    check(hipEventCreate(&start), "event");
    check(hipEventCreate(&stop), "event");
    for (const unsigned lanes : {1U, 2U, 4U, 8U}) {
        float ms[2] = {0.0F, 0.0F};
        for (unsigned arm = 0; arm < 2U && screen.usable(); ++arm) {
            // ~0.5 s of the same work first, so the timed loop runs at the sustained clock (the one-row head is
            // bandwidth-bound, the lanes pass 1 is not).
            for (unsigned i = 0; i < 4U * iterations * lanes; ++i)
                check(screen.enqueue(dx, one.device, nullptr), "warm");
            for (unsigned i = 0; i <= iterations; ++i) {
                if (i == 1U) check(hipEventRecord(start, nullptr), "record");
                if (arm == 0U) {
                    for (unsigned l = 0; l < lanes; ++l)
                        check(screen.enqueue(dx + std::size_t{l} * width, one.device + l, nullptr), "one-row");
                } else {
                    check(screen.enqueue_lanes(dx, lanes, width, many.device, nullptr), "lanes");
                }
            }
            check(hipEventRecord(stop, nullptr), "record");
            check(hipEventSynchronize(stop), "sync");
            check(hipEventElapsedTime(&ms[arm], start, stop), "elapsed");
        }
        bool ok = screen.usable();
        for (unsigned l = 0; ok && l < lanes; ++l)
            ok = one.host[l].token == many.host[l].token && one.host[l].status == many.host[l].status;
        if (!ok) ++failures;
        std::printf("%s screened-head-lanes-hip bench rows=%u width=%u lanes=%u pass1_lanes=%u: one-row x%u %.1f us, "
                    "lanes %.1f us per head set\n",
                    ok ? "PASS" : "FAIL", rows, width, lanes, hs::pass1_lanes(width, lanes), lanes,
                    static_cast<double>(ms[0]) * 1000.0 / iterations, static_cast<double>(ms[1]) * 1000.0 / iterations);
    }
    check(hipEventDestroy(start), "event");
    check(hipEventDestroy(stop), "event");
    check(hipFree(dw), "free");
    check(hipFree(dx), "free");
}

}  // namespace

int main() {
    std::mt19937 rng(4321U);
    for (const unsigned width : {2304U, 512U})
        for (const unsigned bits : {8U, 4U})
            for (const unsigned lanes : {1U, 2U, 3U, 4U, 5U, 8U}) run(width, bits, 6000U, false, lanes, rng);
    for (const unsigned lanes : {2U, 4U}) run(2304U, 8U, 20000U, true, lanes, rng);
    bench(rng);
    if (failures != 0) {
        std::fprintf(stderr, "screened_head_lanes_hip: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("screened_head_lanes_hip: PASS\n");
    return 0;
}
