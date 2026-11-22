// EXACT screened greedy LM head (core/hip_screened_head.hpp), device test.
// 1. The head transform G = row_transforms::bf16_softcap is monotone non-decreasing over every finite BF16 input
//    (exhaustive, 65,536 patterns, for each tested cap): the screen's exactness relies on it.
// 2. Screened token and status equal the full head (dense_linear tight rows + Bf16Softcap + first-max picker) for
//    random rows, exact ties, a peaked row, a non-finite activation (fallback) and more candidates than the capacity.
#include "core/hip_screened_head.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
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

__global__ void monotone_kernel(float cap, unsigned* bad) {
    const unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= 65536U) return;
    const float v = __uint_as_float(b << 16U);
    if (!isfinite(v) || b == 0x7f7fU) return;  // largest finite has no finite successor
    // Next BF16 value up: +0 / -0 -> smallest positive; negatives move toward zero.
    unsigned n = 0U;
    if (b == 0x0000U || b == 0x8000U) n = 0x0001U;
    else n = (b & 0x8000U) != 0U ? b - 1U : b + 1U;
    if (n == 0x8000U) n = 0x0000U;
    const float w = __uint_as_float(n << 16U);
    if (!(strix::core::row_transforms::bf16_softcap(v, cap) <= strix::core::row_transforms::bf16_softcap(w, cap)))
        atomicAdd(bad, 1U);
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

void monotone(float cap) {
    auto* bad = device_alloc<unsigned>(1);
    check(hipMemset(bad, 0, sizeof(unsigned)), "memset");
    monotone_kernel<<<256, 256>>>(cap, bad);
    unsigned host = 0U;
    check(hipMemcpy(&host, bad, sizeof host, hipMemcpyDeviceToHost), "copy");
    std::printf("screened-head-hip monotone cap=%g: %u violations\n", static_cast<double>(cap), host);
    if (host != 0U) ++failures;
    check(hipFree(bad), "free");
}

std::uint16_t to_bf16(float v) { return static_cast<std::uint16_t>(strix::weight::scalar_float::bf16_encode(v)); }

struct Case {
    const char* name;
    unsigned rows;
    int kind;  // 0 random, 1 tie of the best row, 2 peaked, 3 non-finite x, 4 identical rows (over capacity)
};

void run(const Case& c, unsigned width, float cap, unsigned bits, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> w(std::size_t{c.rows} * width);
    if (c.kind == 4) {
        std::vector<std::uint16_t> row(width);
        for (auto& v : row) v = to_bf16(0.02F * normal(rng));
        for (unsigned r = 0; r < c.rows; ++r) std::copy_n(row.data(), width, w.data() + std::size_t{r} * width);
    } else {
        for (auto& v : w) v = to_bf16(0.02F * normal(rng));
    }
    std::vector<float> x(width);
    for (auto& v : x) v = normal(rng);
    if (c.kind == 3) x[width / 2U] = std::numeric_limits<float>::quiet_NaN();
    // Ties / peaks need the incumbent logits first: copy row 0 (aligned with x) over a later row.
    if (c.kind == 1 || c.kind == 2) {
        for (unsigned k = 0; k < width; ++k) {
            const float boost = c.kind == 2 ? 0.2F : 0.05F;
            w[k] = to_bf16((x[k] > 0.0F ? boost : -boost) + 0.01F * normal(rng));
        }
        std::copy_n(w.data(), width, w.data() + std::size_t{c.rows - 3U} * width);
    }
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dx = device_alloc<float>(width);
    auto* dy = device_alloc<float>(c.rows);
    check(hipMemcpy(dw, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "upload w");
    check(hipMemcpy(dx, x.data(), width * 4U, hipMemcpyHostToDevice), "upload x");
    // Full head: incumbent GEMV, transform, host first-max (the shared rule).
    const dl::View view{reinterpret_cast<const std::uint8_t*>(dw), width, c.rows, std::size_t{width} * 2U};
    check(dl::enqueue_bf16_tight_rows(view, dx, dy, 1U, nullptr), "full gemv");
    softcap_kernel<<<(c.rows + 255U) / 256U, 256>>>(dy, c.rows, cap);
    std::vector<float> y(c.rows);
    check(hipMemcpy(y.data(), dy, c.rows * 4U, hipMemcpyDeviceToHost), "copy y");
    const auto full = sh::pick_reference(y);
    // Screened head.
    hs::ResidentScreen screen(dw, c.rows, width, cap, nullptr, bits);
    void* mapped = nullptr;
    check(hipHostMalloc(&mapped, sizeof(HipGreedyRowsReply), hipHostMallocMapped | hipHostMallocCoherent), "reply");
    void* device_reply = nullptr;
    check(hipHostGetDevicePointer(&device_reply, mapped, 0), "map reply");
    if (!screen.usable()) {
        std::printf("FAIL %s: screen not usable\n", c.name);
        ++failures;
    } else {
        for (unsigned repeat = 0; repeat < 2U; ++repeat) {  // the second launch reuses the scratch state
            check(screen.enqueue(dx, static_cast<HipGreedyRowsReply*>(device_reply), nullptr), "screened");
            check(hipDeviceSynchronize(), "sync");
            const auto got = *static_cast<HipGreedyRowsReply*>(mapped);
            const bool ok = got.token == full.token && got.status == full.status;
            if (!ok) ++failures;
            std::printf("%s screened-head-hip %-10s K=%u bits=%u rows=%u: full {%u,%u} screened {%u,%u}\n",
                        ok ? "PASS" : "FAIL", c.name, width, bits, c.rows, full.token, full.status, got.token,
                        got.status);
        }
    }
    check(hipHostFree(mapped), "free reply");
    check(hipFree(dw), "free");
    check(hipFree(dx), "free");
    check(hipFree(dy), "free");
}

// Informational timing at the Inkling 2 Mini head shape (200,058 x 2,304 BF16, one dominant row so the candidate set is
// small): the full GEMV + transform versus the five screened launches. Also checks token equality once more.
void bench(std::mt19937& rng) {
    constexpr unsigned rows = 200058U, width = 2304U, iterations = 20U;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(width);
    for (auto& v : x) v = normal(rng);
    // 1,024 random rows repeated (cheap to generate; repeated rows only add exact ties far below the dominant row).
    std::vector<std::uint16_t> w(std::size_t{rows} * width);
    constexpr unsigned distinct = 1024U;
    for (std::size_t i = 0; i < std::size_t{distinct} * width; ++i) w[i] = to_bf16(0.02F * normal(rng));
    for (unsigned r = distinct; r < rows; ++r)
        std::copy_n(w.data() + std::size_t{r % distinct} * width, width, w.data() + std::size_t{r} * width);
    for (unsigned k = 0; k < width; ++k) w[std::size_t{777U} * width + k] = to_bf16(x[k] > 0.0F ? 0.01F : -0.01F);
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dx = device_alloc<float>(width);
    auto* dy = device_alloc<float>(rows);
    check(hipMemcpy(dw, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "upload w");
    check(hipMemcpy(dx, x.data(), width * 4U, hipMemcpyHostToDevice), "upload x");
    w = {};
    hs::ResidentScreen screen(dw, rows, width, 30.0F, nullptr);
    void* mapped = nullptr;
    check(hipHostMalloc(&mapped, sizeof(HipGreedyRowsReply), hipHostMallocMapped | hipHostMallocCoherent), "reply");
    void* device_reply = nullptr;
    check(hipHostGetDevicePointer(&device_reply, mapped, 0), "map reply");
    hipEvent_t start{}, stop{};
    check(hipEventCreate(&start), "event");
    check(hipEventCreate(&stop), "event");
    const dl::View view{reinterpret_cast<const std::uint8_t*>(dw), width, rows, std::size_t{width} * 2U};
    float full_ms = 0.0F, screened_ms = 0.0F;
    for (unsigned arm = 0; arm < 2U && screen.usable(); ++arm) {
        for (unsigned i = 0; i <= iterations; ++i) {
            if (i == 1U) check(hipEventRecord(start, nullptr), "record");
            if (arm == 0U) {
                check(dl::enqueue_bf16_tight_rows(view, dx, dy, 1U, nullptr), "full gemv");
                softcap_kernel<<<(rows + 255U) / 256U, 256>>>(dy, rows, 30.0F);
            } else {
                check(screen.enqueue(dx, static_cast<HipGreedyRowsReply*>(device_reply), nullptr), "screened");
            }
        }
        check(hipEventRecord(stop, nullptr), "record");
        check(hipEventSynchronize(stop), "sync");
        check(hipEventElapsedTime(arm == 0U ? &full_ms : &screened_ms, start, stop), "elapsed");
    }
    std::vector<float> y(rows);
    check(hipMemcpy(y.data(), dy, rows * 4U, hipMemcpyDeviceToHost), "copy y");
    const auto full = sh::pick_reference(y);
    const auto got = *static_cast<HipGreedyRowsReply*>(mapped);
    const bool ok = screen.usable() && got.token == full.token && got.status == full.status;
    if (!ok) ++failures;
    std::printf("%s screened-head-hip bench rows=%u width=%u: full %.1f us, screened %.1f us per head (plane %zu B)\n",
                ok ? "PASS" : "FAIL", rows, width, static_cast<double>(full_ms) * 1000.0 / iterations,
                static_cast<double>(screened_ms) * 1000.0 / iterations, screen.device_bytes());
    check(hipEventDestroy(start), "event");
    check(hipEventDestroy(stop), "event");
    check(hipHostFree(mapped), "free reply");
    check(hipFree(dw), "free");
    check(hipFree(dx), "free");
    check(hipFree(dy), "free");
}

}  // namespace

int main() {
    for (const float cap : {30.0F, 15.0F, 50.0F}) monotone(cap);
    std::mt19937 rng(1234U);
    const Case cases[] = {{"random", 6000U, 0}, {"tie", 6000U, 1},   {"peaked", 6000U, 2},
                          {"nan-x", 3000U, 3},  {"overflow", 20000U, 4}};
    for (const unsigned width : {2304U, 512U})
        for (const unsigned bits : {8U, 4U})
            for (const auto& c : cases) run(c, width, 30.0F, bits, rng);
    bench(rng);
    if (failures != 0) {
        std::fprintf(stderr, "screened_head_hip: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("screened_head_hip: PASS\n");
    return 0;
}
