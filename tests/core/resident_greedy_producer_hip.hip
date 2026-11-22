// core::ResidentGreedy device producer hook (HIP): a device picker writing through device_reply(row) and published by
// produced(rows) yields, through token(), what enqueue() yields on the full logits row. The producer here is the
// shared EXACT screened head (core/hip_screened_head.hpp) against the full head (dense_linear tight rows +
// bf16_softcap + the shared greedy rows picker). Also checks the hook's refusals. No model.
#include "core/hip_screened_head.hpp"
#include "core/resident_greedy.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

namespace hs = strix::core::hip_screened_head;
namespace dl = strix::core::dense_linear;

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

// kind 0 random rows, 1 one peaked row, 2 non-finite activation (no finite pick).
void run(unsigned rows, unsigned width, int kind, unsigned reply_row, std::mt19937& rng) {
    constexpr float cap = 30.0F;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> w(std::size_t{rows} * width);
    for (auto& v : w) v = to_bf16(0.02F * normal(rng));
    std::vector<float> x(width);
    for (auto& v : x) v = normal(rng);
    if (kind == 1)
        for (unsigned k = 0; k < width; ++k) w[std::size_t{rows / 3U} * width + k] = to_bf16(x[k] > 0.0F ? 0.2F : -0.2F);
    if (kind == 2) x[width / 2U] = std::numeric_limits<float>::quiet_NaN();
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dx = device_alloc<float>(width);
    auto* dy = device_alloc<float>(rows);
    check(hipMemcpy(dw, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "upload w");
    check(hipMemcpy(dx, x.data(), width * 4U, hipMemcpyHostToDevice), "upload x");
    hipStream_t stream{};
    check(hipStreamCreate(&stream), "stream");

    // Incumbent: full logits row, then the shared greedy rows picker.
    strix::core::ResidentGreedy full(reply_row + 1U, rows), screened(reply_row + 1U, rows);
    const dl::View view{reinterpret_cast<const std::uint8_t*>(dw), width, rows, std::size_t{width} * 2U};
    check(dl::enqueue_bf16_tight_rows(view, dx, dy, 1U, stream), "full gemv");
    softcap_kernel<<<(rows + 255U) / 256U, 256, 0, stream>>>(dy, rows, cap);
    full.enqueue(std::span<const float>(dy, rows), stream);

    // Producer: the screened head writes row reply_row of the second picker's mapped replies.
    hs::ResidentScreen screen(dw, rows, width, cap, stream);
    if (!screen.usable()) {
        std::printf("FAIL resident-greedy-producer rows=%u width=%u: screen not usable\n", rows, width);
        ++failures;
    } else {
        check(screen.enqueue(dx, screened.device_reply(reply_row), stream), "screened");
        screened.produced(reply_row + 1U);
        check(hipStreamSynchronize(stream), "sync");
        const auto want = full.token(0);
        const auto got = screened.token(reply_row);
        const bool ok = want == got;
        if (!ok) ++failures;
        std::printf("%s resident-greedy-producer kind=%d rows=%u width=%u reply_row=%u: full %u producer %u\n",
                    ok ? "PASS" : "FAIL", kind, rows, width, reply_row, want, got);
    }
    check(hipStreamDestroy(stream), "stream");
    check(hipFree(dw), "free");
    check(hipFree(dx), "free");
    check(hipFree(dy), "free");
}

void refusals() {
    strix::core::ResidentGreedy greedy(3U, 1000U);
    int refused = 0;
    try { (void)greedy.device_reply(3U); } catch (const std::out_of_range&) { ++refused; }
    try { greedy.produced(0U); } catch (const std::invalid_argument&) { ++refused; }
    try { greedy.produced(4U); } catch (const std::invalid_argument&) { ++refused; }
    try { (void)greedy.token(0U); } catch (const std::out_of_range&) { ++refused; }  // nothing produced yet
    const bool ok = refused == 4;
    if (!ok) ++failures;
    std::printf("%s resident-greedy-producer refusals %d/4\n", ok ? "PASS" : "FAIL", refused);
}

}  // namespace

int main() {
    std::mt19937 rng(20261122U);
    for (const unsigned width : {2304U, 512U})
        for (const int kind : {0, 1, 2})
            for (const unsigned reply_row : {0U, 2U}) run(6000U, width, kind, reply_row, rng);
    refusals();
    if (failures != 0) {
        std::fprintf(stderr, "resident_greedy_producer_hip: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("resident_greedy_producer_hip: PASS\n");
    return 0;
}
