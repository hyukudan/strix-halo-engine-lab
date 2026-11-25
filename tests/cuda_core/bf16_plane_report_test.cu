// The planes a shared BF16 dense rows dispatch reports (enqueue_bf16_dense_rows, `planes_report`) are the planes of the
// tall split that ran, never the requested count: a tall MMA3 dispatch on the tall tile reports 3 planes (or 1 where the
// input is declared BF16-exact); a projection the planner sends to the FP32 row grid (input width divisible by 32,
// output width 17) splits no activation and reports 0 whatever is requested; short rows and the device-profile rows
// planner report 0 and never leave the caller's previous value; a failed dispatch reports 0. Outputs of the FP32 grid
// do not depend on the declaration. No model files are required.
#include "cuda/core/dense/bf16_rows.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace cc = strix::cuda_core;
namespace core = strix::core;

namespace {

constexpr int kStale = 99;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
struct Managed {
    explicit Managed(std::size_t count) : size(count) { check(cudaMallocManaged(&data, count * sizeof(T))); }
    ~Managed() { cudaFree(data); }
    Managed(const Managed&) = delete;
    Managed& operator=(const Managed&) = delete;
    T* data = nullptr;
    std::size_t size = 0;
};

// BF16-exact activations (low 16 bits zero) and small BF16 weights from a fixed LCG: deterministic on every machine.
std::uint32_t next(std::uint32_t& seed) {
    seed = 1664525U * seed + 1013904223U;
    return seed >> 8U;
}

struct Weights {
    Weights(unsigned outputs, unsigned width, std::uint32_t& seed) : matrix(std::size_t{outputs} * width), out(outputs) {
        for (std::size_t i = 0; i < matrix.size; ++i)
            matrix.data[i] = static_cast<std::uint16_t>(0x3C00U | ((next(seed) << 15U) & 0x8000U) | (next(seed) & 0x01FFU));
    }
    Managed<std::uint16_t> matrix;
    unsigned out;
};

struct Outcome {
    cudaError_t status;
    int planes;
};

struct Case {
    unsigned width, rows;
    std::vector<unsigned> outputs;
};

// One dispatch of `test` with the declared operand class (and an optional device profile); `result` receives each
// projection's output so a caller can compare two declarations bit by bit.
Outcome dispatch(const Case& test, core::Bf16Operand operand, const strix::device::Profile* profile,
                 std::vector<std::vector<float>>& result) {
    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> context(cc::dense_big_ctx_create(),
                                                                          cc::dense_big_ctx_destroy);
    if (!context) throw std::runtime_error("dense context");
    std::uint32_t seed = 0x5EEDU + test.width;
    Managed<float> input(std::size_t{test.rows} * test.width);
    for (std::size_t i = 0; i < input.size; ++i)
        input.data[i] = std::bit_cast<float>((next(seed) & 0xFFFFU) << 16U | 0x3C000000U);  // BF16-exact, no Inf/NaN
    std::vector<std::unique_ptr<Weights>> weights;
    std::vector<std::unique_ptr<Managed<float>>> outputs;
    std::vector<core::Bf16RowProjection> projections;
    for (const unsigned width : test.outputs) {
        weights.push_back(std::make_unique<Weights>(width, test.width, seed));
        outputs.push_back(std::make_unique<Managed<float>>(std::size_t{test.rows} * width));
        projections.push_back({weights.back()->matrix.data, width, outputs.back()->data, false});
    }
    int planes = kStale;
    const auto status = cc::enqueue_bf16_dense_rows(input.data, test.width, test.rows, projections, nullptr,
                                                    cc::DenseArith::mma3, context.get(), profile, operand, &planes);
    check(cudaDeviceSynchronize());
    result.clear();
    for (const auto& output : outputs) result.emplace_back(output->data, output->data + output->size);
    return {status, planes};
}

bool run() {
    constexpr auto exact = core::Bf16Operand::bf16_exact, fp32 = core::Bf16Operand::fp32;
    std::vector<std::vector<float>> a, b;
    bool ok = true;
    const auto expect = [&](const char* name, const Case& test, core::Bf16Operand operand, int wanted,
                            const strix::device::Profile* profile = nullptr) {
        const auto outcome = dispatch(test, operand, profile, a);
        const bool passed = outcome.status == cudaSuccess && outcome.planes == wanted;
        std::printf("bf16 plane report %-32s planes %d (expected %d) %s\n", name, outcome.planes, wanted, passed ? "ok" : "FAIL");
        ok &= passed;
    };
    // Tall tile (width divisible by 32, outputs divisible by 16).
    expect("tall mma3 fp32 operand", {64, 64, {32}}, fp32, 3);
    expect("tall mma3 bf16-exact operand", {64, 64, {32}}, exact, 1);
    // The FP32 row grid: output width 17 is not a multiple of 16, so the projection splits nothing.
    expect("fp32 grid bf16-exact operand", {64, 64, {17}}, exact, 0);
    expect("fp32 grid fp32 operand", {64, 64, {17}}, fp32, 0);
    {
        // The declaration does not change what the grid computes.
        const auto x = dispatch({64, 64, {17}}, exact, nullptr, a), y = dispatch({64, 64, {17}}, fp32, nullptr, b);
        const bool same = x.status == cudaSuccess && y.status == cudaSuccess && a.size() == b.size() &&
                          !std::memcmp(a[0].data(), b[0].data(), a[0].size() * sizeof(float));
        std::printf("bf16 plane report fp32 grid output equal across operands %s\n", same ? "ok" : "FAIL");
        ok &= same;
    }
    // A mixed dispatch reports the split that ran (the tall projection's), not the grid's absence of one.
    expect("mixed tall + grid bf16-exact", {64, 64, {32, 17}}, exact, 1);
    expect("mixed grid + tall fp32", {64, 64, {17, 32}}, fp32, 3);
    // Short rows: no tall split, and the previous value is cleared.
    expect("short rows bf16-exact", {64, 8, {32}}, exact, 0);
    expect("short rows fp32 operand", {64, 8, {32}}, fp32, 0);
    // The device-profile rows planner returns before the tall plan: it must not leave the stale value either.
    int device = 0, units = 0;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&units, cudaDevAttrMultiProcessorCount, device));
    strix::device::Profile profile{};
    profile.units = static_cast<unsigned>(units);
    expect("short rows device profile", {64, 8, {32}}, exact, 0, &profile);
    // A rejected dispatch (input width not a multiple of 8) reports 0, not the stale value.
    const auto rejected = dispatch({36, 64, {32}}, exact, nullptr, a);
    const bool rejected_ok = rejected.status != cudaSuccess && rejected.planes == 0;
    std::printf("bf16 plane report rejected dispatch                planes %d (expected 0) %s\n", rejected.planes,
                rejected_ok ? "ok" : "FAIL");
    return ok && rejected_ok;
}

}  // namespace

int main() {
    try {
        if (!run()) {
            std::fprintf(stderr, "CUDA BF16 plane report: FAILED\n");
            return 1;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA BF16 plane report: exception: %s\n", error.what());
        return 1;
    }
    std::printf("CUDA BF16 plane report: PASSED\n");
    return 0;
}
