// EXACT screened greedy LM head (src/cuda/core/head_screened.cuh), device test and bench. Mirrors the HIP device test.
// 1. The head transform G = row_transforms::bf16_softcap (nvcc tanhf) is monotone non-decreasing over every finite BF16
//    input (exhaustive, 65,536 patterns, for each tested cap): the screen's exactness relies on it.
// 2. The mapped pass 2 (dense_bf16_fp32_r1_mapped, trait of the incumbent R=1 row kernel) stores bit-identical logits to
//    the incumbent dense_bf16_fp32_r1_epilogue(softcap): mapped candidate ids and the all-rows form.
// 3. Screened Pick (value bits, index, non-finite count) equals the full head (incumbent kernel + head_argmax pick) for
//    random rows, exact ties, a peaked row, a non-finite activation (fallback), more candidates than the capacity
//    (fallback) and an overflow-guard activation (fallback).
// 4. Bench at the Inkling 2 Mini head shape (200,058 x 2,304): full vs screened per head. REFUSED unless 1-3 PASS.
// usage: cuda_core_screened_head_test [--bench]   (no flag: tests only)
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../../src/cuda/core/head_screened.cuh"
#include "strix/weight/formats/scalar_float.hpp"

namespace {

namespace hs = strix::cuda::core::head_screened;
namespace hp = strix::cuda::core::head_argmax;
namespace rt = strix::core::row_transforms;
namespace cc = strix::cuda_core;
namespace ha = strix::head_argmax;

int failures = 0;

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                           \
        }                                                                                           \
    } while (0)

struct EmitPick {
    ha::Pick* out;
    __device__ void operator()(std::uint32_t row, const ha::Pick& p) const { out[row] = p; }
};

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
    if (!(rt::bf16_softcap(v, cap) <= rt::bf16_softcap(w, cap))) atomicAdd(bad, 1U);
}

template <class T>
T* device_alloc(std::size_t count) {
    void* p = nullptr;
    CK(cudaMalloc(&p, count * sizeof(T)));
    return static_cast<T*>(p);
}

void monotone(float cap) {
    auto* bad = device_alloc<unsigned>(1);
    CK(cudaMemset(bad, 0, sizeof(unsigned)));
    monotone_kernel<<<256, 256>>>(cap, bad);
    unsigned host = 0U;
    CK(cudaMemcpy(&host, bad, sizeof host, cudaMemcpyDeviceToHost));
    std::printf("screened-head-cuda monotone cap=%g: %u violations\n", static_cast<double>(cap), host);
    if (host != 0U) ++failures;
    CK(cudaFree(bad));
}

std::uint16_t to_bf16(float v) { return static_cast<std::uint16_t>(strix::weight::scalar_float::bf16_encode(v)); }

std::uint32_t bits(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    return u;
}

// The full head: incumbent R=1 kernel with the softcap store, then the shared pick (the two launches of head_argmax).
struct FullHead {
    unsigned rows;
    float* y;
    ha::Pick* partial;
    ha::Pick* out;
    cudaError_t enqueue(const std::uint16_t* w, const float* x, unsigned width, float cap, cudaStream_t s) const {
        if (const auto e = cc::dense_bf16_fp32_r1_epilogue(w, x, y, width, rows, cc::Bf16Fp32Store::softcap, cap, s);
            e != cudaSuccess)
            return e;
        hp::View v;
        v.logits = y;
        v.count = rows;
        v.rows = 1U;
        v.partial = partial;
        return hp::pick_rows(v, EmitPick{out}, s);
    }
};

FullHead make_full(unsigned rows) {
    FullHead f{rows, device_alloc<float>(rows), device_alloc<ha::Pick>(ha::kMaxChunks), device_alloc<ha::Pick>(1)};
    return f;
}
void free_full(FullHead& f) {
    CK(cudaFree(f.y));
    CK(cudaFree(f.partial));
    CK(cudaFree(f.out));
}

ha::Pick read_pick(const ha::Pick* d) {
    ha::Pick p;
    CK(cudaMemcpy(&p, d, sizeof p, cudaMemcpyDeviceToHost));
    return p;
}
bool same(const ha::Pick& a, const ha::Pick& b) {
    return a.index == b.index && a.nonfinite == b.nonfinite && (a.index == ha::kNone || bits(a.value) == bits(b.value));
}

struct Case {
    const char* name;
    unsigned rows;
    int kind;  // 0 random, 1 tie of the best row, 2 peaked, 3 non-finite x, 4 identical rows (over capacity), 5 overflow x
};

std::vector<std::uint16_t> make_weights(const Case& c, unsigned width, std::vector<float>& x, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> w(std::size_t{c.rows} * width);
    if (c.kind == 4) {
        std::vector<std::uint16_t> row(width);
        for (auto& v : row) v = to_bf16(0.02F * normal(rng));
        for (unsigned r = 0; r < c.rows; ++r) std::copy_n(row.data(), width, w.data() + std::size_t{r} * width);
    } else {
        for (auto& v : w) v = to_bf16(0.02F * normal(rng));
    }
    x.resize(width);
    for (auto& v : x) v = normal(rng);
    if (c.kind == 3) x[width / 2U] = std::numeric_limits<float>::quiet_NaN();
    if (c.kind == 5)
        for (auto& v : x) v *= 1.0e30F;  // nx * max A >= 1e37: the overflow guard
    // Ties / peaks: copy a row aligned with x over a later row.
    if (c.kind == 1 || c.kind == 2) {
        for (unsigned k = 0; k < width; ++k) {
            const float boost = c.kind == 2 ? 0.2F : 0.05F;
            w[k] = to_bf16((x[k] > 0.0F ? boost : -boost) + 0.01F * normal(rng));
        }
        std::copy_n(w.data(), width, w.data() + std::size_t{c.rows - 3U} * width);
    }
    return w;
}

// Mapped pass 2 == the incumbent rows, bitwise: random ids (mapped form) and count > capacity (all rows).
void mapped_identity(const std::uint16_t* dw, const float* dx, unsigned rows, unsigned width, float cap, const float* full) {
    std::mt19937 rng(77U);
    constexpr unsigned count = 1500U, capacity = 2048U;
    std::vector<std::uint32_t> ids(count);
    for (auto& i : ids) i = rng() % rows;
    auto* did = device_alloc<std::uint32_t>(count);
    auto* dcount = device_alloc<std::uint32_t>(1);
    auto* dy = device_alloc<float>(std::max(rows, capacity));
    CK(cudaMemcpy(did, ids.data(), count * 4U, cudaMemcpyHostToDevice));
    std::vector<float> got(rows);
    for (unsigned mode = 0; mode < 2U; ++mode) {
        const std::uint32_t n = mode == 0U ? count : 0xffffffffU;
        CK(cudaMemcpy(dcount, &n, 4U, cudaMemcpyHostToDevice));
        for (const unsigned blocks : {7U, 1024U}) {  // the grid never changes y
            CK(cudaMemset(dy, 0, std::size_t{std::max(rows, capacity)} * 4U));
            CK(cc::dense_bf16_fp32_r1_mapped_epilogue(dw, dx, dy, did, dcount, capacity, rows, width,
                                                      cc::Bf16Fp32Store::softcap, cap, blocks, nullptr));
            CK(cudaMemcpy(got.data(), dy, std::size_t{mode == 0U ? count : rows} * 4U, cudaMemcpyDeviceToHost));
            std::size_t bad = 0;
            for (unsigned g = 0; g < (mode == 0U ? count : rows); ++g) {
                const unsigned src = mode == 0U ? ids[g] : g;
                if (bits(got[g]) != bits(full[src])) ++bad;
            }
            if (bad != 0U) ++failures;
            std::printf("%s screened-head-cuda mapped-rows %s blocks=%u: %zu mismatching logits\n", bad ? "FAIL" : "PASS",
                        mode == 0U ? "ids" : "all", blocks, bad);
        }
    }
    CK(cudaFree(did));
    CK(cudaFree(dcount));
    CK(cudaFree(dy));
}

void run(const Case& c, unsigned width, float cap, unsigned bits_per_code, std::mt19937& rng) {
    std::vector<float> x;
    const auto w = make_weights(c, width, x, rng);
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dx = device_alloc<float>(width);
    CK(cudaMemcpy(dw, w.data(), w.size() * 2U, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), width * 4U, cudaMemcpyHostToDevice));
    auto full = make_full(c.rows);
    CK(full.enqueue(dw, dx, width, cap, nullptr));
    CK(cudaDeviceSynchronize());
    const auto want = read_pick(full.out);
    std::vector<float> yfull(c.rows);
    CK(cudaMemcpy(yfull.data(), full.y, c.rows * 4U, cudaMemcpyDeviceToHost));
    if (bits_per_code == 8U && width == 2304U && (c.kind == 0 || c.kind == 3)) mapped_identity(dw, dx, c.rows, width, cap, yfull.data());
    hs::ResidentScreen screen(dw, c.rows, width, cap, nullptr, bits_per_code);
    auto* dpick = device_alloc<ha::Pick>(1);
    if (!screen.usable()) {
        std::printf("FAIL %s: screen not usable\n", c.name);
        ++failures;
    } else {
        for (unsigned repeat = 0; repeat < 2U; ++repeat) {  // the second launch reuses the scratch state
            CK(cudaMemset(dpick, 0xff, sizeof(ha::Pick)));
            CK(screen.enqueue(dx, EmitPick{dpick}, nullptr));
            CK(cudaDeviceSynchronize());
            const auto got = read_pick(dpick);
            const bool ok = same(got, want);
            if (!ok) ++failures;
            std::printf("%s screened-head-cuda %-10s K=%u bits=%u rows=%u: full {%u,%u} screened {%u,%u}\n",
                        ok ? "PASS" : "FAIL", c.name, width, bits_per_code, c.rows, want.index, want.nonfinite, got.index,
                        got.nonfinite);
        }
    }
    CK(cudaFree(dpick));
    free_full(full);
    CK(cudaFree(dw));
    CK(cudaFree(dx));
}

bool run_tests() {
    for (const float cap : {30.0F, 15.0F, 50.0F}) monotone(cap);
    std::mt19937 rng(1234U);
    const Case cases[] = {{"random", 6000U, 0}, {"tie", 6000U, 1},      {"peaked", 6000U, 2},
                          {"nan-x", 3000U, 3},  {"capacity", 20000U, 4}, {"overflow", 3000U, 5}};
    for (const unsigned width : {2304U, 512U})
        for (const unsigned bits_per_code : {8U, 4U})
            for (const auto& c : cases) run(c, width, 30.0F, bits_per_code, rng);
    return failures == 0;
}

// Informational timing at the Inkling 2 Mini head shape (200,058 x 2,304 BF16, one dominant row so the candidate set is
// small): the full head (incumbent R=1 kernel + pick) versus the five screened launches, plain stream and one graph replay.
void bench() {
    constexpr unsigned rows = 200058U, width = 2304U, iterations = 50U;
    std::mt19937 rng(4321U);
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
    CK(cudaMemcpy(dw, w.data(), w.size() * 2U, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), width * 4U, cudaMemcpyHostToDevice));
    w = {};
    auto full = make_full(rows);
    hs::ResidentScreen screen(dw, rows, width, 30.0F, nullptr);
    if (!screen.usable()) {
        std::printf("FAIL screened-head-cuda bench: screen not usable\n");
        ++failures;
        return;
    }
    auto* dpick = device_alloc<ha::Pick>(1);
    cudaStream_t stream{};
    CK(cudaStreamCreate(&stream));
    cudaEvent_t start{}, stop{};
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));
    CK(full.enqueue(dw, dx, width, 30.0F, stream));
    CK(cudaStreamSynchronize(stream));
    const auto want = read_pick(full.out);
    const auto time_us = [&](auto&& launch) {
        for (unsigned i = 0; i < 3U; ++i) launch();
        CK(cudaStreamSynchronize(stream));
        CK(cudaEventRecord(start, stream));
        for (unsigned i = 0; i < iterations; ++i) launch();
        CK(cudaEventRecord(stop, stream));
        CK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CK(cudaEventElapsedTime(&ms, start, stop));
        return static_cast<double>(ms) * 1000.0 / iterations;
    };
    const double full_us = time_us([&] { CK(full.enqueue(dw, dx, width, 30.0F, stream)); });
    const auto base = screen.grids();
    double best_us = 1.0e30;
    for (const unsigned mult : {1U, 2U}) {  // pass-1 grid sweep around the device default (informational)
        auto g = base;
        g.pass1 = base.pass1 * mult / 2U;
        screen.set_grids(g);
        const double us = time_us([&] { CK(screen.enqueue(dx, EmitPick{dpick}, stream)); });
        std::printf("screened-head-cuda bench pass1 blocks=%u: %.1f us\n", g.pass1, us);
        best_us = std::min(best_us, us);
    }
    screen.set_grids(base);
    CK(cudaMemset(dpick, 0xff, sizeof(ha::Pick)));
    const double screened_us = time_us([&] { CK(screen.enqueue(dx, EmitPick{dpick}, stream)); });
    const auto got = read_pick(dpick);
    // Graph replay: the five launches captured once (no host read inside).
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CK(screen.enqueue(dx, EmitPick{dpick}, stream));
    CK(cudaStreamEndCapture(stream, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));
    CK(cudaMemset(dpick, 0xff, sizeof(ha::Pick)));
    const double graph_us = time_us([&] { CK(cudaGraphLaunch(exec, stream)); });
    const auto got_graph = read_pick(dpick);
    const bool ok = same(got, want) && same(got_graph, want);
    if (!ok) ++failures;
    std::printf("%s screened-head-cuda bench rows=%u width=%u: full %.1f us, screened %.1f us (graph %.1f us) per head, "
                "best pass1 sweep %.1f us (plane+scratch %zu B)\n",
                ok ? "PASS" : "FAIL", rows, width, full_us, screened_us, graph_us, best_us, screen.device_bytes());
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    CK(cudaStreamDestroy(stream));
    CK(cudaFree(dpick));
    free_full(full);
    CK(cudaFree(dw));
    CK(cudaFree(dx));
}

}  // namespace

int main(int argc, char** argv) {
    bool want_bench = false;
    for (int i = 1; i < argc; ++i) want_bench = want_bench || std::strcmp(argv[i], "--bench") == 0;
    const bool passed = run_tests();
    if (!passed) {
        std::fprintf(stderr, "screened_head_cuda: %d failure(s)%s\n", failures,
                     want_bench ? "; bench REFUSED (tests did not PASS)" : "");
        return 1;
    }
    std::printf("screened_head_cuda: tests PASS\n");
    if (want_bench) {
        bench();
        if (failures != 0) {
            std::fprintf(stderr, "screened_head_cuda: bench identity failure\n");
            return 1;
        }
    }
    std::printf("screened_head_cuda: PASS\n");
    return 0;
}
