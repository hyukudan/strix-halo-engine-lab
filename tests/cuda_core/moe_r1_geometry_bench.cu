// No-model bench of the shared R1 MoE decode kernels (microscaled_r1 gate_up + down_combine) over launch geometries:
// columns per warp (microscaled_r1.hpp Geometry). Synthetic experts of a top-8 MoE shape
// (hidden 2304, intermediate 1152, 128 experts) in NVFP4 and MXFP8; 16 routed id sets rotate so the expert bytes come
// from DRAM, not L2. Every geometry must give the activation and output bytes of the one-column geometry (bitwise);
// then us per launch, registers and local memory per variant. --check-only = identity only.

#include "moe_core/microscaled_r1.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace packed = strix::weight::microscaled;
namespace r1 = strix::moe_core::microscaled_r1;
namespace grouped = strix::moe_core::device_grouped;
using packed::MatrixFormat;

constexpr unsigned kHidden = 2304, kInter = 1152, kExperts = 128, kTop = 8, kSets = 16;
bool g_fail = false;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error %s: %s\nRESULT FAIL\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <class T>
T* upload(const std::vector<T>& host) {
    T* p = nullptr;
    check(cudaMalloc(&p, host.size() * sizeof(T)), "malloc");
    check(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}

template <MatrixFormat F>
struct Pack {
    static constexpr unsigned ratio = F == MatrixFormat::nvfp4 ? 2U : 1U, block = F == MatrixFormat::nvfp4 ? 16U : 32U;
    static std::uint8_t code(std::mt19937& rng) {
        if constexpr (F == MatrixFormat::nvfp4) return static_cast<std::uint8_t>(rng());
        std::uint8_t c = static_cast<std::uint8_t>(rng());
        return (c & 0x7fU) == 0x7fU ? static_cast<std::uint8_t>(c & 0xf0U) : c;  // no E4M3 NaN
    }
    static std::uint8_t scale(std::mt19937& rng) {  // E4M3 0x30..0x3f (NVFP4), E8M0 120..131 (MXFP8)
        return F == MatrixFormat::nvfp4 ? static_cast<std::uint8_t>(0x30U + rng() % 16U) : static_cast<std::uint8_t>(120U + rng() % 12U);
    }
    static packed::PackedMatrix matrix(std::mt19937& rng, unsigned rows, unsigned cols, packed::RowOrder order) {
        std::vector<std::uint8_t> codes(std::size_t{rows} * cols / ratio), scales(std::size_t{rows} * cols / block);
        for (auto& c : codes) c = code(rng);
        for (auto& s : scales) s = scale(rng);
        return {upload(codes), upload(scales), rows, cols, F == MatrixFormat::nvfp4 ? 1.0F / 64.0F : 1.0F, order};
    }
};

struct Buffers {
    grouped::ExpertWeights* matrices;
    unsigned* ids[kSets];
    float *weights, *input, *output;
    std::uint16_t* activation;
};

template <MatrixFormat F>
Buffers make(std::uint64_t& expert_bytes) {
    std::mt19937 rng(F == MatrixFormat::nvfp4 ? 41U : 43U);
    std::vector<grouped::ExpertWeights> experts(kExperts);
    for (auto& e : experts) {
        e.gate_up = Pack<F>::matrix(rng, 2U * kInter, kHidden, packed::RowOrder::interleaved_halves);
        e.down = Pack<F>::matrix(rng, kHidden, kInter, packed::RowOrder::contiguous);
    }
    expert_bytes = (3ULL * kInter * kHidden) / Pack<F>::ratio + (3ULL * kInter * kHidden) / Pack<F>::block;
    Buffers b{};
    b.matrices = upload(experts);
    std::vector<unsigned> all(kExperts);
    for (unsigned i = 0; i < kExperts; ++i) all[i] = i;
    for (unsigned s = 0; s < kSets; ++s) {
        std::vector<unsigned> ids;
        for (unsigned row = 0; row < r1::kMaxRows; ++row) {
            std::shuffle(all.begin(), all.end(), rng);
            ids.insert(ids.end(), all.begin(), all.begin() + kTop);
        }
        b.ids[s] = upload(ids);
    }
    std::uniform_real_distribution<float> u(-1.0F, 1.0F), w(0.05F, 0.3F);
    std::vector<float> weights(r1::kMaxRows * kTop), input(std::size_t{r1::kMaxRows} * kHidden);
    for (auto& x : weights) x = w(rng);
    for (auto& x : input) x = u(rng);
    b.weights = upload(weights);
    b.input = upload(input);
    check(cudaMalloc(&b.output, sizeof(float) * r1::kMaxRows * kHidden), "malloc");
    check(cudaMalloc(&b.activation, sizeof(std::uint16_t) * r1::kMaxRows * kTop * kInter), "malloc");
    return b;
}

r1::Args args_of(const Buffers& b, unsigned set) {
    return {b.matrices, b.ids[set], b.weights, nullptr, b.input, b.activation, b.output, kExperts, kHidden, kInter,
            30.0F, 30.0F};
}

std::uint64_t fnv(const void* p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ static_cast<const unsigned char*>(p)[i]) * 1099511628211ULL;
    return h;
}

struct Reference {
    std::vector<std::uint16_t> activation;
    std::vector<float> output;
};

template <MatrixFormat F, unsigned GC, unsigned DC>
void variant(const Buffers& b, std::uint64_t expert_bytes, unsigned rows, int iters, bool check_only, Reference& ref) {
    const auto gate = r1::detail::gate_up<F, kTop, GC>;
    const auto down = r1::detail::down_combine<F, kTop, DC>;
    const dim3 gate_grid(r1::gate_blocks(kInter, GC), rows * kTop), down_grid(r1::down_blocks(kHidden, DC), rows);
    const unsigned threads = r1::detail::kThreads;
    check(cudaMemset(b.output, 0, sizeof(float) * rows * kHidden), "memset");
    gate<<<gate_grid, threads>>>(args_of(b, 0));
    down<<<down_grid, threads>>>(args_of(b, 0));
    check(cudaDeviceSynchronize(), "identity run");
    std::vector<std::uint16_t> act(std::size_t{rows} * kTop * kInter);
    std::vector<float> out(std::size_t{rows} * kHidden);
    check(cudaMemcpy(act.data(), b.activation, act.size() * sizeof(act[0]), cudaMemcpyDeviceToHost), "copy");
    check(cudaMemcpy(out.data(), b.output, out.size() * sizeof(out[0]), cudaMemcpyDeviceToHost), "copy");
    const bool first = ref.activation.empty();
    if (first) ref = {act, out};
    const bool equal = act == ref.activation && std::memcmp(out.data(), ref.output.data(), out.size() * sizeof(float)) == 0;
    cudaFuncAttributes ga{}, da{};
    check(cudaFuncGetAttributes(&ga, gate), "attr");
    check(cudaFuncGetAttributes(&da, down), "attr");
    const bool ok = equal && ga.localSizeBytes == 0 && da.localSizeBytes == 0;
    if (!ok) g_fail = true;
    const char* fmt = F == MatrixFormat::nvfp4 ? "nvfp4" : "mxfp8";
    std::printf("BITS %s rows=%u g%u d%u regs gate=%d down=%d local=%zu/%zu out=%016llx %s %s\n", fmt, rows, GC, DC,
                ga.numRegs, da.numRegs, ga.localSizeBytes, da.localSizeBytes,
                static_cast<unsigned long long>(fnv(out.data(), out.size() * sizeof(float))),
                first ? "REF" : (equal ? "EQUAL" : "DIFF"), ok ? "PASS" : "FAIL");
    if (check_only) return;
    cudaEvent_t e0, e1, e2;
    check(cudaEventCreate(&e0), "event");
    check(cudaEventCreate(&e1), "event");
    check(cudaEventCreate(&e2), "event");
    for (int i = 0; i < 20; ++i) gate<<<gate_grid, threads>>>(args_of(b, static_cast<unsigned>(i) % kSets));
    check(cudaEventRecord(e0), "record");
    for (int i = 0; i < iters; ++i) gate<<<gate_grid, threads>>>(args_of(b, static_cast<unsigned>(i) % kSets));
    check(cudaEventRecord(e1), "record");
    for (int i = 0; i < iters; ++i) down<<<down_grid, threads>>>(args_of(b, static_cast<unsigned>(i) % kSets));
    check(cudaEventRecord(e2), "record");
    check(cudaEventSynchronize(e2), "sync");
    float g_ms = 0.0F, d_ms = 0.0F;
    check(cudaEventElapsedTime(&g_ms, e0, e1), "elapsed");
    check(cudaEventElapsedTime(&d_ms, e1, e2), "elapsed");
    const double g_us = 1000.0 * g_ms / iters, d_us = 1000.0 * d_ms / iters;
    // expert bytes per launch counted per routed assignment (rows > 1 may share experts: then L2 serves some of them)
    const double gate_mb = rows * kTop * expert_bytes * (2.0 / 3.0) / 1e6, down_mb = rows * kTop * expert_bytes / 3.0 / 1e6;
    std::printf("TIME %s rows=%u g%u d%u gate_us=%.2f down_us=%.2f sum_us=%.2f gate_GBs=%.0f down_GBs=%.0f\n", fmt, rows,
                GC, DC, g_us, d_us, g_us + d_us, gate_mb * 1e3 / g_us, down_mb * 1e3 / d_us);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
}

template <MatrixFormat F>
void run_format(int iters, bool check_only) {
    std::uint64_t expert_bytes = 0;
    const Buffers b = make<F>(expert_bytes);
    for (const unsigned rows : {1U, 4U}) {
    Reference ref;
    variant<F, 1, 1>(b, expert_bytes, rows, iters, check_only, ref);  // reference geometry: first
    variant<F, 2, 2>(b, expert_bytes, rows, iters, check_only, ref);
    variant<F, 2, 4>(b, expert_bytes, rows, iters, check_only, ref);
    variant<F, 4, 4>(b, expert_bytes, rows, iters, check_only, ref);
    // the backend default geometry of this build must also match
    variant<F, r1::detail::kGateColumns, r1::detail::kDownColumns>(b, expert_bytes, rows, iters, check_only, ref);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 200;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--check-only")) check_only = true;
    }
    std::printf("GEOMETRY default g%u d%u\n", r1::detail::kGateColumns, r1::detail::kDownColumns);
    run_format<MatrixFormat::nvfp4>(iters, check_only);
    run_format<MatrixFormat::mxfp8>(iters, check_only);
    std::printf("RESULT %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
