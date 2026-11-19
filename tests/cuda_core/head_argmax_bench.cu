// No-model check + timing of the shared greedy head pick (src/cuda/core/head_argmax.cuh).
// Identity: for vocabularies 248,320 / 154,880 / 129,280 and 1 / 4 / 8 / 16 rows of random logits on a coarse grid
// (many equal values) with planted ties at the maximum, +0 / -0 rows and rows holding NaN / Inf, every row's Pick equals
// the host reference (strix/head_argmax.hpp) word for word, for both launch forms: plain, and PDL (hp::Launch::pdl, the
// dense core's form) right after a producer kernel that writes the logits, so the pick's dependency wait is on the
// path. Timing: us per call of both shared forms and of the one-block-per-row form it replaces (the baseline arm below),
// median of `iters` CUDA-graph replays.
// usage: cuda_core_head_argmax_bench [--check-only] [--iters N]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "../../src/cuda/core/head_argmax.cuh"

namespace ha = strix::head_argmax;
namespace hp = strix::cuda::core::head_argmax;

namespace {
#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

struct EmitPick {
    ha::Pick* out;
    __device__ void operator()(std::uint32_t row, const ha::Pick& p) const { out[row] = p; }
};

// Baseline arm: the replaced form (one block of 1024 threads per row, one load in flight per thread).
// Producer for the PDL arm: writes the rows the pick reads (stale contents are zeros).
__global__ void produce_kernel(const float* src, float* dst, std::size_t n) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) dst[i] = src[i];
}

__global__ __launch_bounds__(1024) void one_block_kernel(const float* logits, std::uint32_t count, ha::Pick* out) {
    const float* x = logits + std::size_t{blockIdx.x} * count;
    ha::Pick p;
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float y = x[i];
        if (!isfinite(y)) ++p.nonfinite;
        else if (ha::better(y, i, p.value, p.index)) { p.value = y; p.index = i; }
    }
    __shared__ ha::Pick s[1024];
    s[threadIdx.x] = p;
    __syncthreads();
    for (unsigned d = 512U; d; d >>= 1U) {
        if (threadIdx.x < d) s[threadIdx.x] = ha::merge(s[threadIdx.x], s[threadIdx.x + d]);
        __syncthreads();
    }
    if (threadIdx.x == 0U) out[blockIdx.x] = s[0];
}

bool same(const ha::Pick& a, const ha::Pick& b) {
    return a.index == b.index && a.nonfinite == b.nonfinite && std::memcmp(&a.value, &b.value, 4) == 0;
}

float graph_us(cudaStream_t s, int iters, const auto& enqueue) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    enqueue();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ge, s));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> t;
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(a, s));
        CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(b, s));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms * 1000.0F);
    }
    std::sort(t.begin(), t.end());
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return t[t.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    bool check_only = false;
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--check-only") check_only = true;
        else if (a == "--iters" && i + 1 < argc) iters = std::max(1, std::atoi(argv[++i]));
    }
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device %s SMs %d\n", prop.name, prop.multiProcessorCount);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::mt19937 rng(1119U);
    int failures = 0;
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    for (const std::uint32_t vocab : {248320U, 154880U, 129280U}) {
        for (const std::uint32_t rows : {1U, 4U, 8U, 16U}) {
            std::vector<float> x(std::size_t{rows} * vocab);
            std::uniform_real_distribution<float> u(-12.0F, 12.0F);
            for (auto& v : x) v = std::round(u(rng) * 8.0F) / 8.0F;
            for (std::uint32_t r = 0; r < rows; ++r) {
                float* row = x.data() + std::size_t{r} * vocab;
                for (int k = 0; k < 5; ++k) row[rng() % vocab] = 13.0F;  // planted ties at the maximum
                if (r % 4 == 1) { row[rng() % vocab] = nan; row[rng() % vocab] = -inf; }
                if (r % 4 == 2) std::fill(row, row + vocab, 0.0F), row[vocab / 2] = -0.0F, row[7] = -0.0F;
                if (r % 4 == 3) row[rng() % vocab] = inf;
            }
            float *d_x = nullptr, *d_y = nullptr;
            ha::Pick *d_out = nullptr, *d_pdl = nullptr, *d_base = nullptr, *d_ws = nullptr, *d_ws2 = nullptr;
            CK(cudaMalloc(&d_x, x.size() * 4));
            CK(cudaMalloc(&d_y, x.size() * 4));
            CK(cudaMemset(d_y, 0, x.size() * 4));
            CK(cudaMalloc(&d_pdl, rows * sizeof(ha::Pick)));
            CK(cudaMalloc(&d_ws2, ha::workspace_bytes(rows)));
            CK(cudaMalloc(&d_out, rows * sizeof(ha::Pick)));
            CK(cudaMalloc(&d_base, rows * sizeof(ha::Pick)));
            CK(cudaMalloc(&d_ws, ha::workspace_bytes(rows)));
            CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            hp::View v;
            v.logits = d_x;
            v.count = vocab;
            v.rows = rows;
            v.partial = d_ws;
            hp::View vp = v;  // PDL arm reads the producer's output
            vp.logits = d_y;
            vp.partial = d_ws2;
            const auto produce_then_pick = [&] {
                produce_kernel<<<256, 256, 0, s>>>(d_x, d_y, x.size());
                CK(cudaGetLastError());
                CK(hp::pick_rows<hp::Launch::pdl>(vp, EmitPick{d_pdl}, s));
            };
            CK(hp::pick_rows(v, EmitPick{d_out}, s));
            produce_then_pick();
            one_block_kernel<<<rows, 1024, 0, s>>>(d_x, vocab, d_base);
            CK(cudaGetLastError());
            CK(cudaStreamSynchronize(s));
            std::vector<ha::Pick> got(rows), pdl(rows), base(rows);
            CK(cudaMemcpy(got.data(), d_out, rows * sizeof(ha::Pick), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(pdl.data(), d_pdl, rows * sizeof(ha::Pick), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(base.data(), d_base, rows * sizeof(ha::Pick), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (std::uint32_t r = 0; r < rows; ++r) {
                const ha::Pick ref = ha::reference({x.data() + std::size_t{r} * vocab, vocab});
                if (!same(got[r], ref) || !same(pdl[r], ref) || !same(base[r], ref)) ++bad;
            }
            failures += bad;
            if (check_only) {
                std::printf("pick vocab %u rows %u: %s\n", vocab, rows, bad ? "MISMATCH" : "ids equal");
            } else {
                const float shared_us = graph_us(s, iters, [&] { CK(hp::pick_rows(v, EmitPick{d_out}, s)); });
                const float produce_us = graph_us(s, iters, [&] {
                    produce_kernel<<<256, 256, 0, s>>>(d_x, d_y, x.size());
                    CK(cudaGetLastError());
                });
                const float pdl_us = graph_us(s, iters, produce_then_pick) - produce_us;
                const float base_us = graph_us(s, iters, [&] { one_block_kernel<<<rows, 1024, 0, s>>>(d_x, vocab, d_base); });
                std::printf("pick vocab %u rows %u: shared %.2f us, shared-pdl %.2f us (after producer), one-block %.2f us, %s\n",
                            vocab, rows, shared_us, pdl_us, base_us, bad ? "MISMATCH" : "ids equal");
            }
            CK(cudaFree(d_x));
            CK(cudaFree(d_out));
            CK(cudaFree(d_base));
            CK(cudaFree(d_ws));
            CK(cudaFree(d_y));
            CK(cudaFree(d_pdl));
            CK(cudaFree(d_ws2));
        }
    }
    std::printf("%s head pick: %d mismatching rows\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
