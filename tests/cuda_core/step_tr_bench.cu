// No-model per-class decode step cost table T(R), R = 1, 2, 4, 8, 16 rows, at Inkling 2 Mini shapes (synthetic weights).
// Classes (each timed alone, CUDA events around a captured graph that replays >= 32 launch sequences):
//   proj_attn  q 3072x2304 + k 1024x2304 + v 1024x2304 (one dense_rows_multi launch) + o 2304x3072 (dense_rows), BF16
//   proj_mlp   shared MLP gate+up 1152x2304 (dense_rows_multi) + down 2304x1152 + router 128x2304 (dense_rows), BF16
//   moe        128 experts, top-8 per row, NVFP4 gate/up 2304x2304 (gate+up interleaved) + down 2304x1152, shared
//              microscaled R1 rows launch (moe_core::microscaled_r1::launch, <= 8 routed rows per launch; R = 16 is two)
//   head       BF16 200058x2304: R = 1 r1 FP32 epilogue (softcap 30); R > 1 dense_rows (MMA) + softcap; then greedy pick
//              (shared argmax_rows = partial + final)
//   attention  not covered (the decode attention launcher is bound to the model's prepared state, no simple synthetic setup)
// Arithmetic class: BASE cost table. R = 1 projections use the FP32 R1 form, R > 1 the production default (mma3); no
// identity claim. L2 is defeated by rotating >= 2x the device L2 bytes of weight copies; allocations fit inside half of the initially free device memory.
// usage: strix-cuda-step-tr-bench [--replays N]   rc != 0 on any CUDA error or after the 30 s wall cap.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "core/pointwise.hpp"
#include "core/row_transforms.hpp"
#include "cuda/core/dense/rows_bf16_epilogue.cuh"
#include "moe_core/microscaled_r1.hpp"
#include "strix/cuda_core/dense.hpp"

namespace cc = strix::cuda_core;
namespace r1 = strix::moe_core::microscaled_r1;
namespace dg = strix::moe_core::device_grouped;
using strix::weight::microscaled::MatrixFormat;
using strix::weight::microscaled::PackedOperand;

namespace {
#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                           \
        }                                                                                           \
    } while (0)

constexpr unsigned kHidden = 2304, kQ = 3072, kKv = 1024, kMlp = 1152, kRouter = 128, kExperts = 128, kTop = 8;
constexpr unsigned kVocab = 200058;
constexpr float kSoftcap = 30.0F;
constexpr unsigned kAttnVisits = 48, kMoeLayers = 24;  // spec text; the model config header does not carry layer counts
constexpr unsigned kRs[] = {1, 2, 4, 8, 16};
constexpr unsigned kMinLaunches = 32;
std::size_t g_mem_cap = 0;
constexpr double kWallCapS = 30.0;

const auto g_start = std::chrono::steady_clock::now();
void wall_check() {
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
    if (s > kWallCapS) {
        std::fprintf(stderr, "wall-clock cap %.0f s exceeded\n", kWallCapS);
        std::exit(4);
    }
}

std::size_t g_bytes = 0;
std::vector<void*> g_allocs;
void* dev_alloc(std::size_t n) {
    if (n > g_mem_cap - g_bytes) {
        std::fprintf(stderr, "synthetic benchmark exceeds the device memory budget\n");
        std::exit(5);
    }
    void* p = nullptr;
    CK(cudaMalloc(&p, n));
    g_bytes += n;
    g_allocs.push_back(p);
    return p;
}
void dev_free_all() {
    for (void* p : g_allocs) CK(cudaFree(p));
    g_allocs.clear();
    g_bytes = 0;
}

__device__ __forceinline__ std::uint32_t mix(std::uint64_t i, std::uint32_t seed) {
    std::uint64_t z = i * 0x9E3779B97F4A7C15ULL + seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return static_cast<std::uint32_t>((z ^ (z >> 31)) >> 16);
}
__global__ void fill_bf16(std::uint16_t* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        const float v = (static_cast<float>(mix(i, seed) & 0xffffU) / 65535.0F - 0.5F) * 0.06F;
        p[i] = static_cast<std::uint16_t>(__float_as_uint(v) >> 16);
    }
}
__global__ void fill_f32(float* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x)
        p[i] = static_cast<float>(mix(i, seed) & 0xffffU) / 32767.5F - 1.0F;
}
__global__ void fill_bytes(std::uint8_t* p, std::size_t n, std::uint32_t seed, int scales) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        const std::uint32_t h = mix(i, seed);
        p[i] = scales ? static_cast<std::uint8_t>(32U + (h & 31U)) : static_cast<std::uint8_t>(h);
    }
}
std::uint16_t* make_bf16(std::size_t n, std::uint32_t seed) {
    auto* p = static_cast<std::uint16_t*>(dev_alloc(n * 2));
    fill_bf16<<<4096, 256>>>(p, n, seed);
    CK(cudaGetLastError());
    return p;
}
float* make_f32(std::size_t n, std::uint32_t seed) {
    auto* p = static_cast<float*>(dev_alloc(n * 4));
    fill_f32<<<1024, 256>>>(p, n, seed);
    CK(cudaGetLastError());
    return p;
}

int g_replays = 15;
std::size_t g_l2 = 0;

// Captures `steps` calls of enqueue(step) into one graph; median over replays of event time / steps, in us.
template <class F>
double graph_us(cudaStream_t s, unsigned steps, F&& enqueue) {
    CK(cudaStreamSynchronize(s));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (unsigned i = 0; i < steps; ++i) {
        const cudaError_t e = enqueue(i);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(e));
            std::exit(3);
        }
    }
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < 2; ++i) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    std::vector<double> t;
    for (int i = 0; i < g_replays; ++i) {
        wall_check();
        CK(cudaEventRecord(a, s));
        CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(b, s));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(static_cast<double>(ms) * 1000.0 / steps);
    }
    std::sort(t.begin(), t.end());
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
    return t[t.size() / 2];
}

unsigned copies_for(std::size_t bytes) { return static_cast<unsigned>(std::max<std::size_t>(1, (2 * g_l2 + bytes - 1) / bytes)); }
unsigned steps_for(unsigned copies, unsigned launches_per_step) {
    return std::max({copies, (kMinLaunches + launches_per_step - 1) / launches_per_step, 1U});
}
cc::DenseWeight bf16w(const std::uint16_t* p, unsigned out, unsigned in) {
    cc::DenseWeight w;
    w.data = p;
    w.format = cc::WeightFormat::bf16;
    w.out = out;
    w.in = in;
    return w;
}
cc::DenseArith arith_for(unsigned r) { return r == 1 ? cc::DenseArith::fp32 : cc::DenseArith::mma3; }

void line(const char* cls, unsigned r, double us, double bytes, const char* extra = "") {
    std::printf("class=%s R=%u us=%.2f us_per_row=%.2f gbps=%.1f%s\n", cls, r, us, us / r, bytes / (us * 1e3), extra);
    std::fflush(stdout);
}

struct Table {
    double attn[5], mlp[5], moe[5], head[5];
};

struct Io {  // activations for the dense classes (rows <= 16)
    float *x_hidden, *x_q, *x_mlp, *y_q, *y_k, *y_v, *y_o, *y_g, *y_u, *y_d, *y_r;
};

void run_proj(cudaStream_t s, Table& tab) {
    constexpr std::size_t qkv_o = (std::size_t{kQ} + 2 * kKv) * kHidden + std::size_t{kHidden} * kQ;
    constexpr std::size_t mlp = 3 * std::size_t{kMlp} * kHidden + std::size_t{kRouter} * kHidden;
    const unsigned ca = copies_for(qkv_o * 2), cm = copies_for(mlp * 2);
    struct Set { const std::uint16_t *q, *k, *v, *o, *g, *u, *d, *r; };
    std::vector<Set> A(ca), M(cm);
    std::uint32_t seed = 1;
    for (auto& c : A) c = {make_bf16(std::size_t{kQ} * kHidden, seed++), make_bf16(std::size_t{kKv} * kHidden, seed++),
                           make_bf16(std::size_t{kKv} * kHidden, seed++), make_bf16(std::size_t{kHidden} * kQ, seed++)};
    for (auto& c : M) c = {nullptr, nullptr, nullptr, nullptr, make_bf16(std::size_t{kMlp} * kHidden, seed++),
                           make_bf16(std::size_t{kMlp} * kHidden, seed++), make_bf16(std::size_t{kHidden} * kMlp, seed++),
                           make_bf16(std::size_t{kRouter} * kHidden, seed++)};
    Io io{};
    io.x_hidden = make_f32(16U * kHidden, 91);
    io.x_q = make_f32(16U * kQ, 92);
    io.x_mlp = make_f32(16U * kMlp, 93);
    io.y_q = static_cast<float*>(dev_alloc(16U * kQ * 4));
    io.y_k = static_cast<float*>(dev_alloc(16U * kKv * 4));
    io.y_v = static_cast<float*>(dev_alloc(16U * kKv * 4));
    io.y_o = static_cast<float*>(dev_alloc(16U * kHidden * 4));
    io.y_g = static_cast<float*>(dev_alloc(16U * kMlp * 4));
    io.y_u = static_cast<float*>(dev_alloc(16U * kMlp * 4));
    io.y_d = static_cast<float*>(dev_alloc(16U * kHidden * 4));
    io.y_r = static_cast<float*>(dev_alloc(16U * kRouter * 4));
    CK(cudaDeviceSynchronize());
    for (int i = 0; i < 5; ++i) {
        const unsigned r = kRs[i];
        const auto ar = arith_for(r);
        tab.attn[i] = graph_us(s, steps_for(ca, 2), [&](unsigned st) {
            const auto& c = A[st % ca];
            const cc::DenseWeight w[3] = {bf16w(c.q, kQ, kHidden), bf16w(c.k, kKv, kHidden), bf16w(c.v, kKv, kHidden)};
            float* y[3] = {io.y_q, io.y_k, io.y_v};
            auto e = cc::dense_rows_multi(w, y, 3, io.x_hidden, r, nullptr, s, ar);
            if (e != cudaSuccess) return e;
            return cc::dense_rows(bf16w(c.o, kHidden, kQ), io.x_q, io.y_o, r, nullptr, s, ar);
        });
        line("proj_attn", r, tab.attn[i], static_cast<double>(qkv_o) * 2);
        tab.mlp[i] = graph_us(s, steps_for(cm, 3), [&](unsigned st) {
            const auto& c = M[st % cm];
            const cc::DenseWeight w[2] = {bf16w(c.g, kMlp, kHidden), bf16w(c.u, kMlp, kHidden)};
            float* y[2] = {io.y_g, io.y_u};
            auto e = cc::dense_rows_multi(w, y, 2, io.x_hidden, r, nullptr, s, ar);
            if (e != cudaSuccess) return e;
            e = cc::dense_rows(bf16w(c.d, kHidden, kMlp), io.x_mlp, io.y_d, r, nullptr, s, ar);
            if (e != cudaSuccess) return e;
            return cc::dense_rows(bf16w(c.r, kRouter, kHidden), io.x_hidden, io.y_r, r, nullptr, s, ar);
        });
        line("proj_mlp", r, tab.mlp[i], static_cast<double>(mlp) * 2);
    }
    dev_free_all();
}

void run_moe(cudaStream_t s, Table& tab) {
    using Op = PackedOperand<MatrixFormat::nvfp4>;
    using strix::weight::microscaled::PackedMatrix;
    using strix::weight::microscaled::RowOrder;
    constexpr std::size_t gu_rows = 2U * kMlp, gu_cols = kHidden, dn_rows = kHidden, dn_cols = kMlp;
    constexpr std::size_t gu_codes = gu_rows * gu_cols / 2, gu_scales = gu_rows * gu_cols / Op::kBlockColumns;
    constexpr std::size_t dn_codes = dn_rows * dn_cols / 2, dn_scales = dn_rows * dn_cols / Op::kBlockColumns;
    constexpr std::size_t per_expert = gu_codes + gu_scales + dn_codes + dn_scales;
    const unsigned nc = std::max(2U, copies_for(per_expert * kExperts));
    struct Copy { dg::ExpertWeights* experts; };
    std::vector<Copy> copy(nc);
    for (unsigned c = 0; c < nc; ++c) {
        auto* codes = static_cast<std::uint8_t*>(dev_alloc((gu_codes + dn_codes) * kExperts));
        auto* scales = static_cast<std::uint8_t*>(dev_alloc((gu_scales + dn_scales) * kExperts));
        fill_bytes<<<4096, 256>>>(codes, (gu_codes + dn_codes) * kExperts, 100 + c, 0);
        fill_bytes<<<4096, 256>>>(scales, (gu_scales + dn_scales) * kExperts, 200 + c, 1);
        CK(cudaGetLastError());
        std::vector<dg::ExpertWeights> host(kExperts);
        for (unsigned e = 0; e < kExperts; ++e) {
            auto* cp = codes + e * (gu_codes + dn_codes);
            auto* sp = scales + e * (gu_scales + dn_scales);
            host[e].gate_up = PackedMatrix{cp, sp, static_cast<unsigned>(gu_rows), static_cast<unsigned>(gu_cols), 0.375F,
                                           RowOrder::interleaved_halves};
            host[e].down = PackedMatrix{cp + gu_codes, sp + gu_scales, static_cast<unsigned>(dn_rows),
                                        static_cast<unsigned>(dn_cols), 0.375F, RowOrder::contiguous};
        }
        copy[c].experts = static_cast<dg::ExpertWeights*>(dev_alloc(sizeof(dg::ExpertWeights) * kExperts));
        CK(cudaMemcpy(copy[c].experts, host.data(), sizeof(dg::ExpertWeights) * kExperts, cudaMemcpyHostToDevice));
    }
    float* input = make_f32(16U * kHidden, 7);
    float* output = static_cast<float*>(dev_alloc(16U * kHidden * 4));
    auto* act = static_cast<std::uint16_t*>(dev_alloc(std::size_t{8} * kTop * kMlp * 2));
    std::mt19937 rng(12345);
    for (int i = 0; i < 5; ++i) {
        const unsigned r = kRs[i];
        const unsigned steps = steps_for(nc, 2 * ((r + 7) / 8));
        std::vector<unsigned> ids(std::size_t{steps} * r * kTop);
        double union_sum = 0;
        for (unsigned st = 0; st < steps; ++st) {
            std::vector<bool> seen(kExperts, false);
            for (unsigned row = 0; row < r; ++row) {
                unsigned pool[kExperts];
                for (unsigned e = 0; e < kExperts; ++e) pool[e] = e;
                for (unsigned k = 0; k < kTop; ++k) {
                    std::swap(pool[k], pool[k + rng() % (kExperts - k)]);
                    ids[(std::size_t{st} * r + row) * kTop + k] = pool[k];
                    seen[pool[k]] = true;
                }
            }
            union_sum += static_cast<double>(std::count(seen.begin(), seen.end(), true));
        }
        const double uni = union_sum / steps;
        auto* d_ids = static_cast<unsigned*>(dev_alloc(ids.size() * 4));
        CK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> hw(std::size_t{r} * kTop, 0.125F);
        auto* d_w = static_cast<float*>(dev_alloc(hw.size() * 4));
        CK(cudaMemcpy(d_w, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
        const double us = graph_us(s, steps, [&](unsigned st) {
            for (unsigned done = 0; done < r; done += 8) {
                const unsigned rows = std::min(8U, r - done);
                r1::Args a{copy[st % nc].experts, d_ids + (std::size_t{st} * r + done) * kTop, d_w + std::size_t{done} * kTop,
                           nullptr, input + std::size_t{done} * kHidden, act, output + std::size_t{done} * kHidden, kExperts,
                           kHidden, kMlp, 4.0F, 25.0F};
                const auto e = r1::launch<MatrixFormat::nvfp4, kTop>(a, s, rows);
                if (e != cudaSuccess) return e;
            }
            return cudaSuccess;
        });
        tab.moe[i] = us;
        char extra[96];
        std::snprintf(extra, sizeof extra, " union_experts=%.1f/%u", uni, kExperts);
        line("moe", r, us, uni * static_cast<double>(per_expert), extra);
    }
    dev_free_all();
}

void run_head(cudaStream_t s, Table& tab) {
    const std::size_t wbytes = std::size_t{kVocab} * kHidden * 2;
    const unsigned nc = copies_for(wbytes);
    std::vector<const std::uint16_t*> w(nc);
    for (unsigned c = 0; c < nc; ++c) w[c] = make_bf16(std::size_t{kVocab} * kHidden, 300 + c);
    float* x = make_f32(16U * kHidden, 55);
    auto* logits = static_cast<float*>(dev_alloc(std::size_t{16} * kVocab * 4));
    auto* ws = dev_alloc(cc::kArgmaxWorkspaceBytes);
    auto* tok = static_cast<std::uint32_t*>(dev_alloc(16 * 4));
    CK(cudaDeviceSynchronize());
    for (int i = 0; i < 5; ++i) {
        const unsigned r = kRs[i];
        tab.head[i] = graph_us(s, steps_for(nc, 3), [&](unsigned st) {
            const auto* wp = w[st % nc];
            cudaError_t e;
            if (r == 1) {
                e = cc::dense_bf16_fp32_r1_epilogue(wp, x, logits, kHidden, kVocab, cc::Bf16Fp32Store::softcap, kSoftcap, s);
            } else {
                e = cc::dense_rows(bf16w(wp, kVocab, kHidden), x, logits, r, nullptr, s, cc::DenseArith::mma3);
                if (e != cudaSuccess) return e;
                e = strix::core::pointwise::elements(std::size_t{r} * kVocab,
                                                     strix::core::row_transforms::Bf16Softcap{logits, kSoftcap}, s);
            }
            if (e != cudaSuccess) return e;
            return cc::argmax_rows(logits, kVocab, r, tok, ws, s);
        });
        line("head", r, tab.head[i], static_cast<double>(wbytes));
    }
    dev_free_all();
}
}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--replays")) g_replays = std::max(15, std::atoi(argv[i + 1]));
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, dev));
    std::size_t free_bytes = 0, total_bytes = 0;
    CK(cudaMemGetInfo(&free_bytes, &total_bytes));
    g_mem_cap = free_bytes / 2U;
    int l2 = 0, core_khz = 0, mem_khz = 0, bus = 0;
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    CK(cudaDeviceGetAttribute(&core_khz, cudaDevAttrClockRate, dev));
    CK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
    CK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));
    g_l2 = static_cast<std::size_t>(l2);
    std::printf("device=%s sm_%d%d sms=%d l2_bytes=%d max_core_khz=%d max_mem_khz=%d bus_bits=%d\n", p.name, p.major, p.minor,
                p.multiProcessorCount, l2, core_khz, mem_khz, bus);
    std::printf("memory_budget_bytes=%zu total_bytes=%zu\n", g_mem_cap, total_bytes);
    std::printf("state=BASE (R=1 FP32 R1 forms, R>1 mma3 defaults; synthetic weights; no identity claim); power/clock state "
                "not queryable here, record it beside this output\n");
    std::printf("timing: graph of >= %u launch sequences, median of %d replays, weight copies rotated >= 2x L2\n", kMinLaunches,
                g_replays);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    Table tab{};
    run_proj(s, tab);
    run_moe(s, tab);
    run_head(s, tab);
    std::printf("attention: not covered\n");
    for (int i = 0; i < 5; ++i) {
        const double step = kAttnVisits * tab.attn[i] + kMoeLayers * (tab.mlp[i] + tab.moe[i]) + tab.head[i];
        std::printf("PREDICTION step R=%u us=%.0f us_per_row=%.0f (%u x proj_attn + %u x (proj_mlp + moe) + head; attention, "
                    "norms, mixes, recurrences excluded; visit/layer counts from the spec text)\n",
                    kRs[i], step, step / kRs[i], kAttnVisits, kMoeLayers);
    }
    CK(cudaStreamDestroy(s));
    CK(cudaDeviceSynchronize());
    std::printf("total_wall_s=%.1f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count());
    return 0;
}
