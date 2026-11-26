// EXACT screened greedy LM head (src/cuda/core/head_screened.cuh), device test and bench. Mirrors the HIP device test.
// 1. The head transform G = row_transforms::bf16_softcap (nvcc tanhf) is monotone non-decreasing over every finite BF16
//    input (exhaustive, 65,536 patterns, for each tested cap): the screen's exactness relies on it.
// 2. The mapped pass 2 (dense_bf16_fp32_r1_mapped, trait of the incumbent R=1 row kernel) stores bit-identical logits to
//    the incumbent dense_bf16_fp32_r1_epilogue(softcap): mapped candidate ids and the all-rows form.
// 3. Screened Pick (value bits, index, non-finite count) equals the full head (incumbent kernel + head_argmax pick) for
//    random rows, exact ties, a peaked row, a non-finite activation (fallback), more candidates than the capacity
//    (fallback) and an overflow-guard activation (fallback).
// 4. The incumbent forms an engine binds: the BF16 row projection dispatch (enqueue_bf16_dense_rows, FP32 rows kernel for
//    a vocabulary that is not a multiple of 16), the separate softcap pass (core::enqueue_bf16_softcap_head) and the shared
//    pick over R rows. The screened head over R lanes (strided activations, per-lane fallback) gives the same Pick per
//    lane, and lane l equals the R=1 head of its row (R = 1, 2, 3, 4, 5, 8).
// 5. Bench at the Inkling 2 Mini head shape (200,058 x 2,304): full vs screened per head, then R = 1, 2, 4, 8, 16, 32 lanes against
//    the incumbent multi-row head. REFUSED unless 1-4 PASS.
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
#include <span>
#include <vector>

#include "../../src/cuda/core/head_screened.cuh"
#include "core/bf16_lm_head.hpp"
#include "core/resident_greedy.hpp"
#include "cuda/core/dense/bf16_rows.hpp"
#include "strix/weight/formats/scalar_float.hpp"

namespace {

namespace hs = strix::cuda::core::head_screened;
namespace hp = strix::cuda::core::head_argmax;
namespace rt = strix::core::row_transforms;
namespace cc = strix::cuda_core;
namespace ha = strix::head_argmax;
namespace plan = strix::core::screened_head_plan;

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
struct EmitPick2 {
    ha::Pick2* out;
    __device__ void operator()(std::uint32_t row, const ha::Pick2& p) const { out[row] = p; }
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

// The engine form over R activation rows (contiguous x [R][width]): the BF16 row projection dispatch an engine binds
// (MMA3 requested; a vocabulary that is not a multiple of 16 runs the FP32 rows kernel), the separate softcap pass, then
// the shared pick over R rows (out[r]).
struct EngineHead {
    unsigned rows, lanes;
    float* y;
    ha::Pick* partial;
    ha::Pick* out;
    ha::Pick2* partial2;
    ha::Pick2* out2;
    cudaError_t enqueue(const std::uint16_t* w, const float* x, unsigned width, float cap, cudaStream_t s) const {
        const strix::core::Bf16RowProjection projection{w, rows, y};
        if (const auto e = strix::core::enqueue_bf16_softcap_head(lanes, projection, cap, s,
                [&](std::span<const strix::core::Bf16RowProjection> p) {
                    return cc::enqueue_bf16_dense_rows(x, width, lanes, p, s, cc::DenseArith::mma3, nullptr);
                });
            e != cudaSuccess)
            return e;
        hp::View v;
        v.logits = y;
        v.stride = rows;
        v.count = rows;
        v.rows = lanes;
        v.partial = partial;
        if (const auto e = hp::pick_rows(v, EmitPick{out}, s); e != cudaSuccess) return e;
        return hp::pick2_rows(v, partial2, EmitPick2{out2}, s);  // the runner-up trait over the same logits
    }
};

EngineHead make_engine(unsigned rows, unsigned lanes) {
    return {rows, lanes, device_alloc<float>(std::size_t{rows} * lanes),
            device_alloc<ha::Pick>(std::size_t{ha::kMaxChunks} * lanes), device_alloc<ha::Pick>(lanes),
            device_alloc<ha::Pick2>(ha::workspace2_bytes(lanes) / sizeof(ha::Pick2) + 1U), device_alloc<ha::Pick2>(lanes)};
}
void free_engine(EngineHead& f) {
    CK(cudaFree(f.y));
    CK(cudaFree(f.partial));
    CK(cudaFree(f.out));
    CK(cudaFree(f.partial2));
    CK(cudaFree(f.out2));
}

std::vector<ha::Pick2> read_picks2(const ha::Pick2* d, unsigned n) {
    std::vector<ha::Pick2> p(n);
    CK(cudaMemcpy(p.data(), d, n * sizeof(ha::Pick2), cudaMemcpyDeviceToHost));
    return p;
}

std::vector<ha::Pick> read_picks(const ha::Pick* d, unsigned n) {
    std::vector<ha::Pick> p(n);
    CK(cudaMemcpy(p.data(), d, n * sizeof(ha::Pick), cudaMemcpyDeviceToHost));
    return p;
}

ha::Pick read_pick(const ha::Pick* d) {
    ha::Pick p;
    CK(cudaMemcpy(&p, d, sizeof p, cudaMemcpyDeviceToHost));
    return p;
}
bool same(const ha::Pick& a, const ha::Pick& b) {
    return a.index == b.index && a.nonfinite == b.nonfinite && (a.index == ha::kNone || bits(a.value) == bits(b.value));
}
bool same2(const ha::Pick2& a, const ha::Pick2& b) {
    return same(a.first, b.first) && a.index2 == b.index2 && (a.index2 == ha::kNone || bits(a.value2) == bits(b.value2));
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
    // The shared resident picker reads a reply the screened head produced on the device (enqueue_produced) and returns
    // the same token as its own pick over the full transformed row (full.y holds the softcap logits).
    if (screen.usable()) {
        strix::core::ResidentGreedy produced(1U, c.rows), full_pick(1U, c.rows);
        produced.enqueue_produced([&](std::uint32_t* token) {
            CK(screen.enqueue(dx, strix::cuda::core::head_argmax::EmitIndexOrNone{token}, nullptr));
        }, nullptr);
        full_pick.enqueue(std::span<const float>(full.y, c.rows), nullptr);
        CK(cudaDeviceSynchronize());
        const auto a = produced.token(0), b = full_pick.token(0);
        const bool ok = a == b;
        if (!ok) ++failures;
        std::printf("%s screened-head-cuda resident-greedy %-10s K=%u bits=%u rows=%u: produced %u full %u\n",
                    ok ? "PASS" : "FAIL", c.name, width, bits_per_code, c.rows, a, b);
        // A producer that returns its launch status is checked: the status-returning form picks the same token, and a
        // rejected launch after a successful submission throws and leaves no published reply (not the stale token).
        produced.enqueue_produced([&](std::uint32_t* token) {
            return screen.enqueue(dx, strix::cuda::core::head_argmax::EmitIndexOrNone{token}, nullptr);
        }, nullptr);
        CK(cudaDeviceSynchronize());
        const bool status_ok = produced.token(0) == b;
        bool threw = false, none = false;
        try {
            produced.enqueue_produced([](std::uint32_t*) { return cudaErrorInvalidValue; }, nullptr);
        } catch (const std::runtime_error&) { threw = true; }
        try { (void)produced.token(0); } catch (const std::out_of_range&) { none = true; }
        const bool failed_ok = status_ok && threw && none;
        if (!failed_ok) ++failures;
        std::printf("%s screened-head-cuda resident-greedy-status %-10s: status-form %d rejected-throws %d no-stale-reply %d\n",
                    failed_ok ? "PASS" : "FAIL", c.name, status_ok ? 1 : 0, threw ? 1 : 0, none ? 1 : 0);
    }
    CK(cudaFree(dpick));
    free_full(full);
    CK(cudaFree(dw));
    CK(cudaFree(dx));
}

// R lanes against the engine form (one launch set over R rows) and against the R=1 head of every row. Rows: a vocabulary
// that is not a multiple of 16 (the FP32 rows kernel, as at 200,058). Lane kinds by l % 5: random, a tie of the best row,
// a peaked row, a non-finite activation (that lane falls back to all rows, the others stay screened), an overflow
// activation (lane 0 stays random).
void lanes_case(unsigned lanes, unsigned width, unsigned bits_per_code, std::mt19937& rng) {
    constexpr unsigned rows = 6002U;
    constexpr float cap = 30.0F;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> w(std::size_t{rows} * width);
    for (auto& v : w) v = to_bf16(0.02F * normal(rng));
    const unsigned stride = width + 8U;  // strided lanes for the screen, contiguous rows for the engine form
    std::vector<float> xs(std::size_t{lanes} * stride, 0.0F), xc(std::size_t{lanes} * width);
    for (unsigned l = 0; l < lanes; ++l) {
        float* x = xs.data() + std::size_t{l} * stride;
        for (unsigned k = 0; k < width; ++k) x[k] = normal(rng);
        const unsigned kind = l % 5U;
        if (kind == 1U || kind == 2U) {  // a row aligned with this lane, copied once more further down (tie)
            const unsigned row = 100U + 7U * l;
            const float boost = kind == 2U ? 0.2F : 0.05F;
            for (unsigned k = 0; k < width; ++k)
                w[std::size_t{row} * width + k] = to_bf16((x[k] > 0.0F ? boost : -boost) + 0.01F * normal(rng));
            std::copy_n(w.data() + std::size_t{row} * width, width, w.data() + std::size_t{rows - 3U - l} * width);
        }
        if (kind == 3U) x[width / 3U] = std::numeric_limits<float>::quiet_NaN();
        if (kind == 4U)
            for (unsigned k = 0; k < width; ++k) x[k] *= 1.0e30F;
        std::copy_n(x, width, xc.data() + std::size_t{l} * width);
    }
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dxs = device_alloc<float>(xs.size());
    auto* dxc = device_alloc<float>(xc.size());
    CK(cudaMemcpy(dw, w.data(), w.size() * 2U, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxs, xs.data(), xs.size() * 4U, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxc, xc.data(), xc.size() * 4U, cudaMemcpyHostToDevice));
    auto engine = make_engine(rows, lanes);
    CK(engine.enqueue(dw, dxc, width, cap, nullptr));
    CK(cudaDeviceSynchronize());
    const auto want = read_picks(engine.out, lanes);
    const auto want2 = read_picks2(engine.out2, lanes);
    auto single = make_full(rows);
    std::vector<ha::Pick> r1(lanes);
    for (unsigned l = 0; l < lanes; ++l) {
        CK(single.enqueue(dw, dxc + std::size_t{l} * width, width, cap, nullptr));
        CK(cudaDeviceSynchronize());
        r1[l] = read_pick(single.out);
    }
    hs::ResidentScreen screen(dw, rows, width, cap, nullptr, bits_per_code, lanes);
    auto* dpick = device_alloc<ha::Pick>(lanes);
    if (!screen.usable()) {
        std::printf("FAIL screened-head-cuda lanes=%u: screen not usable\n", lanes);
        ++failures;
    } else {
        for (unsigned repeat = 0; repeat < 2U; ++repeat) {  // the second launch reuses the scratch state
            CK(cudaMemset(dpick, 0xff, lanes * sizeof(ha::Pick)));
            CK(screen.enqueue(dxs, EmitPick{dpick}, nullptr, lanes, stride));
            CK(cudaDeviceSynchronize());
            const auto got = read_picks(dpick, lanes);
            std::vector<hs::State> states(lanes);
            CK(cudaMemcpy(states.data(), hs::lane_state(screen.launch(dxs, lanes, stride)), lanes * sizeof(hs::State),
                          cudaMemcpyDeviceToHost));
            unsigned bad = 0, fallback = 0, candidates = 0;
            for (unsigned l = 0; l < lanes; ++l) {
                if (!same(got[l], want[l]) || !same(want[l], r1[l])) ++bad;
                const auto o = strix::core::screened_head_plan::outcome(states[l], rows, hs::ResidentScreen::kCapacity);
                fallback += o.fallback ? 1U : 0U;
                if (!o.fallback) candidates = std::max(candidates, o.candidates);
            }
            if (bad != 0U) ++failures;
            std::printf("%s screened-head-cuda lanes=%u K=%u bits=%u rows=%u: %u lane(s) differ from the engine form or "
                        "the R=1 head; %u fallback lane(s), max %u candidates\n",
                        bad ? "FAIL" : "PASS", lanes, width, bits_per_code, rows, bad, fallback, candidates);
        }
        // Runner-up trait: the shared top-2 of every lane equals the engine form's pick2 (first and second).
        auto* dpick2 = device_alloc<ha::Pick2>(lanes);
        CK(cudaMemset(dpick2, 0xff, lanes * sizeof(ha::Pick2)));
        CK(screen.enqueue2(dxs, EmitPick2{dpick2}, nullptr, lanes, stride));
        CK(cudaDeviceSynchronize());
        const auto got2 = read_picks2(dpick2, lanes);
        unsigned bad2 = 0, candidates2 = 0;
        std::vector<hs::State> states2(lanes);
        CK(cudaMemcpy(states2.data(), hs::lane_state(screen.launch(dxs, lanes, stride, true)), lanes * sizeof(hs::State),
                      cudaMemcpyDeviceToHost));
        for (unsigned l = 0; l < lanes; ++l) {
            if (!same2(got2[l], want2[l]) || !same(got2[l].first, want[l])) ++bad2;
            const auto o = strix::core::screened_head_plan::outcome(states2[l], rows, hs::ResidentScreen::kCapacity);
            if (!o.fallback) candidates2 = std::max(candidates2, o.candidates);
        }
        if (bad2 != 0U) ++failures;
        std::printf("%s screened-head-cuda runner-up lanes=%u K=%u bits=%u: %u lane(s) differ from the engine top-2, "
                    "max %u candidates\n",
                    bad2 ? "FAIL" : "PASS", lanes, width, bits_per_code, bad2, candidates2);
        CK(cudaFree(dpick2));
    }
    CK(cudaFree(dpick));
    free_full(single);
    free_engine(engine);
    CK(cudaFree(dw));
    CK(cudaFree(dxs));
    CK(cudaFree(dxc));
}

bool run_tests() {
    for (const float cap : {30.0F, 15.0F, 50.0F}) monotone(cap);
    std::mt19937 rng(1234U);
    const Case cases[] = {{"random", 6000U, 0}, {"tie", 6000U, 1},      {"peaked", 6000U, 2},
                          {"nan-x", 3000U, 3},  {"capacity", 20000U, 4}, {"overflow", 3000U, 5}};
    for (const unsigned width : {2304U, 512U})
        for (const unsigned bits_per_code : {8U, 4U})
            for (const auto& c : cases) run(c, width, 30.0F, bits_per_code, rng);
    for (const unsigned lanes : {1U, 2U, 3U, 4U, 8U})
        for (const unsigned bits_per_code : {8U, 4U}) lanes_case(lanes, 2304U, bits_per_code, rng);
    lanes_case(5U, 512U, 8U, rng);
    return failures == 0;
}

// Informational timing at the Inkling 2 Mini head shape (200,058 x 2,304 BF16, one dominant row so the candidate set is
// small): the full head (incumbent R=1 kernel + pick) versus the five screened launches, plain stream and one graph replay.
void bench() {
    constexpr unsigned rows = 200058U, width = 2304U, iterations = 50U;
    std::mt19937 rng(4321U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    constexpr unsigned max_lanes = 32U;
    std::vector<float> xl(std::size_t{max_lanes} * width);
    for (auto& v : xl) v = normal(rng);
    // 1,024 random rows repeated (cheap to generate; repeated rows only add exact ties far below the dominant rows).
    std::vector<std::uint16_t> w(std::size_t{rows} * width);
    constexpr unsigned distinct = 1024U;
    for (std::size_t i = 0; i < std::size_t{distinct} * width; ++i) w[i] = to_bf16(0.02F * normal(rng));
    for (unsigned r = distinct; r < rows; ++r)
        std::copy_n(w.data() + std::size_t{r % distinct} * width, width, w.data() + std::size_t{r} * width);
    // Lane l has its own dominant row 777 + 1000 l (aligned with its activation).
    for (unsigned l = 0; l < max_lanes; ++l)
        for (unsigned k = 0; k < width; ++k)
            w[std::size_t{777U + 1000U * l} * width + k] = to_bf16(xl[std::size_t{l} * width + k] > 0.0F ? 0.01F : -0.01F);
    auto* dw = device_alloc<std::uint16_t>(w.size());
    auto* dxl = device_alloc<float>(xl.size());
    float* dx = dxl;  // lane 0 is the R=1 activation
    CK(cudaMemcpy(dw, w.data(), w.size() * 2U, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxl, xl.data(), xl.size() * 4U, cudaMemcpyHostToDevice));
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
    // Lanes: the engine form over R rows (one projection launch set, softcap pass, R-row pick) versus the screened head
    // over R lanes (pass 1 reads the plane from memory once: lane groups of pass1_lanes share it in L2), plus the
    // schedule identity of the pass-1 bounds against the R=1 form.
    {
        hs::ResidentScreen wide(dw, rows, width, 30.0F, stream, 8U, max_lanes);
        auto* dpicks = device_alloc<ha::Pick>(max_lanes);
        for (const unsigned lanes : {1U, 2U, 4U, 8U, 16U, 32U}) {
            auto engine = make_engine(rows, lanes);
            CK(engine.enqueue(dw, dxl, width, 30.0F, stream));
            CK(cudaStreamSynchronize(stream));
            const auto want_l = read_picks(engine.out, lanes);
            const double engine_us = time_us([&] { CK(engine.enqueue(dw, dxl, width, 30.0F, stream)); });
            CK(cudaMemset(dpicks, 0xff, max_lanes * sizeof(ha::Pick)));
            const double lanes_us = time_us([&] { CK(wide.enqueue(dxl, EmitPick{dpicks}, stream, lanes, width)); });
            const auto got_l = read_picks(dpicks, lanes);
            bool lanes_ok = wide.usable();
            for (unsigned l = 0; l < lanes; ++l) lanes_ok = lanes_ok && same(got_l[l], want_l[l]);
            // Schedule identity: the pass-1 bounds (upper[] of every row) and Lstar of each lane are bitwise the R=1
            // form's (one-row tiles, one group) on the same activation.
            const auto lane_launch = wide.launch(dxl, lanes, width);
            const auto lp = hs::detail::parts(lane_launch);
            std::vector<float> got_upper(rows), want_upper(rows);
            std::vector<hs::State> lane_states(lanes);
            CK(cudaMemcpy(lane_states.data(), lp.state, lanes * sizeof(hs::State), cudaMemcpyDeviceToHost));
            unsigned upper_diff = 0, lkey_diff = 0;
            for (unsigned l = 0; l < lanes; ++l) {
                CK(cudaMemcpy(got_upper.data(), lp.upper + std::size_t{l} * lp.strides.upper, rows * sizeof(float),
                              cudaMemcpyDeviceToHost));
                CK(screen.enqueue(dxl + std::size_t{l} * width, EmitPick{dpick}, stream));
                CK(cudaStreamSynchronize(stream));
                const auto rp = hs::detail::parts(screen.launch(dxl + std::size_t{l} * width));
                CK(cudaMemcpy(want_upper.data(), rp.upper, rows * sizeof(float), cudaMemcpyDeviceToHost));
                hs::State one{};
                CK(cudaMemcpy(&one, rp.state, sizeof(hs::State), cudaMemcpyDeviceToHost));
                if (std::memcmp(got_upper.data(), want_upper.data(), rows * sizeof(float)) != 0) ++upper_diff;
                if (one.lkey != lane_states[l].lkey) ++lkey_diff;
            }
            lanes_ok = lanes_ok && upper_diff == 0U && lkey_diff == 0U;
            if (!lanes_ok) ++failures;
            const unsigned groups = plan::pass1_groups(lanes, lane_launch.pass1_lanes);
            std::printf("%s screened-head-cuda pass1 schedule lanes=%u: %u group(s) x %u lanes, tile rows %u, blocks %u "
                        "(resident %u, cap %u); upper/Lstar vs R=1: %u/%u lane(s) differ\n",
                        upper_diff == 0U && lkey_diff == 0U ? "PASS" : "FAIL", lanes, groups, lane_launch.pass1_lanes,
                        plan::pass1_tile_rows(lane_launch.pass1_lanes),
                        plan::pass1_blocks(lane_launch.grids.pass1, lane_launch.pass1_resident, groups),
                        lane_launch.pass1_resident, lane_launch.grids.pass1, upper_diff, lkey_diff);
            auto* dpicks2 = device_alloc<ha::Pick2>(lanes);
            const double top2_us = time_us([&] { CK(wide.enqueue2(dxl, EmitPick2{dpicks2}, stream, lanes, width)); });
            CK(cudaFree(dpicks2));
            std::printf("%s screened-head-cuda bench lanes=%u (pass1 lanes %u): engine head %.1f us, screened %.1f us "
                        "(%.2fx), runner-up trait %.1f us\n",
                        lanes_ok ? "PASS" : "FAIL", lanes, hs::pass1_lanes(width, lanes), engine_us, lanes_us,
                        lanes_us / engine_us, top2_us);
            free_engine(engine);
        }
        CK(cudaFree(dpicks));
    }
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    CK(cudaStreamDestroy(stream));
    CK(cudaFree(dpick));
    free_full(full);
    CK(cudaFree(dw));
    CK(cudaFree(dxl));
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
