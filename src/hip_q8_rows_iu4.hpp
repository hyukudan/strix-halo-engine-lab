#pragma once
// TURBO+ onfly INT4 g64 decode projections for the q8_rows sites (rows <= 8). A site opts in with
// `static constexpr unsigned turbo_group` (one bit of strix::turbo_onfly_groups); the launch uses the INT4 copies
// (q6p::iu4 registry, built at load by hip_qwen_onfly_i4.cpp) when EVERY segment of the site has one, else the
// Q8_0 kernels run unchanged. Shared kernel over the one W4A16 small-M core (core/w4a16.hpp): activations (BF16 / F32
// rows, or the dense A8 slot codes * deltas) are staged once per block as FP16, F32 accumulation, a block of S waves
// splits K and reduces its partial sums in LDS in a fixed order (deterministic), Site::store is the epilogue.
// Included from hip_q8_rows_gemv.hpp (namespace strix::q8_rows).
#include "core/iu4_onfly.hpp"
#include "strix/onfly_i4.hpp"
#include "core/w4a16.hpp"

namespace strix::q8_rows {

inline std::atomic<std::uint64_t>& iu4_dispatches() { static std::atomic<std::uint64_t> n{0}; return n; }
template <class Site> constexpr unsigned iu4_group_of() {
    if constexpr (requires { Site::turbo_group; }) return Site::turbo_group; else return 0U;
}
constexpr std::uint32_t kIu4StageBytes = 40U * 1024U;

struct Iu4Weights { const unsigned char* base[kMaxSegments]; };

// Grid: one block per (segment, 64 output columns); S waves split the K groups (G = K / 64 groups, G / S per wave).
template <class Site, unsigned S, unsigned G, bool FromSlot>
__global__ __launch_bounds__(32 * S) void project_rows_iu4_kernel(Site site, Iu4Weights weights, const std::int8_t* codes,
                                                                  const float* deltas) {
    constexpr unsigned R = Site::rows, K = G * 64U, MaxG = G / S, Threads = 32U * S;
    static_assert(G % S == 0U && MaxG % 2U == 0U && R >= 1U && R <= 8U && R * K * 2U <= kIu4StageBytes);
    static_assert(K * 2U >= S * 256U, "partial sums alias the staged activations");
    __shared__ __attribute__((aligned(16))) _Float16 xs[R * K];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid / 32U;
    // Block -> (segment m, 64-column quad).
    unsigned m = 0, quad = blockIdx.x;
    for (;; ++m) {
        const unsigned q = ((site.segments.first[m + 1U] - site.segments.first[m]) + 63U) / 64U;
        if (quad < q || m + 1U == site.segments.count) break;
        quad -= q;
    }
    const std::uint32_t n_real = site.segments.first[m + 1U] - site.segments.first[m], n_pad = (n_real + 63U) / 64U * 64U;
    // Stage the activations as FP16 (finite values saturate, NaN propagates).
    for (unsigned r = 0; r < R; ++r)
        for (unsigned i = tid * 8U; i < K; i += Threads * 8U) {
            float x[8];
            if constexpr (FromSlot) {
                const float delta = deltas[(i / 32U) * dense_a8_pitch<R>() + r];
                const std::int8_t* c = codes + static_cast<std::uint64_t>(r) * K + i;
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) x[j] = static_cast<float>(c[j]) * delta;
            } else load8(row_input(site, r) + i, x);
            _Float16 h[8];
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) h[j] = static_cast<_Float16>(x[j] != x[j] ? x[j] : fminf(65504.0F, fmaxf(-65504.0F, x[j])));
            *reinterpret_cast<uint4*>(xs + r * K + i) = *reinterpret_cast<const uint4*>(h);
        }
    __syncthreads();
    const auto* wq = weights.base[m];
    const auto* ws = reinterpret_cast<const std::uint16_t*>(wq + core::onfly::packed_bytes(n_pad, K));
    float res[R][2];
    const unsigned k0 = wave * MaxG * 4U, k1 = k0 + MaxG * 4U;
    core::w4a16::wave_rows<R, MaxG * 4U, MaxG>(wq, ws, n_pad, K, k0, k1, quad * 4U, xs + k0 * 16U, K, lane, R,
        [&](unsigned, unsigned r, unsigned column, float value) { res[r][column & 1U] = value; });
    __syncthreads();  // every wave is done reading the staged rows: reuse them for the partial sums
    float* part = reinterpret_cast<float*>(xs);
    const unsigned rel = (lane >> 3U) * 16U + 2U * (lane & 7U);
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        part[(r * S + wave) * 64U + rel] = res[r][0];
        part[(r * S + wave) * 64U + rel + 1U] = res[r][1];
    }
    __syncthreads();
    for (unsigned t = tid; t < R * 64U; t += Threads) {
        const unsigned r = t / 64U, c = t % 64U;
        float sum = part[(r * S) * 64U + c];
#pragma unroll
        for (unsigned s = 1; s < S; ++s) sum += part[(r * S + s) * 64U + c];
        if (quad * 64U + c < n_real) site.store(site.segments.first[m] + quad * 64U + c, r, sum);
    }
}

// True when the INT4 form launched. slot != nullptr: activations from the dense A8 slot (producer-quantized rows).
template <class Site> bool launch_iu4_rows(const Site& site, hipStream_t stream, bool any_order, const DenseA8Slot* slot) {
    if constexpr (Site::rows > 8U || !iu4_group_of<Site>()) return false;
    else {
        if (!q6p::iu4::count()) return false;
        const unsigned width = site.width;
        if (width % 64U || site.segments.count == 0U || Site::rows * width * 2U > kIu4StageBytes) return false;
        Iu4Weights w{};
        unsigned blocks = 0;
        for (unsigned m = 0; m < site.segments.count; ++m) {
            const auto* t = q6p::iu4::find(site.segments.weights[m]);
            const unsigned n = site.segments.first[m + 1U] - site.segments.first[m];
            if (!t || t->cols != width || t->rows != n || site.segments.types[m]) return false;
            w.base[m] = t->base;
            blocks += (n + 63U) / 64U;
        }
        const auto run = [&]<unsigned S, unsigned G>() {
            const auto go = [&](auto kernel, const std::int8_t* codes, const float* deltas) {
                if (any_order && !slot)
                    hipExtLaunchKernelGGL(kernel, dim3(blocks), dim3(32U * S), 0, stream, nullptr, nullptr, hipExtAnyOrderLaunch, site, w,
                        codes, deltas);
                else hipLaunchKernelGGL(kernel, dim3(blocks), dim3(32U * S), 0, stream, site, w, codes, deltas);
            };
            if (slot) go(project_rows_iu4_kernel<Site, S, G, true>, slot->codes, slot->deltas);
            else go(project_rows_iu4_kernel<Site, S, G, false>, nullptr, nullptr);
        };
        switch (width / 64U) {
        case 32U: run.template operator()<4U, 32U>(); break;
        case 40U: run.template operator()<4U, 40U>(); break;
        case 64U: run.template operator()<8U, 64U>(); break;
        case 96U: run.template operator()<8U, 96U>(); break;
        default: return false;
        }
        iu4_dispatches().fetch_add(1U, std::memory_order_relaxed);
        return true;
    }
}

}  // namespace strix::q8_rows
