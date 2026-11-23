#pragma once
// Shared EXACT chunk-partial combine of the FP32 online-softmax attention forms (F4 selected, exact class): a row's key
// list is split into cnt <= kMaxChunks chunks, each chunk wrote its partial (m, l, acc[kDim]) (running max, running sum
// of unnormalised probabilities, unnormalised FP32 accumulator); the combine merges them in chunk order:
//   big = max over chunks cc = 0 .. cnt-1 ascending (a < b ? b : a, from -inf)
//   e_cc = exp(m_cc - big)                                  (the engine's declared exponential)
//   L = fmaf chain over cc ascending from +0 of (l_cc, e_cc)
//   o[d] = (fmaf chain over cc ascending from +0 of (acc_cc[d], e_cc)) / L
// Host statement of these chains: chunk_exact_combine_rule.hpp (tests/chunk_exact_combine_tests.cpp).
// Identity (EXACT): any code shape that evaluates these chains gives the same bits. The FAST flash forms
// (sparse_flash_keysplit.hpp, sparse_flash_rowgroup.hpp) merge in other orders and with exp2: other bits.
//
// Spec S (engine data, no behaviour): static constexpr unsigned kDim (multiple of 32: one wave, lane owns kDim / 32
// consecutive dims), kMaxChunks (1 .. 32: chunk cc's m and l are held by lane cc); static float exp(float)
// (device-callable). Partial layout: pt[cc * (2 + kDim) + {0: m, 1: l, 2 + d: acc[d]}], cc < kMaxChunks.
//
// combine_lanes<S>(pt, cnt, lane, out): one wave per (row, head); every chunk's (m, l) and the lane's dims of acc are
// loaded up front (independent loads), m and l broadcast by readlane; out() -> float* o (the row's output, evaluated
// after the denominator chain), o[lane * kDim / 32 + i] written by its lane. cnt == 0 (an empty row) returns before any
// load and before out(): the output row is left as it was (the engines skip such rows; no 0 / 0 NaN is written).
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::attn_core::chunk_exact {

template <class S> inline constexpr unsigned kPartial = 2U + S::kDim;
template <class S> inline constexpr unsigned kDimsPerLane = S::kDim / 32U;
template <class S> inline constexpr bool kSpecOk = S::kDim % 32U == 0 && S::kMaxChunks >= 1U && S::kMaxChunks <= 32U;

__device__ __forceinline__ float max_of(float a, float b) { return a < b ? b : a; }
__device__ inline float as_float(std::uint32_t x) { return __builtin_bit_cast(float, x); }
__device__ inline std::uint32_t as_u32(float x) { return __builtin_bit_cast(std::uint32_t, x); }
__device__ __forceinline__ float readlane(float v, int lane) {
    return as_float(static_cast<std::uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(as_u32(v)), lane)));
}

template <class S, class Out> __device__ __forceinline__ void combine_lanes(const float* pt, unsigned cnt, unsigned lane, Out out) {
    static_assert(kSpecOk<S>, "chunk_exact: kDim multiple of 32, 1 <= kMaxChunks <= 32");
    constexpr unsigned kMaxChunks = S::kMaxChunks, kDimsPerLane = chunk_exact::kDimsPerLane<S>, kPartial = chunk_exact::kPartial<S>;
    if (cnt == 0U) return;  // empty row: no chunk, no partial read, the output callback is not invoked, o is untouched
    const float mc = lane < cnt ? pt[lane * kPartial] : -__builtin_inff();
    const float lc = lane < cnt ? pt[lane * kPartial + 1] : 0.0F;
    float ac[kMaxChunks][kDimsPerLane];
#pragma unroll
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        for (unsigned i = 0; i < kDimsPerLane; ++i) ac[cc][i] = cc < cnt ? pt[cc * kPartial + 2 + lane * kDimsPerLane + i] : 0.0F;
    float big = -__builtin_inff();
#pragma unroll
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        if (cc < cnt) big = max_of(big, readlane(mc, static_cast<int>(cc)));
    const float ec_own = S::exp(mc - big);  // lane cc: e of chunk cc
    float L = 0.0F, e[kMaxChunks];
#pragma unroll
    for (unsigned cc = 0; cc < kMaxChunks; ++cc)
        if (cc < cnt) {
            e[cc] = readlane(ec_own, static_cast<int>(cc));
            L = __builtin_fmaf(readlane(lc, static_cast<int>(cc)), e[cc], L);
        }
    float* o = out();
    for (unsigned i = 0; i < kDimsPerLane; ++i) {
        float a = 0.0F;
#pragma unroll
        for (unsigned cc = 0; cc < kMaxChunks; ++cc)
            if (cc < cnt) a = __builtin_fmaf(ac[cc][i], e[cc], a);
        o[lane * kDimsPerLane + i] = a / L;
    }
}

} // namespace strix::attn_core::chunk_exact
