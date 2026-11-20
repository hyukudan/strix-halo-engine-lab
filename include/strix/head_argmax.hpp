#pragma once
// Greedy head pick, the one contract every backend body implements (CUDA: src/cuda/core/head_argmax.cuh).
// Pick = first maximum over the FINITE logits of a span (lowest index on ties, +0 == -0) plus the count of non-finite
// logits skipped. Each engine maps a Pick to its own output words (an adapter), never to a different rule.
// merge() is associative and commutative, so the pick does not depend on how the span is cut into chunks: the chunk
// count is launch geometry, never numerics (tests/head_argmax_tests.cpp proves it on planted ties).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_HEAD_ARGMAX_HD __host__ __device__
#else
#define STRIX_HEAD_ARGMAX_HD
#endif

namespace strix::head_argmax {

inline constexpr std::uint32_t kNone = 0xffffffffU;

struct Pick {
    float value = 0.0F;          // unspecified (0) when index == kNone
    std::uint32_t index = kNone;  // global index (first_index + position); kNone: no finite logit
    std::uint32_t nonfinite = 0;  // non-finite logits skipped
};
static_assert(sizeof(Pick) == 12, "Pick is three 32-bit words");

// true when candidate (v, i) beats (bv, bi).
STRIX_HEAD_ARGMAX_HD inline bool better(float v, std::uint32_t i, float bv, std::uint32_t bi) {
    if (i == kNone) return false;
    if (bi == kNone) return true;
    return v > bv || (v == bv && i < bi);
}

STRIX_HEAD_ARGMAX_HD inline Pick merge(Pick a, const Pick& b) {
    const std::uint32_t skipped = a.nonfinite + b.nonfinite;
    if (better(b.value, b.index, a.value, a.index)) a = b;
    a.nonfinite = skipped;
    return a;
}

// Host reference over one span.
inline Pick reference(std::span<const float> logits, std::uint32_t first_index = 0) {
    Pick p;
    for (std::size_t j = 0; j < logits.size(); ++j) {
        const float v = logits[j];
        const auto i = static_cast<std::uint32_t>(first_index + j);
        if (!std::isfinite(v)) ++p.nonfinite;
        else if (better(v, i, p.value, p.index)) { p.value = v; p.index = i; }
    }
    return p;
}

// Launch geometry: chunks per row (one partial pick each) and the partial workspace they need.
inline constexpr std::uint32_t kMaxChunks = 128;
[[nodiscard]] constexpr std::uint32_t default_chunks(std::uint64_t count) {
    const std::uint64_t c = count / 1024U;
    return c < 1U ? 1U : (c > kMaxChunks ? kMaxChunks : static_cast<std::uint32_t>(c));
}
[[nodiscard]] constexpr std::size_t workspace_bytes(std::uint32_t rows, std::uint32_t chunks = kMaxChunks) {
    return std::size_t{rows} * chunks * sizeof(Pick);
}
// Chunk c of `chunks` covers [lo, hi) of a row of `count` logits (the device body uses the same cut).
STRIX_HEAD_ARGMAX_HD inline void chunk_range(std::uint32_t count, std::uint32_t chunks, std::uint32_t c, std::uint32_t& lo,
                                             std::uint32_t& hi) {
    const std::uint32_t per = (count + chunks - 1U) / chunks;
    const std::uint64_t l = std::uint64_t{c} * per, h = l + per;
    lo = static_cast<std::uint32_t>(l < count ? l : count);
    hi = static_cast<std::uint32_t>(h < count ? h : count);
}

// Split-vocabulary pair (rank 0 owns the lower ids; MiniMax / GLM transfer form): per row each rank sends one PairWord,
// its Pick's value bits and global index, or index kNone when its half saw a non-finite logit (or no finite one). Both
// ranks then compute the same pair_pick: kNone (the row fails on BOTH ranks, after the exchange) when either word is
// kNone, else merge() of the two, so rank 0 wins ties. Host form of the device candidate / pick kernels of
// src/xport/channel_impl.hip (argmax_candidate_kernel, argmax_pick_kernel), which implement the same rule.
struct PairWord {
    std::uint32_t value_bits = 0;
    std::uint32_t index = kNone;
};
static_assert(sizeof(PairWord) == 8, "PairWord is the 8-byte wire word of one row");
inline PairWord pair_word(const Pick& p) {
    PairWord w;
    std::memcpy(&w.value_bits, &p.value, sizeof(float));
    w.index = p.nonfinite != 0U ? kNone : p.index;
    return w;
}
inline std::uint32_t pair_pick(const PairWord& c0, const PairWord& c1) {
    if (c0.index == kNone || c1.index == kNone) return kNone;
    Pick p0, p1;
    std::memcpy(&p0.value, &c0.value_bits, sizeof(float));
    std::memcpy(&p1.value, &c1.value_bits, sizeof(float));
    p0.index = c0.index;
    p1.index = c1.index;
    return merge(p0, p1).index;
}

}  // namespace strix::head_argmax
