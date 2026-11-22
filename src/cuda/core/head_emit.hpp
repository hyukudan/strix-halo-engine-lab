#pragma once

#include "strix/head_argmax.hpp"
#include "strix/logprob_partials.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

namespace strix::cuda::core::head {

namespace ha = ::strix::head_argmax;
inline constexpr std::uint32_t kMaxCandidates = 64;

struct Candidate {
    float value;
    std::uint32_t id;
};
static_assert(sizeof(Candidate) == 8);
inline constexpr Candidate kNoCandidate{0.0F, ha::kNone};

// The compact readback. Maximum and centered sum retain the information needed to score a logit
// even when the addition in lse has rounded away log(sum). An empty finite set has max/lse = -inf and sum = 0.
struct RowLse {
    float maximum;
    float sum;
    float lse;
    std::uint32_t nonfinite;
};
static_assert(sizeof(RowLse) == 16);

struct DraftChoice {
    std::uint32_t id;
    float confidence;
};
static_assert(sizeof(DraftChoice) == 8);

// A contiguous vocabulary slice per row. Stride is in floats, zero means count.
struct Rows {
    const float* logits;
    std::uint32_t rows;
    std::uint32_t count;
    std::uint32_t stride = 0;
    std::uint32_t first = 0;
};

// Independent host ordering reference. The requested output width is capped at kMaxCandidates.
// Unused ranks retain the sentinel. As in the pick contract, nonfinite logits never enter the list.
inline void reference_candidates(std::span<const float> logits, std::uint32_t first, std::span<Candidate> out) {
    if (out.empty()) return;
    std::fill(out.begin(), out.end(), kNoCandidate);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float value = logits[i];
        const auto id = static_cast<std::uint32_t>(first + i);
        if (!std::isfinite(value) || !ha::better(value, id, out.back().value, out.back().id)) continue;
        const auto insertion = std::lower_bound(out.begin(), out.end(), Candidate{value, id},
            [](const Candidate& lhs, const Candidate& rhs) { return ha::better(lhs.value, lhs.id, rhs.value, rhs.id); });
        std::move_backward(insertion, out.end() - 1, out.end());
        *insertion = {value, id};
    }
}

inline RowLse reference_lse(std::span<const float> logits) {
    // Reuse the established CPU lane-order oracle. The CUDA exp implementation can differ in its last bits.
    const auto partial = ::strix::logprob_partial_host(logits, 0, 0);
    return {partial.max, partial.sum, partial.max + std::log(partial.sum), partial.nonfinite};
}

}  // namespace strix::cuda::core::head
