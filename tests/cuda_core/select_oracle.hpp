#pragma once

#include "attn_core/select_spec.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

namespace select_oracle {
namespace sel = strix::cuda::core::select;
inline constexpr float ninf = -std::numeric_limits<float>::infinity();

// Host references for the two FP32 pooled-key arithmetic contracts.
inline float relu_tree(std::span<const float> query, std::span<const float, 128> key) {
    float total = 0.0F;
    for (std::size_t h = 0; h < query.size(); h += 128) {
        std::array<float, 32> plane{};
        for (std::size_t j = 0; j < plane.size(); ++j) {
            const float even = query[h + j] * key[j] + query[h + j + 64] * key[j + 64];
            const float odd = query[h + j + 32] * key[j + 32] + query[h + j + 96] * key[j + 96];
            plane[j] = even + odd;
        }
        for (std::size_t n = 16; n != 0; n /= 2)
            for (std::size_t j = 0; j < n; ++j) plane[j] += plane[j + n];
        total += std::fmax(plane.front(), 0.0F);
    }
    return total / std::sqrt(128.0F);
}

inline float weighted_fma(std::span<const float> query, std::span<const float, 128> key,
                          std::span<const float> weights) {
    float score = 0.0F;
    for (std::size_t h = 0; h < weights.size(); ++h) {
        float dot = 0.0F;
        for (std::size_t j = 0; j < key.size(); ++j) dot = std::fma(query[h * 128 + j], key[j], dot);
        score = std::fma(weights[h] * 0.17677669529663687F, std::fmax(0.0F, dot * 0.088388347648318F), score);
    }
    return score;
}

inline std::uint32_t float_key(float value, sel::KeyOrder order) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if (order != sel::KeyOrder::raw_float && value == 0.0F) bits = 0;
    return (bits >> 31U) != 0U ? ~bits : bits ^ 0x80000000U;
}

inline std::vector<unsigned> rank(std::span<const std::uint32_t> keys, unsigned budget) {
    std::vector<unsigned> ids(keys.size());
    std::iota(ids.begin(), ids.end(), 0U);
    std::stable_sort(ids.begin(), ids.end(), [&](unsigned a, unsigned b) { return keys[a] > keys[b]; });
    ids.resize(std::min<std::size_t>(budget, ids.size()));
    std::sort(ids.begin(), ids.end());
    return ids;
}

struct Result {
    std::vector<unsigned> ids;
    std::vector<float> scores;
};

inline Result pick(sel::PickSpec spec, unsigned groups, unsigned tail, std::span<const float> scores,
                   std::span<const std::uint32_t> ordered = {}, std::span<const std::uint32_t> ids = {},
                   std::span<const std::uint8_t> pool = {}, unsigned pool_width = 0) {
    Result out;
    out.scores.assign(scores.begin(), scores.end());
    const auto emit = [&](unsigned id) {
        const unsigned actual = ids.empty() ? id : ids[id];
        for (unsigned i = 0; i < spec.expansion; ++i) out.ids.push_back(actual * spec.expansion + i);
    };
    if (groups <= spec.top_k) {
        for (unsigned i = 0; i < groups; ++i) emit(i);
    } else {
        const unsigned end = groups - static_cast<unsigned>(spec.force_newest);
        const unsigned budget = spec.top_k - static_cast<unsigned>(spec.force_newest);
        const bool eligible = spec.order == sel::KeyOrder::eligible_float;
        std::vector<unsigned> candidates;
        std::vector<std::uint32_t> keys;
        for (unsigned i = 0; i < end; ++i) {
            if (eligible && !(scores[i] > ninf)) continue;
            candidates.push_back(i);
            const auto value = !pool.empty() && pool[i / pool_width] == 0 ? ninf :
                spec.order == sel::KeyOrder::ordered ? 0.0F : scores[i];
            keys.push_back(spec.order == sel::KeyOrder::ordered ? ordered[i] : float_key(value, spec.order));
        }
        const auto selected = rank(keys, budget);
        if (eligible && selected.size() < budget) {
            for (std::size_t i = selected.size(); i < budget; ++i) emit(0);
            out.scores[0] = ninf;
        }
        for (unsigned index : selected) {
            const unsigned id = candidates[index];
            emit(id);
            if (eligible) out.scores[id] = ninf;
        }
        if (spec.force_newest) emit(end);
    }
    for (unsigned i = 0; i < tail; ++i) out.ids.push_back(groups * spec.expansion + i);
    return out;
}

inline std::vector<std::uint8_t> pool(std::span<const float> scores, unsigned width, unsigned budget) {
    const auto size = (scores.size() + width - 1U) / width;
    std::vector<std::uint32_t> keys(size);
    for (std::size_t b = 0; b < size; ++b) {
        float best = ninf;
        for (std::size_t i = b * width; i < std::min(scores.size(), (b + 1U) * width); ++i)
            if (scores[i] > best || (scores[i] == 0.0F && best == 0.0F && !std::signbit(scores[i]))) best = scores[i];
        keys[b] = float_key(b + 1U == size ? -ninf : best, sel::KeyOrder::raw_float);
    }
    std::vector<std::uint8_t> candidates(size);
    for (unsigned b : rank(keys, budget))
        candidates[b] = static_cast<std::uint8_t>(keys[b] > float_key(ninf, sel::KeyOrder::raw_float));
    return candidates;
}
} // namespace select_oracle
