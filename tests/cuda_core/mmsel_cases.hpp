#pragma once

#include "strix/moe/route_policy.hpp"

#include <bit>
#include <cstdio>
#include <random>

namespace mmsel {

#include "mmsel_minimax_frozen.inc"

namespace moe = strix::moe;
struct Policy {
    // The MiniMax-M3 router (= include/strix/minimax_moe.hpp kRoutePolicy, bound from its model TOML):
    // sigmoid + bias, serial float keys, selection order, 128 / 4, renormalised, x 2.0.
    static constexpr moe::RoutePolicy value{moe::RouteScore::sigmoid, true, moe::RouteKey::float_serial,
                                            moe::RouteOrder::selection, 128U, 4U, 1U, 1U, true, true, 1e-20F, 2.0F};
};

inline constexpr unsigned kExperts = frozen::kExperts, kTop = frozen::kTopE, kRows = 259U;
inline constexpr unsigned kPairs = kRows * kExperts, kSlots = kRows * kTop;
inline constexpr std::uint32_t kGuardId = 0xcafef00dU;
inline constexpr float kGuardWeight = -12345.0F;

struct Inputs {
    float interleaved[2U * kPairs];
    float unbiased[kPairs];
    float biased[kPairs];

    moe::RouteScorePlanes<2> paired() { return {interleaved, interleaved + 1U}; }
    moe::RouteScorePlanes<1> split() { return {unbiased, biased}; }
};

struct Outputs {
    std::uint32_t ids[kSlots + 2U];
    float weights[kSlots + 2U];

    void reset() {
        std::fill_n(ids, kSlots + 2U, kGuardId);
        std::fill_n(weights, kSlots + 2U, kGuardWeight);
    }
};

inline void fill_inputs(Inputs& in) {
    std::mt19937 rng(0x4d4d5345U);
    for (unsigned i = 0U; i < kPairs; ++i) {
        in.unbiased[i] = static_cast<float>(rng() % 1024U) / 1024.0F;
        in.biased[i] = static_cast<float>(static_cast<int>(rng() % 65U) - 32) / 8.0F;
    }
    constexpr float inf = std::numeric_limits<float>::infinity();
    constexpr float nan = std::bit_cast<float>(0x7fc12345U);
    // All choices tie at signed zero. The selected scores distinguish -0 from +0.
    std::fill_n(in.unbiased, kExperts, 0.0F);
    std::fill_n(in.biased, kExperts, 0.0F);
    in.unbiased[0] = -0.0F;
    in.biased[0] = -0.0F;
    // A first eligible NaN is picked. A later NaN is skipped by strict comparison.
    // The two positive infinities tie and the remaining negative infinities tie.
    std::fill_n(in.biased + kExperts, kExperts, -inf);
    in.biased[kExperts] = nan;
    in.biased[kExperts + 5U] = nan;
    in.biased[kExperts + 7U] = inf;
    in.biased[2U * kExperts - 1U] = inf;
    // The winning choices drive selection independently of the unbiased scores.
    std::fill_n(in.biased + 2U * kExperts, kExperts, -inf);
    for (unsigned e = 11U; e <= 14U; ++e) in.biased[2U * kExperts + e] = static_cast<float>(e);
    in.unbiased[2U * kExperts + 11U] = nan;
    // Exercise infinity in the denominator and nonfinite quotient results.
    std::fill_n(in.biased + 3U * kExperts, 2U * kExperts, 0.0F);
    in.unbiased[3U * kExperts] = inf;
    in.unbiased[4U * kExperts] = inf;
    in.unbiased[4U * kExperts + 1U] = -inf;
    // Reordering this sum can change the last bit.
    std::fill_n(in.biased + 5U * kExperts, kExperts, 0.0F);
    in.unbiased[5U * kExperts] = 0.5F;
    std::fill_n(in.unbiased + 5U * kExperts + 1U, 3U, 0x1p-25F);
    // Tiny scores test the epsilon and preservation of subnormal inputs.
    std::fill_n(in.biased + 6U * kExperts, kExperts, 0.0F);
    std::fill_n(in.unbiased + 6U * kExperts, 4U, std::numeric_limits<float>::denorm_min());
    for (unsigned i = 0U; i < kPairs; ++i) {
        in.interleaved[2U * i] = in.unbiased[i];
        in.interleaved[2U * i + 1U] = in.biased[i];
    }
}

inline void compare(const Outputs& actual, const Outputs& expected, unsigned rows, const char* layout) {
    for (unsigned i = 0U; i < kSlots + 2U; ++i) {
        const bool active = i > 0U && i <= rows * kTop;
        const auto id = active ? expected.ids[i] : kGuardId;
        const auto weight = std::bit_cast<std::uint32_t>(active ? expected.weights[i] : kGuardWeight);
        if (actual.ids[i] == id && std::bit_cast<std::uint32_t>(actual.weights[i]) == weight) continue;
        std::fprintf(stderr, "%s rows=%u buffer index=%u: got %u/%08x, expected %u/%08x\n", layout, rows, i,
                     actual.ids[i], std::bit_cast<std::uint32_t>(actual.weights[i]), id, weight);
        throw std::runtime_error("route output bytes differ");
    }
}

inline void check_known_rows(const Outputs& result, unsigned rows) {
    if (rows == 0U) return;
    constexpr std::uint32_t known[][kTop] = {{0U, 1U, 2U, 3U}, {0U, 7U, 127U, 1U}, {14U, 13U, 12U, 11U}};
    for (unsigned row = 0U; row < std::min(rows, 3U); ++row) {
        if (!std::equal(known[row], known[row] + kTop, result.ids + 1U + row * kTop))
            throw std::runtime_error("serial tie or NaN selection differs");
    }
    if (std::bit_cast<std::uint32_t>(result.weights[1]) != 0x80000000U ||
        std::bit_cast<std::uint32_t>(result.weights[2]) != 0U)
        throw std::runtime_error("signed-zero weights differ");
}

}  // namespace mmsel
