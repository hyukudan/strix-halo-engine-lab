// Shared by the round-store device fixtures (tests/core/round_fold_hip.hip, round_store_incumbent_hip.hip): checked
// hip calls, buffers, NaN-bearing operands and the incumbent / separate rounds on the host. Uses only entries of the
// shared projection headers that every revision of the store policy has, so the incumbent fixture builds on both sides
// of a change to the policy.
#pragma once

#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/fma_bf16_rows.hpp"
#include "strix/attn/group_indexer.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace round_fold_support {
namespace sf = strix::weight::scalar_float;

inline void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FAIL round-fold %s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

template <class T> T* upload(const std::vector<T>& host) {
    T* d = nullptr;
    ok(hipMalloc(&d, host.size() * sizeof(T) + 16U), "malloc");
    ok(hipMemcpy(d, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}

template <class T> std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> host(n);
    ok(hipMemcpy(host.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return host;
}

inline std::vector<float> noise(std::size_t n, std::mt19937& rng, float scale) {
    std::normal_distribution<float> dist(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// NaN-bearing operands: FP32 input rows (a few rows carry NaN / Inf, the others stay finite) and BF16 weights (a few
// output rows carry NaN / Inf / 0 patterns).
inline void nan_operands(std::vector<float>& x, std::vector<std::uint16_t>& w, unsigned width, unsigned rows,
                         unsigned outputs) {
    constexpr std::uint32_t nans[] = {0x7FC00000U, 0xFFC00000U, 0x7FC12345U, 0x7F800001U, 0xFFBFFFFFU, 0x7FFF0001U};
    for (unsigned k = 0; k < 6U; ++k) {
        const unsigned row = (37U + 611U * k) % rows;
        x[std::size_t{row} * width + (k * 131U) % width] = sf::u2f(nans[k]);
    }
    const unsigned inf_row = (rows * 3U) / 4U;
    x[std::size_t{inf_row} * width + 5U] = sf::u2f(0x7F800000U);   // + Inf against a zero weight below: Inf x 0
    x[std::size_t{inf_row} * width + 6U] = sf::u2f(0x7F800000U);   // + Inf twice with weights +1 / -1: Inf - Inf
    constexpr std::uint16_t wnan[] = {0x7FC0U, 0xFFC1U, 0x7F81U, 0x7F80U, 0xFF80U};
    for (unsigned k = 0; k < 5U; ++k) w[std::size_t{(29U + 257U * k) % outputs} * width + (k * 17U) % width] = wnan[k];
    for (unsigned o = 0; o < outputs; o += 97U) {
        w[std::size_t{o} * width + 5U] = 0x0000U;
        w[std::size_t{o} * width + 6U] = o % 2U ? 0x3F80U : 0xBF80U;
    }
}

// The incumbent store function and the separate pass's function, on the host.
inline std::uint32_t incumbent_bits(float v) { return sf::f2u(strix::attn::group_indexer::bf16_round(v)); }
inline std::uint32_t separate_bits(float v) { return sf::f2u(sf::bf16_round(v)); }

inline bool is_nan_bits(std::uint32_t u) { return (u & 0x7fffffffU) > 0x7f800000U; }

// Elements of `raw` (completed FP32 sums) the two functions store differently: NaNs with payload bits below BF16 width.
inline std::size_t discriminating(const std::vector<float>& raw) {
    std::size_t n = 0;
    for (const float v : raw) n += incumbent_bits(v) != separate_bits(v);
    return n;
}

inline std::uint64_t fnv1a(const void* data, std::size_t bytes) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}

// Weights and activations of a projection case: normal noise, NaN / Inf operands when nan is set.
struct Operands {
    std::vector<float> x;
    std::vector<std::uint16_t> wall;  // one leading element: a 2-byte offset gives unaligned weights
    unsigned width, outputs, rows;
};
inline Operands make_operands(unsigned width, unsigned outputs, unsigned rows, bool nan, std::mt19937& rng) {
    Operands o{noise(std::size_t{rows} * width, rng, 1.0F), std::vector<std::uint16_t>(std::size_t{outputs} * width + 1U),
               width, outputs, rows};
    for (auto& b : o.wall) b = sf::bf16_encode(noise(1, rng, 0.05F)[0]);
    if (nan) {
        std::vector<std::uint16_t> body(o.wall.begin() + 1, o.wall.end());
        nan_operands(o.x, body, width, rows, outputs);
        std::copy(body.begin(), body.end(), o.wall.begin() + 1);
    }
    return o;
}
}  // namespace round_fold_support
