#pragma once
// F3 route projection traits: router logit z[r][e] = x[r] . weight[e] as a lane-strided chain, then a score transform
// and the shared selection (route_policy.hpp). Host + device; the device forms live in src/moe_core.
//
// Chain contract (RouteLaneDot<Acc, Lanes>): lane l in [0, Lanes) folds elements l, l + Lanes, l + 2 Lanes, ... in
// ascending order from +0 with Acc::step(acc, w, x); the lanes then combine by the butterfly v = v + v[l ^ d] for
// d = Lanes / 2 .. 1 (lane 0 holds the logit). How a kernel stages the operands (LDS chunks, register tiles, products in
// flight) is free; the per-lane order and the step are the contract.
//
// Accumulation traits:
//   RouteAccMulAdd: acc + w * x with two roundings (product, then sum). The two operations are pinned apart in the body
//                   (intrinsics on CUDA, a contraction pragma on Clang, a materialised product elsewhere), so the bits
//                   do not depend on the translation unit's -ffp-contract flag.
//   RouteAccFma:    fma(w, x, acc), one rounding (the chain of RouteStridedFmaXor32 in route_policy.hpp).
//
// Per-row bias class (RouteRowBias): rows marked in image_rows select with bias_image, the others with bias; the bias
// changes the selection only (route_policy.hpp). A null image_rows means every row uses bias.
#include "strix/moe/route_policy.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

// Chain steps inline before any other optimisation, as the literal expression would (device: always-inline).
#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MOE_ROUTE_CHAIN __host__ __device__ __forceinline__
#else
#define STRIX_MOE_ROUTE_CHAIN inline
#endif

namespace strix::moe {

struct RouteAccMulAdd {
    STRIX_MOE_ROUTE_CHAIN static float step(float acc, float w, float x) {
#if defined(__CUDA_ARCH__)
        return __fadd_rn(acc, __fmul_rn(w, x));
#elif defined(__clang__)
        {
#pragma clang fp contract(off)
            return acc + w * x;
        }
#else
        // Other host compilers (GCC contracts across statements under -ffp-contract=fast, the default for GNU modes):
        // the rounded product is materialised so no fused multiply-add can form.
        volatile float product = w * x;
        return acc + product;
#endif
    }
};

struct RouteAccFma {
    STRIX_MOE_ROUTE_CHAIN static float step(float acc, float w, float x) { return ::fmaf(w, x, acc); }
};

template <class Acc, unsigned Lanes = 32U>
struct RouteLaneDot {
    static_assert(Lanes > 0U && (Lanes & (Lanes - 1U)) == 0U, "route projection: lanes must be a power of two");
    static constexpr unsigned lanes = Lanes;

    STRIX_MOE_ROUTE_CHAIN static float step(float acc, float w, float x) { return Acc::step(acc, w, x); }

    // Butterfly over the wave, in place: xor_shuffle(v, d) returns v of lane (own ^ d). The in-place form updates the
    // caller's element at every step, as a literal loop over that element does.
    template <class XorShuffle>
    STRIX_MOE_ROUTE_CHAIN static void reduce_into(float& v, const XorShuffle& xor_shuffle) {
#if defined(__HIPCC__) || defined(__CUDACC__)
#pragma unroll
#endif
        for (unsigned d = Lanes / 2U; d != 0U; d /= 2U) v = v + xor_shuffle(v, static_cast<int>(d));
    }
    template <class XorShuffle>
    STRIX_MOE_ROUTE_CHAIN static float reduce(float v, const XorShuffle& xor_shuffle) {
        reduce_into(v, xor_shuffle);
        return v;
    }

    // Host emulation of the whole wave; returns lane 0.
    static float dot_ref(std::span<const float> x, std::span<const float> weight) {
        if (x.empty() || x.size() != weight.size())
            throw std::invalid_argument("route projection: expected equal nonempty input and weight spans");
        std::array<float, Lanes> lane{};
        for (std::size_t i = 0; i < x.size(); ++i) lane[i % Lanes] = Acc::step(lane[i % Lanes], weight[i], x[i]);
        for (unsigned d = Lanes / 2U; d != 0U; d /= 2U) {
            std::array<float, Lanes> next{};
            for (unsigned l = 0; l < Lanes; ++l) next[l] = lane[l] + lane[l ^ d];
            lane = next;
        }
        return lane[0];
    }
};

// The bias of row r: bias_image when image_rows marks the row, else bias.
STRIX_MOE_ROUTE_CHAIN const float* route_row_bias(const std::uint8_t* image_rows, std::size_t r, const float* bias_image,
                                                      const float* bias) {
    return image_rows && image_rows[r] ? bias_image : bias;
}

struct RouteRowBias {
    const float* bias = nullptr;
    const float* bias_image = nullptr;
    const std::uint8_t* image_rows = nullptr;

    STRIX_MOE_ROUTE_HD const float* row(std::size_t r) const { return route_row_bias(image_rows, r, bias_image, bias); }
};

// Score transform of a logit by the policy's RouteScore (softmax needs the whole row: not a per-logit transform).
template <class Math>
STRIX_MOE_ROUTE_HD inline float route_logit_score(RouteScore score, float z) {
    return score == RouteScore::sqrt_softplus ? sqrt_softplus_score<Math>(z) : sigmoid_score<Math>(z);
}

// CPU oracle of the projected router: scores[r][e] = score(Dot(x[r], weight[e])), then route_select_ref per row with
// the row's bias class. x: [rows][hidden]; weight: [experts][hidden]; bias spans: [experts] (bias_image may be empty
// when image_rows is empty). Outputs: scores [rows][experts], ids / weights [rows][top_k].
template <class Dot, class Math>
void route_project_rows_ref(const RoutePolicy& policy, std::span<const float> x, std::size_t rows, std::size_t hidden,
                            std::span<const float> weight, std::span<const float> bias, std::span<const float> bias_image,
                            std::span<const std::uint8_t> image_rows, std::span<float> scores,
                            std::span<std::uint32_t> ids, std::span<float> weights) {
    if (const char* why = validate(policy)) throw std::invalid_argument(why);
    if (policy.score == RouteScore::softmax) throw std::invalid_argument("route projection: softmax is a row transform");
    const std::size_t experts = policy.experts, top_k = policy.top_k;
    // Check extents by division before multiplying them into offsets or span sizes.
    // validate(policy) guarantees experts and top_k are nonzero.
    if (rows == 0U || hidden == 0U || rows > x.size() / hidden || experts > weight.size() / hidden ||
        rows > scores.size() / experts || rows > ids.size() / top_k || rows > weights.size() / top_k ||
        (policy.selection_bias && bias.size() < experts) ||
        (!image_rows.empty() && (image_rows.size() < rows || bias_image.size() < experts)))
        throw std::invalid_argument("route projection: inconsistent row geometry");
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t e = 0; e < experts; ++e)
            scores[r * experts + e] =
                route_logit_score<Math>(policy.score, Dot::dot_ref(x.subspan(r * hidden, hidden), weight.subspan(e * hidden, hidden)));
    // The bias of a row is a validated span: bias_image on an image-marked row, else bias; empty when the policy has
    // no selection bias (no span is formed over an operand that was not validated).
    const std::span<const float> all(scores.data(), rows * experts);
    for (std::size_t r = 0; r < rows; ++r) {
        std::span<const float> b;
        if (policy.selection_bias) b = (!image_rows.empty() && image_rows[r] != 0U ? bias_image : bias).first(experts);
        route_select_ref(policy, all.subspan(r * experts, experts), b, ids.subspan(r * top_k, top_k),
                         weights.subspan(r * top_k, top_k));
    }
}

}  // namespace strix::moe
