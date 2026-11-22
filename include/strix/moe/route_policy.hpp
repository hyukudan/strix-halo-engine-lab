#pragma once
// MoE router policy: the vendor-free contract every family router binds (host oracle, HIP and CUDA kernels).
//
// A router row is: logits z[e] (the family's product order) -> scores s[e] by the score transform (the family's Math
// trait supplies exp / log1p / sqrt, so the bits stay the family's) -> choice c[e] = s[e] (+ bias[e] when the policy
// has a selection bias) -> top_k experts by the selection key -> output order -> weights.
//
// Declared order (the bits of every binding):
//   selection: k passes; pass k scans the experts in ascending id, skips the chosen ones, and takes e when there is no
//     pick yet or route_better(key, c[e], e, c[pick], pick). float_serial: c[e] > c[pick] (lower id on ties; a NaN
//     taken as the first candidate stays). ordered_bits: route_ordered_key(c) (-0 == +0, a total order on the bits).
//     For finite choices both keys give the same experts: (choice desc, id asc).
//   order: selection order, or ids sorted ascending.
//   weights: sum = ((0 + s[id_0]) + s[id_1]) + ... in OUTPUT order; w_k = (s[id_k] / (sum + renorm_eps)) * scale.
//     renormalize == false: w_k = s[id_k] * scale. top_k == 1 && !renorm_single: w_0 = s[id_0] * scale.
//   groups: n_group = topk_group = 1 only (no bound family uses group-limited routing; anything else is rejected).
//
// Bound families (presets below, each the bits of the source named): DS41, GLM-5, MiniMax-M3. A family whose binding
// lives in its model package keeps its preset there as data (a RoutePolicy value checked by validate).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MOE_ROUTE_HD __host__ __device__
#else
#define STRIX_MOE_ROUTE_HD
#endif

namespace strix::moe {

enum class RouteScore : std::uint8_t { softmax, sigmoid, sqrt_softplus };
enum class RouteKey : std::uint8_t { float_serial, ordered_bits };
enum class RouteOrder : std::uint8_t { selection, ascending_id };

// Device execution traits. Checked rows write status and use sentinel IDs on failure.
// Explicit round-to-nearest keeps the arithmetic of bindings that require it.
template <bool CheckedStatus = false, bool RoundNearest = false>
struct RouteWaveTraits {
    static constexpr bool checked_status = CheckedStatus;
    static constexpr bool round_nearest = RoundNearest;
};

struct RoutePolicy {
    RouteScore score = RouteScore::sigmoid;
    bool selection_bias = false;
    RouteKey key = RouteKey::float_serial;
    RouteOrder order = RouteOrder::selection;
    std::uint32_t experts = 0;
    std::uint32_t top_k = 0;
    std::uint32_t n_group = 1;
    std::uint32_t topk_group = 1;
    bool renormalize = true;
    bool renorm_single = true;  // false: a single pick keeps its raw score (DS41)
    float renorm_eps = 1e-20F;
    float scale = 1.0F;
};

// nullptr when the policy is bindable, else the reason.
constexpr const char* validate(const RoutePolicy& p) {
    if (p.experts == 0U || p.top_k == 0U || p.top_k > p.experts) return "route policy: need 0 < top_k <= experts";
    if (p.n_group != 1U || p.topk_group != 1U) return "route policy: group-limited routing is not bound (n_group = topk_group = 1)";
    return nullptr;
}

namespace preset {
// src/ds41_moe.cpp reference_router; HIP route_block_topk (hip_ds41_formats.hip); CUDA ds41_core_moe.cu.
constexpr RoutePolicy ds41(std::uint32_t experts, std::uint32_t top_k, float scale) {
    return {RouteScore::sqrt_softplus, true, RouteKey::float_serial, RouteOrder::selection, experts, top_k, 1U, 1U, true, false, 1e-20F, scale};
}
// src/glm/host/glm_host.cpp route; HIP glm_tier_a.hip route_kernel (route_wave_topk). 288 experts, top-8, x 2.5.
inline constexpr RoutePolicy glm5{RouteScore::sigmoid, true, RouteKey::ordered_bits, RouteOrder::ascending_id, 288U, 8U, 1U, 1U, true, true, 1e-20F, 2.5F};
// src/minimax_moe.cpp route_row; HIP hip_minimax_moe.hip route_topk. 128 experts, top-4, x 2.0.
inline constexpr RoutePolicy minimax_m3{RouteScore::sigmoid, true, RouteKey::float_serial, RouteOrder::selection, 128U, 4U, 1U, 1U, true, true, 1e-20F, 2.0F};
}  // namespace preset

// Score transforms of one logit. Math supplies exp / log1p / sqrt with the family's bits.
template <class Math>
STRIX_MOE_ROUTE_HD inline float sigmoid_score(float z) {
    return 1.0F / (1.0F + Math::exp(-z));
}
template <class Math>
STRIX_MOE_ROUTE_HD inline float sqrt_softplus_score(float z) {
    return Math::sqrt(z > 20.0F ? z : Math::log1p(Math::exp(z)));
}

// Selection key: -0 is +0; negatives below positives; a total order on the remaining bits.
STRIX_MOE_ROUTE_HD inline std::uint32_t route_ordered_key(float v) {
    std::uint32_t u = __builtin_bit_cast(std::uint32_t, v);
    if (u == 0x80000000U) u = 0;
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}
// True when candidate a (id ai) replaces the current pick b (id bi).
STRIX_MOE_ROUTE_HD inline bool route_better(RouteKey key, float a, std::uint32_t ai, float b, std::uint32_t bi) {
    if (key == RouteKey::float_serial) return a > b;
    const std::uint32_t ka = route_ordered_key(a), kb = route_ordered_key(b);
    return ka > kb || (ka == kb && ai < bi);
}

inline void validate_route_row(const RoutePolicy& p, std::size_t scores, std::size_t bias,
                               std::size_t ids, std::size_t weights) {
    if (const char* why = validate(p)) throw std::invalid_argument(why);
    if (scores < p.experts || (p.selection_bias && bias < p.experts) || ids < p.top_k || weights < p.top_k)
        throw std::invalid_argument("route policy: row spans are smaller than the policy");
}

// Shared single-row selection. Callers validate spans and supply experts bytes of scratch.
// Inputs are already transformed scores. Bias changes selection only.
inline void route_select_row(const RoutePolicy& p, const float* scores, const float* bias,
                             std::uint32_t* ids, float* weights, std::uint8_t* taken) {
    for (std::uint32_t e = 0; e < p.experts; ++e) taken[e] = 0U;
    for (std::uint32_t k = 0; k < p.top_k; ++k) {
        std::uint32_t pick = p.experts;
        float best = 0.0F;
        for (std::uint32_t e = 0; e < p.experts; ++e) {
            if (taken[e]) continue;
            const float c = p.selection_bias ? scores[e] + bias[e] : scores[e];
            if (pick == p.experts || route_better(p.key, c, e, best, pick)) {
                pick = e;
                best = c;
            }
        }
        taken[pick] = 1U;
        ids[k] = pick;
    }
    if (p.order == RouteOrder::ascending_id) {
        for (std::uint32_t k = 1; k < p.top_k; ++k) {
            const auto expert = ids[k];
            auto slot = k;
            while (slot > 0U && ids[slot - 1U] > expert) {
                ids[slot] = ids[slot - 1U];
                --slot;
            }
            ids[slot] = expert;
        }
    }
    const bool renorm = p.renormalize && (p.top_k > 1U || p.renorm_single);
    float sum = 0.0F;
    for (std::uint32_t k = 0; k < p.top_k; ++k) sum = sum + scores[ids[k]];
    const float denominator = sum + p.renorm_eps;
    for (std::uint32_t k = 0; k < p.top_k; ++k) {
        const float s = scores[ids[k]];
        weights[k] = (renorm ? s / denominator : s) * p.scale;
    }
}

inline void route_select_ref(const RoutePolicy& p, std::span<const float> scores, std::span<const float> bias,
                             std::span<std::uint32_t> ids, std::span<float> weights) {
    validate_route_row(p, scores.size(), bias.size(), ids.size(), weights.size());
    std::vector<std::uint8_t> taken(p.experts);
    route_select_row(p, scores.data(), bias.data(), ids.data(), weights.data(), taken.data());
}

// Host transform of one row: softmax (m = max over e ascending, p_e = exp(z_e - m), sum ascending from 0, s_e = p_e / sum),
// sigmoid or sqrt_softplus per element.
template <class Math>
void route_scores_ref(RouteScore score, std::span<const float> logits, std::span<float> scores) {
    if (score == RouteScore::softmax) {
        float m = -std::numeric_limits<float>::infinity();
        for (const float z : logits) m = z > m ? z : m;
        float sum = 0.0F;
        for (std::size_t e = 0; e < logits.size(); ++e) {
            scores[e] = Math::exp(logits[e] - m);
            sum = sum + scores[e];
        }
        for (std::size_t e = 0; e < logits.size(); ++e) scores[e] = scores[e] / sum;
        return;
    }
    if (score == RouteScore::sigmoid) {
        for (std::size_t e = 0; e < logits.size(); ++e) scores[e] = sigmoid_score<Math>(logits[e]);
        return;
    }
    // sqrt_softplus needs Math::log1p and Math::sqrt; a sigmoid-only Math (GLM, MiniMax) need not supply them.
    if constexpr (requires { Math::log1p(0.0F); Math::sqrt(0.0F); }) {
        for (std::size_t e = 0; e < logits.size(); ++e) scores[e] = sqrt_softplus_score<Math>(logits[e]);
    } else {
        throw std::invalid_argument("route policy: Math has no log1p / sqrt for sqrt_softplus");
    }
}

// Reuse one score buffer over token-major rows. Weights retain the order and
// normalization of the route policy, independent of any later grouping.
template <class Math>
void route_rows_ref(const RoutePolicy& policy, std::span<const float> logits, std::span<const float> bias,
                    std::span<std::uint32_t> ids, std::span<float> weights) {
    if (const char* why = validate(policy)) throw std::invalid_argument(why);
    const auto rows = logits.size() / policy.experts;
    if (logits.size() % policy.experts != 0 || (policy.selection_bias && bias.size() != policy.experts) ||
        ids.size() % policy.top_k != 0 || ids.size() / policy.top_k != rows || weights.size() != ids.size()) {
        throw std::invalid_argument("route policy: invalid token-major geometry");
    }
    if (rows == 0) return;
    std::vector<float> scores(policy.experts);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto first = row * policy.top_k;
        route_scores_ref<Math>(policy.score, logits.subspan(row * policy.experts, policy.experts), scores);
        route_select_ref(policy, scores, bias, ids.subspan(first, policy.top_k), weights.subspan(first, policy.top_k));
    }
}

}  // namespace strix::moe
