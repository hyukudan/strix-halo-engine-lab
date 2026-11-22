#pragma once
// F9 `recur`, kind gated_delta: the arithmetic hooks of one decode step (host and device, no runtime include).
//
// One step on a value head, state S[k][v] (lane = key index k, D keys), for every value column v:
//   s[k]  = S[k][v] * decay(k)                 (gate)
//   u     = tree_k(lane_term(s[k], key[k]))    (halving tree t[l] = t[l] + t[l + h], h = D/2 .. 1)
//   d     = (value[v] - u) * beta
//   s[k]  = update(key[k], d, s[k]);  observe(s[k]);  S[k][v] = s[k]
//   out[v] = tree_k(lane_term(s[k], query[k]))
// Policies are traits of that one form (CORE-SPEC F9 `recur::Spec{kind = gated_delta, decay = ...}`):
//   ScalarGate (default): the scalar per-value-head gate of the GDN step, decay = expf(gate[vh]); the lane term is the
//     plain product and the update is `s + k * d` written as one expression, so it contracts exactly as the binder's TU
//     contracts today (the expressions are the former recurrent_step_kernel text, verbatim).
//   PerKeyChannelGate<Exp>: the per-key-channel (diagonal) gate of KDA, decay = Exp::exp(gate[vh * D + k]) with the
//     binder's declared exp; the lane term is `0.0F + s * x` (product rounded, then +0: an exact or underflowed -0 term
//     becomes +0) and the update an explicit fmaf, both independent of the binder's contraction flags; a non-finite
//     updated state element sets a sticky flag (no effect on any output bit).
// The operand layout is the GDN one: qkv = [q heads | k heads | v heads] (head stride D), gate per value head (scalar) or
// per value head and key (vector), beta per value head; key head = value head % key heads. A binder with independent
// rows flattens them into value heads (vh = row * heads + head).
#include "strix/moe/declared_exp.hpp"

#include <cmath>  // expf (ScalarGate)
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_RECUR_HD __host__ __device__
#else
#define STRIX_RECUR_HD
#endif

namespace strix::ops::recur {

inline constexpr std::uint32_t kStepThreads = 128;  // lanes per block = keys per head (D)
inline constexpr std::uint32_t kStepValueTile = 32;  // value columns per block

// Default: today's GDN step (scalar gate per value head).
struct ScalarGate {
    STRIX_RECUR_HD float decay(const float* gate, std::uint32_t value_head, std::uint32_t /*head_dim*/,
                               std::uint32_t /*lane*/) const {
        return expf(
            gate[value_head]);
    }
    STRIX_RECUR_HD float lane_term(float state, float x) const { return state * x; }
    STRIX_RECUR_HD float update(float key, float delta, float state) const { return state + key * delta; }
    STRIX_RECUR_HD void observe(float /*state*/) const {}
};

// The declared exp (basic IEEE operations, same bits on host and device when the TU does not contract).
struct DeclaredExp {
    STRIX_RECUR_HD static float exp(float x) { return ::strix::moe::declared_math::declared_exp(x); }
};
// Identity exp: the binder already stores the decay factor (not its logarithm) in the gate plane.
struct FactorExp {
    STRIX_RECUR_HD static float exp(float x) { return x; }
};

// KDA: per-key-channel decay, zero-seeded lane terms, fused update, sticky non-finite flag (nullptr = off).
template <class Exp = DeclaredExp> struct PerKeyChannelGate {
    unsigned* finite_flag = nullptr;
    STRIX_RECUR_HD float decay(const float* gate, std::uint32_t value_head, std::uint32_t head_dim,
                               std::uint32_t lane) const {
        return Exp::exp(gate[static_cast<std::uint64_t>(value_head) * head_dim + lane]);
    }
    STRIX_RECUR_HD float lane_term(float state, float x) const {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        const float product = state * x;
        return 0.0F + product;
    }
    STRIX_RECUR_HD float update(float key, float delta, float state) const { return __builtin_fmaf(key, delta, state); }
    STRIX_RECUR_HD void observe(float state) const {
        if (finite_flag != nullptr && !__builtin_isfinite(state)) {
#if defined(__HIP_DEVICE_COMPILE__)
            atomicOr(finite_flag, 1U);
#else
            *finite_flag |= 1U;
#endif
        }
    }
};

// Host reference of one step on one value head in the device order (f32 state, in place). D = head_dim must be a power
// of two; q, k index the key head, v the value head. Writes out[0..D).
template <class Policy>
inline void gated_delta_step_ref(const Policy& policy, const float* q, const float* k, const float* v, float beta,
                                 const float* gate, std::uint32_t value_head, std::uint32_t head_dim, float* state,
                                 float* out) {
    constexpr std::uint32_t kMax = 256;
    float tree[kMax];
    float s[kMax];
    for (std::uint32_t col = 0; col < head_dim; ++col) {
        for (std::uint32_t l = 0; l < head_dim; ++l) {
            s[l] = state[static_cast<std::size_t>(l) * head_dim + col] * policy.decay(gate, value_head, head_dim, l);
            tree[l] = policy.lane_term(s[l], k[l]);
        }
        for (std::uint32_t h = head_dim / 2U; h > 0; h >>= 1U)
            for (std::uint32_t l = 0; l < h; ++l) tree[l] = tree[l] + tree[l + h];
        const float delta = (v[col] - tree[0]) * beta;
        for (std::uint32_t l = 0; l < head_dim; ++l) {
            s[l] = policy.update(k[l], delta, s[l]);
            policy.observe(s[l]);
            state[static_cast<std::size_t>(l) * head_dim + col] = s[l];
            tree[l] = policy.lane_term(s[l], q[l]);
        }
        for (std::uint32_t h = head_dim / 2U; h > 0; h >>= 1U)
            for (std::uint32_t l = 0; l < h; ++l) tree[l] = tree[l] + tree[l + h];
        out[col] = tree[0];
    }
}

}  // namespace strix::ops::recur
