#pragma once
// F9 `recur`, kind gated_delta: the gate prologue that produces the beta and gate planes the step consumes
// (gated_delta_policy.hpp, ScalarGate: decay = expf(gate[vh])). Per value head h:
//   beta[h] = Round(sigmoid(beta[h]))                         sigmoid(x) = 1 / (1 + expf(-x))
//   gate[h] = a[h] * softplus(gate[h] + dt_bias[h])           softplus(x) = x (x > 20), expf(x) (x < -20), log1pf(expf(x))
// `a` is the stored decay rate (already negative), dt_bias the stored bias; both are read through the binder's Load hook.
// The expressions are the former GDN decode prologue text verbatim; operation and load order (beta first, then dt, then a)
// are part of the form. SoftplusGate is the only gate policy today; a binder with another gate function adds a policy.
// Hooks: Round::round(float) (the binder's activation rounding), Load::load(const void*, Load::Kind, std::uint64_t).
#include "ops/recur/gated_delta_policy.hpp"

#include <cmath>  // expf, log1pf
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_RECUR_FHD __host__ __device__ __forceinline__
#else
#define STRIX_RECUR_FHD inline
#endif

namespace strix::ops::recur {

namespace gate_prologue {

STRIX_RECUR_HD inline float softplus(
    float value) {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return expf(value);
    }
    return log1pf(
        expf(value));
}

STRIX_RECUR_HD inline float sigmoid(float value) { return 1.0F / (1.0F + expf(-value)); }

}  // namespace gate_prologue

// Default (and only) gate prologue policy: sigmoid beta, softplus log-decay.
struct SoftplusGate {
    template <class Round> STRIX_RECUR_HD static float beta(float raw) { return Round::round(gate_prologue::sigmoid(raw)); }
    STRIX_RECUR_HD static float log_decay(float a, float biased) { return a * gate_prologue::softplus(biased); }
};

// One value head of the prologue, in place on the beta / gate planes (the former transform_beta_gate_body).
template <class Gate, class Round, class Load>
STRIX_RECUR_FHD void gated_delta_gate_body(float* beta, float* gate, const void* dt, typename Load::Kind dt_kind,
                                           const void* a, typename Load::Kind a_kind, std::uint32_t head) {
    beta[head] = Gate::template beta<Round>(beta[head]);
    const float biased = gate[head] + Load::load(dt, dt_kind, head);
    gate[head] = Gate::log_decay(Load::load(a, a_kind, head), biased);
}

#if defined(__HIPCC__) || defined(__CUDACC__)
// Launch: one thread per value head, grid ceil(heads / block).
template <class Gate, class Round, class Load>
__global__ void gated_delta_gate_kernel(float* beta, float* gate, const void* dt, typename Load::Kind dt_kind, const void* a,
                                        typename Load::Kind a_kind, std::uint32_t heads) {
    const auto head =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (head >= heads) {
        return;
    }

    gated_delta_gate_body<Gate, Round, Load>(beta, gate, dt, dt_kind, a, a_kind, head);
}

// Rows form: grid.y = row, planes [row][heads].
template <class Gate, class Round, class Load>
__global__ void gated_delta_gate_rows_kernel(float* beta, float* gate, const void* dt, typename Load::Kind dt_kind,
                                             const void* a, typename Load::Kind a_kind, std::uint32_t heads) {
    const auto head = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (head >= heads) return;
    const auto off = static_cast<std::uint64_t>(blockIdx.y) * heads;
    gated_delta_gate_body<Gate, Round, Load>(beta + off, gate + off, dt, dt_kind, a, a_kind, head);
}
#endif

}  // namespace strix::ops::recur
