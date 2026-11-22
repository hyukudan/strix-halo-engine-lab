#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace strix::cpu {

// One head of the serial gated delta rule, in the key-ascending order of
// tests/gdn_chunk/gdn_ref.hpp. Q and K are already normalized, Q is scaled,
// and state is key-major. Compile with contraction disabled for these bits.
inline void gdn_reference(std::span<const float> q, std::span<const float> k, std::span<const float> v,
                          float log_decay, float beta, std::span<float> state, std::span<float> output) {
    if (q.empty() || v.empty() || k.size() != q.size() || output.size() != v.size() ||
        state.size() % v.size() != 0 || state.size() / v.size() != k.size())
        throw std::invalid_argument("invalid CPU gated delta rule shape");

    const float decay = std::exp(log_decay);
    for (std::size_t channel = 0; channel < v.size(); ++channel) {
        float prediction = 0.0F;
        for (std::size_t key = 0; key < k.size(); ++key) {
            auto& value = state[key * v.size() + channel];
            value *= decay;
            prediction += value * k[key];
        }
        const float delta = (v[channel] - prediction) * beta;
        float result = 0.0F;
        for (std::size_t key = 0; key < k.size(); ++key) {
            auto& value = state[key * v.size() + channel];
            value += k[key] * delta;
            result += value * q[key];
        }
        output[channel] = result;
    }
}

}  // namespace strix::cpu
