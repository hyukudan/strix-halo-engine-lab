#include "strix/projection_reduction.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {
void environment() {
    if (!std::numeric_limits<float>::is_iec559 || std::fegetround() != FE_TONEAREST) {
        throw std::invalid_argument("projection reference requires IEEE float nearest-even");
    }
    volatile float tiny = std::numeric_limits<float>::denorm_min();
    volatile float two = 2.0F;
    if (tiny * two == 0.0F) {
        throw std::invalid_argument("projection reference requires gradual subnormals");
    }
}
void finite(float value) {
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite projection arithmetic");
}
float add(float left, float right) {
    const float result = left + right;
    finite(result);
    return result;
}
} // namespace

float projection_reduce_lds(std::span<const float, projection_threads> values) {
    environment();
    std::array<float, projection_threads> partials{};
    std::copy(values.begin(), values.end(), partials.begin());
    for (float value : partials) finite(value);
    for (unsigned offset = projection_threads / 2; offset != 0; offset /= 2) {
        for (unsigned lane = 0; lane < offset; ++lane) {
            partials[lane] = add(partials[lane], partials[lane + offset]);
        }
    }
    return partials[0];
}

float projection_reduce_shuffle_tail(std::span<const float, projection_threads> values) {
    environment();
    std::array<float, projection_threads> shared{};
    std::copy(values.begin(), values.end(), shared.begin());
    for (float value : shared) finite(value);
    for (unsigned offset = 128; offset >= 32; offset /= 2) {
        for (unsigned lane = 0; lane < offset; ++lane) {
            shared[lane] = add(shared[lane], shared[lane + offset]);
        }
    }
    std::array<float, 32> registers{};
    std::copy_n(shared.begin(), registers.size(), registers.begin());
    for (unsigned offset = 16; offset != 0; offset /= 2) {
        // Snapshot is the simultaneous shuffle's input; no forward aliasing.
        const auto before = registers;
        for (unsigned lane = 0; lane < offset; ++lane) {
            registers[lane] = add(before[lane], before[lane + offset]);
        }
    }
    return registers[0];
}

float projection_dot_reference(std::span<const float> weights,
                               std::span<const float> activations) {
    environment();
    if (weights.empty() || weights.size() != activations.size()) {
        throw std::invalid_argument("projection reference needs equal nonempty inputs");
    }
    std::array<float, projection_threads> partials{};
    for (unsigned lane = 0; lane < projection_threads; ++lane) {
        for (std::size_t index = lane; index < weights.size(); index += projection_threads) {
            finite(weights[index]);
            finite(activations[index]);
            partials[lane] = std::fma(weights[index], activations[index], partials[lane]);
            finite(partials[lane]);
        }
    }
    return projection_reduce_lds(partials);
}
} // namespace strix
