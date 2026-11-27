#pragma once

#include "strix/attn/group_indexer.hpp"
#include "strix/core/logit_epilogue.hpp"

#include <algorithm>
#include <limits>
#include <span>

namespace strix::core {

inline void multimax_f32_reference(std::span<float> logits,
                                    std::span<const float, 4> ranges,
                                    std::span<const float, 4> strengths, bool preserve_mask = false) {
    const auto relu = [](float value) { return value <= 0.0F ? 0.0F : value; };
    for (float& logit : logits) {
        if (preserve_mask && logit == -std::numeric_limits<float>::infinity()) continue;
        const float source = logit;
        const float lower = relu(ranges[0] - source);
        const float upper = relu(source - ranges[1]);
        const float lower_square = relu(ranges[2] - source);
        const float upper_square = relu(source - ranges[3]);
        // Each multiply rounds separately from the following addition.
        const volatile float a = strengths[0] * lower;
        const volatile float b = strengths[1] * upper;
        const volatile float c = strengths[2] * (lower_square * lower_square);
        const volatile float d = strengths[3] * (upper_square * upper_square);
        logit = (((source + a) + b) + c) + d;
    }
}

inline void apply_logit_epilogue(std::span<float> logits, const LinearQuadraticHinges& policy,
                                 bool preserve_mask = false) {
    multimax_f32_reference(logits, policy.thresholds, policy.gains, preserve_mask);
}

// Vocabulary epilogue with four learned hinges: a lower and upper linear
// correction followed by a lower and upper squared correction. All hinges
// evaluate the original logit, and every primitive stores to BF16.
inline void multimax_bf16_reference(std::span<float> logits,
                                    std::span<const float, 4> ranges,
                                    std::span<const float, 4> strengths) {
    using attn::group_indexer::bf16_round;
    for (float& logit : logits) {
        const float source = bf16_round(logit);
        float result = source;
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const float delta = i % 2 == 0 ? ranges[i] - source : source - ranges[i];
            float hinge = bf16_round(std::max(delta, 0.0F));
            if (i >= 2) hinge = bf16_round(hinge * hinge);
            const float correction = bf16_round(strengths[i] * hinge);
            result = bf16_round(result + correction);
        }
        logit = result;
    }
}

} // namespace strix::core
