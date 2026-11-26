#pragma once

#include "strix/attn/group_indexer.hpp"

#include <algorithm>
#include <span>

namespace strix::core {

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
