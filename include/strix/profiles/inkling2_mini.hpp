#pragma once

#include "strix/engine_profile.hpp"
#include "strix/weight/levers.hpp"

namespace strix::inkling2 {

// The resident binding supplies its consumer formats and complete-model receipts. The lever is the shared
// generic policy weights.attention_head.q8_0@1; records under its former model-named id still resolve to it.
inline constexpr weight::WeightLoadPolicy kMiniWeightLoadPolicy = weight::with_legacy(
    weight::kAttentionHeadQ8Policy, {"inkling2-mini-attention-head-bf16-q8_0", 1});

// The current entry points still resolve to the native path.
inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
