#pragma once

#include "strix/engine_profile.hpp"
#include "strix/weight/levers.hpp"

namespace strix::inkling2 {

// The resident binding supplies its consumer formats and complete-model receipts.
inline constexpr weight::WeightLoadPolicy kMiniWeightLoadPolicy{
    {"inkling2-mini-attention-head-bf16-q8_0", 1}, weight::kQ8WeightStorage,
};

// The current entry points still resolve to the native path.
inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
