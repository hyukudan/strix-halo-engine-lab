#pragma once

#include "strix/engine_profile.hpp"

namespace strix::inkling2 {

// All three tiers currently resolve to the native path. Add qualified levers here.
inline constexpr profile::Engine kMiniProfile{
    "inkling2-mini", "", {}, false, "inkling2-mini-turbo-v1",
};

}  // namespace strix::inkling2
