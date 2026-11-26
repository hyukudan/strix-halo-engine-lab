#pragma once

#include "strix/engine_profile.hpp"

namespace strix::ernie {

// Inspection metadata. The reference execution and native qualification are pending.
inline constexpr profile::Name kDefaultProfile = profile::Name::standard;
inline constexpr std::array<profile::Lever, 0> kProfileRows{};
inline constexpr profile::Engine kProfileEngine{"ernie", "", kProfileRows};

} // namespace strix::ernie
