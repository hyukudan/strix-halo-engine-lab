#pragma once
#include <string_view>
namespace qwen_contract {
inline constexpr std::string_view historical_decode = "exact-v5";
inline constexpr std::string_view baseline_decode = historical_decode;
inline constexpr bool historical_prepared = true;
inline constexpr bool baseline_prepared = historical_prepared;
inline constexpr bool baseline_tiled = false;
}
