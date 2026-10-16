#pragma once
#include <cstdint>
namespace strix {
// Scheduling only: identical expanded arena, launch identities and arithmetic.
enum class MoeTokenSchedule : std::uint8_t { serial_expanded = 0, fork_join = 1 };
} // namespace strix
