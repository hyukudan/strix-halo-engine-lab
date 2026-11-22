#pragma once

#include "strix/exec/descriptor.hpp"

namespace strix::exec {

static_assert(static_cast<std::uint32_t>(step_valid) == session::step_replay_valid);

// X2c's current transport embeds the incumbent position/token pair. Copy
// fields explicitly because StepRow orders its first two words differently.
[[nodiscard]] constexpr session::StepReplayRow to_step_replay_row(const StepRow& row) noexcept {
    return {step_input(row), row.slot, row.parent, row.ancestors, row.flags, row.rng_lo, row.rng_hi};
}

}  // namespace strix::exec
