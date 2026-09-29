#pragma once

#include "strix/launch_dispatch.hpp"

namespace strix {

inline constexpr std::uint64_t
    position_prepare_launcher_identity =
        0x5354524958504f53ULL;

void launch_position_prepare(
    const ExecutionDispatchContext& context,
    void* user_data);

[[nodiscard]] ExecutionLauncher
make_position_prepare_launcher();

void register_default_runtime_launchers(
    ExecutionLauncherRegistry& registry);

}  // namespace strix
