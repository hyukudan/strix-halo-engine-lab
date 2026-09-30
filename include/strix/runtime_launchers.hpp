#pragma once

#include "strix/launch_dispatch.hpp"

namespace strix {

inline constexpr std::uint64_t
    position_prepare_launcher_identity =
        0x5354524958504f53ULL;

inline constexpr std::uint64_t
    position_verify_window_launcher_identity =
        0x5354524958565057ULL;

void launch_position_prepare(
    const ExecutionDispatchContext& context,
    void* user_data);

[[nodiscard]] ExecutionLauncher
make_position_prepare_launcher();

void launch_position_verify_window(
    const ExecutionDispatchContext& context,
    void* user_data);

[[nodiscard]] ExecutionLauncher
make_position_verify_window_launcher();

void register_default_runtime_launchers(
    ExecutionLauncherRegistry& registry);

void register_verify_runtime_launchers(
    ExecutionLauncherRegistry& registry);

}  // namespace strix
