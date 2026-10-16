#include "strix/runtime_launchers.hpp"

#include "strix/dispatch_resources.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strix {

namespace {
SessionStateAddress prepare_position_metadata(const ExecutionDispatchContext& context) {
    if (context.op !=
        SerialOp::position_prepare) {
        throw std::invalid_argument(
            "position launcher attached to wrong operator");
    }

    const auto state =
        resolve_dispatch_state(
            context,
            SessionStateComponent::position_history);

    if (state.storage !=
            SessionStateStorage::host ||
        state.bytes <
            sizeof(std::uint64_t) ||
        state.address == 0) {
        throw std::invalid_argument(
            "position launcher requires writable host position state");
    }

    return state;
}
void preflight_position_prepare(const ExecutionDispatchContext& context, void*) {
    (void)prepare_position_metadata(context);
}
} // namespace

void launch_position_prepare(const ExecutionDispatchContext& context, void*) {
    const auto state = prepare_position_metadata(context);
    const auto position =
        context.position;
    std::memcpy(
        reinterpret_cast<void*>(
            state.address),
        &position,
        sizeof(position));
}


void launch_position_verify_window(
    const ExecutionDispatchContext& context,
    void*) {
    if (context.op !=
        SerialOp::position_prepare) {
        throw std::invalid_argument(
            "verify position launcher attached to wrong operator");
    }
    if (context.logical_rows == 0 ||
        context.input_tokens.size() !=
            context.logical_rows) {
        throw std::invalid_argument(
            "verify position launcher requires one token per logical row");
    }

    const auto state =
        resolve_dispatch_state(
            context,
            SessionStateComponent::position_history);
    if (state.storage !=
            SessionStateStorage::host ||
        state.bytes <
            sizeof(std::uint64_t) ||
        state.address == 0) {
        throw std::invalid_argument(
            "verify position launcher requires writable host position state");
    }

    const auto advance =
        static_cast<std::uint64_t>(
            context.logical_rows - 1U);
    if (advance >
        std::numeric_limits<std::uint64_t>::max() -
            context.position) {
        throw std::overflow_error(
            "verify position window overflow");
    }
    const auto final_position =
        context.position + advance;

    std::memcpy(
        reinterpret_cast<void*>(
            state.address),
        &final_position,
        sizeof(final_position));
}

ExecutionLauncher
make_position_verify_window_launcher() {
    return ExecutionLauncher{
        "runtime.position_verify_window",
        position_verify_window_launcher_identity,
        &launch_position_verify_window,
        nullptr,
    };
}

ExecutionLauncher
make_position_prepare_launcher() {
    return ExecutionLauncher{
        "runtime.position_prepare",
        position_prepare_launcher_identity,
        &launch_position_prepare,
        nullptr,
        &preflight_position_prepare,
    };
}

void register_default_runtime_launchers(
    ExecutionLauncherRegistry& registry) {
    registry.add_runtime_launcher(
        SerialOp::position_prepare,
        make_position_prepare_launcher());
}

void register_verify_runtime_launchers(
    ExecutionLauncherRegistry& registry) {
    registry.add_runtime_launcher(
        SerialOp::position_prepare,
        make_position_verify_window_launcher());
}


}  // namespace strix
