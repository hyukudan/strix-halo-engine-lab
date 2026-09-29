#include "strix/runtime_launchers.hpp"

#include "strix/dispatch_resources.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strix {

void launch_position_prepare(
    const ExecutionDispatchContext& context,
    void*) {
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

    const auto position =
        context.position;
    std::memcpy(
        reinterpret_cast<void*>(
            state.address),
        &position,
        sizeof(position));
}

ExecutionLauncher
make_position_prepare_launcher() {
    return ExecutionLauncher{
        "runtime.position_prepare",
        position_prepare_launcher_identity,
        &launch_position_prepare,
        nullptr,
    };
}

}  // namespace strix
