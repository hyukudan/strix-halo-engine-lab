#pragma once

#include "strix/distributed_control_wire.hpp"

#include <cstdint>

namespace strix {

class DistributedControlServerHandler {
public:
    virtual ~DistributedControlServerHandler() = default;

    [[nodiscard]] virtual bool execute_remote_step(
        const DistributedRemoteStepCommand& command) = 0;

    virtual void receive_boundary_transfer(
        const DistributedBoundaryTransfer& transfer) = 0;
};

enum class DistributedControlServerEvent : std::uint8_t {
    remote_step,
    boundary_transfer,
};

class DistributedControlServer {
public:
    DistributedControlServer(
        DistributedControlByteStream& stream,
        DistributedControlServerHandler& handler,
        DistributedControlWireLimits limits = {});

    [[nodiscard]] DistributedControlServerEvent
    serve_one();

private:
    DistributedControlByteStream* stream_ = nullptr;
    DistributedControlServerHandler* handler_ = nullptr;
    DistributedControlWireLimits limits_{};
};

}  // namespace strix
