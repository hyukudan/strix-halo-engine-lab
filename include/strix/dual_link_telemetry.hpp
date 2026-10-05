#pragma once

#include "strix/dual_link_runtime.hpp"
#include "strix/transport.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace strix::transport {

struct DirectTelemetryStart {
    TransferHandle handle = 0;
    std::size_t link_index = 0;
    LinkDirection direction =
        LinkDirection::outbound;
    std::uint64_t payload_bytes = 0;
    double submit_seconds = 0.0;
    double one_way_latency_seconds = 0.0;
    std::uint64_t in_flight_bytes = 0;
};

class DualLinkCompletionTelemetry {
public:
    void register_transfer(
        const DirectTelemetryStart& start);

    void observe_completion(
        const Completion& completion,
        double event_seconds,
        std::uint64_t in_flight_bytes,
        DualLinkRuntimeScheduler& scheduler);

    [[nodiscard]] bool tracked(
        TransferHandle handle) const noexcept;

private:
    std::unordered_map<
        TransferHandle,
        DirectTelemetryStart> starts_;
};

}  // namespace strix::transport
