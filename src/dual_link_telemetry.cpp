#include "strix/dual_link_telemetry.hpp"

#include <cmath>
#include <stdexcept>

namespace strix::transport {

void DualLinkCompletionTelemetry::register_transfer(
    const DirectTelemetryStart& start) {
    if (start.handle == 0 ||
        start.payload_bytes == 0 ||
        !std::isfinite(
            start.submit_seconds) ||
        start.submit_seconds < 0.0 ||
        !std::isfinite(
            start.one_way_latency_seconds) ||
        start.one_way_latency_seconds < 0.0) {
        throw std::invalid_argument(
            "dual-link telemetry transfer start is invalid");
    }
    const auto inserted =
        starts_.emplace(
            start.handle,
            start);
    if (!inserted.second) {
        throw std::logic_error(
            "dual-link telemetry handle is already tracked");
    }
}

bool DualLinkCompletionTelemetry::tracked(
    TransferHandle handle) const noexcept {
    return starts_.find(handle) !=
        starts_.end();
}

void DualLinkCompletionTelemetry::observe_completion(
    const Completion& completion,
    double event_seconds,
    std::uint64_t in_flight_bytes,
    DualLinkRuntimeScheduler& scheduler) {
    const auto found =
        starts_.find(
            completion.handle);
    if (found == starts_.end()) {
        throw std::invalid_argument(
            "dual-link telemetry completion references unknown handle");
    }
    const auto start =
        found->second;

    if (!std::isfinite(
            event_seconds) ||
        event_seconds <=
            start.submit_seconds) {
        throw std::invalid_argument(
            "dual-link telemetry completion time is invalid");
    }
    if (completion.kind !=
            CompletionKind::failed &&
        completion.bytes !=
            start.payload_bytes) {
        throw std::invalid_argument(
            "dual-link telemetry completion bytes mismatch");
    }

    bool terminal = false;
    bool sample = false;
    switch (completion.kind) {
    case CompletionKind::receive_complete:
        sample =
            start.direction ==
            LinkDirection::inbound;
        terminal = false;
        break;
    case CompletionKind::remote_consumer_complete:
        sample =
            start.direction ==
            LinkDirection::outbound;
        terminal = true;
        break;
    case CompletionKind::receive_reposted:
        sample =
            start.direction ==
            LinkDirection::inbound;
        terminal = true;
        break;
    case CompletionKind::failed:
        starts_.erase(found);
        return;
    case CompletionKind::local_buffer_reusable:
    case CompletionKind::remote_transport_ack:
        return;
    }

    if (sample) {
        scheduler.observe(
            LiveLinkSample{
                start.link_index,
                start.direction,
                start.payload_bytes,
                event_seconds -
                    start.submit_seconds,
                start.one_way_latency_seconds,
                in_flight_bytes,
            });
    }

    if (terminal) {
        starts_.erase(found);
    }
}

}  // namespace strix::transport
