#include "strix/dual_link_telemetry.hpp"

#include <iostream>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::transport::DualLinkRuntimeScheduler
scheduler() {
    std::vector<strix::transport::DirectionalLinkProfile>
        links(1);
    links[0].outbound.id = "usb4-a";
    links[0].inbound.id = "usb4-a";
    links[0].outbound.peer_node = 1;
    links[0].inbound.peer_node = 1;
    links[0].outbound.health =
        strix::transport::LinkHealth::qualified;
    links[0].inbound.health =
        strix::transport::LinkHealth::qualified;
    links[0].outbound.payload_bytes_per_second = 1.0e9;
    links[0].inbound.payload_bytes_per_second = 1.0e9;
    links[0].outbound.one_way_latency_seconds = 10e-6;
    links[0].inbound.one_way_latency_seconds = 10e-6;
    return {std::move(links)};
}

}  // namespace

int main() {
    using namespace strix::transport;

    auto runtime = scheduler();
    DualLinkCompletionTelemetry telemetry;

    telemetry.register_transfer(
        DirectTelemetryStart{
            10,
            0,
            LinkDirection::outbound,
            1024,
            1.0,
            20e-6,
            4096,
        });
    telemetry.observe_completion(
        Completion{
            10,
            CompletionKind::remote_transport_ack,
            1024,
        },
        1.001,
        2048,
        runtime);
    expect(
        telemetry.tracked(10),
        "transport ACK alone keeps outbound telemetry active");

    telemetry.observe_completion(
        Completion{
            10,
            CompletionKind::remote_consumer_complete,
            1024,
        },
        1.002,
        0,
        runtime);
    expect(
        !telemetry.tracked(10) &&
            runtime.links()[0].
                outbound.in_flight_bytes == 0,
        "remote consumer completion samples and retires outbound transfer");

    telemetry.register_transfer(
        DirectTelemetryStart{
            20,
            0,
            LinkDirection::inbound,
            2048,
            2.0,
            15e-6,
            8192,
        });
    telemetry.observe_completion(
        Completion{
            20,
            CompletionKind::receive_complete,
            2048,
        },
        2.001,
        4096,
        runtime);
    expect(
        telemetry.tracked(20) &&
            runtime.links()[0].
                inbound.in_flight_bytes == 4096,
        "receive-ready samples inbound transfer without retiring repost lifecycle");

    telemetry.observe_completion(
        Completion{
            20,
            CompletionKind::receive_reposted,
            2048,
        },
        2.003,
        0,
        runtime);
    expect(
        !telemetry.tracked(20) &&
            runtime.links()[0].
                inbound.in_flight_bytes == 0,
        "repost completion produces final inbound scheduler sample");

    telemetry.register_transfer(
        DirectTelemetryStart{
            30,
            0,
            LinkDirection::outbound,
            512,
            3.0,
            10e-6,
            0,
        });
    telemetry.observe_completion(
        Completion{
            30,
            CompletionKind::failed,
            0,
        },
        3.001,
        0,
        runtime);
    expect(
        !telemetry.tracked(30),
        "failed transfer is retired without publishing throughput sample");

    if (failures == 0) {
        std::cout
            << "dual-link telemetry tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
