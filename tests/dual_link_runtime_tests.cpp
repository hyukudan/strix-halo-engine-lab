#include "strix/dual_link_runtime.hpp"

#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::transport::LinkProfile profile(
    const char* id,
    double rate,
    double latency,
    bool independent = true) {
    strix::transport::LinkProfile value;
    value.id = id;
    value.peer_node = 1;
    value.health =
        strix::transport::LinkHealth::qualified;
    value.payload_bytes_per_second = rate;
    value.one_way_latency_seconds = latency;
    value.independent_for_striping =
        independent;
    return value;
}

}  // namespace

int main() {
    using namespace strix::transport;

    std::vector<DirectionalLinkProfile> links{
        {
            profile("usb4-a", 1.0e9, 10e-6),
            profile("usb4-a", 0.7e9, 12e-6),
        },
        {
            profile("usb4-b", 0.8e9, 11e-6),
            profile("usb4-b", 1.2e9, 9e-6),
        },
    };

    DualLinkRuntimePolicy policy;
    policy.transfer.minimum_stripe_bytes =
        1ULL << 20U;
    policy.metric_alpha = 0.5;
    policy.flow_switch_hysteresis_fraction =
        0.10;

    DualLinkRuntimeScheduler scheduler(
        links,
        policy);

    const auto outbound =
        scheduler.plan(
            1,
            LinkDirection::outbound,
            4096,
            TransferPriority::normal);
    const auto inbound =
        scheduler.plan(
            2,
            LinkDirection::inbound,
            4096,
            TransferPriority::normal);

    expect(outbound.kind ==
               PlanKind::single_link &&
               outbound.slices[0].link_index == 0,
           "outbound flow selects faster outbound physical link");
    expect(inbound.kind ==
               PlanKind::single_link &&
               inbound.slices[0].link_index == 1,
           "inbound flow can prefer the opposite physical link");

    const auto large =
        scheduler.plan(
            3,
            LinkDirection::outbound,
            8ULL << 20U,
            TransferPriority::bulk);
    expect(large.kind ==
               PlanKind::striped &&
               large.slices.size() == 2,
           "large bulk flow stripes only when two qualified independent links win");

    scheduler.observe(
        LiveLinkSample{
            0,
            LinkDirection::outbound,
            1ULL << 20U,
            0.004,
            30e-6,
            8ULL << 20U,
        });
    const auto adapted =
        scheduler.plan(
            4,
            LinkDirection::outbound,
            4096,
            TransferPriority::normal);
    expect(adapted.kind ==
               PlanKind::single_link &&
               adapted.slices[0].link_index == 1,
           "live directional metrics can move new outbound flows to the other link");

    const auto sticky_before =
        scheduler.plan(
            10,
            LinkDirection::inbound,
            4096,
            TransferPriority::normal);
    expect(sticky_before.slices[0].link_index == 1,
           "flow establishes initial link affinity");

    scheduler.observe(
        LiveLinkSample{
            0,
            LinkDirection::inbound,
            1ULL << 20U,
            0.00086,
            9e-6,
            0,
        });
    const auto sticky_after =
        scheduler.plan(
            10,
            LinkDirection::inbound,
            4096,
            TransferPriority::normal);
    expect(sticky_after.kind ==
               PlanKind::single_link &&
               sticky_after.slices[0].link_index == 1,
           "small improvement stays on existing flow link inside hysteresis");

    scheduler.set_link_health(
        0,
        LinkHealth::failed);
    const auto failed_over =
        scheduler.plan(
            20,
            LinkDirection::outbound,
            8ULL << 20U,
            TransferPriority::bulk);
    expect(
        failed_over.kind ==
                PlanKind::single_link &&
            failed_over.slices.size() == 1 &&
            failed_over.slices[0].link_index == 1,
        "failed physical link immediately downgrades striped traffic to surviving link");

    const auto sticky_failed_over =
        scheduler.plan(
            10,
            LinkDirection::inbound,
            4096,
            TransferPriority::normal);
    expect(
        sticky_failed_over.kind ==
                PlanKind::single_link &&
            sticky_failed_over.slices[0].link_index == 1,
        "failed link cannot retain stale flow affinity");

    scheduler.set_link_health(
        0,
        LinkHealth::qualified);
    scheduler.observe(
        LiveLinkSample{
            0,
            LinkDirection::outbound,
            1ULL << 20U,
            0.0005,
            0.0,
            0,
        });
    const auto recovered =
        scheduler.plan(
            21,
            LinkDirection::outbound,
            4096,
            TransferPriority::normal);
    expect(
        recovered.kind ==
                PlanKind::single_link &&
            recovered.slices[0].link_index == 0,
        "re-qualified link re-enters selection for new flows after fresh telemetry");

    if (failures == 0) {
        std::cout << "dual-link runtime tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
