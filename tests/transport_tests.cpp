#include "strix/transport.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    using namespace strix::transport;

    expect(can_transition(BufferOwnership::producer_owned,
                          BufferOwnership::producer_released),
           "producer release is valid");
    expect(!can_transition(BufferOwnership::producer_owned,
                           BufferOwnership::transport_owned),
           "transport cannot steal producer-owned buffer");
    expect(can_transition(BufferOwnership::consumer_owned,
                          BufferOwnership::consumer_released),
           "consumer release precedes repost");
    expect(can_transition(BufferOwnership::consumer_released,
                          BufferOwnership::repost_pending),
           "released receive buffer enters repost pending");
    expect(can_transition(BufferOwnership::repost_pending,
                          BufferOwnership::reusable),
           "only reposted receive buffer becomes reusable");
    expect(!can_transition(BufferOwnership::consumer_owned,
                           BufferOwnership::reusable),
           "consumer-owned receive buffer cannot skip repost lifecycle");
    expect(can_transition(BufferOwnership::reusable,
                          BufferOwnership::producer_owned),
           "reusable buffer can return to producer");

    DescriptorLimits limits;
    limits.protocol_version = 3;
    limits.connection_epoch = 9;
    limits.maximum_bytes = 1024;

    TransferDescriptor good;
    good.protocol_version = 3;
    good.connection_epoch = 9;
    good.model_plan_id = 0x101;
    good.request_id = 17;
    good.generation = 4;
    good.boundary_id = 0x202;
    good.sequence = 11;
    good.shape_fingerprint = 0x303;
    good.dtype_id = 1;
    good.byte_length = 512;
    expect(validate_descriptor(good, limits), "bounded current-epoch descriptor validates");

    auto stale = good;
    stale.connection_epoch = 8;
    expect(!validate_descriptor(stale, limits), "stale epoch is rejected");

    auto zero_epoch_limits = limits;
    zero_epoch_limits.connection_epoch = 0;
    expect(!validate_descriptor(good, zero_epoch_limits),
           "uninitialized expected epoch is rejected");

    auto wrong_shape = good;
    wrong_shape.shape_fingerprint = 0;
    expect(!validate_descriptor(wrong_shape, limits),
           "missing shape identity is rejected");

    auto oversized = good;
    oversized.byte_length = 2048;
    expect(!validate_descriptor(oversized, limits), "oversized payload is rejected");


    GpuCommitRecord commit;
    commit.protocol_version = good.protocol_version;
    commit.commit_word = gpu_commit_word;
    commit.connection_epoch =
        good.connection_epoch;
    commit.sequence = good.sequence;
    commit.payload_bytes = 256;
    commit.stripe_index = 1;
    commit.stripe_count = 2;
    expect(validate_gpu_commit_record(commit, good),
           "GPU commit record validates epoch sequence and stripe metadata");

    auto stale_commit = commit;
    --stale_commit.connection_epoch;
    expect(!validate_gpu_commit_record(stale_commit, good),
           "GPU commit record rejects stale epoch");

    auto bad_stripe = commit;
    bad_stripe.stripe_index = 2;
    expect(!validate_gpu_commit_record(bad_stripe, good),
           "GPU commit record rejects invalid stripe index");

    DirectLinkLifecycle lifecycle;
    lifecycle.physical_link_id = "usb4-a";
    lifecycle.state =
        DirectLinkLifecycleState::arm_allowed;
    lifecycle.connection_epoch = 9;
    lifecycle.control_link_ready = true;
    lifecycle.peer_reconciled = true;
    lifecycle.exclusive_lease_available = true;
    expect(direct_link_armable(lifecycle),
           "direct link is armable only after control reconciliation and lease");

    lifecycle.state =
        DirectLinkLifecycleState::direct_ready;
    lifecycle.receive_ring_primed = true;
    lifecycle.completion_rearm_qualified = true;
    expect(direct_link_usable(lifecycle),
           "qualified direct link becomes usable after RX priming and completion rearm");

    lifecycle.receive_ring_primed = false;
    expect(!direct_link_usable(lifecycle),
           "direct link is unusable before receive ring is primed");

    std::vector<LinkProfile> links(2);
    links[0].id = "a";
    links[0].peer_node = 1;
    links[0].health = LinkHealth::qualified;
    links[0].payload_bytes_per_second = 1'000'000'000.0;
    links[0].one_way_latency_seconds = 20e-6;
    links[0].independent_for_striping = true;

    links[1].id = "b";
    links[1].peer_node = 1;
    links[1].health = LinkHealth::qualified;
    links[1].payload_bytes_per_second = 1'000'000'000.0;
    links[1].one_way_latency_seconds = 30e-6;
    links[1].independent_for_striping = true;

    SchedulerPolicy policy;
    policy.minimum_stripe_bytes = 1ULL << 20U;
    policy.stripe_reassembly_seconds = 5e-6;

    const auto small = plan_transfer(
        links, 4096, TransferPriority::latency_critical, policy);
    expect(small.kind == PlanKind::single_link && small.slices.size() == 1 &&
               small.slices[0].link_index == 0,
           "small latency-critical transfer stays on fastest single link");

    const auto large = plan_transfer(
        links, 16ULL << 20U, TransferPriority::bulk, policy);
    expect(large.kind == PlanKind::striped && large.slices.size() == 2,
           "large transfer stripes across qualified independent links");
    expect(large.slices[0].byte_length + large.slices[1].byte_length ==
               (16ULL << 20U),
           "striped plan covers the payload exactly");

    links[0].in_flight_bytes = 2ULL << 20U;
    const auto imbalanced_stripe = plan_transfer(
        links, 16ULL << 20U, TransferPriority::bulk, policy);
    expect(imbalanced_stripe.kind == PlanKind::striped,
           "queued link can still participate when striping remains faster");
    expect(imbalanced_stripe.slices[0].byte_length <
               imbalanced_stripe.slices[1].byte_length,
           "queued striped split compensates for link backlog");
    links[0].in_flight_bytes = 0;

    links[1].independent_for_striping = false;
    const auto shared = plan_transfer(
        links, 16ULL << 20U, TransferPriority::bulk, policy);
    expect(shared.kind == PlanKind::single_link,
           "shared or unqualified pair does not stripe");

    links[1].independent_for_striping = true;
    links[0].in_flight_bytes = 64ULL << 20U;
    const auto queued = plan_transfer(
        links, 4096, TransferPriority::normal, policy);
    expect(queued.kind == PlanKind::single_link &&
               queued.slices[0].link_index == 1,
           "flow scheduling accounts for in-flight bytes");

    bool rejected_zero = false;
    try {
        static_cast<void>(plan_transfer(
            links, 0, TransferPriority::normal, policy));
    } catch (const std::invalid_argument&) {
        rejected_zero = true;
    }
    expect(rejected_zero, "zero-length transfer is rejected");

    if (failures == 0) {
        std::cout << "transport tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
