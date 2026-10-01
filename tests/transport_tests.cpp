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


    TransferDescriptor striped_descriptor = good;
    striped_descriptor.byte_length = 600;

    StripedMessageTracker tracker(
        striped_descriptor,
        2);

    GpuCommitRecord stripe0;
    stripe0.protocol_version =
        striped_descriptor.protocol_version;
    stripe0.commit_word = gpu_commit_word;
    stripe0.connection_epoch =
        striped_descriptor.connection_epoch;
    stripe0.sequence =
        striped_descriptor.sequence;
    stripe0.payload_bytes = 256;
    stripe0.stripe_index = 0;
    stripe0.stripe_count = 2;

    GpuCommitRecord stripe1 = stripe0;
    stripe1.payload_bytes = 344;
    stripe1.stripe_index = 1;

    expect(tracker.observe_commit(stripe0) &&
               !tracker.complete() &&
               tracker.committed_bytes() == 256,
           "striped tracker keeps partial logical message unpublished");
    expect(tracker.observe_commit(stripe1) &&
               tracker.complete() &&
               !tracker.invalid() &&
               tracker.committed_bytes() == 600,
           "striped tracker publishes only complete logical message");

    StripedMessageTracker duplicate_tracker(
        striped_descriptor,
        2);
    expect(duplicate_tracker.observe_commit(stripe0),
           "first stripe is accepted");
    expect(!duplicate_tracker.observe_commit(stripe0) &&
               duplicate_tracker.invalid(),
           "duplicate stripe invalidates logical message");

    StripedMessageTracker short_tracker(
        striped_descriptor,
        2);
    auto short1 = stripe1;
    short1.payload_bytes = 300;
    expect(short_tracker.observe_commit(stripe0),
           "short fixture first stripe accepted");
    expect(!short_tracker.observe_commit(short1) &&
               short_tracker.invalid() &&
               !short_tracker.complete(),
           "stripe byte sum mismatch never publishes partial tensor");

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


    std::vector<LinkProfile> completion_links(2);
    completion_links[0].id = "wire-fast-consumer-slow";
    completion_links[0].peer_node = 1;
    completion_links[0].health =
        LinkHealth::qualified;
    completion_links[0].payload_bytes_per_second =
        2'000'000'000.0;
    completion_links[0].one_way_latency_seconds =
        10e-6;
    completion_links[0].consumer_completion_seconds =
        80e-6;

    completion_links[1].id = "wire-slower-consumer-fast";
    completion_links[1].peer_node = 1;
    completion_links[1].health =
        LinkHealth::qualified;
    completion_links[1].payload_bytes_per_second =
        1'500'000'000.0;
    completion_links[1].one_way_latency_seconds =
        15e-6;
    completion_links[1].consumer_completion_seconds =
        10e-6;

    SchedulerPolicy receive_policy;
    receive_policy.minimum_stripe_bytes =
        1ULL << 30U;
    receive_policy.completion_target =
        LinkCompletionTarget::receive_ready;
    const auto receive_choice =
        plan_transfer(
            completion_links,
            4096,
            TransferPriority::normal,
            receive_policy);
    expect(receive_choice.slices[0].link_index == 0,
           "receive-ready scheduling prefers faster wire");

    auto consumer_policy = receive_policy;
    consumer_policy.completion_target =
        LinkCompletionTarget::remote_consumer_complete;
    const auto consumer_choice =
        plan_transfer(
            completion_links,
            4096,
            TransferPriority::normal,
            consumer_policy);
    expect(consumer_choice.slices[0].link_index == 1,
           "consumer-complete scheduling can prefer lower end-to-end cost");

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
