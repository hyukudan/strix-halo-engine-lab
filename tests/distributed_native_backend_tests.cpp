#include "strix/distributed_native_backend.hpp"

#include <array>
#include <deque>
#include <iostream>
#include <optional>
#include <string>
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

class NativeBackend final
    : public strix::transport::DirectTransportNativeBackend {
public:
    strix::transport::BackendCapabilities capabilities() const override {
        strix::transport::BackendCapabilities value;
        value.name = "fake-native";
        value.maximum_payload_bytes = 1U << 20U;
        value.external_dma = true;
        value.supports_gpu_transport_pool = true;
        value.requires_exclusive_lease = true;
        value.explicit_receive_repost = true;
        value.gpu_commit_record = true;
        value.buffer_mode =
            strix::transport::TransportBufferMode::imported_gpu_dma_buf;
        return value;
    }

    strix::transport::DirectNativeCapabilities
    direct_capabilities() const override {
        return {true,true,true,true,true,true,true};
    }

    void import_link_pool(
        const strix::transport::DirectTransportLease&,
        const strix::transport::DirectTransportLinkPoolBinding&) override {}

    void release_link_pool(
        const strix::transport::DirectTransportLease&) noexcept override {}

    strix::transport::TransferHandle prepare_send(
        const strix::transport::TransferDescriptor& descriptor,
        const strix::transport::TransportBufferView&) override {
        const auto handle = next_handle++;
        sends.push_back({handle,descriptor.byte_length});
        return handle;
    }

    strix::transport::TransferHandle prepare_receive(
        const strix::transport::TransferDescriptor& descriptor,
        const strix::transport::TransportBufferView&) override {
        const auto handle = next_handle++;
        receives.push_back({handle,descriptor.byte_length});
        return handle;
    }

    void submit(
        strix::transport::TransferHandle handle) override {
        for (const auto& item : sends) {
            if (item.first == handle) {
                events.push_back({
                    {handle,strix::transport::CompletionKind::local_buffer_reusable,item.second},
                    false});
                return;
            }
        }
        for (const auto& item : receives) {
            if (item.first == handle) {
                events.push_back({
                    {handle,strix::transport::CompletionKind::receive_complete,item.second},
                    true});
                return;
            }
        }
        throw std::invalid_argument("unknown handle");
    }

    void cancel(
        strix::transport::TransferHandle) override {}

    void publish_gpu_commit(
        strix::transport::TransferHandle,
        const strix::transport::GpuCommitRecord&) override {
        ++commits;
    }

    void publish_consumer_credit(
        strix::transport::TransferHandle,
        std::uint64_t) override {
        ++credits;
    }

    void repost_receive(
        strix::transport::TransferHandle handle) override {
        for (const auto& item : receives) {
            if (item.first == handle) {
                events.push_back({
                    {handle,strix::transport::CompletionKind::receive_reposted,item.second},
                    true});
                return;
            }
        }
    }

    std::optional<strix::transport::DirectNativeEvent>
    poll_direct_event() override {
        if (events.empty()) return std::nullopt;
        auto value = events.front();
        events.pop_front();
        return value;
    }

    std::uint64_t next_handle = 100;
    int commits = 0;
    int credits = 0;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> sends;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> receives;
    std::deque<strix::transport::DirectNativeEvent> events;
};

class Control final
    : public strix::DistributedNativeControlPlane {
public:
    void execute_remote_step(
        const strix::DistributedStepPlacement& step) override {
        events.push_back("remote:"+std::to_string(step.step_id));
    }

    void publish_boundary_transfer(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer& transfer) override {
        published = transfer;
        events.push_back("publish");
    }

    strix::DistributedBoundaryTransfer await_boundary_transfer(
        const strix::DistributedExecutionBoundary&) override {
        events.push_back("await");
        return inbound;
    }

    strix::DistributedBoundaryTransfer inbound;
    strix::DistributedBoundaryTransfer published;
    std::vector<std::string> events;
};

class Local final
    : public strix::DistributedNativeLocalExecutor {
public:
    void execute_local_step(
        const strix::DistributedStepPlacement& step) override {
        steps.push_back(step.step_id);
    }
    std::vector<std::uint32_t> steps;
};

class Staging final
    : public strix::DistributedBoundaryStagingProvider {
public:
    strix::transport::TransportBufferView stage_outbound(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&) override {
        ++outbound;
        return tx;
    }

    strix::transport::TransportBufferView stage_inbound(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&) override {
        ++inbound;
        return rx;
    }

    void consume_inbound(
        const strix::DistributedExecutionBoundary&,
        const strix::transport::TransportBufferView&) override {
        ++consumed;
    }

    strix::transport::TransportBufferView tx;
    strix::transport::TransportBufferView rx;
    int outbound = 0;
    int inbound = 0;
    int consumed = 0;
};

strix::transport::DirectLinkLifecycle link() {
    strix::transport::DirectLinkLifecycle value;
    value.physical_link_id = "usb4-a";
    value.state =
        strix::transport::DirectLinkLifecycleState::direct_ready;
    value.connection_epoch = 9;
    value.control_link_ready = true;
    value.peer_reconciled = true;
    value.receive_ring_primed = true;
    value.exclusive_lease_available = true;
    value.dma_buf_import_supported = true;
    value.completion_rearm_qualified = true;
    return value;
}

strix::transport::DirectTransportPoolMemoryBinding pools() {
    strix::transport::DirectTransportPoolMemoryBinding value;
    value.binding_identity = 0x100;
    value.total_gpu_bytes = 8192;
    value.links.resize(1);
    value.links[0].tx.address = 0x10000000;
    value.links[0].tx.bytes = 4096;
    value.links[0].tx.allocation_generation = 1;
    value.links[0].tx.domain =
        strix::transport::TransportMemoryDomain::gpu_transport_pool;
    value.links[0].tx.access =
        strix::transport::TransportBufferAccess::read_write;
    value.links[0].rx = value.links[0].tx;
    value.links[0].rx.address = 0x10001000;
    value.links[0].rx.allocation_generation = 2;
    return value;
}

strix::DistributedBoundaryTransfer transfer(
    strix::transport::LinkDirection direction) {
    strix::DistributedBoundaryTransfer value;
    value.flow.flow_id = 1;
    value.flow.direction = direction;
    value.flow.priority =
        strix::transport::TransferPriority::normal;
    value.descriptor.connection_epoch = 9;
    value.descriptor.model_plan_id = 1;
    value.descriptor.request_id = 2;
    value.descriptor.generation = 3;
    value.descriptor.boundary_id = 4;
    value.descriptor.sequence = 5;
    value.descriptor.shape_fingerprint = 6;
    value.descriptor.dtype_id = 7;
    value.descriptor.byte_length = 128;
    value.link_plan.kind =
        strix::transport::PlanKind::single_link;
    value.link_plan.slices.push_back({0,0,128});
    return value;
}

}  // namespace

int main() {
    NativeBackend native;
    auto lifecycle = link();
    auto memory = pools();
    strix::transport::DirectTransportLeaseRegistry leases;
    auto session =
        strix::transport::DirectTransportNativeLinkSession::open(
            native,leases,lifecycle,memory,0);

    Control control;
    Local local;
    Staging staging;
    staging.tx = memory.links[0].tx;
    staging.rx = memory.links[0].rx;
    control.inbound =
        transfer(strix::transport::LinkDirection::inbound);

    std::array<strix::transport::DirectTransportNativeLinkSession*,1>
        sessions{&session};
    strix::SingleLinkNativeDistributedBackend backend(
        sessions,control,local,staging);

    strix::DistributedStepPlacement local_step;
    local_step.step_id = 1;
    backend.execute_local_step(local_step);
    expect(local.steps == std::vector<std::uint32_t>({1}),
           "native distributed backend delegates local step");

    strix::DistributedStepPlacement remote_step;
    remote_step.step_id = 2;
    backend.execute_remote_step(remote_step);
    expect(control.events.front() == "remote:2",
           "native distributed backend delegates remote step control");

    strix::DistributedExecutionBoundary boundary;
    boundary.boundary_id = 4;
    boundary.payload_bytes = 128;

    const auto outbound =
        transfer(strix::transport::LinkDirection::outbound);
    backend.send_boundary(boundary,outbound);
    expect(staging.outbound == 1 &&
               native.commits == 1 &&
               control.published.descriptor.sequence == 5,
           "single-link native send stages, publishes control metadata and commit");

    const auto observed =
        backend.receive_boundary(boundary);
    expect(observed.flow.direction ==
               strix::transport::LinkDirection::inbound &&
               staging.inbound == 1 &&
               staging.consumed == 1 &&
               native.credits == 1 &&
               session.receive_state(native.receives.back().first).ownership ==
                   strix::transport::BufferOwnership::reusable,
           "single-link native receive completes consume-credit-repost lifecycle");

    auto striped = outbound;
    striped.link_plan.kind =
        strix::transport::PlanKind::striped;
    striped.link_plan.slices = {{0,0,64},{0,64,64}};
    bool striped_rejected = false;
    try {
        backend.send_boundary(boundary,striped);
    } catch (const std::logic_error&) {
        striped_rejected = true;
    }
    expect(striped_rejected,
           "native distributed backend fails closed on striped plan before reassembly support");

    if (failures == 0) {
        std::cout << "distributed native backend tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
