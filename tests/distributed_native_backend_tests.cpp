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
        receive_descriptors.push_back({
            handle,
            descriptor,
        });
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
        const strix::transport::GpuCommitRecord& record) override {
        ++commits;
        published_commits.push_back(
            record);
    }

    strix::transport::GpuCommitRecord read_gpu_commit(
        strix::transport::TransferHandle handle) override {
        for (const auto& item : receive_descriptors) {
            if (item.first != handle) {
                continue;
            }
            strix::transport::GpuCommitRecord record;
            record.protocol_version =
                item.second.protocol_version;
            record.commit_word =
                strix::transport::gpu_commit_word;
            record.connection_epoch =
                item.second.connection_epoch;
            record.sequence =
                item.second.sequence;
            record.payload_bytes =
                static_cast<std::uint32_t>(
                    item.second.byte_length);
            record.stripe_index =
                receive_stripe_index;
            record.stripe_count =
                receive_stripe_count;
            return record;
        }
        throw std::invalid_argument(
            "unknown receive handle commit");
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
    std::uint16_t receive_stripe_index = 0;
    std::uint16_t receive_stripe_count = 1;
    std::vector<strix::transport::GpuCommitRecord>
        published_commits;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> sends;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> receives;
    std::vector<std::pair<
        std::uint64_t,
        strix::transport::TransferDescriptor>>
        receive_descriptors;
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

class StripedStaging final
    : public strix::DistributedBoundaryStagingProvider {
public:
    strix::transport::TransportBufferView stage_outbound(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&) override {
        throw std::logic_error("striped staging requires slice API");
    }

    strix::transport::TransportBufferView stage_inbound(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&) override {
        throw std::logic_error("striped staging requires slice API");
    }

    void consume_inbound(
        const strix::DistributedExecutionBoundary&,
        const strix::transport::TransportBufferView&) override {
        throw std::logic_error("striped staging requires slice API");
    }

    strix::transport::TransportBufferView stage_outbound_slice(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&,
        const strix::transport::LinkSlice& slice,
        std::uint16_t,
        std::uint16_t) override {
        ++outbound_slices;
        auto view = tx[slice.link_index];
        view.bytes = slice.byte_length;
        return view;
    }

    strix::transport::TransportBufferView stage_inbound_slice(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&,
        const strix::transport::LinkSlice& slice,
        std::uint16_t,
        std::uint16_t) override {
        ++inbound_slices;
        auto view = rx[slice.link_index];
        view.bytes = slice.byte_length;
        return view;
    }

    void consume_inbound_slice(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&,
        const strix::transport::LinkSlice& slice,
        std::uint16_t stripe_index,
        std::uint16_t stripe_count,
        const strix::transport::TransportBufferView&) override {
        consumed_offsets.push_back(
            slice.byte_offset);
        consumed_indices.push_back(
            stripe_index);
        observed_stripe_count =
            stripe_count;
    }

    void finish_inbound(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&) override {
        ++finished;
    }

    std::array<
        strix::transport::TransportBufferView,
        2> tx;
    std::array<
        strix::transport::TransportBufferView,
        2> rx;
    int outbound_slices = 0;
    int inbound_slices = 0;
    int finished = 0;
    std::uint16_t observed_stripe_count = 0;
    std::vector<std::uint64_t> consumed_offsets;
    std::vector<std::uint16_t> consumed_indices;
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

strix::transport::DirectTransportPoolMemoryBinding pools_two() {
    auto value = pools();
    value.binding_identity = 0x200;
    value.total_gpu_bytes = 16384;
    value.links.resize(2);
    value.links[1] =
        value.links[0];
    value.links[1].tx.address = 0x20000000;
    value.links[1].tx.allocation_generation = 3;
    value.links[1].rx.address = 0x20001000;
    value.links[1].rx.allocation_generation = 4;
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

    {
        NativeBackend native0;
        NativeBackend native1;
        native0.receive_stripe_index = 0;
        native0.receive_stripe_count = 2;
        native1.receive_stripe_index = 1;
        native1.receive_stripe_count = 2;

        auto lifecycle0 = link();
        auto lifecycle1 = link();
        lifecycle0.physical_link_id = "usb4-a";
        lifecycle1.physical_link_id = "usb4-b";
        auto memory2 = pools_two();

        strix::transport::DirectTransportLeaseRegistry
            leases2;
        auto session0 =
            strix::transport::DirectTransportNativeLinkSession::open(
                native0,
                leases2,
                lifecycle0,
                memory2,
                0);
        auto session1 =
            strix::transport::DirectTransportNativeLinkSession::open(
                native1,
                leases2,
                lifecycle1,
                memory2,
                1);

        Control striped_control;
        Local striped_local;
        StripedStaging striped_staging;
        striped_staging.tx = {
            memory2.links[0].tx,
            memory2.links[1].tx,
        };
        striped_staging.rx = {
            memory2.links[0].rx,
            memory2.links[1].rx,
        };

        auto striped =
            transfer(
                strix::transport::LinkDirection::outbound);
        striped.link_plan.kind =
            strix::transport::PlanKind::striped;
        striped.link_plan.slices = {
            {0,0,64},
            {1,64,64},
        };

        auto inbound_striped =
            striped;
        inbound_striped.flow.direction =
            strix::transport::LinkDirection::inbound;
        striped_control.inbound =
            inbound_striped;

        std::array<
            strix::transport::DirectTransportNativeLinkSession*,
            2> striped_sessions{
                &session0,
                &session1,
            };
        strix::NativeDistributedBackend striped_backend(
            striped_sessions,
            striped_control,
            striped_local,
            striped_staging);

        striped_backend.send_boundary(
            boundary,
            striped);
        expect(
            native0.published_commits.size() == 1 &&
                native1.published_commits.size() == 1 &&
                native0.published_commits[0].stripe_index == 0 &&
                native1.published_commits[0].stripe_index == 1 &&
                native0.published_commits[0].stripe_count == 2 &&
                native1.published_commits[0].stripe_count == 2,
            "striped native send publishes one indexed commit per physical link");

        const auto striped_observed =
            striped_backend.receive_boundary(
                boundary);
        expect(
            striped_observed.link_plan.kind ==
                    strix::transport::PlanKind::striped &&
                striped_staging.inbound_slices == 2 &&
                striped_staging.consumed_offsets ==
                    std::vector<std::uint64_t>({0,64}) &&
                striped_staging.consumed_indices ==
                    std::vector<std::uint16_t>({0,1}) &&
                striped_staging.observed_stripe_count == 2 &&
                striped_staging.finished == 1 &&
                native0.credits == 1 &&
                native1.credits == 1,
            "striped native receive validates all commits before reassembly/credit");

        auto overlapping =
            striped;
        overlapping.link_plan.slices = {
            {0,0,80},
            {1,64,64},
        };
        bool overlap_rejected = false;
        try {
            striped_backend.send_boundary(
                boundary,
                overlapping);
        } catch (const std::invalid_argument&) {
            overlap_rejected = true;
        }
        expect(
            overlap_rejected,
            "striped native backend rejects overlapping/non-contiguous slices");
    }

    if (failures == 0) {
        std::cout << "distributed native backend tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
