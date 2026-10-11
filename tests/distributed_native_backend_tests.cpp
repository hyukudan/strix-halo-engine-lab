#include "strix/distributed_native_backend.hpp"
#include "direct_native_fixture.hpp"

#include <array>
#include <deque>
#include <functional>
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
        return synthetic_native_capabilities(native_identity);
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
        send_descriptors.push_back(descriptor);
        return handle;
    }

    strix::transport::TransferHandle prepare_receive(
        const strix::transport::TransferDescriptor& descriptor,
        const strix::transport::TransportBufferView&) override {
        if (fail_receive_prepare) {
            throw std::runtime_error("injected receive prepare failure");
        }
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
        if (suppress_events) return;
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
        strix::transport::TransferHandle handle) override {
        canceled.push_back(handle);
    }

    void publish_gpu_commit(
        strix::transport::TransferHandle,
        const strix::transport::GpuCommitRecord& record) override {
        ++commits;
        published_commits.push_back(
            record);
    }

    strix::transport::GpuCommitRecord read_gpu_commit(
        strix::transport::TransferHandle handle) override {
        if (commit_override) return *commit_override;
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
        ++reposts;
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
    strix::transport::DirectNativeInterfaceIdentity native_identity =
        {1, "synthetic-kernel", "synthetic-driver-build", "synthetic-abi-v1",
         "synthetic-local-node", "synthetic-peer-node", "usb4-a", 9};
    int commits = 0;
    int credits = 0;
    int reposts = 0;
    bool fail_receive_prepare = false;
    bool suppress_events = false;
    std::optional<strix::transport::GpuCommitRecord> commit_override;
    std::vector<strix::transport::TransferHandle> canceled;
    std::vector<strix::transport::TransferDescriptor> send_descriptors;
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

struct ConsumerProofState {
    int pending_polls = 0;
    int polls = 0;
    int destroyed = 0;
    bool fail = false;
    bool throw_on_poll = false;
    std::function<void()> check_ownership;
};

class ConsumerProof final : public strix::DistributedInboundCompletion {
public:
    explicit ConsumerProof(std::shared_ptr<ConsumerProofState> state)
        : state_(std::move(state)) {}

    ~ConsumerProof() override { ++state_->destroyed; }

    strix::DistributedInboundCompletionStatus poll() override {
        ++state_->polls;
        if (state_->check_ownership) state_->check_ownership();
        if (state_->throw_on_poll) {
            throw std::runtime_error("injected completion query failure");
        }
        if (state_->polls <= state_->pending_polls) {
            return strix::DistributedInboundCompletionStatus::pending;
        }
        return state_->fail
            ? strix::DistributedInboundCompletionStatus::failed
            : strix::DistributedInboundCompletionStatus::complete;
    }

private:
    std::shared_ptr<ConsumerProofState> state_;
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
        if (throw_on_consume) throw std::runtime_error("injected consumer launch failure");
    }

    std::unique_ptr<strix::DistributedInboundCompletion> finish_inbound(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&) override {
        ++finished;
        if (throw_on_finish) throw std::runtime_error("injected finish failure");
        if (omit_proof) return nullptr;
        return std::make_unique<ConsumerProof>(proof);
    }

    strix::transport::TransportBufferView tx;
    strix::transport::TransportBufferView rx;
    int outbound = 0;
    int inbound = 0;
    int consumed = 0;
    int finished = 0;
    bool omit_proof = false;
    bool throw_on_consume = false;
    bool throw_on_finish = false;
    std::shared_ptr<ConsumerProofState> proof =
        std::make_shared<ConsumerProofState>();
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

    std::unique_ptr<strix::DistributedInboundCompletion> finish_inbound(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer&) override {
        ++finished;
        return std::make_unique<ConsumerProof>(proof);
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
    std::shared_ptr<ConsumerProofState> proof =
        std::make_shared<ConsumerProofState>();
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
    value.expected_native_interface = synthetic_native_identity(value);
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
    value.descriptor.connection_epoch = 77;
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

class SingleFixture {
public:
    explicit SingleFixture(strix::DistributedNativeBackendPolicy policy = {})
        : lifecycle(link()), memory(pools()),
          session(strix::transport::DirectTransportNativeLinkSession::open(
              native, leases, lifecycle, memory, 0)),
          sessions{&session}, backend(sessions, control, local, staging, policy) {
        staging.tx = memory.links[0].tx;
        staging.rx = memory.links[0].rx;
        boundary.boundary_id = 4;
        boundary.payload_bytes = 128;
        control.inbound = transfer(strix::transport::LinkDirection::inbound);
        control.inbound.native_slices = {{lifecycle.connection_epoch, 1}};
    }

    NativeBackend native;
    strix::transport::DirectLinkLifecycle lifecycle;
    strix::transport::DirectTransportPoolMemoryBinding memory;
    strix::transport::DirectTransportLeaseRegistry leases;
    strix::transport::DirectTransportNativeLinkSession session;
    Control control;
    Local local;
    Staging staging;
    std::array<strix::transport::DirectTransportNativeLinkSession*, 1> sessions;
    strix::NativeDistributedBackend backend;
    strix::DistributedExecutionBoundary boundary;
};

bool receive_rejected(SingleFixture& fixture) {
    try {
        static_cast<void>(fixture.backend.receive_boundary(fixture.boundary));
    } catch (const std::exception&) {
        return true;
    }
    return false;
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
               control.published.descriptor.sequence == 5 &&
               control.published.descriptor.connection_epoch == 77 &&
               control.published.native_slices[0].connection_epoch == 9 &&
               control.published.native_slices[0].sequence == 1 &&
               native.published_commits[0].connection_epoch == 9 &&
               native.published_commits[0].sequence == 1,
           "single-link native send stages, publishes control metadata and commit");

    control.inbound = control.published;
    control.inbound.flow.direction = strix::transport::LinkDirection::inbound;
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
        lifecycle1.connection_epoch = 23;
        lifecycle1.expected_native_interface = synthetic_native_identity(lifecycle1);
        native1.native_identity = *lifecycle1.expected_native_interface;
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
                native1.published_commits[0].stripe_count == 2 &&
                native0.published_commits[0].connection_epoch == 9 &&
                native1.published_commits[0].connection_epoch == 23 &&
                striped_control.published.descriptor.connection_epoch == 77 &&
                striped_control.published.native_slices[1].connection_epoch == 23,
            "striped native send publishes one indexed commit per physical link");

        striped_control.inbound = striped_control.published;
        striped_control.inbound.flow.direction =
            strix::transport::LinkDirection::inbound;
        striped_staging.proof->pending_polls = 3;
        striped_staging.proof->check_ownership = [&] {
            expect(striped_staging.finished == 1 &&
                       striped_staging.consumed_indices.size() == 2 &&
                       native0.credits == 0 && native1.credits == 0 &&
                       native0.reposts == 0 && native1.reposts == 0 &&
                       session0.receive_state(native0.receives.back().first).ownership ==
                           strix::transport::BufferOwnership::consumer_owned &&
                       session1.receive_state(native1.receives.back().first).ownership ==
                           strix::transport::BufferOwnership::consumer_owned,
                   "both stripes remain consumer-owned after CPU callbacks until proof completes");
        };
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
                striped_staging.proof->polls == 4 &&
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

    {
        SingleFixture fixture;
        fixture.staging.proof->pending_polls = 2;
        fixture.staging.proof->check_ownership = [&] {
            expect(fixture.staging.consumed == 1 && fixture.staging.finished == 1 &&
                       fixture.native.credits == 0 && fixture.native.reposts == 0 &&
                       fixture.session.receive_state(fixture.native.receives.back().first).ownership ==
                           strix::transport::BufferOwnership::consumer_owned,
                   "delayed consumer proof withholds RX release, credit and repost");
        };
        static_cast<void>(fixture.backend.receive_boundary(fixture.boundary));
        expect(fixture.staging.proof->polls == 3 && fixture.native.credits == 1 &&
                   fixture.native.reposts == 1,
               "consumer completion proof explicitly authorizes retirement");
        const auto received_count = fixture.native.receives.size();
        expect(receive_rejected(fixture) &&
                   fixture.native.receives.size() == received_count &&
                   fixture.staging.consumed == 1 && fixture.native.credits == 1,
               "duplicate physical message is rejected before staging or consumption");
    }

    for (int mode = 0; mode < 6; ++mode) {
        SingleFixture fixture({16, 3});
        fixture.staging.proof->pending_polls = mode == 0 ? 10 : 0;
        fixture.staging.proof->fail = mode == 1;
        fixture.staging.proof->throw_on_poll = mode == 2;
        fixture.staging.omit_proof = mode == 3;
        fixture.staging.throw_on_consume = mode == 4;
        fixture.staging.throw_on_finish = mode == 5;
        expect(receive_rejected(fixture) && fixture.native.credits == 0 &&
                   fixture.native.reposts == 0 &&
                   fixture.session.receive_state(fixture.native.receives.back().first).ownership ==
                       strix::transport::BufferOwnership::consumer_owned,
               "pending/failed/throwing/missing proof and callback errors quarantine consumer-owned RX");
        expect(fixture.session.receive_quarantined(),
               "failed consumer completion quarantines physical RX lease");
        if (mode < 3) {
            expect(fixture.staging.proof->destroyed == 0,
                   "failed completion proof remains retained after receive failure");
        }
        const auto received_count = fixture.native.receives.size();
        ++fixture.control.inbound.native_slices[0].sequence;
        expect(receive_rejected(fixture) && fixture.native.receives.size() == received_count,
               "quarantined consumer prevents further native backend work");
        strix::NativeDistributedBackend another_backend(
            fixture.sessions, fixture.control, fixture.local, fixture.staging);
        bool pool_reuse_rejected = false;
        try { static_cast<void>(another_backend.receive_boundary(fixture.boundary)); }
        catch (const std::logic_error&) { pool_reuse_rejected = true; }
        expect(pool_reuse_rejected && fixture.native.receives.size() == received_count,
               "new adapter cannot bypass quarantined physical RX pool");
    }

    {
        SingleFixture fixture;
        fixture.control.inbound.native_slices[0].connection_epoch = 8;
        expect(receive_rejected(fixture) && fixture.native.receives.empty() &&
                   fixture.staging.inbound == 0 && fixture.staging.consumed == 0,
               "stale physical lease epoch fails before RX preparation");
    }

    {
        SingleFixture fixture;
        auto first = transfer(strix::transport::LinkDirection::outbound);
        auto second = first;
        second.descriptor.request_id = 99;
        fixture.backend.send_boundary(fixture.boundary, first);
        const auto first_published = fixture.control.published;
        Control second_control;
        strix::NativeDistributedBackend second_backend(
            fixture.sessions, second_control, fixture.local, fixture.staging);
        second_backend.send_boundary(fixture.boundary, second);
        const auto second_published = second_control.published;
        expect(first_published.descriptor.sequence == second_published.descriptor.sequence &&
                   first_published.descriptor.request_id != second_published.descriptor.request_id &&
                   first_published.native_slices[0].sequence == 1 &&
                   second_published.native_slices[0].sequence == 2 &&
                   fixture.native.send_descriptors[0].request_id == first.descriptor.request_id &&
                   fixture.native.send_descriptors[1].request_id == second.descriptor.request_id,
               "serialized request adapters with equal logical sequence reserve distinct physical sequence");
        fixture.control.inbound = first_published;
        fixture.control.inbound.flow.direction = strix::transport::LinkDirection::inbound;
        fixture.native.commit_override = fixture.native.published_commits[1];
        expect(receive_rejected(fixture) && fixture.staging.consumed == 0 &&
                   fixture.native.credits == 0,
               "commit from a foreign pending request fails before any payload consumption");
    }

    {
        SingleFixture fixture({1, 1});
        fixture.backend.send_boundary(fixture.boundary,
            transfer(strix::transport::LinkDirection::outbound));
        static_cast<void>(fixture.backend.receive_boundary(fixture.boundary));
        expect(fixture.native.credits == 1 && fixture.native.reposts == 1,
               "completion on final permitted poll is accepted without cancellation");
    }

    {
        SingleFixture fixture({1, 1});
        fixture.native.suppress_events = true;
        expect(receive_rejected(fixture) && fixture.native.canceled.size() == 1 &&
                   fixture.session.receive_quarantined() && fixture.staging.consumed == 0,
               "RX timeout cancels once and quarantines without unproven cancellation reuse");
    }

    for (int mode = 0; mode < 3; ++mode) {
        SingleFixture fixture;
        strix::transport::GpuCommitRecord record;
        record.commit_word = strix::transport::gpu_commit_word;
        record.connection_epoch = mode == 0 ? 8 : 9;
        record.sequence = mode == 1 ? 2 : 1;
        record.payload_bytes = mode == 2 ? 127 : 128;
        fixture.native.commit_override = record;
        expect(receive_rejected(fixture) && fixture.staging.consumed == 0 &&
                   fixture.native.credits == 0 && fixture.native.reposts == 0,
               "stale epoch, foreign sequence and short stripe commits fail closed");
    }

    {
        NativeBackend native0;
        NativeBackend native1;
        native1.fail_receive_prepare = true;
        auto lifecycle0 = link();
        auto lifecycle1 = link();
        lifecycle1.physical_link_id = "usb4-b";
        lifecycle1.connection_epoch = 23;
        lifecycle1.expected_native_interface = synthetic_native_identity(lifecycle1);
        native1.native_identity = *lifecycle1.expected_native_interface;
        auto memory2 = pools_two();
        strix::transport::DirectTransportLeaseRegistry leases2;
        auto session0 = strix::transport::DirectTransportNativeLinkSession::open(
            native0, leases2, lifecycle0, memory2, 0);
        auto session1 = strix::transport::DirectTransportNativeLinkSession::open(
            native1, leases2, lifecycle1, memory2, 1);
        Control control2;
        Local local2;
        StripedStaging staging2;
        staging2.rx = {memory2.links[0].rx, memory2.links[1].rx};
        control2.inbound = transfer(strix::transport::LinkDirection::inbound);
        control2.inbound.link_plan.kind = strix::transport::PlanKind::striped;
        control2.inbound.link_plan.slices = {{0,0,64},{1,64,64}};
        control2.inbound.native_slices = {{9,1},{23,1}};
        std::array<strix::transport::DirectTransportNativeLinkSession*, 2> sessions2{
            &session0, &session1};
        strix::NativeDistributedBackend backend2(sessions2, control2, local2, staging2);
        bool rejected = false;
        try { static_cast<void>(backend2.receive_boundary(boundary)); }
        catch (const std::runtime_error&) { rejected = true; }
        expect(rejected && native0.canceled.size() == 1 && native1.canceled.empty() &&
                   staging2.consumed_indices.empty() && native0.credits == 0,
               "partial RX preparation cancels the earlier prepared slice");
    }

    if (failures == 0) {
        std::cout << "distributed native backend tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
