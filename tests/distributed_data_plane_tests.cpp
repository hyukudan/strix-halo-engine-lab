#include "strix/distributed_data_plane.hpp"

#include <cstdint>
#include <iostream>
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

strix::transport::DualLinkRuntimeScheduler
scheduler() {
    std::vector<strix::transport::DirectionalLinkProfile>
        links(2);
    for (std::size_t index = 0;
         index < links.size();
         ++index) {
        auto& profile =
            links[index];
        profile.outbound.id =
            "usb4-" +
            std::to_string(index);
        profile.inbound.id =
            profile.outbound.id;
        profile.outbound.peer_node = 1;
        profile.inbound.peer_node = 1;
        profile.outbound.health =
            strix::transport::LinkHealth::qualified;
        profile.inbound.health =
            strix::transport::LinkHealth::qualified;
        profile.outbound.payload_bytes_per_second =
            index == 0 ? 2.0e9 : 1.5e9;
        profile.inbound.payload_bytes_per_second =
            profile.outbound.payload_bytes_per_second;
        profile.outbound.one_way_latency_seconds = 10e-6;
        profile.inbound.one_way_latency_seconds = 10e-6;
        profile.outbound.independent_for_striping = true;
        profile.inbound.independent_for_striping = true;
    }

    strix::transport::DualLinkRuntimePolicy policy;
    policy.transfer.minimum_stripe_bytes =
        1ULL << 20U;
    return strix::transport::DualLinkRuntimeScheduler(
        std::move(links),
        policy);
}

strix::DistributedPlacementPlan
placement(
    std::uint32_t local_node,
    std::uint32_t source_node,
    std::uint32_t destination_node,
    std::uint64_t bytes) {
    strix::DistributedPlacementPlan plan;
    plan.local_node = local_node;
    plan.plan_identity = 0xabcULL;
    plan.executable = true;
    plan.has_remote_work = true;

    strix::DistributedExecutionBoundary boundary;
    boundary.boundary_id = 0x1234ULL;
    boundary.source_step = 3;
    boundary.destination_step = 4;
    boundary.source_node = source_node;
    boundary.destination_node = destination_node;
    boundary.payload_bytes = bytes;
    plan.boundaries.push_back(boundary);
    return plan;
}

}  // namespace

int main() {
    {
        auto runtime = scheduler();
        auto plan =
            placement(
                0,
                0,
                1,
                64ULL * 1024ULL);
        const auto& boundary =
            plan.boundaries.front();

        const auto transfer =
            strix::plan_distributed_boundary_transfer(
                plan,
                boundary,
                runtime,
                9,
                11,
                2,
                1,
                7);

        expect(
            transfer.flow.direction ==
                    strix::transport::LinkDirection::outbound &&
                transfer.flow.priority ==
                    strix::transport::TransferPriority::latency_critical &&
                transfer.descriptor.model_plan_id ==
                    plan.plan_identity &&
                transfer.descriptor.boundary_id ==
                    boundary.boundary_id &&
                transfer.link_plan.kind ==
                    strix::transport::PlanKind::single_link &&
                transfer.link_plan.slices.front().link_index == 0,
            "outbound distributed boundary binds descriptor and live link plan");
    }

    {
        auto runtime = scheduler();
        auto plan =
            placement(
                1,
                0,
                1,
                2ULL << 20U);
        const auto& boundary =
            plan.boundaries.front();

        auto transfer =
            strix::plan_distributed_boundary_transfer(
                plan,
                boundary,
                runtime,
                9,
                11,
                2,
                1,
                7);

        strix::DistributedRequestState request;
        request.placement_plan_identity =
            plan.plan_identity;
        request.request_id = 11;
        request.generation = 2;
        request.next_receive_sequence = 1;

        strix::validate_distributed_boundary_transfer_receive(
            request,
            plan,
            boundary,
            transfer,
            1);
        expect(
            !request.failed &&
                request.next_receive_sequence == 2 &&
                transfer.flow.direction ==
                    strix::transport::LinkDirection::inbound,
            "receive-side data plane validates flow and advances request sequence");

        request.next_receive_sequence = 1;
        request.failed = false;
        ++transfer.flow.flow_id;
        bool rejected = false;
        try {
            strix::validate_distributed_boundary_transfer_receive(
                request,
                plan,
                boundary,
                transfer,
                1);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected && request.failed,
            "foreign flow identity poisons distributed receive request");
    }

    {
        auto runtime=scheduler();
        const auto plan=placement(0,0,1,4096);
        const auto& boundary=plan.boundaries.front();
        auto transfer=strix::plan_distributed_boundary_transfer(plan,boundary,runtime,9,17,1,1,1);
        const auto sender=strix::classify_distributed_boundary_flow(plan,boundary);
        expect(sender.flow_id==strix::classify_distributed_boundary_flow(plan,boundary,{},0).flow_id,
            "legacy flow and explicit matching endpoint have identical hashes");
        const auto legacy_receiver=placement(1,0,1,4096);
        expect(strix::classify_distributed_boundary_flow(legacy_receiver,legacy_receiver.boundaries.front()).flow_id==
            strix::classify_distributed_boundary_flow(legacy_receiver,legacy_receiver.boundaries.front(),{},1).flow_id,
            "legacy node1 receive flow retains its exact matching-endpoint hash");
        transfer.flow=strix::classify_distributed_boundary_flow(plan,boundary,{},1);
        strix::DistributedRequestState request;
        request.placement_plan_identity=plan.plan_identity;request.request_id=17;request.generation=1;
        request.next_receive_sequence=1;
        strix::validate_distributed_boundary_transfer_receive(request,plan,boundary,transfer,1);
        expect(!request.failed && request.next_receive_sequence==2 && plan.local_node==0 &&
            transfer.descriptor.model_plan_id==plan.plan_identity,
            "explicit receive endpoint1 validates canonical plan0 without retagging its wire identity");
        request.next_receive_sequence=1;request.failed=false;
        transfer.flow=sender;
        bool rejected=false;
        try {strix::validate_distributed_boundary_transfer_receive(request,plan,boundary,transfer,1);}
        catch(const std::exception&) {rejected=true;}
        expect(rejected,"raw outbound sender flow cannot bypass explicit inbound normalization");
        rejected=false;
        try {(void)strix::classify_distributed_boundary_flow(plan,boundary,{},2);}
        catch(const std::exception&) {rejected=true;}
        expect(rejected,"foreign endpoint cannot classify a boundary it does not own");
    }
    if (failures == 0) {
        std::cout
            << "distributed data plane tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
