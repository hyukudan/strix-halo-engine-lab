#include "strix/distributed_flow.hpp"

#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::DistributedPlacementPlan plan_with(
    std::uint64_t bytes,
    bool outbound) {
    strix::DistributedPlacementPlan plan;
    plan.local_node = 0;
    plan.compiled_execution_identity =
        0x100ULL;
    plan.manifest_layout_identity =
        0x200ULL;
    plan.tensor_access_identity =
        0x300ULL;
    plan.plan_identity =
        0x400ULL;
    plan.executable = true;
    plan.has_remote_work = true;

    strix::DistributedExecutionBoundary boundary;
    boundary.boundary_id =
        0x500ULL + bytes;
    boundary.source_step = 1;
    boundary.destination_step = 2;
    boundary.source_node =
        outbound ? 0U : 1U;
    boundary.destination_node =
        outbound ? 1U : 0U;
    boundary.payload_bytes =
        bytes;
    plan.boundaries.push_back(
        boundary);
    return plan;
}

}  // namespace

int main() {
    {
        const auto plan =
            plan_with(4096, true);
        const auto flow =
            strix::classify_distributed_boundary_flow(
                plan,
                plan.boundaries[0]);
        expect(
            flow.flow_id != 0 &&
                flow.direction ==
                    strix::transport::
                        LinkDirection::outbound &&
                flow.priority ==
                    strix::transport::
                        TransferPriority::
                            latency_critical,
            "small outbound boundary gets stable latency-critical flow class");

        const auto repeated =
            strix::classify_distributed_boundary_flow(
                plan,
                plan.boundaries[0]);
        expect(
            repeated.flow_id ==
                flow.flow_id,
            "same placement boundary produces stable flow identity");
    }

    {
        const auto plan =
            plan_with(
                8ULL << 20U,
                false);
        const auto flow =
            strix::classify_distributed_boundary_flow(
                plan,
                plan.boundaries[0]);
        expect(
            flow.direction ==
                    strix::transport::
                        LinkDirection::inbound &&
                flow.priority ==
                    strix::transport::
                        TransferPriority::bulk,
            "large inbound boundary maps to bulk scheduler class");
    }

    {
        const auto plan =
            plan_with(
                256ULL * 1024ULL,
                true);
        const auto flow =
            strix::classify_distributed_boundary_flow(
                plan,
                plan.boundaries[0]);
        expect(
            flow.priority ==
                strix::transport::
                    TransferPriority::normal,
            "mid-sized boundary maps to normal priority");
    }

    {
        auto plan =
            plan_with(
                4096,
                true);
        auto foreign =
            plan.boundaries[0];
        ++foreign.boundary_id;
        bool rejected = false;
        try {
            static_cast<void>(
                strix::classify_distributed_boundary_flow(
                    plan,
                    foreign));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected,
            "boundary not owned by placement plan is rejected");
    }

    if (failures == 0) {
        std::cout
            << "distributed flow tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
