#include "strix/distributed_placement.hpp"

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
    strix::CompiledSerialExecution compiled;
    compiled.execution_identity = 0x100ULL;
    compiled.execution.steps = {
        {0, strix::SerialOp::token_embedding, {}, {}, strix::PersistentState::none, strix::PersistentState::none, true, false},
        {1, strix::SerialOp::mlp_hyper_mix, {}, {0}, strix::PersistentState::none, strix::PersistentState::none, true, false},
        {2, strix::SerialOp::moe_experts, {}, {1}, strix::PersistentState::none, strix::PersistentState::none, true, false},
        {3, strix::SerialOp::moe_combine, {}, {2}, strix::PersistentState::none, strix::PersistentState::none, true, false},
    };

    const strix::ExecutionBufferUse hidden0{
        {strix::ExecutionBufferClass::hidden, 0},
        1,
    };
    const strix::ExecutionBufferUse hidden1{
        {strix::ExecutionBufferClass::hidden, 1},
        1,
    };
    const strix::ExecutionBufferUse hidden2{
        {strix::ExecutionBufferClass::hidden, 2},
        1,
    };
    compiled.io.steps = {
        {0, strix::SerialOp::token_embedding, {}, {hidden0}, {}},
        {1, strix::SerialOp::mlp_hyper_mix, {hidden0}, {hidden1}, {}},
        {2, strix::SerialOp::moe_experts, {hidden1}, {hidden2}, {}},
        {3, strix::SerialOp::moe_combine, {hidden2}, {}, {}},
    };
    compiled.arena.allocations = {
        {hidden0, 0, 1, 0, 512},
        {hidden1, 1, 2, 512, 512},
        {hidden2, 2, 3, 1024, 512},
    };
    compiled.arena.total_bytes = 1536;

    strix::ModelManifest manifest;
    manifest.architecture = "fixture";
    for (std::uint32_t index = 0; index < 3; ++index) {
        strix::ManifestTensor tensor;
        tensor.name = "tensor." + std::to_string(index);
        tensor.dimensions = {16, 16};
        tensor.ggml_type = 1;
        tensor.byte_size = 512;
        tensor.absolute_offset = 4096 + index * 512;
        tensor.role = strix::TensorRole::dense_projection;
        if (index == 2) {
            tensor.placement.kind =
                strix::PlacementKind::remote;
            tensor.placement.node = 1;
        }
        manifest.total_tensor_bytes += tensor.byte_size;
        manifest.tensors.push_back(std::move(tensor));
    }
    manifest.layout_identity =
        strix::model_manifest_layout_identity(manifest);

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity = 0x200ULL;
    access.accesses = {
        {0, strix::SerialOp::token_embedding, 0, {0}},
        {1, strix::SerialOp::mlp_hyper_mix, 0, {1}},
        {2, strix::SerialOp::moe_experts, 0, {2}},
    };

    const auto plan =
        strix::build_distributed_placement_plan(
            compiled,
            manifest,
            access,
            0);

    expect(plan.executable &&
               plan.has_remote_work &&
               plan.remote_nodes ==
                   std::vector<std::uint32_t>({1}),
           "distributed placement identifies one remote execution node");
    expect(plan.steps[2].node == 1 &&
               plan.steps[3].node == 1,
           "tensor-free continuation inherits the remote dependency placement");
    expect(plan.boundaries.size() == 1 &&
               plan.boundaries[0].source_step == 1 &&
               plan.boundaries[0].destination_step == 2 &&
               plan.boundaries[0].payload_bytes == 512,
           "distributed placement creates one activation boundary with exact arena bytes");

    const std::vector<std::uint32_t>
        available{1};
    const auto distributed =
        strix::select_distributed_execution_mode(
            plan,
            available);
    expect(distributed.mode ==
               strix::DistributedExecutionMode::distributed,
           "available remote node selects distributed execution");

    const std::vector<std::uint32_t>
        unavailable{};
    const auto no_fallback =
        strix::select_distributed_execution_mode(
            plan,
            unavailable);
    expect(no_fallback.mode ==
               strix::DistributedExecutionMode::unavailable &&
               no_fallback.missing_remote_nodes ==
                   std::vector<std::uint32_t>({1}),
           "missing remote node fails closed without local materialization");

    strix::DistributedFallbackPolicy fallback;
    fallback.local_model_copy_available = true;
    const auto local =
        strix::select_distributed_execution_mode(
            plan,
            unavailable,
            fallback);
    expect(local.mode ==
               strix::DistributedExecutionMode::local_fallback,
           "explicit local model copy enables deterministic fallback");

    const auto descriptor =
        strix::make_distributed_boundary_descriptor(
            plan,
            plan.boundaries[0],
            9,
            10,
            11,
            12,
            13);
    expect(descriptor.model_plan_id ==
               plan.plan_identity &&
               descriptor.boundary_id ==
                   plan.boundaries[0].boundary_id &&
               descriptor.byte_length == 512,
           "distributed boundary descriptor binds plan/boundary/sequence and payload");

    // Explicit compute placement does not forge remote weight ownership.
    auto local_manifest=manifest;
    for(auto& tensor:local_manifest.tensors) tensor.placement={};
    local_manifest.layout_identity=strix::model_manifest_layout_identity(local_manifest);
    auto local_access=access;local_access.manifest_layout_identity=local_manifest.layout_identity;
    auto explicit_source=compiled;explicit_source.io.hidden_slots=3;
    const std::vector<std::uint32_t> nodes{0,0,1,1};
    const auto explicit_plan=strix::build_distributed_placement_plan(explicit_source,local_manifest,local_access,nodes,0);
    const auto repeated=strix::build_distributed_placement_plan(explicit_source,local_manifest,local_access,nodes,0);
    expect(explicit_plan.executable && explicit_plan.plan_identity==repeated.plan_identity &&
        explicit_plan.steps[2].node==1U && explicit_plan.boundaries.size()==1U &&
        explicit_plan.boundaries[0].payloads==std::vector<strix::ExecutionBufferUse>{hidden1} &&
        explicit_plan.boundaries[0].payload_bytes==512U,
        "explicit compute builder derives stable actual boundaries while all weights remain local");
    const std::vector<std::uint32_t> different{0,1,1,1};
    expect(strix::build_distributed_placement_plan(explicit_source,local_manifest,local_access,different,0).plan_identity!=explicit_plan.plan_identity,
        "different compute intervals produce different placement identities");
    auto arena_changed=explicit_source;arena_changed.arena.total_bytes+=512U;
    expect(strix::build_distributed_placement_plan(arena_changed,local_manifest,local_access,nodes,0).plan_identity!=explicit_plan.plan_identity,
        "explicit identity covers actual arena geometry despite a caller-reused compiled ID");
    for(const auto& bad:std::vector<std::vector<std::uint32_t>>{{},{0,1},{0,0,2,1}}) {
        bool rejected=false;
        try {(void)strix::build_distributed_placement_plan(explicit_source,local_manifest,local_access,bad,0);}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"explicit placement rejects missing rows and unknown physical node IDs");
    }

    if (failures == 0) {
        std::cout << "distributed placement tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
