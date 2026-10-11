#include "strix/graph_eligibility.hpp"
#include "strix/compiled_execution.hpp"
#include "strix/compiled_speculative_verify.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template<class Exception, class Function>
void rejects(Function function, const char* message) {
    try { function(); expect(false, message); }
    catch (const Exception&) {}
    catch (...) { expect(false, "wrong graph error category"); }
}
} // namespace

int main() {
    // Portable metadata fixture only: no pointers are dereferenced or GPU launched.
    strix::ModelManifest manifest;
    strix::ManifestTensor tensor;
    tensor.name = "fixture.weight";
    tensor.dimensions = {32, 32};
    tensor.ggml_type = 1;
    tensor.byte_size = 2048;
    manifest.tensors.push_back(tensor);
    manifest.total_tensor_bytes = tensor.byte_size;
    manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
    strix::ModelRuntimeBinding weights;
    weights.manifest_layout_identity = manifest.layout_identity;
    weights.model_content_identity.bytes[0] = 1;
    weights.tensors.push_back({0, strix::RuntimeTensorStorage::resident, 0x60000000, 2048, 7, 0});
    strix::ExecutionArenaPlan arena;
    arena.alignment = 256;
    arena.total_bytes = 4096;
    strix::ExecutionMemoryBinding scratch{0x1234, 0x70000000, 4096, 3};
    strix::SerialSessionStateLayout layout;
    layout.context_tokens = 32;
    layout.device_alignment = 256;
    layout.host_alignment = 64;
    layout.device_total_bytes = 4096;
    layout.host_total_bytes = 64;
    layout.layout_identity = 0x5678;
    strix::SessionStateMemoryBinding state;
    state.layout_identity = layout.layout_identity;
    state.device_base = 0x71000000;
    state.device_bytes = 4096;
    state.device_allocation_generation = 5;
    state.host_base = 0x72000000;
    state.host_bytes = 64;
    state.host_allocation_generation = 9;
    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity = manifest.layout_identity;
    access.plan_identity = 0x4444;
    strix::ExecutionRuntimeResources resources;
    resources.scratch = &scratch;
    resources.model_manifest = &manifest;
    resources.model_weights = &weights;
    resources.tensor_access = &access;
    resources.session_state_layout = &layout;
    resources.session_state_memory = &state;

    const strix::GraphExecutionContract contract{0x1234, 0x4321, 1, true};
    strix::GraphDynamicArguments args;
    args.mode = strix::GraphArgumentMode::device_data;
    args.memory = {0x1234, 0x73000000, 64, 17};
    args.token_offset = 0;
    args.position_offset = 8;
    args.device_argument_abi_identity = 0xabcd;
    strix::GraphMemoryRequest budget;
    budget.sessions = 3;
    budget.shapes_per_session = 2;
    budget.eligibility_classes_per_shape = 4;
    budget.workspace_bytes_per_instance = 1024;
    budget.executable_bytes_per_instance = 2048;
    budget.capture_peak_bytes = 8192;
    budget.budget_bytes = 24 * (1024 + 2048 + 64) + 8192;
    const auto capture = strix::prepare_graph_capture_plan(contract, arena, resources, args, budget);
    expect(capture.instance_count == 24 && capture.charged_bytes == budget.budget_bytes,
        "all session/shape/eligibility instances charge workspace, executable, input and capture bytes");
    const auto original = strix::assess_graph_replay_eligibility(capture, contract, arena, resources, args, 0);
    expect(original.candidate_eligible && !original.native_replay_qualified,
        "portable fixture eligibility remains experimental and unqualified for native replay");
    expect(strix::assess_graph_replay_eligibility(capture, contract, arena, resources, args, 31).candidate_eligible,
        "absolute position can change only as explicit dynamic device data, not a capture scalar");
    expect(!strix::assess_graph_replay_eligibility(capture, contract, arena, resources, args, 32).candidate_eligible,
        "current allocation capacity remains a per-replay gate");

    auto changed_args = args;
    ++changed_args.memory.allocation_generation;
    expect(!strix::assess_graph_replay_eligibility(capture, contract, arena, resources, changed_args, 0).candidate_eligible,
        "device argument allocation generation change invalidates capture at the same address");
    changed_args = args;
    changed_args.mode = strix::GraphArgumentMode::baked_scalars;
    rejects<std::invalid_argument>([&] { (void)strix::prepare_graph_capture_plan(contract, arena, resources, changed_args, budget); },
        "eager token/position scalar arguments do not become dynamic by capture");
    changed_args = args;
    changed_args.device_argument_abi_identity = 0;
    rejects<std::invalid_argument>([&] { (void)strix::prepare_graph_capture_plan(contract, arena, resources, changed_args, budget); },
        "device address declaration without a dynamic argument ABI is insufficient");
    changed_args = args;
    changed_args.position_offset = 0;
    rejects<std::invalid_argument>([&] { (void)strix::prepare_graph_capture_plan(contract, arena, resources, changed_args, budget); },
        "overlapping token/absolute-position storage rejected");

    auto moved_scratch = scratch;
    ++moved_scratch.allocation_generation;
    auto current = resources;
    current.scratch = &moved_scratch;
    expect(!strix::assess_graph_replay_eligibility(capture, contract, arena, current, args, 0).candidate_eligible,
        "scratch generation tampering cannot reuse captured addresses");
    auto moved_state = state;
    ++moved_state.device_allocation_generation;
    current = resources;
    current.session_state_memory = &moved_state;
    expect(!strix::assess_graph_replay_eligibility(capture, contract, arena, current, args, 0).candidate_eligible,
        "session state allocation generation is part of capture identity");
    auto moved_weights = weights;
    ++moved_weights.tensors[0].allocation_generation;
    current = resources;
    current.model_weights = &moved_weights;
    expect(!strix::assess_graph_replay_eligibility(capture, contract, arena, current, args, 0).candidate_eligible,
        "weight allocation generation is part of graph binding identity");
    for (const auto storage : {strix::RuntimeTensorStorage::registered_file_mapped, strix::RuntimeTensorStorage::file_mapped}) {
        moved_weights = weights;
        moved_weights.tensors[0].storage = storage;
        current.model_weights = &moved_weights;
        expect(!strix::assess_graph_replay_eligibility(capture, contract, arena, current, args, 0).candidate_eligible,
            "GPU visibility of mapped weights does not eliminate host residency decisions");
    }
    auto forged = capture;
    ++forged.instance_count;
    expect(!strix::assess_graph_replay_eligibility(forged, contract, arena, resources, args, 0).candidate_eligible,
        "tampered graph instance accounting is not accepted");
    auto unsafe = contract;
    unsafe.whole_plan_graph_safe = false;
    expect(!strix::assess_graph_replay_eligibility(capture, unsafe, arena, resources, args, 0).candidate_eligible,
        "unsafe current compiled plan cannot reuse otherwise intact capture");
    auto small_budget = budget;
    --small_budget.budget_bytes;
    rejects<std::invalid_argument>([&] { (void)strix::prepare_graph_capture_plan(contract, arena, resources, args, small_budget); },
        "all retained instances must fit before capture preparation");
    small_budget = budget;
    small_budget.sessions = std::numeric_limits<std::uint64_t>::max();
    rejects<std::overflow_error>([&] { (void)strix::prepare_graph_capture_plan(contract, arena, resources, args, small_budget); },
        "graph instance multiplication cannot silently wrap");

    strix::CompiledSerialExecution serial;
    serial.execution_identity = contract.execution_identity;
    serial.kernel_plan_identity = contract.kernel_plan_identity;
    serial.arena = arena;
    // Actual initial serial/verify compilers retain unsafe runtime/dynamic steps.
    rejects<std::invalid_argument>([&] { (void)strix::prepare_compiled_serial_graph_capture(serial, resources, args, budget); },
        "compiled serial consumer preserves eager path while native dynamic graph support is absent");
    serial.whole_plan_graph_safe = true;
    rejects<std::invalid_argument>([&] { (void)strix::prepare_compiled_serial_graph_capture(serial, resources, args, budget); },
        "changing a blanket graph flag cannot bypass missing/unsafe compiled steps");
    strix::CompiledSpeculativeVerifyExecution verify;
    verify.execution_identity = contract.execution_identity;
    verify.kernel_plan_identity = contract.kernel_plan_identity;
    verify.candidate_depth = 1;
    verify.arena = arena;
    rejects<std::invalid_argument>([&] { (void)strix::prepare_compiled_verify_graph_capture(verify, resources, args, budget); },
        "compiled verify consumer cannot self-promote graph safety");
    verify.whole_plan_graph_safe = true;
    rejects<std::invalid_argument>([&] { (void)strix::prepare_compiled_verify_graph_capture(verify, resources, args, budget); },
        "blanket verify flag cannot create an executable dynamic device-data graph ABI");

    if (failures == 0) { std::cout << "graph eligibility portable tests passed; native replay unqualified\n"; }
    return failures == 0 ? 0 : 1;
}
