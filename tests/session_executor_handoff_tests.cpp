#include "strix/session_executor_handoff.hpp"

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <class Work> void rejects(Work&& work, const char* message) {
    bool rejected = false;
    try { work(); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, message);
}

// Registry whose variants carry `prefix` in their ids: two prefixes give two executions with the
// same steps, IO and arena and different kernel bindings (arithmetic), i.e. siblings.
strix::kernels::KernelRegistry make_registry(const strix::ModelContract& model,
    const strix::ExecutionBindingRequest& request, const std::string& prefix) {
    strix::kernels::KernelRegistry registry;
    std::set<std::string> added;
    for (const auto& step : strix::build_serial_decode_execution_plan(model).steps) {
        const auto family = strix::operator_family_for(step.op);
        if (!family) continue;
        const auto shape = strix::execution_shape_for(model, step.op, request);
        const auto key = std::to_string(static_cast<unsigned>(*family)) + "." + std::to_string(shape.auxiliary_size);
        if (!added.insert(key).second) continue;
        strix::kernels::KernelVariant variant;
        variant.id = prefix + key;
        variant.family = *family;
        variant.phase = strix::kernels::ExecutionPhase::decode;
        variant.shape = shape;
        variant.wave = strix::kernels::WaveSize::wave32;
        variant.resources.workgroup_threads = 256;
        variant.numerics.input_precision = "bf16";
        variant.numerics.accumulator_precision = "fp32";
        variant.numerics.output_precision = "bf16";
        variant.numerics.row_invariant = true;
        variant.numerics.deterministic = true;
        variant.qualification = strix::kernels::QualificationStatus::oracle_matched;
        variant.graph_safe = true;
        variant.input_layout = "test";
        variant.output_layout = "test";
        variant.compiler_identity = "test";
        registry.add(variant);
    }
    return registry;
}

} // namespace

int main() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    strix::ExecutionBindingRequest binding_request;
    binding_request.logical_rows = 1;
    binding_request.request_concurrency = 1;
    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;
    const auto registry_a = make_registry(model, binding_request, "decode.");
    const auto registry_b = make_registry(model, binding_request, "prefill.");
    const auto decode = strix::compile_serial_execution(model, registry_a, binding_request, buffer_request, 256);
    const auto prefill = strix::compile_serial_execution(model, registry_b, binding_request, buffer_request, 256);
    const auto wide_arena = strix::compile_serial_execution(model, registry_b, binding_request, buffer_request, 512);

    expect(decode.execution_identity != prefill.execution_identity &&
           decode.kernel_plan_identity != prefill.kernel_plan_identity,
           "different kernel bindings give different execution identities");
    strix::validate_sibling_serial_executions(prefill, decode);
    strix::validate_sibling_serial_executions(decode, prefill);
    rejects([&] { strix::validate_sibling_serial_executions(decode, decode); },
        "an execution is not its own sibling");
    rejects([&] { strix::validate_sibling_serial_executions(wide_arena, decode); },
        "a different arena layout is not a sibling");
    auto extra_step = prefill;
    extra_step.execution.steps.pop_back();
    rejects([&] { strix::validate_sibling_serial_executions(extra_step, decode); },
        "a different step list is not a sibling");
    auto other_io = prefill;
    for (auto& step : other_io.io.steps)
        if (!step.writes.empty()) { step.writes.pop_back(); break; }
    rejects([&] { strix::validate_sibling_serial_executions(other_io, decode); },
        "different buffer IO is not a sibling");
    auto other_state = prefill;
    for (auto& step : other_state.execution.steps)
        if (step.writes_state != strix::PersistentState::none) { step.writes_state = strix::PersistentState::none; break; }
    rejects([&] { strix::validate_sibling_serial_executions(other_state, decode); },
        "different persistent state access is not a sibling");

    // Runtime fixtures (shape-only; no device memory is touched).
    strix::ModelManifest manifest;
    manifest.architecture = "fixture";
    strix::ManifestTensor tensor;
    tensor.name = "token_embd.weight";
    tensor.dimensions = {64, 128};
    tensor.ggml_type = 1;
    tensor.byte_size = 16'384;
    tensor.absolute_offset = 4096;
    tensor.role = strix::TensorRole::embedding;
    tensor.placement.kind = strix::PlacementKind::local;
    manifest.tensors.push_back(tensor);
    manifest.total_tensor_bytes = tensor.byte_size;
    manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
    strix::ModelRuntimeBinding weights;
    weights.manifest_layout_identity = manifest.layout_identity;
    weights.model_content_identity.bytes[0] = 0x5a;
    weights.tensors.push_back(strix::RuntimeTensorBinding{0, strix::RuntimeTensorStorage::resident,
        static_cast<std::uintptr_t>(0x60000000ULL), tensor.byte_size, 7, 0});
    strix::SerialSessionStateLayout state_layout;
    state_layout.context_tokens = 32;
    state_layout.device_alignment = 256;
    state_layout.host_alignment = 64;
    state_layout.device_total_bytes = 4096;
    state_layout.host_total_bytes = 64;
    state_layout.layout_identity = 0x5678ULL;
    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity = state_layout.layout_identity;
    state_memory.device_base = static_cast<std::uintptr_t>(0x71000000ULL);
    state_memory.device_bytes = state_layout.device_total_bytes;
    state_memory.device_allocation_generation = 5;
    state_memory.host_base = static_cast<std::uintptr_t>(0x72000000ULL);
    state_memory.host_bytes = state_layout.host_total_bytes;
    state_memory.host_allocation_generation = 9;
    strix::ExecutionTensorAccessPlan tensor_access;
    tensor_access.manifest_layout_identity = manifest.layout_identity;
    tensor_access.plan_identity = 0x4444ULL;

    strix::ExecutionMemoryBinding decode_scratch;
    decode_scratch.execution_identity = decode.execution_identity;
    decode_scratch.arena_base = static_cast<std::uintptr_t>(0x70000000ULL);
    decode_scratch.arena_bytes = decode.arena.total_bytes;
    decode_scratch.allocation_generation = 3;
    const auto prefill_scratch = strix::sibling_scratch_binding(prefill, decode, decode_scratch);
    expect(prefill_scratch.execution_identity == prefill.execution_identity &&
           prefill_scratch.arena_base == decode_scratch.arena_base &&
           prefill_scratch.arena_bytes == decode_scratch.arena_bytes &&
           prefill_scratch.allocation_generation == decode_scratch.allocation_generation,
           "the sibling scratch is the same arena allocation under the sibling identity");
    rejects([&] { (void)strix::sibling_scratch_binding(wide_arena, decode, decode_scratch); },
        "a non-sibling cannot view the scratch arena");
    auto foreign_scratch = decode_scratch; foreign_scratch.execution_identity = prefill.execution_identity;
    rejects([&] { (void)strix::sibling_scratch_binding(prefill, decode, foreign_scratch); },
        "the owner scratch must belong to the owner execution");

    strix::ExecutionRuntimeResources decode_resources;
    decode_resources.scratch = &decode_scratch;
    decode_resources.model_manifest = &manifest;
    decode_resources.model_weights = &weights;
    decode_resources.tensor_access = &tensor_access;
    decode_resources.session_state_layout = &state_layout;
    decode_resources.session_state_memory = &state_memory;
    auto prefill_resources = decode_resources;
    prefill_resources.scratch = &prefill_scratch;
    const auto decode_ids = strix::build_runtime_execution_identities(decode.execution_identity, decode.arena, decode_resources);
    const auto prefill_ids = strix::build_runtime_execution_identities(prefill.execution_identity, prefill.arena, prefill_resources);
    expect(decode_ids.compatibility_identity != prefill_ids.compatibility_identity,
           "sibling executors have distinct compatibility identities");

    strix::SerialSessionContext session;
    session.execution_identity = decode.execution_identity;
    session.state_layout_identity = state_layout.layout_identity;
    session.runtime_compatibility_identity = decode_ids.compatibility_identity;
    const strix::SessionExecutor to_decode{&decode, &decode_resources};
    const strix::SessionExecutor to_prefill{&prefill, &prefill_resources};

    strix::hand_off_serial_session(session, to_decode, to_prefill);
    expect(session.execution_identity == prefill.execution_identity &&
           session.runtime_compatibility_identity == prefill_ids.compatibility_identity &&
           session.next_position == 0 && session.health == strix::SerialSessionHealth::ready,
           "handoff moves only the executor identities");
    rejects([&] { strix::hand_off_serial_session(session, to_decode, to_prefill); },
        "the source executor must own the session");

    // A session with a recorded prompt keeps its lineage (and the prefill producer identity).
    session.arithmetic_lineage = strix::prepare_serial_row_lineage({}, 0, prefill.execution_identity, prefill.kernel_plan_identity);
    session.next_position = session.committed_tokens = 1;
    const auto lineage = session.arithmetic_lineage;
    strix::hand_off_serial_session(session, to_prefill, to_decode);
    expect(session.execution_identity == decode.execution_identity &&
           session.runtime_compatibility_identity == decode_ids.compatibility_identity &&
           session.arithmetic_lineage.intervals == lineage.intervals &&
           session.arithmetic_lineage.history_identity == lineage.history_identity &&
           session.arithmetic_lineage.intervals[0].numerical_plan_identity == prefill.execution_identity,
           "handoff back to decode keeps the prefill producer identity in the lineage");

    const auto unchanged = [&](const strix::SerialSessionContext& before, const char* message) {
        expect(session.execution_identity == before.execution_identity &&
               session.runtime_compatibility_identity == before.runtime_compatibility_identity, message);
    };
    auto other_access = tensor_access;
    auto other_resources = prefill_resources;
    other_resources.tensor_access = &other_access;
    auto before = session;
    rejects([&] { strix::hand_off_serial_session(session, to_decode, {&prefill, &other_resources}); },
        "executors must share one tensor access binding");
    unchanged(before, "a rejected handoff leaves the session unchanged");
    auto other_state_memory = state_memory;
    other_resources = prefill_resources;
    other_resources.session_state_memory = &other_state_memory;
    rejects([&] { strix::hand_off_serial_session(session, to_decode, {&prefill, &other_resources}); },
        "executors must share one state memory binding");
    rejects([&] { strix::hand_off_serial_session(session, to_decode, {&wide_arena, &prefill_resources}); },
        "the target executor must be a sibling");
    other_resources = prefill_resources;
    other_resources.scratch = &decode_scratch;
    rejects([&] { strix::hand_off_serial_session(session, to_decode, {&prefill, &other_resources}); },
        "the target scratch must be bound to the target execution");
    unchanged(before, "a rejected target binding leaves the session unchanged");
    rejects([&] { strix::hand_off_serial_session(session, to_decode, {nullptr, &prefill_resources}); },
        "both executors are required");
    auto tainted = session; tainted.arithmetic_lineage.tainted = true;
    rejects([&] { strix::hand_off_serial_session(tainted, to_decode, to_prefill); },
        "a tainted lineage cannot be handed off");
    auto poisoned = session; poisoned.health = strix::SerialSessionHealth::poisoned;
    rejects([&] { strix::hand_off_serial_session(poisoned, to_decode, to_prefill); },
        "a poisoned session cannot be handed off");
    auto stale = session; ++stale.runtime_compatibility_identity;
    rejects([&] { strix::hand_off_serial_session(stale, to_decode, to_prefill); },
        "a session with another compatibility identity is not owned by the source executor");

    std::cout << "session executor handoff failures=" << failures << '\n';
    return failures ? 1 : 0;
}
