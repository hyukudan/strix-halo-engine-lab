#include "strix/prepared_execution.hpp"
#include "strix/prepared_serial_session.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/model_tensor_index.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template <typename Function> bool rejects(Function&& function) {
    try { function(); } catch (const std::exception&) { return true; }
    return false;
}
strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv | PersistentState::recurrent_matrix | PersistentState::convolution_history |
           PersistentState::sparse_index | PersistentState::ngram_history | PersistentState::position_history;
}
struct Storage {
    std::vector<std::byte> scratch_bytes, device_bytes, host_bytes;
    std::array<float, 1> tensor_bytes{1.0F};
    strix::ExecutionMemoryBinding scratch;
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding weights;
    strix::ExecutionTensorAccessPlan access;
    strix::SerialSessionStateLayout layout;
    strix::SessionStateMemoryBinding state;
    strix::ExecutionRuntimeResources resources;
};
std::uintptr_t allocate_aligned(std::vector<std::byte>& storage, std::uint64_t bytes, std::uint64_t alignment) {
    require(alignment && bytes && bytes <= std::numeric_limits<std::size_t>::max() - (alignment - 1U),
            "fixture aligned allocation size is invalid");
    storage.resize(static_cast<std::size_t>(bytes + alignment - 1U));
    void* pointer = storage.data();
    auto remaining = storage.size();
    require(std::align(static_cast<std::size_t>(alignment), static_cast<std::size_t>(bytes), pointer, remaining) != nullptr,
            "fixture cannot align the retained actual allocation");
    return reinterpret_cast<std::uintptr_t>(pointer);
}

// Actual host allocations and the existing compiler/validators; launcher stubs
// inspect metadata only. Tiny synthetic resource pools are not model evidence.
struct Fixture {
    std::shared_ptr<Storage> owner = std::make_shared<Storage>();
    strix::CompiledSerialExecution compiled;
    strix::EagerExecutionRequest request;
    Fixture() {
        const auto model = strix::ModelContract::qwen38_flash_next();
        strix::kernels::KernelRegistry variants;
        strix::ExecutionBindingRequest selection;
        std::set<std::string> added;
        for (const auto& step : strix::build_serial_decode_execution_plan(model).steps) {
            const auto family = strix::operator_family_for(step.op);
            if (!family) continue;
            const auto shape = strix::execution_shape_for(model, step.op, selection);
            const auto key = "fixture." + std::to_string(static_cast<unsigned>(*family)) + "." +
                std::to_string(shape.hidden_size) + "." + std::to_string(shape.auxiliary_size);
            if (!added.insert(key).second) continue;
            strix::kernels::KernelVariant variant;
            variant.id = key; variant.family = *family; variant.phase = strix::kernels::ExecutionPhase::decode;
            variant.shape = shape; variant.resources.workgroup_threads = 32;
            variant.numerics = {"bf16", "fp32", "bf16", true, true};
            variant.qualification = strix::kernels::QualificationStatus::oracle_matched;
            variant.graph_safe = true; variant.input_layout = "rows"; variant.output_layout = "rows";
            variant.compiler_identity = "synthetic-host"; variants.add(variant);
        }
        compiled = strix::compile_serial_execution(model, variants, selection, {});
        const auto scratch_base = allocate_aligned(owner->scratch_bytes, compiled.arena.total_bytes, compiled.arena.alignment);
        owner->scratch = {compiled.execution_identity, scratch_base,
                          compiled.arena.total_bytes, 3};
        owner->manifest.architecture = "host-fixture";
        const auto requirements = strix::build_serial_execution_tensor_requirements(compiled.execution);
        std::set<std::string> tensor_names;
        for (const auto& requirement : requirements) {
            const auto& selector = requirement.selector;
            std::string needle(selector.name_contains);
            if (needle.empty() && !selector.name_contains_any.empty()) needle = selector.name_contains_any.front();
            const auto name = "fixture." + std::to_string(static_cast<unsigned>(*selector.role)) + "." +
                (selector.layer ? std::to_string(*selector.layer) : "global") + "." + needle;
            if (!tensor_names.insert(name).second) continue;
            strix::ManifestTensor tensor;
            tensor.name = name; tensor.dimensions = {1, 1}; tensor.ggml_type = 0; tensor.byte_size = 4;
            tensor.role = *selector.role; tensor.layer = selector.layer;
            tensor.placement.kind = strix::PlacementKind::local;
            owner->manifest.tensors.push_back(tensor);
            owner->manifest.total_tensor_bytes += 4;
        }
        owner->manifest.layout_identity = strix::model_manifest_layout_identity(owner->manifest);
        owner->weights.manifest_layout_identity = owner->manifest.layout_identity;
        owner->weights.model_content_identity.bytes[0] = 1;
        for (std::size_t index = 0; index < owner->manifest.tensors.size(); ++index)
            owner->weights.tensors.push_back({static_cast<std::uint32_t>(index), strix::RuntimeTensorStorage::resident,
                reinterpret_cast<std::uintptr_t>(owner->tensor_bytes.data()), 4, 7, 0});
        const auto tensor_index = strix::build_model_tensor_index(owner->manifest);
        owner->access = strix::compile_execution_tensor_access_plan(
            compiled.execution, owner->manifest, tensor_index, requirements);
        owner->layout.context_tokens = 16; owner->layout.layout_identity = 0x5678;
        owner->layout.device_total_bytes = 5U * owner->layout.device_alignment;
        owner->layout.host_total_bytes = 2U * owner->layout.host_alignment;
        const std::array components{strix::SessionStateComponent::attention_kv, strix::SessionStateComponent::recurrent_matrix,
            strix::SessionStateComponent::gdn_convolution, strix::SessionStateComponent::ple_convolution,
            strix::SessionStateComponent::sparse_index, strix::SessionStateComponent::ngram_history,
            strix::SessionStateComponent::position_history};
        const std::array flags{strix::PersistentState::attention_kv, strix::PersistentState::recurrent_matrix,
            strix::PersistentState::convolution_history, strix::PersistentState::convolution_history,
            strix::PersistentState::sparse_index, strix::PersistentState::ngram_history, strix::PersistentState::position_history};
        for (std::size_t index = 0; index < components.size(); ++index) {
            const bool host = index >= 5U;
            const auto offset = static_cast<std::uint64_t>(host ? index - 5U : index) *
                (host ? owner->layout.host_alignment : owner->layout.device_alignment);
            owner->layout.allocations.push_back({components[index], flags[index], host ? strix::SessionStateStorage::host :
                strix::SessionStateStorage::device, offset, 16U, false});
        }
        const auto device_base = allocate_aligned(owner->device_bytes, owner->layout.device_total_bytes, owner->layout.device_alignment);
        const auto host_base = allocate_aligned(owner->host_bytes, owner->layout.host_total_bytes, owner->layout.host_alignment);
        owner->state = {owner->layout.layout_identity, device_base, owner->layout.device_total_bytes, 5,
                        host_base, owner->layout.host_total_bytes, 9};
        require(owner->scratch.arena_base % compiled.arena.alignment == 0U &&
                owner->state.device_base % owner->layout.device_alignment == 0U &&
                owner->state.host_base % owner->layout.host_alignment == 0U,
                "fixture bound pools do not honor declared alignment");
        owner->resources = {&owner->scratch, &owner->manifest, &owner->weights, &owner->access,
                            &owner->layout, &owner->state, nullptr, nullptr, nullptr};
        request.state.valid = all_state(); request.input_token_id = 3; request.position = 2;
        request.state.generations = {7, 8, 9, 10, 11, 12};
    }
};

struct Trace final : strix::EagerExecutionDispatcher {
    std::vector<strix::ExecutionDispatchContext> calls;
    std::optional<std::uint32_t> fail_step;
    void record(const strix::ExecutionDispatchContext& context) {
        if (context.prepared_tensor_views) {
            require(context.resources != nullptr, "prepared lookup cache has no resources");
            for (const auto& access : context.resources->tensor_access->accesses) {
                if (access.step_id != context.step_id) continue;
                const auto& cached = strix::resolve_dispatch_tensor_access(context, access.binding_slot);
                require(cached.tensor_indices == access.tensor_indices && cached.op == context.op,
                        "prepared dispatch access differs from its admitted source");
                for (const auto index : cached.tensor_indices) {
                    const auto& actual = context.resources->model_weights->tensors[index];
                    const auto& indexed = strix::resolve_dispatch_tensor(context, index);
                    const auto& named = strix::resolve_dispatch_tensor(context,
                        context.resources->model_manifest->tensors[index].name);
                    require(indexed.address == actual.address && indexed.bytes == actual.bytes &&
                        indexed.allocation_generation == actual.allocation_generation && &indexed == &named,
                        "prepared tensor index/name lookup changed the admitted physical view");
                }
            }
        }
        calls.push_back(context);
        if (fail_step == context.step_id) throw std::runtime_error("injected dispatch failure");
    }
    void dispatch_kernel(const strix::ExecutionDispatchContext& context) override {
        require(context.kernel_id && !context.kernel_id->empty(), "kernel context lost selected id"); record(context);
    }
    void dispatch_runtime_step(const strix::ExecutionDispatchContext& context) override {
        require(!context.kernel_id, "runtime context gained a kernel id"); record(context);
    }
};
template <typename Left, typename Right> bool same_span(const Left& left, const Right& right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}
void same_calls(const Trace& reference, const Trace& prepared) {
    require(reference.calls.size() == prepared.calls.size(), "dispatch call counts differ");
    for (std::size_t index = 0; index < reference.calls.size(); ++index) {
        const auto& a = reference.calls[index]; const auto& b = prepared.calls[index];
        require(a.execution_identity == b.execution_identity && a.step_id == b.step_id && a.op == b.op &&
            a.layer == b.layer && a.kernel_id == b.kernel_id && same_span(a.reads, b.reads) && same_span(a.writes, b.writes) &&
            same_span(a.temporaries, b.temporaries) && a.memory == b.memory && a.resources == b.resources &&
            a.session_state_layout == b.session_state_layout && a.session_state_memory == b.session_state_memory &&
            a.reads_state == b.reads_state && a.writes_state == b.writes_state && a.logical_rows == b.logical_rows &&
            a.request_concurrency == b.request_concurrency && a.input_token_id == b.input_token_id &&
            same_span(a.input_tokens, b.input_tokens) && a.position == b.position, "prepared context changed a dispatch field");
        require(a.arena != b.arena && a.arena->total_bytes == b.arena->total_bytes &&
                a.arena->allocations.size() == b.arena->allocations.size(), "prepared arena is not an independent copy");
        for (std::size_t allocation = 0; allocation < a.arena->allocations.size(); ++allocation) {
            const auto& x = a.arena->allocations[allocation]; const auto& y = b.arena->allocations[allocation];
            require(x.use == y.use && x.offset == y.offset && x.bytes == y.bytes && x.first_step == y.first_step &&
                    x.last_step == y.last_step, "prepared arena layout changed");
        }
    }
}
void equivalence() {
    Fixture fixture;
    strix::PreparedEagerExecution prepared(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
    require(&prepared.compiled() != &fixture.compiled &&
            prepared.compiled().execution_identity == fixture.compiled.execution_identity,
            "prepared compiled accessor does not expose its independent const copy");
    const auto ple = std::find_if(fixture.compiled.execution.steps.begin(), fixture.compiled.execution.steps.end(),
        [](const auto& step) { return step.op == strix::SerialOp::ple; });
    require(ple != fixture.compiled.execution.steps.end() && prepared.ple_step_id() == ple->id,
            "prepared PLE step cache differs from the first original PLE step");
    prepared.validate_dynamic(fixture.owner->resources, fixture.request);
    require(prepared.compatibility_identity() == strix::build_runtime_execution_identities(
                fixture.compiled.execution_identity, fixture.compiled.arena, fixture.owner->resources).compatibility_identity,
            "prepared compatibility accessor differs from its validated resource identity");
    Trace baseline, candidate;
    const auto reference = strix::execute_compiled_serial_eager(fixture.compiled, fixture.owner->resources, fixture.request, baseline);
    const auto result = prepared.run(fixture.owner->resources, fixture.request, candidate);
    same_calls(baseline, candidate);
    require(reference.completed_steps == result.completed_steps && reference.kernel_steps == result.kernel_steps &&
        reference.runtime_steps == result.runtime_steps && reference.state.valid == result.state.valid &&
        reference.state.generations == result.state.generations && same_span(reference.completion_order, result.completion_order),
        "prepared counts/state/order differ from original eager execution");
    const auto order = result.completion_order;
    auto next = fixture.request; next.position = 3; next.input_token_id = 5; next.state = result.state;
    const std::array<std::uint32_t, 1> next_tokens{6}; next.input_tokens = next_tokens;
    Trace repeated;
    const auto repeat = prepared.run(fixture.owner->resources, next, repeated);
    Trace repeated_baseline;
    const auto next_reference = strix::execute_compiled_serial_eager(fixture.compiled, fixture.owner->resources, next, repeated_baseline);
    same_calls(repeated_baseline, repeated);
    require(repeat.state.generations == next_reference.state.generations,
            "prepared repeated request did not preserve evolving state");
    require(repeat.completion_order.data() == order.data() && same_span(repeat.completion_order, order),
            "prepared completion span is not stable across runs");
    fixture.compiled.execution.steps.clear(); fixture.compiled.bindings.bindings.clear(); fixture.compiled.io.steps.clear();
    fixture.compiled.arena.allocations.clear(); fixture.compiled.arena.total_bytes = 0;
    Trace after_external_mutation;
    const auto independent = prepared.run(fixture.owner->resources, next, after_external_mutation);
    require(independent.completed_steps == reference.completed_steps && same_span(independent.completion_order, order),
            "caller mutation changed the prepared immutable plan");
}
void dispatch_failure() {
    Fixture fixture;
    strix::PreparedEagerExecution prepared(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
    for (const auto failing : {0U, 1U, static_cast<unsigned>(fixture.compiled.execution.steps.size() / 2U)}) {
        Trace baseline, candidate; baseline.fail_step = failing; candidate.fail_step = failing;
        const auto before = fixture.request.state;
        require(rejects([&] { static_cast<void>(strix::execute_compiled_serial_eager(
            fixture.compiled, fixture.owner->resources, fixture.request, baseline)); }) &&
            rejects([&] { static_cast<void>(prepared.run(fixture.owner->resources, fixture.request, candidate)); }),
            "dispatch failure was swallowed");
        same_calls(baseline, candidate);
        require(fixture.request.state.valid == before.valid && fixture.request.state.generations == before.generations,
                "failed dispatch published request state");
    }
}
void dynamic_rejection() {
    Fixture fixture;
    strix::PreparedEagerExecution prepared(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
    const auto rejected = [&](const strix::ExecutionRuntimeResources& resources, const strix::EagerExecutionRequest& request) {
        Trace unused;
        require(rejects([&] { prepared.validate_dynamic(resources, request); }),
                "dynamic-only preflight admitted the rejected request");
        require(rejects([&] { static_cast<void>(prepared.run(resources, request, unused)); }) && unused.calls.empty(),
                "dynamic admission dispatched work before rejection");
    };
    auto request = fixture.request; ++request.logical_rows; rejected(fixture.owner->resources, request);
    request = fixture.request; ++request.request_concurrency; rejected(fixture.owner->resources, request);
    request = fixture.request; request.input_token_id = static_cast<std::uint32_t>(fixture.compiled.vocab_size);
    rejected(fixture.owner->resources, request);
    std::array<std::uint32_t, 2> tokens{3, 4};
    request = fixture.request; request.input_tokens = tokens; rejected(fixture.owner->resources, request);
    request = fixture.request; tokens[0] = static_cast<std::uint32_t>(fixture.compiled.vocab_size);
    request.input_tokens = std::span<const std::uint32_t>(tokens.data(), 1); rejected(fixture.owner->resources, request);
    request = fixture.request; request.position = fixture.owner->layout.context_tokens; rejected(fixture.owner->resources, request);
    request = fixture.request; request.state.valid = strix::PersistentState::none; rejected(fixture.owner->resources, request);
    for (std::size_t index = 0; index < fixture.request.state.generations.size(); ++index) {
        request = fixture.request; request.state.generations[index] = std::numeric_limits<std::uint64_t>::max();
        rejected(fixture.owner->resources, request);
    }
    auto scratch = fixture.owner->scratch; ++scratch.allocation_generation;
    auto resources = fixture.owner->resources; resources.scratch = &scratch; rejected(resources, fixture.request);
    scratch = fixture.owner->scratch; ++scratch.arena_base; rejected(resources, fixture.request);
    auto state = fixture.owner->state; ++state.device_allocation_generation;
    resources = fixture.owner->resources; resources.session_state_memory = &state; rejected(resources, fixture.request);
    state = fixture.owner->state; ++state.host_allocation_generation; rejected(resources, fixture.request);
    auto weights = fixture.owner->weights; ++weights.tensors[0].allocation_generation;
    resources = fixture.owner->resources; resources.model_weights = &weights; rejected(resources, fixture.request);
    weights = fixture.owner->weights; ++weights.model_content_identity.bytes[0]; rejected(resources, fixture.request);
    auto access = fixture.owner->access; ++access.plan_identity;
    resources = fixture.owner->resources; resources.tensor_access = &access; rejected(resources, fixture.request);
    auto layout = fixture.owner->layout; layout.allocations.pop_back();
    resources = fixture.owner->resources; resources.session_state_layout = &layout; rejected(resources, fixture.request);
    resources = fixture.owner->resources; resources.model_weights = nullptr; rejected(resources, fixture.request);
    auto manifest = fixture.owner->manifest; manifest.tensors[0].name += ".changed";
    resources = fixture.owner->resources; resources.model_manifest = &manifest; rejected(resources, fixture.request);
    manifest = fixture.owner->manifest; manifest.tensors[0].dimensions[0] = 2; rejected(resources, fixture.request);
    access = fixture.owner->access; access.accesses[0].tensor_indices[0] = 1;
    resources = fixture.owner->resources; resources.tensor_access = &access; rejected(resources, fixture.request);

    // Mutations under the SAME source handles and stale claimed identities must
    // also be rejected: identity hashes do not cover these structural fields.
    const auto saved_name = fixture.owner->manifest.tensors[0].name;
    fixture.owner->manifest.tensors[0].name += ".mutated";
    rejected(fixture.owner->resources, fixture.request);
    fixture.owner->manifest.tensors[0].name = saved_name;
    const auto saved_index = fixture.owner->access.accesses[0].tensor_indices[0];
    fixture.owner->access.accesses[0].tensor_indices[0] = saved_index + 1U;
    rejected(fixture.owner->resources, fixture.request);
    fixture.owner->access.accesses[0].tensor_indices[0] = saved_index;

}
void preparation_rejection() {
    Fixture fixture;
    auto compiled = fixture.compiled; compiled.execution.steps[0].dependencies = {1};
    require(rejects([&] { strix::PreparedEagerExecution invalid(compiled, fixture.owner->resources, fixture.request, fixture.owner); }),
            "preparation admitted a future dependency");
    compiled = fixture.compiled; ++compiled.io.steps[0].step_id;
    require(rejects([&] { strix::PreparedEagerExecution invalid(compiled, fixture.owner->resources, fixture.request, fixture.owner); }),
            "preparation admitted stale IO identity");
    compiled = fixture.compiled; compiled.arena.allocations.clear();
    require(rejects([&] { strix::PreparedEagerExecution invalid(compiled, fixture.owner->resources, fixture.request, fixture.owner); }),
            "preparation admitted missing arena generations");
    auto request = fixture.request; ++request.logical_rows;
    require(rejects([&] { strix::PreparedEagerExecution invalid(fixture.compiled, fixture.owner->resources, request, fixture.owner); }),
            "preparation admitted mismatched prototype shape");
    require(rejects([&] { strix::PreparedEagerExecution invalid(fixture.compiled, fixture.owner->resources, fixture.request, {}); }),
            "preparation admitted absent retained ownership");
}
void retained_owner() {
    Fixture fixture;
    std::weak_ptr<Storage> lifetime = fixture.owner;
    const auto* resources = &fixture.owner->resources;
    {
        strix::PreparedEagerExecution prepared(fixture.compiled, *resources, fixture.request, fixture.owner);
        fixture.owner.reset(); require(!lifetime.expired(), "actual prepared owners were not retained");
        Trace dispatcher;
        require(prepared.run(*resources, fixture.request, dispatcher).completed_steps == fixture.compiled.execution.steps.size(),
                "retained owner cannot supply the prepared resources");
    }
    require(lifetime.expired(), "prepared destruction leaked retained owners");
}

strix::SerialSessionContext initial_session(const Fixture& fixture, const strix::PreparedEagerExecution& prepared) {
    strix::SerialSessionContext session;
    session.execution_identity = fixture.compiled.execution_identity;
    session.state_layout_identity = fixture.owner->layout.layout_identity;
    session.runtime_compatibility_identity = prepared.compatibility_identity();
    session.state = fixture.request.state;
    return session;
}
void same_session(const strix::SerialSessionContext& a, const strix::SerialSessionContext& b) {
    require(a.execution_identity == b.execution_identity && a.state_layout_identity == b.state_layout_identity &&
        a.runtime_compatibility_identity == b.runtime_compatibility_identity && a.next_position == b.next_position &&
        a.committed_tokens == b.committed_tokens && a.health == b.health && a.state.valid == b.state.valid &&
        a.state.generations == b.state.generations && a.arithmetic_lineage.intervals == b.arithmetic_lineage.intervals &&
        a.arithmetic_lineage.history_identity == b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted == b.arithmetic_lineage.tainted, "prepared session lifecycle differs from legacy");
}
void session_adapter() {
    Fixture fixture;
    strix::PreparedEagerExecution prepared(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
    auto reference_session = initial_session(fixture, prepared);
    auto candidate_session = reference_session;
    for (const auto token : {3U, 4U}) {
        Trace reference_dispatcher, candidate_dispatcher;
        const auto before = candidate_session;
        bool reference_hook = false, candidate_hook = false;
        const auto reference = strix::execute_serial_token(fixture.compiled, reference_session,
            fixture.owner->resources, token, reference_dispatcher, [&] {
                same_session(reference_session, before); reference_hook = true;
            });
        const auto candidate = strix::execute_prepared_serial_token(prepared, candidate_session,
            fixture.owner->resources, token, candidate_dispatcher, [&] {
                same_session(candidate_session, before); candidate_hook = true;
            });
        require(reference_hook && candidate_hook, "session completion hook was not called");
        same_calls(reference_dispatcher, candidate_dispatcher);
        same_session(reference_session, candidate_session);
        require(reference.completed_steps == candidate.completed_steps &&
                same_span(reference.completion_order, candidate.completion_order), "session submission results differ");
        require(strix::arithmetic_lineage_valid(candidate_session.arithmetic_lineage, candidate_session.next_position),
                "prepared session failed to publish contiguous arithmetic lineage");
    }
    // Rejection occurs before dispatch and leaves a usable session untouched.
    {
        auto session = initial_session(fixture, prepared); const auto before = session;
        auto scratch = fixture.owner->scratch; ++scratch.allocation_generation;
        auto resources = fixture.owner->resources; resources.scratch = &scratch;
        Trace dispatcher;
        require(rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(
            prepared, session, resources, 3U, dispatcher)); }) && dispatcher.calls.empty(),
            "prepared session dispatched a changed allocation generation");
        same_session(session, before);
    }
    for (const bool fail_completion : {false, true}) {
        auto reference = initial_session(fixture, prepared); auto candidate = reference;
        const auto before = candidate;
        Trace a, b;
        if (!fail_completion) { a.fail_step = 1U; b.fail_step = 1U; }
        const auto completion = [&] {
            if (fail_completion) throw std::runtime_error("injected completion failure");
        };
        require(rejects([&] { static_cast<void>(strix::execute_serial_token(
            fixture.compiled, reference, fixture.owner->resources, 3U, a, completion)); }) &&
            rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(
            prepared, candidate, fixture.owner->resources, 3U, b, completion)); }),
            "session dispatch/completion failure was swallowed");
        same_calls(a, b); same_session(reference, candidate);
        auto expected = before; expected.health = strix::SerialSessionHealth::poisoned;
        expected.arithmetic_lineage.tainted = true;
        same_session(candidate, expected);
        Trace unused;
        require(rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(
            prepared, candidate, fixture.owner->resources, 3U, unused)); }) && unused.calls.empty(),
            "prepared session reused poisoned state");
    }
}
} // namespace

int main() try {
    static_assert(!std::is_copy_constructible_v<strix::PreparedEagerExecution> &&
                  !std::is_move_constructible_v<strix::PreparedEagerExecution>);
    static_assert(std::is_same_v<decltype(std::declval<const strix::PreparedEagerExecution&>().compiled()),
                                 const strix::CompiledSerialExecution&>);
    equivalence(); dispatch_failure(); dynamic_rejection(); preparation_rejection(); retained_owner(); session_adapter();
    std::cout << "prepared eager host contract fixtures passed; no native work\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
