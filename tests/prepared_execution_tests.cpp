#include "strix/prepared_execution.hpp"
#include "strix/prepared_arena_views.hpp"
#include "strix/moe_fork_join_arena.hpp"
#include "strix/generation_output_dispatch.hpp"
#include "strix/prepared_serial_session.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/ple_lookup.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
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
// Observes the existing protocol without adding a production admission hook.
// Metadata and implementation identity stay immutable while a call is active.
struct CountingPle final : strix::PleLookupBackend {
    enum class StartFailure { none, throws, zero };
    mutable std::uint64_t identity_calls = 0;
    std::uint64_t start_calls = 0, finish_calls = 0, cancel_calls = 0;
    StartFailure start_failure = StartFailure::none;
    bool fail_finish = false, active = false;
    std::uint64_t pending_bytes = 0;

    std::uint64_t implementation_identity() const noexcept override {
        ++identity_calls;
        return 0x505245504c454331ULL;
    }
    strix::PleLookupHandle start(const strix::PleLookupRequest& request) override {
        ++start_calls;
        require(!active && request.table_manifest && request.table_runtime &&
                request.table_manifest->ggml_type == 30U && request.row_ids.size() == 1U &&
                request.row_bytes == 4U && request.row_ids.front() == 0U &&
                request.table_runtime->bytes >= request.row_bytes,
                "counting PLE received an invalid retained table request");
        if (start_failure == StartFailure::throws) throw std::runtime_error("injected PLE start failure");
        if (start_failure == StartFailure::zero) return 0;
        pending_bytes = request.row_bytes;
        active = true;
        return 17;
    }
    void finish(strix::PleLookupHandle handle, std::span<std::byte> destination) override {
        ++finish_calls;
        require(active && handle == 17U && destination.size() == pending_bytes,
                "counting PLE finish lost its active request");
        if (fail_finish) throw std::runtime_error("injected PLE finish failure");
        std::fill(destination.begin(), destination.end(), std::byte{0x2a});
        active = false;
    }
    void cancel(strix::PleLookupHandle handle) noexcept override {
        ++cancel_calls;
        if (handle == 17U) active = false;
    }
    void clear_counts() {
        require(!active, "counting PLE counters reset with live work");
        identity_calls = start_calls = finish_calls = cancel_calls = 0;
    }
};
struct Storage {
    std::vector<std::byte> scratch_bytes, device_bytes, host_bytes;
    std::array<float, 1> tensor_bytes{1.0F};
    strix::ExecutionMemoryBinding scratch;
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding weights;
    strix::ExecutionTensorAccessPlan access;
    strix::SerialSessionStateLayout layout;
    strix::SessionStateMemoryBinding state;
    CountingPle ple_backend;
    strix::PleHashContract ple_contract;
    strix::PleLookupPrefetchState ple_prefetch;
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

struct PleFixture : Fixture {
    PleFixture() {
        auto& data = *owner;
        data.ple_contract.ngram_size = 2;
        data.ple_contract.heads_per_ngram = 1;
        data.ple_contract.head_count = 1;
        data.ple_contract.head_dim = 2;
        data.ple_contract.row_bytes = 4;
        data.ple_contract.eos_token_id = 0;
        data.ple_contract.multipliers[0] = 3;
        data.ple_contract.multipliers[1] = 5;
        data.ple_contract.head_vocab_sizes[0] = 1;
        data.ple_contract.minimum_table_rows = 1;
        const auto ple = std::find_if(compiled.execution.steps.begin(), compiled.execution.steps.end(),
            [](const auto& step) { return step.op == strix::SerialOp::ple; });
        require(ple != compiled.execution.steps.end(), "counting PLE fixture has no PLE step");
        const auto* access = strix::find_execution_tensor_access(data.access, ple->id, 0);
        require(access && access->tensor_indices.size() == 1U, "counting PLE table is not unique");
        auto& table = data.manifest.tensors.at(access->tensor_indices.front());
        table.dimensions = {2, 1}; table.ggml_type = 30; // One actual four-byte BF16 row.
        require(table.byte_size == 4U && data.tensor_bytes.size() * sizeof(float) >= table.byte_size,
                "counting PLE table backing is undersized");
        data.manifest.layout_identity = strix::model_manifest_layout_identity(data.manifest);
        data.weights.manifest_layout_identity = data.manifest.layout_identity;
        data.access = strix::compile_execution_tensor_access_plan(compiled.execution, data.manifest,
            strix::build_model_tensor_index(data.manifest),
            strix::build_serial_execution_tensor_requirements(compiled.execution));
        data.layout.context_tokens = 128; // Repeated accepted tokens and between-call mutation cases.
        data.resources.ple_lookup = &data.ple_backend;
        data.resources.ple_hash_contract = &data.ple_contract;
        data.resources.ple_prefetch = &data.ple_prefetch;
        request.position = 0;
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
struct PleTrace final : strix::EagerExecutionDispatcher {
    CountingPle& backend;
    Trace trace;
    bool consume = true;
    std::optional<std::uint64_t> identities_at_first_dispatch;
    explicit PleTrace(CountingPle& lookup) : backend(lookup) {}
    void before() {
        if (!identities_at_first_dispatch) identities_at_first_dispatch = backend.identity_calls;
    }
    void after(const strix::ExecutionDispatchContext& context) const {
        if (consume && context.op == strix::SerialOp::ple && context.resources &&
            context.resources->ple_prefetch && context.resources->ple_prefetch->active) {
            std::array<std::byte, 16> bytes{};
            require(context.resources->ple_prefetch->lookup_bytes == 4U,
                    "counting PLE expected exactly one full row");
            const auto hash = strix::finish_ple_lookup_prefetch(context.input_token_id, context.position,
                backend, *context.resources->ple_prefetch, bytes);
            require(hash.row_count == 1U && hash.context_count == 1U &&
                    bytes.front() == std::byte{0x2a}, "counting PLE finish did not consume its row");
        }
    }
    void dispatch_kernel(const strix::ExecutionDispatchContext& context) override {
        before(); trace.dispatch_kernel(context); after(context);
    }
    void dispatch_runtime_step(const strix::ExecutionDispatchContext& context) override {
        before(); trace.dispatch_runtime_step(context); after(context);
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

strix::EagerExecutionRequest session_request(const strix::SerialSessionContext& session, std::uint32_t token = 3U) {
    strix::EagerExecutionRequest request;
    request.logical_rows = 1; request.request_concurrency = 1;
    request.input_token_id = token; request.position = session.next_position; request.state = session.state;
    return request;
}
struct AdmissionCounts { std::uint64_t validation = 0, prefetch_start = 0; };
AdmissionCounts admission_counts(const strix::PreparedEagerExecution& prepared,
    const strix::ExecutionRuntimeResources& resources, const strix::EagerExecutionRequest& request,
    CountingPle& backend) {
    backend.clear_counts();
    prepared.validate_dynamic(resources, request);
    const auto validation = backend.identity_calls;
    require(validation != 0U && backend.start_calls == 0U, "dynamic admission did not query PLE identity");
    backend.clear_counts();
    strix::PleLookupPrefetchState scratch;
    require(prepared.ple_step_id().has_value(), "counting PLE calibration lost its prepared step");
    strix::start_ple_lookup_prefetch(*resources.ple_hash_contract, *prepared.ple_step_id(),
        request.input_token_id, request.position, *resources.model_manifest, *resources.model_weights,
        *resources.tensor_access, *resources.session_state_layout, *resources.session_state_memory,
        backend, scratch);
    const auto start = backend.identity_calls;
    strix::cancel_ple_lookup_prefetch(backend, scratch);
    require(start != 0U && !scratch.active && !backend.active,
            "counting PLE calibration leaked its independent prefetch");
    backend.clear_counts();
    return {validation, start};
}
void require_poisoned(const strix::SerialSessionContext& actual, const strix::SerialSessionContext& before) {
    auto expected = before;
    expected.health = strix::SerialSessionHealth::poisoned;
    expected.arithmetic_lineage.tainted = true;
    same_session(actual, expected);
}
void public_run_admission() {
    PleFixture fixture;
    auto& data = *fixture.owner;
    strix::PreparedEagerExecution prepared(fixture.compiled, data.resources, fixture.request, fixture.owner);
    const auto counts = admission_counts(prepared, data.resources, fixture.request, data.ple_backend);
    prepared.validate_dynamic(data.resources, fixture.request);
    const auto name = data.manifest.tensors.front().name;
    data.manifest.tensors.front().name += ".between-public-calls";
    PleTrace rejected(data.ple_backend);
    require(rejects([&] { static_cast<void>(prepared.run(data.resources, fixture.request, rejected)); }) &&
            rejected.trace.calls.empty() && data.ple_backend.start_calls == 0U,
            "public run reused an earlier dynamic admission");
    data.manifest.tensors.front().name = name;
    auto request = fixture.request;
    const std::uint32_t* stable_order = nullptr;
    for (unsigned iteration = 0; iteration < 2U; ++iteration) {
        data.ple_backend.clear_counts();
        PleTrace dispatcher(data.ple_backend);
        const auto result = prepared.run(data.resources, request, dispatcher);
        require(dispatcher.identities_at_first_dispatch == counts.validation &&
                data.ple_backend.start_calls == 0U && !data.ple_prefetch.active,
                "public run did not perform exactly one independent full admission");
        if (!stable_order) stable_order = result.completion_order.data();
        require(result.completion_order.data() == stable_order &&
                result.completed_steps == fixture.compiled.execution.steps.size(),
                "public run changed immutable result storage");
        request.state = result.state; ++request.position;
    }
}
void ple_between_call_rejection() {
    PleFixture fixture;
    auto& data = *fixture.owner;
    strix::PreparedEagerExecution prepared(fixture.compiled, data.resources, fixture.request, fixture.owner);
    auto session = initial_session(fixture, prepared);
    {
        PleTrace first(data.ple_backend);
        static_cast<void>(strix::execute_prepared_serial_token(prepared, session, data.resources, 3U, first));
        require(data.ple_backend.finish_calls == 1U && !data.ple_prefetch.active,
                "between-call fixture did not complete its first PLE token");
    }
    const auto manifest = data.manifest;
    const auto weights = data.weights;
    const auto access = data.access;
    const auto scratch = data.scratch;
    const auto layout = data.layout;
    const auto state = data.state;
    const auto resources = data.resources;
    auto alternate_manifest = manifest;
    auto alternate_weights = weights;
    auto alternate_access = access;
    const std::vector<std::pair<const char*, std::function<void()>>> mutations{
        {"same-handle manifest name", [&] { data.manifest.tensors.front().name += ".changed"; }},
        {"same-handle manifest dimensions", [&] { ++data.manifest.tensors.front().dimensions.front(); }},
        {"same-handle manifest role", [&] { data.manifest.tensors.front().role =
            data.manifest.tensors.front().role == strix::TensorRole::unknown ? strix::TensorRole::embedding : strix::TensorRole::unknown; }},
        {"same-handle manifest residency", [&] { data.manifest.tensors.front().residency =
            data.manifest.tensors.front().residency == strix::ResidencyPolicy::resident ?
                strix::ResidencyPolicy::file_backed_candidate : strix::ResidencyPolicy::resident; }},
        {"same-handle manifest file offset", [&] { ++data.manifest.tensors.front().absolute_offset; }},
        {"same-handle access tensor index", [&] { ++data.access.accesses.front().tensor_indices.front(); }},
        {"weight generation", [&] { ++data.weights.tensors.front().allocation_generation; }},
        {"weight relocation", [&] { ++data.weights.tensors.front().address; }},
        {"scratch generation", [&] { ++data.scratch.allocation_generation; }},
        {"scratch relocation", [&] { data.scratch.arena_base += fixture.compiled.arena.alignment; }},
        {"device state generation", [&] { ++data.state.device_allocation_generation; }},
        {"host state generation", [&] { ++data.state.host_allocation_generation; }},
        {"device state relocation", [&] { data.state.device_base += data.layout.device_alignment; }},
        {"host state relocation", [&] { data.state.host_base += data.layout.host_alignment; }},
        {"state layout allocation offset", [&] { data.layout.allocations.front().offset += data.layout.device_alignment; }},
        {"equal manifest at another descriptor handle", [&] { data.resources.model_manifest = &alternate_manifest; }},
        {"equal weights at another descriptor handle", [&] { data.resources.model_weights = &alternate_weights; }},
        {"equal access at another descriptor handle", [&] { data.resources.tensor_access = &alternate_access; }},
    };
    for (const auto& [name, mutate] : mutations) {
        const auto before = session;
        const auto starts = data.ple_backend.start_calls;
        mutate();
        PleTrace rejected(data.ple_backend);
        require(rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(
                    prepared, session, data.resources, 3U, rejected)); }) &&
                rejected.trace.calls.empty() && data.ple_backend.start_calls == starts &&
                !data.ple_prefetch.active && !data.ple_backend.active, name);
        same_session(session, before);
        PleTrace public_rejected(data.ple_backend);
        const auto request = session_request(session);
        require(rejects([&] { static_cast<void>(prepared.run(data.resources, request, public_rejected)); }) &&
                public_rejected.trace.calls.empty() && data.ple_backend.start_calls == starts,
                "public run admitted a between-call mutation");
        data.manifest = manifest; data.weights = weights; data.access = access; data.scratch = scratch;
        data.layout = layout; data.state = state; data.resources = resources;
        PleTrace recovery(data.ple_backend);
        static_cast<void>(strix::execute_prepared_serial_token(prepared, session, data.resources, 3U, recovery));
        require(session.next_position == before.next_position + 1U &&
                session.health == strix::SerialSessionHealth::ready && !data.ple_prefetch.active,
                "rejected between-call mutation left sticky admission state");
    }
}
void ple_preflight_guards() {
    PleFixture fixture;
    auto& data = *fixture.owner;
    strix::PreparedEagerExecution prepared(fixture.compiled, data.resources, fixture.request, fixture.owner);
    const auto base = initial_session(fixture, prepared);
    const auto rejected = [&](strix::SerialSessionContext session,
                              const strix::ExecutionRuntimeResources& resources, std::uint32_t token = 3U) {
        const auto before = session;
        const auto starts = data.ple_backend.start_calls;
        PleTrace dispatcher(data.ple_backend);
        require(rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(
            prepared, session, resources, token, dispatcher)); }) && dispatcher.trace.calls.empty() &&
            data.ple_backend.start_calls == starts && !data.ple_prefetch.active && !data.ple_backend.active,
            "session preflight invoked PLE or dispatch before rejecting an invalid entry");
        same_session(session, before);
    };
    auto session = base; session.state.valid = strix::PersistentState::none; rejected(session, data.resources);
    for (std::size_t index = 0; index < session.state.generations.size(); ++index) {
        session = base; session.state.generations[index] = std::numeric_limits<std::uint64_t>::max();
        rejected(session, data.resources);
    }
    session = base; session.next_position = data.layout.context_tokens; rejected(session, data.resources);
    rejected(base, data.resources, static_cast<std::uint32_t>(fixture.compiled.vocab_size));
    const std::vector<std::function<void(strix::ExecutionRuntimeResources&)>> missing{
        [](auto& r) { r.scratch = nullptr; }, [](auto& r) { r.model_manifest = nullptr; },
        [](auto& r) { r.model_weights = nullptr; }, [](auto& r) { r.tensor_access = nullptr; },
        [](auto& r) { r.session_state_layout = nullptr; }, [](auto& r) { r.session_state_memory = nullptr; },
    };
    for (const auto& remove : missing) { auto resources = data.resources; remove(resources); rejected(base, resources); }
    session = base; session.runtime_compatibility_identity = 0; rejected(session, data.resources);
    session = base; ++session.runtime_compatibility_identity; rejected(session, data.resources);
    session = base; ++session.state_layout_identity; rejected(session, data.resources);
    session = base; ++session.execution_identity; rejected(session, data.resources);
    session = base; session.health = strix::SerialSessionHealth::poisoned; rejected(session, data.resources);
    session = base; session.committed_tokens = std::numeric_limits<std::uint64_t>::max(); rejected(session, data.resources);
}
void ple_failure_boundaries() {
    for (const auto failure : {CountingPle::StartFailure::throws, CountingPle::StartFailure::zero}) {
        PleFixture fixture; auto& data = *fixture.owner;
        strix::PreparedEagerExecution prepared(fixture.compiled, data.resources, fixture.request, fixture.owner);
        auto session = initial_session(fixture, prepared); const auto before = session;
        data.ple_backend.start_failure = failure;
        PleTrace dispatcher(data.ple_backend);
        require(rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(
            prepared, session, data.resources, 3U, dispatcher)); }) && dispatcher.trace.calls.empty() &&
            data.ple_backend.start_calls == 1U && data.ple_backend.cancel_calls == 0U &&
            !data.ple_prefetch.active && !data.ple_backend.active,
            "PLE start failure crossed its pre-dispatch transaction boundary");
        same_session(session, before);
        data.ple_backend.start_failure = CountingPle::StartFailure::none;
        PleTrace retry(data.ple_backend);
        static_cast<void>(strix::execute_prepared_serial_token(prepared, session, data.resources, 3U, retry));
        require(session.next_position == 1U && data.ple_backend.finish_calls == 1U,
                "pre-dispatch PLE start rejection could not be retried");
    }
    for (const unsigned location : {0U, 1U, 2U}) {
        PleFixture fixture; auto& data = *fixture.owner;
        strix::PreparedEagerExecution prepared(fixture.compiled, data.resources, fixture.request, fixture.owner);
        const auto ple = *prepared.ple_step_id();
        require(ple > 0U && ple + 1U < fixture.compiled.execution.steps.size(),
                "PLE failure fixture cannot cover both sides of its runtime step");
        auto session = initial_session(fixture, prepared); const auto before = session;
        PleTrace dispatcher(data.ple_backend);
        dispatcher.trace.fail_step = location == 0U ? ple - 1U : location == 1U ? ple : ple + 1U;
        bool completed = false;
        require(rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(
            prepared, session, data.resources, 3U, dispatcher, [&] { completed = true; })); }) && !completed,
            "dispatch failure with active PLE reached completion publication");
        require(data.ple_backend.cancel_calls == (location < 2U ? 1U : 0U) &&
                data.ple_backend.finish_calls == (location < 2U ? 0U : 1U) &&
                !data.ple_prefetch.active && !data.ple_backend.active,
                "dispatch failure canceled an already consumed PLE or leaked an active one");
        require_poisoned(session, before);
        const auto starts = data.ple_backend.start_calls;
        PleTrace poisoned(data.ple_backend);
        require(rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(
            prepared, session, data.resources, 3U, poisoned)); }) && poisoned.trace.calls.empty() &&
            data.ple_backend.start_calls == starts, "poisoned PLE session restarted work");
    }
    for (const unsigned failure : {0U, 1U, 2U}) {
        PleFixture fixture; auto& data = *fixture.owner;
        strix::PreparedEagerExecution prepared(fixture.compiled, data.resources, fixture.request, fixture.owner);
        auto session = initial_session(fixture, prepared); const auto before = session;
        PleTrace dispatcher(data.ple_backend);
        dispatcher.consume = failure != 0U;
        data.ple_backend.fail_finish = failure == 1U;
        bool completed = false;
        const auto complete = [&] {
            completed = true; same_session(session, before);
            throw std::runtime_error("injected post-PLE completion failure");
        };
        require(rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(
            prepared, session, data.resources, 3U, dispatcher, complete)); }),
            "unconsumed/finish/completion failure was swallowed");
        require(completed == (failure == 2U) &&
                data.ple_backend.finish_calls == (failure == 0U ? 0U : 1U) &&
                data.ple_backend.cancel_calls == (failure == 2U ? 0U : 1U) &&
                !data.ple_prefetch.active && !data.ple_backend.active,
                "PLE failure completion/cancel counts differ from the original transaction");
        require_poisoned(session, before);
    }
}
void ple_single_admission_and_lifetime() {
    PleFixture fixture;
    std::weak_ptr<Storage> lifetime = fixture.owner;
    auto* data = fixture.owner.get();
    {
        strix::PreparedEagerExecution prepared(fixture.compiled, data->resources, fixture.request, fixture.owner);
        auto session = initial_session(fixture, prepared);
        const auto counts = admission_counts(prepared, data->resources, session_request(session), data->ple_backend);
        fixture.owner.reset();
        require(!lifetime.expired(), "prepared PLE backend/resources owner was not retained");
        const std::uint32_t* stable_order = nullptr;
        for (std::uint32_t token = 3; token < 7U; ++token) {
            data->ple_backend.clear_counts();
            const auto before = session;
            PleTrace dispatcher(data->ple_backend);
            bool completed = false;
            const auto result = strix::execute_prepared_serial_token(prepared, session, data->resources, token, dispatcher, [&] {
                same_session(session, before);
                require(!data->ple_prefetch.active && data->ple_backend.finish_calls == 1U,
                        "completion callback preceded PLE consumption");
                completed = true;
            });
            require(dispatcher.identities_at_first_dispatch == counts.validation + counts.prefetch_start,
                    "prepared session did not perform exactly one full admission before PLE and first dispatch");
            require(completed && data->ple_backend.start_calls == 1U && data->ple_backend.finish_calls == 1U &&
                    data->ple_backend.cancel_calls == 0U && !data->ple_prefetch.active && !data->ple_backend.active,
                    "successful repeated PLE token has incomplete protocol counts");
            if (!stable_order) stable_order = result.completion_order.data();
            require(result.completion_order.data() == stable_order &&
                    result.completion_order.size() == fixture.compiled.execution.steps.size(),
                    "prepared PLE result lost its immutable complete submission order");
            for (std::size_t index = 0; index < result.completion_order.size(); ++index)
                require(result.completion_order[index] == fixture.compiled.execution.steps[index].id,
                        "prepared PLE submission order differs from the original compiled plan");
            require(session.next_position == before.next_position + 1U &&
                    session.committed_tokens == before.committed_tokens + 1U &&
                    strix::arithmetic_lineage_valid(session.arithmetic_lineage, session.next_position),
                    "successful repeated PLE token did not publish contiguous session state");
        }
        data->ple_backend.clear_counts();
        PleTrace public_dispatch(data->ple_backend);
        const auto direct = prepared.run(data->resources, session_request(session), public_dispatch);
        require(public_dispatch.identities_at_first_dispatch == counts.validation &&
                direct.completion_order.data() == stable_order,
                "public run after session calls reused a session-only admission or changed its result view");
    }
    require(lifetime.expired(), "prepared destruction leaked the PLE backend/resources owner");
}

// Arena-index fixtures are separate from T0-T5. All addresses below refer to
// retained actual host allocations; these dispatchers never execute kernels.
bool same_address(const strix::ExecutionBufferAddress& a, const strix::ExecutionBufferAddress& b) {
    return a.use == b.use && a.address == b.address && a.bytes == b.bytes &&
           a.allocation_generation == b.allocation_generation;
}
template<class Exception, class Function>
void arena_rejects(Function&& function, const char* fragment) {
    bool caught = false;
    try { function(); }
    catch (const Exception& error) {
        caught = std::string(error.what()).find(fragment) != std::string::npos;
    }
    require(caught, "arena rejection did not reach the intended typed admission gate");
}
struct ArenaTrace final : strix::EagerExecutionDispatcher {
    std::vector<strix::ExecutionDispatchContext> calls;
    std::vector<strix::ExecutionBufferAddress> buffers;
    const strix::MoeForkJoinArenaPlan* fork = nullptr;
    std::optional<bool> prepared_path;
    const strix::PreparedArenaViews* views = nullptr;
    void record(const strix::ExecutionDispatchContext& context) {
        if (!prepared_path) prepared_path = context.prepared_arena_views != nullptr;
        require(*prepared_path == (context.prepared_arena_views != nullptr), "arena cache presence changed during dispatch");
        if (context.prepared_arena_views) {
            if (!views) views = context.prepared_arena_views;
            require(views == context.prepared_arena_views, "forwarded dispatch changed arena cache owner");
        }
        const auto append = [&](const auto uses, auto resolver) {
            for (std::size_t i = 0; i < uses.size(); ++i) {
                const auto actual = resolver(context, i);
                const auto reference = strix::resolve_execution_buffer(context.execution_identity,
                    *context.arena, *context.resources->scratch, uses[i]);
                require(same_address(actual, reference), "indexed arena address differs from legacy full-use lookup");
                if (fork) require(same_address(actual, fork->resolve(*context.resources->scratch, uses[i])),
                    "indexed fork arena address differs from fork plan");
                buffers.push_back(actual);
            }
        };
        append(context.reads, strix::resolve_dispatch_read);
        append(context.writes, strix::resolve_dispatch_write);
        append(context.temporaries, strix::resolve_dispatch_temporary);
        calls.push_back(context);
    }
    void dispatch_kernel(const strix::ExecutionDispatchContext& context) override { record(context); }
    void dispatch_runtime_step(const strix::ExecutionDispatchContext& context) override { record(context); }
};
void same_arena_trace(const ArenaTrace& a, const ArenaTrace& b) {
    require(a.calls.size() == b.calls.size() && a.buffers.size() == b.buffers.size(), "arena sweep counts differ");
    for (std::size_t i = 0; i < a.buffers.size(); ++i)
        require(same_address(a.buffers[i], b.buffers[i]), "arena sweep changed full use/address/extent/generation");
}
void bind_fixture_arena(Fixture& fixture) {
    const auto base = allocate_aligned(fixture.owner->scratch_bytes,
        fixture.compiled.arena.total_bytes, fixture.compiled.arena.alignment);
    fixture.owner->scratch = {fixture.compiled.execution_identity, base, fixture.compiled.arena.total_bytes, 3};
}
strix::CompiledSerialExecution parallel_arena_compiled() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    strix::ExecutionBindingRequest request; request.parallel_moe_router = true;
    strix::ExecutionBufferRequest buffers; buffers.parallel_moe_router = true;
    strix::kernels::KernelRegistry registry;
    std::set<std::string> added;
    for (const auto& step : strix::build_serial_decode_execution_plan(model).steps) {
        const auto family = strix::operator_family_for(step.op);
        if (!family) continue;
        const auto shape = strix::execution_shape_for(model, step.op, request);
        auto id = "arena.fixture." + std::to_string(static_cast<unsigned>(*family)) + "." +
            std::to_string(shape.hidden_size) + "." + std::to_string(shape.auxiliary_size);
        if (step.op == strix::SerialOp::moe_route) id += ".parallel";
        if (!added.insert(id).second) continue;
        strix::kernels::KernelVariant variant;
        variant.id = id; variant.family = *family; variant.phase = strix::kernels::ExecutionPhase::decode;
        variant.shape = shape; variant.resources.workgroup_threads = 32;
        variant.numerics = {"bf16", "fp32", "bf16", true, true};
        variant.qualification = strix::kernels::QualificationStatus::oracle_matched;
        variant.graph_safe = true; variant.input_layout = "rows"; variant.output_layout = "rows";
        variant.compiler_identity = "synthetic-host"; registry.add(variant);
    }
    return strix::compile_serial_execution(model, registry, request, buffers);
}
void arena_parity_sweeps() {
    for (const bool parallel : {false, true}) {
        Fixture fixture;
        if (parallel) { fixture.compiled = parallel_arena_compiled(); bind_fixture_arena(fixture); }
        strix::PreparedEagerExecution prepared(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
        ArenaTrace reference, candidate;
        const auto a = strix::execute_compiled_serial_eager(fixture.compiled, fixture.owner->resources, fixture.request, reference);
        const auto b = prepared.run(fixture.owner->resources, fixture.request, candidate);
        require(reference.calls.size() == 438U && reference.buffers.size() == (parallel ? 1643U : 1595U),
                "canonical arena sweep did not visit every declared IO entry");
        require(reference.prepared_path == false && candidate.prepared_path == true,
                "arena sweep did not exercise both legacy and indexed resolvers");
        same_arena_trace(reference, candidate);
        require(a.state.generations == b.state.generations && same_span(a.completion_order, b.completion_order),
                "arena indexing changed state or completion order");
        const auto stable_order = b.completion_order.data();
        // Caller-owned arena metadata is independent of the immutable prepared snapshot.
        ++fixture.compiled.arena.allocations.front().offset;
        auto next = fixture.request; ++next.position; next.state = b.state;
        ArenaTrace repeated;
        const auto c = prepared.run(fixture.owner->resources, next, repeated);
        same_arena_trace(candidate, repeated);
        require(c.completion_order.data() == stable_order, "caller arena mutation invalidated prepared order span");
    }
}
void arena_fork_join() {
    Fixture fixture;
    const auto fork = strix::prepare_moe_fork_join_arena(fixture.compiled, std::numeric_limits<std::uint64_t>::max());
    fixture.compiled = fork.rebind_execution(fixture.compiled); bind_fixture_arena(fixture);
    strix::PreparedEagerExecution prepared(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
    ArenaTrace reference, candidate; reference.fork = &fork; candidate.fork = &fork;
    static_cast<void>(strix::execute_compiled_serial_eager(fixture.compiled, fixture.owner->resources, fixture.request, reference));
    static_cast<void>(prepared.run(fixture.owner->resources, fixture.request, candidate));
    same_arena_trace(reference, candidate);
    require(fork.layers().size() > 1U, "fork arena fixture lacks inter-layer reuse");
    const auto first = fork.resolve(fixture.owner->scratch, fork.layers()[0].shared_output);
    const auto second = fork.resolve(fixture.owner->scratch, fork.layers()[1].shared_output);
    require(first.use != second.use && first.address == second.address &&
            first.address == fixture.owner->scratch.arena_base + fork.serial_arena_bytes(),
            "shared outputs did not preserve distinct logical generations at the reusable join region");
}
void arena_context_rejections() {
    Fixture fixture;
    strix::PreparedEagerExecution a(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
    strix::PreparedEagerExecution b(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
    ArenaTrace at, bt;
    static_cast<void>(a.run(fixture.owner->resources, fixture.request, at));
    static_cast<void>(b.run(fixture.owner->resources, fixture.request, bt));
    const auto found = std::find_if(at.calls.begin(), at.calls.end(), [](const auto& c) { return !c.reads.empty(); });
    require(found != at.calls.end(), "arena fixture has no read context");
    const auto good = *found;
    const auto resolve = [&](const auto& c) { return strix::resolve_dispatch_read(c, 0); };
    const auto reject = [&](const auto& c, const char* message) {
        arena_rejects<std::invalid_argument>([&] { static_cast<void>(resolve(c)); }, message);
    };
    auto bad = good; bad.execution_identity = 0; reject(bad, "execution identity");
    bad = good; ++bad.execution_identity; reject(bad, "execution identity");
    bad = good; bad.arena = &fixture.compiled.arena; reject(bad, "arena handle");
    bad = good; bad.arena = bt.calls[good.step_id].arena; reject(bad, "arena handle");
    bad = bt.calls[good.step_id]; bad.arena = good.arena; reject(bad, "arena handle");
    bad = good; bad.step_id = static_cast<std::uint32_t>(at.calls.size());
    arena_rejects<std::out_of_range>([&] { static_cast<void>(resolve(bad)); }, "step");
    bad = good; bad.step_id = good.step_id == 0U ? 1U : 0U; reject(bad, "dispatch step");
    bad = good; bad.op = good.op == strix::SerialOp::lm_head ? strix::SerialOp::position_prepare : strix::SerialOp::lm_head;
    reject(bad, "dispatch step");
    bad = good; bad.layer = good.layer.value_or(0U) + 1U; reject(bad, "dispatch step");
    // Equal copied spans must remain legal; pointer equality is not the contract.
    std::vector<strix::ExecutionBufferUse> reads(good.reads.begin(), good.reads.end());
    bad = good; bad.reads = reads;
    require(same_address(resolve(good), resolve(bad)), "equal copied IO span was rejected");
    for (unsigned mutation = 0; mutation < 3U; ++mutation) {
        reads.assign(good.reads.begin(), good.reads.end());
        if (mutation == 0U) ++reads[0].generation;
        if (mutation == 1U) ++reads[0].buffer.slot;
        if (mutation == 2U) reads[0].buffer.buffer_class = reads[0].buffer.buffer_class == strix::ExecutionBufferClass::logits ?
            strix::ExecutionBufferClass::hidden : strix::ExecutionBufferClass::logits;
        bad = good; bad.reads = reads; reject(bad, "buffer generation");
    }
    reads.assign(good.reads.begin(), good.reads.end()); reads.push_back(reads.front());
    bad = good; bad.reads = reads; reject(bad, "IO sizes");
    bad = good; bad.writes = {}; // This read-bearing step also writes an output.
    require(!good.writes.empty(), "read context selected for missing-span test has no write");
    reject(bad, "IO sizes");
    arena_rejects<std::out_of_range>([&] { static_cast<void>(strix::resolve_dispatch_read(good, good.reads.size())); }, "read index");
    arena_rejects<std::out_of_range>([&] { static_cast<void>(good.prepared_arena_views->resolve(good,
        strix::PreparedBufferKind::read, good.reads.size(), fixture.owner->scratch)); }, "buffer index");
    arena_rejects<std::invalid_argument>([&] { static_cast<void>(good.prepared_arena_views->resolve(good,
        static_cast<strix::PreparedBufferKind>(999), 0, fixture.owner->scratch)); }, "buffer kind");
}
void arena_current_memory_and_fault_order() {
    Fixture fixture;
    strix::PreparedEagerExecution prepared(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
    ArenaTrace trace; static_cast<void>(prepared.run(fixture.owner->resources, fixture.request, trace));
    const auto found = std::find_if(trace.calls.begin(), trace.calls.end(), [](const auto& c) { return !c.reads.empty(); });
    require(found != trace.calls.end(), "current-binding fixture has no read");
    const auto good = *found;
    std::vector<std::byte> alternative_storage;
    auto alternative = fixture.owner->scratch;
    alternative.arena_base = allocate_aligned(alternative_storage, alternative.arena_bytes, fixture.compiled.arena.alignment);
    ++alternative.allocation_generation;
    auto context = good; context.memory = &alternative;
    const auto expected = strix::resolve_dispatch_read(good, 0);
    require(same_address(strix::resolve_dispatch_read(context, 0), expected), "context.memory displaced resources scratch priority");
    auto resources = fixture.owner->resources; resources.scratch = &alternative; context.resources = &resources;
    const auto relocated = strix::resolve_dispatch_read(context, 0);
    require(relocated.address - alternative.arena_base == expected.address - fixture.owner->scratch.arena_base &&
            relocated.allocation_generation == alternative.allocation_generation,
            "standalone arena resolver cached a physical address or generation");
    // These are standalone metadata resolver probes, not admission of rebound prepared runs.
    for (unsigned fault = 0; fault < 5U; ++fault) {
        auto broken = alternative;
        if (fault == 0U) broken.arena_base = 0;
        if (fault == 1U) --broken.arena_bytes;
        if (fault == 2U) broken.allocation_generation = 0;
        if (fault == 3U) broken.arena_base = std::numeric_limits<std::uintptr_t>::max();
        if (fault == 4U) ++broken.execution_identity;
        resources.scratch = &broken;
        if (fault == 3U) arena_rejects<std::overflow_error>([&] {
            static_cast<void>(strix::resolve_dispatch_read(context, 0)); }, "address range overflows");
        else arena_rejects<std::invalid_argument>([&] {
            static_cast<void>(strix::resolve_dispatch_read(context, 0)); }, fault == 0U ? "null arena base" :
                fault == 1U ? "smaller than compiled" : fault == 2U ? "generation must be non-zero" : "binding identity");
        // A valid span/arena enters the cache. Binding validation precedes its context checks.
        auto wrong_context = context; wrong_context.execution_identity = 0;
        if (fault == 3U) arena_rejects<std::overflow_error>([&] {
            static_cast<void>(strix::resolve_dispatch_read(wrong_context, 0)); }, "address range overflows");
        else arena_rejects<std::invalid_argument>([&] {
            static_cast<void>(strix::resolve_dispatch_read(wrong_context, 0)); }, fault == 0U ? "null arena base" :
                fault == 1U ? "smaller than compiled" : fault == 2U ? "generation must be non-zero" : "binding identity");
    }
}
void arena_admission_before_ple() {
    PleFixture fixture;
    strix::PreparedEagerExecution prepared(fixture.compiled, fixture.owner->resources, fixture.request, fixture.owner);
    std::vector<std::byte> other_storage;
    const auto other_base = allocate_aligned(other_storage, fixture.owner->scratch.arena_bytes, fixture.compiled.arena.alignment);
    auto accepted = initial_session(fixture, prepared);
    PleTrace initial_dispatch(fixture.owner->ple_backend);
    static_cast<void>(strix::execute_prepared_serial_token(prepared, accepted,
        fixture.owner->resources, 3U, initial_dispatch));
    for (unsigned fault = 0; fault < 3U; ++fault) {
        auto session = accepted; const auto before = session;
        auto memory = fixture.owner->scratch;
        if (fault == 0U) --memory.arena_bytes;
        if (fault == 1U) memory.arena_base = other_base;
        if (fault == 2U) ++memory.allocation_generation;
        auto resources = fixture.owner->resources; resources.scratch = &memory;
        fixture.owner->ple_backend.clear_counts(); ArenaTrace unused;
        require(rejects([&] { static_cast<void>(strix::execute_prepared_serial_token(prepared, session, resources, 3U, unused)); }) &&
                unused.calls.empty() && fixture.owner->ple_backend.start_calls == 0U &&
                !fixture.owner->ple_prefetch.active, "changed scratch reached PLE or dispatch before admission");
        same_session(session, before);
    }
}
void arena_preparation_rejections() {
    Fixture fixture;
    for (unsigned fault = 0; fault < 6U; ++fault) {
        auto broken = fixture.compiled;
        if (fault == 0U) broken.arena.allocations.push_back(broken.arena.allocations.front());
        if (fault == 1U) broken.arena.allocations.front().offset = broken.arena.total_bytes + 1U;
        if (fault == 2U) { broken.arena.allocations.front().offset = broken.arena.total_bytes;
                          broken.arena.allocations.front().bytes = 1U; }
        if (fault == 3U) ++broken.io.steps.front().step_id;
        if (fault == 4U) broken.io.steps.front().op = strix::SerialOp::lm_head;
        if (fault == 5U) {
            // A7 validates even allocations not referenced by any dispatch IO.
            auto unused = broken.arena.allocations.front();
            unused.use.generation = std::numeric_limits<std::uint64_t>::max();
            unused.offset = broken.arena.total_bytes + 1U;
            broken.arena.allocations.push_back(unused);
        }
        const auto* gate = fault == 0U ? "duplicate allocation uses" :
            (fault == 3U || fault == 4U) ? "execution IO identity" : "allocation exceeds compiled arena";
        arena_rejects<std::invalid_argument>([&] { strix::PreparedEagerExecution rejected(broken,
                    fixture.owner->resources, fixture.request, fixture.owner); }, gate);
    }
}
void arena_state_only_and_lifetime() {
    Fixture fixture;
    std::weak_ptr<Storage> weak = fixture.owner;
    auto* storage = fixture.owner.get();
    {
        strix::PreparedEagerExecution prepared(fixture.compiled, storage->resources, fixture.request, fixture.owner);
        fixture.owner.reset();
        require(!weak.expired(), "prepared arena did not retain actual allocation owner");
        const std::uint32_t* order = nullptr;
        for (std::uint32_t i = 0; i < 3U; ++i) {
            auto request = fixture.request; request.position += i; request.input_token_id += i;
            ArenaTrace baseline, candidate;
            const auto reference = strix::execute_compiled_serial_eager(fixture.compiled, storage->resources, request, baseline);
            const auto result = prepared.run(storage->resources, request, candidate);
            same_arena_trace(baseline, candidate);
            require(reference.state.generations == result.state.generations && reference.state.valid == result.state.valid,
                    "repeated arena run changed published state");
            if (!order) order = result.completion_order.data();
            require(order == result.completion_order.data(), "arena owner retention changed immutable result span");
        }
        ArenaTrace forwarded;
        strix::GenerationOutputDispatcher wrapper(forwarded, prepared.compiled());
        {
            strix::GenerationOutputDispatcher::StateOnlyScope scope(wrapper);
            static_cast<void>(prepared.run(storage->resources, fixture.request, wrapper));
        }
        require(wrapper.omitted_heads() == 1U && forwarded.calls.size() + 1U == prepared.compiled().execution.steps.size() &&
                forwarded.prepared_path == true && forwarded.calls.back().step_id + 2U == prepared.compiled().execution.steps.size(),
                "state-only prefix lost the cache or omitted more than its terminal head");
        auto session = strix::SerialSessionContext{};
        session.execution_identity = prepared.compiled().execution_identity;
        session.state_layout_identity = storage->layout.layout_identity;
        session.runtime_compatibility_identity = prepared.compatibility_identity(); session.state = fixture.request.state;
        auto baseline_session = session;
        for (const auto token : {3U, 4U}) {
            ArenaTrace dispatched, baseline_dispatch;
            const auto reference = strix::execute_serial_token(fixture.compiled, baseline_session,
                storage->resources, token, baseline_dispatch);
            const auto result = strix::execute_prepared_serial_token(prepared, session, storage->resources, token, dispatched);
            same_arena_trace(baseline_dispatch, dispatched); same_session(baseline_session, session);
            require(result.state.generations == reference.state.generations &&
                    dispatched.prepared_path == true && result.completion_order.data() == order &&
                    dispatched.calls.size() == 438U, "session wrapper lost indexed arena or immutable order");
        }
    }
    require(weak.expired(), "prepared arena view leaked its retained allocation owner");
}

} // namespace

int main() try {
    static_assert(!std::is_copy_constructible_v<strix::PreparedEagerExecution> &&
                  !std::is_move_constructible_v<strix::PreparedEagerExecution>);
    static_assert(std::is_same_v<decltype(std::declval<const strix::PreparedEagerExecution&>().compiled()),
                                 const strix::CompiledSerialExecution&>);
    static_assert(!std::is_copy_constructible_v<strix::PreparedArenaViews> &&
                  !std::is_move_constructible_v<strix::PreparedArenaViews> &&
                  !std::is_constructible_v<strix::PreparedArenaViews, const strix::CompiledSerialExecution&>);
    equivalence(); dispatch_failure(); dynamic_rejection(); preparation_rejection(); retained_owner(); session_adapter();
    public_run_admission(); ple_between_call_rejection(); ple_preflight_guards();
    ple_failure_boundaries(); ple_single_admission_and_lifetime();
    arena_parity_sweeps(); arena_fork_join(); arena_context_rejections();
    arena_current_memory_and_fault_order(); arena_admission_before_ple();
    arena_preparation_rejections(); arena_state_only_and_lifetime();
    std::cout << "prepared eager host contract fixtures passed; no native work\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
