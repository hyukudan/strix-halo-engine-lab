#include "strix/hip_moe_overlap_dispatcher.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/prepared_execution.hpp"
#include "strix/prepared_serial_session.hpp"
#include "strix/generation_output_dispatch.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/model_tensor_index.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <set>
#include <vector>
#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
void check(hipError_t e, const char* text) {
    if (e != hipSuccess) throw std::runtime_error(std::string(text) + ": " + hipGetErrorString(e));
}
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
template<class Function> void reject(Function&& work) {
    bool caught = false;
    try { work(); } catch (const std::logic_error&) { caught = true; }
    require(caught, "invalid sibling protocol was accepted");
}
struct Resources {
    std::shared_ptr<const strix::HipSetupQuiescenceGuard> guard =
        std::make_shared<strix::HipSetupQuiescenceGuard>();
    hipStream_t main = nullptr;
    std::uint32_t* memory = nullptr;
    Resources() {
        try {
            guard->require_original();
            check(hipStreamCreateWithFlags(&main, hipStreamNonBlocking), "fixture main stream");
            check(hipMalloc(reinterpret_cast<void**>(&memory), 8U * sizeof(std::uint32_t)), "fixture memory");
            guard->run([&] { check(hipMemset(memory, 0xa5, 8U * sizeof(std::uint32_t)), "fixture guard initialization"); });
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() noexcept {
        guard->recover_or_fail_stop();
        if (memory && hipFree(memory) != hipSuccess) std::_Exit(3);
        if (main && hipStreamDestroy(main) != hipSuccess) std::_Exit(3);
        memory = nullptr; main = nullptr;
    }
    ~Resources() { cleanup(); }
    strix::HipLaunchContext context() const {
        strix::HipLaunchContext c;
        c.stream = main;
        c.deferred_moe_status = {0x4d4f454f564c5031ULL, reinterpret_cast<std::uintptr_t>(memory + 6), 8, 3};
        return c;
    }
};
__global__ void producer(std::uint32_t* memory, std::uint32_t seed) {
    if (!threadIdx.x && !blockIdx.x) memory[0] = seed;
}
__global__ void branch(std::uint32_t* memory, unsigned output, std::uint32_t iterations, std::uint32_t factor) {
    if (!threadIdx.x && !blockIdx.x) {
        auto value = memory[0] * factor;
        for (std::uint32_t i = 0; i < iterations; ++i) value = value * 1664525U + 1013904223U;
        memory[output] = value;
    }
}
__global__ void combine(std::uint32_t* memory, unsigned output) {
    if (!threadIdx.x && !blockIdx.x) memory[output] = memory[1] ^ memory[2];
}
std::uint32_t expected(std::uint32_t seed, std::uint32_t count, std::uint32_t factor) {
    auto value = seed * factor;
    for (std::uint32_t i = 0; i < count; ++i) value = value * 1664525U + 1013904223U;
    return value;
}
void positive(bool shared_slow) {
    auto resources = std::make_shared<Resources>();
    strix::HipMoeSiblingStreams streams(resources->main, resources, resources->guard);
    const auto context = resources->context();
    for (std::uint64_t token = 0; token < 3; ++token) {
        streams.begin_token(token,token,2);
        strix::HipLaunchScope original(context);
        const auto* outer_context = strix::current_hip_launch_context();
        std::array<std::uint32_t,2> oracle{};
        for (std::uint32_t layer = 0; layer < 2; ++layer) {
            const auto seed = static_cast<std::uint32_t>(token * 31U + layer + 1U);
            hipLaunchKernelGGL(producer,dim3(1),dim3(1),0,resources->main,resources->memory,seed);
            check(hipGetLastError(), "fixture producer launch");
            streams.fork_shared(layer);
            const auto side_iterations = shared_slow ? 20000U : 17U;
            const auto main_iterations = shared_slow ? 17U : 20000U;
            {
                strix::HipSharedExpertSideScope side(resources->main,streams.side_stream());
                require(strix::hip_execution_stream() == streams.side_stream() && strix::hip_deferred_moe_status() == nullptr,
                        "side context retained the D1 single-publisher record");
                reject([&] { strix::HipLaunchScope nested(context); });
                hipLaunchKernelGGL(branch,dim3(1),dim3(1),0,strix::hip_execution_stream(),resources->memory,1U,side_iterations,3U);
                check(hipGetLastError(), "fixture shared launch");
            }
            require(strix::current_hip_launch_context() == outer_context,
                    "side scope failed to restore exact TLS context");
            streams.record_shared_done(layer);
            streams.begin_experts(layer);
            require(strix::hip_execution_stream() == resources->main && strix::hip_deferred_moe_status() == resources->memory + 6,
                    "main D1 context was not restored");
            hipLaunchKernelGGL(branch,dim3(1),dim3(1),0,resources->main,resources->memory,2U,main_iterations,7U);
            check(hipGetLastError(), "fixture routed launch");
            streams.join_before_combine(layer);
            hipLaunchKernelGGL(combine,dim3(1),dim3(1),0,resources->main,resources->memory,3U+layer);
            check(hipGetLastError(), "fixture combine launch");
            oracle[layer] = expected(seed,side_iterations,3U) ^ expected(seed,main_iterations,7U);
        }
        require(streams.stats().committed_tokens == token, "enqueued branch counted as committed");
        streams.verify_before_commit();
        // This is a protocol fixture, not a real D1/session commit. The actual
        // dispatcher adapter requires the root's live SerialSessionContext.
        streams.mark_committed(token+1U,token+1U);
        std::array<std::uint32_t,8> result{};
        resources->guard->run([&] { check(hipMemcpy(result.data(),resources->memory,sizeof(result),hipMemcpyDeviceToHost), "fixture result"); });
        require(result[3] == oracle[0] && result[4] == oracle[1], "event fork/join failed to protect shared input/output");
        require(result[5] == 0xa5a5a5a5U && result[6] == 0xa5a5a5a5U && result[7] == 0xa5a5a5a5U,
                "sibling kernels modified guard or main-only D1 record");
    }
    const auto stats = streams.stats();
    require(stats.fork_records == 6 && stats.join_waits == 6 && stats.verified_tokens == 3 &&
            stats.committed_tokens == 3 && stats.committed_layers == 6 && !streams.failed(), "sibling protocol count mismatch");
}
void negative_scope_and_protocol() {
    auto resources = std::make_shared<Resources>();
    strix::HipMoeSiblingStreams streams(resources->main, resources, resources->guard);
    reject([&] { strix::HipSharedExpertSideScope side(resources->main,streams.side_stream()); });
    auto context = resources->context();
    {
        strix::HipLaunchScope original(context);
        const auto* previous = strix::current_hip_launch_context();
        reject([&] { strix::HipSharedExpertSideScope side(resources->main,resources->main); });
        try {
            strix::HipSharedExpertSideScope side(resources->main,streams.side_stream());
            throw std::runtime_error("test restoration");
        } catch (const std::runtime_error&) {}
        require(strix::current_hip_launch_context() == previous && strix::hip_deferred_moe_status() == resources->memory + 6,
                "exception failed to restore original context");
    }
    streams.begin_token(0,0,1);
    reject([&] { streams.verify_before_commit(); });
    reject([&] { streams.begin_experts(0); });
    streams.fork_shared(0);
    reject([&] { streams.join_before_combine(0); });
    reject([&] { streams.mark_committed(1,1); });
    streams.abort();
    require(streams.failed() && streams.stats().committed_tokens == 0, "incomplete branch published");
    reject([&] { streams.begin_token(1,0,1); });
}
void graph_rejected() {
    auto resources = std::make_shared<Resources>();
    strix::HipMoeSiblingStreams streams(resources->main,resources,resources->guard);
    check(hipStreamBeginCapture(resources->main,hipStreamCaptureModeThreadLocal), "fixture graph begin");
    bool failed = false;
    std::exception_ptr unexpected;
    try { streams.begin_token(0,0,1); } catch (const std::logic_error&) { failed = true; }
    catch (...) { unexpected = std::current_exception(); }
    hipGraph_t graph = nullptr;
    check(hipStreamEndCapture(resources->main,&graph), "fixture graph end");
    std::size_t nodes = 99;
    check(hipGraphGetNodes(graph,nullptr,&nodes), "fixture graph node count");
    check(hipGraphDestroy(graph), "fixture graph destroy");
    if (unexpected) std::rethrow_exception(unexpected);
    require(failed && nodes == 0, "sibling graph mode was not rejected before enqueue");
}
namespace adapter_fixture {
strix::PersistentState all_state() {
    using strix::PersistentState;
    return PersistentState::attention_kv | PersistentState::recurrent_matrix | PersistentState::convolution_history |
           PersistentState::sparse_index | PersistentState::ngram_history | PersistentState::position_history;
}
struct Storage {
    std::vector<std::byte> device_bytes, host_bytes;
    std::shared_ptr<Resources> gpu = std::make_shared<Resources>();
    void* scratch_allocation = nullptr;
    std::uint32_t* ledger = nullptr;
    ~Storage() {
        gpu->guard->recover_or_fail_stop();
        if (scratch_allocation && hipFree(scratch_allocation) != hipSuccess) std::_Exit(3);
        if (ledger && hipFree(ledger) != hipSuccess) std::_Exit(3);
    }
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

// Canonical compiled DAG with tiny synthetic weight/state metadata, a real
// expanded GPU arena, and toy GPU callbacks. No model arithmetic evidence.
struct Fixture {
    std::shared_ptr<Storage> owner = std::make_shared<Storage>();
    strix::CompiledSerialExecution compiled;
    strix::EagerExecutionRequest request;
    std::optional<strix::MoeForkJoinArenaPlan> plan;
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
        plan.emplace(strix::prepare_moe_fork_join_arena(compiled, std::numeric_limits<std::uint64_t>::max()));
        compiled = plan->rebind_execution(compiled);
        require(compiled.arena.alignment <= 256U, "fixture GPU guard alignment is insufficient");
        check(hipMalloc(&owner->scratch_allocation, static_cast<std::size_t>(compiled.arena.total_bytes + 512U)), "adapter scratch");
        check(hipMalloc(reinterpret_cast<void**>(&owner->ledger), (plan->layers().size() + 2U) * sizeof(std::uint32_t)), "adapter ledger");
        owner->gpu->guard->run([&] {
            check(hipMemset(owner->scratch_allocation, 0xa5, static_cast<std::size_t>(compiled.arena.total_bytes + 512U)), "adapter scratch poison");
            check(hipMemset(owner->ledger, 0xa5, (plan->layers().size() + 2U) * sizeof(std::uint32_t)), "adapter ledger poison");
        });
        const auto scratch_base = reinterpret_cast<std::uintptr_t>(owner->scratch_allocation) + 256U;
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
        request.state.valid = all_state(); request.input_token_id = 3; request.position = 0;
        request.state.generations = {7, 8, 9, 10, 11, 12};
    }
};


__global__ void toy_branch(const std::uint32_t* input, std::uint32_t* output,
                           std::uint32_t* temporary, std::uint32_t factor) {
    if (!threadIdx.x && !blockIdx.x) {
        auto value = *input * factor;
        for (unsigned i = 0; i < 1000U; ++i) value = value * 1664525U + 1013904223U;
        *temporary = value; *output = value ^ 0x12345678U;
    }
}
__global__ void toy_combine(const std::uint32_t* shared, const std::uint32_t* routed,
                            std::uint32_t* output, std::uint32_t* ledger) {
    if (!threadIdx.x && !blockIdx.x) *ledger = *output = *shared ^ *routed;
}
struct Callbacks {
    Fixture& fixture;
    bool prepared = false, fail_routed = false;
    std::uint32_t head_calls = 0, cached_calls = 0, publication_calls = 0;
    static void launch(const strix::ExecutionDispatchContext& c, void* opaque) {
        auto& self = *static_cast<Callbacks*>(opaque);
        require((c.prepared_tensor_views != nullptr) == self.prepared, "adapter lost prepared tensor cache");
        if (c.prepared_tensor_views) {
            ++self.cached_calls;
            for (const auto& access : c.resources->tensor_access->accesses) {
                if (access.step_id == c.step_id) {
                    const auto& actual = strix::resolve_dispatch_tensor_access(c,access.binding_slot);
                    require(actual.tensor_indices == access.tensor_indices, "adapter altered prepared access");
                }
            }
        }
        if (c.op == strix::SerialOp::lm_head) ++self.head_calls;
        for (const auto& layer : self.fixture.plan->layers()) {
            const auto address = [&](strix::ExecutionBufferUse use) {
                return reinterpret_cast<std::uint32_t*>(self.fixture.plan->resolve(*c.memory,use).address);
            };
            if (c.step_id == layer.mix_step) {
                require(strix::hip_execution_stream() == self.fixture.owner->gpu->main, "producer left main");
                hipLaunchKernelGGL(producer,dim3(1),dim3(1),0,strix::hip_execution_stream(),address(layer.input),layer.layer+1U);
            } else if (c.step_id == layer.shared_step || c.step_id == layer.experts_step) {
                const bool shared = c.step_id == layer.shared_step;
                require((strix::hip_deferred_moe_status() == nullptr) == shared, "adapter D1 publisher scope differs");
                hipLaunchKernelGGL(toy_branch,dim3(1),dim3(1),0,strix::hip_execution_stream(),address(layer.input),
                    address(shared ? layer.shared_output : layer.routed_output),
                    address(shared ? layer.shared_temporary : layer.routed_temporary),shared ? 3U : 7U);
                check(hipGetLastError(),"adapter branch launch");
                if (!shared && self.fail_routed) throw std::runtime_error("injected routed callback after enqueue");
            } else if (c.step_id == layer.combine_step) {
                hipLaunchKernelGGL(toy_combine,dim3(1),dim3(1),0,strix::hip_execution_stream(),
                    address(layer.shared_output),address(layer.routed_output),
                    reinterpret_cast<std::uint32_t*>(strix::resolve_dispatch_write(c,0).address),
                    self.fixture.owner->ledger+layer.layer+1U);
            } else continue;
            check(hipGetLastError(),"adapter toy launch");
            break;
        }
    }
};
void actual_adapter(bool prepared_mode, bool omit_head, unsigned failure) {
    Fixture fixture;
    auto callbacks = std::make_shared<Callbacks>(Callbacks{fixture,prepared_mode,failure == 1U});
    strix::ExecutionLauncherRegistry registry;
    std::set<std::string> kernels;
    std::set<strix::SerialOp> runtime_ops;
    for (const auto& binding : fixture.compiled.bindings.bindings) {
        if (binding.host_or_runtime_step) {
            if (runtime_ops.insert(binding.op).second)
                registry.add_runtime_launcher(binding.op,{"fixture.runtime",77,&Callbacks::launch,callbacks.get()});
        } else if (kernels.insert(*binding.kernel_id).second)
            registry.add_kernel_launcher({*binding.kernel_id,77,&Callbacks::launch,callbacks.get()});
    }
    const auto launches = strix::compile_launch_plan(fixture.compiled,registry);
    strix::HipMoeOverlapDispatcher overlap(fixture.compiled,*fixture.plan,launches,fixture.owner->resources,
        fixture.owner->gpu->main,fixture.owner,callbacks,fixture.owner->gpu->guard);
    strix::GenerationOutputDispatcher output(overlap,fixture.compiled);
    std::unique_ptr<strix::PreparedEagerExecution> prepared;
    if (prepared_mode) prepared = std::make_unique<strix::PreparedEagerExecution>(
        fixture.compiled,fixture.owner->resources,fixture.request,fixture.owner);
    strix::SerialSessionContext session;
    session.execution_identity = fixture.compiled.execution_identity;
    session.state_layout_identity = fixture.owner->layout.layout_identity;
    session.runtime_compatibility_identity = strix::build_runtime_execution_identities(
        fixture.compiled.execution_identity,fixture.compiled.arena,fixture.owner->resources).compatibility_identity;
    session.state = fixture.request.state;
    auto context = fixture.owner->gpu->context();
    context.deferred_moe_status.execution_identity = fixture.compiled.execution_identity;
    bool caught = false;
    try {
        strix::HipLaunchScope scope(context);
        overlap.begin_token(session);
        const auto hook = [&] {
            overlap.verify_before_commit(output.state_only_active());
            if (failure == 2U) throw std::runtime_error("injected completion before publication");
            ++callbacks->publication_calls; // Stub D1 boundary, not real D1 publication.
        };
        const auto execute = [&] {
            if (prepared) static_cast<void>(strix::execute_prepared_serial_token(*prepared,session,
                fixture.owner->resources,3,output,hook));
            else static_cast<void>(strix::execute_serial_token(fixture.compiled,session,
                fixture.owner->resources,3,output,hook));
        };
        if (omit_head) { strix::GenerationOutputDispatcher::StateOnlyScope intent(output); execute(); }
        else execute();
        overlap.mark_committed(session);
    } catch (const std::exception&) { overlap.abort(); caught = true; }
    require(caught == (failure != 0U), "adapter transaction unexpected failure outcome");
    require(session.committed_tokens == (failure ? 0U : 1U) && session.next_position == (failure ? 0U : 1U) &&
            session.health == (failure ? strix::SerialSessionHealth::poisoned : strix::SerialSessionHealth::ready) &&
            session.arithmetic_lineage.tainted == (failure != 0U) &&
            callbacks->publication_calls == (failure ? 0U : 1U), "adapter published a failed token");
    require(overlap.stats().committed_tokens == (failure ? 0U : 1U), "adapter counters invented a commit");
    if (!failure) {
        require(callbacks->head_calls == (omit_head ? 0U : 1U) &&
                output.omitted_heads() == (omit_head ? 1U : 0U) && (!prepared_mode || callbacks->cached_calls > 0U),
                "adapter prefix/prepared composition differs");
        std::vector<std::uint32_t> ledger(fixture.plan->layers().size()+2U);
        check(hipMemcpy(ledger.data(),fixture.owner->ledger,ledger.size()*sizeof(std::uint32_t),hipMemcpyDeviceToHost),"adapter ledger read");
        for (const auto& layer : fixture.plan->layers())
            require(ledger[layer.layer+1U] == (expected(layer.layer+1U,1000U,3U) ^ expected(layer.layer+1U,1000U,7U)),
                    "actual adapter fork/join mixed aliased or incomplete branch data");
        require(ledger.front() == 0xa5a5a5a5U && ledger.back() == 0xa5a5a5a5U,"adapter ledger guards changed");
    }
    std::array<std::byte,256> front{},back{};
    check(hipMemcpy(front.data(),fixture.owner->scratch_allocation,front.size(),hipMemcpyDeviceToHost),"adapter front guard");
    check(hipMemcpy(back.data(),reinterpret_cast<void*>(fixture.owner->scratch.arena_base+fixture.owner->scratch.arena_bytes),back.size(),hipMemcpyDeviceToHost),"adapter tail guard");
    for (const auto& guard : {front,back})
        require(std::all_of(guard.begin(),guard.end(),[](auto b) { return b == std::byte{0xa5}; }),"adapter scratch guard changed");
}
} // namespace adapter_fixture

} // namespace
int main() try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || !devices) return 77;
    positive(false); positive(true); negative_scope_and_protocol(); graph_rejected();
    for (const bool prepared : {false,true}) for (const bool omit : {false,true})
        adapter_fixture::actual_adapter(prepared,omit,0);
    adapter_fixture::actual_adapter(false,false,1);
    adapter_fixture::actual_adapter(true,true,2);
    std::cout << "MoE sibling event/scope diagnostic fixtures completed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
