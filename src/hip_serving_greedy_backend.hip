#include "strix/hip_serving_greedy_backend.hpp"

#include "strix/deferred_moe_status.hpp"
#include "strix/generation_output_dispatch.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_serving_prefix_cache.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/prepared_execution.hpp"
#include "strix/prepared_serial_session.hpp"
#include "strix/serial_decode_replay.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {
void serving_require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void serving_check(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
// The shared transfer backend admits the full snapshot, queues its D2D
// ranges on this request's owned stream and completes/drains before return.
using ServingPrefixTransfer = HipSessionSnapshotTransferBackend;
bool empty_binding(const ExecutionMemoryBinding& binding) noexcept {
    return !binding.execution_identity && !binding.arena_base && !binding.arena_bytes &&
        !binding.allocation_generation;
}
} // namespace

struct HipServingGreedyBackend::Impl {
    // Retained owners and original-device proof outlive all borrowed userdata.
    std::shared_ptr<const void> owner;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    CompiledSerialExecution serial;
    ModelContract model;
    HipLauncherCatalogConfig config;
    ExecutionRuntimeResources resources;
    SerialSessionContext& session;
    const HipLaunchContext scope;
    const bool use_prepared;
    RuntimeExecutionIdentities identities;
    HipLauncherCatalog catalog;
    ExecutionLauncherRegistry registry;
    CompiledLaunchPlan launch_plan;
    std::unique_ptr<DirectLaunchDispatcher> direct;
    std::unique_ptr<GenerationOutputDispatcher> output_dispatcher;
    // Also supplies deep immutable admission for the ordinary serial path.
    std::unique_ptr<PreparedEagerExecution> prepared;
    std::unique_ptr<HipGreedyOutput> output;
    ExecutionBufferAddress logits;
    std::uint64_t epoch = 0;
    bool broken = false;
    std::shared_ptr<HipServingPrefixCache> prefix_cache;
    std::vector<std::uint32_t> request_prefix;
    bool prefix_started=false,capture_enabled=false,capture_attempted=false,selected_valid=false;
    std::uint32_t selected_token=0;
    std::uint64_t selected_position=0;

    Impl(CompiledSerialExecution compiled, ModelContract contract, HipLauncherCatalogConfig cfg,
        const ExecutionRuntimeResources& original, SerialSessionContext& live_session,
        HipLaunchContext stream, std::shared_ptr<const void> retained, bool prepared_mode,
        std::shared_ptr<HipServingPrefixCache> pool)
        : owner(std::move(retained)), guard(std::make_shared<HipSetupQuiescenceGuard>()),
          serial(std::move(compiled)), model(std::move(contract)), config(std::move(cfg)),
          resources(original), session(live_session), scope(stream), use_prepared(prepared_mode),
          catalog(model, config),prefix_cache(std::move(pool)) {
        serving_require(owner && model.vocab_size && model.vocab_size <= std::numeric_limits<std::uint32_t>::max() &&
            model.vocab_size <= std::numeric_limits<std::size_t>::max()/sizeof(float) &&
            model.vocab_size == serial.vocab_size && serial.logical_rows == 1 && serial.request_concurrency == 1,
            "serving backend requires retained owners and one canonical uint32-vocabulary C1 plan");
        serving_require(config.deferred_moe_status,"serving backend requires an explicit per-session D1 record");
        serving_require(!serial.device_graph_inputs && !config.enable_device_graph_inputs,
            "serving backend requires eager original-resource execution");
        enter_worker();
        identities = build_runtime_execution_identities(serial.execution_identity, serial.arena, resources);
        if(prefix_cache) serving_require(prefix_cache->device_index()==guard->device_index() &&
            prefix_cache->model_content()==resources.model_weights->model_content_identity &&
            prefix_cache->vocabulary_size()==model.vocab_size,"serving prefix pool belongs to a different model or device");
        serving_require(session.health==SerialSessionHealth::ready && !session.arithmetic_lineage.tainted &&
            session.execution_identity==serial.execution_identity && session.state_layout_identity==resources.session_state_layout->layout_identity &&
            session.runtime_compatibility_identity==identities.compatibility_identity,
            "serving constructor session does not match its original runtime owners");
        serving_require(!serial.execution.steps.empty() && serial.execution.steps.back().op == SerialOp::lm_head &&
            serial.io.steps.size() == serial.execution.steps.size() && serial.io.steps.back().writes.size() == 1,
            "serving backend requires a complete terminal LM head");
        validate_scope();
        try {
        catalog.register_for_compiled_execution(serial, registry, &resources);
        serving_require(catalog.audit(serial, registry).missing_steps == 0,
            "serving backend native catalog coverage is incomplete");
        launch_plan = compile_launch_plan(serial, registry);
        direct = std::make_unique<DirectLaunchDispatcher>(launch_plan);
        output_dispatcher = std::make_unique<GenerationOutputDispatcher>(*direct, serial);
        EagerExecutionRequest prototype;
        prototype.position = session.next_position;
        prototype.state = session.state;
        prepared = std::make_unique<PreparedEagerExecution>(serial, resources, prototype, owner);
        logits = resolve_execution_buffer(serial.execution_identity, serial.arena, *resources.scratch,
            serial.io.steps.back().writes.front());
        serving_require(logits.bytes == model.vocab_size * sizeof(float),
            "serving backend logits must be the exact FP32 vocabulary view");
        output = std::make_unique<HipGreedyOutput>(static_cast<std::uint32_t>(model.vocab_size),
            *resources.scratch, owner, guard);
        complete(); // Check a pre-zeroed record; never silently clear it.
        } catch (...) { guard->recover_or_fail_stop(); throw; }
    }

    ~Impl() {
        // Destruction may run on another worker; establish the original device
        // before draining/freeing the selectors and releasing caller owners.
        if (current_hip_launch_context()) guard->recover_or_fail_stop();
        else {
            try {
                enter_worker();
                serving_check(hipStreamSynchronize(scope.stream), "drain serving stream");
            } catch (...) { guard->recover_or_fail_stop(); }
        }
    }

    void enter_worker() const {
        serving_require(!current_hip_launch_context(), "serving backend cannot borrow a nested HIP launch scope");
        serving_check(hipSetDevice(guard->device_index()), "bind serving worker device");
        guard->require_original();
    }

    void validate_scope() const {
        serving_require(scope.stream && scope.stream != hipStreamLegacy && scope.stream != hipStreamPerThread &&
            !scope.graph_capture && scope.arguments == GraphDynamicArguments{} && empty_binding(scope.status) &&
            empty_binding(scope.ngram_input) && empty_binding(scope.ngram_output) && !scope.context_tokens && !scope.vocabulary_size,
            "serving backend requires its exclusive eager stream with empty graph bindings");
        hipStreamCaptureStatus captured = hipStreamCaptureStatusNone;
        serving_check(hipStreamIsCapturing(scope.stream, &captured), "query serving capture state");
        serving_require(captured == hipStreamCaptureStatusNone, "serving backend cannot enter active stream capture");
        const auto& record = scope.deferred_moe_status;
        if (!config.deferred_moe_status) {
            serving_require(empty_binding(record), "serving deferred configuration and record differ");
            return;
        }
        serving_require(record.execution_identity == serial.execution_identity && record.arena_base &&
            record.arena_base % alignof(std::uint64_t) == 0 && record.arena_bytes == sizeof(DeferredMoeFirstError) &&
            record.allocation_generation && record.arena_bytes <= std::numeric_limits<std::uintptr_t>::max() - record.arena_base,
            "serving deferred status has stale identity, generation or extent");
        const auto disjoint = [&](std::uintptr_t address, std::uint64_t bytes) {
            if (!bytes) return;
            serving_require(address && bytes <= std::numeric_limits<std::uintptr_t>::max() - address &&
                !(record.arena_base < address + bytes && address < record.arena_base + record.arena_bytes),
                "serving deferred status aliases a retained owner");
        };
        disjoint(resources.scratch->arena_base, resources.scratch->arena_bytes);
        disjoint(resources.session_state_memory->device_base, resources.session_state_memory->device_bytes);
        disjoint(resources.session_state_memory->host_base, resources.session_state_memory->host_bytes);
        for (const auto& tensor : resources.model_weights->tensors)
            if (tensor.storage != RuntimeTensorStorage::remote) disjoint(tensor.address, tensor.bytes);
    }

    void admit(std::uint32_t token, std::uint64_t position) const {
        if (broken) throw std::logic_error("serving backend is permanently failed");
        enter_worker();
        validate_scope();
        serving_require(position == session.next_position && session.health == SerialSessionHealth::ready &&
            !session.arithmetic_lineage.tainted && session.execution_identity == serial.execution_identity &&
            session.runtime_compatibility_identity == identities.compatibility_identity &&
            session.state_layout_identity == resources.session_state_layout->layout_identity,
            "serving request differs from its ready live session");
        serving_require(token < serial.vocab_size && position < serial.max_position_embeddings &&
            position < resources.session_state_layout->context_tokens &&
            session.committed_tokens != std::numeric_limits<std::uint64_t>::max(),
            "serving input or position exceeds its admitted capacity");
        if(prefix_started && position<request_prefix.size())
            serving_require(request_prefix[static_cast<std::size_t>(position)]==token,
                "serving prefix input differs from the immutable cache request witness");
        EagerExecutionRequest request;
        request.input_token_id = token;
        request.position = position;
        request.state = session.state;
        prepared->validate_dynamic(resources, request);
    }

    void complete() {
        // This local remains alive until success or a proved recovery drain.
        DeferredMoeFirstError record{};
        try {
            if (config.deferred_moe_status)
                serving_check(hipMemcpyAsync(&record, reinterpret_cast<const void*>(scope.deferred_moe_status.arena_base),
                    sizeof(record), hipMemcpyDeviceToHost, scope.stream), "read serving D1 status");
            serving_check(hipStreamSynchronize(scope.stream), "complete serving producer");
            guard->require_original();
            if (config.deferred_moe_status) check_deferred_moe_status(record);
        } catch (...) { guard->recover_or_fail_stop(); throw; }
    }

    template<class Delivery>
    void run(std::uint32_t token, std::uint64_t position, bool state_only, Delivery&& delivery) {
        admit(token, position); // Pure rejection leaves the session untouched.
        bool attempted = false;
        selected_valid=false;
        try {
            serving_check(hipPeekAtLastError(), "serving preexisting HIP error");
            const HipLaunchScope active(scope);
            const auto before_commit = [&] {
                complete();
                delivery();
            };
            const auto dispatch = [&] {
                attempted = true;
                if (use_prepared)
                    (void)execute_prepared_serial_token(*prepared, session, resources, token,
                        *output_dispatcher, before_commit);
                else
                    (void)execute_serial_token(serial, session, resources, token, *output_dispatcher, before_commit);
            };
            if (state_only) {
                const GenerationOutputDispatcher::StateOnlyScope omit(*output_dispatcher);
                dispatch();
            } else dispatch();
        } catch (...) {
            if (attempted) {
                broken = true;
                guard->recover_or_fail_stop();
                session.health = SerialSessionHealth::poisoned;
                session.arithmetic_lineage.tainted = true;
            }
            throw;
        }
    }
};

HipServingGreedyBackend::HipServingGreedyBackend(CompiledSerialExecution serial, ModelContract model,
    HipLauncherCatalogConfig config, const ExecutionRuntimeResources& resources, SerialSessionContext& session,
    HipLaunchContext scope, std::shared_ptr<const void> owner, bool prepared_eager,
    std::shared_ptr<HipServingPrefixCache> prefix_cache)
    : impl_(std::make_unique<Impl>(std::move(serial), std::move(model), std::move(config), resources,
        session, scope, std::move(owner), prepared_eager,std::move(prefix_cache))) {}
HipServingGreedyBackend::~HipServingGreedyBackend() = default;
std::uint64_t HipServingGreedyBackend::output_reservation_bytes(std::uint32_t vocabulary) {
    return HipGreedyOutput::required_scratch_bytes(vocabulary);
}

std::vector<float> HipServingGreedyBackend::execute(std::uint32_t token, std::uint64_t position) {
    // Reserve delivery storage before any model state can be consumed.
    std::vector<float> values(static_cast<std::size_t>(impl_->model.vocab_size));
    impl_->run(token, position, false, [&] {
        serving_check(hipMemcpyAsync(values.data(), reinterpret_cast<const void*>(impl_->logits.address),
            static_cast<std::size_t>(impl_->logits.bytes), hipMemcpyDeviceToHost, impl_->scope.stream),
            "copy serving full logits");
        serving_check(hipStreamSynchronize(impl_->scope.stream), "complete serving full logits");
        impl_->guard->require_original();
        validate_serial_decode_engine_logits(values, impl_->model.vocab_size);
    });
    return values;
}

std::uint32_t HipServingGreedyBackend::execute_greedy(std::uint32_t token, std::uint64_t position,
                                                     std::uint64_t vocabulary) {
    serving_require(vocabulary == impl_->model.vocab_size, "serving greedy vocabulary differs from its model");
    if (impl_->epoch == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("serving greedy output epoch exhausted");
    std::uint32_t result = 0;
    impl_->run(token, position, false, [&] {
        result = impl_->output->select(impl_->logits, static_cast<std::uint32_t>(vocabulary), position, ++impl_->epoch);
    });
    impl_->selected_valid=true;impl_->selected_token=result;impl_->selected_position=position;
    return result;
}

void HipServingGreedyBackend::execute_state(std::uint32_t token, std::uint64_t position) {
    impl_->run(token, position, true, [] {});
}
std::optional<GreedyPrefixRestore> HipServingGreedyBackend::try_restore_greedy_prefix(const SerialGreedyRequest& request) {
    auto& s=*impl_;if(!s.prefix_cache) return std::nullopt;
    if(s.broken || s.prefix_started) throw std::logic_error("serving prefix request was already started or failed");
    s.enter_worker();s.validate_scope();
    serving_require(!request.prefix_tokens.empty() && request.vocab_size==s.model.vocab_size &&
        request.context_capacity==s.resources.session_state_layout->context_tokens &&
        request.prefix_tokens.size()<=request.context_capacity && request.eos_token_id<request.vocab_size &&
        s.session.next_position==0U && s.session.committed_tokens==0U &&
        std::all_of(request.prefix_tokens.begin(),request.prefix_tokens.end(),[&](auto t){return t<request.vocab_size;}),
        "serving cached request differs from its original fresh session");
    s.admit(request.prefix_tokens.front(),0U); // deep source admission before any restore
    serving_require(!s.resources.ple_prefetch || !s.resources.ple_prefetch->active,"serving restore cannot borrow active PLE work");
    // Reserve the exact input witness BEFORE copying any cached target state.
    std::vector<std::uint32_t> prefix(request.prefix_tokens.begin(),request.prefix_tokens.end());
    s.request_prefix=std::move(prefix);s.prefix_started=true;s.capture_enabled=request.max_new_tokens!=0U;
    try {
        s.complete();ServingPrefixTransfer transfer(s.scope.stream,s.guard);
        const auto result=s.prefix_cache->try_restore(s.model,request,s.session,*s.resources.session_state_layout,
            *s.resources.session_state_memory,transfer);
        s.complete();return result;
    } catch(...) {
        s.broken=true;s.guard->recover_or_fail_stop();s.session.health=SerialSessionHealth::poisoned;
        s.session.arithmetic_lineage.tainted=true;throw;
    }
}
void HipServingGreedyBackend::capture_greedy_prefix(std::span<const std::uint32_t> prefix,std::uint32_t pending,std::uint64_t vocabulary) {
    auto& s=*impl_;if(!s.prefix_cache || !s.capture_enabled || s.capture_attempted) return;
    try {s.enter_worker();s.validate_scope();}
    catch(...) {
        // Model consumption already completed before this hook. No new work
        // was queued here: poison without synchronizing a foreign capture.
        s.broken=true;s.session.health=SerialSessionHealth::poisoned;s.session.arithmetic_lineage.tainted=true;throw;
    }
    try {
        serving_require(!s.broken && s.prefix_started && vocabulary==s.model.vocab_size && pending<vocabulary &&
            prefix.size()==s.request_prefix.size() && std::equal(prefix.begin(),prefix.end(),s.request_prefix.begin()) &&
            s.session.next_position==prefix.size() && s.selected_valid && s.selected_token==pending &&
            s.selected_position+1U==prefix.size(),"serving cache capture has no actual terminal finite-selection witness");
        serving_require(!s.resources.ple_prefetch || !s.resources.ple_prefetch->active,"serving capture requires retired PLE work");
        s.complete();ServingPrefixTransfer transfer(s.scope.stream,s.guard);
        s.prefix_cache->capture(s.model,prefix,pending,s.session,*s.resources.session_state_layout,*s.resources.session_state_memory,transfer);
        s.complete();s.capture_attempted=true;
    } catch(...) {
        // The prefix model calls already committed. Never leave that consumed
        // session healthy while reporting a failed cache-copy request.
        s.broken=true;s.guard->recover_or_fail_stop();s.session.health=SerialSessionHealth::poisoned;
        s.session.arithmetic_lineage.tainted=true;throw;
    }
}
bool HipServingGreedyBackend::failed() const noexcept { return impl_->broken; }
int HipServingGreedyBackend::device_index() const noexcept { return impl_->guard->device_index(); }

} // namespace strix
