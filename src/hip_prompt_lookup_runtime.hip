#include "strix/hip_prompt_lookup_runtime.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/prepared_tensor_views.hpp"
#include "strix/ple_lookup.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {
void require_lookup(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
void lookup_check(hipError_t value) {
    if (value != hipSuccess) throw std::runtime_error(std::string("prompt lookup HIP: ") + hipGetErrorString(value));
}
// Snapshot methods are synchronous, but an unsuccessful HIP transfer still
// requires proven quiescence before its snapshot owner can unwind/free.
class LookupTransfer final : public SessionSnapshotTransferBackend {
public:
    LookupTransfer(std::shared_ptr<const HipSetupQuiescenceGuard> guard,hipStream_t stream)
        :guard_(std::move(guard)),backend_(stream,guard_) {}
    void capture_batch(std::span<const SessionSnapshotTransfer> transfers) override {batch(transfers,true);}
    void restore_batch(std::span<const SessionSnapshotTransfer> transfers) override {batch(transfers,false);}
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    void capture_device(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override { copy(a,b,n,&HipSessionSnapshotTransferBackend::capture_device); }
    void capture_host(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override { copy(a,b,n,&HipSessionSnapshotTransferBackend::capture_host); }
    void restore_device(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override { copy(a,b,n,&HipSessionSnapshotTransferBackend::restore_device); }
    void restore_host(std::uintptr_t a,std::uintptr_t b,std::uint64_t n) override { copy(a,b,n,&HipSessionSnapshotTransferBackend::restore_host); }
private:
    void batch(std::span<const SessionSnapshotTransfer> transfers,bool capture) {
        try {
            guard_->require_original();
            if(capture) backend_.capture_batch(transfers);else backend_.restore_batch(transfers);
        } catch(...) {failed_=true;guard_->recover_or_fail_stop();throw;}
    }
    void copy(std::uintptr_t a,std::uintptr_t b,std::uint64_t n,
        void (HipSessionSnapshotTransferBackend::*method)(std::uintptr_t,std::uintptr_t,std::uint64_t)) {
        try { guard_->require_original(); (backend_.*method)(a,b,n); }
        catch(...) { failed_=true; guard_->recover_or_fail_stop(); throw; }
    }
    std::shared_ptr<const HipSetupQuiescenceGuard> guard_;
    HipSessionSnapshotTransferBackend backend_;
    bool failed_=false;
};
bool same_point(const SerialSessionContext& a, const SerialSessionContext& b) {
    return a.execution_identity == b.execution_identity && a.state_layout_identity == b.state_layout_identity &&
        a.runtime_compatibility_identity == b.runtime_compatibility_identity && a.next_position == b.next_position &&
        a.committed_tokens == b.committed_tokens && a.health == b.health && a.state.valid == b.state.valid &&
        a.state.generations == b.state.generations && a.arithmetic_lineage.intervals == b.arithmetic_lineage.intervals &&
        a.arithmetic_lineage.history_identity == b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted == b.arithmetic_lineage.tainted;
}
}
struct HipPromptLookupRuntime::Impl {
    // Owners outlive selectors, registry userdata and native completion.
    std::shared_ptr<const void> owner;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    CompiledSerialExecution serial;
    ModelContract model;
    HipLauncherCatalogConfig config;
    ExecutionRuntimeResources resources;
    CompiledKnownPrefixWindow compiled_window;
    std::shared_ptr<HipKnownPrefixWindow> window;
    ExecutionMemoryBinding window_memory;
    HipLaunchContext scope, serial_scope;
    RuntimeMemoryAllocator& allocator;
    AllocationGenerationSource& generations;
    LookupTransfer transfer;
    RuntimeExecutionIdentities original_identities;
    HipLauncherCatalog catalog;
    ExecutionLauncherRegistry registry;
    CompiledLaunchPlan launch_plan;
    std::unique_ptr<DirectLaunchDispatcher> dispatcher;
    std::unique_ptr<const PreparedTensorViews> metadata;
    std::unique_ptr<HipGreedyOutput> serial_select, row0_select, row1_select;
    std::uint64_t epoch = 0;
    bool failed = false, stopped_on_eos = false;
    std::optional<std::uint32_t> pending;
    struct Point { SerialSessionContext session; std::uint32_t token = 0; bool valid = false; };
    Point base, verified, replayed;
    HipPromptLookupAdaptivePolicy adaptive_policy;
    HipPromptLookupSnapshotPolicy snapshot_policy;
    std::uint64_t latest_snapshot_payload_bytes = 0;
    HipPromptLookupAdaptiveStats adaptive_stats;
    std::optional<ObservedSpeculationController> adaptive_controller;
    struct AdaptiveContext {
        std::uint64_t proposal = 0, bucket = 0, snapshot_alignment = 0;
        std::uint32_t maximum_depth = 0;
        std::vector<std::uint32_t> eos;
        bool valid = false;
    } adaptive_context;
    struct AdaptiveMeasurement {
        std::chrono::steady_clock::time_point started{};
        std::uint64_t position = 0, committed = 0, emitted = 0;
        std::uint32_t depth = 0;
        bool active = false;
    } adaptive_measurement;

    Impl(CompiledSerialExecution s, ModelContract m, HipLauncherCatalogConfig cfg,
        const ExecutionRuntimeResources& r, CompiledKnownPrefixWindow w, std::shared_ptr<HipKnownPrefixWindow> native,
        ExecutionMemoryBinding memory, HipLaunchContext serial_stream, HipLaunchContext stream, std::shared_ptr<const void> retained,
        std::shared_ptr<const HipSetupQuiescenceGuard> original, RuntimeMemoryAllocator& a, AllocationGenerationSource& g,
        HipPromptLookupAdaptivePolicy adaptive, HipPromptLookupSnapshotPolicy snapshot)
        : owner(std::move(retained)), guard(std::move(original)), serial(std::move(s)), model(std::move(m)),
          config(std::move(cfg)), resources(r), compiled_window(std::move(w)), window(std::move(native)),
          window_memory(memory), scope(stream), serial_scope(serial_stream), allocator(a), generations(g), transfer(guard,stream.stream), catalog(model,config) {
        adaptive_policy=adaptive;
        snapshot_policy=snapshot;
        validate_session_snapshot_policy(snapshot.format);
        require_lookup(snapshot.format!=SessionSnapshotPolicy::occupied_kv_v1 || snapshot.max_payload_bytes,
            "occupied KV lookup snapshot requires a positive payload budget");
        if(adaptive.enabled) {
            require_lookup(adaptive.context_bucket_tokens && adaptive.context_bucket_tokens<=1048576U,
                "prompt lookup adaptive context bucket must be within one and 1048576 tokens");
            adaptive_controller.emplace(adaptive.controller);
            adaptive_stats.enabled=true;
        }
        require_lookup(owner && guard && window && model.vocab_size && model.vocab_size <= std::numeric_limits<std::uint32_t>::max(),
            "prompt lookup requires retained native owners and a uint32 vocabulary");
        require_lookup(!serial.device_graph_inputs && !config.enable_device_graph_inputs && config.gdn_recurrent_layout == HipGdnRecurrentLayout::baseline,
            "prompt lookup requires the existing eager baseline recurrent layout");
        guard->require_original(); validate_scope();
        original_identities=build_runtime_execution_identities(serial.execution_identity,serial.arena,resources);
        require_lookup(!serial.execution.steps.empty() && serial.execution.steps.back().op==SerialOp::lm_head &&
            serial.io.steps.size()==serial.execution.steps.size() && serial.io.steps.back().writes.size()==1U,
            "prompt lookup requires an actual complete terminal LM head");
        validate_compiled_known_prefix_window(compiled_window,serial);
        require_lookup(compiled_window.output_policy==KnownWindowOutputPolicy::all_rows&&
            window->output_policy()==KnownWindowOutputPolicy::all_rows&&compiled_window.logical_rows==2U&&window->logical_rows()==2U,
            "prompt lookup verification requires an all-rows two-row native window");
        validate_known_prefix_window_formats(compiled_window,serial,*resources.model_manifest,*resources.tensor_access);
        validate_execution_memory_binding(compiled_window.window_execution_identity,compiled_window.arena.physical,window_memory);
        catalog.register_for_compiled_execution(serial,registry,&resources);
        launch_plan=compile_launch_plan(serial,registry);
        dispatcher=std::make_unique<DirectLaunchDispatcher>(launch_plan);
        metadata=std::make_unique<const PreparedTensorViews>(serial.execution_identity,serial.execution,
            *resources.model_manifest,*resources.model_weights,*resources.tensor_access,owner);
        serial_select=std::make_unique<HipGreedyOutput>(static_cast<std::uint32_t>(model.vocab_size),*resources.scratch,owner,guard);
        row0_select=std::make_unique<HipGreedyOutput>(static_cast<std::uint32_t>(model.vocab_size),window_memory,owner,guard);
        row1_select=std::make_unique<HipGreedyOutput>(static_cast<std::uint32_t>(model.vocab_size),window_memory,owner,guard);
        complete(scope); complete(serial_scope); // Admit a pre-zeroed status, never clear or hide a prior D1 error.
    }
    ~Impl() { guard->recover_or_fail_stop(); }
    void validate_scope() const {
        const auto empty=[](const ExecutionMemoryBinding& view) {
            return !view.execution_identity && !view.arena_base && !view.arena_bytes && !view.allocation_generation;
        };
        require_lookup(!current_hip_launch_context() && scope.stream && scope.stream!=hipStreamLegacy &&
            scope.stream!=hipStreamPerThread && !scope.graph_capture && scope.arguments==GraphDynamicArguments{} &&
            empty(scope.status) && empty(scope.ngram_input) && empty(scope.ngram_output) && !scope.context_tokens && !scope.vocabulary_size,
            "prompt lookup requires an exclusive eager scope with empty graph bindings");
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        lookup_check(hipStreamIsCapturing(scope.stream,&capture));
        require_lookup(capture==hipStreamCaptureStatusNone,"prompt lookup cannot enter active capture");
        require_lookup(serial_scope.stream==scope.stream,"prompt lookup serial and window scopes must share the owned stream");
        validate_record(scope,compiled_window.window_execution_identity);
        validate_record(serial_scope,serial.execution_identity);
        if(config.deferred_moe_status && resources.scratch && resources.session_state_memory && resources.model_weights) {
            for(const auto* record:{&scope.deferred_moe_status,&serial_scope.deferred_moe_status}) {
                const auto disjoint=[&](std::uintptr_t address,std::uint64_t bytes) {
                    require_lookup(bytes<=std::numeric_limits<std::uintptr_t>::max()-address &&
                        !(record->arena_base<address+bytes && address<record->arena_base+record->arena_bytes),
                        "prompt lookup D1 record aliases retained storage");
                };
                disjoint(window_memory.arena_base,window_memory.arena_bytes);
                disjoint(resources.scratch->arena_base,resources.scratch->arena_bytes);
                disjoint(resources.session_state_memory->device_base,resources.session_state_memory->device_bytes);
                disjoint(resources.session_state_memory->host_base,resources.session_state_memory->host_bytes);
                for(const auto& tensor:resources.model_weights->tensors) if(tensor.storage!=RuntimeTensorStorage::remote) disjoint(tensor.address,tensor.bytes);
            }
        }
        if(config.deferred_moe_status) require_lookup(scope.deferred_moe_status.arena_base!=serial_scope.deferred_moe_status.arena_base,
            "prompt lookup serial and window D1 records must be distinct");
    }
    void validate_record(const HipLaunchContext& binding,std::uint64_t identity) const {
        require_lookup(!binding.graph_capture && binding.arguments==GraphDynamicArguments{} &&
            !binding.status.arena_base && !binding.status.arena_bytes && !binding.status.execution_identity && !binding.status.allocation_generation &&
            !binding.ngram_input.arena_base && !binding.ngram_input.arena_bytes && !binding.ngram_input.execution_identity && !binding.ngram_input.allocation_generation &&
            !binding.ngram_output.arena_base && !binding.ngram_output.arena_bytes && !binding.ngram_output.execution_identity && !binding.ngram_output.allocation_generation &&
            !binding.context_tokens && !binding.vocabulary_size,"prompt lookup serial scope contains graph bindings");
        const auto& status=binding.deferred_moe_status;
        require_lookup(!config.deferred_moe_status || (status.execution_identity==identity &&
            status.arena_base && status.arena_base%8U==0 && status.arena_bytes==sizeof(DeferredMoeFirstError) && status.allocation_generation &&
            status.arena_bytes<=std::numeric_limits<std::uintptr_t>::max()-status.arena_base),
            "prompt lookup deferred status must be its retained execution record");
        require_lookup(config.deferred_moe_status || (!status.execution_identity && !status.arena_base && !status.arena_bytes && !status.allocation_generation),
            "prompt lookup unexpected deferred status record");
    }
    void complete(const HipLaunchContext& completed_scope) {
        DeferredMoeFirstError status{};
        try {
        if(config.deferred_moe_status) lookup_check(hipMemcpyAsync(&status,
            reinterpret_cast<const void*>(completed_scope.deferred_moe_status.arena_base),sizeof(status),hipMemcpyDeviceToHost,completed_scope.stream));
        lookup_check(hipStreamSynchronize(completed_scope.stream));
        guard->require_original();
        if(config.deferred_moe_status) check_deferred_moe_status(status);
        } catch(...) { guard->recover_or_fail_stop(); throw; }
    }
    void admit(const SerialSessionContext& session) const {
        if(failed) throw std::logic_error("prompt lookup target is failed");
        validate_scope(); guard->require_original();
        require_lookup(session.health==SerialSessionHealth::ready && !session.arithmetic_lineage.tainted,
            "prompt lookup requires ready untainted target state");
        require_lookup(resources.session_state_layout && session.next_position<=resources.session_state_layout->context_tokens &&
            session.next_position<=model.max_position_embeddings,"prompt lookup frontier exceeds context capacity");
        validate_execution_runtime_resources(serial.execution_identity,serial.arena,resources,
            session.next_position ? session.next_position-1U : 0U);
        const auto current=build_runtime_execution_identities(serial.execution_identity,serial.arena,resources);
        require_lookup(current.compatibility_identity==original_identities.compatibility_identity &&
            current.binding_identity==original_identities.binding_identity,
            "prompt lookup original resource binding changed");
        metadata->validate_source_metadata(*resources.model_manifest,*resources.model_weights,*resources.tensor_access);
        require_lookup(!resources.ple_prefetch || !resources.ple_prefetch->active,"prompt lookup requires inactive PLE lookup");
    }
    void caller_resources(const ExecutionRuntimeResources& current) const {
        require_lookup(current.scratch==resources.scratch && current.model_manifest==resources.model_manifest &&
            current.model_weights==resources.model_weights && current.tensor_access==resources.tensor_access &&
            current.session_state_layout==resources.session_state_layout && current.session_state_memory==resources.session_state_memory &&
            current.ple_lookup==resources.ple_lookup && current.ple_hash_contract==resources.ple_hash_contract &&
            current.ple_prefetch==resources.ple_prefetch,"prompt lookup caller resource handles changed");
    }
    const Point& frontier(const SerialSessionContext& session) const {
        for(const auto* point:{&replayed,&verified,&base}) if(point->valid && same_point(point->session,session)) return *point;
        throw std::invalid_argument("prompt lookup has no exact logits frontier for this restored state");
    }
    std::uint32_t select(HipGreedyOutput& output,const ExecutionBufferAddress& view,std::uint64_t position) {
        if(epoch==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("prompt lookup selector epoch exhausted");
        return output.select(view,static_cast<std::uint32_t>(model.vocab_size),position,++epoch);
    }
    ExecutionBufferAddress serial_logits() const {
        return resolve_execution_buffer(serial.execution_identity,serial.arena,*resources.scratch,serial.io.steps.back().writes.front());
    }
    void serial_step(SerialSessionContext& session,std::uint32_t token) {
        const auto position=session.next_position;
        const auto committed=session.committed_tokens;
        bool attempted=false;
        try {
            admit(session);
            attempted=true;
            { const HipLaunchScope active(serial_scope);
              (void)execute_serial_token(serial,session,resources,token,*dispatcher,[&]{complete(serial_scope);}); }
            replayed={session,select(*serial_select,serial_logits(),position),true};
        } catch(...) {
            if(attempted) { failed=true; guard->recover_or_fail_stop(); }
            if(session.committed_tokens!=committed) { session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true; }
            throw;
        }
    }
    static void count(std::uint64_t& value,std::uint64_t increment=1) noexcept {
        value=increment>std::numeric_limits<std::uint64_t>::max()-value ?
            std::numeric_limits<std::uint64_t>::max() : value+increment;
    }
    void censor_adaptive() noexcept {
        if(!adaptive_measurement.active) return;
        count(adaptive_stats.censored_cycles);
        adaptive_controller->cancel_observation();
        adaptive_measurement.active=false;
    }
    void start_adaptive(const SerialSessionContext& session,const DraftProposalStateContext& draft_state,
        GreedyDraftExecutionRequest& request,std::chrono::steady_clock::time_point started) {
        require_lookup(!adaptive_measurement.active,"finish the previous adaptive prompt lookup cycle first");
        require_lookup(request.token_budget!=0,"draft speculation token budget must be non-zero");
        const auto remaining=std::min(resources.session_state_layout->context_tokens,model.max_position_embeddings)-session.next_position;
        const auto cap=draft_state.phase==DraftProposalStatePhase::poisoned ? 0U :
            static_cast<std::uint32_t>(std::min<std::uint64_t>(request.max_candidate_depth,
                std::min<std::uint64_t>(remaining,request.token_budget-1U)));
        const auto bucket=session.next_position/adaptive_policy.context_bucket_tokens;
        // Execution, numerical and allocation identities are fixed for this
        // retained runtime and checked by admit(). EOS is compared in full;
        // neither a hash collision nor an unrelated bucket reuses cost data.
        if(!adaptive_context.valid || adaptive_context.proposal!=draft_state.proposal_identity ||
            adaptive_context.bucket!=bucket || adaptive_context.maximum_depth!=cap ||
            adaptive_context.snapshot_alignment!=request.snapshot_alignment ||
            !std::equal(adaptive_context.eos.begin(),adaptive_context.eos.end(),request.eos_tokens.begin(),request.eos_tokens.end())) {
            std::vector<std::uint32_t> eos(request.eos_tokens.begin(),request.eos_tokens.end());
            adaptive_context={draft_state.proposal_identity,bucket,request.snapshot_alignment,cap,std::move(eos),true};
            adaptive_controller->reset();
            count(adaptive_stats.context_resets);
        }
        const auto selected=adaptive_controller->select(cap);
        request.max_candidate_depth=selected;
        adaptive_measurement={started,session.next_position,session.committed_tokens,0,selected,true};
        adaptive_stats.selected_depth=selected;
        count(adaptive_stats.depth_selections[selected]);
    }
    void finish_adaptive(const SerialSessionContext& session) noexcept {
        if(!adaptive_measurement.active) return;
        const auto ended=std::chrono::steady_clock::now();
        const auto& m=adaptive_measurement;
        const bool complete=!pending && !failed && session.health==SerialSessionHealth::ready && !session.arithmetic_lineage.tainted &&
            session.execution_identity==serial.execution_identity && session.state_layout_identity==resources.session_state_layout->layout_identity &&
            session.runtime_compatibility_identity==original_identities.compatibility_identity &&
            session.committed_tokens>=m.committed && session.next_position>=m.position && m.emitted &&
            session.committed_tokens-m.committed==m.emitted && session.next_position-m.position==m.emitted;
        const double seconds=std::chrono::duration<double>(ended-m.started).count();
        const double total=adaptive_stats.complete_cycle_seconds[m.depth]+seconds;
        if(!complete || !std::isfinite(total) || !adaptive_controller->observe(m.depth,seconds,m.emitted)) {
            censor_adaptive();return;
        }
        count(adaptive_stats.completed_cycles);
        count(adaptive_stats.samples[m.depth]);
        count(adaptive_stats.emitted_tokens[m.depth],m.emitted);
        count(adaptive_stats.committed_tokens[m.depth],session.committed_tokens-m.committed);
        adaptive_stats.complete_cycle_seconds[m.depth]=total;
        adaptive_measurement.active=false;
    }
};
HipPromptLookupRuntime::HipPromptLookupRuntime(CompiledSerialExecution serial,ModelContract model,
    HipLauncherCatalogConfig config,const ExecutionRuntimeResources& resources,CompiledKnownPrefixWindow compiled_window,
    std::shared_ptr<HipKnownPrefixWindow> window,ExecutionMemoryBinding scratch,HipLaunchContext serial_scope,HipLaunchContext scope,
    std::shared_ptr<const void> owner,std::shared_ptr<const HipSetupQuiescenceGuard> guard,
    RuntimeMemoryAllocator& allocator,AllocationGenerationSource& generations,HipPromptLookupAdaptivePolicy adaptive,
    HipPromptLookupSnapshotPolicy snapshot)
    :impl_(std::make_unique<Impl>(std::move(serial),std::move(model),std::move(config),resources,std::move(compiled_window),
        std::move(window),scratch,serial_scope,scope,std::move(owner),std::move(guard),allocator,generations,adaptive,snapshot)) {}
HipPromptLookupRuntime::~HipPromptLookupRuntime()=default;
void HipPromptLookupRuntime::prime_from_serial_logits(SerialSessionContext& session) {
    impl_->admit(session);
    require_lookup(session.next_position>0,"prompt lookup prime requires a completed serial input");
    if(impl_->adaptive_policy.enabled) {
        impl_->censor_adaptive();impl_->adaptive_controller->reset();impl_->adaptive_context.valid=false;
    }
    try {
        impl_->complete(impl_->serial_scope);
        impl_->base={session,impl_->select(*impl_->serial_select,impl_->serial_logits(),session.next_position-1U),true};
        impl_->verified.valid=false;impl_->replayed.valid=false;impl_->pending.reset();impl_->stopped_on_eos=false;
    } catch(...) {
        impl_->failed=true;impl_->guard->recover_or_fail_stop();
        session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
        throw;
    }
}
std::vector<std::uint32_t> HipPromptLookupRuntime::verify_target_candidates(SerialSessionContext& session,
    const ExecutionRuntimeResources& resources,std::span<const std::uint32_t> candidates) {
    impl_->admit(session);
    impl_->caller_resources(resources);
    require_lookup(candidates.size()<=2U,"prompt lookup target supports only depths zero, one or two");
    for(auto token:candidates) require_lookup(token<impl_->model.vocab_size,"prompt lookup candidate outside vocabulary");
    const auto first=impl_->frontier(session).token;
    impl_->base={session,first,true};impl_->verified.valid=false;impl_->replayed.valid=false;
    std::vector<std::uint32_t> result;result.reserve(candidates.size()+1U);result.push_back(first);
    if(candidates.empty()) return result;
    if(candidates.size()==1U) { impl_->serial_step(session,candidates[0]);result.push_back(impl_->replayed.token);return result; }
    const auto position=session.next_position,committed=session.committed_tokens;
    bool attempted=false;
    try {
        attempted=true;
        const auto native=impl_->window->execute(session,{candidates[0],candidates[1]});
        require_lookup(native.logits.size()==2U&&native.logit_row_indices.size()==2U&&
            native.logit_row_indices[0]==0U&&native.logit_row_indices[1]==1U&&
            native.omitted_lm_heads==0U,"prompt lookup verification requires every mapped candidate head");
        result.push_back(impl_->select(*impl_->row0_select,native.logits[0],position));
        result.push_back(impl_->select(*impl_->row1_select,native.logits[1],position+1U));
        impl_->verified={session,result.back(),true};
        return result;
    } catch(...) {
        if(attempted) { impl_->failed=true;impl_->guard->recover_or_fail_stop(); }
        if(session.committed_tokens!=committed) {session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;}
        throw;
    }
}
void HipPromptLookupRuntime::replay_target_token(SerialSessionContext& session,const ExecutionRuntimeResources& resources,std::uint32_t token) {
    impl_->caller_resources(resources);
    impl_->serial_step(session,token);
}
GreedyDraftExecutionResult HipPromptLookupRuntime::cycle(SerialSessionContext& session,PromptLookupDraftBackend& lookup,
    DraftProposalStateContext& draft_state,const GreedyDraftExecutionRequest& request) {
    // Disabled mode performs no clock reads and retains the fixed request.
    const auto started=impl_->adaptive_policy.enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    impl_->admit(session);
    require_lookup(!impl_->pending,"consume the pending bonus before another prompt lookup cycle");
    require_lookup(request.max_candidate_depth<=2U,"prompt lookup maps depths only to zero, one or two target rows");
    require_lookup(draft_state.proposal_identity==lookup.proposal_identity(),"prompt lookup proposal owner changed");
    auto selected_request=request;
    selected_request.snapshot_policy=impl_->snapshot_policy.format;
    selected_request.max_snapshot_payload_bytes=impl_->snapshot_policy.max_payload_bytes;
    // Admit current occupied bytes before proposing/altering drafter state.
    // The generic executor repeats the budget check before snapshot allocation.
    if(selected_request.snapshot_policy==SessionSnapshotPolicy::occupied_kv_v1 &&
        selected_request.max_candidate_depth && selected_request.token_budget>1U &&
        session.next_position<impl_->resources.session_state_layout->context_tokens) {
        SerialSessionSnapshotRequest snapshot;
        snapshot.compiled_execution_identity=session.execution_identity;
        snapshot.runtime_compatibility_identity=session.runtime_compatibility_identity;
        snapshot.next_position=session.next_position; snapshot.committed_tokens=session.committed_tokens;
        snapshot.state=session.state; snapshot.alignment=selected_request.snapshot_alignment;
        snapshot.policy=selected_request.snapshot_policy;
        const auto plan=build_serial_session_snapshot_plan(impl_->model,*impl_->resources.session_state_layout,snapshot);
        require_lookup(plan.device_bytes<=UINT64_MAX-plan.host_bytes &&
            plan.device_bytes+plan.host_bytes<=impl_->snapshot_policy.max_payload_bytes,
            "occupied KV lookup snapshot exceeds its payload budget");
    }
    if(impl_->adaptive_policy.enabled) impl_->start_adaptive(session,draft_state,selected_request,started);
    GreedyDraftExecutionResult result;
    try {
        result=execute_greedy_speculation_with_draft(impl_->model,impl_->serial,impl_->resources,session,selected_request,
            impl_->allocator,impl_->generations,impl_->transfer,draft_state,lookup,*this);
    } catch(...) {
        impl_->censor_adaptive();
        if(impl_->transfer.failed()) {
            impl_->failed=true; session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
        }
        throw;
    }
    // The existing executor deliberately leaves a poisoned drafter unused at
    // depth zero. A healthy lookup still needs its emitted history reconciled.
    if(result.proposal_depth_limit==0U && draft_state.phase==DraftProposalStatePhase::synchronized) {
        try {
            begin_draft_proposal(draft_state,session);finish_draft_proposal(draft_state);
            lookup.reconcile_after_target(draft_state,session,result.target_cycle.plan.emitted_tokens);
            reconcile_draft_proposal_state(draft_state,session);
        } catch(...) { impl_->censor_adaptive();poison_draft_proposal_state(draft_state); throw; }
    }
    if(result.target_cycle.plan.emitted_tokens.empty()) {
        impl_->censor_adaptive();
        throw std::invalid_argument("prompt lookup cycle emitted no frontier");
    }
    impl_->pending=result.target_cycle.plan.emitted_tokens.back();
    if(result.target_cycle.snapshot_captured) {
        const auto& snapshot=result.target_cycle.plan.pre_window_snapshot;
        impl_->latest_snapshot_payload_bytes=snapshot.device_bytes+snapshot.host_bytes;
    }
    impl_->stopped_on_eos=result.target_cycle.plan.verification.stopped_on_eos;
    if(impl_->adaptive_measurement.active) impl_->adaptive_measurement.emitted=result.target_cycle.plan.emitted_tokens.size();
    return result;
}
void HipPromptLookupRuntime::consume_pending_bonus(SerialSessionContext& session,const PromptLookupDraftBackend& lookup,
    DraftProposalStateContext& draft_state) {
    impl_->admit(session);
    require_lookup(impl_->pending && !impl_->stopped_on_eos,"no resumable prompt lookup bonus is pending");
    require_lookup(impl_->frontier(session).token==*impl_->pending,"pending bonus differs from exact target frontier");
    require_lookup(draft_state.proposal_identity==lookup.proposal_identity(),"prompt lookup bonus belongs to a different draft owner");
    const bool disabled=draft_state.phase==DraftProposalStatePhase::poisoned;
    if(!disabled) validate_draft_proposal_state(draft_state,session);
    if(!disabled && draft_state.generation==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("draft lifecycle generation exhausted");
    impl_->serial_step(session,*impl_->pending);
    if(!disabled) {
        auto refreshed=lookup.make_state(session);refreshed.generation=draft_state.generation+1U;
        draft_state=refreshed;
    }
    impl_->pending.reset();
}
std::optional<std::uint32_t> HipPromptLookupRuntime::pending_bonus() const noexcept { return impl_->pending; }
const HipPromptLookupSnapshotPolicy& HipPromptLookupRuntime::snapshot_policy() const noexcept {return impl_->snapshot_policy;}
std::uint64_t HipPromptLookupRuntime::latest_snapshot_payload_bytes() const noexcept {return impl_->latest_snapshot_payload_bytes;}
void HipPromptLookupRuntime::finish_adaptive_cycle(const SerialSessionContext& session) noexcept {
    if(impl_->adaptive_policy.enabled) impl_->finish_adaptive(session);
}
const HipPromptLookupAdaptiveStats& HipPromptLookupRuntime::adaptive_stats() const noexcept { return impl_->adaptive_stats; }
} // namespace strix
