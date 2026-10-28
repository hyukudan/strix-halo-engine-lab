#include "strix/hip_serving_greedy_backend.hpp"

#include "strix/deferred_moe_status.hpp"
#include "strix/hip_known_prefix_window.hpp"
#include "strix/generation_output_dispatch.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_serving_prefix_cache.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/prepared_execution.hpp"
#include "strix/prepared_serial_session.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/serial_decode_replay.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
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

std::uint64_t serving_add(std::uint64_t a,std::uint64_t b) {
    if(b>UINT64_MAX-a) throw std::overflow_error("serving prefill reservation overflow");
    return a+b;
}
std::uint64_t serving_mul(std::uint64_t a,std::uint64_t b) {
    if(a && b>UINT64_MAX/a) throw std::overflow_error("serving prefill reservation overflow");
    return a*b;
}
std::uint64_t arena_metadata(const KnownWindowArena& arena) {
    auto bytes=serving_mul(arena.physical.allocations.size(),sizeof(ExecutionArenaAllocation));
    bytes=serving_add(bytes,serving_mul(arena.row_stride.size(),sizeof(std::uint64_t)));
    bytes=serving_add(bytes,serving_mul(arena.row_view.size(),sizeof(ExecutionArenaPlan)));
    for(const auto& row:arena.row_view) bytes=serving_add(bytes,serving_mul(row.allocations.size(),sizeof(ExecutionArenaAllocation)));
    return bytes;
}
struct ServingPrefillAllocation final {
    std::shared_ptr<const void> source_owner;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    HipLaunchContext scope;
    void* raw=nullptr;void* status=nullptr;
    std::uintptr_t aligned=0;
    std::uint64_t bytes=0,scratch_generation=0,state_status_generation=0,terminal_status_generation=0;
    ServingPrefillAllocation(const HipServingPrefillPlan& plan,HipLaunchContext stream,
        std::shared_ptr<const void> retained,std::shared_ptr<const HipSetupQuiescenceGuard> device)
        :source_owner(std::move(retained)),guard(std::move(device)),scope(stream),bytes(plan.workspace_bytes) {
        serving_require(source_owner && guard && bytes && plan.alignment && !(plan.alignment&(plan.alignment-1U)),
            "serving prefill needs an actual retained source and nonempty aligned workspace");
        const auto allocated=serving_add(bytes,plan.alignment-1U);
        serving_require(allocated<=SIZE_MAX,"serving prefill native allocation exceeds host size domain");
        AllocationGenerationSource generations;scratch_generation=generations.next();
        state_status_generation=generations.next();terminal_status_generation=generations.next();
        try {
            guard->run([&] {
                serving_check(hipMalloc(&raw,static_cast<std::size_t>(allocated)),"allocate serving prefill workspace");
                const auto base=reinterpret_cast<std::uintptr_t>(raw);
                serving_require(plan.alignment-1U<=UINTPTR_MAX-base,"serving prefill alignment overflows pointer");
                aligned=(base+plan.alignment-1U)&~static_cast<std::uintptr_t>(plan.alignment-1U);
                serving_check(hipMalloc(&status,16U),"allocate two serving prefill D1 records");
                serving_check(hipMemsetAsync(reinterpret_cast<void*>(aligned),0,static_cast<std::size_t>(bytes),scope.stream),"initialize serving prefill workspace");
                serving_check(hipMemsetAsync(status,0,16U,scope.stream),"initialize serving prefill D1 records once");
            });
        } catch(...) {release();throw;}
    }
    ~ServingPrefillAllocation() {release();}
    void release() noexcept {
        if(!raw && !status) return;
        guard->recover_or_fail_stop();
        if(status && hipFree(status)!=hipSuccess) std::_Exit(3);
        status=nullptr;
        if(raw && hipFree(raw)!=hipSuccess) std::_Exit(3);
        raw=nullptr;
    }
    ExecutionMemoryBinding scratch(std::uint64_t id) const {return {id,aligned,bytes,scratch_generation};}
    HipLaunchContext context(std::uint64_t id,bool terminal) const {
        auto result=scope;
        result.deferred_moe_status={id,reinterpret_cast<std::uintptr_t>(status)+(terminal?8U:0U),8U,
            terminal?terminal_status_generation:state_status_generation};return result;
    }
};
struct ServingPrefillRuntime final {
    std::shared_ptr<const HipServingPrefillPlan> plan;
    std::shared_ptr<ServingPrefillAllocation> memory;
    std::unique_ptr<HipKnownPrefixWindow> state,terminal;
    std::unique_ptr<HipGreedyOutput> output;
    ExecutionBufferAddress logits;
    HipServingPrefillStats counters;
    ServingPrefillRuntime(std::shared_ptr<const HipServingPrefillPlan> prepared,const CompiledSerialExecution& serial,
        const ModelContract& model,const HipLauncherCatalogConfig& config,const ExecutionRuntimeResources& resources,
        HipLaunchContext scope,std::shared_ptr<const void> owner,std::shared_ptr<const HipSetupQuiescenceGuard> guard)
        :plan(std::move(prepared)) {
        serving_require(plan && plan->rows && config.deferred_moe_status && config.activation==HipActivationStorage::bf16 &&
            config.gdn_recurrent_storage==GdnRecurrentStorage::key_value && config.gdn_recurrent_layout==HipGdnRecurrentLayout::baseline &&
            !config.qsa_hierarchical_selection && config.qsa_gqa_heads_per_block==1U &&
            config.wave8_q8_gdn_input && config.wave8_q8_gdn_output && config.virtual_q8_gdn_input &&
            config.prefetch_q8_gdn_input && config.virtual_q8_gdn_output,
            "serving prefill requires its admitted paired wave8 BF16 profile and baseline state/QSA layouts");
        const auto build=[&](bool final) {
            const auto& tiled=final?plan->terminal_tiled:plan->state_tiled;
            const auto& small=final?plan->terminal_window:plan->state_window;
            if(tiled) validate_compiled_known_prefix_prefill(*tiled,serial);
            else {validate_compiled_known_prefix_window(small,serial);
                validate_known_prefix_window_formats(small,serial,*resources.model_manifest,*resources.tensor_access);}
            const auto policy=tiled?tiled->output_policy:small.output_policy;
            serving_require(policy==(final?KnownWindowOutputPolicy::terminal_row:KnownWindowOutputPolicy::state_only),
                "serving prefill intent plan changed");
        };
        build(false);build(true);
        memory=std::make_shared<ServingPrefillAllocation>(*plan,scope,std::move(owner),guard);
        const auto create=[&](bool final) {
            const auto& tiled=final?plan->terminal_tiled:plan->state_tiled;
            const auto& small=final?plan->terminal_window:plan->state_window;
            const auto id=tiled?tiled->prefill_execution_identity:small.window_execution_identity;
            if(tiled) return std::make_unique<HipKnownPrefixWindow>(*tiled,serial,model,config,resources,
                memory->scratch(id),memory->context(id,final),memory);
            return std::make_unique<HipKnownPrefixWindow>(small,serial,model,config,resources,
                memory->scratch(id),memory->context(id,final),memory);
        };
        state=create(false);terminal=create(true);
        const auto id=plan->terminal_tiled?plan->terminal_tiled->prefill_execution_identity:plan->terminal_window.window_execution_identity;
        const auto& arena=plan->terminal_tiled?plan->terminal_tiled->arena:plan->terminal_window.arena;
        logits=resolve_execution_buffer(id,arena.row_view.back(),memory->scratch(id),serial.io.steps.back().writes.front());
        output=std::make_unique<HipGreedyOutput>(static_cast<std::uint32_t>(model.vocab_size),memory->scratch(id),memory,guard);
    }
    void count(std::uint64_t& n,std::uint64_t delta=1U) noexcept {
        if(delta>UINT64_MAX-n) {n=UINT64_MAX;counters.saturated=true;}else n+=delta;
    }
};
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
    HipSharedExpertLauncherConfig cohort_shared,cohort_shared_fallback;
    ExecutionLauncherRegistry registry;
    CompiledLaunchPlan launch_plan;
    std::unique_ptr<DirectLaunchDispatcher> direct;
    std::unique_ptr<GenerationOutputDispatcher> output_dispatcher;
    // Also supplies deep immutable admission for the ordinary serial path.
    std::unique_ptr<PreparedEagerExecution> prepared;
    std::unique_ptr<HipGreedyOutput> output;
    ExecutionBufferAddress logits;
    std::uint64_t epoch = 0;
    bool broken = false, cohort_pending = false;
    std::shared_ptr<HipServingPrefixCache> prefix_cache;
    std::unique_ptr<ServingPrefillRuntime> prefill;
    std::optional<SerialPrefixProducerRecipe> prefix_recipe;
    std::vector<std::uint32_t> request_prefix;
    bool prefix_started=false,capture_enabled=false,capture_attempted=false,selected_valid=false;
    std::uint32_t selected_token=0;
    std::uint64_t selected_position=0;

    Impl(CompiledSerialExecution compiled, ModelContract contract, HipLauncherCatalogConfig cfg,
        const ExecutionRuntimeResources& original, SerialSessionContext& live_session,
        HipLaunchContext stream, std::shared_ptr<const void> retained, bool prepared_mode,
        std::shared_ptr<HipServingPrefixCache> pool,std::shared_ptr<const HipServingPrefillPlan> prefill_plan)
        : owner(std::move(retained)), guard(std::make_shared<HipSetupQuiescenceGuard>()),
          serial(std::move(compiled)), model(std::move(contract)), config(std::move(cfg)),
          resources(original), session(live_session), scope(stream), use_prepared(prepared_mode),
          catalog(model, config),prefix_cache(std::move(pool)) {
        serving_require(owner && model.vocab_size && model.vocab_size <= std::numeric_limits<std::uint32_t>::max() &&
            model.vocab_size <= std::numeric_limits<std::size_t>::max()/sizeof(float) &&
            model.vocab_size == serial.vocab_size && serial.logical_rows == 1 && serial.request_concurrency == 1,
            "serving backend requires retained owners and one canonical uint32-vocabulary C1 plan");
        serving_require(config.deferred_moe_status,"serving backend requires an explicit per-session D1 record");
        if(prefill_plan && prefix_cache) {
            serving_require(prefill_plan->state_tiled.has_value()==prefill_plan->terminal_tiled.has_value(),
                "serving prefix recipe requires the same compiled plan family for both intents");
            prefix_recipe.emplace(prefill_plan->state_tiled ?
                SerialPrefixProducerRecipe::known_prefill(serial,*prefill_plan->state_tiled,*prefill_plan->terminal_tiled) :
                SerialPrefixProducerRecipe::known_window(serial,prefill_plan->state_window,prefill_plan->terminal_window));
        }
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
        cohort_shared={static_cast<std::uint32_t>(model.hidden_size),static_cast<std::uint32_t>(model.shared_expert_intermediate_size),
            config.activation,config.prefetch_q8_shared_expert,config.shared_expert_prefetch_layers,config.shared_expert_prefetch_stats};
        cohort_shared_fallback=cohort_shared;cohort_shared_fallback.prefetch_q8=false;cohort_shared_fallback.prefetch_eligible_layers.clear();
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
        if(prefill_plan) prefill=std::make_unique<ServingPrefillRuntime>(std::move(prefill_plan),serial,model,config,
            resources,scope,owner,guard);
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
        serving_require(!cohort_pending,"serving backend already has a cohort target in flight");
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
    std::shared_ptr<HipServingPrefixCache> prefix_cache,std::shared_ptr<const HipServingPrefillPlan> prefill)
    : impl_(std::make_unique<Impl>(std::move(serial), std::move(model), std::move(config), resources,
        session, scope, std::move(owner), prepared_eager,std::move(prefix_cache),std::move(prefill))) {}
HipServingGreedyBackend::~HipServingGreedyBackend() = default;
std::uint64_t HipServingGreedyBackend::output_reservation_bytes(std::uint32_t vocabulary) {
    return HipGreedyOutput::required_scratch_bytes(vocabulary);
}

HipServingPrefillPlan HipServingGreedyBackend::prepare_prefill(const ModelContract& model,
    const CompiledSerialExecution& serial,const ModelManifest& manifest,const ExecutionTensorAccessPlan& access,
    std::uint32_t rows,std::uint64_t workspace_budget,std::uint64_t metadata_budget) {
    serving_require(model.validate().empty() && model.vocab_size && model.vocab_size<=UINT32_MAX &&
        serial.logical_rows==1U && serial.request_concurrency==1U && !serial.device_graph_inputs,
        "serving prefill source must be a valid eager C1 uint32-vocabulary model");
    serving_require((rows==2U || rows==4U || rows==8U || (rows>=16U && rows<=2048U)) && workspace_budget && metadata_budget,
        "serving prefill requires closed rows2/4/8 or16..2048 and explicit positive payload budgets");
    HipServingPrefillPlan p;p.rows=rows;p.alignment=serial.arena.alignment;
    if(rows>=16U) {
        p.state_tiled=compile_known_prefix_prefill(model,serial,manifest,access,
            {rows,true,true,true,false,KnownWindowOutputPolicy::state_only,workspace_budget,metadata_budget,false,false,false,false,false,false,false});
        p.terminal_tiled=compile_known_prefix_prefill(model,serial,manifest,access,
            {rows,true,true,true,false,KnownWindowOutputPolicy::terminal_row,workspace_budget,metadata_budget,false,false,false,false,false,false,false});
        p.workspace_bytes=std::max(p.state_tiled->arena.physical.total_bytes,p.terminal_tiled->arena.physical.total_bytes);
        p.metadata_payload_bytes=serving_add(p.state_tiled->host_metadata_payload_bytes,p.terminal_tiled->host_metadata_payload_bytes);
    } else {
        p.state_window=compile_known_prefix_window(model,serial,manifest,access,{true,true,true,rows,false,KnownWindowOutputPolicy::state_only});
        p.terminal_window=compile_known_prefix_window(model,serial,manifest,access,{true,true,true,rows,false,KnownWindowOutputPolicy::terminal_row});
        serving_require(known_window_has_promoted_steps(p.state_window),"serving prefill source admits no real promoted rows operator");
        p.workspace_bytes=std::max(p.state_window.arena.physical.total_bytes,p.terminal_window.arena.physical.total_bytes);
        p.metadata_payload_bytes=serving_add(arena_metadata(p.state_window.arena),arena_metadata(p.terminal_window.arena));
    }
    serving_require(p.workspace_bytes<=workspace_budget && p.metadata_payload_bytes<=metadata_budget,
        "serving prefill workspace/known metadata payload exceeds explicit budget");
    p.runtime_payload_bytes=serving_mul(serving_mul(rows,serial.execution.steps.size()),sizeof(std::uint32_t));
    p.runtime_payload_bytes=serving_add(p.runtime_payload_bytes,sizeof(ExecutionBufferAddress)+sizeof(std::uint32_t));
    p.per_request_reservation_bytes=serving_add(serving_add(p.workspace_bytes,p.alignment-1U),16U);
    p.per_request_reservation_bytes=serving_add(p.per_request_reservation_bytes,output_reservation_bytes(static_cast<std::uint32_t>(model.vocab_size)));
    p.per_request_reservation_bytes=serving_add(p.per_request_reservation_bytes,serving_add(p.metadata_payload_bytes,p.runtime_payload_bytes));
    return p;
}
std::uint32_t HipServingGreedyBackend::prefill_chunk_rows() const noexcept {return impl_->prefill?impl_->prefill->plan->rows:0U;}
HipServingPrefillStats HipServingGreedyBackend::prefill_stats() const noexcept {return impl_->prefill?impl_->prefill->counters:HipServingPrefillStats{};}
std::optional<std::uint32_t> HipServingGreedyBackend::execute_prefill_chunk(std::span<const std::uint32_t> tokens,
    std::uint64_t position,std::uint64_t vocabulary,bool terminal_output) {
    auto& s=*impl_;serving_require(s.prefill && tokens.size()==s.prefill->plan->rows && vocabulary==s.model.vocab_size,
        "serving prefill call differs from admitted native rows/vocabulary");
    s.admit(tokens.front(),position);
    serving_require(position<=s.resources.session_state_layout->context_tokens &&
        tokens.size()<=s.resources.session_state_layout->context_tokens-position &&
        tokens.size()<=UINT64_MAX-s.session.committed_tokens,"serving prefill exceeds admitted state frontier");
    for(std::size_t i=0;i<tokens.size();++i) {
        serving_require(tokens[i]<vocabulary,"serving prefill input token exceeds vocabulary");
        if(s.prefix_started && position+i<s.request_prefix.size())
            serving_require(tokens[i]==s.request_prefix[static_cast<std::size_t>(position+i)],
                "serving prefill differs from immutable known-prefix cache witness");
    }
    serving_require(!terminal_output || s.epoch!=UINT64_MAX,"serving prefill selection epoch exhausted");
    std::optional<std::uint32_t> selected;s.selected_valid=false;
    try {
        const auto validate=[&] {
            if(terminal_output) selected=s.prefill->output->select(s.prefill->logits,static_cast<std::uint32_t>(vocabulary),
                position+tokens.size()-1U,++s.epoch);
        };
        auto& native=terminal_output?*s.prefill->terminal:*s.prefill->state;
        const auto result=native.execute_validated(s.session,tokens,validate);
        s.prefill->count(s.prefill->counters.chunks);s.prefill->count(s.prefill->counters.consumed_rows,tokens.size());
        s.prefill->count(terminal_output?s.prefill->counters.terminal_chunks:s.prefill->counters.state_only_chunks);
        s.prefill->count(s.prefill->counters.r2_dispatches,result.r2_dispatches);
        s.prefill->count(s.prefill->counters.shared_rows_dispatches,result.shared_rows_dispatches);
        s.prefill->count(s.prefill->counters.gdn_rows_dispatches,result.gdn_rows_dispatches);
        s.prefill->count(s.prefill->counters.omitted_lm_heads,result.omitted_lm_heads);
        if(selected) {s.selected_valid=true;s.selected_token=*selected;s.selected_position=position+tokens.size()-1U;}
        return selected;
    } catch(...) {
        s.broken=true;s.guard->recover_or_fail_stop();s.session.health=SerialSessionHealth::poisoned;
        s.session.arithmetic_lineage.tainted=true;throw;
    }
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
            *s.resources.session_state_memory,transfer,s.prefix_recipe?&*s.prefix_recipe:nullptr);
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
        s.prefix_cache->capture(s.model,prefix,pending,s.session,*s.resources.session_state_layout,*s.resources.session_state_memory,transfer,s.prefix_recipe?&*s.prefix_recipe:nullptr);
        s.complete();s.capture_attempted=true;
    } catch(...) {
        // The prefix model calls already committed. Never leave that consumed
        // session healthy while reporting a failed cache-copy request.
        s.broken=true;s.guard->recover_or_fail_stop();s.session.health=SerialSessionHealth::poisoned;
        s.session.arithmetic_lineage.tainted=true;throw;
    }
}
HipServingGreedyBackend::CohortKey HipServingGreedyBackend::cohort_key() const noexcept {
    return {impl_->serial.execution_identity,impl_->resources.model_weights,&impl_->session,impl_->guard->device_index()};
}
HipServingCohortMember HipServingGreedyBackend::admit_cohort_target(const SerialGreedyTargetInput& target) {
    serving_require(!impl_->prefill,"serving native prefill cannot enter a tensor cohort");
    impl_->admit(target.token,target.position);
    serving_require(target.ticket && target.vocab_size==impl_->model.vocab_size &&
        (target.state_only || impl_->epoch!=UINT64_MAX),"serving cohort ticket, vocabulary or selector epoch differs");
    return {&impl_->serial,&impl_->model,&impl_->config,&impl_->resources,&impl_->session,
        &impl_->launch_plan,impl_->direct.get(),impl_->scope,impl_->owner,impl_->guard,
        &impl_->cohort_shared,&impl_->cohort_shared_fallback};
}
void HipServingGreedyBackend::begin_cohort_target() noexcept {
    impl_->cohort_pending=true; impl_->selected_valid=false;
}
std::optional<std::uint32_t> HipServingGreedyBackend::complete_cohort_target(const SerialGreedyTargetInput& target) {
    serving_require(impl_->cohort_pending,"serving cohort completion has no pending target");
    impl_->complete();
    if(target.state_only) return {};
    return impl_->output->select(impl_->logits,static_cast<std::uint32_t>(target.vocab_size),target.position,++impl_->epoch);
}
void HipServingGreedyBackend::publish_cohort_target(const SerialGreedyTargetInput& target,
    std::optional<std::uint32_t> selected) noexcept {
    impl_->cohort_pending=false;
    impl_->selected_valid=selected.has_value();
    if(selected) { impl_->selected_token=*selected; impl_->selected_position=target.position; }
}
void HipServingGreedyBackend::poison_cohort_target() noexcept {
    impl_->broken=true; impl_->cohort_pending=false; impl_->selected_valid=false;
    if(impl_->resources.ple_lookup && impl_->resources.ple_prefetch)
        cancel_ple_lookup_prefetch(*impl_->resources.ple_lookup,*impl_->resources.ple_prefetch);
    // The original device is established even on a different scheduler worker.
    try { impl_->enter_worker(); } catch(...) { std::_Exit(3); }
    impl_->guard->recover_or_fail_stop();
    impl_->session.health=SerialSessionHealth::poisoned;
    impl_->session.arithmetic_lineage.tainted=true;
}
bool HipServingGreedyBackend::failed() const noexcept { return impl_->broken; }
int HipServingGreedyBackend::device_index() const noexcept { return impl_->guard->device_index(); }

} // namespace strix
