#include "strix/hip_serving_cohort.hpp"
#include "strix/hip_serving_greedy_backend.hpp"
#include "strix/hip_shared_serving_cohort.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/runtime_allocator.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
namespace strix {
namespace {
void cohort_require(bool value,const char* message) { if(!value) throw std::invalid_argument(message); }
void cohort_check(hipError_t error,const char* message) {
    if(error!=hipSuccess) throw std::runtime_error(std::string(message)+": "+hipGetErrorString(error));
}
void cohort_observe(SerialGreedyStepBackend& backend,const std::function<void()>& call) {
    if(auto* observer=dynamic_cast<ServingCohortObservation*>(&backend)) observer->observe_serving_target(call);
    else call();
}
struct CohortRange { std::uintptr_t address; std::uint64_t bytes; };
bool cohort_overlap(CohortRange a,CohortRange b) {
    cohort_require(a.address && a.bytes && a.bytes<=UINTPTR_MAX-a.address &&
        b.address && b.bytes && b.bytes<=UINTPTR_MAX-b.address,"serving cohort retained range overflows");
    return a.address<b.address+b.bytes && b.address<a.address+a.bytes;
}
ExecutionDispatchContext cohort_context(const HipServingCohortMember& member,
    const SerialGreedyTargetInput& input,std::size_t index) {
    const auto& compiled=*member.serial; const auto& step=compiled.execution.steps[index];
    const auto& io=compiled.io.steps[index]; const auto& binding=compiled.bindings.bindings[index];
    ExecutionDispatchContext c;
    c.execution_identity=compiled.execution_identity; c.step_id=step.id; c.op=step.op; c.layer=step.layer;
    if(binding.kernel_id) c.kernel_id=*binding.kernel_id;
    c.reads=io.reads; c.writes=io.writes; c.temporaries=io.temporaries; c.arena=&compiled.arena;
    c.memory=member.resources->scratch; c.resources=member.resources;
    c.session_state_layout=member.resources->session_state_layout; c.session_state_memory=member.resources->session_state_memory;
    c.reads_state=step.reads_state; c.writes_state=step.writes_state;
    c.logical_rows=1; c.request_concurrency=1; c.input_token_id=input.token; c.position=input.position;
    return c;
}
const HipSharedExpertLauncherConfig& cohort_shared_config(const HipServingCohortMember& member,const ExecutionDispatchContext& c) {
    cohort_require(member.shared_config && member.shared_fallback_config,"serving cohort lacks immutable shared configs");
    return c.kernel_id && c.kernel_id->ends_with(shared_expert_prefetch::kernel_suffix)
        ? *member.shared_config : *member.shared_fallback_config;
}
} // namespace
struct HipServingCohort::Impl {
    std::shared_ptr<const void> model_owner;
    std::shared_ptr<HipSetupQuiescenceGuard> guard=std::make_shared<HipSetupQuiescenceGuard>();
    Resolver resolver;
    hipStream_t stream=nullptr;
    std::array<hipEvent_t,4> ready{};
    hipEvent_t done=nullptr;
    HipSharedServingStaging staging;
    AllocationGenerationSource generations;
    Stats counters;
    bool failed=false;
    Impl(std::shared_ptr<const void> owner,std::uint32_t rows,Resolver resolve)
        :model_owner(std::move(owner)),resolver(std::move(resolve)) {
        cohort_require(!current_hip_launch_context(),"serving cohort setup requires no active launch scope");
        cohort_require(model_owner!=nullptr,"serving cohort requires retained shared model owners");
        const auto bytes=HipServingCohort::requested_reservation_bytes(rows);
        staging.capacity_rows=rows; staging.bytes=bytes;
        // Exactly one fixed allocation for this retained owner; no rebinding or
        // reuse while any request/event can borrow it.
        staging.allocation_generation=generations.next();
        if(!resolver) resolver=[](SerialGreedyStepBackend& backend) { return dynamic_cast<HipServingGreedyBackend*>(&backend); };
        try {
            cohort_check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"create serving cohort stream");
            void* address=nullptr;
            cohort_check(hipMalloc(&address,static_cast<std::size_t>(bytes)),"allocate serving cohort staging");
            staging.address=reinterpret_cast<std::uintptr_t>(address);
            for(auto& event:ready) cohort_check(hipEventCreateWithFlags(&event,hipEventDisableTiming),"create serving cohort ready event");
            cohort_check(hipEventCreateWithFlags(&done,hipEventDisableTiming),"create serving cohort done event");
        } catch(...) { cleanup(); throw; }
    }
    void enter() const {
        cohort_check(hipSetDevice(guard->device_index()),"establish serving cohort original device");
        guard->require_original();
    }
    void cleanup() noexcept {
        if(!guard) return;
        if(hipSetDevice(guard->device_index())!=hipSuccess) std::_Exit(3);
        guard->recover_or_fail_stop();
        for(auto& event:ready) if(event) { if(hipEventDestroy(event)!=hipSuccess) std::_Exit(3); event=nullptr; }
        if(done) { if(hipEventDestroy(done)!=hipSuccess) std::_Exit(3); done=nullptr; }
        if(staging.address) { if(hipFree(reinterpret_cast<void*>(staging.address))!=hipSuccess) std::_Exit(3); staging.address=0; }
        if(stream) { if(hipStreamDestroy(stream)!=hipSuccess) std::_Exit(3); stream=nullptr; }
    }
    ~Impl() { cleanup(); }
    void count(std::uint64_t& value) noexcept { if(value==UINT64_MAX) counters.saturated=true; else ++value; }
    void native_range(CohortRange range) const {
        (void)cohort_overlap(range,range);
        hipPointerAttribute_t attribute{};
        cohort_check(hipPointerGetAttributes(&attribute,reinterpret_cast<const void*>(range.address)),"serving cohort native owner device");
        cohort_require(attribute.type==hipMemoryTypeDevice && !attribute.isManaged && attribute.device==guard->device_index(),
            "serving cohort mutable owners must belong to the original device");
        hipDeviceptr_t base{}; std::size_t bytes=0;
        cohort_check(hipMemGetAddressRange(&base,&bytes,reinterpret_cast<hipDeviceptr_t>(range.address)),"serving cohort native owner extent");
        const auto address=reinterpret_cast<std::uintptr_t>(base);
        cohort_require(range.address>=address && range.address-address<=bytes && range.bytes<=bytes-(range.address-address),
            "serving cohort mutable owner exceeds its actual native allocation");
    }
};
HipServingCohort::HipServingCohort(std::shared_ptr<const void> owner,std::uint32_t rows,Resolver resolver)
    :impl_(std::make_unique<Impl>(std::move(owner),rows,std::move(resolver))) {}
HipServingCohort::~HipServingCohort()=default;
std::uint64_t HipServingCohort::requested_reservation_bytes(std::uint32_t rows) { return hip_shared_serving_staging_bytes(rows); }
HipServingCohort::Stats HipServingCohort::stats() const noexcept { return impl_->counters; }
bool HipServingCohort::compatible(std::span<SerialGreedyStepBackend* const> backends) const noexcept {
    try {
        if(impl_->failed || (backends.size()!=2U && backends.size()!=4U) || backends.size()>impl_->staging.capacity_rows) return false;
        HipServingGreedyBackend::CohortKey first{};
        std::array<const HipServingGreedyBackend*,4> resolved{};
        std::array<const SerialSessionContext*,4> sessions{};
        for(std::size_t i=0;i<backends.size();++i) {
            if(!backends[i]) return false;
            const auto* native=impl_->resolver(*backends[i]);
            if(!native || native->failed()) return false;
            const auto key=native->cohort_key();
            if(key.device!=impl_->guard->device_index() || !key.execution_identity || !key.weights) return false;
            for(std::size_t j=0;j<i;++j) if(native==resolved[j] || key.session==sessions[j]) return false;
            resolved[i]=native; sessions[i]=key.session;
            if(!i) first=key;
            else if(key.execution_identity!=first.execution_identity || key.weights!=first.weights || key.device!=first.device) return false;
            for(std::size_t j=0;j<i;++j) if(backends[j]==backends[i]) return false;
        }
        return true;
    } catch(...) { return false; }
}
void HipServingCohort::execute(std::span<ServingCohortTarget> targets) {
    validate_serving_cohort_targets(targets);
    std::array<SerialGreedyStepBackend*,4> bases{};
    for(std::size_t row=0;row<targets.size();++row) bases[row]=targets[row].backend;
    cohort_require(compatible(std::span<SerialGreedyStepBackend* const>(bases.data(),targets.size())),
        "serving tensor cohort contains incompatible native backends");
    cohort_require(!current_hip_launch_context(),"serving cohort cannot borrow an active launch scope");
    auto& s=*impl_; s.enter();
    hipStreamCaptureStatus capture{};
    cohort_check(hipStreamIsCapturing(s.stream,&capture),"serving cohort capture query");
    cohort_require(capture==hipStreamCaptureStatusNone,"serving cohort rejects active capture");
    std::array<HipServingGreedyBackend*,4> native{};
    std::array<HipServingCohortMember,4> members{};
    std::array<ArithmeticLineage,4> lineage;
    std::array<ExecutionTelemetry*,4> telemetry{};
    std::array<ExecutionCorrelation,4> correlations{};
    std::array<PersistentStateGenerations,4> proposed{};
    std::array<ExecutionDispatchContext,4> contexts{};
    std::array<std::array<CohortRange,4>,4> writes{};
    for(std::size_t row=0;row<targets.size();++row) {
        native[row]=s.resolver(*targets[row].backend);
        cohort_observe(*targets[row].backend,[&] {
            members[row]=native[row]->admit_cohort_target(targets[row].input);
            telemetry[row]=current_execution_telemetry(); correlations[row]=current_execution_correlation();
        });
        auto& member=members[row]; const auto& state=*member.resources->session_state_memory;
        cohort_require(member.model->hidden_size==2560U && member.model->shared_expert_intermediate_size==640U &&
            member.config->activation==HipActivationStorage::bf16,"serving cohort shared profile is not canonical BF16");
        cohort_check(hipStreamIsCapturing(member.scope.stream,&capture),"serving request capture query");
        cohort_require(capture==hipStreamCaptureStatusNone,"serving cohort request is being captured");
        writes[row]={CohortRange{member.resources->scratch->arena_base,member.resources->scratch->arena_bytes},
            CohortRange{state.device_base,state.device_bytes},CohortRange{state.host_base,state.host_bytes},
            CohortRange{member.scope.deferred_moe_status.arena_base,member.scope.deferred_moe_status.arena_bytes}};
        for(std::size_t field=0;field<4U;++field) {
            if(field!=2U) s.native_range(writes[row][field]);
            (void)cohort_overlap(writes[row][field],writes[row][field]);
            cohort_require(!cohort_overlap(writes[row][field],{s.staging.address,s.staging.bytes}),"serving staging aliases a request owner");
            for(std::size_t prior=0;prior<field;++prior)
                cohort_require(!cohort_overlap(writes[row][field],writes[row][prior]),"serving request mutable owners alias");
            for(std::size_t previous=0;previous<row;++previous) for(const auto other:writes[previous])
                cohort_require(!cohort_overlap(writes[row][field],other),"serving cohort sessions share a mutable owner");
        }
        for(std::size_t previous=0;previous<row;++previous) {
            cohort_require(member.session!=members[previous].session && member.scope.stream!=members[previous].scope.stream &&
                member.resources->ple_lookup!=members[previous].resources->ple_lookup &&
                member.resources->ple_prefetch!=members[previous].resources->ple_prefetch,
                "serving cohort sessions share stream, history, PLE or session metadata");
        }
        for(const auto& weight:member.resources->model_weights->tensors) if(weight.address && weight.bytes) {
            cohort_require(!cohort_overlap({s.staging.address,s.staging.bytes},{weight.address,weight.bytes}),"serving staging aliases immutable weights");
            for(const auto range:writes[row]) cohort_require(!cohort_overlap(range,{weight.address,weight.bytes}),"serving mutable owner aliases weights");
        }
        EagerExecutionRequest request;
        request.input_token_id=targets[row].input.token; request.position=targets[row].input.position; request.state=member.session->state;
        validate_eager_execution_preflight(member.serial->execution,member.serial->bindings,member.serial->io,member.serial->arena,
            request,member.resources->session_state_layout,member.resources->session_state_memory,member.resources);
        lineage[row]=prepare_serial_row_lineage(member.session->arithmetic_lineage,request.position,
            member.serial->execution_identity,member.serial->kernel_plan_identity);
        proposed[row]=request.state;
    }
    // Every original native metadata callback and the new cross-request shared
    // admission run before PLE starts or any target mutation. No forged window.
    const auto steps=members[0].serial->execution.steps.size();
    for(std::size_t step=0;step<steps;++step) {
        for(std::size_t row=0;row<targets.size();++row) {
            auto& member=members[row];
            cohort_require(member.serial->execution.steps.size()==steps && member.launches->steps.size()==steps,
                "serving cohort step coverage differs");
            contexts[row]=cohort_context(member,targets[row].input,step);
            const auto& launch=member.launches->steps[step];
            cohort_require(launch.step_id==step && launch.op==contexts[row].op && launch.function && launch.preflight,
                "serving cohort requires complete original native admission callbacks");
            cohort_observe(*targets[row].backend,[&] {
                const HipLaunchScope scope(member.scope);
                launch.preflight(contexts[row],launch.user_data);
            });
        }
        if(contexts[0].op==SerialOp::shared_expert) {
            const auto& config=cohort_shared_config(members[0],contexts[0]);
            shared_expert_serving_cohort_preflight(std::span<const ExecutionDispatchContext>(contexts.data(),targets.size()),config,s.staging);
        }
    }
    bool submitted=false;
    for(std::size_t row=0;row<targets.size();++row) native[row]->begin_cohort_target();
    try {
        cohort_check(hipPeekAtLastError(),"serving cohort stale error before submission");
        for(std::size_t row=0;row<targets.size();++row) {
            const auto& member=members[row]; const auto& resources=*member.resources;
            if(resources.ple_lookup && resources.ple_hash_contract && resources.ple_prefetch) {
                const auto found=std::find_if(member.serial->execution.steps.begin(),member.serial->execution.steps.end(),
                    [](const auto& step){return step.op==SerialOp::ple;});
                cohort_require(found!=member.serial->execution.steps.end(),"serving cohort PLE resources have no PLE step");
                start_ple_lookup_prefetch(*resources.ple_hash_contract,found->id,targets[row].input.token,
                    targets[row].input.position,*resources.model_manifest,*resources.model_weights,*resources.tensor_access,
                    *resources.session_state_layout,*resources.session_state_memory,*resources.ple_lookup,*resources.ple_prefetch);
            }
        }
        for(std::size_t step=0;step<steps;++step) {
            for(std::size_t row=0;row<targets.size();++row) contexts[row]=cohort_context(members[row],targets[row].input,step);
            if(contexts[0].op==SerialOp::shared_expert) {
                submitted=true;
                for(std::size_t row=0;row<targets.size();++row) {
                    cohort_check(hipEventRecord(s.ready[row],members[row].scope.stream),"record serving shared input ready");
                    cohort_check(hipStreamWaitEvent(s.stream,s.ready[row],0),"cohort waits for request input");
                }
                HipLaunchContext cohort_scope; cohort_scope.stream=s.stream; // No request D1 publisher.
                {
                    const HipLaunchScope active(cohort_scope);
                    const auto& config=cohort_shared_config(members[0],contexts[0]);
                    shared_expert_serving_cohort_launch(std::span<const ExecutionDispatchContext>(contexts.data(),targets.size()),config,s.staging);
                }
                cohort_check(hipEventRecord(s.done,s.stream),"record shared cohort scatter complete");
                for(std::size_t row=0;row<targets.size();++row)
                    cohort_check(hipStreamWaitEvent(members[row].scope.stream,s.done,0),"request waits before shared combine");
                s.count(s.counters.shared_tensor_steps);
            } else {
                for(std::size_t row=0;row<targets.size();++row) {
                    const auto& context=contexts[row];
                    const bool omitted=targets[row].input.state_only && step+1U==steps;
                    if(omitted) {
                        cohort_require(context.op==SerialOp::lm_head && context.reads_state==PersistentState::none &&
                            context.writes_state==PersistentState::none,"serving state-only may omit only the terminal stateless head");
                    } else {
                        cohort_observe(*targets[row].backend,[&] {
                            auto correlation=current_execution_correlation();
                            correlation.execution_identity=context.execution_identity; correlation.position=context.position;
                            correlation.row=0; correlation.rows=1; correlation.step_id=context.step_id;
                            correlation.op=context.op; correlation.layer=context.layer.value_or(UINT32_MAX);
                            correlation.phase=ExecutionPhase::operator_step;
                            const ExecutionCorrelationScope activity(correlation);
                            const HipLaunchScope active(members[row].scope);
                            submitted=true;
                            if(members[row].serial->bindings.bindings[step].host_or_runtime_step)
                                members[row].dispatcher->dispatch_runtime_step(context);
                            else members[row].dispatcher->dispatch_kernel(context);
                            s.count(s.counters.fallback_r1_steps);
                        });
                    }
                }
            }
            for(std::size_t row=0;row<targets.size();++row) proposed[row].publish_write(contexts[row].writes_state);
        }
        for(std::size_t row=0;row<targets.size();++row) {
            const auto& resources=*members[row].resources;
            cohort_require(!resources.ple_prefetch || !resources.ple_prefetch->active,"serving cohort left an unconsumed PLE lookup");
            cohort_observe(*targets[row].backend,[&] {
                targets[row].selected=native[row]->complete_cohort_target(targets[row].input);
            });
        }
        cohort_check(hipStreamQuery(s.stream),"prove serving cohort stream complete before publication");
        // No fallible operation after this publication boundary. All statuses,
        // selectors and both request/cohort stream dependencies have completed.
        for(std::size_t row=0;row<targets.size();++row) {
            auto& session=*members[row].session;
            session.state=proposed[row]; session.arithmetic_lineage=std::move(lineage[row]);
            ++session.next_position; ++session.committed_tokens;
            native[row]->publish_cohort_target(targets[row].input,targets[row].selected);
            // Native telemetry callbacks are noexcept and retained by each
            // request owner. Do not invoke the fallible wrapper callback here.
            auto correlation=correlations[row];correlation.execution_identity=session.execution_identity;
            correlation.position=targets[row].input.position;correlation.row=0;correlation.rows=1;
            correlation.step_id=UINT32_MAX;correlation.phase=ExecutionPhase::commit;
            const ExecutionTelemetryBinding activity(telemetry[row],correlation);
            observe_execution_activity(ExecutionActivityKind::token_commit);
        }
        s.count(s.counters.cohorts);
        for(std::size_t row=0;row<targets.size();++row) s.count(s.counters.targets);
    } catch(...) {
        if(submitted) { s.failed=true; s.guard->recover_or_fail_stop(); }
        for(std::size_t row=0;row<targets.size();++row) {
            const auto& resources=*members[row].resources;
            if(resources.ple_lookup && resources.ple_prefetch) cancel_ple_lookup_prefetch(*resources.ple_lookup,*resources.ple_prefetch);
            targets[row].selected.reset();
            if(submitted) native[row]->poison_cohort_target();
            else native[row]->publish_cohort_target(targets[row].input,{});
        }
        throw;
    }
}
} // namespace strix
