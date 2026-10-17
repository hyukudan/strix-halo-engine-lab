#include "strix/known_prefix_prefill.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/execution_telemetry.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/runtime_launchers.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
void require(bool value,const char* message) {if(!value) throw std::invalid_argument(message);}
std::uint64_t add(std::uint64_t a,std::uint64_t b,const char* message) {
    if(b>UINT64_MAX-a) throw std::overflow_error(message);
    return a+b;
}
constexpr std::array flags{PersistentState::attention_kv,PersistentState::recurrent_matrix,
    PersistentState::convolution_history,PersistentState::sparse_index,PersistentState::ngram_history,PersistentState::position_history};
bool reads_position(PersistentState state) {return (static_cast<std::uint32_t>(state)&static_cast<std::uint32_t>(PersistentState::position_history))!=0;}
}
KnownWindowResult execute_known_prefix_prefill_eager(const CompiledKnownPrefixPrefill& plan,
    const CompiledSerialExecution& serial,const ExecutionRuntimeResources& serial_resources,
    const ExecutionMemoryBinding& memory,SerialSessionContext& session,std::span<const std::uint32_t> tokens,
    const CompiledLaunchPlan& launch,std::span<const KnownWindowR2Entry> pairs,
    std::span<const KnownWindowRowsEntry> wide,const KnownWindowExecutionHooks& hooks) {
    validate_compiled_known_prefix_prefill(plan,serial);
    const auto rows=plan.logical_rows;const auto steps=serial.execution.steps.size();
    require(tokens.size()==rows,"known prefill token count differs from compiled T");
    require(hooks.preflight_scope&&hooks.complete&&hooks.quiesce_or_fail_stop,"known prefill requires scope/completion/fail-stop hooks");
    if(session.health!=SerialSessionHealth::ready) throw std::logic_error("known prefill requires a ready serial session");
    require(session.execution_identity==serial.execution_identity&&session.execution_identity,"known prefill source session identity differs");
    validate_execution_runtime_resources(serial.execution_identity,serial.arena,serial_resources,session.next_position);
    const auto identities=build_runtime_execution_identities(serial.execution_identity,serial.arena,serial_resources);
    require(session.runtime_compatibility_identity==identities.compatibility_identity&&
        session.state_layout_identity==serial_resources.session_state_layout->layout_identity&&
        plan.tile8.manifest_layout_identity==serial_resources.model_manifest->layout_identity&&
        plan.tile8.tensor_access_plan_identity==serial_resources.tensor_access->plan_identity,"known prefill resource/session identity differs");
    const auto post=add(session.next_position,rows,"known prefill position overflow");
    const auto committed=add(session.committed_tokens,rows,"known prefill committed-token overflow");
    if(post>serial.max_position_embeddings||post>serial_resources.session_state_layout->context_tokens)
        throw std::out_of_range("known prefill exceeds allocated context capacity");
    for(auto token:tokens) require(serial.vocab_size&&token<serial.vocab_size,"known prefill input token exceeds vocabulary");
    auto lineage=prepare_prefill_chunk_lineage(session.arithmetic_lineage,session.next_position,post,
        plan.prefill_execution_identity,plan.prefill_kernel_plan_identity,plan.prefill_execution_identity);
    validate_execution_memory_binding(plan.prefill_execution_identity,plan.arena.physical,memory);
    const auto disjoint=[&](std::uintptr_t base,std::uint64_t bytes) {
        if(!bytes) return;
        require(base&&bytes<=UINTPTR_MAX-base,"known prefill borrowed owner extent overflows");
        require(base>=memory.arena_base+memory.arena_bytes||memory.arena_base>=base+bytes,
            "known prefill scratch aliases retained serial/state/weight storage");
    };
    disjoint(serial_resources.scratch->arena_base,serial_resources.scratch->arena_bytes);
    const auto& state_memory=*serial_resources.session_state_memory;
    disjoint(state_memory.device_base,state_memory.device_bytes);disjoint(state_memory.host_base,state_memory.host_bytes);
    for(const auto& tensor:serial_resources.model_weights->tensors) if(tensor.address) disjoint(tensor.address,tensor.bytes);
    require(!serial_resources.ple_prefetch||!serial_resources.ple_prefetch->active,"known prefill cannot borrow an active PLE lookup");
    require(launch.execution_identity==plan.prefill_execution_identity&&launch.plan_identity&&launch.steps.size()==steps,
        "known prefill requires a fresh source launcher plan");
    for(std::size_t i=0;i<steps;++i) {
        const auto& actual=launch.steps[i];const auto& expected=serial.execution.steps[i];
        require(actual.step_id==expected.id&&actual.op==expected.op&&actual.function&&actual.preflight&&actual.implementation_identity&&
            actual.runtime_step==serial.bindings.bindings[i].host_or_runtime_step,"known prefill launcher metadata or preflight differs");
        if(reads_position(expected.reads_state)) {
            const bool position=expected.id==1U&&expected.op==SerialOp::position_prepare&&actual.runtime_step&&
                actual.implementation_identity==position_prepare_launcher_identity&&actual.function==&launch_position_prepare;
            const bool qsa=expected.op==SerialOp::indexed_attention&&!actual.runtime_step&&
                actual.implementation_identity!=position_prepare_launcher_identity&&actual.implementation_identity!=position_verify_window_launcher_identity&&
                std::find(hooks.position_reader_implementations.begin(),hooks.position_reader_implementations.end(),actual.implementation_identity)!=hooks.position_reader_implementations.end();
            require(position||qsa,"known prefill position reader is not admitted");
        }
    }
    std::array<const KnownWindowR2Entry*,known_window_r2_variant_slots> pair_by_id{};
    std::array<const KnownWindowRowsEntry*,known_window_r2_variant_slots> wide_by_id{};
    for(const auto& entry:pairs) {
        const auto id=static_cast<std::uint32_t>(entry.variant);
        require(id&&id<=3U&&!pair_by_id[id]&&entry.config&&entry.preflight&&entry.launch,"known prefill pair entry table is ambiguous or invalid");
        pair_by_id[id]=&entry;
    }
    for(const auto& entry:wide) {
        const auto id=static_cast<std::uint32_t>(entry.variant);
        require(id>=4U&&id<known_window_r2_variant_slots&&!wide_by_id[id]&&entry.config&&entry.preflight&&entry.launch,
            "known prefill wide entry table is ambiguous or invalid");
        wide_by_id[id]=&entry;
    }
    ExecutionRuntimeResources resources=serial_resources;resources.scratch=&memory;
    const auto context=[&](std::size_t step,std::uint32_t row) {
        const auto& s=serial.execution.steps[step];const auto& io=serial.io.steps[step];const auto& binding=serial.bindings.bindings[step];
        ExecutionDispatchContext c;c.execution_identity=plan.prefill_execution_identity;c.step_id=s.id;c.op=s.op;c.layer=s.layer;
        c.reads=io.reads;c.writes=io.writes;c.temporaries=io.temporaries;c.arena=&plan.arena.row_view[row];c.memory=&memory;
        c.session_state_layout=resources.session_state_layout;c.session_state_memory=resources.session_state_memory;c.resources=&resources;
        c.reads_state=s.reads_state;c.writes_state=s.writes_state;c.position=session.next_position+row;c.input_token_id=tokens[row];
        if(!binding.host_or_runtime_step) c.kernel_id=std::string_view(*binding.kernel_id);
        return c;
    };
    // Full T admission is streamed on the host. No T*steps context table or T
    // copies of compiled topology/catalogs are created.
    validate_execution_runtime_resources(plan.prefill_execution_identity,plan.arena.physical,resources,session.next_position);
    EagerExecutionRequest request;request.input_token_id=tokens[0];request.position=session.next_position;request.state=session.state;
    validate_eager_execution_preflight(serial.execution,serial.bindings,serial.io,plan.arena.row_view[0],request,
        resources.session_state_layout,resources.session_state_memory,&resources);
    // Topology/IO/state shape is identical for every R1 view. The immutable
    // compiled T arena validation above checks ALL row offsets, sizes and uses;
    // global token/capacity checks cover every request row. Actual per-row deep
    // launcher admission below still resolves each live view before mutation.
    require(steps&&serial.execution.steps.back().op==SerialOp::lm_head,"known prefill requires a complete terminal head");
    require(steps<=UINT32_MAX/rows,"known prefill logical work counter overflows");
    KnownWindowResult result;result.execution.state=session.state;
    result.logit_row_indices=plan.logit_row_indices;result.logits.resize(plan.logit_row_indices.size());
    result.omitted_lm_heads=rows-static_cast<std::uint32_t>(plan.logit_row_indices.size());
    result.logical_row_steps=static_cast<std::uint32_t>(rows*steps)-result.omitted_lm_heads;
    result.execution.completion_order.reserve(rows*steps);
    for(std::uint32_t row=0;row<rows;++row) {
        const auto c=context(steps-1U,row);require(c.writes.size()==1U,"known prefill head needs one output");
        const auto output=resolve_dispatch_write(c,0);
        require(serial.vocab_size<=UINT64_MAX/sizeof(float)&&output.bytes>=serial.vocab_size*sizeof(float),"known prefill logit row is undersized");
        const auto found=std::find(plan.logit_row_indices.begin(),plan.logit_row_indices.end(),row);
        if(found!=plan.logit_row_indices.end()) result.logits[static_cast<std::size_t>(found-plan.logit_row_indices.begin())]=output;
    }
    const auto position=resolve_dispatch_state(context(0,0),SessionStateComponent::position_history);
    require(position.storage==SessionStateStorage::host&&position.address&&position.bytes>=8U&&position.bytes<=UINTPTR_MAX-position.address,
        "known prefill requires bounded host position history");
    const auto materialize=[&](const ExecutionDispatchContext& c,std::uint64_t value) {
        const auto current=resolve_dispatch_state(c,SessionStateComponent::position_history);
        require(current.storage==position.storage&&current.address==position.address&&current.bytes==position.bytes&&
            current.allocation_generation==position.allocation_generation,"known prefill position binding changed after admission");
        std::memcpy(reinterpret_cast<void*>(current.address),&value,sizeof(value));
    };
    std::array<std::uint64_t,6> writes{};
    hooks.preflight_scope();
    for(std::size_t step=0;step<steps;++step) {
        for(std::size_t f=0;f<flags.size();++f) if(
            (static_cast<std::uint32_t>(serial.execution.steps[step].writes_state)&static_cast<std::uint32_t>(flags[f])))
            writes[f]=add(writes[f],rows,"known prefill state write count overflow");
        const auto mode=plan.tile8.mode[step];const auto id=static_cast<std::uint32_t>(plan.tile8.r2_variant[step]);
        const bool multi=mode!=KnownWindowStepMode::r1_row_loop;
        const bool rows_entry=known_window_rows_mode(mode);
        if(multi) require(!reads_position(serial.execution.steps[step].reads_state)&&(rows_entry?wide_by_id[id]!=nullptr:pair_by_id[id]!=nullptr),
            "known prefill selected closed implementation is missing");
        const auto width=multi?(rows_entry?8U:2U):1U;
        for(std::uint32_t first=0;first<rows;) {
            const auto count=rows-first>=width?width:1U;
            std::array<ExecutionDispatchContext,8> pack{};for(std::uint32_t r=0;r<count;++r) pack[r]=context(step,first+r);
            if(count==8U) wide_by_id[id]->preflight(std::span<const ExecutionDispatchContext>(pack.data(),8),wide_by_id[id]->config);
            else if(count==2U) pair_by_id[id]->preflight(std::span<const ExecutionDispatchContext,2>(pack.data(),2),pair_by_id[id]->config);
            else launch.steps[step].preflight(pack[0],launch.steps[step].user_data);
            first+=count;
        }
    }
    for(std::size_t f=0;f<flags.size();++f) require(writes[f]<=UINT64_MAX-session.state.generations[f],"known prefill state generation overflow");
    if(hooks.preflight_plan) hooks.preflight_plan();
    DirectLaunchDispatcher dispatcher(launch);
    const auto ple=std::find_if(serial.execution.steps.begin(),serial.execution.steps.end(),[](const auto& s){return s.op==SerialOp::ple;});
    if(resources.ple_lookup&&resources.ple_hash_contract&&resources.ple_prefetch) {
        require(ple!=serial.execution.steps.end(),"known prefill has PLE resources without PLE step");
        start_ple_lookup_prefetch(*resources.ple_hash_contract,ple->id,tokens[0],session.next_position,*resources.model_manifest,
            *resources.model_weights,*resources.tensor_access,*resources.session_state_layout,*resources.session_state_memory,
            *resources.ple_lookup,*resources.ple_prefetch);
    }
    try {
        for(std::size_t step=0;step<steps;++step) {
            const auto& source=serial.execution.steps[step];
            if(!result.execution.state.contains(source.reads_state)) throw std::logic_error("known prefill persistent state disappeared after admission");
            const auto mode=plan.tile8.mode[step];const auto id=static_cast<std::uint32_t>(plan.tile8.r2_variant[step]);
            const bool rows_entry=known_window_rows_mode(mode);
            const auto width=mode==KnownWindowStepMode::r1_row_loop?1U:(rows_entry?8U:2U);
            // All tiles of this source step run before advancing to its next
            // consumer/layer. Recurrent/conv/PLE/QSA rows remain increasing.
            for(std::uint32_t first=0;first<rows;) {
                const auto count=rows-first>=width?width:1U;
                std::array<ExecutionDispatchContext,8> pack{};for(std::uint32_t r=0;r<count;++r) pack[r]=context(step,first+r);
                auto correlation=current_execution_correlation();correlation.execution_identity=plan.prefill_execution_identity;
                correlation.position=pack[0].position;correlation.step_id=source.id;correlation.op=source.op;correlation.layer=source.layer.value_or(UINT32_MAX);
                correlation.rows=count;correlation.row=count==1U?first:UINT32_MAX;correlation.phase=ExecutionPhase::operator_step;
                ExecutionCorrelationScope observed(correlation);
                if(count==8U) {
                    wide_by_id[id]->launch(std::span<const ExecutionDispatchContext>(pack.data(),8),wide_by_id[id]->config);
                    if(mode==KnownWindowStepMode::shared_rows) ++result.shared_rows_dispatches;
                    else if(mode==KnownWindowStepMode::gdn_rows) ++result.gdn_rows_dispatches;
                    else if(mode==KnownWindowStepMode::hyper_mix_rows) ++result.hyper_mix_rows_dispatches;
                    else if(mode==KnownWindowStepMode::routed_expert_union) ++result.routed_expert_union_dispatches;
                } else if(count==2U) {
                    pair_by_id[id]->launch(std::span<const ExecutionDispatchContext,2>(pack.data(),2),pair_by_id[id]->config);++result.r2_dispatches;
                } else {
                    if(step+1U==steps&&!known_window_produces_logits(plan.output_policy,first,rows)) {++first;continue;}
                    if(reads_position(source.reads_state)) materialize(pack[0],pack[0].position);
                    if(first&&source.op==SerialOp::ple&&resources.ple_prefetch&&resources.ple_prefetch->active)
                        throw std::logic_error("known prefill first row did not consume PLE lookup");
                    if(launch.steps[step].runtime_step) dispatcher.dispatch_runtime_step(pack[0]);else dispatcher.dispatch_kernel(pack[0]);
                    ++result.row_loop_dispatches;
                }
                if(count>1U) ++result.r2_variant_dispatches[id];
                if(launch.steps[step].runtime_step) ++result.execution.runtime_steps;else ++result.execution.kernel_steps;
                ++result.execution.completed_steps;result.execution.completion_order.push_back(source.id);
                for(std::uint32_t r=0;r<count;++r) result.execution.state.publish_write(source.writes_state);
                first+=count;
            }
        }
        if(resources.ple_prefetch&&resources.ple_prefetch->active) throw std::logic_error("known prefill left PLE lookup unconsumed");
        hooks.complete();materialize(context(steps-1U,rows-1U),post-1U);
        std::uint64_t final_position=0;std::memcpy(&final_position,reinterpret_cast<const void*>(position.address),sizeof(final_position));
        if(final_position!=post-1U) throw std::logic_error("known prefill final host position differs from completed row frontier");
    } catch(...) {
        if(resources.ple_lookup&&resources.ple_prefetch) cancel_ple_lookup_prefetch(*resources.ple_lookup,*resources.ple_prefetch);
        session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
        try {hooks.quiesce_or_fail_stop();} catch(...) {std::terminate();}throw;
    }
    session.state=result.execution.state;session.next_position=post;session.committed_tokens=committed;session.arithmetic_lineage=std::move(lineage);
    auto correlation=current_execution_correlation();correlation.execution_identity=serial.execution_identity;correlation.position=post-rows;
    correlation.rows=rows;correlation.row=UINT32_MAX;correlation.phase=ExecutionPhase::commit;
    ExecutionCorrelationScope observed(correlation);observe_execution_activity(ExecutionActivityKind::token_commit,0,rows);
    return result;
}
} // namespace strix
