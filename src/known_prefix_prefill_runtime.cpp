#include "strix/known_prefix_prefill.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_telemetry.hpp"
#include "strix/known_window_cohort.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/runtime_launchers.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

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
    std::span<const KnownWindowRowsEntry> wide,const KnownWindowExecutionHooks& hooks,
    std::span<const KnownWindowRowsEntry> full) {
    // STRIX_PREFILL_TIMING=1: host milliseconds of each phase of every block on stderr.
    static const bool timing=[]{const char* v=std::getenv("STRIX_PREFILL_TIMING");return v&&v[0]=='1';}();
    auto mark=std::chrono::steady_clock::now();
    const auto lap=[&](const char* name) {
        if(!timing) return;
        const auto now=std::chrono::steady_clock::now();
        std::fprintf(stderr,"prefill host %-10s %8.2f ms\n",name,std::chrono::duration<double,std::milli>(now-mark).count());mark=now;
    };
    // A repeat block of the same owner and launch plan skips the static plan check and the
    // per-row launcher preflight (about 120 ms at T=64); every launcher still validates its
    // context before enqueueing, and tokens, positions, capacity and session are checked below.
    const void* preflight_key=&launch;
    const bool preflight_known=hooks.preflight_memo&&
        std::find(hooks.preflight_memo->begin(),hooks.preflight_memo->end(),preflight_key)!=hooks.preflight_memo->end();
    if(!preflight_known) validate_compiled_known_prefix_prefill(plan,serial);
    lap("plan");
    const auto rows=plan.logical_rows;const auto steps=serial.execution.steps.size();
    require(tokens.size()==rows,"known prefill token count differs from compiled T");
    require(hooks.preflight_scope&&hooks.complete&&hooks.quiesce_or_fail_stop,"known prefill requires scope/completion/fail-stop hooks");
    if(session.health!=SerialSessionHealth::ready) throw std::logic_error("known prefill requires a ready serial session");
    require(session.execution_identity==serial.execution_identity&&session.execution_identity,"known prefill source session identity differs");
    validate_execution_runtime_resources(serial.execution_identity,serial.arena,serial_resources,session.next_position);
    // Every row context binds these model objects: the launch's tensor lookups skip re-validating
    // the whole binding (about 1000 lookups per row, 61 % of the host time of a 4K prompt).
    std::optional<ValidatedModelBindingScope> binding_scope;
    if(!serial_resources.distributed_stage) binding_scope.emplace(*serial_resources.model_manifest,*serial_resources.model_weights);
    // Likewise each row's arena plan is indexed once instead of searched linearly per buffer.
    const ArenaLookupScope arena_lookup;
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
    std::array<const KnownWindowRowsEntry*,known_window_r2_variant_slots> full_by_id{};
    for(const auto& entry:full) {
        const auto id=static_cast<std::uint32_t>(entry.variant);
        require(id>=4U&&id<known_window_r2_variant_slots&&!full_by_id[id]&&entry.config&&entry.preflight&&entry.launch,
            "known prefill full-T entry table is ambiguous or invalid");
        full_by_id[id]=&entry;
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
    const bool headless=hooks.skip_head;
    if(!headless) {result.logit_row_indices=plan.logit_row_indices;result.logits.resize(plan.logit_row_indices.size());}
    result.omitted_lm_heads=rows-static_cast<std::uint32_t>(result.logit_row_indices.size());
    result.logical_row_steps=static_cast<std::uint32_t>(rows*steps)-result.omitted_lm_heads;
    result.execution.completion_order.reserve(rows*steps);
    for(std::uint32_t row=0;row<rows&&!headless;++row) {
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
    lap("admit");
    hooks.preflight_scope();
    lap("scope");
    std::array<double,32> mode_ms{};
    for(std::size_t step=0;step<steps;++step) {
        for(std::size_t f=0;f<flags.size();++f) if(
            (static_cast<std::uint32_t>(serial.execution.steps[step].writes_state)&static_cast<std::uint32_t>(flags[f])))
            writes[f]=add(writes[f],rows,"known prefill state write count overflow");
        const auto mode=plan.tile8.mode[step];const auto id=static_cast<std::uint32_t>(plan.tile8.r2_variant[step]);
        const bool multi=mode!=KnownWindowStepMode::r1_row_loop;
        const bool rows_entry=known_window_rows_mode(mode);
        // QSA rows read each row's own context position, never the host word.
        if(multi&&!((!reads_position(serial.execution.steps[step].reads_state)||mode==KnownWindowStepMode::qsa_rows)&&(rows_entry?wide_by_id[id]!=nullptr:pair_by_id[id]!=nullptr)))
            throw std::invalid_argument("known prefill selected closed implementation is missing (variant "+std::to_string(id)+
                ", mode "+std::to_string(static_cast<unsigned>(mode))+")");
        // Memoized blocks skip every step preflight, including the position-dependent QSA/PLE ones
        // (re-running those costs ~17 ms of idle GPU per 64-row block). Their launchers still admit
        // each context before enqueueing, so an out-of-capacity block fails during launch (session
        // poisoned) instead of before it; capacity and positions are checked above for every block.
        if(preflight_known||(headless&&step+1U==steps)) continue;
        const auto step_begin=timing?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        const auto step_done=[&]{
            if(timing) mode_ms[static_cast<std::size_t>(mode)%mode_ms.size()]+=
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-step_begin).count();
        };
        if(rows_entry&&full_by_id[id]&&rows%8U==0U) {
            std::vector<ExecutionDispatchContext> all(rows);for(std::uint32_t r=0;r<rows;++r) all[r]=context(step,r);
            // Position-independent full-T entries: the compiled T arena (validated above for every row) makes tile t's
            // views tile 0's plus t row strides, so the plan preflight admits the first and last two tiles; each
            // launch still admits all of its tiles before enqueueing that step. QSA and PLE depend on positions, and the
            // routed MoE workspace is sized for all T rows: those admit every tile here.
            const bool edges=mode!=KnownWindowStepMode::qsa_rows&&mode!=KnownWindowStepMode::ple_rows&&
                mode!=KnownWindowStepMode::routed_expert_union&&rows>32U;
            if(edges) {
                const std::span<const ExecutionDispatchContext> view(all);
                full_by_id[id]->preflight(view.first(16U),full_by_id[id]->config);
                full_by_id[id]->preflight(view.last(16U),full_by_id[id]->config);
            } else full_by_id[id]->preflight(all,full_by_id[id]->config);
            step_done();
            continue;
        }
        const auto width=multi?(rows_entry?8U:2U):1U;
        for(std::uint32_t first=0;first<rows;) {
            const auto count=rows-first>=width?width:1U;
            std::array<ExecutionDispatchContext,8> pack{};for(std::uint32_t r=0;r<count;++r) pack[r]=context(step,first+r);
            if(count==8U) wide_by_id[id]->preflight(std::span<const ExecutionDispatchContext>(pack.data(),8),wide_by_id[id]->config);
            else if(count==2U) pair_by_id[id]->preflight(std::span<const ExecutionDispatchContext,2>(pack.data(),2),pair_by_id[id]->config);
            else launch.steps[step].preflight(pack[0],launch.steps[step].user_data);
            first+=count;
        }
        step_done();
    }
    if(timing&&!preflight_known)
        for(std::size_t m=0;m<mode_ms.size();++m) if(mode_ms[m]>0.5)
            std::fprintf(stderr,"prefill preflight mode %2zu %8.2f ms\n",m,mode_ms[m]);
    for(std::size_t f=0;f<flags.size();++f) require(writes[f]<=UINT64_MAX-session.state.generations[f],"known prefill state generation overflow");
    lap("preflight");
    if(hooks.preflight_plan) hooks.preflight_plan();
    lap("metadata");
    if(hooks.preflight_memo&&!preflight_known) hooks.preflight_memo->push_back(preflight_key);
    DirectLaunchDispatcher dispatcher(launch);
    const auto ple=std::find_if(serial.execution.steps.begin(),serial.execution.steps.end(),[](const auto& s){return s.op==SerialOp::ple;});
    if(resources.ple_lookup&&resources.ple_hash_contract&&resources.ple_prefetch) {
        require(ple!=serial.execution.steps.end(),"known prefill has PLE resources without PLE step");
        start_ple_lookup_prefetch(*resources.ple_hash_contract,ple->id,tokens[0],session.next_position,*resources.model_manifest,
            *resources.model_weights,*resources.tensor_access,*resources.session_state_layout,*resources.session_state_memory,
            *resources.ple_lookup,*resources.ple_prefetch);
    }
    std::array<double,32> launch_ms{};std::array<std::uint32_t,32> launch_steps{};
    const auto observe=[&](std::size_t step) {
        if(!hooks.after_step) return;
        hooks.after_step(step,[&](std::uint32_t row) {
            require(row<rows,"known prefill observer row is outside the block");return context(step,row);});
    };
    try {
        for(std::size_t step=0;step<steps;++step) {
            const auto& source=serial.execution.steps[step];
            if(!result.execution.state.contains(source.reads_state)) throw std::logic_error("known prefill persistent state disappeared after admission");
            const auto mode=plan.tile8.mode[step];const auto id=static_cast<std::uint32_t>(plan.tile8.r2_variant[step]);
            // STRIX_PREFILL_TIMING=1: host enqueue time per step mode (admission inside each launch included).
            const auto launch_begin=timing?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            struct LaunchLap {
                bool on;std::chrono::steady_clock::time_point begin;double& total;std::uint32_t& count;
                ~LaunchLap(){if(on){total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();++count;}}
            } launch_lap{timing,launch_begin,launch_ms[static_cast<std::size_t>(mode)%launch_ms.size()],launch_steps[static_cast<std::size_t>(mode)%launch_steps.size()]};
            const bool rows_entry=known_window_rows_mode(mode);
            if(headless&&step+1U==steps) break;
            if(rows_entry&&full_by_id[id]&&rows%8U==0U) {
                std::vector<ExecutionDispatchContext> all(rows);for(std::uint32_t r=0;r<rows;++r) all[r]=context(step,r);
                auto correlation=current_execution_correlation();correlation.execution_identity=plan.prefill_execution_identity;
                correlation.position=all[0].position;correlation.step_id=source.id;correlation.op=source.op;correlation.layer=source.layer.value_or(UINT32_MAX);
                correlation.rows=rows;correlation.row=UINT32_MAX;correlation.phase=ExecutionPhase::operator_step;
                ExecutionCorrelationScope observed(correlation);
                full_by_id[id]->launch(all,full_by_id[id]->config);
                // Counted per 8-row tile, like the tile path, so receipts compare across both.
                const auto tiles=rows/8U;
                if(mode==KnownWindowStepMode::shared_rows) result.shared_rows_dispatches+=tiles;
                else if(mode==KnownWindowStepMode::gdn_rows) result.gdn_rows_dispatches+=tiles;
                else if(mode==KnownWindowStepMode::hyper_mix_rows) result.hyper_mix_rows_dispatches+=tiles;
                else if(mode==KnownWindowStepMode::routed_expert_union) result.routed_expert_union_dispatches+=tiles;
                result.r2_variant_dispatches[id]+=tiles;
                if(launch.steps[step].runtime_step) ++result.execution.runtime_steps;else ++result.execution.kernel_steps;
                ++result.execution.completed_steps;result.execution.completion_order.push_back(source.id);
                for(std::uint32_t r=0;r<rows;++r) result.execution.state.publish_write(source.writes_state);
                observe(step);
                continue;
            }
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
            observe(step);
        }
        if(resources.ple_prefetch&&resources.ple_prefetch->active) throw std::logic_error("known prefill left PLE lookup unconsumed");
        if(timing)
            for(std::size_t m=0;m<launch_ms.size();++m) if(launch_ms[m]>0.5)
                std::fprintf(stderr,"prefill launch mode %2zu %8.2f ms (%u steps)\n",m,launch_ms[m],launch_steps[m]);
        lap("launch");
        hooks.complete();lap("complete");materialize(context(steps-1U,rows-1U),post-1U);
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

std::vector<KnownWindowResult> execute_known_prefix_prefill_segments_eager(const CompiledKnownPrefixPrefill& plan,
    const CompiledSerialExecution& serial,const ExecutionMemoryBinding& memory,std::span<const KnownPrefillSegment> segments,
    const CompiledLaunchPlan& launch,std::span<const KnownWindowR2Entry> pairs,std::span<const KnownWindowRowsEntry> wide,
    const KnownWindowExecutionHooks& hooks,std::span<const KnownWindowRowsEntry> full,const KnownPrefillSegmentHooks& segment_hooks) {
    const auto count=segments.size();
    require(count>=1U,"segmented prefill requires a segment");
    require(hooks.preflight_scope&&hooks.complete&&hooks.quiesce_or_fail_stop,"segmented prefill requires scope/completion/fail-stop hooks");
    require(!hooks.preflight_only&&!hooks.before_step&&!hooks.after_step&&!hooks.record_actual_window_lineage,
        "segmented prefill takes its observers through the segment hooks");
    require(plan.output_policy!=KnownWindowOutputPolicy::all_rows,"segmented prefill does not deliver every row's logits");
    const void* preflight_key=&launch;
    const bool preflight_known=hooks.preflight_memo&&
        std::find(hooks.preflight_memo->begin(),hooks.preflight_memo->end(),preflight_key)!=hooks.preflight_memo->end();
    if(!preflight_known) validate_compiled_known_prefix_prefill(plan,serial);
    const auto steps=serial.execution.steps.size();
    require(steps&&serial.execution.steps.back().op==SerialOp::lm_head,"segmented prefill requires a complete terminal head");
    // Segment geometry: consecutive arena rows, each a multiple of 8 (every full-T entry and 8-row tile of a
    // segment then starts on the same tile boundary as in its own block).
    std::vector<std::uint32_t> offset(count),rows(count);
    std::uint32_t total=0;
    for(std::size_t s=0;s<count;++s) {
        const auto& g=segments[s];
        require(g.session&&g.resources&&g.tokens.size()>=8U&&g.tokens.size()%8U==0U&&g.tokens.size()<=plan.logical_rows,
            "segmented prefill segment needs a session, resources and 8..T rows in multiples of 8");
        require(!g.head||plan.output_policy!=KnownWindowOutputPolicy::state_only,"segmented prefill head on a state-only plan");
        offset[s]=total;rows[s]=static_cast<std::uint32_t>(g.tokens.size());
        require(rows[s]<=plan.logical_rows-total,"segmented prefill rows exceed the compiled T");
        total+=rows[s];
    }
    require(launch.execution_identity==plan.prefill_execution_identity&&launch.plan_identity&&launch.steps.size()==steps,
        "segmented prefill requires a fresh source launcher plan");
    for(std::size_t i=0;i<steps;++i) {
        const auto& actual=launch.steps[i];const auto& expected=serial.execution.steps[i];
        require(actual.step_id==expected.id&&actual.op==expected.op&&actual.function&&actual.preflight&&actual.implementation_identity&&
            actual.runtime_step==serial.bindings.bindings[i].host_or_runtime_step,"segmented prefill launcher metadata or preflight differs");
        if(reads_position(expected.reads_state)) {
            const bool position=expected.id==1U&&expected.op==SerialOp::position_prepare&&actual.runtime_step&&
                actual.implementation_identity==position_prepare_launcher_identity&&actual.function==&launch_position_prepare;
            const bool qsa=expected.op==SerialOp::indexed_attention&&!actual.runtime_step&&
                actual.implementation_identity!=position_prepare_launcher_identity&&actual.implementation_identity!=position_verify_window_launcher_identity&&
                std::find(hooks.position_reader_implementations.begin(),hooks.position_reader_implementations.end(),actual.implementation_identity)!=hooks.position_reader_implementations.end();
            require(position||qsa,"segmented prefill position reader is not admitted");
        }
    }
    std::array<const KnownWindowR2Entry*,known_window_r2_variant_slots> pair_by_id{};
    std::array<const KnownWindowRowsEntry*,known_window_r2_variant_slots> wide_by_id{},full_by_id{};
    for(const auto& entry:pairs) {
        const auto id=static_cast<std::uint32_t>(entry.variant);
        require(id&&id<=3U&&!pair_by_id[id]&&entry.config&&entry.preflight&&entry.launch,"segmented prefill pair entry table is ambiguous or invalid");
        pair_by_id[id]=&entry;
    }
    for(const auto& entry:wide) {
        const auto id=static_cast<std::uint32_t>(entry.variant);
        require(id>=4U&&id<known_window_r2_variant_slots&&!wide_by_id[id]&&entry.config&&entry.preflight&&entry.launch,
            "segmented prefill wide entry table is ambiguous or invalid");
        wide_by_id[id]=&entry;
    }
    for(const auto& entry:full) {
        const auto id=static_cast<std::uint32_t>(entry.variant);
        require(id>=4U&&id<known_window_r2_variant_slots&&!full_by_id[id]&&entry.config&&entry.preflight&&entry.launch,
            "segmented prefill full-T entry table is ambiguous or invalid");
        full_by_id[id]=&entry;
    }
    // Per segment: exactly the admission of its block alone, over its own resources with the block arena as scratch.
    std::vector<ExecutionRuntimeResources> resources(count);
    std::vector<std::uint64_t> post(count),committed(count);
    std::vector<ArithmeticLineage> lineage(count);
    const auto& lead=*segments[0].resources;
    require(lead.model_manifest&&lead.model_weights&&lead.tensor_access&&lead.session_state_layout&&!lead.distributed_stage,
        "segmented prefill requires whole-model resources");
    const auto disjoint=[&](std::uintptr_t base,std::uint64_t bytes) {
        if(!bytes) return;
        require(base&&bytes<=UINTPTR_MAX-base,"segmented prefill borrowed owner extent overflows");
        require(base>=memory.arena_base+memory.arena_bytes||memory.arena_base>=base+bytes,
            "segmented prefill scratch aliases retained serial/state/weight storage");
    };
    validate_execution_memory_binding(plan.prefill_execution_identity,plan.arena.physical,memory);
    for(const auto& tensor:lead.model_weights->tensors) if(tensor.address) disjoint(tensor.address,tensor.bytes);
    for(std::size_t s=0;s<count;++s) {
        const auto& g=segments[s];auto& session=*g.session;const auto& own=*g.resources;
        if(session.health!=SerialSessionHealth::ready) throw std::logic_error("segmented prefill requires ready serial sessions");
        require(session.execution_identity==serial.execution_identity&&session.execution_identity,"segmented prefill source session identity differs");
        require(own.model_manifest==lead.model_manifest&&own.model_weights==lead.model_weights&&own.tensor_access==lead.tensor_access&&
            own.session_state_layout&&own.session_state_memory&&own.scratch&&!own.distributed_stage&&
            own.session_state_layout->layout_identity==lead.session_state_layout->layout_identity,
            "segmented prefill sessions must share one model and one state layout");
        for(std::size_t o=0;o<s;++o)
            require(segments[o].session!=g.session&&segments[o].resources->session_state_memory!=own.session_state_memory,
                "segmented prefill segments must be distinct sessions");
        validate_execution_runtime_resources(serial.execution_identity,serial.arena,own,session.next_position);
        const auto identities=build_runtime_execution_identities(serial.execution_identity,serial.arena,own);
        require(session.runtime_compatibility_identity==identities.compatibility_identity&&
            session.state_layout_identity==own.session_state_layout->layout_identity&&
            plan.tile8.manifest_layout_identity==own.model_manifest->layout_identity&&
            plan.tile8.tensor_access_plan_identity==own.tensor_access->plan_identity,"segmented prefill resource/session identity differs");
        post[s]=add(session.next_position,rows[s],"segmented prefill position overflow");
        committed[s]=add(session.committed_tokens,rows[s],"segmented prefill committed-token overflow");
        if(post[s]>serial.max_position_embeddings||post[s]>own.session_state_layout->context_tokens)
            throw std::out_of_range("segmented prefill exceeds allocated context capacity");
        for(auto token:g.tokens) require(serial.vocab_size&&token<serial.vocab_size,"segmented prefill input token exceeds vocabulary");
        const auto numerical=g.lineage_execution_identity?g.lineage_execution_identity:plan.prefill_execution_identity;
        const auto reduction=g.lineage_kernel_plan_identity?g.lineage_kernel_plan_identity:plan.prefill_kernel_plan_identity;
        lineage[s]=prepare_prefill_chunk_lineage(session.arithmetic_lineage,session.next_position,post[s],numerical,reduction,numerical);
        disjoint(own.scratch->arena_base,own.scratch->arena_bytes);
        const auto& state_memory=*own.session_state_memory;
        disjoint(state_memory.device_base,state_memory.device_bytes);disjoint(state_memory.host_base,state_memory.host_bytes);
        require(!own.ple_prefetch||!own.ple_prefetch->active,"segmented prefill cannot borrow an active PLE lookup");
        resources[s]=own;resources[s].scratch=&memory;
        validate_execution_runtime_resources(plan.prefill_execution_identity,plan.arena.physical,resources[s],session.next_position);
        EagerExecutionRequest request;request.input_token_id=g.tokens[0];request.position=session.next_position;request.state=session.state;
        validate_eager_execution_preflight(serial.execution,serial.bindings,serial.io,plan.arena.row_view[offset[s]],request,
            resources[s].session_state_layout,resources[s].session_state_memory,&resources[s]);
    }
    std::optional<ValidatedModelBindingScope> binding_scope;
    binding_scope.emplace(*lead.model_manifest,*lead.model_weights);
    const ArenaLookupScope arena_lookup;
    const auto context=[&](std::size_t step,std::size_t s,std::uint32_t row) {
        const auto& source=serial.execution.steps[step];const auto& io=serial.io.steps[step];const auto& binding=serial.bindings.bindings[step];
        const auto& session=*segments[s].session;
        ExecutionDispatchContext c;c.execution_identity=plan.prefill_execution_identity;c.step_id=source.id;c.op=source.op;c.layer=source.layer;
        c.reads=io.reads;c.writes=io.writes;c.temporaries=io.temporaries;c.arena=&plan.arena.row_view[offset[s]+row];c.memory=&memory;
        c.session_state_layout=resources[s].session_state_layout;c.session_state_memory=resources[s].session_state_memory;c.resources=&resources[s];
        c.reads_state=source.reads_state;c.writes_state=source.writes_state;c.position=session.next_position+row;c.input_token_id=segments[s].tokens[row];
        if(!binding.host_or_runtime_step) c.kernel_id=std::string_view(*binding.kernel_id);
        return c;
    };
    std::vector<KnownWindowResult> results(count);
    std::vector<SessionStateAddress> positions(count);
    for(std::size_t s=0;s<count;++s) {
        auto& result=results[s];result.execution.state=segments[s].session->state;
        if(segments[s].head) {
            const auto c=context(steps-1U,s,rows[s]-1U);require(c.writes.size()==1U,"segmented prefill head needs one output");
            const auto output=resolve_dispatch_write(c,0);
            require(serial.vocab_size<=UINT64_MAX/sizeof(float)&&output.bytes>=serial.vocab_size*sizeof(float),"segmented prefill logit row is undersized");
            result.logit_row_indices={rows[s]-1U};result.logits={output};
        }
        result.omitted_lm_heads=rows[s]-static_cast<std::uint32_t>(result.logit_row_indices.size());
        result.logical_row_steps=static_cast<std::uint32_t>(rows[s]*steps)-result.omitted_lm_heads;
        result.execution.completion_order.reserve(rows[s]*steps);
        positions[s]=resolve_dispatch_state(context(0,s,0),SessionStateComponent::position_history);
        require(positions[s].storage==SessionStateStorage::host&&positions[s].address&&positions[s].bytes>=8U&&
            positions[s].bytes<=UINTPTR_MAX-positions[s].address,"segmented prefill requires bounded host position history");
    }
    const auto materialize=[&](std::size_t s,const ExecutionDispatchContext& c,std::uint64_t value) {
        const auto current=resolve_dispatch_state(c,SessionStateComponent::position_history);
        const auto& position=positions[s];
        require(current.storage==position.storage&&current.address==position.address&&current.bytes==position.bytes&&
            current.allocation_generation==position.allocation_generation,"segmented prefill position binding changed after admission");
        std::memcpy(reinterpret_cast<void*>(current.address),&value,sizeof(value));
    };
    const auto shared_step=[&](std::size_t step) {
        const auto mode=plan.tile8.mode[step];
        return segment_hooks.shared&&count>1U&&step+1U<steps&&known_window_rows_mode(mode)&&
            full_by_id[static_cast<std::uint32_t>(plan.tile8.r2_variant[step])]&&segment_hooks.shared(step);
    };
    const bool any_head=std::any_of(segments.begin(),segments.end(),[](const auto& g){return g.head;});
    // Admission: per segment as its block alone; shared steps also over all rows (the launch admits again).
    for(std::size_t step=0;step<steps;++step) {
        const auto mode=plan.tile8.mode[step];const auto id=static_cast<std::uint32_t>(plan.tile8.r2_variant[step]);
        const bool multi=mode!=KnownWindowStepMode::r1_row_loop;
        const bool rows_entry=known_window_rows_mode(mode);
        if(multi&&!((!reads_position(serial.execution.steps[step].reads_state)||mode==KnownWindowStepMode::qsa_rows)&&(rows_entry?wide_by_id[id]!=nullptr:pair_by_id[id]!=nullptr)))
            throw std::invalid_argument("segmented prefill selected closed implementation is missing (variant "+std::to_string(id)+")");
        if(step+1U==steps) require(mode==KnownWindowStepMode::r1_row_loop,"segmented prefill terminal head must be a row step");
        if(preflight_known||(step+1U==steps&&!any_head)) continue;
        if(shared_step(step)) {
            std::vector<ExecutionDispatchContext> all;all.reserve(total);
            for(std::size_t s=0;s<count;++s) for(std::uint32_t r=0;r<rows[s];++r) all.push_back(context(step,s,r));
            // The rows of several sessions: per-segment session and position rules (see known_window_cohort.hpp).
            const KnownWindowCohortScope cohort;
            full_by_id[id]->preflight(all,full_by_id[id]->config);
            continue;
        }
        for(std::size_t s=0;s<count;++s) {
            if(step+1U==steps) {
                if(segments[s].head) launch.steps[step].preflight(context(step,s,rows[s]-1U),launch.steps[step].user_data);
                continue;
            }
            if(rows_entry&&full_by_id[id]) {
                std::vector<ExecutionDispatchContext> all(rows[s]);for(std::uint32_t r=0;r<rows[s];++r) all[r]=context(step,s,r);
                // As a block alone: position-independent full-T entries admit their first and last two tiles here.
                const bool edges=mode!=KnownWindowStepMode::qsa_rows&&mode!=KnownWindowStepMode::ple_rows&&
                    mode!=KnownWindowStepMode::routed_expert_union&&rows[s]>32U;
                const std::span<const ExecutionDispatchContext> view(all);
                if(edges) {
                    full_by_id[id]->preflight(view.first(16U),full_by_id[id]->config);
                    full_by_id[id]->preflight(view.last(16U),full_by_id[id]->config);
                } else full_by_id[id]->preflight(view,full_by_id[id]->config);
                continue;
            }
            const auto width=multi?(rows_entry?8U:2U):1U;
            for(std::uint32_t first=0;first<rows[s];) {
                const auto n=rows[s]-first>=width?width:1U;
                std::array<ExecutionDispatchContext,8> pack{};for(std::uint32_t r=0;r<n;++r) pack[r]=context(step,s,first+r);
                if(n==8U) wide_by_id[id]->preflight(std::span<const ExecutionDispatchContext>(pack.data(),8),wide_by_id[id]->config);
                else if(n==2U) pair_by_id[id]->preflight(std::span<const ExecutionDispatchContext,2>(pack.data(),2),pair_by_id[id]->config);
                else launch.steps[step].preflight(pack[0],launch.steps[step].user_data);
                first+=n;
            }
        }
    }
    for(std::size_t s=0;s<count;++s)
        for(std::size_t f=0;f<flags.size();++f) {
            std::uint64_t writes=0;
            for(std::size_t step=0;step<steps;++step)
                if(static_cast<std::uint32_t>(serial.execution.steps[step].writes_state)&static_cast<std::uint32_t>(flags[f])) writes=add(writes,rows[s],"segmented prefill state write count overflow");
            require(writes<=UINT64_MAX-segments[s].session->state.generations[f],"segmented prefill state generation overflow");
        }
    hooks.preflight_scope();
    if(hooks.preflight_plan) hooks.preflight_plan();
    if(hooks.preflight_memo&&!preflight_known) hooks.preflight_memo->push_back(preflight_key);
    DirectLaunchDispatcher dispatcher(launch);
    const auto observe=[&](std::size_t step,std::size_t s) {
        if(!segment_hooks.after_step) return;
        segment_hooks.after_step(step,s,[&,step,s](std::uint32_t row) {
            require(row<rows[s],"segmented prefill observer row is outside the segment");return context(step,s,row);});
    };
    const auto record=[&](std::size_t s,std::size_t step,std::uint32_t n) {
        auto& result=results[s];const auto& source=serial.execution.steps[step];
        if(launch.steps[step].runtime_step) ++result.execution.runtime_steps;else ++result.execution.kernel_steps;
        ++result.execution.completed_steps;result.execution.completion_order.push_back(source.id);
        for(std::uint32_t r=0;r<n;++r) result.execution.state.publish_write(source.writes_state);
    };
    const auto count_tiles=[&](KnownWindowResult& result,KnownWindowStepMode mode,std::uint32_t id,std::uint32_t tiles) {
        if(mode==KnownWindowStepMode::shared_rows) result.shared_rows_dispatches+=tiles;
        else if(mode==KnownWindowStepMode::gdn_rows) result.gdn_rows_dispatches+=tiles;
        else if(mode==KnownWindowStepMode::hyper_mix_rows) result.hyper_mix_rows_dispatches+=tiles;
        else if(mode==KnownWindowStepMode::routed_expert_union) result.routed_expert_union_dispatches+=tiles;
        result.r2_variant_dispatches[id]+=tiles;
    };
    try {
        for(std::size_t step=0;step<steps;++step) {
            const auto& source=serial.execution.steps[step];
            for(std::size_t s=0;s<count;++s)
                if(!results[s].execution.state.contains(source.reads_state)) throw std::logic_error("segmented prefill persistent state disappeared after admission");
            const auto mode=plan.tile8.mode[step];const auto id=static_cast<std::uint32_t>(plan.tile8.r2_variant[step]);
            const bool rows_entry=known_window_rows_mode(mode);
            if(step+1U==steps) {
                // Terminal head: the last row of each segment that asked for it, as the row loop of its block.
                for(std::size_t s=0;s<count;++s) {
                    if(!segments[s].head) continue;
                    const auto c=context(step,s,rows[s]-1U);
                    if(reads_position(source.reads_state)) materialize(s,c,c.position);
                    if(launch.steps[step].runtime_step) dispatcher.dispatch_runtime_step(c);else dispatcher.dispatch_kernel(c);
                    ++results[s].row_loop_dispatches;record(s,step,1U);observe(step,s);
                }
                break;
            }
            if(shared_step(step)) {
                std::vector<ExecutionDispatchContext> all;all.reserve(total);
                for(std::size_t s=0;s<count;++s) for(std::uint32_t r=0;r<rows[s];++r) all.push_back(context(step,s,r));
                auto correlation=current_execution_correlation();correlation.execution_identity=plan.prefill_execution_identity;
                correlation.position=all[0].position;correlation.step_id=source.id;correlation.op=source.op;correlation.layer=source.layer.value_or(UINT32_MAX);
                correlation.rows=total;correlation.row=UINT32_MAX;correlation.phase=ExecutionPhase::operator_step;
                ExecutionCorrelationScope observed(correlation);
                {
                    const KnownWindowCohortScope cohort;
                    full_by_id[id]->launch(all,full_by_id[id]->config);
                }
                for(std::size_t s=0;s<count;++s) {count_tiles(results[s],mode,id,rows[s]/8U);record(s,step,rows[s]);observe(step,s);}
                continue;
            }
            for(std::size_t s=0;s<count;++s) {
                auto& result=results[s];
                auto correlation=current_execution_correlation();correlation.execution_identity=plan.prefill_execution_identity;
                correlation.step_id=source.id;correlation.op=source.op;correlation.layer=source.layer.value_or(UINT32_MAX);
                correlation.phase=ExecutionPhase::operator_step;
                if(rows_entry&&full_by_id[id]) {
                    std::vector<ExecutionDispatchContext> all(rows[s]);for(std::uint32_t r=0;r<rows[s];++r) all[r]=context(step,s,r);
                    correlation.position=all[0].position;correlation.rows=rows[s];correlation.row=UINT32_MAX;
                    ExecutionCorrelationScope observed(correlation);
                    full_by_id[id]->launch(all,full_by_id[id]->config);
                    count_tiles(result,mode,id,rows[s]/8U);record(s,step,rows[s]);observe(step,s);
                    continue;
                }
                const auto width=mode==KnownWindowStepMode::r1_row_loop?1U:(rows_entry?8U:2U);
                for(std::uint32_t first=0;first<rows[s];) {
                    const auto n=rows[s]-first>=width?width:1U;
                    std::array<ExecutionDispatchContext,8> pack{};for(std::uint32_t r=0;r<n;++r) pack[r]=context(step,s,first+r);
                    correlation.position=pack[0].position;correlation.rows=n;correlation.row=n==1U?first:UINT32_MAX;
                    ExecutionCorrelationScope observed(correlation);
                    if(n==8U) {
                        wide_by_id[id]->launch(std::span<const ExecutionDispatchContext>(pack.data(),8),wide_by_id[id]->config);
                        if(mode==KnownWindowStepMode::shared_rows) ++result.shared_rows_dispatches;
                        else if(mode==KnownWindowStepMode::gdn_rows) ++result.gdn_rows_dispatches;
                        else if(mode==KnownWindowStepMode::hyper_mix_rows) ++result.hyper_mix_rows_dispatches;
                        else if(mode==KnownWindowStepMode::routed_expert_union) ++result.routed_expert_union_dispatches;
                    } else if(n==2U) {
                        pair_by_id[id]->launch(std::span<const ExecutionDispatchContext,2>(pack.data(),2),pair_by_id[id]->config);++result.r2_dispatches;
                    } else {
                        if(reads_position(source.reads_state)) materialize(s,pack[0],pack[0].position);
                        if(launch.steps[step].runtime_step) dispatcher.dispatch_runtime_step(pack[0]);else dispatcher.dispatch_kernel(pack[0]);
                        ++result.row_loop_dispatches;
                    }
                    if(n>1U) ++result.r2_variant_dispatches[id];
                    record(s,step,n);
                    first+=n;
                }
                observe(step,s);
            }
        }
        for(std::size_t s=0;s<count;++s)
            if(resources[s].ple_prefetch&&resources[s].ple_prefetch->active) throw std::logic_error("segmented prefill left a PLE lookup unconsumed");
        hooks.complete();
        for(std::size_t s=0;s<count;++s) {
            materialize(s,context(steps-1U,s,rows[s]-1U),post[s]-1U);
            std::uint64_t final_position=0;std::memcpy(&final_position,reinterpret_cast<const void*>(positions[s].address),sizeof(final_position));
            if(final_position!=post[s]-1U) throw std::logic_error("segmented prefill final host position differs from completed row frontier");
        }
    } catch(...) {
        for(std::size_t s=0;s<count;++s) {
            if(resources[s].ple_lookup&&resources[s].ple_prefetch) cancel_ple_lookup_prefetch(*resources[s].ple_lookup,*resources[s].ple_prefetch);
            auto& session=*segments[s].session;session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
        }
        try {hooks.quiesce_or_fail_stop();} catch(...) {std::terminate();}throw;
    }
    for(std::size_t s=0;s<count;++s) {
        auto& session=*segments[s].session;
        session.state=results[s].execution.state;session.next_position=post[s];session.committed_tokens=committed[s];
        session.arithmetic_lineage=std::move(lineage[s]);
        auto correlation=current_execution_correlation();correlation.execution_identity=serial.execution_identity;correlation.position=post[s]-rows[s];
        correlation.rows=rows[s];correlation.row=UINT32_MAX;correlation.phase=ExecutionPhase::commit;
        ExecutionCorrelationScope observed(correlation);observe_execution_activity(ExecutionActivityKind::token_commit,0,rows[s]);
    }
    return results;
}
} // namespace strix
