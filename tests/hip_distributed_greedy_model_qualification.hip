#include "hip_distributed_greedy_model_qualification.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/session_state_memory.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>
namespace strix::qualification {
namespace {
void require_case(bool value,const char* reason) {if(!value) throw std::invalid_argument(reason);}
void hip_case(hipError_t value) {if(value!=hipSuccess) throw std::runtime_error("distributed sequence qualification HIP failure");}
void disjoint(std::uintptr_t a,std::uint64_t an,std::uintptr_t b,std::uint64_t bn) {
    require_case(a && b && an && bn && an<=UINTPTR_MAX-a && bn<=UINTPTR_MAX-b && !(a<b+bn && b<a+an),
        "qualification reference and distributed mutable owners overlap or overflow");
}
struct OwnedRange {SessionStateAddress actual,reference;};
std::vector<OwnedRange> owned_ranges(const HipDistributedSequenceSetup& setup,const DistributedSequenceReference& reference) {
    std::vector<OwnedRange> ranges;
    const auto& b=setup.participant->current_binding();const auto& peer=b.peers[setup.local_node];
    const auto& layout=*setup.resources.session_state_layout;const auto& memory=*setup.resources.session_state_memory;
    const auto& ref_layout=*reference.resources.session_state_layout;const auto& ref_memory=*reference.resources.session_state_memory;
    require_case(layout.layout_identity==ref_layout.layout_identity && layout.recurrent_storage==ref_layout.recurrent_storage,
        "qualification physical state layouts differ");
    for(std::uint32_t i=peer.first_step;i<=peer.last_step;++i) {
        const auto& step=setup.serial.execution.steps[i];
        if(step.op==SerialOp::gated_delta_net) {
            require_case(step.layer.has_value(),"GDN owned slice has no layer");
            for(auto component:{SessionStateComponent::recurrent_matrix,SessionStateComponent::gdn_convolution})
                ranges.push_back({resolve_gdn_layer_state(layout,memory,setup.model,*step.layer,component),
                    resolve_gdn_layer_state(ref_layout,ref_memory,setup.model,*step.layer,component)});
        } else if(step.op==SerialOp::indexed_attention) {
            require_case(step.layer.has_value(),"attention owned slice has no layer");
            for(auto component:{SessionStateComponent::attention_kv,SessionStateComponent::sparse_index})
                ranges.push_back({resolve_indexed_attention_layer_state(layout,memory,setup.model,*step.layer,component),
                    resolve_indexed_attention_layer_state(ref_layout,ref_memory,setup.model,*step.layer,component)});
        } else if(step.op==SerialOp::ple) {
            for(auto component:{SessionStateComponent::ple_convolution,SessionStateComponent::ngram_history})
                ranges.push_back({resolve_session_state(layout,memory,component),resolve_session_state(ref_layout,ref_memory,component)});
        }
    }
    // Prefix position_prepare and the suffix controller materialization both
    // write the current input position. This compares payload, not a fabricated
    // suffix-local position generation or healthy ordinary-session frontier.
    ranges.push_back({resolve_session_state(layout,memory,SessionStateComponent::position_history),
        resolve_session_state(ref_layout,ref_memory,SessionStateComponent::position_history)});
    return ranges;
}
void compare_ranges(const std::vector<OwnedRange>& ranges,HipSetupQuiescenceGuard& guard) {
    std::array<std::byte,65536> actual{},reference{};
    for(const auto& range:ranges) {
        require_case(range.actual.bytes==range.reference.bytes && range.actual.storage==range.reference.storage,
            "qualification owned state geometry differs");
        for(std::uint64_t offset=0;offset<range.actual.bytes;) {
            const auto bytes=static_cast<std::size_t>(std::min<std::uint64_t>(actual.size(),range.actual.bytes-offset));
            guard.run([&] {
                if(range.actual.storage==SessionStateStorage::device) {
                    hip_case(hipMemcpy(actual.data(),reinterpret_cast<const void*>(range.actual.address+offset),bytes,hipMemcpyDeviceToHost));
                    hip_case(hipMemcpy(reference.data(),reinterpret_cast<const void*>(range.reference.address+offset),bytes,hipMemcpyDeviceToHost));
                } else {
                    std::memcpy(actual.data(),reinterpret_cast<const void*>(range.actual.address+offset),bytes);
                    std::memcpy(reference.data(),reinterpret_cast<const void*>(range.reference.address+offset),bytes);
                }
            });
            require_case(std::equal(actual.begin(),actual.begin()+bytes,reference.begin()),
                "qualification locally owned state payload differs from full serial reference");
            offset+=bytes;
        }
    }
}
} // namespace


DistributedGreedyQualificationResult qualify_hip_distributed_greedy_request(HipDistributedSequenceSetup setup,
    const DistributedSequenceReference& reference,DistributedGreedyRequestIdentity request,
    std::span<const std::uint32_t> prompt,std::shared_ptr<DistributedControlByteStream> control,
    const SerialSessionContext& ordinary_anchor) {
    require_case(setup.participant && setup.actual_model_owner && reference.actual_owner && reference.session &&
        &ordinary_anchor!=reference.session && setup.streaming && setup.local_node<=1U &&
        setup.resources.session_state_layout && setup.resources.session_state_memory && setup.resources.scratch &&
        reference.resources.session_state_layout && reference.resources.session_state_memory && reference.resources.scratch &&
        reference.resources.model_weights && reference.resources.model_manifest && reference.resources.tensor_access &&
        reference.resources.model_weights->model_content_identity==request.model_content_identity &&
        reference.session->health==SerialSessionHealth::ready && !reference.session->arithmetic_lineage.tainted &&
        !reference.session->next_position && !reference.session->committed_tokens &&
        reference.scope.stream && !reference.scope.graph_capture && !current_hip_launch_context(),
        "greedy qualification needs fresh actual native and independent full-reference owners");
    const auto original=ordinary_anchor;
    const auto binding=setup.participant->current_binding();
    require_case(reference.session->state.valid==binding.initial_global_state.valid &&
        reference.session->state.generations==binding.initial_global_state.generations,
        "greedy reference and actual initial generations differ");
    const auto& a=*setup.resources.session_state_memory;const auto& b=*reference.resources.session_state_memory;
    disjoint(a.device_base,a.device_bytes,b.device_base,b.device_bytes);
    disjoint(a.host_base,a.host_bytes,b.host_base,b.host_bytes);
    disjoint(setup.resources.scratch->arena_base,setup.resources.scratch->arena_bytes,
        reference.resources.scratch->arena_base,reference.resources.scratch->arena_bytes);
    require_case(setup.scope.stream!=reference.scope.stream && setup.resources.ple_lookup!=reference.resources.ple_lookup &&
        setup.resources.ple_prefetch!=reference.resources.ple_prefetch,"greedy reference must retain independent streams/PLE");
    if(reference.config.deferred_moe_status) {
        require_case(setup.scope.deferred_moe_status.arena_bytes==8U && reference.scope.deferred_moe_status.arena_bytes==8U,
            "greedy qualification requires exact independent D1 records");
        disjoint(setup.scope.deferred_moe_status.arena_base,8U,reference.scope.deferred_moe_status.arena_base,8U);
    }
    HipSetupQuiescenceGuard guard;
    const auto ranges=owned_ranges(setup,reference);compare_ranges(ranges,guard);
    // Constructor checks actual transport, prompt SHA, roles and bounded row
    // budgets BEFORE the serial oracle or distributed execution mutates state.
    HipDistributedGreedyRequest actual(setup,request,prompt,std::move(control));
    HipLauncherCatalog catalog(setup.model,reference.config);ExecutionLauncherRegistry registry;
    catalog.register_for_compiled_execution(setup.serial,registry,&reference.resources);
    const auto launches=compile_launch_plan(setup.serial,registry);DirectLaunchDispatcher dispatch(launches);
    class ReferenceBackend final:public SerialGreedyStepBackend {
    public:
        ReferenceBackend(const CompiledSerialExecution& serial,const DistributedSequenceReference& reference,
            DirectLaunchDispatcher& dispatch,HipSetupQuiescenceGuard& guard,std::uint64_t vocab)
            :serial_(serial),reference_(reference),dispatch_(dispatch),guard_(guard) {
            require_case(vocab && vocab<=SIZE_MAX/sizeof(float) && vocab<=reference.maximum_logit_payload_bytes/sizeof(float),
                "greedy full serial row exceeds explicit reference payload budget");
            row_.resize(static_cast<std::size_t>(vocab));
        }
        std::vector<float> execute(std::uint32_t token,std::uint64_t position) override {
            run(token,position);return row_;
        }
        std::uint32_t execute_greedy(std::uint32_t token,std::uint64_t position,std::uint64_t vocab) override {
            require_case(vocab==row_.size(),"reference greedy vocabulary changed");
            run(token,position);return select_greedy_token(row_,vocab);
        }
        void execute_state(std::uint32_t token,std::uint64_t position) override {run(token,position);}
    private:
        void run(std::uint32_t token,std::uint64_t position) {
            require_case(reference_.session->next_position==position,"reference cursor frontier changed");
            guard_.run([&] {
                const HipLaunchScope scope(reference_.scope);
                (void)execute_serial_token(serial_,*reference_.session,reference_.resources,token,dispatch_,[&] {
                    hip_case(hipStreamSynchronize(reference_.scope.stream));
                    if(reference_.config.deferred_moe_status) {
                        DeferredMoeFirstError error{};
                        hip_case(hipMemcpy(&error,reinterpret_cast<const void*>(reference_.scope.deferred_moe_status.arena_base),
                            sizeof(error),hipMemcpyDeviceToHost));check_deferred_moe_status(error);
                    }
                });
            });
            const auto logits=resolve_execution_buffer(serial_.execution_identity,serial_.arena,
                *reference_.resources.scratch,serial_.io.steps.back().writes.front());
            require_case(logits.bytes==row_.size()*sizeof(float),"reference full output extent changed");
            guard_.run([&] {hip_case(hipMemcpy(row_.data(),reinterpret_cast<const void*>(logits.address),logits.bytes,hipMemcpyDeviceToHost));});
            for(const float value:row_) require_case(std::isfinite(value),"reference terminal row is nonfinite");
        }
        const CompiledSerialExecution& serial_;const DistributedSequenceReference& reference_;
        DirectLaunchDispatcher& dispatch_;HipSetupQuiescenceGuard& guard_;std::vector<float> row_;
    } backend(setup.serial,reference,dispatch,guard,request.vocab_size);
    const auto expected=execute_serial_greedy_token_request({prompt,request.vocab_size,request.context_capacity,
        request.max_new_tokens,request.eos_token_id},backend);
    DistributedGreedyQualificationResult receipt;receipt.local_node=setup.local_node;
    if(!setup.local_node) {
        const auto observed=actual.execute_coordinator();
        require_case(!observed.cancelled && observed.finish_confirmed &&
            observed.generation.generated_tokens==expected.generated_tokens &&
            observed.generation.steps_executed==expected.steps_executed &&
            observed.generation.final_position==expected.final_position && observed.generation.stop_reason==expected.stop_reason,
            "actual native distributed greedy cursor differs from independent serial output/frontier");
        receipt.consumed_inputs=observed.generation.steps_executed;
        receipt.emitted_tokens=observed.generation.generated_tokens.size();receipt.finish_confirmed=observed.finish_confirmed;
    } else {
        const auto observed=actual.serve_terminal();
        require_case(observed.finish_confirmed && observed.consumed_inputs==expected.steps_executed &&
            observed.emitted_tokens==expected.generated_tokens.size() &&
            observed.stop_reason==distributed_greedy_finish_reason(expected.stop_reason) &&
            observed.pending_present==!expected.generated_tokens.empty() &&
            (!observed.pending_present || observed.pending_token==expected.generated_tokens.back()),
            "actual terminal finish differs from independent serial consumed/emitted/pending frontier");
        receipt.consumed_inputs=observed.consumed_inputs;receipt.emitted_tokens=observed.emitted_tokens;
        receipt.finish_confirmed=observed.finish_confirmed;
    }
    const auto frontier=setup.participant->frontier();
    require_case(frontier.has_value() && setup.participant->phase()==DistributedTokenPhase::committed &&
        frontier->commit().next_position==expected.final_position && frontier->commit().committed_tokens==expected.steps_executed &&
        frontier->commit().global_state.valid==reference.session->state.valid &&
        frontier->commit().global_state.generations==reference.session->state.generations,
        "greedy final distributed generations/frontier differ from independent serial");
    compare_ranges(ranges,guard);
    require_case(ordinary_anchor.health==SerialSessionHealth::poisoned && ordinary_anchor.arithmetic_lineage.tainted &&
        ordinary_anchor.next_position==original.next_position && ordinary_anchor.committed_tokens==original.committed_tokens &&
        ordinary_anchor.state.valid==original.state.valid && ordinary_anchor.state.generations==original.state.generations &&
        ordinary_anchor.arithmetic_lineage.history_identity==original.arithmetic_lineage.history_identity &&
        ordinary_anchor.arithmetic_lineage.intervals==original.arithmetic_lineage.intervals,
        "distributed greedy request republished its quarantined ordinary serial anchor");
    receipt.real_native_generation_completed=true;receipt.selected_sequence_matches_serial=true;
    receipt.locally_owned_state_equal=true;return receipt;
}
} // namespace strix::qualification
