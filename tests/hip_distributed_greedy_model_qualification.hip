#include "hip_distributed_greedy_model_qualification.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/session_state_memory.hpp"
#include "strix/distributed_stage_state.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
namespace strix::qualification {
namespace {
void require_case(bool value,const char* reason) {if(!value) throw std::invalid_argument(reason);}
void hip_case(hipError_t value) {if(value!=hipSuccess) throw std::runtime_error("distributed sequence qualification HIP failure");}
void disjoint(std::uintptr_t a,std::uint64_t an,std::uintptr_t b,std::uint64_t bn) {
    if(!an || !bn) {
        require_case((an || !a) && (bn || !b),"qualification absent pool carries a nonzero address");
        return;
    }
    require_case(a && b && an && bn && an<=UINTPTR_MAX-a && bn<=UINTPTR_MAX-b && !(a<b+bn && b<a+an),
        "qualification reference and distributed mutable owners overlap or overflow");
}
} // namespace


namespace {
DistributedGreedyQualificationResult qualify_greedy(HipDistributedSequenceSetup setup,
    const DistributedSequenceReference& reference,DistributedGreedyRequestIdentity request,
    std::span<const std::uint32_t> prompt,std::shared_ptr<DistributedControlByteStream> control,
    const SerialSessionContext* ordinary_anchor) {
    require_case(setup.participant && setup.actual_model_owner && reference.actual_owner && reference.session &&
        setup.streaming && setup.local_node<=1U &&
        setup.resources.session_state_layout && setup.resources.session_state_memory && setup.resources.scratch &&
        reference.resources.session_state_layout && reference.resources.session_state_memory && reference.resources.scratch &&
        reference.resources.model_weights && reference.resources.model_manifest && reference.resources.tensor_access &&
        reference.resources.model_weights->model_content_identity==request.model_content_identity &&
        reference.session->health==SerialSessionHealth::ready && !reference.session->arithmetic_lineage.tainted &&
        !reference.session->next_position && !reference.session->committed_tokens &&
        reference.scope.stream && !reference.scope.graph_capture && !current_hip_launch_context(),
        "greedy qualification needs fresh actual native and independent full-reference owners");
    require_case(setup.stage_owner?!ordinary_anchor:ordinary_anchor!=nullptr,
        "greedy stage qualification cannot fabricate an ordinary anchor");
    std::optional<SerialSessionContext> original;
    if(ordinary_anchor) {
        require_case(ordinary_anchor!=reference.session,"greedy reference aliases the ordinary anchor");
        original=*ordinary_anchor;
    }
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
    // Constructor checks actual transport, prompt SHA, roles and bounded row
    // budgets BEFORE the serial oracle or distributed execution mutates state.
    HipDistributedGreedyRequest actual(setup,request,prompt,std::move(control));
    compare_hip_distributed_owned_state(setup,reference);
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
    compare_hip_distributed_owned_state(setup,reference);
    if(ordinary_anchor) require_case(ordinary_anchor->health==SerialSessionHealth::poisoned && ordinary_anchor->arithmetic_lineage.tainted &&
        ordinary_anchor->next_position==original->next_position && ordinary_anchor->committed_tokens==original->committed_tokens &&
        ordinary_anchor->state.valid==original->state.valid && ordinary_anchor->state.generations==original->state.generations &&
        ordinary_anchor->arithmetic_lineage.history_identity==original->arithmetic_lineage.history_identity &&
        ordinary_anchor->arithmetic_lineage.intervals==original->arithmetic_lineage.intervals,
        "distributed greedy request republished its quarantined ordinary serial anchor");
    receipt.stage_owned_resources=static_cast<bool>(setup.stage_owner);
    if(setup.stage_owner) {
        receipt.stage_resource_identity=setup.stage_owner->plan().identity();
        receipt.state_partitioned=setup.stage_owner->reservation().state_partitioned;
        if(setup.resources.session_state_memory->stage_mapping)
            receipt.state_mapping_identity=setup.resources.session_state_memory->stage_mapping->identity();
    }
    receipt.real_native_generation_completed=true;receipt.selected_sequence_matches_serial=true;
    receipt.locally_owned_state_equal=true;return receipt;
}
} // namespace
DistributedGreedyQualificationResult qualify_hip_distributed_greedy_request(HipDistributedSequenceSetup setup,
    const DistributedSequenceReference& reference,DistributedGreedyRequestIdentity request,
    std::span<const std::uint32_t> prompt,std::shared_ptr<DistributedControlByteStream> control,
    const SerialSessionContext& ordinary_anchor) {
    return qualify_greedy(std::move(setup),reference,std::move(request),prompt,std::move(control),&ordinary_anchor);
}
DistributedGreedyQualificationResult qualify_hip_distributed_stage_greedy_request(HipDistributedSequenceSetup setup,
    const DistributedSequenceReference& reference,DistributedGreedyRequestIdentity request,
    std::span<const std::uint32_t> prompt,std::shared_ptr<DistributedControlByteStream> control) {
    require_case(static_cast<bool>(setup.stage_owner),"greedy qualification requires an actual retained stage owner");
    return qualify_greedy(std::move(setup),reference,std::move(request),prompt,std::move(control),nullptr);
}
} // namespace strix::qualification
