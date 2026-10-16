#include "hip_distributed_sequence_model_qualification.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/session_state_memory.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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

namespace {
DistributedSequenceQualificationResult qualify_sequence_inputs(HipDistributedSequenceSetup setup,
    const DistributedSequenceReference& reference,std::span<const std::uint32_t> inputs,const ModelContentIdentity& pin,
    const SerialSessionContext& ordinary_anchor,bool streaming) {
    require_case(pin.valid() && setup.participant && setup.actual_model_owner && reference.actual_owner && reference.session &&
        setup.local_node<=1U && setup.resources.model_weights && reference.resources.model_weights &&
        setup.resources.scratch && setup.resources.session_state_memory && setup.resources.session_state_layout &&
        reference.resources.scratch && reference.resources.session_state_memory && reference.resources.session_state_layout &&
        setup.resources.model_weights->model_content_identity==pin && reference.resources.model_weights->model_content_identity==pin,
        "qualification requires actual pinned model, native setup and independent full serial reference");
    require_case(reference.resources.model_manifest && reference.resources.tensor_access &&
        reference.scope.stream && !reference.scope.graph_capture && !current_hip_launch_context() &&
        reference.session->health==SerialSessionHealth::ready && !reference.session->arithmetic_lineage.tainted &&
        setup.streaming==streaming && !inputs.empty() && inputs.size()<=UINT32_MAX &&
        (streaming?inputs.size()<=setup.max_target_inputs:(inputs.size()>=2U && inputs.size()<=setup.maximum_inputs)) && setup.model.vocab_size &&
        setup.model.vocab_size<=UINT64_MAX/sizeof(float) && reference.maximum_logit_payload_bytes &&
        (streaming?1U:inputs.size())<=reference.maximum_logit_payload_bytes/(setup.model.vocab_size*sizeof(float)),
        "qualification reference or retained terminal-row budget is invalid");
    const auto binding=setup.participant->current_binding();
    require_case((streaming || inputs.size()<=16U) && inputs.front()==binding.input_token &&
        inputs.size()-1U<=UINT64_MAX-binding.generation,"qualification known input sequence does not bind its participant");
    for(const auto token:inputs) require_case(token<setup.model.vocab_size,"qualification known input is outside vocabulary");
    for(const auto& peer:binding.peers)
        require_case(binding.position<=peer.context_capacity && inputs.size()<=peer.context_capacity-binding.position,
            "qualification known input sequence exceeds a peer context");
    const auto original_anchor=ordinary_anchor;
    require_case(&ordinary_anchor!=reference.session,"qualification ordinary anchor aliases the serial reference");
    require_case(reference.session->next_position==binding.position && reference.session->committed_tokens==binding.committed_tokens &&
        reference.session->state.valid==binding.initial_global_state.valid &&
        reference.session->state.generations==binding.initial_global_state.generations,
        "qualification reference and distributed initial logical frontiers differ");
    const auto& a=*setup.resources.session_state_memory;const auto& b=*reference.resources.session_state_memory;
    disjoint(a.device_base,a.device_bytes,b.device_base,b.device_bytes);
    disjoint(a.host_base,a.host_bytes,b.host_base,b.host_bytes);
    disjoint(setup.resources.scratch->arena_base,setup.resources.scratch->arena_bytes,
        reference.resources.scratch->arena_base,reference.resources.scratch->arena_bytes);
    require_case(setup.resources.ple_prefetch!=reference.resources.ple_prefetch &&
        setup.resources.ple_lookup!=reference.resources.ple_lookup && setup.scope.stream!=reference.scope.stream,
        "qualification must retain independent PLE lifecycle owners");
    if(reference.config.deferred_moe_status) {
        require_case(setup.scope.deferred_moe_status.arena_bytes==8U && reference.scope.deferred_moe_status.arena_bytes==8U,
            "qualification needs exact independent D1 records");
        disjoint(setup.scope.deferred_moe_status.arena_base,8U,reference.scope.deferred_moe_status.arena_base,8U);
    }
    HipSetupQuiescenceGuard guard;
    // Actual missing/expired native capability fails before model/reference
    // execution. This object is used below; no replacement participant is made.
    HipDistributedKnownSequence sequence(setup);
    const auto data_boundary=std::find_if(setup.canonical_plan.boundaries.begin(),setup.canonical_plan.boundaries.end(),
        [](const auto& value) {return value.payload_bytes && !value.payloads.empty();});
    require_case(data_boundary!=setup.canonical_plan.boundaries.end(),"qualification has no actual native data boundary");
    const auto data_index=static_cast<std::size_t>(data_boundary-setup.canonical_plan.boundaries.begin());
    const auto ranges=owned_ranges(setup,reference);
    compare_ranges(ranges,guard); // Actual starting owned payloads, not metadata alone.
    for(unsigned fault=0;fault<7U;++fault) {
        auto bad=setup;
        const char* reason="";
        if(fault==0U) {
            if(setup.layer_boundary_position_dependencies && bad.canonical_plan.boundaries.size()>1U) {
                const auto replacement=data_index?0U:1U;
                const auto id=bad.canonical_plan.boundaries[replacement].boundary_id;
                bad.canonical_plan.boundaries[replacement]=bad.canonical_plan.boundaries[data_index];
                bad.canonical_plan.boundaries[replacement].boundary_id=id;
            } else bad.canonical_plan.boundaries.push_back(bad.canonical_plan.boundaries[data_index]);
            reason=setup.layer_boundary_position_dependencies?"exactly one data boundary":"exactly one boundary";
        }
        if(fault==1U) {++bad.canonical_plan.boundaries[data_index].destination_step;reason="last prefix step";}
        if(fault==2U) {++bad.canonical_plan.plan_identity;reason="canonical placement metadata";}
        if(fault==3U) {bad.local_node=2U;reason="bounded actual model";}
        if(fault==4U) {bad.transport.step_control=bad.transport.token_control;reason="distinct bounded kind5";}
        if(fault==5U) {bad.transport.actual_owner.reset();reason="real one-link transport";}
        if(fault==6U) {bad.local_node^=1U;reason="participant and model";}
        bool rejected=false;
        try {HipDistributedKnownSequence invalid(std::move(bad));}
        catch(const std::invalid_argument& error) {rejected=std::string(error.what()).find(reason)!=std::string::npos;}
        require_case(rejected && setup.participant->phase()==DistributedTokenPhase::preparing,
            "qualification invalid cut/endpoint/wire/owner did not reject at its intended pre-mutation gate");
    }
    if(setup.layer_boundary_position_dependencies) {
        // A structural guard is tested before canonical-ID comparison; no
        // forged boundary is offered to actual GPU or native transport APIs.
        const auto negative=[&](HipDistributedSequenceSetup bad,const char* reason) {
            bool rejected=false;
            try {HipDistributedKnownSequence invalid(std::move(bad));}
            catch(const std::invalid_argument& error) {rejected=std::string(error.what()).find(reason)!=std::string::npos;}
            require_case(rejected && setup.participant->phase()==DistributedTokenPhase::preparing,
                "qualification layer-cut admission reached the wrong gate or mutated its participant");
        };
        const auto move_cut=[&](HipDistributedSequenceSetup& bad,std::uint32_t cut) {
            auto& data=bad.canonical_plan.boundaries[data_index];data.source_step=cut;data.destination_step=cut+1U;
            for(auto& step:bad.canonical_plan.steps) {step.node=step.step_id<=cut?0U:1U;step.remote=step.node!=0U;}
        };
        auto interior=setup;
        require_case(data_boundary->source_step>0U,"qualification complete-layer cut has no interior predecessor");
        move_cut(interior,data_boundary->source_step-1U);
        negative(std::move(interior),"complete adjacent layers");
        for(const auto& step:setup.serial.execution.steps) if(step.op==SerialOp::ple && step.id>0U) {
            auto before_ple=setup;move_cut(before_ple,step.id-1U);
            negative(std::move(before_ple),"complete adjacent layers");break;
        }
        for(std::size_t z=0;z<setup.canonical_plan.boundaries.size();++z) {
            if(z==data_index) continue;
            for(unsigned mutation=0;mutation<4U;++mutation) {
                auto bad=setup;auto& edge=bad.canonical_plan.boundaries[z];
                if(mutation==0U) edge.source_step=data_boundary->source_step;
                if(mutation==1U) edge.destination_step=data_boundary->destination_step;
                if(mutation==2U) edge.payloads=data_boundary->payloads;
                if(mutation==3U) edge.boundary_id=0U;
                negative(std::move(bad),mutation==3U?"canonical boundary identity":"unsupported crossing dependency");
            }
        }
        if(setup.canonical_plan.boundaries.size()>1U) {
            auto strict=setup;strict.layer_boundary_position_dependencies=false;
            negative(std::move(strict),"exactly one boundary");
        }
    }
    {
        auto low_budget=setup;low_budget.maximum_logit_payload_bytes=1;
        bool rejected=false;
        try {
            HipDistributedKnownSequence limited(std::move(low_budget));
            if(streaming) (void)limited.execute_input(inputs.front());
            else (void)limited.execute(inputs);
        }
        catch(const std::invalid_argument& error) {rejected=std::string(error.what()).find("payload budget")!=std::string::npos;}
        require_case(rejected && setup.participant->phase()==DistributedTokenPhase::preparing,
            "qualification sequence budget did not reject before preparation or any wire send");
    }
    compare_ranges(ranges,guard);
    std::vector<std::byte> encoded;
    for(const auto token:inputs) for(unsigned shift=0;shift<32;shift+=8)
        encoded.push_back(static_cast<std::byte>((token>>shift)&0xffU));
    DistributedSequenceQualificationResult receipt;
    receipt.model_content=pin;receipt.input_u32le_sha256=sha256_bytes(encoded);
    receipt.local_node=setup.local_node;receipt.known_inputs=static_cast<std::uint32_t>(inputs.size());
    receipt.streaming_inputs=streaming;
    receipt.layer_boundary_position_dependencies=setup.layer_boundary_position_dependencies;
    for(const auto& edge:setup.canonical_plan.boundaries) {
        if(is_materialized_input_position_dependency(setup.canonical_plan,edge)) ++receipt.logical_position_dependencies;
        else ++receipt.data_boundaries;
    }
    require_case(receipt.data_boundaries==1U && receipt.logical_position_dependencies<=64U &&
        receipt.data_boundaries+receipt.logical_position_dependencies==setup.canonical_plan.boundaries.size(),
        "qualification did not account for every canonical dependency");
    HipLauncherCatalog catalog(setup.model,reference.config);
    ExecutionLauncherRegistry registry;catalog.register_for_compiled_execution(setup.serial,registry,&reference.resources);
    const auto launches=compile_launch_plan(setup.serial,registry);
    DirectLaunchDispatcher dispatch(launches);
    // Streaming qualification retains one reusable reference row. Legacy
    // known-input qualification still allocates ALL row payloads before work.
    std::vector<std::vector<float>> expected(streaming?1U:inputs.size());
    for(auto& row:expected) row.resize(static_cast<std::size_t>(setup.model.vocab_size));
    const auto reference_step=[&](std::size_t i,std::vector<float>& row) {
        guard.run([&] {
            const HipLaunchScope scope(reference.scope);
            (void)execute_serial_token(setup.serial,*reference.session,reference.resources,inputs[i],dispatch,[&] {
                hip_case(hipStreamSynchronize(reference.scope.stream));
                if(reference.config.deferred_moe_status) {
                    DeferredMoeFirstError error{};
                    hip_case(hipMemcpy(&error,reinterpret_cast<const void*>(reference.scope.deferred_moe_status.arena_base),
                        sizeof(error),hipMemcpyDeviceToHost));
                    check_deferred_moe_status(error);
                }
            });
        });
        const auto logits=resolve_execution_buffer(setup.serial.execution_identity,setup.serial.arena,
            *reference.resources.scratch,setup.serial.io.steps.back().writes.front());
        require_case(logits.bytes==row.size()*sizeof(float),"qualification reference is not a full FP32 row");
        guard.run([&] {hip_case(hipMemcpy(row.data(),reinterpret_cast<const void*>(logits.address),
            logits.bytes,hipMemcpyDeviceToHost));});
        for(const auto value:row) require_case(std::isfinite(value),"qualification reference row is nonfinite");
    };
    const auto check_row=[&](std::size_t i,const DistributedTokenCommit& commit,std::uint32_t token,
        std::uint32_t firstmax,std::span<const float> row,const std::vector<float>& wanted) {
        require_case(token==inputs[i] && commit.next_position==binding.position+i+1U &&
            commit.committed_tokens==binding.committed_tokens+i+1U && commit.decision_identity,
            "qualification committed frontier does not represent the actual known input prefix");
        if(setup.local_node) {
            require_case(row.size()==wanted.size() && std::memcmp(row.data(),wanted.data(),wanted.size()*sizeof(float))==0,
                "qualification actual native terminal FP32 row differs bitwise from serial");
            const auto selected=static_cast<std::uint32_t>(std::max_element(wanted.begin(),wanted.end())-wanted.begin());
            require_case(firstmax==selected,"qualification terminal firstmax tie policy differs");
        } else require_case(row.empty(),"coordinator fabricated a local terminal FP32 row");
    };
    require_case(sequence.completed_inputs()==0U && sequence.last_terminal_logits().empty(),
        "qualification published a terminal row before any transaction");
    // Both wrong-mode APIs must reject without preparation, transport or state
    // mutation. A mode flag never relaxes the literal execute(span) contract.
    {
        bool rejected=false;
        try {
            if(streaming) (void)sequence.execute(inputs);
            else (void)sequence.execute_input(inputs.front());
        } catch(const std::invalid_argument&) {rejected=true;}
        require_case(rejected && sequence.completed_inputs()==0U &&
            setup.participant->phase()==DistributedTokenPhase::preparing,
            "qualification cross-mode entry did not reject before preparation");
    }
    if(streaming) {
        for(unsigned fault=0;fault<2U;++fault) {
            auto bad=setup;
            if(!fault) bad.max_target_inputs=0U;
            else {
                std::uint64_t capacity=setup.serial.max_position_embeddings;
                for(const auto& peer:binding.peers) capacity=std::min(capacity,peer.context_capacity);
                require_case(capacity>=binding.position && capacity-binding.position<UINT64_MAX,
                    "qualification needs a representable streaming capacity-overflow witness");
                bad.max_target_inputs=capacity-binding.position+1U;
            }
            bool rejected=false;
            try {HipDistributedKnownSequence invalid(std::move(bad));}
            catch(const std::invalid_argument& error) {
                const std::string reason=error.what();
                rejected=reason.find(!fault?"positive target limit":"capacity")!=std::string::npos ||
                    (fault && reason.find("generation bound")!=std::string::npos);
            }
            require_case(rejected && setup.participant->phase()==DistributedTokenPhase::preparing,
                "qualification streaming limit did not reject before target preparation");
        }
        bool rejected=false;
        try {(void)sequence.execute_input(static_cast<std::uint32_t>(setup.model.vocab_size));}
        catch(const std::invalid_argument&) {rejected=true;}
        require_case(rejected && sequence.completed_inputs()==0U && sequence.last_terminal_logits().empty() &&
            setup.participant->phase()==DistributedTokenPhase::preparing,
            "qualification invalid streaming input changed its initial participant");
        if(setup.model.vocab_size>1U) {
            const auto other=static_cast<std::uint32_t>((static_cast<std::uint64_t>(inputs.front())+1U)%setup.model.vocab_size);
            bool wrong_first=false;
            try {(void)sequence.execute_input(other);}
            catch(const std::invalid_argument& error) {
                wrong_first=std::string(error.what()).find("negotiated binding")!=std::string::npos;
            }
            require_case(wrong_first && sequence.completed_inputs()==0U &&
                setup.participant->phase()==DistributedTokenPhase::preparing,
                "qualification streaming first input did not bind the negotiated token");
        }
        const float* terminal_storage=nullptr;
        for(std::size_t i=0;i<inputs.size();++i) {
            reference_step(i,expected.front());
            const auto completion=sequence.execute_input(inputs[i]);
            const auto row=sequence.last_terminal_logits();
            check_row(i,completion.commit,completion.input_token,completion.terminal_firstmax,row,expected.front());
            require_case(completion.terminal_peer==(setup.local_node==1U) && sequence.completed_inputs()==i+1U &&
                setup.participant->phase()==DistributedTokenPhase::committed &&
                completion.commit.binding_identity==setup.participant->binding_identity(),
                "qualification streaming completion did not follow the actual committed binding");
            if(setup.local_node) {
                require_case(completion.terminal_firstmax_bits==std::bit_cast<std::uint32_t>(row[completion.terminal_firstmax]),
                    "qualification streaming firstmax bits differ from the actual full terminal row");
                if(!i) terminal_storage=row.data();
                require_case(row.data()==terminal_storage,"qualification streaming terminal row was reallocated");
            } else require_case(completion.terminal_firstmax==0U && completion.terminal_firstmax_bits==0U,
                "qualification coordinator fabricated a terminal selection");
        }
        if(inputs.size()==setup.max_target_inputs) {
            const auto previous=sequence.last_terminal_logits();
            bool rejected_limit=false;
            try {(void)sequence.execute_input(inputs.back());}
            catch(const std::invalid_argument&) {rejected_limit=true;}
            require_case(rejected_limit && sequence.completed_inputs()==inputs.size() &&
                sequence.last_terminal_logits().data()==previous.data() &&
                setup.participant->phase()==DistributedTokenPhase::committed,
                "qualification exhausted streaming limit performed another transaction");
        }
    } else {
        for(std::size_t i=0;i<inputs.size();++i) reference_step(i,expected[i]);
        const auto result=sequence.execute(inputs);
        require_case(result.inputs.size()==inputs.size() && result.coordinator_confirmation==(setup.local_node==0U) &&
            sequence.completed_inputs()==inputs.size(),"qualification sequence did not confirm every known input");
        for(std::size_t i=0;i<result.inputs.size();++i) {
            const auto& row=result.inputs[i];
            check_row(i,row.commit,row.input_token,row.terminal_firstmax,row.terminal_logits,expected[i]);
        }
    }
    const auto frontier=setup.participant->frontier();
    require_case(frontier.has_value() && frontier->commit().global_state.valid==reference.session->state.valid &&
        frontier->commit().global_state.generations==reference.session->state.generations,
        "qualification final global write generations differ from full serial");
    compare_ranges(ranges,guard);
    require_case(ordinary_anchor.health==SerialSessionHealth::poisoned && ordinary_anchor.arithmetic_lineage.tainted &&
        ordinary_anchor.execution_identity==original_anchor.execution_identity &&
        ordinary_anchor.state_layout_identity==original_anchor.state_layout_identity &&
        ordinary_anchor.runtime_compatibility_identity==original_anchor.runtime_compatibility_identity &&
        ordinary_anchor.next_position==original_anchor.next_position && ordinary_anchor.committed_tokens==original_anchor.committed_tokens &&
        ordinary_anchor.state.valid==original_anchor.state.valid && ordinary_anchor.state.generations==original_anchor.state.generations &&
        ordinary_anchor.arithmetic_lineage.intervals==original_anchor.arithmetic_lineage.intervals &&
        ordinary_anchor.arithmetic_lineage.history_identity==original_anchor.arithmetic_lineage.history_identity,
        "qualification distributed continuation republished or changed its quarantined ordinary anchor");
    receipt.native_sequence_completed=true;receipt.terminal_rows_bitwise_equal=setup.local_node==1U;
    receipt.locally_owned_state_equal=true;
    return receipt;
}
} // namespace
DistributedSequenceQualificationResult qualify_hip_distributed_known_sequence(HipDistributedSequenceSetup setup,
    const DistributedSequenceReference& reference,std::span<const std::uint32_t> inputs,const ModelContentIdentity& pin,
    const SerialSessionContext& ordinary_anchor) {
    return qualify_sequence_inputs(std::move(setup),reference,inputs,pin,ordinary_anchor,false);
}
DistributedSequenceQualificationResult qualify_hip_distributed_streaming_inputs(HipDistributedSequenceSetup setup,
    const DistributedSequenceReference& reference,std::span<const std::uint32_t> inputs,const ModelContentIdentity& pin,
    const SerialSessionContext& ordinary_anchor) {
    return qualify_sequence_inputs(std::move(setup),reference,inputs,pin,ordinary_anchor,true);
}
DistributedSequenceQualificationResult qualify_hip_distributed_layer_boundary_inputs(HipDistributedSequenceSetup setup,
    const DistributedSequenceReference& reference,std::span<const std::uint32_t> inputs,const ModelContentIdentity& pin,
    const SerialSessionContext& ordinary_anchor) {
    require_case(setup.layer_boundary_position_dependencies,"qualification requires an explicit actual complete-layer setup");
    const bool streaming=setup.streaming;
    return qualify_sequence_inputs(std::move(setup),reference,inputs,pin,ordinary_anchor,streaming);
}
} // namespace strix::qualification
