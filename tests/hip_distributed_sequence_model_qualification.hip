#include "hip_distributed_sequence_model_qualification.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/session_state_memory.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
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

DistributedSequenceQualificationResult qualify_hip_distributed_known_sequence(HipDistributedSequenceSetup setup,
    const DistributedSequenceReference& reference,std::span<const std::uint32_t> inputs,const ModelContentIdentity& pin,
    const SerialSessionContext& ordinary_anchor) {
    require_case(pin.valid() && setup.participant && setup.actual_model_owner && reference.actual_owner && reference.session &&
        setup.local_node<=1U && setup.resources.model_weights && reference.resources.model_weights &&
        setup.resources.scratch && setup.resources.session_state_memory && setup.resources.session_state_layout &&
        reference.resources.scratch && reference.resources.session_state_memory && reference.resources.session_state_layout &&
        setup.resources.model_weights->model_content_identity==pin && reference.resources.model_weights->model_content_identity==pin,
        "qualification requires actual pinned model, native setup and independent full serial reference");
    require_case(reference.resources.model_manifest && reference.resources.tensor_access &&
        reference.scope.stream && !reference.scope.graph_capture && !current_hip_launch_context() &&
        reference.session->health==SerialSessionHealth::ready && !reference.session->arithmetic_lineage.tainted &&
        inputs.size()>=2U && inputs.size()<=setup.maximum_inputs && setup.model.vocab_size &&
        setup.model.vocab_size<=UINT64_MAX/sizeof(float) && reference.maximum_logit_payload_bytes &&
        inputs.size()<=reference.maximum_logit_payload_bytes/(setup.model.vocab_size*sizeof(float)),
        "qualification reference or retained terminal-row budget is invalid");
    const auto binding=setup.participant->current_binding();
    require_case(inputs.size()<=16U && inputs.front()==binding.input_token &&
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
    const auto ranges=owned_ranges(setup,reference);
    compare_ranges(ranges,guard); // Actual starting owned payloads, not metadata alone.
    for(unsigned fault=0;fault<7U;++fault) {
        auto bad=setup;
        const char* reason="";
        if(fault==0U) {bad.canonical_plan.boundaries.push_back(bad.canonical_plan.boundaries.front());reason="exactly one boundary";}
        if(fault==1U) {++bad.canonical_plan.boundaries.front().destination_step;reason="last prefix step";}
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
    {
        auto low_budget=setup;low_budget.maximum_logit_payload_bytes=1;
        HipDistributedKnownSequence limited(std::move(low_budget));
        bool rejected=false;
        try {(void)limited.execute(inputs);}
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
    HipLauncherCatalog catalog(setup.model,reference.config);
    ExecutionLauncherRegistry registry;catalog.register_for_compiled_execution(setup.serial,registry,&reference.resources);
    const auto launches=compile_launch_plan(setup.serial,registry);
    DirectLaunchDispatcher dispatch(launches);
    std::vector<std::vector<float>> expected(inputs.size());
    for(auto& row:expected) row.resize(static_cast<std::size_t>(setup.model.vocab_size));
    for(std::size_t i=0;i<inputs.size();++i) {
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
        require_case(logits.bytes==expected[i].size()*sizeof(float),"qualification reference is not a full FP32 row");
        guard.run([&] {hip_case(hipMemcpy(expected[i].data(),reinterpret_cast<const void*>(logits.address),
            logits.bytes,hipMemcpyDeviceToHost));});
        for(const auto value:expected[i]) require_case(std::isfinite(value),"qualification reference row is nonfinite");
    }
    const auto result=sequence.execute(inputs);
    require_case(result.inputs.size()==inputs.size() && result.coordinator_confirmation==(setup.local_node==0U),
        "qualification sequence did not confirm every known input");
    for(std::size_t i=0;i<result.inputs.size();++i) {
        const auto& row=result.inputs[i];
        require_case(row.input_token==inputs[i] && row.commit.next_position==binding.position+i+1U &&
            row.commit.committed_tokens==binding.committed_tokens+i+1U && row.commit.decision_identity,
            "qualification committed frontier does not represent the actual known input prefix");
        if(setup.local_node) {
            require_case(row.terminal_logits.size()==expected[i].size() &&
                std::memcmp(row.terminal_logits.data(),expected[i].data(),expected[i].size()*sizeof(float))==0,
                "qualification actual native terminal FP32 row differs bitwise from serial");
            const auto selected=static_cast<std::uint32_t>(std::max_element(expected[i].begin(),expected[i].end())-expected[i].begin());
            require_case(row.terminal_firstmax==selected,"qualification terminal firstmax tie policy differs");
        } else require_case(row.terminal_logits.empty(),"coordinator fabricated a local terminal FP32 row");
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
} // namespace strix::qualification
