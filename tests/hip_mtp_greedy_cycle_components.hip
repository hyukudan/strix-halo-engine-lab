#include "hip_mtp_greedy_cycle_components.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/hip_launch_context.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
namespace strix::tests {
namespace {
constexpr std::uint64_t cycle_limit=256U;
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
std::uint64_t add(std::uint64_t a,std::uint64_t b){if(b>UINT64_MAX-a)throw std::overflow_error("MTP cycle fixture byte overflow");return a+b;}
std::uint64_t mul(std::uint64_t a,std::uint64_t b){if(b && a>UINT64_MAX/b)throw std::overflow_error("MTP cycle fixture byte overflow");return a*b;}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
void append(std::vector<std::byte>& frame,std::uint64_t value){for(unsigned shift=0;shift<64U;shift+=8U)frame.push_back(static_cast<std::byte>((value>>shift)&255U));}
void append(std::vector<std::byte>& frame,const ModelContentIdentity& value){for(auto b:value.bytes)frame.push_back(static_cast<std::byte>(b));}
template<class T>ModelContentIdentity digest(const std::vector<T>& v){return sha256_bytes(std::as_bytes(std::span<const T>(v)));}
template<class T>void payload(std::vector<std::byte>& frame,const std::vector<T>& v){append(frame,v.size());append(frame,digest(v));}
float decode(std::uint16_t v){return std::bit_cast<float>(static_cast<std::uint32_t>(v)<<16U);}
void finite_hidden(const std::vector<std::uint16_t>& v){for(auto b:v)require(std::isfinite(decode(b)),"MTP cycle fixture hidden reference/output is nonfinite");}
void finite_logits(const std::vector<float>& v){for(auto b:v)require(std::isfinite(b),"MTP cycle fixture full logits are nonfinite");}
bool is_eos(std::uint32_t token,const std::vector<std::uint32_t>& eos){return std::find(eos.begin(),eos.end(),token)!=eos.end();}
void read(MtpDeviceView view,void* out,std::uint64_t bytes,hipStream_t stream){
    require(view.address && view.allocation_generation && view.bytes>=bytes &&
        bytes<=std::numeric_limits<std::uintptr_t>::max()-view.address,"MTP cycle fixture native view is invalid");
    check(hipMemcpyAsync(out,reinterpret_cast<const void*>(view.address),static_cast<std::size_t>(bytes),hipMemcpyDeviceToHost,stream));
}
void require_coverage(const HipMtpGreedyCycleCoverage& needed,const HipMtpGreedyCycleCoverage& actual){
    require((!needed.accepted_window||actual.accepted_window) && (!needed.rejected_window_restored||actual.rejected_window_restored) &&
        (!needed.eos||actual.eos) && (!needed.budget||actual.budget) && (!needed.continuation||actual.continuation),
        "MTP cycle fixture independent traces omit required coverage");
}
HipMtpGreedyCycleCoverage coverage(const HipMtpGreedyCycleGolden& golden){
    HipMtpGreedyCycleCoverage out;
    for(std::size_t i=0;i<golden.cycles.size();++i){const auto& row=golden.cycles[i];
        out.accepted_window|=row.attempted_window&&row.accepted_window&&!row.restored_checkpoint;
        out.rejected_window_restored|=row.attempted_window&&!row.accepted_window&&row.restored_checkpoint;
        out.eos|=row.eos;out.budget|=row.terminal_budget;out.continuation|=i!=0U;
    }return out;
}
}
ModelContentIdentity hip_mtp_greedy_cycle_golden_identity(const HipMtpGreedyCycleGolden& g){
    static_assert(sizeof(float)==4U && std::numeric_limits<float>::is_iec559);
    require(std::endian::native==std::endian::little && g.cycles.size()<=cycle_limit,"MTP cycle fixture framing contract is invalid");
    std::vector<std::byte> frame;frame.reserve(static_cast<std::size_t>(mul(add(g.cycles.size(),1U),512U)));
    append(frame,0x4d54504359434c31ULL);append(frame,g.independent_source_content);append(frame,g.target_content);append(frame,g.sidecar_content);
    append(frame,g.independent_reference_identity);append(frame,g.structural_plan_identity);append(frame,g.semantic_identity);append(frame,g.numerical_profile_identity);append(frame,g.cycles.size());
    for(const auto& r:g.cycles){
        append(frame,r.initial_source_position);append(frame,r.final_source_position);append(frame,r.final_committed_tokens);append(frame,r.consumed_target_rows);
        append(frame,r.reconciled_next_source_position);append(frame,r.initial_pending_token);append(frame,r.final_pending_token);
        append(frame,r.maximum_candidates);append(frame,r.token_budget);append(frame,r.proposed_candidates);append(frame,r.accepted_candidates);
        append(frame,r.target_r2_dispatches);append(frame,r.target_projection_launches_saved);
        append(frame,r.eos);append(frame,r.terminal_budget);append(frame,r.attempted_window);append(frame,r.accepted_window);append(frame,r.restored_checkpoint);
        payload(frame,r.eos_tokens);payload(frame,r.emitted_tokens);payload(frame,r.final_target_wide_bf16);payload(frame,r.reconciled_mtp_wide_bf16);payload(frame,r.reconciled_full_logits);
    }return sha256_bytes(frame);
}
HipMtpGreedyCycleComponentReceipt qualify_hip_mtp_greedy_cycles(HipMtpGreedyRuntime& greedy,HipMtpRuntime& native,
    HipMtpTargetHiddenTicket initial,const HipMtpGreedyCycleGolden& g,const HipMtpGreedyCycleComponentRequest& request){
    require(&greedy.native_runtime()==&native,"MTP cycle fixture requires its exact actual native predictor instance");
    const auto& owner=native.resources();owner.validate();owner.require_idle();
    require(current_hip_launch_context()==nullptr,"MTP cycle fixture must not borrow a launch scope");
    const auto stream=static_cast<hipStream_t>(owner.stream());hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    check(hipStreamIsCapturing(stream,&capture));require(capture==hipStreamCaptureStatusNone,"MTP cycle fixture must not enter capture");check(hipPeekAtLastError());
    require(!native.failed()&&!greedy.failed()&&initial.valid(),"MTP cycle fixture requires live actual runtimes and a committed target ticket");
    require(request.maximum_cycles && request.maximum_cycles<=cycle_limit && !g.cycles.empty() &&
        g.cycles.size()<=request.maximum_cycles,"MTP cycle fixture cycle bound is invalid");
    require(g.independent_source_content.valid()&&g.independent_reference_identity&&request.expected_payload.valid()&&
        g.target_content==owner.shared_target().target_model_content_identity && initial.target_content_identity()==g.target_content &&
        g.sidecar_content==owner.shared_target().sidecar_model_content_identity && g.structural_plan_identity==owner.plan().plan_identity &&
        g.semantic_identity==owner.plan().forward_semantics.contract_identity && g.numerical_profile_identity==native.numerical_profile_identity(),
        "MTP cycle fixture independent provenance/model/plan/profile differs");
    const auto width=mul(owner.model().hidden_size,owner.model().hc_count);const auto vocab=owner.model().vocab_size;
    require(width&&width<=SIZE_MAX/2U&&vocab&&vocab<=UINT32_MAX&&vocab<=SIZE_MAX/4U,"MTP cycle fixture geometry is invalid");
    std::uint64_t bytes=add(mul(width,4U),mul(vocab,4U));
    bytes=add(bytes,mul(add(g.cycles.size(),1U),512U));
    bytes=add(bytes,mul(g.cycles.size(),add(sizeof(HipMtpGreedyCycleComponentRowReceipt),sizeof(HipMtpGreedyCycleGoldenRow))));
    for(const auto& r:g.cycles){
        bytes=add(bytes,add(mul(add(r.eos_tokens.size(),r.emitted_tokens.size()),4U),
            add(mul(add(r.final_target_wide_bf16.size(),r.reconciled_mtp_wide_bf16.size()),2U),mul(r.reconciled_full_logits.size(),4U))));
    }
    require(request.host_payload_budget_bytes&&bytes<=request.host_payload_budget_bytes,"MTP cycle fixture exceeds host payload budget");
    for(std::size_t i=0;i<g.cycles.size();++i){const auto& r=g.cycles[i];
        require(r.token_budget && r.maximum_candidates<=owner.budget().maximum_candidate_depth &&
            r.initial_source_position<owner.context_capacity()&&r.final_source_position<owner.context_capacity()&&
            r.initial_pending_token<vocab&&r.final_pending_token<vocab&&r.proposed_candidates<=r.maximum_candidates&&
            r.accepted_candidates<=r.proposed_candidates&&r.emitted_tokens.size()<=r.token_budget&&
            r.consumed_target_rows==r.emitted_tokens.size()&&r.final_source_position==add(r.initial_source_position,r.consumed_target_rows)&&
            r.reconciled_next_source_position==r.final_source_position,
            "MTP cycle fixture trace counters/frontier/depth are inconsistent");
        require(r.final_target_wide_bf16.size()==width&&r.reconciled_mtp_wide_bf16.size()==width&&r.reconciled_full_logits.size()==vocab,
            "MTP cycle fixture reference requires full wide states and vocabulary");
        for(auto token:r.eos_tokens)require(token<vocab,"MTP cycle fixture EOS is outside vocabulary");
        for(std::size_t j=0;j<r.emitted_tokens.size();++j)require(r.emitted_tokens[j]<vocab&&
            (j+1U==r.emitted_tokens.size()||!is_eos(r.emitted_tokens[j],r.eos_tokens)),"MTP cycle fixture output token/EOS ordering differs");
        require(r.eos==is_eos(r.final_pending_token,r.eos_tokens)&&r.terminal_budget==(r.emitted_tokens.size()>=r.token_budget)&&
            (r.emitted_tokens.empty()?r.final_pending_token==r.initial_pending_token:r.final_pending_token==r.emitted_tokens.back())&&
            (!r.accepted_window||(r.attempted_window&&!r.restored_checkpoint))&&(!r.restored_checkpoint||r.attempted_window),
            "MTP cycle fixture expected flags or pending selection differ");
        if(i){const auto& prior=g.cycles[i-1U];require(!prior.eos&&r.initial_source_position==prior.final_source_position&&
            r.initial_pending_token==prior.final_pending_token,"MTP cycle fixture continuation must consume prior pending without duplicate emission");}
        finite_hidden(r.final_target_wide_bf16);finite_hidden(r.reconciled_mtp_wide_bf16);finite_logits(r.reconciled_full_logits);
    }
    require_coverage(request.required_coverage,coverage(g));
    require(initial.source_position()==g.cycles.front().initial_source_position&&initial.predicted_token()==g.cycles.front().initial_pending_token&&
        native.next_source_position()==initial.source_position(),"MTP cycle fixture initial real frontier differs");
    const auto reference=hip_mtp_greedy_cycle_golden_identity(g);require(reference==request.expected_payload,"MTP cycle fixture full reference digest differs");
    HipMtpGreedyCycleComponentReceipt receipt;receipt.reference_payload=reference;receipt.cycles.reserve(g.cycles.size());
    std::vector<std::uint16_t> target(static_cast<std::size_t>(width)),mtp(static_cast<std::size_t>(width));
    std::vector<float> logits(static_cast<std::size_t>(vocab));auto current=std::move(initial);
    try{
        for(std::size_t i=0;i<g.cycles.size();++i){const auto& expected=g.cycles[i];
            require(current.source_position()==expected.initial_source_position&&current.predicted_token()==expected.initial_pending_token&&
                expected.final_committed_tokens==add(current.committed_session().committed_tokens,expected.consumed_target_rows),
                "MTP cycle fixture continuation did not retain actual prior ticket/counter frontier");
            const auto actual=greedy.cycle(current,expected.maximum_candidates,expected.token_budget,expected.eos_tokens);
            require(actual.emitted_tokens==expected.emitted_tokens&&actual.proposed_candidates==expected.proposed_candidates&&
                actual.accepted_candidates==expected.accepted_candidates&&actual.consumed_target_rows==expected.consumed_target_rows&&
                actual.eos==expected.eos&&actual.terminal_budget==expected.terminal_budget&&
                actual.attempted_target_window==expected.attempted_window&&actual.accepted_target_window==expected.accepted_window&&
                actual.target_checkpoint_restored==expected.restored_checkpoint&&actual.target_r2_dispatches==expected.target_r2_dispatches&&
                actual.target_projection_launches_saved==expected.target_projection_launches_saved,
                "MTP cycle fixture actual emitted outputs/counters/restore flags differ");
            require(actual.final_target.valid()&&actual.final_target.same_producer(current)&&
                actual.final_target.source_position()==expected.final_source_position&&actual.final_target.predicted_token()==expected.final_pending_token&&
                actual.final_target.committed_session().next_position==add(expected.final_source_position,1U)&&
                actual.final_target.committed_session().committed_tokens==expected.final_committed_tokens&&
                actual.final_target.committed_session().health==SerialSessionHealth::ready&&
                !actual.final_target.committed_session().arithmetic_lineage.tainted&&
                native.next_source_position()==expected.reconciled_next_source_position&&!native.failed()&&!greedy.failed(),
                "MTP cycle fixture final real target/predictor frontier differs");
            read(actual.final_target.view(),target.data(),mul(width,2U),stream);
            read(owner.view(HipMtpBuffer::wide_current),mtp.data(),mul(width,2U),stream);
            read(owner.view(HipMtpBuffer::logits),logits.data(),mul(vocab,4U),stream);check(hipStreamSynchronize(stream));
            finite_hidden(target);finite_hidden(mtp);finite_logits(logits);
            require(target==expected.final_target_wide_bf16&&mtp==expected.reconciled_mtp_wide_bf16&&
                std::memcmp(logits.data(),expected.reconciled_full_logits.data(),static_cast<std::size_t>(mul(vocab,4U)))==0,
                "MTP cycle fixture complete target/reconciled predictor payload differs bitwise");
            receipt.cycles.push_back({actual.final_target.source_position(),actual.consumed_target_rows,
                digest(actual.emitted_tokens),digest(target),digest(mtp),digest(logits)});
            receipt.coverage.accepted_window|=actual.attempted_target_window&&actual.accepted_target_window&&!actual.target_checkpoint_restored;
            receipt.coverage.rejected_window_restored|=actual.attempted_target_window&&!actual.accepted_target_window&&actual.target_checkpoint_restored;
            receipt.coverage.eos|=actual.eos;receipt.coverage.budget|=actual.terminal_budget;receipt.coverage.continuation|=i!=0U;
            current=actual.final_target;
        }
        require_coverage(request.required_coverage,receipt.coverage);receipt.final_target=std::move(current);return receipt;
    }catch(...){native.cancel();throw;}
}
HipMtpGreedyCycleComponentReceipt qualify_hip_mtp_history_capacity_fallback(
    HipMtpGreedyRuntime& greedy,HipMtpRuntime& native,std::shared_ptr<HipMtpTargetHidden> target,
    std::shared_ptr<HipKnownPrefixWindow> window,const ExecutionRuntimeResources& resources,
    std::shared_ptr<const void> target_owner,HipMtpWindowVerificationPolicy policy,
    AllocationGenerationSource& generations,HipMtpTargetHiddenTicket initial,
    const HipMtpGreedyCycleGolden& golden,const HipMtpGreedyCycleComponentRequest& request){
    require(&greedy.native_runtime()==&native&&target&&window&&target_owner&&initial.valid()&&
        resources.session_state_layout&&!golden.cycles.empty(),"MTP history fallback fixture requires real complete owners and trace");
    const auto& first=golden.cycles.front();const auto rows=window->logical_rows();
    const auto& session=initial.committed_session();
    const auto capacity=std::min(resources.session_state_layout->context_tokens,target->model().max_position_embeddings);
    require(policy.capture.maximum_history_intervals&&session.arithmetic_lineage.intervals.size()>=policy.capture.maximum_history_intervals&&
        session.next_position<=capacity&&rows<=capacity-session.next_position&&target->retained_rows()>=rows+2U&&
        first.maximum_candidates>=rows-1U&&first.proposed_candidates>=rows-1U&&first.token_budget>=rows&&
        !is_eos(initial.predicted_token(),first.eos_tokens)&&!first.attempted_window&&!first.accepted_window&&!first.restored_checkpoint&&
        first.consumed_target_rows>0U&&first.target_r2_dispatches==0U&&first.target_projection_launches_saved==0U,
        "MTP history fallback trace must exhaust history but otherwise admit a full window and expect live serial consumption");
    // Actual constructor owns the real policy/capture pools. This is not a
    // boolean mock of use_window; before the fix this reaches Capture::begin,
    // throws and poisons the transaction instead of completing the golden.
    auto verifier=std::make_shared<HipMtpWindowVerifier>(std::move(target),std::move(window),resources,
        std::move(target_owner),policy,generations);
    greedy.use_window_verifier(verifier);
    auto receipt=qualify_hip_mtp_greedy_cycles(greedy,native,std::move(initial),golden,request);
    require(!verifier->failed()&&!greedy.failed()&&!native.failed(),"MTP history fallback must leave actual owners usable");
    return receipt;
}
} // namespace strix::tests
