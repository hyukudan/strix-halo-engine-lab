#include "strix/hip_mtp_window_verifier.hpp"
#include "strix/mtp_target_window_policy.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
namespace strix {
namespace {
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
bool session_equal(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.state_layout_identity==b.state_layout_identity &&
        a.runtime_compatibility_identity==b.runtime_compatibility_identity && a.next_position==b.next_position &&
        a.committed_tokens==b.committed_tokens && a.health==b.health && a.state.valid==b.state.valid &&
        a.state.generations==b.state.generations && a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted && a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals;
}
}
struct HipMtpWindowVerifier::Impl {
    std::shared_ptr<HipMtpTargetHidden> target;
    std::shared_ptr<HipKnownPrefixWindow> window;
    ExecutionRuntimeResources resources;
    std::shared_ptr<const void> resource_owner;
    std::array<std::shared_ptr<HipMtpTargetWindowCapture>,2> captures;
    std::unique_ptr<HipMtpTargetCheckpoint> checkpoint;
    std::uint64_t retained=0,maximum_history_intervals=0;
    bool busy=false,failed=false,bounded=false;
    Impl(std::shared_ptr<HipMtpTargetHidden> t,std::shared_ptr<HipKnownPrefixWindow> w,
         const ExecutionRuntimeResources& r,std::shared_ptr<const void> owner,
         HipMtpWindowVerificationPolicy policy,AllocationGenerationSource& generations)
        :target(std::move(t)),window(std::move(w)),resources(r),resource_owner(std::move(owner)),
         maximum_history_intervals(policy.capture.maximum_history_intervals),bounded(policy.bounded_window_rollback) {
        require(target && window && resource_owner && resources.session_state_layout &&
            resources.session_state_memory==&target->memory_owner()->state() &&
            resources.scratch==&target->memory_owner()->scratch() &&
            resources.model_weights==&target->weights_owner()->binding() &&
            window->output_policy()==KnownWindowOutputPolicy::all_rows &&
            (window->logical_rows()==2 || window->logical_rows()==4 || window->logical_rows()==8),
            "MTP window verifier requires actual closed all-row target owners");
        const auto reservation=bounded?plan_hip_mtp_target_window_rollback(target->model(),*resources.session_state_layout,0,window->logical_rows(),true).reservation:
            plan_hip_mtp_target_checkpoint(target->model(),*resources.session_state_layout,policy.snapshot_policy);
        const auto upper=mtp_native::sum(reservation.retained_bytes,
            mtp_native::product(policy.capture.retained_byte_budget,2));
        require(policy.total_retained_byte_budget && reservation.retained_bytes<=policy.checkpoint_retained_byte_budget &&
            upper<=policy.total_retained_byte_budget,"MTP window verification exceeds upfront combined retained reservation");
        for(auto& capture:captures) {
            capture=window->prepare_mtp_target_capture(policy.capture);
            target->validate_window_capture(*capture);
            retained=mtp_native::sum(retained,capture->retained_payload_bytes());
        }
        checkpoint=std::make_unique<HipMtpTargetCheckpoint>(target->model(),target->compiled(),*resources.session_state_layout,
            target->memory_owner(),target->weights_owner(),target->allocator(),static_cast<hipStream_t>(target->stream()),
            target,target->guard(),policy.snapshot_policy,policy.checkpoint_retained_byte_budget,generations,
            bounded?captures.front():std::shared_ptr<const HipMtpTargetWindowCapture>{});
        retained=mtp_native::sum(retained,checkpoint->reserved_bytes());
        require(retained<=policy.total_retained_byte_budget,"MTP actual verification pools exceed admitted reservation");
    }
};
HipMtpWindowVerifier::HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden> target,
    std::shared_ptr<HipKnownPrefixWindow> window,const ExecutionRuntimeResources& resources,
    std::shared_ptr<const void> owner,HipMtpWindowVerificationPolicy policy,AllocationGenerationSource& generations)
    :impl_(std::make_unique<Impl>(std::move(target),std::move(window),resources,std::move(owner),policy,generations)){}
HipMtpWindowVerifier::~HipMtpWindowVerifier()=default;
std::shared_ptr<HipMtpTargetHidden> HipMtpWindowVerifier::target_owner() const noexcept {return impl_->target;}
std::shared_ptr<HipKnownPrefixWindow> HipMtpWindowVerifier::window_owner() const noexcept {return impl_->window;}
std::uint32_t HipMtpWindowVerifier::logical_rows() const noexcept {return impl_->window->logical_rows();}
bool HipMtpWindowVerifier::failed() const noexcept {return impl_->failed || impl_->checkpoint->failed();}
std::uint64_t HipMtpWindowVerifier::retained_payload_bytes() const noexcept {return impl_->retained;}
HipMtpVerifiedTargetRows HipMtpWindowVerifier::verify(SerialSessionContext& session,DirectLaunchDispatcher& dispatcher,
    const HipMtpTargetHiddenTicket& initial,std::span<const std::uint32_t> candidates,
    std::uint32_t budget,std::span<const std::uint32_t> eos) {
    auto& i=*impl_;
    require(!i.busy && !failed() && budget && initial.valid() &&
        initial.source_position()+1==session.next_position && session_equal(initial.committed_session(),session),
        "MTP window verifier requires exact idle committed target frontier");
    const auto vocabulary=i.target->model().vocab_size;
    for(auto token:candidates)require(token<vocabulary,"MTP target candidate outside vocabulary");
    for(auto token:eos)require(token<vocabulary,"MTP target EOS outside vocabulary");
    auto is_eos=[&](std::uint32_t token){return std::find(eos.begin(),eos.end(),token)!=eos.end();};
    const auto rows=i.window->logical_rows();
    require(candidates.size()<rows,"MTP verifier admits only one closed window per cycle");
    auto pending=initial.predicted_token();
    HipMtpVerifiedTargetRows out;out.committed_rows.reserve(candidates.size()+1);out.emitted_tokens.reserve(candidates.size()+1);
    if(is_eos(pending))return out;
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,i.target->model().max_position_embeddings);
    require(session.next_position<capacity,"MTP verifier target context is full");
    // Capture::begin needs room for one actual known-window chunk. Capacity
    // exhaustion is an ordinary serial fallback, before pool/checkpoint work.
    const bool use_window=candidates.size()+1>=rows && budget>=rows && capacity-session.next_position>=rows &&
        session.arithmetic_lineage.intervals.size()<i.maximum_history_intervals;
    std::shared_ptr<HipMtpTargetWindowCapture> capture;
    if(use_window) {
        for(const auto& pool:i.captures)if(pool->ready_for_reuse()){capture=pool;break;}
        require(!!capture,"MTP window capture pools are retained by previous batch readers");
        i.target->validate_window_capture(*capture);
    }
    i.busy=true;bool attempted_target=false;
    try {
        if(use_window) {
            std::vector<std::uint32_t> inputs;inputs.reserve(rows);inputs.push_back(pending);
            inputs.insert(inputs.end(),candidates.begin(),candidates.begin()+static_cast<std::ptrdiff_t>(rows-1));
            if(i.bounded)i.checkpoint->capture_window(session);else i.checkpoint->capture(session);
            auto native=i.window->execute_mtp_captured(session,inputs,capture);
            // Successful return proves actual target consumption. Rejected
            // admission leaves the original ready session untouched; dispatch
            // failures are already poisoned/drained by the native producer.
            attempted_target=true;out.attempted_window=true;
            out.r2_dispatches=native.execution.r2_dispatches;
            out.projection_launches_saved=native.execution.projection_launches_saved;
            auto provisional=std::make_shared<const HipMtpTargetWindowBatch>(std::move(native.batch));
            std::vector<std::uint32_t> selected;selected.reserve(rows);
            for(std::uint32_t row=0;row<rows;++row)selected.push_back(provisional->predicted_token(row));
            const auto decision=assess_mtp_target_window(inputs,selected,pending,budget,eos,vocabulary);
            if(decision.accepts_complete_chunk()) {
                out.committed_rows=i.target->accept_verified_window(provisional,initial,session,eos,budget);
                out.emitted_tokens=std::move(selected);out.accepted_candidates=rows-1;out.accepted_window=true;
                i.checkpoint->discard();i.busy=false;return out;
            }
            // No provisional row has been promoted. Restore the exact actual
            // checkpoint, then select/reverify the serial greedy continuation.
            if(i.bounded)i.checkpoint->restore_window(session,*provisional);else i.checkpoint->restore(session);
            provisional.reset();out.restored_checkpoint=true;
            require(session_equal(initial.committed_session(),session),"MTP rejected window did not restore exact target frontier");
        }
        for(std::size_t index=0;out.emitted_tokens.size()<budget && session.next_position<capacity;++index) {
            auto ticket=i.target->execute_serial(session,i.resources,dispatcher,pending);
            attempted_target=true;
            pending=ticket.predicted_token();out.committed_rows.push_back(std::move(ticket));out.emitted_tokens.push_back(pending);
            const bool accepted=index<candidates.size() && candidates[index]==pending;
            if(accepted)++out.accepted_candidates;
            if(is_eos(pending) || !accepted)break;
        }
        i.busy=false;return out;
    } catch(...) {
        i.failed=true;i.busy=false;
        // Producers/checkpoint own cleanup for their submitted operations.
        // Do not synchronize or poison a purely rejected predispatch scope.
        // A failure after any successful target consumption still cannot
        // publish a reusable partially completed verification transaction.
        if(attempted_target || session.health==SerialSessionHealth::poisoned){
            session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
            i.target->guard()->recover_or_fail_stop();
        }
        throw;
    }
}
} // namespace strix
