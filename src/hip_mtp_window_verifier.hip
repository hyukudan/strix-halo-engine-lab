#include "strix/host_phase_timing.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include "strix/mtp_target_window_policy.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/arithmetic_lineage.hpp"
#include "strix/session_state_memory.hpp"
#include "strix/hip_generation_output.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string_view>
#include <optional>
#include <stdexcept>
namespace strix {
namespace {
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
void write_position_word(const ExecutionRuntimeResources& resources,std::uint64_t value) {
    for(const auto& allocation:resources.session_state_layout->allocations) {
        if(allocation.component!=SessionStateComponent::position_history) continue;
        require(allocation.storage==SessionStateStorage::host && allocation.bytes>=sizeof(value) &&
            resources.session_state_memory->host_base,"MTP prefix commit requires the host position word");
        std::memcpy(reinterpret_cast<void*>(resources.session_state_memory->host_base+allocation.offset),&value,sizeof(value));
        return;
    }
    throw std::invalid_argument("MTP prefix commit found no position history");
}
bool session_equal(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.state_layout_identity==b.state_layout_identity &&
        a.runtime_compatibility_identity==b.runtime_compatibility_identity && a.next_position==b.next_position &&
        a.committed_tokens==b.committed_tokens && a.health==b.health && a.state.valid==b.state.valid &&
        a.state.generations==b.state.generations && a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted && a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals;
}
}
struct HipMtpWindowVerifier::Impl {
    struct Owner {
        std::shared_ptr<HipKnownPrefixWindow> window;
        std::array<std::shared_ptr<HipMtpTargetWindowCapture>,4> captures;
        std::uint32_t rows=0;
        HipMtpWindowOwnerStats stats;
        std::shared_ptr<void> greedy_partials,greedy_replies;
        bool warmed=false; // first window of an owner pays one-time admission and first-touch costs
    };
    std::shared_ptr<HipMtpTargetHidden> target;
    std::vector<Owner> owners; // ascending rows
    std::shared_ptr<HipKnownPrefixWindow> window; // largest owner; the checkpoint path has exactly one
    ExecutionRuntimeResources resources;
    std::shared_ptr<const void> resource_owner;
    std::unique_ptr<HipMtpTargetCheckpoint> checkpoint;
    std::uint64_t retained=0,maximum_history_intervals=0;
    bool busy=false,failed=false,bounded=false,prefix=false;
    // Adaptive choice: owner cost per cycle (draft included), serial cost per emitted token.
    float min_probability=0.0F;
    double serial_ms_ema=0.0;std::uint64_t serial_cycles=0;
    // Draft probabilities are not acceptance rates: per probability bin, verified
    // drafts (every draft up to the first rejection) and how many were accepted.
    std::array<double,10> tested{},accepted{};
    Owner* last=nullptr; // owner of the last prefix window; its stream carries the PLE commit
    std::optional<std::chrono::steady_clock::time_point> last_end;
    Impl(std::shared_ptr<HipMtpTargetHidden> t,std::vector<std::shared_ptr<HipKnownPrefixWindow>> w,
         const ExecutionRuntimeResources& r,std::shared_ptr<const void> owner,
         HipMtpWindowVerificationPolicy policy,AllocationGenerationSource& generations)
        :target(std::move(t)),resources(r),resource_owner(std::move(owner)),
         maximum_history_intervals(policy.capture.maximum_history_intervals),bounded(policy.bounded_window_rollback),
         min_probability(policy.min_probability) {
        require(target && resource_owner && resources.session_state_layout && !w.empty() &&
            resources.session_state_memory==&target->memory_owner()->state() &&
            resources.scratch==&target->memory_owner()->scratch() &&
            resources.model_weights==&target->weights_owner()->binding() &&
            min_probability>=0.0F && min_probability<=1.0F,"MTP window verifier requires actual target owners");
        for(const auto& candidate:w) require(!!candidate,"MTP window verifier requires actual closed all-row target owners");
        std::sort(w.begin(),w.end(),[](const auto& a,const auto& b){return a->logical_rows()<b->logical_rows();});
        for(const auto& candidate:w) {
            require(candidate->output_policy()==KnownWindowOutputPolicy::all_rows &&
                (candidate->logical_rows()==2 || candidate->logical_rows()==4 || candidate->logical_rows()==8),
                "MTP window verifier requires actual closed all-row target owners");
            require(owners.empty() || (owners.back().rows<candidate->logical_rows() && candidate->defers_state() &&
                owners.front().window->defers_state()),"MTP window owners must have distinct rows and prefix acceptance");
            owners.push_back({candidate,{},candidate->logical_rows(),{candidate->logical_rows(),0,0.0},{},{},false});
        }
        window=owners.back().window;
        prefix=window->defers_state();
        for(std::size_t o=1;o<owners.size();++o) owners[o].window->share_deferred(*owners.front().window);
        const auto reservation=bounded?plan_hip_mtp_target_window_rollback(target->model(),*resources.session_state_layout,0,window->logical_rows(),true).reservation:
            plan_hip_mtp_target_checkpoint(target->model(),*resources.session_state_layout,policy.snapshot_policy);
        const auto upper=mtp_native::sum(reservation.retained_bytes,
            mtp_native::product(policy.capture.retained_byte_budget,4U*owners.size()));
        require(policy.total_retained_byte_budget && reservation.retained_bytes<=policy.checkpoint_retained_byte_budget &&
            upper<=policy.total_retained_byte_budget,"MTP window verification exceeds upfront combined retained reservation");
        // Every row's argmax runs on the window stream right after its head rows and lands in
        // host-mapped memory: one completion per window instead of a selector round trip per
        // row. STRIX_MTP_GREEDY_ROWS=0 keeps the per-row selectors.
        const char* greedy_rows=std::getenv("STRIX_MTP_GREEDY_ROWS");
        if(!greedy_rows || std::string_view(greedy_rows)!="0") for(auto& o:owners) {
            const auto vocab=static_cast<std::uint32_t>(target->model().vocab_size);
            const auto bytes=hip_greedy_rows_scratch_bytes(o.rows,vocab);
            void* partials=nullptr;void* replies=nullptr;
            require(hipMalloc(&partials,bytes)==hipSuccess,"MTP greedy rows scratch");
            o.greedy_partials=std::shared_ptr<void>(partials,[](void* p){(void)hipFree(p);});
            require(hipHostMalloc(&replies,sizeof(HipGreedyRowsReply)*o.rows,hipHostMallocMapped|hipHostMallocCoherent)==hipSuccess,
                "MTP greedy rows replies");
            o.greedy_replies=std::shared_ptr<void>(replies,[](void* p){(void)hipHostFree(p);});
            ExecutionBufferAddress view{};view.address=reinterpret_cast<std::uintptr_t>(partials);view.bytes=bytes;
            o.window->attach_greedy_rows(view,static_cast<HipGreedyRowsReply*>(replies));
        }
        for(auto& o:owners) for(auto& capture:o.captures) {
            capture=o.window->prepare_mtp_target_capture(policy.capture);
            target->validate_window_capture(*capture);
            retained=mtp_native::sum(retained,capture->retained_payload_bytes());
        }
        // Completed serial metadata admits capture, but bounded rollback belongs
        // to the actual known-window producer queue and its retained owner.
        // Full snapshots retain the original serial-target queue contract.
        if(prefix) return; // deferred GDN + PLE snapshots replace the state checkpoint
        const auto& captures=owners.back().captures;
        const auto checkpoint_stream=bounded?captures.front()->source_stream():target->stream();
        const std::shared_ptr<const void> checkpoint_stream_owner=bounded?
            std::shared_ptr<const void>(window):std::shared_ptr<const void>(target);
        checkpoint=std::make_unique<HipMtpTargetCheckpoint>(target->model(),target->compiled(),*resources.session_state_layout,
            target->memory_owner(),target->weights_owner(),target->allocator(),static_cast<hipStream_t>(checkpoint_stream),
            checkpoint_stream_owner,target->guard(),policy.snapshot_policy,policy.checkpoint_retained_byte_budget,generations,
            bounded?captures.front():std::shared_ptr<const HipMtpTargetWindowCapture>{});
        retained=mtp_native::sum(retained,checkpoint->reserved_bytes());
        require(retained<=policy.total_retained_byte_budget,"MTP actual verification pools exceed admitted reservation");
    }
    static std::size_t bin(float p){return std::min<std::size_t>(9U,static_cast<std::size_t>(std::max(p,0.0F)*10.0F));}
    // Calibrated acceptance: the bin's observed rate with a two-draft prior at p.
    double acceptance(float p) const {const auto b=bin(p);return (accepted[b]+2.0*p)/(tested[b]+2.0);}
    void calibrate(std::span<const float> p,std::size_t accepted_drafts,std::size_t verified_rows) {
        for(std::size_t j=0;j<p.size() && j<verified_rows && j<=accepted_drafts;++j){
            const auto b=bin(p[j]);tested[b]+=1.0;if(j<accepted_drafts)accepted[b]+=1.0;}
    }
    // Expected emitted tokens of an R-row prefix window when draft j is accepted
    // with its calibrated probability, given that every earlier draft was accepted.
    double expected_tokens(std::uint32_t rows,std::span<const float> p) const {
        double expected=1.0,run=1.0;
        for(std::uint32_t j=1;j<rows && j<=p.size();++j){run*=acceptance(p[j-1U]);expected+=run;}
        return expected;
    }
    // nullptr = serial target steps.
    template<class Fits> Owner* choose(std::span<const float> p,std::size_t candidates,Fits fits) {
        auto largest=[&]()->Owner*{for(auto o=owners.rbegin();o!=owners.rend();++o) if(fits(*o)) return &*o;return nullptr;};
        if(min_probability<=0.0F || p.size()!=candidates) return largest();
        std::uint32_t want=1;
        while(want-1U<p.size() && p[want-1U]>=min_probability) ++want;
        Owner* cap=&owners.back();
        for(auto& o:owners) if(o.rows>=want){cap=&o;break;}
        const bool serial=want==1U;
        // Unmeasured options are tried once before the cost model ranks them.
        if(fits(*cap) && !cap->stats.cycles) return cap;
        if(serial && !serial_cycles) return nullptr;
        Owner* best=nullptr;double best_rate=0.0;
        for(auto& o:owners) {
            if(o.rows>cap->rows || !fits(o) || !o.stats.cycles) continue;
            const auto rate=expected_tokens(o.rows,p)/o.stats.cycle_ms_ema;
            if(!best || rate>best_rate){best=&o;best_rate=rate;}
        }
        if(serial && (!best || 1.0/serial_ms_ema>best_rate)) return nullptr;
        return best?best:largest();
    }
    // Wall time between consecutive verified cycles, attributed to this cycle's choice.
    void observe(Owner* owner,std::size_t emitted) {
        const auto now=std::chrono::steady_clock::now();
        if(last_end && emitted) {
            const auto ms=std::chrono::duration<double,std::milli>(now-*last_end).count();
            static const bool log_cycles=std::getenv("STRIX_CYCLE_LOG")!=nullptr;
            if(log_cycles) std::fprintf(stderr,"cycle rows=%u emitted=%zu ms=%.2f\n",owner?owner->rows:1U,emitted,ms);
            constexpr double alpha=0.25;
            // A sample is capped at twice the running estimate: one slow cycle (page faults,
            // a stalled host) must not price an owner out of every later choice.
            auto update=[&](double& ema,std::uint64_t& count,double value){
                ema=count?ema+alpha*(std::min(value,2.0*ema)-ema):value;++count;};
            if(owner) {
                // The owner's first window is not a cost sample; it stays unmeasured and is tried again.
                if(owner->warmed) update(owner->stats.cycle_ms_ema,owner->stats.cycles,ms);
                owner->warmed=true;
            } else update(serial_ms_ema,serial_cycles,ms/static_cast<double>(emitted));
        }
        last_end=now;
    }
};
HipMtpWindowVerifier::HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden> target,
    std::shared_ptr<HipKnownPrefixWindow> window,const ExecutionRuntimeResources& resources,
    std::shared_ptr<const void> owner,HipMtpWindowVerificationPolicy policy,AllocationGenerationSource& generations)
    :HipMtpWindowVerifier(std::move(target),std::vector<std::shared_ptr<HipKnownPrefixWindow>>{std::move(window)},
        resources,std::move(owner),policy,generations){}
HipMtpWindowVerifier::HipMtpWindowVerifier(std::shared_ptr<HipMtpTargetHidden> target,
    std::vector<std::shared_ptr<HipKnownPrefixWindow>> windows,const ExecutionRuntimeResources& resources,
    std::shared_ptr<const void> owner,HipMtpWindowVerificationPolicy policy,AllocationGenerationSource& generations)
    :impl_(std::make_unique<Impl>(std::move(target),std::move(windows),resources,std::move(owner),policy,generations)){}
HipMtpWindowVerifier::~HipMtpWindowVerifier()=default;
std::shared_ptr<HipMtpTargetHidden> HipMtpWindowVerifier::target_owner() const noexcept {return impl_->target;}
std::shared_ptr<HipKnownPrefixWindow> HipMtpWindowVerifier::window_owner() const noexcept {return impl_->window;}
std::vector<std::shared_ptr<HipKnownPrefixWindow>> HipMtpWindowVerifier::window_owners() const {
    std::vector<std::shared_ptr<HipKnownPrefixWindow>> out;for(const auto& o:impl_->owners)out.push_back(o.window);return out;
}
std::vector<HipMtpWindowOwnerStats> HipMtpWindowVerifier::owner_stats() const {
    std::vector<HipMtpWindowOwnerStats> out{{1U,impl_->serial_cycles,impl_->serial_ms_ema}};
    for(const auto& o:impl_->owners)out.push_back(o.stats);
    return out;
}
std::uint32_t HipMtpWindowVerifier::logical_rows() const noexcept {return impl_->window->logical_rows();}
bool HipMtpWindowVerifier::failed() const noexcept {return impl_->failed || (impl_->checkpoint && impl_->checkpoint->failed());}
std::uint64_t HipMtpWindowVerifier::retained_payload_bytes() const noexcept {return impl_->retained;}
HipMtpVerifiedTargetRows HipMtpWindowVerifier::verify(SerialSessionContext& session,DirectLaunchDispatcher& dispatcher,
    const HipMtpTargetHiddenTicket& initial,std::span<const std::uint32_t> candidates,
    std::uint32_t budget,std::span<const std::uint32_t> eos,std::span<const float> probabilities) {
    auto& i=*impl_;
    require(!i.busy && !failed() && budget && initial.valid() &&
        initial.source_position()+1==session.next_position && session_equal(initial.committed_session(),session),
        "MTP window verifier requires exact idle committed target frontier");
    const auto vocabulary=i.target->model().vocab_size;
    for(auto token:candidates)require(token<vocabulary,"MTP target candidate outside vocabulary");
    for(auto token:eos)require(token<vocabulary,"MTP target EOS outside vocabulary");
    auto is_eos=[&](std::uint32_t token){return std::find(eos.begin(),eos.end(),token)!=eos.end();};
    require(candidates.size()<i.window->logical_rows(),"MTP verifier admits only one closed window per cycle");
    auto pending=initial.predicted_token();
    HipMtpVerifiedTargetRows out;out.committed_rows.reserve(candidates.size()+1);out.emitted_tokens.reserve(candidates.size()+1);
    if(is_eos(pending))return out;
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,i.target->model().max_position_embeddings);
    require(session.next_position<capacity,"MTP verifier target context is full");
    // Capture::begin needs room for one actual known-window chunk. Capacity
    // exhaustion is an ordinary serial fallback, before pool/checkpoint work.
    // Adaptive prefix windows may pad unconfident rows: they are never accepted.
    const bool pad=i.prefix && i.min_probability>0.0F;
    const auto room=capacity-session.next_position;
    const bool history=session.arithmetic_lineage.intervals.size()<i.maximum_history_intervals;
    auto* owner=i.choose(probabilities,candidates.size(),[&](const Impl::Owner& o){
        return budget>=o.rows && room>=o.rows && history && (pad?!candidates.empty():candidates.size()+1>=o.rows);});
    const bool use_window=owner!=nullptr;
    const auto rows=use_window?owner->rows:1U;
    out.chosen_rows=rows;
    std::shared_ptr<HipMtpTargetWindowCapture> capture;
    if(use_window) {
        for(const auto& pool:owner->captures)if(pool->ready_for_reuse()){capture=pool;break;}
        require(!!capture,"MTP window capture pools are retained by previous batch readers");
        i.target->validate_window_capture(*capture);
    }
    i.busy=true;bool attempted_target=false;
    try {
        if(use_window && i.prefix) {
            // The previous owner's PLE commit runs on its own stream.
            if(i.last && i.last!=owner && hipStreamSynchronize(static_cast<hipStream_t>(i.last->captures.front()->source_stream()))!=hipSuccess)
                throw std::runtime_error("MTP window owner switch could not complete the previous commit");
            std::vector<std::uint32_t> inputs;inputs.reserve(rows);inputs.push_back(pending);
            for(std::uint32_t r=1;r<rows;++r) inputs.push_back(candidates[std::min<std::size_t>(r-1U,candidates.size()-1U)]);
            const auto previous=session;
            host_phase_mark("4 verify prologue");
            auto native=owner->window->execute_mtp_captured(session,inputs,capture);
            host_phase_mark("7 window launch+wait");
            attempted_target=true;out.attempted_window=true;
            out.r2_dispatches=native.execution.r2_dispatches;
            out.projection_launches_saved=native.execution.projection_launches_saved;
            auto provisional=std::make_shared<const HipMtpTargetWindowBatch>(std::move(native.batch));
            std::vector<std::uint32_t> selected;selected.reserve(rows);
            for(std::uint32_t row=0;row<rows;++row)selected.push_back(provisional->predicted_token(row));
            // Longest accepted prefix: row a+1's input equals row a's prediction.
            std::uint32_t a=0;
            while(a+1U<rows && a<candidates.size() && a+1U<budget && inputs[a+1U]==selected[a] && !is_eos(selected[a])) ++a;
            const auto n_keep=a+1U;
            host_phase_mark("7a select/accept");
            owner->window->commit_prefix(n_keep);i.last=owner;
            host_phase_mark("7b commit prefix");
            if(n_keep<rows) {
                const auto dropped=rows-n_keep;
                session.next_position-=dropped;session.committed_tokens-=dropped;
                const auto& last=session.arithmetic_lineage.intervals.back();
                session.arithmetic_lineage=prepare_known_window_chunk_lineage(previous.arithmetic_lineage,last.begin,
                    last.begin+n_keep,last.numerical_plan_identity,last.reduction_plan_identity,last.state_update_identity);
                write_position_word(i.resources,session.next_position-1U);
            }
            host_phase_mark("7c lineage");
            out.committed_rows=i.target->accept_verified_prefix(provisional,initial,session,n_keep);
            host_phase_mark("7d accept prefix");
            out.emitted_tokens.assign(selected.begin(),selected.begin()+n_keep);
            out.accepted_candidates=a;out.accepted_window=n_keep==rows;
            if(probabilities.size()==candidates.size())i.calibrate(probabilities,a,rows-1U);
            i.observe(owner,n_keep);
            host_phase_mark("7e calibrate/observe");
            i.busy=false;return out;
        }
        if(i.prefix && i.last) i.last->window->flush_deferred();
        if(use_window) {
            std::vector<std::uint32_t> inputs;inputs.reserve(rows);inputs.push_back(pending);
            inputs.insert(inputs.end(),candidates.begin(),candidates.begin()+static_cast<std::ptrdiff_t>(rows-1));
            if(i.bounded)i.checkpoint->capture_window(session);else i.checkpoint->capture(session);
            auto native=owner->window->execute_mtp_captured(session,inputs,capture);
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
                i.checkpoint->discard();i.observe(owner,rows);i.busy=false;return out;
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
        if(probabilities.size()==candidates.size())i.calibrate(probabilities,out.accepted_candidates,out.emitted_tokens.size());
        i.observe(use_window?owner:nullptr,out.emitted_tokens.size());
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
