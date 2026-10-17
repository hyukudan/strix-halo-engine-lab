#include "strix/host_phase_timing.hpp"
#include "strix/hip_mtp_greedy_runtime.hpp"
#include "strix/mtp_sidecar_binding.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include <algorithm>
#include <stdexcept>
namespace strix {
namespace {
void require(bool ok,const char* msg){if(!ok)throw std::invalid_argument(msg);}
bool session_matches(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.state_layout_identity==b.state_layout_identity &&
        a.runtime_compatibility_identity==b.runtime_compatibility_identity &&
        a.next_position==b.next_position && a.committed_tokens==b.committed_tokens &&
        a.health==b.health && a.state.valid==b.state.valid && a.state.generations==b.state.generations &&
        a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted &&
        a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals;
}
}
struct HipMtpGreedyRuntime::Impl {
    std::shared_ptr<HipMtpRuntime> mtp;
    QualifiedMtpDraftPlan plan;
    ActivatedMtpDraftPlan activation;
    std::shared_ptr<HipMtpTargetHidden> target;
    std::shared_ptr<HipMtpWindowVerifier> window;
    SerialSessionContext* session;
    ExecutionRuntimeResources resources;
    DirectLaunchDispatcher* dispatcher;
    std::shared_ptr<const void> owner;
    DraftProposalStateContext draft;
    std::unique_ptr<QualifiedMtpForwardRuntimeAdapter> adapter;
    std::unique_ptr<QualifiedMtpDraftBackend> backend;
    bool busy=false,failed=false,cycled=false;
    float min_probability=0.0F;
    Impl(std::shared_ptr<HipMtpRuntime> m,QualifiedMtpDraftPlan q,ActivatedMtpDraftPlan a,
         std::shared_ptr<HipMtpTargetHidden> t,SerialSessionContext& s,const ExecutionRuntimeResources& r,
         DirectLaunchDispatcher& d,std::shared_ptr<const void> o)
        :mtp(std::move(m)),plan(std::move(q)),activation(std::move(a)),target(std::move(t)),
         session(&s),resources(r),dispatcher(&d),owner(std::move(o)) {
        require(mtp && target && owner,"MTP greedy consumer requires retained actual owners");
        require_qualified_mtp_draft_plan(plan);
        const auto& p=mtp->resources();
        require(plan.native_arithmetic_profile_identity==mtp->numerical_profile_identity() &&
            activation.native_arithmetic_profile_identity==mtp->numerical_profile_identity(),
            "MTP greedy qualification/activation does not bind actual native arithmetic profile");
        require(plan.base.plan_identity==p.plan().plan_identity &&
            plan.base.proposal_identity==p.binding().compatibility_identity,
            "MTP greedy qualified plan differs from actual native owner");
        draft=make_mtp_draft_proposal_state(p.binding(),s);
        adapter=std::make_unique<QualifiedMtpForwardRuntimeAdapter>(p.model(),activation,
            MtpForwardPhysicalResources{&p.manifest(),&p.runtime_binding()},*mtp,*mtp,*mtp);
        backend=std::make_unique<QualifiedMtpDraftBackend>(plan,p.binding(),p.state_memory(),*adapter);
    }
};
HipMtpGreedyRuntime::HipMtpGreedyRuntime(std::shared_ptr<HipMtpRuntime> m,
    QualifiedMtpDraftPlan q,ActivatedMtpDraftPlan a,std::shared_ptr<HipMtpTargetHidden> t,
    SerialSessionContext& s,const ExecutionRuntimeResources& r,DirectLaunchDispatcher& d,
    std::shared_ptr<const void> owner)
    :impl_(std::make_unique<Impl>(std::move(m),std::move(q),std::move(a),std::move(t),s,r,d,std::move(owner))){}
HipMtpGreedyRuntime::~HipMtpGreedyRuntime()=default;
bool HipMtpGreedyRuntime::failed() const noexcept {
    return impl_->failed || impl_->mtp->failed() || (impl_->window && impl_->window->failed());
}
const HipMtpRuntime& HipMtpGreedyRuntime::native_runtime() const noexcept {return *impl_->mtp;}
void HipMtpGreedyRuntime::use_window_verifier(std::shared_ptr<HipMtpWindowVerifier> window) {
    auto& i=*impl_;require(!i.busy && !failed() && !i.cycled && window && window->target_owner()==i.target,
        "MTP window verifier must retain the same actual idle target before the first cycle");
    i.window=std::move(window);
}
void HipMtpGreedyRuntime::set_draft_min_probability(float p) {
    auto& i=*impl_;require(!i.busy && !i.cycled && p>=0.0F && p<=1.0F,"MTP draft minimum probability is set once in [0, 1] before cycles");
    i.min_probability=p;
}
void HipMtpGreedyRuntime::prime_prompt(std::span<const HipMtpReplayRow> rows) {
    auto& i=*impl_;require(!i.busy && !failed(),"MTP greedy consumer is busy or failed");
    i.busy=true;
    try {i.mtp->prime_committed_rows(i.plan,i.activation,rows);i.busy=false;}
    catch(...){i.mtp->cancel();i.failed=true;i.busy=false;poison_draft_proposal_state(i.draft);throw;}
}
HipMtpTargetHiddenTicket HipMtpGreedyRuntime::prefill_prompt(std::span<const std::uint32_t> tokens) {
    auto& i=*impl_;
    require(!i.busy && !failed() && !i.cycled && !tokens.empty() &&
        i.session->next_position==0 && i.mtp->next_source_position()==0 &&
        i.resources.session_state_layout && i.target->retained_rows()>=3,
        "MTP prompt ingestion requires fresh actual target/predictor owners");
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,
        i.mtp->resources().model().max_position_embeddings);
    require(tokens.size()<=capacity && tokens.size()<=i.mtp->resources().context_capacity(),
        "MTP prompt exceeds retained target/predictor capacity");
    for(auto token:tokens) require(token<i.mtp->resources().model().vocab_size,
        "MTP prompt token is outside target vocabulary");
    i.busy=true;
    try {
        HipMtpTargetHiddenTicket previous;
        for(auto token:tokens) {
            auto current=i.target->execute_serial(*i.session,i.resources,*i.dispatcher,token);
            if(previous.valid()) {
                const HipMtpReplayRow row{previous,token};
                i.mtp->prime_committed_rows(i.plan,i.activation,std::span<const HipMtpReplayRow>(&row,1));
            }
            previous=std::move(current);
        }
        i.draft=make_mtp_draft_proposal_state(i.mtp->resources().binding(),*i.session);
        i.busy=false;return previous;
    } catch(...) {i.mtp->cancel();i.failed=true;i.busy=false;poison_draft_proposal_state(i.draft);throw;}
}
HipMtpGreedyCycleResult HipMtpGreedyRuntime::cycle(const HipMtpTargetHiddenTicket& initial,
    std::uint32_t max_candidates,std::uint32_t budget,std::span<const std::uint32_t> eos) {
    auto& i=*impl_;require(!i.busy && !failed() && budget,"MTP greedy cycle requires idle healthy owner and nonzero budget");
    require(initial.valid() && session_matches(initial.committed_session(),*i.session),
        "MTP greedy cycle requires exact current committed target ticket");
    require(initial.target_content_identity()==i.mtp->resources().shared_target().target_model_content_identity,
        "MTP greedy cycle target content differs from bound target");
    auto is_eos=[&](std::uint32_t t){return std::find(eos.begin(),eos.end(),t)!=eos.end();};
    require(i.resources.session_state_layout && i.target->retained_rows()>=3,
        "MTP greedy cycle requires bounded target context and at least three retained target rows");
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,
        i.mtp->resources().model().max_position_embeddings);
    require(i.session->next_position<capacity &&
        initial.source_position()<i.mtp->resources().context_capacity(),
        "MTP greedy cycle has no remaining target/predictor capacity");
    const auto remaining=std::min(capacity-i.session->next_position,
        i.mtp->resources().context_capacity()-initial.source_position());
    auto limit=std::min(max_candidates,i.mtp->resources().budget().maximum_candidate_depth);
    if(i.window)limit=std::min(limit,i.window->logical_rows()-1);
    const auto depth=static_cast<std::uint32_t>(std::min<std::uint64_t>(
        std::min(limit,budget-1),std::min<std::uint64_t>(remaining-1,i.target->retained_rows()-3)));
    HipMtpGreedyCycleResult out;
    out.emitted_tokens.reserve(static_cast<std::size_t>(depth)+2);
    std::vector<HipMtpReplayRow> replay;replay.reserve(static_cast<std::size_t>(depth)+1);
    out.final_target=initial;
    auto pending=initial.predicted_token();
    out.eos=is_eos(pending);
    if(out.eos)return out;
    i.busy=true;
    try {
        host_phase_mark("0 cycle start",true);
        validate_draft_proposal_state(i.draft,*i.session);
        i.cycled=true;
        i.mtp->begin_cycle(pending,initial);
        begin_draft_proposal(i.draft,*i.session);
        host_phase_mark("1 begin cycle");
        const auto candidates=i.mtp->draft_head()?i.mtp->propose_chain(i.plan,i.activation,*i.session,i.draft,depth,i.min_probability):
            i.backend->propose_candidates(*i.session,i.draft,depth);
        host_phase_mark("2 draft chain");
        out.proposed_candidates=static_cast<std::uint32_t>(candidates.size());
        finish_draft_proposal(i.draft);
        host_phase_mark("3 finish proposal");
        if(i.window) {
            auto verified=i.window->verify(*i.session,*i.dispatcher,initial,candidates,budget,eos,
                i.mtp->draft_head()?i.mtp->draft_probabilities():std::span<const float>{});
            host_phase_mark("8 verify tail");
            require(!verified.committed_rows.empty() && verified.committed_rows.size()==verified.emitted_tokens.size(),
                "MTP target verifier returned no actual aligned committed output rows");
            for(const auto& ticket:verified.committed_rows) {
                replay.push_back({out.final_target,ticket.input_token()});out.final_target=ticket;
            }
            out.emitted_tokens=std::move(verified.emitted_tokens);
            out.consumed_target_rows=verified.committed_rows.size();
            out.accepted_candidates=verified.accepted_candidates;
            out.attempted_target_window=verified.attempted_window;out.accepted_target_window=verified.accepted_window;
            out.target_checkpoint_restored=verified.restored_checkpoint;
            out.target_r2_dispatches=verified.r2_dispatches;
            out.target_projection_launches_saved=verified.projection_launches_saved;
            out.target_window_rows=verified.chosen_rows;
            out.eos=is_eos(out.emitted_tokens.back());out.terminal_budget=out.emitted_tokens.size()>=budget;
        } else {
        for(std::size_t index=0;;++index) {
            replay.push_back({out.final_target,pending});
            auto completed=i.target->execute_serial(*i.session,i.resources,*i.dispatcher,pending);
            ++out.consumed_target_rows;
            pending=completed.predicted_token();out.final_target=std::move(completed);
            out.emitted_tokens.push_back(pending);
            const bool accepted=index<candidates.size() && candidates[index]==pending;
            if(accepted)++out.accepted_candidates;
            out.eos=is_eos(pending);out.terminal_budget=out.emitted_tokens.size()>=budget;
            if(out.eos || out.terminal_budget || !accepted)break;
        }
        }
        host_phase_mark("9 cycle bookkeeping");
        i.mtp->set_reconciliation_rows(replay,out.final_target);
        i.backend->reconcile_after_target(i.draft,*i.session,out.emitted_tokens);
        reconcile_draft_proposal_state(i.draft,*i.session);
        host_phase_mark("a reconcile");
        i.busy=false;return out;
    } catch(...) {
        i.mtp->cancel();i.failed=true;i.busy=false;poison_draft_proposal_state(i.draft);
        // The target producer already poisons native execution failures. A
        // reconciliation/proposal failure disables this consumer and publishes
        // no speculative outputs; completed target history is never rolled back.
        throw;
    }
}
} // namespace strix
