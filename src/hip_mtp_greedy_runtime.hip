#include "strix/host_phase_timing.hpp"
#include "strix/hip_mtp_greedy_runtime.hpp"
#include "strix/mtp_sidecar_binding.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include <algorithm>
#include <cstdlib>
#include <functional>
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
    // The drafts of the cycle in progress (between begin_cycle_draft and finish_cycle).
    std::vector<std::uint32_t> candidates;
    std::vector<float> probabilities;
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
    i.window->prepare_admission(*i.session);
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
std::uint32_t HipMtpGreedyRuntime::begin_cycle_draft(const HipMtpTargetHiddenTicket& initial,
    std::uint32_t max_candidates,std::uint32_t budget,std::span<const std::uint32_t> eos,bool cost_model) {
    auto& i=*impl_;require(!i.busy && !failed() && budget,"MTP greedy cycle requires idle healthy owner and nonzero budget");
    require(initial.valid() && session_matches(initial.committed_session(),*i.session),
        "MTP greedy cycle requires exact current committed target ticket");
    require(initial.target_content_identity()==i.mtp->resources().shared_target().target_model_content_identity,
        "MTP greedy cycle target content differs from bound target");
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
    i.candidates.clear();i.probabilities.clear();
    const auto pending=initial.predicted_token();
    if(std::find(eos.begin(),eos.end(),pending)!=eos.end())return depth;
    i.busy=true;
    try {
        host_phase_mark("0 cycle start",true);
        validate_draft_proposal_state(i.draft,*i.session);
        i.cycled=true;
        i.mtp->begin_cycle(pending,initial);
        begin_draft_proposal(i.draft,*i.session);
        host_phase_mark("1 begin cycle");
        // STRIX_MTP_DRAFT_WORTH=0: draft until min_probability only, without asking the window cost model.
        static const bool worth=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_WORTH");return !v || v[0]!='0';}();
        const std::function<bool(std::span<const float>)> keep=worth && cost_model && i.window?
            std::function<bool(std::span<const float>)>([&](std::span<const float> p){return i.window->worth_drafting(p);}):
            std::function<bool(std::span<const float>)>{};
        const auto sampler=i.target->token_sampler();
        require(!sampler || !depth || i.mtp->draft_head(),"MTP speculative sampling requires the draft chain");
        i.candidates=i.mtp->draft_head()?i.mtp->propose_chain(i.plan,i.activation,*i.session,i.draft,depth,i.min_probability,keep):
            i.backend->propose_candidates(*i.session,i.draft,depth);
        if(i.mtp->draft_head()){const auto p=i.mtp->draft_probabilities();i.probabilities.assign(p.begin(),p.end());}
        host_phase_mark("2 draft chain");
        finish_draft_proposal(i.draft);
        host_phase_mark("3 finish proposal");
    } catch(...) {fail_cycle();throw;}
    return depth;
}
std::span<const std::uint32_t> HipMtpGreedyRuntime::drafted_candidates() const noexcept {return impl_->candidates;}
std::span<const float> HipMtpGreedyRuntime::drafted_probabilities() const noexcept {return impl_->probabilities;}
bool HipMtpGreedyRuntime::drafting() const noexcept {return impl_->busy;}
HipMtpWindowVerifier* HipMtpGreedyRuntime::window_verifier() const noexcept {return impl_->window.get();}
SerialSessionContext& HipMtpGreedyRuntime::target_session() const noexcept {return *impl_->session;}
DirectLaunchDispatcher& HipMtpGreedyRuntime::target_dispatcher() const noexcept {return *impl_->dispatcher;}
void HipMtpGreedyRuntime::fail_cycle() noexcept {
    auto& i=*impl_;
    i.mtp->cancel();i.failed=true;i.busy=false;poison_draft_proposal_state(i.draft);
}
HipMtpGreedyCycleResult HipMtpGreedyRuntime::finish_cycle(const HipMtpTargetHiddenTicket& initial,
    HipMtpVerifiedTargetRows verified,std::uint32_t budget,std::span<const std::uint32_t> eos) {
    auto& i=*impl_;require(i.busy && !i.failed,"MTP greedy cycle finish requires a drafted cycle");
    auto is_eos=[&](std::uint32_t t){return std::find(eos.begin(),eos.end(),t)!=eos.end();};
    HipMtpGreedyCycleResult out;
    out.final_target=initial;
    out.proposed_candidates=static_cast<std::uint32_t>(i.candidates.size());
    try {
        host_phase_mark("8 verify tail");
        require(!verified.committed_rows.empty() && verified.committed_rows.size()==verified.emitted_tokens.size(),
            "MTP target verifier returned no actual aligned committed output rows");
        std::vector<HipMtpReplayRow> replay;replay.reserve(verified.committed_rows.size());
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
        host_phase_mark("9 cycle bookkeeping");
        i.mtp->set_reconciliation_rows(replay,out.final_target);
        i.backend->reconcile_after_target(i.draft,*i.session,out.emitted_tokens);
        reconcile_draft_proposal_state(i.draft,*i.session);
        host_phase_mark("a reconcile");
        i.busy=false;return out;
    } catch(...) {fail_cycle();throw;}
}
HipMtpGreedyCycleResult HipMtpGreedyRuntime::cycle(const HipMtpTargetHiddenTicket& initial,
    std::uint32_t max_candidates,std::uint32_t budget,std::span<const std::uint32_t> eos) {
    auto& i=*impl_;
    auto is_eos=[&](std::uint32_t t){return std::find(eos.begin(),eos.end(),t)!=eos.end();};
    const auto depth=begin_cycle_draft(initial,max_candidates,budget,eos,true);
    if(!i.busy){HipMtpGreedyCycleResult out;out.final_target=initial;out.eos=true;return out;}
    if(i.window) {
        HipMtpVerifiedTargetRows verified;
        try {
            verified=i.window->verify(*i.session,*i.dispatcher,initial,i.candidates,budget,eos,
                i.mtp->draft_head()?std::span<const float>(i.probabilities):std::span<const float>{},
                i.target->token_sampler()?i.mtp->sampled_drafts():std::span<const TokenDraft>{});
        } catch(...) {fail_cycle();throw;}
        return finish_cycle(initial,std::move(verified),budget,eos);
    }
    static_cast<void>(depth);
    HipMtpGreedyCycleResult out;
    out.emitted_tokens.reserve(i.candidates.size()+2);
    std::vector<HipMtpReplayRow> replay;replay.reserve(i.candidates.size()+1);
    out.final_target=initial;
    out.proposed_candidates=static_cast<std::uint32_t>(i.candidates.size());
    auto pending=initial.predicted_token();
    try {
        const auto& candidates=i.candidates;
        for(std::size_t index=0;;++index) {
            replay.push_back({out.final_target,pending});
            const auto drafts=i.target->token_sampler()?i.mtp->sampled_drafts():std::span<const TokenDraft>{};
            auto completed=i.target->execute_serial(*i.session,i.resources,*i.dispatcher,pending,
                index<drafts.size()?&drafts[index]:nullptr);
            ++out.consumed_target_rows;
            pending=completed.predicted_token();out.final_target=std::move(completed);
            out.emitted_tokens.push_back(pending);
            const bool accepted=index<candidates.size() && candidates[index]==pending;
            if(accepted)++out.accepted_candidates;
            out.eos=is_eos(pending);out.terminal_budget=out.emitted_tokens.size()>=budget;
            if(out.eos || out.terminal_budget || !accepted)break;
        }
        host_phase_mark("9 cycle bookkeeping");
        i.mtp->set_reconciliation_rows(replay,out.final_target);
        i.backend->reconcile_after_target(i.draft,*i.session,out.emitted_tokens);
        reconcile_draft_proposal_state(i.draft,*i.session);
        host_phase_mark("a reconcile");
        i.busy=false;return out;
    } catch(...) {
        // The target producer already poisons native execution failures. A
        // reconciliation/proposal failure disables this consumer and publishes
        // no speculative outputs; completed target history is never rolled back.
        fail_cycle();
        throw;
    }
}
std::vector<HipMtpGreedyCycleResult> hip_mtp_greedy_cohort_cycle(std::span<const HipMtpGreedyCohortMember> members,
    HipMtpGreedyCohortStats* stats) {
    std::vector<HipMtpGreedyCycleResult> out(members.size());
    std::vector<bool> open(members.size(),false);
    struct Item {std::size_t member;std::uint32_t rows;};
    std::vector<Item> items;
    // Rows per session: at most 8 / sessions (4 for two, 2 for three or four), so that the sessions of a
    // cycle fill one 8-row cohort; drafts beyond that window are not drafted.
    // STRIX_COHORT_MAX_ROWS=2|4|8 overrides it.
    static const std::uint32_t max_rows_override=[]{const char* v=std::getenv("STRIX_COHORT_MAX_ROWS");
        const auto n=v?std::atoi(v):0;return n==2||n==4||n==8?static_cast<std::uint32_t>(n):0U;}();
    const std::uint32_t cap=max_rows_override?max_rows_override:members.size()>=3U?2U:members.size()==2U?4U:8U;
    try {
        for(std::size_t m=0;m<members.size();++m) {
            const auto& c=members[m];
            require(c.runtime && c.initial && c.runtime->window_verifier(),"MTP cohort cycle requires window-verified greedy runtimes");
            for(std::size_t o=0;o<m;++o)require(members[o].runtime!=c.runtime,"MTP cohort cycle requires distinct sessions");
            static_cast<void>(c.runtime->begin_cycle_draft(*c.initial,std::min(c.maximum_candidates,cap-1U),c.budget,c.eos,false));
            if(!c.runtime->drafting()){out[m].final_target=*c.initial;out[m].eos=true;continue;}
            open[m]=true;
            auto rows=c.runtime->window_verifier()->cohort_rows(c.runtime->target_session(),
                c.runtime->drafted_candidates(),c.runtime->drafted_probabilities(),c.budget);
            if(rows>cap && c.runtime->window_verifier()->cohort_fits(c.runtime->target_session(),cap,
                c.runtime->drafted_candidates().size(),c.budget)) rows=cap;
            items.push_back({m,rows});
        }
        // Pack sessions into cohorts of 4 or 8 rows: larger windows first, each into the first cohort
        // with room; a lone session, or one whose owner does not fit, verifies alone.
        std::vector<Item> solo;
        std::vector<std::vector<Item>> groups;
        std::stable_sort(items.begin(),items.end(),[](const Item& a,const Item& b){return a.rows>b.rows;});
        for(const auto& item:items) {
            if(!item.rows || item.rows>=8U){solo.push_back(item);continue;}
            bool placed=false;
            for(auto& g:groups) {
                std::uint32_t sum=0;for(const auto& x:g)sum+=x.rows;
                if(sum+item.rows<=8U){g.push_back(item);placed=true;break;}
            }
            if(!placed)groups.push_back({item});
        }
        const auto fits=[&](const Item& x,std::uint32_t rows) {
            const auto& r=*members[x.member].runtime;
            return r.window_verifier()->cohort_fits(r.target_session(),rows,r.drafted_candidates().size(),members[x.member].budget);
        };
        for(auto& g:groups) {
            for(;;) {
                if(g.size()<2U) break;
                std::uint32_t sum=0;for(const auto& x:g)sum+=x.rows;
                if(sum==4U || sum==8U) break;
                // Round up to the next cohort size with the padding rows of a smaller window.
                const auto target=sum<4U?4U:8U;
                bool grown=false;
                for(auto x=g.rbegin();x!=g.rend() && !grown;++x)
                    for(const auto rows:{x->rows*2U,x->rows*4U})
                        if(rows<=8U && sum-x->rows+rows==target && fits(*x,rows)){x->rows=rows;grown=true;break;}
                if(!grown){solo.push_back(g.back());g.pop_back();}
            }
            if(g.size()==1U){solo.push_back(g.front());g.clear();}
        }
        for(const auto& g:groups) {
            if(g.empty()) continue;
            std::uint32_t total=0;for(const auto& x:g)total+=x.rows;
            // Primary: the idle owner of the cohort's rows of any member that verifies with fewer rows.
            std::shared_ptr<HipKnownPrefixWindow> primary;
            for(const auto& x:g) if(x.rows!=total && (primary=members[x.member].runtime->window_verifier()->owner_with_rows(total))) break;
            require(!!primary,"MTP cohort has no idle owner of its total rows");
            std::vector<HipMtpCohortVerifyRequest> requests;
            for(const auto& x:g) {
                const auto& c=members[x.member];auto& r=*c.runtime;
                requests.push_back({r.window_verifier(),&r.target_session(),c.initial,r.drafted_candidates(),
                    r.drafted_probabilities(),c.budget,c.eos,x.rows});
            }
            auto verified=HipMtpWindowVerifier::verify_cohort(requests,*primary);
            for(std::size_t k=0;k<g.size();++k) {
                const auto& c=members[g[k].member];
                out[g[k].member]=c.runtime->finish_cycle(*c.initial,std::move(verified[k]),c.budget,c.eos);
                open[g[k].member]=false;
            }
            if(stats){++stats->cohort_windows;stats->cohort_sessions+=g.size();stats->cohort_rows+=total;}
        }
        for(const auto& x:solo) {
            const auto& c=members[x.member];auto& r=*c.runtime;
            auto verified=r.window_verifier()->verify(r.target_session(),r.target_dispatcher(),*c.initial,r.drafted_candidates(),
                c.budget,c.eos,r.drafted_probabilities());
            out[x.member]=r.finish_cycle(*c.initial,std::move(verified),c.budget,c.eos);
            open[x.member]=false;
            if(stats)++stats->solo_windows;
        }
        return out;
    } catch(...) {
        for(std::size_t m=0;m<members.size();++m) if(open[m]) members[m].runtime->fail_cycle();
        throw;
    }
}
} // namespace strix
