#include "strix/known_window_cohort.hpp"
#include "strix/host_phase_timing.hpp"
#include "strix/hip_mtp_greedy_runtime.hpp"
#include "strix/mtp_sidecar_binding.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
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
    std::vector<float> probabilities,runner_ups;
    // A draft chain between draft_begin and draft_end (begin_cycle_draft runs all phases in order).
    struct DraftState {
        std::uint32_t depth=0,pending=0;bool cost_model=false,survival=false,full_chain=false,open=false;
        std::chrono::steady_clock::time_point start{};
    } ds;
    std::unique_ptr<MtpNgramDraft> ngram;
    std::vector<double> ngram_acceptance;
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
void HipMtpGreedyRuntime::enable_ngram_draft(std::span<const std::uint32_t> prompt,MtpNgramDraftConfig config) {
    auto& i=*impl_;
    require(!i.busy && !i.cycled && i.window && !i.window->survival_controller() && !i.target->token_sampler(),
        "MTP n-gram draft extension needs the window verifier with the min-p controller, greedy selection and no cycle yet");
    i.ngram=std::make_unique<MtpNgramDraft>(config);
    i.ngram->reset(prompt);
}
const MtpNgramDraftStats* HipMtpGreedyRuntime::ngram_stats() const noexcept {
    return impl_->ngram?&impl_->ngram->stats():nullptr;
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
    const auto depth=draft_begin(initial,max_candidates,budget,eos,cost_model);
    if(!drafting()) return depth;
    while(draft_launch_step() && draft_after_step()) {}
    draft_end();
    return depth;
}
std::uint32_t HipMtpGreedyRuntime::draft_begin(const HipMtpTargetHiddenTicket& initial,
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
        // The survival controller owns the chain cut (no min-p stop) and sees runner-up probabilities.
        // STRIX_MTP_DIAG_FULL_CHAIN=1 drafts the full depth without any cut.
        const bool full_chain=hip_mtp_full_chain_diagnostic();
        const bool survival=cost_model && i.window && i.window->survival_controller() && !full_chain;
        // Without a min-p cut (and without the survival controller) the chain runs sync-free: no keep test.
        const std::function<bool(std::span<const float>)> keep=((worth && i.min_probability>0.0F) || survival) && cost_model && i.window && !full_chain?
            std::function<bool(std::span<const float>)>([&](std::span<const float> p){
                return i.window->worth_drafting(p,i.mtp->draft_runner_up_probabilities());}):
            std::function<bool(std::span<const float>)>{};
        const auto sampler=i.target->token_sampler();
        require(!sampler || !depth || i.mtp->draft_head(),"MTP speculative sampling requires the draft chain");
        i.ds=Impl::DraftState{};
        i.ds.depth=depth;i.ds.pending=pending;i.ds.cost_model=cost_model;i.ds.survival=survival;i.ds.full_chain=full_chain;
        i.ds.start=std::chrono::steady_clock::now();
        if(i.mtp->draft_head())
            i.ds.open=i.mtp->propose_chain_begin(i.plan,i.activation,*i.session,i.draft,depth,
                full_chain || survival?0.0F:i.min_probability,keep);
        else i.candidates=i.backend->propose_candidates(*i.session,i.draft,depth);
    } catch(...) {fail_cycle();throw;}
    return depth;
}
bool HipMtpGreedyRuntime::draft_launch_step() {
    auto& i=*impl_;
    if(!i.busy || !i.ds.open) return false;
    try {return i.mtp->propose_chain_launch_step();} catch(...) {fail_cycle();throw;}
}
bool HipMtpGreedyRuntime::draft_after_step() {
    auto& i=*impl_;
    if(!i.busy || !i.ds.open) return false;
    try {return i.mtp->propose_chain_after_step();} catch(...) {fail_cycle();throw;}
}
void HipMtpGreedyRuntime::draft_end() {
    auto& i=*impl_;require(i.busy && !i.failed,"MTP draft end requires a begun draft");
    const auto depth=i.ds.depth;const auto pending=i.ds.pending;const bool cost_model=i.ds.cost_model;
    const bool survival=i.ds.survival,full_chain=i.ds.full_chain;
    try {
        if(i.ds.open) {i.candidates=i.mtp->propose_chain_finish();i.ds.open=false;}
        else if(i.mtp->draft_head()) i.candidates.clear();
        if(survival) i.window->note_draft(static_cast<std::uint32_t>(i.candidates.size()),
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-i.ds.start).count());
        i.runner_ups.clear();
        if(i.mtp->draft_head()){
            const auto p=i.mtp->draft_probabilities();i.probabilities.assign(p.begin(),p.end());
            const auto q=i.mtp->draft_runner_up_probabilities();i.runner_ups.assign(q.begin(),q.end());
        }
        if(i.ngram && cost_model && !survival && !full_chain && i.window && i.probabilities.size()==i.candidates.size()) {
            // Keep the drafts up to the first one below the minimum probability; a context n-gram may
            // continue them up to the draft depth. The target verifies every row either way.
            std::size_t confident=0;
            while(confident<i.candidates.size() && i.probabilities[confident]>=i.min_probability) ++confident;
            auto extension=i.ngram->begin_cycle(pending,i.candidates,confident,depth,i.ngram_acceptance);
            if(!extension.empty()) {
                i.candidates.resize(confident);i.probabilities.resize(confident);
                if(i.runner_ups.size()>confident) i.runner_ups.resize(confident);
                for(std::size_t r=0;r<extension.size();++r) {
                    i.candidates.push_back(extension[r]);
                    i.probabilities.push_back(static_cast<float>(i.ngram_acceptance[r]));
                    if(!i.runner_ups.empty()) i.runner_ups.push_back(0.0F);
                }
                i.window->set_external_rows(confident,i.ngram_acceptance);
                i.ngram->used(extension.size());
            }
        }
        host_phase_mark("2 draft chain");
        finish_draft_proposal(i.draft);
        host_phase_mark("3 finish proposal");
    } catch(...) {fail_cycle();throw;}
}
std::span<const std::uint32_t> HipMtpGreedyRuntime::drafted_candidates() const noexcept {return impl_->candidates;}
std::span<const TokenDraft> HipMtpGreedyRuntime::drafted_distributions() const noexcept {
    const auto& i=*impl_;return i.target->token_sampler()?i.mtp->sampled_drafts():std::span<const TokenDraft>{};
}
std::span<const float> HipMtpGreedyRuntime::drafted_probabilities() const noexcept {return impl_->probabilities;}
bool HipMtpGreedyRuntime::drafting() const noexcept {return impl_->busy;}
HipMtpWindowVerifier* HipMtpGreedyRuntime::window_verifier() const noexcept {return impl_->window.get();}
SerialSessionContext& HipMtpGreedyRuntime::target_session() const noexcept {return *impl_->session;}
DirectLaunchDispatcher& HipMtpGreedyRuntime::target_dispatcher() const noexcept {return *impl_->dispatcher;}
void HipMtpGreedyRuntime::fail_cycle() noexcept {
    auto& i=*impl_;
    i.mtp->cancel();i.failed=true;i.busy=false;poison_draft_proposal_state(i.draft);
}
namespace {
// STRIX_MTP_DRAFT_LOG=1: the cycle's outcome after its draft lines (source position, window rows, drafts, accepted, emitted ids).
void log_cycle(const HipMtpTargetHiddenTicket& initial,const HipMtpGreedyCycleResult& out) {
    static const bool log_cycles=std::getenv("STRIX_MTP_DRAFT_LOG")!=nullptr;
    if(!log_cycles) return;
    std::fprintf(stderr,"cycle at=%llu rows=%u proposed=%u accepted=%u emitted=",static_cast<unsigned long long>(initial.source_position()),
        out.target_window_rows,out.proposed_candidates,out.accepted_candidates);
    for(std::size_t e=0;e<out.emitted_tokens.size();++e) std::fprintf(stderr,e?",%u":"%u",out.emitted_tokens[e]);
    std::fprintf(stderr,"\n");
}
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
        log_cycle(initial,out);
        if(i.ngram)i.ngram->finish_cycle(out.emitted_tokens);
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
                i.target->token_sampler()?i.mtp->sampled_drafts():std::span<const TokenDraft>{},
                i.mtp->draft_head()?std::span<const float>(i.runner_ups):std::span<const float>{});
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
        log_cycle(initial,out);
        if(i.ngram)i.ngram->finish_cycle(out.emitted_tokens);
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
namespace {
struct CohortItem {std::size_t member;std::uint32_t rows;};
// Global row budget (STRIX_COHORT_ROWS): items (largest first) fill groups of at most `budget` rows; a group's
// items fill chunks of at most eight rows; a chunk of two or more items runs on an idle owner of its total rows
// (4 or 8) that no member of the group verifies with, else its items run alone in the group's stream of steps.
// A group of one item verifies alone.
std::vector<HipMtpGreedyCycleResult> budget_cohorts(std::span<const HipMtpGreedyCohortMember> members,
    std::vector<CohortItem>& items,std::vector<HipMtpGreedyCycleResult>& out,std::vector<bool>& open,
    HipMtpGreedyCohortStats* stats,std::uint32_t budget) {
    const auto sum_of=[](const std::vector<CohortItem>& v){std::uint32_t s=0;for(const auto& x:v)s+=x.rows;return s;};
    std::stable_sort(items.begin(),items.end(),[](const CohortItem& a,const CohortItem& b){return a.rows>b.rows;});
    std::vector<CohortItem> solo;
    std::vector<std::vector<CohortItem>> groups;
    for(const auto& item:items) {
        if(!item.rows){solo.push_back(item);continue;}
        bool placed=false;
        for(auto& g:groups) if(sum_of(g)+item.rows<=budget){g.push_back(item);placed=true;break;}
        if(!placed)groups.push_back({item});
    }
    for(auto& g:groups) {
        if(g.size()<2U){solo.insert(solo.end(),g.begin(),g.end());continue;}
        // Chunks: first fit of the (descending) items into bins of eight rows; a bin of six rows gives its
        // last item a chunk of its own, so every chunk of two or more items has 4 or 8 rows.
        std::vector<std::vector<CohortItem>> bins;
        for(const auto& item:g) {
            bool placed=false;
            for(auto& b:bins) if(sum_of(b)+item.rows<=8U){b.push_back(item);placed=true;break;}
            if(!placed)bins.push_back({item});
        }
        std::vector<std::vector<CohortItem>> chunks;
        for(auto& b:bins) {
            const auto sum=sum_of(b);
            if(sum!=4U && sum!=8U && b.size()>=2U){const auto last=b.back();b.pop_back();chunks.push_back(b);chunks.push_back({last});}
            else chunks.push_back(b);
        }
        // Primaries: owners no member verifies with in this cycle, one per chunk.
        std::vector<const HipKnownPrefixWindow*> used;
        for(const auto& x:g) used.push_back(members[x.member].runtime->window_verifier()->owner_with_rows(x.rows).get());
        std::vector<std::shared_ptr<HipKnownPrefixWindow>> keep;
        std::vector<HipKnownPrefixWindow*> primaries;
        std::vector<std::size_t> sizes;
        std::vector<CohortItem> order;
        for(const auto& c:chunks) {
            if(c.size()>=2U) {
                const auto total=sum_of(c);
                std::shared_ptr<HipKnownPrefixWindow> primary;
                for(const auto& x:g) {
                    auto o=members[x.member].runtime->window_verifier()->owner_with_rows(total);
                    if(o && std::find(used.begin(),used.end(),o.get())==used.end()){primary=std::move(o);break;}
                }
                if(primary) {
                    used.push_back(primary.get());primaries.push_back(primary.get());keep.push_back(std::move(primary));
                    sizes.push_back(c.size());order.insert(order.end(),c.begin(),c.end());
                    continue;
                }
            }
            for(const auto& x:c){primaries.push_back(nullptr);sizes.push_back(1U);order.push_back(x);}
        }
        std::vector<HipMtpCohortVerifyRequest> requests;
        std::uint32_t total=0;
        for(const auto& x:order) {
            const auto& c=members[x.member];auto& r=*c.runtime;total+=x.rows;
            requests.push_back({r.window_verifier(),&r.target_session(),c.initial,r.drafted_candidates(),
                r.drafted_probabilities(),c.budget,c.eos,x.rows,r.drafted_distributions()});
        }
        auto verified=HipMtpWindowVerifier::verify_cohort(requests,primaries,sizes);
        for(std::size_t k=0;k<order.size();++k) {
            const auto& c=members[order[k].member];
            out[order[k].member]=c.runtime->finish_cycle(*c.initial,std::move(verified[k]),c.budget,c.eos);
            open[order[k].member]=false;
        }
        if(stats){++stats->cohort_windows;stats->cohort_sessions+=order.size();stats->cohort_rows+=total;stats->cohort_chunks+=sizes.size();}
    }
    for(const auto& x:solo) {
        const auto& c=members[x.member];auto& r=*c.runtime;
        auto verified=r.window_verifier()->verify(r.target_session(),r.target_dispatcher(),*c.initial,r.drafted_candidates(),
            c.budget,c.eos,r.drafted_probabilities(),r.drafted_distributions());
        out[x.member]=r.finish_cycle(*c.initial,std::move(verified),c.budget,c.eos);
        open[x.member]=false;
        if(stats)++stats->solo_windows;
    }
    return out;
}
}
namespace {
std::vector<HipMtpGreedyCycleResult> cohort_cycle(std::span<const HipMtpGreedyCohortMember> members,
    HipMtpGreedyCohortStats* stats);
// STRIX_COHORT_AB=1 (measurement): per cycle parity, the mean wall time of a cohort cycle (drafts, windows,
// acceptance and reconciliation) and the tokens it emitted, printed at exit; odd cycles run without the
// features of STRIX_COHORT_AB_OFF.
struct CohortAbTiming {
    double ms[2]{}; std::uint64_t cycles[2]{}, tokens[2]{};
    ~CohortAbTiming() {
        for (unsigned p = 0; p < 2U; ++p) if (cycles[p])
            std::fprintf(stderr, "cohort ab %s: %llu cycles x %.3f ms, %.3f tokens/cycle, %.2f tok/s\n", p ? "odd (features off)" : "even",
                static_cast<unsigned long long>(cycles[p]), ms[p] / static_cast<double>(cycles[p]),
                static_cast<double>(tokens[p]) / static_cast<double>(cycles[p]), 1000.0 * static_cast<double>(tokens[p]) / ms[p]);
    }
};
}
std::vector<HipMtpGreedyCycleResult> hip_mtp_greedy_cohort_cycle(std::span<const HipMtpGreedyCohortMember> members,
    HipMtpGreedyCohortStats* stats) {
    ++known_window_cohort_cycle();
    if (!known_window_cohort_ab()) return cohort_cycle(members, stats);
    static CohortAbTiming timing;
    const auto parity = known_window_cohort_cycle() & 1U;
    const auto start = std::chrono::steady_clock::now();
    auto out = cohort_cycle(members, stats);
    timing.ms[parity] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    ++timing.cycles[parity];
    for (const auto& r : out) timing.tokens[parity] += r.emitted_tokens.size();
    return out;
}
namespace {
std::vector<HipMtpGreedyCycleResult> cohort_cycle(std::span<const HipMtpGreedyCohortMember> members,
    HipMtpGreedyCohortStats* stats) {
    std::vector<HipMtpGreedyCycleResult> out(members.size());
    std::vector<bool> open(members.size(),false);
    using Item=CohortItem;
    std::vector<Item> items;
    // Rows per session: at most 8 / sessions (4 for two, 2 for three or four), so that the sessions of a
    // cycle fill one 8-row cohort; drafts beyond that window are not drafted.
    // STRIX_COHORT_MAX_ROWS=2|4|8 overrides it.
    static const std::uint32_t max_rows_override=[]{const char* v=std::getenv("STRIX_COHORT_MAX_ROWS");
        const auto n=v?std::atoi(v):0;return n==2||n==4||n==8?static_cast<std::uint32_t>(n):0U;}();
    // STRIX_COHORT_ROWS=16|32: a global row budget per cycle instead of one 8-row cohort. Each session keeps
    // the rows it gets in an 8-row cohort (8 / sessions, at least 2), so up to four sessions run as before and
    // more fill one group of up to the budget instead of several cohorts; the sessions of a group run in one
    // stream of steps as chunks of at most eight rows (verify_cohort over chunks). Measured: more rows per
    // session (4 at four sessions, 8 at two) cost more than the drafts they accept.
    static const std::uint32_t budget_rows=[]{const char* v=std::getenv("STRIX_COHORT_ROWS");
        const auto n=v?std::atoi(v):0;return n==16||n==32?static_cast<std::uint32_t>(n):0U;}();
    const std::uint32_t cap=max_rows_override?max_rows_override:members.size()>=3U?2U:members.size()==2U?4U:8U;
    try {
        // Drafts: every session's chain in phases, step k of all sessions (each on its own stream) enqueued
        // before any wait (STRIX_COHORT_DRAFT_INTERLEAVE=0: one session's chain after another). Each session's
        // launches and waits are those of its own chain.
        static const bool interleave=[]{const char* v=std::getenv("STRIX_COHORT_DRAFT_INTERLEAVE");return !v || v[0]!='0';}();
        const bool interleaved=interleave && !known_window_cohort_feature_off("draft");
        for(std::size_t m=0;m<members.size();++m) {
            const auto& c=members[m];
            require(c.runtime && c.initial && c.runtime->window_verifier(),"MTP cohort cycle requires window-verified greedy runtimes");
            for(std::size_t o=0;o<m;++o)require(members[o].runtime!=c.runtime,"MTP cohort cycle requires distinct sessions");
            if(interleaved) {
                static_cast<void>(c.runtime->draft_begin(*c.initial,std::min(c.maximum_candidates,cap-1U),c.budget,c.eos,false));
                if(c.runtime->drafting()) open[m]=true;
            } else {
                static_cast<void>(c.runtime->begin_cycle_draft(*c.initial,std::min(c.maximum_candidates,cap-1U),c.budget,c.eos,false));
                if(c.runtime->drafting()) open[m]=true;
            }
        }
        if(interleaved) {
            std::vector<bool> stepping(open);
            for(bool any=true;any;) {
                any=false;
                std::vector<bool> launched(members.size(),false);
                for(std::size_t m=0;m<members.size();++m) if(stepping[m]) launched[m]=members[m].runtime->draft_launch_step();
                for(std::size_t m=0;m<members.size();++m) {
                    if(!stepping[m]) continue;
                    stepping[m]=launched[m] && members[m].runtime->draft_after_step();
                    any=any || stepping[m];
                }
            }
            for(std::size_t m=0;m<members.size();++m) if(open[m]) members[m].runtime->draft_end();
        }
        for(std::size_t m=0;m<members.size();++m) {
            const auto& c=members[m];
            if(!open[m]){out[m].final_target=*c.initial;out[m].eos=true;continue;}
            auto rows=c.runtime->window_verifier()->cohort_rows(c.runtime->target_session(),
                c.runtime->drafted_candidates(),c.runtime->drafted_probabilities(),c.budget);
            if(rows>cap && c.runtime->window_verifier()->cohort_fits(c.runtime->target_session(),cap,
                c.runtime->drafted_candidates().size(),c.budget)) rows=cap;
            items.push_back({m,rows});
        }
        if(budget_rows) return budget_cohorts(members,items,out,open,stats,budget_rows);
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
                    r.drafted_probabilities(),c.budget,c.eos,x.rows,r.drafted_distributions()});
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
                c.budget,c.eos,r.drafted_probabilities(),r.drafted_distributions());
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
}
} // namespace strix
