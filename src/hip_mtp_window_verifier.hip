#include "strix/host_phase_timing.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include "strix/mtp_target_window_policy.hpp"
#include "strix/mtp_window_controller.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/arithmetic_lineage.hpp"
#include "strix/session_state_memory.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_token_sampling.hpp"
#include "strix/hip_token_sampling_device.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
bool hip_mtp_survival_controller() noexcept {
    static const bool enabled=[]{const char* v=std::getenv("STRIX_MTP_CONTROLLER");return v && std::string_view(v)=="survival";}();
    return enabled;
}
namespace {
double env_number(const char* name,double fallback) {const char* v=std::getenv(name);return v && *v?std::atof(v):fallback;}
double clamped_logit(double p) {p=std::clamp(p,1e-6,1.0-1e-6);return std::log(p/(1.0-p));}
double sigmoid(double z) {return 1.0/(1.0+std::exp(-std::clamp(z,-30.0,30.0)));}
// P(draft k equals the target's greedy token | drafts 0..k-1 did): a logistic over the draft's own
// probability, its margin over the runner-up and its depth. The draft probability is a softmax over the draft
// vocabulary, not an acceptance rate. Prior weights are fitted offline on full-chain logs (every start
// position drafted to depth 7 on EN/ES/FR/code/agent prompts, Q4 draft, auto vocabulary); the history of
// earlier probabilities added nothing there once depth is known. Online, only the bias adapts to the
// recent workload (SGD on verified drafts, pulled back to the prior).
struct SurvivalCalibrator {
    static constexpr std::size_t width=5;
    // bias, logit p1, log(p1/p2) capped at 12, depth 1, depth >= 2
    static constexpr std::array<double,width> prior{0.1956,1.0131,-0.4434,-0.3505,-0.7112};
    // Acceptance of a draft not made yet, from its depth and the previous draft's probability:
    // bias, logit p(k-1), depth 1, depth >= 2.
    static constexpr std::array<double,4> lookahead{0.3911,0.1133,0.1019,-0.1015};
    double bias=prior[0],rate=0.0,pull=0.0;
    static std::array<double,width> features(std::size_t k,float p1,float p2) {
        const double margin=std::min(12.0,std::log(std::max(static_cast<double>(p1),1e-6)/std::max(static_cast<double>(p2),1e-6)));
        return {1.0,clamped_logit(p1),margin,k==1?1.0:0.0,k>=2?1.0:0.0};
    }
    double probability(std::size_t k,float p1,float p2) const {
        const auto x=features(k,p1,p2);double z=bias;
        for(std::size_t f=1;f<width;++f) z+=prior[f]*x[f];
        return sigmoid(z);
    }
    void update(std::size_t k,float p1,float p2,bool right) {
        if(rate>0.0) bias-=rate*(probability(k,p1,p2)-(right?1.0:0.0)+pull*(bias-prior[0]));
    }
    static double next(std::size_t k,float previous) {
        return sigmoid(lookahead[0]+lookahead[1]*clamped_logit(previous)+(k==1?lookahead[2]:0.0)+(k>=2?lookahead[3]:0.0));
    }
};
}
bool hip_mtp_full_chain_diagnostic() noexcept {
    static const bool enabled=[]{const char* v=std::getenv("STRIX_MTP_DIAG_FULL_CHAIN");return v && v[0]=='1';}();
    return enabled;
}
struct HipMtpWindowVerifier::Impl {
    struct Owner {
        std::shared_ptr<HipKnownPrefixWindow> window;
        std::array<std::shared_ptr<HipMtpTargetWindowCapture>,4> captures;
        std::uint32_t rows=0;
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
    // Min-p controller (D-022): owner cost per cycle (draft included), serial cost per emitted token,
    // per-bin acceptance; host logic in MtpMinPWindowController.
    float min_probability=0.0F;
    bool timing_independent=false;
    std::vector<float> sampled_logits;
    std::unique_ptr<MtpMinPWindowController> minp;
    std::unique_ptr<HipTokenSamplerDevice> device_sampler; // sampling-v1 on the GPU (STRIX_SAMPLE_PATH=host: host)
    Owner* last=nullptr; // owner of the last prefix window; its stream carries the PLE commit
    // Survival controller (STRIX_MTP_CONTROLLER=survival). An option is an owner and its active rows: with
    // STRIX_MTP_ACTIVE_ROWS=1 the R4 owner also verifies 3 active rows and the R8 owner 6 (the rest are
    // padding outside the expert union and acceptance). Costs are separated: per option the cycle time
    // minus the draft chain (verify, commit, reconcile and host work), per draft step, per serial row.
    struct Option {Owner* owner=nullptr;std::uint32_t active=0;std::uint64_t cycles=0,last_used=0,reprobe_after=0;double cost_ms_ema=0.0;bool probing=false;};
    bool survival=false;
    std::vector<Option> options; // ascending active rows
    SurvivalCalibrator calibrator;
    double explore=0.0; // one more draft after a first draft whose calibrated acceptance is below this
    double draft_ms_ema=0.0,serial_row_ms_ema=0.0;std::uint64_t draft_samples=0,serial_rows=0;
    double rho=0.0; // long-run tokens per ms (EMA over cycles): the price of a millisecond in survival_best
    // An option's cost is only sampled when it is chosen, so one slow sample (page faults, a stalled host) could
    // price it out for the rest of the run. An option whose estimate is implausible (above a larger option's, or
    // for the largest one, more than 1.25x the next smaller one scaled by rows) and unused for reprobe_cycles
    // cycles is chosen once more; that sample replaces a higher estimate. A probe that confirms the estimate
    // doubles the option's interval (STRIX_MTP_REPROBE=N cycles, 0 = never).
    std::uint64_t cycle_index=0,reprobe_cycles=64;
    double chain_ms=0.0;std::uint32_t chain_steps=0; // the current cycle's draft chain (note_draft)
    std::optional<std::chrono::steady_clock::time_point> last_end;
    Impl(std::shared_ptr<HipMtpTargetHidden> t,std::vector<std::shared_ptr<HipKnownPrefixWindow>> w,
         const ExecutionRuntimeResources& r,std::shared_ptr<const void> owner,
         HipMtpWindowVerificationPolicy policy,AllocationGenerationSource& generations)
        :target(std::move(t)),resources(r),resource_owner(std::move(owner)),
         maximum_history_intervals(policy.capture.maximum_history_intervals),bounded(policy.bounded_window_rollback),
         min_probability(policy.min_probability),timing_independent(policy.timing_independent_schedule) {
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
            owners.push_back({candidate,{},candidate->logical_rows(),{},{},false});
        }
        window=owners.back().window;
        prefix=window->defers_state();
        {
            // STRIX_MTP_WINDOW_COST=legacy: EMA cost estimates without re-probes (the D-022 behaviour before D-042).
            const char* cost=std::getenv("STRIX_MTP_WINDOW_COST");
            const auto model=cost && std::string_view(cost)=="legacy"?MtpWindowCostEstimate::Model::legacy:MtpWindowCostEstimate::Model::robust;
            std::vector<std::uint32_t> rows;for(const auto& o:owners) rows.push_back(o.rows);
            minp=std::make_unique<MtpMinPWindowController>(std::move(rows),min_probability,model,64,timing_independent);
        }
        // The survival controller prices options by measured time: never with a timing-independent schedule.
        survival=prefix && hip_mtp_survival_controller() && !timing_independent;
        if(survival) {
            const bool active_rows=env_number("STRIX_MTP_ACTIVE_ROWS",0.0)!=0.0;
            for(auto& o:owners) {
                if(active_rows && (o.rows==4U || o.rows==8U)) options.push_back({&o,o.rows==4U?3U:6U});
                options.push_back({&o,o.rows});
            }
            explore=env_number("STRIX_MTP_EXPLORE",0.0);
            reprobe_cycles=static_cast<std::uint64_t>(std::max(0.0,env_number("STRIX_MTP_REPROBE",64.0)));
            if(env_number("STRIX_MTP_SURVIVAL_ADAPT",1.0)!=0.0) {calibrator.rate=0.01;calibrator.pull=0.01;}
        }
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
    // Survival controller. Calibrated acceptance of each draft so far.
    std::vector<double> survival_acceptance(std::span<const float> p,std::span<const float> p2) const {
        std::vector<double> out(p.size());
        for(std::size_t k=0;k<p.size();++k) out[k]=calibrator.probability(k,p[k],k<p2.size()?p2[k]:0.0F);
        return out;
    }
    // Unmeasured options are priced just under the cheapest measured one, so each is tried.
    double option_cost(const Option& o) const {
        if(o.cycles) return o.cost_ms_ema;
        const Option* cheapest=nullptr;
        for(const auto& m:options) if(m.cycles && (!cheapest || m.cost_ms_ema<cheapest->cost_ms_ema)) cheapest=&m;
        return cheapest?0.9*cheapest->cost_ms_ema:1.0;
    }
    // Best option for drafts with acceptance P (n=P.size() drafts paid): E[N] = 1 + sum_j prod_{i<j} P_i over
    // the option's active rows against C = option cost + n draft steps. The value is E[N] - rho*C, rho the
    // long-run rate (a cycle is worth its tokens minus the tokens its time would earn at that rate), or
    // E[N]/C before the first cycle. Serial target steps earn one token per serial row cost.
    // Returns the option (nullptr = serial) and its value.
    template<class Fits> std::pair<Option*,double> survival_best(std::span<const double> P,Fits fits) {
        std::vector<double> expected(P.size()+1U,1.0);double run=1.0;
        for(std::size_t j=0;j<P.size();++j) {run*=P[j];expected[j+1U]=expected[j]+run;}
        const double drafts=draft_samples?draft_ms_ema*static_cast<double>(P.size()):0.0;
        const auto value=[&](double tokens,double ms){return rho>0.0?tokens-rho*ms:tokens/ms;};
        Option* best=nullptr;double best_value=serial_rows?value(expected.back(),serial_row_ms_ema*expected.back()+drafts):0.0;
        for(auto& o:options) {
            if(!fits(o)) continue;
            const auto v=value(expected[std::min<std::size_t>(o.active-1U,P.size())],option_cost(o)+drafts);
            if(v>best_value || (!best && !serial_rows)) {best=&o;best_value=v;}
        }
        return {best,best_value};
    }
    // Keep drafting while some longer chain (future drafts at the lookahead acceptance) has a better best
    // option than the drafts so far; after a weak first draft, one exploration step regardless.
    bool survival_keep(std::span<const float> p,std::span<const float> p2) {
        if(p.empty()) return true;
        auto P=survival_acceptance(p,p2);
        const auto any=[](const Option&){return true;};
        const auto now=survival_best(P,any).second;
        const auto next=SurvivalCalibrator::next(p.size(),p.back());
        for(std::size_t m=p.size();m<owners.back().rows-1U;++m) {
            P.push_back(next);
            if(survival_best(P,any).second>now) return true;
        }
        return p.size()==1U && P.front()<explore;
    }
    bool implausible(const Option& o) const {
        const Option* smaller=nullptr;
        for(const auto& m:options) {
            if(!m.cycles || &m==&o) continue;
            if(m.active>o.active && m.cost_ms_ema<o.cost_ms_ema) return true;
            if(m.active<o.active && (!smaller || m.active>smaller->active)) smaller=&m;
        }
        const bool largest=std::none_of(options.begin(),options.end(),[&](const Option& m){return m.cycles && m.active>o.active;});
        return largest && smaller && o.cost_ms_ema>1.25*smaller->cost_ms_ema*o.active/smaller->active;
    }
    template<class Fits> Option* survival_choose(std::span<const float> p,std::span<const float> p2,std::size_t candidates,Fits fits) {
        ++cycle_index;
        const auto use=[&](Option* o){if(o) o->last_used=cycle_index;return o;};
        // Options never measured and holding the drafts are tried first, largest first.
        for(auto o=options.rbegin();o!=options.rend();++o) if(!o->cycles && o->active<=candidates+1U && fits(*o)) return use(&*o);
        if(reprobe_cycles) for(auto o=options.rbegin();o!=options.rend();++o)
            if(o->cycles && cycle_index-o->last_used>std::max(o->reprobe_after,reprobe_cycles) && implausible(*o) && fits(*o)) {
                o->probing=true;return use(&*o);}
        return use(survival_best(survival_acceptance(p,p2),fits).first);
    }
    void survival_calibrate(std::span<const float> p,std::span<const float> p2,std::size_t accepted_drafts,std::size_t verified_rows) {
        for(std::size_t j=0;j<p.size() && j<verified_rows && j<=accepted_drafts;++j)
            calibrator.update(j,p[j],j<p2.size()?p2[j]:0.0F,j<accepted_drafts);
    }
    void survival_observe(Option* option,std::size_t emitted) {
        const auto now=std::chrono::steady_clock::now();
        if(last_end && emitted) {
            const auto ms=std::chrono::duration<double,std::milli>(now-*last_end).count();
            static const bool log_cycles=std::getenv("STRIX_CYCLE_LOG")!=nullptr;
            if(log_cycles) std::fprintf(stderr,"cycle rows=%u emitted=%zu ms=%.2f draft_ms=%.2f drafts=%u\n",
                option?option->active:1U,emitted,ms,chain_ms,chain_steps);
            constexpr double alpha=0.25;
            auto update=[&](double& ema,std::uint64_t& count,double value){
                ema=count?ema+alpha*(std::min(value,2.0*ema)-ema):value;++count;};
            if(chain_steps) update(draft_ms_ema,draft_samples,chain_ms/chain_steps);
            const auto rate=static_cast<double>(emitted)/std::max(ms,1e-3);
            rho=rho>0.0?rho+0.1*(rate-rho):rate;
            const auto rest=std::max(ms-chain_ms,0.0);
            if(option) {
                // An owner's first window pays one-time admission and first touch: not a cost sample.
                if(option->owner->warmed) {
                    if(option->probing) {
                        option->reprobe_after=rest<0.9*option->cost_ms_ema?reprobe_cycles:2U*std::max(option->reprobe_after,reprobe_cycles);
                        option->cost_ms_ema=std::min(option->cost_ms_ema,rest);++option->cycles;
                    }
                    else update(option->cost_ms_ema,option->cycles,rest);
                }
                option->owner->warmed=true;option->probing=false;
            } else update(serial_row_ms_ema,serial_rows,rest/static_cast<double>(emitted));
        }
        last_end=now;chain_ms=0.0;chain_steps=0;
    }
    // nullptr = serial target steps.
    template<class Fits> Owner* choose(std::span<const float> p,std::size_t candidates,Fits fits) {
        std::array<std::uint8_t,8> mask{};
        for(std::size_t o=0;o<owners.size();++o) mask[o]=fits(owners[o])?1U:0U;
        const auto chosen=minp->choose(p,candidates,std::span<const std::uint8_t>(mask.data(),owners.size()));
        return chosen==MtpMinPWindowController::serial?nullptr:&owners[chosen];
    }
    // A cohort cycle is not a sample of any owner's cost; the next cycle measures from here.
    void skip_sample() {last_end=std::chrono::steady_clock::now();}
    Owner* owner_of(std::uint32_t rows) {for(auto& o:owners) if(o.rows==rows) return &o;return nullptr;}
    bool fits(const SerialSessionContext& session,const Owner& o,std::size_t candidates,std::uint32_t budget) const {
        const auto capacity=std::min(resources.session_state_layout->context_tokens,target->model().max_position_embeddings);
        if(session.next_position>=capacity) return false;
        const bool pad=prefix && min_probability>0.0F;
        const bool history=session.arithmetic_lineage.intervals.size()<maximum_history_intervals;
        return budget>=o.rows && capacity-session.next_position>=o.rows && history &&
            (pad?candidates!=0U:candidates+1U>=o.rows);
    }
    // Wall time between consecutive verified cycles, attributed to this cycle's choice.
    void observe(Owner* owner,std::size_t emitted) {
        const auto now=std::chrono::steady_clock::now();
        if(last_end && emitted) {
            const auto ms=std::chrono::duration<double,std::milli>(now-*last_end).count();
            static const bool log_cycles=std::getenv("STRIX_CYCLE_LOG")!=nullptr;
            if(log_cycles) std::fprintf(stderr,"cycle rows=%u emitted=%zu ms=%.2f\n",owner?owner->rows:1U,emitted,ms);
            minp->observe(owner?static_cast<std::size_t>(owner-owners.data()):MtpMinPWindowController::serial,emitted,ms);
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
    std::vector<HipMtpWindowOwnerStats> out;
    for(const auto& o:impl_->minp->owner_stats()) out.push_back({o.rows,o.cycles,o.cycle_ms,o.probes});
    return out;
}
bool HipMtpWindowVerifier::robust_cost_model() const noexcept {return impl_->minp->model()==MtpWindowCostEstimate::Model::robust;}
std::uint32_t HipMtpWindowVerifier::logical_rows() const noexcept {return impl_->window->logical_rows();}
void HipMtpWindowVerifier::prepare_admission(const SerialSessionContext& session) {
    // STRIX_WINDOW_SETUP_PREFLIGHT=0 leaves each owner's admission to its first window.
    static const bool enabled=[]{const char* v=std::getenv("STRIX_WINDOW_SETUP_PREFLIGHT");return !v || v[0]!='0';}();
    auto& i=*impl_;
    if(!enabled || i.busy || failed()) return;
    const auto capacity=std::min(i.resources.session_state_layout->context_tokens,i.target->model().max_position_embeddings);
    for(auto& o:i.owners) {
        if(session.next_position+o.rows>capacity) continue;
        const std::vector<std::uint32_t> tokens(o.rows,0U);
        o.window->preflight_admission(session,tokens);
    }
}
std::uint32_t HipMtpWindowVerifier::cohort_rows(const SerialSessionContext& session,std::span<const std::uint32_t> candidates,
    std::span<const float> probabilities,std::uint32_t budget) const {
    const auto& i=*impl_;
    if(!i.prefix || i.busy || failed() || candidates.empty()) return 0U;
    std::uint32_t want=2U;
    if(i.min_probability>0.0F && probabilities.size()==candidates.size()) {
        std::uint32_t lead=1U;
        while(lead-1U<probabilities.size() && probabilities[lead-1U]>=i.min_probability) ++lead;
        want=std::max(want,lead);
    } else want=std::max<std::uint32_t>(want,static_cast<std::uint32_t>(candidates.size())+1U);
    for(const auto& o:i.owners) if(o.rows>=want && i.fits(session,o,candidates.size(),budget)) return o.rows;
    for(auto o=i.owners.rbegin();o!=i.owners.rend();++o) if(i.fits(session,*o,candidates.size(),budget)) return o->rows;
    return 0U;
}
bool HipMtpWindowVerifier::cohort_fits(const SerialSessionContext& session,std::uint32_t rows,std::size_t candidates,
    std::uint32_t budget) const {
    auto& i=*impl_;
    const auto* owner=i.owner_of(rows);
    return i.prefix && !i.busy && !failed() && owner && i.fits(session,*owner,candidates,budget);
}
std::shared_ptr<HipKnownPrefixWindow> HipMtpWindowVerifier::owner_with_rows(std::uint32_t rows) const {
    auto* owner=impl_->owner_of(rows);
    return owner?owner->window:nullptr;
}
std::vector<HipMtpVerifiedTargetRows> HipMtpWindowVerifier::verify_cohort(std::span<const HipMtpCohortVerifyRequest> requests,
    HipKnownPrefixWindow& primary) {
    require(requests.size()>=2U,"MTP cohort verification requires two or more sessions");
    struct Prepared {
        Impl* impl=nullptr;Impl::Owner* owner=nullptr;
        std::vector<std::uint32_t> inputs;
        SerialSessionContext previous;
        std::shared_ptr<HipMtpTargetWindowCapture> capture;
    };
    std::vector<Prepared> prepared(requests.size());
    std::vector<HipKnownPrefixCohortMember> members(requests.size());
    for(std::size_t a=0;a<requests.size();++a) {
        const auto& q=requests[a];
        require(q.verifier && q.session && q.initial,"MTP cohort verification request is incomplete");
        auto& i=*q.verifier->impl_;
        for(std::size_t b=0;b<a;++b)require(requests[b].verifier!=q.verifier && requests[b].session!=q.session,
            "MTP cohort verification requires distinct sessions");
        const auto& session=*q.session;const auto& initial=*q.initial;
        require(i.prefix && !i.busy && !q.verifier->failed() && q.budget && initial.valid() &&
            initial.source_position()+1==session.next_position && session_equal(initial.committed_session(),session),
            "MTP cohort verifier requires exact idle committed prefix-owner frontiers");
        const auto vocabulary=i.target->model().vocab_size;
        for(auto token:q.candidates)require(token<vocabulary,"MTP target candidate outside vocabulary");
        for(auto token:q.eos)require(token<vocabulary,"MTP target EOS outside vocabulary");
        require(q.candidates.size()<i.window->logical_rows(),"MTP verifier admits only one closed window per cycle");
        const auto pending=initial.predicted_token();
        require(std::find(q.eos.begin(),q.eos.end(),pending)==q.eos.end(),"MTP cohort session has an EOS pending token");
        auto* owner=i.owner_of(q.rows);
        require(owner && i.fits(session,*owner,q.candidates.size(),q.budget),"MTP cohort session owner does not fit");
        auto& p=prepared[a];p.impl=&i;p.owner=owner;
        for(const auto& pool:owner->captures)if(pool->ready_for_reuse()){p.capture=pool;break;}
        require(!!p.capture,"MTP window capture pools are retained by previous batch readers");
        i.target->validate_window_capture(*p.capture);
        p.inputs.reserve(q.rows);p.inputs.push_back(pending);
        for(std::uint32_t r=1;r<q.rows;++r) p.inputs.push_back(q.candidates[std::min<std::size_t>(r-1U,q.candidates.size()-1U)]);
        members[a]={owner->window.get(),q.session,p.inputs,p.capture};
    }
    for(auto& p:prepared) {
        // The previous owner's PLE commit runs on its own stream.
        if(p.impl->last && p.impl->last!=p.owner &&
           hipStreamSynchronize(static_cast<hipStream_t>(p.impl->last->captures.front()->source_stream()))!=hipSuccess)
            throw std::runtime_error("MTP window owner switch could not complete the previous commit");
    }
    for(std::size_t a=0;a<requests.size();++a){prepared[a].previous=*requests[a].session;prepared[a].impl->busy=true;}
    bool attempted=false;
    try {
        host_phase_mark("4 cohort verify prologue");
        auto natives=HipKnownPrefixWindow::execute_mtp_cohort(primary,members);
        host_phase_mark("7 cohort window launch+wait");
        attempted=true;
        std::vector<HipMtpVerifiedTargetRows> out(requests.size());
        for(std::size_t a=0;a<requests.size();++a) {
            const auto& q=requests[a];auto& p=prepared[a];auto& i=*p.impl;auto& session=*q.session;
            const auto rows=q.rows;
            auto is_eos=[&](std::uint32_t token){return std::find(q.eos.begin(),q.eos.end(),token)!=q.eos.end();};
            auto& result=out[a];result.chosen_rows=rows;result.attempted_window=true;
            result.r2_dispatches=natives[a].execution.r2_dispatches;
            result.projection_launches_saved=natives[a].execution.projection_launches_saved;
            auto provisional=std::make_shared<const HipMtpTargetWindowBatch>(std::move(natives[a].batch));
            std::vector<std::uint32_t> selected;selected.reserve(rows);
            for(std::uint32_t row=0;row<rows;++row)selected.push_back(provisional->predicted_token(row));
            // Longest accepted prefix: row a+1's input equals row a's prediction.
            std::uint32_t k=0;
            while(k+1U<rows && k<q.candidates.size() && k+1U<q.budget && p.inputs[k+1U]==selected[k] && !is_eos(selected[k])) ++k;
            const auto n_keep=k+1U;
            p.owner->window->commit_prefix(n_keep);i.last=p.owner;
            if(n_keep<rows) {
                const auto dropped=rows-n_keep;
                session.next_position-=dropped;session.committed_tokens-=dropped;
                const auto& last=session.arithmetic_lineage.intervals.back();
                session.arithmetic_lineage=prepare_known_window_chunk_lineage(p.previous.arithmetic_lineage,last.begin,
                    last.begin+n_keep,last.numerical_plan_identity,last.reduction_plan_identity,last.state_update_identity);
                write_position_word(i.resources,session.next_position-1U);
            }
            result.committed_rows=i.target->accept_verified_prefix(provisional,*q.initial,session,n_keep);
            result.emitted_tokens.assign(selected.begin(),selected.begin()+n_keep);
            result.accepted_candidates=k;result.accepted_window=n_keep==rows;
            if(q.probabilities.size()==q.candidates.size())i.minp->calibrate(q.probabilities,k,rows-1U);
            i.skip_sample();
            provisional.reset();
            natives[a]={};
            p.capture.reset();
            i.busy=false;
        }
        host_phase_mark("7e cohort accept");
        return out;
    } catch(...) {
        for(std::size_t a=0;a<requests.size();++a) {
            auto& i=*prepared[a].impl;i.failed=true;i.busy=false;
            auto& session=*requests[a].session;
            if(attempted || session.health==SerialSessionHealth::poisoned){
                session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
                i.target->guard()->recover_or_fail_stop();
            }
        }
        throw;
    }
}
void HipMtpWindowVerifier::begin_session() {
    auto& i=*impl_;
    require(!i.busy && !failed(),"MTP window verifier session restarts only while idle and healthy");
    for(auto& o:i.owners) o.window->discard_deferred();
    i.last=nullptr;i.last_end.reset();
}
bool HipMtpWindowVerifier::failed() const noexcept {return impl_->failed || (impl_->checkpoint && impl_->checkpoint->failed());}
std::uint64_t HipMtpWindowVerifier::retained_payload_bytes() const noexcept {return impl_->retained;}
bool HipMtpWindowVerifier::worth_drafting(std::span<const float> probabilities,std::span<const float> runner_ups) const {
    return impl_->survival?impl_->survival_keep(probabilities,runner_ups):impl_->minp->worth_drafting(probabilities);
}
bool HipMtpWindowVerifier::survival_controller() const noexcept {return impl_->survival;}
void HipMtpWindowVerifier::note_draft(std::uint32_t steps,double ms) noexcept {impl_->chain_steps=steps;impl_->chain_ms=ms;}
std::vector<HipMtpWindowOptionStats> HipMtpWindowVerifier::option_stats() const {
    std::vector<HipMtpWindowOptionStats> out;
    if(!impl_->survival) return out;
    out.push_back({1U,1U,impl_->serial_rows,impl_->serial_row_ms_ema});
    for(const auto& o:impl_->options) out.push_back({o.owner->rows,o.active,o.cycles,o.cost_ms_ema});
    out.push_back({0U,0U,impl_->draft_samples,impl_->draft_ms_ema});
    return out;
}
HipMtpVerifiedTargetRows HipMtpWindowVerifier::verify(SerialSessionContext& session,DirectLaunchDispatcher& dispatcher,
    const HipMtpTargetHiddenTicket& initial,std::span<const std::uint32_t> candidates,
    std::uint32_t budget,std::span<const std::uint32_t> eos,std::span<const float> probabilities,
    std::span<const TokenDraft> drafts,std::span<const float> runner_ups) {
    auto& i=*impl_;
    if(hip_mtp_full_chain_diagnostic()) budget=std::min(budget,1U); // no owner fits: one serial row
    const auto sampler=i.target->token_sampler();
    require(sampler?drafts.size()==candidates.size():drafts.empty(),"MTP sampled verification needs one draft distribution per candidate");
    for(std::size_t k=0;k<drafts.size();++k)
        require(drafts[k].token==candidates[k] && drafts[k].position==session.next_position+1U+k,"MTP sampled draft does not match its candidate");
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
    const bool pad=i.prefix && (i.min_probability>0.0F || i.survival);
    const auto room=capacity-session.next_position;
    const bool history=session.arithmetic_lineage.intervals.size()<i.maximum_history_intervals;
    const bool survival=i.survival && probabilities.size()==candidates.size();
    Impl::Option* option=nullptr;Impl::Owner* owner=nullptr;
    if(survival) {
        option=i.survival_choose(probabilities,runner_ups,candidates.size(),[&](const Impl::Option& o){
            return budget>=o.active && room>=o.owner->rows && history && !candidates.empty();});
        owner=option?option->owner:nullptr;
    } else owner=i.choose(probabilities,candidates.size(),[&](const Impl::Owner& o){
        return budget>=o.rows && room>=o.rows && history && (pad?!candidates.empty():candidates.size()+1>=o.rows);});
    const bool use_window=owner!=nullptr;
    const auto rows=use_window?owner->rows:1U;
    // Rows whose drafts can be accepted; the owner's later rows are padding (outside the expert union
    // when fewer than its physical rows).
    const auto active=option?option->active:rows;
    out.chosen_rows=active;
    const auto calibrate=[&](std::size_t accepted,std::size_t verified) {
        if(probabilities.size()!=candidates.size()) return;
        if(survival) i.survival_calibrate(probabilities,runner_ups,accepted,verified);
        else i.minp->calibrate(probabilities,accepted,verified);
    };
    const auto observe=[&](bool window,std::size_t emitted) {
        if(survival) i.survival_observe(window?option:nullptr,emitted);
        else i.observe(window?owner:nullptr,emitted);
    };
    std::shared_ptr<HipMtpTargetWindowCapture> capture;
    if(use_window) {
        for(const auto& pool:owner->captures)if(pool->ready_for_reuse()){capture=pool;break;}
        require(!!capture,"MTP window capture pools are retained by previous batch readers");
        i.target->validate_window_capture(*capture);
        // Sampled rows: decided in order from row 0 (the token at next_position + 1) until the first
        // row without a draft, a rejection, EOS or the budget; later rows are never accepted.
        if(sampler) capture->set_row_selection([&,first=session.next_position+1U,stream=capture->source_stream()](
            std::span<const ExecutionBufferAddress> logits,std::span<std::uint32_t> predictions) {
            const bool timed=sample_timing_enabled();
            double readback=0.0;
            const auto start=std::chrono::steady_clock::now();
            std::vector<TokenDecision> decisions;
            const auto n=std::min(active,budget);
            if(device_token_sampling_enabled(sampler->policy())) {
                // Every row on the device at once; the stopping rule of TokenSampler::decide_rows on the results.
                if(!i.device_sampler) i.device_sampler=std::make_unique<HipTokenSamplerDevice>(static_cast<std::uint32_t>(vocabulary),8U);
                std::vector<DeviceSampleRow> device_rows(n);
                for(std::uint32_t r=0;r<n;++r) device_rows[r]={logits[r],first+r,r<drafts.size()?&drafts[r]:nullptr};
                const auto all=i.device_sampler->decide(*sampler,device_rows,stream);
                for(std::uint32_t r=0;r<n;++r) {
                    decisions.push_back(all[r]);sampler->count(all[r]);
                    if(!device_rows[r].draft || !all[r].accepted || std::find(eos.begin(),eos.end(),all[r].token)!=eos.end()) break;
                }
            } else decisions=sampler->decide_rows(n,first,drafts,eos,[&](std::uint32_t r) {
                const auto t0=std::chrono::steady_clock::now();
                read_hip_logits_row(logits[r],static_cast<std::uint32_t>(vocabulary),stream,i.sampled_logits);
                if(timed) readback+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                return std::span<const float>(i.sampled_logits);});
            if(timed) record_sample_timing(SampleTimingKind::window,rows,readback,
                std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()-readback,
                static_cast<std::uint32_t>(decisions.size()));
            for(std::size_t r=0;r<decisions.size();++r) predictions[r]=decisions[r].token;
        });
        else capture->set_row_selection({});
    }
    // The selection borrows this call's locals: never leave it attached past the call.
    struct ClearSelection {HipMtpTargetWindowCapture* c;~ClearSelection(){if(c)c->set_row_selection({});}} clear_selection{sampler?capture.get():nullptr};
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
            auto native=owner->window->execute_mtp_captured(session,inputs,capture,{},active);
            host_phase_mark("7 window launch+wait");
            attempted_target=true;out.attempted_window=true;
            out.r2_dispatches=native.execution.r2_dispatches;
            out.projection_launches_saved=native.execution.projection_launches_saved;
            auto provisional=std::make_shared<const HipMtpTargetWindowBatch>(std::move(native.batch));
            std::vector<std::uint32_t> selected;selected.reserve(rows);
            for(std::uint32_t row=0;row<rows;++row)selected.push_back(provisional->predicted_token(row));
            // Longest accepted prefix: row a+1's input equals row a's prediction.
            std::uint32_t a=0;
            while(a+1U<active && a<candidates.size() && a+1U<budget && inputs[a+1U]==selected[a] && !is_eos(selected[a])) ++a;
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
            out.accepted_candidates=a;out.accepted_window=n_keep==active;
            calibrate(a,active-1U);
            observe(true,n_keep);
            host_phase_mark("7e calibrate/observe");
            provisional.reset();
            host_phase_mark("7f release provisional");
            {auto done=std::move(native);}
            host_phase_mark("7g release native");
            capture.reset();
            host_phase_mark("7h release capture");
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
            auto ticket=i.target->execute_serial(session,i.resources,dispatcher,pending,
                index<drafts.size()?&drafts[index]:nullptr);
            attempted_target=true;
            pending=ticket.predicted_token();out.committed_rows.push_back(std::move(ticket));out.emitted_tokens.push_back(pending);
            const bool accepted=index<candidates.size() && candidates[index]==pending;
            if(accepted)++out.accepted_candidates;
            if(is_eos(pending) || !accepted)break;
        }
        calibrate(out.accepted_candidates,out.emitted_tokens.size());
        observe(use_window,out.emitted_tokens.size());
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
