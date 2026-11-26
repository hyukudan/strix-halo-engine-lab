#include "strix/token_sampling.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix {
namespace {

constexpr double tail_cutoff = 40.0;

void require(bool ok,const char* message) {if(!ok) throw std::invalid_argument(message);}

void validate_forced_rejection(const TokenSamplerOptions& options) {
    require(options.force_reject_row==TokenSamplerOptions::forced_rejection_off ||
        (options.acceptance==SampledAcceptance::coupled &&
         (options.force_reject_row==TokenSamplerOptions::forced_rejection_rotate || options.force_reject_row<8U)),
        "forced rejection requires coupled acceptance and a row 0..7 or rotate");
}

struct Entry {
    double z = 0.0; // (logit - max logit) / temperature, <= 0
    std::uint32_t token = 0;
};

bool before(const Entry& a,const Entry& b) noexcept {return a.z>b.z || (a.z==b.z && a.token<b.token);}

// The first double delta whose rounded division delta/temperature reaches the cutoff. Comparing a logit's
// (logit - max) against it is the same decision as the general path's (logit - max)/temperature >= cutoff for every
// double, without a division per entry. (Comparing against the rounded product alone could move entries across the
// boundary.)
double cutoff_delta(double cutoff,double temperature) {
    const auto infinity=std::numeric_limits<double>::infinity();
    double delta=cutoff*temperature;
    while(delta/temperature<cutoff) delta=std::nextafter(delta,infinity);
    for(double previous=std::nextafter(delta,-infinity);previous/temperature>=cutoff;
        previous=std::nextafter(delta,-infinity)) delta=previous;
    return delta;
}

// Whole-row weights (ids = row indices, no top-k / top-p) in the arithmetic of the general path: exp of the same z,
// the total summed in token order. One pass: with min-p the kept set is small, so the buffers start small and grow;
// without it every entry above the tail cut is kept.
void row_weights(std::span<const float> logits,const TokenSamplingPolicy& policy,
    std::vector<std::uint32_t>& tokens,std::vector<double>& weights,double& total) {
    require(token_sampling_enabled(policy) && !logits.empty() && logits.size()<=std::numeric_limits<std::uint32_t>::max(),
        "token distribution requires a positive temperature and a non-empty row");
    float max=-std::numeric_limits<float>::infinity();
    for(const auto value:logits) {
        require(std::isfinite(value),"token distribution requires finite logits");
        max=std::max(max,value);
    }
    const double temperature=policy.temperature;
    double cutoff=-tail_cutoff;
    if(policy.min_p>0.0F) cutoff=std::max(cutoff,std::log(static_cast<double>(policy.min_p)));
    const double min_delta=cutoff_delta(cutoff,temperature);
    const std::size_t initial=policy.min_p>0.0F ? std::min<std::size_t>(logits.size(),64U) : logits.size();
    tokens.reserve(initial);weights.reserve(initial);
    total=0.0;
    for(std::size_t i=0;i<logits.size();++i) {
        const double delta=static_cast<double>(logits[i])-static_cast<double>(max);
        if(delta<min_delta) continue;
        const double weight=std::exp(delta/temperature);  // divided by the total by the caller
        tokens.push_back(static_cast<std::uint32_t>(i));
        weights.push_back(weight);
        total+=weight;
    }
}

}  // namespace

bool token_sampling_enabled(const TokenSamplingPolicy& policy) noexcept {return policy.temperature>0.0F;}

void validate_token_sampling_policy(const TokenSamplingPolicy& policy) {
    require(std::isfinite(policy.temperature) && policy.temperature>=0.0F && policy.temperature<=100.0F,
        "sampling temperature must be finite in [0, 100]");
    require(std::isfinite(policy.top_p) && policy.top_p>0.0F && policy.top_p<=1.0F,"sampling top-p must lie in (0, 1]");
    require(std::isfinite(policy.min_p) && policy.min_p>=0.0F && policy.min_p<1.0F,"sampling min-p must lie in [0, 1)");
}

std::uint64_t token_sampling_policy_identity(const TokenSamplingPolicy& policy) {
    validate_token_sampling_policy(policy);
    std::uint64_t hash=14695981039346656037ULL;
    auto mix=[&](std::uint64_t value){for(unsigned s=0;s<64;s+=8){hash^=(value>>s)&0xffU;hash*=1099511628211ULL;}};
    const bool canonical_zero = policy.identity == SamplingPolicyIdentity::canonical_zero_v2;
    const auto float_bits = [canonical_zero](float value) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        return canonical_zero && bits == 0x80000000U ? 0U : bits;
    };
    mix(canonical_zero ? 2U : 1U);
    mix(float_bits(policy.temperature));
    mix(policy.top_k);
    mix(std::bit_cast<std::uint32_t>(policy.top_p));
    mix(float_bits(policy.min_p));
    return hash?hash:1U;
}

SpeculationRngIdentity token_sampling_rng(const TokenSamplingPolicy& policy,std::uint64_t request_identity) {
    require(request_identity!=0,"sampling request identity must be non-zero");
    return {policy.seed,request_identity,token_sampling_policy_identity(policy)};
}

bool token_sampling_request_identity_counts() {
    static const bool counter=[] {
        const char* v=std::getenv("STRIX_SAMPLING_REQUEST_IDENTITY");
        if(!v || !*v || std::strcmp(v,"0")==0 || std::strcmp(v,"single")==0) return false;
        // The counter mode gave each request an identity from its admission order, so the same request with the
        // same seed drew different tokens depending on what arrived before it. It is retired: a client that wants
        // distinct streams sends distinct seeds, which reproduce on every entry point.
        require(std::strcmp(v,"counter")!=0,"STRIX_SAMPLING_REQUEST_IDENTITY=counter is retired: the tokens of a "
            "request must not depend on its arrival order; send a distinct seed per request instead");
        require(false,"STRIX_SAMPLING_REQUEST_IDENTITY must be single (counter is retired)");
        return false;
    }();
    return counter;
}

std::uint64_t next_token_sampling_request_identity() {
    static std::atomic<std::uint64_t> next{1};
    return token_sampling_request_identity_counts()?next.fetch_add(1,std::memory_order_relaxed):1U;
}

TokenSamplerOptions validate_sampling_startup_environment(SampledAcceptance unset_acceptance) {
    (void)token_sampling_request_identity_counts();
    return token_sampler_options_from_environment(unset_acceptance);
}

TokenSamplerOptions validate_sampling_startup_environment_only(SampledAcceptance supported, const char* route) {
    auto options=validate_sampling_startup_environment(supported);
    if(options.acceptance!=supported)
        throw std::invalid_argument(std::string(route)+" implements "+(supported==SampledAcceptance::coupled?"coupled":"rejection")+
            " acceptance only: STRIX_MTP_SAMPLED_ACCEPT="+(options.acceptance==SampledAcceptance::coupled?"coupled":"rejection")+
            " is refused (unset it or name the supported rule)");
    return options;
}

TokenSamplerOptions token_sampler_options_from_environment(SampledAcceptance unset_acceptance) {
    TokenSamplerOptions out;
    out.acceptance=unset_acceptance;
    if(const char* v=std::getenv("STRIX_MTP_SAMPLED_ACCEPT"); v && *v) {
        if(std::strcmp(v,"coupled")==0) out.acceptance=SampledAcceptance::coupled;
        else if(std::strcmp(v,"rejection")==0) out.acceptance=SampledAcceptance::rejection;
        else require(false,"STRIX_MTP_SAMPLED_ACCEPT must be rejection or coupled");
    }
    if(const char* v=std::getenv("STRIX_MTP_SAMPLED_ARGMAX_BELOW"); v && *v) {
        char* end=nullptr;
        const double t=std::strtod(v,&end);
        require(end && *end=='\0' && std::isfinite(t) && t>=0.0 && t<=100.0,
            "STRIX_MTP_SAMPLED_ARGMAX_BELOW must be a temperature in [0, 100]");
        out.argmax_draft_below=static_cast<float>(t);
    }
    if(const char* v=std::getenv("STRIX_MTP_SAMPLED_FORCE_REJECT"); v && *v) {
        if(std::strcmp(v,"rotate")==0) out.force_reject_row=TokenSamplerOptions::forced_rejection_rotate;
        else {
            require(v[1]=='\0' && v[0]>='0' && v[0]<='7',"STRIX_MTP_SAMPLED_FORCE_REJECT must be a row 0..7 or rotate");
            out.force_reject_row=static_cast<std::uint32_t>(v[0]-'0');
        }
    }
    validate_forced_rejection(out);
    return out;
}

double TokenDistribution::probability(std::uint32_t token) const {
    const auto it=std::lower_bound(by_token_.begin(),by_token_.end(),token,
        [&](std::uint32_t index,std::uint32_t value){return tokens_[index]<value;});
    return it!=by_token_.end() && tokens_[*it]==token?probabilities_[*it]:0.0;
}

std::uint32_t TokenDistribution::sample(double uniform) const {
    require(!tokens_.empty() && std::isfinite(uniform) && uniform>=0.0 && uniform<1.0,
        "token sampling needs a non-empty distribution and a uniform in [0, 1)");
    double cumulative=0.0;
    for(std::size_t i=0;i<tokens_.size();++i) {
        cumulative+=probabilities_[i];
        if(uniform<cumulative) return tokens_[i];
    }
    return tokens_.back(); // rounding: every kept entry has positive probability
}


// The distribution of a whole logits row (ids = row indices) without top-k / top-p: the same z, tail cut, exp and
// probabilities per entry as the general path below, the total summed in token order. The kept entries are ascending
// by construction, so neither the entry sort nor the by-token sort is needed. Internal (no public declaration).
TokenDistribution make_row_distribution(std::span<const float> logits,const TokenSamplingPolicy& policy) {
    TokenDistribution out;
    double total=0.0;
    row_weights(logits,policy,out.tokens_,out.probabilities_,total);
    for(auto& probability:out.probabilities_) probability/=total;
    out.by_token_.resize(out.tokens_.size());
    for(std::size_t i=0;i<out.by_token_.size();++i) out.by_token_[i]=static_cast<std::uint32_t>(i);
    return out;
}

std::uint32_t sample_token_row(std::span<const float> logits,std::span<const std::uint32_t> ids,
    const TokenSamplingPolicy& policy,double uniform) {
    if(ids.empty() && !policy.top_k && policy.top_p>=1.0F && logits.size()>=65536U) {
        validate_token_sampling_policy(policy);
        require(std::isfinite(uniform) && uniform>=0.0 && uniform<1.0,
            "token sampling needs a non-empty distribution and a uniform in [0, 1)");
        std::vector<std::uint32_t> tokens;std::vector<double> weights;double total=0.0;
        row_weights(logits,policy,tokens,weights,total);
        // Each weight is divided before it is added (the stored probabilities of the distribution); scaling the
        // uniform instead would move rounded CDF boundaries.
        double cumulative=0.0;
        for(std::size_t i=0;i<tokens.size();++i) {
            cumulative+=weights[i]/total;
            if(uniform<cumulative) return tokens[i];
        }
        return tokens.back();
    }
    return make_token_distribution(logits,ids,policy).sample(uniform);
}

TokenDistribution make_token_distribution(std::span<const float> logits,std::span<const std::uint32_t> ids,
    const TokenSamplingPolicy& policy) {
    require(token_sampling_enabled(policy),"token distribution requires a positive temperature");
    validate_token_sampling_policy(policy);
    require(!logits.empty() && (ids.empty() || ids.size()==logits.size()) &&
        logits.size()<=std::numeric_limits<std::uint32_t>::max(),"token distribution geometry is invalid");
    if(ids.empty() && !policy.top_k && policy.top_p>=1.0F && logits.size()>=65536U)
        return make_row_distribution(logits,policy);
    float max=-std::numeric_limits<float>::infinity();
    for(const auto value:logits) {
        require(std::isfinite(value),"token distribution requires finite logits");
        max=std::max(max,value);
    }
    const double temperature=policy.temperature;
    double cutoff=-tail_cutoff;
    if(policy.min_p>0.0F) cutoff=std::max(cutoff,std::log(static_cast<double>(policy.min_p)));
    std::vector<Entry> kept;
    for(std::size_t i=0;i<logits.size();++i) {
        const double z=(static_cast<double>(logits[i])-static_cast<double>(max))/temperature;
        if(z>=cutoff) kept.push_back({z,ids.empty()?static_cast<std::uint32_t>(i):ids[i]});
    }
    if(policy.top_k && kept.size()>policy.top_k) {
        std::nth_element(kept.begin(),kept.begin()+policy.top_k-1,kept.end(),before);
        kept.resize(policy.top_k);
    }
    // Sums run in token id order (or the top-p order below), never in a library-dependent order.
    // (Already ascending when the ids are the row indices and no top-k selection reordered them: skip the sort.)
    const auto by_id=[](const Entry& a,const Entry& b){return a.token<b.token;};
    if(!std::is_sorted(kept.begin(),kept.end(),by_id)) std::sort(kept.begin(),kept.end(),by_id);
    for(std::size_t i=1;i<kept.size();++i) require(kept[i-1].token!=kept[i].token,"token distribution ids must be distinct");
    TokenDistribution out;
    std::vector<double> weights(kept.size());
    double total=0.0;
    for(std::size_t i=0;i<kept.size();++i) {weights[i]=std::exp(kept[i].z);total+=weights[i];}
    if(policy.top_p<1.0F) {
        const double target=static_cast<double>(policy.top_p)*total;
        std::vector<std::uint32_t> order(kept.size());
        for(std::size_t i=0;i<order.size();++i) order[i]=static_cast<std::uint32_t>(i);
        auto rank=[&](std::uint32_t a,std::uint32_t b){return before(kept[a],kept[b]);};
        // Sorted prefixes grow until they reach the requested mass; the prefix is unique under (z desc, id asc).
        std::size_t sorted=std::min<std::size_t>(order.size(),1024U),count=0;
        double mass=0.0;
        for(;;) {
            std::partial_sort(order.begin(),order.begin()+static_cast<std::ptrdiff_t>(sorted),order.end(),rank);
            mass=0.0;count=0;
            while(count<sorted) {mass+=weights[order[count]];++count;if(mass>=target) break;}
            if(mass>=target || sorted==order.size()) break;
            sorted=std::min(order.size(),sorted*8U);
        }
        out.tokens_.reserve(count);out.probabilities_.reserve(count);
        for(std::size_t i=0;i<count;++i) {
            out.tokens_.push_back(kept[order[i]].token);out.probabilities_.push_back(weights[order[i]]/mass);
        }
    } else {
        out.tokens_.reserve(kept.size());out.probabilities_.reserve(kept.size());
        for(std::size_t i=0;i<kept.size();++i) {out.tokens_.push_back(kept[i].token);out.probabilities_.push_back(weights[i]/total);}
    }
    out.by_token_.resize(out.tokens_.size());
    for(std::size_t i=0;i<out.by_token_.size();++i) out.by_token_[i]=static_cast<std::uint32_t>(i);
    if(!std::is_sorted(out.tokens_.begin(),out.tokens_.end()))  // identity order when the tokens are ascending
        std::sort(out.by_token_.begin(),out.by_token_.end(),[&](std::uint32_t a,std::uint32_t b){return out.tokens_[a]<out.tokens_[b];});
    return out;
}

TokenDistribution make_token_distribution_from_probabilities(std::span<const std::uint32_t> tokens,
    std::span<const double> probabilities) {
    require(tokens.size()==probabilities.size(),"token distribution geometry is invalid");
    TokenDistribution out;
    for(std::size_t i=0;i<tokens.size();++i) if(probabilities[i]>0.0) {
        out.tokens_.push_back(tokens[i]);out.probabilities_.push_back(probabilities[i]);
    }
    out.by_token_.resize(out.tokens_.size());
    for(std::size_t i=0;i<out.by_token_.size();++i) out.by_token_[i]=static_cast<std::uint32_t>(i);
    std::sort(out.by_token_.begin(),out.by_token_.end(),[&](std::uint32_t a,std::uint32_t b){return out.tokens_[a]<out.tokens_[b];});
    for(std::size_t i=1;i<out.by_token_.size();++i)
        require(out.tokens_[out.by_token_[i-1]]!=out.tokens_[out.by_token_[i]],"token distribution ids must be distinct");
    return out;
}

TokenDecision decide_token(const TokenDistribution& p,const TokenDraft* draft,
    double u_plain,double u_accept,double u_residual) {
    require(!p.empty(),"token decision requires a target distribution");
    TokenDecision out;
    if(!draft) {out.token=p.sample(u_plain);return out;}
    out.drafted=true;
    const auto& q=draft->distribution;
    const double qd=q.probability(draft->token),pd=p.probability(draft->token);
    require(qd>0.0,"draft token must have positive draft probability");
    if(u_accept*qd<pd) {out.token=draft->token;out.accepted=true;return out;}
    out.residual=true;
    const auto tokens=p.tokens();const auto probabilities=p.probabilities();
    double mass=0.0;
    for(std::size_t i=0;i<tokens.size();++i) mass+=std::max(0.0,probabilities[i]-q.probability(tokens[i]));
    if(!(mass>0.0)) {out.token=p.sample(u_residual);return out;} // p == q up to rounding
    const double target=u_residual*mass;
    double cumulative=0.0;std::uint32_t last=draft->token;
    for(std::size_t i=0;i<tokens.size();++i) {
        const double r=std::max(0.0,probabilities[i]-q.probability(tokens[i]));
        if(!(r>0.0)) continue;
        cumulative+=r;last=tokens[i];
        if(target<cumulative) {out.token=tokens[i];return out;}
    }
    out.token=last;
    return out;
}

TokenDecision decide_token_coupled(const TokenDistribution& p,const TokenDraft* draft,double u_plain) {
    require(!p.empty(),"token decision requires a target distribution");
    TokenDecision out;
    out.token=p.sample(u_plain);
    if(draft) {out.drafted=true;out.accepted=out.token==draft->token;}
    return out;
}

TokenSampler::TokenSampler(TokenSamplingPolicy policy):TokenSampler(policy,token_sampler_options_from_environment()) {}

TokenSampler::TokenSampler(TokenSamplingPolicy policy,TokenSamplerOptions options)
    :policy_(policy),draft_policy_(policy),options_(options),rng_(token_sampling_rng(policy,options.request_identity)) {
    require(token_sampling_enabled(policy_),"token sampler requires a positive temperature");
    require(std::isfinite(options_.argmax_draft_below) && options_.argmax_draft_below>=0.0F,
        "argmax draft temperature must be finite and non-negative");
    // Forcing a rejection keeps the token only under coupled acceptance (rejection sampling would change the
    // distribution), so the test switch is refused otherwise.
    validate_forced_rejection(options_);
    // Below the gate a coupled draft is q's first entry by (z desc, id asc): the same transform with top-k 1.
    if(coupled() && policy_.temperature<options_.argmax_draft_below) draft_policy_.top_k=1U;
}

TokenDraft TokenSampler::draft(std::span<const float> logits,std::span<const std::uint32_t> ids,std::uint64_t position) {
    TokenDraft out;out.position=position;
    out.distribution=make_token_distribution(logits,ids,draft_policy_);
    out.token=out.distribution.sample(draft_uniform(position));
    ++stats_.drafts;
    return out;
}

TokenDecision TokenSampler::decide(std::span<const float> logits,std::uint64_t position,const TokenDraft* draft) {
    const auto out=decide_uncounted(logits,position,draft);
    count(out);
    return out;
}

TokenDecision TokenSampler::decide_uncounted(std::span<const float> logits,std::uint64_t position,const TokenDraft* draft) const {
    require(!draft || draft->position==position,"draft position differs from the decided position");
    const auto p=make_token_distribution(logits,{},policy_);
    const auto out=coupled()?decide_token_coupled(p,draft,
        speculation_uniform01(rng_,position,SpeculationRandomDrawClass::serial_target)):decide_token(p,draft,
        speculation_uniform01(rng_,position,SpeculationRandomDrawClass::serial_target),
        draft?speculation_uniform01(rng_,position,SpeculationRandomDrawClass::draft_acceptance):0.0,
        draft?speculation_uniform01(rng_,position,SpeculationRandomDrawClass::residual_fallback):0.0);
    return out;
}

TokenSampler::Uniforms TokenSampler::uniforms(std::uint64_t position,bool drafted) const {
    Uniforms u;
    u.plain=speculation_uniform01(rng_,position,SpeculationRandomDrawClass::serial_target);
    if(drafted && !coupled()) {
        u.accept=speculation_uniform01(rng_,position,SpeculationRandomDrawClass::draft_acceptance);
        u.residual=speculation_uniform01(rng_,position,SpeculationRandomDrawClass::residual_fallback);
    }
    return u;
}

double TokenSampler::draft_uniform(std::uint64_t position) const {
    // Coupled: the draft for position n uses the uniform row n decides with, so a draft equal in distribution to
    // the target row is drawn as the very token that row will sample.
    return speculation_uniform01(rng_,position,
        coupled()?SpeculationRandomDrawClass::serial_target:SpeculationRandomDrawClass::draft_proposal);
}

void TokenSampler::count(const TokenDecision& decision) noexcept {
    if(!decision.drafted) ++stats_.plain; else if(decision.accepted) ++stats_.accepted; else ++stats_.rejected;
    if(decision.forced) ++stats_.forced;
}

void TokenSampler::force_rejection(TokenDecision& decision,std::uint32_t row) const noexcept {
    const auto forced=options_.force_reject_row;
    if(forced==TokenSamplerOptions::forced_rejection_off || !decision.drafted || !decision.accepted) return;
    const auto target=forced==TokenSamplerOptions::forced_rejection_rotate?
        static_cast<std::uint32_t>((windows_?windows_-1U:0U)%8U):forced;
    // Coupled only (the constructor refuses it otherwise): the token stays the plain sample, which equals the draft.
    if(row==target) {decision.accepted=false;decision.forced=true;}
}

std::vector<TokenDecision> TokenSampler::decide_rows(std::uint32_t max_rows,std::uint64_t first_position,
    std::span<const TokenDraft> drafts,std::span<const std::uint32_t> eos,
    const std::function<std::span<const float>(std::uint32_t)>& logits) {
    std::vector<TokenDecision> out;
    begin_window();
    for(std::uint32_t r=0;r<max_rows;++r) {
        const auto* draft=r<drafts.size()?&drafts[r]:nullptr;
        auto decision=decide_uncounted(logits(r),first_position+r,draft);
        force_rejection(decision,r);
        count(decision);
        out.push_back(decision);
        const auto token=out.back().token;
        if(!draft || !out.back().accepted || std::find(eos.begin(),eos.end(),token)!=eos.end()) break;
    }
    return out;
}

}  // namespace strix
