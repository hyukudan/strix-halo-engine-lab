#include "strix/token_sampling.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

constexpr double tail_cutoff = 40.0;
constexpr std::uint64_t single_request_identity = 1;

void require(bool ok,const char* message) {if(!ok) throw std::invalid_argument(message);}

struct Entry {
    double z = 0.0; // (logit - max logit) / temperature, <= 0
    std::uint32_t token = 0;
};

bool before(const Entry& a,const Entry& b) noexcept {return a.z>b.z || (a.z==b.z && a.token<b.token);}

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
    mix(1U); // transform revision: sampling-v1
    mix(std::bit_cast<std::uint32_t>(policy.temperature));
    mix(policy.top_k);
    mix(std::bit_cast<std::uint32_t>(policy.top_p));
    mix(std::bit_cast<std::uint32_t>(policy.min_p));
    return hash?hash:1U;
}

SpeculationRngIdentity token_sampling_rng(const TokenSamplingPolicy& policy) {
    return {policy.seed,single_request_identity,token_sampling_policy_identity(policy)};
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

TokenDistribution make_token_distribution(std::span<const float> logits,std::span<const std::uint32_t> ids,
    const TokenSamplingPolicy& policy) {
    require(token_sampling_enabled(policy),"token distribution requires a positive temperature");
    validate_token_sampling_policy(policy);
    require(!logits.empty() && (ids.empty() || ids.size()==logits.size()) &&
        logits.size()<=std::numeric_limits<std::uint32_t>::max(),"token distribution geometry is invalid");
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
    std::sort(kept.begin(),kept.end(),[](const Entry& a,const Entry& b){return a.token<b.token;});
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

TokenSampler::TokenSampler(TokenSamplingPolicy policy):policy_(policy),rng_(token_sampling_rng(policy)) {
    require(token_sampling_enabled(policy_),"token sampler requires a positive temperature");
}

TokenDraft TokenSampler::draft(std::span<const float> logits,std::span<const std::uint32_t> ids,std::uint64_t position) {
    TokenDraft out;out.position=position;
    out.distribution=make_token_distribution(logits,ids,policy_);
    out.token=out.distribution.sample(speculation_uniform01(rng_,position,SpeculationRandomDrawClass::draft_proposal));
    ++stats_.drafts;
    return out;
}

TokenDecision TokenSampler::decide(std::span<const float> logits,std::uint64_t position,const TokenDraft* draft) {
    require(!draft || draft->position==position,"draft position differs from the decided position");
    const auto p=make_token_distribution(logits,{},policy_);
    const auto out=decide_token(p,draft,
        speculation_uniform01(rng_,position,SpeculationRandomDrawClass::serial_target),
        draft?speculation_uniform01(rng_,position,SpeculationRandomDrawClass::draft_acceptance):0.0,
        draft?speculation_uniform01(rng_,position,SpeculationRandomDrawClass::residual_fallback):0.0);
    if(!draft) ++stats_.plain; else if(out.accepted) ++stats_.accepted; else ++stats_.rejected;
    return out;
}

TokenSampler::Uniforms TokenSampler::uniforms(std::uint64_t position,bool drafted) const {
    Uniforms u;
    u.plain=speculation_uniform01(rng_,position,SpeculationRandomDrawClass::serial_target);
    if(drafted) {
        u.accept=speculation_uniform01(rng_,position,SpeculationRandomDrawClass::draft_acceptance);
        u.residual=speculation_uniform01(rng_,position,SpeculationRandomDrawClass::residual_fallback);
    }
    return u;
}

double TokenSampler::draft_uniform(std::uint64_t position) const {
    return speculation_uniform01(rng_,position,SpeculationRandomDrawClass::draft_proposal);
}

void TokenSampler::count(const TokenDecision& decision) noexcept {
    if(!decision.drafted) ++stats_.plain; else if(decision.accepted) ++stats_.accepted; else ++stats_.rejected;
}

std::vector<TokenDecision> TokenSampler::decide_rows(std::uint32_t max_rows,std::uint64_t first_position,
    std::span<const TokenDraft> drafts,std::span<const std::uint32_t> eos,
    const std::function<std::span<const float>(std::uint32_t)>& logits) {
    std::vector<TokenDecision> out;
    for(std::uint32_t r=0;r<max_rows;++r) {
        const auto* draft=r<drafts.size()?&drafts[r]:nullptr;
        out.push_back(decide(logits(r),first_position+r,draft));
        const auto token=out.back().token;
        if(!draft || !out.back().accepted || std::find(eos.begin(),eos.end(),token)!=eos.end()) break;
    }
    return out;
}

}  // namespace strix
