#include "strix/mtp_window_controller.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix {
void MtpWindowCostEstimate::add(double ms) {
    if(model_==Model::legacy) {
        constexpr double alpha=0.25;
        value_=samples_?value_+alpha*(std::min(ms,2.0*value_)-value_):ms;
        ++samples_;return;
    }
    recent_[samples_%recent_.size()]=ms;++samples_;
    const auto n=static_cast<std::size_t>(std::min<std::uint64_t>(samples_,recent_.size()));
    std::array<double,8> sorted{};
    for(std::size_t i=0;i<n;++i) { // insertion sort of at most 8 samples
        std::size_t j=i;
        while(j && sorted[j-1U]>recent_[i]) {sorted[j]=sorted[j-1U];--j;}
        sorted[j]=recent_[i];
    }
    value_=provisional()?sorted[0]:(n%2U?sorted[n/2U]:0.5*(sorted[n/2U-1U]+sorted[n/2U]));
}

MtpMinPWindowController::MtpMinPWindowController(std::vector<std::uint32_t> owner_rows,float min_probability,
    MtpWindowCostEstimate::Model model,std::uint64_t reprobe_cycles)
    :serial_(model),model_(model),min_probability_(min_probability),reprobe_cycles_(reprobe_cycles) {
    if(owner_rows.empty() || !std::is_sorted(owner_rows.begin(),owner_rows.end()) || owner_rows.front()<2U ||
        std::adjacent_find(owner_rows.begin(),owner_rows.end())!=owner_rows.end() ||
        !(min_probability>=0.0F && min_probability<=1.0F))
        throw std::invalid_argument("MTP window controller requires distinct ascending owners and a probability in [0, 1]");
    for(const auto rows:owner_rows) {Owner o;o.rows=rows;o.cost=MtpWindowCostEstimate(model);o.interval=reprobe_cycles;owners_.push_back(o);}
}
namespace {
std::size_t bin(float p) {return std::min<std::size_t>(9U,static_cast<std::size_t>(std::max(p,0.0F)*10.0F));}
}
// Calibrated acceptance: the bin's observed rate with a two-draft prior at p.
double MtpMinPWindowController::acceptance(float p) const {const auto b=bin(p);return (accepted_[b]+2.0*p)/(tested_[b]+2.0);}
void MtpMinPWindowController::calibrate(std::span<const float> p,std::size_t accepted_drafts,std::size_t verified_rows) {
    for(std::size_t j=0;j<p.size() && j<verified_rows && j<=accepted_drafts;++j){
        const auto b=bin(p[j]);tested_[b]+=1.0;if(j<accepted_drafts)accepted_[b]+=1.0;}
}
// Expected emitted tokens of an R-row prefix window when draft j is accepted
// with its calibrated probability, given that every earlier draft was accepted.
double MtpMinPWindowController::expected_tokens(std::uint32_t rows,std::span<const float> p) const {
    double expected=1.0,run=1.0;
    for(std::uint32_t j=1;j<rows && j<=p.size();++j){run*=acceptance(p[j-1U]);expected+=run;}
    return expected;
}
bool MtpMinPWindowController::worth_drafting(std::span<const float> p) const {
    if(min_probability_<=0.0F || p.empty()) return true;
    const auto usable=p.size()+1U;
    double best=serial_.samples()?1.0/serial_.value():0.0;
    const Owner* next=nullptr;
    for(const auto& o:owners_) {
        if(o.rows<=usable) {if(o.cost.samples()) best=std::max(best,expected_tokens(o.rows,p)/o.cost.value());}
        else if(!next) next=&o;
    }
    if(!next) return false;
    if(!next->cost.samples()) return true;
    std::vector<float> assumed(p.begin(),p.end());assumed.resize(next->rows-1U,p.back());
    return expected_tokens(next->rows,assumed)/next->cost.value()>best;
}
bool MtpMinPWindowController::implausible(const Owner& o) const {
    if(!o.cost.samples() || o.cost.provisional()) return true;
    return std::any_of(owners_.begin(),owners_.end(),[&](const Owner& m){
        return m.rows>o.rows && m.cost.samples() && !m.cost.provisional() && m.cost.value()<o.cost.value();});
}
std::size_t MtpMinPWindowController::choose(std::span<const float> p,std::size_t candidates,std::span<const std::uint8_t> fits) {
    if(fits.size()!=owners_.size()) throw std::invalid_argument("MTP window controller fits mask differs from its owners");
    const auto n=owners_.size();
    const auto largest=[&]()->std::size_t{for(std::size_t o=n;o-->0;) if(fits[o]) return o;return serial;};
    if(min_probability_<=0.0F || p.size()!=candidates) return largest();
    ++choices_;
    const auto use=[&](std::size_t o){if(o!=serial) owners_[o].last_used=choices_;return o;};
    if(model_==MtpWindowCostEstimate::Model::robust && reprobe_cycles_) {
        for(std::size_t o=n;o-->0;) {
            auto& owner=owners_[o];
            if(fits[o] && choices_-owner.last_used>owner.interval && implausible(owner)) {
                owner.interval*=2U;++owner.probes;return use(o);
            }
        }
    }
    std::uint32_t want=1;
    while(want-1U<p.size() && p[want-1U]>=min_probability_) ++want;
    std::size_t cap=n-1U;
    for(std::size_t o=0;o<n;++o) if(owners_[o].rows>=want){cap=o;break;}
    const bool serial_only=want==1U;
    // Unmeasured options are tried once before the cost model ranks them.
    if(fits[cap] && !owners_[cap].cost.samples()) return use(cap);
    if(serial_only && !serial_.samples()) return serial;
    std::size_t best=serial;double best_rate=0.0;
    for(std::size_t o=0;o<n;++o) {
        const auto& owner=owners_[o];
        if(owner.rows>owners_[cap].rows || !fits[o] || !owner.cost.samples()) continue;
        const auto rate=expected_tokens(owner.rows,p)/owner.cost.value();
        if(best==serial || rate>best_rate){best=o;best_rate=rate;}
    }
    if(serial_only && (best==serial || 1.0/serial_.value()>best_rate)) return serial;
    return use(best!=serial?best:largest());
}
void MtpMinPWindowController::observe(std::size_t owner,std::size_t emitted,double ms) {
    if(!emitted) return;
    if(owner==serial) {serial_.add(ms/static_cast<double>(emitted));return;}
    auto& o=owners_.at(owner);
    if(o.warmed) o.cost.add(ms);
    o.warmed=true;
}
std::vector<MtpMinPWindowController::OwnerStats> MtpMinPWindowController::owner_stats() const {
    std::vector<OwnerStats> out{{1U,serial_.samples(),serial_.value(),0U}};
    for(const auto& o:owners_) out.push_back({o.rows,o.cost.samples(),o.cost.value(),o.probes});
    return out;
}
} // namespace strix
