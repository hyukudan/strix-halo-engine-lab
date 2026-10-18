#include "strix/mtp_window_controller.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

namespace {
int failures=0;
void expect(bool pass,const char* message) {
    if(!pass) {std::cerr<<"FAIL: "<<message<<'\n';++failures;}
}
using strix::MtpMinPWindowController;
using strix::MtpWindowCostEstimate;
using Model=MtpWindowCostEstimate::Model;

struct Run {std::map<std::uint32_t,std::uint64_t> rows;std::uint64_t tokens=0,probes=0;double ms=0.0;};
// One synthetic decode against the controller, driven like HipMtpGreedyRuntime + the verifier:
// - a chain of up to 7 drafts stops after the first draft below min-p or when worth_drafting says no;
// - prefix windows of 2/4/8 rows (padding allowed), every draft at or above min-p is right;
// - cycle wall time = owner cost + 1 ms per draft; serial rows cost 45 ms each.
// script[c] gives the draft probabilities of cycle c (the last entry repeats); spike replaces the n-th cost
// sample of R4 (the second R4 window: the first one is the warm-up).
Run simulate(Model model,const std::vector<std::vector<float>>& script,std::size_t cycles,double r4_spike) {
    MtpMinPWindowController c({2U,4U,8U},0.8F,model);
    const std::array<double,3> cost{45.0,55.0,75.0};
    Run out;std::uint32_t r4_windows=0;
    for(std::size_t cycle=0;cycle<cycles;++cycle) {
        const auto& p_all=script[std::min(cycle,script.size()-1U)];
        std::vector<float> p;
        for(std::size_t k=0;k<7U;++k) {
            p.push_back(p_all[std::min(k,p_all.size()-1U)]);
            if(k+1U<7U && (p.back()<0.8F || !c.worth_drafting(p))) break;
        }
        std::size_t accepted=0;while(accepted<p.size() && p[accepted]>=0.8F) ++accepted;
        const std::array<std::uint8_t,3> fits{1U,1U,1U};
        const auto chosen=c.choose(p,p.size(),fits);
        std::size_t emitted;double ms;
        if(chosen==MtpMinPWindowController::serial) {
            emitted=accepted+1U;ms=45.0*static_cast<double>(emitted)+static_cast<double>(p.size());
            c.calibrate(p,accepted,emitted);
            if(cycle>=100U) ++out.rows[1];
        } else {
            const std::uint32_t rows=chosen==0?2U:chosen==1?4U:8U;
            emitted=std::min<std::size_t>(accepted,rows-1U)+1U;
            ms=cost[chosen]+static_cast<double>(p.size());
            if(rows==4U && ++r4_windows==2U) ms=r4_spike;
            c.calibrate(p,emitted-1U,rows-1U);
            if(cycle>=100U) ++out.rows[rows];
        }
        c.observe(chosen,emitted,ms);
        if(cycle>=100U) {out.tokens+=emitted;out.ms+=ms;}
    }
    for(const auto& s:c.owner_stats()) out.probes+=s.probes;
    return out;
}
double share(const Run& r,std::uint32_t rows) {
    std::uint64_t total=0;for(const auto& [k,v]:r.rows) total+=v;
    return total?static_cast<double>(r.rows.count(rows)?r.rows.at(rows):0U)/static_cast<double>(total):0.0;
}
} // namespace

int main() {
    {
        MtpWindowCostEstimate legacy(Model::legacy);
        legacy.add(100.0);legacy.add(300.0);
        expect(std::abs(legacy.value()-125.0)<1e-9,"legacy estimate: first sample, then EMA 0.25 with the 2x cap");
        MtpWindowCostEstimate robust(Model::robust);
        robust.add(111.0);expect(robust.provisional() && robust.value()==111.0,"robust: one sample is provisional");
        robust.add(55.0);expect(robust.value()==55.0,"robust: provisional estimate is the minimum, a spike cannot inflate it");
        robust.add(56.0);robust.add(54.0);robust.add(500.0);
        expect(!robust.provisional() && robust.value()==56.0,"robust: median of recent samples ignores one outlier");
        for(int i=0;i<8;++i) robust.add(60.0);
        expect(robust.value()==60.0,"robust: the window of 8 samples follows a lasting change");
    }
    // The stall seen on hardware (code prompt, min-p controller): R4's first cost sample is a 111 ms spike, R8
    // gets only its warm-up window, R2 is measured next; from then on worth_drafting stops every chain after
    // one draft because R4 looks too slow, R8 is never reached and every cycle is R2.
    std::vector<std::vector<float>> script{
        {0.99F,0.99F,0.99F,0.5F}, {0.99F,0.99F,0.99F,0.5F}, // R4 warm-up, then the 111 ms sample
        {0.99F},                                           // full chain: R8 warm-up (not a sample)
        {0.5F}, {0.5F},                                    // weak first drafts: R2 warm-up and sample
        {0.99F}};                                          // then a confident workload
    const auto stuck=simulate(Model::legacy,script,400,111.0);
    expect(share(stuck,2)>0.95,"legacy: the spike keeps every later cycle in R2 (stall reproduced)");
    const auto fixed=simulate(Model::robust,script,400,111.0);
    expect(share(fixed,2)<0.05 && share(fixed,8)>0.9,"robust: re-probes recover R4/R8 and the workload runs in R8");
    expect(fixed.probes>=1 && fixed.probes<=8,"robust: probes are few and bounded");
    const double stuck_rate=static_cast<double>(stuck.tokens)/stuck.ms,fixed_rate=static_cast<double>(fixed.tokens)/fixed.ms;
    expect(fixed_rate>1.5*stuck_rate,"robust: tokens per ms recover");
    // Without the spike both cost models settle on the same windows.
    const auto clean_legacy=simulate(Model::legacy,script,400,55.0),clean_robust=simulate(Model::robust,script,400,55.0);
    expect(share(clean_legacy,8)>0.9 && share(clean_robust,8)>0.9,"no spike: both models run the confident workload in R8");
    // A probe never picks an owner that does not fit.
    {
        MtpMinPWindowController c({2U,4U,8U},0.8F,Model::robust,4);
        const std::vector<float> p{0.5F};
        const std::array<std::uint8_t,3> only_r2{1U,0U,0U};
        for(int i=0;i<40;++i) {
            const auto o=c.choose(p,1U,only_r2);
            expect(o==0U || o==MtpMinPWindowController::serial,"probe respects the fits mask");
            c.observe(o,1U,45.0);
        }
    }
    if(failures) {std::cerr<<failures<<" failure(s)\n";return 1;}
    std::cout<<"mtp window controller: legacy R2 share "<<share(stuck,2)<<", robust R8 share "<<share(fixed,8)
             <<" with "<<fixed.probes<<" probes; tok/ms "<<stuck_rate<<" -> "<<fixed_rate<<'\n';
    return 0;
}
