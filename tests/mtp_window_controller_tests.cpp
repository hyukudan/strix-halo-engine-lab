#include "strix/mtp_window_controller.hpp"
#include "strix/mtp_batch_window_controller.hpp"

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
    // A slot first observes cheap R2 cycles and slow deeper windows (for example
    // while other requests are scheduled), then serves a confident request.
    // Those individually plausible costs can suppress all deeper drafting: a
    // larger owner's fit-mask re-probe cannot help if the chain never reaches it.
    for(const auto model:{Model::legacy,Model::robust}) {
        MtpMinPWindowController reused({2U,4U,8U},0.7F,model,4);
        for(int k=0;k<10;++k) {
            reused.observe(0,2,20.0);reused.observe(1,4,200.0);reused.observe(2,8,300.0);
        }
        const std::vector<float> p{0.99F};
        expect(!reused.worth_drafting(p),"retained workload costs stop the next request at one draft");
        for(int k=0;k<200;++k)reused.calibrate(p,0,1);
        reused.set_external_rows(0,std::array<double,1>{0.01});
        reused.begin_session();
        MtpMinPWindowController fresh({2U,4U,8U},0.7F,model,4);
        expect(reused.worth_drafting(p),"request reset reopens deeper drafting from the prior");
        for(const auto& s:reused.owner_stats())
            expect(!s.cycles && !s.probes && s.cycle_ms==0.0,"reset clears costs and exploration history");
        const std::array<std::uint8_t,3> all{1U,1U,1U};
        const std::vector<float> chain{0.99F,0.85F,0.72F,0.4F};
        for(int k=0;k<80;++k) {
            const auto a=reused.choose(chain,chain.size(),all),b=fresh.choose(chain,chain.size(),all);
            expect(a==b,"reused request choices equal a fresh controller after reset");
            expect(reused.worth_drafting(p)==fresh.worth_drafting(p),"reset restores acceptance and external-row priors");
            reused.calibrate(chain,2,3);fresh.calibrate(chain,2,3);
            reused.observe(a,3,55.0);fresh.observe(b,3,55.0);
        }
    }
    // An implausible middle owner can also stay trapped: one fast re-probe
    // cannot displace a full median window of slow samples, and the larger
    // owner is never drafted even though its retained cost is competitive.
    {
        MtpMinPWindowController reused({2U,4U,8U},0.7F,Model::robust,64);
        for(int k=0;k<10;++k) {
            reused.observe(0,2,20.0);reused.observe(1,4,90.0);reused.observe(2,8,35.0);
        }
        const std::vector<float> p{0.99F};
        const std::array<std::uint8_t,3> all{1U,1U,1U};
        std::size_t probes=0,large=0;
        for(int k=0;k<128;++k) {
            const auto o=reused.choose(p,1,all);
            if(o==1U)++probes;
            if(o==2U)++large;
            reused.observe(o,2,o==0U?20.0:o==1U?25.0:35.0);
        }
        expect(probes==1U && large==0U,"a short reused session only gets one middle-owner re-probe");
        expect(reused.owner_stats()[2].cycle_ms==90.0 && !reused.worth_drafting(p),
            "one fast probe leaves the slow median and the deeper draft barrier intact");
        reused.begin_session();
        expect(reused.worth_drafting(p),"a new request clears the contaminated median without waiting for more probes");
    }
    {
        strix::MtpBatchWindowController joint(4);
        const std::vector<float> high(7,0.99F),low(7,0.01F);
        std::vector<strix::MtpBatchWindowController::Request> q{{0,8,high},{1,8,high}};
        const auto good=joint.choose(q,16);
        expect(good==std::vector<std::uint32_t>({8,8}),"joint selection retains deep, high acceptance windows");
        q[0].probabilities=low;q[1].probabilities=low;
        expect(joint.choose(q,16)==std::vector<std::uint32_t>({2,2}),"joint selection avoids low acceptance padding");
        q[0].limit=4;q[1].limit=2;q[0].probabilities=high;
        expect(joint.choose(q,16)==std::vector<std::uint32_t>({2,2}),"six physical rows cannot fall through to serial chunks");
        const std::vector<strix::MtpBatchWindowController::Request> tails{{0,2,{}},{1,2,{}},{2,2,{}}};
        expect(joint.choose(tails,16)==std::vector<std::uint32_t>({2,2,2}),"three R2-only tails retain admissible chunks");
        const std::vector<strix::MtpBatchWindowController::Request> triple{{0,8,{}},{1,8,{}},{2,8,{}}};
        const auto probe=joint.plan(triple,16);
        expect(probe==std::vector<std::uint32_t>({4,2,2}),"three-request exploration pads a valid eight-row cohort");
        q={{0,8,high},{1,8,high},{2,8,high},{3,8,high}};
        const auto bounded=joint.choose(q,16);
        std::uint32_t total=0;for(auto r:bounded) total+=r;
        expect(total<=16U,"joint exhaustive search respects the global row budget");
        q.resize(2);q[0].probabilities=low;q[1].probabilities=low;
        const std::array<std::uint32_t,2> rows{8,8},miss{0,0};
        for(int n=0;n<100;++n) joint.observe(q,rows,miss,200.0,30.0,14U);
        for(int k=0;k<9;++k) static_cast<void>(joint.plan(q,16)); // bounded exploration
        const auto no_draft=joint.plan(q,16);
        expect(no_draft==std::vector<std::uint32_t>({0,0}),"joint planning skips expensive unproductive drafts");
        joint.reset(0);joint.reset(1);
    }
    if(failures) {std::cerr<<failures<<" failure(s)\n";return 1;}
    std::cout<<"mtp window controller: legacy R2 share "<<share(stuck,2)<<", robust R8 share "<<share(fixed,8)
             <<" with "<<fixed.probes<<" probes; tok/ms "<<stuck_rate<<" -> "<<fixed_rate<<'\n';
    return 0;
}
