#include "strix/speculation_controller.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    std::vector<strix::SpeculationDepthSample> samples{
        // Serial baseline: 10 ms/token.
        {0, true, 0.0, 0.010, 0.0, 0.0, 0.0, 1.0},

        // 4-token draft: 15 ms cycle, expected 2 committed => 7.5 ms/token.
        {4, true, 0.003, 0.009, 0.002, 0.0, 0.001, 2.0},

        // Deeper draft has more acceptance, but worse full-cycle cost/token.
        {8, true, 0.007, 0.017, 0.004, 0.0, 0.001, 3.0},
    };

    auto decision =
        strix::choose_speculation_depth(samples);
    expect(decision.selected_depth == 4 &&
               decision.speculation_enabled,
           "controller chooses lowest complete cost per committed token");

    strix::SpeculationControllerPolicy policy;
    policy.current_depth = 0;
    policy.switch_hysteresis_fraction = 0.30;

    decision =
        strix::choose_speculation_depth(samples, policy);
    expect(decision.selected_depth == 0 &&
               !decision.speculation_enabled &&
               !decision.switched,
           "hysteresis prevents a marginally insufficient switch");

    policy.switch_hysteresis_fraction = 0.20;
    decision =
        strix::choose_speculation_depth(samples, policy);
    expect(decision.selected_depth == 4 &&
               decision.switched,
           "controller switches when gain exceeds hysteresis");

    // Make every positive depth slower than serial.
    samples[1].verification_seconds = 0.020;
    samples[2].verification_seconds = 0.030;
    policy.current_depth = 4;
    policy.switch_hysteresis_fraction = 0.03;
    decision =
        strix::choose_speculation_depth(samples, policy);
    expect(decision.selected_depth == 0 &&
               !decision.speculation_enabled &&
               decision.switched,
           "depth zero wins when speculation loses end to end");

    auto unqualified = samples;
    unqualified[1].qualified = false;
    unqualified[2].qualified = false;
    policy.current_depth.reset();
    decision =
        strix::choose_speculation_depth(unqualified, policy);
    expect(decision.selected_depth == 0 &&
               decision.qualified_costs.size() == 1,
           "unqualified positive depths never enter selection");

    bool missing_serial = false;
    try {
        const std::vector<strix::SpeculationDepthSample> invalid{
            {4, true, 0.001, 0.001, 0.0, 0.0, 0.0, 2.0},
        };
        static_cast<void>(
            strix::choose_speculation_depth(invalid));
    } catch (const std::invalid_argument&) {
        missing_serial = true;
    }
    expect(missing_serial,
           "controller requires a qualified serial baseline");

    {
        auto fast_prompt = samples[1];
        fast_prompt.depth = 4;
        fast_prompt.qualified = true;
        fast_prompt.proposal_seconds = 0.002;
        fast_prompt.verification_seconds = 0.008;
        fast_prompt.commit_replay_seconds = 0.001;
        fast_prompt.control_seconds = 0.001;
        fast_prompt.expected_committed_tokens = 2.0;
        fast_prompt.proposal_identity = 101;

        auto slower_mtp = fast_prompt;
        slower_mtp.proposal_seconds = 0.005;
        slower_mtp.proposal_identity = 202;

        const std::vector<strix::SpeculationDepthSample>
            multi_proposal{
                samples.front(),
                fast_prompt,
                slower_mtp,
            };

        strix::SpeculationControllerPolicy
            multi_policy;
        multi_policy.current_depth = 4;
        multi_policy.current_proposal_identity = 202;
        multi_policy.switch_hysteresis_fraction = 0.10;

        const auto multi_decision =
            strix::choose_speculation_depth(
                multi_proposal,
                multi_policy);
        expect(multi_decision.selected_depth == 4 &&
                   multi_decision.selected_proposal_identity == 101 &&
                   multi_decision.switched,
               "controller compares distinct drafters at the same depth");
    }

    bool duplicate = false;
    try {
        auto invalid = samples;
        invalid.push_back(samples.front());
        static_cast<void>(
            strix::choose_speculation_depth(invalid));
    } catch (const std::invalid_argument&) {
        duplicate = true;
    }
    expect(duplicate,
           "controller rejects duplicate proposal/depth observations");

    {
        // These are synthetic host costs, not measurements or qualification
        // of any native depth. Real lookup records only after bonus consumption.
        strix::ObservedSpeculationController observed;
        for (std::uint32_t depth=0; depth<3; ++depth) {
            for (int repetition=0; repetition<2; ++repetition) {
                expect(observed.select(2)==depth,"observed bootstrap measures serial before each positive depth");
                const double seconds=depth==0 ? .010 : (depth==1 ? .018 : .021);
                expect(observed.observe(depth,seconds,depth+1U),"completed full-cycle sample admitted");
            }
        }
        expect(observed.select(2)==2,"observed controller compares full wall cost per consumed token");
        expect(observed.observe(2,.021,3),"selected completed depth-two cycle recorded");
        expect(observed.costs()[0].samples==2 && observed.costs()[2].committed_tokens==9,
            "observed controller retains real sample and consumed-token counts");
        observed.reset();
        expect(observed.costs()[2].samples==0 && observed.select(2)==0,
            "new runtime cost context discards all previous cost estimates");
        observed.cancel_observation();
        expect(observed.select(0)==0,"depth zero remains available under an effective zero cap");
        observed.cancel_observation();
        expect(observed.costs()[0].samples==0,"censored terminal frontier is not a free serial baseline");
    }
    {
        strix::ObservedSpeculationPolicy observed_policy;
        observed_policy.bootstrap_samples_per_depth=1;
        observed_policy.reexplore_interval=4;
        strix::ObservedSpeculationController observed(observed_policy);
        expect(observed.select(1)==0 && observed.observe(0,.010,1),"synthetic completed baseline");
        expect(observed.select(1)==1 && observed.observe(1,.0196,2),"synthetic marginal positive-depth cost");
        expect(observed.select(1)==0,"three percent hysteresis retains serial at a two percent gain");
        expect(observed.observe(0,.010,1),"record first exploitation cycle");
        expect(observed.select(1)==0 && observed.observe(0,.010,1),"record second exploitation cycle");
        expect(observed.select(1)==0 && observed.observe(0,.010,1),"bounded exploration includes actual depth zero");
        for(int cycle=0;cycle<4;++cycle) {
            expect(observed.select(1)==0,"preferred serial between exploration observations");
            expect(observed.observe(0,.010,1),"complete serial cycle before next exploration");
        }
        expect(observed.select(1)==1,"bounded exploration revisits the other available depth");
        observed.cancel_observation();
    }
    {
        strix::ObservedSpeculationPolicy observed_policy;
        observed_policy.bootstrap_samples_per_depth=1;
        observed_policy.reexplore_interval=100;
        strix::ObservedSpeculationController observed(observed_policy);
        expect(observed.select(1)==0 && observed.observe(0,.010,1),"actual serial denominator is one");
        expect(observed.select(1)==1 && observed.observe(1,.014,2),"positive-depth useful work can amortize full cost");
        expect(observed.select(1)==1,"beneficial observed depth selected after serial baseline");
        // A long bonus consumption/publication time must remain in the same
        // unsplit scope, rather than comparing just the cheap verification.
        expect(observed.observe(1,.100,1),"one accepted output still pays its complete observed wall cost");
        expect(observed.select(1)==0,"full cycle slowdown returns to depth zero");
        observed.cancel_observation();
    }
    {
        strix::ObservedSpeculationController observed;
        expect(!observed.observe(0,.010,1),"unselected sample cannot enter online costs");
        expect(observed.select(2)==0,"start censored/invalid measurement checks");
        for(double invalid:{0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
            expect(!observed.observe(0,invalid,1),"nonfinite or empty full wall observation rejected");
        expect(!observed.observe(0,.010,0) && !observed.observe(0,.010,2) && !observed.observe(1,.010,1),
            "zero/incorrect baseline denominator or foreign selected depth rejected");
        expect(observed.costs()[0].samples==0,"invalid observations leave all cost totals unchanged");
        bool pending_rejected=false;
        try { static_cast<void>(observed.select(2)); } catch(const std::logic_error&) { pending_rejected=true; }
        expect(pending_rejected,"cannot select twice across an unfinished observed cycle");
        observed.cancel_observation();
        bool cap_rejected=false;
        try { static_cast<void>(observed.select(3)); } catch(const std::invalid_argument&) { cap_rejected=true; }
        expect(cap_rejected,"unsupported observed depth rejected before selection");
        expect(observed.select(0)==0 && observed.observe(0,.010,1),"healthy baseline survives cancelled observation");
    }
    {
        bool invalid_policy=false;
        try { strix::ObservedSpeculationPolicy p; p.bootstrap_samples_per_depth=0;
            strix::ObservedSpeculationController observed(p); }
        catch(const std::invalid_argument&) { invalid_policy=true; }
        expect(invalid_policy,"zero bootstrap repetitions rejected before execution");
        invalid_policy=false;
        try { strix::ObservedSpeculationPolicy p; p.reexplore_interval=0;
            strix::ObservedSpeculationController observed(p); }
        catch(const std::invalid_argument&) { invalid_policy=true; }
        expect(invalid_policy,"zero reexploration interval rejected before execution");
    }

    if (failures == 0) {
        std::cout << "speculation controller tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
