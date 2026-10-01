#include "strix/speculation_controller.hpp"

#include <iostream>
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

    if (failures == 0) {
        std::cout << "speculation controller tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
