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
           "controller rejects duplicate depth observations");

    if (failures == 0) {
        std::cout << "speculation controller tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
