#include "strix/speculation_measurement.hpp"

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

strix::SpeculationCycleMeasurement measurement(
    std::uint64_t proposal,
    std::uint32_t depth,
    double proposal_s,
    double verify_s,
    double committed,
    bool correct = true) {
    strix::SpeculationCycleMeasurement value;
    value.campaign_identity = 0xabcULL;
    value.proposal_identity = proposal;
    value.depth = depth;
    value.correctness_passed = correct;
    value.proposal_seconds = proposal_s;
    value.verification_seconds = verify_s;
    value.commit_replay_seconds = 0.001;
    value.uncovered_communication_seconds = 0.0;
    value.control_seconds = 0.001;
    value.committed_tokens = committed;
    return value;
}

}  // namespace

int main() {
    std::vector<strix::SpeculationCycleMeasurement>
        measurements;

    measurements.push_back(
        measurement(0,0,0.0,0.010,1.0));
    measurements.push_back(
        measurement(0,0,0.0,0.011,1.0));
    measurements.push_back(
        measurement(0,0,0.0,0.009,1.0));

    measurements.push_back(
        measurement(101,4,0.002,0.008,2.0));
    measurements.push_back(
        measurement(101,4,0.003,0.009,2.2));
    measurements.push_back(
        measurement(101,4,0.0025,0.0085,1.8));

    measurements.push_back(
        measurement(202,4,0.006,0.008,2.0));
    measurements.push_back(
        measurement(202,4,0.005,0.009,2.0));
    measurements.push_back(
        measurement(202,4,0.006,0.008,2.0));

    strix::SpeculationMeasurementPolicy policy;
    policy.campaign_identity = 0xabcULL;
    policy.minimum_repetitions = 3;

    const auto samples =
        strix::build_speculation_samples_from_measurements(
            measurements,
            policy);
    expect(
        samples.size() == 3,
        "measurement aggregation emits one sample per proposal/depth");

    const auto decision =
        strix::choose_speculation_depth(
            samples);
    expect(
        decision.selected_depth == 4 &&
            decision.selected_proposal_identity == 101,
        "aggregated campaign feeds controller and selects faster drafter");

    auto with_bad =
        measurements;
    with_bad.back().correctness_passed =
        false;
    const auto bad_samples =
        strix::build_speculation_samples_from_measurements(
            with_bad,
            policy);
    bool mtp_unqualified = false;
    for (const auto& sample :
         bad_samples) {
        if (sample.proposal_identity == 202) {
            mtp_unqualified =
                !sample.qualified;
        }
    }
    expect(
        mtp_unqualified,
        "one correctness failure disqualifies the entire proposal/depth campaign group");

    auto foreign =
        measurements;
    foreign.front().campaign_identity =
        0xdefULL;
    bool foreign_rejected = false;
    try {
        static_cast<void>(
            strix::build_speculation_samples_from_measurements(
                foreign,
                policy));
    } catch (const std::invalid_argument&) {
        foreign_rejected = true;
    }
    expect(
        foreign_rejected,
        "foreign campaign measurement cannot contaminate controller evidence");

    auto too_few =
        measurements;
    too_few.erase(
        too_few.begin() + 5);
    const auto few_samples =
        strix::build_speculation_samples_from_measurements(
            too_few,
            policy);
    bool prompt_unqualified = false;
    for (const auto& sample :
         few_samples) {
        if (sample.proposal_identity == 101) {
            prompt_unqualified =
                !sample.qualified;
        }
    }
    expect(
        prompt_unqualified,
        "proposal/depth group below minimum repetitions remains unqualified");

    if (failures == 0) {
        std::cout
            << "speculation measurement tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
