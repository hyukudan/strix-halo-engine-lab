#include "strix/speculation_measurement.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix {
namespace {

using Key =
    std::pair<std::uint64_t, std::uint32_t>;

struct Group {
    std::vector<double> proposal;
    std::vector<double> verification;
    std::vector<double> commit_replay;
    std::vector<double> communication;
    std::vector<double> control;
    double committed_sum = 0.0;
    bool correctness_clean = true;
};

void require_component(
    double value,
    const char* message) {
    if (!std::isfinite(value) ||
        value < 0.0) {
        throw std::invalid_argument(message);
    }
}

double median(
    std::vector<double> values) {
    if (values.empty()) {
        throw std::logic_error(
            "speculation measurement median requires values");
    }
    std::sort(
        values.begin(),
        values.end());
    const auto middle =
        values.size() / 2U;
    if (values.size() % 2U != 0U) {
        return values[middle];
    }
    return (
        values[middle - 1U] +
        values[middle]) /
        2.0;
}

}  // namespace

std::vector<SpeculationDepthSample>
build_speculation_samples_from_measurements(
    std::span<const SpeculationCycleMeasurement> measurements,
    const SpeculationMeasurementPolicy& policy) {
    if (policy.campaign_identity == 0 ||
        policy.minimum_repetitions == 0 ||
        measurements.empty()) {
        throw std::invalid_argument(
            "speculation measurement policy/campaign is invalid");
    }

    std::map<Key, Group> groups;
    for (const auto& measurement :
         measurements) {
        if (measurement.campaign_identity !=
            policy.campaign_identity) {
            throw std::invalid_argument(
                "speculation measurement belongs to foreign campaign");
        }
        if (measurement.depth == 0 &&
            measurement.proposal_identity != 0) {
            throw std::invalid_argument(
                "serial baseline measurement cannot have proposal identity");
        }
        if (measurement.depth != 0 &&
            measurement.proposal_identity == 0) {
            throw std::invalid_argument(
                "positive-depth measurement requires proposal identity");
        }

        require_component(
            measurement.proposal_seconds,
            "invalid measured proposal seconds");
        require_component(
            measurement.verification_seconds,
            "invalid measured verification seconds");
        require_component(
            measurement.commit_replay_seconds,
            "invalid measured commit/replay seconds");
        require_component(
            measurement.uncovered_communication_seconds,
            "invalid measured communication seconds");
        require_component(
            measurement.control_seconds,
            "invalid measured control seconds");
        if (!std::isfinite(
                measurement.committed_tokens) ||
            measurement.committed_tokens <= 0.0 ||
            (measurement.depth == 0 &&
             std::abs(
                 measurement.committed_tokens -
                 1.0) > 1e-12)) {
            throw std::invalid_argument(
                "invalid measured committed-token count");
        }

        auto& group =
            groups[{
                measurement.proposal_identity,
                measurement.depth,
            }];
        group.proposal.push_back(
            measurement.proposal_seconds);
        group.verification.push_back(
            measurement.verification_seconds);
        group.commit_replay.push_back(
            measurement.commit_replay_seconds);
        group.communication.push_back(
            measurement.uncovered_communication_seconds);
        group.control.push_back(
            measurement.control_seconds);
        group.committed_sum +=
            measurement.committed_tokens;
        if (!std::isfinite(
                group.committed_sum)) {
            throw std::overflow_error(
                "speculation committed-token aggregation overflow");
        }
        group.correctness_clean =
            group.correctness_clean &&
            measurement.correctness_passed;
    }

    std::vector<SpeculationDepthSample>
        result;
    result.reserve(
        groups.size());

    for (auto& [key, group] :
         groups) {
        const auto repetitions =
            group.proposal.size();
        SpeculationDepthSample sample;
        sample.proposal_identity =
            key.first;
        sample.depth =
            key.second;
        sample.qualified =
            repetitions >=
                policy.minimum_repetitions &&
            group.correctness_clean;
        sample.proposal_seconds =
            median(
                std::move(group.proposal));
        sample.verification_seconds =
            median(
                std::move(group.verification));
        sample.commit_replay_seconds =
            median(
                std::move(group.commit_replay));
        sample.uncovered_communication_seconds =
            median(
                std::move(group.communication));
        sample.control_seconds =
            median(
                std::move(group.control));
        sample.expected_committed_tokens =
            group.committed_sum /
            static_cast<double>(
                repetitions);
        result.push_back(
            sample);
    }

    return result;
}

}  // namespace strix
