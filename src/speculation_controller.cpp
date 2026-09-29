#include "strix/speculation_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace strix {
namespace {

double checked_component(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(name);
    }
    return value;
}

SpeculationDepthCost cost(const SpeculationDepthSample& sample) {
    if (!std::isfinite(sample.expected_committed_tokens) ||
        sample.expected_committed_tokens <= 0.0) {
        throw std::invalid_argument(
            "expected committed tokens must be finite and positive");
    }

    const double cycle =
        checked_component(sample.proposal_seconds, "invalid proposal cost") +
        checked_component(sample.verification_seconds, "invalid verification cost") +
        checked_component(sample.commit_replay_seconds, "invalid commit/replay cost") +
        checked_component(
            sample.uncovered_communication_seconds,
            "invalid uncovered communication cost") +
        checked_component(sample.control_seconds, "invalid control cost");

    if (!(cycle > 0.0) || !std::isfinite(cycle)) {
        throw std::invalid_argument(
            "qualified speculation cycle cost must be finite and positive");
    }

    return SpeculationDepthCost{
        sample.depth,
        cycle,
        sample.expected_committed_tokens,
        cycle / sample.expected_committed_tokens,
    };
}

}  // namespace

SpeculationDecision choose_speculation_depth(
    std::span<const SpeculationDepthSample> samples,
    const SpeculationControllerPolicy& policy) {
    if (!std::isfinite(policy.switch_hysteresis_fraction) ||
        policy.switch_hysteresis_fraction < 0.0 ||
        policy.switch_hysteresis_fraction >= 1.0) {
        throw std::invalid_argument(
            "switch hysteresis must be finite and within [0, 1)");
    }

    std::unordered_set<std::uint32_t> depths;
    depths.reserve(samples.size());

    SpeculationDecision decision;
    bool qualified_depth_zero = false;

    for (const auto& sample : samples) {
        if (!depths.insert(sample.depth).second) {
            throw std::invalid_argument(
                "speculation depth samples must have unique depths");
        }
        if (!sample.qualified) {
            continue;
        }

        auto evaluated = cost(sample);
        if (sample.depth == 0) {
            if (std::abs(sample.expected_committed_tokens - 1.0) > 1e-12) {
                throw std::invalid_argument(
                    "depth-zero serial baseline must commit exactly one token");
            }
            qualified_depth_zero = true;
        }
        decision.qualified_costs.push_back(evaluated);
    }

    if (!qualified_depth_zero) {
        throw std::invalid_argument(
            "speculation controller requires a qualified depth-zero baseline");
    }

    std::sort(
        decision.qualified_costs.begin(),
        decision.qualified_costs.end(),
        [](const auto& left, const auto& right) {
            return left.depth < right.depth;
        });

    const auto best_it = std::min_element(
        decision.qualified_costs.begin(),
        decision.qualified_costs.end(),
        [](const auto& left, const auto& right) {
            if (left.seconds_per_committed_token !=
                right.seconds_per_committed_token) {
                return left.seconds_per_committed_token <
                       right.seconds_per_committed_token;
            }
            return left.depth < right.depth;
        });

    std::uint32_t selected = best_it->depth;
    bool switched = false;

    if (policy.current_depth.has_value()) {
        const auto current_it = std::find_if(
            decision.qualified_costs.begin(),
            decision.qualified_costs.end(),
            [&](const auto& item) {
                return item.depth == *policy.current_depth;
            });

        if (current_it != decision.qualified_costs.end() &&
            current_it->depth != best_it->depth) {
            const double required =
                current_it->seconds_per_committed_token *
                (1.0 - policy.switch_hysteresis_fraction);
            if (best_it->seconds_per_committed_token >= required) {
                selected = current_it->depth;
            } else {
                switched = true;
            }
        } else if (current_it == decision.qualified_costs.end()) {
            switched = true;
        }
    } else {
        switched = selected != 0;
    }

    decision.selected_depth = selected;
    decision.speculation_enabled = selected != 0;
    decision.switched = switched;
    return decision;
}

}  // namespace strix
