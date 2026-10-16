#include "strix/speculation_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

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
        sample.proposal_identity,
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

    std::set<std::pair<std::uint64_t, std::uint32_t>>
        proposal_depths;

    SpeculationDecision decision;
    bool qualified_depth_zero = false;

    for (const auto& sample : samples) {
        if (!proposal_depths.insert(
                {sample.proposal_identity,
                 sample.depth}).second) {
            throw std::invalid_argument(
                "speculation samples must have unique proposal/depth pairs");
        }
        if (sample.depth == 0 &&
            sample.proposal_identity != 0) {
            throw std::invalid_argument(
                "depth-zero serial baseline cannot have proposal identity");
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
            if (left.depth != right.depth) {
                return left.depth < right.depth;
            }
            return left.proposal_identity <
                   right.proposal_identity;
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
            if (left.depth != right.depth) {
                return left.depth < right.depth;
            }
            return left.proposal_identity <
                   right.proposal_identity;
        });

    std::uint32_t selected_depth =
        best_it->depth;
    std::uint64_t selected_proposal =
        best_it->proposal_identity;
    bool switched = false;

    if (policy.current_proposal_identity.has_value() &&
        !policy.current_depth.has_value()) {
        throw std::invalid_argument(
            "current proposal identity requires current depth");
    }

    if (policy.current_depth.has_value()) {
        auto current_it =
            decision.qualified_costs.end();

        if (policy.current_proposal_identity.has_value()) {
            current_it = std::find_if(
                decision.qualified_costs.begin(),
                decision.qualified_costs.end(),
                [&](const auto& item) {
                    return item.depth ==
                               *policy.current_depth &&
                           item.proposal_identity ==
                               *policy.current_proposal_identity;
                });
        } else {
            const auto first =
                std::find_if(
                    decision.qualified_costs.begin(),
                    decision.qualified_costs.end(),
                    [&](const auto& item) {
                        return item.depth ==
                            *policy.current_depth;
                    });
            if (first !=
                decision.qualified_costs.end()) {
                const auto another =
                    std::find_if(
                        first + 1,
                        decision.qualified_costs.end(),
                        [&](const auto& item) {
                            return item.depth ==
                                *policy.current_depth;
                        });
                if (another ==
                    decision.qualified_costs.end()) {
                    current_it = first;
                }
            }
        }

        const bool same_choice =
            current_it !=
                decision.qualified_costs.end() &&
            current_it->depth ==
                best_it->depth &&
            current_it->proposal_identity ==
                best_it->proposal_identity;

        if (current_it !=
                decision.qualified_costs.end() &&
            !same_choice) {
            const double required =
                current_it->seconds_per_committed_token *
                (1.0 - policy.switch_hysteresis_fraction);
            if (best_it->seconds_per_committed_token >=
                required) {
                selected_depth =
                    current_it->depth;
                selected_proposal =
                    current_it->proposal_identity;
            } else {
                switched = true;
            }
        } else if (current_it ==
                   decision.qualified_costs.end()) {
            switched = true;
        }
    } else {
        switched = selected_depth != 0;
    }

    decision.selected_depth =
        selected_depth;
    decision.selected_proposal_identity =
        selected_proposal;
    decision.speculation_enabled =
        selected_depth != 0;
    decision.switched = switched;
    return decision;
}

ObservedSpeculationController::ObservedSpeculationController(ObservedSpeculationPolicy policy)
    : policy_(policy) {
    if (!policy_.bootstrap_samples_per_depth || policy_.bootstrap_samples_per_depth > 64U ||
        !policy_.reexplore_interval || policy_.reexplore_interval > 1048576U ||
        !std::isfinite(policy_.switch_hysteresis_fraction) ||
        policy_.switch_hysteresis_fraction < 0.0 || policy_.switch_hysteresis_fraction >= 1.0)
        throw std::invalid_argument("invalid observed speculation policy");
}

std::uint32_t ObservedSpeculationController::select(std::uint32_t maximum_depth) {
    if (maximum_depth > 2U) throw std::invalid_argument("observed speculation supports only depths zero, one or two");
    if (pending_) throw std::logic_error("finish the previous observed speculation cycle before selection");
    exploring_ = false;
    for (std::uint32_t depth = 0; depth <= maximum_depth; ++depth) {
        if (costs_[depth].samples < policy_.bootstrap_samples_per_depth) {
            pending_ = true; selected_depth_ = depth; return depth;
        }
    }
    // Sum wall time / sum actually consumed tokens weights observations by
    // useful work. The strict 3% default threshold matches the fixed-sample
    // controller, including retaining the current choice at the boundary.
    const auto rate = [&](std::uint32_t depth) {
        return costs_[depth].complete_cycle_seconds / static_cast<double>(costs_[depth].committed_tokens);
    };
    std::uint32_t best = 0;
    for (std::uint32_t depth = 1; depth <= maximum_depth; ++depth)
        if (rate(depth) < rate(best)) best = depth;
    if (preferred_depth_ > maximum_depth ||
        rate(best) < rate(preferred_depth_) * (1.0 - policy_.switch_hysteresis_fraction))
        preferred_depth_ = best;
    selected_depth_ = preferred_depth_;
    if (since_exploration_ >= policy_.reexplore_interval) {
        selected_depth_ = exploration_cursor_ % (maximum_depth + 1U);
        exploring_ = true;
    }
    pending_ = true;
    return selected_depth_;
}

bool ObservedSpeculationController::observe(std::uint32_t depth, double seconds,
                                           std::uint64_t committed) noexcept {
    if (!pending_ || depth != selected_depth_ || depth > 2U || !std::isfinite(seconds) || seconds <= 0.0 ||
        !committed || committed > depth + 1U) return false;
    auto& cost = costs_[depth];
    const double total = cost.complete_cycle_seconds + seconds;
    if (cost.samples == std::numeric_limits<std::uint64_t>::max() ||
        committed > std::numeric_limits<std::uint64_t>::max() - cost.committed_tokens || !std::isfinite(total)) return false;
    ++cost.samples; cost.committed_tokens += committed; cost.complete_cycle_seconds = total;
    if (exploring_) { since_exploration_ = 0; exploration_cursor_ = (depth + 1U) % 3U; }
    else if (since_exploration_ != std::numeric_limits<std::uint64_t>::max()) ++since_exploration_;
    pending_ = false; exploring_ = false;
    return true;
}

void ObservedSpeculationController::cancel_observation() noexcept { pending_ = false; exploring_ = false; }
void ObservedSpeculationController::reset() noexcept {
    costs_ = {}; since_exploration_ = 0; preferred_depth_ = 0; exploration_cursor_ = 0;
    selected_depth_ = 0; pending_ = false; exploring_ = false;
}

}  // namespace strix
