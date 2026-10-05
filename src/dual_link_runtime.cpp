#include "strix/dual_link_runtime.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix::transport {
namespace {

void validate_policy(
    const DualLinkRuntimePolicy& policy) {
    if (!std::isfinite(policy.metric_alpha) ||
        policy.metric_alpha <= 0.0 ||
        policy.metric_alpha > 1.0 ||
        !std::isfinite(
            policy.flow_switch_hysteresis_fraction) ||
        policy.flow_switch_hysteresis_fraction < 0.0) {
        throw std::invalid_argument(
            "dual-link runtime policy is invalid");
    }
}

void validate_directional_profile(
    const DirectionalLinkProfile& profile) {
    if (profile.outbound.id.empty() ||
        profile.inbound.id.empty() ||
        profile.outbound.id !=
            profile.inbound.id ||
        profile.outbound.peer_node !=
            profile.inbound.peer_node) {
        throw std::invalid_argument(
            "dual-link directional profile requires one physical link identity/peer");
    }
}

LinkProfile& choose(
    DirectionalLinkProfile& profile,
    LinkDirection direction) {
    return direction ==
               LinkDirection::outbound
        ? profile.outbound
        : profile.inbound;
}

const LinkProfile& choose(
    const DirectionalLinkProfile& profile,
    LinkDirection direction) {
    return direction ==
               LinkDirection::outbound
        ? profile.outbound
        : profile.inbound;
}

double ewma(
    double old_value,
    double new_value,
    double alpha) {
    if (!std::isfinite(old_value) ||
        old_value <= 0.0) {
        return new_value;
    }
    return old_value * (1.0 - alpha) +
           new_value * alpha;
}

}  // namespace

DualLinkRuntimeScheduler::DualLinkRuntimeScheduler(
    std::vector<DirectionalLinkProfile> links,
    DualLinkRuntimePolicy policy)
    : links_(std::move(links)),
      policy_(policy) {
    validate_policy(policy_);
    if (links_.empty()) {
        throw std::invalid_argument(
            "dual-link runtime scheduler requires at least one link");
    }
    for (const auto& link : links_) {
        validate_directional_profile(link);
    }
}

void DualLinkRuntimeScheduler::observe(
    const LiveLinkSample& sample) {
    if (sample.link_index >=
            links_.size() ||
        sample.payload_bytes == 0 ||
        !std::isfinite(sample.elapsed_seconds) ||
        sample.elapsed_seconds <= 0.0 ||
        !std::isfinite(
            sample.one_way_latency_seconds) ||
        sample.one_way_latency_seconds < 0.0) {
        throw std::invalid_argument(
            "dual-link live sample is invalid");
    }

    auto& profile =
        choose(
            links_[sample.link_index],
            sample.direction);
    const auto rate =
        static_cast<double>(
            sample.payload_bytes) /
        sample.elapsed_seconds;
    if (!std::isfinite(rate) ||
        rate <= 0.0) {
        throw std::invalid_argument(
            "dual-link live sample produced invalid payload rate");
    }

    profile.payload_bytes_per_second =
        ewma(
            profile.payload_bytes_per_second,
            rate,
            policy_.metric_alpha);
    profile.one_way_latency_seconds =
        ewma(
            profile.one_way_latency_seconds,
            sample.one_way_latency_seconds,
            policy_.metric_alpha);
    profile.in_flight_bytes =
        sample.in_flight_bytes;
}

void DualLinkRuntimeScheduler::set_link_health(
    std::size_t link_index,
    LinkHealth health) {
    if (link_index >= links_.size()) {
        throw std::out_of_range(
            "dual-link health update index is out of range");
    }

    auto& physical =
        links_[link_index];
    physical.outbound.health =
        health;
    physical.inbound.health =
        health;

    if (health == LinkHealth::failed ||
        health == LinkHealth::absent) {
        const auto erase_link =
            [link_index](auto& sticky) {
                for (auto it = sticky.begin();
                     it != sticky.end();) {
                    if (it->second ==
                        link_index) {
                        it =
                            sticky.erase(it);
                    } else {
                        ++it;
                    }
                }
            };
        erase_link(
            outbound_flow_link_);
        erase_link(
            inbound_flow_link_);
    }
}

std::vector<LinkProfile>
DualLinkRuntimeScheduler::directional_links(
    LinkDirection direction) const {
    std::vector<LinkProfile> result;
    result.reserve(links_.size());
    for (const auto& link : links_) {
        result.push_back(
            choose(link, direction));
    }
    return result;
}

TransferPlan DualLinkRuntimeScheduler::plan(
    std::uint64_t flow_id,
    LinkDirection direction,
    std::uint64_t bytes,
    TransferPriority priority) {
    if (flow_id == 0) {
        throw std::invalid_argument(
            "dual-link flow identity must be non-zero");
    }

    const auto profiles =
        directional_links(direction);
    auto best =
        plan_transfer(
            profiles,
            bytes,
            priority,
            policy_.transfer);

    if (best.kind !=
            PlanKind::single_link ||
        best.slices.size() != 1) {
        return best;
    }

    auto& sticky =
        direction ==
                LinkDirection::outbound
            ? outbound_flow_link_
            : inbound_flow_link_;
    const auto previous =
        sticky.find(flow_id);

    if (previous != sticky.end() &&
        previous->second <
            profiles.size() &&
        previous->second !=
            best.slices[0].link_index) {
        const auto sticky_seconds =
            estimate_link_completion_seconds(
                profiles[previous->second],
                bytes,
                policy_.transfer.
                    completion_target);
        const auto permitted =
            best.estimated_seconds *
            (1.0 +
             policy_.
                 flow_switch_hysteresis_fraction);
        if (std::isfinite(sticky_seconds) &&
            sticky_seconds <=
                permitted) {
            TransferPlan retained;
            retained.kind =
                PlanKind::single_link;
            retained.estimated_seconds =
                sticky_seconds;
            retained.slices.push_back(
                LinkSlice{
                    previous->second,
                    0,
                    bytes,
                });
            return retained;
        }
    }

    sticky[flow_id] =
        best.slices[0].link_index;
    return best;
}

}  // namespace strix::transport
