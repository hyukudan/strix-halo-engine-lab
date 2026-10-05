#pragma once

#include "strix/transport.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace strix::transport {

enum class LinkDirection : std::uint8_t {
    outbound,
    inbound,
};

struct DirectionalLinkProfile {
    LinkProfile outbound;
    LinkProfile inbound;
};

struct LiveLinkSample {
    std::size_t link_index = 0;
    LinkDirection direction =
        LinkDirection::outbound;
    std::uint64_t payload_bytes = 0;
    double elapsed_seconds = 0.0;
    double one_way_latency_seconds = 0.0;
    std::uint64_t in_flight_bytes = 0;
};

struct DualLinkRuntimePolicy {
    SchedulerPolicy transfer;
    double metric_alpha = 0.25;
    double flow_switch_hysteresis_fraction = 0.05;
};

class DualLinkRuntimeScheduler {
public:
    explicit DualLinkRuntimeScheduler(
        std::vector<DirectionalLinkProfile> links,
        DualLinkRuntimePolicy policy = {});

    void observe(
        const LiveLinkSample& sample);

    void set_link_health(
        std::size_t link_index,
        LinkHealth health);

    [[nodiscard]] TransferPlan plan(
        std::uint64_t flow_id,
        LinkDirection direction,
        std::uint64_t bytes,
        TransferPriority priority);

    [[nodiscard]] const std::vector<DirectionalLinkProfile>&
    links() const noexcept {
        return links_;
    }

private:
    [[nodiscard]] std::vector<LinkProfile>
    directional_links(
        LinkDirection direction) const;

    std::vector<DirectionalLinkProfile> links_;
    DualLinkRuntimePolicy policy_;
    std::unordered_map<std::uint64_t, std::size_t>
        outbound_flow_link_;
    std::unordered_map<std::uint64_t, std::size_t>
        inbound_flow_link_;
};

}  // namespace strix::transport
