#pragma once
// The negotiated pair link layout as data (backend-neutral, host-only): how many cables carry host streams, how many
// host streams each cable carries, their common raw link rate, and whether a GPU-direct device exchange is open.
// A pair speed receipt is valid only on the layout it was timed on (the host stream count per cable sets the exposed
// exchange time), so an engine admits a measured pair execution only on a layout listed with a receipt and spells
// layout_key() into its execution key. Examples: USB4 layout 1 (data0 + data1 host on both cables) through
// usb4-auto = "usb4-host-2x2x40"; layout 2bq (data0 host, data1 gpu_direct on both cables) through usb4-auto =
// "usb4-host-2x1x40", through usb4gpu = "usb4-host-2x1x40+direct".
#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

namespace strix::dist {

struct LinkLayout {
    unsigned cables = 0;          // distinct XDomains (one per cable) carrying host streams
    unsigned host_per_cable = 0;  // host streams on every cable; 0 when the cables differ
    unsigned link_gbps = 0;       // common raw link rate in Gb/s; 0 when the streams differ or a rate is unknown
    bool direct = false;          // a GPU-direct device exchange is open beside the host streams
    // Every field determined: a uniform layout a receipt can name.
    [[nodiscard]] bool known() const noexcept { return cables != 0 && host_per_cable != 0 && link_gbps != 0; }
    // "usb4-host-<cables>x<per cable>x<Gb/s>[+direct]"; "unknown" for a layout that is not uniform.
    [[nodiscard]] std::string key() const {
        if (!known()) return "unknown";
        std::string k = "usb4-host-" + std::to_string(cables) + "x" + std::to_string(host_per_cable) + "x" +
                        std::to_string(link_gbps);
        if (direct) k += "+direct";
        return k;
    }
};

// Classifies the negotiated host streams (any range of records with .xdomain and .link_gbps, e.g.
// Usb4Topology::streams). A stream without an XDomain or with a fractional rate makes the layout unknown.
template <class Streams>
[[nodiscard]] LinkLayout classify_link(const Streams& streams, bool direct) {
    LinkLayout layout;
    layout.direct = direct;
    std::vector<std::string> domains;
    std::vector<unsigned> per_domain;
    double rate = -1.0;
    for (const auto& stream : streams) {
        const double gbps = stream.link_gbps;
        if (std::string_view(stream.xdomain).empty() || !(gbps > 0.0) || gbps != std::floor(gbps)) return {0, 0, 0, direct};
        if (rate < 0.0) rate = gbps;
        else if (gbps != rate) return {0, 0, 0, direct};
        const auto at = std::find(domains.begin(), domains.end(), std::string(stream.xdomain));
        if (at == domains.end()) {
            domains.emplace_back(stream.xdomain);
            per_domain.push_back(1U);
        } else {
            ++per_domain[static_cast<std::size_t>(at - domains.begin())];
        }
    }
    if (domains.empty()) return layout;
    const bool even = std::all_of(per_domain.begin(), per_domain.end(), [&](unsigned n) { return n == per_domain.front(); });
    layout.cables = static_cast<unsigned>(domains.size());
    layout.host_per_cable = even ? per_domain.front() : 0U;
    layout.link_gbps = static_cast<unsigned>(rate);
    return layout;
}

// Whether `layout` is one of the measured layouts (keys as spelled by LinkLayout::key()).
template <class Keys>
[[nodiscard]] bool measured_layout(const LinkLayout& layout, const Keys& keys) {
    if (!layout.known()) return false;
    const auto key = layout.key();
    return std::any_of(std::begin(keys), std::end(keys), [&](std::string_view k) { return k == key; });
}

}  // namespace strix::dist
