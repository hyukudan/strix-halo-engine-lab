#pragma once

#include <array>
#include <cstdint>
#include <string_view>

// Link classes of the device declaration (attach.link, links.<name>.class). A class is the one-line way to state what
// connects two units; it carries NOMINAL useful rates, used only until a measured rate is declared, and a wider
// uncertainty than a measurement. Rates are useful payload GB/s (decimal), not the signalling rate.
namespace strix::device {

struct LinkClass {
    std::string_view name;
    double gbytes_ps = 0.0;    // nominal useful payload rate, one direction
    double rtt_us = 0.0;       // nominal round trip of one small message
    double uncertainty = 0.0;  // relative half-width of the nominal rate
};

// 1gbe is the measured useful rate of the cluster's 1 GbE path; 2.5/10/25/100 GbE are estimates; usb4 is the measured
// pair cable; oculink/pcie are per-direction PCIe payload estimates. Each is replaced by attach.gbps_measured (device
// declaration) or links.<name>.gbps_measured (strix-topology-v1) when probed. This is the ONE class table: the
// `interconnect = ...` classes of profiles.hpp are a view of it.
inline constexpr std::array<LinkClass, 13> link_classes{{
    {"1gbe", 0.118, 200.0, 0.25},   {"2.5gbe", 0.29, 150.0, 0.30},  {"5gbe", 0.58, 120.0, 0.30},
    {"10gbe", 1.1, 100.0, 0.30},    {"25gbe", 2.8, 60.0, 0.35},     {"100gbe", 11.5, 20.0, 0.35},
    {"usb4", 4.58, 30.0, 0.30},     {"usb4-gen2", 1.1, 40.0, 0.50}, {"pcie3x4", 3.2, 5.0, 0.30},
    {"pcie4x4", 7.0, 5.0, 0.30},    {"oculink4x4", 7.0, 5.0, 0.30}, {"pcie4x16", 25.0, 5.0, 0.30},
    {"pcie5x16", 50.0, 5.0, 0.30},
}};

[[nodiscard]] constexpr const LinkClass* find_link_class(std::string_view name) noexcept {
    for (const auto& c : link_classes)
        if (c.name == name) return &c;
    return nullptr;
}
[[nodiscard]] constexpr bool known_link_class(std::string_view name) noexcept { return find_link_class(name) != nullptr; }

// Decision tables that were tuned against a protocol-level rate on a given class keep that rate as a named
// CALIBRATION row here (one place) instead of a private constant in the consumer. Every consumer applies one rule
// (link_rate_mbytes_ps): a rate measured and declared for the link > the consumer's calibration for the link's class >
// the class nominal above > 0 (unknown). With no declaration, or with the lab's two-node usb4 declaration (nominal),
// every consumer reads exactly the number it read before this table existed; another class or a declared measurement
// is a declaration, not a code change. A calibration row changes only with a re-measured decision table (each
// threshold flip is its own measured change).
enum class LinkUse {
    relay_host_path,   // bulk relay over the host transport, >= 16 MB pieces (handoff_policy, pair/state_owners)
    layer_cut_handoff, // STRIX_HANDOFF_LINK layer-cut handoff rate (pair/layercut)
    host_egress,       // prefill host egress shared by every destination (pair/state_owners)
};
struct LinkCalibration {
    LinkUse use;
    std::string_view link_class;
    double mbytes_ps = 0.0;  // MB/s (decimal), the consumer's own unit, kept bit-exact
    std::string_view why;
};
inline constexpr std::array<LinkCalibration, 6> link_calibrations{{
    {LinkUse::relay_host_path, "usb4", 4400.0, "measured 2026-10-31 host path, 4 rings, 35-36 Gb/s at >= 16 MB pieces"},
    {LinkUse::layer_cut_handoff, "1gbe", 118.0, "layer-cut handoff table (measured 1 GbE useful rate)"},
    {LinkUse::layer_cut_handoff, "2.5gbe", 290.0, "layer-cut handoff table (nominal)"},
    {LinkUse::layer_cut_handoff, "10gbe", 1180.0, "layer-cut handoff table (nominal)"},
    {LinkUse::layer_cut_handoff, "usb4", 4600.0, "layer-cut handoff table (pair cable)"},
    {LinkUse::host_egress, "1gbe", 125.0, "prefill host egress line rate (1 GbE payload ~117.6 MB/s)"},
}};
[[nodiscard]] constexpr const LinkCalibration* find_link_calibration(LinkUse use, std::string_view name) noexcept {
    for (const auto& c : link_calibrations)
        if (c.use == use && c.link_class == name) return &c;
    return nullptr;
}
// MB/s for `use` over a link of class `name`; `measured_gbytes_ps` > 0 is a declared measurement and wins.
[[nodiscard]] constexpr double link_rate_mbytes_ps(LinkUse use, std::string_view name, double measured_gbytes_ps = 0.0) noexcept {
    if (measured_gbytes_ps > 0.0) return measured_gbytes_ps * 1000.0;
    if (const auto* c = find_link_calibration(use, name)) return c->mbytes_ps;
    if (const auto* c = find_link_class(name)) return c->gbytes_ps * 1000.0;
    return 0.0;
}
// The class a pair consumer assumes when nothing is declared: today's lab pair cable / host port.
inline constexpr std::string_view legacy_pair_link_class = "usb4";
inline constexpr std::string_view legacy_host_link_class = "1gbe";

} // namespace strix::device
