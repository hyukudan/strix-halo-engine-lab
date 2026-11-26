#pragma once

#include "strix/device/topology.hpp"
#include "strix/profile/multi_device.hpp"

#include <algorithm>
#include <string>
#include <string_view>

namespace strix::hy4 {
namespace md = ::strix::profile::multi_device;

// The runtime binds the same rank layout on CLI and HTTP. A deployment receipt can only match
// a declared pair with a measured link. The topology ID excludes the local-rank selection.
inline md::Query pair_query(const device::Topology& topology, ::strix::profile::Name tier,
                            std::uint64_t prompt, std::uint64_t context, std::uint64_t decode_tokens, std::string_view transport,
                            std::string_view arch, std::string_view layout, const md::BandReceipt* receipt) {
    md::Query q;
    q.tier = tier;
    q.devices = topology.size();
    q.phase = decode_tokens != 0 ? md::Phase::decode : md::Phase::prefill;
    if (prompt != 0) q.prompt_rows = prompt;
    if (q.devices != 2 || topology.links.empty() || layout.empty() || arch.empty() ||
        !std::ranges::all_of(topology.nodes, [arch](const auto& node) {
            return !node.host.empty() && node.arch == arch && node.memory_gbytes_ps > 0.0;
        }) || topology.nodes[0].host == topology.nodes[1].host ||
        !std::ranges::all_of(topology.links, [](const auto& link) {
            return link.link_class == "usb4" && link.measured && link.rtt_measured;
        }) || transport != "usb4auto")
        return q;
    q.execution = {"hy4-flash", device::topology_id(topology), std::string(transport) + "/channel",
                   std::string(layout), std::string(arch), context, decode_tokens, receipt};
    return q;
}

inline md::Decision resolve_pair_placement(const model_description::ModelDescription& model, const md::Query& query) {
    return md::resolve(model, query);
}

inline std::string tensor_layout(const ::strix::profile::Resolution& tier, std::uint32_t prompt_rows) {
    std::string key = "hip/hy4-tp-slots8x2-bf16-v1/mtp-split-union-r8/rows" + std::to_string(prompt_rows);
    // A nested tier reuses the FAST receipt only when the effective execution is the same.
    for (const auto& entry : tier.entries) {
        if (entry.lever->omit_builtin_fingerprint && entry.value == entry.lever->builtin) continue;
        key += "/" + std::string(entry.lever->name) + "=" + entry.value;
    }
    key += "/" + tier.weights.identity();
    for (const auto& setting : tier.engine_overrides) key += "/" + setting;
    for (const auto* variant : tier.identity_variants()) key += "/" + std::string(variant->id) + "@" + std::to_string(variant->version);
    return key;
}
} // namespace strix::hy4
