#pragma once

#include "glm_rank_defaults.hpp"
#include "strix/device/hip_gate.hpp"

#include <algorithm>
#include <cstdio>

namespace strix::glm::rank_defaults {

// This receipt used one integrated 8060S per rank, the host transport, and two
// streams per 40 Gb/s cable. Unknown/mixed bindings decline in the pair vote.
inline bool measured_prefill_pair(const strix::device::Visible& device, const auto& streams, bool direct) {
    if (device.vendor != strix::device::Vendor::amd || !device.integrated || device.arch != "gfx1151" ||
        device.name != "AMD Radeon 8060S Graphics" || direct) return false;
    if (streams.size() != 4U || std::any_of(streams.begin(), streams.end(), [](const auto& stream) {
            return stream.xdomain.empty() || stream.link_gbps != 40.0;
        })) return false;
    const auto& first = streams.front().xdomain;
    const auto second = std::find_if(streams.begin(), streams.end(), [&](const auto& stream) {
        return stream.xdomain != first;
    });
    return second != streams.end() && std::count_if(streams.begin(), streams.end(), [&](const auto& stream) {
        return stream.xdomain == first;
    }) == 2 && std::count_if(second, streams.end(), [&](const auto& stream) {
        return stream.xdomain == second->xdomain;
    }) == 2;
}

inline bool measured_prefill_pair(const auto& streams, bool direct) {
    const auto* bound = strix::device::hip_bound();
    return bound && measured_prefill_pair(bound->visible, streams, direct);
}

// Pipelined (2-stage), replicated KV, full query chunks. The scratch storage (routed or full arena) is not an
// arithmetic choice; it is spelled in the execution key instead (PrefillWork::routed_scratch).
inline bool measured_prefill_layout(unsigned pipeline) {
    const auto& layout = strix::glm::pf::capacity_layout();
    return pipeline == 2 && !layout.sharded && layout.query_window == 0;
}
inline bool routed_prefill_scratch() { return strix::glm::pf::capacity_layout().routed; }

// Per-prompt receipt of the pf_wmma ladder decision (FAST band row / TURBO rows). When the prompt reached the ladder
// (voted pair, ordinary workload), the second line is the shared strix-ladder/v1 route of that prompt's regime: it
// names the deciding row (or why none applies).
inline void report_prefill_wmma(unsigned rank, const Request& r, const Effective& e, const EnvGet& env,
                                const PrefillWork& work, bool selected) {
    const char* profile = r.product.profile ? strix::profile::name(*r.product.profile) : "default";
    // The first unmet condition of select_prefill_wmma, in its order (the pair_ready vote folds device, link, layout
    // and the whole-prompt route of the entry point).
    const char* why = selected ? "matched 8060S USB4 TP2 ladder row"
        : !work.pair_ready ? "outside measured execution: pair device / link / layout / prompt route"
        : !ordinary_lossy_prefill(r, e) ? "outside measured execution: not the ordinary pair workload"
        : !measured_prefill_switches(e, env) ? "outside measured execution: prefill composition"
        : work.rows != kTurboPrefillRows ? "outside measured execution: chunk"
        : work.lanes != 1 ? "outside measured execution: lanes" : "no ladder row applies";
    std::fprintf(stderr, "rank %u glm-pf-wmma-x profile=%s prompt=%llu chunk=%u scratch=%s effective=%s (%s)\n",
                 rank, profile, static_cast<unsigned long long>(work.prompt), work.rows,
                 work.routed_scratch ? "routed" : "full", selected ? "wmma" : "exact-cores", why);
    if (!work.pair_ready || !ordinary_lossy_prefill(r, e)) return;
    const auto product = resolve_product(r, env, e.baseline, work);
    const auto* row = product ? product->find("pf_wmma") : nullptr;
    if (row != nullptr && !row->ladder.empty()) std::fprintf(stderr, "rank %u %s\n", rank, row->ladder.c_str());
}

}  // namespace strix::glm::rank_defaults
