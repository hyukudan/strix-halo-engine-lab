#pragma once
// Topology binding of the shared layer-chain planner (strix/pair/chain.hpp): the ranks, memory, rates and links of a
// strix-topology-v1 declaration (strix/device/topology.hpp) become chain devices and links, so every engine whose
// placement is a contiguous layer chain gets its plan for N ranks, unequal memory, unequal speed and any link class
// from the declaration alone. The engine supplies only its model table (chain::ModelShape), the relative per-layer
// cost shape and the whole-model prefill rate of its reference node (receipt or flag); no lab constant lives here.
// Ranks sit on a line in rank order (rank i links to rank i+1); a topology that is not a line is refused by name.
// Home: shared (host shared). Bound by: qwen35 dense pair app (apps/hip_qwen35_pair_main.hip).
#include <algorithm>
#include <cstdint>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "strix/device/process_topology.hpp"
#include "strix/device/topology.hpp"
#include "strix/pair/chain.hpp"

namespace strix::pair::chain {
// One rank as the planner sees it (from the declaration, or measured once and cached).
struct RankDecl {
    std::string name;
    double capacity_bytes = 0.0;       // usable memory for weights + state (0: undeclared)
    double prefill_rows_per_s = 0.0;   // whole-model prefill rate of this rank alone (0: never runs layers)
    double read_bytes_per_ms = 0.0;    // decode stream rate (bytes/ms); 0: no decode prediction
    double block_fixed_ms = 0.0;
};
struct HopDecl { double bytes_per_ms = 0.0, latency_ms = 0.0; };  // rank i <-> rank i+1, both directions

// Ranks of the declaration. `reference_rows_per_s` (> 0) is the whole-model prefill rate of the fastest declared node
// (by fp16_tflops); every other node is scaled by its fp16_tflops over that maximum, so a slow rank gets fewer layers.
// A node without a compute declaration (or no node with one) runs at the reference rate: equal nodes stay equal.
inline std::vector<RankDecl> ranks(const device::Topology& topology, double reference_rows_per_s) {
    double max_tflops = 0.0;
    for (const auto& n : topology.nodes) max_tflops = std::max(max_tflops, n.fp16_tflops);
    std::vector<RankDecl> out;
    out.reserve(topology.nodes.size());
    for (const auto& n : topology.nodes) {
        RankDecl r;
        r.name = n.id;
        r.capacity_bytes = double(n.memory_bytes);
        r.prefill_rows_per_s = max_tflops > 0.0 && n.fp16_tflops > 0.0 ? reference_rows_per_s * n.fp16_tflops / max_tflops
                                                                        : reference_rows_per_s;
        r.read_bytes_per_ms = n.memory_gbytes_ps * 1e6;
        out.push_back(std::move(r));
    }
    return out;
}
// True when every node declares its memory: the planner then admits each stage (weights + state) against it.
inline bool memory_declared(const device::Topology& topology) {
    return std::all_of(topology.nodes.begin(), topology.nodes.end(), [](const auto& n) { return n.memory_bytes > 0U; });
}
// Line hops rank r <-> r + 1: the declared rate of the link(s) between them (measured over class nominal, parallel
// links summed: Topology::direct_gbytes_ps) and the declared round trip as latency.
inline std::vector<HopDecl> line_hops(const device::Topology& topology) {
    std::vector<HopDecl> hops;
    for (std::uint32_t r = 0; r + 1U < topology.size(); ++r) {
        const auto ls = topology.links_between(r, r + 1U);
        if (ls.empty())
            throw std::invalid_argument("ranks must be declared on a line: no link between " + topology.nodes[r].id + " and " +
                                        topology.nodes[r + 1U].id);
        hops.push_back({topology.direct_gbytes_ps(r, r + 1U) * 1e6, ls.front()->rtt_us / 1000.0});
    }
    return hops;
}
// Per-rank multiplier of per-layer costs measured on the fastest declared node (by fp16_tflops): max over the rank's
// own rate. A node without a compute declaration, or a topology where none declares one, gets exactly 1.0, so equal or
// undeclared nodes leave every cost bit unchanged. Used by pipeline routes that plan from measured cost files.
inline std::vector<double> stage_cost_scale(const device::Topology& topology) {
    double max_tflops = 0.0;
    for (const auto& n : topology.nodes) max_tflops = std::max(max_tflops, n.fp16_tflops);
    std::vector<double> scale(topology.nodes.size(), 1.0);
    for (std::size_t r = 0; r < topology.nodes.size(); ++r)
        if (max_tflops > 0.0 && topology.nodes[r].fp16_tflops > 0.0) scale[r] = max_tflops / topology.nodes[r].fp16_tflops;
    return scale;
}
// The same for the process topology (device/process_topology.hpp) when a caller resolved it; empty (= equal ranks)
// before any resolution, so a route that never resolved a topology keeps the equal-rank balance.
inline std::vector<double> process_stage_cost_scale() {
    const auto* topology = device::resolved_process_topology();
    return topology ? stage_cost_scale(*topology) : std::vector<double>{};
}
// Port of the line link (lower, lower + 1) for a transport that needs one: base + lower (lower rank listens).
inline std::uint32_t line_port(std::uint32_t base, std::uint32_t lower) { return base + lower; }

inline std::vector<Device> devices(std::span<const RankDecl> ranks, std::uint32_t layers, std::span<const double> shape) {
    std::vector<Device> out;
    out.reserve(ranks.size());
    for (const auto& r : ranks) {
        Device d;
        d.name = r.name;
        if (r.prefill_rows_per_s > 0.0) d.layer_ms_per_row = scaled_costs(r.prefill_rows_per_s, layers, shape);
        d.block_fixed_ms = r.block_fixed_ms;
        d.capacity_bytes = r.capacity_bytes;
        d.read_bytes_per_ms = r.read_bytes_per_ms;
        out.push_back(std::move(d));
    }
    return out;
}
inline std::vector<Link> links(std::span<const HopDecl> hops) {
    std::vector<Link> out;
    for (const auto& h : hops) out.push_back({h.bytes_per_ms, h.bytes_per_ms, h.latency_ms});
    return out;
}

// Explicit override, one value for CLI and server: "auto" (default), "single" (one rank prefills and decodes),
// "cut:A[,B...]" (exactly that many interior cuts; 0 leaves one to the planner).
inline void apply_override(std::string_view text, Request& request) {
    if (text.empty() || text == "auto") { request.split = Split::auto_plan; return; }
    if (text == "single") { request.split = Split::device_only; return; }
    if (!text.starts_with("cut:")) throw std::invalid_argument("placement override must be auto, single or cut:A[,B...]");
    request.split = Split::layer_cut;
    request.cuts.clear();
    std::uint32_t value = 0;
    bool digits = false;
    for (std::size_t i = 4; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == ',') {
            if (!digits) throw std::invalid_argument("placement override cut list is empty or malformed");
            request.cuts.push_back(value);
            value = 0; digits = false;
        } else if (text[i] >= '0' && text[i] <= '9') {
            const auto digit = static_cast<std::uint32_t>(text[i] - '0');
            // Refuse instead of wrapping: "cut:4294967299" must not become cut 3.
            if (value > (UINT32_MAX - digit) / 10U) throw std::invalid_argument("placement override cut value out of range");
            value = value * 10U + digit; digits = true;
        } else {
            throw std::invalid_argument("placement override cut list is malformed");
        }
    }
}

// One effective-config line (CLI = serve): the decision and its reason.
inline std::string describe(const Plan& p, const std::vector<Device>& ranks, std::uint64_t rows) {
    std::ostringstream o;
    o << "placement " << (p.valid ? "valid" : "refused") << " stages=" << p.stages() << " ranks=";
    for (std::uint32_t s = 0; s < p.stages(); ++s) o << (s ? "," : "") << ranks[p.first + s].name;
    o << " cuts=";
    for (std::size_t i = 0; i < p.cuts.size(); ++i) o << (i ? "," : "") << p.cuts[i];
    o << " block=" << p.block << " decoder=" << (p.decoder < ranks.size() ? ranks[p.decoder].name : std::string("?"))
      << " prefill_ms=" << p.wall_ms << " prefill_rows_per_s=" << p.rows_per_s(rows)
      << " decode_ms_per_token=" << p.decode_ms_per_token << " reason=\"" << p.reason << "\"";
    return o.str();
}
} // namespace strix::pair::chain
