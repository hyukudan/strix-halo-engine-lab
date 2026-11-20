#pragma once
// Hy4-flash (hy_v4_flash) binding of the shared placement planners. The model supplies only its table: the role of each
// tensor naming map component (src/weight/adapters/hy4.cpp), its state layout and its tensor-parallel exchange points. Every
// decision (cuts, ranks, admission, step estimate) is the shared planners': layer chain (strix/pair/chain.hpp) and exact
// tensor parallelism (strix/pair/column_split.hpp), fed through strix/pair/placement_inputs.hpp.
// Exactness contract of the tensor-parallel split (ONE oracle for 1 and N ranks): attention is split by heads and o_proj by
// the matching K rows, experts by 256-element slices of their inner dimension (gate/up columns, down K), the shared expert and
// the dense layer-0 FFN by columns / K, the head by vocabulary rows; each sublayer ends in ONE exchange of partial sums that
// every rank adds in rank order, and the one-rank engine adds the same partials in the same order.
// Home: specific (model table: Hy4 MLA + LatentMoE + Engram layout, src/weight/adapters/hy4.cpp components); placement is
// shared (include/strix/pair).
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "strix/pair/chain.hpp"
#include "strix/pair/column_split.hpp"
#include "strix/pair/placement_inputs.hpp"

namespace strix::hy4::placement {
namespace place = strix::pair::place;
namespace chain = strix::pair::chain;
namespace tp = strix::pair::tp;
using place::Role;

// Hyper-parameters the plan needs (hy_v4_flash config; the loader resolves the same symbols from the pack metadata).
struct Facts {
    std::uint32_t layers = 42;          // trunk blocks (block 42 is the next-token block)
    std::uint32_t experts = 512, top_k = 16;
    std::uint32_t hidden = 3072, streams = 4, latent = 1536, vocab = 120832;
    std::uint32_t kv_row = 576;         // MLA latent 512 + rope 64, bf16 per cached token
    std::uint32_t index_key = 128;      // sparse-attention indexer key, bf16 per cached token
    std::uint32_t window = 512;         // sliding-attention layers keep this many latent rows
    std::uint32_t lookup_heads = 16, lookup_dim = 96;  // Engram rows gathered per token and Engram layer (F8: 1 byte each)
    std::uint32_t slice_units = 6;      // expert inner dim 1536 in 256-element K blocks: the indivisible tensor-parallel unit
};

// Role of a component (tensor naming map key) under placement.
[[nodiscard]] inline Role role_of(std::string_view component, std::int64_t /*layer*/) {
    if (component.starts_with("experts.")) return Role::experts;
    if (component == "attention.q_b" || component == "attention.k_b" || component == "attention.v_b" || component == "attention.gate" ||
        component == "attention.o" || component == "attention.sinks" || component == "shared_expert" || component == "dense_ffn")
        return Role::split;
    if (component == "engram.table") return Role::lookup;
    if (component == "embedding") return Role::embed;
    if (component == "head") return Role::head;
    return Role::replicated;  // q_a, kv_a, norms, hyper connections, router, latent, indexer, Engram projections, head mix/norm
}
[[nodiscard]] inline place::Inputs inputs(std::span<const place::Record> records, const Facts& f) {
    return place::inputs(records, f.layers, role_of);
}
// Layers that own a component (from the table itself, not a hard-coded list).
[[nodiscard]] inline std::vector<std::uint32_t> layers_with(std::span<const place::Record> records, std::string_view component,
                                                            const Facts& f) {
    std::vector<std::uint32_t> out;
    for (const auto& r : records)
        if (r.component == component && r.layer >= 0 && r.layer < static_cast<std::int64_t>(f.layers))
            out.push_back(static_cast<std::uint32_t>(r.layer));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
// State: sparse-attention layers (those with an indexer) cache latent + indexer key per token; sliding layers keep a fixed
// window of latent rows. Seam: the residual streams (bf16) of a row.
[[nodiscard]] inline place::StateLayout state_layout(std::span<const place::Record> records, const Facts& f) {
    place::StateLayout s;
    s.seam_bytes_per_row = double(f.streams) * f.hidden * 2.0;
    s.per_row.assign(f.layers, 0.0);
    s.fixed.assign(f.layers, double(f.window) * f.kv_row * 2.0);
    for (const auto l : layers_with(records, "indexer", f)) {
        s.per_row[l] = double(f.kv_row) * 2.0 + double(f.index_key) * 2.0;
        s.fixed[l] = 0.0;
    }
    return s;
}
// Engram table: one global tensor whose rows are per Engram layer; placed as equal slices with the Engram layers.
[[nodiscard]] inline std::vector<double> lookup_slices(const place::Inputs& in, std::span<const place::Record> records, const Facts& f) {
    const auto owners = layers_with(records, "engram.proj", f);
    std::vector<double> s(f.layers, 0.0);
    if (owners.empty()) return s;
    for (const auto l : owners) s[l] = place::at(in.global, Role::lookup) / static_cast<double>(owners.size());
    return s;
}
[[nodiscard]] inline std::vector<double> lookup_reads(std::span<const place::Record> records, const Facts& f) {
    std::vector<double> r(f.layers, 0.0);
    for (const auto l : layers_with(records, "engram.proj", f)) r[l] = double(f.lookup_heads) * f.lookup_dim;
    return r;
}
// Exchange points of one tensor-parallel decode step over `ranks` ranks (bytes each rank sends): per layer the attention
// partial (hidden fp32) and the FFN partial (layer 0 dense: hidden fp32; MoE: latent partial + shared-expert partial, one
// message); per token the Engram rows a rank owns for its peers and its vocabulary share of the logits.
[[nodiscard]] inline std::vector<tp::Trip> tp_trips(std::span<const place::Record> records, const Facts& f, std::uint32_t ranks) {
    if (ranks == 0U) throw std::invalid_argument("hy4 placement: ranks must be > 0");
    std::vector<tp::Trip> trips;
    const auto moe = layers_with(records, "router", f);
    for (std::uint32_t l = 0; l < f.layers; ++l) {
        trips.push_back({double(f.hidden) * 4.0});
        const bool routed = std::binary_search(moe.begin(), moe.end(), l);
        trips.push_back({routed ? double(f.latent + f.hidden) * 4.0 : double(f.hidden) * 4.0});
    }
    const auto engram = layers_with(records, "engram.proj", f);
    trips.push_back({double(engram.size()) * f.lookup_heads * f.lookup_dim / ranks});
    trips.push_back({double(f.vocab) * 4.0 / ranks});
    return trips;
}

// One decode plan row (PREDICTION: bytes over the declared read rate + the per-token small-kernel time + trips).
struct PlanRow {
    std::string name;
    std::uint32_t ranks = 0;
    bool valid = false;
    double decode_ms = 0.0;
    double resident_bytes_per_rank = 0.0;  // largest rank
    std::string reason;
};
struct Node {
    double read_bytes_per_ms = 0.0;      // measured GEMV read rate
    double capacity_bytes = 0.0;         // device-mapped budget (GTT wall)
    double prefill_rows_per_s = 0.0;     // whole-model prefill rate of one node (chain cost seed)
};
struct Budget {
    double small_ms_per_token = 0.0;     // non-GEMV kernels per token (norms, mixes, routing), every rank
    double context_rows = 0.0;           // KV rows reserved
    double workspace_bytes = 0.0;        // per rank
    bool drafting = false;               // next-token block resident and split with the trunk
};
// Plans for `n` equal nodes on a line: one node (Engram table file-backed, not resident), layer chain over n (Engram slices
// resident with their layers), exact tensor parallelism over the best k <= n ranks (Engram slices split per rank).
[[nodiscard]] inline std::vector<PlanRow> decode_plans(std::span<const place::Record> records, const Facts& f, const Node& node,
                                                       std::uint32_t n, const tp::TripCost& link, const Budget& b) {
    if (n == 0U || !(node.read_bytes_per_ms > 0.0)) throw std::invalid_argument("hy4 placement: nodes and read rate must be > 0");
    const auto in = inputs(records, f);
    const auto state = state_layout(records, f);
    const auto active = place::expert_union_fraction(f.top_k, f.experts, 1U);
    double kv = 0.0;
    for (std::uint32_t l = 0; l < f.layers; ++l) kv += state.per_row[l] * b.context_rows + state.fixed[l];
    const double beyond = b.drafting ? place::sum(in.beyond) : 0.0;
    const auto slices = lookup_slices(in, records, f);
    const auto reads = lookup_reads(records, f);
    std::vector<PlanRow> rows;
    // Layer chain (1 rank = the one-node plan). Decode reads come from the chain's own prediction.
    for (const std::uint32_t k : {1U, n}) {
        if (k == 1U && !rows.empty()) continue;
        if (k > 1U && n == 1U) break;
        // One node: the Engram table is file-backed (rows prefetched by token id), so no slice is resident.
        const auto shape = place::chain_shape(in, state, active, k == 1U ? std::span<const double>{} : std::span<const double>(slices), reads);
        std::vector<chain::Device> devices(k);
        std::vector<double> cost_shape(shape.read_bytes_per_token.begin(), shape.read_bytes_per_token.end());
        for (std::uint32_t r = 0; r < k; ++r) {
            devices[r].name = "rank" + std::to_string(r);
            devices[r].layer_ms_per_row = chain::scaled_costs(node.prefill_rows_per_s > 0.0 ? node.prefill_rows_per_s : 1.0, f.layers, cost_shape);
            devices[r].capacity_bytes = node.capacity_bytes - b.workspace_bytes - (r + 1U == k ? beyond : 0.0);
            devices[r].read_bytes_per_ms = node.read_bytes_per_ms;
        }
        std::vector<chain::Link> links(k > 1U ? k - 1U : 0U, chain::Link{link.bytes_per_ms, link.bytes_per_ms, link.latency_ms});
        chain::Request q;
        q.split = k == 1U ? chain::Split::device_only : chain::Split::layer_cut;
        q.cuts.assign(k - 1U, 0U);
        q.rows = 4096U;      // a 4K prompt in 512-row blocks: the flow shop balances the cut by the cost shape
        q.blocks = {512U};
        q.decoders = {k - 1U};
        q.state_rows = static_cast<std::uint64_t>(b.context_rows);
        q.layer_owned_state = k > 1U;
        const auto choice = chain::choose(devices, links, shape, q);
        PlanRow row;
        row.name = k == 1U ? "one-node (Engram file-backed)" : "layer-chain";
        row.ranks = k;
        row.valid = choice.best.valid;
        row.reason = choice.best.reason;
        // chain decode = whole read on one rate; the stages run in sequence, so add the per-token hops and small kernels.
        row.decode_ms = choice.best.decode_ms_per_token + b.small_ms_per_token + 2.0 * (k - 1U) * link.latency_ms;
        double worst = 0.0;
        for (std::uint32_t s = 0; s < choice.best.stages(); ++s) {
            double w = 0.0;
            for (auto l = choice.best.cuts[s]; l < choice.best.cuts[s + 1U]; ++l) w += shape.weight_bytes[l];
            if (s == 0U) w += shape.embed_bytes;
            if (s + 1U == choice.best.stages()) w += shape.head_bytes + beyond;
            worst = std::max(worst, w);
        }
        row.resident_bytes_per_rank = worst + kv / static_cast<double>(k);
        rows.push_back(row);
    }
    if (n > 1U) {
        const auto trips = tp_trips(records, f, 2U);
        double engram_rows = 0.0;
        for (const auto r : reads) engram_rows += r;
        const auto step = place::tp_step(in, active, trips, engram_rows, b.small_ms_per_token);
        const std::vector<double> rates(n, node.read_bytes_per_ms);
        const auto g = tp::choose_ranks(step, rates, f.slice_units, link);
        const auto k = static_cast<std::uint32_t>(g.ranks.size());
        const auto shares = tp::split_units(f.slice_units, std::vector<double>(k, node.read_bytes_per_ms));
        double largest = 0.0;
        for (const auto& s : shares) largest = std::max(largest, double(s.size()) / f.slice_units);
        PlanRow row;
        row.name = "tensor-parallel";
        row.ranks = k;
        row.decode_ms = g.est.group_ms;
        row.resident_bytes_per_rank = place::tp_resident_bytes(in, largest, 1.0 / k, false) + beyond * largest + kv;
        row.valid = row.resident_bytes_per_rank + b.workspace_bytes <= node.capacity_bytes;
        row.reason = row.valid ? "fits" : "weights and state do not fit a rank";
        rows.push_back(row);
    }
    return rows;
}
}  // namespace strix::hy4::placement
