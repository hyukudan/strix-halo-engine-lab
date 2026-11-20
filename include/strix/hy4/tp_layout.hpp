#pragma once
// Hy4-flash binding of exact tensor parallelism over two ranks: where each rank writes its share inside the one-node
// device buffers, so that the shared pair gather (strix/pair/gather_link.hpp: slot[rank] = own share, slot[1 - rank]
// receives the peer's) leaves the one-node layout in place with no join pass.
//   half(n, rank)       rows / heads / vocabulary columns [rank n/2, (rank + 1) n/2): the two slots of a buffer are its
//                       two halves, so the gathered buffer IS the one-node buffer (o by heads, o_proj / FFN / head rows).
//   moe(rank, ...)      one gather per MoE layer of [8 routed slot rows | 768 shared-expert activations] per rank. Rank 0
//                       writes [act0 | raw0], rank 1 [raw1 | act1]: after the swap the 16 slot rows raw0 raw1 are
//                       contiguous (the combine reads all 16 in slot order, as one node) and the activation halves sit
//                       act_pitch elements apart (one 2-row copy into the one-node activation buffer).
// Host only. Home: specific (Hy4 head / slot / shared-expert layout); the gather, its route and the planner are shared.
// Host test: tests/hy4_tp_layout_tests.cpp.
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "strix/pair/split_plan.hpp"

namespace strix::hy4::tp {

struct Range {
    std::size_t first = 0, count = 0;
};

// Rank `rank` of 2 owns [rank n/2, (rank + 1) n/2) of n units (heads, rows, columns).
[[nodiscard]] inline Range half(std::size_t n, std::uint32_t rank) {
    if (rank > 1U) throw std::invalid_argument("hy4 tp: two ranks");
    if (n == 0U || n % 2U != 0U) throw std::invalid_argument("hy4 tp: a split dimension must be even");
    return {std::size_t{rank} * (n / 2U), n / 2U};
}

struct MoeLayout {
    std::size_t slot = 0;       // elements per rank slot (gather message = slot x 4 bytes)
    std::size_t raw_all = 0;    // first element of the 16 joined slot rows [top_k][latent]
    std::size_t raw_mine = 0;   // where this rank writes its top_k / 2 slot rows
    std::size_t act_mine = 0;   // where this rank writes its shared_ff / 2 activations
    std::size_t act_first = 0;  // activation half 0 (rank 0's) after the gather
    std::size_t act_pitch = 0;  // elements from activation half 0 to half 1
    std::size_t slot_first = 0; // this rank's routed slot index (ids + slot_first)
    std::size_t slots = 0;      // routed slots per rank
};

[[nodiscard]] inline MoeLayout moe(std::uint32_t rank, std::size_t top_k, std::size_t latent, std::size_t shared_ff) {
    const Range k = half(top_k, rank);
    const Range a = half(shared_ff, rank);
    if (latent == 0U) throw std::invalid_argument("hy4 tp: latent width");
    MoeLayout m;
    m.slots = k.count;
    m.slot_first = k.first;
    m.slot = a.count + k.count * latent;
    m.raw_all = a.count;
    m.raw_mine = m.raw_all + k.first * latent;
    m.act_first = 0U;
    m.act_pitch = a.count + top_k * latent;
    m.act_mine = rank == 0U ? 0U : m.act_pitch;
    return m;
}

// Gathers per decode step and the bytes one rank sends per step (o, o_proj rows, FFN, MoE, head).
struct StepCount {
    std::uint32_t gathers = 0;
    std::uint64_t bytes = 0;
};
struct Dims {
    std::uint32_t layers = 0, dense_layers = 0, hidden = 0, vocab = 0, v_head = 0, dense_ff = 0, top_k = 0, latent = 0,
                  shared_ff = 0;
    std::uint64_t heads_sum = 0;  // sum over layers of the attention heads
};
// Dims of a model config (Config of src/hy4/ref: layers, dense[], heads[], hidden, vocab, v_head, dense_ff, top_k,
// latent, shared_ff).
template <class C>
[[nodiscard]] Dims dims_of(const C& c) {
    Dims d;
    d.layers = c.layers;
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        d.dense_layers += c.dense[l] != 0U ? 1U : 0U;
        d.heads_sum += c.heads[l];
    }
    d.hidden = c.hidden;
    d.vocab = c.vocab;
    d.v_head = c.v_head;
    d.dense_ff = c.dense_ff;
    d.top_k = c.top_k;
    d.latent = c.latent;
    d.shared_ff = c.shared_ff;
    return d;
}

[[nodiscard]] inline StepCount step_count(const Dims& d);

// One decode pass as shared split stages (strix/pair/split_plan.hpp, priced by pair::split::plan): per layer the
// replicated attention inputs, the head-split attention (join o), o_proj rows (join), then either the dense FFN rows
// (two joins) or the replicated router / latent down, the shared gate/up rows with the routed slots (one join) and the
// replicated latent up / shared down; last the vocabulary rows (join logits). bytes_of(name) = bytes of a loaded tensor
// (0 when absent). The joins are the gathers step() runs: their count equals step_count(dims_of(c)).gathers.
template <class C, class BytesOf>
[[nodiscard]] std::vector<pair::split::Stage> decode_stages(const C& c, BytesOf&& bytes_of) {
    namespace ps = pair::split;
    std::vector<ps::Stage> out;
    const auto blk = [](std::uint32_t l, const char* n) { return "blk." + std::to_string(l) + "." + n; };
    const auto sum = [&](std::uint32_t l, std::initializer_list<const char*> names) {
        double b = 0.0;
        for (const char* n : names) b += static_cast<double>(bytes_of(blk(l, n)));
        return b;
    };
    const auto add = [&](std::string_view name, std::uint32_t l, ps::Axis axis, ps::Join join, std::uint32_t units, double bytes,
                         double join_row) -> ps::Stage& {
        ps::Stage s;
        s.name = name;
        s.layer = l;
        s.axis = axis;
        s.join = join;
        s.units = units;
        s.bytes = bytes;
        s.join_bytes_per_row = join_row;
        out.push_back(s);
        return out.back();
    };
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::uint32_t H = c.heads[l];
        add("attn inputs", l, ps::Axis::replicated, ps::Join::none, 1U, sum(l, {"attn_q_a.weight", "attn_kv_a_mqa.weight"}), 0.0);
        add("attn heads", l, ps::Axis::heads, ps::Join::gather, H,
            sum(l, {"attn_gate.weight", "attn_q_b.weight", "attn_k_b.weight", "attn_v_b.weight"}), 4.0 * H * c.v_head);
        add("o_proj rows", l, ps::Axis::columns, ps::Join::gather, c.hidden / 64U, sum(l, {"attn_output.weight"}), 4.0 * c.hidden);
        if (c.dense[l] != 0U) {
            add("ffn gate/up rows", l, ps::Axis::columns, ps::Join::gather, c.dense_ff / 64U, sum(l, {"ffn_gate.weight", "ffn_up.weight"}),
                4.0 * c.dense_ff);
            add("ffn down rows", l, ps::Axis::columns, ps::Join::gather, c.hidden / 64U, sum(l, {"ffn_down.weight"}), 4.0 * c.hidden);
        } else {
            add("router + latent down", l, ps::Axis::replicated, ps::Join::none, 1U,
                sum(l, {"ffn_gate_inp.weight", "ffn_routed_down.weight"}), 0.0);
            add("shared gate/up rows", l, ps::Axis::columns, ps::Join::none, c.shared_ff / 64U,
                sum(l, {"ffn_gate_shexp.weight", "ffn_up_shexp.weight"}), 0.0);
            auto& e = add("routed slots", l, ps::Axis::slots, ps::Join::slots, c.top_k,
                          sum(l, {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) / c.experts, 4.0 * c.latent);
            e.pool = c.experts;
            add("latent up + shared down", l, ps::Axis::replicated, ps::Join::none, 1U,
                sum(l, {"ffn_routed_up.weight", "ffn_down_shexp.weight"}), 0.0);
        }
    }
    add("head rows", c.layers, ps::Axis::vocab, ps::Join::gather, c.vocab / 64U, static_cast<double>(bytes_of(std::string("output.weight"))),
        4.0 * c.vocab);
    return out;
}

[[nodiscard]] inline StepCount step_count(const Dims& d) {
    if (d.dense_layers > d.layers) throw std::invalid_argument("hy4 tp: dense layers");
    StepCount s;
    const std::uint32_t moe_layers = d.layers - d.dense_layers;
    s.gathers = 2U * d.layers + 2U * d.dense_layers + moe_layers + 1U;
    s.bytes = 4U * (d.heads_sum / 2U * d.v_head + std::uint64_t{d.layers} * (d.hidden / 2U) +
                    std::uint64_t{d.dense_layers} * (d.dense_ff / 2U + d.hidden / 2U) +
                    std::uint64_t{moe_layers} * moe(0U, d.top_k, d.latent, d.shared_ff).slot + d.vocab / 2U);
    return s;
}

}  // namespace strix::hy4::tp
