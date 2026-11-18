#pragma once
// Stage balance of a layer pipeline over N stages on N ranks, shared by every engine's pipeline route: stage s
// runs layers [cuts[s-1], cuts[s]) on its own rank with that rank's per-layer costs, the last stage also runs the
// tail (head, final norm). The best cut set minimises the slowest stage; ties keep the lexicographically smallest
// cut set. Costs are per-row (or per-chunk) times measured by the engine, scaled per rank by the declared speed.
// balance_split is the two-stage, equal-rank form (one cost vector for both stages): it calls balance_chain, which
// enumerates the same candidates in the same order with the same additions, so its cut is bit-identical.
// Home: shared (host shared). Bound by: Qwen MoE pair (qwen_prefill_balance.hpp choose_cut, rank speeds from the
// topology declaration through pair/chain_topology.hpp stage_cost_scale).
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::pair {
struct StageSplit { std::uint32_t cut = 0; double a = 0, b = 0; bool half = false; };
// N-stage result: cuts[s] is the end layer of stage s (s < N - 1); half[s] puts the front sub-block (attention) of
// layer cuts[s] on stage s too; work[s] is the stage's cost (the last one with the tail).
struct ChainSplit { std::vector<std::uint32_t> cuts; std::vector<bool> half; std::vector<double> work; };
// Candidate bound of one search (lexicographic enumeration with exact prefix pruning); exhaustion throws.
inline constexpr std::uint64_t max_balance_candidates = 50'000'000U;

// costs[s]: per-layer costs of the rank running stage s (all the same length L); front[s]: per-layer front sub-block
// costs on that rank (empty span: no half cuts for any stage). A boundary is a whole cut c (1 <= c < L, legal(c)) or a
// half cut at c (legal_half(c)); boundaries of consecutive stages lie in increasing layers (every stage owns at least
// one layer's back part). Stage work = (tail on the last stage) + its layers in order, + front of its end layer when
// that end is a half cut, - front of its begin layer when that begin is a half cut.
template<class Legal, class LegalHalf> ChainSplit balance_chain(std::span<const std::span<const double>> costs,
    std::span<const std::span<const double>> front, double tail, Legal legal, LegalHalf legal_half) {
    const auto stages = static_cast<std::uint32_t>(costs.size());
    if (stages == 0U) throw std::invalid_argument("pair pipeline needs at least one stage");
    const auto layers = static_cast<std::uint32_t>(costs[0].size());
    for (const auto& c : costs)
        if (c.size() != layers) throw std::invalid_argument("pair pipeline stage costs differ in layer count");
    const bool halves = !front.empty();
    if (halves) {
        if (front.size() != stages) throw std::invalid_argument("pair pipeline needs one front cost vector per stage");
        for (const auto& f : front)
            if (f.size() != layers) throw std::invalid_argument("pair pipeline front costs differ in layer count");
    }
    // Boundary position p = 2 * cut + half; begin 0, end 2 * layers.
    const auto stage_work = [&](std::uint32_t s, std::uint32_t begin, std::uint32_t end) {
        double work = s + 1U == stages ? tail : 0.0;
        const auto first = begin / 2U, last = end / 2U;
        for (auto layer = first; layer < last; ++layer) work += costs[s][layer];
        if (end % 2U != 0U) {
            if (!(front[s][last] <= costs[s][last]))
                throw std::invalid_argument("pair pipeline half-layer cut is outside the layer costs");
            work += front[s][last];
        }
        if (begin % 2U != 0U) work -= front[s][first];
        return work;
    };
    ChainSplit best;
    double score = std::numeric_limits<double>::infinity();
    std::vector<std::uint32_t> at(stages + 1U, 0U);   // boundary positions; at[0] = 0, at[stages] = 2 L
    std::vector<double> work(stages, 0.0);
    at[stages] = 2U * layers;
    std::uint64_t candidates = 0;
    // Places the end of stage s (s < stages - 1) or, for the last stage, scores the candidate. `running` is the max
    // of the placed stages 0 .. s - 1 (same comparison form as the two-stage `a > b ? a : b`).
    std::function<void(std::uint32_t, double)> place = [&](std::uint32_t s, double running) {
        if (s + 1U == stages) {
            if (++candidates > max_balance_candidates) throw std::invalid_argument("pair pipeline balance search limit exceeded");
            work[s] = stage_work(s, at[s], at[stages]);
            const auto worst = s == 0U ? work[s] : (running > work[s] ? running : work[s]);
            if (worst < score) {
                score = worst;
                best.cuts.assign(stages - 1U, 0U);
                best.half.assign(stages - 1U, false);
                for (std::uint32_t k = 0; k + 1U < stages; ++k) {
                    best.cuts[k] = at[k + 1U] / 2U;
                    best.half[k] = at[k + 1U] % 2U != 0U;
                }
                best.work = work;
            }
            return;
        }
        const auto from = s == 0U ? 1U : at[s] / 2U + 1U;
        for (auto cut = from; cut + (stages - 2U - s) < layers; ++cut) {
            for (std::uint32_t h = 0; h < (halves ? 2U : 1U); ++h) {
                if (h == 0U ? !legal(cut) : !legal_half(cut)) continue;
                at[s + 1U] = 2U * cut + h;
                work[s] = stage_work(s, at[s], at[s + 1U]);
                const auto next = s == 0U ? work[s] : (running > work[s] ? running : work[s]);
                if (next >= score) continue;   // exact: every completion's max is at least this prefix's max (NaN: no prune)
                place(s + 1U, next);
            }
        }
    };
    place(0U, 0.0);
    if (!(score < std::numeric_limits<double>::infinity())) throw std::invalid_argument("pair pipeline has no legal cut");
    return best;
}
template<class Legal> ChainSplit balance_chain(std::span<const std::span<const double>> costs, double tail, Legal legal) {
    return balance_chain(costs, {}, tail, legal, [](std::uint32_t) { return false; });
}
namespace detail {
inline StageSplit two_stage(const ChainSplit& c) { return {c.cuts.at(0), c.work.at(0), c.work.at(1), c.half.at(0)}; }
} // namespace detail
inline StageSplit predict_split(std::span<const double> layers, double tail, std::uint32_t cut) {
    StageSplit split{cut, 0, tail};
    for (std::size_t layer = 0; layer < layers.size(); ++layer) (layer < cut ? split.a : split.b) += layers[layer];
    return split;
}
template<class Legal> StageSplit balance_split(std::span<const double> layers, double tail, Legal legal) {
    if (layers.size() < 2U) throw std::invalid_argument("pair pipeline has no legal cut");
    const std::span<const double> both[2] = {layers, layers};
    return detail::two_stage(balance_chain(std::span<const std::span<const double>>(both), tail, legal));
}
// Half-layer cuts: `front[layer]` is the cost of the first sub-block (attention) of a layer, part of layers[layer].
// A half cut at `cut` gives stage A layers [0, cut) plus front[cut].
inline StageSplit predict_split(std::span<const double> layers, std::span<const double> front, double tail,
    std::uint32_t cut, bool half) {
    auto split = predict_split(layers, tail, cut);
    if (!half) return split;
    if (cut >= layers.size() || front.size() != layers.size() || !(front[cut] <= layers[cut]))
        throw std::invalid_argument("pair pipeline half-layer cut is outside the layer costs");
    split.a += front[cut]; split.b -= front[cut]; split.half = true;
    return split;
}
template<class Legal, class LegalHalf> StageSplit balance_split(std::span<const double> layers,
    std::span<const double> front, double tail, Legal legal, LegalHalf legal_half) {
    if (layers.size() < 2U) throw std::invalid_argument("pair pipeline has no legal cut");
    const std::span<const double> both[2] = {layers, layers};
    if (front.size() != layers.size()) {   // the half candidate of the two-stage form throws on bad front costs
        for (std::uint32_t cut = 1; cut < layers.size(); ++cut)
            if (legal_half(cut)) throw std::invalid_argument("pair pipeline half-layer cut is outside the layer costs");
        return detail::two_stage(balance_chain(std::span<const std::span<const double>>(both), tail, legal));
    }
    const std::span<const double> fronts[2] = {front, front};
    return detail::two_stage(balance_chain(std::span<const std::span<const double>>(both),
        std::span<const std::span<const double>>(fronts), tail, legal, legal_half));
}
// Heterogeneous chain: stage s runs layers [cuts[s-1], cuts[s]) (cuts[-1] = 0, the last cut = layer count) on its
// own device, whose per-layer costs are costs[s] (each covering every layer of the model); the last stage also runs
// `tail`. With one shared vector and two stages this is predict_split.
inline std::vector<double> predict_chain(std::span<const std::span<const double>> costs,
                                         std::span<const std::uint32_t> cuts, double tail) {
    if (costs.empty() || cuts.size() != costs.size()) throw std::invalid_argument("pipeline chain needs one cut per stage");
    std::vector<double> work(costs.size(), 0.0);
    std::uint32_t begin = 0;
    for (std::size_t s = 0; s < costs.size(); ++s) {
        if (cuts[s] <= begin || cuts[s] > costs[s].size()) throw std::invalid_argument("pipeline chain cut out of order");
        for (std::uint32_t layer = begin; layer < cuts[s]; ++layer) work[s] += costs[s][layer];
        begin = cuts[s];
    }
    if (begin != costs.back().size()) throw std::invalid_argument("pipeline chain does not end at the last layer");
    work.back() += tail;
    return work;
}
} // namespace strix::pair
