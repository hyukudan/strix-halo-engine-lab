#pragma once

// Route sources and the expert union of the microscaled R1 decode family (B8 "route + plan as one device operation").
//
// The incumbent decode MoE visit is four dependent launches: router projection -> checked route (one small launch:
// top-k ids, renormalised weights, status word) -> gate_up -> down_combine. For R1 the plan of a routed row is the
// route itself (the kernels address the experts by id), so route + plan is one operation, and it runs inside gate_up:
//
//   Inline<Policy, Math, Traits>   a route source of gate_up. Before its first weight read, every gate_up CTA runs the
//                                  checked route body (moe_core/route_wave_kernel.hpp route_wave_body, the function of
//                                  the route launch, so the ids / weights / status bits have one definition) on its
//                                  row's logits into shared memory, then reads its expert from there. One CTA per
//                                  routed row publishes that row's route (ids, weights, status) to the R1 Args buffers,
//                                  which down_combine reads after gate_up completed. The checked route launch and its
//                                  dependency boundary disappear; the route arithmetic is unchanged (CUDA).
//
// Rows launches (decode verify windows) add the expert union: the routed assignments of all rows are grouped by
// distinct expert, ascending id, and each expert's weights are read once for every row that selected it
// (microscaled_r1 dot_inputs keeps each row's lane chunks, FMA order and butterfly, so each row's bits are those of the
// one-row launch). expert_union() below is the host statement of that plan; the device kernels build the same groups
// in shared memory from the route (gate_up_union with or without an Inline source, down_union).
//
// Legality (microscaled_r1.hpp admits_route): the policy's experts / top_k equal the launch's, the route is checked
// (status words; a failed row routes nowhere), the publish targets are the Args route buffers, and no buffer the
// route writes overlaps a buffer any CTA of the launch reads (logits, bias, input) or writes (activation).

#include "strix/moe/route_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include "moe_core/route_wave_kernel.hpp"
#endif

namespace strix::moe_core::r1_route {

// A route source: the router logits ([rows][experts], predecessor output), the selection bias ([experts], constant)
// and the route buffers it publishes ([rows][top_k] ids and weights, [rows] status), which are the R1 Args buffers.
template <class Policy, class Math, class Traits>
struct Inline {
    const float* logits;
    const float* bias;
    std::uint32_t* ids;
    float* weights;
    unsigned* failure;
};

// Distinct experts of a union launch: at most one per routed assignment and at most the expert count.
inline constexpr unsigned union_groups(unsigned rows, unsigned top_k, unsigned experts) {
    return rows * top_k < experts ? rows * top_k : experts;
}

// Largest expert count the device union keeps as a presence bitmask in shared memory, and the widest union launch
// (the verify / replay widths: 2, 4, 8, 16 rows).
inline constexpr unsigned kMaxUnionExperts = 1024U;
inline constexpr unsigned kMaxUnionRows = 16U;

// Host statement of the expert union of a routed rows launch. Assignment a = row * top_k + slot participates when its
// row is valid (row_valid empty or row_valid[row] != 0), its status word is zero (status empty = all routed) and
// ids[a] < experts. Groups are ordered by ascending expert id; a group lists its assignments in ascending order.
struct UnionGroup {
    std::uint32_t expert;
    std::vector<std::uint32_t> assignments;
};

inline std::vector<UnionGroup> expert_union(std::span<const std::uint32_t> ids, std::span<const unsigned> status,
                                            std::span<const unsigned char> row_valid, unsigned rows, unsigned top_k,
                                            unsigned experts) {
    std::vector<std::vector<std::uint32_t>> by_expert(experts);
    for (unsigned row = 0; row < rows; ++row) {
        if (!row_valid.empty() && row_valid[row] == 0U) continue;
        if (!status.empty() && status[row] != 0U) continue;
        for (unsigned slot = 0; slot < top_k; ++slot) {
            const auto a = row * top_k + slot;
            if (ids[a] < experts) by_expert[ids[a]].push_back(a);
        }
    }
    std::vector<UnionGroup> groups;
    for (std::uint32_t e = 0; e < experts; ++e) {
        if (!by_expert[e].empty()) groups.push_back({e, std::move(by_expert[e])});
    }
    return groups;
}

#if defined(__CUDACC__) && !defined(__HIPCC__)
// route_wave_body sites (which wave routes which row).
// CtaRow: wave 0 routes row r of a rows launch (the gate_up CTA's own row); the other waves idle.
struct CtaRow {
    static constexpr unsigned kWaves = 1U;
    unsigned r, rows;
    __device__ unsigned row(unsigned wave) const { return wave == 0U ? r : rows; }
};

// WaveRows: wave w routes row w (a union CTA routes every row of the launch; rows <= kWaves).
template <unsigned Waves>
struct WaveRows {
    static constexpr unsigned kWaves = Waves;
    __device__ static unsigned row(unsigned wave) { return wave; }
};

// Route the rows the site assigns to this CTA's waves into ids / weights / status (any address space; the R1 kernels
// pass shared memory). Same body and bits as the checked route launch.
template <class Policy, class Math, class Traits, class Site>
__device__ __forceinline__ void route_rows(const Inline<Policy, Math, Traits>& route, std::uint32_t* ids,
                                           float* weights, unsigned* status, unsigned rows, Site site) {
    static_assert(Traits::checked_status, "an inline R1 route publishes its status word");
    route_wave_body<Policy::value, Math, Traits, Site>(route.logits, route.bias, ids, weights, status, rows, site);
}
#endif

}  // namespace strix::moe_core::r1_route
