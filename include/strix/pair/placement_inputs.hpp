#pragma once
// Placement inputs from a pack's tensor table (model- and vendor-agnostic, host only). A binding hands over its tensors as
// records {layer, component, bytes} (layer -1 = global; from the shared tensor naming map's claims, or from GGUF names through
// block_of) and a role per component; this header turns them into what the shared planners read:
// - the layer chain (strix/pair/chain.hpp): per-layer resident bytes and decode bytes read per token (chain::ModelShape);
// - exact tensor parallelism (strix/pair/column_split.hpp): the bytes a decode step splits over the ranks and the bytes every
//   rank reads whole (tp::StepModel), and the bytes each rank keeps resident.
// Roles say how a component behaves under placement, not what it computes:
//   split      divided over tensor-parallel ranks (head groups, column / K slices); read whole by one rank per step
//   experts    expert banks: divided over ranks (expert slices), a step reads the fraction of experts its rows touch
//   replicated kept whole on every tensor-parallel rank (small projections, norms, routers, state builders)
//   lookup     row-gathered tables (embedding-like sidecars): resident where placed, a step reads only the gathered rows
//   embed      token embedding: resident on the first stage, a step reads one row per token
//   head       output head: divided by vocabulary over tensor-parallel ranks, read whole per token on one rank
// Blocks at or above `layers` (next-token / draft blocks) are kept apart as `beyond`: not trunk layers, resident where
// drafting runs.
// Home: shared (host shared). Bound by: qwen35 pair plan (block_of); a family placement reads the tensor naming map components.
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
#include "strix/pair/chain.hpp"
#include "strix/pair/column_split.hpp"

namespace strix::pair::place {
enum class Role : std::uint8_t { split, experts, replicated, lookup, embed, head, count };
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::count);
using RoleBytes = std::array<double, kRoles>;
[[nodiscard]] inline double& at(RoleBytes& bytes, Role role) { return bytes[static_cast<std::size_t>(role)]; }
[[nodiscard]] inline double at(const RoleBytes& bytes, Role role) { return bytes[static_cast<std::size_t>(role)]; }
[[nodiscard]] inline double sum(const RoleBytes& bytes) {
    double s = 0.0;
    for (const double b : bytes) s += b;
    return s;
}

// Owner of a GGUF tensor by name: "blk.<l>." -> l, any other name -> -1 (global). "blk." without digits and a dot throws.
[[nodiscard]] inline std::int64_t block_of(std::string_view name) {
    if (!name.starts_with("blk.")) return -1;
    std::int64_t layer = 0;
    std::size_t i = 4;
    bool digits = false;
    for (; i < name.size() && name[i] >= '0' && name[i] <= '9'; ++i) {
        layer = layer * 10 + static_cast<std::int64_t>(name[i] - '0');
        digits = true;
        if (layer > 0xffffffffLL) throw std::invalid_argument("block tensor index out of range");
    }
    if (!digits || i >= name.size() || name[i] != '.') throw std::invalid_argument("malformed block tensor name");
    return layer;
}

struct Record {
    std::int64_t layer = -1;       // -1: global
    std::string_view component;    // binding key for the role
    double bytes = 0.0;
};
struct Inputs {
    std::uint32_t layers = 0;
    std::vector<RoleBytes> layer;  // trunk layers [0, layers)
    RoleBytes global{};
    RoleBytes beyond{};            // blocks >= layers
    [[nodiscard]] double total(Role role) const {
        double s = at(global, role);
        for (const auto& l : layer) s += at(l, role);
        return s;
    }
    [[nodiscard]] double trunk_bytes() const {
        double s = 0.0;
        for (const auto& l : layer) s += sum(l);
        return s;
    }
};
// `role_of(component, layer)` -> Role. Records of one (layer, component) may repeat; bytes add.
template<class Records, class RoleOf> [[nodiscard]] Inputs inputs(const Records& records, std::uint32_t layers, RoleOf&& role_of) {
    Inputs in;
    in.layers = layers;
    in.layer.assign(layers, RoleBytes{});
    for (const Record& r : records) {
        if (!(r.bytes >= 0.0) || !std::isfinite(r.bytes)) throw std::invalid_argument("placement record bytes must be finite and >= 0");
        const Role role = role_of(r.component, r.layer);
        if (role >= Role::count) throw std::invalid_argument("placement role out of range");
        if (r.layer < 0) at(in.global, role) += r.bytes;
        else if (r.layer < static_cast<std::int64_t>(layers)) at(in.layer[static_cast<std::size_t>(r.layer)], role) += r.bytes;
        else at(in.beyond, role) += r.bytes;
    }
    return in;
}

// Expected fraction of an expert bank touched by `rows` routed rows of top-`k` of `experts`, routes independent and uniform:
// 1 - (1 - k/E)^rows. One row is exactly k/E; for verify rows it is an UPPER bound (real routes of neighbouring rows correlate).
[[nodiscard]] inline double expert_union_fraction(std::uint32_t k, std::uint32_t experts, std::uint32_t rows) {
    if (experts == 0U || k == 0U || k > experts || rows == 0U) throw std::invalid_argument("expert union: need 0 < k <= experts, rows > 0");
    return 1.0 - std::pow(1.0 - static_cast<double>(k) / static_cast<double>(experts), static_cast<double>(rows));
}

// Decode bytes read per step for each trunk layer: everything but the expert banks and lookup tables, the expert banks times
// `expert_fraction`, plus `lookup_read[l]` (gathered rows; empty = none). Embedding row and head are added by the caller's
// step (chain::ModelShape::head_bytes; the embedding row is negligible and left out).
[[nodiscard]] inline std::vector<double> read_bytes_per_token(const Inputs& in, double expert_fraction, std::span<const double> lookup_read = {}) {
    if (!lookup_read.empty() && lookup_read.size() != in.layers) throw std::invalid_argument("lookup reads must cover every layer");
    std::vector<double> out(in.layers, 0.0);
    for (std::uint32_t l = 0; l < in.layers; ++l) {
        const auto& b = in.layer[l];
        out[l] = sum(b) - at(b, Role::experts) - at(b, Role::lookup) + at(b, Role::experts) * expert_fraction +
                 (lookup_read.empty() ? 0.0 : lookup_read[l]);
    }
    return out;
}

// Layer-chain model shape. `lookup_resident[l]`: global lookup bytes the binding places with layer l (e.g. per-layer slices of
// one table; empty = none placed per layer). State and seam come from the model's layout.
struct StateLayout {
    double seam_bytes_per_row = 0.0;
    std::vector<double> per_row;   // per layer, prefix state per row (empty = none)
    std::vector<double> fixed;     // per layer, fixed-size state (empty = none)
};
[[nodiscard]] inline chain::ModelShape chain_shape(const Inputs& in, const StateLayout& state, double expert_fraction,
                                                   std::span<const double> lookup_resident = {}, std::span<const double> lookup_read = {}) {
    if (!lookup_resident.empty() && lookup_resident.size() != in.layers) throw std::invalid_argument("lookup slices must cover every layer");
    chain::ModelShape m;
    m.layers = in.layers;
    m.seam_bytes_per_row = state.seam_bytes_per_row;
    m.state_bytes_per_row = state.per_row.empty() ? std::vector<double>(in.layers, 0.0) : state.per_row;
    m.state_fixed_bytes = state.fixed.empty() ? std::vector<double>(in.layers, 0.0) : state.fixed;
    if (m.state_bytes_per_row.size() != in.layers || m.state_fixed_bytes.size() != in.layers)
        throw std::invalid_argument("state layout must cover every layer");
    m.weight_bytes.assign(in.layers, 0.0);
    for (std::uint32_t l = 0; l < in.layers; ++l)
        m.weight_bytes[l] = sum(in.layer[l]) + (lookup_resident.empty() ? 0.0 : lookup_resident[l]);
    m.embed_bytes = at(in.global, Role::embed);
    m.head_bytes = at(in.global, Role::head) + at(in.global, Role::replicated) + at(in.global, Role::split);
    m.read_bytes_per_token = read_bytes_per_token(in, expert_fraction, lookup_read);
    return m;
}

// Exact tensor-parallel decode step: split = head-split / sliced / expert-slice bytes the step reads (experts at
// `expert_fraction`) plus the vocabulary-split head; replicated = what every rank reads whole (+ `replicated_read_extra`, e.g.
// gathered lookup rows). `trips`: the binding's all-reduce points of one step, in order.
[[nodiscard]] inline tp::StepModel tp_step(const Inputs& in, double expert_fraction, std::span<const tp::Trip> trips,
                                           double replicated_read_extra = 0.0, double fixed_ms = 0.0) {
    tp::StepModel m;
    for (const auto& b : in.layer) {
        m.split_bytes += at(b, Role::split) + at(b, Role::experts) * expert_fraction;
        m.replicated_bytes += at(b, Role::replicated);
    }
    m.split_bytes += at(in.global, Role::split) + at(in.global, Role::head);
    m.replicated_bytes += at(in.global, Role::replicated) + replicated_read_extra;
    m.fixed_ms = fixed_ms;
    m.trips.assign(trips.begin(), trips.end());
    return m;
}
// Bytes one tensor-parallel rank keeps resident when it owns `fraction` of every divided role (split, experts, head) and
// `lookup_fraction` of the lookup tables; replicated roles and the embedding stay whole. `with_beyond`: the rank also holds
// its share of the blocks above the trunk (drafting on).
[[nodiscard]] inline double tp_resident_bytes(const Inputs& in, double fraction, double lookup_fraction, bool with_beyond) {
    const auto rank_part = [&](const RoleBytes& b) {
        return (at(b, Role::split) + at(b, Role::experts) + at(b, Role::head)) * fraction + at(b, Role::replicated) +
               at(b, Role::embed) + at(b, Role::lookup) * lookup_fraction;
    };
    double s = rank_part(in.global);
    for (const auto& b : in.layer) s += rank_part(b);
    if (with_beyond) s += rank_part(in.beyond);
    return s;
}
}  // namespace strix::pair::place
