#pragma once
// Short-context rule of indexed sparse attention (backend-neutral host policy; plain C++, no device headers).
//
// An indexed sparse layer scores the visible key units (pools / blocks of `unit_tokens` positions) with an indexer
// query and attends the `topk_units` best. While a query sees at most `topk_units` scored units the selection is every
// unit in position order (the exact top-k of a set no larger than k is the whole set), so the indexer QUERY side
// (query projection, head weights, scores, top-k) is dead work: no kernel reads it. The KEY side still runs (the cache
// must hold index keys for later positions). One predicate for the token path, verify rows and windows, so a step and
// its replay agree.
//
// Geometry per model (declaration only):
//   unit_tokens          positions per scored unit;
//   topk_units           units the selector keeps;
//   partial_unit_scored  true: the newest unfinished unit is scored too (units = ceil((pos + 1) / unit)); false: only
//                        completed units are scored and the unfinished tail is always attended (floor).
#include "strix/env_flags.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>

namespace strix::sparse_select {

struct Geometry {
    std::uint64_t unit_tokens = 1;
    std::uint64_t topk_units = 0;
    bool partial_unit_scored = true;
};

// Units the selector would rank for a query at absolute position `pos` (causal: positions 0..pos are visible).
constexpr std::uint64_t scored_units(const Geometry& g, std::uint64_t pos) {
    const std::uint64_t visible = pos + 1;
    return g.partial_unit_scored ? (visible + g.unit_tokens - 1) / g.unit_tokens : visible / g.unit_tokens;
}

// The selection of this query is every visible unit in order (no indexer query, score or top-k is needed).
constexpr bool selection_is_identity(const Geometry& g, std::uint64_t pos) { return scored_units(g, pos) <= g.topk_units; }

// The same for every row of a step (decode rows, verify rows, a window): `pos_of(row)` gives a row's position.
template <class Row, class PosOf> bool step_selection_is_identity(const Geometry& g, std::span<const Row> rows, PosOf pos_of) {
    for (const Row& r : rows)
        if (!selection_is_identity(g, static_cast<std::uint64_t>(pos_of(r)))) return false;
    return true;
}

// STRIX_SELECT_SHORT_SKIP: unset / empty / "1" = on (skip the dead indexer query work), "0" = off (compute it anyway,
// the previous form); other values fail closed. Engines read it once at startup (not per layer call).
inline bool query_skip_enabled() {
    const char* e = strix::envcfg::text("STRIX_SELECT_SHORT_SKIP");
    if (e == nullptr || *e == '\0' || std::strcmp(e, "1") == 0) return true;
    if (std::strcmp(e, "0") == 0) return false;
    throw std::invalid_argument("STRIX_SELECT_SHORT_SKIP must be 0 or 1");
}

}  // namespace strix::sparse_select
