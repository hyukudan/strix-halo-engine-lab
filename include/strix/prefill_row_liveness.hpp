#pragma once

#include <cstdint>
#include <span>

// Row liveness of a prompt pass (shared host policy, every engine and backend).
//
// A prompt pass runs rows through a stack of layers. A row's work at a layer is live only when something reads what
// it produces: a later row (through the caches and recurrent states the layer writes), a later layer of the same row
// that is itself live, or a consumer outside the stack (draft-model taps, final-norm rows, logits, digests).
//
// Every prompt row writes the caches of the layers that own state, so every row must run up to the LAST layer that
// owns state (K/V or recurrent). A layer above it owns nothing that another row reads: for a row nobody reads after
// the stack, the layers above the last state owner, the final norm and the head are dead work. At the last owner
// itself, such a row needs only the state writes (the cache-only form of strix/prefill_headless.hpp, which is this rule
// applied to a stack whose last layer owns state).
//
// The rule is declared once from the stack's state ownership (a model description's per-layer states, or a family's
// producer map) plus the pass's readers; engines bind the resulting plan. It changes which rows run, never how a row is
// computed: rows that run keep their bits as long as the backend's kernels give a row the same bytes whatever the other
// rows of the call are (`tail_row_invariant`); without that, a pass that must still deliver its last row runs in full.
namespace strix::prefill_liveness {

// Who reads the rows of the pass once the layers have run.
struct Readers {
    std::span<const std::uint32_t> taps{}; // layers whose output a consumer reads for every row (draft taps); >= layers: none
    bool every_row = false;                // every row's final output is read (final-norm rows, all-row logits, a pipeline seam)
    bool head_row = true;                  // the pass's last row continues to the final norm and the head (now or deferred)
    bool tail_row_invariant = false;       // one row's bytes do not depend on the rows beside it in a call (linears, attention)
};

struct Plan {
    unsigned layers = 0;         // layers in the stack
    unsigned full_layers = 0;    // layers [0, full_layers) run for every row
    bool cache_only_last = false; // rows past the last owner are unread: layer full_layers - 1 needs only its state writes for them
    bool tail = false;           // the last row alone runs the rest: [full_layers, layers), the final norm and the head
    [[nodiscard]] constexpr bool cut() const noexcept { return full_layers < layers; }
    // Some rows stop early: above the cut, or after the state writes of the last owner.
    [[nodiscard]] constexpr bool narrows() const noexcept { return cut() || cache_only_last; }
    // Nothing after the state writes of layer full_layers - 1 is read: no final norm, no head, no last row kept.
    [[nodiscard]] constexpr bool ends_at_state() const noexcept { return narrows() && !tail; }
};

// `owns_state(l)` is true when layer l writes state that a later row reads (its own K/V or a recurrent state).
template <class OwnsState>
[[nodiscard]] constexpr Plan plan(unsigned layers, OwnsState&& owns_state, const Readers& r) {
    Plan p{layers, layers, false, false};
    if (layers == 0 || r.every_row) return p;
    unsigned owners = 0, tapped = 0;
    for (unsigned l = 0; l < layers; ++l)
        if (owns_state(l)) owners = l + 1;
    for (const auto t : r.taps)
        if (t < layers && t + 1 > tapped) tapped = t + 1;
    const unsigned depth = owners > tapped ? owners : tapped;
    // The last row must run past the cut with the others absent; only kernels that are row-invariant allow it.
    if (depth == 0 || (r.head_row && !r.tail_row_invariant)) return p;
    const bool cache_only = depth == owners && tapped < owners;
    if (depth >= layers && !cache_only) return p;
    p.full_layers = depth;
    p.cache_only_last = cache_only;
    p.tail = r.head_row;
    return p;
}

// The same rule over a model graph declaration (any type with `layers[i].states`, src/model_graph/model_graph.hpp): a
// layer owns state when it declares any (attention K/V, latent / compressed caches, recurrent state).
template <class Graph>
[[nodiscard]] constexpr Plan plan_graph(const Graph& graph, const Readers& r) {
    return plan(unsigned(graph.layers.size()), [&](unsigned l) { return !graph.layers[l].states.empty(); }, r);
}

// Row-layers the plan removes from a pass of `rows` rows (receipt counter; the cache-only layer is not counted).
[[nodiscard]] constexpr std::uint64_t skipped_row_layers(const Plan& p, unsigned rows) noexcept {
    if (!p.cut() || rows == 0) return 0;
    const std::uint64_t unread = p.tail ? rows - 1U : rows;
    return unread * (p.layers - p.full_layers);
}

}  // namespace strix::prefill_liveness
