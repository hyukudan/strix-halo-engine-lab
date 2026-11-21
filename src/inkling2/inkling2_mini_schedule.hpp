#pragma once

#include "core/looped_layers.hpp"

namespace strix::inkling2 {

using core::LayerVisit;
inline constexpr unsigned kPhysicalLayers = 24;
inline constexpr unsigned kCoreFirst = 6;
inline constexpr unsigned kCoreLast = 18;
inline constexpr unsigned kCorePasses = 3;
inline constexpr unsigned kResidualBlockLayers = 6;
using MiniLayers = core::LoopedLayers<kPhysicalLayers, kCoreFirst, kCoreLast, kCorePasses>;
inline constexpr unsigned kLogicalVisits = MiniLayers::visits;

constexpr auto mini_schedule() { return MiniLayers::schedule(); }
constexpr bool valid_mini_visit(LayerVisit visit) {
    constexpr auto schedule = mini_schedule();
    return visit.logical_layer < schedule.size() &&
           schedule[visit.logical_layer].physical_layer == visit.physical_layer &&
           schedule[visit.logical_layer].pass == visit.pass;
}
constexpr bool is_core(unsigned physical_layer) { return MiniLayers::is_core(physical_layer); }
constexpr unsigned state_slot(unsigned physical_layer, unsigned pass) {
    return MiniLayers::state_slot(physical_layer, pass);
}

constexpr bool starts_residual_block(LayerVisit visit) {
    return visit.physical_layer % kResidualBlockLayers == 0;
}

constexpr bool is_full_attention(unsigned physical_layer) {
    return physical_layer % kResidualBlockLayers == kResidualBlockLayers - 1U;
}

// A repeated pass reads the incoming residual at its second core block
// boundary, plus the hidden output at the end of that pass.
constexpr bool captures_message_boundary(LayerVisit visit) {
    return visit.physical_layer == kCoreFirst + kResidualBlockLayers &&
           visit.pass + 1U < kCorePasses;
}

// The official reference numbers pass-zero slots by physical layer, then
// appends caches for repeats. Takes a visit from mini_schedule().
constexpr unsigned reference_state_slot(LayerVisit visit) {
    return visit.pass == 0 ? visit.physical_layer
        : kPhysicalLayers + (visit.pass - 1U) * MiniLayers::core_width
              + (visit.physical_layer - kCoreFirst);
}

}  // namespace strix::inkling2
