#pragma once

#include <array>
#include <stdexcept>

namespace strix::core {

struct LayerVisit {
    unsigned physical_layer;
    unsigned pass;
    unsigned logical_layer;
};

// Weight identity is physical. Persistent state identity is a visit in
// execution order and is independent of the transient residual snapshots.
template <unsigned Layers, unsigned First, unsigned End, unsigned Passes>
struct LoopedLayers {
    static_assert(First < End && End <= Layers && Passes > 0);
    static constexpr unsigned core_width = End - First;
    static constexpr unsigned core_end_visit = First + Passes * core_width;
    static constexpr unsigned visits = core_end_visit + Layers - End;

    static constexpr bool is_core(unsigned layer) {
        return layer >= First && layer < End;
    }

    static constexpr unsigned state_slot(unsigned layer, unsigned pass) {
        if (layer >= Layers || pass >= (is_core(layer) ? Passes : 1U)) {
            throw std::out_of_range("invalid looped layer/pass");
        }
        if (is_core(layer)) return layer + pass * core_width;
        return layer < First ? layer : layer + (Passes - 1U) * core_width;
    }

    static constexpr std::array<LayerVisit, visits> schedule() {
        std::array<LayerVisit, visits> plan{};
        for (unsigned ordinal = 0; ordinal < visits; ++ordinal) {
            unsigned layer = ordinal;
            unsigned pass = 0;
            if (ordinal >= core_end_visit) {
                layer = End + ordinal - core_end_visit;
            } else if (ordinal >= First) {
                layer = First + (ordinal - First) % core_width;
                pass = (ordinal - First) / core_width;
            }
            plan[ordinal] = {layer, pass, ordinal};
        }
        return plan;
    }

    // The adapter captures the previous pass before restoring the entry.
    // Persistent state is only addressed by run_layer's visit.
    template <class Adapter>
    static void run(Adapter& adapter) {
        constexpr auto plan = schedule();
        for (const LayerVisit visit : plan) {
            if (visit.physical_layer == First) {
                if (visit.pass == 0) {
                    adapter.save_core_entry();
                } else {
                    adapter.capture_previous_pass(visit.pass - 1);
                    adapter.restore_core_entry();
                }
            }
            adapter.run_layer(visit);
        }
    }
};

}  // namespace strix::core
