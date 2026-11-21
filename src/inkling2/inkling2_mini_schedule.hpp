#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>

namespace strix::inkling2 {

// Physical weights are shared across the three core passes. Logical visits
// have independent runtime attention/convolution state.
struct LayerVisit {
    unsigned physical_layer;
    unsigned pass;
    unsigned logical_layer;
};

inline constexpr unsigned kPhysicalLayers = 24;
inline constexpr unsigned kLogicalVisits = 48;
inline constexpr unsigned kCoreFirst = 6;
inline constexpr unsigned kCoreLast = 18;
inline constexpr unsigned kCorePasses = 3;

constexpr std::array<LayerVisit, kLogicalVisits> mini_schedule() {
    std::array<LayerVisit, kLogicalVisits> out{};
    unsigned i = 0;
    for (unsigned layer = 0; layer < kCoreFirst; ++layer)
        out[i] = {layer, 0, i}, ++i;
    for (unsigned pass = 0; pass < kCorePasses; ++pass)
        for (unsigned layer = kCoreFirst; layer < kCoreLast; ++layer)
            out[i] = {layer, pass, i}, ++i;
    for (unsigned layer = kCoreLast; layer < kPhysicalLayers; ++layer)
        out[i] = {layer, 0, i}, ++i;
    return out;
}

constexpr bool is_core(unsigned physical_layer) {
    return physical_layer >= kCoreFirst && physical_layer < kCoreLast;
}

constexpr unsigned state_slot(unsigned physical_layer, unsigned pass) {
    if (physical_layer >= kPhysicalLayers || (is_core(physical_layer) ? pass >= kCorePasses : pass != 0))
        throw std::out_of_range("invalid Inkling 2 Mini layer/pass");
    if (physical_layer < kCoreFirst) return physical_layer;
    if (is_core(physical_layer)) return kCoreFirst + pass * (kCoreLast - kCoreFirst) + physical_layer - kCoreFirst;
    return kCoreFirst + kCorePasses * (kCoreLast - kCoreFirst) + physical_layer - kCoreLast;
}

} // namespace strix::inkling2
