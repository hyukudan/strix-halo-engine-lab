#pragma once
// Owned-layer ordinals (backend-neutral planning). A rank that owns layers [begin, end) of a model keeps per-layer state
// only for the layers of one kind inside that range: ordinal[l] = index of layer l among the OWNED layers of that kind
// (0..count-1), kNone for a layer of another kind or outside the range. Engines size state / backup / scratch by `count`
// and address slice ordinal[l]; the whole model (begin 0, end = layers) gives the global ordinals.
// Home: shared (host planning; used by the HIP Qwen 3.5 engine's GDN state, src/qwen35/hip/engine.hip).
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strix {

struct OwnedLayerOrdinals {
    static constexpr std::uint32_t kNone = ~0U;
    std::vector<std::uint32_t> ordinal;  // [layers]
    std::uint32_t count = 0;             // owned layers of the kind
    [[nodiscard]] bool has(std::uint32_t layer) const { return layer < ordinal.size() && ordinal[layer] != kNone; }
};

template <class IsKind>
OwnedLayerOrdinals owned_layer_ordinals(std::uint32_t layers, std::uint32_t begin, std::uint32_t end, IsKind&& is_kind) {
    if (begin >= end || end > layers) throw std::invalid_argument("owned_layer_ordinals: need 0 <= begin < end <= layers");
    OwnedLayerOrdinals o;
    o.ordinal.assign(layers, OwnedLayerOrdinals::kNone);
    for (std::uint32_t l = begin; l < end; ++l)
        if (is_kind(l)) o.ordinal[l] = o.count++;
    return o;
}

}  // namespace strix
