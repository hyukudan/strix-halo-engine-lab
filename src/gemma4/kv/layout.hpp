#pragma once

#include "gemma4/contracts/g4_types.hpp"
#include <stdexcept>

namespace strix::gemma4::gpu {
inline void validate_config(const ModelConfig& c) {
    if (!c.hidden || c.hidden > hidden || !c.layers || c.layers > layers || c.layer.size() != c.layers || !c.inter ||
        c.inter > inter || !c.vocab || c.vocab > vocab || !c.heads || c.heads > heads || !c.kv_heads ||
        c.kv_heads > kv_heads || c.heads / c.kv_heads != group || c.heads % c.kv_heads || !c.ple_dim ||
        c.ple_dim > ple_dim || !c.window || c.window > window || !c.max_positions || c.max_positions > max_positions ||
        c.eps != norm_eps)
        throw std::runtime_error("invalid g4-v1 device configuration");
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const auto& a = c.layer[l];
        if (!a.head_dim || a.head_dim % 2 || a.head_dim > (a.attention == AttentionKind::global ? 512U : 256U) ||
            a.producer > l)
            throw std::runtime_error("invalid device attention geometry");
        const auto& p = c.layer[a.producer];
        if (p.producer != a.producer || p.attention != a.attention || p.head_dim != a.head_dim)
            throw std::runtime_error("invalid device KV producer alias");
    }
}
inline bool same_config(const ModelConfig& a, const ModelConfig& b) {
    if (a.hidden != b.hidden || a.layers != b.layers || a.inter != b.inter || a.vocab != b.vocab ||
        a.heads != b.heads || a.kv_heads != b.kv_heads || a.ple_dim != b.ple_dim || a.window != b.window ||
        a.max_positions != b.max_positions || a.eps != b.eps || a.layer.size() != b.layer.size())
        return false;
    for (std::size_t l = 0; l < a.layer.size(); ++l)
        if (a.layer[l].attention != b.layer[l].attention || a.layer[l].producer != b.layer[l].producer ||
            a.layer[l].head_dim != b.layer[l].head_dim)
            return false;
    return true;
}
struct Visibility {
    std::uint64_t first, last;
};
STRIX_G4_HD inline Visibility visible(std::uint64_t position, bool global, std::uint32_t capacity) {
    return {global || position < capacity ? 0 : position - capacity + 1, position};
}
inline std::size_t cache_bytes(const ModelConfig& c, std::uint32_t admitted) {
    validate_config(c);
    if (!admitted || admitted > c.max_positions)
        throw std::runtime_error("invalid admitted device context");
    std::size_t bytes = 0;
    for (std::uint32_t l = 0; l < c.layers; ++l)
        if (c.layer[l].producer == l) {
            const auto slots = c.layer[l].attention == AttentionKind::global ? admitted : c.window;
            bytes +=
                std::size_t(slots) * (2ULL * c.kv_heads * c.layer[l].head_dim * sizeof(float) + sizeof(std::uint64_t));
        }
    return bytes;
}
} // namespace strix::gemma4::gpu
