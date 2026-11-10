#pragma once
#include "glm_deploy_provider.hpp"
#include <map>
#include <string_view>

namespace strix::glm::deployment {
// Header-only upper bound on the existing TP2 loader. Raw MoE slices share
// 1-GiB arenas; other EXL3 linears reserve both raw and prepared representations.
// Retained source tensors are never added again as a full model file mapping.
template<class Tensors>
Owners header_owners(const Tensors& tensors, bool mtp, bool fast_moe) {
    using deploy::checked_add; using deploy::checked_mul; using deploy::aligned;
    std::uint64_t arena = 0, largest = 0, prepared = 0, whole = 0;
    const auto arena_add = [&](std::uint64_t n) {
        n = aligned(n, 256);
        // Isolate large requests in the bound so a late vocab/embedding
        // allocation cannot inflate the rollover allowance of every expert.
        // Two chunks also cover a discarded preceding arena and final slack.
        if (n >= (32ULL << 20)) prepared = checked_add(prepared, checked_add(aligned(n, 2ULL << 20), 2ULL << 30));
        else { arena = checked_add(arena, n); largest = std::max(largest, n); }
    };
    for (const auto& [name, t] : tensors) {
        const std::string_view n(name);
        if (!mtp && n.find(".layers.45.") != n.npos) continue;
        if (n.ends_with(".suh") || n.ends_with(".svh") || n.ends_with(".mul1")) continue;
        if (n.ends_with(".trellis")) {
            const auto base = name.substr(0, name.size()-8);
            const auto& suh = tensors.at(base + ".suh");
            const auto& svh = tensors.at(base + ".svh");
            const bool out = n.find("qkv_proj.") != n.npos || n.find("q_b_proj.") != n.npos ||
                (n.find("mlp.") != n.npos && !n.ends_with("down_proj.trellis")) || base == "lm_head";
            const bool in = n.find("o_proj.") != n.npos || n.ends_with("down_proj.trellis");
            const auto tr = out || in ? checked_add(t.bytes, 1) / 2 : t.bytes;
            const auto sh = in ? checked_add(suh.bytes, 1) / 2 : suh.bytes;
            const auto sv = out ? checked_add(svh.bytes, 1) / 2 : svh.bytes;
            const auto bytes = checked_add(tr, checked_add(sh, sv));
            // Three KDA q/k/v slices require independent padding/prepared owners.
            const auto padded = checked_add(bytes, 3 * 768);
            arena_add(padded);
            const bool raw_only = fast_moe && n.find(".mlp.experts.") != n.npos;
            if (!raw_only) prepared = checked_add(prepared, checked_add(aligned(padded, 2ULL << 20), 6ULL << 20));
            // Input/output assembly can read the complete unsliced trellis.
            whole = std::max(whole, t.bytes);
        } else {
            const bool half = n.find("kv_b_proj.weight") != n.npos || n.find("f_b_proj.weight") != n.npos ||
                n.find("g_b_proj.weight") != n.npos || n.find("b_proj.weight") != n.npos ||
                n.find("conv1d.weight") != n.npos || n.find("A_log") != n.npos || n.find("dt_bias") != n.npos;
            arena_add(half ? checked_add(t.bytes, 1) / 2 : t.bytes);
            // R1b packed originals are retained. Include every eligible copy.
            if (n.find("f_a_proj.weight") != n.npos || n.find("g_a_proj.weight") != n.npos ||
                n.find("b_proj.weight") != n.npos || n.ends_with("mlp.gate.weight")) arena_add(t.bytes);
        }
    }
    if (!arena && !prepared) throw std::invalid_argument("GLM deployment: empty tensor inventory");
    // An arena rollover loses less than the largest sub-GiB request; one final
    // arena is partly empty. Oversized allocations were charged separately.
    const auto chunk = 1ULL << 30;
    const auto arenas = arena ? checked_add(arena / (chunk-largest), 1) : 0;
    const auto padding = checked_mul(arenas, largest);
    Owners result;
    result.weights = checked_add(prepared, checked_add(arena, checked_add(padding, chunk)));
    // Expert view tables, HC constants and other small loader metadata.
    result.weights = checked_add(result.weights, 16ULL << 20);
    result.staging = checked_add(chunk, checked_add(whole, 256ULL << 20));
    return result;
}
} // namespace strix::glm::deployment
