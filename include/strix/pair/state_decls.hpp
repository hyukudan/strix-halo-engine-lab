#pragma once
// Prompt-state declarations for strix/pair/state_owners.hpp: tensors only (owner, bytes, layers). MiniMax and GLM are
// bound from their model TOML in include/strix/{minimax,glm}/pair_state.hpp (same namespace); DeepSeek V4.1 and Qwen
// stay literals here because their description readers reject keys outside their generated contract.
// No family decides here what crosses which link; the shared planner does. Sizes are the consumer's import contract
// today (file:line in each piece); a format change updates the row here and the host test that pins its totals.
#include <array>
#include <cstddef>
#include <cstdint>

#include "strix/deepseek41_attention_state.hpp"
#include "strix/pair/state_owners.hpp"

namespace strix::pair::state::decl {

// DeepSeek V4.1 pair: the whole serialized state is the same on both ranks; the serve app already takes ONE copy and
// rank 0 relays it (apps/deepseek41_serve_main.cpp:2874-2875). Four KV sources (three ratio-2 at N/2 entries, one at
// N), FP4 KV 288 B + FP4 IK 68 B per entry (include/strix/ds41_attention.hpp:28-36), token ids 4 B, 40 window rings
// of capacity 136 x 528 B. rows = N. This is a tensor/token capacity estimate, NOT serialized traffic:
// carry/top-k capacity assumes three 1024-float carries and eight full 512-entry top-k vectors, and excludes framing.
// Use ds41_serialized_bytes with the actual state for wire traffic (short/absent vectors included).
inline constexpr std::array<Piece, 7> ds41_pieces{{
    {"kv_ratio2", 0, Owner::shared, Extent::rows, 288, 3, 0, 1, 2, 0, "src/ds41_attention.cpp:130-146"},
    {"kv_full", 1, Owner::shared, Extent::rows, 288, 1, 0, 1, 1, 0, "src/ds41_attention.cpp:130-146"},
    {"ik_ratio2", 2, Owner::shared, Extent::rows, 68, 3, 0, 1, 2, 0, "include/strix/ds41_attention.hpp:28-36"},
    {"ik_full", 3, Owner::shared, Extent::rows, 68, 1, 0, 1, 1, 0, "include/strix/ds41_attention.hpp:28-36"},
    {"token_ids", 4, Owner::shared, Extent::rows, 4, 1, 0, 1, 1, 0, "src/hip_deepseek41_batch_forward.hip:1490-1526"},
    {"window_rings", 5, Owner::shared, Extent::fixed, 136U * 528U, 40, 0, 1, 1, 0, "src/hip_ds41_attention.hip:122,1375-1395"},
    {"carry_topk_capacity", 6, Owner::shared, Extent::fixed, 28672, 1, 0, 1, 1, 0, "src/hip_deepseek41_device_forward.hip:1247-1263"},
}};
inline constexpr Family ds41{"deepseek-v4.1", "N", ds41_pieces, 2, 256, false, true,
                             "shipped relay: apps/deepseek41_serve_main.cpp:2874-2875"};

// Reuse the serializer's layout, including every vector length, header and digest; no second wire-size formula.
[[nodiscard]] inline std::uint64_t ds41_serialized_bytes(const deepseek41::AttentionState& state) {
    deepseek41::AttentionFrame frame;
    frame.token_count = state.tokens.size();
    for (const auto& layer : state.layers) {
        deepseek41::AttentionFrameLayer f;
        f.ring_bytes = layer.ring.size();
        f.kv_bytes = layer.kv.size();
        f.idx_bytes = layer.idx_keys.size();
        f.carry_floats = layer.carry.size();
        f.topk_count = layer.topk.size();
        frame.layers.push_back(f);
    }
    deepseek41::plan_attention_frame(frame);
    return frame.total;
}

// Qwen3.8-Flash-Next: 48 layers, QSA at (layer + 1) % 4 == 0 (apps/hip_qwen_prefill_mtp_handoff.inc:165), GDN on the
// other 36. On a pair each layer's state belongs to the stage that runs it (src/distributed_stage_state.cpp:76-104);
// PLE, n-gram ids and the MTP layer belong to stage A (input stage, apps/hip_qwen_prefill_mtp_handoff.inc:167-176);
// the position is replicated (8 B). rows = N; MTP rows stop one short.
inline constexpr std::array<Piece, 14> qwen_pieces{{
    {"qsa_kv", 0, Owner::stage, Extent::rows, 2048, 12, 3, 4, 1, 0, "src/hip_indexed_attention_launcher.hip:4384-4440"},
    {"qsa_ik", 1, Owner::stage, Extent::rows, 256, 12, 3, 4, 1, 0, "apps/hip_prefix_state_cache.hpp:349-373"},
    {"gdn_s_0", 2, Owner::stage, Extent::fixed, 48U * 128U * 128U * 4U, 12, 0, 4, 1, 0, "src/hip_gdn_launcher.hip:1490-1508"},
    {"gdn_s_1", 2, Owner::stage, Extent::fixed, 48U * 128U * 128U * 4U, 12, 1, 4, 1, 0, "src/hip_gdn_launcher.hip:1490-1508"},
    {"gdn_s_2", 2, Owner::stage, Extent::fixed, 48U * 128U * 128U * 4U, 12, 2, 4, 1, 0, "src/hip_gdn_launcher.hip:1490-1508"},
    {"gdn_conv_0", 3, Owner::stage, Extent::fixed, 10240U * 4U * 2U, 12, 0, 4, 1, 0, "src/hip_gdn_launcher.hip:1217-1248"},
    {"gdn_conv_1", 3, Owner::stage, Extent::fixed, 10240U * 4U * 2U, 12, 1, 4, 1, 0, "src/hip_gdn_launcher.hip:1217-1248"},
    {"gdn_conv_2", 3, Owner::stage, Extent::fixed, 10240U * 4U * 2U, 12, 2, 4, 1, 0, "src/hip_gdn_launcher.hip:1217-1248"},
    {"ple", 4, Owner::stage, Extent::fixed, 10240U * 9U * 2U, 1, 0, 1, 1, 0, "src/hip_ple_launcher.hip:1066-1094"},
    {"mtp_kv", 5, Owner::stage, Extent::rows, 2048, 1, 0, 1, 1, 1, "apps/hip_prefix_state_cache.hpp:317-331"},
    {"mtp_ik", 6, Owner::stage, Extent::rows, 256, 1, 0, 1, 1, 1, "apps/hip_prefix_state_cache.hpp:317-331"},
    {"mtp_held", 7, Owner::stage, Extent::fixed, 4U * 2560U * 2U, 1, 0, 1, 1, 0, "apps/hip_prefix_state_cache.hpp:317-331"},
    {"ngram_ids", 8, Owner::stage, Extent::fixed, 8, 1, 0, 1, 1, 0, "src/memory_plan.cpp:167-179"},
    {"position", 9, Owner::replica, Extent::fixed, 8, 1, 0, 1, 1, 0, "src/memory_plan.cpp:167-179"},
}};
inline constexpr Family qwen_pair{"qwen3.8-flash-next", "N", qwen_pieces, 2, 64, false, false, "no shared pieces"};
inline constexpr Family qwen_single{"qwen3.8-flash-next", "N", qwen_pieces, 1, 64, false, false, "single destination"};

inline constexpr std::array<const Family*, 3> all{&ds41, &qwen_pair, &qwen_single};

}  // namespace strix::pair::state::decl
