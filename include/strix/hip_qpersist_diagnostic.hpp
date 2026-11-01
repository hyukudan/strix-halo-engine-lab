#pragma once

#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <span>
#include <string>

namespace strix::qpersist_diagnostic {
// Component qualification only. Caller owns disjoint device buffers and an
// explicit stream through completion. No allocation, fence or status reset.
struct AttentionRow {
    const float* query = nullptr;
    const std::uint32_t* selected = nullptr;
    float* scores = nullptr;
    float* output = nullptr;
    std::uint32_t count = 0;
};
struct Attention {
    std::span<const AttentionRow> rows;
    const std::uint16_t* kv = nullptr;
    std::uint32_t heads = 0, kv_heads = 0, dim = 0, maximum = 0;
    // Optional dense codes [row][heads*dim], deltas [(head*dim+feature)/32][64].
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
};
void enqueue_scores(const Attention&, bool tiled, hipStream_t);
void enqueue_tail(const Attention&, bool fused, hipStream_t);
// Window attention v2 (STRIX_QSA_WINDOW_ATTN_V2) score as the engine launches it for verify windows: the operand-pinned
// KV-head-grouped score (12 query heads per wave; 8 slots at 1-2 rows, 16 at 4-8) in attention_score_rows_wave_kernel<R,
// 4>'s operand order (head_dim 256). R1 is exercised here only as a kernel shape; the engine routes 2/4/8-row windows.
bool window_v2_admitted(const Attention&);
void enqueue_window_v2_scores(const Attention&, hipStream_t);
struct Cache {
    const float* index = nullptr;
    const float* key = nullptr;
    const float* value = nullptr;
    std::uint16_t* raw_state = nullptr;
    std::uint16_t* kv_state = nullptr;
    std::uint32_t index_dim = 0, kv_width = 0;
    std::uint64_t position = 0, capacity = 0;
    std::uint32_t* status = nullptr;
};
void enqueue_cache(const Cache&, bool fused, hipStream_t);
// MTP draft attention (Q7 arm E): one draft step's score and tail launches as the engine issues
// them. Score variants: "w4" (production), "w8", "w16", "g2s8", "g4s8", "g4s16"; tail variants:
// "incumbent" (production), "g32", "s8", "s16", "s32", "s64", "p32", "v2", "v4". Optional dynamic selection
// (position/status/ratio/topk) as the graph path passes it; null position = the MTP path.
struct DraftAttention {
    const float* query = nullptr;
    const std::uint32_t* selected = nullptr;
    float* scores = nullptr;
    float* output = nullptr;
    const std::uint16_t* kv = nullptr;
    std::uint32_t heads = 0, kv_heads = 0, dim = 0, maximum = 0, count = 0;
    const std::uint64_t* position = nullptr;
    std::uint64_t context = 0;
    std::uint32_t* status = nullptr;
    std::uint32_t ratio = 0, topk = 0;
};
bool draft_attention_admitted(const DraftAttention&, const char* score_variant, const char* tail_variant);
void enqueue_draft_scores(const DraftAttention&, const char* variant, hipStream_t);
void enqueue_draft_tail(const DraftAttention&, const char* variant, hipStream_t);
// The MTP draft step's score where it runs the one-block form (no STRIX_QPERSIST_DRAFT_SCORE), as the engine launches
// it: attention_score_kernel, or with v2 (STRIX_QSA_WINDOW_ATTN_V2) four query heads per K row in its operand order
// (head_dim 256); tail = also the engine's unfused softmax and value/gate launches. No dynamic position.
void enqueue_draft_one_block(const DraftAttention&, bool v2, bool tail, hipStream_t);
// STRIX_QSA_WINDOW_ATTN_V2 resolution and routing as the engine decides them.
bool window_v2_switch(const char* setting, const char* defaults, bool baseline);
struct WindowV2Route {
    bool enabled = true, verify_window = true, int8 = false, fast = false, window_grouped = false;
    unsigned rows = 2U, score_slots = 4U, heads = 24U, kv_heads = 2U, dim = 256U;
};
bool window_v2_routed(const WindowV2Route&);
bool draft_v2_routed(bool enabled, bool draft, bool qpersist_draft_score, unsigned heads, unsigned kv_heads, unsigned dim);
// Two-launch tails ("v2", "v4") only: the softmax launch (value == false) or the value launch alone.
void enqueue_draft_tail_stage(const DraftAttention&, const char* variant, bool value, hipStream_t);
// The engine's resolution of STRIX_QPERSIST_DRAFT_ATTN / _SCORE / _TAIL (null = unset): "incumbent" or
// "<score>+<tail> (default|explicit)"; throws on an unknown value.
std::string draft_attention_selection(const char* attn, const char* score, const char* tail);
} // namespace strix::qpersist_diagnostic
