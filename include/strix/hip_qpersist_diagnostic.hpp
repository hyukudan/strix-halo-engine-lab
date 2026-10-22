#pragma once

#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <span>

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
} // namespace strix::qpersist_diagnostic
