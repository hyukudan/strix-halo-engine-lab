#pragma once
// minimax-m3 MSA decode kernels for gfx1151 (docs/minimax-msa.md). Every kernel follows the declared arithmetic of
// strix/minimax_msa.hpp, so its output equals the host reference bit for bit. Rows are consecutive positions of one
// call (1..8 for decode, more for prefill windows); every row's visible positions must already be in the caches.
// Head counts are per rank (tensor parallel: 2 KV heads, 2 index heads, 32 query heads; or the full 4/4/64).
#include <cstdint>

#include <hip/hip_runtime.h>

#include "strix/minimax_msa.hpp"

namespace strix::minimax::msa::hip {

// K1: head norm + RoPE in place for q and the index q; K, V and the index key of each row written to the BF16 caches
// at slot = position (caches are indexed by position, [capacity][kv_heads][128] and [capacity][128]).
struct PrepView {
    std::uint32_t rows, q_heads, kv_heads, idx_heads;
    float* q;                     // R x q_heads x 128, raw in, normed/roped out
    const float* k;               // R x kv_heads x 128 raw
    const float* v;               // R x kv_heads x 128
    float* iq;                    // R x idx_heads x 128, raw in, normed/roped out
    const float* ik;              // R x 128 raw
    const float* q_w;             // 128 stored Gemma weights (scale 1 + w), as float
    const float* k_w;
    const float* iq_w;
    const float* ik_w;
    const RopeRow* rope;          // R
    const std::uint64_t* pos;     // R
    std::uint16_t* k_cache;       // capacity x kv_heads x 128
    std::uint16_t* v_cache;
    std::uint16_t* ik_cache;      // capacity x 128
};
void enqueue_prep(const PrepView& v, hipStream_t stream);

// K2: per (row, index head, block) the block score max_t dot128(iq, ik_t) * kScale over visible positions.
// A thread owns one position; the key is read once for every row and head of the call.
struct ScoreView {
    std::uint32_t rows, idx_heads, max_blocks;  // max_blocks >= every row's block_count(pos + 1)
    const float* iq;              // R x idx_heads x 128 (normed/roped)
    const std::uint16_t* ik_cache;
    const std::uint64_t* pos;     // R
    float* scores;                // R x idx_heads x max_blocks (local block index when sharded)
    // Context-parallel indexer: this call scores only the global blocks b = j * shard_count + shard_rank, written at
    // local index j. Default: one shard (every block).
    std::uint32_t shard_rank = 0, shard_count = 1;
};
void enqueue_scores(const ScoreView& v, hipStream_t stream);

// K3: selection per (row, head): select_blocks() of the host reference, ascending block ids.
struct SelectView {
    std::uint32_t rows, heads, max_blocks;
    const std::uint64_t* pos;
    const float* scores;          // R x heads x max_blocks
    std::uint32_t* idx;           // R x heads x kTopK
    std::uint32_t* idx_count;     // R x heads
};
void enqueue_select(const SelectView& v, hipStream_t stream);

// Context-parallel selection, exact: each shard keeps its best kTopK - 1 blocks (the newest excluded) as
// (order key, global id); the head's owner merges both shards' lists and adds the newest block. The result equals
// enqueue_select on the full scores (the global top kTopK - 1 is always inside the union of the shards' lists).
// The candidate lists are what the shards exchange (kTopK x 8 B per row and head).
struct CandidateView {
    std::uint32_t rows, heads, max_blocks, shard_rank, shard_count;
    const std::uint64_t* pos;
    const float* scores;          // R x heads x max_blocks, local block index (ScoreView with the same shard)
    std::uint32_t* cand_key;      // R x heads x kTopK
    std::uint32_t* cand_id;       // R x heads x kTopK (global block ids, ascending)
    std::uint32_t* cand_count;    // R x heads
};
void enqueue_candidates(const CandidateView& v, hipStream_t stream);

struct MergeView {
    std::uint32_t rows, heads;
    const std::uint64_t* pos;
    const std::uint32_t* cand_key[2];   // per shard, R x heads x kTopK (shard order does not matter)
    const std::uint32_t* cand_id[2];
    const std::uint32_t* cand_count[2];
    std::uint32_t* idx;           // R x heads x kTopK
    std::uint32_t* idx_count;     // R x heads
};
void enqueue_merge(const MergeView& v, hipStream_t stream);

// K4: attention of the kGroup query heads of each local KV head over its selected blocks. One block per
// (row, KV head, chunk) writes the chunk partial (m, l, acc[128]); then one wave per (row, query head) combines.
struct AttentionView {
    std::uint32_t rows, kv_heads;
    const float* q;               // R x (kv_heads * kGroup) x 128 (normed/roped)
    const std::uint16_t* k_cache; // capacity x kv_heads x 128
    const std::uint16_t* v_cache;
    const std::uint64_t* pos;
    const std::uint32_t* idx;     // R x kv_heads x kTopK
    const std::uint32_t* idx_count;
    float* partial;               // R x (kv_heads * kGroup) x kMaxChunks x (2 + 128)
    float* o;                     // R x (kv_heads * kGroup) x 128
};
void enqueue_attention(const AttentionView& v, hipStream_t stream);

}  // namespace strix::minimax::msa::hip
