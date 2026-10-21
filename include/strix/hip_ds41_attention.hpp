#pragma once
// gfx1151 kernels of the deepseek41 decode attention; arithmetic = the host references in ds41_attention.hpp.
// All pointers are device memory; nothing allocates or synchronizes. Rows of one call are consecutive positions.
#include "strix/ds41_attention.hpp"

#include <hip/hip_runtime.h>

namespace strix::ds41::attn::hip {

// RoPE tables are per row: the rows' own positions (and, for compress, the group's first position).
struct WindowView {
    const float* kv;              // R x 512
    const float* norm;            // 512
    const RopeRow* rope;          // R, at pos[r]
    const std::uint64_t* pos;     // R
    std::uint32_t rows, capacity;
    std::uint8_t* ring;           // capacity x 528
};
void enqueue_window_write(const WindowView& v, hipStream_t stream);

struct CompressView {
    std::uint32_t ratio, rows;
    const float* kv;              // R x 512
    const float* score;           // R x 512 (ratio 2)
    const float* norm;            // 512
    const std::uint16_t* wk;      // 128 x 512 F16
    const float* knorm;           // 128
    const RopeRow* rope;          // R, at the group's first position (pos / ratio * ratio)
    const std::uint64_t* pos;     // R
    const float* state_in;        // ratio 2: 1024 floats (kv, score) of the pending group's first token, valid flag below
    const std::uint32_t* state_valid_in;
    float* state_out;             // ratio 2: the pending group after the last row
    std::uint32_t* state_valid_out;
    std::uint8_t* kv_cache;       // entries of 288 B, index pos / ratio
    std::uint8_t* idx_cache;      // entries of 68 B
};
void enqueue_compress(const CompressView& v, hipStream_t stream);
// STRIX_DS41_COMPRESS_AHEAD=1 (default off): the index key weights of each thread are read three 64-weight steps ahead,
// the first ones before the latent barrier; the same chains, the same bytes. Tests: 0 off, 1 on, -1 the environment.
void set_compress_ahead(int mode);

struct IndexView {
    std::uint32_t rows, ratio, max_count;   // max_count >= every row's (pos + 1) / ratio
    const float* q;               // R x 32 x 128
    const float* w;               // R x 32
    const RopeRow* rope;          // R
    const std::uint64_t* pos;     // R
    const std::uint8_t* idx_cache;
    float* scores;                // R x max_count
    void* qprep = nullptr;        // scratch R x kIdxPrepBytes (required): quantised queries, scales and head weights
};
inline constexpr unsigned kIdxPrepBytes = (kIdxHeads * 32 + kIdxHeads * 4 + kIdxHeads) * 4;
void enqueue_index_scores(const IndexView& v, hipStream_t stream);

struct SelectView {
    std::uint32_t rows, ratio, max_count, topk, cand_blocks, block_size;
    Pool pool;
    const std::uint64_t* pos;     // R (count = (pos + 1) / ratio)
    const float* scores;          // R x max_count
    std::uint8_t* candidates;     // R x ceil(max_count / block_size): written by the source, read by users
    std::uint32_t* keys;          // scratch R x max_count
    std::uint32_t* idx;           // R x topk, ascending
    std::uint32_t* idx_count;     // R
};
void enqueue_select(const SelectView& v, hipStream_t stream);

// heads: the query heads of this call (all 64, or a tensor-parallel rank's 32): q, sink, o and partial hold them
// contiguously (head index local to the call); the shared KV head is the same for every head.
struct AttentionView {
    std::uint32_t rows, capacity, topk;
    const float* q;               // R x heads x 512
    const float* sink;            // heads
    const RopeRow* rope;          // R
    const std::uint64_t* pos;     // R
    const std::uint8_t* ring;
    const std::uint8_t* kv_cache; // may be null when idx_count is null (window-only layer)
    const std::uint32_t* idx;     // R x topk
    const std::uint32_t* idx_count; // R, or null
    float* partial;               // scratch R x heads x kMaxChunks x (2 + 512)
    float* o;                     // R x heads x 512
    std::uint32_t* tickets = nullptr; // R x kAttnGroupCount counters, zero before the first call and left zero by every
                                  // call: the last chunk block of each (row, head group) combines; null: combine kernel
    std::uint32_t heads = kHeads; // multiple of 8
    // optional A8 of o (int8 per 32, enqueue_quantize_a8's bits) written by the combine: codes R x heads x 512,
    // deltas / sums R x heads x 16
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    std::int32_t* sums = nullptr;
};
inline constexpr unsigned kAttnGroupCount = 8;   // at most 8 head groups per row (8 heads per block)
void enqueue_attention(const AttentionView& v, hipStream_t stream);
#if defined(STRIX_DS41_ATTN_TIMING)
// timing build: slot k (0 window .. 6 combine) x 65536 blocks x {wall0, wall1, cycles0, cycles1}
void set_timing_buffer(void* p);
#endif

}  // namespace strix::ds41::attn::hip
