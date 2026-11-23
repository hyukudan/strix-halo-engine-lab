#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Frozen K5c baselines from src/cuda/models/minimax/mm_rtx_kernels.cu
// and its header at 29f8c9e64. The original score and pick guards are retained.
namespace strix::minimax::rtxpf::cuda::k5c_baseline {
constexpr unsigned kBlock = 128, kTopK = 16;
constexpr float kScale = 0.08838834764831845F;

// One prompt sequence: rows [row0, row0 + rows) have positions [pos0, pos0 + rows). Default pos0=0 preserves full prefill.
// Caches are this sequence's, every layer: k/v [layers][cap][4][128] BF16, ik [layers][cap][128] BF16.
struct SeqView {
    std::uint32_t row0, rows, cap;
    std::uint16_t *k, *v, *ik;
    std::uint32_t pos0 = 0;  // absolute position of row0; zero preserves whole-prompt calls
};


__device__ __forceinline__ float u2f(std::uint32_t b) { return __uint_as_float(b); }
__device__ __forceinline__ float bf16_to_float(std::uint16_t b) { return u2f(static_cast<std::uint32_t>(b) << 16U); }
__device__ __forceinline__ float fmaxs(float a, float b) { return a < b ? b : a; }  // std::max(a, b)
__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000U); }
__device__ __forceinline__ float shx(float v, int d) { return __shfl_xor_sync(0xffffffffU, v, d, 32); }

// dot128 of one thread: the 32 lane chains (4 dims each, in order) and the butterfly's tree for lanes[0].
__device__ __forceinline__ float dot128_thread(const float* a, const float* b) {
    float c[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        float acc = 0.0F;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc = __fmaf_rn(a[4 * l + i], b[4 * l + i], acc);
        c[l] = acc;
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1)
#pragma unroll
        for (int l = 0; l < d; ++l) c[l] = c[l] + c[l + d];
    return c[0];
}

// Index block scores: one warp per (row, index head g, block b); lane j scores keys b*128 + j + 32 m (full dot128 per
// lane, the reference's lane chains + tree), max over the visible keys (comparisons only feed the selection).
__global__ void index_scores_kernel(const SeqView* seqs, const std::uint32_t* row_seq, std::size_t rows, unsigned max_blocks,
                                    unsigned layer, const float* iq, float* scores) {
    __shared__ float qs[4][128];
    const unsigned w = threadIdx.x / 32, lane = threadIdx.x & 31U;
    const std::size_t item = static_cast<std::size_t>(blockIdx.x) * 4 + w;  // (row, g, b)
    const std::size_t row = item / (4 * max_blocks);
    const unsigned g = static_cast<unsigned>(item / max_blocks % 4), b = static_cast<unsigned>(item % max_blocks);
    const bool live = row < rows;
    SeqView sq{};
    std::uint64_t count = 0;
    if (live) {
        sq = seqs[row_seq[row]];
        count = row - sq.row0 + sq.pos0 + 1;
        for (unsigned i = lane; i < 128; i += 32) qs[w][i] = iq[(row * 4 + g) * 128 + i];
    }
    __syncwarp();
    if (!live || static_cast<std::uint64_t>(b) * kBlock >= count) return;
    const std::uint64_t blocks = (count + kBlock - 1) / kBlock;
    if (blocks <= kTopK) return;  // dense: no scores needed
    float best = neg_inf();
    float kv[128];
    for (unsigned m = 0; m < 4; ++m) {
        const std::uint64_t t = static_cast<std::uint64_t>(b) * kBlock + lane + 32 * m;
        if (t >= count) break;
        const std::uint16_t* kc = sq.ik + (static_cast<std::size_t>(layer) * sq.cap + t) * 128;
#pragma unroll 8
        for (int i = 0; i < 128; ++i) kv[i] = bf16_to_float(kc[i]);
        best = fmaxs(best, dot128_thread(qs[w], kv) * kScale);
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) best = fmaxs(best, shx(best, d));
    if (lane == 0) scores[(row * 4 + g) * max_blocks + b] = best;
}

// Selection per (row, g): all blocks when there are at most 16; else the newest + the best 15 others (score desc, id asc),
// ascending block order.
__global__ void select_kernel(const SeqView* seqs, const std::uint32_t* row_seq, std::size_t rows, unsigned max_blocks,
                              const float* scores, std::uint32_t* sel, std::uint32_t* sel_count) {
    const std::size_t item = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= rows * 4) return;
    const std::size_t row = item / 4;
    const SeqView sq = seqs[row_seq[row]];
    const std::uint64_t count = row - sq.row0 + sq.pos0 + 1, blocks = (count + kBlock - 1) / kBlock;
    std::uint32_t* out = sel + item * kTopK;
    if (blocks <= kTopK) {
        for (std::uint32_t b = 0; b < blocks; ++b) out[b] = b;
        sel_count[item] = static_cast<std::uint32_t>(blocks);
        return;
    }
    const float* sc = scores + item * max_blocks;
    const auto newest = static_cast<std::uint32_t>(blocks - 1);
    std::uint32_t pick[kTopK];
    for (unsigned k = 0; k + 1 < kTopK; ++k) {
        std::uint32_t best = 0xffffffffU;
        for (std::uint32_t b = 0; b < newest; ++b) {
            bool taken = false;
            for (unsigned j = 0; j < k; ++j) taken |= pick[j] == b;
            if (taken) continue;
            if (best == 0xffffffffU || sc[b] > sc[best]) best = b;  // ties keep the smaller id
        }
        pick[k] = best;
    }
    pick[kTopK - 1] = newest;
    for (unsigned i = 1; i < kTopK; ++i)  // ascending
        for (unsigned j = i; j > 0 && pick[j - 1] > pick[j]; --j) {
            const std::uint32_t tmp = pick[j];
            pick[j] = pick[j - 1];
            pick[j - 1] = tmp;
        }
    for (unsigned i = 0; i < kTopK; ++i) out[i] = pick[i];
    sel_count[item] = kTopK;
}

// Index block scores, tiled (same arithmetic as index_scores_kernel): CTA = (32-row tile of ONE sequence, index head g,
// block b); the block's 128 keys and the tile's 32 query rows staged in shared memory as FP32; thread t scores rows
// t / 8 against keys (t % 8) + 8 j: per dot the reference's 32 lane chains (one float4 of each operand) and its tree;
// max over the visible keys (comparison only, so the reduction order does not matter). Key rows are padded to 33 float4
// (the 8 lanes of a quarter-warp read 8 different keys: at 32 float4 per row they all hit the same banks).
constexpr unsigned kKs4 = 33;
__global__ __launch_bounds__(256) void index_scores_tiled_kernel(const SeqView* seqs, const std::uint32_t* row_seq, const uint2* tiles,
                                                                 unsigned max_blocks, unsigned layer, const float* iq, float* scores) {
    extern __shared__ float4 smem4[];
    float4* ks = smem4;                 // [128 keys][kKs4 float4]
    float4* qs = smem4 + 128 * kKs4;    // [32 rows][32 float4]
    const uint2 tile = tiles[blockIdx.x];  // (first row, rows)
    const unsigned g = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const SeqView sq = seqs[row_seq[tile.x]];
    const std::uint64_t last_count = tile.x + tile.y - sq.row0 + sq.pos0;  // visible positions of the tile's last row
    if (static_cast<std::uint64_t>(b) * kBlock >= last_count || (last_count + kBlock - 1) / kBlock <= kTopK) return;
    for (unsigned i = t; i < 128 * 32; i += 256) {
        const unsigned key = i / 32, q4 = i % 32;
        const std::uint64_t pos = static_cast<std::uint64_t>(b) * kBlock + key;
        float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        if (pos < last_count) {
            const std::uint16_t* kc = sq.ik + (static_cast<std::size_t>(layer) * sq.cap + pos) * 128 + 4 * q4;
            v = make_float4(bf16_to_float(kc[0]), bf16_to_float(kc[1]), bf16_to_float(kc[2]), bf16_to_float(kc[3]));
        }
        ks[key * kKs4 + q4] = v;
    }
    for (unsigned i = t; i < 32 * 32; i += 256) {
        const unsigned r = i / 32, q4 = i % 32;
        qs[i] = r < tile.y ? *reinterpret_cast<const float4*>(iq + ((static_cast<std::size_t>(tile.x) + r) * 4 + g) * 128 + 4 * q4)
                           : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    }
    __syncthreads();
    const unsigned r = t / 8;
    const std::uint64_t count = static_cast<std::uint64_t>(tile.x) + r - sq.row0 + sq.pos0 + 1;
    float best = neg_inf();
    if (r < tile.y && (count + kBlock - 1) / kBlock > kTopK) {
        for (unsigned j = 0; j < 16; ++j) {
            const unsigned key = (t % 8) + 8 * j;
            if (static_cast<std::uint64_t>(b) * kBlock + key >= count) continue;
            float c[32];
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                const float4 a = qs[r * 32 + l], k = ks[key * kKs4 + l];
                float acc = __fmaf_rn(a.x, k.x, 0.0F);
                acc = __fmaf_rn(a.y, k.y, acc);
                acc = __fmaf_rn(a.z, k.z, acc);
                acc = __fmaf_rn(a.w, k.w, acc);
                c[l] = acc;
            }
#pragma unroll
            for (int d = 16; d >= 1; d >>= 1)
#pragma unroll
                for (int l = 0; l < d; ++l) c[l] = c[l] + c[l + d];
            best = fmaxs(best, c[0] * kScale);
        }
    }
#pragma unroll
    for (int d = 4; d >= 1; d >>= 1) best = fmaxs(best, __shfl_xor_sync(0xffffffffU, best, d, 8));
    if ((t % 8) == 0 && r < tile.y && (count + kBlock - 1) / kBlock > kTopK && static_cast<std::uint64_t>(b) * kBlock < count)
        scores[((static_cast<std::size_t>(tile.x) + r) * 4 + g) * max_blocks + b] = best;
}

} // namespace strix::minimax::rtxpf::cuda::k5c_baseline
