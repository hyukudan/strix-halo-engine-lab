#pragma once
// minimax-m3 decode sparse attention (MSA): per-head Gemma RMS norm and neox RoPE, the 4-head block indexer
// (max of q.k over each 128-position block), top-16 block selection per index head with the newest block forced,
// and grouped-query attention of 16 query heads per KV head over the selected blocks only.
// Host references whose arithmetic is the declared order of the planned device kernels (docs/minimax-msa.md), so a
// device implementation can be checked bit for bit; the scalar helpers below are shared by host and device code.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#if defined(__HIPCC__)
#define MMSA_HD __host__ __device__ inline
#else
#define MMSA_HD inline
#endif

namespace strix::minimax::msa {

inline constexpr unsigned kHeadDim = 128, kRotaryDim = 64, kRotaryHalf = 32;
inline constexpr unsigned kQHeads = 64, kKvHeads = 4, kGroup = kQHeads / kKvHeads;
inline constexpr unsigned kIdxHeads = 4, kIdxDim = 128;  // index head g selects for KV head g
inline constexpr unsigned kBlock = 128, kTopK = 16, kInitBlocks = 0, kLocalBlocks = 1;
inline constexpr unsigned kTile = 16, kChunkBlocks = 1, kMaxChunks = kTopK / kChunkBlocks;  // a chunk = one selected block
inline constexpr unsigned kLanes = 32, kDimsPerLane = kHeadDim / kLanes;
inline constexpr float kNormEps = 1e-6F;
inline constexpr float kScale = 0.08838834764831845F;  // 128^-0.5, main and index heads

MMSA_HD std::uint32_t f2u(float x) { return __builtin_bit_cast(std::uint32_t, x); }
MMSA_HD float u2f(std::uint32_t x) { return __builtin_bit_cast(float, x); }
MMSA_HD float bf16_to_float(std::uint16_t b) { return u2f(static_cast<std::uint32_t>(b) << 16U); }
MMSA_HD std::uint16_t float_to_bf16(float x) {  // round to nearest even (finite inputs)
    const std::uint32_t b = f2u(x);
    return static_cast<std::uint16_t>((b + 0x7FFFU + ((b >> 16U) & 1U)) >> 16U);
}
MMSA_HD float pow2f(int e) { return u2f(static_cast<std::uint32_t>(e + 127) << 23U); }
// exp: the deepseek41 declared exp (Cody-Waite + degree-8 polynomial, all fmaf); 0 below -87.
MMSA_HD float exp_f(float x) {
    if (x < -87.0F) return 0.0F;
    if (x > 88.0F) return __builtin_inff();
    const float n = __builtin_rintf(x * 1.44269504088896341F);
    float r = __builtin_fmaf(n, -0.693145751953125F, x);
    r = __builtin_fmaf(n, -1.42860676533018e-06F, r);
    float p = 2.48015873e-05F;
    p = __builtin_fmaf(p, r, 1.98412698e-04F);
    p = __builtin_fmaf(p, r, 1.38888889e-03F);
    p = __builtin_fmaf(p, r, 8.33333333e-03F);
    p = __builtin_fmaf(p, r, 4.16666667e-02F);
    p = __builtin_fmaf(p, r, 1.66666667e-01F);
    p = __builtin_fmaf(p, r, 0.5F);
    p = __builtin_fmaf(p, r, 1.0F);
    p = __builtin_fmaf(p, r, 1.0F);
    return p * pow2f(static_cast<int>(n));
}

// One wave holds a 128-vector, lane l owning dims 4l..4l+3. A "lane chain" is acc = fmaf(a, b, acc) over those 4
// dims in order from 0; the "butterfly" is v = v + v[lane ^ d] for d = 16, 8, 4, 2, 1 (every lane ends equal).
MMSA_HD void butterfly(float* lanes) {
    for (unsigned d = 16; d >= 1; d >>= 1U) {
        float next[kLanes];
        for (unsigned l = 0; l < kLanes; ++l) next[l] = lanes[l] + lanes[l ^ d];
        for (unsigned l = 0; l < kLanes; ++l) lanes[l] = next[l];
    }
}
MMSA_HD float dot128(const float* a, const float* b) {
    float lanes[kLanes];
    for (unsigned l = 0; l < kLanes; ++l) {
        float acc = 0.0F;
        for (unsigned i = 0; i < kDimsPerLane; ++i) acc = __builtin_fmaf(a[l * kDimsPerLane + i], b[l * kDimsPerLane + i], acc);
        lanes[l] = acc;
    }
    butterfly(lanes);
    return lanes[0];
}

// Per-position RoPE row (neox halves over dims 0..63, theta from the config): cos/sin computed in double.
struct RopeRow {
    std::array<float, kRotaryHalf> cos{}, sin{};
};
RopeRow rope_row(std::uint64_t position, double theta);

// In place on one 128 head: Gemma RMS norm with the stored weight w (scale = 1 + w), then neox RoPE.
//   r = 1 / sqrt(butterfly(lane chain of x*x) / 128 + eps); y_i = (x_i * r) * (1 + w_i);
//   i < 32: y_i' = fmaf(-y_{i+32}, sin_i, y_i * cos_i), y_{i+32}' = fmaf(y_i, sin_i, y_{i+32} * cos_i).
void head_norm_rope(float* x, const float* gemma_w, const RopeRow& rope);

// Index block scores of one row at position p (count = p + 1 visible positions): for block b,
// score_b = max over its visible positions t of dot128(q, k_t) * kScale. keys: [count][128].
void index_block_scores(const float* q, const float* keys, std::uint64_t count, std::span<float> out);
MMSA_HD std::uint64_t block_count(std::uint64_t count) { return (count + kBlock - 1) / kBlock; }

// Selection: all blocks when there are at most kTopK; else the newest block, then the best kTopK - 1 others by
// score (ties: the smaller block id). Returned in ascending block order.
std::vector<std::uint32_t> select_blocks(std::span<const float> block_scores);

// Attention of `heads` query heads (one KV group) over the selected blocks: keys/values [count][128] (already
// normed/roped keys; values as stored). Keys are visited in ascending position inside ascending blocks; chunks of
// kChunkBlocks blocks run an online softmax over tiles of kTile keys:
//   s_j = dot128(q, k_j) * kScale; tm = max over the tile; m' = max(m, tm); c = exp_f(m - m') (0 on the first tile);
//   l = l * c, then l = l + p_j with p_j = exp_f(s_j - m') in key order; acc = acc * c, then
//   acc = fmaf(p_j, v_j, acc) in key order; m = m'.
// Chunks combine in chunk order: M = max m_c; e_c = exp_f(m_c - M); L = fmaf(l_c, e_c, L);
// o = (fmaf chain over chunks of acc_c * e_c) / L.
void sparse_attention(const float* q, unsigned heads, const float* keys, const float* values, std::uint64_t count,
                      std::span<const std::uint32_t> blocks, float* out);

}  // namespace strix::minimax::msa
