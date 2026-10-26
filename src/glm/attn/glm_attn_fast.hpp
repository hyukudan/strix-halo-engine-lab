#pragma once
// GLM-5.3-Flash attention decode path, optimized schedule (design notes).
// Every launcher here produces the SAME BITS as the Tier A host emulation (src/glm/host/glm_host.cpp) and the
// batch-1/2 device kernels (src/glm/kernels/*) given the same inputs: the per-op orders R1, R2, R4, R5, R8-R13 of
// design notes are unchanged; only the schedule differs (which lane holds which partial, LDS staging, R rows
// sharing one weight read, fused launches). Build with -ffp-contract=off -fno-fast-math -fno-gpu-flush-denormals-to-zero.
// Layouts are the engine's (unchanged): KDA state [slot][head][k][v] fp32, conv state [slot][3 * C][3] fp32, DSA caches
// as glm_dsa.hpp DsaSlot. All pointers are device pointers. Rows = independent sequence slots unless stated.
#include "glm/kernels/glm_dsa.hpp"
#include "glm/kernels/glm_tier_a.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::glm::fast {

// F16 bits -> float, branch-free, equal to f16_to_float (glm_math.hpp) for all 65,536 inputs (gate: exhaustive check):
// the same fields, selected instead of branched (normal: exponent rebias; 0/subnormal: m * 2^-24 exactly, sign OR-ed;
// inf/NaN: payload << 13). Used inside the chain loops, where the branchy form costs registers and scratch.
__host__ __device__ inline float f16_bits_to_float(std::uint32_t h) {
    const std::uint32_t sign = (h & 0x8000U) << 16U, e = (h >> 10U) & 0x1fU, m = h & 0x3ffU;
    const std::uint32_t normal = sign | ((e + 112U) << 23U) | (m << 13U);
    const std::uint32_t special = sign | 0x7f800000U | (m << 13U);
    const std::uint32_t sub = sign | __builtin_bit_cast(std::uint32_t, static_cast<float>(m) * 5.9604644775390625e-8F);
    return __builtin_bit_cast(float, e == 0x1fU ? special : (e == 0 ? sub : normal));
}
// Gate hook: out[i] = f16_bits_to_float(i) on the device for i < 65536.
void f16_table(float* out, hipStream_t s);

inline constexpr unsigned kMaxRows = 8;
inline constexpr unsigned kMaxTok = kIdxTopk + kKpool - 1;  // 2051: selected pools x 4 + tail <= 3
inline constexpr unsigned kTokStride = 2052;                // per-row stride of token lists / scores / probabilities

// ---------------------------------------------------------------- R1 dense chains (F16 weights), fused launch
// y[r][o] = chain_i fmaf(x[r][i], W[o][i]) (R1), for up to 4 matrices in one launch; rows <= 8 share each weight read.
// Requirements: in % 32 == 0, w 16-byte aligned. x rows have stride x_stride floats, y rows y_stride floats.
// A job may have `groups` independent groups (e.g. one per head): group g uses w + g * w_gstride (elements),
// x + g * x_gstride, y + g * y_gstride and has n outputs.
struct DenseJob {
    const std::uint16_t* w = nullptr;  // [n][in] F16, first row of this rank's slice
    const float* x = nullptr;
    float* y = nullptr;
    unsigned n = 0, in = 0, x_stride = 0, y_stride = 0;
    unsigned groups = 1;
    std::size_t w_gstride = 0, x_gstride = 0, y_gstride = 0;
};
void dense_f16(const DenseJob* jobs, unsigned njobs, unsigned rows, hipStream_t s);
// Schedule of the dense chains and the MLA scores (same bits either way): kStaged (default; dense for in % 256 == 0: 16-row
// weight tiles streamed through double-buffered LDS by dedicated loader waves) or kLane (one lane streams its own row; dense always uses it for in < 512).
// Environment STRIX_GLM_ATTN_SCHED=lane forces kLane at first use; set_dense_schedule overrides (bench A/B).
enum class DenseSchedule { kStaged, kLane };
void set_dense_schedule(DenseSchedule d);
// Fused KDA step: lanes per state column (2 = round 3 schedule, 4 or 8 = more waves per head; same bits). Environment
// +16 = interleaved, +32 = 2-barrier trees, +64 = column split (s suffix). 1 = auto (default):
// 2c (column quad) at one row, cqs (column-quad split + shared prep) for 2..8 independent rows (attention measurement, same bits);
// kSeq stays 2. The column-quad paths need 16-byte aligned S and snapshot bases.
// STRIX_GLM_KDA_LANES at first use; this overrides it (gate / bench).
void set_kda_lanes(unsigned q);
// Indexer score selector: 1 (default), 2 or 4 (RG=4 uses RG=2 to fit 64 KiB LDS).
// 3 = two keys x two rows, 5 = two keys x one row; streamed queries, histogram-only LDS.
// STRIX_GLM_IDX_RG is read at first use (exactly 1/2/3/4/5; unset = 6 = auto: rk1 / rk2 / rg1 by pools and rows when the
// caller passes same_slot = true to indexer_score, else rg1); this setter overrides it.
void set_indexer_score_rows(unsigned rg);
// PV CTA heads x dims: 0 = 4 x 16 (default), 1 = 8 x 8, 2 = 16 x 4; same chains and tree.
// STRIX_GLM_MLA_PV is read at first use (exactly 0/1/2); this setter overrides it.
void set_mla_pv_variant(unsigned v);
// Staged dense tile: 16 (default) or 8 outputs per block (bench A/B; same bits).
void set_dense_tile(unsigned outputs);
// MLA scores schedule (same bits): auto by default (staged when mla_attention / mla_scores get max_tokens <= 1024 and rows
// <= 4, else lane); STRIX_GLM_ATTN_SCORES=staged / lane forces one;
// or this setter enables the staged form. STRIX_GLM_ATTN_SCHED=lane forces lane for everything.
void set_scores_schedule(DenseSchedule d);
void set_scores_auto();  // back to the attention auto schedule (default)

// ---------------------------------------------------------------- KDA (R8-R10), split variants add a finalizer
// rows independent slots: identical to dev::kda_step (prep + update + out) without the scratch round trips.
//   qkv [rows][3C], f / gate [rows][C], b [rows][H]; S [rows][H][128][128]; conv_state [rows][3C][3]; y [rows][C].
void kda_step(const dev::KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b, float* S,
              float* conv_state, float* y, unsigned* flag, unsigned rows, hipStream_t s);
// Benchmark hook: copy state through registers with the default Q2 ownership.
void kda_state_copy_floor(float* S, unsigned heads, unsigned rows, hipStream_t s);
// npos consecutive positions of ONE slot (speculative verify): equal to npos sequential kda_step calls with rows = 1.
// The recurrent and conv state stay in registers between positions; S / conv_state are the slot's own buffers.
// snap (optional, may be null): S after each position, [npos][H][128][128]; conv_snap (optional): [npos][3C][3].
void kda_steps_seq(const dev::KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b, float* S,
                   float* conv_state, float* y, float* snap, float* conv_snap, unsigned* flag, unsigned npos,
                   hipStream_t s);

// ---------------------------------------------------------------- DSA indexer (R4, R11, R12, R5, section 1.4)
// Per row r: slot slots[r], query position tpos[r] (device). The row's token t = tpos[r] must already be appended
// (dsa_store) before indexer_score / mla_* read the caches.
struct DsaRows {
    const dev::DsaSlot* slots = nullptr;  // [rows]
    const unsigned* tpos = nullptr;       // [rows]
    unsigned rows = 0;
};
// Cache writes for the rows' new tokens: LayerNorm(key) (R4), gate, latent -> bf16 (C3), pooled key on pool completion
// (R12, C4). key / gate are the R1 outputs of wk / compress gate ([rows][128]); latent [rows][512] normed.
// Also clears the per-row radix histogram hist0 [rows][256] used by indexer_score. Two launches (cache writes, then
// pool completion), so rows may append consecutive positions of the same slot in one call.
void dsa_store(const DsaRows& d, const float* latent, const float* key, const float* gate, const dev::IndexerDev& w,
               std::uint32_t* hist0, unsigned* flag, hipStream_t s, unsigned flag_stride = 0);
// flag_stride 0: every row ORs into *flag; 1: flag[r] per row (C3 of row r, C4 of the row closing the pool).
// R11 scores of every visible pool (P_r = (t_r + 1) / 4) -> ordered keys [rows][key_stride] + top-byte histogram.
// wraw = R1 output of weights_proj [rows][32] (the * 32^-1/2 scale is applied here, as in the host). qI [rows][32][128].
// p_bound: grid bound (>= every P_r), key_stride >= p_bound.
void indexer_score(const DsaRows& d, const float* qI, const float* wraw, std::uint32_t* keys, unsigned key_stride,
                   std::uint32_t* hist0, unsigned p_bound, hipStream_t s, bool same_slot = false);
// R5 exact top-k (k = min(512, P)) + canonical token list: tokens [rows][kTokStride], n [rows].
void indexer_select(const DsaRows& d, const std::uint32_t* keys, unsigned key_stride, const std::uint32_t* hist0,
                    std::uint32_t* tokens, std::uint32_t* n, hipStream_t s);

// ---------------------------------------------------------------- absorbed MLA (R13), H rank heads
// q [rows][H][256]; kv_b rank rows [H * 512][512] F16 (K rows h*512 + d, V rows h*512 + 256 + d).
// Workspace: qt [rows][H][512], mkey [rows][H], sc / p [rows][H][kTokStride], ob [rows][H][512]. o [rows][H][256].
struct MlaWork {
    float* qt;
    std::uint32_t* mkey;
    float* sc;
    float* p;
    float* ob;
};
// max_tokens (optional, 0 = unknown): host upper bound on n[r] (visible tokens); picks the scores schedule when no
// schedule is forced (auto: staged when max_tokens <= 1024 and rows <= 4, repeated measurements on both nodes; same bits).
void mla_attention(const DsaRows& d, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                   const std::uint16_t* kv_b, unsigned heads, const MlaWork& w, float* o, hipStream_t s, unsigned max_tokens = 0);
// The stages alone (bench hooks; mla_attention = these in order).
void mla_absorb(const float* q, const std::uint16_t* kv_b, unsigned heads, float* qt, std::uint32_t* mkey, unsigned rows,
                hipStream_t s);
void mla_scores(const DsaRows& d, const std::uint32_t* tokens, const std::uint32_t* n, const float* qt, unsigned heads,
                float* sc, std::uint32_t* mkey, hipStream_t s, unsigned max_tokens = 0);
void mla_softmax(const std::uint32_t* n, const float* sc, const std::uint32_t* mkey, unsigned heads, float* p,
                 unsigned rows, hipStream_t s);
void mla_pv(const DsaRows& d, const std::uint32_t* tokens, const std::uint32_t* n, const float* p, unsigned heads,
            float* ob, hipStream_t s);
void mla_out(const float* ob, const std::uint16_t* kv_b, unsigned heads, float* o, unsigned rows, hipStream_t s);

}  // namespace strix::glm::fast
