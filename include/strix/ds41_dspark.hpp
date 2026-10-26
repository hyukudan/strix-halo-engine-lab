#pragma once

// deepseek41 DSpark drafter: host reference of the draft model's own layers in the declared
// arithmetic of the device kernels (hip_ds41_dspark.hpp), so that host and device drafts are
// bitwise comparable, plus the greedy verify protocol of a draft window.
//
// The drafter (three stages stored as mtp.0-2 in the draft file) reads the target's hidden states:
// for every committed target row, the mean over the four hyper-connection copies of the input
// stream of target layers 37, 38 and 39, concatenated (15,360 values). main_proj + main_norm turn
// it into main_x; every stage keeps a 128-entry FP8 ring of its own kv(main_x). A draft step embeds
// a block of five tokens (the last committed token, then four noise tokens) with the target's
// token embedding, runs the three stages (window-only attention over the ring plus the whole block,
// bidirectional inside the block; 128 MXFP4 experts top-3 plus a shared expert), and the target's
// output head with a sequential Markov bias picks one token per block row. A confidence head scores
// every drafted position.
//
// Numerical identity "ds41-dspark-v1" (declared here, not the model's training arithmetic):
// - Q8_0 linears use A8 activations (ds41_formats.hpp) and the declared Q8_0 product.
// - F16/F32 weights use FP32 products in the "wave order": lane l accumulates the terms l, l + 32,
//   ... in ascending order (acc = acc + a * b from 0, no fused operations), then the lanes are
//   combined by the butterfly v[l] += v[l ^ d], d = 16, 8, 4, 2, 1. Sums of squares use the same order.
// - The routed and shared experts are the MoE window of ds41_moe.hpp.
// Draft numerics never change the output tokens: the target verifies every drafted token greedily.

#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_DSPARK_HD __host__ __device__
#else
#define STRIX_DSPARK_HD
#endif

namespace strix::ds41::dspark {

inline constexpr std::uint32_t hidden = 5120;
inline constexpr std::uint32_t hc = 4;
inline constexpr std::uint32_t stream_width = hc * hidden;        // 20480
inline constexpr std::uint32_t mix_count = (2U + hc) * hc;        // 24: pre[4], post[4], comb[4][4]
inline constexpr std::uint32_t block_size = 5;
inline constexpr std::uint32_t window = 128;
inline constexpr std::uint32_t heads = 64;
inline constexpr std::uint32_t head_dim = 512;
inline constexpr std::uint32_t rope_dim = 64;
inline constexpr std::uint32_t q_rank = 1280;
inline constexpr std::uint32_t q_width = heads * head_dim;        // 32768
inline constexpr std::uint32_t o_groups = 8;
inline constexpr std::uint32_t o_group_width = q_width / o_groups; // 4096
inline constexpr std::uint32_t o_rank = 1024;
inline constexpr std::uint32_t o_width = o_groups * o_rank;       // 8192
inline constexpr std::uint32_t stages = 3;
inline constexpr std::uint32_t target_layers = 3;                 // 37, 38, 39
inline constexpr std::uint32_t main_width = target_layers * hidden; // 15360
inline constexpr std::uint32_t vocab = 129280;
inline constexpr std::uint32_t markov_rank = 256;
inline constexpr std::uint32_t confidence_width = hidden + markov_rank; // 5376
inline constexpr std::uint32_t experts = 128;
inline constexpr std::uint32_t top_k = 3;
inline constexpr std::uint32_t inter = 2304;
inline constexpr std::uint32_t sinkhorn_iterations = 20;
inline constexpr std::uint32_t noise_token = 128799;
inline constexpr std::uint32_t entry_bytes = head_dim + head_dim / 32U; // 512 E4M3 codes + 16 E8M0 scales
inline constexpr float norm_eps = 1e-20F;
inline constexpr float hc_eps = 1e-6F;
inline constexpr float route_scale = 1.5F;
inline constexpr float swiglu_limit = 10.0F;
inline constexpr float rope_theta = 10000.0F;
inline constexpr float attention_scale = 0.0441941738241592F; // 512^-0.5 rounded to FP32
inline constexpr float fp8_max = 448.0F;
inline constexpr float fp8_max_inverse = 1.0F / 448.0F;
inline constexpr float fp8_amax_floor = 1e-4F;

// ---- scalar helpers shared by host and device ----

STRIX_DSPARK_HD inline float sigmoid(float x) { return 1.0F / (1.0F + declared_exp(-x)); }

// E4M3 (finite variant) encoding: round to nearest even, saturating at +-448.
STRIX_DSPARK_HD inline std::uint8_t encode_e4m3(float x) {
    const auto bits = __builtin_bit_cast(std::uint32_t, x);
    const std::uint8_t sign = (bits >> 31U) != 0U ? 0x80U : 0U;
    float a = x < 0.0F ? -x : x;
    if (!(a == a)) return 0x7fU;
    if (a > fp8_max) a = fp8_max;
    std::uint32_t code = 0;
    if (a < 0.015625F) {
        code = static_cast<std::uint32_t>(__builtin_rintf(a * 512.0F));
    } else {
        int e = static_cast<int>((__builtin_bit_cast(std::uint32_t, a) >> 23U) & 0xffU) - 127;
        const float mantissa = a * detail::pow2(-e);
        float q = __builtin_rintf((mantissa - 1.0F) * 8.0F);
        if (q == 8.0F) { e = e + 1; q = 0.0F; }
        code = (static_cast<std::uint32_t>(e + 7) << 3U) | static_cast<std::uint32_t>(q);
    }
    return static_cast<std::uint8_t>(code | sign);
}
STRIX_DSPARK_HD inline float decode_e4m3(std::uint8_t code) {
    const std::uint32_t e = (code >> 3U) & 15U, m = code & 7U;
    float v = e == 0U ? static_cast<float>(m) * 0.001953125F
                      : __builtin_bit_cast(float, ((e + 120U) << 23U) | (m << 20U));
    if (e == 15U && m == 7U) v = __builtin_bit_cast(float, 0x7fc00000U);
    return (code & 0x80U) != 0U ? -v : v;
}
// FP8 group scale exponent of an amax: 2^ceil(log2(max(amax, 1e-4) / 448)), as a biased E8M0 byte.
STRIX_DSPARK_HD inline std::uint8_t fp8_scale_code(float amax) {
    const float a = (amax > fp8_amax_floor ? amax : fp8_amax_floor) * fp8_max_inverse;
    const auto b = __builtin_bit_cast(std::uint32_t, a);
    const int e = static_cast<int>((b >> 23U) & 0xffU) - 127 + ((b & 0x7fffffU) != 0U ? 1 : 0);
    return static_cast<std::uint8_t>(e + 127);
}
STRIX_DSPARK_HD inline float scale_value(std::uint8_t code) { return detail::pow2(static_cast<int>(code) - 127); }
// One FP8 element: code = E4M3(clamp(x / s, +-448)).
STRIX_DSPARK_HD inline std::uint8_t fp8_element(float x, float s) {
    float v = x / s;
    v = v < -fp8_max ? -fp8_max : (v > fp8_max ? fp8_max : v);
    return encode_e4m3(v);
}
// RoPE on one pair (adjacent elements 2i, 2i + 1), no fused operations.
STRIX_DSPARK_HD inline void rope_pair(float& x0, float& x1, float c, float s, bool inverse) {
    const float a = x0, b = x1;
    if (!inverse) {
        x0 = a * c - b * s;
        x1 = a * s + b * c;
    } else {
        x0 = a * c + b * s;
        x1 = b * c - a * s;
    }
}
// 1 / sqrt(sum_squares / width + eps) with the correctly rounded root.
STRIX_DSPARK_HD inline float rstd_of(float sum_squares, std::uint32_t width, float eps) {
    return 1.0F / declared_sqrt(sum_squares / static_cast<float>(width) + eps);
}
// Hyper-connection split of one row's 24 mixes (already multiplied by the stream's rstd):
//   pre_j = sigmoid(m_j * scale0 + base_j) + hc_eps; post_j = 2 * sigmoid(m_{4+j} * scale1 + base_{4+j});
//   comb_jk = m_{8+4j+k} * scale2 + base_{8+4j+k}; row softmax (max, exp, ((c0 + c1) + c2) + c3) then + hc_eps;
//   column normalization c / (colsum + hc_eps); then 19 times: rows c / (rowsum + hc_eps), columns likewise.
STRIX_DSPARK_HD inline void sinkhorn_core(const float* mixes, const float* scale, const float* base, float* out) {
    for (std::uint32_t j = 0; j < hc; ++j) out[j] = sigmoid(mixes[j] * scale[0] + base[j]) + hc_eps;
    for (std::uint32_t j = 0; j < hc; ++j) out[hc + j] = 2.0F * sigmoid(mixes[hc + j] * scale[1] + base[hc + j]);
    float c[hc][hc];
    for (std::uint32_t j = 0; j < hc; ++j)
        for (std::uint32_t k = 0; k < hc; ++k) c[j][k] = mixes[2U * hc + j * hc + k] * scale[2] + base[2U * hc + j * hc + k];
    for (std::uint32_t j = 0; j < hc; ++j) {
        float m = c[j][0];
        for (std::uint32_t k = 1; k < hc; ++k) m = c[j][k] > m ? c[j][k] : m;
        for (std::uint32_t k = 0; k < hc; ++k) c[j][k] = declared_exp(c[j][k] - m);
        const float sum = ((c[j][0] + c[j][1]) + c[j][2]) + c[j][3];
        for (std::uint32_t k = 0; k < hc; ++k) c[j][k] = c[j][k] / sum + hc_eps;
    }
    for (std::uint32_t it = 0; it < sinkhorn_iterations; ++it) {
        if (it != 0U)
            for (std::uint32_t j = 0; j < hc; ++j) {
                const float sum = ((c[j][0] + c[j][1]) + c[j][2]) + c[j][3];
                for (std::uint32_t k = 0; k < hc; ++k) c[j][k] = c[j][k] / (sum + hc_eps);
            }
        for (std::uint32_t k = 0; k < hc; ++k) {
            const float sum = ((c[0][k] + c[1][k]) + c[2][k]) + c[3][k];
            for (std::uint32_t j = 0; j < hc; ++j) c[j][k] = c[j][k] / (sum + hc_eps);
        }
    }
    for (std::uint32_t j = 0; j < hc; ++j)
        for (std::uint32_t k = 0; k < hc; ++k) out[2U * hc + j * hc + k] = c[j][k];
}
// Argmax order: larger value first, the lower index on ties; NaN never wins.
STRIX_DSPARK_HD inline bool better(float value, std::uint32_t index, float best, std::uint32_t best_index) {
    return value > best || (value == best && index < best_index);
}

// ---- weights (views of the file bytes) ----

struct Q8Matrix {
    std::span<const std::uint8_t> bytes; // `outputs` rows of `width` Q8_0 values
    std::uint32_t width = 0, outputs = 0;
    [[nodiscard]] std::uint64_t row_bytes() const noexcept { return width / 32U * q8_0_block_bytes; }
};
struct F16Matrix {
    std::span<const std::uint16_t> bits; // `outputs` rows of `width` F16 values
    std::uint32_t width = 0, outputs = 0;
};
struct StageWeights {
    F16Matrix hc_attn_fn, hc_ffn_fn;                       // [24][20480]
    std::span<const float> hc_attn_scale, hc_attn_base;    // [3], [24]
    std::span<const float> hc_ffn_scale, hc_ffn_base;
    std::span<const float> attn_norm, ffn_norm;            // [5120]
    std::span<const float> sinks;                          // [64]
    std::span<const float> q_norm;                         // [1280]
    std::span<const float> kv_norm;                        // [512]
    Q8Matrix q_a, q_b, kv, out_a, out_b;
    std::span<const float> router, bias, bias_image;       // [128][5120], [128], [128]
    ExpertTensors routed, shared;                          // MXFP4 [128] / Q8_0 [1]
};
struct Weights {
    Q8Matrix main_proj;                                    // [5120][15360]
    std::span<const float> main_norm;                      // [5120]
    std::array<StageWeights, stages> stage;
    std::span<const float> final_norm;                     // mtp.2.norm [5120]
    F16Matrix markov_embed, markov_head;                   // [129280][256]
    std::span<const float> confidence;                     // [5376]
    F16Matrix token_embd;                                  // target [129280][5120]
    Q8Matrix output;                                       // target head [129280][5120]
};

// ---- declared operations (host side; each is one or a few device kernels) ----

// Wave-order sum of term(i) for i < n (see the header comment).
template <class Term>
float wave_sum(std::uint32_t n, Term term) {
    float lane[32] = {};
    for (std::uint32_t i = 0; i < n; ++i) lane[i % 32U] = lane[i % 32U] + term(i);
    for (std::uint32_t d = 16; d != 0U; d /= 2U) {
        float next[32];
        for (std::uint32_t l = 0; l < 32U; ++l) next[l] = lane[l] + lane[l ^ d];
        for (std::uint32_t l = 0; l < 32U; ++l) lane[l] = next[l];
    }
    return lane[0];
}
// Chunked order of the long sums (RMSNorm, hyper-connection mixes): the n terms are split into
// chunks of 640 (one chunk when n <= 640); each chunk is a wave_sum of its own terms, and the chunk
// sums are added in ascending order from the first: ((c0 + c1) + c2) + ... One wave per chunk.
inline constexpr std::uint32_t sum_chunk = 640;
template <class Term>
float chunk_sum(std::uint32_t n, Term term) {
    if (n <= sum_chunk) return wave_sum(n, term);
    float total = 0.0F;
    for (std::uint32_t c = 0; c * sum_chunk < n; ++c) {
        const std::uint32_t first = c * sum_chunk, count = n - first < sum_chunk ? n - first : sum_chunk;
        const float part = wave_sum(count, [&](std::uint32_t i) { return term(first + i); });
        total = c == 0U ? part : total + part;
    }
    return total;
}
[[nodiscard]] float f16_value(std::uint16_t bits) noexcept;

// Main input of one committed target row: mean over the copies of each target layer's input stream
// (streams[l] = [4][5120] of layer 37 + l), ((x0 + x1) + x2) + x3 then * 0.25, concatenated by layer.
[[nodiscard]] std::vector<float> main_hidden_row(std::span<const float> layer37, std::span<const float> layer38,
                                                 std::span<const float> layer39);
// RMSNorm per row: rstd = 1 / sqrt(chunk_sum(x * x) / width + eps); y = (x * rstd) * w.
[[nodiscard]] std::vector<float> rms_norm(std::span<const float> x, std::uint32_t rows, std::span<const float> weight);
// Q8_0 linear: per row A8 quantization, then the declared Q8_0 product for every output.
[[nodiscard]] std::vector<float> q8_linear(const Q8Matrix& w, std::span<const float> x, std::uint32_t rows);
// Hyper-connection coefficients of each row of streams [rows][4][5120]: mix_j = chunk_sum(fn_j * x) * rstd
// with rstd = 1 / sqrt(chunk_sum(x * x) / 20480 + norm_eps); then the Sinkhorn split.
// Output per row: pre[4], post[4], comb[16] (comb[j * 4 + k]).
[[nodiscard]] std::vector<float> hc_mixes(std::span<const float> streams, std::uint32_t rows, const F16Matrix& fn,
                                          std::span<const float> scale, std::span<const float> base);
[[nodiscard]] std::array<float, mix_count> sinkhorn_split(std::span<const float, mix_count> mixes, std::span<const float> scale,
                                                          std::span<const float> base);
// y[r] = ((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3 with p = coefficients[r * stride + 0..3].
[[nodiscard]] std::vector<float> hc_pre(std::span<const float> streams, std::uint32_t rows, std::span<const float> coefficients,
                                        std::uint32_t stride);
// y[r][k] = post_k * x[r] + (((c0k * s0 + c1k * s1) + c2k * s2) + c3k * s3), with post/comb from mixes.
[[nodiscard]] std::vector<float> hc_post(std::span<const float> x, std::span<const float> residual, std::uint32_t rows,
                                         std::span<const float> mixes);
// RoPE table of the window-only layers (theta 10000, no YaRN), computed in double and rounded once:
// cos/sin of position * theta^(-2i/64) for the 32 pairs.
struct RopeRow { std::array<float, rope_dim / 2U> c{}, s{}; };
[[nodiscard]] RopeRow rope_row(std::uint64_t position);
// Rotate the last 64 values of every `head_width` chunk of each row (row r at position first + r).
void apply_rope(std::span<float> x, std::uint32_t rows, std::uint32_t head_width, std::uint64_t first, bool inverse);
// FP8 entry of a 512-value vector: 16 groups of 32 with an E8M0 scale each (fp8_scale_code of the
// group's amax), codes fp8_element(x, scale); bytes = 512 codes then 16 scale codes.
void fp8_entry(std::span<const float> x, std::span<std::uint8_t> entry);
[[nodiscard]] float entry_value(std::span<const std::uint8_t> entry, std::uint32_t i) noexcept;
// kv entries of rows of a stage input: kv_norm(kv(x)), RoPE at first + r, FP8 entry.
[[nodiscard]] std::vector<std::uint8_t> kv_entries(const StageWeights& w, std::span<const float> x, std::uint32_t rows,
                                                   std::uint64_t first);

// Per-stage ring of main_x kv entries: position p lives in slot p % 128; positions
// [max(0, next - 128), next) are present.
struct Ring {
    std::vector<std::uint8_t> entries = std::vector<std::uint8_t>(static_cast<std::size_t>(window) * entry_bytes);
    std::uint64_t next = 0;
    // First position ever appended (a ring started at a later position, e.g. a sequence continued from an imported
    // target state): only positions [max(start, next - 128), next) are keys. 0 = the usual ring.
    std::uint64_t start = 0;
};
struct State { std::array<Ring, stages> rings; };

// Block attention of a draft step: rows r < R at positions first + r, queries q [R][64][512] after
// RoPE. Keys of every row: the ring's positions in ascending order, then the R block entries (every
// block row sees all of them). Per (row, head): s_j = wave_dot(q, k_j) * scale; m = max s_j;
// e_j = exp(s_j - m); sum = (sum_j e_j, in key order from 0) + exp(sink - m);
// o[d] = (sum_j e_j * v_j[d], in key order from 0) / sum; then the inverse RoPE of o.
[[nodiscard]] std::vector<float> block_attention(std::span<const float> q, std::uint32_t rows, const Ring& ring,
                                                 std::span<const std::uint8_t> block_entries, std::span<const float> sinks,
                                                 std::uint64_t first);
// Grouped output: per row and group g, A8 of o[g * 4096 ..] times rows g * 1024 .. of out_a, then out_b.
[[nodiscard]] std::vector<float> output_projection(const StageWeights& w, std::span<const float> o, std::uint32_t rows);
// MoE of rows x (FP32, the ffn_norm output): router, shared expert, routed window, combine.
[[nodiscard]] std::vector<float> moe(const StageWeights& w, std::span<const float> x, std::uint32_t rows);

// ---- drafter ----

// main_x rows of committed target rows (main_hidden [n][15360]): main_norm(main_proj(.)).
[[nodiscard]] std::vector<float> main_rows(const Weights& w, std::span<const float> main_hidden, std::uint32_t rows);
// Appends main_x rows at positions state.next .. (to every stage's ring).
void append(const Weights& w, State& state, std::span<const float> main_x, std::uint32_t rows);

// Draft vocabulary: the head and the Markov argmax cover the token ids below `core` plus the
// `extra` ids (each at most once, all >= core). The default is the whole vocabulary. A restricted
// vocabulary is a separate draft identity (ds41-dspark-v1 with that vocabulary); it never changes
// output tokens, only which tokens the drafter can propose. Argmax ties still go to the lowest id.
struct Vocab {
    std::uint32_t core = vocab;
    std::vector<std::uint32_t> extra;
    [[nodiscard]] std::uint32_t size() const noexcept { return core + static_cast<std::uint32_t>(extra.size()); }
    [[nodiscard]] std::uint32_t id(std::uint32_t j) const noexcept { return j < core ? j : extra[j - core]; }
};

struct Draft {
    std::array<std::uint32_t, block_size + 1U> tokens{}; // tokens[0] = the input token, then 5 drafts
    std::array<float, block_size> confidence{};
    std::vector<float> logits;                            // [5][vocabulary size]: head logits before the Markov bias
    std::vector<float> stage_out[stages];                 // streams after each stage [5][4][5120]
    std::vector<float> head_in;                           // hc_pre of the last stage [5][5120]
    std::vector<float> markov;                            // Markov embeddings [5][256]
    std::array<std::uint32_t, block_size> candidates{};   // vocabulary rows evaluated per Markov step
};
// One draft step from `token` at position state.rings[0].next (the ring holds every earlier
// position still in the window): block positions next .. next + 4.
[[nodiscard]] Draft draft(const Weights& w, const State& state, std::uint32_t token, bool keep_logits, const Vocab& vocabulary = {});
// The head of a draft from the last stage's streams and pre mix (exposed for the device fixture).
void draft_head(const Weights& w, std::span<const float> streams, std::span<const float> pre_mix, Draft& out, bool keep_logits,
                const Vocab& vocabulary = {});
// Markov bias of one vocabulary row: lane l sums elements 8l..8l+7 in ascending order from 0 (one
// 16-byte load per lane), then the butterfly.
[[nodiscard]] float markov_dot(const std::uint16_t* head_row, const float* e) noexcept;

// ---- greedy verify protocol ----

// A window verifies row 0 = the last committed token t at position p and rows 1..k = drafts
// d_1..d_k at p + 1..p + k; target[j] is the target's greedy token after row j. The longest prefix of
// drafts equal to the target's tokens is accepted. Rows 0..accepted become committed (their target
// state and main hidden rows are kept, the rest rolled back), and target[accepted] is the next t at
// position p + accepted + 1. Tokens equal target-only greedy decoding by construction.
struct VerifyOutcome {
    std::uint32_t accepted = 0;       // drafts accepted, 0..k
    std::uint32_t committed_rows = 1; // accepted + 1
    std::uint32_t next_token = 0;
};
[[nodiscard]] VerifyOutcome verify_greedy(std::span<const std::uint32_t> drafts, std::span<const std::uint32_t> target);

// Window choice from the confidence head (an input to the cost model, never the acceptor):
// survival[j] = prod_{i <= j} sigmoid(confidence[i]); E[tokens | k drafts] = 1 + sum_{j < k} survival[j];
// returns the k in [min_drafts, 5] maximizing E / cost[k] (cost[k] = time of a k + 1 row window plus
// the draft step), the smaller k on ties.
[[nodiscard]] std::uint32_t choose_drafts(std::span<const float, block_size> confidence, std::span<const float, block_size + 1U> cost,
                                          std::uint32_t min_drafts);

} // namespace strix::ds41::dspark
