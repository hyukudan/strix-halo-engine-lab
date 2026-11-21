#pragma once
// Compressed group indexer contract (shared): which earlier tokens a query of a sparse-attention layer may attend to,
// chosen by scoring POOLED keys, one per group of `ratio` consecutive token slots.
//
// Stages (every engine binds the shared cores; this header holds the shape, the traits and the scalar arithmetic that
// both the host oracle and the device policies use, so host and device give the same bits):
//   (a) token key per slot (model data, e.g. RMSNorm(Wk x) [128]);
//   (b) pooling of a COMPLETE group into one key (trait Pooling); a group key is a deterministic function of its
//       ratio token keys, so it is computed once when the group closes and cached (decode reads G = N / ratio keys);
//   (c) score_g = sum_h w_h relu(q_h . k_g) (trait ScoreScale: where the softmax-style scale sits);
//   (d) top-k over the visible groups (trait OwnGroup), ties -> lower group index (src/attn_core/select.hpp:119-180);
//   (e) expansion: selected groups ascending -> g * ratio + {0..ratio-1}, then the query's own tail tokens up to t
//       (select.hpp:111-116 append_tail; the CUDA/HIP row-group union consumes these lists).
// A group key carries the RoPE phase of its FIRST slot (group_rope_slot); RoPE itself is the shared rotary stage.
//
// Tie rule (declared): larger score wins; equal scores -> lower group index; selected groups are emitted ascending.
// Reference implementations that use torch.topk do not specify the order among equal scores, so the selected SET can
// differ from such a reference only when several groups share the score at the cut. Scores here are never -0
// (acc starts at +0 and every step is an fma whose exact-zero result rounds to +0), so the uint32 key map below orders
// them exactly as float comparison does.
// Reduction order (declared, settles the fp32-order identity risk): dot_h = fma chain over i ascending from +0;
// score = fma(w_h, relu(dot_h [* c]), acc) over h ascending from +0 (score_group). The device EXACT scan
// (src/attn_core/group_score_scan.hpp, every variant) runs these statements and is bit-identical to the host oracle;
// a FAST score (WMMA, other accumulation order) must re-score the band around the cut with score_group before the
// select (src/attn_core/index_score_band.hpp), so the selected set is the declared one. GLM's ordered_key is score_key.
// Against a reference with another reduction order (cuBLAS) near-ties at the cut can flip: the gate measures recall.
//
// Host/vendor-free (core_audit rule S2): no HIP/CUDA headers; STRIX_GI_HD marks the functions device code may call.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__CUDACC__) && defined(_MSC_VER) && !defined(__HIPCC__)
#define STRIX_GI_HD __host__ __device__ __forceinline__
#elif defined(__HIPCC__) || defined(__CUDACC__)
// always_inline: device binders get these statements inlined before simplification, like hand-written kernel code.
#define STRIX_GI_HD __host__ __device__ __attribute__((always_inline))
#else
#define STRIX_GI_HD
#endif

namespace strix::attn::group_indexer {

// How a complete group of `ratio` token keys becomes one key.
enum class Pooling : std::uint8_t {
    GatedSoftmaxApe,  // v = Wv k, g = Wg k + ape[slot] (fp32), softmax over slots per channel, RMSNorm, bf16
    GatedSoftmax,     // GLM DSA: per-token bf16 key + bf16 gate, + fp32 ape, softmax over slots per channel, no norm
    Mean,             // Qwen QSA: mean of the raw keys, then RMSNorm (qwen_qsa.cu:192-219)
};
// Which groups a query at slot t may score.
enum class OwnGroup : std::uint8_t {
    CompleteOnly,  // GLM, QSA: P = (t + 1) / ratio groups (the own group counts once complete); top `budget`; tail =
                   // tokens P * ratio .. t (0 .. ratio-1 tokens)
    AlwaysOwn,     // P = t / ratio (the own group is never scored); top `budget - 1`; tail = own group up to t
};
// Where the 1/sqrt scale of the score sits.
enum class ScoreScale : std::uint8_t {
    InsideRelu,    // GLM: acc = fma(w_h, relu(dot * c), acc)
    InHeadWeight,  // w_h already carries heads^-0.5 * head_dim^-0.5; acc = fma(w_h, relu(dot), acc)
};

struct Shape {
    unsigned ratio;         // token slots per group
    unsigned heads;         // indexer query heads
    unsigned head_dim;      // indexer key / query width
    unsigned group_budget;  // groups attended per query (own group included for AlwaysOwn)
    Pooling pooling;
    OwnGroup own;
    ScoreScale scale;
};
// GLM DSA indexer as it runs today (glm_dims.hpp:8, glm_attn_fast.hip:771-781). GLM's decode select binds it through
// src/attn_core/group_select_kernel.hpp (GlmSelectView); the other GLM stages are descriptive here.
inline constexpr Shape kGlmDsa{4U, 32U, 128U, 512U, Pooling::GatedSoftmax, OwnGroup::CompleteOnly, ScoreScale::InsideRelu};

STRIX_GI_HD constexpr unsigned visible_groups(const Shape& s, std::uint32_t t) {
    return s.own == OwnGroup::AlwaysOwn ? t / s.ratio : (t + 1U) / s.ratio;
}
// Groups the top-k picks among the visible ones.
STRIX_GI_HD constexpr unsigned pick_count(const Shape& s) {
    return s.own == OwnGroup::AlwaysOwn ? s.group_budget - 1U : s.group_budget;
}
// First token of the tail appended after the selected groups (the tail runs to t inclusive).
STRIX_GI_HD constexpr std::uint32_t tail_begin(const Shape& s, std::uint32_t t) { return visible_groups(s, t) * s.ratio; }
// Largest token list of one query (AlwaysOwn, budget 512, ratio 4: 511 * 4 + 4 = 2048; GLM / QSA: 512 * 4 + 3 = 2051).
STRIX_GI_HD constexpr unsigned max_tokens(const Shape& s) {
    return s.own == OwnGroup::AlwaysOwn ? s.group_budget * s.ratio : s.group_budget * s.ratio + s.ratio - 1U;
}
// Every visible group is selected -> the query attends densely (causal). At budget 512, ratio 4 this holds for t < 2048, which is the
// reference's "kv_len <= 2048 -> dense" switch stated per query.
STRIX_GI_HD constexpr bool dense(const Shape& s, std::uint32_t t) { return visible_groups(s, t) <= pick_count(s); }
// Slot whose rotary phase the pooled key of group g carries.
STRIX_GI_HD constexpr std::uint32_t group_rope_slot(const Shape& s, std::uint32_t g) { return g * s.ratio; }
// Groups closed by a context of n token slots (pooled-key cache length).
STRIX_GI_HD constexpr std::uint32_t closed_groups(const Shape& s, std::uint32_t n) { return n / s.ratio; }
// Token keys a decoder must keep besides the pooled cache: the open group plus the rows a verify step may roll back
// (a group closed by a rejected draft token is dropped and pooled again from these keys).
STRIX_GI_HD constexpr unsigned token_ring_slots(const Shape& s, unsigned max_verify_rows) { return s.ratio - 1U + max_verify_rows; }

STRIX_GI_HD inline float fma3(float a, float b, float c) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return fmaf(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
STRIX_GI_HD inline float relu(float x) { return x > 0.0F ? x : 0.0F; }
// Round to bfloat16, nearest even; NaN stays NaN.
STRIX_GI_HD inline float bf16_round(float x) {
    const std::uint32_t b = __builtin_bit_cast(std::uint32_t, x);
    if ((b & 0x7F800000U) == 0x7F800000U && (b & 0x007FFFFFU) != 0U) return __builtin_bit_cast(float, b | 0x00400000U);
    const std::uint32_t r = (b + 0x7FFFU + ((b >> 16U) & 1U)) & 0xFFFF0000U;
    return __builtin_bit_cast(float, r);
}
// Order-preserving uint32 key of a score (larger score -> larger key); -0 maps like +0.
STRIX_GI_HD inline std::uint32_t score_key(float s) {
    const std::uint32_t b = __builtin_bit_cast(std::uint32_t, s == 0.0F ? 0.0F : s);
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}

// One channel of a gated pooling: softmax over the R slot logits (max-subtracted, exp from the caller's declared exp),
// weighted sum of the R slot values. The expression tree for R = 4 is GLM dsa_pool_channel's
// (((e0 + e1) + e2) + e3, ((p0 v0 + p1 v1) + p2 v2) + p3 v3).
template <unsigned R, class Exp>
STRIX_GI_HD inline float gated_pool_channel(const float (&logits)[R], const float (&values)[R], Exp exp_fn) {
    static_assert(R >= 2U, "a group pools at least two slots");
    float maximum = logits[0];
    for (unsigned j = 1; j < R; ++j) maximum = logits[j] > maximum ? logits[j] : maximum;
    float exponent[R], probability[R];
    for (unsigned j = 0; j < R; ++j) exponent[j] = exp_fn(logits[j] - maximum);
    float sum = exponent[0];
    for (unsigned j = 1; j < R; ++j) sum = sum + exponent[j];
    // Statement order as GLM's pre-lift kernels (all probabilities, then the products): same bits in any order, and
    // the same order keeps GLM's device code byte-identical.
    for (unsigned j = 0; j < R; ++j) probability[j] = exponent[j] / sum;
    float acc = probability[0] * values[0];
    for (unsigned j = 1; j < R; ++j) acc = acc + probability[j] * values[j];
    return acc;
}

// Score of one group key k [head_dim] for a query q [heads][head_dim] with head weights w [heads]:
// acc = +0; for h ascending: dot = fma chain over i ascending from +0; acc = fma(w_h, relu(dot [* c]), acc).
// `c` is used only for ScoreScale::InsideRelu.
STRIX_GI_HD inline float score_group(const Shape& s, const float* q, const float* k, const float* w, float c) {
    float acc = 0.0F;
    for (unsigned h = 0; h < s.heads; ++h) {
        const float* qh = q + static_cast<std::size_t>(h) * s.head_dim;
        float dot = 0.0F;
        for (unsigned i = 0; i < s.head_dim; ++i) dot = fma3(qh[i], k[i], dot);
        acc = fma3(w[h], relu(s.scale == ScoreScale::InsideRelu ? dot * c : dot), acc);
    }
    return acc;
}

// ------------------------------------------------------------------ host oracle (src/attn/group_indexer_ref.cpp)

// InHeadWeight head-weight scale heads^-0.5 * head_dim^-0.5 as the fp32 the reference multiplies by.
float head_weight_scale(const Shape& s);

// GatedSoftmaxApe pooling weights of one layer: wv, wg [head_dim][head_dim] row-major (out, in) like a torch Linear weight, ape
// [ratio][head_dim], norm [head_dim]; all fp32.
struct GatedApeWeights {
    const float* wv;
    const float* wg;
    const float* ape;
    const float* norm;
    float eps;
};
// Pools one COMPLETE group (Pooling::GatedSoftmaxApe): keys [ratio][head_dim] (the cached bf16 token keys as float)
// -> out [head_dim], bf16-rounded, before RoPE. Projections: fma chain over the input index ascending from +0.
// RMSNorm: ss = fma chain of x * x, r = 1 / sqrt(ss / head_dim + eps), out = bf16(norm * (x * r)).
void pool_group_gated_ape(const Shape& s, const float* keys, const GatedApeWeights& w, float (*exp_fn)(float), float* out);

// Scores of the visible groups 0 .. visible-1 of one query: keys [visible][head_dim] rotated, q [heads][head_dim]
// rotated, w [heads] (scale applied for InHeadWeight). Throws on a NaN score.
void score_row(const Shape& s, const float* q, const float* w, float c, const float* group_keys, unsigned visible,
               float* scores);

// Selected groups, ascending: min(pick_count, visible) groups by the declared tie rule.
std::vector<std::uint32_t> select_groups(const Shape& s, const float* scores, unsigned visible);

// Token slots query t attends: the selected groups expanded, then tail_begin(t) .. t. Strictly ascending, all <= t.
std::vector<std::uint32_t> expand_tokens(const Shape& s, const std::vector<std::uint32_t>& groups, std::uint32_t t);

}  // namespace strix::attn::group_indexer
