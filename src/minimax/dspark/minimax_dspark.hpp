#pragma once

// MiniMax M3.1 Flash DSpark drafter: host reference of the draft model in the declared arithmetic of the device
// kernels (hip_minimax_dspark.hpp), so host and device drafts are bitwise comparable, plus the greedy verify protocol.
// Plan and sources: qualification notes (ROCm/ATOM atom/models/minimax_m3_dspark.py, tokenspeed
// minimax_m3_dspark.py).
//
// The draft (5 dense layers, checkpoint `<model>/dspark/`) has no embedding and no head; it uses the target's.
// - Context: for every committed target row j at position p_j, a_j = concat(h_3, h_17, h_31, h_45, h_59) (the residual
//   stream after those target layers, before any final norm), c_j = hidden_norm(fc a_j); every draft layer keeps
//   K = RoPE(k_norm(k_proj c_j), p_j) and V = v_proj c_j of its own, in a ring of 4096 positions (p mod 4096).
// - Block from the anchor token x0 at position P: rows k = 0..6 at positions P + k hold [x0, noise x 6] embedded with
//   the target's table; five pre-norm layers (GQA 64 q / 4 kv heads of 128, per-head Gemma q/k norms, NeoX RoPE on the
//   first 64 lanes, attention over the context positions [P - 4095, P - 1] plus all seven block rows, non-causal;
//   dense SwiGLU-OAI MLP of 12288); final norm -> y [7 x 6144]; the target head gives base logits.
// - Markov (vanilla, rank 256): t_0 = x0, t_{k+1} = argmax_v(base_k[v] + W2[v] . W1[t_k]); drafts t_1..t_7.
// - Confidence: c_k = sigmoid(w . [y_k ; W1[t_k]] + b), the estimate that t_{k+1} survives.
//
// Numerical identity "mm-dspark-v1" (declared here, not the model's training arithmetic):
// - MXFP8 linears: weight = E4M3(code) * 2^(e8m0 - 127), exact in FP32; BF16 weights exact in FP32; activations FP32.
//   Dot products in the "chunk-16 wave order": lane l (0..31) takes the chunks c = l, l + 32, ... of 16 consecutive
//   inputs, inside a chunk in ascending order (acc = acc + w * x from 0, no fused operations); then the butterfly
//   v[l] = v[l] + v[l ^ d], d = 16, 8, 4, 2, 1.
// - 128-wide dots (q/k norm sums of squares, attention scores): lane l takes l, l + 32, l + 64, l + 96, then the
//   butterfly. Markov dots (256): lane l takes 8l..8l + 7 ascending, then the butterfly.
// - RMSNorm over 6144: 256 threads, thread t takes t, t + 256, ... ascending, butterfly per wave of 32, the eight wave
//   sums added in ascending order; rstd = 1 / sqrt(mean + 1e-6) (correctly rounded); y = (x * rstd) * (1 + w).
// - RoPE: cos/sin of p * theta^(-2i/64) computed in double and rounded once; pairs (i, i + 32), i < 32.
// - Context and block K/V are stored as BF16 (round to nearest even); Q stays FP32.
// - Attention per (row, head): s_j = dot128(q, k_j) * 128^-0.5 over the keys in order (context positions ascending,
//   then block rows 0..6); m = max; e_j = exp(s_j - m); the keys in chunks of `attention_chunk` (64): per chunk the sum
//   and o[d] = sum_j e_j v_j[d] accumulated sequentially in key order from 0, then the chunk results added in ascending
//   chunk order from 0; o[d] = o[d] / sum. (One chunk covering every key is the v1 order: plain key-order sums.)
// - SwiGLU-OAI: g = min(gate, 7), u = clamp(up, -7, 7), (g * (1 / (1 + exp(-1.702 g)))) * (u + 1).
// - exp is declared_exp (IEEE basic operations only); argmax: larger value first, the lower id on ties.
// Draft numerics never change the output tokens: the target verifies every drafted token greedily.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#if defined(__HIPCC__)
#define STRIX_MMDS_HD __host__ __device__
#else
#define STRIX_MMDS_HD
#endif

namespace strix::minimax::dspark {

inline constexpr std::uint32_t hidden = 6144;
inline constexpr std::uint32_t layers = 5;
inline constexpr std::uint32_t heads = 64;
inline constexpr std::uint32_t kv_heads = 4;
inline constexpr std::uint32_t head_dim = 128;
inline constexpr std::uint32_t group = heads / kv_heads;           // 16 q heads per kv head
inline constexpr std::uint32_t q_width = heads * head_dim;         // 8192
inline constexpr std::uint32_t kv_width = kv_heads * head_dim;     // 512
inline constexpr std::uint32_t rope_dim = 64;
inline constexpr std::uint32_t inter = 12288;
inline constexpr std::uint32_t block = 7;                          // dspark_block_size: rows and drafts per pass
inline constexpr std::uint32_t window = 4096;                      // sliding window; context = the last 4095 positions
inline constexpr std::uint32_t context_keys = window - 1;
inline constexpr std::uint32_t taps = 5;
inline constexpr std::array<std::uint32_t, taps> tap_layers = {3, 17, 31, 45, 59};
inline constexpr std::uint32_t tap_width = taps * hidden;          // 30720
inline constexpr std::uint32_t vocab = 200064;
inline constexpr std::uint32_t markov_rank = 256;
inline constexpr std::uint32_t confidence_width = hidden + markov_rank; // 6400
inline constexpr std::uint32_t noise_token = 200058;
inline constexpr float norm_eps = 1e-6F;
inline constexpr float swiglu_alpha = 1.702F;
inline constexpr float swiglu_limit = 7.0F;
inline constexpr double rope_theta = 10000000.0;
inline constexpr float attention_scale = 0.08838834764831845F;     // 128^-0.5 rounded to FP32
inline constexpr std::uint32_t attention_chunk = 64;               // keys per attention partial (see the order below)

// ---- scalar helpers shared by host and device ----

namespace detail {
STRIX_MMDS_HD inline float pow2(int k) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(k + 127) << 23U); }
} // namespace detail

// exp(x) from basic IEEE operations only (the declared_exp of ds41_moe.hpp, copied so this module stands alone):
// Cody-Waite reduction by ln 2, a degree-7 Taylor polynomial in Horner form, a two-step power-of-two scaling.
STRIX_MMDS_HD inline float declared_exp(float x) {
    if (!(x == x)) return x;
    if (x > 88.0F) return __builtin_huge_valf();
    if (x < -103.0F) return 0.0F;
    const float n = __builtin_rintf(x * 1.44269504F);
    const float r = (x - n * 0.693359375F) - n * -2.12194440e-4F;
    float p = 1.98412698e-4F;
    p = p * r + 1.38888889e-3F;
    p = p * r + 8.33333333e-3F;
    p = p * r + 4.16666667e-2F;
    p = p * r + 1.66666667e-1F;
    p = p * r + 0.5F;
    p = p * r + 1.0F;
    p = p * r + 1.0F;
    const int k = static_cast<int>(n), k1 = k / 2, k2 = k - k1;
    return (p * detail::pow2(k1)) * detail::pow2(k2);
}
STRIX_MMDS_HD inline float sigmoid(float x) { return 1.0F / (1.0F + declared_exp(-x)); }
STRIX_MMDS_HD inline float swiglu_oai(float gate, float up) {
    const float g = gate > swiglu_limit ? swiglu_limit : gate;
    const float u = up < -swiglu_limit ? -swiglu_limit : (up > swiglu_limit ? swiglu_limit : up);
    return (g * (1.0F / (1.0F + declared_exp(-swiglu_alpha * g)))) * (u + 1.0F);
}
STRIX_MMDS_HD inline float bf16_value(std::uint16_t bits) {
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(bits) << 16U);
}
// Round to nearest even; NaN stays a quiet NaN.
STRIX_MMDS_HD inline std::uint16_t bf16_bits(float x) {
    const auto u = __builtin_bit_cast(std::uint32_t, x);
    if ((u & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
// E4M3 "fn" (no infinities; S.1111.111 is NaN), the checkpoint's F8_E4M3.
STRIX_MMDS_HD inline float e4m3_value(std::uint8_t code) {
    const std::uint32_t e = (code >> 3U) & 15U, m = code & 7U;
    float v = e == 0U ? static_cast<float>(m) * 0.001953125F : __builtin_bit_cast(float, ((e + 120U) << 23U) | (m << 20U));
    if (e == 15U && m == 7U) v = __builtin_bit_cast(float, 0x7fc00000U);
    return (code & 0x80U) != 0U ? -v : v;
}
// E8M0 block scale 2^(code - 127); the loader rejects 255 (NaN) and codes outside [1, 254].
STRIX_MMDS_HD inline float e8m0_value(std::uint8_t code) {
    const int k = static_cast<int>(code) - 127;
    return k < -126 ? detail::pow2(k + 64) * detail::pow2(-64) : detail::pow2(k);
}
// Argmax order: larger value first, the lower index on ties; NaN never wins.
STRIX_MMDS_HD inline bool better(float value, std::uint32_t index, float best, std::uint32_t best_index) {
    return value > best || (value == best && index < best_index);
}

// ---- weights (views of the file bytes) ----

struct Mxfp8Matrix {
    const std::uint8_t* codes = nullptr;   // [outputs][width] E4M3
    const std::uint8_t* scales = nullptr;  // [outputs][width / 32] E8M0
    std::uint32_t outputs = 0, width = 0;
};
struct Bf16Matrix {
    const std::uint16_t* bits = nullptr;   // [outputs][width]
    std::uint32_t outputs = 0, width = 0;
};
struct LayerWeights {
    const std::uint16_t *input_norm = nullptr, *post_norm = nullptr;  // [6144]
    const std::uint16_t *q_norm = nullptr, *k_norm = nullptr;         // [128]
    Mxfp8Matrix q, k, v, o, gate, up, down;
};
struct Weights {
    Bf16Matrix fc;                                     // [6144][30720]
    const std::uint16_t *hidden_norm = nullptr, *final_norm = nullptr;
    std::array<LayerWeights, layers> layer{};
    const std::uint16_t *markov_w1 = nullptr, *markov_w2 = nullptr;  // [vocab][256]
    const std::uint16_t* confidence_w = nullptr;      // [6400]
    float confidence_b = 0.0F;
};

// The draft checkpoint directory (`<model>/dspark`, or any directory with the same safetensors), read into host memory
// and checked against the expected manifest (97 tensors, names, dtypes, shapes, E8M0 codes in [1, 254]).
class DraftFile {
public:
    // block_linears = false: the 35 MXFP8 block linears are checked but not read (their Mxfp8Matrix views are null), for
    // callers that supply them through DeviceDrafter::set_linear_hook (measurement: 1.66 GB less to read and upload).
    static std::unique_ptr<DraftFile> open(const std::filesystem::path& dir, bool block_linears = true);
    ~DraftFile();
    [[nodiscard]] const Weights& weights() const noexcept { return weights_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
    // Raw bytes of one tensor by its checkpoint name (for device upload); throws if absent.
    [[nodiscard]] std::span<const std::uint8_t> tensor(const std::string& name) const;

private:
    DraftFile() = default;
    struct Blob;
    std::vector<std::unique_ptr<Blob>> blobs_;
    Weights weights_;
    std::uint64_t bytes_ = 0;
};
inline const std::string checkpoint_prefix = "language_model.model.dspark.";

// Rows of a plain BF16 tensor (the target's embedding) from a safetensors directory with an index; host memory.
std::vector<float> read_bf16_rows(const std::filesystem::path& dir, const std::string& name, std::span<const std::uint32_t> rows);

// ---- declared operations (host reference; the device kernels reproduce them bitwise) ----

// out[r][o] = W[o] . x[r] for `rows` rows (x row stride = W.width).
void mxfp8_linear(const Mxfp8Matrix& w, const float* x, std::size_t rows, float* out);
void bf16_linear(const Bf16Matrix& w, const float* x, std::size_t rows, float* out);
// Gemma RMSNorm over 6144 of `rows` rows.
void rms_norm(const float* x, const std::uint16_t* weight, std::size_t rows, float* out);
// cos/sin of position p for i = 0..31 (FP32 rounded once from double).
void rope_table(std::uint64_t position, float* cos_out, float* sin_out);
// Per-head Gemma norm then NeoX RoPE on the first 64 lanes, in place, for `count` heads of one row.
void head_norm_rope(float* x, std::uint32_t count, const std::uint16_t* weight, const float* cos_v, const float* sin_v);

// Context ring of one sequence: K and V of every layer at positions p mod 4096, with the position each slot holds.
struct Ring {
    std::vector<std::uint16_t> k, v;                   // [layers][window][512] BF16
    std::vector<std::int64_t> position;                // [window], -1 = empty
    Ring();
};

// Context of committed rows: taps [n][30720] f32 at positions pos0 .. pos0 + n - 1.
void append_context(const Weights& w, Ring& ring, const float* taps_rows, std::size_t n, std::uint64_t pos0);
// The block pass from the anchor token at `position` with its 7 embedded rows ([x0, noise x 6], f32 [7][6144]);
// returns y = final_norm(...) [7][6144]. Throws if a context position of the window is missing from the ring.
void block_forward(const Weights& w, const Ring& ring, const float* embedded, std::uint64_t position, float* y,
                   std::uint32_t chunk = attention_chunk);

// One Markov step over the vocabulary range [v0, v1): candidate (value, id) of argmax(base[v - v0] + W2[v] . W1[prev]).
struct Candidate {
    float value = 0.0F;
    std::uint32_t id = 0;
};
Candidate markov_step(const Weights& w, std::uint32_t prev, const float* base, std::uint32_t v0, std::uint32_t v1);
[[nodiscard]] inline Candidate combine(Candidate a, Candidate b) { return better(b.value, b.id, a.value, a.id) ? b : a; }
// Confidence of block row k given its previous token.
float confidence(const Weights& w, const float* y_row, std::uint32_t prev);

struct Draft {
    std::array<std::uint32_t, block> tokens{};
    std::array<float, block> confidence{};
};
// Full-vocabulary Markov chain from base logits [7][vocab] and the block's y.
Draft sample(const Weights& w, std::uint32_t anchor, const float* base, const float* y);

// ---- verify protocol ----

// Window rows t, d_1..d_k at positions p..p+k; `greedy` = the target's argmax after each row (k + 1 values).
// accepted = longest prefix with d_j == greedy[j - 1]; the emitted tokens are greedy[0..accepted].
struct Verdict {
    std::uint32_t accepted = 0;
    std::vector<std::uint32_t> emitted;
};
Verdict verify_greedy(std::span<const std::uint32_t> drafts, std::span<const std::uint32_t> greedy);

// Window length by expected tokens per cost: survival s_i = prod_{j < i} sigmoid confidence (already probabilities),
// expected emitted tokens of a k-draft window = 1 + sum_{i=1..k} s_i; cost[k] = measured cost of a (k + 1)-row verify
// plus the draft when k > 0. Returns the k in [min_k, cost.size() - 1] maximizing expected / cost (lowest k on ties).
std::uint32_t choose_window(std::span<const float> confidence, std::span<const double> cost, std::uint32_t min_k);

// Window controller learned online (the DS SurvivalController reduced to MiniMax's needs): per draft depth i and
// confidence bin b (8 bins of the confidence head's sigmoid), p(i, b) = P(draft i accepted | drafts < i accepted).
// Short windows never observe the deep depths, so each cell is shrunk towards its bin's rate pooled over all depths,
// m = (ok + 4 q_b) / (tries + 4) with q_b = (ok_b + 1.2) / (tries_b + 2), and read optimistically,
// p = min(1, m + 0.5 sqrt(m (1 - m) / (tries + 4))) (without both, fully accepted text kept short windows: mmbench B json
// auto -9 % vs fixed:7). choose() returns the k in [0, 7] maximizing expected emitted tokens per cost,
// E[k] = 1 + sum_{i<=k} prod_{j<i} p(j, b_j), cost[k] = draft (k > 0) + verify of k + 1 rows (measured by the caller);
// the first call returns the full window; every `explore_every`-th call returns the best window + 2 (perf review of
// measured: a full-window probe every 16 cycles was the largest single loss on prose, 5-12 tokens per 256), so the
// next depths keep being observed and pooling covers the rest.
class SurvivalController {
public:
    explicit SurvivalController(std::uint32_t explore_every = 16) : explore_(explore_every) {}
    [[nodiscard]] std::uint32_t choose(std::span<const float> confidence, std::span<const double> cost);
    // One verify outcome: the window drafted k tokens (with their row confidences) and `accepted` of them were accepted.
    void observe(std::span<const float> confidence, std::uint32_t k, std::uint32_t accepted);
    [[nodiscard]] double probability(std::uint32_t depth, float confidence) const;

private:
    static std::uint32_t bin(float c) { return c >= 1.0F ? 7U : (c <= 0.0F ? 0U : static_cast<std::uint32_t>(c * 8.0F)); }
    std::array<std::array<std::uint32_t, 8>, block> ok_{}, tries_{};
    std::uint32_t explore_, calls_ = 0;
};

// Pre-draft gate for the auto window (report §9, measurement): the neural draft runs before the window is chosen, so where drafting
// does not pay, a k = 0 cycle still costs the draft (prose-es on 7a548a43: 17 + 57 ms against 55 ms plain). Each drafting
// cycle's excess over plain decoding in the same time, x = emitted - seconds / t_plain, is accumulated since the gate was
// made; drafting pauses for `backoff` cycles (8, doubling up to 128 while it keeps losing) when
// mean(x) + z sd(x) / sqrt(n) < 0 with n >= min_cycles, and the backoff resets to 8 when mean(x) - z sd(x) / sqrt(n) > 0.
// The sums decay by 0.97 per drafting cycle (effective window ~33 cycles) so a prose -> code switch is followed.
// t_plain = EMA (0.8 / 0.2) of the cycles that did not draft; before the first one, the caller's estimate.
// Across ranks only the decisions must agree: proposal() on a drafting cycle (agreed by the caller with its other
// per-cycle agreement, rank 0's value wins) and pause() with the agreed value; draft() is then local on every rank, so
// a paused cycle needs no exchange (s10: a per-cycle agree cost ~1 ms).
class DraftGate {
public:
    // agreed = false (default): observe() starts a pause itself, sums without decay (the previous form, serve's).
    // agreed = true: the decayed sums above, and pauses only through proposal() / pause() (decisions agreed by the caller).
    explicit DraftGate(double z = 1.0, std::uint32_t min_cycles = 12, bool agreed = false)
        : z_(z), min_(min_cycles), agreed_(agreed) {}
    // Once per cycle, before drafting: false = decode this cycle without a draft (no exchange).
    [[nodiscard]] bool draft();
    void observe(bool drafted, std::uint32_t emitted, double seconds, double plain_estimate);
    // On the evidence so far: cycles to pause after the current one (0 = keep drafting).
    [[nodiscard]] std::uint32_t proposal();
    // Start the agreed pause (0 = none).
    void pause(std::uint32_t cycles);
    [[nodiscard]] std::uint32_t paused_cycles() const { return paused_; }

private:
    double z_;
    std::uint32_t min_;
    bool agreed_;
    double n_ = 0, sx_ = 0, sxx_ = 0, t_plain_ = 0;
    std::uint32_t backoff_ = 8, skip_ = 0, paused_ = 0;
};

} // namespace strix::minimax::dspark
