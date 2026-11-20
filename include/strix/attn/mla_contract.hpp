#pragma once
// MLA attention contract (shared): absorbed multi-head latent attention over a per-row key selection, for every engine
// whose attention keeps one latent row per token (DSA + SWA layers, GLM-5 DSA). Host/vendor-free (core_audit rule
// S2): the shape, the traits and the scalar index arithmetic that the host oracle (src/attn/mla_ref.cpp) and the device
// policies (src/attn_core/sparse_flash_mla.hpp) both use.
//
// Per token the cache holds ONE key row shared by every head: [kv_lora latent (RMS-normed) | rope (RoPE applied)].
// Weights kv_b: rows h * (nope + v) + d, columns l < kv_lora (nn.Linear layout): rows d < nope = W_k[h], the next v rows
// = W_v[h]. For a query row with q[h] = [nope | rope] (RoPE already applied by the shared rotary stage):
//   (1) absorb   qt[h][l] = sum_{d < nope} q[h][d] W_k[h][d][l]  (l < kv_lora);  qt[h][kv_lora + i] = q[h][nope + i]
//   (2) score    s[h][j]  = scale * qt[h] . key_row(pos_j)        (width d_qk = kv_lora + rope; scale = (nope+rope)^-1/2)
//   (3) softmax  p = exp(s - m) / (sum_j exp(s_j - m) + [sink] exp(z_h - m)),  m = max(max_j s_j, [sink] z_h)
//                (learned sink z_h: one logit per head in the denominator only, gpt-oss form; no value)
//   (4) latent   o[h][l] = sum_j p[h][j] latent(pos_j)[l]          (width d_v = kv_lora; the rope part has no value)
//   (5) out      y[h][e] = sum_l o[h][l] W_v[h][e][l]              (e < v)
//   (6) gate     y[h][e] *= sigmoid(g[h][e]),  g = gate_proj(x)    (elementwise; acts on the v space, i.e. AFTER (5):
//                it is an epilogue of the out projection, never of the latent flash core)
// (1)-(6) equal the unabsorbed reference (expand k = [W_k latent | k_rope], v = W_v latent per head) by associativity.
//
// Selection (which cache positions a row at position t attends; ascending, unique, all <= t):
//   List:   the indexer's list (include/strix/attn/group_indexer.hpp), at most `budget` positions.
//   Window: [t + 1 - n, t] with n = min(t + 1, window)  (window 512: 511 earlier keys + self). A window IS a list whose
//           entries are computed, so both layer kinds run the same shared sparse core with a different policy.
// Accumulation order: the host oracle sums every product in increasing index order in fp64. The device core is a
// FAST identity (bf16 operands rounded RNE, WMMA order, online exp2 softmax): compared to the oracle by tolerance,
// never bitwise. Engines whose exact path has its own declared order keep it (GLM exact: glm_attn_fast.hip).
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MLA_HD __host__ __device__
#else
#define STRIX_MLA_HD
#endif

namespace strix::attn::mla {

struct Geometry {
    unsigned heads;    // query heads of the layer (all ranks)
    unsigned kv_lora;  // latent width (value width in the absorbed form)
    unsigned rope;     // rotary key/query dims appended to the latent (0 = NoPE MLA)
    unsigned nope;     // per-head non-rotary query/key dims (absorbed into the latent query)
    unsigned v;        // per-head value dims (out projection width)
    constexpr unsigned d_qk() const { return kv_lora + rope; }  // key row / absorbed query width
    constexpr unsigned d_v() const { return kv_lora; }          // latent value width
    constexpr unsigned head_qk() const { return nope + rope; }  // unabsorbed query/key width per head
    constexpr unsigned kv_b_rows() const { return heads * (nope + v); }
};

enum class Select : std::uint8_t { List, Window };
enum class Sink : std::uint8_t { None, Learned };
enum class Gate : std::uint8_t { None, SigmoidElementwise };

struct Selection {
    Select kind;
    unsigned budget;  // List: max positions per row; Window: the window (positions per row at most)
};

struct LayerSpec {
    Geometry g;
    Selection sel;
    Sink sink;
    Gate gate;
    double rope_theta;  // RoPE base of the layer kind (the rotary stage's input; recorded, not applied here)
    double scale() const;  // (nope + rope)^-1/2, defined in mla_ref.cpp (exact for 256: 0.0625)
};

// Window arithmetic (device policies call these: one definition of the selection).
STRIX_MLA_HD constexpr unsigned window_count(unsigned t, unsigned window) { return t + 1U < window ? t + 1U : window; }
STRIX_MLA_HD constexpr unsigned window_key(unsigned t, unsigned n, unsigned j) { return t + 1U - n + j; }

// Presets (GLM: include/strix/glm/engine.hpp DsaLayerWeights, glm_math.hpp kMlaScale). A family bound from its model
// package keeps its LayerSpec there as data.
// GLM-5 DSA in this engine: NoPE MLA (key = latent only), k_nope 256, v 256, 64 heads, top-k list; no sink, no gate.
constexpr LayerSpec glm5_dsa() { return {{64U, 512U, 0U, 256U, 256U}, {Select::List, 2048U}, Sink::None, Gate::None, 0.0}; }

// Shared HIP sparse flash core fit (src/attn_core/sparse_flash_wmma.hpp Geometry): heads per rank 16 | 32, value width
// a multiple of 64 up to 512, key-only tail a multiple of 64 (one 16-wide k step per dim quarter per 64).
constexpr bool hip_core_fits(const LayerSpec& s, unsigned ranks) {
    if (ranks == 0U || s.g.heads % ranks != 0U) return false;
    const unsigned h = s.g.heads / ranks;
    return (h == 16U || h == 32U) && s.g.d_v() % 64U == 0U && s.g.d_v() <= 512U && s.g.rope % 64U == 0U;
}

// Host oracle (src/attn/mla_ref.cpp), fp64 accumulation in increasing index order. Pointers are row-major; a TP rank
// passes a spec with its head count and kv_b / sinks / q / gate pointers offset to its first head.
namespace ref {
struct Weights {
    const float* kv_b;   // [kv_b_rows][kv_lora]
    const float* sinks;  // [heads] natural-unit logits; required when spec.sink == Learned
};
struct Cache {
    const float* latent;  // [positions][kv_lora], RMS-normed
    const float* krope;   // [positions][rope], RoPE applied (unused when rope == 0)
    unsigned positions;
};
// Positions attended by the row at t (ascending). List: validates `list` (ascending, unique, <= t, <= budget) and
// copies it; Window: computes it. `out` holds at least budget entries. Throws std::invalid_argument on a bad list.
unsigned select(const LayerSpec& s, unsigned t, const std::uint32_t* list, unsigned list_n, std::uint32_t* out);
// (1) q [heads][head_qk] -> qt [heads][d_qk]
void absorb(const LayerSpec& s, const Weights& w, const float* q, double* qt);
// (2)-(4) qt [heads][d_qk] over the positions -> o [heads][kv_lora] (normalised; no keys -> zeros)
void latent_attend(const LayerSpec& s, const Weights& w, const Cache& c, const double* qt, const std::uint32_t* pos,
                   unsigned n, double* o);
// (5)-(6) o [heads][kv_lora] -> y [heads][v]; gate [heads][v] logits (required when spec.gate != None)
void out_project(const LayerSpec& s, const Weights& w, const double* o, const float* gate, double* y);
// One decode row at position t: (1)-(6). y [heads][v].
void attend_row(const LayerSpec& s, const Weights& w, const Cache& c, const float* q, unsigned t, const std::uint32_t* list,
                unsigned list_n, const float* gate, double* y);
// The same row by the unabsorbed reference form (per-head expanded keys / values, explicit sink column).
void attend_row_unabsorbed(const LayerSpec& s, const Weights& w, const Cache& c, const float* q, unsigned t,
                           const std::uint32_t* list, unsigned list_n, const float* gate, double* y);
// A prefill chunk: rows t0 .. t0 + rows - 1; q [rows][heads][head_qk], lists [rows][list_stride] with list_n [rows]
// (ignored for Window), gate [rows][heads][v], y [rows][heads][v]. Row r equals attend_row at t0 + r.
void attend_chunk(const LayerSpec& s, const Weights& w, const Cache& c, const float* q, unsigned t0, unsigned rows,
                  const std::uint32_t* lists, unsigned list_stride, const unsigned* list_n, const float* gate, double* y);
}  // namespace ref

}  // namespace strix::attn::mla
