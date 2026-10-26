#pragma once
// GLM-5.3-Flash Tier A host emulation (design notes sections 1.2 and 1.4): one function per device op,
// in exactly the declared operation order. Device kernels must equal these bit for bit given the same inputs.
// Layouts: activations FP32 row-major; plain weights in their file layout [out][in] (F16 / BF16 as uint16, F32 as
// float); the residual stream of one row is 4 copies x 4096, copy-major.
#include "glm/common/glm_math.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix::glm::host {

enum class WType { F16, BF16, F32 };
struct Mat {  // a plain weight [out][in]
    const void* data = nullptr;
    WType type = WType::F16;
    unsigned out = 0, in = 0;
    float at(unsigned o, unsigned i) const {
        const std::size_t k = static_cast<std::size_t>(o) * in + i;
        switch (type) {
            case WType::F16: return f16_to_float(static_cast<const std::uint16_t*>(data)[k]);
            case WType::BF16: return bf16_to_float(static_cast<const std::uint16_t*>(data)[k]);
            default: return static_cast<const float*>(data)[k];
        }
    }
};
struct Vec {  // a small vector weight (norm weights, biases)
    const void* data = nullptr;
    WType type = WType::BF16;
    unsigned n = 0;
    float at(unsigned i) const { return Mat{data, type, 1, n}.at(0, i); }
};

struct NonFinite : std::runtime_error {
    using std::runtime_error::runtime_error;
};
void check_finite(std::span<const float> v, const char* checkpoint);  // C1/C2/C4/C5

// R1: y[o] = chain over i of fmaf(x[i], W[o][i]) for o in [o0, o0 + n).
void dense(std::span<const float> x, const Mat& w, std::span<float> y, unsigned o0 = 0);
// R2/R3 RMSNorm over n = x.size(): weighted (w != null) or unweighted.
void rmsnorm(std::span<const float> x, const Vec* w, float eps, std::span<float> y);

// ---- mHC (R6, R7)
struct HcWeights {
    Mat fn;  // [24][16384] F32
    const float* base = nullptr;   // [24]
    const float* scale = nullptr;  // [3]
};
struct HcMix {
    std::array<float, kHc> pre{}, post{};
    std::array<float, kHc * kHc> comb{};  // comb[j * 4 + i]: input stream j -> output stream i
};
HcMix hc_mix(std::span<const float> X, const HcWeights& w, Identity id = Identity::kR1);  // R1b: blocked fn chains
void hc_collapse(std::span<const float> X, const HcMix& m, std::span<float> x);
void hc_expand(std::span<const float> X, const HcMix& m, std::span<const float> y, std::span<float> Xout);
void hc_final(std::span<const float> X, std::span<float> h);  // (((X0 + X1) + X2) + X3) * 0.25

// ---- KDA (R8-R10), per rank: `heads` heads (32 on a TP2 rank, 64 single), channels head-major.
struct KdaState {
    unsigned heads = 0;
    std::vector<float> S;     // [heads][128 k][128 v]
    std::vector<float> conv;  // [3 * heads * 128 channels (q | k | v)][3] oldest first
};
struct KdaInputs {           // one row, rank-local
    std::span<const float> qkv;      // [3 * heads * 128] pre-conv projections (q | k | v)
    std::span<const float> f;        // [heads * 128] f_b(f_a(x))
    std::span<const float> gate;     // [heads * 128] g_b(g_a(x))
    std::span<const float> b;        // [heads] b_proj(x)
};
struct KdaWeights {
    const std::uint16_t* conv = nullptr;  // [3 * heads * 128][4] BF16 rank rows (q | k | v)
    const float* A_log = nullptr;         // [heads]
    const float* dt_bias = nullptr;       // [heads * 128]
    Vec o_norm;                           // [128] BF16
};
// 4a + 4b + 4c for one row: updates st, writes y [heads * 128] (the o_proj input).
void kda_step(KdaState& st, const KdaInputs& in, const KdaWeights& w, std::span<float> y);

// ---- indexer (R4, R11, R12, R5, section 1.4) and MLA (R13): per layer cache of one sequence
struct DsaCache {
    std::vector<std::uint16_t> latent;  // [T][512] bf16
    std::vector<std::uint16_t> ikey;    // [T][128] bf16
    std::vector<std::uint16_t> igate;   // [T][128] bf16
    std::vector<float> pooled;          // [P][128] fp32
    unsigned tokens() const { return static_cast<unsigned>(ikey.size() / kIdxDim); }
    unsigned pools() const { return static_cast<unsigned>(pooled.size() / kIdxDim); }
    void truncate(unsigned t) {  // rollback: keep positions [0, t)
        latent.resize(static_cast<std::size_t>(t) * kKvLora);
        ikey.resize(static_cast<std::size_t>(t) * kIdxDim);
        igate.resize(static_cast<std::size_t>(t) * kIdxDim);
        pooled.resize(static_cast<std::size_t>(t / kKpool) * kIdxDim);
    }
};
struct IndexerWeights {
    Mat wk, gate, weights_proj;  // [128][4096], [128][4096], [32][4096] F16
    Vec k_norm_w, k_norm_b;      // [128] F16
    const float* ape = nullptr;  // [4][128] F32
};
// Appends the row's latent (already normed), indexer key and gate to the cache, completes a pool when due (R12).
void dsa_append(DsaCache& c, std::span<const float> latent, std::span<const float> x, const IndexerWeights& w);
// Canonical token list of the query at position t (section 1.4): selected pools ascending, then the tail.
std::vector<std::uint32_t> indexer_select(const DsaCache& c, unsigned t, std::span<const float> qI /*[32*128]*/,
                                          std::span<const float> x, const IndexerWeights& w);
// R13 for `heads` rank heads: q [heads][256]; kv_b rank rows [heads * 512][512] F16 (K rows h*512 + d, V rows
// h*512 + 256 + d); writes o [heads][256].
void mla_attention(const DsaCache& c, std::span<const std::uint32_t> tokens, std::span<const float> q, const Mat& kv_b,
                   unsigned heads, std::span<float> o);

// ---- router (R1, R5, R14), swiglu (R15), combine (R16), head (R18)
struct Route {
    std::array<std::uint32_t, kTopE> id{};  // ascending expert id
    std::array<float, kTopE> w{};
};
Route route(std::span<const float> x, const Mat& router, const float* bias, Identity id = Identity::kR1);
// R1b router logits (identity experiment): y[o] = r1b_dot over the 4096 inputs.
void dense_r1b(std::span<const float> x, const Mat& w, std::span<float> y);
std::vector<std::uint32_t> topk_ordered(std::span<const float> score, unsigned k);  // R5: indices by key, then sorted
inline float swiglu(float g, float u) {
    const float gg = g > kClamp ? kClamp : g;
    const float uu = u > kClamp ? kClamp : (u < -kClamp ? -kClamp : u);
    return silu(gg) * uu;
}
void moe_combine(const Route& r, std::span<const float> expert_out /*[8][4096] in route order*/,
                 std::span<const float> shared, std::span<float> y);
struct ArgMax {
    float value = 0.0F;
    std::uint32_t id = 0;
};
ArgMax argmax_local(std::span<const float> logits, std::uint32_t id0);
ArgMax argmax_merge(ArgMax a, ArgMax b);

}  // namespace strix::glm::host
