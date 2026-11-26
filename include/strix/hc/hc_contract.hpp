#pragma once

// Hyper-connection (HC) contract: one declared arithmetic for the multi-stream residual of every model that
// keeps `streams` copies of the hidden state (DeepSeek-V4 mHC with the Sinkhorn comb, Hy-V4 iHC without it).
// This header is the identity-defining specification; the host oracle (hc_ref.hpp) and every device body
// (HIP src/core, CUDA src/cuda/core) compute exactly these operations in exactly this order.
//
// Layout: the residual is copy-major, element (c, i) at c * hidden + i (c < streams). A site's weights are
// fn [mix_rows][streams * hidden] (F16 or F32, row-major), base [mix_rows], scale [scale_count]; mix rows are
// pre (streams), then post (streams, sublayer sites only), then comb (streams x streams, row j = source copy,
// column k = destination copy, Comb::sinkhorn only).
//
// Operation order (FP32 accumulation, every product and sum rounded on its own: no contraction to FMA):
//   n = streams * hidden; dot(a, b) = the chunked lane dot of ds41_layer.hpp (chunks of 640 in ascending order,
//   lane l of a chunk takes elements l, l + 32, ... with acc = acc + a * b, butterfly 16, 8, 4, 2, 1);
//   r = 1 / declared_sqrt(dot(x, x) / n + norm_eps);                    (rsqrt AFTER the dot: scale-after-dot)
//   mix[m] = dot(fn[m], x) * r;                                          (fn F16 -> F32 exactly)
//   pre[j]  = declared_sigmoid(mix[j] * scale[0] + base[j]) + pre_eps;
//   post[j] = post_magnitude * declared_sigmoid(mix[N + j] * scale[1] + base[N + j]) + post_eps;
//   Comb::sinkhorn: c[j][k] = mix[2N + jN + k] * scale[2] + base[2N + jN + k]; per row j: max, exp(c - max),
//     sum from 0 in ascending k, c = c / sum + comb_eps; then normalise columns; then iters - 1 rounds of
//     (rows, columns), each c = c / (sum from 0 ascending + comb_eps);
//   Comb::identity: no comb rows, no stage (NOT the same bits as an identity matrix: 0 * inf and -0 + +0);
//   collapse: x_out[i] = store(0 + pre[0] * x[0][i] + ... + pre[N-1] * x[N-1][i]), ascending c;
//   expand:   out[k][i] = store(post[k] * sub[i] + s), s = residual[k][i] (identity) or
//             s = 0 + comb[0][k] * residual[0][i] + ... ascending j (sinkhorn);
//   store(v) = v (Store::f32) or the nearest bfloat16, ties to even (Store::bf16).
//
// Traits not covered by this oracle (declared differences): the GLM
// order (normalise before the dot, serial FMA chains) is a different identity; its binding adds a trait together with
// its own fixture, never a silent unification.
//
// Declared arithmetic (declared_exp/sqrt/sigmoid, the lane and chunked dots, f16/bf16 conversion) is homed in
// ds41_layer.hpp / ds41_moe.hpp today and used from there (pending move to a model-free numeric header: it needs
// a DS41 byte identity and is not owned by this package). GLM keeps a verbatim copy (src/glm/common/glm_math.hpp).

#include "strix/ds41_layer.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_HC_HD __host__ __device__
#else
#define STRIX_HC_HD
#endif

namespace strix::hc {

enum class Comb : std::uint8_t { sinkhorn, identity };
enum class FnType : std::uint8_t { f16, f32 };
enum class Store : std::uint8_t { f32, bf16 };
enum class Site : std::uint8_t { sublayer, head };  // head: pre-only mix, collapse only

inline constexpr std::uint32_t max_streams = 8;

struct Spec {
    std::uint32_t streams = 4;
    std::uint32_t hidden = 0;
    Site site = Site::sublayer;
    Comb comb = Comb::sinkhorn;
    std::uint32_t sinkhorn_iters = 0;
    FnType fn = FnType::f32;
    Store store = Store::f32;
    float post_magnitude = 2.0F;
    float pre_eps = 0.0F, post_eps = 0.0F, comb_eps = 0.0F, norm_eps = 0.0F;
    bool norm_eps_outside_sqrt = false;

    [[nodiscard]] constexpr std::uint32_t post_rows() const { return site == Site::head ? 0U : streams; }
    [[nodiscard]] constexpr std::uint32_t comb_rows() const {
        return site == Site::head || comb == Comb::identity ? 0U : streams * streams;
    }
    [[nodiscard]] constexpr std::uint32_t mix_rows() const { return streams + post_rows() + comb_rows(); }
    [[nodiscard]] constexpr std::uint32_t scale_count() const {
        return 1U + (post_rows() != 0U ? 1U : 0U) + (comb_rows() != 0U ? 1U : 0U);
    }
    [[nodiscard]] constexpr std::size_t flat() const { return std::size_t{streams} * hidden; }
    [[nodiscard]] constexpr bool valid() const {
        return streams >= 1U && streams <= max_streams && hidden > 0U &&
               (comb_rows() == 0U || sinkhorn_iters >= 1U);
    }
};

// ---- per-value pieces shared by the oracle and the device bodies ----------------------------------------
STRIX_HC_HD inline float store_value(float v, Store store) {
    return store == Store::bf16 ? ds41::round_to_bf16(v) : v;
}
STRIX_HC_HD inline float pre_coef(float mix, float scale, float base, float eps) {
    return ds41::declared_sigmoid(mix * scale + base) + eps;
}
STRIX_HC_HD inline float post_coef(float mix, float scale, float base, float magnitude, float eps) {
    return magnitude * ds41::declared_sigmoid(mix * scale + base) + eps;
}
// Collapse of element i: sum over c ascending of pre[c] * x[c][i], from 0, then the store rounding.
STRIX_HC_HD inline float collapse_at(const float* x, std::size_t hidden, std::size_t i, const float* pre,
                                     std::uint32_t streams, Store store) {
    float y = 0.0F;
    for (std::uint32_t c = 0; c < streams; ++c) y = y + pre[c] * x[c * hidden + i];
    return store_value(y, store);
}
// Expand of copy k, element i. comb == nullptr is Comb::identity (the stage is skipped).
STRIX_HC_HD inline float expand_at(float sub, const float* residual, std::size_t hidden, std::size_t i, std::uint32_t k,
                                   float post, const float* comb, std::uint32_t streams, Store store) {
    float s;
    if (comb == nullptr) {
        s = residual[k * hidden + i];
    } else {
        s = 0.0F;
        for (std::uint32_t j = 0; j < streams; ++j) s = s + comb[j * streams + k] * residual[j * hidden + i];
    }
    return store_value(post * sub + s, store);
}

} // namespace strix::hc
