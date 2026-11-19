#pragma once
// Group pooling of the compressed group indexer: the per-lane statements shared by the host oracle test and the HIP
// block core (src/attn_core/group_pool.hpp). Contract and traits: strix/attn/group_indexer.hpp (Pooling).
//
// Lane = output channel o of the pooled key (head_dim lanes). Two traits:
//   GatedSoftmax     (GLM DSA): logits = bf16 gate + ape, values = bf16 key, both cached per token; gated channel.
//   GatedSoftmaxApe:            values = Wv tk, logits = Wg tk + ape (fp32 fma chains over the input index ascending,
//                               from +0), gated channel, then RMSNorm (serial fma chain of x * x over the channel index
//                               ascending, r = 1 / sqrt(ss / head_dim + eps)) and bf16(norm * (x * r)). RoPE of the
//                               group key (group_rope_slot) is the caller's rotary stage, applied to the bf16 output.
// Contraction: these statements state the declared tree. A HIP TU that needs the oracle's bits puts
// `#pragma clang fp contract(off)` before including this header (the repo's declared-arithmetic convention); GLM's
// TUs keep their default, so their bits are what they were before the lift.
#include "strix/attn/group_indexer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace strix::attn::group_pool {
namespace gi = strix::attn::group_indexer;

STRIX_GI_HD inline float bf16_value(std::uint16_t b) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U); }

// Direct-call functor of a declared transcendental (e.g. Fn<declared_exp>{}): inlined like a direct call.
template <float (*F)(float)> struct Fn {
    STRIX_GI_HD float operator()(float x) const { return F(x); }
};

STRIX_GI_HD inline float sqrt_rn(float x) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return sqrtf(x);
#else
    return std::sqrt(x);
#endif
}

// GatedSoftmax: one channel of one complete group whose R token rows start at key / gate (row stride `dim`);
// ape [R][dim]. Same statements as the former GLM dsa_pool_channel.
template <unsigned R, class Exp>
STRIX_GI_HD inline float gated_bf16_channel(const std::uint16_t* key, const std::uint16_t* gate, const float* ape,
                                            unsigned channel, unsigned dim, Exp exp_fn) {
    float logits[R], values[R];
    for (unsigned j = 0; j < R; ++j) {
        const unsigned offset = j * dim + channel;
        logits[j] = bf16_value(gate[offset]) + ape[offset];
        values[j] = bf16_value(key[offset]);
    }
    return gi::gated_pool_channel<R>(logits, values, exp_fn);
}

// Weights of the GatedSoftmaxApe pool as plain fp32 rows (contract struct gi::GatedApeWeights, (out, in) row-major).
// A binder with quantised weights supplies the same five members (v, g, ape, norm, eps) with an exact dequantisation.
struct F32Ape {
    gi::GatedApeWeights w;
    unsigned dim;
    STRIX_GI_HD float v(unsigned o, unsigned i) const { return w.wv[static_cast<std::size_t>(o) * dim + i]; }
    STRIX_GI_HD float g(unsigned o, unsigned i) const { return w.wg[static_cast<std::size_t>(o) * dim + i]; }
    STRIX_GI_HD float ape(unsigned j, unsigned o) const { return w.ape[static_cast<std::size_t>(j) * dim + o]; }
    STRIX_GI_HD float norm(unsigned o) const { return w.norm[o]; }
    STRIX_GI_HD float eps() const { return w.eps; }
};

// GatedSoftmaxApe, phase 1 (lane o): pooled channel o before the norm. keys [R][dim] = the cached bf16 token keys as float.
template <unsigned R, class Wt, class Exp>
STRIX_GI_HD inline float gated_ape_channel(const float* keys, unsigned dim, const Wt& w, unsigned o, Exp exp_fn) {
    float logits[R], values[R];
    for (unsigned j = 0; j < R; ++j) {
        const float* k = keys + static_cast<std::size_t>(j) * dim;
        float v = 0.0F, g = 0.0F;
        for (unsigned i = 0; i < dim; ++i) {
            v = gi::fma3(w.v(o, i), k[i], v);
            g = gi::fma3(w.g(o, i), k[i], g);
        }
        values[j] = v;
        logits[j] = g + w.ape(j, o);
    }
    return gi::gated_pool_channel<R>(logits, values, exp_fn);
}

// Phase 2 (every lane, the same serial chain): 1 / RMS of the pooled vector x [dim].
STRIX_GI_HD inline float rms_inv(const float* x, unsigned dim, float eps) {
    float ss = 0.0F;
    for (unsigned i = 0; i < dim; ++i) ss = gi::fma3(x[i], x[i], ss);
    return 1.0F / sqrt_rn(ss / static_cast<float>(dim) + eps);
}

// Phase 3 (lane o): the bf16-valued normalised channel.
STRIX_GI_HD inline float rms_out(float norm, float x, float r) { return gi::bf16_round(norm * (x * r)); }

}  // namespace strix::attn::group_pool
