// GLM-5.3-Flash Tier A host emulation (see glm_host.hpp; orders R1-R18 of design notes).
// Build with -ffp-contract=off -fno-fast-math: every fmaf below is the only fused operation.
#include "glm/host/glm_host.hpp"

#include <algorithm>
#include <numeric>
#include <string>

namespace strix::glm::host {

namespace {

template <class T> float maxf(T x, unsigned n) {
    float m = x(0);
    for (unsigned i = 1; i < n; ++i) m = x(i) > m ? x(i) : m;
    return m;
}
inline float relu(float s) { return s > 0.0F ? s : 0.0F; }  // max(s, +0): -0 and negatives give +0

}  // namespace

void check_finite(std::span<const float> v, const char* checkpoint) {
    for (std::size_t i = 0; i < v.size(); ++i)
        if (!is_finite(v[i])) throw NonFinite(std::string("non-finite at ") + checkpoint + " index " + std::to_string(i));
}

void dense(std::span<const float> x, const Mat& w, std::span<float> y, unsigned o0) {
    if (x.size() != w.in || o0 + y.size() > w.out) throw std::invalid_argument("dense geometry");
    for (unsigned o = 0; o < y.size(); ++o)
        y[o] = chain_dot([&](unsigned i) { return x[i]; }, [&](unsigned i) { return w.at(o0 + o, i); }, w.in);
}

void rmsnorm(std::span<const float> x, const Vec* w, float eps, std::span<float> y) {
    const unsigned n = static_cast<unsigned>(x.size());
    const float ss = n >= 512 ? tree_sumsq<256>([&](unsigned i) { return x[i]; }, n)
                              : tree_sumsq<128>([&](unsigned i) { return x[i]; }, n);
    const float r = inv_sqrt(ss / static_cast<float>(n) + eps);
    for (unsigned i = 0; i < n; ++i) y[i] = w ? (x[i] * r) * w->at(i) : x[i] * r;
}

// ---------------------------------------------------------------- mHC
HcMix hc_mix(std::span<const float> X, const HcWeights& w, Identity id) {
    constexpr unsigned n = kHc * kHidden;
    std::vector<float> F(n);
    rmsnorm(X, nullptr, kEps, F);  // unweighted, eps = rms_norm_eps, BEFORE fn (R6)
    std::array<float, kHcMix> z{};
    for (unsigned m = 0; m < kHcMix; ++m)
        z[m] = id == Identity::kR1b ? r1b_dot([&](unsigned i) { return F[i]; }, [&](unsigned i) { return w.fn.at(m, i); }, n)
                                    : chain_dot([&](unsigned i) { return F[i]; }, [&](unsigned i) { return w.fn.at(m, i); }, n);
    HcMix r;
    for (unsigned j = 0; j < kHc; ++j) r.pre[j] = sigmoid(z[j] * w.scale[0] + w.base[j]) + kHcEps;
    for (unsigned j = 0; j < kHc; ++j) r.post[j] = 2.0F * sigmoid(z[kHc + j] * w.scale[1] + w.base[kHc + j]);
    auto& c = r.comb;
    for (unsigned k = 0; k < kHc * kHc; ++k) c[k] = z[2 * kHc + k] * w.scale[2] + w.base[2 * kHc + k];
    for (unsigned j = 0; j < kHc; ++j) {  // row softmax + eps
        const float m = maxf([&](unsigned k) { return c[j * kHc + k]; }, kHc);
        float e[kHc];
        for (unsigned k = 0; k < kHc; ++k) e[k] = declared_exp(c[j * kHc + k] - m);
        const float sum = ((e[0] + e[1]) + e[2]) + e[3];
        for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = e[k] / sum + kHcEps;
    }
    auto cols = [&] {
        float s[kHc];
        for (unsigned k = 0; k < kHc; ++k) s[k] = (((c[0 * kHc + k] + c[1 * kHc + k]) + c[2 * kHc + k]) + c[3 * kHc + k]) + kHcEps;
        for (unsigned j = 0; j < kHc; ++j)
            for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / s[k];
    };
    auto rows = [&] {
        float s[kHc];
        for (unsigned j = 0; j < kHc; ++j) s[j] = (((c[j * kHc + 0] + c[j * kHc + 1]) + c[j * kHc + 2]) + c[j * kHc + 3]) + kHcEps;
        for (unsigned j = 0; j < kHc; ++j)
            for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / s[j];
    };
    cols();
    for (unsigned it = 1; it < kHcIters; ++it) {
        rows();
        cols();
    }
    return r;
}

void hc_collapse(std::span<const float> X, const HcMix& m, std::span<float> x) {
    for (unsigned d = 0; d < kHidden; ++d)
        x[d] = ((m.pre[0] * X[d] + m.pre[1] * X[kHidden + d]) + m.pre[2] * X[2 * kHidden + d]) + m.pre[3] * X[3 * kHidden + d];
}

void hc_expand(std::span<const float> X, const HcMix& m, std::span<const float> y, std::span<float> Xout) {
    for (unsigned i = 0; i < kHc; ++i)
        for (unsigned d = 0; d < kHidden; ++d) {
            const float mix = ((m.comb[0 * kHc + i] * X[d] + m.comb[1 * kHc + i] * X[kHidden + d]) +
                               m.comb[2 * kHc + i] * X[2 * kHidden + d]) +
                              m.comb[3 * kHc + i] * X[3 * kHidden + d];
            Xout[i * kHidden + d] = m.post[i] * y[d] + mix;
        }
}

void hc_final(std::span<const float> X, std::span<float> h) {
    for (unsigned d = 0; d < kHidden; ++d)
        h[d] = (((X[d] + X[kHidden + d]) + X[2 * kHidden + d]) + X[3 * kHidden + d]) * 0.25F;
}

// ---------------------------------------------------------------- KDA
void kda_step(KdaState& st, const KdaInputs& in, const KdaWeights& w, std::span<float> y) {
    const unsigned H = st.heads, C = H * kKdaDim, CH = 3 * C;
    std::vector<float> a(CH);
    for (unsigned ch = 0; ch < CH; ++ch) {  // 4a: R8 conv (oldest first) + SiLU, state shift
        float* s = &st.conv[static_cast<std::size_t>(ch) * 3];
        const std::uint16_t* cw = w.conv + static_cast<std::size_t>(ch) * kConvK;
        float acc = __builtin_fmaf(bf16_to_float(cw[0]), s[0], 0.0F);
        acc = __builtin_fmaf(bf16_to_float(cw[1]), s[1], acc);
        acc = __builtin_fmaf(bf16_to_float(cw[2]), s[2], acc);
        acc = __builtin_fmaf(bf16_to_float(cw[3]), in.qkv[ch], acc);
        a[ch] = silu(acc);
        s[0] = s[1];
        s[1] = s[2];
        s[2] = in.qkv[ch];
    }
    for (unsigned h = 0; h < H; ++h) {
        float q[kKdaDim], k[kKdaDim], v[kKdaDim], g[kKdaDim], o[kKdaDim];
        for (unsigned i = 0; i < kKdaDim; ++i) {
            q[i] = a[h * kKdaDim + i];
            k[i] = a[C + h * kKdaDim + i];
            v[i] = a[2 * C + h * kKdaDim + i];
        }
        const float sq = tree_sumsq<128>([&](unsigned i) { return q[i]; }, kKdaDim);
        const float sk = tree_sumsq<128>([&](unsigned i) { return k[i]; }, kKdaDim);
        const float nq = declared_sqrt(sq + kL2Eps), nk = declared_sqrt(sk + kL2Eps);
        for (unsigned i = 0; i < kKdaDim; ++i) {
            q[i] = (q[i] / nq) * kInvSqrt128;
            k[i] = k[i] / nk;
        }
        const float expA = declared_exp(w.A_log[h]);
        for (unsigned i = 0; i < kKdaDim; ++i)
            g[i] = kGateLow * sigmoid(expA * (in.f[h * kKdaDim + i] + w.dt_bias[h * kKdaDim + i]));
        const float beta = sigmoid(in.b[h]);
        float* S = &st.S[static_cast<std::size_t>(h) * kKdaDim * kKdaDim];  // [k][v]
        for (unsigned kk = 0; kk < kKdaDim; ++kk) {                       // 4b: decay key rows
            const float dec = declared_exp(g[kk]);
            for (unsigned vv = 0; vv < kKdaDim; ++vv) S[kk * kKdaDim + vv] = S[kk * kKdaDim + vv] * dec;
        }
        for (unsigned vv = 0; vv < kKdaDim; ++vv) {
            const float u = tree_sum<128>([&](unsigned kk) { return S[kk * kKdaDim + vv] * k[kk]; }, kKdaDim);
            const float dlt = (v[vv] - u) * beta;
            for (unsigned kk = 0; kk < kKdaDim; ++kk) S[kk * kKdaDim + vv] = __builtin_fmaf(k[kk], dlt, S[kk * kKdaDim + vv]);
            o[vv] = tree_sum<128>([&](unsigned kk) { return S[kk * kKdaDim + vv] * q[kk]; }, kKdaDim);
        }
        check_finite({S, static_cast<std::size_t>(kKdaDim) * kKdaDim}, "C2 kda state");
        const float ms = tree_sumsq<128>([&](unsigned i) { return o[i]; }, kKdaDim) / static_cast<float>(kKdaDim);
        const float r = inv_sqrt(ms + kEps);  // 4c: R10
        for (unsigned i = 0; i < kKdaDim; ++i)
            y[h * kKdaDim + i] = ((o[i] * r) * w.o_norm.at(i)) * sigmoid(in.gate[h * kKdaDim + i]);
    }
}

// ---------------------------------------------------------------- indexer + MLA
void dsa_append(DsaCache& c, std::span<const float> latent, std::span<const float> x, const IndexerWeights& w) {
    for (unsigned l = 0; l < kKvLora; ++l) {
        const std::uint16_t b = float_to_bf16(latent[l]);
        if (!bf16_is_finite(b)) throw NonFinite("C3 latent cache");
        c.latent.push_back(b);
    }
    float key[kIdxDim], gs[kIdxDim], d[kIdxDim];
    dense(x, w.wk, key);
    dense(x, w.gate, gs);
    const float mu = tree_sum<128>([&](unsigned i) { return key[i]; }, kIdxDim) / static_cast<float>(kIdxDim);
    for (unsigned i = 0; i < kIdxDim; ++i) d[i] = key[i] - mu;
    const float var = tree_sumsq<128>([&](unsigned i) { return d[i]; }, kIdxDim) / static_cast<float>(kIdxDim);
    const float r = inv_sqrt(var + kLnEps);
    for (unsigned i = 0; i < kIdxDim; ++i) {
        const std::uint16_t kb = float_to_bf16((d[i] * r) * w.k_norm_w.at(i) + w.k_norm_b.at(i));
        const std::uint16_t gb = float_to_bf16(gs[i]);
        if (!bf16_is_finite(kb) || !bf16_is_finite(gb)) throw NonFinite("C3 indexer cache");
        c.ikey.push_back(kb);
        c.igate.push_back(gb);
    }
    const unsigned T = c.tokens();
    if (T % kKpool == 0) {  // pool (T/4 - 1) completed: R12
        const unsigned t0 = T - kKpool;
        for (unsigned ch = 0; ch < kIdxDim; ++ch) {
            float l[kKpool], e[kKpool], kv[kKpool];
            for (unsigned j = 0; j < kKpool; ++j) {
                l[j] = bf16_to_float(c.igate[static_cast<std::size_t>(t0 + j) * kIdxDim + ch]) + w.ape[j * kIdxDim + ch];
                kv[j] = bf16_to_float(c.ikey[static_cast<std::size_t>(t0 + j) * kIdxDim + ch]);
            }
            const float m = maxf([&](unsigned j) { return l[j]; }, kKpool);
            for (unsigned j = 0; j < kKpool; ++j) e[j] = declared_exp(l[j] - m);
            const float sum = ((e[0] + e[1]) + e[2]) + e[3];
            float p[kKpool];
            for (unsigned j = 0; j < kKpool; ++j) p[j] = e[j] / sum;
            const float K = ((p[0] * kv[0] + p[1] * kv[1]) + p[2] * kv[2]) + p[3] * kv[3];
            if (!is_finite(K)) throw NonFinite("C4 pooled key");
            c.pooled.push_back(K);
        }
    }
}

std::vector<std::uint32_t> topk_ordered(std::span<const float> score, unsigned k) {
    std::vector<std::uint32_t> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0U);
    std::stable_sort(idx.begin(), idx.end(), [&](std::uint32_t a, std::uint32_t b) {
        const std::uint32_t ka = ordered_key(score[a]), kb = ordered_key(score[b]);
        return ka != kb ? ka > kb : a < b;
    });
    idx.resize(std::min<std::size_t>(k, idx.size()));
    std::sort(idx.begin(), idx.end());
    return idx;
}

std::vector<std::uint32_t> indexer_select(const DsaCache& c, unsigned t, std::span<const float> qI,
                                          std::span<const float> x, const IndexerWeights& w) {
    const unsigned P = (t + 1) / kKpool;
    if (P > c.pools()) throw std::logic_error("indexer: pools not built up to the query position");
    std::vector<std::uint32_t> out;
    if (P > 0) {
        float wh[kIdxHeads];
        dense(x, w.weights_proj, wh);
        for (unsigned h = 0; h < kIdxHeads; ++h) wh[h] = wh[h] * kInvSqrt32;
        std::vector<float> I(P);
        for (unsigned p = 0; p < P; ++p) {
            const float* K = &c.pooled[static_cast<std::size_t>(p) * kIdxDim];
            float acc = 0.0F;
            for (unsigned h = 0; h < kIdxHeads; ++h) {
                const float dot = chain_dot([&](unsigned i) { return qI[h * kIdxDim + i]; }, [&](unsigned i) { return K[i]; }, kIdxDim);
                acc = __builtin_fmaf(wh[h], relu(dot * kInvSqrt128), acc);
            }
            I[p] = acc;
        }
        for (std::uint32_t p : topk_ordered(I, std::min(kIdxPools, P)))
            for (unsigned j = 0; j < kKpool; ++j) out.push_back(p * kKpool + j);
    }
    for (unsigned pos = P * kKpool; pos <= t; ++pos) out.push_back(pos);  // tail: (t + 1) mod 4 positions
    return out;
}

void mla_attention(const DsaCache& c, std::span<const std::uint32_t> tokens, std::span<const float> q, const Mat& kv_b,
                   unsigned heads, std::span<float> o) {
    const unsigned n = static_cast<unsigned>(tokens.size());
    if (n == 0) throw std::logic_error("mla: empty token list");
    std::vector<float> s(n);
    for (unsigned h = 0; h < heads; ++h) {
        float qt[kKvLora], ob[kKvLora];
        for (unsigned l = 0; l < kKvLora; ++l)
            qt[l] = chain_dot([&](unsigned d) { return q[h * kMlaDim + d]; }, [&](unsigned d) { return kv_b.at(h * 2 * kMlaDim + d, l); }, kMlaDim);
        for (unsigned t = 0; t < n; ++t) {
            const std::uint16_t* lat = &c.latent[static_cast<std::size_t>(tokens[t]) * kKvLora];
            s[t] = chain_dot([&](unsigned l) { return qt[l]; }, [&](unsigned l) { return bf16_to_float(lat[l]); }, kKvLora) * kMlaScale;
        }
        const float m = maxf([&](unsigned t) { return s[t]; }, n);
        for (unsigned t = 0; t < n; ++t) s[t] = declared_exp(s[t] - m);
        const float sum = tree_sum<256>([&](unsigned t) { return s[t]; }, n);
        for (unsigned t = 0; t < n; ++t) s[t] = s[t] / sum;
        for (unsigned l = 0; l < kKvLora; ++l)
            ob[l] = tree_fma<256>([&](unsigned t) { return s[t]; },
                                  [&](unsigned t) { return bf16_to_float(c.latent[static_cast<std::size_t>(tokens[t]) * kKvLora + l]); }, n);
        for (unsigned d = 0; d < kMlaDim; ++d)
            o[h * kMlaDim + d] = chain_dot([&](unsigned l) { return ob[l]; }, [&](unsigned l) { return kv_b.at(h * 2 * kMlaDim + kMlaDim + d, l); }, kKvLora);
    }
}

// ---------------------------------------------------------------- router, combine, head
void dense_r1b(std::span<const float> x, const Mat& w, std::span<float> y) {
    if (x.size() != w.in || y.size() > w.out || w.in % kR1bBlock != 0 || w.in / kR1bBlock > kR1bMaxBlocks ||
        ((w.in / kR1bBlock) & (w.in / kR1bBlock - 1)) != 0)
        throw std::invalid_argument("dense_r1b geometry (K = 256 * 2^j, <= 64 blocks)");
    for (unsigned o = 0; o < y.size(); ++o)
        y[o] = r1b_dot([&](unsigned i) { return x[i]; }, [&](unsigned i) { return w.at(o, i); }, w.in);
}

Route route(std::span<const float> x, const Mat& router, const float* bias, Identity id) {
    std::vector<float> s(kExperts), choice(kExperts);
    if (id == Identity::kR1b) dense_r1b(x, router, s);
    else dense(x, router, s);
    for (unsigned e = 0; e < kExperts; ++e) {
        s[e] = sigmoid(s[e]);
        choice[e] = s[e] + bias[e];
    }
    const auto ids = topk_ordered(choice, kTopE);
    Route r;
    float den = 0.0F;
    for (unsigned k = 0; k < kTopE; ++k) {
        r.id[k] = ids[k];
        den = den + s[ids[k]];
    }
    den = den + kRouteDenEps;
    for (unsigned k = 0; k < kTopE; ++k) r.w[k] = (s[ids[k]] / den) * kRouteScale;
    return r;
}

void moe_combine(const Route& r, std::span<const float> expert_out, std::span<const float> shared, std::span<float> y) {
    for (unsigned d = 0; d < kHidden; ++d) {
        float acc = 0.0F;
        for (unsigned k = 0; k < kTopE; ++k) acc = __builtin_fmaf(r.w[k], expert_out[static_cast<std::size_t>(k) * kHidden + d], acc);
        y[d] = acc + shared[d];
    }
}

ArgMax argmax_local(std::span<const float> logits, std::uint32_t id0) {
    check_finite(logits, "C5 logits");
    ArgMax a{logits[0], id0};
    for (std::size_t i = 1; i < logits.size(); ++i)
        if (logits[i] > a.value) a = {logits[i], id0 + static_cast<std::uint32_t>(i)};
    return a;
}
ArgMax argmax_merge(ArgMax a, ArgMax b) {
    if (b.value > a.value || (b.value == a.value && b.id < a.id)) return b;
    return a;
}

}  // namespace strix::glm::host
