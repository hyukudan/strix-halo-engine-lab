#pragma once
// CPU references of the gated delta rule (test only): the serial recurrence of gdn.hpp (any float type, key order ascending, the
// incumbent kernel's order) and the chunk form of strix/cuda_core/gdn_chunk.hpp (float, FMA like the kernels).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "strix/cuda_core/gdn_chunk.hpp"

namespace gdnref {
constexpr int D = 128;
constexpr int C = static_cast<int>(strix::cuda_core::gdn_chunk::kChunk);

struct Problem {
    int key_heads = 2, value_heads = 6, rows = 0;
    std::vector<float> q, k;     // [rows][key_heads][D] (q pre-scaled, both l2-normalised)
    std::vector<float> v;        // [rows][value_heads][D]
    std::vector<float> g, beta;  // [rows][value_heads] log decay, beta
    [[nodiscard]] const float* qr(int r, int h) const { return &q[(static_cast<std::size_t>(r) * key_heads + h % key_heads) * D]; }
    [[nodiscard]] const float* kr(int r, int h) const { return &k[(static_cast<std::size_t>(r) * key_heads + h % key_heads) * D]; }
    [[nodiscard]] const float* vr(int r, int h) const { return &v[(static_cast<std::size_t>(r) * value_heads + h) * D]; }
};
// state [value_heads][D k][D v]; out [rows][value_heads][D] (pre-norm o); rows [r0, r1)
template <class T>
void serial(const Problem& p, int r0, int r1, std::vector<T>& S, std::vector<T>& out) {
    if (out.size() != static_cast<std::size_t>(p.rows) * p.value_heads * D) out.assign(static_cast<std::size_t>(p.rows) * p.value_heads * D, T(0));
    for (int r = r0; r < r1; ++r)
        for (int h = 0; h < p.value_heads; ++h) {
            const T decay = std::exp(static_cast<T>(p.g[static_cast<std::size_t>(r) * p.value_heads + h]));
            const T beta = static_cast<T>(p.beta[static_cast<std::size_t>(r) * p.value_heads + h]);
            const float *q = p.qr(r, h), *k = p.kr(r, h), *v = p.vr(r, h);
            T* s = &S[static_cast<std::size_t>(h) * D * D];
            for (int c = 0; c < D; ++c) {
                T kv = 0;
                for (int e = 0; e < D; ++e) {
                    s[e * D + c] = s[e * D + c] * decay;
                    kv = kv + s[e * D + c] * static_cast<T>(k[e]);
                }
                const T dl = (static_cast<T>(v[c]) - kv) * beta;
                T acc = 0;
                for (int e = 0; e < D; ++e) {
                    s[e * D + c] = s[e * D + c] + static_cast<T>(k[e]) * dl;
                    acc = acc + s[e * D + c] * static_cast<T>(q[e]);
                }
                out[(static_cast<std::size_t>(r) * p.value_heads + h) * D + c] = acc;
            }
        }
}
// Chunk form over rows [r0, r1) (chunks start at r0; the last chunk is padded), float + fma, the kernels' algebra.
inline void chunked(const Problem& p, int r0, int r1, std::vector<float>& S, std::vector<float>& out) {
    if (out.size() != static_cast<std::size_t>(p.rows) * p.value_heads * D) out.assign(static_cast<std::size_t>(p.rows) * p.value_heads * D, 0.f);
    for (int h = 0; h < p.value_heads; ++h) {
        float* s = &S[static_cast<std::size_t>(h) * D * D];
        for (int c0 = r0; c0 < r1; c0 += C) {
            float G[C], beta[C], kb[C], A[C][C] = {}, T[C][C] = {}, P[C][C] = {};
            std::vector<float> K(C * D, 0.f), Q(C * D, 0.f), V(C * D, 0.f), W(C * D, 0.f), U(C * D, 0.f);
            float acc = 0.f;
            for (int i = 0; i < C; ++i) {
                const int r = c0 + i;
                const bool in = r < r1;
                G[i] = acc = acc + (in ? p.g[static_cast<std::size_t>(r) * p.value_heads + h] : 0.f);
                beta[i] = in ? p.beta[static_cast<std::size_t>(r) * p.value_heads + h] : 0.f;
                for (int e = 0; e < D && in; ++e) {
                    K[i * D + e] = p.kr(r, h)[e];
                    Q[i * D + e] = p.qr(r, h)[e];
                    V[i * D + e] = p.vr(r, h)[e];
                }
            }
            for (int i = 0; i < C; ++i) {
                kb[i] = beta[i] * std::exp(G[i]);
                for (int j = 0; j <= i; ++j) {
                    float kk = 0.f, qk = 0.f;
                    for (int e = 0; e < D; ++e) {
                        kk = std::fma(K[i * D + e], K[j * D + e], kk);
                        qk = std::fma(Q[i * D + e], K[j * D + e], qk);
                    }
                    const float dec = std::exp(G[i] - G[j]);
                    if (j < i) A[i][j] = beta[i] * dec * kk;
                    P[i][j] = dec * qk;
                }
            }
            for (int i = 0; i < C; ++i)
                for (int m = 0; m < C; ++m) {
                    float x = i == m ? 1.f : 0.f;
                    for (int j = 0; j < i; ++j) x = std::fma(-A[i][j], T[j][m], x);
                    T[i][m] = x;
                }
            for (int i = 0; i < C; ++i)
                for (int e = 0; e < D; ++e) {
                    float w = 0.f, u = 0.f;
                    for (int j = 0; j <= i; ++j) {
                        w = std::fma(T[i][j] * kb[j], K[j * D + e], w);
                        u = std::fma(T[i][j] * beta[j], V[j * D + e], u);
                    }
                    W[i * D + e] = w;
                    U[i * D + e] = u;
                }
            std::vector<float> Uc(C * D), O(C * D);
            for (int i = 0; i < C; ++i)
                for (int c = 0; c < D; ++c) {
                    float ws = 0.f, qs = 0.f;
                    for (int e = 0; e < D; ++e) {
                        ws = std::fma(W[i * D + e], s[e * D + c], ws);
                        qs = std::fma(Q[i * D + e], s[e * D + c], qs);
                    }
                    Uc[i * D + c] = U[i * D + c] - ws;
                    O[i * D + c] = std::exp(G[i]) * qs;
                }
            for (int i = 0; i < C; ++i)
                for (int c = 0; c < D; ++c) {
                    float o = O[i * D + c];
                    for (int j = 0; j < C; ++j) o = std::fma(j <= i ? P[i][j] : 0.f, Uc[j * D + c], o);
                    if (c0 + i < r1) out[(static_cast<std::size_t>(c0 + i) * p.value_heads + h) * D + c] = o;
                }
            const float dlast = std::exp(G[C - 1]);
            for (int i = 0; i < C; ++i) {
                const float f = std::exp(G[C - 1] - G[i]);
                for (int c = 0; c < D; ++c) Uc[i * D + c] *= f;
            }
            for (int e = 0; e < D; ++e)
                for (int c = 0; c < D; ++c) {
                    float x = dlast * s[e * D + c];
                    for (int i = 0; i < C; ++i) x = std::fma(K[i * D + e], Uc[i * D + c], x);
                    s[e * D + c] = x;
                }
        }
    }
}
}  // namespace gdnref
