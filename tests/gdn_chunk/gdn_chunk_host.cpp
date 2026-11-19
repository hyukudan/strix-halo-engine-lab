// Host oracle of the FAST chunked GDN (strix/cuda_core/gdn_chunk.hpp, gdn-chunk16-f32-v1). Invariants:
//  1. the chunk algebra equals the serial recurrence: chunk(float) vs serial(double) relative RMS of o and of the final state is
//     within the bound and of the same order as the incumbent serial(float) error (rows 1..300, partial chunks, decays from
//     ~0.05 to ~0.999, betas 0..1);
//  2. padding rows (beta 0, g 0, zero q/k/v) leave the state unchanged and emit nothing: chunk calls of any length compose with
//     each other and with serial decode rows (prefill -> decode hand-off);
//  3. workspace layout: blocks float4-aligned, workspace_floats covers every (chunk, head) block.
//  4. tf32 scan (gdn-chunk16-tf32-v1): the fragment maps of gdn_chunk.hpp tc are bijections and a C fragment is the next
//     product's A fragment under perm; every kernel fragment load at the tc strides is bank-conflict free; a lane-level
//     emulation of chunk_scan_tc_kernel's loop (TF32 operands) matches serial(double) within 3e-3, and is visibly wrong
//     without the permutation (negative control).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "gdn_ref.hpp"

namespace gc = strix::cuda_core::gdn_chunk;
using gdnref::D;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++fails;
}
template <class A, class B>
static double rel(const std::vector<A>& a, const std::vector<B>& b) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

static gdnref::Problem make(int rows, unsigned seed, float g_lo, float g_hi) {
    gdnref::Problem p;
    p.rows = rows;
    std::mt19937 rng(seed);
    std::normal_distribution<float> n01(0.f, 1.f);
    std::uniform_real_distribution<float> ug(g_lo, g_hi), ub(-3.f, 3.f);
    auto unit = [&](std::vector<float>& x, int heads, float scale) {
        x.resize(static_cast<std::size_t>(rows) * heads * D);
        for (std::size_t o = 0; o < x.size(); o += D) {
            double ss = 0;
            for (int e = 0; e < D; ++e) ss += static_cast<double>((x[o + e] = n01(rng))) * x[o + e];
            const float inv = static_cast<float>(scale / std::sqrt(ss + 1e-6));
            for (int e = 0; e < D; ++e) x[o + e] *= inv;
        }
    };
    unit(p.q, p.key_heads, 1.0f / std::sqrt(128.0f));
    unit(p.k, p.key_heads, 1.0f);
    p.v.resize(static_cast<std::size_t>(rows) * p.value_heads * D);
    for (float& x : p.v) x = n01(rng);
    p.g.resize(static_cast<std::size_t>(rows) * p.value_heads);
    p.beta.resize(p.g.size());
    for (std::size_t i = 0; i < p.g.size(); ++i) {
        p.g[i] = -std::exp(ug(rng));                        // log decay in [-exp(g_hi), -exp(g_lo)]
        p.beta[i] = 1.0f / (1.0f + std::exp(-ub(rng)));     // sigmoid(b)
    }
    return p;
}
static std::vector<float> init_state(int vh, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.f, 0.3f);
    std::vector<float> s(static_cast<std::size_t>(vh) * D * D);
    for (float& x : s) x = n(rng);
    return s;
}

// ---- 4. tf32 scan (gdn-chunk16-tf32-v1, src/cuda_core/gdn/gdn_chunk.cu chunk_scan_tc_kernel): lane-level emulation with the
// shared fragment maps (gdn_chunk.hpp tc), the kernel's shared-memory index expressions and TF32 rounding of every product operand.
namespace tc = gc::tc;
constexpr int TC = static_cast<int>(gc::kChunk);
constexpr int WTS = static_cast<int>(tc::kWtStride), QS = static_cast<int>(tc::kQStride), KS = static_cast<int>(tc::kKStride),
              US = static_cast<int>(tc::kUStride), PTS = static_cast<int>(tc::kPtStride);
constexpr int SWT = static_cast<int>(tc::Stage::wt), SQ = static_cast<int>(tc::Stage::q), SK = static_cast<int>(tc::Stage::k),
              SU = static_cast<int>(tc::Stage::u), SPT = static_cast<int>(tc::Stage::pt), SG = static_cast<int>(tc::Stage::g);

static float tf32(float x) {  // cvt.rna.tf32.f32: nearest, ties away from zero, 10 mantissa bits kept
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    if ((u & 0x7f800000u) != 0x7f800000u) u = (u + 0x1000u) & 0xffffe000u;
    float r;
    std::memcpy(&r, &u, sizeof r);
    return r;
}
// mma.m16n8k8 of one warp from the lane fragments: d += A B, A / B assembled through the A / B maps, d scattered by the C map
static void mma(float (&d)[32][4], const float (&a)[32][4], const float (&b)[32][2]) {
    float A[16][8] = {}, B[8][8] = {};
    for (int l = 0; l < 32; ++l) {
        for (int e = 0; e < 4; ++e) A[tc::a_row(l, e)][tc::a_kslot(l, e)] = a[l][e];
        for (int e = 0; e < 2; ++e) B[tc::b_kslot(l, e)][tc::b_col(l)] = b[l][e];
    }
    for (int l = 0; l < 32; ++l)
        for (int e = 0; e < 4; ++e) {
            double s = d[l][e];
            for (int k = 0; k < 8; ++k) s += static_cast<double>(A[tc::c_row(l, e)][k]) * B[k][tc::c_col(l, e)];
            d[l][e] = static_cast<float>(s);
        }
}
// K1 of chunk c, head h into one operand stage at the tc strides (the chunk algebra of gdn_ref.hpp chunked, float + fma)
static void stage_chunk(const gdnref::Problem& p, int h, int c, int c0, std::vector<float>& st) {
    std::fill(st.begin(), st.end(), 0.f);
    float G[TC], beta[TC], kb[TC], A[TC][TC] = {}, T[TC][TC] = {}, P[TC][TC] = {}, acc = 0.f;
    std::vector<float> K(TC * D, 0.f), Q(TC * D, 0.f), V(TC * D, 0.f);
    for (int i = 0; i < TC; ++i) {
        const int r = c * TC + i;
        const bool in = r < p.rows;
        G[i] = acc = acc + (in ? p.g[static_cast<std::size_t>(r) * p.value_heads + h] : 0.f);
        beta[i] = in ? p.beta[static_cast<std::size_t>(r) * p.value_heads + h] : 0.f;
        for (int e = 0; e < D && in; ++e) {
            K[i * D + e] = p.kr(r, h)[e];
            Q[i * D + e] = p.qr(r, h)[e];
            V[i * D + e] = p.vr(r, h)[e];
        }
    }
    for (int i = 0; i < TC; ++i) {
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
    for (int i = 0; i < TC; ++i)
        for (int m = 0; m < TC; ++m) {
            float x = i == m ? 1.f : 0.f;
            for (int j = 0; j < i; ++j) x = std::fma(-A[i][j], T[j][m], x);
            T[i][m] = x;
        }
    for (int i = 0; i < TC; ++i) {
        for (int e = 0; e < D; ++e) {
            float w = 0.f, u = 0.f;
            for (int j = 0; j <= i; ++j) {
                w = std::fma(T[i][j] * kb[j], K[j * D + e], w);
                u = std::fma(T[i][j] * beta[j], V[j * D + e], u);
            }
            st[static_cast<std::size_t>(SWT + e * WTS + i)] = w;
            if (e >= c0 && e < c0 + static_cast<int>(tc::kCtaCols)) st[static_cast<std::size_t>(SU + i * US + e - c0)] = u;
            st[static_cast<std::size_t>(SQ + i * QS + e)] = Q[i * D + e];
            st[static_cast<std::size_t>(SK + i * KS + e)] = K[i * D + e];
        }
        for (int j = 0; j <= i; ++j) st[static_cast<std::size_t>(SPT + j * PTS + i)] = P[i][j];
        st[static_cast<std::size_t>(SG + i)] = G[i];
    }
}
// The kernel's chunk loop for value head h, both 64-column CTA tiles, 4 warps each. S: [D k][D v] of head h (in/out);
// out: [rows][value_heads][D]. use_perm = false feeds C fragments as A without the permutation (negative control).
static void emulate_tc(const gdnref::Problem& p, int h, float* S, std::vector<float>& out, bool use_perm) {
    const int nch = (p.rows + TC - 1) / TC;
    std::vector<float> st(tc::Stage::floats);
    auto from_c = [&](int e) { return use_perm ? tc::a_from_c(e) : e; };
    static float acc[tc::kWarps][D / 8][32][4];
    for (int c0 = 0; c0 < D; c0 += static_cast<int>(tc::kCtaCols)) {
        for (int w = 0; w < static_cast<int>(tc::kWarps); ++w)
            for (int nt = 0; nt < D / 8; ++nt)
                for (int l = 0; l < 32; ++l)
                    for (int e = 0; e < 4; ++e) acc[w][nt][l][e] = S[(8 * nt + tc::c_col(l, e)) * D + c0 + 16 * w + tc::c_row(l, e)];
        for (int c = 0; c < nch; ++c) {
            stage_chunk(p, h, c, c0, st);
            const float* s = st.data();
            for (int w = 0; w < static_cast<int>(tc::kWarps); ++w) {
                float x[2][32][4] = {}, y[2][32][4] = {}, u[2][32][4], o[2][32][4], a[32][4], b[32][2], ua[2][32][4];
                for (int nt = 0; nt < D / 8; ++nt) {
                    for (int l = 0; l < 32; ++l)
                        for (int e = 0; e < 4; ++e) a[l][e] = tf32(acc[w][nt][l][from_c(e)]);
                    for (int ni = 0; ni < 2; ++ni) {
                        for (int l = 0; l < 32; ++l) {
                            const int k0 = 8 * nt + 2 * (l & 3), i = 8 * ni + (l >> 2);
                            b[l][0] = tf32(s[SWT + k0 * WTS + i]);
                            b[l][1] = tf32(s[SWT + (k0 + 1) * WTS + i]);
                        }
                        mma(x[ni], a, b);
                        for (int l = 0; l < 32; ++l) {
                            const int k0 = 8 * nt + 2 * (l & 3), i = 8 * ni + (l >> 2);
                            b[l][0] = tf32(s[SQ + i * QS + k0]);
                            b[l][1] = tf32(s[SQ + i * QS + k0 + 1]);
                        }
                        mma(y[ni], a, b);
                    }
                }
                for (int ni = 0; ni < 2; ++ni)
                    for (int l = 0; l < 32; ++l)
                        for (int e = 0; e < 4; ++e) {
                            const int i = 8 * ni + tc::c_col(l, e);
                            u[ni][l][e] = s[SU + i * US + 16 * w + tc::c_row(l, e)] - x[ni][l][e];
                            o[ni][l][e] = std::exp(s[SG + i]) * y[ni][l][e];
                        }
                for (int kj = 0; kj < 2; ++kj) {
                    for (int l = 0; l < 32; ++l)
                        for (int e = 0; e < 4; ++e) a[l][e] = tf32(u[kj][l][from_c(e)]);
                    for (int ni = 0; ni < 2; ++ni) {
                        for (int l = 0; l < 32; ++l) {
                            const int j0 = 8 * kj + 2 * (l & 3), i = 8 * ni + (l >> 2);
                            b[l][0] = tf32(s[SPT + j0 * PTS + i]);
                            b[l][1] = tf32(s[SPT + (j0 + 1) * PTS + i]);
                        }
                        mma(o[ni], a, b);
                    }
                }
                for (int ni = 0; ni < 2; ++ni)
                    for (int l = 0; l < 32; ++l)
                        for (int e = 0; e < 4; ++e) {
                            const int r = c * TC + 8 * ni + tc::c_col(l, e);
                            if (r < p.rows)
                                out[(static_cast<std::size_t>(r) * static_cast<std::size_t>(p.value_heads) + static_cast<std::size_t>(h)) * D +
                                    static_cast<std::size_t>(c0 + 16 * w + tc::c_row(l, e))] = o[ni][l][e];
                        }
                const float glast = s[SG + TC - 1], dlast = std::exp(glast);
                for (int kj = 0; kj < 2; ++kj) {
                    for (int l = 0; l < 32; ++l)
                        for (int e = 0; e < 4; ++e) u[kj][l][e] *= std::exp(glast - s[SG + 8 * kj + tc::c_col(l, e)]);
                    for (int l = 0; l < 32; ++l)
                        for (int e = 0; e < 4; ++e) ua[kj][l][e] = tf32(u[kj][l][from_c(e)]);
                }
                for (int nt = 0; nt < D / 8; ++nt) {
                    for (int l = 0; l < 32; ++l)
                        for (int e = 0; e < 4; ++e) acc[w][nt][l][e] *= dlast;
                    for (int kj = 0; kj < 2; ++kj) {
                        for (int l = 0; l < 32; ++l) {
                            const int j0 = 8 * kj + 2 * (l & 3), kk = 8 * nt + (l >> 2);
                            b[l][0] = tf32(s[SK + j0 * KS + kk]);
                            b[l][1] = tf32(s[SK + (j0 + 1) * KS + kk]);
                        }
                        mma(acc[w][nt], ua[kj], b);
                    }
                }
            }
        }
        for (int w = 0; w < static_cast<int>(tc::kWarps); ++w)
            for (int nt = 0; nt < D / 8; ++nt)
                for (int l = 0; l < 32; ++l)
                    for (int e = 0; e < 4; ++e) S[(8 * nt + tc::c_col(l, e)) * D + c0 + 16 * w + tc::c_row(l, e)] = acc[w][nt][l][e];
    }
}
template <class F>
static bool distinct_banks(F addr, int words) {  // 32-bit words: no two on one bank
    bool seen[32] = {};
    for (int l = 0; l < words; ++l) {
        const int b = addr(l) & 31;
        if (seen[b]) return false;
        seen[b] = true;
    }
    return true;
}

int main() {
    // 3. layout
    {
        using L = gc::Layout;
        const bool al = L::wt % 4 == 0 && L::u % 4 == 0 && L::pt % 4 == 0 && L::g % 4 == 0 && L::block % 4 == 0;
        const bool cover = gc::workspace_floats(48, 4096) == gc::block_offset(255, 47, 48) + L::block &&
                           gc::workspace_floats(48, 4097) == gc::block_offset(256, 47, 48) + L::block && gc::chunks(1) == 1 &&
                           gc::chunks(16) == 1 && gc::chunks(17) == 2;
        check(al && cover && L::block == 4368, "layout: float4-aligned blocks, workspace covers every (chunk, head), 4,368 floats/block");
    }
    // 1. chunk algebra vs serial, several lengths and decay regimes
    struct Case { int rows; float glo, ghi; const char* name; };
    const Case cases[] = {{1, -7.f, -1.f, "rows 1"},          {15, -7.f, -1.f, "rows 15 (one partial chunk)"},
                          {16, -7.f, -1.f, "rows 16"},        {17, -7.f, -1.f, "rows 17"},
                          {300, -7.f, 1.f, "rows 300 mixed decays 0.07..0.999"},
                          {300, -9.f, -5.f, "rows 300 long memory (decay > 0.993)"}};
    unsigned seed = 1;
    for (const Case& cs : cases) {
        const auto p = make(cs.rows, seed++, cs.glo, cs.ghi);
        const auto s0 = init_state(p.value_heads, seed++);
        std::vector<double> sd(s0.begin(), s0.end()), od;
        std::vector<float> sf = s0, of, sc = s0, oc;
        gdnref::serial<double>(p, 0, p.rows, sd, od);
        gdnref::serial<float>(p, 0, p.rows, sf, of);
        gdnref::chunked(p, 0, p.rows, sc, oc);
        const double eo = rel(oc, od), es = rel(sc, sd), fo = rel(of, od), fs = rel(sf, sd);
        char msg[200];
        std::snprintf(msg, sizeof msg, "%s: chunk o %.2e state %.2e | serial-float o %.2e state %.2e (bound 2e-5 and <= 8x serial-float + 1e-6)",
                      cs.name, eo, es, fo, fs);
        check(eo < 2e-5 && es < 2e-5 && eo <= 8 * fo + 1e-6 && es <= 8 * fs + 1e-6, msg);
    }
    // 2. composition: chunk calls of 37 + 64 + 23 rows, then 20 serial decode rows == serial(double) over all 144 rows
    {
        const auto p = make(144, 99, -7.f, 0.5f);
        const auto s0 = init_state(p.value_heads, 98);
        std::vector<double> sd(s0.begin(), s0.end()), od;
        gdnref::serial<double>(p, 0, 144, sd, od);
        std::vector<float> sc = s0, oc;
        gdnref::chunked(p, 0, 37, sc, oc);
        gdnref::chunked(p, 37, 101, sc, oc);
        gdnref::chunked(p, 101, 124, sc, oc);
        gdnref::serial<float>(p, 124, 144, sc, oc);
        char msg[160];
        std::snprintf(msg, sizeof msg, "composition 37+64+23 chunk rows + 20 serial rows: o %.2e state %.2e (bound 2e-5)", rel(oc, od),
                      rel(sc, sd));
        check(rel(oc, od) < 2e-5 && rel(sc, sd) < 2e-5, msg);
    }
    // 4a. fragment maps: bijections onto their tiles; the A element (row, k-slot) of a[e] = c[a_from_c(e)] is the C element
    //     (row, perm(k-slot)); the B k-slots the kernel loads (keys 2t, 2t + 1) are perm(t), perm(t + 4)
    {
        int ca[16][8] = {}, cb[8][8] = {}, cc[16][8] = {};
        bool reuse = true, bload = true, perm_ok = true;
        for (int l = 0; l < 32; ++l) {
            for (int e = 0; e < 4; ++e) {
                ++ca[tc::a_row(l, e)][tc::a_kslot(l, e)];
                ++cc[tc::c_row(l, e)][tc::c_col(l, e)];
                const int f = tc::a_from_c(e);
                reuse = reuse && tc::a_row(l, e) == tc::c_row(l, f) && tc::perm(tc::a_kslot(l, e)) == tc::c_col(l, f);
            }
            for (int e = 0; e < 2; ++e) {
                ++cb[tc::b_kslot(l, e)][tc::b_col(l)];
                bload = bload && tc::perm(tc::b_kslot(l, e)) == 2 * (l & 3) + e;
            }
        }
        bool bij = true;
        for (int r = 0; r < 16; ++r)
            for (int k = 0; k < 8; ++k) bij = bij && ca[r][k] == 1 && cc[r][k] == 1 && (r >= 8 || cb[r][k] == 1);
        bool pb[8] = {};
        for (int s = 0; s < 8; ++s) pb[tc::perm(s)] = true;
        for (bool x : pb) perm_ok = perm_ok && x;
        check(bij && reuse && bload && perm_ok, "tf32 fragment maps: A/B/C bijections, C fragment = A fragment under perm, B loads at perm");
    }
    // 4b. every fragment load of the kernel at the tc strides is bank-conflict free; stage offsets 16-byte aligned for cp.async
    {
        bool ok = true;
        for (int e = 0; e < 2; ++e) {
            ok = ok && distinct_banks([&](int l) { return (2 * (l & 3) + e) * WTS + (l >> 2); }, 32);  // Wt b0 / b1
            ok = ok && distinct_banks([&](int l) { return (2 * (l & 3) + e) * KS + (l >> 2); }, 32);   // K b0 / b1
            ok = ok && distinct_banks([&](int l) { return (2 * (l & 3) + e) * PTS + (l >> 2); }, 32);  // Pt b0 / b1
        }
        for (int e = 0; e < 4; ++e)  // U~ at the C fragment
            ok = ok && distinct_banks([&](int l) { return (2 * (l & 3) + (e & 1)) * US + (l >> 2) + 8 * (e >> 1); }, 32);
        for (int half = 0; half < 2; ++half)  // Q float2 (keys 2t, 2t + 1): 16 lanes x 2 words = 32 banks per half warp
            ok = ok && distinct_banks([&](int w) { const int l = 16 * half + (w >> 1); return (l >> 2) * QS + 2 * (l & 3) + (w & 1); }, 32);
        const bool al = WTS % 4 == 0 && QS % 4 == 0 && KS % 4 == 0 && US % 4 == 0 && PTS % 4 == 0 && SQ % 4 == 0 && SK % 4 == 0 &&
                        SU % 4 == 0 && SPT % 4 == 0 && SG % 4 == 0 && tc::Stage::floats % 4 == 0 &&
                        tc::kSmemBytes == 2 * tc::Stage::floats * sizeof(float) && US >= static_cast<int>(tc::kCtaCols);
        check(ok && al, "tf32 smem strides: fragment loads bank-conflict free, cp.async offsets 16-byte aligned");
    }
    // 4c. the kernel's chunk loop (emulated) vs serial(double): rows 1..300, partial chunks, both CTA tiles, heads 0 and 5
    {
        struct TCase { int rows; float glo, ghi; };
        const TCase tcases[] = {{1, -7.f, -1.f}, {17, -7.f, -1.f}, {40, -7.f, 1.f}, {300, -9.f, -5.f}};
        unsigned sd0 = 500;
        double worst_o = 0, worst_s = 0;
        for (const TCase& cs : tcases) {
            const auto p = make(cs.rows, sd0++, cs.glo, cs.ghi);
            const auto s0 = init_state(p.value_heads, sd0++);
            std::vector<double> sd(s0.begin(), s0.end()), od;
            gdnref::serial<double>(p, 0, p.rows, sd, od);
            std::vector<float> st = s0, ot(od.size(), 0.f);
            for (const int h : {0, 5}) emulate_tc(p, h, &st[static_cast<std::size_t>(h) * D * D], ot, true);
            for (const int h : {0, 5}) {
                std::vector<float> o1, s1(st.begin() + static_cast<std::ptrdiff_t>(h) * D * D, st.begin() + static_cast<std::ptrdiff_t>(h + 1) * D * D);
                std::vector<double> o2, s2(sd.begin() + static_cast<std::ptrdiff_t>(h) * D * D, sd.begin() + static_cast<std::ptrdiff_t>(h + 1) * D * D);
                for (int r = 0; r < p.rows; ++r)
                    for (int e = 0; e < D; ++e) {
                        const std::size_t o = (static_cast<std::size_t>(r) * p.value_heads + static_cast<std::size_t>(h)) * D + static_cast<std::size_t>(e);
                        o1.push_back(ot[o]);
                        o2.push_back(od[o]);
                    }
                worst_o = std::max(worst_o, rel(o1, o2));
                worst_s = std::max(worst_s, rel(s1, s2));
            }
        }
        char msg[200];
        std::snprintf(msg, sizeof msg, "tf32 scan emulation vs serial(double): worst o %.2e state %.2e (bound 3e-3; TF32 operands, rel 2^-11)",
                      worst_o, worst_s);
        check(worst_o < 3e-3 && worst_s < 3e-3, msg);
        // negative control: C fragments fed as A WITHOUT the permutation must be visibly wrong (the test sees the map)
        const auto p = make(40, 777, -7.f, 1.f);
        const auto s0 = init_state(p.value_heads, 778);
        std::vector<double> sd(s0.begin(), s0.end()), od;
        gdnref::serial<double>(p, 0, p.rows, sd, od);
        std::vector<float> st = s0, ot(od.size(), 0.f);
        emulate_tc(p, 0, st.data(), ot, false);
        std::vector<float> o1;
        std::vector<double> o2;
        for (int r = 0; r < p.rows; ++r)
            for (int e = 0; e < D; ++e) {
                o1.push_back(ot[static_cast<std::size_t>(r) * p.value_heads * D + static_cast<std::size_t>(e)]);
                o2.push_back(od[static_cast<std::size_t>(r) * p.value_heads * D + static_cast<std::size_t>(e)]);
            }
        std::snprintf(msg, sizeof msg, "tf32 negative control (no perm): o rel %.2e (must exceed 5e-2)", rel(o1, o2));
        check(rel(o1, o2) > 5e-2, msg);
    }
    std::printf("gdn_chunk_host: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
