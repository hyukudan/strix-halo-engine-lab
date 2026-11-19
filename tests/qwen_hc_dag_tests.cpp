// CPU oracle of the exact prefill HC DAG (qwen_hc.cu [P] projection and [O] output) against a lane-by-lane model of the CUDA kernel
// structure in include/strix/cuda/qwen_hc_exact_shape.hpp: the transpose-reduce fold of the low-output warps, the injection block
// (8 warps x 2 rows, plain xor butterfly), the output staging at pitch kOutLowPitch and the rolled two-half epilogue.
// Bitwise comparisons only; every (row, output) must be produced exactly once. The sigmoid is any fixed function here (the same on
// both sides): the test is about order, ownership and layout, not about amd_expf.
#include "strix/cuda/qwen_hc_exact_shape.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace hs = strix::cuda::qwen_hc::exact_shape;
using u32 = std::uint32_t;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}
static u32 bits(float f) { u32 u; std::memcpy(&u, &f, 4); return u; }
static float sig(float x) { return 1.0F / (1.0F + std::exp(-x)); }

constexpr u32 W = hs::kWide, RK = hs::kRank, S = hs::kStreams;
static u32 feature(u32 tile, u32 sub, u32 lane, u32 j) { return tile * 640U + sub * 128U + 4U * lane + j; }

// ---- oracle: one (row, output) of the projection, HIP DAG (leaf chains, xor tree 16..1, serial tile sum) -------------------------
static float oracle_total(const float* x, const float* c) {
    float total = 0.0F;
    for (u32 t = 0; t < 16; ++t) {
        float leaf[32];
        for (u32 l = 0; l < 32; ++l) {
            float a = 0.0F;
            for (u32 s = 0; s < 5; ++s)
                for (u32 j = 0; j < 4; ++j) a = std::fma(x[feature(t, s, l, j)], c[feature(t, s, l, j)], a);
            leaf[l] = a;
        }
        for (u32 d = 16; d > 0; d >>= 1) {
            float n[32];
            for (u32 l = 0; l < 32; ++l) n[l] = leaf[l] + leaf[l ^ d];
            std::memcpy(leaf, n, sizeof(leaf));
        }
        total = total + leaf[0];
    }
    return total;
}

// ---- model of a low-output warp: 32 lanes x kProjItems chains, transpose-reduce fold (fold_level), lane -> (row, output) -----------
static void model_low_warp(const std::vector<float>& x, u32 rows, u32 row0, const std::vector<float>& coef, u32 first,
                           std::vector<float>& out_total, std::vector<int>& owned) {
    constexpr u32 R = hs::kProjRows, O = hs::kProjOutputs, N = hs::kProjItems;
    std::vector<float> acc(32 * N, 0.0F), total(32 * 4, 0.0F);
    for (u32 t = 0; t < 16; ++t) {
        for (u32 s = 0; s < 5; ++s)
            for (u32 lane = 0; lane < 32; ++lane)
                for (u32 r = 0; r < R; ++r)
                    for (u32 j = 0; j < 4; ++j)
                        for (u32 p = 0; p < O; ++p) {
                            const u32 f = feature(t, s, lane, j);
                            const float xv = row0 + r < rows ? x[(row0 + r) * W + f] : 0.0F;
                            float& a = acc[lane * N + r * O + p];
                            a = std::fma(xv, coef[(first + p) * W + f], a);
                        }
        for (u32 level = 0; level < 5; ++level) {  // fold_level<level>
            const u32 distance = 16U >> level, half = 64U >> level;
            std::vector<float> next = acc;
            for (u32 lane = 0; lane < 32; ++lane) {
                const bool upper = (lane & distance) != 0U;
                const u32 partner = lane ^ distance;
                const bool pupper = (partner & distance) != 0U;
                for (u32 i = 0; i < half; ++i) {
                    const float keep = upper ? acc[lane * N + i + half] : acc[lane * N + i];
                    const float send = pupper ? acc[partner * N + i] : acc[partner * N + i + half];  // partner's `send`
                    next[lane * N + i] = keep + send;
                }
            }
            acc = next;
        }
        for (u32 lane = 0; lane < 32; ++lane) {
            for (u32 i = 0; i < 4; ++i) total[lane * 4 + i] = total[lane * 4 + i] + acc[lane * N + i];
            for (u32 i = 0; i < N; ++i) acc[lane * N + i] = 0.0F;
        }
    }
    for (u32 b = 0; b < 32; ++b) {
        const u32 r = 8U * ((b >> 4) & 1U) + 4U * ((b >> 3) & 1U) + 2U * ((b >> 2) & 1U) + ((b >> 1) & 1U);
        if (row0 + r >= rows) continue;
        for (u32 i = 0; i < 4; ++i) {
            const u32 o = first + 4U * (b & 1U) + i;
            out_total[(row0 + r) * RK + o] = total[b * 4 + i];
            ++owned[(row0 + r) * RK + o];
        }
    }
}

// ---- model of the injection block: warp w = rows 2w, 2w + 1 x 4 outputs, xor butterfly per item, lane i < 8 stores item i --------
static void model_inject_block(const std::vector<float>& x, u32 rows, u32 row0, const std::vector<float>& inj,
                               std::vector<float>& out_total, std::vector<int>& owned) {
    constexpr u32 IR = hs::kProjInjRows, IN = hs::kProjInjItems;
    for (u32 warp = 0; warp < hs::kProjWarps; ++warp) {
        std::vector<float> acc(32 * IN, 0.0F), tot(32 * IN, 0.0F);
        for (u32 t = 0; t < 16; ++t) {
            for (u32 s = 0; s < 5; ++s)
                for (u32 lane = 0; lane < 32; ++lane)
                    for (u32 rr = 0; rr < IR; ++rr)
                        for (u32 j = 0; j < 4; ++j)
                            for (u32 p = 0; p < S; ++p) {
                                const u32 f = feature(t, s, lane, j), row = row0 + IR * warp + rr;
                                const float xv = row < rows ? x[row * W + f] : 0.0F;
                                float& a = acc[lane * IN + rr * S + p];
                                a = std::fma(xv, inj[p * W + f], a);
                            }
            for (u32 i = 0; i < IN; ++i) {
                for (u32 d = 16; d > 0; d >>= 1) {
                    float n[32];
                    for (u32 lane = 0; lane < 32; ++lane) n[lane] = acc[lane * IN + i] + acc[(lane ^ d) * IN + i];
                    for (u32 lane = 0; lane < 32; ++lane) acc[lane * IN + i] = n[lane];
                }
                for (u32 lane = 0; lane < 32; ++lane) { tot[lane * IN + i] = tot[lane * IN + i] + acc[lane * IN + i]; acc[lane * IN + i] = 0.0F; }
            }
        }
        for (u32 lane = 0; lane < 32; ++lane) {
            float v = tot[lane * IN + 0];
            for (u32 i = 1; i < IN; ++i) v = lane == i ? tot[lane * IN + i] : v;
            const u32 row = row0 + IR * warp + lane / S;
            if (lane >= IN || row >= rows) continue;
            out_total[row * S + lane % S] = v;
            ++owned[row * S + lane % S];
        }
    }
}

// ---- chunk pipeline of hc_projection_kernel (hs::proj_pipeline, the same template the kernel runs) ------------------------------
// Every chunk's body sees its own coefficients and its own x tile (stored before a barrier), no request overwrites registers that were
// not consumed yet, no store overwrites a buffer without a barrier after its last read, and every decode of chunk k > 0 runs after a
// whole body that follows the request of chunk k (the load-to-use distance the schedule exists for).
static void test_proj_pipeline() {
    constexpr int none = -1;
    int raw = none, next = none, coef = none, epoch = 0, bodies = 0, expect_body = 0;
    int buf[2] = {none, none}, stored_epoch[2] = {-1, -1}, read_epoch[2] = {-1, -1};
    int requests = 0, decodes = 0, stores = 0, barriers = 0, bodies_at_request = 0;
    bool ok = true;
    hs::proj_pipeline(
        [&](u32 k) {
            ok &= (raw == none || coef == raw) && (next == none || buf[next & 1] == next);  // previous words decoded, tile stored
            raw = next = static_cast<int>(k);
            bodies_at_request = bodies;
            ++requests;
        },
        [&]() {
            ok &= raw != none && (raw == 0 || bodies > bodies_at_request);
            coef = raw;
            ++decodes;
        },
        [&](u32 k) {
            const int b = static_cast<int>(k & 1U);
            ok &= next == static_cast<int>(k) && read_epoch[b] < epoch;
            buf[b] = next;
            stored_epoch[b] = epoch;
            ++stores;
        },
        [&](u32 k) {
            const int b = static_cast<int>(k & 1U);
            ok &= static_cast<int>(k) == expect_body && coef == static_cast<int>(k) && buf[b] == static_cast<int>(k) && stored_epoch[b] < epoch;
            read_epoch[b] = epoch;
            ++expect_body;
            ++bodies;
        },
        [&]() { ++epoch; ++barriers; });
    const int n = static_cast<int>(hs::kProjChunks);
    ok &= expect_body == n && requests == n && decodes == n && stores == n && barriers == n + 1;
    check(ok, "projection pipeline: chunk k reads its own coefficients and x tile; decode of k > 0 one body after its request");
}

int main() {
    test_proj_pipeline();
    std::mt19937 rng(0x48c2u);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<int> code(-127, 127);
    const u32 rows = 21;  // two projection row blocks (16 + 5: partial block)
    std::vector<float> x(static_cast<std::size_t>(rows) * W), coef(static_cast<std::size_t>(RK) * W), inj(static_cast<std::size_t>(S) * W);
    for (auto& v : x) v = nd(rng);
    for (u32 o = 0; o < RK; ++o)
        for (u32 b = 0; b < W / 32; ++b) {
            const float scale = std::ldexp(1.0F + static_cast<float>(code(rng) & 63) / 64.0F, -9);
            for (u32 i = 0; i < 32; ++i) coef[o * W + b * 32 + i] = scale * static_cast<float>(code(rng));  // exact product
        }
    for (auto& v : inj) v = 0.01F * nd(rng);

    // [P] low outputs: two warps (16 outputs) of every row block; injection block of every row block.
    {
        std::vector<float> got(static_cast<std::size_t>(rows) * RK, 0.0F), gi(static_cast<std::size_t>(rows) * S, 0.0F);
        std::vector<int> own(got.size(), 0), owni(gi.size(), 0);
        const u32 firsts[2] = {0U, 4U * hs::kProjWarps * hs::kProjOutputs + 7U * hs::kProjOutputs};  // block 0 warp 0, block 4 warp 7
        for (u32 row0 = 0; row0 < rows; row0 += hs::kProjRows) {
            for (u32 first : firsts) model_low_warp(x, rows, row0, coef, first, got, own);
            model_inject_block(x, rows, row0, inj, gi, owni);
        }
        bool same = true, once = true;
        for (u32 r = 0; r < rows; ++r)
            for (u32 first : firsts)
                for (u32 o = first; o < first + hs::kProjOutputs; ++o) {
                    same &= bits(got[r * RK + o]) == bits(oracle_total(&x[r * W], &coef[o * W]));
                    once &= own[r * RK + o] == 1;
                }
        check(same && once, "projection low warps: transpose-reduce fold == oracle tree, each (row, output) once");
        bool isame = true, ionce = true;
        for (u32 r = 0; r < rows; ++r)
            for (u32 o = 0; o < S; ++o) {
                isame &= bits(gi[r * S + o]) == bits(oracle_total(&x[r * W], &inj[o * W]));
                ionce &= owni[r * S + o] == 1;
            }
        check(isame && ionce, "projection injection block (8 warps x 2 rows, xor butterfly) == oracle tree, each (row, output) once");
    }

    // [O] output: staging map, bank degree, chains, rolled epilogue ownership and mix.
    {
        constexpr u32 P = hs::kOutLowPitch, T = hs::kOutThreads, OR = hs::kOutRows;
        std::vector<int> written(RK * OR, 0);
        u32 worst = 0, worst32 = 0;
        for (u32 i = 0; i < hs::kOutStagePerThread; ++i)
            for (u32 w = 0; w < T / 32; ++w) {
                u32 bank[32] = {}, bank32[32] = {};
                for (u32 lane = 0; lane < 32; ++lane) {
                    const u32 idx = w * 32 + lane + T * i, r = idx / RK, k = idx % RK;
                    ++written[k * OR + r];
                    worst = std::max(worst, ++bank[(k * P + r) % 32]);
                    worst32 = std::max(worst32, ++bank32[(k * OR + r) % 32]);
                }
            }
        bool once = true;
        for (int c : written) once &= c == 1;
        std::printf("       staging write bank degree: pitch %u -> %u-way (pitch %u -> %u-way)\n", P, worst, OR, worst32);
        check(once && worst <= 4U && (P * 4U) % 16U == 0U, "output staging: each lows[k][r] written once, <= 4-way, float4-aligned rows");

        bool slots_ok = true;
        std::vector<int> slot_use(RK * P / 4, 0);
        for (u32 r = 0; r < hs::kOutHalfRows; ++r)
            for (u32 t = 0; t < T; ++t) { const u32 sidx = r * T + t; slots_ok &= sidx < slot_use.size() && ++slot_use[sidx] == 1; }
        check(slots_ok, "output epilogue: float4 slots thread-private, inside the lows buffer");

        const u32 orows = 37;  // two row blocks (32 + 5)
        std::vector<float> lowv(static_cast<std::size_t>(orows) * RK), up(static_cast<std::size_t>(S) * 8 * RK), nrm(static_cast<std::size_t>(orows) * S * 8);
        for (auto& v : lowv) v = 0.1F * nd(rng);
        for (auto& v : up) v = std::ldexp(static_cast<float>(code(rng)), -10);
        for (auto& v : nrm) v = nd(rng);
        constexpr u32 F = 8;  // features modelled (the mapping is per feature)
        auto cf = [&](u32 s, u32 f, u32 k) { return up[(s * F + f) * RK + k]; };
        bool same = true, own_ok = true;
        std::vector<int> owned(static_cast<std::size_t>(orows) * F, 0);
        for (u32 row0 = 0; row0 < orows; row0 += OR)
            for (u32 f = 0; f < F; ++f) {
                std::vector<float> lows(RK * P, 0.0F);
                for (u32 r = 0; r < OR; ++r)
                    for (u32 k = 0; k < RK; ++k) lows[k * P + r] = row0 + r < orows ? lowv[(row0 + r) * RK + k] : 0.0F;
                float g[4][OR] = {};
                for (u32 k = 0; k < RK; ++k)
                    for (u32 r = 0; r < OR; ++r)
                        for (u32 s = 0; s < S; ++s) g[s][r] = std::fma(lows[k * P + r], cf(s, f, k), g[s][r]);
                for (u32 h = 0; h < 2; ++h) {
                    if (row0 + hs::kOutHalfRows * h >= orows) break;
                    float slot[hs::kOutHalfRows][4];
                    for (u32 r = 0; r < hs::kOutHalfRows; ++r)
                        for (u32 s = 0; s < S; ++s) slot[r][s] = g[s][hs::kOutHalfRows * h + r];
                    for (u32 r0 = 0; r0 < hs::kOutHalfRows; r0 += hs::kOutMixRows)
                        for (u32 i = 0; i < hs::kOutMixRows; ++i) {
                            const u32 row = row0 + hs::kOutHalfRows * h + r0 + i;
                            float mixed = 0.0F;
                            for (u32 s = 0; s < S; ++s) mixed = std::fma(row < orows ? nrm[(row * S + s) * F + f] : 0.0F, sig(slot[r0 + i][s]), mixed);
                            if (row >= orows) continue;
                            ++owned[row * F + f];
                            float want = 0.0F;  // oracle: wide2_hyper_output DAG
                            for (u32 s = 0; s < S; ++s) {
                                float gg = 0.0F;
                                for (u32 k = 0; k < RK; ++k) gg = std::fma(lowv[row * RK + k], cf(s, f, k), gg);
                                want = std::fma(nrm[(row * S + s) * F + f], sig(gg), want);
                            }
                            same &= bits(mixed * 0.25F) == bits(want * 0.25F);
                        }
                }
            }
        for (int c : owned) own_ok &= c == 1;
        check(same && own_ok, "output: chains via pitch-36 lows + rolled 2-row epilogue == oracle, each (row, feature) once");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
