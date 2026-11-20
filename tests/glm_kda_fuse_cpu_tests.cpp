// Host invariants of the KDA prefill scan v6 (src/glm/prefill/glm_prefill.hip kda_scan_v6_kernel), emulated on the CPU with
// the declared arithmetic of glm_math.hpp (build with -ffp-contract=off -fno-fast-math, as the device code):
//   1. the in-wave norm butterfly over lane-owned elements l + 32 m (levels 64 / 32 inside the lane, 16 .. 1 by xor partner)
//      gives lds_tree<128>'s sum bitwise on EVERY lane;
//   2. the fused prep mapping (wave w = batch token b0 + w; lane l = channels l + 32 m of q / k / g and the own value columns
//      l, l + 32 of value tile y; conv window from the chunk rows or the conv state) reproduces kda_prep_rows_kernel's
//      scratch (q, k, v, decay = declared_exp(g)) and beta bitwise, for both value tiles of every head;
//   3. the non-finite test taken once on the final state equals v5's test after every token (non-finite is absorbing),
//      with injected inf / NaN / zero decay / overflow inputs; both outcomes must occur.
#include "glm/common/glm_math.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strix::glm;

namespace {
constexpr unsigned kD = 128, kConv = 4, kTB = 8, kV = 64;
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}
std::uint32_t bits(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}
bool same(float a, float b) { return bits(a) == bits(b) || (std::isnan(a) && std::isnan(b)); }

float lds_tree128(const float* e) {  // = lds_tree<128>: s[l] = s[l] + s[l + h], h = 64 .. 1
    float s[kD];
    std::memcpy(s, e, sizeof(s));
    for (unsigned h = kD / 2; h > 0; h >>= 1U)
        for (unsigned l = 0; l < h; ++l) s[l] = s[l] + s[l + h];
    return s[0];
}
void butterfly128(const float* e, float* lane_out) {  // = norm_tree128 on all 32 lanes; e indexed by element
    float t[32], n[32];
    for (unsigned l = 0; l < 32; ++l) t[l] = (e[l] + e[l + 64]) + (e[l + 32] + e[l + 96]);
    for (unsigned hh = 16; hh > 0; hh >>= 1U) {
        for (unsigned l = 0; l < 32; ++l) n[l] = t[l] + t[l ^ hh];
        std::memcpy(t, n, sizeof(t));
    }
    std::memcpy(lane_out, t, sizeof(t));
}

void test_butterfly(std::mt19937& g) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<int> ex(-30, 30);
    bool ok = true;
    for (unsigned trial = 0; trial < 20000 && ok; ++trial) {
        float e[kD], lanes[32];
        for (float& x : e) {
            const float v = nd(g) * std::ldexp(1.0F, ex(g));
            x = std::fmaf(v, v, 0.0F);  // the norms' elements
        }
        if (trial % 97 == 0) e[trial % kD] = 3.0e38F;                // overflow to +inf in the tree
        if (trial % 389 == 0) e[(trial / 7) % kD] = NAN;
        const float ref = lds_tree128(e);
        butterfly128(e, lanes);
        for (float x : lanes) ok = ok && same(x, ref);
    }
    check(ok, "norm butterfly (lane-owned l + 32 m, xor levels 16..1) == lds_tree<128> on every lane, 20000 vectors");
}

struct Prep {
    unsigned H, C;
    std::vector<float> qkv, f, b, conv_state, A_log, dt_bias;
    std::vector<std::uint16_t> conv_w;
};

// kda_prep_rows_kernel, element by element: scratch [C][H][4][128] (q, k, v, g) and beta [C][H].
void prep_reference(const Prep& p, std::vector<float>& scr, std::vector<float>& beta) {
    const unsigned H = p.H, CH = H * kD;
    scr.assign(static_cast<std::size_t>(p.C) * H * 4 * kD, 0.0F);
    beta.assign(static_cast<std::size_t>(p.C) * H, 0.0F);
    for (unsigned r = 0; r < p.C; ++r)
        for (unsigned h = 0; h < H; ++h) {
            float out[3][kD], eq[kD], ek[kD];
            for (unsigned i = 0; i < kD; ++i) {
                for (unsigned part = 0; part < 3; ++part) {
                    const unsigned ch = part * CH + h * kD + i;
                    float x[3];
                    for (unsigned j = 0; j < 3; ++j) {
                        const int m = static_cast<int>(r) - 3 + static_cast<int>(j);
                        x[j] = m >= 0 ? p.qkv[static_cast<std::size_t>(m) * 3 * CH + ch] : p.conv_state[static_cast<std::size_t>(ch) * 3 + static_cast<unsigned>(3 + m)];
                    }
                    const float nw = p.qkv[static_cast<std::size_t>(r) * 3 * CH + ch];
                    const std::uint16_t* cw = p.conv_w.data() + static_cast<std::size_t>(ch) * kConv;
                    float acc = std::fmaf(bf16_to_float(cw[0]), x[0], 0.0F);
                    acc = std::fmaf(bf16_to_float(cw[1]), x[1], acc);
                    acc = std::fmaf(bf16_to_float(cw[2]), x[2], acc);
                    acc = std::fmaf(bf16_to_float(cw[3]), nw, acc);
                    out[part][i] = silu(acc);
                }
                eq[i] = std::fmaf(out[0][i], out[0][i], 0.0F);
                ek[i] = std::fmaf(out[1][i], out[1][i], 0.0F);
            }
            const float nq = declared_sqrt(lds_tree128(eq) + kL2Eps), nk = declared_sqrt(lds_tree128(ek) + kL2Eps);
            const float expA = declared_exp(p.A_log[h]);
            float* sc = scr.data() + (static_cast<std::size_t>(r) * H + h) * 4 * kD;
            for (unsigned i = 0; i < kD; ++i) {
                sc[i] = (out[0][i] / nq) * kInvSqrt128;
                sc[kD + i] = out[1][i] / nk;
                sc[2 * kD + i] = out[2][i];
                sc[3 * kD + i] = kGateLow * sigmoid(expA * (p.f[static_cast<std::size_t>(r) * CH + h * kD + i] + p.dt_bias[h * kD + i]));
            }
            beta[static_cast<std::size_t>(r) * H + h] = sigmoid(p.b[static_cast<std::size_t>(r) * H + h]);
        }
}

// kda_scan_v6_kernel's load + finish for WG (h, y), token r = b0 + w, all 32 lanes of wave w; returns false on a mismatch.
bool fused_matches(const Prep& p, const std::vector<float>& scr, const std::vector<float>& beta) {
    const unsigned H = p.H, CH = H * kD;
    bool ok = true;
    for (unsigned h = 0; h < H; ++h)
        for (unsigned y = 0; y < kD / kV; ++y)
            for (unsigned b0 = 0; b0 < p.C; b0 += kTB)
                for (unsigned w = 0; w < kTB; ++w) {
                    const unsigned r = b0 + w;
                    if (r >= p.C) continue;  // wave-uniform skip (load / finish return)
                    float out[32][10], eq[kD], ek[kD];
                    for (unsigned l = 0; l < 32; ++l)
                        for (unsigned c = 0; c < 10; ++c) {
                            const unsigned ch = c < 8 ? (c / 4) * CH + h * kD + l + 32U * (c % 4) : 2U * CH + h * kD + y * kV + l + 32U * (c - 8);
                            float rx[4];
                            for (unsigned j = 0; j < 3; ++j) {
                                const int m = static_cast<int>(r) - 3 + static_cast<int>(j);
                                rx[j] = m >= 0 ? p.qkv[static_cast<std::size_t>(m) * 3 * CH + ch] : p.conv_state[static_cast<std::size_t>(ch) * 3 + static_cast<unsigned>(3 + m)];
                            }
                            rx[3] = p.qkv[static_cast<std::size_t>(r) * 3 * CH + ch];
                            const std::uint16_t* cw = p.conv_w.data() + static_cast<std::size_t>(ch) * kConv;
                            // the kernel's packed taps: (tap 2j) | (tap 2j + 1) << 16, unpacked by shift / mask
                            const std::uint32_t p0 = static_cast<std::uint32_t>(cw[0]) | (static_cast<std::uint32_t>(cw[1]) << 16U);
                            const std::uint32_t p1 = static_cast<std::uint32_t>(cw[2]) | (static_cast<std::uint32_t>(cw[3]) << 16U);
                            float acc = std::fmaf(std::bit_cast<float>(p0 << 16U), rx[0], 0.0F);
                            acc = std::fmaf(std::bit_cast<float>(p0 & 0xffff0000U), rx[1], acc);
                            acc = std::fmaf(std::bit_cast<float>(p1 << 16U), rx[2], acc);
                            acc = std::fmaf(std::bit_cast<float>(p1 & 0xffff0000U), rx[3], acc);
                            out[l][c] = silu(acc);
                            if (c < 4) eq[l + 32U * c] = std::fmaf(out[l][c], out[l][c], 0.0F);
                            else if (c < 8) ek[l + 32U * (c - 4)] = std::fmaf(out[l][c], out[l][c], 0.0F);
                        }
                    float tq[32], tk[32];
                    butterfly128(eq, tq);
                    butterfly128(ek, tk);
                    const float expA = declared_exp(p.A_log[h]);
                    const float* sc = scr.data() + (static_cast<std::size_t>(r) * H + h) * 4 * kD;
                    for (unsigned l = 0; l < 32; ++l) {
                        const float nq = declared_sqrt(tq[l] + kL2Eps), nk = declared_sqrt(tk[l] + kL2Eps);
                        for (unsigned m = 0; m < 4; ++m) {
                            const unsigned i = l + 32U * m;
                            ok = ok && same((out[l][m] / nq) * kInvSqrt128, sc[i]) && same(out[l][4 + m] / nk, sc[kD + i]);
                            const float rf = p.f[static_cast<std::size_t>(r) * CH + h * kD + i];
                            ok = ok && same(declared_exp(kGateLow * sigmoid(expA * (rf + p.dt_bias[h * kD + i]))), declared_exp(sc[3 * kD + i]));
                        }
                        ok = ok && same(out[l][8], sc[2 * kD + y * kV + l]) && same(out[l][9], sc[2 * kD + y * kV + l + 32U]);
                    }
                    ok = ok && same(sigmoid(p.b[static_cast<std::size_t>(r) * H + h]), beta[static_cast<std::size_t>(r) * H + h]);
                }
    return ok;
}

// kda_scan_v6_kernel's LDS schedule for WG (h, y), simulated slot by slot: rows of the next batch staged in place into the
// other buffer half (stage_load / stage_store element map), conv history hist[3][320] (the conv state first, then raw rows
// 5..7 of each batch, copied by waves 5-7 after barrier B), finish() reading windows / f / b before barrier B and writing
// the prepared row w in place after it (waves in reverse order: each wave touches only its own row and hist row).
// Bitwise checks: (a) every value finish() reads is the input kda_prep_rows_kernel reads for that token (qkv row t - 3 + i
// or the conv state, f, b), (b) for every token the scan reads that token's prepared q, k, exp g, v and beta.
bool staged_matches(const Prep& p, const std::vector<float>& scr, const std::vector<float>& beta) {
    const unsigned H = p.H, C = p.C, CH = H * kD, kRaw = 2 * kD + kV;
    const float stale = std::bit_cast<float>(0x7fc0deadU);
    bool ok = true;
    for (unsigned h = 0; h < H; ++h)
        for (unsigned y = 0; y < kD / kV; ++y) {
            std::vector<float> sq(2 * kTB * kD, stale), sk(sq), sd(sq), sv(2 * kTB * kV, stale), sb(2 * kTB, stale), hist(3 * kRaw, stale);
            const unsigned vch = 2 * CH + h * kD + y * kV;
            auto slot = [](unsigned buf, unsigned row) -> std::size_t { return static_cast<std::size_t>(buf) * kTB + row; };
            auto hcol = [](unsigned c, unsigned l) -> unsigned { return c < 8 ? (c / 4) * kD + l + 32 * (c % 4) : 2 * kD + l + 32 * (c - 8); };
            auto stage = [&](unsigned buf, unsigned b1) {  // tokens j = 0..7 of the running batch stage row b1 + j
                for (unsigned j = 0; j < kTB && b1 + j < C; ++j)
                    for (unsigned tid = 0; tid < 256; ++tid) {
                        const unsigned w = tid / 32, l = tid % 32, r = b1 + j;
                        const std::size_t row = static_cast<std::size_t>(r) * 3 * CH;
                        const float ea = p.qkv[row + (tid / kD) * CH + h * kD + tid % kD];
                        if (w < 4) {
                            sq[slot(buf, j) * kD + tid] = ea;
                            sd[slot(buf, j) * kD + tid] = p.f[static_cast<std::size_t>(r) * CH + h * kD + tid];
                        } else {
                            sk[slot(buf, j) * kD + tid - kD] = ea;
                            if (w < 6) sv[slot(buf, j) * kV + tid - kD] = p.qkv[row + vch + (tid - kD)];
                            else if (w == 6 && l == 0) sb[slot(buf, j)] = p.b[static_cast<std::size_t>(r) * H + h];
                        }
                    }
            };
            auto raw = [&](unsigned buf, int row, unsigned c, unsigned l) -> float {
                if (row < 0) return hist[static_cast<std::size_t>(3 + row) * kRaw + hcol(c, l)];
                const std::size_t at = slot(buf, static_cast<unsigned>(row));
                if (c < 4) return sq[at * kD + l + 32 * c];
                if (c < 8) return sk[at * kD + l + 32 * (c - 4)];
                return sv[at * kV + l + 32 * (c - 8)];
            };
            auto finish = [&](unsigned buf, unsigned b1) {
                for (unsigned w = 0; w < kTB; ++w) {  // before barrier B: reads only
                    const unsigned t = b1 + w;
                    if (t >= C) continue;
                    for (unsigned l = 0; l < 32; ++l) {
                        for (unsigned c = 0; c < 10; ++c) {
                            const unsigned ch = c < 8 ? (c / 4) * CH + h * kD + l + 32 * (c % 4) : vch + l + 32 * (c - 8);
                            for (unsigned i = 0; i < 4; ++i) {
                                const int m = static_cast<int>(t) - 3 + static_cast<int>(i);
                                const float want = m >= 0 ? p.qkv[static_cast<std::size_t>(m) * 3 * CH + ch]
                                                          : p.conv_state[static_cast<std::size_t>(ch) * 3 + static_cast<unsigned>(3 + m)];
                                ok = ok && bits(raw(buf, static_cast<int>(w + i) - 3, c, l)) == bits(want);
                            }
                        }
                        for (unsigned m = 0; m < 4; ++m)
                            ok = ok && bits(sd[slot(buf, w) * kD + l + 32 * m]) == bits(p.f[static_cast<std::size_t>(t) * CH + h * kD + l + 32 * m]);
                    }
                    ok = ok && bits(sb[slot(buf, w)]) == bits(p.b[static_cast<std::size_t>(t) * H + h]);
                }
                for (unsigned wr = 0; wr < kTB; ++wr) {  // after barrier B, waves in reverse order
                    const unsigned w = kTB - 1 - wr, t = b1 + w;
                    if (t >= C) continue;
                    if (w >= kTB - 3)
                        for (unsigned l = 0; l < 32; ++l)
                            for (unsigned c = 0; c < 10; ++c) hist[(w - (kTB - 3)) * kRaw + hcol(c, l)] = raw(buf, static_cast<int>(w), c, l);
                    const float* sc = scr.data() + (static_cast<std::size_t>(t) * H + h) * 4 * kD;
                    for (unsigned i = 0; i < kD; ++i) {
                        sq[slot(buf, w) * kD + i] = sc[i];
                        sk[slot(buf, w) * kD + i] = sc[kD + i];
                        sd[slot(buf, w) * kD + i] = declared_exp(sc[3 * kD + i]);
                    }
                    for (unsigned i = 0; i < kV; ++i) sv[slot(buf, w) * kV + i] = sc[2 * kD + y * kV + i];
                    sb[slot(buf, w)] = beta[static_cast<std::size_t>(t) * H + h];
                }
            };
            for (unsigned e = 0; e < 3 * kRaw; ++e) {  // first batch: hist = the conv state
                const unsigned i = e / kRaw, c = e % kRaw;
                const unsigned ch = c < 2 * kD ? (c / kD) * CH + h * kD + c % kD : vch + (c - 2 * kD);
                hist[e] = p.conv_state[static_cast<std::size_t>(ch) * 3 + i];
            }
            stage(0, 0);
            unsigned cur = 0;
            for (unsigned b0 = 0; b0 < C; b0 += kTB, cur ^= 1U) {
                finish(cur, b0);
                const unsigned nb = C - b0 < kTB ? C - b0 : kTB;
                for (unsigned j = 0; j < nb; ++j) {  // what the scan reads for token b0 + j
                    const std::size_t rh = static_cast<std::size_t>(b0 + j) * H + h;
                    const float* sc = scr.data() + rh * 4 * kD;
                    for (unsigned i = 0; i < kD; ++i)
                        ok = ok && bits(sq[slot(cur, j) * kD + i]) == bits(sc[i]) && bits(sk[slot(cur, j) * kD + i]) == bits(sc[kD + i]) &&
                             bits(sd[slot(cur, j) * kD + i]) == bits(declared_exp(sc[3 * kD + i]));
                    for (unsigned i = 0; i < kV; ++i) ok = ok && bits(sv[slot(cur, j) * kV + i]) == bits(sc[2 * kD + y * kV + i]);
                    ok = ok && bits(sb[slot(cur, j)]) == bits(beta[rh]);
                }
                if (b0 + kTB < C) stage(cur ^ 1U, b0 + kTB);  // staged during the tokens above, into the other half
            }
        }
    return ok;
}

void test_prep(std::mt19937& g) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<unsigned> u16(0, 0xffffU);
    for (unsigned C : {1U, 2U, 3U, 4U, 9U, 13U, 17U, 24U, 41U}) {
        Prep p;
        p.H = 2;
        p.C = C;
        const unsigned CH = p.H * kD;
        auto fill = [&](std::vector<float>& v, std::size_t n, float scale) {
            v.resize(n);
            for (float& x : v) x = nd(g) * scale;
        };
        fill(p.qkv, static_cast<std::size_t>(C) * 3 * CH, 2.0F);
        fill(p.f, static_cast<std::size_t>(C) * CH, 1.5F);
        fill(p.b, static_cast<std::size_t>(C) * p.H, 2.0F);
        fill(p.conv_state, static_cast<std::size_t>(3) * CH * 3, 2.0F);
        fill(p.A_log, p.H, 1.0F);
        fill(p.dt_bias, CH, 1.0F);
        p.conv_w.resize(static_cast<std::size_t>(3) * CH * kConv);
        for (auto& w : p.conv_w) {  // finite BF16 weights of moderate size: sign, exponent 120..130, random mantissa
            const unsigned sgn = u16(g) & 0x8000U, e = 120U + u16(g) % 11U, man = u16(g) & 0x7fU;
            w = static_cast<std::uint16_t>(sgn | (e << 7U) | man);
        }
        std::vector<float> scr, beta;
        prep_reference(p, scr, beta);
        char tag[160];
        std::snprintf(tag, sizeof(tag), "fused prep mapping == kda_prep_rows_kernel (q, k, v, exp g, beta), H=2, C=%u, both value tiles", C);
        check(fused_matches(p, scr, beta), tag);
        std::snprintf(tag, sizeof(tag), "v6 LDS stage / conv history schedule reads the prep inputs and serves the prepared rows, H=2, C=%u", C);
        check(staged_matches(p, scr, beta), tag);
    }
}

// One value column of the scan, 128 keys, v5 expressions; returns {per-token flag, final-state flag}.
void scan_flags(const std::vector<float>& d, const std::vector<float>& k, const std::vector<float>& q, const std::vector<float>& v,
                const std::vector<float>& bt, unsigned T, bool& per_token, bool& final_state) {
    float st[kD];
    for (unsigned i = 0; i < kD; ++i) st[i] = 0.25F * static_cast<float>(i % 7) - 0.5F;
    per_token = false;
    for (unsigned t = 0; t < T; ++t) {
        float e[kD];
        for (unsigned i = 0; i < kD; ++i) st[i] = st[i] * d[t * kD + i];
        for (unsigned i = 0; i < kD; ++i) e[i] = 0.0F + st[i] * k[t * kD + i];
        const float dlt = (v[t] - lds_tree128(e)) * bt[t];
        for (unsigned i = 0; i < kD; ++i) {
            st[i] = std::fmaf(k[t * kD + i], dlt, st[i]);
            per_token = per_token || !is_finite(st[i]);
        }
        for (unsigned i = 0; i < kD; ++i) e[i] = 0.0F + st[i] * q[t * kD + i];
        (void)lds_tree128(e);
    }
    final_state = false;
    for (float x : st) final_state = final_state || !is_finite(x);
}

void test_absorbing(std::mt19937& g) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_real_distribution<float> ud(0.0F, 1.0F);
    const float specials[] = {INFINITY, -INFINITY, NAN, 0.0F, -0.0F, 3.0e38F, -3.0e38F, 1.0e-45F, 1.0e20F};
    constexpr unsigned T = 24;
    unsigned agree = 0, flagged = 0, clean = 0, trials = 20000;
    for (unsigned trial = 0; trial < trials; ++trial) {
        std::vector<float> d(T * kD), k(T * kD), q(T * kD), v(T), bt(T);
        for (unsigned i = 0; i < T * kD; ++i) {
            d[i] = declared_exp(kGateLow * ud(g));
            k[i] = nd(g) * 0.1F;
            q[i] = nd(g) * 0.1F;
        }
        for (unsigned t = 0; t < T; ++t) {
            v[t] = nd(g);
            bt[t] = ud(g);
        }
        const unsigned n_inj = trial % 4;  // 0 .. 3 injected specials anywhere (some cancel or overflow, some not)
        for (unsigned j = 0; j < n_inj; ++j) {
            const float s = specials[g() % (sizeof(specials) / sizeof(specials[0]))];
            const unsigned t = static_cast<unsigned>(g() % T), i = static_cast<unsigned>(g() % kD);
            switch (g() % 5) {
                case 0: d[t * kD + i] = s; break;
                case 1: k[t * kD + i] = s; break;
                case 2: q[t * kD + i] = s; break;
                case 3: v[t] = s; break;
                default: bt[t] = s; break;
            }
        }
        bool a = false, b = false;
        scan_flags(d, k, q, v, bt, T, a, b);
        agree += a == b ? 1U : 0U;
        flagged += a ? 1U : 0U;
        clean += a ? 0U : 1U;
    }
    char tag[200];
    std::snprintf(tag, sizeof(tag), "non-finite absorbing: final-state flag == per-token flag in %u/%u scans (%u flagged, %u clean)", agree,
                  trials, flagged, clean);
    check(agree == trials && flagged > 1000 && clean > 1000, tag);
}
}  // namespace

int main() {
    std::mt19937 g(20261117U);
    test_butterfly(g);
    test_prep(g);
    test_absorbing(g);
    std::printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
