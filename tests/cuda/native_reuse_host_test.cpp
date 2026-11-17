// Host check of src/cuda/core/native_reuse.hpp, the shared contract of the native SIMT expert route
// (cuda-native-iq-fp32@1), on the six native GSQ storage types with the shared decoders (quant::decode_row_group):
//   codes     : each decoded group equals s[j / 16] * float(c[j]) bitwise (32 int8 codes, 2 FP32 sub-scales);
//   fold      : the register transpose-fold gives every item the xor-butterfly root bits at its owner lane;
//   seed      : a -0 leaf is the only case where the two lane seeds differ;
//   down/proj : decode once per tile + register blocking + fold (and the code view) == the per-chunk route, bit for bit.
// No GPU. Release build, ctest; failures print and return 1 (no dialogs).
#include "../../src/cuda/core/native_reuse.hpp"
#include "../../src/cuda/weight/ggml_sources.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace nr = strix::cuda::core::native_reuse;
namespace qt = strix::cuda::quant;
using u8 = std::uint8_t;
using h16 = std::uint16_t;
using u32 = std::uint32_t;

static int g_fail = 0;
static u32 bits(float f) { u32 b; std::memcpy(&b, &f, 4); return b; }
static bool same(float a, float b) { return (std::isnan(a) && std::isnan(b)) || bits(a) == bits(b); }
static void check(bool ok, const char* what, u32 type, u32 i) {
    if (!ok && g_fail++ < 10) std::printf("FAIL %s type %u index %u\n", what, type, i);
}
static constexpr u32 kTypes[] = {qt::t_iq3_xxs, qt::t_iq4_nl, qt::t_iq3_s, qt::t_iq2_s, qt::t_iq4_xs, qt::t_q2_0};

// Rows of random bytes; every block's FP16 d is a random finite value (both signs, subnormals included), and the first
// block of row 0 gets -0.
static std::vector<u8> random_rows(u32 type, u32 rows, u32 width, std::mt19937& rng) {
    const auto geo = qt::geometry(type);
    std::vector<u8> bytes(std::size_t(rows) * width / geo.block_elements * geo.block_bytes);
    for (auto& b : bytes) b = u8(rng());
    for (std::size_t off = 0; off < bytes.size(); off += geo.block_bytes) {
        h16 d = h16(rng());
        if ((d & 0x7C00U) == 0x7C00U) d &= 0xBBFFU;  // no inf / NaN
        if ((d & 0x7C00U) > 0x4800U) d = h16((d & 0x83FFU) | 0x3000U);  // keep products in range
        if (off == 0) d = 0x8000U;
        std::memcpy(bytes.data() + off, &d, 2);
    }
    return bytes;
}

// Code view of group g: same scale expressions and grid lookups as quant::decode_group, sign folded into the code.
struct Codes { std::int8_t c[32]; float s[2]; };
static Codes factor(u32 t, const u8* row, u32 g) {
    namespace qd = qt::detail;
    const auto geo = qt::geometry(t);
    const u32 per = geo.block_elements / 32U, sub = g % per;
    const u8* x = row + std::size_t(g / per) * geo.block_bytes;
    Codes v{};
    const float d = qd::fp16(qd::u16(x));
    auto signed_byte = [](u32 grid_value, bool negative) { return std::int8_t(negative ? -int(grid_value) : int(grid_value)); };
    switch (t) {
    case qt::t_iq4_nl:
        v.s[0] = v.s[1] = d;
        for (int j = 0; j < 16; ++j) { v.c[j] = std::int8_t(qd::kvalue(x[2 + j] & 0xf)); v.c[j + 16] = std::int8_t(qd::kvalue(x[2 + j] >> 4)); }
        break;
    case qt::t_q2_0:
        v.s[0] = v.s[1] = d;
        for (int k = 0; k < 32; ++k) { const int j = int(32 * sub) + k; v.c[k] = std::int8_t(int((x[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1); }
        break;
    case qt::t_iq4_xs: {
        const h16 scales_h = qd::u16(x + 2); const u8* qs = x + 8 + 16 * sub;
        const int ls = ((x[4 + sub / 2] >> 4 * (sub % 2)) & 0xf) | (((scales_h >> 2 * sub) & 3) << 4);
        v.s[0] = v.s[1] = d * (ls - 32);
        for (int j = 0; j < 16; ++j) { v.c[j] = std::int8_t(qd::kvalue(qs[j] & 0xf)); v.c[j + 16] = std::int8_t(qd::kvalue(qs[j] >> 4)); }
        break;
    }
    case qt::t_iq2_s: {
        const u8* qs = x + 2 + 4 * sub; const u8* qh = x + 66; const u8* scales = x + 74; const u8* signs = x + 34 + 4 * sub;
        v.s[0] = d * (0.5f + (scales[sub] & 0xf)) * 0.25f;
        v.s[1] = d * (0.5f + (scales[sub] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const auto grid = qd::iq2s_grid(qs[l] | (qh[sub] << (8 - 2 * l) & 0x300));
            for (int j = 0; j < 8; ++j) v.c[8 * l + j] = signed_byte(qd::grid_byte(grid, j), signs[l] & (1U << j));
        }
        break;
    }
    case qt::t_iq3_xxs: {
        const u8* qs = x + 2 + 8 * sub; const u32 aux32 = qd::u32(x + 66 + 4 * sub);
        v.s[0] = v.s[1] = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const u8 signs = qd::ksigns((aux32 >> 7 * l) & 127);
            const auto g1 = qd::iq3xxs_grid(qs[2 * l]), g2 = qd::iq3xxs_grid(qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                v.c[8 * l + j] = signed_byte(qd::grid_byte(g1, j), signs & (1U << j));
                v.c[8 * l + j + 4] = signed_byte(qd::grid_byte(g2, j), signs & (1U << (j + 4)));
            }
        }
        break;
    }
    case qt::t_iq3_s: {
        const u8* qs = x + 2 + 8 * sub; const u8 qh = x[66 + sub]; const u8* signs = x + 74 + 4 * sub; const u8 sb = x[106 + sub / 2];
        v.s[0] = v.s[1] = d * (1 + 2 * ((sub & 1) ? (sb >> 4) : (sb & 0xf)));
        for (int l = 0; l < 4; ++l) {
            const auto g1 = qd::iq3s_grid(qs[2 * l] | ((qh << (8 - 2 * l)) & 256)), g2 = qd::iq3s_grid(qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256));
            for (int j = 0; j < 4; ++j) {
                v.c[8 * l + j] = signed_byte(qd::grid_byte(g1, j), signs[l] & (1U << j));
                v.c[8 * l + j + 4] = signed_byte(qd::grid_byte(g2, j), signs[l] & (1U << (j + 4)));
            }
        }
        break;
    }
    default: break;
    }
    return v;
}
static void expand(const Codes& v, float* w) { for (int j = 0; j < 32; ++j) w[j] = v.s[j >> 4] * float(v.c[j]); }

static void test_codes(std::mt19937& rng) {
    for (u32 t : kTypes) {
        const u32 width = 2560U, rows = 64U;
        const auto bytes = random_rows(t, rows, width, rng);
        const auto row_bytes = std::size_t(width) / qt::geometry(t).block_elements * qt::geometry(t).block_bytes;
        for (u32 r = 0; r < rows; ++r)
            for (u32 g = 0; g < width / 32U; ++g) {
                float ref[32], cv[32];
                qt::decode_row_group(t, bytes.data() + r * row_bytes, g, ref);
                expand(factor(t, bytes.data() + r * row_bytes, g), cv);
                for (int j = 0; j < 32; ++j) check(bits(ref[j]) == bits(cv[j]), "code view", t, r * width + 32 * g + j);
            }
    }
}

static float special(std::mt19937& rng) {
    static const float k[] = {0.0F, -0.0F, INFINITY, -INFINITY, NAN, 1e-45F, -1e-45F, 1.0F, -1.0F};
    const u32 pick = rng() % 16U;
    if (pick < 9U) return k[pick];
    return std::ldexp(float(int(rng() % 2001U) - 1000), int(rng() % 40U) - 20);
}

template <unsigned N> static void test_fold(std::mt19937& rng, u32 trials) {
    using F = nr::TransposeFold<N>;
    for (u32 trial = 0; trial < trials; ++trial) {
        float acc[32][N];
        for (auto& lane : acc) for (auto& x : lane) x = (trial & 1U) ? special(rng) : std::ldexp(float(int(rng() % 2001U) - 1000), int(rng() % 40U) - 20);
        float root[N];
        for (unsigned i = 0; i < N; ++i) { float column[32]; for (unsigned l = 0; l < 32U; ++l) column[l] = acc[l][i]; root[i] = nr::butterfly_root(column); }
        nr::transpose_fold_simulate<N>(acc);
        for (unsigned i = 0; i < N; ++i) check(same(root[i], acc[F::owner_lane(i)][F::owner_slot(i)]), "fold", N, i);
    }
}

// Down: ffn 640 (20 groups, LaneSeed::direct), FP32 inputs. One tile of `rows` assignments of one expert.
static void test_down(u32 t, std::mt19937& rng) {
    constexpr u32 ffn = 640U, groups = ffn / 32U, outputs = 32U, rows = 37U, chunk = 8U;
    const auto w = random_rows(t, outputs, ffn, rng);
    const auto rb = std::size_t(ffn) / qt::geometry(t).block_elements * qt::geometry(t).block_bytes;
    std::vector<float> h(std::size_t(rows) * ffn);
    for (auto& x : h) x = std::ldexp(float(int(rng() % 4001U) - 2000), int(rng() % 16U) - 12);
    std::vector<float> ref(std::size_t(rows) * outputs), blk(ref.size());
    for (u32 base = 0; base < rows; base += chunk)  // per-chunk decode, the route before the down hoist
        for (u32 o = 0; o < outputs; ++o)
            for (u32 a = base; a < std::min(rows, base + chunk); ++a) {
                float v[32];
                for (u32 l = 0; l < 32U; ++l)
                    v[l] = nr::lane_value(nr::LaneSeed::direct, l, groups, [&](u32 g) {
                        float wv[32]; qt::decode_row_group(t, w.data() + o * rb, g, wv);
                        return nr::group_chain(wv, [&](u32 j) { return h[a * ffn + 32 * g + j]; });
                    });
                ref[a * outputs + o] = nr::butterfly_root(v);
            }
    // Decode once per tile, 4 rows x 4 outputs per thread (16 items), fold, read at the owner lane.
    std::vector<float> dec(std::size_t(outputs) * ffn);
    for (u32 o = 0; o < outputs; ++o) for (u32 g = 0; g < groups; ++g) qt::decode_row_group(t, w.data() + o * rb, g, &dec[o * ffn + 32 * g]);
    using F = nr::TransposeFold<16>;
    for (u32 a0 = 0; a0 < rows; a0 += 4U)
        for (u32 o0 = 0; o0 < outputs; o0 += 4U) {
            float acc[32][16] = {};
            for (u32 l = 0; l < groups; ++l)
                for (u32 i = 0; i < 16U; ++i) {
                    const u32 a = std::min(rows - 1U, a0 + i / 4U), o = o0 + i % 4U;
                    acc[l][i] = nr::group_chain(&dec[o * ffn + 32 * l], [&](u32 j) { return h[a * ffn + 32 * l + j]; });
                }
            nr::transpose_fold_simulate<16>(acc);
            for (u32 i = 0; i < 16U; ++i) {
                const u32 a = a0 + i / 4U, o = o0 + i % 4U;
                if (a < rows) blk[a * outputs + o] = acc[F::owner_lane(i)][F::owner_slot(i)];
            }
        }
    for (std::size_t i = 0; i < ref.size(); ++i) check(same(ref[i], blk[i]), "down tile", t, u32(i));
}

// Project: hidden 2560 (80 groups, LaneSeed::zero_seeded), BF16 inputs, gate and up. Per-chunk decode vs code view
// resident per tile + stage k = lane set {l + 32 k} outer + 2 rows x (2 outputs x 2 matrices) per thread + fold.
static void test_project(u32 t, std::mt19937& rng) {
    constexpr u32 hidden = 2560U, groups = hidden / 32U, outputs = 8U, rows = 21U, chunk = 8U;
    const auto wg = random_rows(t, outputs, hidden, rng), wu = random_rows(t, outputs, hidden, rng);
    const auto rb = std::size_t(hidden) / qt::geometry(t).block_elements * qt::geometry(t).block_bytes;
    std::vector<h16> x(std::size_t(rows) * hidden);
    for (auto& v : x) { v = h16(rng()); if ((v & 0x7F80U) == 0x7F80U) v &= 0xBF7FU; if ((v & 0x7F80U) > 0x4300U) v = h16((v & 0x807FU) | 0x3C00U); }
    auto xf = [&](u32 a, u32 f) { const u32 b = u32(x[a * hidden + f]) << 16U; float r; std::memcpy(&r, &b, 4); return r; };
    std::vector<float> ref(std::size_t(rows) * outputs * 2U), blk(ref.size());
    for (u32 base = 0; base < rows; base += chunk)
        for (u32 o = 0; o < outputs; ++o)
            for (u32 m = 0; m < 2U; ++m)
                for (u32 a = base; a < std::min(rows, base + chunk); ++a) {
                    float v[32];
                    for (u32 l = 0; l < 32U; ++l)
                        v[l] = nr::lane_value(nr::LaneSeed::zero_seeded, l, groups, [&](u32 g) {
                            float wv[32]; qt::decode_row_group(t, (m ? wu : wg).data() + o * rb, g, wv);
                            return nr::group_chain(wv, [&](u32 j) { return xf(a, 32 * g + j); });
                        });
                    ref[(a * outputs + o) * 2U + m] = nr::butterfly_root(v);
                }
    std::vector<Codes> codes(std::size_t(outputs) * 2U * groups);
    for (u32 o = 0; o < outputs; ++o) for (u32 m = 0; m < 2U; ++m) for (u32 g = 0; g < groups; ++g)
        codes[(o * 2U + m) * groups + g] = factor(t, (m ? wu : wg).data() + o * rb, g);
    using F = nr::TransposeFold<8>;
    for (u32 a0 = 0; a0 < rows; a0 += 2U)
        for (u32 o0 = 0; o0 < outputs; o0 += 2U) {
            float part[32][8] = {};  // lane partial per item, +0 seed
            for (u32 k = 0; k < 3U; ++k)
                for (u32 l = 0; l < 32U; ++l) {
                    const u32 g = l + 32U * k;
                    if (g >= groups) continue;
                    for (u32 i = 0; i < 8U; ++i) {
                        const u32 a = std::min(rows - 1U, a0 + i / 4U), o = o0 + (i % 4U) / 2U, m = i % 2U;
                        float wv[32]; expand(codes[(o * 2U + m) * groups + g], wv);
                        part[l][i] = nr::add_rn(part[l][i], nr::group_chain(wv, [&](u32 j) { return xf(a, 32 * g + j); }));
                    }
                }
            nr::transpose_fold_simulate<8>(part);
            for (u32 i = 0; i < 8U; ++i) {
                const u32 a = a0 + i / 4U, o = o0 + (i % 4U) / 2U, m = i % 2U;
                if (a < rows) blk[(a * outputs + o) * 2U + m] = part[F::owner_lane(i)][F::owner_slot(i)];
            }
        }
    for (std::size_t i = 0; i < ref.size(); ++i) check(same(ref[i], blk[i]), "project tile", t, u32(i));
}

static void test_seed() {
    float w[32], x[32];
    for (int j = 0; j < 32; ++j) { w[j] = -0x1p-100F; x[j] = 0x1p-100F; }  // every product underflows to -0
    const float leaf = nr::group_chain(w, [&](unsigned j) { return x[j]; });
    check(bits(leaf) == 0x80000000U, "underflow leaf is -0", 0, 0);
    const float d = nr::lane_value(nr::LaneSeed::direct, 0, 20, [&](unsigned) { return leaf; });
    const float z = nr::lane_value(nr::LaneSeed::zero_seeded, 0, 20, [&](unsigned) { return leaf; });
    check(bits(d) == 0x80000000U && bits(z) == 0U, "seeds differ only for a -0 leaf", 0, 0);
}

// Tree order (native_reuse tree_position / TreeCounter): the binary counter over the bit-reversed lane positions is the
// lane-0 butterfly root bit for bit, also for signed zeros, infinities and NaN (register slots and Deep = volatile float).
template <class Deep> static void test_tree_order(std::mt19937& rng, u32 trials) {
    for (u32 trial = 0; trial < trials; ++trial) {
        float v[32][4];
        for (auto& lane : v) for (auto& x : lane) x = (trial & 1U) ? special(rng) : std::ldexp(float(int(rng() % 2001U) - 1000), int(rng() % 40U) - 20);
        nr::TreeCounter<4, Deep> tree;
        float leaf[4] = {};
        for (u32 m = 0; m < 32U; ++m) {
            for (u32 i = 0; i < 4U; ++i) leaf[i] = v[nr::tree_position(m)][i];
            tree.push(m, leaf);
        }
        for (u32 i = 0; i < 4U; ++i) {
            float column[32];
            for (u32 l = 0; l < 32U; ++l) column[l] = v[l][i];
            check(same(nr::butterfly_root(column), leaf[i]), "tree order", trial, i);
        }
    }
    bool perm[32] = {};
    for (u32 m = 0; m < 32U; ++m) perm[nr::tree_position(m)] = true;
    check(std::all_of(perm, perm + 32, [](bool b) { return b; }), "tree_position is a permutation", 0, 0);
}

// Host replica of the per-tile kernels of src/cuda/core/moe_a8.cuh (native_project_kernel / native_down_kernel): the same
// lane map (rows wr * 8 + rl + 4 i, outputs per output warp), the stage schedule walked with next_stage exactly as the
// kernel's double buffer does, the leaf rule of each route and TreeCounter<4, volatile float>. Every (row, output) of the
// tile must be written exactly once and equal the butterfly route (lane_value + butterfly_root) bit for bit.
static void test_tile_mapping(nr::LaneSeed seed, u32 t, u32 count, bool specials, std::mt19937& rng) {
    const bool project = seed == nr::LaneSeed::zero_seeded;
    const u32 width = project ? 2560U : 640U, groups = width / 32U, outputs = project ? 32U : 64U;
    const u32 mats = project ? 2U : 1U;
    const auto wbytes = random_rows(t, outputs * mats, width, rng);
    const auto rb = std::size_t(width) / qt::geometry(t).block_elements * qt::geometry(t).block_bytes;
    std::vector<float> dec(std::size_t(outputs) * mats * width), x(std::size_t(count) * width);
    for (u32 r = 0; r < outputs * mats; ++r)
        for (u32 g = 0; g < groups; ++g) qt::decode_row_group(t, wbytes.data() + r * rb, g, &dec[std::size_t(r) * width + 32U * g]);
    for (u32 a = 0; a < count; ++a)
        for (u32 f = 0; f < width; ++f) {
            float val = std::ldexp(float(int(rng() % 4001U) - 2000), int(rng() % 16U) - 12);
            if (project) { u32 b; std::memcpy(&b, &val, 4); b &= 0xFFFF0000U; std::memcpy(&val, &b, 4); }  // BF16 values
            if (specials && a == count - 1U && f % 97U == 0U) val = special(rng);  // signed zeros / inf / NaN on the last row
            x[std::size_t(a) * width + f] = val;
        }
    auto wrow = [&](u32 mat, u32 o) { return &dec[(std::size_t(mat) * outputs + o) * width]; };
    auto reference = [&](u32 a, u32 mat, u32 o) {
        float v[32];
        for (u32 l = 0; l < 32U; ++l)
            v[l] = nr::lane_value(seed, l, groups, [&](u32 g) {
                return nr::group_chain(wrow(mat, o) + 32U * g, [&](u32 j) { return x[std::size_t(a) * width + 32U * g + j]; });
            });
        return nr::butterfly_root(v);
    };
    std::vector<u32> written(std::size_t(count) * outputs * mats, 0U);
    for (u32 warp = 0; warp < 32U; ++warp)
        for (u32 lane = 0; lane < 32U; ++lane) {
            const u32 r0 = (warp & 7U) * 8U + (lane & 3U);
            if ((warp & 7U) * 8U >= count) continue;  // inactive warp: never stores
            u32 rows[2] = {r0, r0 + 4U}, item_o[4], item_m[4], item_r[4];
            for (u32 i = 0; i < 2U; ++i)
                for (u32 k = 0; k < 2U; ++k) {
                    const u32 it = 2U * i + k;
                    item_r[it] = rows[i];
                    item_o[it] = project ? (warp >> 3) * 8U + (lane >> 2) : (warp >> 3) * 16U + (lane >> 2) + 8U * k;
                    item_m[it] = project ? k : 0U;
                }
            nr::TreeCounter<4, volatile float> tree;
            float leaf[4];
            u32 sm = 0, sg = 0, stages = 0;  // the stage the double buffer holds (stage 0 = group 0)
            bool more = true;
            for (u32 m = 0; m < 32U; ++m) {
                const u32 l = nr::tree_position(m), nt = nr::position_groups(seed, l, groups);
                for (float& f : leaf) f = 0.0F;
                for (u32 tt = 0; tt < nt; ++tt) {
                    check(more && sm == m && sg == l + 32U * tt, "stage schedule", t, m);
                    for (u32 it = 0; it < 4U; ++it) {
                        const u32 a = std::min(item_r[it], count - 1U);  // rows >= count compute on any data, never stored
                        const float* w = wrow(item_m[it], item_o[it]) + 32U * sg;
                        float acc = 0.0F;
                        for (u32 j = 0; j < 32U; ++j) acc = nr::fma_rn(w[j], x[std::size_t(a) * width + 32U * sg + j], acc);
                        leaf[it] = project ? nr::add_rn(leaf[it], acc) : acc;
                    }
                    more = nr::next_stage(seed, groups, m, tt, sm, sg);
                    ++stages;
                }
                tree.push(m, leaf);
            }
            check(!more && stages == groups, "every group staged once", t, stages);
            for (u32 it = 0; it < 4U; ++it) {
                if (item_r[it] >= count) continue;
                const std::size_t idx = (std::size_t(item_r[it]) * outputs + item_o[it]) * mats + item_m[it];
                ++written[idx];
                check(same(reference(item_r[it], item_m[it], item_o[it]), leaf[it]), project ? "project tile map" : "down tile map", t, u32(idx));
            }
        }
    check(std::all_of(written.begin(), written.end(), [](u32 n) { return n == 1U; }), "tile map writes every item once", t, count);
}

static void print_budgets() {
    // Candidate decoded-tile mappings (planning numbers; sm_120 limits 99 KiB shared per block, 64 regs at 1024 threads).
    struct Row { const char* name; nr::TileBudget b; unsigned threads; };
    const Row rows_[] = {
        {"project fp32-resident stage32 o8 r16", {8, 2, 32, 128, 16, 32, 2, 2, 4, true}, 1024},
        {"project codes-resident all80 o8 r16", {8, 2, 80, 40, 16, 32, 2, 2, 4, true}, 1024},
        {"project codes-resident all80 o8 r16 ra4", {8, 2, 80, 40, 16, 32, 2, 4, 4, true}, 512},
        {"down fp32-resident all20 o32 r8", {32, 1, 20, 128, 8, 20, 4, 2, 4, false}, 1024},
        {"down codes-resident all20 o32 r16", {32, 1, 20, 40, 16, 20, 4, 4, 4, false}, 1024},
    };
    std::printf("%-42s %7s %7s %7s %5s %8s %8s\n", "mapping", "w_smem", "x_smem", "smem", "regs", "B/FMA32", "B/FMA8");
    for (const auto& r : rows_)
        std::printf("%-42s %7u %7u %7u %5u %8.2f %8.2f  threads %u\n", r.name, r.b.weight_smem(), r.b.x_smem(), r.b.smem(),
                    r.b.accumulator_regs(), r.b.smem_bytes_per_fma(4), r.b.smem_bytes_per_fma(1), r.threads);
    std::printf("fold shuffles per lane: N=4 %u, N=8 %u, N=16 %u, N=32 %u (plain butterfly: 5 per item)\n",
                nr::TransposeFold<4>::shuffles(), nr::TransposeFold<8>::shuffles(), nr::TransposeFold<16>::shuffles(), nr::TransposeFold<32>::shuffles());
}

int main() {
    std::mt19937 rng(20261117U);
    test_codes(rng);
    test_fold<1>(rng, 64); test_fold<2>(rng, 64); test_fold<4>(rng, 64); test_fold<8>(rng, 64);
    test_fold<16>(rng, 64); test_fold<32>(rng, 64); test_fold<128>(rng, 16);
    test_seed();
    for (u32 t : {qt::t_iq4_nl, qt::t_q2_0}) test_down(t, rng);  // down types of the native packs (ffn 640)
    for (u32 t : {qt::t_iq3_xxs, qt::t_iq3_s, qt::t_iq2_s, qt::t_iq4_xs, qt::t_q2_0}) test_project(t, rng);
    test_tree_order<float>(rng, 512); test_tree_order<volatile float>(rng, 512);
    // Fixture shapes of the tile kernels: T = 1 and 9 (old capture records), a full tile, a tile whose last 8-row warp is
    // partial (37), the single-row second tile of T = 65, and signed-zero / non-finite inputs.
    for (u32 count : {1U, 9U, 37U, 64U}) {
        for (u32 t : {qt::t_iq4_nl, qt::t_q2_0}) test_tile_mapping(nr::LaneSeed::direct, t, count, count == 37U, rng);
        for (u32 t : {qt::t_iq3_xxs, qt::t_iq3_s, qt::t_iq2_s, qt::t_iq4_xs}) test_tile_mapping(nr::LaneSeed::zero_seeded, t, count, count == 37U, rng);
    }
    test_tile_mapping(nr::LaneSeed::direct, qt::t_iq4_nl, 1U, true, rng);
    test_tile_mapping(nr::LaneSeed::zero_seeded, qt::t_iq3_s, 1U, true, rng);
    print_budgets();
    std::printf("native_reuse host test: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
