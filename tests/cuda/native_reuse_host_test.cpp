// Host check of src/cuda/core/native_reuse.hpp, the shared contract of the native SIMT expert route
// (cuda-native-iq-fp32@1), on the six native GSQ storage types with the shared decoders (quant::decode_row_group):
//   codes     : each decoded group equals s[j / 16] * float(c[j]) bitwise (32 int8 codes, 2 FP32 sub-scales);
//   fold      : the register transpose-fold gives every item the xor-butterfly root bits at its owner lane;
//   seed      : a -0 leaf is the only case where the two lane seeds differ;
//   down/proj : decode once per tile + register blocking + fold (and the code view) == the per-chunk route, bit for bit.
// No GPU. Release build, ctest; failures print and return 1 (no dialogs).
#include "../../src/cuda/core/native_reuse.hpp"
#include "../../src/cuda/core/native_mma.hpp"
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
// thread map (native_reuse::MicroTile: 256 threads, rows 8 w + 4 rh + i, outputs og + 16 k), the stage schedule walked with
// next_stage exactly as the kernel's double buffer does, the leaf rule of each route and the kernel's TreeCounter<16, ...>.
// Every (row, output, matrix) of the tile must be written exactly once and equal the butterfly route (lane_value +
// butterfly_root) bit for bit.
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
    using PT = nr::MicroTile<2>;
    using DT = nr::MicroTile<4>;
    check(PT::tile_outputs == 32U && DT::tile_outputs == 64U && PT::threads == 256U, "micro-tile block shape", t, 0);
    for (u32 th = 0; th < 256U; ++th) {
        if ((th >> 5) * 8U >= count) continue;  // inactive warp: never stores
        u32 item_o[16], item_m[16], item_r[16];
        for (u32 i = 0; i < 4U; ++i)
            for (u32 c = 0; c < 4U; ++c) {
                const u32 it = 4U * i + c;  // project: c = 2 k + mat; down: c = k
                item_r[it] = project ? PT::row(th, i) : DT::row(th, i);
                item_o[it] = project ? PT::output(th, c >> 1) : DT::output(th, c);
                item_m[it] = project ? (c & 1U) : 0U;
            }
        nr::TreeCounter<16, volatile float, volatile float> tree;
        float leaf[16];
        u32 sm = 0, sg = 0, stages = 0;  // the stage the double buffer holds (stage 0 = group 0)
        bool more = true;
        for (u32 m = 0; m < 32U; ++m) {
            const u32 l = nr::tree_position(m), nt = nr::position_groups(seed, l, groups);
            for (float& f : leaf) f = 0.0F;
            for (u32 tt = 0; tt < nt; ++tt) {
                check(more && sm == m && sg == l + 32U * tt, "stage schedule", t, m);
                for (u32 it = 0; it < 16U; ++it) {
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
        for (u32 it = 0; it < 16U; ++it) {
            if (item_r[it] >= count) continue;
            const std::size_t idx = (std::size_t(item_r[it]) * outputs + item_o[it]) * mats + item_m[it];
            ++written[idx];
            check(same(reference(item_r[it], item_m[it], item_o[it]), leaf[it]), project ? "project tile map" : "down tile map", t, u32(idx));
        }
    }
    check(std::all_of(written.begin(), written.end(), [](u32 n) { return n == 1U; }), "tile map writes every item once", t, count);
}

// The micro-tile's shared-memory reads, per quarter warp (the LDS.128 phase): X rows and W outputs at pitch 36 floats must
// hit distinct 16-byte bank quads (or the same address), for every k-step q; and the 256 threads cover each tile item once.
template <unsigned K> static void test_micro_tile_banks() {
    using MT = nr::MicroTile<K>;
    std::vector<u32> cover(std::size_t(MT::tile_rows) * MT::tile_outputs, 0U);
    for (u32 th = 0; th < MT::threads; ++th)
        for (u32 i = 0; i < MT::rows; ++i)
            for (u32 k = 0; k < K; ++k) ++cover[std::size_t(MT::row(th, i)) * MT::tile_outputs + MT::output(th, k)];
    check(std::all_of(cover.begin(), cover.end(), [](u32 n) { return n == 1U; }), "micro-tile covers every item once", K, 0);
    for (u32 q = 0; q < 8U; ++q)
        for (u32 base = 0; base < MT::threads; base += 8U) {
            for (u32 i = 0; i < MT::rows + K; ++i) {  // X read i (i < rows) or W read k = i - rows
                u32 addr[8], quad[8];
                for (u32 l = 0; l < 8U; ++l) {
                    const u32 th = base + l, line = i < MT::rows ? MT::row(th, i) : MT::output(th, i - MT::rows);
                    addr[l] = line * 36U + 4U * q;
                    quad[l] = (addr[l] / 4U) % 8U;
                }
                for (u32 l = 0; l < 8U; ++l)
                    for (u32 n = 0; n < l; ++n) check(addr[l] == addr[n] || quad[l] != quad[n], "micro-tile bank quad", K, base + l);
            }
        }
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

// ---- gsq-iq-bf16mma-cuda-v1 (native_mma.hpp): fragment geometry, exact conversions, hi/lo split, tile DAG ------------
// The device sum order inside one MMA is the hardware's, so this oracle checks (a) that the kernel's register code pairs
// every logical k exactly once on both operands, (b) that the operands are exact, and (c) the contract's error bound on
// the tile DAG (FP32 half sums, fma group scales) against a double reference, beside the FP32-chain class for scale.
namespace nm = strix::cuda::core::native_mma;
static float bf16v(float v) { return nm::bf16_float(nm::bf16_rn(v)); }
static double elem(u32 reg, u32 e) { return double(nm::bf16_float((reg >> (16U * e)) & 0xFFFFU)); }
static u32 pair(float lo, float hi) { return nm::bf16_rn(lo) | (nm::bf16_rn(hi) << 16U); }

static void test_mma_geometry(std::mt19937& rng) {
    std::uniform_int_distribution<int> ci(-127, 127), xi(-200, 200);
    for (int trial = 0; trial < 64; ++trial) {
        float x[16][32]; int code[8][32];
        for (auto& r : x) for (float& v : r) v = bf16v(float(xi(rng)) * 0.25F);
        for (auto& r : code) for (int& c : r) c = ci(rng);
        for (u32 h = 0; h < 2U; ++h) {
            double A[16][16] = {}, B[16][8] = {}; int fa[16][16] = {}, fb[16][8] = {};
            for (u32 lane = 0; lane < 32U; ++lane) {
                const u32 g = lane / 4U, k = 16U * h + 4U * (lane % 4U);
                u32 a[4];
                nm::a_regs(pair(x[g][k], x[g][k + 1]), pair(x[g][k + 2], x[g][k + 3]), pair(x[g + 8][k], x[g + 8][k + 1]),
                           pair(x[g + 8][k + 2], x[g + 8][k + 3]), a);
                for (u32 r = 0; r < 4U; ++r) for (u32 e = 0; e < 2U; ++e) {
                    const u32 row = nm::a_row(lane, r), p = nm::a_slot(lane, r, e);
                    A[row][p] = elem(a[r], e); ++fa[row][p];
                }
                const u32 wi = nm::word_index(lane, h), col = nm::b_col(lane);
                u32 word = 0;
                for (u32 j = 0; j < 4U; ++j) word |= (u32(code[col][4U * wi + j]) & 255U) << (8U * j);
                u32 b[2]; nm::code_pairs(word, b[0], b[1]);
                for (u32 r = 0; r < 2U; ++r) for (u32 e = 0; e < 2U; ++e) {
                    const u32 p = nm::b_slot(lane, r, e);
                    B[p][col] = elem(b[r], e); ++fb[p][col];
                }
            }
            for (u32 row = 0; row < 16U; ++row) for (u32 p = 0; p < 16U; ++p) {
                check(fa[row][p] == 1 && A[row][p] == double(x[row][nm::logical_k(h, p)]), "mma A slot", h, row * 16U + p);
                if (row < 8U) check(fb[p][row] == 1 && B[p][row] == double(code[row][nm::logical_k(h, p)]), "mma B slot", h, p * 8U + row);
            }
            for (u32 r = 0; r < 16U; ++r) for (u32 c = 0; c < 8U; ++c) {
                double d = 0.0, ref = 0.0;
                for (u32 p = 0; p < 16U; ++p) d += A[r][p] * B[p][c];
                for (u32 k = 16U * h; k < 16U * h + 16U; ++k) ref += double(x[r][k]) * double(code[c][k]);
                check(d == ref, "mma half sum", h, r * 8U + c);
            }
        }
    }
    int fc[16][8] = {};
    for (u32 lane = 0; lane < 32U; ++lane) for (u32 i = 0; i < 4U; ++i) {
        ++fc[nm::c_row(lane, i)][nm::c_col(lane, i)];
        check(nm::b_col(nm::scale_lane(lane, i)) == nm::c_col(lane, i), "mma scale lane", lane, i);
    }
    for (u32 r = 0; r < 16U; ++r) for (u32 c = 0; c < 8U; ++c) check(fc[r][c] == 1, "mma C element", r, c);
}

static void test_mma_conversions(std::mt19937& rng) {
    for (int c = -128; c <= 127; ++c) {
        const float f = nm::bf16_float(nm::code_bf16(c));
        check(f == float(c) && bits(f) == bits(float(c)), "code bf16 exact", u32(c + 128), 0);
    }
    std::uniform_int_distribution<u32> any;
    for (int i = 0; i < 4096; ++i) {
        const u32 w = any(rng); u32 b0, b1; nm::code_pairs(w, b0, b1);
        for (u32 j = 0; j < 4U; ++j)
            check(elem(j < 2U ? b0 : b1, j & 1U) == double(std::int8_t((w >> (8U * j)) & 255U)), "code pairs", w, j);
    }
    // bf16_rn against a direct nearest-even choice; specials: +-0, +-inf, max finite (overflows), subnormals.
    std::vector<u32> pats = {0U, 0x80000000U, 0x7F800000U, 0xFF800000U, 0x7F7FFFFFU, 0x00000001U, 0x807FFFFFU, 0x00018000U, 0x3F808000U, 0x3F818000U};
    for (int i = 0; i < 200000; ++i) pats.push_back(any(rng));
    for (u32 u : pats) {
        if ((u & 0x7FFFFFFFU) > 0x7F800000U) continue;
        float x; std::memcpy(&x, &u, 4);
        const u32 lo = u >> 16U, up = lo + 1U;
        const double dl = std::fabs(double(x) - double(nm::bf16_float(lo)));
        const double du = std::isinf(nm::bf16_float(up)) ? std::ldexp(1.0, 128) - std::fabs(double(x)) : std::fabs(double(nm::bf16_float(up)) - double(x));
        const u32 want = (u & 0xFFFFU) == 0U ? lo : (dl < du || (dl == du && !(lo & 1U))) ? lo : up;
        check(nm::bf16_rn(x) == want, "bf16 rn", u, 0);
    }
    // Split: |x - (hi + lo)| <= 2^-17 |x| for normal x; +-0 reconstruct exactly.
    std::normal_distribution<float> nd(0.0F, 1.0F); std::uniform_int_distribution<int> ex(-100, 100);
    double worst = 0.0;
    for (int i = 0; i < 200000; ++i) {
        const float x = std::ldexp(nd(rng), ex(rng));
        if (x == 0.0F || !std::isnormal(x)) continue;
        u32 hi, lo; nm::split(x, hi, lo);
        const double rel = std::fabs(double(x) - (double(nm::bf16_float(hi)) + double(nm::bf16_float(lo)))) / std::fabs(double(x));
        worst = std::max(worst, rel);
        check(rel <= std::ldexp(1.0, -17), "split bound", bits(x), 0);
    }
    for (float z : {0.0F, -0.0F}) { u32 hi, lo; nm::split(z, hi, lo); check(nm::bf16_float(hi) + nm::bf16_float(lo) == 0.0F && (hi & 0x7FFFU) == 0U && lo == 0U, "split zero", bits(z), 0); }
    std::printf("mma split: worst relative residual %.3g (bound 2^-17 = %.3g)\n", worst, std::ldexp(1.0, -17));
}

static double g_mma_worst = 0.0, g_fp32_worst = 0.0;
// One tile of `count` rows through the project and down DAG of the tensor-core kernels, at reduced widths with the same
// stage / block geometry (hidden 512: 4 project stages, 4 down blocks; ffn 128: 2 project blocks, 2 down stages).
static void test_mma_tile(u32 tg, u32 td, u32 count, std::mt19937& rng) {
    constexpr u32 hidden = 512U, ffn = 128U;
    const auto gg = qt::geometry(tg), gd = qt::geometry(td);
    const std::size_t grow = hidden / gg.block_elements * gg.block_bytes, drow = ffn / gd.block_elements * gd.block_bytes;
    const auto gate = random_rows(tg, ffn, hidden, rng), up = random_rows(tg, ffn, hidden, rng), down = random_rows(td, hidden, ffn, rng);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(std::size_t(count) * hidden);
    for (float& v : x) v = bf16v(nd(rng));
    if (count > 1U) x[1] = -0.0F;
    const bool gh = tg == qt::t_iq2_s, dh = td == qt::t_iq2_s;
    // Model of one (row, output): FP32 half sums of exact products, then fma with the group scale; double reference.
    auto dot = [&](bool halves, u32 t, const u8* wrow, const float* in, u32 width, bool split_in, double& ref, double& mag, float& fp32) {
        float acc = 0.0F; ref = 0.0; mag = 0.0; fp32 = 0.0F;
        for (u32 g = 0; g < width / 32U; ++g) {
            const Codes cw = factor(t, wrow, g);
            double half[2] = {0.0, 0.0};
            for (u32 k = 0; k < 32U; ++k) {
                double v = double(in[32U * g + k]);
                if (split_in) { u32 hi, lo; nm::split(in[32U * g + k], hi, lo); v = double(nm::bf16_float(hi)) + double(nm::bf16_float(lo)); }
                half[k >> 4] += v * double(cw.c[k]);
                const double w = double(cw.s[k >> 4]) * double(cw.c[k]);
                ref += w * double(in[32U * g + k]); mag += std::fabs(w * double(in[32U * g + k]));
                fp32 = std::fmaf(cw.s[k >> 4] * float(cw.c[k]), in[32U * g + k], fp32);
            }
            if (halves) { acc = std::fmaf(float(half[0]), cw.s[0], acc); acc = std::fmaf(float(half[1]), cw.s[1], acc); }
            else acc = std::fmaf(float(half[0] + half[1]), cw.s[0], acc);
        }
        return acc;
    };
    auto judge = [&](float got, double ref, double mag, float fp32, u32 groups, bool split_in, const char* what, u32 i) {
        if (!std::isfinite(mag) || mag == 0.0) return;
        const double tol = (2.0 * groups + 4.0) * std::ldexp(1.0, -24) * mag + (split_in ? std::ldexp(1.0, -17) * mag : 0.0);
        check(std::fabs(double(got) - ref) <= tol, what, count, i);
        g_mma_worst = std::max(g_mma_worst, std::fabs(double(got) - ref) / mag);
        g_fp32_worst = std::max(g_fp32_worst, std::fabs(double(fp32) - ref) / mag);
    };
    std::vector<int> wp(std::size_t(count) * ffn, 0), wd(std::size_t(count) * hidden, 0);
    std::vector<float> h(std::size_t(count) * ffn, 0.0F);
    for (u32 by = 0; by < ffn / nm::kProjectOutputs; ++by)
        for (u32 warp = 0; warp < nm::kThreads / 32U; ++warp)
            for (u32 lane = 0; lane < 32U; ++lane)
                for (u32 m = 0; m < nm::m_blocks(count); ++m)
                    for (u32 i = 0; i < 4U; ++i) {
                        const u32 row = 16U * m + nm::c_row(lane, i), o = nm::project_output(by, warp, nm::c_col(lane, i));
                        if (row >= count) continue;
                        ++wp[std::size_t(row) * ffn + o];
                        double ref, mag; float fp32;
                        const float gv = dot(gh, tg, gate.data() + o * grow, &x[std::size_t(row) * hidden], hidden, false, ref, mag, fp32);
                        judge(gv, ref, mag, fp32, hidden / 32U, false, "mma project gate", row * ffn + o);
                        const float uv = dot(gh, tg, up.data() + o * grow, &x[std::size_t(row) * hidden], hidden, false, ref, mag, fp32);
                        judge(uv, ref, mag, fp32, hidden / 32U, false, "mma project up", row * ffn + o);
                        h[std::size_t(row) * ffn + o] = gv / (1.0F + std::exp(-gv)) * uv;
                    }
    for (std::size_t i = 0; i < wp.size(); ++i) check(wp[i] == 1, "mma project coverage", count, u32(i));
    for (u32 by = 0; by < hidden / nm::kDownOutputs; ++by)
        for (u32 warp = 0; warp < nm::kThreads / 32U; ++warp)
            for (u32 n = 0; n < 2U; ++n)
                for (u32 lane = 0; lane < 32U; ++lane)
                    for (u32 m = 0; m < nm::m_blocks(count); ++m)
                        for (u32 i = 0; i < 4U; ++i) {
                            const u32 row = 16U * m + nm::c_row(lane, i), o = nm::down_output(by, warp, n, nm::c_col(lane, i));
                            if (row >= count) continue;
                            ++wd[std::size_t(row) * hidden + o];
                            double ref, mag; float fp32;
                            const float dv = dot(dh, td, down.data() + o * drow, &h[std::size_t(row) * ffn], ffn, true, ref, mag, fp32);
                            judge(dv, ref, mag, fp32, ffn / 32U, true, "mma down", row * hidden + o);
                        }
    for (std::size_t i = 0; i < wd.size(); ++i) check(wd[i] == 1, "mma down coverage", count, u32(i));
    // Staging coverage of the kernels' loops: project copies word t % 64 of rows t / 64 + 4 i; down thread t owns
    // values 16 (t % 4) .. of row t / 4 (hi and lo planes alike).
    std::vector<int> sp(std::size_t(count) * nm::kProjectK, 0), sd(std::size_t(count) * nm::kDownK, 0);
    for (u32 t = 0; t < nm::kThreads; ++t) {
        for (u32 r = t >> 6; r < count; r += nm::kThreads / 64U) { ++sp[r * nm::kProjectK + 2U * (t & 63U)]; ++sp[r * nm::kProjectK + 2U * (t & 63U) + 1U]; }
        if ((t >> 2) < count) for (u32 k = 0; k < 16U; ++k) ++sd[(t >> 2) * nm::kDownK + 16U * (t & 3U) + k];
    }
    for (std::size_t i = 0; i < sp.size(); ++i) check(sp[i] == 1, "mma project staging", count, u32(i));
    for (std::size_t i = 0; i < sd.size(); ++i) check(sd[i] == 1, "mma down staging", count, u32(i));
}

// Tile-major raster of the FAST kernels (native_mma::tile_major) on their launch grids (tiles, ffn / 64) and
// (tiles, hidden / 128): every (tile, slice) is owned by exactly one block, the slices of a tile are consecutive in dispatch
// order, the blocks past tile_count are exactly (tiles - tile_count) x slices, and each tile still writes every output of
// both kernels once (project_output / down_output unchanged).
static void test_mma_raster() {
    constexpr u32 ffn = 640U, hidden = 2560U;
    for (u32 slices : {ffn / nm::kProjectOutputs, hidden / nm::kDownOutputs}) {
        const bool project = slices == ffn / nm::kProjectOutputs;
        for (u32 tiles : {1U, 2U, 9U, 65U, 1262U, 1775U}) {
            std::vector<u8> seen(std::size_t(tiles) * slices, 0);
            const u32 tile_count = tiles > 2U ? tiles - 2U : tiles;
            u32 exited = 0, prev_tile = 0;
            for (u32 by = 0; by < slices; ++by)
                for (u32 bx = 0; bx < tiles; ++bx) {
                    const nm::TileSlice ts = nm::tile_major(bx, by, tiles, slices);
                    const u32 l = bx + tiles * by;
                    check(ts.tile < tiles && ts.slice < slices, "mma raster range", tiles, l);
                    if (ts.tile < tiles && ts.slice < slices) ++seen[std::size_t(ts.tile) * slices + ts.slice];
                    check(ts.slice == l % slices && (l == 0U || ts.tile == prev_tile + (ts.slice == 0U ? 1U : 0U)),
                          "mma raster tile-major order", tiles, l);
                    prev_tile = ts.tile;
                    exited += ts.tile >= tile_count ? 1U : 0U;
                }
            for (std::size_t i = 0; i < seen.size(); ++i) check(seen[i] == 1, "mma raster bijection", tiles, u32(i));
            check(exited == (tiles - tile_count) * slices, "mma raster early exit", tiles, exited);
            std::vector<u8> out(project ? ffn : hidden, 0);
            for (u32 s = 0; s < slices; ++s)
                for (u32 w = 0; w < nm::kThreads / 32U; ++w)
                    for (u32 n = 0; n < (project ? 1U : 2U); ++n)
                        for (u32 c = 0; c < 8U; ++c) {
                            const u32 o = project ? nm::project_output(s, w, c) : nm::down_output(s, w, n, c);
                            if (o < out.size()) ++out[o];
                        }
            for (std::size_t i = 0; i < out.size(); ++i) check(out[i] == 1, "mma raster output cover", slices, u32(i));
        }
    }
}

// Decode-tile form of the tensor-core expert kernels (native_mma::decode_form / ring_group): (1) a D-deep ring hands the
// groups to the MMAs in order 0 .. groups - 1, each loaded before use, never past the last group, and D = 1 is the tile
// form's one-group lookahead (next = group + 1, the last re-loads itself); (2) a decode-form step has <= 16 rows, and with
// distinct top-k experts per row every tile (one expert's rows) fits the kDecodeMBlocks m16 blocks.
static void test_mma_decode_form(std::mt19937& rng) {
    for (u32 depth = 1; depth <= 4U; ++depth)
        for (u32 groups = 1; groups <= 96U; ++groups) {
            std::vector<u32> ring(depth);
            for (u32 i = 0; i < depth; ++i) ring[i] = nm::ring_group(0U, i, groups);
            for (u32 group = 0; group < groups; ++group) {
                check(ring[0] == group, "mma ring order", depth, groups * 1000U + group);
                for (u32 i = 0; i + 1U < depth; ++i) ring[i] = ring[i + 1U];
                ring[depth - 1U] = nm::ring_group(group, depth, groups);
                check(ring[depth - 1U] < groups, "mma ring bound", depth, groups * 1000U + group);
                if (depth == 1U) check(ring[0] == (group + 1U < groups ? group + 1U : group), "mma ring tile form", groups, group);
            }
        }
    check(nm::kTileMBlocks * 16U == 64U && nm::kTileAhead == 1U, "mma tile form constants", 0U, 0U);
    for (u32 rows = 1; rows <= 64U; ++rows) {
        check(nm::decode_form(rows) == (rows <= 16U * nm::kDecodeMBlocks), "mma decode form", 0U, rows);
        if (!nm::decode_form(rows)) continue;
        std::vector<u32> per_expert(512U, 0U);
        for (u32 r = 0; r < rows; ++r) {
            std::vector<u32> pick;
            while (pick.size() < 10U) {
                const u32 e = u32(rng() % 512U);
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
            for (u32 e : pick) ++per_expert[e];
        }
        for (u32 e = 0; e < 512U; ++e) check(nm::m_blocks(per_expert[e]) <= nm::kDecodeMBlocks, "mma decode tile capacity", rows, e);
    }
}

int main() {
    std::mt19937 rng(20261117U);
    test_mma_decode_form(rng);
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
    test_micro_tile_banks<2>(); test_micro_tile_banks<4>();
    test_mma_geometry(rng);
    test_mma_conversions(rng);
    test_mma_raster();
    for (u32 count : {1U, 9U, 37U, 64U}) {
        u32 n = 0;
        for (u32 tg : {qt::t_iq3_xxs, qt::t_iq3_s, qt::t_iq2_s, qt::t_iq4_xs, qt::t_q2_0})
            test_mma_tile(tg, (n++ & 1U) ? qt::t_q2_0 : qt::t_iq4_nl, count, rng);
    }
    std::printf("mma tile DAG: worst |err| / sum|terms| %.3g (FP32 chain class on the same items %.3g)\n", g_mma_worst, g_fp32_worst);
    print_budgets();
    std::printf("native_reuse host test: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
