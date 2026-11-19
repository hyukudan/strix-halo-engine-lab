// CPU oracle of the TURBO integer expert contract gsq-iq-i8mma-turbo-cuda-v1 (src/cuda/core/turbo_i8.hpp).
// 1. The kernel's operand placement (8-byte A row loads, packed code words) put through the PTX ISA fragment layout of
//    mma.sync m16n8k32 / m16n8k16 .s8 (written out here independently of the header) gives the logical group dot.
// 2. The magic accumulator start converts every reachable group dot exactly.
// 3. A whole tile, emulated in the kernel's order (groups, m16 blocks, lanes, C elements, scale owner lanes), equals the
//    contract's reference loop bit for bit, for one-scale and two-scale weights and partial tiles with garbage rows.
// 4. Sensitivity: the unpermuted placement with the same loads gives wrong dots.
// Raster: 5. the shared tile-major map is a bijection of the TURBO grids with the slices of one tile on consecutive blocks.
#include "../../src/cuda/core/native_mma.hpp"
#include "../../src/cuda/core/turbo_i8.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace ti = strix::cuda::core::turbo_i8;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}

// PTX ISA (independent statement): m16n8k32 .s8, g = lane / 4, q = lane % 4.
// A reg r: row g + 8 (r & 1), k 4 q + 16 (r >> 1) + e.  B reg r: column g, k 4 q + 16 r + e.
// m16n8k16 .s8: A reg r: row g + 8 r, k 4 q + e.  B reg 0: column g, k 4 q + e.  C element i: row g + 8 (i >> 1), col 2 q + (i & 1).
struct Frag32 { std::uint32_t a[32][4], b[32][2]; };
static std::int8_t byte_of(std::uint32_t w, unsigned e) { return static_cast<std::int8_t>((w >> (8U * e)) & 255U); }
static std::uint32_t pack(const std::int8_t* p) {
    std::uint32_t w = 0;
    for (unsigned e = 0; e < 4U; ++e) w |= std::uint32_t(std::uint8_t(p[e])) << (8U * e);
    return w;
}
// D (int32 bits, C = magic) per (lane, element) of one IMMA, from the registers through the ISA layout.
static void isa_k32(const Frag32& f, std::uint32_t (&d)[32][4]) {
    int A[16][32] = {}, B[32][8] = {};
    for (unsigned lane = 0; lane < 32U; ++lane)
        for (unsigned e = 0; e < 4U; ++e) {
            const unsigned g = lane / 4U, q = lane % 4U;
            for (unsigned r = 0; r < 4U; ++r) A[g + 8U * (r & 1U)][4U * q + 16U * (r >> 1U) + e] = byte_of(f.a[lane][r], e);
            for (unsigned r = 0; r < 2U; ++r) B[4U * q + 16U * r + e][g] = byte_of(f.b[lane][r], e);
        }
    for (unsigned lane = 0; lane < 32U; ++lane)
        for (unsigned i = 0; i < 4U; ++i) {
            const unsigned row = lane / 4U + 8U * (i >> 1U), col = 2U * (lane % 4U) + (i & 1U);
            int dot = 0;
            for (unsigned k = 0; k < 32U; ++k) dot += A[row][k] * B[k][col];
            d[lane][i] = ti::kMagicBits + static_cast<std::uint32_t>(dot);
        }
}
static void isa_k16(const std::uint32_t (&a)[32][2], const std::uint32_t (&b)[32], std::uint32_t (&d)[32][4]) {
    int A[16][16] = {}, B[16][8] = {};
    for (unsigned lane = 0; lane < 32U; ++lane)
        for (unsigned e = 0; e < 4U; ++e) {
            const unsigned g = lane / 4U, q = lane % 4U;
            for (unsigned r = 0; r < 2U; ++r) A[g + 8U * r][4U * q + e] = byte_of(a[lane][r], e);
            B[4U * q + e][g] = byte_of(b[lane], e);
        }
    for (unsigned lane = 0; lane < 32U; ++lane)
        for (unsigned i = 0; i < 4U; ++i) {
            const unsigned row = lane / 4U + 8U * (i >> 1U), col = 2U * (lane % 4U) + (i & 1U);
            int dot = 0;
            for (unsigned k = 0; k < 16U; ++k) dot += A[row][k] * B[k][col];
            d[lane][i] = ti::kMagicBits + static_cast<std::uint32_t>(dot);
        }
}

// A tile problem: rows x K activation codes (logical k order, row pitch = K) with deltas per group, weight codes per
// output (logical k) with one scale per (output, group[, half]).
struct Problem {
    unsigned rows, K, outs; bool halves;
    std::vector<std::int8_t> x, w;
    std::vector<float> ex, sw;  // ex[row * groups + g], sw[(out * groups + g) * 2 + h]
};
static Problem make(std::mt19937& rng, unsigned rows, unsigned K, unsigned outs, bool halves) {
    Problem p{rows, K, outs, halves, {}, {}, {}, {}};
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(1e-4F, 2e-2F), sign(-1.0F, 1.0F);
    p.x.resize(std::size_t(rows) * K); p.w.resize(std::size_t(outs) * K);
    for (auto& c : p.x) c = static_cast<std::int8_t>(code(rng));
    for (auto& c : p.w) c = static_cast<std::int8_t>(code(rng));
    p.ex.resize(std::size_t(rows) * (K / 32U)); p.sw.resize(std::size_t(outs) * (K / 32U) * 2U);
    for (auto& e : p.ex) e = scale(rng);
    for (auto& s : p.sw) s = scale(rng) * (sign(rng) < 0.0F ? -1.0F : 1.0F);
    return p;
}
// The contract, written as its definition.
static float reference(const Problem& p, unsigned row, unsigned out) {
    float acc = 0.0F;
    for (unsigned g = 0; g < p.K / 32U; ++g) {
        const float e = p.ex[row * (p.K / 32U) + g];
        if (!p.halves) {
            int dot = 0;
            for (unsigned k = 0; k < 32U; ++k) dot += p.x[row * p.K + 32U * g + k] * p.w[out * p.K + 32U * g + k];
            acc = ti::step(acc, static_cast<float>(dot), e, p.sw[(out * (p.K / 32U) + g) * 2U]);
        } else {
            for (unsigned h = 0; h < 2U; ++h) {
                int dot = 0;
                for (unsigned k = 16U * h; k < 16U * h + 16U; ++k) dot += p.x[row * p.K + 32U * g + k] * p.w[out * p.K + 32U * g + k];
                acc = ti::step(acc, static_cast<float>(dot), e, p.sw[(out * (p.K / 32U) + g) * 2U + h]);
            }
        }
    }
    return acc;
}
// The kernel's order: per group, per m16 block of the tile (rows >= count hold garbage), per n8 block of outputs; operands
// loaded exactly as turbo_group / turbo_weights do; scale from the owner lane (shuffle). permuted=false: sensitivity arm
// (identity words q, q + 4 against the permuted 8-byte A loads).
static std::vector<float> emulate(const Problem& p, unsigned count, bool permuted, std::mt19937& rng) {
    const unsigned mblocks = ti::m_blocks(count), groups = p.K / 32U;
    std::vector<std::int8_t> staged(std::size_t(16U * mblocks) * p.K);
    std::uniform_int_distribution<int> junk(-128, 127);
    for (unsigned r = 0; r < 16U * mblocks; ++r)
        for (unsigned k = 0; k < p.K; ++k) staged[std::size_t(r) * p.K + k] = r < count ? p.x[r * p.K + k] : static_cast<std::int8_t>(junk(rng));
    std::vector<float> out(std::size_t(count) * p.outs, 0.0F), acc(std::size_t(16U * mblocks) * p.outs, 0.0F);
    for (unsigned g = 0; g < groups; ++g)
        for (unsigned nb = 0; nb < p.outs / 8U; ++nb) {
            // turbo_weights: lane reads words of weight row nb*8 + b_col(lane); scale of its own row, then the owner lanes'.
            float own[32][2];
            std::uint32_t words[32][2];
            for (unsigned lane = 0; lane < 32U; ++lane) {
                const unsigned o = 8U * nb + ti::b_col(lane);
                const std::int8_t* code = &p.w[std::size_t(o) * p.K + 32U * g];
                for (unsigned r = 0; r < 2U; ++r)
                    words[lane][r] = pack(code + 4U * (p.halves ? ti::k16_word(lane, r) : (permuted ? ti::k32_word(lane, r) : lane % 4U + 4U * r)));
                own[lane][0] = p.sw[(o * groups + g) * 2U];
                own[lane][1] = p.sw[(o * groups + g) * 2U + (p.halves ? 1U : 0U)];
            }
            for (unsigned m = 0; m < mblocks; ++m) {
                std::uint32_t d[2][32][4];
                const unsigned passes = p.halves ? 2U : 1U;
                for (unsigned h = 0; h < passes; ++h) {
                    if (!p.halves) {
                        Frag32 f;
                        for (unsigned lane = 0; lane < 32U; ++lane) {
                            const unsigned row0 = 16U * m + lane / 4U;
                            const unsigned k = 32U * g + ti::k32_a_byte(lane);  // one 8-byte load per row
                            const std::int8_t* r0 = &staged[std::size_t(row0) * p.K + k];
                            const std::int8_t* r1 = &staged[std::size_t(row0 + 8U) * p.K + k];
                            ti::a_regs(pack(r0), pack(r0 + 4), pack(r1), pack(r1 + 4), f.a[lane]);
                            f.b[lane][0] = words[lane][0]; f.b[lane][1] = words[lane][1];
                        }
                        isa_k32(f, d[0]);
                    } else {
                        std::uint32_t a[32][2], b[32];
                        for (unsigned lane = 0; lane < 32U; ++lane) {
                            const unsigned row0 = 16U * m + lane / 4U, k = 32U * g + ti::k16_a_byte(lane, h);
                            a[lane][0] = pack(&staged[std::size_t(row0) * p.K + k]);
                            a[lane][1] = pack(&staged[std::size_t(row0 + 8U) * p.K + k]);
                            b[lane] = words[lane][h];
                        }
                        isa_k16(a, b, d[h]);
                    }
                }
                for (unsigned lane = 0; lane < 32U; ++lane)
                    for (unsigned i = 0; i < 4U; ++i) {
                        const unsigned row = 16U * m + ti::c_row(lane, i), col = ti::c_col(lane, i);
                        const unsigned owner = ti::scale_lane(lane, i);
                        if (ti::b_col(owner) != col) { check(false, "scale owner lane holds the C column's weight row"); return out; }
                        const float e = row < count ? p.ex[row * groups + g] : 1.0F;
                        float& a = acc[std::size_t(row) * p.outs + 8U * nb + col];
                        for (unsigned h = 0; h < passes; ++h) a = ti::step(a, ti::dot_float(d[h][lane][i]), e, own[owner][h]);
                    }
            }
        }
    for (unsigned r = 0; r < count; ++r)
        for (unsigned o = 0; o < p.outs; ++o) out[std::size_t(r) * p.outs + o] = acc[std::size_t(r) * p.outs + o];
    return out;
}

int main() {
    // Placement: the header's permuted k32 map is a bijection that A and B share, and matches the kernel loads.
    {
        bool bij = true, loads = true;
        unsigned seen = 0;
        for (unsigned p = 0; p < 32U; ++p) seen |= 1U << ti::k32_logical(p);
        bij = seen == 0xFFFFFFFFU;
        for (unsigned lane = 0; lane < 32U; ++lane)
            for (unsigned e = 0; e < 4U; ++e) {
                loads &= ti::k32_logical(ti::k32_a_phys(lane, 0U, e)) == ti::k32_a_byte(lane) + e;
                loads &= ti::k32_logical(ti::k32_a_phys(lane, 2U, e)) == ti::k32_a_byte(lane) + 4U + e;
                for (unsigned r = 0; r < 2U; ++r) loads &= ti::k32_logical(ti::k32_b_phys(lane, r, e)) == 4U * ti::k32_word(lane, r) + e;
                for (unsigned h = 0; h < 2U; ++h) loads &= 16U * h + ti::k16_phys(lane, e) == 4U * ti::k16_word(lane, h) + e &&
                                                         ti::k16_a_byte(lane, h) + e == 16U * h + ti::k16_phys(lane, e);
            }
        check(bij, "k32 logical map is a bijection of 0..31");
        check(loads, "A row loads and B words carry the logical k of their ISA slots");
    }
    // Exact group conversion over the whole reachable range.
    {
        bool exact = true;
        for (int dot = -ti::kMaxDot; dot <= ti::kMaxDot; dot += 7)
            exact &= ti::dot_float(ti::kMagicBits + static_cast<std::uint32_t>(dot)) == static_cast<float>(dot);
        for (int dot : {-ti::kMaxDot, -1, 0, 1, ti::kMaxDot})
            exact &= ti::dot_float(ti::kMagicBits + static_cast<std::uint32_t>(dot)) == static_cast<float>(dot);
        exact &= std::signbit(ti::dot_float(ti::kMagicBits)) == false;
        check(exact, "magic accumulator start converts every group dot exactly (+0 at 0)");
    }
    // Tile DAG: kernel order == contract, bitwise; and the unpermuted placement is caught.
    std::mt19937 rng(20261117U);
    unsigned cases = 0, compared = 0;
    for (bool halves : {false, true})
        for (unsigned count : {1U, 9U, 16U, 37U, 64U}) {
            const Problem p = make(rng, count, 256U, 16U, halves);
            const auto got = emulate(p, count, true, rng);
            bool same = true;
            for (unsigned r = 0; r < count; ++r)
                for (unsigned o = 0; o < p.outs; ++o) {
                    const float want = reference(p, r, o), have = got[std::size_t(r) * p.outs + o];
                    same &= std::memcmp(&want, &have, 4) == 0;
                    ++compared;
                }
            check(same, halves ? "two-scale tile equals the contract bitwise" : "one-scale tile equals the contract bitwise");
            ++cases;
        }
    {
        const Problem p = make(rng, 16U, 256U, 16U, false);
        const auto bad = emulate(p, 16U, false, rng);
        bool differs = false;
        for (unsigned r = 0; r < 16U; ++r)
            for (unsigned o = 0; o < 16U; ++o) differs |= bad[std::size_t(r) * 16U + o] != reference(p, r, o);
        check(differs, "sensitivity: unpermuted words with 8-byte loads give wrong dots");
    }
    // The step is one product rounding and one FMA rounding (not a fused triple, not two-rounded fma).
    {
        const float acc = 1.0F, dot = 3.0F, e = 1.0F / 3.0F, s = 3.0F;
        const float p = e * s;  // rounded once
        check(ti::step(acc, dot, e, s) == std::fma(dot, p, acc), "step = fma(dot, fl(e * s), acc)");
    }
    {   // 5. tile-major raster of the TURBO grids (project ffn/64 = 10 slices, down hidden/128 = 20 slices)
        bool bij = true, consecutive = true;
        for (unsigned slices : {10U, 20U})
            for (unsigned grid_x : {1U, 7U, 64U, 1601U}) {
                std::vector<unsigned char> seen(std::size_t(grid_x) * slices, 0U);
                for (unsigned by = 0; by < slices; ++by)
                    for (unsigned bx = 0; bx < grid_x; ++bx) {
                        const auto ts = strix::cuda::core::native_mma::tile_major(bx, by, grid_x, slices);
                        if (ts.tile >= grid_x || ts.slice >= slices) { bij = false; continue; }
                        seen[std::size_t(ts.tile) * slices + ts.slice] += 1U;
                        const unsigned l = bx + grid_x * by;
                        consecutive &= ts.tile == l / slices && ts.slice == l % slices;
                    }
                for (unsigned char c : seen) bij &= c == 1U;
            }
        check(bij, "tile-major map is a bijection of the TURBO grids");
        check(consecutive, "consecutive blocks are the slices of one tile");
    }
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("turbo_i8_host_test: ok (%u tiles, %u outputs bitwise)\n", cases, compared);
    return 0;
}
