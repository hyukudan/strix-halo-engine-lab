// Host oracle of the NVFP4 tile GEMM staging (no GPU). Invariants:
//  1. rec_slot (header) inverts an independent transcription of the decode record layout; repack_dst is a bijection.
//  2. The operands the kernel's fragment reads deliver (PTX m16n8k64 e2m1 block-scale semantics, lane map of the validated
//     operator) are exactly the weight / activation elements: a GEMM over the reconstructed fragments equals the double reference.
//  3. The CTA / warp / accumulator coordinates write every (row, output) exactly once, in plain and in fused gate/up (GLU) mode,
//     and GLU pairs gate and up of the same channels.
//  4. Quantiser: zero rows and padding rows are zero; per-row scale rule; cache entry size = tile size.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#include "strix/cuda_core/operand_cache_plan.hpp"
#include "tile_ref.hpp"

namespace {
int g_fail = 0;
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)
namespace T = strix::cuda::core::nvfp4::tile;
namespace N = strix::cuda::core::nvfp4;
using namespace tile_ref;

std::vector<std::uint8_t> repack(const Weight& w) {
    std::vector<std::uint8_t> t(w.rec.size(), 0);
    for (std::size_t i = 0; i < w.rec.size(); ++i) t[T::repack_dst(i, w.in)] = w.rec[i];
    return t;
}
std::uint32_t nib(const std::uint8_t* p, std::uint32_t n) { return (n & 1) ? p[n / 2] >> 4 : p[n / 2] & 15; }

// operands as the mma sees them: decoded A [tiles16 * 16][in] (value * scale) from W tile records via a_frag / a_scale
std::vector<double> fragments_a(const std::vector<std::uint8_t>& tile, std::uint32_t out, std::uint32_t in) {
    std::vector<double> A(static_cast<std::size_t>(out) * in, 0.0);
    const std::uint32_t KS = in / T::kK;
    for (std::uint32_t m = 0; m < out / 16; ++m)
        for (std::uint32_t ks = 0; ks < KS; ++ks) {
            const std::uint8_t* rec = &tile[(static_cast<std::size_t>(m) * KS + ks) * T::kWRec];
            for (std::uint32_t kb = 0; kb < 2; ++kb)
                for (std::uint32_t lane = 0; lane < 32; ++lane)
                    for (std::uint32_t q = 0; q < 4; ++q)
                        for (std::uint32_t n = 0; n < 8; ++n) {
                            const std::uint32_t g = lane >> 2, t = lane & 3, row = g + 8 * (q & 1);
                            const std::uint32_t kl = 32 * (q >> 1) + 8 * t + n;             // k inside the 64 block
                            const std::uint32_t slane = 4 * (row % 8) + row / 8;            // lane supplying row's scale
                            const std::uint8_t sc = rec[T::a_scale(kb, slane) + kl / 16];  // byte kl/16 of the u32
                            const std::uint32_t code = nib(rec + T::a_frag(kb, lane) + 4 * q, n);
                            A[static_cast<std::size_t>(16 * m + row) * in + ks * 128 + kb * 64 + kl] =
                                static_cast<double>(N::e2m1(static_cast<std::uint8_t>(code))) * N::e4m3(sc);
                        }
        }
    return A;
}
std::vector<double> fragments_b(const std::vector<std::uint8_t>& tile, std::uint32_t rows_pad, std::uint32_t in) {
    std::vector<double> B(static_cast<std::size_t>(rows_pad) * in, 0.0);
    const std::uint32_t KS = in / T::kK;
    for (std::uint32_t nt = 0; nt < rows_pad / 8; ++nt)
        for (std::uint32_t ks = 0; ks < KS; ++ks) {
            const std::uint8_t* rec = &tile[(static_cast<std::size_t>(nt) * KS + ks) * T::kXRec];
            for (std::uint32_t kb = 0; kb < 2; ++kb)
                for (std::uint32_t lane = 0; lane < 32; ++lane)
                    for (std::uint32_t q = 0; q < 2; ++q)
                        for (std::uint32_t n = 0; n < 8; ++n) {
                            const std::uint32_t g = lane >> 2, t = lane & 3, kl = 32 * q + 8 * t + n;
                            const std::uint8_t sc = rec[T::b_scale(kb, 4 * g) + kl / 16];
                            const std::uint32_t code = nib(rec + T::b_frag(kb, lane) + 4 * q, n);
                            B[static_cast<std::size_t>(8 * nt + g) * in + ks * 128 + kb * 64 + kl] =
                                static_cast<double>(N::e2m1(static_cast<std::uint8_t>(code))) * N::e4m3(sc);
                        }
        }
    return B;
}
double fdot(const std::vector<double>& A, std::uint32_t o, const std::vector<double>& B, std::uint32_t r, std::uint32_t in) {
    double s = 0;
    for (std::uint32_t k = 0; k < in; ++k) s += A[static_cast<std::size_t>(o) * in + k] * B[static_cast<std::size_t>(r) * in + k];
    return s;
}
bool close(double a, double b) { return std::fabs(a - b) <= 1e-9 * (1.0 + std::fabs(b)); }

void test_layouts(std::uint32_t out, std::uint32_t in) {
    std::map<std::size_t, int> seen;
    for (std::uint32_t o = 0; o < out; ++o) {
        for (std::uint32_t k = 0; k < in; k += 2) {
            const auto s = T::rec_slot(rec_code(o, k, in), in);
            CHECK(!s.scale && s.row == o && s.k == k);
            ++seen[T::w_code_byte(o, k, in)];
        }
        for (std::uint32_t k = 0; k < in; k += 16) {
            const auto s = T::rec_slot(rec_scale(o, k, in), in);
            CHECK(s.scale && s.row == o && s.k == k / 16);
            ++seen[T::w_scale_byte(o, k / 16, in)];
        }
    }
    CHECK(seen.size() == T::w_tile_bytes(out, in));
    for (const auto& kv : seen) CHECK(kv.second == 1 && kv.first < T::w_tile_bytes(out, in));
    std::vector<int> hit(T::w_tile_bytes(out, in), 0);
    for (std::size_t i = 0; i < hit.size(); ++i) ++hit[T::repack_dst(i, in)];
    for (int h : hit) CHECK(h == 1);
    CHECK(strix::cuda_core::operand_cache::entry_bytes(strix::cuda_core::operand_cache::Kind::e2m1_tile, out, in) == T::w_tile_bytes(out, in));
}

void test_gemm(std::uint32_t out, std::uint32_t in, std::uint32_t rows, std::uint32_t seed) {
    const Weight w = random_weight(out, in, seed);
    const Act a = quantize(random_x(rows, in, seed + 7), rows, in);
    const auto A = fragments_a(repack(w), out, in);
    const auto B = fragments_b(a.tile, a.rows_pad, in);
    // coverage + values through the kernel's coordinates
    std::vector<int> cover(static_cast<std::size_t>(rows) * out, 0);
    const std::uint32_t tiles_m = (out + T::kCtaOut - 1) / T::kCtaOut, tiles_n = a.rows_pad / T::kCtaRows;
    for (std::uint32_t tm = 0; tm < tiles_m; ++tm)
        for (std::uint32_t tn = 0; tn < tiles_n; ++tn)
            for (std::uint32_t warp = 0; warp < 8; ++warp)
                for (std::uint32_t lane = 0; lane < 32; ++lane)
                    for (std::uint32_t mt = 0; mt < 4; ++mt)
                        for (std::uint32_t nt = 0; nt < 8; ++nt)
                            for (std::uint32_t z = 0; z < 4; ++z) {
                                const std::uint32_t o = T::out_row(tm, warp & 1, mt, lane >> 2, z);
                                const std::uint32_t r = T::act_row(tn, warp >> 1, nt, lane & 3, z);
                                if (o >= out || r >= rows) continue;
                                ++cover[static_cast<std::size_t>(r) * out + o];
                                const double got = fdot(A, o, B, r, in) * w.gs * a.rs[r];
                                CHECK(close(got, dot(w, a, o, r) * w.gs * a.rs[r]));
                            }
    for (int c : cover) CHECK(c == 1);
    // padding rows of the X tile are zero operands
    for (std::uint32_t r = rows; r < a.rows_pad; ++r)
        for (std::uint32_t k = 0; k < in; ++k) CHECK(B[static_cast<std::size_t>(r) * in + k] == 0.0);
}

void test_glu(std::uint32_t ch, std::uint32_t in, std::uint32_t rows, std::uint32_t seed) {
    const Weight wg = random_weight(ch, in, seed), wu = random_weight(ch, in, seed + 1);
    const Act a = quantize(random_x(rows, in, seed + 9), rows, in);
    const auto G = fragments_a(repack(wg), ch, in), U = fragments_a(repack(wu), ch, in);
    const auto B = fragments_b(a.tile, a.rows_pad, in);
    std::vector<int> cover(static_cast<std::size_t>(rows) * ch, 0);
    const std::uint32_t tiles_m = (ch + T::kCtaOut / 2 - 1) / (T::kCtaOut / 2), tiles_n = a.rows_pad / T::kCtaRows;
    for (std::uint32_t tm = 0; tm < tiles_m; ++tm) {
        for (std::uint32_t s = 0; s < 8; s += 2) CHECK(T::glu_slot_tile(tm, s) == T::glu_slot_tile(tm, s + 1));  // gate, up: same tile
        for (std::uint32_t tn = 0; tn < tiles_n; ++tn)
            for (std::uint32_t warp = 0; warp < 8; ++warp)
                for (std::uint32_t lane = 0; lane < 32; ++lane)
                    for (std::uint32_t p = 0; p < 2; ++p)
                        for (std::uint32_t nt = 0; nt < 8; ++nt)
                            for (std::uint32_t z = 0; z < 4; ++z) {
                                const std::uint32_t c = T::glu_channel(tm, warp & 1, p, lane >> 2, z);
                                const std::uint32_t r = T::act_row(tn, warp >> 1, nt, lane & 3, z);
                                if (c >= ch || r >= rows) continue;
                                ++cover[static_cast<std::size_t>(r) * ch + c];
                                // acc[2p] reads slot wm*4+2p (gate), acc[2p+1] slot wm*4+2p+1 (up): both channel tile c / 16
                                const double gate = fdot(G, c, B, r, in) * wg.gs * a.rs[r], up = fdot(U, c, B, r, in) * wu.gs * a.rs[r];
                                const double ref = silu(dot(wg, a, c, r) * wg.gs * a.rs[r]) * (dot(wu, a, c, r) * wu.gs * a.rs[r]);
                                CHECK(close(silu(gate) * up, ref));
                            }
    }
    for (int c : cover) CHECK(c == 1);
}

void test_quantize_and_contract() {
    const std::uint32_t rows = 9, in = 256;
    const auto x = random_x(rows, in, 3);
    const Act a = quantize(x, rows, in);
    CHECK(a.rows_pad == 256 && a.rs[3] == 0.f);  // all-zero row: rs 0, zero codes
    for (std::uint32_t k = 0; k < in; ++k) CHECK(x_value(a, 3, k) == 0.0);
    for (std::uint32_t r = 0; r < rows; ++r) {
        if (r == 3) continue;
        float amax = 0;
        for (std::uint32_t k = 0; k < in; ++k) amax = std::fmax(amax, std::fabs(x[r * in + k]));
        CHECK(a.rs[r] == amax / 2688.0f);
        for (std::uint32_t q = 0; q < in / 16; ++q) CHECK(N::e4m3(a.scales[r * in / 16 + q]) <= 448.f);
        // row-max element is representable to within one e2m1 step of its block
        double err = 0;
        for (std::uint32_t k = 0; k < in; ++k) err = std::fmax(err, std::fabs(x_value(a, r, k) * a.rs[r] - x[r * in + k]) / amax);
        CHECK(err < 0.26);
    }
    T::GemmArgs g;
    CHECK(!T::valid(g));
    std::uint8_t b = 0;
    float f = 0;
    g.w = g.x = &b;
    g.rs = g.y = &f;
    g.out = 48;
    g.in = 256;
    g.rows = 40;
    g.rows_pad = 256;
    CHECK(T::valid(g));
    g.rows_pad = 128;
    CHECK(!T::valid(g));
    g.rows_pad = 256;
    g.in = 200;
    CHECK(!T::valid(g));
    g.in = 256;
    g.epi = T::Epi::glu;
    CHECK(!T::valid(g));  // GLU needs the up tile
    g.w2 = &b;
    CHECK(T::valid(g));
    g.epi = T::Epi::residual;
    CHECK(!T::valid(g));  // residual needs aux
}
}  // namespace

int main() {
    test_layouts(48, 256);
    test_layouts(32, 384);
    test_gemm(48, 256, 40, 11);    // partial weight tile, one activation tile with padding
    test_gemm(256, 384, 300, 21);  // two weight tiles x two activation tiles
    test_glu(80, 128, 20, 31);     // GLU with a partial 64-channel tile
    test_quantize_and_contract();
    if (g_fail) {
        std::fprintf(stderr, "nvfp4_tile_host: %d failures\n", g_fail);
        return 1;
    }
    std::printf("nvfp4_tile_host: PASS (layouts, fragments, coordinates, GLU, quantiser, contract)\n");
    return 0;
}
