// Host oracle of the NVFP4 tile GEMM staging (no GPU). Invariants:
//  1. rec_slot (header) inverts an independent transcription of the decode record layout; repack_dst is a bijection.
//  2. The operands the kernel's fragment reads deliver (PTX m16n8k64 e2m1 block-scale semantics, lane map of the validated
//     operator) are exactly the weight / activation elements: a GEMM over the reconstructed fragments equals the double reference.
//  3. The CTA / warp / accumulator coordinates write every (row, output) exactly once, in plain and in fused gate/up (GLU) mode,
//     and GLU pairs gate and up of the same channels.
//  4. Quantiser: zero rows and padding rows are zero; per-row scale rule; cache entry size = tile size.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include "strix/weight/operand_cache_plan.hpp"
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
    CHECK(strix::weight::operand_cache::entry_bytes(strix::weight::operand_cache::Kind::e2m1_tile, out, in) == T::w_tile_bytes(out, in));
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
    std::vector<float> ybuf(40 * 48), rsbuf(256), auxbuf(40 * 48);
    g.w = g.x = &b;
    g.rs = rsbuf.data();
    g.y = ybuf.data();
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
    // epilogue aliasing contract: aux is y (in place) or disjoint from it; rs never overlaps y
    g.aux = ybuf.data();
    CHECK(T::valid(g));
    g.aux = auxbuf.data();
    CHECK(T::valid(g));
    g.aux = ybuf.data() + 1;
    CHECK(!T::valid(g));  // partial overlap: another thread's store could land before this thread's aux load
    g.aux = ybuf.data();
    g.rs = ybuf.data() + 40 * 48 - 1;
    CHECK(!T::valid(g));  // row scales inside y
    g.rs = rsbuf.data();
    CHECK(T::valid(g));
}

// Epilogue ownership: gemm_kernel issues all of a thread's rs / aux loads before its y stores. That is bitwise equal to the
// per-element load-after-store order iff every output element of the launch is touched (read as aux, written as y) by exactly one
// (CTA, thread) and every row scale it loads is a valid row. Enumerate the kernel's own coordinate functions over every CTA, warp,
// lane and accumulator slot and check that each in-range (row, out) element is owned exactly once.
// Host model of the RMSNorm folded into the quantiser (sm120 norm_quantize_kernel = cuda_core norm_kernel<rms> arithmetic): thread t
// adds x[t + 256 i]^2 for i < nreg in ascending order (padding terms 0), 32-lane xor butterfly, the 8 warp sums added 0..7.
std::vector<float> norm_rows(const std::vector<float>& x, const std::vector<float>& gain, std::uint32_t rows, std::uint32_t in,
                             std::uint32_t nreg, float eps) {
    std::vector<float> y(x.size());
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* xr = x.data() + static_cast<std::size_t>(r) * in;
        float lane[256];
        for (std::uint32_t t = 0; t < 256; ++t) {
            float p = 0.0f;
            for (std::uint32_t i = 0; i < nreg; ++i) {
                const std::uint32_t idx = t + 256 * i;
                const float v = idx < in ? xr[idx] : 0.0f;
                const float sq = v * v;
                p = p + sq;
            }
            lane[t] = p;
        }
        float warp[8];
        for (std::uint32_t w = 0; w < 8; ++w) {
            float* l = lane + 32 * w;
            for (std::uint32_t off = 16; off > 0; off >>= 1) {
                float n[32];
                for (std::uint32_t k = 0; k < 32; ++k) n[k] = l[k] + l[k ^ off];
                std::memcpy(l, n, sizeof n);
            }
            warp[w] = l[0];
        }
        float ss = warp[0];
        for (std::uint32_t w = 1; w < 8; ++w) ss = ss + warp[w];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(in) + eps);
        for (std::uint32_t k = 0; k < in; ++k) {
            const float v = xr[k] * inv;
            y[static_cast<std::size_t>(r) * in + k] = v * gain[k];
        }
    }
    return y;
}

void test_norm_quant_fold() {
    // contract of the fused producer: width a multiple of kK, <= 8192 (32 registers x 256 threads, in * 4 bytes of shared memory)
    CHECK(T::norm_quantize_fits(5120) && T::norm_quantize_fits(T::kK) && T::norm_quantize_fits(T::kNormQuantMaxIn));
    CHECK(!T::norm_quantize_fits(0) && !T::norm_quantize_fits(5120 + 64) && !T::norm_quantize_fits(T::kNormQuantMaxIn + T::kK));
    static_assert(T::kNormQuantMaxIn * sizeof(float) <= 48u * 1024u && T::kNormQuantMaxIn <= 32u * 256u);
    // the register bucket is free: padding terms add +0, so every bucket >= ceil(in / 256) gives rmsnorm's bits, hence the same tile
    for (const std::uint32_t in : {384u, 5120u, 8192u}) {
        const std::uint32_t rows = 6, need = (in + 255) / 256;
        auto x = random_x(rows, in, 7 + in);
        std::vector<float> g(in);
        for (std::uint32_t k = 0; k < in; ++k) g[k] = 0.25f + static_cast<float>((k * 40503u) % 997) / 400.f;
        const auto y0 = norm_rows(x, g, rows, in, need, 1e-6f);
        for (const std::uint32_t nreg : {8u, 16u, 24u, 32u}) {
            if (nreg < need) continue;
            const auto y1 = norm_rows(x, g, rows, in, nreg, 1e-6f);
            CHECK(std::memcmp(y0.data(), y1.data(), y0.size() * sizeof(float)) == 0);
            const Act a0 = quantize(y0, rows, in), a1 = quantize(y1, rows, in);
            CHECK(a0.tile == a1.tile && std::memcmp(a0.rs.data(), a1.rs.data(), a0.rs.size() * sizeof(float)) == 0);
        }
        CHECK(quantize(y0, rows, in).rs[3] == 0.f);  // the all-zero row normalises to zeros: rs 0
    }
}

void test_epilogue_ownership(std::uint32_t out, std::uint32_t rows, bool glu) {
    constexpr std::uint32_t kThreads = 256;  // gemm_kernel: 8 warps, wm = warp & 1, wn = warp >> 1
    const std::uint32_t rp = T::rows_padded(rows), tiles_n = rp / T::kCtaRows;
    const std::uint32_t tiles_m = glu ? (out + T::kCtaOut / 2 - 1) / (T::kCtaOut / 2) : (out + T::kCtaOut - 1) / T::kCtaOut;
    std::vector<std::uint8_t> owner(static_cast<std::size_t>(rows) * out, 0);
    for (std::uint32_t tm = 0; tm < tiles_m; ++tm)
        for (std::uint32_t tn = 0; tn < tiles_n; ++tn)
            for (std::uint32_t tid = 0; tid < kThreads; ++tid) {
                const std::uint32_t lane = tid & 31, warp = tid >> 5, wm = warp & 1, wn = warp >> 1, g = lane >> 2, t = lane & 3;
                for (std::uint32_t nt = 0; nt < 8; ++nt)
                    for (std::uint32_t z = 0; z < 4; ++z) {
                        const std::uint32_t r = T::act_row(tn, wn, nt, t, z);
                        CHECK(r == T::act_row(tn, wn, nt, t, z & 1));  // slot z uses the hoisted scale rsv[nt][z & 1]
                        if (r >= rows) continue;
                        for (std::uint32_t k = 0; k < (glu ? 2u : 4u); ++k) {
                            const std::uint32_t o = glu ? T::glu_channel(tm, wm, k, g, z) : T::out_row(tm, wm, k, g, z);
                            if (o >= out) continue;
                            ++owner[static_cast<std::size_t>(r) * out + o];
                        }
                    }
            }
    for (std::uint8_t c : owner) CHECK(c == 1);
}

// Producer of one gemm_kernel stage: the hoisted form (stage_chunk: base + ks * step, resolved once per thread) must address the
// same bytes as the per-stage form ((t * KS + ks) * rec + off) for every chunk and stage, and one stage must fill each shared byte
// of [0, kStageWRecs * kWRec + kStageXRecs * kXRec) exactly once.
void test_stage_chunks(std::uint32_t out, std::uint32_t in, std::uint32_t rows, bool glu) {
    const std::uint32_t KS = in / T::kK, rows_pad = T::rows_padded(rows), wtiles = out / 16, ntiles = rows_pad / 8;
    const std::uint32_t tiles_m = glu ? (2 * out + T::kCtaOut - 1) / T::kCtaOut : (out + T::kCtaOut - 1) / T::kCtaOut;
    const std::uint32_t tiles_n = rows_pad / T::kCtaRows, stage = T::kStageWRecs * T::kWRec + T::kStageXRecs * T::kXRec;
    static_assert(T::kStageChunks * 16 == T::kStageWRecs * T::kWRec + T::kStageXRecs * T::kXRec, "stage chunks");
    for (std::uint32_t tm = 0; tm < tiles_m; ++tm)
        for (std::uint32_t tn = 0; tn < tiles_n; ++tn) {
            std::vector<int> cover(stage, 0);
            for (std::uint32_t c = 0; c < T::kStageChunks; ++c) {
                const T::StageChunk q = T::stage_chunk(c, tm, tn, glu, KS, wtiles, ntiles);
                std::uint32_t t, rec, off, dst, which;
                if (c < T::kStageChunksW) {  // per-stage form of the original kernel loop
                    const std::uint32_t s = c / (T::kWRec / 16);
                    off = (c % (T::kWRec / 16)) * 16;
                    t = glu ? T::glu_slot_tile(tm, s) : T::a_slot_tile(tm, s);
                    which = glu && (s & 1) ? 1U : 0U;
                    rec = T::kWRec;
                    dst = s * T::kWRec + off;
                    CHECK(q.ok == (t < wtiles));
                } else {
                    const std::uint32_t c2 = c - T::kStageChunksW, j = c2 / (T::kXRec / 16);
                    off = (c2 % (T::kXRec / 16)) * 16;
                    t = tn * T::kStageXRecs + j;
                    which = 2;
                    rec = T::kXRec;
                    dst = T::kStageWRecs * T::kWRec + j * T::kXRec + off;
                    CHECK(q.ok == (t < ntiles));
                }
                CHECK(q.which == which && q.dst == dst && q.step == rec);
                for (const std::uint32_t ks : {0U, 1U, KS - 1})
                    CHECK(q.base + static_cast<std::size_t>(ks) * q.step ==
                          (static_cast<std::size_t>(q.ok ? t : 0) * KS + ks) * rec + off);
                for (std::uint32_t b = 0; b < 16; ++b) ++cover[q.dst + b];
            }
            for (const int n : cover) CHECK(n == 1);
        }
}

// Feed default + spelling: every GemmArgs runs the bulk feed unless a caller asks for lsu (the rollback); strict parse (the feed
// is scheduling only, so valid() admits both and nothing else).
void test_feed_default() {
    static_assert(T::kDefaultFeed == T::Feed::bulk, "the FAST4 tile default feed is bulk (measured); lsu is the rollback");
    const T::GemmArgs fresh{};
    CHECK(fresh.feed == T::kDefaultFeed);
    T::Feed f = T::Feed::lsu;
    CHECK(T::parse_feed("bulk", &f) && f == T::Feed::bulk);
    CHECK(T::parse_feed("lsu", &f) && f == T::Feed::lsu);
    for (const char* bad : {"", "Bulk", "bulk ", "lsux", "tma", "on", "1"}) {
        f = T::Feed::lsu;
        CHECK(!T::parse_feed(bad, &f) && f == T::Feed::lsu);
    }
    CHECK(!T::parse_feed(nullptr, &f));
    CHECK(std::strcmp(T::feed_name(T::Feed::bulk), "bulk") == 0 && std::strcmp(T::feed_name(T::Feed::lsu), "lsu") == 0);
}

// Feed::bulk invariant: copying the kStageRecs records of a stage (stage_record: one bulk copy of `bytes` from base + ks * bytes)
// writes, for every stage ks, exactly the (source byte -> shared byte) pairs the lsu feed's chunks write (stage_chunk), each shared
// byte once; !ok records are exactly the zero-filled chunks; stage_bytes is the sum of the copied records (the mbarrier
// transaction count) and every bulk size / offset is 16-byte aligned.
void test_stage_records(std::uint32_t out, std::uint32_t in, std::uint32_t rows, bool glu) {
    const std::uint32_t KS = in / T::kK, rows_pad = T::rows_padded(rows), wtiles = out / 16, ntiles = rows_pad / 8;
    const std::uint32_t tiles_m = glu ? (2 * out + T::kCtaOut - 1) / T::kCtaOut : (out + T::kCtaOut - 1) / T::kCtaOut;
    const std::uint32_t tiles_n = rows_pad / T::kCtaRows, stage = T::kStageChunks * 16;
    struct Src {
        std::uint32_t which;
        std::size_t byte;
        bool ok;
    };
    for (std::uint32_t tm = 0; tm < tiles_m; ++tm)
        for (std::uint32_t tn = 0; tn < tiles_n; ++tn)
            for (const std::uint32_t ks : {0U, 1U, KS - 1}) {
                std::vector<Src> lsu(stage, Src{9, 0, false}), bulk(stage, Src{9, 0, false});
                for (std::uint32_t c = 0; c < T::kStageChunks; ++c) {
                    const T::StageChunk q = T::stage_chunk(c, tm, tn, glu, KS, wtiles, ntiles);
                    for (std::uint32_t b = 0; b < 16; ++b)
                        lsu[q.dst + b] = Src{q.which, q.base + static_cast<std::size_t>(ks) * q.step + b, q.ok};
                }
                std::uint32_t copied = 0;
                std::vector<int> cover(stage, 0);
                for (std::uint32_t i = 0; i < T::kStageRecs; ++i) {
                    const T::StageRecord r = T::stage_record(i, tm, tn, glu, KS, wtiles, ntiles);
                    CHECK(r.dst % 16 == 0 && r.bytes % 16 == 0 && r.base % 16 == 0 && r.dst + r.bytes <= stage);
                    if (r.ok) copied += r.bytes;
                    for (std::uint32_t b = 0; b < r.bytes; ++b) {
                        bulk[r.dst + b] = Src{r.which, r.base + static_cast<std::size_t>(ks) * r.bytes + b, r.ok};
                        ++cover[r.dst + b];
                    }
                }
                CHECK(copied == T::stage_bytes(tm, tn, glu, KS, wtiles, ntiles));
                for (std::uint32_t b = 0; b < stage; ++b) {
                    CHECK(cover[b] == 1 && bulk[b].ok == lsu[b].ok);
                    if (lsu[b].ok) CHECK(bulk[b].which == lsu[b].which && bulk[b].byte == lsu[b].byte);
                }
            }
}
}  // namespace

// GatedHeadNorm (arch/sm120/nvfp4_tile.cuh) is bitwise equal to gdn_chunk.cu chunk_norm_kernel only if, in the 256-thread row CTA,
// warp w at register step i holds exactly the 32 elements that chunk_norm's warp (position >> 5) of one head holds, in the same lane
// order (warp_sum_xor is order-sensitive), each (head, quarter) once, and the warp-uniform guard 256 i + 32 w < in equals every
// lane's idx < in. Checked for every register bucket the launcher can pick.
void test_gated_norm_fold_mapping() {
    static_assert(2 * T::kGatedNormHead == 256 && T::kK % T::kGatedNormHead == 0);
    for (const std::uint32_t in : {384u, 6144u, 8192u}) {
        CHECK(T::norm_quantize_fits(in) && in % T::kGatedNormHead == 0);
        const std::uint32_t need = (in + 255) / 256, heads = in / T::kGatedNormHead;
        for (const std::uint32_t nreg : {8u, 16u, 24u, 32u}) {
            if (nreg < need) continue;
            std::vector<int> seen(static_cast<std::size_t>(heads) * 4, 0);
            for (std::uint32_t i = 0; i < nreg; ++i)
                for (std::uint32_t w = 0; w < 8; ++w) {
                    const bool warp_live = 256 * i + 32 * w < in;
                    const std::uint32_t h = 2 * i + (w >> 2), q = w & 3;
                    if (warp_live) ++seen[static_cast<std::size_t>(h) * 4 + q];
                    for (std::uint32_t lane = 0; lane < 32; ++lane) {
                        const std::uint32_t idx = 256 * i + 32 * w + lane;
                        CHECK((idx < in) == warp_live);
                        if (idx >= in) continue;
                        const std::uint32_t pos = idx % T::kGatedNormHead;  // chunk_norm thread of this element
                        CHECK(idx / T::kGatedNormHead == h && (pos >> 5) == q && (pos & 31) == lane);
                    }
                }
            for (const int s : seen) CHECK(s == 1);
        }
    }
}

int main() {
    test_layouts(48, 256);
    test_layouts(32, 384);
    test_gemm(48, 256, 40, 11);    // partial weight tile, one activation tile with padding
    test_gemm(256, 384, 300, 21);  // two weight tiles x two activation tiles
    test_glu(80, 128, 20, 31);     // GLU with a partial 64-channel tile
    test_quantize_and_contract();
    test_stage_chunks(48, 256, 40, false);    // partial weight tile (wtiles 3 < 8 slots), padded rows
    test_stage_chunks(256, 384, 300, false);  // two weight tiles x two activation tiles
    test_stage_chunks(80, 128, 20, true);     // GLU, partial 64-channel tile (w2 slots)
    test_feed_default();
    test_stage_records(48, 256, 40, false);    // bulk feed: partial weight tile (!ok records), padded rows
    test_stage_records(256, 384, 300, false);  // bulk feed: two weight tiles x two activation tiles, KS 3
    test_stage_records(80, 128, 20, true);     // bulk feed: GLU, partial 64-channel tile, KS 1
    test_stage_records(1040, 640, 520, true);  // bulk feed: GLU, 17 x 64-channel tiles (last partial), three activation tiles
    test_epilogue_ownership(48, 40, false);     // partial weight tile, padded rows
    test_epilogue_ownership(256, 300, false);   // two weight tiles x two activation tiles
    test_epilogue_ownership(80, 20, true);      // GLU, partial 64-channel tile
    test_epilogue_ownership(5120, 4096, false); // engine residual shape (down / o)
    test_norm_quant_fold();
    test_gated_norm_fold_mapping();
    if (g_fail) {
        std::fprintf(stderr, "nvfp4_tile_host: %d failures\n", g_fail);
        return 1;
    }
    std::printf("nvfp4_tile_host: PASS (layouts, fragments, coordinates, GLU, quantiser, contract, stage producer, bulk stage records, epilogue ownership, norm fold, gated norm fold)\n");
    return 0;
}
