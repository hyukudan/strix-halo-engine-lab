// Kernel bench of the shared dense flash prefill core (src/attn_core/dense_flash_prefill.hpp) on the shapes the engines
// launch, synthetic data, no model. Per case: one warm-up launch, then REPS launches timed with hip events (median ms),
// TFLOP/s of the visited work (QK + PV over the visible keys) and a 64-bit FNV digest of the output bits (two builds of the
// core with the same digest per case produce the same bits per row).
//   mha      MHA sliding form: D 256, bf16, one head x 64 positions per work-group, 64 heads, window 512 (Hy4-flash shape)
//   g4s      Gemma4 E4B sliding layer form: D 256, f16, GQA group 4 (8 waves), window 512
//   g4g      Gemma4 E4B global layer form: D 512, f16, GQA group 4 (16 waves), full causal
//   qw       Qwen3.5 27B wide-window form (dense_decode_flash.hpp <256, 6>): D 256, bf16, group 6 (12 waves)
//   v80 v64  vision encoder form: D 80 / 64, f16 two-term (hi + lo) operands, 64 positions, 1024 rows from position
//            4095 (each row sees 4096-5119 keys, the size of a bidirectional encoder row)
//   ikl ikf  one-slice lane-score form: D 128, bf16, GQA group 3 (12 waves), 64 positions, 8192 rows from position 0,
//            window 1024 (ikl) / full causal (ikf)
//   ibl ibf  the same with a relative bias projected inside the flash (width 16, extent 1024)
//   o80 o64  one-term FP16 vision forms on a causal stand-in (1024 rows from position 4095)
//   b80 b64  the vision encoder's own one-term geometry (src/vision/encoder.hip BidirPolicy): bidirectional over n
//            keys (row r at logical position n - 1 + r, window n, logical key p backed by key p % n, mask:: bidir
//            functions), its scales 1/sqrt(80) and 1/8, n off the 64-row and 16-key tiles (1000, 1031, 37 rows)
//   b64-s    the 37-row tail of b64 alone (own selector)
//   b72 b72-s  the encoder's D72 heads: padded to 80 (the last eight Q/K/V components zero), scale 1/sqrt(72), n 1013 / 37
//   xp       pinned rescale-boundary cases of the lane-score form (Special::pin_*): deterministic Q/K/V, the expected
//            output bits asserted literally in addition to the cross-build digest
// Usage: strix-dense-flash-bench [REPS] [CASE] [nocap]   (CASE: one name above; default all; nocap: a core header
//        without the rescale probe, i.e. the incumbent comparison build, skips the capture assertions)
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "attn_core/dense_flash_prefill.hpp"
#include "attn_core/masks.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace dense = strix::attn_core::dense;
namespace tile = strix::attn_core::tile;

struct BArgs {
    const float* q;
    float* out;
    const std::uint16_t *k, *vt;
    unsigned* failure;
    unsigned rows, p0, k0, window, vstride, heads, kv_heads;
    const float* feat = nullptr;          // projected relative bias (Rel policies only): [row][head][16] FP32
    const std::uint16_t* rw = nullptr;    // [16][extent] BF16
    unsigned extent = 0U;
    unsigned* probe = nullptr;            // rescale capture (Probed policies only)
};
template <unsigned D, unsigned G, unsigned Pos, tile::Operand Op>
struct Common {
    using Args = BArgs;
    using Shape = dense::Shape<D, G, (D >= 128U ? D / 128U : 1U), Pos>;
    static constexpr float kScaleLog2 = (D == 512U ? 0.044194173824159220F : 0.0625F) * 1.4426950408889634F;
    static constexpr tile::Operand kOp = Op;
    __device__ static bool valid(const Args& a) { return a.rows != 0U && a.p0 >= a.k0; }
    __host__ __device__ static unsigned rows(const Args& a) { return a.rows; }
    __device__ static unsigned position(const Args& a) { return a.p0; }
    __device__ static unsigned window(const Args& a) { return a.window; }
    __device__ static const float* query(const Args& a, unsigned row, unsigned h) { return a.q + (std::size_t(row) * a.heads + h) * D; }
    __device__ static float* out(const Args& a, unsigned row, unsigned h) { return a.out + (std::size_t(row) * a.heads + h) * D; }
    __device__ static unsigned* failure(const Args& a) { return a.failure; }
};
template <unsigned D, unsigned G, unsigned Pos, tile::Operand Op>
struct Classic : Common<D, G, Pos, Op> {
    using Args = BArgs;
    __device__ static const std::uint16_t* key(const Args& a, unsigned p, unsigned h) { return a.k + (std::size_t(p - a.k0) * a.kv_heads + h) * D; }
    __device__ static const std::uint16_t* value_t(const Args& a, unsigned h, unsigned d) { return a.vt + std::size_t(h * D + d) * a.vstride - a.k0; }
};
template <unsigned D>  // two-term operands (lo terms = the same arrays: timing only)
struct Classic2 : Classic<D, 1U, 64U, tile::Operand::f16> {
    using Args = BArgs;
    using Shape = dense::Shape<D, 1U, 1U, 64U>;
    static constexpr unsigned kTerms = 2U;
    __device__ static const std::uint16_t* key_lo(const Args& a, unsigned p, unsigned h) { return a.k + (std::size_t(p - a.k0) * a.kv_heads + h) * D; }
    __device__ static const std::uint16_t* value_t_lo(const Args& a, unsigned h, unsigned d) { return a.vt + std::size_t(h * D + d) * a.vstride - a.k0; }
};
// One-term vision encoder policy: its scale (src/vision/encoder.hip BidirPolicy<D, DP, 1>; the bench's Classic uses 1/16
// for every D < 512). D 72 runs padded to DP 80.
template <unsigned D, unsigned DP = D>
struct Vision1 : Classic<DP, 1U, 64U, tile::Operand::f16> {
    using Args = BArgs;
    static constexpr float kScaleLog2 =
        (D == 64U ? 0.125F : D == 72U ? 0.11785113019775792F : 0.11180339887498948F) * 1.44269504088896341F;
    static_assert(D == 64U || D == 72U || D == 80U);
    static_assert(DP == D || (D == 72U && DP == 80U));
};
template <unsigned D, unsigned G>  // one-slice form with the relative bias projected in the flash
struct Rel : Classic<D, G, 64U, tile::Operand::bf16> {
    using Args = BArgs;
    static constexpr unsigned kBiasWidth = 16U, kBiasMaxExtent = 1024U;
    __device__ static strix::attn_core::RelativeBiasProjection bias_projection(const Args& a) {
        return {a.feat, a.rw, a.heads, a.extent};
    }
};
// Test-only capture of the lane-score rescale (dense_flash_prefill.hpp DenseRescaleProbe): per wave, key tile (first
// kProbeTiles), fragment and lane the kernel stores the visit marker, the wave's rescale vote, this lane's alpha and the
// 8 accumulator values it saw: incoming, the operand handed to the matrix unit, and the unconditional multiply.
constexpr unsigned kProbeTiles = 6U, kProbeWords = 27U;
template <class B> struct Probed : B {
    using Unprobed = B;  // the production policy: every timed pass and the nocap arm run this one
    static constexpr bool kRescaleProbe = true;
    __device__ static void probe(const BArgs& a, unsigned t, unsigned f, bool rescale, float alpha, const tile::Acc& in,
                                 const tile::Acc& post, const tile::Acc& incumbent) {
        if (t >= kProbeTiles) return;
        const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5;
        const std::size_t w = (std::size_t(blockIdx.y) * gridDim.x + blockIdx.x) * B::Shape::waves + wave;
        unsigned* r = a.probe + (((w * kProbeTiles + t) * B::Shape::frags + f) * 32U + lane) * kProbeWords;
        r[0] = 1U;
        r[1] = rescale ? 1U : 0U;
        r[2] = __builtin_bit_cast(unsigned, alpha);
        for (unsigned e = 0; e < 8U; ++e) {
            r[3U + e] = __builtin_bit_cast(unsigned, in[e]);
            r[11U + e] = __builtin_bit_cast(unsigned, post[e]);
            r[19U + e] = __builtin_bit_cast(unsigned, incumbent[e]);
        }
    }
};
template <class P> struct UnprobedOf { using type = P; };
template <class P> requires requires { typename P::Unprobed; } struct UnprobedOf<P> { using type = typename P::Unprobed; };
constexpr unsigned kN = 192U;  // MHA case: own columns per head before the shared 64

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
std::uint16_t round_op(float x, bool f16) {
    if (f16) {
        const _Float16 h = static_cast<_Float16>(x);
        std::uint16_t b;
        std::memcpy(&b, &h, 2);
        return b;
    }
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}
template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ok(hipMalloc(&d, h.size() * sizeof(T)), "malloc");
    ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}

// Exceptional operand values (review r17: bit identity of the lane-score form against the incumbent kernel):
//   nan       V holds NaN operands of several payloads, quiet and signalling encodings, both signs;
//   overflow  the first key tile's V is the largest finite operand (the FP32 PV sum overflows to Inf), a later key
//             dominates the scores (alpha underflows to 0: Inf * 0 = NaN accumulators) and the following tiles keep
//             that maximum (alpha exactly 1 across the wave: the shortcut's case);
//   tiny      V holds operands whose products are FP32 subnormals, and signed zeros.
// Pinned rescale-boundary cases (deterministic data, expected output asserted literally; apply_pin / check_pin):
// Q is 1.0 everywhere and K is 0 except one dominating key, so every row of a wave sees the same score sequence: tile 0
// (keys 0-15) alpha 0 (maximum from -Inf), tile 1 alpha exactly 1 for the whole wave (rescale skipped), tile 2 (key 40)
// the maximum jumps (alpha 0, or 2^-115.4), tiles 3.. alpha exactly 1 for the whole wave again. V is 0 except tile 0:
//   pin_ovf_nan   BF16: V = +-largest finite, the FP32 PV sum overflows to +-Inf (tile 0), survives tile 1 at alpha 1, meets
//                 alpha 0 (Inf * 0) and is a live NaN at every later alpha 1: every output is a quiet NaN, failure bit 1;
//   pin_inf_nan   the same with V = +-Inf operands (reachable in FP16 too);
//   pin_ovf_hold  BF16 without the dominating key: the accumulator holds +-Inf at every alpha 1: outputs +-Inf bits;
//   pin_inf_hold  the same with V = +-Inf operands;
//   pin_sub       V = a tiny operand (FP16 2^-24, BF16 2^-30): tile 0 accumulates 16 V exactly, tile 2 rescales it by
//                 2^-115.4 (the multiply yields an FP32 subnormal), tiles 3.. keep alpha 1: the matrix accumulate that
//                 follows the multiply flushes the subnormal, the output is +0 (a kernel that never rescales outputs
//                 16 V, nonzero), so no FP32 subnormal accumulator reaches the next boundary;
//   pin_subop     BF16 without the dominating key and V = the smallest BF16 subnormal: the PV accumulator of tile 0 would
//                 be the FP32 subnormal 16 * 2^-133 at every alpha 1; the matrix unit gives +0 (same bits as the
//                 incumbent kernel), asserted.
// Ordinary cases (none) must leave the failure word 0; exceptional cases report it for the cross-build comparison.
enum class Special : unsigned { none, nan, overflow, tiny, pin_ovf_nan, pin_inf_nan, pin_ovf_hold, pin_inf_hold, pin_sub, pin_subop };

struct Case {
    const char* name;
    unsigned D, heads, kv_heads, rows, p0, window;
    bool f16;
    Special special = Special::none;
    bool bidir = false;  // rows = n keys, p0 = bidir_position(n), window = bidir_window(n), key p holds key p % n
    unsigned valid_dim = 0U;  // components >= valid_dim of Q / K / V are zero (the encoder's D72 padded to 80); 0 = all
};

bool g_flag_failed = false;
bool g_capture = true;  // false: the header under test has no rescale probe (the incumbent comparison build)

void apply_special(const Case& c, unsigned keys, unsigned KH, std::vector<std::uint16_t>& kfull,
                   std::vector<std::uint16_t>& vrows) {
    const unsigned D = c.D;
    const auto v_at = [&](unsigned p, unsigned h, unsigned d) -> std::uint16_t& {
        return vrows[(std::size_t(p) * KH + h) * D + d];
    };
    if (c.special == Special::nan) {
        const std::uint16_t bf[5] = {0x7FC1U, 0x7F81U, 0xFFC5U, 0xFF8AU, 0x7FFFU};  // quiet, signalling, both signs
        const std::uint16_t hf[5] = {0x7E01U, 0x7C01U, 0xFE05U, 0xFC0AU, 0x7FFFU};
        for (unsigned p = 5U; p < keys; p += 97U)
            for (unsigned h = 0; h < KH; ++h)
                for (unsigned d = 3U; d < D; d += 7U) v_at(p, h, d) = (c.f16 ? hf : bf)[(p + d) % 5U];
    } else if (c.special == Special::overflow) {
        const std::uint16_t big = c.f16 ? std::uint16_t{0x7BFFU} : std::uint16_t{0x7F7FU};
        const std::uint16_t nbig = static_cast<std::uint16_t>(big | 0x8000U);
        for (unsigned p = 0; p < 16U && p < keys; ++p)
            for (unsigned h = 0; h < KH; ++h)
                for (unsigned d = 0; d < D; ++d) v_at(p, h, d) = d % 3U == 2U ? nbig : big;
        const std::uint16_t dom = round_op(30.0F, c.f16);  // a dominating key at position 40
        if (keys > 40U)
            for (unsigned h = 0; h < KH; ++h)
                for (unsigned d = 0; d < D; ++d) kfull[(std::size_t(40U) * KH + h) * D + d] = dom;
    } else if (c.special == Special::tiny) {
        for (unsigned p = 0; p < keys; ++p)
            for (unsigned h = 0; h < KH; ++h)
                for (unsigned d = 0; d < D; ++d)
                    if ((p + 3U * d) % 5U == 0U)
                        v_at(p, h, d) = (p + d) % 4U == 0U ? std::uint16_t{0x8000U}
                                                           : static_cast<std::uint16_t>(1U + (p * 7U + d) % 127U);
    }
}

constexpr unsigned kPinRowsFrom = 64U;  // rows at or after this position: every wave of them sees tiles 0..2 whole

constexpr bool is_pin(Special s) { return s >= Special::pin_ovf_nan; }

void apply_pin(const Case& c, unsigned keys, unsigned KH, std::vector<float>& q, std::vector<std::uint16_t>& kfull,
               std::vector<std::uint16_t>& vrows) {
    const unsigned D = c.D;
    std::fill(q.begin(), q.end(), 1.0F);
    std::fill(kfull.begin(), kfull.end(), std::uint16_t{0U});
    std::fill(vrows.begin(), vrows.end(), std::uint16_t{0U});
    const bool ovf = c.special == Special::pin_ovf_nan || c.special == Special::pin_ovf_hold;
    const bool inf = c.special == Special::pin_inf_nan || c.special == Special::pin_inf_hold;
    const bool hold = c.special == Special::pin_ovf_hold || c.special == Special::pin_inf_hold || c.special == Special::pin_subop;
    const std::uint16_t mag = ovf ? (c.f16 ? std::uint16_t{0x7BFFU} : std::uint16_t{0x7F7FU})
                                  : (c.f16 ? std::uint16_t{0x7C00U} : std::uint16_t{0x7F80U});
    const std::uint16_t tiny = c.f16 || c.special == Special::pin_subop ? std::uint16_t{0x0001U} : round_op(0x1p-30F, false);
    for (unsigned p = 0; p < 16U; ++p)
        for (unsigned h = 0; h < KH; ++h)
            for (unsigned d = 0; d < D; ++d) {
                std::uint16_t v = tiny;
                if (ovf || inf) v = d % 3U == 2U ? static_cast<std::uint16_t>(mag | 0x8000U) : mag;
                vrows[(std::size_t(p) * KH + h) * D + d] = v;
            }
    if (hold) return;
    // Dominating key at position 40: score 1280 * scale, ~346 (alpha 0: exp2 underflows) or ~115.4 (alpha 2^-115.4).
    const float kd = c.special == Special::pin_sub ? (c.f16 ? 16.0F : 10.0F) : (c.f16 ? 48.0F : 30.0F);
    if (keys > 40U)
        for (unsigned h = 0; h < KH; ++h)
            for (unsigned d = 0; d < D; ++d) kfull[(std::size_t(40U) * KH + h) * D + d] = round_op(kd, c.f16);
}

// The literal expectation from the analysis above: every row at or after kPinRowsFrom, every head, every column.
bool check_pin(const Case& c, const std::vector<float>& host, unsigned H, unsigned failure) {
    const unsigned D = c.D;
    bool good = true;
    unsigned shown = 0U, bad_n = 0U;
    for (unsigned r = 0; r < c.rows; ++r) {
        if (c.p0 + r < kPinRowsFrom) continue;
        for (unsigned h = 0; h < H; ++h)
            for (unsigned d = 0; d < D; ++d) {
                const float x = host[(std::size_t(r) * H + h) * D + d];
                std::uint32_t b;
                std::memcpy(&b, &x, 4);
                bool ok_bits = false;
                switch (c.special) {
                case Special::pin_ovf_nan:
                case Special::pin_inf_nan: ok_bits = (b & 0x7fc00000U) == 0x7fc00000U; break;  // quiet NaN
                case Special::pin_ovf_hold:
                case Special::pin_inf_hold: ok_bits = b == (d % 3U == 2U ? 0xff800000U : 0x7f800000U); break;
                case Special::pin_sub:
                case Special::pin_subop: ok_bits = b == 0U; break;
                default: break;
                }
                if (!ok_bits) {
                    ++bad_n;
                    good = false;
                    if (shown++ < 3U)
                        std::printf("FAIL pin %s row %u head %u col %u: bits %08x (expected %s)\n", c.name, r, h, d, b,
                                    c.special == Special::pin_sub || c.special == Special::pin_subop ? "+0" : "quiet NaN / signed Inf");
                }
            }
    }
    const bool want_failure = c.special != Special::pin_sub && c.special != Special::pin_subop;
    if ((failure & 1U) != (want_failure ? 1U : 0U)) {
        std::printf("FAIL pin %s: failure word %u, bit 1 expected %s\n", c.name, failure, want_failure ? "set" : "clear");
        good = false;
    }
    if (bad_n) std::printf("FAIL pin %s: %u outputs differ from the literal expectation\n", c.name, bad_n);
    return good;
}

// Capture assertions of a pinned case (Probed policy), rows at or after kPinRowsFrom only. Wave-wide sequence per key tile
// t (see the table above): vote 1 and alpha 0 at tile 0, then the case's own boundary tiles, vote 0 and alpha exactly 1 in
// every lane everywhere else. For every record the operand handed to the matrix unit equals the unconditional multiply bit
// for bit (the identity the skip relies on, raw, before any WMMA or division can hide it), and:
//   Inf cases    tile 0 incoming +0; tiles 1.. incoming = +-Inf by column (d % 3 == 2 negative) until the Inf * 0 multiply
//                of a dominating key's tile 2 (alpha 0), whose operand is a quiet NaN; every later incoming value is a
//                quiet NaN equal to the operand handed on;
//   sub cases    incoming at tiles 1 and 2 = 16 V exactly, the tile 2 multiply yields a nonzero FP32 subnormal, the matrix
//                accumulate flushes it: incoming +0 from tile 3 on;
//   sbo case     tile 0's PV is a subnormal sum: incoming +0 from tile 1 on (flushed by the matrix unit).
bool check_capture(const Case& c, unsigned G, unsigned W, unsigned F, unsigned positions, unsigned gx,
                   const std::vector<unsigned>& rec, std::size_t& visits) {
    const bool inf = c.special == Special::pin_ovf_nan || c.special == Special::pin_inf_nan;
    const bool hold = c.special == Special::pin_ovf_hold || c.special == Special::pin_inf_hold || c.special == Special::pin_subop;
    const bool sub = c.special == Special::pin_sub;
    const bool dom = !hold;
    const std::uint32_t sub_in = c.f16 ? 0x35800000U : 0x32800000U;  // 16 tiny V: FP16 2^-24, BF16 2^-30
    bool good = true;
    unsigned bad_n = 0U, shown = 0U;
    const auto fail = [&](unsigned bx, unsigned y, unsigned wv, unsigned t, unsigned f, unsigned lane, const char* what,
                          unsigned got) {
        good = false;
        ++bad_n;
        if (shown++ < 4U)
            std::printf("FAIL capture %s block %u kv %u wave %u tile %u frag %u lane %u: %s (bits %08x)\n", c.name, gx - 1U - bx, y,
                        wv, t, f, lane, what, got);
    };
    visits = 0U;
    for (unsigned y = 0; y < c.kv_heads; ++y)
        for (unsigned bx = 0; bx < gx; ++bx) {
            const unsigned block = gx - 1U - bx;
            if (block * positions < kPinRowsFrom) continue;
            for (unsigned wv = 0; wv < W; ++wv) {
                const std::size_t w = (std::size_t(y) * gx + bx) * W + wv;
                for (unsigned t = 0; t < kProbeTiles; ++t) {
                    const bool seen = t <= 4U * block + wv / G;  // the wave's last visible key tile
                    const auto at = [&](unsigned f, unsigned lane) {
                        return rec.data() + (((w * kProbeTiles + t) * F + f) * 32U + lane) * kProbeWords;
                    };
                    if ((at(0, 0)[0] != 0U) != seen) {
                        fail(bx, y, wv, t, 0, 0, seen ? "tile not visited" : "unexpected visit", at(0, 0)[0]);
                        continue;
                    }
                    if (!seen) continue;
                    const bool jump = dom && t == 2U;
                    const bool want_vote = t == 0U || jump;
                    unsigned any_alpha = 0U;
                    for (unsigned lane = 0; lane < 32U; ++lane) any_alpha |= at(0, lane)[2] != 0x3f800000U ? 1U : 0U;
                    for (unsigned f = 0; f < F; ++f)
                        for (unsigned lane = 0; lane < 32U; ++lane) {
                            const unsigned* r = at(f, lane);
                            ++visits;
                            if (r[0] != 1U) {
                                fail(bx, y, wv, t, f, lane, "tile not visited", r[0]);
                                continue;
                            }
                            if ((r[1] != 0U) != want_vote || (r[1] != 0U) != (any_alpha != 0U))
                                fail(bx, y, wv, t, f, lane, "rescale vote", r[1]);
                            const std::uint32_t al = r[2];
                            const bool al_ok = t == 0U ? al == 0U
                                               : jump  ? (sub ? al >= 0x00800000U && al < 0x3f800000U : al == 0U)
                                                       : al == 0x3f800000U;
                            if (!al_ok) fail(bx, y, wv, t, f, lane, "alpha", al);
                            const bool neg = (16U * f + (lane & 15U)) % 3U == 2U;
                            for (unsigned e = 0; e < 8U; ++e) {
                                const std::uint32_t in = r[3U + e], post = r[11U + e], inc = r[19U + e];
                                if (post != inc) fail(bx, y, wv, t, f, lane, "operand after the rescale differs from the multiply", post);
                                if (r[1] == 0U && post != in) fail(bx, y, wv, t, f, lane, "unit tile changed the accumulator", post);
                                const bool qnan = (in & 0x7fc00000U) == 0x7fc00000U;
                                if (t == 0U) {
                                    if (in != 0U) fail(bx, y, wv, t, f, lane, "initial accumulator not +0", in);
                                } else if (inf) {
                                    const std::uint32_t pinf = neg ? 0xff800000U : 0x7f800000U;
                                    if (t <= 2U ? in != pinf : !qnan)
                                        fail(bx, y, wv, t, f, lane, t <= 2U ? "incoming not the signed Inf" : "incoming not a quiet NaN", in);
                                    if (jump && (post & 0x7fc00000U) != 0x7fc00000U)
                                        fail(bx, y, wv, t, f, lane, "Inf * 0 is not a quiet NaN", post);
                                } else if (hold) {
                                    const std::uint32_t want = c.special == Special::pin_subop ? 0U : (neg ? 0xff800000U : 0x7f800000U);
                                    if (in != want) fail(bx, y, wv, t, f, lane, "incoming accumulator", in);
                                } else if (sub) {
                                    if (t <= 2U ? in != sub_in : in != 0U)
                                        fail(bx, y, wv, t, f, lane, t <= 2U ? "incoming not 16 V" : "subnormal not flushed", in);
                                    if (jump && !((post & 0x7f800000U) == 0U && (post & 0x7fffffffU) != 0U))
                                        fail(bx, y, wv, t, f, lane, "rescaled operand is not a nonzero subnormal", post);
                                }
                            }
                        }
                }
            }
        }
    if (bad_n) std::printf("FAIL capture %s: %u capture checks failed\n", c.name, bad_n);
    return good;
}

template <class P> void run(const Case& c, unsigned reps) {
    const unsigned D = c.D, H = c.heads, KH = c.kv_heads;
    const unsigned k0 = (c.window && c.p0 >= c.window ? c.p0 - c.window + 1U : 0U) & ~15U;
    // Production layout (src/vision/encoder.hip): K / V^T backed for whole 16-key tiles, the V^T pitch a multiple of 16
    // elements, so a V^T row starts on a 16-byte boundary at every position % 16 == 0. One-image bidirectional cases
    // back bidir_capacity(n) logical positions; the others one more tile than the last visible key.
    const unsigned keys = c.bidir ? strix::attn_core::mask::bidir_capacity(c.rows) : (c.p0 + c.rows - k0 + 31U) & ~15U;
    const unsigned vstride = keys;
    if (vstride % 16U != 0U) throw std::runtime_error("V^T pitch is not a whole tile");
    std::mt19937 g(c.rows * 131U + c.p0 + D);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::vector<float> q(std::size_t(c.rows) * H * D);
    for (auto& x : q) x = u(g);
    // Overflow case: non-negative queries, so the all +30 key at position 40 dominates every row that sees it (a later
    // maximum after the overflowing first tile: alpha 0, then tied maxima).
    if (c.special == Special::overflow)
        for (auto& x : q) x = x < 0.0F ? -x : x;
    std::vector<std::uint16_t> kfull(std::size_t(keys) * KH * D), vt(std::size_t(KH) * D * vstride, 0U);
    for (auto& x : kfull) x = round_op(u(g), c.f16);
    std::vector<std::uint16_t> vrows(std::size_t(keys) * KH * D);
    for (auto& x : vrows) x = round_op(u(g), c.f16);
    apply_special(c, keys, KH, kfull, vrows);
    if (is_pin(c.special)) apply_pin(c, keys, KH, q, kfull, vrows);
    if (c.valid_dim)  // padded heads: components >= valid_dim are zero in Q, K and V
        for (unsigned d = c.valid_dim; d < D; ++d) {
            for (std::size_t i = d; i < q.size(); i += D) q[i] = 0.0F;
            for (std::size_t i = d; i < kfull.size(); i += D) kfull[i] = 0U, vrows[i] = 0U;
        }
    if (c.bidir) {  // the encoder's wrapped storage: logical key p (up to 2 n - 2) is backed by key bidir_key(p, n)
        const unsigned n = c.rows;
        if (c.p0 != strix::attn_core::mask::bidir_position(n) || c.window != strix::attn_core::mask::bidir_window(n) ||
            k0 != 0U || keys < strix::attn_core::mask::bidir_capacity(n))
            throw std::runtime_error("bidirectional case geometry");
        const std::size_t row = std::size_t(KH) * D;
        for (unsigned p = n; p < keys; ++p) {
            const unsigned src = strix::attn_core::mask::bidir_key(p, n);
            std::copy_n(kfull.begin() + std::ptrdiff_t(src * row), row, kfull.begin() + std::ptrdiff_t(p * row));
            std::copy_n(vrows.begin() + std::ptrdiff_t(src * row), row, vrows.begin() + std::ptrdiff_t(p * row));
        }
    }
    for (unsigned p = 0; p < keys; ++p)
        for (unsigned h = 0; h < KH; ++h)
            for (unsigned d = 0; d < D; ++d) vt[std::size_t(h * D + d) * vstride + p] = vrows[(std::size_t(p) * KH + h) * D + d];
    if (D == 256U && KH == H)  // MHA case: columns 192..255 of every head = head 0's (the shared rope key of an MLA form)
        for (unsigned p = 0; p < keys; ++p)
            for (unsigned h = 1; h < H; ++h)
                for (unsigned d = kN; d < D; ++d) kfull[(std::size_t(p) * KH + h) * D + d] = kfull[std::size_t(p) * KH * D + d];
    BArgs a{};
    a.q = up(q);
    a.k = up(kfull);
    a.vt = up(vt);
    a.failure = up(std::vector<unsigned>{0U});
    a.rows = c.rows;
    a.p0 = c.p0;
    a.k0 = k0;
    a.window = c.window;
    a.vstride = vstride;
    a.heads = H;
    a.kv_heads = KH;
    if constexpr (dense::DenseProjectedBias<P>) {
        std::vector<float> feat(std::size_t(c.rows) * H * 16U);
        for (auto& x : feat) x = 0.25F * u(g);
        std::vector<std::uint16_t> rw(16U * 1024U);
        for (auto& x : rw) x = round_op(0.25F * u(g), false);
        a.feat = up(feat);
        a.rw = up(rw);
        a.extent = 1024U;
    }
    constexpr bool kProbed = requires { P::kRescaleProbe; };
    const unsigned gx = (c.rows + P::Shape::positions - 1U) / P::Shape::positions;
    std::vector<unsigned> rec;
    if constexpr (kProbed) {
        if (g_capture) {
            rec.assign(std::size_t(gx) * KH * P::Shape::waves * kProbeTiles * P::Shape::frags * 32U * kProbeWords, 0U);
            a.probe = up(rec);
        }
    }
    // Timing always runs the production (unprobed) policy: no capture stores, no extra multiply, no probe callback.
    using U = typename UnprobedOf<P>::type;
    const std::size_t n = std::size_t(c.rows) * H * D;
    float* out = nullptr;
    ok(hipMalloc(&out, n * 4U), "malloc out");
    a.out = out;
    hipEvent_t e0, e1;
    ok(hipEventCreate(&e0), "event");
    ok(hipEventCreate(&e1), "event");
    dense::launch<U>(a, H, nullptr);
    ok(hipDeviceSynchronize(), "warm-up");
    std::vector<float> ms(reps);
    for (unsigned r = 0; r < reps; ++r) {
        ok(hipEventRecord(e0, nullptr), "record");
        dense::launch<U>(a, H, nullptr);
        ok(hipEventRecord(e1, nullptr), "record");
        ok(hipEventSynchronize(e1), "sync");
        ok(hipEventElapsedTime(&ms[r], e0, e1), "elapsed");
    }
    std::vector<float> host(n);
    ok(hipMemcpy(host.data(), out, n * 4U, hipMemcpyDeviceToHost), "download");
    unsigned failure = 0U;
    ok(hipMemcpy(&failure, a.failure, 4U, hipMemcpyDeviceToHost), "failure");
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (const float x : host) {
        std::uint32_t b;
        std::memcpy(&b, &x, 4);
        for (unsigned i = 0; i < 4U; ++i) h = (h ^ ((b >> (8U * i)) & 0xffU)) * 0x100000001b3ULL;
    }
    double work = 0.0;  // visible (row, key) pairs x heads x D x 2 products x 2 flop
    for (unsigned r = 0; r < c.rows; ++r) {
        const unsigned pos = c.p0 + r, lo = c.window && pos + 1U >= c.window ? pos + 1U - c.window : 0U;
        work += double(pos - lo + 1U);
    }
    work *= double(H) * D * 4.0;
    std::sort(ms.begin(), ms.end());
    const float med = ms[reps / 2U];
    std::printf("case %-5s D %u heads %u/%u rows %u p0 %u window %u: median %.3f ms min %.3f max %.3f  %.2f TF/s  digest %016llx failure %u\n",
                c.name, D, H, KH, c.rows, c.p0, c.window, double(med), double(ms.front()), double(ms.back()), work / (double(med) * 1e9),
                static_cast<unsigned long long>(h), failure);
    if (c.special == Special::none && failure != 0U) {
        std::printf("FAIL flag case %s: failure %u on ordinary data\n", c.name, failure);
        g_flag_failed = true;
    }
    if (is_pin(c.special) && !check_pin(c, host, H, failure)) g_flag_failed = true;
    if constexpr (kProbed) {
        if (g_capture && is_pin(c.special)) {
            // Separate untimed instrumented run; the pin check above judged the unprobed output.
            dense::launch<P>(a, H, nullptr);
            ok(hipDeviceSynchronize(), "capture run");
            ok(hipMemcpy(rec.data(), a.probe, rec.size() * 4U, hipMemcpyDeviceToHost), "download probe");
            std::size_t visits = 0U;
            const bool good = check_capture(c, P::Shape::group, P::Shape::waves, P::Shape::frags, P::Shape::positions, gx, rec, visits);
            std::printf("capture %-5s records %zu: %s\n", c.name, visits, good ? "ok" : "FAIL");
            if (!good || visits == 0U) g_flag_failed = true;
        }
    }
    for (const void* p : {static_cast<const void*>(a.q), static_cast<const void*>(a.k), static_cast<const void*>(a.vt),
                          static_cast<const void*>(a.failure), static_cast<const void*>(out), static_cast<const void*>(a.feat),
                          static_cast<const void*>(a.rw), static_cast<const void*>(a.probe)})
        if (p) ok(hipFree(const_cast<void*>(p)), "free");
    ok(hipEventDestroy(e0), "event");
    ok(hipEventDestroy(e1), "event");
}

}  // namespace

int main(int argc, char** argv) {
    const unsigned reps = argc > 1 ? unsigned(std::atoi(argv[1])) : 7U;
    const std::string only = argc > 2 ? argv[2] : "";
    g_capture = !(argc > 3 && std::string(argv[3]) == "nocap");  // nocap: a core header without the rescale probe
    const auto want = [&](const char* n) { return only.empty() || only == n; };
    try {
        // MHA sliding layer-window: 2048 rows, 64 heads, window 512, the halo keys before the rows.
        if (want("mha")) run<Classic<256U, 1U, 64U, tile::Operand::bf16>>({"mha", 256U, 64U, 64U, 2048U, 2048U, 512U, false}, reps);
        if (want("g4s")) run<Classic<256U, 4U, 16U, tile::Operand::f16>>({"g4s", 256U, 8U, 2U, 2048U, 2048U, 512U, true}, reps);
        if (want("g4g")) run<Classic<512U, 4U, 16U, tile::Operand::f16>>({"g4g", 512U, 8U, 2U, 2048U, 2048U, 0U, true}, reps);
        if (want("qw")) run<Classic<256U, 6U, 16U, tile::Operand::bf16>>({"qw", 256U, 24U, 4U, 2048U, 2048U, 0U, false}, reps);
        if (want("v80")) run<Classic2<80U>>({"v80", 80U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
        if (want("v64")) run<Classic2<64U>>({"v64", 64U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
        if (want("ikl")) run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikl", 128U, 24U, 8U, 8192U, 0U, 1024U, false}, reps);
        if (want("ikf")) run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikf", 128U, 24U, 8U, 8192U, 0U, 0U, false}, reps);
        if (want("ibl")) run<Rel<128U, 3U>>({"ibl", 128U, 24U, 8U, 8192U, 0U, 1024U, false}, reps);
        if (want("ibf")) run<Rel<128U, 3U>>({"ibf", 128U, 24U, 8U, 8192U, 0U, 0U, false}, reps);
        // One-term FP16 vision forms on a causal stand-in (the FAST route's lane-score body).
        if (want("o80")) run<Classic<80U, 1U, 64U, tile::Operand::f16>>({"o80", 80U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
        if (want("o64")) run<Classic<64U, 1U, 64U, tile::Operand::f16>>({"o64", 64U, 16U, 16U, 1024U, 4095U, 0U, true}, reps);
        // The encoder's bidirectional one-term geometry and scales; n off the row and key tiles.
        if (want("b80")) run<Vision1<80U>>({"b80", 80U, 16U, 16U, 1000U, 999U, 1000U, true, Special::none, true}, reps);
        if (want("b64")) run<Vision1<64U>>({"b64", 64U, 16U, 16U, 1031U, 1030U, 1031U, true, Special::none, true}, reps);
        if (want("b64-s")) run<Vision1<64U>>({"b64-s", 64U, 16U, 16U, 37U, 36U, 37U, true, Special::none, true}, reps);
        // D72 heads padded to 80 (last eight components zero, scale 1/sqrt(72)), wrapped and off the tiles.
        if (want("b72")) run<Vision1<72U, 80U>>({"b72", 80U, 16U, 16U, 1013U, 1012U, 1013U, true, Special::none, true, 72U}, reps);
        if (want("b72-s")) run<Vision1<72U, 80U>>({"b72-s", 80U, 16U, 16U, 37U, 36U, 37U, true, Special::none, true, 72U}, reps);
        // Exceptional values on the lane-score forms (BF16 Inkling local / full / projected bias, FP16 one-term vision)
        // and on the two-term vision comparison form.
        using S = Special;
        if (want("xn")) {
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikl-n", 128U, 24U, 8U, 2048U, 0U, 1024U, false, S::nan}, reps);
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikf-n", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::nan}, reps);
            run<Rel<128U, 3U>>({"ibf-n", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::nan}, reps);
            run<Classic<80U, 1U, 64U, tile::Operand::f16>>({"o80-n", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::nan}, reps);
            run<Classic2<80U>>({"v80-n", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::nan}, reps);
        }
        if (want("xo")) {
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikl-o", 128U, 24U, 8U, 2048U, 0U, 1024U, false, S::overflow}, reps);
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikf-o", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::overflow}, reps);
            run<Rel<128U, 3U>>({"ibf-o", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::overflow}, reps);
            run<Classic<80U, 1U, 64U, tile::Operand::f16>>({"o80-o", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::overflow}, reps);
            run<Classic2<80U>>({"v80-o", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::overflow}, reps);
        }
        if (want("xt")) {
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikl-t", 128U, 24U, 8U, 2048U, 0U, 1024U, false, S::tiny}, reps);
            run<Classic<128U, 3U, 64U, tile::Operand::bf16>>({"ikf-t", 128U, 24U, 8U, 2048U, 0U, 0U, false, S::tiny}, reps);
            run<Classic<80U, 1U, 64U, tile::Operand::f16>>({"o80-t", 80U, 16U, 16U, 512U, 1023U, 0U, true, S::tiny}, reps);
        }
        // Pinned rescale-boundary cases (deterministic data, literal expected bits): BF16 Inkling form, FP16 vision form.
        if (want("xp")) {
            using B = Probed<Classic<128U, 3U, 64U, tile::Operand::bf16>>;
            using F = Probed<Classic<80U, 1U, 64U, tile::Operand::f16>>;
            run<B>({"pb-ovn", 128U, 24U, 8U, 256U, 0U, 0U, false, S::pin_ovf_nan}, reps);
            run<B>({"pb-inn", 128U, 24U, 8U, 256U, 0U, 0U, false, S::pin_inf_nan}, reps);
            run<B>({"pb-ovh", 128U, 24U, 8U, 256U, 0U, 0U, false, S::pin_ovf_hold}, reps);
            run<B>({"pb-inh", 128U, 24U, 8U, 256U, 0U, 0U, false, S::pin_inf_hold}, reps);
            run<B>({"pb-sub", 128U, 24U, 8U, 256U, 0U, 0U, false, S::pin_sub}, reps);
            run<F>({"pf-inn", 80U, 16U, 16U, 256U, 0U, 0U, true, S::pin_inf_nan}, reps);
            run<F>({"pf-inh", 80U, 16U, 16U, 256U, 0U, 0U, true, S::pin_inf_hold}, reps);
            run<B>({"pb-sbo", 128U, 24U, 8U, 256U, 0U, 0U, false, S::pin_subop}, reps);
            run<F>({"pf-sub", 80U, 16U, 16U, 256U, 0U, 0U, true, S::pin_sub}, reps);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    return g_flag_failed ? 1 : 0;
}
