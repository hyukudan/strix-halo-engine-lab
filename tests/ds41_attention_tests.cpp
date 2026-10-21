// CPU tests of the deepseek41 decode attention references: scalar formats, exp, the layer plan, caches written by
// window/compress, top-k selection against a sort, attention against a double-precision evaluation of the same
// decoded keys, and (STRIX_DS41_ATTN_CASE=DIR) a small case exported from the official reference model.
#include "strix/ds41_attention.hpp"
#include "strix/deepseek41_model.hpp"
#include "strix/ds41_formats.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace a = strix::ds41::attn;
namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what.c_str()); }
}

void scalar_formats() {
    // E4M3: every finite code round-trips; ties to even; saturation
    for (unsigned c = 0; c < 256; ++c) {
        if ((c & 0x7FU) == 0x7FU) continue;
        const float v = a::e4m3_value(static_cast<std::uint8_t>(c));
        const std::uint8_t back = a::e4m3_encode(v);
        expect(back == c || (v == 0.0F && (back & 0x7FU) == 0U), "e4m3 round trip " + std::to_string(c));
    }
    expect(a::e4m3_encode(1.0625F) == 0x38U, "e4m3 tie to even (1.0625 -> 1)");
    expect(a::e4m3_encode(1.1875F) == 0x3AU, "e4m3 tie to even (1.1875 -> 1.25)");
    expect(a::e4m3_encode(1000.0F) == 0x7EU && a::e4m3_encode(-1000.0F) == 0xFEU, "e4m3 saturation");
    expect(a::e4m3_value(0x01U) == 0.001953125F, "e4m3 smallest subnormal");
    // E2M1 thresholds (ties to even)
    const float probes[] = {0.25F, 0.26F, 0.75F, 1.25F, 1.75F, 2.5F, 3.5F, 5.0F, 7.0F};
    const unsigned codes[] = {0, 1, 2, 2, 4, 4, 6, 6, 7};
    for (unsigned i = 0; i < 9; ++i) expect(a::e2m1_encode(probes[i]) == codes[i], "e2m1 " + std::to_string(probes[i]));
    for (unsigned c = 0; c < 16; ++c) expect(a::e2m1_int2(static_cast<std::uint8_t>(c)) == static_cast<int>(2.0F * a::e2m1_value(static_cast<std::uint8_t>(c))), "e2m1 int2");
    expect(a::log2_ceil(1.0F) == 0 && a::log2_ceil(1.5F) == 1 && a::log2_ceil(0.25F) == -2, "log2_ceil");
    // exp: within 2 ulp of the libm value over the softmax range
    double worst = 0;
    for (int i = 0; i <= 200000; ++i) {
        const float x = -87.0F + 175.0F * static_cast<float>(i) / 200000.0F;
        const double ref = std::exp(static_cast<double>(x));
        const float got = a::exp_f(x);
        const double ulp = std::ldexp(1.0, std::ilogb(ref) - 23);
        worst = std::max(worst, std::fabs(got - ref) / ulp);
    }
    std::printf("exp_f: worst %.2f ulp over [-87, 88]\n", worst);
    expect(worst <= 2.0, "exp_f accuracy");
}

#include "deepseek41_real_config.inc"

// The shared codecs against independent definitions (the formats module's earlier ones): every decode code bitwise,
// the E4M3 encoder against a nearest-value search over the 127 magnitudes (ties to the even code) on a sweep of the
// float bit patterns plus every midpoint and its neighbours, the E2M1 encoder likewise.
float oracle_e4m3(std::uint8_t code) {
    const bool negative = (code & 0x80U) != 0U;
    const unsigned exponent = (code >> 3U) & 15U, mantissa = code & 7U;
    float magnitude;
    if (exponent == 15U && mantissa == 7U) magnitude = std::numeric_limits<float>::quiet_NaN();
    else if (exponent == 0U) magnitude = std::ldexp(static_cast<float>(mantissa), -9);
    else magnitude = std::ldexp(static_cast<float>(8U + mantissa), static_cast<int>(exponent) - 10);
    return negative ? -magnitude : magnitude;
}
std::uint8_t oracle_e4m3_encode(float value) {
    if (std::isnan(value)) return 0x7fU;
    const std::uint8_t sign = std::signbit(value) ? 0x80U : 0x00U;
    const double magnitude = std::fabs(static_cast<double>(value));
    if (magnitude >= 448.0) return static_cast<std::uint8_t>(sign | 0x7eU);
    unsigned best = 0;
    for (unsigned c = 1; c < 0x7fU; ++c) {
        const double x = std::fabs(static_cast<double>(oracle_e4m3(static_cast<std::uint8_t>(c))) - magnitude);
        const double y = std::fabs(static_cast<double>(oracle_e4m3(static_cast<std::uint8_t>(best))) - magnitude);
        if (x < y || (x == y && (c & 1U) == 0U)) best = c;
    }
    return static_cast<std::uint8_t>(sign | best);
}
std::uint8_t oracle_e2m1_encode(float value) {
    const double mags[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    const double m = std::min(std::fabs(static_cast<double>(value)), 6.0);
    unsigned best = 0;
    for (unsigned c = 1; c < 8U; ++c) {
        const double x = std::fabs(mags[c] - m), y = std::fabs(mags[best] - m);
        if (x < y || (x == y && (c & 1U) == 0U)) best = c;
    }
    return static_cast<std::uint8_t>((std::signbit(value) ? 8U : 0U) | best);
}
bool same_bits(float x, float y) { return std::bit_cast<std::uint32_t>(x) == std::bit_cast<std::uint32_t>(y) || (std::isnan(x) && std::isnan(y) && std::signbit(x) == std::signbit(y)); }
void codecs() {
    namespace c = strix::ds41::codec;
    namespace f = strix::ds41;
    unsigned bad = 0;
    for (unsigned k = 0; k < 256; ++k) {
        const auto b = static_cast<std::uint8_t>(k);
        const float e8 = k == 255U ? std::numeric_limits<float>::quiet_NaN() : k == 0U ? std::bit_cast<float>(0x00400000U) : std::bit_cast<float>(k << 23U);
        const float e8h = k == 255U ? std::numeric_limits<float>::quiet_NaN() : k < 2U ? std::bit_cast<float>(0x00200000U << k) : std::bit_cast<float>((k - 1U) << 23U);
        bad += same_bits(c::e4m3(b), oracle_e4m3(b)) && same_bits(c::e8m0(b), e8) && same_bits(c::e8m0_half(b), e8h) ? 0U : 1U;
        bad += same_bits(f::e4m3_to_float(b), c::e4m3(b)) && same_bits(f::e8m0_to_float(b), e8) && same_bits(f::e8m0_half_to_float(b), e8h) ? 0U : 1U;
    }
    for (unsigned k = 0; k < 16; ++k) {
        const auto b = static_cast<std::uint8_t>(k);
        const float want = (k & 8U) != 0U ? -0.5F * static_cast<float>(f::e2m1_doubled[k & 7U]) : 0.5F * static_cast<float>(f::e2m1_doubled[k]);
        bad += same_bits(c::e2m1(b), want) && same_bits(f::e2m1_to_float(b), want) && c::e2m1_doubled(b) == f::e2m1_doubled[k] ? 0U : 1U;
    }
    expect(bad == 0U, "codec decodes equal the independent definitions (" + std::to_string(bad) + " differ)");
    std::vector<float> probes;
    for (std::uint64_t u = 0; u < (1ULL << 32); u += 4099U) probes.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(u)));
    const auto around = [&](double mid) {
        float x = static_cast<float>(mid);
        for (int i = 0; i < 3; ++i) x = std::nextafter(x, 0.0F);
        for (int i = 0; i < 7; ++i) { probes.push_back(x); probes.push_back(-x); x = std::nextafter(x, 1e30F); }
    };
    for (unsigned k = 0; k + 1 < 0x7fU; ++k) around((static_cast<double>(oracle_e4m3(static_cast<std::uint8_t>(k))) + oracle_e4m3(static_cast<std::uint8_t>(k + 1))) / 2.0);
    for (const double m : {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 448.0, 464.0}) around(m);
    for (const float x : {0.0F, -0.0F, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) probes.push_back(x);
    unsigned bad8 = 0, bad4 = 0, badf = 0;
    for (const float x : probes) {
        const std::uint8_t e = c::e4m3_encode(x);
        bad8 += e == oracle_e4m3_encode(x) ? 0U : 1U;
        badf += f::float_to_e4m3(x) == e ? 0U : 1U;
        if (!std::isnan(x)) bad4 += c::e2m1_encode(x) == oracle_e2m1_encode(x) ? 0U : 1U;
    }
    expect(bad8 == 0U && badf == 0U && bad4 == 0U, "encoders equal the nearest-value search on " + std::to_string(probes.size()) + " probes (" +
        std::to_string(bad8) + " E4M3, " + std::to_string(badf) + " formats E4M3, " + std::to_string(bad4) + " E2M1 differ)");
    std::printf("codecs: decodes of every code and encoders on %zu probes equal the independent definitions\n", probes.size());
}

// The plan derived from the model contract (real configuration) equals the hand-written released plan.
void contract_plan() {
    using namespace strix::deepseek41;
    const Config config = parse_config_json(real_config_json);
    a::check_contract(config);
    const auto p = a::plan_from_contract(layer_attributes(config));
    const auto want = a::released_plan();
    bool ok = p.size() == want.size();
    for (std::size_t l = 0; ok && l < p.size(); ++l) {
        const auto& x = p[l];
        const auto& y = want[l];
        ok = x.ratio == y.ratio && x.kv_source == y.kv_source && x.index_source == y.index_source && x.owns_kv == y.owns_kv &&
             x.owns_index == y.owns_index && x.candidate_source == y.candidate_source && x.candidate_user == y.candidate_user && x.rope == y.rope;
    }
    expect(ok, "plan from the contract equals the released plan");
}

void plan() {
    const auto p = a::released_plan();
    expect(p[0].ratio == 0 && p[1].ratio == 0 && p[1].kv_source == -1, "window-only layers 0, 1");
    expect(p[2].owns_kv && p[2].owns_index && p[2].ratio == 2, "layer 2 source");
    expect(p[7].kv_source == 2 && p[7].index_source == 2 && !p[7].owns_kv, "layer 7 reuses 2");
    expect(p[13].kv_source == 8 && p[19].kv_source == 14, "ratio-2 reuse");
    expect(p[20].owns_kv && p[20].candidate_source && p[20].ratio == 1, "layer 20 candidate source");
    expect(p[23].kv_source == 20 && p[23].index_source == 20, "layer 23 reuses 20");
    expect(p[24].owns_index && p[24].candidate_user && p[24].kv_source == 20, "layer 24 candidate user");
    expect(p[39].index_source == 36 && p[39].kv_source == 20 && !p[39].owns_index, "layer 39 reuses 36");
}

std::mt19937 rng(41);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

void select_against_sort() {
    for (const unsigned n : {300U, 700U, 5000U, 20000U}) {
        std::vector<float> s(n);
        for (auto& x : s) x = std::round(uni(-50, 50) * 4.0F) / 4.0F;  // many ties
        std::vector<std::uint8_t> cand((n + 7) / 8, 0);
        const auto got = a::select_topk(s, a::Pool::candidate_source, cand);
        // naive: all candidates computed by a full sort of the block maxima
        const std::size_t nb = (n + 7) / 8;
        std::vector<std::pair<float, std::size_t>> blocks(nb, {-std::numeric_limits<float>::infinity(), 0});
        for (std::size_t b = 0; b < nb; ++b) blocks[b].second = b;
        for (std::size_t t = 0; t < n; ++t) blocks[t / 8].first = std::max(blocks[t / 8].first, s[t]);
        blocks[nb - 1].first = std::numeric_limits<float>::infinity();
        std::stable_sort(blocks.begin(), blocks.end(), [](auto x, auto y) { return x.first > y.first; });
        std::vector<std::uint8_t> want(nb, 0);
        for (std::size_t i = 0; i < std::min<std::size_t>(a::kCandBlocks, nb); ++i) want[blocks[i].second] = 1;
        expect(std::equal(want.begin(), want.end(), cand.begin()), "candidate blocks n=" + std::to_string(n));
        std::vector<std::pair<float, std::uint32_t>> all(n);
        for (std::uint32_t t = 0; t < n; ++t) all[t] = {s[t], t};
        std::stable_sort(all.begin(), all.end(), [](auto x, auto y) { return x.first > y.first; });
        std::vector<std::uint32_t> top;
        for (std::size_t i = 0; i < std::min<std::size_t>(a::kTopK, n); ++i) top.push_back(all[i].second);
        std::sort(top.begin(), top.end());
        expect(top == got, "top-k n=" + std::to_string(n));
        const auto user = a::select_topk(s, a::Pool::candidate_user, cand);
        for (const auto t : user) if (n > a::kTopK) expect(cand[t / 8] != 0, "user picks inside candidates");
    }
}

// Window/compress caches, then attention vs double precision on the decoded keys.
void caches_and_attention() {
    const unsigned cap = a::kWindow + a::kMaxRows, npos = 700;
    std::vector<std::uint8_t> ring(cap * a::kWindowEntryBytes), kv(npos * a::kKvEntryBytes), ix(npos * a::kIdxEntryBytes);
    std::vector<float> norm(512), knorm(128);
    std::vector<std::uint16_t> wk(128 * 512);
    for (auto& x : norm) x = uni(0.5F, 1.5F);
    for (auto& x : knorm) x = uni(0.5F, 1.5F);
    for (auto& h : wk) h = static_cast<std::uint16_t>(0x2000U + (rng() & 0x3FFU) + ((rng() & 1U) << 15U));  // ~1e-3 magnitudes
    a::CompressState state;
    std::uint64_t p = 0;
    for (unsigned step = 0; p < npos; ++step) {
        const unsigned rows = std::min<unsigned>(1U + step % 8U, static_cast<unsigned>(npos - p));
        std::vector<float> kvp(rows * 512), sc(rows * 512);
        std::vector<std::uint64_t> pos(rows);
        for (unsigned r = 0; r < rows; ++r) pos[r] = p + r;
        for (auto& x : kvp) x = uni(-2, 2);
        for (auto& x : sc) x = uni(-3, 3);
        a::window_write({kvp, norm, pos, cap, a::RopeKind::compressed}, ring);
        a::compress({1, kvp, {}, norm, wk, knorm, pos}, state, kv, ix);
        p += rows;
    }
    // every written entry decodes to finite values with the norm's scale
    for (unsigned t = 0; t < npos; ++t) {
        const auto e = a::decode_kv_entry(&kv[t * a::kKvEntryBytes]);
        double ss = 0;
        for (const float x : e) { expect(std::isfinite(x), "kv finite"); ss += double(x) * x; }
        expect(ss > 1.0, "kv entry not empty");
    }
    std::vector<float> q(64 * 512), sink(64), o(64 * 512);
    for (auto& x : q) x = uni(-1, 1);
    for (auto& x : sink) x = uni(-2, 2);
    std::vector<std::uint32_t> idx(a::kTopK), cnt{300};
    for (unsigned i = 0; i < 300; ++i) idx[i] = 2 * i;
    const std::vector<std::uint64_t> pos{npos - 1};
    a::attention({q, sink, pos, cap, a::RopeKind::compressed, idx, cnt}, ring, kv, o);
    // double precision evaluation on the same decoded keys
    const auto rope = a::rope_row(a::RopeKind::compressed, npos - 1);
    double worst = 0;
    for (unsigned h = 0; h < 64; ++h) {
        std::vector<double> qq(512);
        std::vector<float> qf(&q[h * 512], &q[h * 512] + 512);
        for (unsigned i = 0; i < 32; ++i) a::rope_pair(qf[448 + 2 * i], qf[449 + 2 * i], rope.c[i], rope.s[i], false);
        for (unsigned d = 0; d < 512; ++d) qq[d] = qf[d];
        std::vector<std::array<float, 512>> keys;
        for (unsigned w = 0; w < 128; ++w) keys.push_back(a::decode_window_entry(&ring[((npos - 128 + w) % cap) * a::kWindowEntryBytes]));
        for (unsigned i = 0; i < 300; ++i) keys.push_back(a::decode_kv_entry(&kv[idx[i] * a::kKvEntryBytes]));
        std::vector<double> s(keys.size());
        double m = -1e300;
        for (std::size_t j = 0; j < keys.size(); ++j) {
            double d = 0;
            for (unsigned k = 0; k < 512; ++k) d += qq[k] * keys[j][k];
            s[j] = d * a::kAttnScale;
            m = std::max(m, s[j]);
        }
        double l = std::exp(sink[h] - m);
        std::vector<double> out(512, 0.0);
        for (std::size_t j = 0; j < keys.size(); ++j) {
            const double e = std::exp(s[j] - m);
            l += e;
            for (unsigned k = 0; k < 512; ++k) out[k] += e * keys[j][k];
        }
        std::vector<float> of(512);
        for (unsigned k = 0; k < 512; ++k) of[k] = static_cast<float>(out[k] / l);
        for (unsigned i = 0; i < 32; ++i) a::rope_pair(of[448 + 2 * i], of[449 + 2 * i], rope.c[i], rope.s[i], true);
        double nrm = 0;
        for (unsigned k = 0; k < 512; ++k) nrm = std::max(nrm, std::fabs(double(of[k])));
        for (unsigned k = 0; k < 512; ++k) worst = std::max(worst, std::fabs(double(o[h * 512 + k]) - of[k]) / nrm);
    }
    std::printf("attention vs double on the same keys: worst error / max |o| = %.3g\n", worst);
    expect(worst < 1e-5, "attention accuracy");
}

// Official reference case (exported by an internal script from the published inference code).
template <class T> std::vector<T> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("missing " + path);
    f.seekg(0, std::ios::end);
    const auto n = static_cast<std::size_t>(f.tellg()) / sizeof(T);
    f.seekg(0);
    std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
    return v;
}
void official_case(const std::string& dir);

}  // namespace

int main() {
    try {
        scalar_formats();
        plan();
        contract_plan();
        codecs();
        select_against_sort();
        caches_and_attention();
        if (const char* dir = std::getenv("STRIX_DS41_ATTN_CASE")) official_case(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", failures == 0 ? "ds41 attention CPU tests pass" : "FAILED");
    return failures == 0 ? 0 : 1;
}

namespace {
// Case layout (little-endian raw files, written by the export script):
//   meta.txt: "P D topk cand_blocks block_size layers..." ; for every layer L listed and token t < P + D:
//   L<L>_kv.f32 [(P+D) x 512] wkv(x); L<L>_ckv.f32 / L<L>_cscore.f32 [(P+D) x 512] compressor projections (KV sources);
//   L<L>_q.f32 [D x 64 x 512] (decode steps), L<L>_iq.f32 [D x 32 x 128], L<L>_iw.f32 [D x 32];
//   weights: L<L>_kvnorm.f32, L<L>_cnorm.f32, L<L>_wk.f16 [128 x 512], L<L>_knorm.f32, L<L>_sink.f32;
//   official outputs: L<L>_win.f32 [128 x 512] window cache after the last step (dequantized),
//   L<L>_ckvcache.f32 / L<L>_kcache.f32 (dequantized compressed KV / index keys), L<L>_topk.i32 [D x topk]
//   (compressed indices, -1 padded), L<L>_o.f32 [D x 64 x 512] attention output after the inverse RoPE.
void official_case(const std::string& dir) {
    std::ifstream meta(dir + "/meta.txt");
    unsigned P, D, topk, cand, bs;
    meta >> P >> D >> topk >> cand >> bs;
    std::vector<int> layers;
    for (int l; meta >> l;) layers.push_back(l);
    const auto plan = a::released_plan();
    const unsigned total = P + D, cap = a::kWindow + a::kMaxRows;
    struct Src { std::vector<std::uint8_t> kv, ix; unsigned ratio; };
    std::vector<Src> srcs(40);
    std::vector<std::vector<std::uint32_t>> topk_of(40);  // per index source, last step's list
    std::vector<std::uint8_t> cand_mask;
    std::printf("official case: P %u D %u topk %u candidates %u x %u, layers", P, D, topk, cand, bs);
    for (const int L : layers) std::printf(" %d", L);
    std::printf("\n");
    // Run each layer token by token (rows of 1), layers interleaved per token like the reference.
    struct LayerState {
        std::vector<std::uint8_t> ring; a::CompressState cs; std::vector<float> kv, ckv, csc, q, iq, iw, kvnorm, cnorm, knorm, sink;
        std::vector<std::uint16_t> wk; double o_err = 0, o_rel = 0, win_err = 0; unsigned topk_same = 0, topk_total = 0;
    };
    std::vector<LayerState> st(40);
    const auto f = [&](int L, const char* name) { return dir + "/L" + std::to_string(L) + "_" + name; };
    for (const int L : layers) {
        auto& s = st[static_cast<std::size_t>(L)];
        s.ring.assign(cap * a::kWindowEntryBytes, 0);
        s.kv = read_bin<float>(f(L, "kv.f32"));
        s.kvnorm = read_bin<float>(f(L, "kvnorm.f32"));
        s.sink = read_bin<float>(f(L, "sink.f32"));
        s.q = read_bin<float>(f(L, "q.f32"));
        const auto& pl = plan[static_cast<std::size_t>(L)];
        if (pl.owns_kv) {
            s.ckv = read_bin<float>(f(L, "ckv.f32"));
            if (pl.ratio == 2) s.csc = read_bin<float>(f(L, "cscore.f32"));
            s.cnorm = read_bin<float>(f(L, "cnorm.f32"));
            s.knorm = read_bin<float>(f(L, "knorm.f32"));
            s.wk = read_bin<std::uint16_t>(f(L, "wk.f16"));
            srcs[static_cast<std::size_t>(L)] = {std::vector<std::uint8_t>(total * a::kKvEntryBytes), std::vector<std::uint8_t>(total * a::kIdxEntryBytes), pl.ratio};
        }
        if (pl.owns_index) { s.iq = read_bin<float>(f(L, "iq.f32")); s.iw = read_bin<float>(f(L, "iw.f32")); }
    }
    for (unsigned t = 0; t < total; ++t) {
        const std::vector<std::uint64_t> pos{t};
        for (const int L : layers) {
            auto& s = st[static_cast<std::size_t>(L)];
            const auto& pl = plan[static_cast<std::size_t>(L)];
            a::window_write({std::span<const float>(&s.kv[t * 512], 512), s.kvnorm, pos, cap, pl.rope}, s.ring);
            if (pl.owns_kv) {
                auto& src = srcs[static_cast<std::size_t>(L)];
                a::compress({pl.ratio, std::span<const float>(&s.ckv[t * 512], 512),
                    pl.ratio == 2 ? std::span<const float>(&s.csc[t * 512], 512) : std::span<const float>{}, s.cnorm, s.wk, s.knorm, pos},
                    s.cs, src.kv, src.ix);
            }
            if (t < P) continue;
            const unsigned step = t - P;
            std::vector<std::uint32_t> idx(topk, 0), cnt{0};
            if (pl.ratio != 0) {
                if (pl.owns_index) {
                    const auto& src = srcs[static_cast<std::size_t>(pl.kv_source)];
                    std::vector<std::vector<float>> sc;
                    a::index_scores({std::span<const float>(&s.iq[step * 32 * 128], 32 * 128), std::span<const float>(&s.iw[step * 32], 32), pos, pl.ratio, pl.rope}, src.ix, sc);
                    const auto pool = pl.candidate_source ? a::Pool::candidate_source : pl.candidate_user ? a::Pool::candidate_user : a::Pool::plain;
                    if (pl.candidate_source) cand_mask.assign((sc[0].size() + bs - 1) / bs + 1, 0);
                    topk_of[static_cast<std::size_t>(L)] = a::select_topk(sc[0], pool, cand_mask, topk, cand, bs);
                    // compare the selected set with the reference's
                    const auto ref = read_bin<std::int32_t>(f(L, "topk.i32"));
                    std::vector<std::uint32_t> rs;
                    for (unsigned i = 0; i < topk; ++i) if (ref[step * topk + i] >= 0) rs.push_back(static_cast<std::uint32_t>(ref[step * topk + i]));
                    std::sort(rs.begin(), rs.end());
                    const auto& mine = topk_of[static_cast<std::size_t>(L)];
                    std::vector<std::uint32_t> both;
                    std::set_intersection(mine.begin(), mine.end(), rs.begin(), rs.end(), std::back_inserter(both));
                    s.topk_same += static_cast<unsigned>(both.size());
                    s.topk_total += static_cast<unsigned>(rs.size());
                }
                const auto& mine = topk_of[static_cast<std::size_t>(pl.index_source)];
                std::copy(mine.begin(), mine.end(), idx.begin());
                cnt[0] = static_cast<std::uint32_t>(mine.size());
            }
            std::vector<float> o(64 * 512);
            const auto& src = pl.ratio != 0 ? srcs[static_cast<std::size_t>(pl.kv_source)] : srcs[0];
            a::attention({std::span<const float>(&s.q[step * 64 * 512], 64 * 512), s.sink, pos, cap, pl.rope, idx,
                pl.ratio != 0 ? std::span<const std::uint32_t>(cnt) : std::span<const std::uint32_t>{}, topk}, s.ring, src.kv, o);
            const auto ref = read_bin<float>(f(L, "o.f32"));
            double num = 0, den = 0;
            for (unsigned i = 0; i < 64 * 512; ++i) {
                const double d = double(o[i]) - ref[step * 64 * 512 + i];
                num += d * d; den += double(ref[step * 64 * 512 + i]) * ref[step * 64 * 512 + i];
                s.o_err = std::max(s.o_err, std::fabs(d));
            }
            s.o_rel = std::max(s.o_rel, std::sqrt(num / den));
        }
    }
    for (const int L : layers) {
        auto& s = st[static_cast<std::size_t>(L)];
        const auto& pl = plan[static_cast<std::size_t>(L)];
        // window cache after the last step vs the reference's (slot = position % 128 there)
        const auto win = read_bin<float>(f(L, "win.f32"));
        double werr = 0, wmax = 0;
        unsigned wcodes = 0;
        for (unsigned slot = 0; slot < 128; ++slot) {
            const unsigned position = total - 128 + ((slot + 128 - total % 128) % 128);
            const auto mine = a::decode_window_entry(&s.ring[(position % cap) * a::kWindowEntryBytes]);
            for (unsigned d = 0; d < 512; ++d) {
                const float r = win[(position % 128) * 512 + d];
                werr = std::max(werr, std::fabs(double(mine[d]) - r));
                wmax = std::max(wmax, std::fabs(double(r)));
                wcodes += mine[d] != r ? 1U : 0U;
            }
        }
        std::printf("layer %2d: window cache max |diff| %.3g (max |v| %.3g, %u/%u values differ)", L, werr, wmax, wcodes, 128U * 512U);
        if (pl.owns_kv) {
            const auto& src = srcs[static_cast<std::size_t>(L)];
            const auto ck = read_bin<float>(f(L, "ckvcache.f32")), kk = read_bin<float>(f(L, "kcache.f32"));
            const unsigned n = total / pl.ratio;
            unsigned diff_kv = 0, diff_k = 0;
            double ekv = 0, ek = 0;
            for (unsigned t = 0; t < n; ++t) {
                const auto mk = a::decode_kv_entry(&src.kv[t * a::kKvEntryBytes]);
                const auto mi = a::decode_idx_entry(&src.ix[t * a::kIdxEntryBytes]);
                for (unsigned d = 0; d < 512; ++d) { diff_kv += mk[d] != ck[t * 512 + d] ? 1U : 0U; ekv = std::max(ekv, std::fabs(double(mk[d]) - ck[t * 512 + d])); }
                for (unsigned d = 0; d < 128; ++d) { diff_k += mi[d] != kk[t * 128 + d] ? 1U : 0U; ek = std::max(ek, std::fabs(double(mi[d]) - kk[t * 128 + d])); }
            }
            std::printf("; compressed KV %u/%u values differ (max %.3g), index keys %u/%u (max %.3g)", diff_kv, n * 512, ekv, diff_k, n * 128, ek);
        }
        if (s.topk_total != 0) std::printf("; top-k overlap %u/%u", s.topk_same, s.topk_total);
        std::printf("; attention output rel L2 %.3g, max |diff| %.3g\n", s.o_rel, s.o_err);
        expect(s.o_rel < 2e-2, "official case: attention output layer " + std::to_string(L));
        if (s.topk_total != 0) expect(s.topk_same * 100U >= s.topk_total * 95U, "official case: top-k overlap layer " + std::to_string(L));
    }
}
}  // namespace
