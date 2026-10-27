// GLM speculative decoding, CPU equivalence tests (design notes section 5, gate S0).
// A toy target built ONLY from the Tier A host ops (KDA layer with 2 heads + DSA/MLA layer with the pooled indexer, random
// weights, D = 256, vocabulary 97) is decoded twice: plain greedy (kda_step / dsa_append per token) and speculative
// (verify rows of one slot through KdaVerify + dsa_append_rows + per-row indexer/attention, greedy acceptance, commit and
// rollback), with drafters that are right, wrong, or right up to a random depth, K = 1..7, prefill in random row chunks.
// Required, every cycle: emitted tokens == plain greedy and the committed rows' logits == the plain run's bit for bit; after commit the DSA caches (latent, key, gate, pooled) equal the
// plain run's prefix byte for byte and the flushed KDA state + conv equal the plain state at the same position bit for bit.
// The run crosses 2,051 tokens so the indexer selects (P > 512 pools), and rollbacks land on every pool phase.
// Plus: accept_greedy cases; MTP layer composition: one call over R rows == R single-row calls (bitwise), and rollback.
// usage: glm_spec_cpu_tests [tokens=2300] [seed=7] [mutant=0]   (mutant 1..3 must FAIL: the checks are not vacuous)
#include "glm/spec/glm_spec_host.hpp"
#include "glm/spec/glm_spec_merge.hpp"

#include <bit>
#include <cmath>
#include <stdexcept>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <random>
#include <limits>
#include <string>
#include <vector>

using namespace strix::glm;
using namespace strix::glm::spec;
using host::WType;

namespace {

int failures = 0;
int g_mutant = 0;  // negative controls: 1 commit one row too many, 2 rows see the whole block, 3 keep a rejected row,
                   // 4 MTP cache truncated to p + m instead of p, 5 a flagged rejected row aborts the cycle
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

std::uint16_t to_f16(float f) {  // round-to-nearest F16 of a small finite value (test weights only)
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t s = (u >> 16) & 0x8000U;
    int e = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
    std::uint32_t m = u & 0x7fffffU;
    if (e <= 0) return static_cast<std::uint16_t>(s);
    std::uint32_t r = (static_cast<std::uint32_t>(e) << 10) | (m >> 13);
    if ((m & 0x1fffU) > 0x1000U || ((m & 0x1fffU) == 0x1000U && (r & 1U))) ++r;
    return static_cast<std::uint16_t>(s | r);
}

struct Rng {
    std::mt19937_64 g;
    explicit Rng(std::uint64_t s) : g(s) {}
    float u(float a) { return std::uniform_real_distribution<float>(-a, a)(g); }
    unsigned n(unsigned k) { return static_cast<unsigned>(g() % k); }
};

std::uint64_t fnv(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ULL) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
std::uint64_t kda_hash(const KdaState& s) {
    return fnv(s.conv.data(), s.conv.size() * 4, fnv(s.S.data(), s.S.size() * 4));
}

// ---------------------------------------------------------------- toy target from Tier A host ops
struct Toy {
    static constexpr unsigned D = 256, V = 97, H = 2, C = H * kKdaDim;
    static constexpr std::uint32_t POISON = V - 1;  // NaN embedding (fires C2 in the KDA layer); never a greedy output
    std::vector<float> emb, w_o1, w_o2, w_head;
    std::vector<std::uint16_t> w_qkv, w_f, w_g, w_b, w_kv, w_q, w_iq, w_ik, w_igate, w_iw, ik_w, ik_b, kv_b, conv, onorm, kvnorm;
    std::vector<float> A_log, dt_bias, ape;
    Mat m_qkv, m_f, m_g, m_b, m_o1, m_kv, m_q, m_iq, m_o2, m_head, m_kvb;
    IndexerWeights iw;
    KdaWeights kw;
    Vec v_kvnorm;

    static void fill16(std::vector<std::uint16_t>& v, std::size_t n, Rng& r, float a) {
        v.resize(n);
        for (auto& x : v) x = to_f16(r.u(a));
    }
    static void fill32(std::vector<float>& v, std::size_t n, Rng& r, float a) {
        v.resize(n);
        for (auto& x : v) x = r.u(a);
    }
    explicit Toy(std::uint64_t seed) {
        Rng r(seed);
        const float a = 1.0F / 16.0F;  // ~1/sqrt(D)
        fill32(emb, static_cast<std::size_t>(V) * D, r, 1.0F);
        for (unsigned d = 0; d < D; ++d) emb[static_cast<std::size_t>(POISON) * D + d] = __builtin_nanf("");
        fill16(w_qkv, static_cast<std::size_t>(3) * C * D, r, a);
        fill16(w_f, static_cast<std::size_t>(C) * D, r, a);
        fill16(w_g, static_cast<std::size_t>(C) * D, r, a);
        fill16(w_b, static_cast<std::size_t>(H) * D, r, a);
        fill32(w_o1, static_cast<std::size_t>(D) * C, r, 1.0F / 16.0F);
        fill16(w_kv, static_cast<std::size_t>(kKvLora) * D, r, a);
        fill16(w_q, static_cast<std::size_t>(kMlaDim) * D, r, a);
        fill16(w_iq, static_cast<std::size_t>(kIdxHeads) * kIdxDim * D, r, a);
        fill16(w_ik, static_cast<std::size_t>(kIdxDim) * D, r, a);
        fill16(w_igate, static_cast<std::size_t>(kIdxDim) * D, r, a);
        fill16(w_iw, static_cast<std::size_t>(kIdxHeads) * D, r, a);
        fill16(ik_w, kIdxDim, r, 1.0F);
        fill16(ik_b, kIdxDim, r, 0.1F);
        fill16(kv_b, static_cast<std::size_t>(2 * kMlaDim) * kKvLora, r, 1.0F / 22.0F);
        fill32(w_o2, static_cast<std::size_t>(D) * kMlaDim, r, 1.0F / 16.0F);
        fill32(w_head, static_cast<std::size_t>(V) * D, r, 1.0F / 16.0F);
        conv.resize(static_cast<std::size_t>(3) * C * kConvK);
        for (auto& x : conv) x = float_to_bf16(r.u(0.5F));
        onorm.resize(kKdaDim);
        for (auto& x : onorm) x = float_to_bf16(1.0F + r.u(0.1F));
        kvnorm.resize(kKvLora);
        for (auto& x : kvnorm) x = float_to_bf16(1.0F + r.u(0.1F));
        fill32(A_log, H, r, 0.5F);
        fill32(dt_bias, C, r, 0.5F);
        fill32(ape, static_cast<std::size_t>(kKpool) * kIdxDim, r, 0.5F);
        m_qkv = {w_qkv.data(), WType::F16, 3 * C, D};
        m_f = {w_f.data(), WType::F16, C, D};
        m_g = {w_g.data(), WType::F16, C, D};
        m_b = {w_b.data(), WType::F16, H, D};
        m_o1 = {w_o1.data(), WType::F32, D, C};
        m_kv = {w_kv.data(), WType::F16, kKvLora, D};
        m_q = {w_q.data(), WType::F16, kMlaDim, D};
        m_iq = {w_iq.data(), WType::F16, kIdxHeads * kIdxDim, D};
        m_o2 = {w_o2.data(), WType::F32, D, kMlaDim};
        m_head = {w_head.data(), WType::F32, V, D};
        m_kvb = {kv_b.data(), WType::F16, 2 * kMlaDim, kKvLora};
        iw = {{w_ik.data(), WType::F16, kIdxDim, D}, {w_igate.data(), WType::F16, kIdxDim, D},
              {w_iw.data(), WType::F16, kIdxHeads, D}, {ik_w.data(), WType::F16, kIdxDim}, {ik_b.data(), WType::F16, kIdxDim},
              ape.data()};
        kw = {conv.data(), A_log.data(), dt_bias.data(), {onorm.data(), WType::BF16, kKdaDim}};
        v_kvnorm = {kvnorm.data(), WType::BF16, kKvLora};
    }
    KdaState fresh_kda() const {
        KdaState s;
        s.heads = H;
        s.S.assign(static_cast<std::size_t>(H) * kKdaDim * kKdaDim, 0.0F);
        s.conv.assign(static_cast<std::size_t>(3) * C * 3, 0.0F);
        return s;
    }
    KdaRow kda_row(std::span<const float> x) const {
        KdaRow k;
        k.qkv.resize(3 * C);
        k.f.resize(C);
        k.gate.resize(C);
        k.b.resize(H);
        host::dense(x, m_qkv, k.qkv);
        host::dense(x, m_f, k.f);
        host::dense(x, m_g, k.gate);
        host::dense(x, m_b, k.b);
        return k;
    }
    void add_proj(std::span<float> x, const Mat& w, std::span<const float> y) const {
        std::vector<float> t(D);
        host::dense(y, w, t);
        for (unsigned d = 0; d < D; ++d) x[d] = x[d] + t[d];
    }
    void latent(std::span<const float> x, std::span<float> lat) const {
        std::vector<float> t(kKvLora);
        host::dense(x, m_kv, t);
        host::rmsnorm(t, &v_kvnorm, kEps, lat);
    }
    // the DSA layer's attention for the row at position t (cache already holds positions <= t)
    void attend(const DsaCache& c, unsigned t, std::span<float> x) const {
        std::vector<float> qI(kIdxHeads * kIdxDim), q(kMlaDim), o(kMlaDim);
        host::dense(x, m_iq, qI);
        host::dense(x, m_q, q);
        const auto tok = host::indexer_select(c, t, qI, x, iw);
        host::mla_attention(c, tok, q, m_kvb, 1, o);
        add_proj(x, m_o2, o);
    }
    struct Out {
        std::uint32_t id;
        std::uint64_t logits;  // FNV of the logits bits: committed rows must match the plain run bit for bit
    };
    Out head(std::span<const float> x) const {
        std::vector<float> l(V);
        host::dense(x, m_head, l);
        return {host::argmax_local(std::span<const float>(l).first(V - 1), 0).id, fnv(l.data(), l.size() * 4)};
    }
};

// plain greedy: one token at position t = cache length
Toy::Out plain_step(const Toy& m, KdaState& kda, DsaCache& c, std::uint32_t tok) {
    std::vector<float> x(m.emb.begin() + static_cast<std::ptrdiff_t>(tok) * Toy::D, m.emb.begin() + static_cast<std::ptrdiff_t>(tok + 1) * Toy::D);
    std::vector<float> y(Toy::C), lat(kKvLora);
    const KdaRow r = m.kda_row(x);
    host::kda_step(kda, r.in(), m.kw, y);
    m.add_proj(x, m.m_o1, y);
    const unsigned t = c.tokens();
    m.latent(x, lat);
    host::dsa_append(c, lat, x, m.iw);
    m.attend(c, t, x);
    return m.head(x);
}

// verify rows of one slot at positions p .. p+R-1 (p = committed cache length): argmax per valid row + the first row
// whose checkpoint fired (R if none; rows from it on have no outputs)
struct VerifyOut {
    std::vector<Toy::Out> o;
    unsigned first = 0;
};
VerifyOut verify_rows(const Toy& m, KdaVerify& kv, DsaCache& c, std::span<const std::uint32_t> toks) {
    const unsigned R = static_cast<unsigned>(toks.size()), D = Toy::D, p = c.tokens();
    std::vector<float> X(static_cast<std::size_t>(R) * D), lat(static_cast<std::size_t>(R) * kKvLora);
    std::vector<KdaRow> rows;
    for (unsigned r = 0; r < R; ++r) {
        std::copy_n(m.emb.begin() + static_cast<std::ptrdiff_t>(toks[r]) * D, D, X.begin() + static_cast<std::ptrdiff_t>(r) * D);
        rows.push_back(m.kda_row(std::span<const float>(&X[static_cast<std::size_t>(r) * D], D)));
    }
    std::vector<std::vector<float>> y;
    unsigned first = kv.run(rows, m.kw, y);
    for (unsigned r = 0; r < first; ++r) {
        std::span<float> xr(&X[static_cast<std::size_t>(r) * D], D);
        m.add_proj(xr, m.m_o1, y[r]);
        m.latent(xr, std::span<float>(&lat[static_cast<std::size_t>(r) * kKvLora], kKvLora));
    }
    if (first > 0) first = dsa_append_rows(c, lat, X, D, m.iw, first);
    VerifyOut out;
    for (unsigned r = 0; r < first; ++r) {
        std::span<float> xr(&X[static_cast<std::size_t>(r) * D], D);
        try {
            m.attend(c, g_mutant == 2 ? std::min(p + R - 1, c.tokens() - 1) : p + r, xr);
            out.o.push_back(m.head(xr));
        } catch (const host::NonFinite&) {
            first = r;
            break;
        }
    }
    out.first = first;
    return out;
}

bool caches_equal_prefix(const DsaCache& a, const DsaCache& plain, unsigned n) {
    if (a.tokens() != n || a.pools() != n / kKpool) return false;
    return std::memcmp(a.latent.data(), plain.latent.data(), static_cast<std::size_t>(n) * kKvLora * 2) == 0 &&
           std::memcmp(a.ikey.data(), plain.ikey.data(), static_cast<std::size_t>(n) * kIdxDim * 2) == 0 &&
           std::memcmp(a.igate.data(), plain.igate.data(), static_cast<std::size_t>(n) * kIdxDim * 2) == 0 &&
           std::memcmp(a.pooled.data(), plain.pooled.data(), static_cast<std::size_t>(n / kKpool) * kIdxDim * 4) == 0;
}

void test_accept() {
    const std::uint32_t d1[] = {5, 6, 7}, t1[] = {5, 6, 7, 8};
    auto a = accept_greedy(d1, t1);
    check(a.m == 3 && a.bonus == 8, "accept_greedy: all drafts accepted, bonus = last target");
    const std::uint32_t t2[] = {5, 9, 7, 8};
    a = accept_greedy(d1, t2);
    check(a.m == 1 && a.bonus == 9, "accept_greedy: first mismatch at depth 2 (later equal drafts ignored)");
    const std::uint32_t t3[] = {4, 6, 7, 8};
    a = accept_greedy(d1, t3);
    check(a.m == 0 && a.bonus == 4, "accept_greedy: none accepted, bonus = target of row 0");
    a = accept_greedy(d1, std::span<const std::uint32_t>(t2).first(2), 2);
    check(a.m == 1 && a.bonus == 9 && !a.abort, "accept_greedy: row 2 flagged but rejected (m = 1): no abort");
    a = accept_greedy(d1, std::span<const std::uint32_t>(t1).first(2), 2);
    check(a.m == 2 && a.abort, "accept_greedy: flagged row 2 would be committed (d1, d2 accepted): abort after emitting d1, d2");
    a = accept_greedy(d1, std::span<const std::uint32_t>(t1).first(0), 0);
    check(a.m == 0 && a.abort, "accept_greedy: row 0 flagged: abort");
    // terminal frontier first (review #2): EOS = 5 emitted by valid row 0 as an accepted draft, row 1 flagged -> no abort
    const std::uint32_t de[] = {5, 6, 7}, te[] = {5};
    a = accept_greedy(de, te, 1, Terminal{5, kNoFlag});
    check(a.m == 1 && a.stop && !a.abort && !a.has_bonus && a.rows == 1,
          "accept_greedy: accepted draft = EOS, the flagged next row is beyond the frontier: stop, no abort, 1 row committed");
    a = accept_greedy(de, te, 1, Terminal{kNoFlag, 1});
    check(a.m == 1 && a.stop && !a.abort && a.rows == 1, "accept_greedy: budget 1 reached by the accepted draft: stop, no abort");
    a = accept_greedy(de, te, 1, Terminal{kNoFlag, 2});
    check(a.m == 1 && a.abort && a.rows == 1, "accept_greedy: budget 2: the flagged row 1 is within the frontier: abort");
    const std::uint32_t tb[] = {5, 9};
    a = accept_greedy(de, tb, 2, Terminal{9, kNoFlag});
    check(a.m == 1 && a.has_bonus && a.bonus == 9 && a.stop && !a.abort && a.rows == 2,
          "accept_greedy: terminal bonus from row 1, row 2 flagged beyond the frontier: stop after the bonus, no abort");
    const std::uint32_t tc[] = {5, 6};
    a = accept_greedy(de, tc, 2, Terminal{6, kNoFlag});
    check(a.m == 2 && a.stop && !a.abort && a.rows == 2, "accept_greedy: second accepted draft = EOS, row 2 flagged: stop, no abort");
    a = accept_greedy(de, tc, 2, Terminal{kNoFlag, 3});
    check(a.m == 2 && a.abort && a.rows == 2, "accept_greedy: same with no terminal: abort at row 2");
}

void test_equivalence(unsigned total, std::uint64_t seed) {
    const Toy m(seed);
    Rng r(seed * 31 + 1);
    const unsigned prompt_len = 37;
    std::vector<std::uint32_t> text;
    for (unsigned i = 0; i < prompt_len; ++i) text.push_back(r.n(Toy::V - 1));
    // ---- plain greedy reference: per-position KDA hashes, final caches
    KdaState pk = m.fresh_kda();
    DsaCache pc;
    std::vector<std::uint64_t> kh;  // kh[t] = KDA state hash after position t
    std::vector<std::uint32_t> g;   // g[t] = greedy next token after position t
    std::vector<std::uint64_t> lh;  // lh[t] = logits hash of position t
    for (unsigned t = 0; t + 1 < total; ++t) {
        const Toy::Out o = plain_step(m, pk, pc, text[t]);
        const std::uint32_t nx = o.id;
        g.push_back(nx);
        lh.push_back(o.logits);
        kh.push_back(kda_hash(pk));
        if (t + 1 >= prompt_len) text.push_back(nx);
    }
    std::printf("     plain: %u positions, %u pools, last greedy %u\n", pc.tokens(), pc.pools(), g.back());
    // ---- speculative run
    KdaVerify kv(m.fresh_kda());
    DsaCache sc;
    unsigned p = 0;  // committed positions [0, p)
    bool ok = true;
    unsigned cycles = 0, drafted = 0, accepted = 0, prefill_chunks = 0, rejects_cross_pool = 0, poisoned = 0, aborts = 0, stops = 0;
    std::array<unsigned, 4> frontier_phase{};
    while (p < prompt_len) {  // prefill through the verify path, every row committed
        const unsigned R = std::min(1 + r.n(8), prompt_len - p);
        if (verify_rows(m, kv, sc, std::span<const std::uint32_t>(&text[p], R)).first != R) {
            check(false, "spec: prefill row flagged");
            return;
        }
        kv.commit(R);
        p += R;
        ++prefill_chunks;
    }
    // position p-1 is the last prompt row: its greedy next token is the first generated one; x0 of a cycle = text[p-1]?
    // Cycles verify [x0 = text[p], d1..dK]: row 0 is the first not-yet-processed position p.
    // After prefill, positions [0, prompt_len) are in the caches; the next row to process is position prompt_len with
    // token text[prompt_len] = g[prompt_len - 1].
    while (p + 2 < total && ok) {  // the bonus of the last cycle needs g[p + K]: p + K <= total - 2
        const unsigned K = std::min<unsigned>(1 + r.n(7), total - 2 - p);
        if (K == 0) break;
        const unsigned mode = r.n(4), good = mode == 0 ? K : (mode == 1 ? 0 : r.n(K + 1));
        std::vector<std::uint32_t> rows{text[p]};
        for (unsigned j = 1; j <= K; ++j) {
            const std::uint32_t right = g[p + j - 1];
            rows.push_back(j <= good ? right : (right + 1 + r.n(Toy::V - 2)) % (Toy::V - 1));
        }
        if (good < K && r.n(4) == 0) {  // a poisoned (NaN) draft after the first wrong one: flagged, rejected, no abort
            rows[1 + good + r.n(K - good)] = Toy::POISON;
            ++poisoned;
        }
        const auto vr = verify_rows(m, kv, sc, rows);
        const auto& vo = vr.o;
        std::vector<std::uint32_t> tgt;
        for (const auto& o : vo) tgt.push_back(o.id);
        Accepted a = accept_greedy(std::span<const std::uint32_t>(rows).subspan(1), tgt, vr.first == rows.size() ? kNoFlag : vr.first,
                                   Terminal{kNoFlag, total - 1 - p});  // output limit: position total - 1
        if (g_mutant == 5 && vr.first < rows.size()) a.abort = true;
        if (a.abort) {
            ++aborts;
            check(false, "spec: cycle aborted on a flagged row although plain greedy never fails (cycle start " + std::to_string(p) + ")");
            ok = false;
            break;
        }
        for (unsigned j = 0; j < a.rows; ++j) ok = ok && vo[j].logits == lh[p + j];  // committed rows: logits bitwise
        // emitted d1..dm, bonus: must equal the plain greedy stream
        for (unsigned j = 0; j < a.m; ++j) ok = ok && rows[1 + j] == g[p + j];
        ok = ok && (!a.has_bonus || a.bonus == g[p + a.m]);
        stops += a.stop;
        if (!ok) {
            check(false, "spec: emitted token or committed-row logits differ from plain at cycle start " + std::to_string(p));
            break;
        }
        const unsigned front = p + a.rows;
        if (a.m < K && (p + K) / kKpool > front / kKpool) ++rejects_cross_pool;  // drafted into a pool, rolled back
        kv.commit(g_mutant == 1 ? std::min(a.rows + 1, K + 1) : a.rows);
        sc.truncate(g_mutant == 3 && a.m < K ? front + 1 : front);
        ++frontier_phase[front % kKpool];
        ++cycles;
        drafted += K;
        accepted += a.m;
        if (!caches_equal_prefix(sc, pc, front)) {
            check(false, "spec: DSA caches differ from plain after commit at frontier " + std::to_string(front));
            ok = false;
            break;
        }
        KdaVerify probe = kv;
        probe.flush(m.kw);
        if (kda_hash(probe.committed()) != kh[front - 1]) {
            check(false, "spec: KDA state/conv differ from plain at position " + std::to_string(front - 1));
            ok = false;
            break;
        }
        p = front;
        if (a.stop) break;
    }
    char b[256];
    std::snprintf(b, sizeof b,
                  "spec == plain greedy, caches + KDA bitwise after every commit (%u cycles, %u drafted, %u accepted, "
                  "%u prefill chunks, %u rollbacks across a pool boundary, frontier phases %u/%u/%u/%u, %u cycles with a "
                  "flagged rejected row, end %u)",
                  cycles, drafted, accepted, prefill_chunks, rejects_cross_pool, frontier_phase[0], frontier_phase[1],
                  frontier_phase[2], frontier_phase[3], poisoned, p);
    check(ok && p + 2 >= total && pc.pools() > kIdxPools && rejects_cross_pool > 0 && poisoned > 0 && aborts == 0 && stops <= 1 &&
              frontier_phase[0] && frontier_phase[1] && frontier_phase[2] && frontier_phase[3],
          b);
}

// ---------------------------------------------------------------- MTP composition
void test_mtp(std::uint64_t seed) {
    Rng r(seed);
    const unsigned D = kHidden, V = 1000;
    std::vector<std::uint16_t> nrm(D), qan(kQLora), kvan(kKvLora), kvb(static_cast<std::size_t>(2 * kMlaDim) * kKvLora),
        router(static_cast<std::size_t>(kExperts) * D), ik(static_cast<std::size_t>(kIdxDim) * D), ig(static_cast<std::size_t>(kIdxDim) * D),
        iwp(static_cast<std::size_t>(kIdxHeads) * D), ikw(kIdxDim), ikb(kIdxDim);
    for (auto& x : nrm) x = float_to_bf16(1.0F + r.u(0.1F));
    for (auto& x : qan) x = float_to_bf16(1.0F + r.u(0.1F));
    for (auto& x : kvan) x = float_to_bf16(1.0F + r.u(0.1F));
    for (auto& x : kvb) x = to_f16(r.u(0.05F));
    for (auto& x : router) x = to_f16(r.u(0.02F));
    for (auto& x : ik) x = to_f16(r.u(0.02F));
    for (auto& x : ig) x = to_f16(r.u(0.02F));
    for (auto& x : iwp) x = to_f16(r.u(0.02F));
    for (auto& x : ikw) x = to_f16(1.0F);
    for (auto& x : ikb) x = to_f16(r.u(0.1F));
    std::vector<float> bias(kExperts), ape(static_cast<std::size_t>(kKpool) * kIdxDim), emb(static_cast<std::size_t>(V) * D);
    for (auto& x : bias) x = r.u(0.01F);
    for (auto& x : ape) x = r.u(0.5F);
    for (auto& x : emb) x = r.u(1.0F);
    // EXL3 sites stand-ins: deterministic sparse mixes (any fixed function will do: the test is composition order)
    auto mix = [](unsigned salt) {
        return [salt](std::span<const float> x, std::span<float> y) {
            const std::size_t n = x.size();
            for (std::size_t o = 0; o < y.size(); ++o) {
                float acc = 0.0F;
                for (unsigned k = 0; k < 8; ++k) {
                    const std::size_t i = (o * 2654435761ULL + k * 40503ULL + salt * 97ULL) % n;
                    acc = __builtin_fmaf(x[i], (k & 1U) ? 0.125F : -0.0625F, acc);
                }
                y[o] = acc;
            }
        };
    };
    MtpWeights w;
    w.embed = [&](std::uint32_t t, std::span<float> e) { std::copy_n(emb.begin() + static_cast<std::ptrdiff_t>(t) * D, D, e.begin()); };
    w.enorm = w.hnorm = w.input_ln = w.post_ln = w.head_norm = {nrm.data(), WType::BF16, D};
    w.q_a_norm = {qan.data(), WType::BF16, kQLora};
    w.kv_a_norm = {kvan.data(), WType::BF16, kKvLora};
    w.eh_proj = mix(1);
    w.q_a = mix(2);
    w.q_b = mix(3);
    w.kv_a = mix(4);
    w.o_proj = mix(5);
    w.idx_wq_b = mix(6);
    w.kv_b = {kvb.data(), WType::F16, 2 * kMlaDim, kKvLora};
    w.heads = 1;
    w.idx = {{ik.data(), WType::F16, kIdxDim, D}, {ig.data(), WType::F16, kIdxDim, D}, {iwp.data(), WType::F16, kIdxHeads, D},
             {ikw.data(), WType::F16, kIdxDim}, {ikb.data(), WType::F16, kIdxDim}, ape.data()};
    w.router = {router.data(), WType::F16, kExperts, D};
    w.router_bias = bias.data();
    w.expert = [&](std::uint32_t e, std::span<const float> x, std::span<float> y) { mix(100 + e)(x, y); };
    w.shared = mix(7);
    w.head = mix(8);
    w.vocab_n = V;
    const unsigned R = 6, T = 23;
    std::vector<float> h(static_cast<std::size_t>(T) * D);
    for (auto& x : h) x = r.u(1.0F);
    std::vector<std::uint32_t> x(T);
    for (auto& t : x) t = r.n(V);
    // A: all rows one by one; B: rows 0..T-R-1 one by one, then the last R rows in one call (catch-up form)
    DsaCache ca, cb;
    std::vector<MtpOut> oa, ob;
    for (unsigned t = 0; t < T; ++t) {
        auto o = mtp_forward(ca, std::span<const float>(&h[static_cast<std::size_t>(t) * D], D), std::span<const std::uint32_t>(&x[t], 1), w);
        oa.push_back(o[0]);
    }
    for (unsigned t = 0; t < T - R; ++t) {
        auto o = mtp_forward(cb, std::span<const float>(&h[static_cast<std::size_t>(t) * D], D), std::span<const std::uint32_t>(&x[t], 1), w);
        ob.push_back(o[0]);
    }
    for (auto& o : mtp_forward(cb, std::span<const float>(&h[static_cast<std::size_t>(T - R) * D], static_cast<std::size_t>(R) * D),
                               std::span<const std::uint32_t>(&x[T - R], R), w))
        ob.push_back(o);
    bool same = caches_equal_prefix(cb, ca, T);
    for (unsigned t = 0; t < T; ++t)
        same = same && oa[t].top.id == ob[t].top.id && std::memcmp(oa[t].h.data(), ob[t].h.data(), D * 4) == 0 &&
               std::memcmp(&oa[t].top.value, &ob[t].top.value, 4) == 0;
    check(same, "mtp: one call over R = 6 rows == 6 single-row calls (h', argmax, MTP caches bitwise)");
    // rollback: truncate to 19 and redo rows 19..22 -> same as A
    cb.truncate(19);
    bool same2 = true;
    for (unsigned t = 19; t < T; ++t) {
        auto o = mtp_forward(cb, std::span<const float>(&h[static_cast<std::size_t>(t) * D], D), std::span<const std::uint32_t>(&x[t], 1), w);
        same2 = same2 && std::memcmp(o[0].h.data(), oa[t].h.data(), D * 4) == 0;
    }
    check(same2 && caches_equal_prefix(cb, ca, T), "mtp: rollback across a pool boundary (truncate 19) + redo == straight run");
}


// Independent single-row MTP reference written from the design formula (not from glm_spec_host.cpp): appends the row at
// the next MTP index, then attends with either a fresh selection or the shared rule (pools of the last catch-up row,
// then every position from 4 * P0 to t). Returns h', argmax, the list used and the fresh list (to show the test
// discriminates).
struct RefRow {
    std::vector<float> h;
    std::uint32_t top = 0;
    std::vector<std::uint32_t> tokens, fresh;
};
RefRow ref_mtp_row(DsaCache& c, std::span<const float> h, std::uint32_t x, const MtpWeights& w,
                   const std::vector<std::uint32_t>* pools, unsigned P0) {
    const unsigned D = kHidden;
    std::vector<float> e(D), cat(2 * D), z(D), xin(D), t1(kQLora), qres(kQLora), kv(kKvLora), lat(kKvLora);
    w.embed(x, e);
    host::rmsnorm(e, &w.enorm, kEps, std::span<float>(cat).first(D));
    host::rmsnorm(h, &w.hnorm, kEps, std::span<float>(cat).last(D));
    w.eh_proj(cat, z);
    host::rmsnorm(z, &w.input_ln, kEps, xin);
    w.q_a(xin, t1);
    host::rmsnorm(t1, &w.q_a_norm, kEps, qres);
    w.kv_a(xin, kv);
    host::rmsnorm(kv, &w.kv_a_norm, kEps, lat);
    host::dsa_append(c, lat, xin, w.idx);
    const unsigned t = c.tokens() - 1;
    std::vector<float> q(w.heads * kMlaDim), qI(kIdxHeads * kIdxDim), o(w.heads * kMlaDim), a(D), y(D), yn(D), sh(D),
        eo(static_cast<std::size_t>(kTopE) * D);
    w.q_b(qres, q);
    w.idx_wq_b(qres, qI);
    RefRow r;
    r.fresh = host::indexer_select(c, t, qI, xin, w.idx);
    if (pools) {
        r.tokens = *pools;
        for (unsigned pos = P0 * kKpool; pos <= t; ++pos) r.tokens.push_back(pos);
    } else {
        r.tokens = r.fresh;
    }
    host::mla_attention(c, r.tokens, q, w.kv_b, w.heads, o);
    w.o_proj(o, a);
    for (unsigned d = 0; d < D; ++d) y[d] = z[d] + a[d];
    host::rmsnorm(y, &w.post_ln, kEps, yn);
    const host::Route rt = host::route(yn, w.router, w.router_bias);
    for (unsigned k = 0; k < kTopE; ++k) w.expert(rt.id[k], yn, std::span<float>(&eo[static_cast<std::size_t>(k) * D], D));
    w.shared(yn, sh);
    host::moe_combine(rt, eo, sh, a);
    for (unsigned d = 0; d < D; ++d) y[d] = y[d] + a[d];
    r.h.resize(D);
    host::rmsnorm(y, &w.head_norm, kEps, r.h);
    std::vector<float> logits(w.vocab_n);
    w.head(r.h, logits);
    r.top = host::argmax_local(logits, w.vocab0).id;
    return r;
}

// MTP cycle protocol (design notes 4.2): at a cycle with committed trunk positions [0, p) (MTP pairs 0 .. p-1 true), after
// accepting m drafts the MTP cache is truncated to p, the m + 1 true pairs (h_j, x_{j+1}), j = p .. p+m, are appended as
// one catch-up pass, then K' - 1 draft steps (h', d') are appended (approximate rows, replaced next cycle). Required:
// after every catch-up the MTP cache == a straight run over the true pairs and the catch-up outputs == the straight run's
// rows bitwise; zero, partial and full acceptance, across pool boundaries and beyond 2,051 rows, sharing on and off.
void test_mtp_cycles(std::uint64_t seed, bool share_on, unsigned T) {
    Rng r(seed);
    const unsigned D = kHidden, V = 1000;
    std::vector<std::uint16_t> nrm(D), qan(kQLora), kvan(kKvLora), kvb(static_cast<std::size_t>(2 * kMlaDim) * kKvLora),
        router(static_cast<std::size_t>(kExperts) * D), ik(static_cast<std::size_t>(kIdxDim) * D), ig(static_cast<std::size_t>(kIdxDim) * D),
        iwp(static_cast<std::size_t>(kIdxHeads) * D), ikw(kIdxDim), ikb(kIdxDim);
    for (auto& x : nrm) x = float_to_bf16(1.0F + r.u(0.1F));
    for (auto& x : qan) x = float_to_bf16(1.0F + r.u(0.1F));
    for (auto& x : kvan) x = float_to_bf16(1.0F + r.u(0.1F));
    for (auto& x : kvb) x = to_f16(r.u(0.05F));
    for (auto& x : router) x = to_f16(r.u(0.02F));
    for (auto& x : ik) x = to_f16(r.u(0.02F));
    for (auto& x : ig) x = to_f16(r.u(0.02F));
    for (auto& x : iwp) x = to_f16(r.u(0.02F));
    for (auto& x : ikw) x = to_f16(1.0F);
    for (auto& x : ikb) x = to_f16(r.u(0.1F));
    std::vector<float> bias(kExperts), ape(static_cast<std::size_t>(kKpool) * kIdxDim), emb(static_cast<std::size_t>(V) * D);
    for (auto& x : bias) x = r.u(0.01F);
    for (auto& x : ape) x = r.u(0.5F);
    for (auto& x : emb) x = r.u(1.0F);
    auto mix = [](unsigned salt) {
        return [salt](std::span<const float> x, std::span<float> y) {
            const std::size_t n = x.size();
            for (std::size_t o = 0; o < y.size(); ++o) {
                float acc = 0.0F;
                for (unsigned k = 0; k < 8; ++k) {
                    const std::size_t i = (o * 2654435761ULL + k * 40503ULL + salt * 97ULL) % n;
                    acc = __builtin_fmaf(x[i], (k & 1U) ? 0.125F : -0.0625F, acc);
                }
                y[o] = acc;
            }
        };
    };
    MtpWeights w;
    w.embed = [&](std::uint32_t t, std::span<float> e) { std::copy_n(emb.begin() + static_cast<std::ptrdiff_t>(t) * D, D, e.begin()); };
    w.enorm = w.hnorm = w.input_ln = w.post_ln = w.head_norm = {nrm.data(), WType::BF16, D};
    w.q_a_norm = {qan.data(), WType::BF16, kQLora};
    w.kv_a_norm = {kvan.data(), WType::BF16, kKvLora};
    w.eh_proj = mix(1);
    w.q_a = mix(2);
    w.q_b = mix(3);
    w.kv_a = mix(4);
    w.o_proj = mix(5);
    w.idx_wq_b = mix(6);
    w.kv_b = {kvb.data(), WType::F16, 2 * kMlaDim, kKvLora};
    w.heads = 1;
    w.idx = {{ik.data(), WType::F16, kIdxDim, D}, {ig.data(), WType::F16, kIdxDim, D}, {iwp.data(), WType::F16, kIdxHeads, D},
             {ikw.data(), WType::F16, kIdxDim}, {ikb.data(), WType::F16, kIdxDim}, ape.data()};
    w.router = {router.data(), WType::F16, kExperts, D};
    w.router_bias = bias.data();
    w.expert = [&](std::uint32_t e, std::span<const float> x, std::span<float> y) { mix(100 + e)(x, y); };
    w.shared = mix(7);
    w.head = mix(8);
    w.vocab_n = V;
    std::vector<float> h(static_cast<std::size_t>(T) * D);  // "trunk" final-normed hiddens of positions 0 .. T-1
    for (auto& x : h) x = r.u(1.0F);
    std::vector<std::uint32_t> x(T + 1);
    for (auto& t : x) t = r.n(V);
    // straight run over the true pairs (h_j, x_{j+1}), j = 0 .. T-1
    DsaCache cs;
    std::vector<MtpOut> os;
    for (unsigned j = 0; j < T; ++j)
        os.push_back(mtp_forward(cs, std::span<const float>(&h[static_cast<std::size_t>(j) * D], D), std::span<const std::uint32_t>(&x[j + 1], 1), w)[0]);
    // cycles
    DsaCache cc;
    MtpShare sh;
    sh.on = share_on;
    const unsigned prompt = 41;  // bootstrap: prompt pairs 0 .. prompt-2 in chunks, then the first cycle starts at p = prompt-1
    unsigned p = 0;
    bool ok = true;
    while (p + 1 < prompt) {
        const unsigned R = std::min(1 + r.n(8), prompt - 1 - p);
        mtp_forward(cc, std::span<const float>(&h[static_cast<std::size_t>(p) * D], static_cast<std::size_t>(R) * D),
                    std::span<const std::uint32_t>(&x[p + 1], R), w, &sh, false);
        p += R;
    }
    unsigned cycles = 0, full = 0, zero = 0, cross = 0, steps = 0, ext_sparse = 0, ext_differ = 0;
    DsaCache cr = cc;  // the independent reference's own MTP cache
    while (p + 9 < T && ok) {
        const unsigned K = 1 + r.n(7), mode = r.n(3), m = mode == 0 ? 0 : (mode == 1 ? K : r.n(K + 1));
        cc.truncate(g_mutant == 4 ? std::min(cc.tokens(), p + m) : p);
        const auto oc = mtp_forward(cc, std::span<const float>(&h[static_cast<std::size_t>(p) * D], static_cast<std::size_t>(m + 1) * D),
                                    std::span<const std::uint32_t>(&x[p + 1], m + 1), w, &sh, false);
        ok = ok && caches_equal_prefix(cc, cs, p + m + 1);
        for (unsigned j = 0; j <= m && ok; ++j)
            ok = std::memcmp(oc[j].h.data(), os[p + j].h.data(), D * 4) == 0 && oc[j].top.id == os[p + j].top.id;
        // independent reference: catch-up rows with fresh selection, the last one's pools for the extensions
        cr.truncate(p);
        RefRow last;
        for (unsigned j = 0; j <= m; ++j)
            last = ref_mtp_row(cr, std::span<const float>(&h[static_cast<std::size_t>(p + j) * D], D), x[p + j + 1], w, nullptr, 0);
        const unsigned tl = p + m, P0 = (tl + 1) / kKpool;
        const std::vector<std::uint32_t> pools(last.tokens.begin(), last.tokens.begin() + static_cast<std::ptrdiff_t>(last.tokens.size() - (tl + 1 - P0 * kKpool)));
        std::vector<float> hp = oc[m].h;  // draft extension: K' - 1 steps from (h', d')
        std::uint32_t d = oc[m].top.id;
        std::vector<float> hr = last.h;
        std::uint32_t dr = last.top;
        const unsigned K2 = 1 + r.n(7);
        for (unsigned s = 1; s < K2 && ok; ++s) {
            auto od = mtp_forward(cc, hp, std::span<const std::uint32_t>(&d, 1), w, &sh, true);
            const RefRow rr = ref_mtp_row(cr, hr, dr, w, share_on ? &pools : nullptr, P0);
            ok = std::memcmp(od[0].h.data(), rr.h.data(), D * 4) == 0 && od[0].top.id == rr.top;
            ext_sparse += cc.tokens() / kKpool > kIdxPools;
            ext_differ += rr.tokens != rr.fresh;
            hp = od[0].h;
            d = od[0].top.id;
            hr = rr.h;
            dr = rr.top;
            ++steps;
        }
        ok = ok && caches_equal_prefix(cc, cr, cc.tokens());
        if ((p + K2 + m) / kKpool > (p + m + 1) / kKpool) ++cross;
        full += m == K;
        zero += m == 0;
        ++cycles;
        p += m + 1;
    }
    char b[400];
    std::snprintf(b, sizeof b, "mtp cycles (sharing %s): catch-up after truncation to p == straight run; every extension step == "
                               "independent reference (h', argmax, caches) (%u cycles, %u zero / %u full acceptance, %u extension steps, "
                               "%u sparse, %u where shared != fresh, %u crossing a pool, end %u)",
                  share_on ? "on" : "off", cycles, zero, full, steps, ext_sparse, ext_differ, cross, p);
    check(ok && zero > 0 && full > 0 && cross > 0 && p + 20 > T && (T < 2100 || (p > 2060 && ext_sparse > 0)) &&
              (!share_on || T < 2100 || ext_differ > 0),
          b);
}

}  // namespace

void test_adaptive() {
    auto table = seeded_cost_table();
    check(valid_cost_table(table), "adaptive measured seed is valid");
    float wire[11] = {};
    store_cost_table(table, wire);
    CostTable merged;
    check(load_cost_table(wire, 17, merged) && merged.version == 17 && merged.catchup_ms == 3.1F,
          "adaptive wire layout includes catch-up and version");
    Calibration calibration;
    check(calibration.probability(1, 0.5F) == 0.90F && calibration.probability(2, 0.5F) == 0.80F &&
          calibration.probability(3, 0.5F) == 0.70F, "calibration virtual priors");
    calibration.observe(1, 0.5F, false);
    check(calibration.probability(1, 0.5F) < 0.9F && calibration.probability(1, 0.1F) == 0.9F,
          "calibration updates only reached confidence bin");
    check(Calibration{}.probability(1, 0.5F) == 0.9F, "calibration resets per prompt");
    check(calibration.probability(2, -1) == calibration.probability(2, 0) &&
          calibration.probability(2, 2) == calibration.probability(2, 1), "confidence bins clamp");
    check(draft_limit(99, 100, 5, 100) == 7 && draft_limit(3, 1, 5, 100) == 0 &&
          draft_limit(3, 0, 5, 100) == 0 && draft_limit(3, 10, 99, 100) == 0 &&
          draft_limit(3, 10, 101, 100) == 0 && draft_limit(3, 3, 5, 100) == 2,
          "adaptive launch limits clamp budget, capacity, K and underflow");
    WindowEstimate w{1, 0.9F, 1.9F, table.verify_ms[2] + table.catchup_ms};
    check(window_extend(table, w, 0.8F, 3), "adaptive extension uses current predicted throughput");
    check(!window_extend(table, w, 0.0F, 3) && !window_extend(table, w, 1, 1), "adaptive closes at low survival or Kmax");
    // Two deliberately unrelated local clocks/publishers; decisions consume only a returned rank-0 packet.
    auto local0 = table, local1 = table;
    bool identical = true;
    for (unsigned i = 0; i < 32; ++i) {
        observe_cost(local0, 2, 40 + i);
        observe_cost(local1, 2, 4000 + 20 * i);
        store_cost_table(local0, wire);
        CostTable a, b;
        identical = identical && load_cost_table(wire, local0.version, a) && load_cost_table(wire, local0.version, b);
        identical = identical && window_extend(a, w, 0.8F, 3) == window_extend(b, w, 0.8F, 3);
    }
    check(identical && local0.verify_ms[2] != local1.verify_ms[2], "identical merged inputs ignore differing local timings");
    {  // a window that only verifies R = 4 keeps R = 3 one slope below it (no stale-seed K lock-in)
        auto t = seeded_cost_table();
        for (unsigned i = 0; i < 64; ++i) observe_cost(t, 4, 60.0);
        const float d = t.verify_ms[4] - t.verify_ms[3];
        check(t.verify_ms[4] < 61.0F && d > 9.0F && d < 11.0F && t.verify_ms[1] < t.verify_ms[3],
              "unsampled verify rows follow the sampled row by the seed slope");
        for (unsigned i = 0; i < 64; ++i) observe_cost(t, 3, 52.0);
        const float d2 = t.verify_ms[4] - t.verify_ms[3];
        check(d2 > 7.0F && d2 < 9.0F && std::fabs((t.verify_ms[3] - t.verify_ms[2]) - d2) < 0.01F,
              "two sampled rows set the slope for the unsampled ones");
        // chat-like acceptance (n2 traces: survival to depth 3 ~0.61, conditional 0.63) closes at K2 under the
        // measured table; code-like (0.85 / 0.86) extends to K3.
        const auto m = seeded_cost_table();
        const float c2 = m.verify_ms[3] + m.step_ms + m.catchup_ms;
        check(!window_extend(m, WindowEstimate{2, 0.61F, 1.0F + 0.84F + 0.61F, c2}, 0.63F, 3) &&
                  window_extend(m, WindowEstimate{2, 0.85F, 1.0F + 0.92F + 0.85F, c2}, 0.86F, 3),
              "measured table: chat stops at K2, code extends to K3");
    }
    observe_cost(table, 2, 1000);
    check(valid_cost_table(table) && table.verify_ms[8] >= table.verify_ms[2], "EMA publication enforces monotone verify costs");
    store_cost_table(table, wire);
    wire[9] = 0;
    check(!load_cost_table(wire, 99, merged), "invalid merged table rejected for fixed-K fallback");
    wire[9] = std::numeric_limits<float>::quiet_NaN();
    check(!load_cost_table(wire, 99, merged), "NaN table rejected");
    store_cost_table(table, wire);
    wire[3] = wire[2] - 1;
    check(!load_cost_table(wire, 99, merged), "nonmonotone merged table rejected");
}


// Retained CommExchange inline arithmetic: an independent oracle for the extracted helper.
void old_inline_merge(std::span<MergeRow> rows, MergeCtl& ctl, std::span<const MergeRow> peer,
                      const MergeCtl& pc, std::uint32_t rank) {
    const MergeCtl& c0 = rank == 0 ? ctl : pc;
    const MergeCtl& c1 = rank == 0 ? pc : ctl;
    MergeCtl out = c0;
    out.replay_flag = c0.replay_flag | c1.replay_flag;
    out.mtp_max = std::max(c0.mtp_max, c1.mtp_max);
    out.mtp_sum = c0.mtp_sum * std::exp(c0.mtp_max - out.mtp_max) + c1.mtp_sum * std::exp(c1.mtp_max - out.mtp_max);
    ctl = out;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const MergeRow p = peer[r];
        const MergeRow& a = rank == 0 ? rows[r] : p;
        const MergeRow& b = rank == 0 ? p : rows[r];
        MergeRow m = (b.value > a.value || (b.value == a.value && b.id < a.id)) ? b : a;
        m.flag = a.flag | b.flag;
        rows[r] = m;
    }
}

void test_merge_rank_order() {
    // All candidates are NaN-free; infinities are excluded here too (head checkpoint owns C5).
    const std::array<MergeRow, 8> r0{{{1, 9, 1}, {2, 8, 2}, {3, 7, 4}, {0.0F, 5, 8},
                                    {-0.0F, 6, 16}, {-4, 3, 32}, {0.0F, 2, 64}, {-0.0F, 1, 128}}};
    const std::array<MergeRow, 8> r1{{{2, 10, 2}, {2, 7, 4}, {3, 8, 8}, {-0.0F, 5, 16},
                                    {0.0F, 6, 32}, {-5, 2, 64}, {-0.0F, 1, 128}, {0.0F, 2, 256}}};
    const std::array<unsigned, 8> ids{10, 7, 7, 5, 6, 3, 1, 1};
    const std::array<float, 6> maxima{-0.0F, 0.0F, -100.25F, 13.125F, 0.0001F, 88.0F};
    for (unsigned n = 1; n <= 8; ++n) for (float m0 : maxima) for (float m1 : maxima) {
        MergeCtl c0{}, c1{};
        c0.cycle = c1.cycle = 17;
        c0.step = c1.step = 3;
        c0.kind = c1.kind = 2;
        c0.replay_flag = 2; c1.replay_flag = 8;
        c0.table_version = 19; c1.table_version = 99;
        for (unsigned j = 0; j < 11; ++j) { c0.table[j] = static_cast<float>(j + 1); c1.table[j] = -1; }
        c0.mtp_max = m0; c1.mtp_max = m1;
        c0.mtp_sum = 1.234567F; c1.mtp_sum = 17.76543F;
        auto expected = r0;
        auto ec = c0;
        old_inline_merge({expected.data(), n}, ec, {r1.data(), n}, c1, 0);
        for (std::uint32_t rank = 0; rank < 2; ++rank) {
            auto mine = rank == 0 ? r0 : r1;
            const auto& peer = rank == 0 ? r1 : r0;
            auto ctl = rank == 0 ? c0 : c1;
            const auto& pc = rank == 0 ? c1 : c0;
            auto oracle = mine;
            auto oc = ctl;
            old_inline_merge({oracle.data(), n}, oc, {peer.data(), n}, pc, rank);
            merge_rank_order({mine.data(), n}, ctl, {peer.data(), n}, pc, rank);
            check(std::memcmp(mine.data(), oracle.data(), n * sizeof(MergeRow)) == 0, "merge row inline oracle");
            check(std::memcmp(&ctl, &oc, sizeof(ctl)) == 0, "merge ctl inline oracle / sum bits");
            check(std::memcmp(mine.data(), expected.data(), n * sizeof(MergeRow)) == 0, "merge both rank orientations");
            check(std::memcmp(&ctl, &ec, sizeof(ctl)) == 0, "merge rank agreement / table ownership");
            check(ctl.replay_flag == 10 && ctl.table_version == 19, "merge replay OR / rank-0 version");
            for (unsigned j = 0; j < n; ++j) {
                check(std::isfinite(mine[j].value), "merge NaN-free candidates");
                check(mine[j].id == ids[j] && mine[j].flag == (r0[j].flag | r1[j].flag), "merge R18 / row OR");
            }
            if (n >= 5) {
                check(std::bit_cast<std::uint32_t>(mine[3].value) == 0, "merge +0 rank-0 tie");
                check(std::bit_cast<std::uint32_t>(mine[4].value) == 0x80000000U, "merge -0 rank-0 tie");
            }
            for (unsigned field = 0; field < 3; ++field) {
                auto bad = pc;
                if (field == 0) ++bad.cycle;
                if (field == 1) ++bad.step;
                if (field == 2) ++bad.kind;
                auto untouched = mine;
                auto before = ctl;
                bool threw = false;
                try { merge_rank_order({mine.data(), n}, ctl, {peer.data(), n}, bad, rank); }
                catch (const std::runtime_error&) { threw = true; }
                check(threw, "merge control mismatch throws");
                check(std::memcmp(&before, &ctl, sizeof(ctl)) == 0 &&
                      std::memcmp(mine.data(), untouched.data(), sizeof(mine)) == 0, "mismatch leaves output untouched");
            }
        }
    }
}

int main(int argc, char** argv) {
    const unsigned total = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 2300;
    const std::uint64_t seed = argc > 2 ? std::stoull(argv[2]) : 7;
    g_mutant = argc > 3 ? std::stoi(argv[3]) : 0;
    try {
        test_merge_rank_order();
        test_adaptive();
        test_accept();
        test_equivalence(total, seed);
        test_mtp(seed + 1);
        if (g_mutant == 0 || g_mutant == 4) {
            test_mtp_cycles(seed + 2, true, 2200);
            test_mtp_cycles(seed + 3, false, 600);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 2;
    }
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
