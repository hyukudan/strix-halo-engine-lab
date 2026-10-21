// deepseek41 decode attention on gfx1151: every kernel against its host reference, bit for bit, on synthetic data
// (window ring and compressed caches written over many verify windows of 1..8 rows, index scores, top-k with the
// candidate pool, attention with window-only and compressed visible sets). STRIX_DS41_ATTN_BENCH=1 adds timings.
#include "strix/hip_ds41_attention.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace a = strix::ds41::attn;
namespace g = strix::ds41::attn::hip;

namespace {
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what.c_str()); }
}
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template <class T> struct Dev {
    T* p = nullptr; std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) { check(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T))); check(hipMemset(p, 0, std::max<std::size_t>(count, 1) * sizeof(T))); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { put(h); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    void put(const std::vector<T>& h) { check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> h(n); check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
};
std::mt19937 rng(4141);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
template <class T> bool same(const std::vector<T>& x, const std::vector<T>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}
std::vector<a::RopeRow> ropes(a::RopeKind k, const std::vector<std::uint64_t>& pos, unsigned ratio = 0) {
    std::vector<a::RopeRow> r;
    for (const auto p : pos) r.push_back(a::rope_row(k, ratio == 0 ? p : p / ratio * ratio));
    return r;
}
std::vector<std::uint16_t> random_f16(std::size_t n) {
    std::vector<std::uint16_t> v(n);
    for (auto& h : v) h = static_cast<std::uint16_t>(0x1C00U + (rng() & 0xFFFU) + ((rng() & 1U) << 15U));
    return v;
}
std::vector<float> random_f(std::size_t n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = uni(lo, hi); return v; }

// Window ring and compression over 0..npos-1 in verify windows of 1..8 rows; host and device caches equal.
void caches(unsigned ratio, unsigned npos) {
    const unsigned cap = a::kWindow + a::kMaxRows;
    const auto norm = random_f(512, 0.5F, 1.5F), cnorm = random_f(512, 0.5F, 1.5F), knorm = random_f(128, 0.5F, 1.5F);
    const auto wk = random_f16(128 * 512);
    std::vector<std::uint8_t> ring(cap * a::kWindowEntryBytes), kv(npos * a::kKvEntryBytes), ix(npos * a::kIdxEntryBytes);
    Dev<std::uint8_t> dring(ring.size()), dkv(kv.size()), dix(ix.size());
    Dev<float> dnorm(norm), dcnorm(cnorm), dknorm(knorm), dstate_a(1024), dstate_b(1024);
    Dev<std::uint16_t> dwk(wk);
    Dev<std::uint32_t> dvalid_a(1), dvalid_b(1);
    a::CompressState state;
    std::uint64_t p = 0;
    bool flip = false;
    for (unsigned step = 0; p < npos; ++step) {
        const unsigned rows = std::min<unsigned>(1U + (step * 5U) % 8U, static_cast<unsigned>(npos - p));
        std::vector<std::uint64_t> pos(rows);
        for (unsigned r = 0; r < rows; ++r) pos[r] = p + r;
        const auto kvp = random_f(rows * 512, -3, 3), sc = random_f(rows * 512, -4, 4);
        a::window_write({kvp, norm, pos, cap, a::RopeKind::compressed}, ring);
        a::compress({ratio, kvp, sc, cnorm, wk, knorm, pos}, state, kv, ix);
        Dev<float> dkvp(kvp), dsc(sc);
        Dev<std::uint64_t> dpos(pos);
        Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos)), dcrope(ropes(a::RopeKind::compressed, pos, ratio));
        g::enqueue_window_write({dkvp.p, dnorm.p, drope.p, dpos.p, rows, cap, dring.p}, nullptr);
        auto& sin = flip ? dstate_b : dstate_a; auto& sout = flip ? dstate_a : dstate_b;
        auto& vin = flip ? dvalid_b : dvalid_a; auto& vout = flip ? dvalid_a : dvalid_b;
        g::enqueue_compress({ratio, rows, dkvp.p, dsc.p, dcnorm.p, dwk.p, dknorm.p, dcrope.p, dpos.p, sin.p, vin.p, sout.p, vout.p, dkv.p, dix.p}, nullptr);
        check(hipDeviceSynchronize());
        if (ratio == 2) {
            const auto vs = vout.get();
            expect((vs[0] != 0U) == state.valid, "compress state valid flag");
        }
        flip = !flip;
        p += rows;
    }
    expect(same(dring.get(), ring), "window ring bytes, ratio " + std::to_string(ratio));
    expect(same(dkv.get(), kv), "compressed KV bytes, ratio " + std::to_string(ratio));
    expect(same(dix.get(), ix), "index key bytes, ratio " + std::to_string(ratio));
    std::printf("caches ratio %u, %u positions: ring/KV/index keys host == device\n", ratio, npos);
}

std::vector<std::uint8_t> random_idx_cache(std::size_t n) {
    std::vector<std::uint8_t> c(n * a::kIdxEntryBytes);
    for (std::size_t t = 0; t < n; ++t) {
        for (unsigned i = 0; i < 64; ++i) c[t * 68 + i] = static_cast<std::uint8_t>(rng());
        for (unsigned b = 0; b < 4; ++b) c[t * 68 + 64 + b] = static_cast<std::uint8_t>(122 + rng() % 6);
    }
    return c;
}

// Index scores and selection for rows ending at `last`, every pool mode; host == device.
void index_and_select(std::uint64_t last, unsigned rows, unsigned ratio, unsigned topk, unsigned cand, unsigned bs, bool bench) {
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = last - (rows - 1) + r;
    const std::uint32_t max_count = static_cast<std::uint32_t>((last + 1) / ratio);
    const auto cache = random_idx_cache(max_count);
    const auto q = random_f(rows * 32 * 128, -2, 2), w = random_f(rows * 32, -1, 1);
    std::vector<std::vector<float>> scores;
    a::index_scores({q, w, pos, ratio, a::RopeKind::compressed}, cache, scores);
    Dev<std::uint8_t> dcache(cache);
    Dev<float> dq(q), dw(w), dscores(std::size_t{rows} * max_count);
    Dev<std::uint64_t> dpos(pos);
    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos));
    Dev<std::uint8_t> dprep(std::size_t{rows} * g::kIdxPrepBytes);
    const g::IndexView iv{rows, ratio, max_count, dq.p, dw.p, drope.p, dpos.p, dcache.p, dscores.p, dprep.p};
    g::enqueue_index_scores(iv, nullptr);
    check(hipDeviceSynchronize());
    const auto ds = dscores.get();
    bool ok = true;
    for (unsigned r = 0; r < rows; ++r) ok = ok && std::memcmp(&ds[std::size_t{r} * max_count], scores[r].data(), scores[r].size() * 4) == 0;
    expect(ok, "index scores rows " + std::to_string(rows) + " count " + std::to_string(max_count));
    const std::uint32_t nb = (max_count + bs - 1) / bs;
    Dev<std::uint8_t> dcand(std::size_t{rows} * nb);
    Dev<std::uint32_t> dkeys(std::size_t{rows} * max_count), didx(std::size_t{rows} * topk), dcnt(rows);
    std::vector<std::vector<std::uint8_t>> cands(rows, std::vector<std::uint8_t>(nb, 0));
    for (const auto pool : {a::Pool::candidate_source, a::Pool::candidate_user, a::Pool::plain}) {
        const g::SelectView sv{rows, ratio, max_count, topk, cand, bs, pool, dpos.p, dscores.p, dcand.p, dkeys.p, didx.p, dcnt.p};
        g::enqueue_select(sv, nullptr);
        check(hipDeviceSynchronize());
        const auto di = didx.get(), dc = dcnt.get();
        const auto dm = dcand.get();
        bool sel_ok = true, cand_ok = true;
        for (unsigned r = 0; r < rows; ++r) {
            const auto mine = a::select_topk(scores[r], pool, cands[r], topk, cand, bs);
            sel_ok = sel_ok && dc[r] == mine.size() && std::equal(mine.begin(), mine.end(), &di[std::size_t{r} * topk]);
            if (pool == a::Pool::candidate_source) {
                const std::uint32_t nbr = (static_cast<std::uint32_t>(scores[r].size()) + bs - 1) / bs;
                cand_ok = cand_ok && std::equal(cands[r].begin(), cands[r].begin() + nbr, &dm[std::size_t{r} * nb]);
            }
        }
        const char* name = pool == a::Pool::plain ? "plain" : pool == a::Pool::candidate_source ? "source" : "user";
        expect(sel_ok, std::string("select ") + name + " count " + std::to_string(max_count));
        if (pool == a::Pool::candidate_source) expect(cand_ok, "candidate blocks count " + std::to_string(max_count));
    }
    std::printf("index scores + select: rows %u, count up to %u, topk %u, candidates %u x %u: host == device\n", rows, max_count, topk, cand, bs);
    if (!bench) return;
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    const auto time = [&](auto f) {
        f(); check(hipDeviceSynchronize());
        std::vector<float> ms;
        for (int i = 0; i < 30; ++i) { check(hipEventRecord(e0)); f(); check(hipEventRecord(e1)); check(hipEventSynchronize(e1)); float t; check(hipEventElapsedTime(&t, e0, e1)); ms.push_back(t); }
        std::sort(ms.begin(), ms.end()); return ms[15];
    };
    const float ts = time([&] { g::enqueue_index_scores(iv, nullptr); });
    const g::SelectView sv{rows, ratio, max_count, topk, cand, bs, a::Pool::candidate_source, dpos.p, dscores.p, dcand.p, dkeys.p, didx.p, dcnt.p};
    const float tsel = time([&] { g::enqueue_select(sv, nullptr); });
    const double bytes = double(max_count) * a::kIdxEntryBytes;  // keys read once per call (rows share them via L2)
    std::printf("  bench count %u rows %u: index scores %.1f us (%.0f GB/s of index keys), select (source) %.1f us\n",
        max_count, rows, ts * 1e3, bytes / (ts * 1e-3) / 1e9, tsel * 1e3);
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
}

// Attention with random caches: window-only and compressed visible sets, 1..8 rows.
void attention(std::uint64_t last, unsigned rows, unsigned nc, bool bench) {
    // compressed keys: 4096, or every position of a long context (selected indices up to the last position, ratio 1)
    const unsigned cap = a::kWindow + a::kMaxRows, ncache = last >= 65536U ? static_cast<unsigned>(last + 1U) : 4096U;
    std::vector<std::uint8_t> ring(cap * a::kWindowEntryBytes), kv(ncache * a::kKvEntryBytes);
    for (std::size_t s = 0; s < cap; ++s) {
        for (unsigned d = 0; d < 512; ++d) { std::uint8_t c; do c = static_cast<std::uint8_t>(rng()); while ((c & 0x7FU) == 0x7FU); ring[s * 528 + d] = c; }
        for (unsigned b = 0; b < 16; ++b) ring[s * 528 + 512 + b] = static_cast<std::uint8_t>(118 + rng() % 4);
    }
    for (std::size_t t = 0; t < ncache; ++t) {
        for (unsigned i = 0; i < 256; ++i) kv[t * 288 + i] = static_cast<std::uint8_t>(rng());
        for (unsigned g2 = 0; g2 < 32; ++g2) kv[t * 288 + 256 + g2] = static_cast<std::uint8_t>(0x30 + rng() % 16);
    }
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = last - (rows - 1) + r;
    const auto q = random_f(std::size_t{rows} * 64 * 512, -0.5F, 0.5F), sink = random_f(64, -2, 2);
    std::vector<std::uint32_t> idx(std::size_t{rows} * a::kTopK), cnt(rows, nc);
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> pick(ncache);
        for (std::uint32_t i = 0; i < ncache; ++i) pick[i] = i;
        std::shuffle(pick.begin(), pick.end(), rng);
        std::sort(pick.begin(), pick.begin() + nc);
        std::copy(pick.begin(), pick.begin() + nc, &idx[std::size_t{r} * a::kTopK]);
    }
    std::vector<float> o(std::size_t{rows} * 64 * 512);
    a::attention({q, sink, pos, cap, a::RopeKind::compressed, idx, nc != 0 ? std::span<const std::uint32_t>(cnt) : std::span<const std::uint32_t>{}}, ring, kv, o);
    Dev<std::uint8_t> dring(ring), dkv(kv);
    Dev<float> dq(q), dsink(sink), dpart(std::size_t{rows} * 64 * a::kMaxChunks * 514), dout(o.size());
    Dev<std::uint64_t> dpos(pos);
    Dev<std::uint32_t> didx(idx), dcnt(cnt);
    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos));
    const g::AttentionView av{rows, cap, a::kTopK, dq.p, dsink.p, drope.p, dpos.p, dring.p, dkv.p, didx.p, nc != 0 ? dcnt.p : nullptr, dpart.p, dout.p};
    g::enqueue_attention(av, nullptr);
    check(hipDeviceSynchronize());
    expect(same(dout.get(), o), "attention rows " + std::to_string(rows) + " pos " + std::to_string(last) + " compressed " + std::to_string(nc));
    // fused combine (tickets): twice, the second call relies on the tickets the first left at zero
    Dev<std::uint32_t> dtick(std::vector<std::uint32_t>(std::size_t{rows} * g::kAttnGroupCount, 0U));
    g::AttentionView avf = av;
    avf.tickets = dtick.p;
    for (int rep = 0; rep < 2; ++rep) {
        check(hipMemset(dout.p, 0xFF, o.size() * sizeof(float)));
        g::enqueue_attention(avf, nullptr);
        check(hipDeviceSynchronize());
        expect(same(dout.get(), o), "attention (fused combine) rows " + std::to_string(rows) + " pos " + std::to_string(last) + " compressed " + std::to_string(nc));
    }
    expect(dtick.get() == std::vector<std::uint32_t>(std::size_t{rows} * g::kAttnGroupCount, 0U), "attention tickets left at zero");
    {  // the combine's A8 of o: quantize_a8's bits (int8 per 32: delta = amax / 127, rint, clamp, exact sum)
        Dev<std::int8_t> dcodes(o.size());
        Dev<float> ddeltas(o.size() / 32U);
        Dev<std::int32_t> dsums(o.size() / 32U);
        g::AttentionView avq = avf;
        avq.codes = dcodes.p; avq.deltas = ddeltas.p; avq.sums = dsums.p;
        g::enqueue_attention(avq, nullptr);
        check(hipDeviceSynchronize());
        std::vector<std::int8_t> codes(o.size());
        std::vector<float> deltas(o.size() / 32U);
        std::vector<std::int32_t> sums(o.size() / 32U);
        for (std::size_t gi = 0; gi < deltas.size(); ++gi) {
            float amax = 0.0F;
            for (unsigned i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(o[gi * 32U + i]));
            const bool tiny = amax > 0.0F && amax < 0x1p-119F;
            const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
            int sum = 0;
            for (unsigned i = 0; i < 32; ++i) {
                const float rounded = tiny ? 0.0F : std::nearbyint(o[gi * 32U + i] / delta);
                const int c = static_cast<int>(std::min(127.0F, std::max(-127.0F, rounded)));
                codes[gi * 32U + i] = static_cast<std::int8_t>(c);
                sum += c;
            }
            deltas[gi] = delta;
            sums[gi] = sum;
        }
        expect(same(dout.get(), o) && same(dcodes.get(), codes) && same(ddeltas.get(), deltas) && same(dsums.get(), sums),
               "attention combine A8 = quantize_a8 of o, rows " + std::to_string(rows));
    }
    // tensor-parallel rank 1: heads 32..63 only (q, sink and o of those heads, contiguous)
    {
        std::vector<float> qh(std::size_t{rows} * 32 * 512), oh(qh.size());
        for (unsigned r = 0; r < rows; ++r) {
            std::copy_n(&q[(std::size_t{r} * 64 + 32) * 512], 32 * 512, &qh[std::size_t{r} * 32 * 512]);
            std::copy_n(&o[(std::size_t{r} * 64 + 32) * 512], 32 * 512, &oh[std::size_t{r} * 32 * 512]);
        }
        Dev<float> dqh(qh), douth(oh.size());
        g::AttentionView avh = avf;
        avh.q = dqh.p; avh.sink = dsink.p + 32; avh.o = douth.p; avh.heads = 32;
        g::enqueue_attention(avh, nullptr);
        check(hipDeviceSynchronize());
        expect(same(douth.get(), oh), "attention heads 32..63 rows " + std::to_string(rows));
    }
    std::printf("attention rows %u last pos %llu compressed %u: host == device (combine kernel and fused)\n", rows, static_cast<unsigned long long>(last), nc);
    if (!bench) return;
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    g::enqueue_attention(avf, nullptr); check(hipDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < 30; ++i) { check(hipEventRecord(e0)); g::enqueue_attention(avf, nullptr); check(hipEventRecord(e1)); check(hipEventSynchronize(e1)); float t; check(hipEventElapsedTime(&t, e0, e1)); ms.push_back(t); }
    std::sort(ms.begin(), ms.end());
    const unsigned nw = static_cast<unsigned>(std::min<std::uint64_t>(last + 1, a::kWindow));
    const double bytes = double(rows) * (nw * double(a::kWindowEntryBytes) + nc * double(a::kKvEntryBytes)) + double(rows) * 64 * 512 * 4 * 2;
    std::printf("  bench: %.1f us per call (%.0f GB/s of keys + q + o)\n", ms[15] * 1e3, bytes / (ms[15] * 1e-3) / 1e9);
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
}

}  // namespace

int main() {
    try {
        const bool bench = std::getenv("STRIX_DS41_ATTN_BENCH") != nullptr;
        caches(2, 333);
        caches(1, 300);
        g::set_compress_ahead(1);  // STRIX_DS41_COMPRESS_AHEAD: the same caches
        caches(2, 333);
        caches(1, 300);
        g::set_compress_ahead(-1);
        index_and_select(99, 1, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);       // count <= topk
        index_and_select(1401, 3, 2, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);     // ratio 2, 701 positions
        index_and_select(20007, 8, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);    // 2501 blocks > 2048
        index_and_select(4000, 2, 1, 64, 8, 8, false);                                          // small pool parameters
        index_and_select(9000, 2, 1, 64, 8, 8, false);                                          // two-level selection, small k
        index_and_select(4600, 3, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);     // two slices, the last short
        attention(5, 1, 0, false);       // window only, partial window
        attention(1000, 3, 0, false);
        attention(1000, 1, 300, false);
        attention(20, 2, 37, false);     // partial window and compressed keys in one tile
        attention(5000, 8, a::kTopK, false);
        // long context: 512K positions (ratio 1: 512K index keys, 128 select slices; ratio 2: 256K), attention at 1M
        index_and_select(524287, 1, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);
        index_and_select(524287, 2, 2, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);
        attention(1048575, 3, a::kTopK, false);
        if (std::getenv("STRIX_DS41_ATTN_LONG") != nullptr) {  // minutes of host reference: 1M positions, 8-row windows
            index_and_select(1048575, 8, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);
            index_and_select(1048575, 8, 2, a::kTopK, a::kCandBlocks, a::kCandBlockSize, false);
            attention(1048575, 8, a::kTopK, false);
        }
        if (bench) {
            for (const std::uint64_t n : {32767ULL, 131071ULL, 262143ULL, 524287ULL}) for (const unsigned rows : {1U, 8U})
                index_and_select(n, rows, 1, a::kTopK, a::kCandBlocks, a::kCandBlockSize, true);
            for (const unsigned rows : {1U, 4U, 8U}) attention(100000, rows, a::kTopK, true);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", failures == 0 ? "ds41 attention device tests pass" : "FAILED");
    return failures == 0 ? 0 : 1;
}
