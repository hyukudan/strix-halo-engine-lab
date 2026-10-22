// deepseek41 decode attention on gfx1151: every kernel against its host reference, bit for bit, on synthetic data
// (window ring and compressed caches written over many verify windows of 1..8 rows, index scores, top-k with the
// candidate pool, attention with window-only and compressed visible sets) and the compress kernel's fast form against
// its default, and the segment launches against the per-segment calls. STRIX_DS41_ATTN_BENCH=1 adds timings.
#include "strix/hip_ds41_attention.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
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
void caches(unsigned ratio, unsigned npos, int compress_fast = 0) {
    g::set_compress_fast(compress_fast);
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
    g::set_compress_fast(-1);
    std::printf("caches ratio %u, %u positions%s: ring/KV/index keys host == device\n", ratio, npos, compress_fast != 0 ? " (compress fast)" : "");
}

// Compress fast (set_compress_fast(1)) against the default kernel, bit for bit (KV and index caches, carry state and
// flag): ratio 1 and 2, rows 1..8, several first positions and carry flags; weights with F16 subnormals and signed
// zeros, and in one set infinities and NaNs (quiet, signalling, negative) with far-apart, infinite and NaN scores.
// STRIX_DS41_ATTN_BENCH=1 adds us per call at rows 1, 2, 6, 8: cold (a 256 MB read before each call, median of 15)
// and warm (back to back, the mean of 50).
__global__ void flush_kernel(const uint4* p, std::size_t n, unsigned* sink) {
    uint4 acc{0, 0, 0, 0};
    for (std::size_t i = blockIdx.x * 256ULL + threadIdx.x; i < n; i += 256ULL * gridDim.x) {
        const uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x & acc.y & acc.z & acc.w) == 0xFFFFFFFFU) *sink = 1U;
}
template <class F> double cold_us(F call) {
    Dev<std::uint8_t> flush(std::size_t{256} << 20U);
    Dev<unsigned> sink(1);
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    std::vector<double> t;
    for (int i = 0; i < 15; ++i) {
        hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, nullptr, reinterpret_cast<const uint4*>(flush.p), flush.n / 16U, sink.p);
        check(hipEventRecord(e0)); call(); check(hipEventRecord(e1)); check(hipEventSynchronize(e1));
        float ms; check(hipEventElapsedTime(&ms, e0, e1)); t.push_back(1e3 * ms);
    }
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
template <class F> double warm_us(F call) {
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0)); check(hipEventCreate(&e1));
    call();
    check(hipEventRecord(e0));
    for (int i = 0; i < 50; ++i) call();
    check(hipEventRecord(e1)); check(hipEventSynchronize(e1));
    float ms; check(hipEventElapsedTime(&ms, e0, e1));
    check(hipEventDestroy(e0)); check(hipEventDestroy(e1));
    return 1e3 * ms / 50.0;
}
template <class T> std::vector<std::uint8_t> bytes_of(const std::vector<T>& v) {
    std::vector<std::uint8_t> b(v.size() * sizeof(T));
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}
void compress_fast(bool bench) {
    const auto norm = random_f(512, 0.5F, 1.5F), knorm = random_f(128, 0.5F, 1.5F);
    Dev<float> dnorm(norm), dknorm(knorm);
    auto wk = random_f16(128 * 512);
    for (std::size_t i = 0; i < wk.size(); i += 37) wk[i] = static_cast<std::uint16_t>((rng() & 0x3FFU) | ((rng() & 1U) << 15U));  // subnormals, +-0
    wk[5] = 0x0000U; wk[9] = 0x8000U; wk[700] = 0x0001U; wk[701] = 0x83FFU;
    auto wk_special = wk;   // rows 3, 40, 41, 77, 127: +inf, -inf, quiet NaN, signalling NaN, negative NaN
    wk_special[3 * 512 + 100] = 0x7C00U; wk_special[40 * 512 + 511] = 0xFC00U; wk_special[41 * 512 + 0] = 0x7E00U;
    wk_special[77 * 512 + 300] = 0x7C01U; wk_special[127 * 512 + 64] = 0xFE01U;
    Dev<std::uint16_t> dwk(wk), dwk_special(wk_special);
    unsigned cases = 0;
    for (const unsigned set : {0U, 1U})
        for (const unsigned ratio : {1U, 2U})
            for (std::uint32_t rows = 1; rows <= 8U; ++rows)
                for (const std::uint64_t first : {0ULL, 1ULL, 1000ULL, 1001ULL, 65535ULL, 500000ULL}) {
                    std::vector<std::uint64_t> pos(rows);
                    for (std::uint32_t r = 0; r < rows; ++r) pos[r] = first + r;
                    auto kvp = random_f(std::size_t{rows} * 512, -3.0F, 3.0F), sc = random_f(std::size_t{rows} * 512, -4.0F, 4.0F);
                    auto state = random_f(1024, -3.0F, 3.0F);
                    if (set == 1U) {   // exp underflow below -87, infinite and NaN scores
                        sc[3] = 95.0F; state[512 + 3] = -20.0F; sc[17] = -1e30F; state[512 + 17] = 0.0F;
                        sc[40] = std::numeric_limits<float>::infinity(); sc[41] = std::numeric_limits<float>::quiet_NaN();
                        if (rows > 1U) { sc[512 + 3] = -100.0F; sc[512 + 5] = 88.5F; sc[5] = -0.5F; }
                    }
                    const std::size_t entries = (first + rows) / ratio + 1U;
                    Dev<float> dkvp(kvp), dsc(sc), state_in(state), state_out(1024);
                    Dev<std::uint32_t> valid_in(std::vector<std::uint32_t>{static_cast<std::uint32_t>(first % 3U != 0U)}), valid_out(1);
                    Dev<std::uint64_t> dpos(pos);
                    Dev<a::RopeRow> drope(ropes(a::RopeKind::compressed, pos, ratio));
                    Dev<std::uint8_t> kvc(entries * a::kKvEntryBytes), idxc(entries * a::kIdxEntryBytes);
                    const g::CompressView v{ratio, rows, dkvp.p, dsc.p, dnorm.p, set == 1U ? dwk_special.p : dwk.p, dknorm.p, drope.p, dpos.p,
                                            state_in.p, valid_in.p, state_out.p, valid_out.p, kvc.p, idxc.p};
                    const auto run = [&](int mode) {
                        check(hipMemset(kvc.p, 0xA5, kvc.n)); check(hipMemset(idxc.p, 0xA5, idxc.n));
                        check(hipMemset(state_out.p, 0xFF, 4096)); check(hipMemset(valid_out.p, 0xFF, 4));
                        g::set_compress_fast(mode);
                        g::enqueue_compress(v, nullptr);
                        check(hipDeviceSynchronize());
                        return std::vector<std::vector<std::uint8_t>>{kvc.get(), idxc.get(), bytes_of(state_out.get()), bytes_of(valid_out.get())};
                    };
                    const auto base = run(0), got = run(1);
                    expect(base == got, "compress fast == default, set " + std::to_string(set) + " ratio " + std::to_string(ratio) + " rows " +
                                            std::to_string(rows) + " first " + std::to_string(first));
                    ++cases;
                    if (bench && set == 0U && first == 1001U && (rows == 1U || rows == 2U || rows == 6U || rows == 8U)) {
                        const auto call = [&] { g::enqueue_compress(v, nullptr); };
                        g::set_compress_fast(0);
                        const double c0 = cold_us(call), w0 = warm_us(call);
                        g::set_compress_fast(1);
                        const double c1 = cold_us(call), w1 = warm_us(call);
                        std::printf("  bench compress ratio %u R%u cold/warm us: default %.1f/%.1f -> fast %.1f/%.1f\n", ratio, rows, c0, w0, c1, w1);
                    }
                }
    g::set_compress_fast(-1);
    std::printf("compress fast == default kernel (KV, index keys, carry state), %u cases: ratio 1 and 2, rows 1..8, F16 specials\n", cases);
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

// Segment launches (enqueue_*_segments, BatchForward STRIX_DS41_BATCH_SEGMENTS=1) against the per-segment calls, bit
// for bit: segments of different rows, positions, ratios and selection paths (one block per row and two-level), every
// compress form (default, ahead, fast) and pool, attention with the combine kernel and with tickets (+ A8, 32 heads);
// every output buffer starts from the same pattern in both runs. More segments than kMaxSegments split the launches.
struct Seg { std::uint64_t last; unsigned rows, ratio; };
void segments() {
    const std::vector<Seg> segs = {{5, 1, 1}, {1000, 3, 2}, {20007, 8, 1}, {4600, 2, 1}, {9001, 1, 2}, {130, 4, 1},
                                   {64, 1, 2}, {3001, 2, 1}, {700, 5, 2}, {41, 1, 1}};   // 10 > kMaxSegments
    const unsigned cap = a::kWindow + a::kMaxRows;
    const auto norm = random_f(512, 0.5F, 1.5F), knorm = random_f(128, 0.5F, 1.5F);
    const auto wk = random_f16(128 * 512);
    Dev<float> dnorm(norm), dknorm(knorm);
    Dev<std::uint16_t> dwk(wk);
    struct Bufs {
        std::vector<std::uint64_t> pos;
        std::unique_ptr<Dev<std::uint64_t>> dpos;
        std::unique_ptr<Dev<a::RopeRow>> drope, dcrope;
        std::unique_ptr<Dev<float>> kv, sc, sin, sout, q, w, scores, aq, part, out, deltas;
        std::unique_ptr<Dev<std::uint32_t>> vin, vout, keys, idx, cnt, tick, aidx, acnt;
        std::unique_ptr<Dev<std::uint8_t>> ring, kvc, ixc, prep, cand, akv, aring;
        std::unique_ptr<Dev<std::int8_t>> codes;
        std::unique_ptr<Dev<std::int32_t>> sums;
        std::uint32_t max_count = 0;
    };
    std::vector<Bufs> b(segs.size());
    for (std::size_t k = 0; k < segs.size(); ++k) {
        const auto& s = segs[k];
        auto& x = b[k];
        for (unsigned r = 0; r < s.rows; ++r) x.pos.push_back(s.last - (s.rows - 1) + r);
        x.dpos = std::make_unique<Dev<std::uint64_t>>(x.pos);
        x.drope = std::make_unique<Dev<a::RopeRow>>(ropes(a::RopeKind::compressed, x.pos));
        x.dcrope = std::make_unique<Dev<a::RopeRow>>(ropes(a::RopeKind::compressed, x.pos, s.ratio));
        x.kv = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 512, -3, 3));
        x.sc = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 512, -4, 4));
        x.sin = std::make_unique<Dev<float>>(random_f(1024, -3, 3));
        x.sout = std::make_unique<Dev<float>>(1024);
        x.vin = std::make_unique<Dev<std::uint32_t>>(std::vector<std::uint32_t>{static_cast<std::uint32_t>(k % 2U)});
        x.vout = std::make_unique<Dev<std::uint32_t>>(1);
        x.ring = std::make_unique<Dev<std::uint8_t>>(std::size_t{cap} * a::kWindowEntryBytes);
        const std::size_t entries = (s.last + 1U) / s.ratio + 1U;
        x.kvc = std::make_unique<Dev<std::uint8_t>>(entries * a::kKvEntryBytes);
        x.max_count = static_cast<std::uint32_t>((s.last + 1U) / s.ratio);
        x.ixc = std::make_unique<Dev<std::uint8_t>>(random_idx_cache(x.max_count + 1U));
        x.q = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 32 * 128, -2, 2));
        x.w = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 32, -1, 1));
        x.scores = std::make_unique<Dev<float>>(std::size_t{s.rows} * x.max_count);
        x.prep = std::make_unique<Dev<std::uint8_t>>(std::size_t{s.rows} * g::kIdxPrepBytes);
        const std::uint32_t nb = (x.max_count + a::kCandBlockSize - 1) / a::kCandBlockSize;
        x.cand = std::make_unique<Dev<std::uint8_t>>(std::size_t{s.rows} * nb);
        x.keys = std::make_unique<Dev<std::uint32_t>>(std::size_t{s.rows} * x.max_count);
        x.idx = std::make_unique<Dev<std::uint32_t>>(std::size_t{s.rows} * a::kTopK);
        x.cnt = std::make_unique<Dev<std::uint32_t>>(s.rows);
        // attention inputs: random ring and compressed keys, selected indices below the cache size
        std::vector<std::uint8_t> ring(std::size_t{cap} * a::kWindowEntryBytes), kv(4096U * a::kKvEntryBytes);
        for (std::size_t e = 0; e < cap; ++e) {
            for (unsigned d = 0; d < 512; ++d) { std::uint8_t c; do c = static_cast<std::uint8_t>(rng()); while ((c & 0x7FU) == 0x7FU); ring[e * 528 + d] = c; }
            for (unsigned c = 0; c < 16; ++c) ring[e * 528 + 512 + c] = static_cast<std::uint8_t>(118 + rng() % 4);
        }
        for (std::size_t t = 0; t < 4096U; ++t) {
            for (unsigned i = 0; i < 256; ++i) kv[t * 288 + i] = static_cast<std::uint8_t>(rng());
            for (unsigned c = 0; c < 32; ++c) kv[t * 288 + 256 + c] = static_cast<std::uint8_t>(0x30 + rng() % 16);
        }
        const unsigned nc = std::min<unsigned>(a::kTopK, static_cast<unsigned>(k * 97U % 600U));
        std::vector<std::uint32_t> idx(std::size_t{s.rows} * a::kTopK), cnt(s.rows, nc);
        for (unsigned r = 0; r < s.rows; ++r) {
            std::vector<std::uint32_t> pick(4096);
            for (std::uint32_t i = 0; i < 4096U; ++i) pick[i] = i;
            std::shuffle(pick.begin(), pick.end(), rng);
            std::sort(pick.begin(), pick.begin() + nc);
            std::copy(pick.begin(), pick.begin() + nc, &idx[std::size_t{r} * a::kTopK]);
        }
        x.aring = std::make_unique<Dev<std::uint8_t>>(ring);
        x.akv = std::make_unique<Dev<std::uint8_t>>(kv);
        x.aidx = std::make_unique<Dev<std::uint32_t>>(idx);
        x.acnt = std::make_unique<Dev<std::uint32_t>>(cnt);
        x.aq = std::make_unique<Dev<float>>(random_f(std::size_t{s.rows} * 64 * 512, -0.5F, 0.5F));
        x.part = std::make_unique<Dev<float>>(std::size_t{s.rows} * 64 * a::kMaxChunks * 514);
        x.out = std::make_unique<Dev<float>>(std::size_t{s.rows} * 64 * 512);
        x.tick = std::make_unique<Dev<std::uint32_t>>(std::size_t{s.rows} * g::kAttnGroupCount);
        x.codes = std::make_unique<Dev<std::int8_t>>(std::size_t{s.rows} * 64 * 512);
        x.deltas = std::make_unique<Dev<float>>(std::size_t{s.rows} * 64 * 16);
        x.sums = std::make_unique<Dev<std::int32_t>>(std::size_t{s.rows} * 64 * 16);
    }
    const auto sink = random_f(64, -2, 2);
    Dev<float> dsink(sink);
    const auto reset = [&] {   // every output to the same pattern
        for (auto& x : b) {
            for (auto* d : {x.ring.get(), x.kvc.get(), x.cand.get()}) check(hipMemset(d->p, 0xA5, d->n));
            for (auto* d : {x.sout.get(), x.scores.get(), x.out.get(), x.deltas.get(), x.part.get()}) check(hipMemset(d->p, 0xA5, d->n * 4));
            for (auto* d : {x.vout.get(), x.keys.get(), x.idx.get(), x.cnt.get()}) check(hipMemset(d->p, 0xA5, d->n * 4));
            check(hipMemset(x.codes->p, 0xA5, x.codes->n)); check(hipMemset(x.sums->p, 0xA5, x.sums->n * 4));
            check(hipMemset(x.prep->p, 0xA5, x.prep->n));
            check(hipMemset(x.tick->p, 0, x.tick->n * 4));
        }
        check(hipDeviceSynchronize());
    };
    const auto snapshot = [&] {
        std::vector<std::vector<std::uint8_t>> out;
        for (auto& x : b) {
            out.push_back(x.ring->get()); out.push_back(x.kvc->get()); out.push_back(x.ixc->get()); out.push_back(x.cand->get());
            out.push_back(bytes_of(x.sout->get())); out.push_back(bytes_of(x.vout->get())); out.push_back(bytes_of(x.scores->get()));
            out.push_back(bytes_of(x.idx->get())); out.push_back(bytes_of(x.cnt->get())); out.push_back(bytes_of(x.out->get()));
            out.push_back(bytes_of(x.codes->get())); out.push_back(bytes_of(x.deltas->get())); out.push_back(bytes_of(x.sums->get()));
            out.push_back(bytes_of(x.tick->get()));
        }
        return out;
    };
    unsigned cases = 0;
    for (const int form : {0, 1, 2}) {   // compress default, ahead, fast
        g::set_compress_ahead(form == 1 ? 1 : 0);
        g::set_compress_fast(form == 2 ? 1 : 0);
        for (const auto pool : {a::Pool::candidate_source, a::Pool::candidate_user, a::Pool::plain})
            for (const int att : {0, 1, 2, 3}) {   // 0 combine kernel, 1 tickets, 2 tickets + A8, 3 tickets, 32 heads
                std::vector<g::WindowView> wv;
                std::vector<g::CompressView> cv;
                std::vector<g::IndexView> iv;
                std::vector<g::SelectView> sv;
                std::vector<g::AttentionView> av;
                for (std::size_t k = 0; k < segs.size(); ++k) {
                    const auto& s = segs[k];
                    auto& x = b[k];
                    wv.push_back({x.kv->p, dnorm.p, x.drope->p, x.dpos->p, s.rows, cap, x.ring->p});
                    cv.push_back({s.ratio, s.rows, x.kv->p, x.sc->p, dnorm.p, dwk.p, dknorm.p, x.dcrope->p, x.dpos->p, x.sin->p, x.vin->p,
                                  x.sout->p, x.vout->p, x.kvc->p, x.ixc->p});
                    iv.push_back({s.rows, s.ratio, x.max_count, x.q->p, x.w->p, x.drope->p, x.dpos->p, x.ixc->p, x.scores->p, x.prep->p});
                    sv.push_back({s.rows, s.ratio, x.max_count, a::kTopK, a::kCandBlocks, a::kCandBlockSize, pool, x.dpos->p, x.scores->p,
                                  x.cand->p, x.keys->p, x.idx->p, x.cnt->p});
                    g::AttentionView v{s.rows, cap, a::kTopK, x.aq->p, dsink.p, x.drope->p, x.dpos->p, x.aring->p, x.akv->p, x.aidx->p,
                                       k % 3U == 0U ? nullptr : x.acnt->p, x.part->p, x.out->p};
                    if (att >= 1) v.tickets = x.tick->p;
                    if (att == 2) { v.codes = x.codes->p; v.deltas = x.deltas->p; v.sums = x.sums->p; }
                    if (att == 3) { v.heads = 32; v.sink = dsink.p + 32; }
                    av.push_back(v);
                }
                // the compress writes index keys that the scores then read: both runs restore them first
                std::vector<std::vector<std::uint8_t>> ixc0;
                for (auto& x : b) ixc0.push_back(x.ixc->get());
                const auto run = [&](bool seg) {
                    reset();
                    for (std::size_t k = 0; k < b.size(); ++k) b[k].ixc->put(ixc0[k]);
                    if (seg) {
                        g::enqueue_window_write_segments(wv, nullptr);
                        g::enqueue_compress_segments(cv, nullptr);
                        g::enqueue_index_scores_segments(iv, nullptr);
                        g::enqueue_select_segments(sv, nullptr);
                        g::enqueue_attention_segments(av, nullptr);
                    } else {
                        for (const auto& v : wv) g::enqueue_window_write(v, nullptr);
                        for (const auto& v : cv) g::enqueue_compress(v, nullptr);
                        for (const auto& v : iv) g::enqueue_index_scores(v, nullptr);
                        for (const auto& v : sv) g::enqueue_select(v, nullptr);
                        for (const auto& v : av) g::enqueue_attention(v, nullptr);
                    }
                    check(hipDeviceSynchronize());
                    return snapshot();
                };
                const auto loop = run(false), seg = run(true);
                expect(loop == seg, "segments == per-segment calls, compress form " + std::to_string(form) + " pool " +
                                        std::to_string(static_cast<int>(pool)) + " attention arm " + std::to_string(att));
                ++cases;
            }
    }
    g::set_compress_ahead(-1);
    g::set_compress_fast(-1);
    std::printf("segment launches == per-segment calls (window, compress, index, select, attention), %zu segments, %u cases\n", segs.size(), cases);
}

}  // namespace

int main() {
    try {
        const bool bench = std::getenv("STRIX_DS41_ATTN_BENCH") != nullptr;
        caches(2, 333);
        caches(1, 300);
        caches(2, 333, 1);
        caches(1, 300, 1);
        compress_fast(bench);
        g::set_compress_ahead(1);  // STRIX_DS41_COMPRESS_AHEAD: the same caches
        caches(2, 333, 0);
        caches(1, 300, 0);
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
        segments();
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
