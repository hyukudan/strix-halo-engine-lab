// Shared MTP chain glue (strix/spec/mtp_chain.hpp) against a fake engine whose "device" is host memory: checks the lazy
// catch-up bookkeeping (position / successor token / hidden row pairing of every mtp_forward call, chunking, carry hidden of
// chain steps, accept() after a partial verify) without any GPU, and that the device-fed chain (engine feed hooks) issues the
// same calls and proposes the same ids as the host-fed chain with at most one synchronizing read per draft.
#include "strix/spec/mtp_chain.hpp"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace sp = strix::spec;

#define MC_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

namespace {
constexpr unsigned kH = 4;
struct FakeRows {
    const std::uint32_t* top = nullptr;
    const float* hidden_post = nullptr;
    const float* hidden_pre = nullptr;
};
struct FakeMtpOut {
    const std::uint32_t* top = nullptr;
    const float* hidden_post = nullptr;
    const float* hidden_pre = nullptr;
};
struct Call {
    std::vector<std::uint32_t> tokens;
    std::vector<float> hidden0;  // first element of each hidden row
    unsigned pos0 = 0;
    bool want_top = false;
};
struct Cfg { unsigned hidden = kH; };
struct FakeEngine {
    using Rows = FakeRows;
    using MtpOut = FakeMtpOut;
    Cfg cfg;
    std::vector<Call> calls;
    std::vector<float> out_h;       // mtp hidden_post rows of the last call: row r first element = 1000 + call index * 10 + r
    std::vector<std::uint32_t> out_top;
    const Cfg& config() const { return cfg; }
    FakeMtpOut mtp_forward(const std::uint32_t* t, const float* hidden, unsigned rows, unsigned pos0, bool want_top) {
        Call c;
        c.tokens.assign(t, t + rows);
        for (unsigned r = 0; r < rows; ++r) c.hidden0.push_back(hidden[r * kH]);
        c.pos0 = pos0;
        c.want_top = want_top;
        calls.push_back(c);
        out_h.assign(rows * kH, 0.0F);
        out_top.assign(rows, 0);
        for (unsigned r = 0; r < rows; ++r) {
            out_h[r * kH] = 1000.0F + float(calls.size()) * 10.0F + float(r);
            out_top[r] = 500U + unsigned(calls.size()) * 10U + r;  // prediction of the call's row r
        }
        return {out_top.data(), out_h.data(), out_h.data()};
    }
    void read_top(const FakeRows& o, unsigned rows, std::uint32_t* host) const { std::memcpy(host, o.top, rows * 4U); }
};
struct HostDev {
    static float* alloc(FakeEngine&, std::size_t n) { return new float[n](); }
    static void release(FakeEngine&, float* p) { delete[] p; }
    static void copy(FakeEngine&, float* d, const float* s, std::size_t n) { std::memcpy(d, s, n * sizeof(float)); }
};
// The same fake with the device-feed hooks: the feed is host memory; a step's input id comes from a slot unless given.
struct FakeFedEngine : FakeEngine {
    static constexpr std::uint32_t kMtpFeedSlots = 64;
    std::uint32_t feed[kMtpFeedSlots]{};
    unsigned feed_reads = 0;
    FakeMtpOut mtp_step_fed(const std::uint32_t* in_host, std::uint32_t in_slot, std::uint32_t out_slot, const float* hidden,
                            std::uint32_t pos, bool want_top) {
        const std::uint32_t t = in_host ? *in_host : feed[in_slot];
        const FakeMtpOut o = mtp_forward(&t, hidden, 1, pos, want_top);
        if (want_top) feed[out_slot] = o.top[0];
        return o;
    }
    void mtp_feed_put(std::uint32_t slot, const std::uint32_t* top) { feed[slot] = *top; }
    void mtp_feed_read(std::uint32_t n, std::uint32_t* host) {
        ++feed_reads;
        std::memcpy(host, feed, n * 4U);
    }
};
static_assert(!sp::MtpFeedEngine<FakeEngine> && sp::MtpFeedEngine<FakeFedEngine>, "feed hook detection");
// The same fake with the KV-only rows hook: rows outside the trailing `outputs` get poisoned outputs (NaN hidden, a top id
// no other row produces), so any read of them changes a later call's hidden operand or a proposal.
struct FakeKvEngine : FakeEngine {
    using FakeEngine::mtp_forward;
    std::vector<std::pair<unsigned, bool>> kv_calls;  // (outputs, want_top) of every 6-argument call
    unsigned kv_only_rows = 0;
    FakeMtpOut mtp_forward(const std::uint32_t* t, const float* hidden, unsigned rows, unsigned pos0, bool want_top, unsigned outputs) {
        const FakeMtpOut o = mtp_forward(t, hidden, rows, pos0, want_top);
        kv_calls.emplace_back(outputs, want_top);
        for (unsigned r = 0; r + std::min(outputs, rows) < rows; ++r) {
            out_h[r * kH] = std::numeric_limits<float>::quiet_NaN();
            out_top[r] = 0xBADu;
            ++kv_only_rows;
        }
        return o;
    }
};
static_assert(!sp::MtpKvRowsEngine<FakeEngine> && sp::MtpKvRowsEngine<FakeKvEngine>, "KV-only rows hook detection");

// The fed fake with the verify pre-launch hooks: the async read snapshots the slots into staging (what the device copy
// sees in stream order); the wait returns that snapshot. `order` records a (async read), l (launch), w (wait).
struct FakePreEngine : FakeFedEngine {
    std::uint32_t staging[kMtpFeedSlots]{};
    std::string order;
    void mtp_feed_read_async(std::uint32_t n) {
        std::memcpy(staging, feed, n * 4U);
        order += 'a';
    }
    void mtp_feed_wait(std::uint32_t n, std::uint32_t* host) {
        ++feed_reads;
        std::memcpy(host, staging, n * 4U);
        order += 'w';
    }
};
static_assert(sp::MtpPrelaunchEngine<FakePreEngine> && !sp::MtpPrelaunchEngine<FakeFedEngine>, "pre-launch hook detection");

// Pre-launch bookkeeping of one scenario run: launches, drafts whose launched ids differ from the returned ones, falls back.
struct PreLog {
    unsigned launches = 0, mismatches = 0, fallbacks = 0, prefix_mismatches = 0;
};

// One scripted sequence (flush chunks, chain steps, catch-up, copy prefixes, drafts without a flush, a k past the feed);
// returns every proposed id in order and the chain's synchronizing reads per draft (max). pre: every draft goes through
// draft_prelaunch with a launch that resolves the feed the way the verify's device gather does (slots read in stream
// order behind the draft), then overwrites those slots (later device work) before the chain waits for its ids.
template <class Eng>
std::vector<std::uint32_t> scenario(Eng& e, std::uint64_t& max_reads_per_draft, bool& fed, PreLog* pre = nullptr) {
    sp::MtpChain<Eng, HostDev> chain(e, false, 3, 8);
    fed = chain.device_feed();
    std::vector<std::uint32_t> all, out;
    max_reads_per_draft = 0;
    auto draft = [&](std::vector<std::uint32_t> prefix, std::uint32_t k) {
        const auto r0 = chain.host_reads();
        if (pre) {
            std::vector<std::uint32_t> launched;
            unsigned calls = 0;
            const sp::PrelaunchFn launch = [&](std::span<const std::uint32_t> pfx, const sp::DraftFeed& f) {
                ++calls;
                if constexpr (requires { e.order; }) e.order += 'l';
                if (!std::equal(pfx.begin(), pfx.end(), prefix.begin(), prefix.end())) ++pre->prefix_mismatches;
                launched.clear();
                for (std::uint32_t j = 0; j < f.k; ++j) {
                    if constexpr (requires { e.feed[0]; }) {
                        launched.push_back(f.slot[j] == sp::DraftFeed::kHost ? f.host_id[j] : e.feed[f.slot[j]]);
                        if (f.slot[j] != sp::DraftFeed::kHost) e.feed[f.slot[j]] = 0xDEADU;
                    }
                }
            };
            if (chain.draft_prelaunch(prefix, k, out, launch)) {
                pre->launches += calls;
                if (calls != 1 || launched != out) ++pre->mismatches;
            } else {
                if (calls != 0) ++pre->mismatches;
                ++pre->fallbacks;
                chain.draft(prefix, k, out);
            }
        } else {
            chain.draft(prefix, k, out);
        }
        max_reads_per_draft = std::max<std::uint64_t>(max_reads_per_draft, chain.host_reads() - r0);
        all.insert(all.end(), out.begin(), out.end());
        all.push_back(0xFFFFFFFFu);
    };
    auto verify_accept = [&](std::vector<std::uint32_t> emitted, float h) {
        std::vector<float> vh(emitted.size() * kH, 0.0F);
        for (std::size_t i = 0; i < emitted.size(); ++i) vh[i * kH] = h + float(i);
        FakeRows vr;
        vr.hidden_post = vr.hidden_pre = vh.data();
        chain.observe_verify(vr, static_cast<std::uint32_t>(emitted.size()));
        chain.accept(emitted);
    };
    const std::uint32_t prompt[5] = {11, 12, 13, 14, 15};
    std::vector<float> hid(5 * kH, 0.0F);
    for (unsigned i = 0; i < 5; ++i) hid[i * kH] = 10.0F * float(i);
    FakeRows pf;
    pf.hidden_post = pf.hidden_pre = hid.data();
    chain.observe_prefill(pf, 5, prompt, 99);
    draft({}, 4);              // flush 3 + 2 rows, 3 chain steps
    verify_accept({out[0], 77}, 100.0F);
    draft({}, 1);              // flush only
    verify_accept({out[0], 78, 79}, 200.0F);
    draft({8, 9}, 3);          // flush, 2 prefix steps, 2 chain steps
    draft({}, 2);              // no flush: draft 0 is the host d1, its step reads it by value
    draft({5}, 1);             // no flush, one prefix step, no chain step
    verify_accept({41}, 300.0F);
    draft({}, 70);             // k + 1 past the feed: host path
    return all;
}

// Long-prompt feed (SpecEngine::prefill_big) and K/V-only catch-up (MtpKvEngine): a fake whose outputs depend only on the
// row's position and input token, and which logs every MTP K/V write (position, successor token, hidden row).
struct KvWrite {
    unsigned pos;
    std::uint32_t token;
    float hidden0;
    bool operator==(const KvWrite&) const = default;
};
struct PosEngine {
    using Rows = FakeRows;
    using MtpOut = FakeMtpOut;
    Cfg cfg;
    std::vector<KvWrite> writes;
    std::vector<std::pair<unsigned, unsigned>> full_calls;  // (pos0, rows) of every mtp_forward
    std::vector<float> out_h;
    std::vector<std::uint32_t> out_top;
    const Cfg& config() const { return cfg; }
    void log(const std::uint32_t* t, const float* hidden, unsigned rows, unsigned pos0) {
        for (unsigned r = 0; r < rows; ++r) writes.push_back({pos0 + r, t[r], hidden[r * kH]});
    }
    FakeMtpOut mtp_forward(const std::uint32_t* t, const float* hidden, unsigned rows, unsigned pos0, bool) {
        log(t, hidden, rows, pos0);
        full_calls.emplace_back(pos0, rows);
        out_h.assign(rows * kH, 0.0F);
        out_top.assign(rows, 0);
        for (unsigned r = 0; r < rows; ++r) {
            out_h[r * kH] = 2000.0F + float(pos0 + r) + 0.25F * float(t[r] % 4U);
            out_top[r] = 7000U + 3U * (pos0 + r) + t[r] % 5U;
        }
        return {out_top.data(), out_h.data(), out_h.data()};
    }
    void read_top(const FakeRows& o, unsigned rows, std::uint32_t* host) const { std::memcpy(host, o.top, rows * 4U); }
};
struct PosKvEngine : PosEngine {
    unsigned kv_rows = 0;
    void mtp_kv(const std::uint32_t* t, const float* hidden, unsigned rows, unsigned pos0) {
        log(t, hidden, rows, pos0);
        kv_rows += rows;
    }
};
struct PosDev {
    template <class E> static float* alloc(E&, std::size_t n) { return new float[n](); }
    template <class E> static void release(E&, float* p) { delete[] p; }
    template <class E> static void copy(E&, float* d, const float* s, std::size_t n) { std::memcpy(d, s, n * sizeof(float)); }
};
static_assert(!sp::MtpKvEngine<FakeEngine> && !sp::MtpKvEngine<PosEngine> && sp::MtpKvEngine<PosKvEngine>, "kv hook detection");

struct FeedResult {
    std::vector<KvWrite> writes;
    std::vector<std::pair<unsigned, unsigned>> full_calls;
    std::vector<std::uint32_t> drafts;
    std::uint64_t rows = 0, kv_calls = 0, calls = 0;
    std::uint64_t calls0 = 0, kv0 = 0, rows0 = 0;  // after the prompt, before the first draft (the exe's spec[] snapshot)
};
// P prompt rows fed as windows of `win` rows (win = 0: one observe_prefill of all rows = prefill_big), then a draft of 4,
// one accept of 3 rows and a second draft (decode catch-up after the prompt).
template <class Eng>
FeedResult feed(unsigned P, unsigned win, bool kv) {
    Eng e;
    sp::MtpChain<Eng, PosDev> chain(e, false, 3, 8);  // chunk 3 rows, ring 8: many forced flushes over the prompt
    chain.set_kv_catchup(kv);
    std::vector<std::uint32_t> prompt(P);
    std::vector<float> hid(std::size_t(P) * kH, 0.0F);
    for (unsigned i = 0; i < P; ++i) {
        prompt[i] = 100U + (i * 37U) % 91U;
        hid[std::size_t(i) * kH] = float(i) + 0.5F;
    }
    const std::uint32_t anchor = 4242;
    FakeRows pf;
    pf.hidden_post = pf.hidden_pre = hid.data();
    if (win == 0) {
        chain.observe_prefill(pf, P, prompt.data(), anchor);
    } else {
        for (unsigned p = 0; p < P; p += win) {
            const unsigned r = std::min(win, P - p);
            FakeRows w;
            w.hidden_post = w.hidden_pre = hid.data() + std::size_t(p) * kH;
            chain.observe_prefill(w, r, prompt.data() + p, p + r == P ? anchor : prompt[p + r]);
        }
    }
    FeedResult res;
    res.calls0 = chain.calls();
    res.kv0 = chain.kv_calls();
    res.rows0 = chain.rows();
    std::vector<std::uint32_t> out;
    chain.draft({}, 4, out);
    res.drafts = out;
    std::vector<float> vh(3 * kH, 0.0F);
    for (unsigned i = 0; i < 3; ++i) vh[i * kH] = 900.0F + float(i);
    FakeRows vr;
    vr.hidden_post = vr.hidden_pre = vh.data();
    chain.observe_verify(vr, 3);
    chain.accept(std::vector<std::uint32_t>{out[0], 61, 62});
    chain.draft({}, 3, out);
    res.drafts.insert(res.drafts.end(), out.begin(), out.end());
    res.writes = e.writes;
    res.full_calls = e.full_calls;
    res.rows = chain.rows();
    res.kv_calls = chain.kv_calls();
    res.calls = chain.calls();
    return res;
}
// The K/V each position ends with = its LAST write (chain steps write speculative rows that a later flush overwrites).
std::vector<KvWrite> final_kv(const std::vector<KvWrite>& w) {
    std::vector<KvWrite> last;
    for (const auto& x : w) {
        if (x.pos >= last.size()) last.resize(x.pos + 1, KvWrite{~0U, 0, 0.0F});
        last[x.pos] = x;
    }
    return last;
}
}  // namespace

int main() {
    FakeEngine e;
    sp::MtpChain<FakeEngine, HostDev> chain(e, false, 3, 8);  // chunk 3 rows, ring 8
    // prefill: positions 0..4 with hidden first-element 10*pos, tokens 11..15, anchor 99 follows position 4
    const std::uint32_t prompt[5] = {11, 12, 13, 14, 15};
    std::vector<float> hid(5 * kH, 0.0F);
    for (unsigned i = 0; i < 5; ++i) hid[i * kH] = 10.0F * float(i);
    FakeRows pf;
    pf.hidden_post = pf.hidden_pre = hid.data();
    chain.observe_prefill(pf, 5, prompt, 99);
    MC_CHECK(e.calls.empty(), "lazy: nothing before the first draft");

    std::vector<std::uint32_t> out;
    chain.draft({}, 2, out);
    // flush: successors 12,13,14 | 15,99 at positions 0..4; the last chunk wants the top
    MC_CHECK(e.calls.size() == 2 + 1, "flush chunks 3+2 then one chain step: got %zu calls", e.calls.size());
    MC_CHECK(e.calls[0].pos0 == 0 && e.calls[0].tokens == std::vector<std::uint32_t>({12, 13, 14}) && !e.calls[0].want_top, "chunk 0");
    MC_CHECK(e.calls[0].hidden0 == std::vector<float>({0.0F, 10.0F, 20.0F}), "chunk 0 hidden rows");
    MC_CHECK(e.calls[1].pos0 == 3 && e.calls[1].tokens == std::vector<std::uint32_t>({15, 99}) && e.calls[1].want_top, "chunk 1");
    MC_CHECK(e.calls[1].hidden0 == std::vector<float>({30.0F, 40.0F}), "chunk 1 hidden rows");
    // d1 = prediction of the last flushed row (call 2, row 1) = 500 + 20 + 1; step feeds d1 at position 5 with carry = that row's hidden
    MC_CHECK(out.size() == 2 && out[0] == 521U, "first draft token is the last flushed row's prediction (%u)", out.empty() ? 0U : out[0]);
    MC_CHECK(e.calls[2].pos0 == 5 && e.calls[2].tokens == std::vector<std::uint32_t>({521}) && e.calls[2].want_top, "chain step");
    MC_CHECK(e.calls[2].hidden0[0] == 1000.0F + 20.0F + 1.0F, "carry = last flushed row hidden (%g)", double(e.calls[2].hidden0[0]));
    MC_CHECK(out[1] == 530U, "second draft = chain step prediction (%u)", out[1]);

    // verify of [anchor 99, 521, 530] accepted 2 rows: emitted = {521, 77}; the next draft catches up positions 5..6 (not 7)
    std::vector<float> vh(3 * kH, 0.0F);
    vh[0] = 100.0F; vh[kH] = 110.0F; vh[2 * kH] = 120.0F;
    FakeRows vr;
    vr.hidden_post = vr.hidden_pre = vh.data();
    chain.observe_verify(vr, 3);
    const std::uint32_t emitted[2] = {521, 77};
    chain.accept(emitted);
    const auto before = e.calls.size();
    chain.draft({}, 1, out);
    MC_CHECK(e.calls.size() == before + 1 && out.size() == 1, "one catch-up call, no chain step for k = 1");
    MC_CHECK(e.calls[before].pos0 == 5 && e.calls[before].tokens == std::vector<std::uint32_t>({521, 77}), "catch-up overwrites the speculative position");
    MC_CHECK(e.calls[before].hidden0 == std::vector<float>({100.0F, 110.0F}), "catch-up hidden rows are the verify rows");
    // prefix (copy tokens) is consumed by the chain: only the last step's prediction is used
    chain.draft(std::vector<std::uint32_t>({8, 9}), 1, out);
    const auto n = e.calls.size();
    MC_CHECK(e.calls[n - 2].pos0 == 7 && e.calls[n - 2].tokens[0] == 8 && !e.calls[n - 2].want_top, "prefix step 1 (no top)");
    MC_CHECK(e.calls[n - 1].pos0 == 8 && e.calls[n - 1].tokens[0] == 9 && e.calls[n - 1].want_top, "prefix step 2 (top)");
    MC_CHECK(out.size() == 1 && out[0] == 500U + unsigned(n) * 10U, "draft after a prefix = last prefix step prediction");
    // ring overflow flushes in order
    chain.reset();
    std::vector<float> big(20 * kH, 1.0F);
    std::vector<std::uint32_t> toks(20, 3);
    FakeRows br;
    br.hidden_post = br.hidden_pre = big.data();
    e.calls.clear();
    chain.observe_prefill(br, 20, toks.data(), 4);
    MC_CHECK(!e.calls.empty() && e.calls[0].pos0 == 0, "ring full forces an in-order flush from position 0");

    // device feed == host feed: same mtp calls (tokens, hidden rows, positions, want_top) and the same proposals
    FakeEngine he;
    FakeFedEngine fe;
    std::uint64_t host_reads = 0, fed_reads = 0;
    bool host_fed = true, fed_fed = false;
    const auto hp = scenario(he, host_reads, host_fed);
    const auto fp = scenario(fe, fed_reads, fed_fed);
    MC_CHECK(!host_fed && fed_fed, "feed detection: host %d fed %d", int(host_fed), int(fed_fed));
    MC_CHECK(hp == fp, "device-fed proposals differ from host-fed (%zu vs %zu ids)", hp.size(), fp.size());
    MC_CHECK(he.calls.size() == fe.calls.size(), "call count %zu vs %zu", he.calls.size(), fe.calls.size());
    for (std::size_t i = 0; i < he.calls.size(); ++i)
        MC_CHECK(he.calls[i].tokens == fe.calls[i].tokens && he.calls[i].hidden0 == fe.calls[i].hidden0 &&
                     he.calls[i].pos0 == fe.calls[i].pos0 && he.calls[i].want_top == fe.calls[i].want_top,
                 "call %zu differs", i);
    MC_CHECK(fed_reads == 70, "the k = 70 draft falls back to one read per position (max %llu)", (unsigned long long)fed_reads);
    MC_CHECK(host_reads == 70, "host path reads once per position (max %llu)", (unsigned long long)host_reads);
    // KV-only rows == every row computed: same calls, same proposals; flush chunks pass outputs 0, the final chunk 1
    {
        FakeEngine pe;
        FakeKvEngine ke;
        std::uint64_t pr = 0, kr = 0;
        bool pfed = false, kfed = false;
        const auto pp = scenario(pe, pr, pfed);
        const auto kp = scenario(ke, kr, kfed);
        MC_CHECK(pp == kp, "KV-only proposals differ (%zu vs %zu ids)", pp.size(), kp.size());
        MC_CHECK(pe.calls.size() == ke.calls.size(), "KV-only call count %zu vs %zu", pe.calls.size(), ke.calls.size());
        for (std::size_t i = 0; i < pe.calls.size(); ++i)
            MC_CHECK(pe.calls[i].tokens == ke.calls[i].tokens && pe.calls[i].hidden0 == ke.calls[i].hidden0 &&
                         pe.calls[i].pos0 == ke.calls[i].pos0 && pe.calls[i].want_top == ke.calls[i].want_top,
                     "KV-only call %zu differs (a poisoned row was read)", i);
        bool saw_none = false;
        for (const auto& [outputs, want_top] : ke.kv_calls) {
            MC_CHECK(outputs == (want_top ? 1U : 0U), "flush chunk outputs %u with want_top %d", outputs, int(want_top));
            saw_none = saw_none || outputs == 0U;
        }
        MC_CHECK(!ke.kv_calls.empty() && saw_none && ke.kv_only_rows > 0U, "KV-only hook engaged (%zu calls, %u rows)",
                 ke.kv_calls.size(), ke.kv_only_rows);
    }
    std::uint64_t fed_small = 0;
    {
        FakeFedEngine f2;
        sp::MtpChain<FakeFedEngine, HostDev> c2(f2, false, 3, 8);
        FakeRows p2;
        p2.hidden_post = p2.hidden_pre = hid.data();
        c2.observe_prefill(p2, 5, prompt, 99);
        c2.draft({}, 6, out);
        fed_small = c2.host_reads();
        MC_CHECK(f2.feed_reads == 1 && fed_small == 1, "a fed draft of 6 reads once (%u)", f2.feed_reads);
        c2.set_device_feed(false);
        const auto before_off = c2.host_reads();
        c2.draft({}, 3, out);
        MC_CHECK(!c2.device_feed() && c2.host_reads() - before_off == 2, "rollback: host reads per step");
    }
    {  // long prompt: one observe_prefill of every row (prefill_big) == 32-row windows (prefill), call for call
        const unsigned P = 77;
        const auto a = feed<PosEngine>(P, 32, false), b = feed<PosEngine>(P, 0, false), c = feed<PosEngine>(P, 5, false);
        MC_CHECK(a.writes == b.writes && a.full_calls == b.full_calls && a.drafts == b.drafts, "prefill_big feed != windows feed");
        MC_CHECK(a.writes == c.writes && a.drafts == c.drafts, "window size changes the feed");
        // every prompt position is written once with its successor (tokens[i + 1], anchor for the last) and its own hidden
        const auto kvw = final_kv(a.writes);
        for (unsigned i = 0; i < P; ++i)
            MC_CHECK(kvw[i].pos == i && kvw[i].hidden0 == float(i) + 0.5F && kvw[i].token == (i + 1 < P ? 100U + ((i + 1) * 37U) % 91U : 4242U),
                     "prompt row %u: pos %u token %u hidden %.2f", i, kvw[i].pos, kvw[i].token, double(kvw[i].hidden0));
        // K/V-only catch-up: the same K/V per position, the same drafts; full calls only where an output is read
        const auto k = feed<PosKvEngine>(P, 0, true), kw = feed<PosKvEngine>(P, 32, true);
        MC_CHECK(final_kv(k.writes) == kvw && k.drafts == a.drafts, "kv catch-up changed the MTP K/V or the drafts");
        MC_CHECK(final_kv(kw.writes) == kvw && kw.drafts == a.drafts, "kv catch-up (windows) changed the K/V or the drafts");
        MC_CHECK(k.rows == a.rows && k.kv_calls > 0, "kv catch-up rows %llu vs %llu, kv calls %llu", (unsigned long long)k.rows,
                 (unsigned long long)a.rows, (unsigned long long)k.kv_calls);
        // counter split (kitfix 11-19, device d0): calls() counts full mtp_forward calls only; a K/V-only chunk counts in
        // kv_calls(). Full catch-up calls == kv run calls + kv-only calls, in total and in the decode window after the prompt
        // (the first draft flushes the residual ring: its non-final chunks are K/V-only), with the same rows.
        MC_CHECK(a.kv_calls == 0 && k.calls + k.kv_calls == a.calls && kw.calls + kw.kv_calls == a.calls,
                 "counter split: full %llu vs kv %llu + kv-only %llu", (unsigned long long)a.calls, (unsigned long long)k.calls,
                 (unsigned long long)k.kv_calls);
        MC_CHECK((k.calls - k.calls0) + (k.kv_calls - k.kv0) == a.calls - a.calls0 && k.rows - k.rows0 == a.rows - a.rows0 &&
                     k.kv_calls - k.kv0 > 0,
                 "decode window: full %llu vs kv %llu + kv-only %llu", (unsigned long long)(a.calls - a.calls0),
                 (unsigned long long)(k.calls - k.calls0), (unsigned long long)(k.kv_calls - k.kv0));
        unsigned prompt_full = 0;  // full calls that cover prompt rows: only the final chunk of the first draft's flush
        for (const auto& fc : k.full_calls) prompt_full += fc.first < P ? 1U : 0U;
        MC_CHECK(prompt_full == 1, "kv catch-up: %u full calls over the prompt (want 1)", prompt_full);
        unsigned inc_full = 0;
        for (const auto& fc : a.full_calls) inc_full += fc.first < P ? 1U : 0U;
        const unsigned forced = (P - 1) / 8, rest = P - 8 * forced;  // ring-full flushes of 8 rows (3 calls), then the draft's
        MC_CHECK(inc_full == forced * 3 + (rest + 2) / 3, "incumbent: %u full calls over the prompt (want %u)", inc_full,
                 forced * 3 + (rest + 2) / 3);
        // rollback: an engine with the hook but kv off issues exactly the incumbent calls
        const auto off = feed<PosKvEngine>(P, 0, false);
        MC_CHECK(off.writes == a.writes && off.full_calls == a.full_calls && off.drafts == a.drafts && off.kv_calls == 0, "kv off != incumbent");
        // an engine without the hook ignores the switch
        const auto nohook = feed<PosEngine>(P, 0, true);
        MC_CHECK(nohook.writes == a.writes && nohook.full_calls == a.full_calls && nohook.kv_calls == 0, "kv switch without hook");
        // default: K/V-only catch-up ON where the engine has the hook (device gate passed), the incumbent calls without it
        PosKvEngine ek;
        PosEngine ep;
        const sp::MtpChain<PosKvEngine, PosDev> dk(ek, false, 3, 8);
        const sp::MtpChain<PosEngine, PosDev> dp(ep, false, 3, 8);
        MC_CHECK(dk.kv_catchup() && !dp.kv_catchup(), "default kv catch-up: hook %d no-hook %d", int(dk.kv_catchup()), int(dp.kv_catchup()));
    }
    // verify pre-launch == device feed: same mtp calls, same proposals; the launched verify's ids (feed slots resolved at
    // launch) are exactly the drafted ids, the ids are read from the copy queued BEFORE the launch, one launch per draft.
    {
        FakePreEngine pe;
        PreLog log;
        std::uint64_t pre_reads = 0;
        bool pre_fed = false;
        const auto pp = scenario(pe, pre_reads, pre_fed, &log);
        MC_CHECK(pre_fed && pp == fp, "pre-launch proposals differ from device-fed (%zu vs %zu ids)", pp.size(), fp.size());
        MC_CHECK(pe.calls.size() == fe.calls.size(), "pre-launch call count %zu vs %zu", pe.calls.size(), fe.calls.size());
        for (std::size_t i = 0; i < fe.calls.size() && i < pe.calls.size(); ++i)
            MC_CHECK(fe.calls[i].tokens == pe.calls[i].tokens && fe.calls[i].hidden0 == pe.calls[i].hidden0 &&
                         fe.calls[i].pos0 == pe.calls[i].pos0 && fe.calls[i].want_top == pe.calls[i].want_top,
                     "pre-launch call %zu differs", i);
        MC_CHECK(log.mismatches == 0 && log.prefix_mismatches == 0, "launched ids differ from the drafts (%u, prefix %u)",
                 log.mismatches, log.prefix_mismatches);
        MC_CHECK(log.launches == 5 && log.fallbacks == 1, "5 pre-launched drafts + the k = 70 fallback (%u, %u)", log.launches,
                 log.fallbacks);
        // drafts 1-3 read (a l w), draft 4 (k = 2, no flush) reads (a l w), draft 5 (one prefix step) reads (a l w)
        MC_CHECK(pe.order == "alwalwalwalwalw", "async read before launch before wait (%s)", pe.order.c_str());
        // k = 1 without a flush or a prefix: draft 0 is the host d1, nothing to read; the launch gets it as a host row
        FakePreEngine p3;
        sp::MtpChain<FakePreEngine, HostDev> c3(p3, false, 3, 8);
        FakeRows p3r;
        p3r.hidden_post = p3r.hidden_pre = hid.data();
        c3.observe_prefill(p3r, 5, prompt, 99);
        c3.draft({}, 1, out);  // flush
        const auto d1 = out[0];
        sp::DraftFeed seen;
        const sp::PrelaunchFn grab = [&](std::span<const std::uint32_t>, const sp::DraftFeed& f) { seen = f; };
        p3.order.clear();
        MC_CHECK(c3.draft_prelaunch({}, 1, out, grab) && out.size() == 1 && out[0] == d1, "k = 1 pre-launch draft");
        MC_CHECK(seen.k == 1 && seen.slot[0] == sp::DraftFeed::kHost && seen.host_id[0] == d1 && p3.order.empty(),
                 "k = 1 host row, no read (%s)", p3.order.c_str());
        c3.set_device_feed(false);
        MC_CHECK(!c3.draft_prelaunch({}, 2, out, grab), "no device feed: pre-launch declined");
        FakeFedEngine f4;
        sp::MtpChain<FakeFedEngine, HostDev> c4(f4, false, 3, 8);
        MC_CHECK(!c4.draft_prelaunch({}, 2, out, grab), "engine without the pre-launch hooks: declined");
    }
    std::puts("spec_mtp_chain_tests: PASS");
    return 0;
}
