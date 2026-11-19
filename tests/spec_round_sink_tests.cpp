// Round trace of the shared speculator (strix/spec/block_drafter.hpp RoundSink / RoundTraceWriter) against a deterministic
// fake target and fake chain: (1) the emitted ids with a sink attached equal the ids without one (copy, chain and hybrid
// blocks) and equal serial greedy; (2) every record is self-consistent (greedy has one id per verify row, the rejected rows
// included; emitted = drafts[0, accepted) + greedy[accepted]) and the records rebuild the stream; (3) a detached sink sees
// no call; (4) RoundTraceWriter's file reads back field for field. argv[1] (optional): also write the hand-made trace of the
// scorer fixture (scripts/bench/spec/recycle_score.py --expect) to that path.
#include "strix/spec/block_drafter.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace sp = strix::spec;

namespace {
#define RS_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

double g_clock = 0.0;  // virtual ms

// response[i] = greedy token at response position i; the prompt is re-emitted with sparse edits ("edit") or not ("prose").
struct World {
    std::vector<std::uint32_t> prompt, response;
};

World make_world(bool edit, std::uint32_t n, std::uint64_t seed) {
    World w;
    for (std::uint32_t i = 0; i < 64; ++i) w.prompt.push_back(std::uint32_t(mix(seed + i) % 900) + 10);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t t = std::uint32_t(mix(seed * 31 + i) % 900) + 10;
        if (edit) {
            t = w.prompt[i % 64];
            if (mix(seed * 7 + i) % 23 == 0) t = std::uint32_t(mix(seed * 13 + i) % 900) + 10;
        }
        w.response.push_back(t);
    }
    return w;
}

struct FakeTarget final : sp::VerifyTarget {
    const World& w;
    std::uint32_t committed_rows = 0;
    std::size_t last_rows = 0;
    explicit FakeTarget(const World& world) : w(world) {}
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        greedy.clear();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto pos = committed_rows + i;
            greedy.push_back(pos + 1 < w.response.size() ? w.response[pos + 1] : 7);
        }
        last_rows = tokens.size();
        g_clock += 10.0 + 0.15 * double(tokens.size());
    }
    void commit(std::uint32_t k) override {
        RS_CHECK(k >= 1 && k <= last_rows, "commit %u of %zu", k, last_rows);
        committed_rows += k;
        g_clock += 0.2;
    }
};

struct FakeChain final : sp::ChainDrafter {
    const World& w;
    double accuracy;
    std::uint32_t stream_len = 0;
    FakeChain(const World& world, double acc) : w(world), accuracy(acc) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        out.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t pos = std::uint64_t(stream_len) + prefix.size() + j;
            const bool right = double(mix(pos * 977 + 5) % 1000) / 1000.0 < accuracy;
            out.push_back(pos < w.response.size() && right ? w.response[pos] : 5);
        }
        g_clock += 1.5 * double(prefix.size() + k);
    }
    void accept(std::span<const std::uint32_t> emitted) override { stream_len += std::uint32_t(emitted.size()); }
};

// Copies the record (the spans die with the call) and counts calls.
struct MemorySink final : sp::RoundSink {
    struct Rec {
        sp::RoundRecord r;
        std::vector<std::uint32_t> drafts, greedy, emitted;
    };
    std::vector<Rec> recs;
    void round(const sp::RoundRecord& r) override {
        Rec c{r, {r.drafts.begin(), r.drafts.end()}, {r.greedy.begin(), r.greedy.end()}, {r.emitted.begin(), r.emitted.end()}};
        c.r.drafts = {};
        c.r.greedy = {};
        c.r.emitted = {};
        recs.push_back(std::move(c));
    }
};

struct Run {
    std::vector<std::uint32_t> out;
    std::uint64_t steps = 0;
};

Run run(const World& w, std::uint32_t n, double acc, bool copy, bool chain, bool hybrid, sp::RoundSink* sink,
        std::uint32_t detach_after = ~0U) {
    g_clock = 0.0;
    sp::BlockConfig cfg;
    cfg.copy = copy; cfg.chain = chain; cfg.hybrid = hybrid;
    cfg.clock_ms = [] { return g_clock; };
    FakeTarget target(w);
    FakeChain fchain(w, acc);
    sp::BlockDrafter drafter(cfg, chain ? &fchain : nullptr);
    drafter.commit(w.prompt);
    Run r;
    r.out.push_back(w.response[0]);
    drafter.commit(std::span<const std::uint32_t>(r.out.data(), 1));
    sp::BlockSpeculator spec(drafter, target);
    spec.set_round_sink(sink);
    spec.start(r.out[0]);
    while (r.out.size() < n) {
        if (r.steps == detach_after) spec.set_round_sink(nullptr);
        sp::StepParams p;
        p.remaining = n - std::uint32_t(r.out.size());
        p.max_rows = 32;
        p.context = w.prompt.size() + target.committed_rows;
        const auto rec = spec.step(r.out, p);
        if (rec.emitted == 0) break;
        ++r.steps;
    }
    return r;
}

void check_records(const MemorySink& s, const Run& r, const char* what) {
    RS_CHECK(s.recs.size() == r.steps, "%s: %zu records for %llu steps", what, s.recs.size(), (unsigned long long)r.steps);
    std::vector<std::uint32_t> stream{r.out[0]};
    std::uint32_t rejected = 0;
    for (const auto& c : s.recs) {
        const auto& R = c.r;
        RS_CHECK(c.drafts.size() == R.k && R.rows == R.k + 1 && c.greedy.size() == R.rows, "%s: k %u rows %u greedy %zu", what, R.k,
                 R.rows, c.greedy.size());
        RS_CHECK(R.accepted <= R.k, "%s: accepted %u > k %u", what, R.accepted, R.k);
        for (std::uint32_t i = 0; i < R.accepted; ++i) RS_CHECK(c.drafts[i] == c.greedy[i], "%s: accepted draft %u != greedy", what, i);
        if (R.accepted < R.k) {
            RS_CHECK(c.drafts[R.accepted] != c.greedy[R.accepted], "%s: first rejected draft equals greedy", what);
            ++rejected;
        }
        RS_CHECK(R.anchor == stream.back(), "%s: anchor %u != last emitted %u", what, R.anchor, stream.back());
        std::vector<std::uint32_t> e(c.drafts.begin(), c.drafts.begin() + R.accepted);
        e.push_back(c.greedy[R.accepted]);
        RS_CHECK(e == c.emitted, "%s: emitted != drafts[0,a) + greedy[a]", what);
        stream.insert(stream.end(), e.begin(), e.end());
    }
    RS_CHECK(stream == r.out, "%s: records do not rebuild the emitted stream", what);
    std::printf("%s: %zu records, %u rejected rounds, %zu tokens\n", what, s.recs.size(), rejected, r.out.size());
}

std::uint32_t get_u32(const std::vector<unsigned char>& b, std::size_t& o) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= std::uint32_t(b.at(o + std::size_t(i))) << (8 * i);
    o += 4;
    return v;
}

void writer_roundtrip(const MemorySink& s, const std::string& path) {
    {
        sp::RoundTraceWriter wr(path, "unit");
        for (const auto& c : s.recs) {
            sp::RoundRecord r = c.r;
            r.drafts = c.drafts;
            r.greedy = c.greedy;
            r.emitted = c.emitted;
            wr.round(r);
        }
        RS_CHECK(wr.records() == s.recs.size(), "writer count");
    }
    std::FILE* f = std::fopen(path.c_str(), "rb");
    RS_CHECK(f, "cannot reopen %s", path.c_str());
    std::vector<unsigned char> b;
    unsigned char tmp[4096];
    for (std::size_t n; (n = std::fread(tmp, 1, sizeof tmp, f)) > 0;) b.insert(b.end(), tmp, tmp + n);
    std::fclose(f);
    std::remove(path.c_str());
    std::size_t o = 0;
    RS_CHECK(std::memcmp(b.data(), "SRT1", 4) == 0, "file magic");
    o = 4;
    RS_CHECK(get_u32(b, o) == 1, "version");
    const std::uint32_t ll = get_u32(b, o);
    RS_CHECK(std::string(b.begin() + std::ptrdiff_t(o), b.begin() + std::ptrdiff_t(o + ll)) == "unit", "label");
    o += ll;
    for (const auto& c : s.recs) {
        RS_CHECK(std::memcmp(b.data() + o, "RREC", 4) == 0, "record magic at %zu", o);
        o += 4;
        const std::uint64_t ctx = std::uint64_t(get_u32(b, o)) | (std::uint64_t(get_u32(b, o)) << 32);
        const std::uint32_t anchor = get_u32(b, o), k = get_u32(b, o), rows = get_u32(b, o), acc = get_u32(b, o);
        const std::uint32_t cp = get_u32(b, o), ext = get_u32(b, o), src = get_u32(b, o);
        o += 8;  // f32 verify_ms, draft_ms
        RS_CHECK(ctx == c.r.context && anchor == c.r.anchor && k == c.r.k && rows == c.r.rows && acc == c.r.accepted && cp == c.r.copy &&
                     ext == c.r.ext && src == std::uint32_t(c.r.source), "record header mismatch");
        for (const std::uint32_t d : c.drafts) RS_CHECK(get_u32(b, o) == d, "draft id");
        for (const std::uint32_t g : c.greedy) RS_CHECK(get_u32(b, o) == g, "greedy id");
    }
    RS_CHECK(o == b.size(), "trailing bytes %zu of %zu", o, b.size());
    std::printf("writer round trip: %zu records, %zu bytes\n", s.recs.size(), b.size());
}

// Scorer fixture: stream s(i) = 100 + i. Round 1 (model, k 6): drafts s1 s2 X s4 s5 s6 -> accepted 2, correction s3; the
// substitution candidate s4 s5 s6 matches 3. Round 2 (model, k 2): s4 Y -> accepted 1 (M = 1 <= 3: counted); its own
// candidate is empty (k - a - 1 = 0). Round 3 (k 1): s6 accepted, not rejected. Round 4 (k 3): Z s9 s10 -> accepted 0
// (correction s8), candidate s9 s10 matches 2; round 5 (k 4): s9..s12 accepted 4 (M = 4 > 2: not counted); last round.
// Expected (scripts/bench/spec/recycle_score.py --expect): rounds 5, rejected 3, f 1/4 = 0.25 (rounds with a successor),
// f_rej 1/3, mean_run (3 + 0 + 2) / 3.
void write_fixture(const std::string& path) {
    auto s = [](std::uint32_t i) { return 100 + i; };
    sp::RoundTraceWriter wr(path, "fixture");
    struct R { std::uint32_t anchor; std::vector<std::uint32_t> d, g; std::uint32_t a; };
    const std::vector<R> rounds = {
        {s(0), {s(1), s(2), 9, s(4), s(5), s(6)}, {s(1), s(2), s(3), 1, 1, 1, 1}, 2},
        {s(3), {s(4), 9}, {s(4), s(5), 1}, 1},
        {s(5), {s(6)}, {s(6), s(7)}, 1},
        {s(7), {9, s(9), s(10)}, {s(8), 1, 1, 1}, 0},
        {s(8), {s(9), s(10), s(11), s(12)}, {s(9), s(10), s(11), s(12), s(13)}, 4},
    };
    for (const auto& x : rounds) {
        sp::RoundRecord r;
        r.source = sp::Source::model;
        r.anchor = x.anchor;
        r.k = std::uint32_t(x.d.size());
        r.rows = r.k + 1;
        r.accepted = x.a;
        r.verify_ms = 13.3 + 0.113 * double(r.k);
        r.draft_ms = 0.51 * double(r.k);
        r.context = 64;
        std::vector<std::uint32_t> e(x.d.begin(), x.d.begin() + x.a);
        e.push_back(x.g[x.a]);
        r.drafts = x.d;
        r.greedy = x.g;
        r.emitted = e;
        wr.round(r);
    }
    std::printf("fixture trace written: %s (%llu records)\n", path.c_str(), (unsigned long long)wr.records());
}
}  // namespace

int main(int argc, char** argv) {
    struct Case { const char* name; bool edit; double acc; bool copy, chain, hybrid; };
    const Case cases[] = {
        {"copy edit", true, 0.0, true, false, false},
        {"chain prose", false, 0.7, false, true, false},
        {"hybrid edit", true, 0.6, true, true, true},
        {"hybrid prose", false, 0.85, true, true, true},
    };
    MemorySink keep;
    for (const auto& c : cases) {
        const World w = make_world(c.edit, 600, c.edit ? 11 : 29);
        const Run off = run(w, 600, c.acc, c.copy, c.chain, c.hybrid, nullptr);
        MemorySink sink;
        const Run on = run(w, 600, c.acc, c.copy, c.chain, c.hybrid, &sink);
        RS_CHECK(on.out == off.out, "%s: ids differ with the sink attached", c.name);
        const std::vector<std::uint32_t> serial(w.response.begin(), w.response.begin() + std::ptrdiff_t(off.out.size()));
        RS_CHECK(off.out == serial, "%s: ids differ from serial greedy", c.name);
        check_records(sink, on, c.name);
        // detached after 5 steps: exactly 5 calls, ids unchanged
        MemorySink part;
        const Run det = run(w, 600, c.acc, c.copy, c.chain, c.hybrid, &part, 5);
        RS_CHECK(det.out == off.out && part.recs.size() == 5, "%s: detach: %zu calls", c.name, part.recs.size());
        if (c.hybrid && c.edit) keep = sink;
    }
    writer_roundtrip(keep, "spec_round_sink_tests.tmp.srt");
    if (argc > 1) write_fixture(argv[1]);
    std::printf("round sink tests ok\n");
    return 0;
}
