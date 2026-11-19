// Block drafting (strix/spec/block_drafter.hpp) against a deterministic fake target and a fake chain drafter with a
// controlled accuracy: whatever the sources / windows / accuracy, the emitted ids equal serial greedy decode and the
// target's committed length equals prompt + emitted - 1 (the anchor is never committed). Costs are a virtual clock.
#include "strix/spec/block_drafter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace sp = strix::spec;

namespace {
#define BD_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

double g_clock = 0.0;  // virtual ms
double g_chain_step = 1.5;  // virtual ms per chain step
// Wall-clock disturbances (determinism test): jitter in [0, g_jitter) ms per call and a one-off g_spike on the first draft
// call (on CUDA the subset head's materialisation lands there). g_fixed = BlockConfig::fixed_costs with the fake's costs.
double g_jitter = 0.0, g_spike = 0.0;
std::uint64_t g_jitter_seed = 0;
bool g_fixed = false;
// Verify pre-launch: the fake chain keeps its drafts in fake device slots g_dev and hands them to the target's launch;
// slots are overwritten right after the launch (later device work) so a late read would be caught.
bool g_prelaunch = false;
std::uint32_t g_dev[64]{};
std::string g_cost_kv = "online";  // context cost of the fixed model (spec::set_context_cost)
// Tree-arbiter cost model of TreeTarget (rv_tree3): a tree window costs the linear window of the same rows + g_tree_extra
// (the measured per-window surcharge), and a window wider than g_step_rows rows pays g_step_ms more (dense row tile). The
// fixed seeds stay linear (10.2 + row_ms r), as on the device. g_gap_mode: RunnerChain margins, 0 none, 1 informative
// (a runner-up that is the target's token has gap 0.3, any other 4.0), 2 uninformative (hash in [0, 6)).
double g_tree_extra = 0.0, g_step_ms = 0.0;
std::uint32_t g_step_rows = 32, g_gap_mode = 0;
double noise(std::uint64_t k) { return g_jitter * double(mix(k * 0x9e3779b9ULL + g_jitter_seed) % 1000) / 1000.0; }

// The "model": response[i] is the greedy token at response position i (a pure function of the position: a valid
// deterministic target). Prompt = `prompt` tokens, which the response re-emits with sparse edits (code-edit class).
struct World {
    std::vector<std::uint32_t> prompt, response;
};

World make_world(const std::string& cls, std::uint32_t prompt_len, std::uint32_t n, std::uint64_t seed) {
    World w;
    for (std::uint32_t i = 0; i < prompt_len; ++i) w.prompt.push_back(std::uint32_t(mix(seed + i) % 900) + 10);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t t;
        if (cls == "edit") {  // copy of the prompt, an edit every ~23 tokens, wraps around
            t = w.prompt[i % prompt_len];
            if (mix(seed * 7 + i) % 23 == 0) t = std::uint32_t(mix(seed * 13 + i) % 900) + 10;
        } else if (cls == "loop") {  // short period
            t = w.prompt[i % 12];
        } else {  // prose: no structure
            t = std::uint32_t(mix(seed * 31 + i) % 900) + 10;
        }
        w.response.push_back(t);
    }
    return w;
}

struct FakeTarget final : sp::VerifyTarget {
    const World& w;
    std::uint32_t committed_rows = 0;  // positions in the target (anchor excluded)
    std::vector<std::uint32_t> last;
    std::uint32_t max_rows_seen = 0;
    std::uint64_t verifies = 0;
    // Backend contract: legal rows per verify (CUDA dense 32 / HIP MoE 16) and KV capacity in positions (prompt included).
    std::uint32_t legal_rows = 32, capacity = ~0U;
    explicit FakeTarget(const World& world) : w(world) {}
    // row i sits at stream position prompt + committed_rows + i; its logits predict response[committed_rows + i]
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        BD_CHECK(tokens.size() >= 1 && tokens.size() <= legal_rows, "verify rows %zu > legal %u", tokens.size(), legal_rows);
        BD_CHECK(w.prompt.size() + committed_rows + tokens.size() <= capacity, "verify rows %zu past capacity %u", tokens.size(), capacity);
        max_rows_seen = std::max<std::uint32_t>(max_rows_seen, std::uint32_t(tokens.size()));
        ++verifies;
        greedy.clear();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto pos = committed_rows + i;
            greedy.push_back(pos + 1 < w.response.size() ? w.response[pos + 1] : 7);
        }
        last.assign(tokens.begin(), tokens.end());
        g_clock += 10.0 + 0.15 * double(tokens.size()) + noise(verifies);  // memory-bound verify: nearly flat in rows
    }
    void commit(std::uint32_t k) override {
        BD_CHECK(k >= 1 && k <= last.size(), "commit %u of %zu", k, last.size());
        committed_rows += k;
        g_clock += 0.2;
    }
    // Pre-launch: the rows are resolved at launch time (as the device gather runs in stream order behind the draft).
    std::vector<std::uint32_t> pending;
    std::uint64_t launched = 0;
    bool verify_launch_fed(std::span<const std::uint32_t> host_rows, const sp::DraftFeed& f) override {
        if (!g_prelaunch) return false;
        BD_CHECK(pending.empty(), "pre-launch while a pre-launched verify is pending");
        pending.assign(host_rows.begin(), host_rows.end());
        for (std::uint32_t j = 0; j < f.k; ++j) pending.push_back(f.slot[j] == sp::DraftFeed::kHost ? f.host_id[j] : g_dev[f.slot[j]]);
        ++launched;
        return true;
    }
    void verify_collect(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        BD_CHECK(std::equal(tokens.begin(), tokens.end(), pending.begin(), pending.end()), "pre-launched rows differ from the block");
        pending.clear();
        verify(tokens, greedy);
    }
};

struct FakeChain final : sp::ChainDrafter {
    const World& w;
    double accuracy;
    std::uint32_t stream_len = 0;  // response tokens emitted so far (the anchor is the last of them)
    std::uint64_t calls = 0;
    FakeChain(const World& world, double acc) : w(world), accuracy(acc) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        ++calls;
        out.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t pos = std::uint64_t(stream_len) + prefix.size() + j;  // response position predicted
            const bool right = double(mix(pos * 977 + 5) % 1000) / 1000.0 < accuracy;
            out.push_back(pos < w.response.size() && right ? w.response[pos] : 5);
        }
        g_clock += g_chain_step * double(prefix.size() + k) + noise(calls + (1ULL << 40)) + (calls == 1 ? g_spike : 0.0);
    }
    void accept(std::span<const std::uint32_t> emitted) override { stream_len += std::uint32_t(emitted.size()); }
    // Same drafts and clock as draft(); draft 0 of every other call is a host row (as the MTP chain's unflushed d1).
    bool draft_prelaunch(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out,
                         const sp::PrelaunchFn& launch) override {
        if (!g_prelaunch || k == 0 || k + 3 > 64) return false;
        draft(prefix, k, out);
        sp::DraftFeed f;
        f.k = k;
        for (std::uint32_t j = 0; j < k; ++j) {
            if (j == 0 && calls % 2 == 1) {
                f.slot[0] = sp::DraftFeed::kHost;
                f.host_id[0] = out[0];
            } else {
                f.slot[j] = 3 + j;
                g_dev[3 + j] = out[j];
            }
        }
        launch(prefix, f);
        for (std::uint32_t j = 0; j < k; ++j) g_dev[3 + j] = 0xDEADU;
        return true;
    }
};

struct Run {
    std::vector<std::uint32_t> out;
    std::uint64_t steps = 0, hybrid = 0, chain_blocks = 0, copy_blocks = 0, serial_steps = 0;
    std::uint32_t max_rows = 0;
    std::vector<std::uint32_t> trace;  // per step: source * 100 + drafts verified
    std::uint64_t prelaunched = 0, launched = 0;
    std::vector<double> rounds;  // every RoundSink record, flattened (fields, ms and the three id lists)
};

// Round trace capture: the pre-launched verify must feed the sink the same record as the read-then-verify order.
struct RoundCapture final : sp::RoundSink {
    std::vector<double>* out = nullptr;
    void round(const sp::RoundRecord& r) override {
        for (const double v : {double(r.context), double(int(r.source)), double(r.anchor), double(r.k), double(r.rows),
                               double(r.accepted), double(r.copy), double(r.ext), r.verify_ms, r.draft_ms})
            out->push_back(v);
        for (const auto ids : {r.drafts, r.greedy, r.emitted}) {
            out->push_back(double(ids.size()));
            for (const std::uint32_t id : ids) out->push_back(double(id));
        }
    }
};

// legal_rows / capacity: the caller binds StepParams::max_rows = min(backend rows, context room), as the engine apps do.
Run run(const World& w, std::uint32_t n_tokens, double accuracy, bool copy, bool chain, bool hybrid, bool seed_costs,
        std::uint32_t legal_rows = 32, std::uint32_t capacity = ~0U,
        std::uint32_t frontier_every = sp::ArbiterConfig{}.frontier_every) {
    sp::BlockConfig cfg;
    cfg.copy = copy; cfg.chain = chain; cfg.hybrid = hybrid;
    cfg.arbiter.frontier_every = frontier_every;
    cfg.clock_ms = [] { return g_clock; };
    if (g_fixed) {  // the fake's true costs as the fixed model: chain g_chain_step per position, verify 10 + 0.15 r + commit 0.2
        cfg.fixed_costs = true;
        cfg.chain_seed_flush_ms = cfg.chain_seed_step_ms = g_chain_step;
        sp::set_context_cost(cfg.arbiter, g_cost_kv);
    }
    FakeChain fc(w, accuracy);
    sp::BlockDrafter drafter(cfg, chain || hybrid ? &fc : nullptr);
    if (g_fixed)
        for (std::uint32_t r = 1; r <= 32; ++r) drafter.seed_verify_ms(r, 10.2 + 0.15 * double(r));
    else if (seed_costs)
        for (const std::uint32_t r : {1U, 2U, 4U, 8U, 16U, 32U}) drafter.seed_verify_ms(r, 10.0 + 0.15 * double(r));
    FakeTarget target(w);
    target.legal_rows = legal_rows;
    target.capacity = capacity;
    // prefill: the first token (greedy after the prompt) is the anchor; the stream = prompt + anchor
    drafter.commit(w.prompt);
    Run r;
    r.out.push_back(w.response[0]);
    drafter.commit(std::span<const std::uint32_t>(&r.out[0], 1));
    fc.stream_len = 1;
    sp::BlockSpeculator spec(drafter, target);
    spec.set_prelaunch(g_prelaunch);
    RoundCapture capture;
    capture.out = &r.rounds;
    spec.set_round_sink(&capture);
    spec.start(r.out[0]);
    target.committed_rows = 0;
    while (r.out.size() < n_tokens) {
        sp::StepParams p;
        p.remaining = n_tokens - std::uint32_t(r.out.size());
        p.context = w.prompt.size() + r.out.size();
        const std::uint64_t used = w.prompt.size() + target.committed_rows;
        p.max_rows = std::uint32_t(std::min<std::uint64_t>(legal_rows, capacity > used ? capacity - used : 0));
        const auto before = r.out.size();
        const auto verifies = target.verifies;
        const auto rec = spec.step(r.out, p);
        if (p.max_rows == 0) {  // no context room: nothing is verified or committed, the caller stops
            BD_CHECK(rec.emitted == 0 && target.verifies == verifies && r.out.size() == before, "step past the context room");
            break;
        }
        BD_CHECK(rec.emitted >= 1 && rec.emitted <= p.remaining, "emitted %u remaining %u", rec.emitted, p.remaining);
        BD_CHECK(r.out.size() == before + rec.emitted, "out size");
        ++r.steps;
        r.trace.push_back(std::uint32_t(rec.source) * 100 + rec.proposed);
        r.max_rows = std::max(r.max_rows, rec.rows);
        r.hybrid += rec.copy > 0 && rec.ext > 0;
        r.chain_blocks += rec.source == sp::Source::model;
        r.copy_blocks += rec.source == sp::Source::lookup;
        r.serial_steps += rec.source == sp::Source::serial;
        // bookkeeping: committed target rows = emitted so far - 1 (the anchor is not in the target)
        BD_CHECK(target.committed_rows + 1 == r.out.size(), "committed %u vs emitted %zu", target.committed_rows, r.out.size());
    }
    r.prelaunched = spec.prelaunched_steps();
    r.launched = target.launched;
    return r;
}

// ---- token tree (sibling rows)
// The tree target computes each row's greedy from the row's PATH (ancestor tokens): a row whose path is the true stream
// predicts the true next token, any other row a hash of its path - so the walk can only use rows whose context is right.
struct TreeTarget final : sp::VerifyTarget {
    const World& w;
    std::uint32_t committed_rows = 0, tree_width = 32;
    double row_ms = 0.15;
    std::uint64_t trees = 0, linear = 0;
    sp::TreeRows last_tree;
    std::vector<std::uint32_t> last_linear;
    explicit TreeTarget(const World& world) : w(world) {}
    [[nodiscard]] std::uint32_t tree_rows() const override { return tree_width; }
    std::uint32_t truth(std::uint64_t pos) const { return pos < w.response.size() ? w.response[pos] : 7; }
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        ++linear;
        greedy.clear();
        for (std::size_t i = 0; i < tokens.size(); ++i) greedy.push_back(truth(committed_rows + i + 1));
        last_linear.assign(tokens.begin(), tokens.end());
        g_clock += 10.0 + row_ms * double(tokens.size()) + (tokens.size() > g_step_rows ? g_step_ms : 0.0);
    }
    void commit(std::uint32_t k) override {
        BD_CHECK(k >= 1 && k <= last_linear.size(), "linear commit %u of %zu", k, last_linear.size());
        committed_rows += k;
        g_clock += 0.2;
    }
    void verify_tree(const sp::TreeRows& t, std::vector<std::uint32_t>& greedy) override {
        ++trees;
        const std::size_t n = t.tokens.size();
        BD_CHECK(n >= 2 && n <= tree_width && t.parent.size() == n && t.depth.size() == n && t.node.size() == n, "tree shape");
        BD_CHECK(t.parent[0] == -1 && t.depth[0] == 0, "row 0 is the anchor");
        std::vector<bool> leaf(n, true), right(n, false);
        for (std::size_t r = 1; r < n; ++r) {
            BD_CHECK(t.parent[r] >= 0 && std::size_t(t.parent[r]) < r, "parent-closed, topological");
            leaf[std::size_t(t.parent[r])] = false;
            BD_CHECK(t.depth[r] == t.depth[std::size_t(t.parent[r])] + 1, "depth = parent depth + 1");
        }
        for (std::size_t r = 1; r < n; ++r) {  // recurrent layout: the parent is the last non-leaf row before r
            std::int32_t last = 0;
            for (std::size_t q = 1; q < r; ++q)
                if (!leaf[q]) last = std::int32_t(q);
            BD_CHECK(t.parent[r] == last, "row %zu parent %d, last non-leaf %d", r, t.parent[r], last);
        }
        greedy.assign(n, 0);
        right[0] = true;  // the anchor is the last emitted (true) token
        for (std::size_t r = 0; r < n; ++r) {
            if (r > 0) right[r] = right[std::size_t(t.parent[r])] && t.tokens[r] == truth(committed_rows + t.depth[r]);
            std::uint64_t h = 0;
            for (std::int64_t q = std::int64_t(r); q >= 0; q = t.parent[std::size_t(q)]) h = mix(h * 1315423911ULL + t.tokens[std::size_t(q)]);
            greedy[r] = right[r] ? truth(committed_rows + t.depth[r] + 1) : std::uint32_t(h % 900) + 10;
        }
        last_tree = t;
        g_clock += 10.0 + row_ms * double(n) + (n > g_step_rows ? g_step_ms : 0.0) + g_tree_extra;
    }
    void commit_path(std::span<const std::uint32_t> path) override {
        BD_CHECK(!path.empty() && path[0] == 0, "path starts at the anchor");
        for (std::size_t i = 1; i < path.size(); ++i) {
            BD_CHECK(last_tree.parent[path[i]] == std::int32_t(path[i - 1]), "path is a root path");
            BD_CHECK(last_tree.tokens[path[i]] == truth(committed_rows + i), "committed row %zu is not the stream", i);
        }
        committed_rows += std::uint32_t(path.size());
        g_clock += 0.2;
    }
};

// Chain with runner-ups: a wrong draft's runner-up is the true token with probability q_true; drafts 5, wrong runner-ups 6.
struct RunnerChain final : sp::ChainDrafter {
    const World& w;
    double accuracy, q_true;
    bool report = true;
    std::uint32_t stream_len = 0;
    std::vector<std::uint32_t> ru;
    std::vector<float> gp;
    std::uint64_t paths = 0;
    RunnerChain(const World& world, double acc, double q) : w(world), accuracy(acc), q_true(q) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        out.clear();
        ru.clear();
        gp.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t pos = std::uint64_t(stream_len) + prefix.size() + j;
            const bool right = double(mix(pos * 977 + 5) % 1000) / 1000.0 < accuracy;
            const bool alt = double(mix(pos * 389 + 11) % 1000) / 1000.0 < q_true;
            const std::uint32_t t = pos < w.response.size() ? w.response[pos] : 7;
            out.push_back(right ? t : 5);
            ru.push_back(!right && alt ? t : 6);
            if (g_gap_mode == 1) gp.push_back(!right && alt ? 0.3f : 4.0f);
            else if (g_gap_mode == 2) gp.push_back(float(mix(pos * 131 + 3) % 600) / 100.0f);
        }
        g_clock += 0.5 * double(prefix.size() + k);
    }
    void accept(std::span<const std::uint32_t> emitted) override { stream_len += std::uint32_t(emitted.size()); }
    void accept_path(std::span<const std::uint32_t> emitted, std::span<const std::uint32_t> rows) override {
        BD_CHECK(rows.size() == emitted.size(), "path rows %zu vs emitted %zu", rows.size(), emitted.size());
        ++paths;
        accept(emitted);
    }
    [[nodiscard]] std::span<const std::uint32_t> runner_ups() const override {
        return report ? std::span<const std::uint32_t>(ru) : std::span<const std::uint32_t>();
    }
    [[nodiscard]] std::span<const float> runner_up_gaps() const override {
        return report ? std::span<const float>(gp) : std::span<const float>();
    }
};

struct TreeRun {
    std::vector<std::uint32_t> out, trace;
    sp::SpecStats stats;
    std::uint64_t steps = 0, sib_rows = 0, taken = 0;
    double ms = 0.0;
};

// tree_width 0 = a linear target (reports no tree rows)
TreeRun run_tree(const World& w, std::uint32_t n_tokens, double acc, double q, std::uint32_t tree_width, bool report,
                 double row_ms, bool copy) {
    g_clock = 0.0;
    sp::BlockConfig cfg;
    cfg.copy = copy;
    cfg.fixed_costs = true;
    cfg.chain_seed_flush_ms = cfg.chain_seed_step_ms = 0.5;
    cfg.clock_ms = [] { return g_clock; };
    RunnerChain ch(w, acc, q);
    ch.report = report;
    sp::BlockDrafter drafter(cfg, &ch);
    for (std::uint32_t r = 1; r <= 32; ++r) drafter.seed_verify_ms(r, 10.2 + row_ms * double(r));
    TreeTarget target(w);
    target.tree_width = tree_width;
    target.row_ms = row_ms;
    drafter.commit(w.prompt);
    TreeRun r;
    r.out.push_back(w.response[0]);
    drafter.commit(std::span<const std::uint32_t>(&r.out[0], 1));
    ch.stream_len = 1;
    sp::BlockSpeculator spec(drafter, target);
    spec.start(r.out[0]);
    while (r.out.size() < n_tokens) {
        sp::StepParams p;
        p.remaining = n_tokens - std::uint32_t(r.out.size());
        p.context = w.prompt.size() + r.out.size();
        p.max_rows = 32;
        const auto before = r.out.size();
        const auto rec = spec.step(r.out, p);
        BD_CHECK(rec.emitted >= 1 && rec.emitted <= p.remaining && r.out.size() == before + rec.emitted, "tree step emitted");
        BD_CHECK(rec.emitted == rec.accepted + 1 + (rec.sibling_taken ? 1U : 0U), "emitted = main + sibling + next");
        BD_CHECK(target.committed_rows + 1 == r.out.size(), "committed %u vs emitted %zu", target.committed_rows, r.out.size());
        ++r.steps;
        r.sib_rows += rec.siblings;
        r.taken += rec.sibling_taken;
        r.trace.push_back(std::uint32_t(rec.source) * 100 + rec.proposed);
    }
    r.stats = drafter.stats();
    r.ms = g_clock;
    return r;
}

void tree_unit_tests() {
    // accept_tree on a hand tree: anchor; depth 1 sibling s1 (leaf) + chain c1; depth 2 chain c2 (no sibling)
    sp::TreeRows t;
    t.tokens = {100, 21, 11, 12};
    t.parent = {-1, 0, 0, 2};
    t.node = {0, -1, 1, 2};
    t.depth = {0, 1, 1, 2};
    auto a = sp::accept_tree(t, std::vector<std::uint32_t>{11, 999, 12, 55});  // chain all right
    BD_CHECK(a.path == (std::vector<std::uint32_t>{0, 2, 3}) && a.main == 2 && !a.sibling, "chain path");
    BD_CHECK(a.emitted == (std::vector<std::uint32_t>{11, 12, 55}), "chain emitted");
    a = sp::accept_tree(t, std::vector<std::uint32_t>{21, 77, 12, 55});  // depth 1 = the runner-up
    BD_CHECK(a.path == (std::vector<std::uint32_t>{0, 1}) && a.main == 0 && a.sibling, "sibling path");
    BD_CHECK(a.emitted == (std::vector<std::uint32_t>{21, 77}), "sibling: runner-up + the target's next token");
    a = sp::accept_tree(t, std::vector<std::uint32_t>{33, 77, 12, 55});  // neither: the correction only
    BD_CHECK(a.path == (std::vector<std::uint32_t>{0}) && a.emitted == (std::vector<std::uint32_t>{33}), "reject at depth 1");
    // layout from a block: siblings at depths 1 and 3 of a 3-chain
    sp::Block b;
    b.drafts = {11, 12, 13};
    b.alts = {21, sp::kNoToken, 23};
    b.sib = {1, 0, 1};
    b.siblings = 2;
    sp::TreeRows u;
    b.tree_rows(100, u);
    BD_CHECK(u.tokens == (std::vector<std::uint32_t>{100, 21, 11, 12, 23, 13}), "layout tokens");
    BD_CHECK(u.parent == (std::vector<std::int32_t>{-1, 0, 0, 2, 3, 3}), "layout parents");
    BD_CHECK(u.depth == (std::vector<std::uint32_t>{0, 1, 1, 2, 3, 3}) && b.rows() == 6, "layout depths / rows");
}

int tree_tests() {
    tree_unit_tests();
    int cases = 0;
    const std::uint32_t n = 1200;
    std::uint64_t placed = 0, taken = 0;
    for (const std::string cls : {"prose", "edit", "loop"})
        for (const double acc : {0.3, 0.63, 0.85})
            for (const double q : {0.0, 0.35, 0.8})
                for (const bool copy : {false, true}) {
                    const auto w = make_world(cls, 40, n, 7 + std::uint64_t(acc * 100));
                    // linear target + runner-ups vs linear target without: identical decisions and ids (probe only)
                    const auto lin = run_tree(w, n, acc, q, 0, true, 0.15, copy);
                    const auto base = run_tree(w, n, acc, q, 0, false, 0.15, copy);
                    BD_CHECK(lin.trace == base.trace && lin.out == base.out && lin.sib_rows == 0, "probe changed a linear run");
                    // cheap rows (RTX dense class): siblings placed; ids == serial
                    const auto tr = run_tree(w, n, acc, q, 32, true, 0.113, copy);
                    for (std::uint32_t i = 0; i < n; ++i)
                        BD_CHECK(tr.out[i] == w.response[i], "%s acc %.2f q %.2f copy %d: token %u differs", cls.c_str(), acc, q,
                                 int(copy), i);
                    // expensive rows (MoE expert-union class, ~5.6 ms per row): a sibling is placed only where it pays
                    // (high q), so the run is never slower than the linear one
                    const auto ex = run_tree(w, n, acc, q, 32, true, 5.6, copy);
                    const auto ex_lin = run_tree(w, n, acc, q, 0, true, 5.6, copy);
                    BD_CHECK(ex.ms <= ex_lin.ms * 1.005 && (q > 0.0 || ex.sib_rows * 100 <= ex.steps),
                             "%s acc %.2f q %.2f: %llu sibling rows in %llu steps at 5.6 ms per row, %.0f vs linear %.0f ms",
                             cls.c_str(), acc, q, (unsigned long long)ex.sib_rows, (unsigned long long)ex.steps, ex.ms, ex_lin.ms);
                    for (std::uint32_t i = 0; i < n; ++i) BD_CHECK(ex.out[i] == w.response[i], "expensive-row run differs at %u", i);
                    // the probe: measured hit rate tracks q (runner-ups at rejections)
                    std::uint64_t trials = 0, hits = 0;
                    for (std::size_t d = 0; d < tr.stats.sibling_trials.size(); ++d) {
                        trials += tr.stats.sibling_trials[d];
                        hits += tr.stats.sibling_hits[d];
                    }
                    if (trials > 200) {
                        const double m = double(hits) / double(trials);
                        BD_CHECK(std::abs(m - q) < 0.08, "%s acc %.2f q %.2f: probe measured %.3f", cls.c_str(), acc, q, m);
                    }
                    BD_CHECK(tr.taken == tr.stats.sibling_taken && tr.sib_rows == tr.stats.sibling_rows, "stats");
                    if (q >= 0.35 && !copy && acc <= 0.63)
                        BD_CHECK(tr.sib_rows > 0 && tr.taken > 0 && tr.ms < lin.ms,
                                 "%s acc %.2f q %.2f: tree must place and win (rows %llu taken %llu, %.0f vs %.0f ms)", cls.c_str(),
                                 acc, q, (unsigned long long)tr.sib_rows, (unsigned long long)tr.taken, tr.ms, lin.ms);
                    if (q == 0.0) BD_CHECK(tr.taken == 0, "q 0 cannot take a sibling");
                    placed += tr.sib_rows;
                    taken += tr.taken;
                    ++cases;
                }
    std::printf("tree tests ok: %d cases, sibling rows %llu, taken %llu\n", cases, (unsigned long long)placed,
                (unsigned long long)taken);
    return cases;
}

// rv_tree3: sibling placement priced by measured walls. The device showed a per-window tree surcharge (+0.5..0.7 ms at
// equal rows) and a step-shaped row cost (8 -> 9 rows +1.2 ms) that the linear seeds do not see; the seeds here stay
// linear too. Invariants: ids == serial; the tree run is never slower than the linear run (1 %: exploration of a shape
// before its wall is known + path dependence, see below); a surcharge no sibling set pays stops placement; the learned
// surcharge equals the modelled one; informative drafter gaps raise the taken / placed ratio and never cost time.
int tree_arbiter_tests() {
    const std::uint32_t n = 1200;
    int cases = 0;
    double worst = 0.0;
    for (const std::string cls : {"prose", "edit", "loop"})
        for (const double acc : {0.3, 0.63, 0.85})
            for (const double q : {0.1, 0.35, 0.8})
                for (const double extra : {0.0, 0.7, 3.0})
                    for (const bool step : {false, true}) {
                        const auto w = make_world(cls, 40, n, 11 + std::uint64_t(acc * 100));
                        g_tree_extra = extra;
                        g_step_rows = step ? 8 : 32;
                        g_step_ms = step ? 1.2 : 0.0;
                        const auto tr = run_tree(w, n, acc, q, 32, true, 0.113, false);
                        const auto lin = run_tree(w, n, acc, q, 0, true, 0.113, false);
                        for (std::uint32_t i = 0; i < n; ++i)
                            BD_CHECK(tr.out[i] == w.response[i] && lin.out[i] == w.response[i], "arbiter run differs at %u", i);
                        worst = std::max(worst, tr.ms / lin.ms);
                        // 1 %: once one tree window changes the acceptance history, the two runs make different chain
                        // decisions; that path dependence alone moves the total by up to ~0.9 % here (cases with <= 10 tree
                        // windows, whose own cost is < 0.1 %). The device gate (on >= off - 0.5 %) is the kit's.
                        BD_CHECK(tr.ms <= lin.ms * 1.01,
                                 "%s acc %.2f q %.2f extra %.1f step %d: tree %.0f vs linear %.0f ms (rows %llu taken %llu windows %llu)",
                                 cls.c_str(), acc, q, extra, int(step), tr.ms, lin.ms, (unsigned long long)tr.sib_rows,
                                 (unsigned long long)tr.taken, (unsigned long long)tr.stats.tree_windows);
                        if (!step && extra > 0.0 && tr.stats.tree_windows > 0 && tr.stats.tree_surcharge_ms != 0.0)
                            BD_CHECK(std::abs(tr.stats.tree_surcharge_ms - extra) < 0.05, "learned surcharge %.3f vs %.1f",
                                     tr.stats.tree_surcharge_ms, extra);
                        if (extra == 3.0 && q <= 0.35)  // a surcharge no sibling set pays: placement stops after exploration
                            BD_CHECK(tr.stats.tree_windows * 20 <= tr.steps, "%s acc %.2f q %.2f: %llu tree windows in %llu steps",
                                     cls.c_str(), acc, q, (unsigned long long)tr.stats.tree_windows, (unsigned long long)tr.steps);
                        ++cases;
                    }
    // Drafter gaps: informative margins (hit 0.3, miss 4.0) vs uninformative ones at the device's en-prose surcharge.
    std::uint64_t rows_i = 0, taken_i = 0, rows_u = 0, taken_u = 0;
    for (const double acc : {0.3, 0.63})
        for (const double q : {0.1, 0.35}) {
            const auto w = make_world("prose", 40, n, 17 + std::uint64_t(acc * 100));
            g_tree_extra = 0.7;
            g_step_rows = 32;
            g_step_ms = 0.0;
            g_gap_mode = 1;
            const auto inf = run_tree(w, n, acc, q, 32, true, 0.113, false);
            g_gap_mode = 2;
            const auto uni = run_tree(w, n, acc, q, 32, true, 0.113, false);
            BD_CHECK(inf.ms <= uni.ms * 1.005, "acc %.2f q %.2f: informative gaps %.0f vs %.0f ms", acc, q, inf.ms, uni.ms);
            for (std::uint32_t i = 0; i < n; ++i) BD_CHECK(inf.out[i] == w.response[i], "gap run differs at %u", i);
            rows_i += inf.sib_rows;
            taken_i += inf.taken;
            rows_u += uni.sib_rows;
            taken_u += uni.taken;
            ++cases;
        }
    BD_CHECK(rows_i > 0 && rows_u > 0 && double(taken_i) / double(rows_i) > 1.5 * double(taken_u) / double(rows_u),
             "gap selector: taken/placed %llu/%llu vs uninformative %llu/%llu", (unsigned long long)taken_i,
             (unsigned long long)rows_i, (unsigned long long)taken_u, (unsigned long long)rows_u);
    g_tree_extra = g_step_ms = 0.0;
    g_step_rows = 32;
    g_gap_mode = 0;
    std::printf("tree arbiter tests ok: %d cases, worst tree/linear %.4f, gap selector taken/placed %llu/%llu vs %llu/%llu\n",
                cases, worst, (unsigned long long)taken_i, (unsigned long long)rows_i, (unsigned long long)taken_u,
                (unsigned long long)rows_u);
    return cases;
}
}  // namespace

int main() {
    const std::uint32_t n = 1500;
    int cases = 0;
    std::uint64_t hybrids = 0, chains = 0, copies = 0;
    double tok_ms_serial = 0.0, tok_ms_all_loop = 0.0, tok_ms_all_prose = 0.0;
    for (const std::string cls : {"edit", "loop", "prose"})
        for (const std::uint32_t plen : {12U, 40U, 700U}) {
            const auto w = make_world(cls, plen, n, 100 + plen);
            for (const double acc : {0.0, 0.55, 0.9})
                for (int mode = 0; mode < 5; ++mode) {  // serial, copy, chain, copy+chain, all
                    const bool copy = mode == 1 || mode >= 3, chain = mode == 2 || mode >= 3, hybrid = mode == 4;
                    for (const bool seed : {false, true}) {
                        g_clock = 0.0;
                        const auto r = run(w, n, acc, copy, chain, hybrid, seed);
                        for (std::uint32_t i = 0; i < n; ++i)
                            BD_CHECK(r.out[i] == w.response[i], "%s plen %u acc %.2f mode %d: token %u differs", cls.c_str(), plen, acc, mode, i);
                        BD_CHECK(r.max_rows <= 32, "rows");
                        if (mode == 0) BD_CHECK(r.steps == n - 1 && r.serial_steps == n - 1, "serial mode must not draft");
                        if (mode == 1) BD_CHECK(r.chain_blocks == 0 && r.hybrid == 0, "copy only");
                        if (mode == 2) BD_CHECK(r.copy_blocks == 0, "chain only");
                        ++cases;
                        hybrids += r.hybrid; chains += r.chain_blocks; copies += r.copy_blocks;
                        if (plen == 700 && acc == 0.9 && seed) {
                            if (cls == "edit" && mode == 0) tok_ms_serial = g_clock / n;
                            if (cls == "edit" && mode == 4) tok_ms_all_loop = g_clock / n;
                            if (cls == "prose" && mode == 4) tok_ms_all_prose = g_clock / n;
                        }
                    }
                }
        }
    {  // a cheap chain makes the hybrid extension pay: short copy spans (period 12) extended by chain tokens
        g_chain_step = 0.02;
        for (const std::string cls : {"edit", "loop"}) {
            const auto w = make_world(cls, 12, n, 77);
            g_clock = 0.0;
            const auto r = run(w, n, 0.9, true, true, true, true);
            for (std::uint32_t i = 0; i < n; ++i) BD_CHECK(r.out[i] == w.response[i], "cheap chain %s: token %u differs", cls.c_str(), i);
            hybrids += r.hybrid;
        }
        g_chain_step = 1.5;
    }
    {  // backend legal rows below max_block (HIP MoE windows: 16 rows, --max-block 32) and a context room that runs out:
       // every verify stays legal (FakeTarget checks), partial accepts commit a prefix, output stays serial-exact
        g_chain_step = 0.02;  // cheap chain: hybrid extensions are offered and must respect the 16-row cap
        for (const std::string cls : {"edit", "loop", "prose"}) {
            const auto w = make_world(cls, 40, n, 911);
            for (int mode = 1; mode < 5; ++mode) {
                const bool copy = mode == 1 || mode >= 3, chain = mode == 2 || mode >= 3, hybrid = mode == 4;
                g_clock = 0.0;
                const auto r16 = run(w, n, 0.9, copy, chain, hybrid, true, 16U);
                BD_CHECK(r16.out.size() == n && r16.max_rows <= 16, "%s mode %d: 16-row backend rows %u", cls.c_str(), mode, r16.max_rows);
                for (std::uint32_t i = 0; i < n; ++i) BD_CHECK(r16.out[i] == w.response[i], "16-row %s mode %d: token %u differs", cls.c_str(), mode, i);
                const std::uint32_t room = 301;  // positions after the prompt: the run stops at 301 emitted (anchor not committed)
                g_clock = 0.0;
                const auto rc = run(w, n, 0.9, copy, chain, hybrid, true, 16U, std::uint32_t(w.prompt.size()) + room - 1U);
                BD_CHECK(rc.out.size() == room, "%s mode %d: context room run emitted %zu, want %u", cls.c_str(), mode, rc.out.size(), room);
                for (std::uint32_t i = 0; i < room; ++i) BD_CHECK(rc.out[i] == w.response[i], "room %s mode %d: token %u differs", cls.c_str(), mode, i);
            }
        }
        g_chain_step = 1.5;
    }
    BD_CHECK(copies > 100 && chains > 100, "sources never used: copy %llu chain %llu", (unsigned long long)copies, (unsigned long long)chains);
    BD_CHECK(hybrids > 0, "hybrid never chosen");
    // The mixed drafter must pay on structure (>= 3x fewer ms per token than serial on the edit class)...
    BD_CHECK(tok_ms_all_loop * 3.0 < tok_ms_serial, "edit class ms/token %.2f vs serial %.2f", tok_ms_all_loop, tok_ms_serial);
    // ...and must not lose to serial on structureless text with an accurate chain (acc 0.9) nor by more than 10 % at acc 0.
    BD_CHECK(tok_ms_all_prose < tok_ms_serial, "prose class ms/token %.2f vs serial %.2f", tok_ms_all_prose, tok_ms_serial);
    {  // inaccurate chain on prose: the arbiter must bar it (probe cost only)
        const auto w = make_world("prose", 700, n, 3);
        g_clock = 0.0;
        const auto r = run(w, n, 0.0, true, true, true, true);
        BD_CHECK(g_clock / n < tok_ms_serial * 1.10, "useless chain costs %.2f ms/token vs serial %.2f", g_clock / n, tok_ms_serial);
        (void)r;
    }
    {  // a chain at ~66 % per-token acceptance on prose must keep drafting: misses in a row are routine, not a retreat
        const auto w = make_world("prose", 700, n, 5);
        g_clock = 0.0;
        const auto r = run(w, n, 0.66, true, true, true, true);
        BD_CHECK(double(r.serial_steps) < 0.15 * double(r.steps), "66 %% chain: %llu of %llu passes serial", (unsigned long long)r.serial_steps, (unsigned long long)r.steps);
    }
    {  // frontier probe on/off (ArbiterConfig::frontier_every): emitted ids equal the serial truth in both forms, every
       // prefix commit stays legal; the probe changes only how deep the chain drafts (verify cost and clock, never tokens)
        const auto w = make_world("prose", 700, n, 11);
        for (const std::uint32_t every : {0U, 8U, 1U}) {
            g_clock = 0.0;
            g_chain_step = 0.65;  // CUDA dense ratio: 0.85 ms chain step per ~13.2 ms verify, on this 10 ms fake
            const auto r = run(w, n, 0.76, false, true, false, true, 32, ~0U, every);
            g_chain_step = 1.5;
            BD_CHECK(r.out.size() == n, "frontier %u: %zu tokens", every, r.out.size());
            for (std::uint32_t i = 0; i < n; ++i)
                BD_CHECK(r.out[i] == w.response[i], "frontier %u: token %u differs from serial", every, i);
            std::printf("frontier %u: %llu steps for %u tokens, %.1f virtual ms\n", every, (unsigned long long)r.steps, n, g_clock);
        }
    }
    {  // Fixed-cost controller (BlockConfig::fixed_costs): one prompt under three wall clocks (exact; jitter < 3 ms per
       // call; jitter + a 12 ms spike on the first draft call). With the context cost off (or fixed) it takes the SAME
       // decision at every step; with the default online fit (a clipped 3-term fit of the verify walls, not per-window walls)
       // the decisions may follow the clock but the spike / jitter must not pin k (the learned form's failure); all emit
       // the serial truth. The learned form is printed beside them.
        const auto w = make_world("prose", 40, n, 21);
        g_chain_step = 0.64;  // per drafted position vs a ~10.5 ms verify: the CUDA dense CJK subset head ratio
        std::vector<std::uint32_t> ref;
        std::uint32_t learned_differs = 0, online_differs = 0;
        std::vector<std::uint32_t> learned_ref, online_ref;
        for (int variant = 0; variant < 3; ++variant)
            for (const int arm : {0, 1, 2}) {  // 0 = fixed, context cost off; 1 = fixed, online (default); 2 = learned
                const bool fixed = arm != 2;
                g_fixed = fixed;
                g_cost_kv = arm == 0 ? "off" : "online";
                g_jitter = variant ? 3.0 : 0.0;
                g_spike = variant == 2 ? 12.0 : 0.0;
                g_jitter_seed = std::uint64_t(variant) * 1000003ULL;
                g_clock = 0.0;
                const auto r = run(w, n, 0.6, true, true, true, true);
                for (std::uint32_t i = 0; i < n; ++i)
                    BD_CHECK(r.out[i] == w.response[i], "determinism variant %d fixed %d: token %u differs", variant, int(fixed), i);
                std::uint32_t hist[33] = {}, serial = 0;
                for (const auto t : r.trace) {
                    if (t / 100 == std::uint32_t(sp::Source::model)) ++hist[std::min<std::uint32_t>(t % 100, 32)];
                    serial += t / 100 == std::uint32_t(sp::Source::serial);
                }
                std::uint32_t mode_k = 0;
                for (std::uint32_t k = 1; k <= 32; ++k) if (hist[k] > hist[mode_k]) mode_k = k;
                std::printf("determinism variant %d %s: %zu steps, serial %u, model k mode %u (%u passes), %.1f virtual ms\n", variant,
                            arm == 0 ? "fixed off   " : arm == 1 ? "fixed online" : "learned     ", r.trace.size(), serial, mode_k,
                            hist[mode_k], g_clock);
                if (arm == 1) {
                    if (variant == 0) online_ref = r.trace;
                    else online_differs += r.trace != online_ref;
                    BD_CHECK(mode_k >= 2 && mode_k <= 5, "fixed online: model k mode %u, want 2..5 (variant %d)", mode_k, variant);
                    BD_CHECK(serial * 10 < r.trace.size(), "fixed online: %u serial of %zu steps", serial, r.trace.size());
                } else if (fixed) {
                    if (variant == 0) ref = r.trace;
                    std::size_t at = 0;
                    while (at < ref.size() && at < r.trace.size() && ref[at] == r.trace[at]) ++at;
                    BD_CHECK(r.trace == ref, "fixed costs: variant %d decision differs at step %zu (%u vs %u)", variant, at,
                             at < r.trace.size() ? r.trace[at] : 0U, at < ref.size() ? ref[at] : 0U);
                    // fake optimum at a 0.6 per-position rate: k = 3 (k = 2..5 within 4 %)
                    BD_CHECK(mode_k >= 2 && mode_k <= 5, "fixed costs: model k mode %u, want 2..5", mode_k);
                    BD_CHECK(serial * 10 < r.trace.size(), "fixed costs: %u serial of %zu steps", serial, r.trace.size());
                } else {
                    if (variant == 0) learned_ref = r.trace;
                    else learned_differs += r.trace != learned_ref;
                }
            }
        std::printf("determinism: fixed costs (context cost off) identical in 3 of 3 clocks; online fit differs in %u of 2 (k mode "
                    "kept); learned costs differ in %u of 2\n", online_differs, learned_differs);
        g_fixed = false;
        g_cost_kv = "online";
        g_jitter = g_spike = 0.0;
        g_chain_step = 1.5;
    }
    {
        // Verify pre-launch is scheduling only: same emitted ids and the same block trace as the read-then-verify order;
        // every chain / hybrid block is pre-launched (launched rows checked == block rows in verify_collect); copy-only
        // and serial blocks are not.
        std::uint64_t pre_steps = 0;
        for (const std::string cls : {"edit", "prose"})
            for (const int mode : {2, 4}) {
                const auto w = make_world(cls, 40U, 600U, 7 + std::uint64_t(mode));
                const bool copy = mode == 4, hybrid = mode == 4;
                g_clock = 0.0;
                g_prelaunch = false;
                const auto off = run(w, 600U, 0.55, copy, true, hybrid, true);
                g_clock = 0.0;
                g_prelaunch = true;
                const auto on = run(w, 600U, 0.55, copy, true, hybrid, true);
                g_prelaunch = false;
                BD_CHECK(on.out == off.out && on.trace == off.trace, "pre-launch %s mode %d: emitted ids or blocks differ", cls.c_str(), mode);
                BD_CHECK(!on.rounds.empty() && on.rounds == off.rounds,
                         "pre-launch %s mode %d: round records differ from the read-then-verify order", cls.c_str(), mode);
                BD_CHECK(off.prelaunched == 0 && off.launched == 0, "pre-launch off: nothing pre-launched");
                BD_CHECK(on.prelaunched == on.launched && on.prelaunched == on.chain_blocks + on.hybrid && on.prelaunched > 0,
                         "pre-launch %s mode %d: %llu pre-launched, %llu chain + %llu hybrid blocks", cls.c_str(), mode,
                         (unsigned long long)on.prelaunched, (unsigned long long)on.chain_blocks, (unsigned long long)on.hybrid);
                pre_steps += on.prelaunched;
            }
        std::printf("verify pre-launch: same ids, blocks and round records as read-then-verify in 4 of 4 runs, %llu pre-launched steps\n",
                    (unsigned long long)pre_steps);
    }
    tree_tests();
    tree_arbiter_tests();
    std::printf("block drafter tests ok: %d cases, copy blocks %llu, chain blocks %llu, hybrid blocks %llu; edit %.2f vs serial %.2f ms/token\n",
                cases, (unsigned long long)copies, (unsigned long long)chains, (unsigned long long)hybrids, tok_ms_all_loop, tok_ms_serial);
    return 0;
}
