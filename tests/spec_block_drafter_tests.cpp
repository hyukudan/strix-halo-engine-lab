// Block drafting (strix/spec/block_drafter.hpp) against a deterministic fake target and a fake chain drafter with a
// controlled accuracy: whatever the sources / windows / accuracy, the emitted ids equal serial greedy decode and the
// target's committed length equals prompt + emitted - 1 (the anchor is never committed). Costs are a virtual clock.
#include "strix/spec/block_drafter.hpp"

#include <algorithm>
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
    std::printf("block drafter tests ok: %d cases, copy blocks %llu, chain blocks %llu, hybrid blocks %llu; edit %.2f vs serial %.2f ms/token\n",
                cases, (unsigned long long)copies, (unsigned long long)chains, (unsigned long long)hybrids, tok_ms_all_loop, tok_ms_serial);
    return 0;
}
