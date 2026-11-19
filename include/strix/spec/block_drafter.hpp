#pragma once

// Block drafting on the shared drafting layer: ONE linear block per verify step, built from mixed sources, verified
// exactly (greedy longest prefix) by any target that can run R rows and commit a prefix.
// Home: shared (strix::spec, model-agnostic: emitted ids + measured walls only; no engine, GGUF or device types).
//
// Sources of the block (the Arbiter of drafting.hpp picks copy vs chain and the window from measured costs; the
// hybrid extension is priced by the same expected-tokens-per-ms rule):
//   copy    ContextProposer longest-suffix / cursor proposals over prompt + emitted text, up to max_block - 1 tokens
//   chain   a ChainDrafter (native MTP: k steps, each fed the previous step's own hidden), depth <= chain_max_k
//   hybrid  copy span, then chain extension: the chain consumes the copy tokens (prefix) to keep its state, then
//           drafts `ext` more tokens. Used only when the copy proposal ended below the window and the extension's
//           expected gain per ms beats the copy-only block.
// Exactness: drafts are arbitrary ids. BlockSpeculator verifies [anchor, drafts...] in one target pass, accepts the
// longest prefix equal to the target's greedy rows, emits accepted drafts + the target's next token (the bonus or the
// correction) and commits accepted + 1 rows. Emitted ids equal serial greedy decode for ANY block.
//
// VERIFY PRE-LAUNCH (scheduling only; same kernels, operands and stream order, so the bits are unchanged): a chain that
// keeps its drafted ids on the device (engine feed slots) lets the target ENQUEUE the verify of [anchor, drafts] right
// behind the draft work, before the ids are read back. The host launch of the verify (a CUDA graph of ~870 nodes takes
// ~0.5 ms of host time) then runs while the GPU drafts instead of with the GPU idle. Both sides are optional
// (ChainDrafter::draft_prelaunch, VerifyTarget::verify_launch_fed); either one declining falls back to the read-then-verify
// order. Off unless BlockSpeculator::set_prelaunch(true).
// Token tree (sibling rows): when the drafter reports its runner-up per chain position (ChainDrafter::runner_ups) and the
// target verifies tree rows (VerifyTarget::tree_rows() > 0), the runner-up at main depth j may be added as a LEAF row
// beside the chain node of depth j (same parent). It pays exactly +1 token when the target's token at depth j is the
// runner-up (the linear block emits the same token as its correction and stops; the sibling row also yields the target's
// next token). Placement is a per-step decision on a SET of siblings (the best-gain prefix): the set is placed only when
// it raises the block's tokens per ms, priced by MEASURED walls: P = T(rows with siblings) - L(chain rows), L = the
// measured linear verify wall per window rows (else the Arbiter's price), T = the measured tree-window wall per rows (else
// L + the learned tree surcharge). A tree window carries a per-window surcharge and the row cost is step shaped (dense row
// tiles), so neither a per-row marginal nor a linear seed can price it (rv_tree3: tree +0.5..0.7 ms at equal rows, 8 -> 9
// rows +1.2 ms on CUDA dense 27B). Where rows are expensive (MoE expert union) no sibling is placed. q_j is learned from
// EVERY rejection that had a runner-up, placed or not, per depth and per drafter logit-gap bucket (top-1 minus top-2
// logit = log p1/p2 of the drafter, ChainDrafter::runner_up_gaps): q = q_depth x lift(gap bucket).
// Row layout: depth order; at each depth the leaf sibling precedes the chain node, so every row's parent is the last
// non-leaf row before it (a recurrent target evaluates rows in order and does not keep a leaf row's state). The accepted
// path (anchor + main prefix [+ sibling]) is committed by VerifyTarget::commit_path; a block without siblings uses
// verify / commit(k) exactly as before.
#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "strix/spec/drafting.hpp"

namespace strix::spec {

// Where the k drafts of one chain call live before they are read back (verify pre-launch). Draft j is in device feed
// slot slot[j], or, when slot[j] == kHost, already on the host as host_id[j].
struct DraftFeed {
    static constexpr std::uint32_t kHost = 0xFFFFFFFFU;
    static constexpr std::uint32_t kMax = 32;  // = the widest verify window
    std::uint32_t k = 0;
    std::array<std::uint32_t, kMax> slot{};
    std::array<std::uint32_t, kMax> host_id{};
};
// prefix = the host tokens the chain consumed before its drafts (the hybrid copy span; empty for a chain-only block).
using PrelaunchFn = std::function<void(std::span<const std::uint32_t> prefix, const DraftFeed& feed)>;

// A learned drafter that continues the stream by a chain of tokens (e.g. a native MTP head).
class ChainDrafter {
public:
    virtual ~ChainDrafter() = default;
    // Writes `k` tokens that follow (stream + prefix) into `out`; prefix = tokens a copy source supplies (the
    // chain state is advanced through them). Must not change committed target state; its own speculative state is
    // overwritten by accept().
    virtual void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) = 0;
    // Optional verify pre-launch: draft() with the same work and the same `out`, but after the draft work is enqueued and
    // BEFORE the ids are read back it calls launch(prefix, feed) exactly once (feed.k == k). Returns false, having done
    // nothing, when this draft cannot stay on the device; the caller then uses draft().
    virtual bool draft_prelaunch(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out,
                                 const PrelaunchFn& launch) {
        (void)prefix;
        (void)k;
        (void)out;
        (void)launch;
        return false;
    }
    // The last verify pass emitted `emitted` (accepted drafts + next token): emitted[i] follows verify row i
    // (row 0 = the anchor). Rows 0..emitted.size()-1 are now committed target positions.
    virtual void accept(std::span<const std::uint32_t> emitted) = 0;
    // Optional: runner-up id (second most likely) of each position written by the last draft(), aligned with its `out`;
    // kNoToken where none. Empty span: the drafter has no runner-ups (no siblings, no probe).
    [[nodiscard]] virtual std::span<const std::uint32_t> runner_ups() const { return {}; }
    // Optional, aligned with runner_ups(): the drafter's top-1 minus top-2 logit at each position (log p1 / p2; NaN or
    // negative where unknown). Empty span: no margins (sibling q uses the depth estimate only).
    [[nodiscard]] virtual std::span<const float> runner_up_gaps() const { return {}; }
    // Tree verify: emitted[i] follows verify row rows[i] (rows[0] = 0, the anchor; a root path of the last tree). A
    // drafter that reads the target's verify rows (hidden states) must override; the default ignores the rows.
    virtual void accept_path(std::span<const std::uint32_t> emitted, std::span<const std::uint32_t> rows) {
        (void)rows;
        accept(emitted);
    }
};

inline constexpr std::uint32_t kNoToken = 0xFFFFFFFFU;

struct BlockConfig {
    ArbiterConfig arbiter = [] {
        ArbiterConfig c;
        c.windows.clear();
        for (std::uint32_t w = 1; w <= 32; ++w) c.windows.push_back(w);  // every row count: no padding rows
        return c;
    }();
    ProposerConfig proposer{};
    std::uint32_t max_block = 32;     // rows of the longest window (anchor + drafts)
    std::uint32_t chain_max_k = 7;    // deepest chain draft (also the hybrid extension)
    bool copy = true, chain = true, hybrid = true;
    double hybrid_switch_in = 1.05;   // the extension must beat the copy-only block by this factor
    double chain_seed_step_ms = 1.5;  // chain wall seed per chain step until measured (engine: MTP layer + lm_head)
    double chain_seed_flush_ms = 1.5; // first step after a verify (catch-up of the accepted rows)
    // true: the seeds ARE the cost model. The chain costs flush + step x (k - 1) and is never learned from walls, and the
    // Arbiter prices verify windows from seed_verify_ms (ArbiterConfig::learn_costs = false) plus the context cost: with it
    // off or fixed (spec::set_context_cost "off" / KV) every decision is a pure function of the seeds and the acceptance
    // history (deterministic for a fixed prompt); the default online fit adds a clipped 3-term fit of the verify walls
    // (ArbiterConfig::cost_fit), never per-window walls. Learned walls froze a depth
    // at its first sample (the subset head's one-off materialisation, ~10 ms, landed in one depth's mean and that depth was
    // never chosen again): measured on CUDA dense 27B, three identical Japanese runs gave 80 / 115 / 99 tok/s and k = 15 /
    // k = 1 / k = 1 where k = 3 was the optimum. false = the previous learned form.
    bool fixed_costs = false;
    // Token-tree siblings (see the header comment): placed only with a tree target and a drafter with runner-ups, and
    // only where the cost rule pays. sibling_prior = Beta prior (hits, misses) of q_j before measurements (mean 0.25,
    // inside the 15-55 % measured on HIP Qwen MoE MTP, decision-register X-014).
    bool siblings = true;
    std::array<double, 2> sibling_prior = {1.0, 3.0};
    // Pseudo-trials of the gap-bucket lift toward 1 (a bucket with few trials keeps the depth estimate).
    double sibling_gap_kappa = 4.0;
    std::function<double()> clock_ms; // wall clock in ms (default steady_clock); tests inject a virtual clock
};

struct StepParams {
    std::uint32_t remaining = ~0U;  // tokens left in the budget
    std::uint32_t max_rows = ~0U;   // legal rows this step (context room)
    std::uint64_t context = 0;      // committed positions (cost bin)
    std::uint32_t tree_rows = 0;    // widest tree window of the target (0: linear only; BlockSpeculator sets it)
};

// Rows of a block as the target sees them: tokens[0] = anchor; parent[r] = parent row (-1 for the anchor, < r);
// depth[r] = stream offset of row r (position committed + depth[r]); node[r] = main depth j (1..) of a chain row, or
// -(j) for the sibling at depth j, 0 for the anchor.
struct TreeRows {
    std::vector<std::uint32_t> tokens;
    std::vector<std::int32_t> parent, node;
    std::vector<std::uint32_t> depth;
};

struct Block {
    std::vector<std::uint32_t> drafts;  // excluding the anchor
    Source source = Source::serial;
    std::uint32_t copy = 0;             // leading drafts from the copy source
    std::uint32_t ext = 0;              // trailing drafts from the chain (hybrid: after the copy span)
    std::uint32_t bucket = 0;           // copy proposal bucket
    Decision decision{};
    double draft_ms = 0.0;
    // Token tree: alts[j] = the drafter's runner-up at main depth j + 1 (kNoToken: none; size == drafts.size() or 0);
    // sib[j] != 0: that runner-up is verified as a leaf row (sibling of drafts[j]).
    std::vector<std::uint32_t> alts;
    std::vector<float> alt_gap;  // aligned with alts when the drafter reports margins (else empty)
    std::vector<std::uint8_t> sib;
    std::uint32_t siblings = 0;  // sibling rows placed
    double sib_gain = 0.0;       // predicted extra tokens of the placed siblings (before calibration)
    [[nodiscard]] std::uint32_t rows() const { return std::uint32_t(drafts.size()) + 1 + siblings; }
    [[nodiscard]] bool hybrid() const { return copy > 0 && ext > 0; }
    // The verify rows (anchor first) in the layout described in the header comment.
    void tree_rows(std::uint32_t anchor, TreeRows& out) const;
};

// Drafter logit-gap buckets of the sibling q (edges in logits: gap = log p1 / p2). -1: no gap (NaN / negative).
inline constexpr std::array<float, 5> kSiblingGapEdges = {0.5f, 1.0f, 2.0f, 3.0f, 5.0f};
inline constexpr std::size_t kSiblingGapBuckets = kSiblingGapEdges.size() + 1;
static_assert(kSiblingGapBuckets == std::tuple_size_v<decltype(SpecStats::sibling_gap_trials)>,
              "SpecStats gap arrays follow the bucket edges");
[[nodiscard]] inline int sibling_gap_bucket(float gap) {
    if (!(gap >= 0.0f)) return -1;
    int b = 0;
    for (const float e : kSiblingGapEdges) b += gap >= e ? 1 : 0;
    return b;
}

// Walk of a verified tree: from the anchor, follow the child row whose token equals the target's greedy at the current
// row (chain node first; a sibling is a leaf). path = rows taken (path[0] = 0); emitted = their tokens after the anchor
// + the target's token after the last row; main = chain drafts accepted; sibling = the path ends on a sibling row.
struct TreeAccept {
    std::vector<std::uint32_t> path, emitted;
    std::uint32_t main = 0;
    bool sibling = false;
};
TreeAccept accept_tree(const TreeRows& rows, std::span<const std::uint32_t> greedy);

class BlockDrafter {
public:
    explicit BlockDrafter(BlockConfig config = {}, ChainDrafter* chain = nullptr);
    // Emitted stream tokens in order, once each (prompt first; the anchor is part of the stream).
    void commit(std::span<const std::uint32_t> tokens) { proposer_.commit(tokens); }
    // Engine's tabled verify wall at `rows` (shape of the cost curve; measured walls rescale it).
    void seed_verify_ms(std::uint32_t rows, double ms) { arbiter_.seed_cost(rows, ms); }
    // launch (optional): handed to ChainDrafter::draft_prelaunch for every chain call of this block.
    [[nodiscard]] Block next(const StepParams& params, const PrelaunchFn* launch = nullptr);
    // After the pass. verify_ms = wall of verify + commit (without drafting). emitted = accepted drafts + next token.
    // path: verify rows of a tree's accepted path (empty: rows 0..emitted.size()-1 of a linear block).
    void observe(const Block& block, std::uint32_t accepted, double verify_ms, std::span<const std::uint32_t> emitted,
                 std::uint64_t context, std::span<const std::uint32_t> path = {});
    [[nodiscard]] const Arbiter& arbiter() const { return arbiter_; }
    [[nodiscard]] const SpecStats& stats() const { return stats_; }
    [[nodiscard]] double now_ms() const;
    [[nodiscard]] double chain_cost_ms(std::uint32_t steps) const { return chain_ms(steps); }
    // Extension bookkeeping (hybrid) for reports.
    std::uint64_t hybrid_blocks = 0, hybrid_ext_proposed = 0, hybrid_ext_accepted = 0;

private:
    [[nodiscard]] double chain_ms(std::uint32_t steps) const;  // chain wall for `steps` chain rows (flush counted in step 1)
    void learn_chain_ms(std::uint32_t steps, double ms);
    BlockConfig config_;
    ChainDrafter* chain_;
    ContextProposer proposer_;
    Arbiter arbiter_;
    Proposal proposal_;
    SpecStats stats_;
    std::vector<double> chain_ms_, chain_w_;  // [steps] measured EWMA
    bool step_learned_ = false;
    double step_ms_;                          // marginal ms per extra chain step
    std::vector<double> model_ms_;            // scratch: Arbiter model_draft_ms table
    std::array<std::array<double, 2>, 16> ext_{};  // hybrid extension (accept, reject) per depth (chain_max_k <= 16)
    std::array<std::array<double, 2>, 16> sib_q_{};  // runner-up at a rejection: (hit, miss) per main depth
    std::array<std::array<double, 2>, kSiblingGapBuckets> gap_q_{};  // the same, per drafter logit-gap bucket
    // Measured verify + commit walls per window rows, linear and tree windows apart (EWMA, at most 16 samples of memory;
    // the first window of each shape is not learned: it carries the one-off graph capture). sur_ = mean tree surcharge
    // (tree wall - linear price at the same rows), the price of a tree shape not yet measured.
    struct WallBin {
        double ms = 0.0;
        std::uint32_t seen = 0;
    };
    std::array<WallBin, 33> lin_wall_{}, tree_wall_{};
    WallBin sur_{};
    // Realised vs predicted sibling tokens over tree windows: the predicted gain is scaled by (taken + 4) / (predicted + 4).
    double gain_pred_ = 0.0, gain_real_ = 0.0;
    // The run's measured rate: emitted tokens and verify + draft ms, decayed (x 0.98 per pass).
    double run_tok_ = 0.0, run_ms_ = 0.0;
    static void learn_wall(WallBin& b, double ms, bool skip_first);
    [[nodiscard]] double lin_price(std::uint32_t rows, std::uint64_t context) const;
    [[nodiscard]] double tree_price(std::uint32_t rows, std::uint64_t context) const;
    void take_runner_ups(Block& b, std::uint32_t first, std::uint32_t n) const;
    void place_siblings(Block& b, const StepParams& params) const;
    [[nodiscard]] double sibling_q(std::uint32_t depth, float gap = -1.0f) const;

public:
    // Learned tree surcharge (ms; 0 until a tree shape has been measured twice) and whether it is measured.
    [[nodiscard]] double tree_surcharge_ms() const { return sur_.ms; }
    [[nodiscard]] std::uint32_t tree_surcharge_samples() const { return sur_.seen; }
};

// What BlockSpeculator needs from an engine: R-row verify of [anchor, drafts...] and a prefix commit.
class VerifyTarget {
public:
    virtual ~VerifyTarget() = default;
    // tokens[0] = anchor (last emitted token, not yet in the target). greedy[i] = argmax after row i (size == tokens).
    virtual void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) = 0;
    // Accept the first k rows of the last verify (1 <= k <= rows); the rest is rolled back.
    virtual void commit(std::uint32_t k) = 0;
    // Optional verify pre-launch: ENQUEUE (no host read, no wait) the verify of host_rows followed by the feed.k drafts of
    // `feed`. Returns false, having enqueued nothing, when unsupported. After true, verify_collect(tokens) with tokens ==
    // host_rows + the drafts completes it with the same result as verify(tokens).
    virtual bool verify_launch_fed(std::span<const std::uint32_t> host_rows, const DraftFeed& feed) {
        (void)host_rows;
        (void)feed;
        return false;
    }
    virtual void verify_collect(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) {
        (void)tokens;
        (void)greedy;
        throw std::logic_error("VerifyTarget::verify_collect without a pre-launched verify");
    }
    // Token-tree rows (optional). 0 = linear windows only (verify_tree / commit_path are never called).
    [[nodiscard]] virtual std::uint32_t tree_rows() const { return 0; }
    // rows.tokens[r] at stream position committed + rows.depth[r], attending the committed prefix and its ancestor rows
    // (rows.parent); greedy[r] = argmax after row r. Must give each row the bits a linear window of its path gives.
    virtual void verify_tree(const TreeRows& rows, std::vector<std::uint32_t>& greedy) {
        (void)rows;
        (void)greedy;
        throw std::logic_error("verify target: no tree rows (tree_rows() == 0)");
    }
    // Accept a root path of the last verify_tree (path[0] = 0, each row the parent of the next) as positions
    // committed .. committed + path.size() - 1; every other row is rolled back. Default: a contiguous row prefix only.
    virtual void commit_path(std::span<const std::uint32_t> path) {
        for (std::size_t i = 0; i < path.size(); ++i)
            if (path[i] != i) throw std::logic_error("verify target: no tree rows (tree_rows() == 0)");
        commit(static_cast<std::uint32_t>(path.size()));
    }
};

struct StepRecord {
    Source source = Source::serial;
    std::uint32_t rows = 1, proposed = 0, accepted = 0, copy = 0, ext = 0, emitted = 0;
    std::uint32_t siblings = 0;  // sibling rows verified
    bool sibling_taken = false;  // the accepted path ended on a sibling (+1 token)
    double verify_ms = 0.0, draft_ms = 0.0;
};

// One verify round of BlockSpeculator (the rollout record): what was drafted, the target's greedy id at EVERY verified row
// (the rejected rows included) and what was emitted (= drafts[0, accepted) + greedy[accepted]). The spans are valid only
// during RoundSink::round. Recording never changes a decision: the sink sees the round after commit.
struct RoundRecord {
    std::uint64_t context = 0;  // StepParams::context of the round (committed positions, cost bin)
    Source source = Source::serial;
    std::uint32_t anchor = 0;   // row 0 of the verify
    std::uint32_t k = 0, rows = 0, accepted = 0, copy = 0, ext = 0;  // rows = k + 1
    double verify_ms = 0.0, draft_ms = 0.0;
    std::span<const std::uint32_t> drafts, greedy, emitted;  // k, rows, accepted + 1 ids
};

class RoundSink {
public:
    virtual ~RoundSink() = default;
    virtual void round(const RoundRecord& record) = 0;  // once per verify, after the target's commit
};

// Binary trace of rounds (little endian; no host names or paths inside):
//   file   : "SRT1", u32 version (1), u32 label bytes, label
//   record : u32 "RREC", u64 context, u32 anchor, k, rows, accepted, copy, ext, u8 source, u8[3] 0, f32 verify_ms,
//            f32 draft_ms (48 bytes), then k u32 drafts, rows u32 greedy (emitted is derived)
// Reader: scripts/bench/spec/recycle_score.py.
class RoundTraceWriter final : public RoundSink {
public:
    RoundTraceWriter(const std::string& path, std::string_view label);  // throws std::runtime_error if it cannot open
    ~RoundTraceWriter() override;
    RoundTraceWriter(const RoundTraceWriter&) = delete;
    RoundTraceWriter& operator=(const RoundTraceWriter&) = delete;
    void round(const RoundRecord& record) override;
    [[nodiscard]] std::uint64_t records() const { return records_; }

private:
    std::FILE* file_ = nullptr;
    std::uint64_t records_ = 0;
    std::vector<unsigned char> buf_;
};

class BlockSpeculator {
public:
    BlockSpeculator(BlockDrafter& drafter, VerifyTarget& target) : drafter_(drafter), target_(target) {}
    // anchor = last emitted token (already committed to the drafter's stream, NOT in the target).
    void start(std::uint32_t anchor) { anchor_ = anchor; }
    // Verify pre-launch (see the top of this file). Bits unchanged; off by default until its timing receipt.
    void set_prelaunch(bool on) { prelaunch_ = on; }
    [[nodiscard]] bool prelaunch() const { return prelaunch_; }
    [[nodiscard]] std::uint64_t prelaunched_steps() const { return prelaunched_; }
    // One step; appends the emitted ids to `out`. Returns the record (emitted == 0: nothing to do).
    StepRecord step(std::vector<std::uint32_t>& out, const StepParams& params);
    [[nodiscard]] std::uint32_t anchor() const { return anchor_; }
    // Optional round trace (nullptr = off, the default: one null check per step). Not owned.
    void set_round_sink(RoundSink* sink) { sink_ = sink; }

private:
    BlockDrafter& drafter_;
    VerifyTarget& target_;
    RoundSink* sink_ = nullptr;
    std::uint32_t anchor_ = 0;
    std::vector<std::uint32_t> rows_, greedy_, host_rows_;
    bool prelaunch_ = false;
    std::uint64_t prelaunched_ = 0;
    TreeRows tree_;
};

}  // namespace strix::spec
