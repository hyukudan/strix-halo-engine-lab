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
#include <cstdint>
#include <cstdio>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "strix/spec/drafting.hpp"

namespace strix::spec {

// A learned drafter that continues the stream by a chain of tokens (e.g. a native MTP head).
class ChainDrafter {
public:
    virtual ~ChainDrafter() = default;
    // Writes `k` tokens that follow (stream + prefix) into `out`; prefix = tokens a copy source supplies (the
    // chain state is advanced through them). Must not change committed target state; its own speculative state is
    // overwritten by accept().
    virtual void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) = 0;
    // The last verify pass emitted `emitted` (accepted drafts + next token): emitted[i] follows verify row i
    // (row 0 = the anchor). Rows 0..emitted.size()-1 are now committed target positions.
    virtual void accept(std::span<const std::uint32_t> emitted) = 0;
};

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
    std::function<double()> clock_ms; // wall clock in ms (default steady_clock); tests inject a virtual clock
};

struct StepParams {
    std::uint32_t remaining = ~0U;  // tokens left in the budget
    std::uint32_t max_rows = ~0U;   // legal rows this step (context room)
    std::uint64_t context = 0;      // committed positions (cost bin)
};

struct Block {
    std::vector<std::uint32_t> drafts;  // excluding the anchor
    Source source = Source::serial;
    std::uint32_t copy = 0;             // leading drafts from the copy source
    std::uint32_t ext = 0;              // trailing drafts from the chain (hybrid: after the copy span)
    std::uint32_t bucket = 0;           // copy proposal bucket
    Decision decision{};
    double draft_ms = 0.0;
    [[nodiscard]] std::uint32_t rows() const { return std::uint32_t(drafts.size()) + 1; }
    [[nodiscard]] bool hybrid() const { return copy > 0 && ext > 0; }
};

class BlockDrafter {
public:
    explicit BlockDrafter(BlockConfig config = {}, ChainDrafter* chain = nullptr);
    // Emitted stream tokens in order, once each (prompt first; the anchor is part of the stream).
    void commit(std::span<const std::uint32_t> tokens) { proposer_.commit(tokens); }
    // Engine's tabled verify wall at `rows` (shape of the cost curve; measured walls rescale it).
    void seed_verify_ms(std::uint32_t rows, double ms) { arbiter_.seed_cost(rows, ms); }
    // Stated confidence of the chain drafter: conditional survival P(draft d accepted | drafts < d accepted), d = 1 ..,
    // as `weight` pseudo-observations per depth (Arbiter::seed_model_survival). An offline acceptance measurement (a host
    // oracle's) becomes the prior the online counts refine. Prices the chain windows only; emitted ids never change.
    void seed_chain_survival(std::span<const double> conditional, double weight = 8.0) {
        arbiter_.seed_model_survival(conditional, weight);
    }
    [[nodiscard]] Block next(const StepParams& params);
    // After the pass. verify_ms = wall of verify + commit (without drafting). emitted = accepted drafts + next token.
    void observe(const Block& block, std::uint32_t accepted, double verify_ms, std::span<const std::uint32_t> emitted,
                 std::uint64_t context);
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
};

// What BlockSpeculator needs from an engine: R-row verify of [anchor, drafts...] and a prefix commit.
class VerifyTarget {
public:
    virtual ~VerifyTarget() = default;
    // tokens[0] = anchor (last emitted token, not yet in the target). greedy[i] = argmax after row i (size == tokens).
    virtual void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) = 0;
    // Accept the first k rows of the last verify (1 <= k <= rows); the rest is rolled back.
    virtual void commit(std::uint32_t k) = 0;
};

struct StepRecord {
    Source source = Source::serial;
    std::uint32_t rows = 1, proposed = 0, accepted = 0, copy = 0, ext = 0, emitted = 0;
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
    std::vector<std::uint32_t> rows_, greedy_;
};

}  // namespace strix::spec
