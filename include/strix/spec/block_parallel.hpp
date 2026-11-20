#pragma once

// Parallel block drafter contract (shared): ONE drafter forward proposes a whole block of tokens. Every engine whose
// drafter fills a block of rows after the anchor with a fixed token and reads every proposal from one pass binds this
// contract: HY4 DFly2 (KV share), DS41 DSpark and Gemma4/Nemotron DFlash (own context rings). The model forward is
// the engine's (BlockForward); the policy, the block layout, the row visibility, the controller binding and the
// verify pricing live here.
// Home: shared (strix::spec, host/vendor-free; no engine, GGUF or device types).
//
// Block layout (rows r = 0 .. block-1 at positions anchor_pos + r):
//   row 0            the anchor (last emitted token, not yet in the target)
//   rows 1 .. n      a known prefix (copy-source tokens; only when prefix_in_block)
//   rows n+1 ..      fill_token (mask / noise id)
//   Proposal j (the token at anchor_pos + 1 + j) is read from row j + 1 (anchor_proposes = false: the row holding the
//   fill token at that position, DFlash / DFly2) or from row j (anchor_proposes = true: each row predicts its successor,
//   DSpark). A prefix of n tokens fixes proposals 0 .. n-1; the forward returns the proposals after it.
// Context keys of the draft attention: own_ring = the drafter's own rings (built from target hidden taps, kept by the
// engine); target_share = the target's cached latent rows of target_layers[i] at the positions the target indexer
// selected for that layer (no drafter KV). Block rows see each other by `mask`.
// Cost: one forward for any k <= max_drafts -> the cost is FLAT in k (BlockConfig flush = pass, step = 0).
// Exactness: drafts are arbitrary ids; BlockSpeculator's greedy verify keeps emitted == serial greedy decode.
#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "strix/spec/block_drafter.hpp"

namespace strix::spec {

enum class KvSource : std::uint8_t { own_ring, target_share };
enum class BlockMask : std::uint8_t { bidirectional, causal };
enum class Fusion : std::uint8_t { none, slot0, every_row };  // where the target hidden enters the block rows

inline constexpr std::size_t max_draft_layers = 8;

struct BlockPolicy {
    std::uint32_t block = 0;          // rows including the anchor
    std::uint32_t fill_token = 0;     // id of the rows after the anchor and the prefix
    KvSource kv = KvSource::own_ring;
    std::uint32_t ring = 0;           // own_ring: context entries per draft layer (0 with target_share)
    std::uint32_t share_budget = 0;   // target_share: max selected context positions per draft layer
    BlockMask mask = BlockMask::bidirectional;
    Fusion fusion = Fusion::none;
    bool correction = false;          // hidden-correction hook (DFly2 hidden_correction)
    bool anchor_proposes = false;
    bool prefix_in_block = false;     // known tokens may replace the leading fill rows (hybrid extension)
    std::uint32_t draft_layers = 0;
    std::uint32_t target_layer_count = 0;
    // target_share: KV + index source layer of draft layer i; own_ring: the target layers whose hidden is tapped.
    std::array<std::uint16_t, max_draft_layers> target_layers{};
};

// Throws std::invalid_argument when the policy is not usable.
void validate(const BlockPolicy& p);
// Proposals one forward returns after a prefix of n known tokens.
[[nodiscard]] std::uint32_t max_drafts(const BlockPolicy& p, std::uint32_t prefix = 0);
// Rows the forward runs for a prefix n and k proposals (the trailing fill rows nobody reads are not run).
[[nodiscard]] std::uint32_t rows_needed(const BlockPolicy& p, std::uint32_t prefix, std::uint32_t k);
// Block row whose output is proposal j.
[[nodiscard]] std::uint32_t proposal_row(const BlockPolicy& p, std::uint32_t j);
// Block input ids: [anchor, prefix..., fill...] of length rows_needed(p, prefix.size(), k).
void block_ids(const BlockPolicy& p, std::uint32_t anchor, std::span<const std::uint32_t> prefix, std::uint32_t k,
               std::vector<std::uint32_t>& out);
// Does block row q attend block row r (context keys are always visible).
[[nodiscard]] bool row_visible(const BlockPolicy& p, std::uint32_t q, std::uint32_t r);

// Presets (policy as data). Sources: hy4/web/config.json + README (DFly2); include/strix/ds41_dspark.hpp:3-23, :66
// (DSpark); src/gemma4/dflash/dflash.hpp:3-20 (DFlash). Binding owners re-check their row rule on their fixture.
[[nodiscard]] BlockPolicy hy4_dfly2();
[[nodiscard]] BlockPolicy ds41_dspark();
[[nodiscard]] BlockPolicy g4_dflash();

// The engine's drafter forward.
struct BlockRequest {
    std::span<const std::uint32_t> ids;  // block_ids(), rows_needed() rows
    std::uint64_t anchor_pos = 0;        // target position of row 0 (= committed target rows)
    std::uint32_t prefix = 0;            // known tokens after the anchor
    std::uint32_t k = 0;                 // proposals wanted after the prefix
};
class BlockForward {
public:
    virtual ~BlockForward() = default;
    // One drafter pass over the block. Writes exactly req.k proposals (the tokens at anchor_pos + 1 + prefix + j) and,
    // when the drafter has a confidence head, one confidence per proposal (else leaves it empty). Must not change
    // committed target state.
    virtual void forward(const BlockRequest& req, std::vector<std::uint32_t>& proposals, std::vector<float>& confidence) = 0;
    // The last verify committed `emitted` (accepted drafts + next token; row emitted.size()-1 of that verify produced the
    // new anchor). target_share drafters pick the hidden of that row; own_ring drafters append those rows to the rings.
    virtual void accept(std::span<const std::uint32_t> emitted) = 0;
};

// The controller binding: a parallel block drafter IS a ChainDrafter whose k steps cost one pass.
class ParallelBlockDrafter final : public ChainDrafter {
public:
    ParallelBlockDrafter(const BlockPolicy& policy, BlockForward& forward);
    // anchor = last emitted token at target position anchor_pos (prompt last token after prefill).
    void start(std::uint32_t anchor, std::uint64_t anchor_pos);
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override;
    void accept(std::span<const std::uint32_t> emitted) override;
    [[nodiscard]] const BlockPolicy& policy() const { return policy_; }
    [[nodiscard]] std::uint64_t passes() const { return passes_; }
    [[nodiscard]] std::uint64_t anchor_pos() const { return anchor_pos_; }
    [[nodiscard]] std::span<const float> last_confidence() const { return confidence_; }

    // The drafter's confidence as a per-row prior (opt-in; unbound = the previous behaviour bit for bit). After the block
    // pass (its cost is flat in k, so it is already paid) a model block without a prefix keeps the k' <= k proposals that
    // maximise (1 + sum_j prod_{i<=j} p_i) / (verify_ms(k' + 1) + draft_ms), p_i = calibrated acceptance of row i given
    // its prefix accepted, learned per confidence bin from the verified rows. A block whose rows' bins lack evidence is
    // not trimmed. k' = 0 (a plain step) keeps one row every explore_every times so the first-row bins keep learning.
    // Only the rows verified change; emitted tokens never do.
    struct RowPriorConfig {
        std::uint32_t bins = 8;           // confidence bins over [0, 1]
        double min_evidence = 8.0;        // decayed labels a bin needs before it prices a row
        double half_life = 256.0;         // labels
        double draft_ms = 0.0;            // the block pass (part of the pass the rate divides by)
        std::uint32_t explore_every = 16;
    };
    void bind_row_prior(std::function<double(std::uint32_t rows)> verify_ms, RowPriorConfig config);
    [[nodiscard]] double row_prior(float confidence) const;  // calibrated acceptance; < 0: no evidence (or unbound)
    [[nodiscard]] std::uint64_t trimmed() const { return trimmed_; }

private:
    [[nodiscard]] std::uint32_t trim(std::uint32_t k);
    void label(std::span<const std::uint32_t> emitted);
    [[nodiscard]] std::size_t bin_of(float confidence) const;
    BlockPolicy policy_;
    BlockForward& forward_;
    std::function<double(std::uint32_t)> row_verify_ms_;
    RowPriorConfig row_cfg_{};
    std::vector<std::array<double, 2>> bins_;  // decayed (accepted, rejected) per confidence bin
    std::vector<std::uint32_t> pending_;       // verified proposals of the last model block (labelled on accept)
    std::vector<float> pending_conf_;
    std::uint64_t trimmed_ = 0, zero_picks_ = 0;
    std::uint32_t anchor_ = 0;
    std::uint64_t anchor_pos_ = 0;
    std::uint64_t passes_ = 0;
    std::vector<std::uint32_t> ids_, proposals_;
    std::vector<float> confidence_;
};

// BlockConfig for a parallel drafter: chain depth = max_drafts, flat cost (flush = pass_ms, step = 0), hybrid only
// when the policy accepts a prefix in the block.
[[nodiscard]] BlockConfig parallel_block_config(const BlockPolicy& p, double pass_ms, BlockConfig base = {});

// Verify pricing for routed-MoE targets (bytes per verify pass, any backend; ms = bytes / effective bandwidth + the
// backend's fixed per-pass time). Prior only: a measured window histogram (moe_plan.hpp mean_distinct_experts)
// replaces uniform_distinct_experts when a routing trace exists.
struct MoeVerifyBytes {
    double fixed_gb = 0.0;          // weights every pass reads once (attention, shared expert, head, router, dense)
    double per_row_gb = 0.0;        // per verify row (KV / selected keys read, activations)
    double expert_gb = 0.0;         // one routed expert of one layer
    std::uint32_t experts = 0, top_k = 0, moe_layers = 0;
};
// Expected distinct experts per layer for `rows` rows, each choosing top_k of experts uniformly and independently:
// E (1 - (1 - k/E)^rows). Upper prior: correlated routing of adjacent tokens touches fewer.
[[nodiscard]] double uniform_distinct_experts(std::uint32_t experts, std::uint32_t top_k, std::uint32_t rows);
[[nodiscard]] double verify_gb(const MoeVerifyBytes& m, std::uint32_t rows);
// Mean emitted tokens per step (accepted + 1) at which a step of (verify + draft) ms equals plain decode.
[[nodiscard]] double break_even_emitted(double plain_ms, double verify_ms, double draft_ms);
// Mean emitted tokens per step for per-depth acceptance alpha and `rows` verify rows: sum_{i<rows} alpha^i.
[[nodiscard]] double expected_emitted(double alpha, std::uint32_t rows);

// Verify seed table of the Arbiter (BlockDrafter::seed_verify_ms / Arbiter::seed_cost) from the bytes of a routed-MoE
// verify pass: ms[R - 1] = fixed + verify_gb(m, R) / gbps for R = 1 .. max_rows, where fixed = plain_ms - verify_gb(m, 1)
// / gbps is the measured non-streaming part of a plain step (launch boundaries, host). A sparse MoE (top_k << experts)
// pays nearly one new expert set per extra row, so its curve is steep where a dense target's is flat; seeding it flat
// makes the seeded-cost Arbiter (fixed_costs) over-draft on low-acceptance text. Scheduling only: tokens never change.
// gbps in GB/s, ms in ms. Throws std::invalid_argument when plain_ms or gbps is not > 0, max_rows is 0 or the bytes of
// one row exceed the plain wall at gbps.
[[nodiscard]] std::vector<double> moe_verify_seed_ms(const MoeVerifyBytes& m, double plain_ms, double gbps,
                                                    std::uint32_t max_rows);
// HY4-flash target, one node (pack geometry + decode kernel trace): non-expert bytes read once per pass (Q8_0 Linears,
// absorbed K_b / V_b, BF16 head, F32 router), selected K/V per row, one routed expert of one layer (Q4_K gate + up
// 1536 x 1536 each, Q6_K down 1536 x 1536 = 4.59 MB), 512 experts top 16 on 41 MoE layers.
[[nodiscard]] MoeVerifyBytes hy4_verify_bytes();

// Verify cost of a routed-MoE target in the terms a pass actually spends (any backend, any topology):
//   ms(R) = fixed_ms + trips x trip_ms + per_row_ms x R + per_expert_ms x reads(R)
// reads(R) = expert-layer reads of the pass: distinct experts of the R rows summed over the MoE layers when the kernel
// groups the rows' experts (grouped), top_k x layers x R when it reads every row's experts. Exchanges are paid once per
// pass. Scheduling only: tokens never change.
struct MoeVerifyCost {
    double fixed_ms = 0.0;       // per pass: weights read once + launch boundaries + host
    double per_expert_ms = 0.0;  // one routed expert of one layer
    double per_row_ms = 0.0;     // per verify row (selected K/V, activations)
    std::uint32_t experts = 0, top_k = 0, layers = 0;
    double trip_ms = 0.0;        // one rank exchange
    std::uint32_t trips = 0;     // exchanges per pass (0 on one node)
    bool grouped = true;
};
// Seed from bytes: every rank reads 1 / ranks of every weight (TP), plain_ms is the R = 1 wall of that topology and fixes
// fixed_ms. One node, grouped: seed_ms(c, R) equals moe_verify_seed_ms(m, plain_ms, gbps, R)[R - 1].
// Throws std::invalid_argument on a bad geometry or when the bytes of one row exceed the wall.
[[nodiscard]] MoeVerifyCost moe_verify_cost(const MoeVerifyBytes& m, double plain_ms, double gbps, std::uint32_t ranks = 1,
                                            double trip_ms = 0.0, std::uint32_t trips = 0, bool grouped = true);
// Prior reads of R rows: grouped = layers x uniform_distinct_experts (an upper bound: adjacent tokens share experts),
// ungrouped = layers x top_k x R.
[[nodiscard]] double prior_reads(const MoeVerifyCost& c, std::uint32_t rows);
[[nodiscard]] double verify_ms(const MoeVerifyCost& c, std::uint32_t rows, double reads);
[[nodiscard]] inline double seed_ms(const MoeVerifyCost& c, std::uint32_t rows) {
    return verify_ms(c, rows, prior_reads(c, rows));
}

// The cost learned online from the walls: a ridge fit of (fixed + trips x trip, per_row, per_expert) toward the seed with
// forgetting, on the reads the engine counted in each pass. A direction the walls do not identify (an ungrouped kernel
// has reads proportional to R; a single window) keeps the seed's split while the price at the observed rows follows the
// walls, so an ill-conditioned fit never invents a slope (kvcost2: collinear online fits). A wall off the prediction by
// more than `clip` is skipped once and clipped when it repeats (one stall cannot move the price). reads(R) is learned per
// content as a ratio to the prior (route overlap of adjacent rows) and extends to an unseen R from the nearest seen R.
// Bind: BlockConfig::fixed_costs = true and BlockDrafter::seed_verify_ms(R, table[R - 1]) after every observe (the
// Arbiter's context term stays on top). Deterministic: a fixed sequence of observations gives bitwise the same prices.
struct MoeFitConfig {
    double half_life = 64.0;        // passes: an older wall weighs half after this many
    double ridge = 4.0;             // pass-equivalents pulling the per-row and per-expert slopes toward the seed
    double ridge_fixed = 0.01;      // the same for the per-pass term: the first wall sets the scale, the seed the shape
    double clip = 0.25;             // relative residual beyond which a wall is skipped once / clipped on repeat
    double reads_half_life = 32.0;  // passes: route-overlap ratio per R
};
class MoeVerifyFit {
public:
    explicit MoeVerifyFit(const MoeVerifyCost& seed, MoeFitConfig config = {});
    void observe(std::uint32_t rows, double reads, double ms);
    void observe(std::uint32_t rows, double ms) { observe(rows, reads(rows), ms); }  // engine without a read counter
    [[nodiscard]] double reads(std::uint32_t rows) const;      // learned, else the prior
    [[nodiscard]] double verify_ms(std::uint32_t rows) const;  // fitted at reads(rows), non-decreasing in rows
    [[nodiscard]] std::vector<double> table(std::uint32_t rows) const;  // [R - 1] = verify_ms(R), R = 1 .. rows
    [[nodiscard]] const MoeVerifyCost& seed() const { return seed_; }
    [[nodiscard]] MoeVerifyCost current() const;  // seed geometry and trips with the fitted coefficients
    [[nodiscard]] std::uint64_t passes() const { return passes_; }
    [[nodiscard]] std::uint64_t skipped() const { return skipped_; }

    static constexpr std::uint32_t max_rows = 32;

private:
    [[nodiscard]] double raw_ms(std::uint32_t rows, double reads) const;
    void solve();
    MoeVerifyCost seed_;
    MoeFitConfig config_;
    double unit_reads_ = 1.0;                    // reads of one row (top_k x layers): feature scale
    std::array<double, 3> theta0_{}, theta_{};   // (fixed + trips x trip, per_row, per_expert x unit_reads_)
    std::array<double, 9> xx_{};
    std::array<double, 3> xy_{};
    std::array<double, max_rows + 1> ratio_{}, ratio_w_{};  // [R]: observed / prior reads
    std::uint64_t passes_ = 0, skipped_ = 0;
    bool last_skipped_ = false;
};

}  // namespace strix::spec
