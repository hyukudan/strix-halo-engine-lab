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

private:
    BlockPolicy policy_;
    BlockForward& forward_;
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

}  // namespace strix::spec
