#pragma once

// Shared speculative-drafting layer: context proposals + one arbiter + one stats line, for every engine.
//
// ContextProposer: an incremental index over one stream's emitted tokens (prompt + every emitted token, the target's
// correction token included as soon as it is emitted) proposing variable-length continuations (up to 64 tokens) from
// an earlier occurrence, with a copy cursor that follows a source run in O(1) while the stream keeps matching it.
//
// Arbiter: per step picks serial, lookup(k) or the engine's learned drafter(k) by expected committed tokens per ms:
//     E(k) = 1 + sum_{j<=k} prod_{i<=j} p_i,   score = E(k) / (V(R) + D(k) + overhead),   R = physical window >= k+1
// with p_i learned per (source, bucket, depth class) from verified outcomes (lookup: every step's proposal scored against the emitted tokens) and
// V(R) learned per window size from the engine's measured walls (seeded by the engine).
//
// Exactness: proposals are arbitrary ids; the engine verifies them with its serial arithmetic, commits the greedy-equal
// prefix + correction and rolls back the rejected rows' state. The layer only reads emitted tokens, so it needs no
// rollback, and no decision of it can change the emitted ids.
#include "strix/lookup/index.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::spec {

inline constexpr std::uint32_t max_proposal = 64;  // longest lookup proposal (verify windows up to 65 rows)

enum class Source : std::uint8_t { serial = 0, lookup = 1, model = 2 };
inline constexpr std::size_t source_count = 3;
[[nodiscard]] const char* source_name(Source source);

struct Proposal {
    std::vector<std::uint32_t> tokens;  // empty: no proposal
    std::uint32_t match_length = 0;     // matched suffix length (resumed: run length followed by the cursor)
    std::uint64_t distance = 0;         // recency of the source occurrence
    bool resumed = false;               // continued from the copy cursor
    std::uint64_t run = 0;              // copy run id (a fresh source starts a new run; resumed proposals keep it)
    [[nodiscard]] std::uint32_t bucket() const;  // survival bucket (fresh match class, or resumed run class)
};
inline constexpr std::uint32_t lookup_buckets = 6;  // fresh: match 2-3, 4-7, 8-15, 16+; resumed: run < 16, >= 16

struct ProposerConfig {
    strix::lookup::IndexConfig index{1U << 17, 64};  // 128K tokens: prompt + output of every suite task
    std::uint32_t min_match = 2;                     // shortest fresh match proposed
};

class ContextProposer {
public:
    explicit ContextProposer(const ProposerConfig& config = {});
    // Emitted tokens, once each, in stream order (prompt first). Advances or drops the copy cursor.
    void commit(std::span<const std::uint32_t> tokens);
    // Continuation of stream + extra (extra = a hypothetical prefix, e.g. a model draft to extend), at most max_len.
    [[nodiscard]] Proposal propose(std::span<const std::uint32_t> extra, std::uint32_t max_len);
    // The same into a caller-owned proposal (its token storage is reused across steps).
    void propose(Proposal& out, std::span<const std::uint32_t> extra, std::uint32_t max_len);
    [[nodiscard]] std::uint64_t tokens() const { return index_->tokens(); }
    [[nodiscard]] std::size_t bytes() const { return index_->bytes(); }

private:
    ProposerConfig config_;
    std::unique_ptr<strix::lookup::Index> index_;
    bool cursor_valid_ = false;
    std::uint64_t cursor_ = 0;  // source position whose successor is predicted to be the next emitted token
    std::uint32_t run_ = 0;     // tokens the cursor has followed (anchor match included)
    std::uint64_t run_id_ = 0;
};

// Survival per depth class: depths 1..8 each, then 9-12, 13-16, 17-24, 25-32, 33-48, 49-64 (one hazard per class).
inline constexpr std::size_t depth_classes = 14;
[[nodiscard]] std::size_t depth_class(std::uint32_t depth);
// Verify costs are kept per context bin: < 2K, 2K-8K, 8K-32K, 32K-128K, >= 128K positions.
inline constexpr std::size_t context_bins = 5;
[[nodiscard]] std::size_t context_bin(std::uint64_t positions);

struct ArbiterConfig {
    std::vector<std::uint32_t> windows{1, 2, 3, 4, 5, 6, 7, 8};  // physical verify windows (rows), ascending, 1 first
    double switch_in = 1.05;            // another drafting arm must beat the current one by this factor
    double half_life_tokens = 256.0;    // evidence decay per emitted token
    std::uint32_t probe_every = 32;     // passes: re-probe a barred / unused learned drafter (at its deepest k)
    std::uint32_t retreat_misses = 2;   // consecutive first-row misses that bar a source (lookup: until new evidence;
                                        // never a black-box drafter: its measured rate already prices its misses)
    double retreat_confidence = 0.01;   // learned drafter: bar only a miss streak at most this probable under its rate
    // ... over this many passes (the evidence horizon), not per pass: per pass, a 0.64 first-row rate barred after 5 misses,
    // which happens by chance ~2.3 times per 580 passes (measured on CUDA dense 27B Japanese prose: 33-566 serial passes per
    // 1,200 tokens, ~4.7 % at 32 serial passes per bar), while a miss pass costs only its drafts. 0 = the per-pass form.
    double retreat_horizon = 256.0;
    double min_lookup_evidence = 4.0;   // observations a lookup bucket needs before lookup may run (fresh buckets:
                                        // depth 1; resumed copy buckets: one hazard per emitted position)
    std::uint32_t min_model_passes = 3; // the learned drafter runs this often before it is compared
    // Frontier probe (per-depth learned drafter only): expected() credits nothing to a depth no proposal has reached, and a
    // drafter that wins at k only learns depths <= k + 1 of what it offers, so after a first-row miss on the first probe it
    // stayed at k = 1 for good (measured: 290 of 291 MTP passes at depth 1, E 1.76, where depth 2 was worth ~0.5 token).
    // While the model arm wins at k and depth k + 1 has decayed evidence below frontier_evidence, it offers k + 1 once every
    // frontier_every passes. 0 = off (the previous form). Drafts never change emitted tokens; this only prices depth.
    std::uint32_t frontier_every = 8;
    double frontier_evidence = 0.5;
    // The probe is charged its cost: it fires only when its priced score (expected tokens / (verify + draft ms)) is within
    // frontier_loss of the pick, so it can cost at most frontier_loss / frontier_every of the throughput per pass.
    double frontier_loss = 0.05;
    // The exploration probe (min_model_passes, probe_every) is charged the same way: it runs at the deepest legal window
    // whose priced excess over the pick stays within frontier_loss x probe_every pick passes (none: the shallowest), so a
    // context where serial wins (K/V-priced verify rows) never pays an unpriced deep pass (128K, k 15: ~8 serial passes).
    // false = the previous form (always the deepest legal window). Drafts never change emitted tokens.
    bool charge_probe = true;
    // Per-depth learned drafter: the cumulative survival S_j is shrunk toward the geometric continuation of the shallower
    // depths (m_j = B_{j-1}^2 / B_{j-2}, m_1 = the prior mean) instead of the flat prior mean, and a depth without evidence is
    // priced at m_j. The flat mean (false, the previous form) pulls deep, little-evidenced S_j up toward 0.5: measured on the
    // CUDA dense 27B Japanese prose, S_4 0.048 was priced 0.12 and the arm ran k = 4-5 instead of the k = 3 optimum (-3.8 %).
    // Lookup sources keep the flat prior (their depth is bounded by the proposal). Only k changes; tokens never do.
    bool model_continuation = true;
    // false: measured walls never enter a decision; the seeded cost model (seed_cost) prices every window, so decide() is a
    // pure function of the seeds and the acceptance history (deterministic for a fixed prompt). Walls are still reported by
    // the caller's stats. true (the previous form): every pass's wall is folded into the window's cost.
    bool learn_costs = true;
    // Context cost of the seeded model (learn_costs = false only; learned walls already carry their context bin). The
    // decode attention re-reads the K/V once per verify row, so a row's cost grows with the context: measured on CUDA
    // dense 27B at 28K a row cost ~2.0 ms where the short-context seed priced 0.113, and the Arbiter drafted k ~7-8 where
    // k ~4 is the optimum. The device's own verify walls price it (cost_fit, default):
    //   V(r, L) = seed(r) + c + beta (r - 1) + kappa r L / 1,024
    // a weighted ridge least squares over every pass's (rows, context, wall) with forgetting, the residual clipped against
    // the current prediction (one stall cannot move it) and the per-row slope kept >= 0 (monotone in rows). It follows a
    // kernel whose rows share one K/V walk by itself (smaller kappa) and corrects the seed's shape on this device. At one
    // context the three terms are collinear: the minimum-norm split gives a long-context residual to kappa and a short one
    // to c / beta, and the prediction at the live context is right either way. No pass seen = the seeds (previous form).
    // A pass whose wall is off the prediction by more than half of it (an isolated stall) is not folded at all, from the
    // first folded pass on (the first pass: by more than the warm-up clip); a second one in a row is (clipped), so a real
    // level change is folded on every other pass while it stays off.
    // Only k changes; tokens never do. Switch for every binder: set_context_cost ("online" | "off" | KV).
    bool cost_fit = true;
    double cost_fit_half_life = 64.0;  // passes: older walls weigh half after this many
    double cost_fit_ridge = 1.0;       // pass-equivalents pulling (c, beta, kappa) toward 0 (= the seeds)
    // Identification of the K/V term. At one context the row slope of the walls over the seeds is s = beta + kappa L and
    // only the context distance between the walls and the seed table separates the two: the seeds' context is known to
    // their context bin [lo, hi) (bin 0 for a table seeded at every context: a context-blind table prices the shortest
    // bin), so kappa lies in [s / (L - lo), s / (L - hi)]. Its condition q = (L - lo) / (L - hi) is infinite while L <= hi
    // (r (L - L_seed) can vanish: collinear with rows; the slope a short run sees is row tiles and drift) and tends to 1
    // with L. The context term (fit or fixed) prices a decision only while q <= cost_fit_condition at the live context
    // and, for the fit, at its weighted mean context; otherwise the price is the seed table, bit for bit (= "off").
    // Bin-0 seeds at 1.5: from 6,144 positions. Regime rule (RTX PRO 6000, Qwen 27B dense, one run per arm, online vs off):
    // at q 1.78 (4.7K) kappa is identified only within a factor 1.8 and the term lost on one content (code -0.75 %, mixed
    // +0.8 / +4.4 % on the others), at q 1.45 (6.6K) +2.5 %, at q 1.08 (28K) +6.0 %; 1.5 holds the off plan on the whole
    // 4-5K regime and keeps both winning regimes open. 0 = no identification test (every context, the previous form).
    double cost_fit_condition = 1.5;
    // > 0: fixed context term instead of the fit (deterministic decisions for A/B runs): every verify row pays this many
    // ms per 1,024 context positions on top of the seed table, at the contexts that identify it (cost_fit_condition).
    // 0 = the fit (cost_fit) or nothing.
    double row_kv_ms_per_kpos = 0.0;
    double cost_weight_cap = 8.0;      // verify-cost EWMA: a new wall weighs >= 1/(cap+1)
    double cost_aging = 0.995;          // per pass, every window's weight decays (stale windows re-learn fast)
    double lookup_overhead_ms = 0.0;    // extra cost of a lookup pass (the proposal itself is computed every step: sunk)
    std::array<double, 2> lookup_prior{1.0, 2.0};  // Beta prior (accept, reject) per depth for lookup
    std::array<double, 2> model_prior{2.0, 2.0};
    double lookup_sd = 1.0;  // lookup: cumulative survival bounded once per depth at mean - lookup_sd * SD
    double model_sd = 0.0;
};

struct StepInput {
    const Proposal* lookup = nullptr;             // null/empty: no lookup candidate this step
    std::uint32_t model_max_k = 0;                // 0: no learned drafter; every depth 1..model_max_k must be runnable
    std::span<const double> model_draft_ms;       // [k-1] = draft wall for k model tokens (required with a drafter)
    // A drafter that picks its own window (confidence stop): its measured committed tokens and complete ms per pass
    // (engine's upper-bound estimate); when ms > 0 the model arm is scored on this rate instead of per-depth survival.
    double model_pass_tokens = 0.0, model_pass_ms = 0.0;
    std::uint32_t remaining = ~0U;                // tokens left in the budget (k <= remaining - 1; 0: nothing to do)
    std::uint32_t max_rows = ~0U;                 // legal physical rows this step (context room, alignment, slots)
    std::uint64_t context = 0;                    // positions already in the context (cost bin)
};

struct Decision {
    Source source = Source::serial;
    std::uint32_t k = 0;      // draft tokens to verify (lookup: a prefix of the proposal)
    std::uint32_t rows = 1;   // physical window (>= k + 1, from ArbiterConfig::windows); 0: nothing to do
    bool padded = false;      // rows > k + 1: the engine masks the padding rows
    double expected_tokens = 1.0;
    double expected_ms = 0.0;
    bool probe = false;       // exploration (forced model measurement or re-probe), not the best estimate
};

struct Outcome {
    Source source = Source::serial;
    std::uint32_t bucket = 0;     // lookup: Proposal::bucket()
    std::uint32_t proposed = 0;   // k
    std::uint32_t accepted = 0;   // draft tokens equal to the target's greedy choice (prefix)
    std::uint32_t rows = 1;       // physical window verified
    double verify_ms = 0.0;       // complete host wall of the pass minus draft: verify + commit + exchange
    double draft_ms = 0.0;        // model draft wall (0 for lookup/serial)
    std::uint64_t context = 0;    // positions in the context when the pass started
};

class Arbiter {
public:
    explicit Arbiter(ArbiterConfig config = {});
    // Engine's tabled verify wall at `rows`: the shape of the cost curve; measured walls set its scale and replace
    // the seed where measured. context: the bin to seed; the default seeds every bin.
    void seed_cost(std::uint32_t rows, double ms, std::uint64_t context = ~0ULL);
    void seed_model_survival(std::span<const double> conditional_by_depth, double weight = 8.0);
    // The step's lookup proposal (if any) is kept and scored against the tokens emitted from now on, whatever runs,
    // one pending proposal per copy run (overlapping cursor proposals are not counted twice).
    [[nodiscard]] Decision decide(const StepInput& input);
    // After the pass: its outcome (costs, model survival) and the tokens it emitted (accepted drafts + correction).
    void observe(const Outcome& outcome, std::span<const std::uint32_t> emitted);
    // The stream restarted elsewhere (new request / non-contiguous position): drop unresolved proposals.
    void reset_pending() { pending_.clear(); }
    // Current estimate: interpolated between measured windows, never below a smaller measured window; with seeded costs
    // (learn_costs = false) plus the context cost (fitted, or rows x row_kv_ms_per_kpos x context / 1,024).
    [[nodiscard]] double verify_ms(std::uint32_t rows, std::uint64_t context = 0) const;
    // The context cost (ArbiterConfig::cost_fit): coefficients of V(r, L) - seed(r) and the decayed pass weight.
    // kv_open: the term prices decisions at the last pass's context (ArbiterConfig::cost_fit_condition); context: the
    // fit's weighted mean context (positions); skipped: isolated outliers not folded.
    struct CostFit {
        const char* mode = "off";  // "online" | "fixed" | "off" | "learned" (walls priced per window)
        double c = 0.0, beta = 0.0, kappa = 0.0, weight = 0.0, context = 0.0;
        bool kv_open = false;
        std::uint64_t skipped = 0;
    };
    [[nodiscard]] CostFit cost_fit() const;
    // ArbiterConfig::cost_fit_condition at `context` against the seed table priced there (true: the K/V term is identified).
    [[nodiscard]] bool kv_identified(std::uint64_t context) const;
    // Posterior mean of the conditional survival at `depth` (decisions bound the cumulative survival, see .cpp).
    [[nodiscard]] double survival(Source source, std::uint32_t bucket, std::uint32_t depth) const;
    [[nodiscard]] const ArbiterConfig& config() const { return config_; }

private:
    struct Beta { double a = 0, b = 0; };
    using Table = std::array<Beta, depth_classes>;
    struct Pending { std::vector<std::uint32_t> tokens; std::uint32_t bucket = 0, matched = 0; std::uint64_t run = 0; };
    // [rows]; seed_lo / seed_hi: the context range the seed table was given for (positions; bin 0 when seeded everywhere)
    struct Costs { std::vector<double> ms, weight, seed; std::uint64_t seed_lo = 0, seed_hi = 2048; };
    [[nodiscard]] double known_ms(const Costs& c, std::uint32_t rows) const;  // measured, scaled seed, or -1
    [[nodiscard]] double table_ms(std::uint32_t rows, std::uint64_t context) const;  // verify_ms without the context term
    [[nodiscard]] bool fitting() const { return !config_.learn_costs && config_.cost_fit && !(config_.row_kv_ms_per_kpos > 0.0); }
    [[nodiscard]] double fit_correction(std::uint32_t rows, std::uint64_t context) const;
    struct FitPass {
        std::uint32_t rows = 0;  // 0 = none
        std::uint64_t context = 0;
        double verify_ms = 0.0;
    };
    [[nodiscard]] double fit_predict(const FitPass& pass, double& base) const;
    void fit_fold(const FitPass& pass);
    void fit_cost(const Outcome& outcome);
    void decay(double tokens);
    void learn(Table& table, std::uint32_t proposed, std::uint32_t accepted, double weight);
    [[nodiscard]] double expected(Source source, std::uint32_t bucket, std::uint32_t k) const;
    [[nodiscard]] std::uint32_t physical(std::uint32_t rows) const;
    ArbiterConfig config_;
    std::array<Table, lookup_buckets> lookup_{};
    std::array<double, lookup_buckets> lookup_evidence_{};
    Table model_{};
    std::vector<Pending> pending_;  // unresolved lookup proposals, oldest first (at most max_proposal)
    std::array<Costs, context_bins> cost_{};
    std::array<double, 9> fit_xx_{};  // decayed sum of w x x^T over the features (1, rows - 1, rows x context / 1,024)
    std::array<double, 3> fit_xy_{}, fit_coef_{};
    double fit_w_ = 0.0, fit_wl_ = 0.0;  // decayed pass weight and weight x context (positions)
    std::uint64_t fit_last_context_ = 0, fit_skipped_ = 0;
    bool fit_last_skipped_ = false;
    FitPass fit_held_{};  // the pass the stall test held back (rows 0 = none)
    Source last_ = Source::serial;
    std::uint64_t passes_ = 0, last_model_ = 0, model_passes_ = 0, model_barred_until_ = 0, last_frontier_ = 0;
    std::uint32_t model_misses_ = 0, lookup_misses_ = 0;
    bool last_black_box_ = false;  // the last model decision used the drafter's measured rate (no retreat for it)
    bool lookup_barred_ = false;
    std::array<Decision, source_count> arms_{};  // best option per source of the last decide() (trace)
    std::array<double, source_count> arm_scores_{};

public:
    [[nodiscard]] const std::array<Decision, source_count>& last_arms() const { return arms_; }
    [[nodiscard]] const std::array<double, source_count>& last_arm_scores() const { return arm_scores_; }
};

// One counters object per stream (or summed over a run) and THE line every engine prints at the end of a run:
//   strix-spec-stats {"passes":..,"emitted":..,"tokens_per_pass":..,"sources":{..},"depth_offered":[..],
//                     "depth_accepted":[..],"rows":{..},"draft_ms":..,"lookup_us":{..},"index_bytes":..
//                     [,"cost_fit":{"mode":..,"c":..,"beta":..,"kappa":..,"weight":..,"kv":"open"|"closed","ctx":..,"skipped":..}]}  (cost_fit: set by the drafter)
struct SpecStats {
    Arbiter::CostFit cost_fit{};  // the Arbiter's context cost after the last pass; mode "off" with no pass: not printed
    bool has_cost_fit = false;
    std::uint64_t passes = 0, emitted = 0;
    std::array<std::uint64_t, source_count> source_passes{}, source_proposed{}, source_accepted{};
    std::array<std::uint64_t, max_proposal> depth_offered{}, depth_accepted{};  // [d-1]
    std::vector<std::uint64_t> rows_passes;   // [rows]
    std::vector<double> rows_ms;              // [rows] summed verify wall
    double draft_ms = 0;
    std::array<std::uint64_t, 32> lookup_us_hist{};  // [i]: proposal host time in [2^(i-3), 2^(i-2)) us
    double lookup_us_max = 0;
    std::size_t index_bytes = 0;
    void record(const Outcome& outcome, std::uint32_t emitted_this_pass);
    void lookup_time(double us);
    [[nodiscard]] double lookup_us_quantile(double q) const;  // upper bucket edge
    [[nodiscard]] std::string json() const;
    [[nodiscard]] std::string line() const { return "strix-spec-stats " + json(); }
};

// The context cost switch of every binder (CLI = serve: one parser, one printed name). "online" = the fit (default),
// "off" = the seeds only (the previous form), a number KV > 0 (<= 10) = fixed row_kv_ms_per_kpos (deterministic).
// Throws std::invalid_argument on anything else.
void set_context_cost(ArbiterConfig& config, const std::string& value);
[[nodiscard]] std::string context_cost_name(const ArbiterConfig& config);  // "online" | "off" | "fixed 0.0699" | "learned"

} // namespace strix::spec
