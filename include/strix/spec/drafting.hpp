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
#include "strix/spec/wide_window.hpp"

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

enum class ArbiterMode : std::uint8_t { Adaptive, ChanceWindow };

struct ChanceWindowTables {
    struct Bin {
        float lower = 0.0f;  // inclusive; first bin starts at 0, last includes 1
        double acceptance = 0.0;  // P(row accepted | its preceding drafts accepted)
    };
    std::vector<Bin> calibration;  // strictly increasing lower edges
    std::vector<std::array<double, context_bins>> verify_ms;  // [physical rows - 1][context_bin]
    void validate(std::uint32_t max_window) const;
};

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
    // Charged re-probe interval (charge_probe, learned drafter that is not the pick): a re-probe whose priced excess over
    // the pick is E waits until E <= probe_loss x the pick's ms over the passes since the last model pass (at least
    // probe_every), so the re-probes of a drafter that loses cost at most probe_loss of decode time (plus one excess per
    // run), while a drafter near break-even (E ~ 0) is re-probed every probe_every passes as before. The bound rests on
    // learned prices: an excess the estimate understates is learned by the probe itself. 0 = the fixed probe_every
    // interval. Measured cause: on prose the fixed 32-pass probes alone cost 4.6 % (closed-loop trace of the CUDA
    // Inkling 2 Mini walls, spec_drafting_tests honest_plain_price_tests).
    // With probe_loss > 0 a probe runs the deepest legal window and opens an episode: the drafter keeps that window while
    // the episode's realized net (emitted tokens x the pick's ms per token - pass walls) stays >= 0, and the next probe
    // waits on the realized loss of the closed episode (at most one pass's excess). Bound: exploration loses at most
    // probe_loss of the pick's decode time plus one episode and the min_model_passes first measurements. Measured cause:
    // q47, an 11 ms plain step: the acceptance estimate of 5-8 probes in one 255-token request stays below break-even
    // (3.77 tokens per 8-row pass) on code that accepts ~4.9, so the estimate alone drafted 6 of 250 passes (88 vs 116 tok/s).
    // The episode's ledger counts every model pass while it is open, allows half a priced excess (one short accept must
    // not end a winning run) and caps its credit at three probe walls (a change of text stops it within a few passes).
    // 64 traced 255-token requests at the q46 walls (spec_drafting_tests honest_plain_price_tests), x plain decode:
    //   allowance 0 / 0.5 / 1 / 2 excess: code 1.19 / 1.21 / 1.23 / 1.25, prose 0.971 / 0.970 / 0.965 / 0.947
    //   (always drafting k=7: code 1.325, prose 0.536); probe_loss 0.010 / 0.015: code 1.208 / 1.213, prose 0.974 / 0.970.
    // Default 0 (the fixed interval). The value an engine runs is data: spec::resolve_probe_loss
    // (strix/spec/probe_policy.hpp) gives kMeasuredProbeLoss only to an engine and backend with a device receipt, and
    // STRIX_SPEC_PROBE_LOSS overrides it for every engine.
    double probe_loss = 0.0;
    // Priced exploration (learned per-depth drafter; replaces the deepest-window choice of charge_probe when true):
    // - a probe never moves the arm hysteresis (switch_in): leaving serial needs the estimate, not a probe's side effect.
    //   Measured: each deep probe left the arbiter on a k = 3 window that tied serial (1.002x priced) for 43 of 187 passes;
    // - the probe runs at the model arm's own best window, or one window up only when that window holds depths below
    //   frontier_evidence AND could reach switch_in x the pick with every unmeasured deeper draft accepted. A deep row the
    //   shallow acceptance cannot pay for is never probed (measured: 10 k = 7 probes whose best case was 0.97x serial);
    // - probes share one cumulative budget: priced excess (probe ms - expected tokens at the pick's ms per token) <=
    //   frontier_loss x (priced ms of the run + probe_every pick passes); over budget the base window, else no probe.
    // Applies to priced decisions only (learn_costs = false: the seeded / frozen table prices every window and depth);
    // in-run learned walls extrapolate an unmeasured draft depth linearly, so only a deep probe can price it there.
    // Every term is a seeded price or an acceptance count. false = the previous form (default; the serial CLI turns it
    // on for a run whose persisted cost table prices every window, BlockConfig::fixed_costs).
    bool priced_exploration = false;
    // Cycle stop-loss (priced decisions only, learn_costs = false): a ledger of the model arm's realized cycles, each one
    // (emitted tokens x the plain row's price - the decision's priced ms: verify window + draft), decayed with this
    // half-life in emitted tokens. When the ledger has at least cycle_min_passes decayed passes and its net is below
    // -cycle_margin x its priced ms, the arm is barred for probe_every passes and the ledger restarts. The acceptance
    // estimate (half_life_tokens) averages over a whole short run; when the text changes, it keeps buying a window
    // the current acceptance cannot pay for (measured on CUDA Inkling 2 Mini prose: 69 drafted passes at 2.20 tokens
    // per pass where R4 + draft needed 2.30, 0.928 x plain decode). Every term is a frozen price or an emitted count,
    // so decisions stay deterministic. 0 = off (default; the serial CLI turns it on with frozen prices).
    double cycle_half_life_tokens = 0.0;
    double cycle_margin = 0.0;
    double cycle_min_passes = 4.0;
    // Measured ladder: while a context bin knows the cost of only one window (measured or seeded), a lookup pass the
    // arbiter picks runs at most at the next window up, which measures it. The one-point form prices every wider window
    // at +10 % per row; measured on HIP Inkling 2 Mini a row costs +38 % (6 rows 105.6 ms against 1 row 37.7 ms, priced
    // 56.5 ms), so an unevidenced 5-draft lookup pass looked profitable, missed and cost 68 ms (prose auto 26.16 against
    // off 26.80 tok/s). With two known windows the existing linear extrapolation prices every wider window from the
    // measured slope. The choice to draft is unchanged; only the depth of the first lookup pass of a bin is capped. The
    // model arm keeps its charged exploration (its draft wall is learned per depth).
    // false = the previous form. Drafts never change emitted tokens.
    bool measured_ladder = true;
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
    // Learned walls (learn_costs) ignore the first `cold_walls` passes of every window width: those passes can carry
    // one-time work of the width (graph capture, lazy allocation, code load). Measured on a CUDA step program: the
    // first one-row pass 33-37 ms against a ~9.5 ms plain step; learned as the plain row's price, it made every
    // drafted window look cheaper than serial, and serial was never measured again (auto drafted on prose at 0.65x
    // of off). An ignored width is priced as if unmeasured (seeds or neighbours); an unmeasured, unseeded plain row
    // still forces a serial pass, so the plain row is learned from a warm pass. A cold model pass is left out of the
    // probe_loss episode's net (it neither closes the episode nor paces the next probe). 0 = every wall counts.
    std::uint32_t cold_walls = 0;
    double lookup_overhead_ms = 0.0;    // extra cost of a lookup pass (the proposal itself is computed every step: sunk)
    std::array<double, 2> lookup_prior{1.0, 2.0};  // Beta prior (accept, reject) per depth for lookup
    std::array<double, 2> model_prior{2.0, 2.0};
    double lookup_sd = 1.0;  // lookup: cumulative survival bounded once per depth at mean - lookup_sd * SD
    double model_sd = 0.0;
    ArbiterMode mode = ArbiterMode::Adaptive;
    ChanceWindowTables chance_window{};
};

struct StepInput {
    const Proposal* lookup = nullptr;             // null/empty: no lookup candidate this step
    std::uint32_t model_max_k = 0;                // 0: no learned drafter; every depth 1..model_max_k must be runnable
    std::span<const double> model_draft_ms;       // [k-1] = draft wall for k model tokens (required with a drafter)
    // A drafter that picks its own window (confidence stop): its measured committed tokens and complete ms per pass
    // (engine's upper-bound estimate); when ms > 0 the model arm is scored on this rate instead of per-depth survival.
    double model_pass_tokens = 0.0, model_pass_ms = 0.0;
    // model_pass_ms is a frozen price (a declared table value, never this run's clock). Under priced rules a black-box
    // arm is admitted only with it; learned routes ignore the flag.
    bool model_pass_frozen = false;
    std::uint32_t remaining = ~0U;                // tokens left in the budget (k <= remaining - 1; 0: nothing to do)
    std::uint32_t max_rows = ~0U;                 // legal physical rows this step (context room, alignment, slots)
    std::uint64_t context = 0;                    // positions already in the context (cost bin)
    bool force_lookup = false;                    // longest legal copy proposal, or serial on a miss
    // ChanceWindow only: one probability per already-drafted ID, in proposal order. Empty omits the model arm.
    // This replaces the pre-draft model inputs. Proposal work is sunk, so decisions price verification only.
    std::span<const float> model_confidence{};
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
    // Drop observations and pending predictions on a greedy/sampled transition. Keep the verify cost seeds.
    void reset_observations();
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
    // ChanceWindow returns the configured cost. Adaptive interpolates between measured windows, never below a smaller
    // measured window. Adaptive seeded costs (learn_costs = false) add a fitted or fixed context cost.
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
    // ArbiterConfig::probe_loss episode state (receipts): an episode is open; the realized loss (ms) of the last closed
    // one (< 0: none closed yet).
    [[nodiscard]] bool episode_open() const noexcept { return episode_open_; }
    [[nodiscard]] double episode_loss_ms() const noexcept { return episode_loss_ms_; }

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
    [[nodiscard]] std::uint32_t track_lookup(const Proposal& lookup);
    [[nodiscard]] Decision decide_chance_window(const StepInput& input);
    [[nodiscard]] std::uint32_t physical(std::uint32_t rows) const;
    // ArbiterConfig::measured_ladder: the widest window a lookup pass runs at `context` (all when off / priced).
    [[nodiscard]] std::uint32_t priced_rows(std::uint64_t context) const;
    ArbiterConfig config_;
    std::array<Table, lookup_buckets> lookup_{};
    std::array<double, lookup_buckets> lookup_evidence_{};
    Table model_{};
    std::vector<Pending> pending_;  // unresolved lookup proposals, oldest first (at most max_proposal)
    std::array<Costs, context_bins> cost_{};
    std::vector<std::uint32_t> width_walls_;  // [rows] walls seen per width (ArbiterConfig::cold_walls), all context bins
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
    // ArbiterConfig::priced_exploration: priced ms of the observed passes, spent probe excess, and the last decision's
    // priced ms and probe excess (consumed by observe()).
    double run_ms_ = 0.0, probe_spent_ms_ = 0.0, decided_ms_ = 0.0, decided_excess_ms_ = 0.0;
    // ArbiterConfig::cycle_half_life_tokens: decayed net (ms), priced ms and passes of the model arm's realized cycles,
    // and the plain row's price at the last decision.
    double cycle_net_ms_ = 0.0, cycle_priced_ms_ = 0.0, cycle_passes_ = 0.0, decided_plain_ms_ = 0.0;
    // ArbiterConfig::probe_loss episodes: realized net (ms) of the open episode at the pick's ms per token, that rate,
    // and the realized loss of the last closed episode (< 0 = none: the priced excess paces the next probe).
    // Allowance (one priced excess) and credit cap (three probe walls) of the open episode.
    double episode_net_ms_ = 0.0, episode_rate_ = 0.0, episode_loss_ms_ = -1.0, episode_allow_ms_ = 0.0, episode_cap_ms_ = 0.0;
    bool episode_open_ = false, episode_pass_ = false;
    std::array<Decision, source_count> arms_{};  // best option per source of the last decide() (trace)
    std::array<double, source_count> arm_scores_{};

public:
    [[nodiscard]] const std::array<Decision, source_count>& last_arms() const { return arms_; }
    [[nodiscard]] const std::array<double, source_count>& last_arm_scores() const { return arm_scores_; }
};

// One counters object per stream (or summed over a run) and THE line every engine prints at the end of a run:
//   strix-spec-stats {"passes":..,"emitted":..,"tokens_per_pass":..,"sources":{..},"depth_offered":[..],
//                     "depth_accepted":[..],"rows":{"R":{"passes":..,"verify_ms":..[,"warm_ms":..]}},"draft_ms":..,"lookup_us":{..},"index_bytes":..
//                     [,"cost_fit":{"mode":..,"c":..,"beta":..,"kappa":..,"weight":..,"kv":"open"|"closed","ctx":..,"skipped":..}]}  (cost_fit: set by the drafter)
struct SpecStats {
    Arbiter::CostFit cost_fit{};  // the Arbiter's context cost after the last pass; mode "off" with no pass: not printed
    bool has_cost_fit = false;
    std::uint64_t passes = 0, emitted = 0;
    std::array<std::uint64_t, source_count> source_passes{}, source_proposed{}, source_accepted{};
    std::array<std::uint64_t, max_proposal> depth_offered{}, depth_accepted{};  // [d-1]
    std::vector<std::uint64_t> rows_passes;   // [rows]
    std::vector<double> rows_ms;              // [rows] summed verify wall
    std::vector<double> rows_first_ms;        // [rows] wall of the first pass of the width (may carry one-time work)
    // Mean wall of `rows` without the width's first pass (one-time work); -1 when the width ran fewer than two passes
    // since the last restart_warm() (the row rule owns its prices: a rule change restarts these, not the totals above).
    [[nodiscard]] double warm_ms(std::size_t rows) const;
    void restart_warm();
    std::vector<std::uint64_t> warm_passes;   // [rows] passes since the last rule change
    std::vector<double> warm_sum_ms;          // [rows] their summed verify wall
    std::vector<double> warm_first_ms;        // [rows] wall of the first of them
    double draft_ms = 0;
    std::array<std::uint64_t, 32> lookup_us_hist{};  // [i]: proposal host time in [2^(i-3), 2^(i-2)) us
    double lookup_us_max = 0;
    std::size_t index_bytes = 0;
    // Token-tree siblings (block_drafter.hpp): [d-1] rejections at main-chain depth d that had a drafter runner-up
    // (trials) and those where the runner-up was the target's token (hits), recorded whether or not the sibling row was
    // placed, so a linear run measures q_d; rows = sibling rows verified, taken = siblings accepted (+1 token each).
    std::array<std::uint64_t, 16> sibling_trials{}, sibling_hits{};
    std::uint64_t sibling_rows = 0, sibling_taken = 0;
    // The same probe per drafter logit-gap bucket (block_drafter.hpp kSiblingGapEdges); tree windows verified and the
    // learned tree surcharge (ms over a linear window of the same rows) used to price siblings.
    std::array<std::uint64_t, 6> sibling_gap_trials{}, sibling_gap_hits{};
    std::uint64_t tree_windows = 0;
    double tree_surcharge_ms = 0.0;
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
// The decision rules of a run whose frozen table prices every window (the serial CLI with BlockConfig::fixed_costs):
// priced exploration plus the cycle stop-loss (half-life 32 emitted tokens, 3 % margin, 4 passes). One place, so every
// binder of a frozen table and its replay tests decide alike.
void set_frozen_price_rules(ArbiterConfig& config);

} // namespace strix::spec
