#pragma once
// Per-regime choice between the drafters (arms) of one decode: for example a bundled multi-token head (MTP, cheap per
// draft, wins on low acceptance) and a block drafter (one forward for any depth, wins only when deep drafts are
// accepted). Each arm is a complete BlockDrafter (its own Arbiter, chain wall and acceptance model, so serial / copy
// / chain / hybrid stay priced per arm); the selector picks the arm of every pass from its MEASURED rate (emitted
// tokens / pass wall, decayed) and keeps the idle arms in step through BlockDrafter::follow (stream + chain accept,
// no drafting). Scheduling only: the verify decides every emitted id, so ids equal those of any single arm.
//
// Policy (all priced from realised passes, never from a model of the other arm):
// - arm 0 starts (the incumbent). An arm with no measurement is probed once after `first_probe` passes;
// - a probe runs the arm for up to `probe_passes` passes; it ends early once its realised net (tokens x the active arm's
//   ms per token - pass walls) is below -probe_stop active passes; at the end it replaces the active arm only when its
//   rate beats the active rate by `switch_in` (hysteresis); an inconclusive probe (rate within probe_band below that bar)
//   of an arm that wins ties (below) runs on up to probe_passes_max passes before it decides (6 passes of text carry +-15 % noise, more than the few %
//   between two good drafters; measured code workload: a 6-pass block probe at 0.94x the bundled rate, 1.03x over the run);
// - a measured arm is probed again every `probe_every` passes (doubled after each losing probe, up to `probe_max`)
//   while the probe ledger (sum of the probes' realised losses) stays <= probe_loss x the decode's wall;
// - regime change: when the active rate falls below `drop` x another arm's last measured rate, that arm is probed now;
// - probe_min_tokens (per arm): probe it only while the active arm emits at least this many tokens per pass. A block
//   drafter whose forward costs ~B chain steps can only pay where ~B drafts are accepted; the adapter declares it;
// - wins_ties (per arm): at the end of an inconclusive episode (rate within probe_band of the bar after probe_passes_max)
//   the probed arm takes over when it declares wins_ties and its rate is at least (1 - tie_band) x the active rate (a
//   genuine tie; an arm measured a few % slower never takes over by the tie rule): the adapter states, from device
//   receipts, that wherever its gate opens this arm wins by a margin the probe cannot resolve (default false: the
//   incumbent keeps a tie);
// - available (per arm, optional): whether the arm can draft a pass at (context, rows) (a drafter with a context limit).
//   An unavailable arm is neither run nor probed; when the active one becomes unavailable the first available arm in
//   preference order takes over (a fallback, counted apart from switches).

#include "strix/spec/block_drafter.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace strix::spec {

struct ArmSelectorConfig {
    double decay = 0.94;             // per own pass: tokens / ms / passes of an arm (memory ~16 passes)
    double switch_in = 1.01;         // a probed arm takes over only when its rate beats the active one by this factor
    std::uint32_t first_probe = 8;   // decode passes before the first probe of an unmeasured arm
    std::uint32_t probe_every = 48;  // passes since an arm last ran before it is probed again
    std::uint32_t probe_max = 768;   // ceiling of the doubled interval
    std::uint32_t probe_passes = 6;  // probe episode length before its first decision
    std::uint32_t probe_passes_max = 18;  // an inconclusive episode (rate within probe_band of the bar) runs on to this
    double probe_band = 0.10;
    double tie_band = 0.02;          // a wins_ties arm takes an inconclusive episode only at >= (1 - tie_band) x the active rate
    double probe_stop = 1.5;         // early end: realised net loss above this many active passes
    double probe_loss = 0.01;        // periodic probes wait while the probe ledger > probe_loss x decode wall
    double drop = 0.80;              // active rate < drop x an arm's last rate: probe that arm now
    std::function<double()> clock_ms;  // wall clock in ms (default steady_clock); tests inject a virtual clock
};

struct ArmStats {
    std::string name;
    std::uint64_t passes = 0, tokens = 0, probes = 0, probe_passes = 0, wins = 0;
    double ms = 0.0;  // wall of this arm's passes
};

class ArmSelector {
public:
    static constexpr std::size_t kNone = ~std::size_t{0};
    explicit ArmSelector(ArmSelectorConfig config = {});
    // Arms in preference order (arm 0 starts); not owned. Returns the arm index.
    using Available = std::function<bool(std::uint64_t context, std::uint32_t rows)>;
    std::size_t add(BlockDrafter& arm, std::string name, double probe_min_tokens = 0.0, Available available = {},
                    bool wins_ties = false);
    void set_row_rule(const RowRule& rule);
    // Stream tokens committed outside the passes (prompt, anchor): every arm.
    void commit(std::span<const std::uint32_t> tokens);
    // The arm that drafts the next pass at `context` committed positions with up to `rows` verify rows (may open a probe
    // episode). Arm 0 must stay available (it is the fallback when no arm reports room).
    [[nodiscard]] std::size_t begin_pass(std::uint64_t context = 0, std::uint32_t rows = ~0U);
    // After the pass of `arm`: emitted = accepted drafts + next token, pass_ms = its whole wall (draft, verify, commit);
    // path = the accepted tree path (empty: linear). The other arms follow the stream.
    void end_pass(std::size_t arm, std::span<const std::uint32_t> emitted, double pass_ms,
                  std::span<const std::uint32_t> path = {});
    [[nodiscard]] BlockDrafter& drafter(std::size_t arm) const { return *arms_.at(arm).drafter; }
    [[nodiscard]] std::size_t size() const { return arms_.size(); }
    [[nodiscard]] std::size_t active() const { return active_; }
    [[nodiscard]] std::size_t probing() const { return probing_; }
    [[nodiscard]] double now_ms() const;
    [[nodiscard]] const ArmStats& stats(std::size_t arm) const { return arms_.at(arm).stats; }
    // Decayed rate of an arm (tokens per ms; 0 = not measured).
    [[nodiscard]] double rate(std::size_t arm) const;
    [[nodiscard]] std::uint64_t switches() const { return switches_; }
    [[nodiscard]] std::uint64_t fallbacks() const { return fallbacks_; }
    [[nodiscard]] double probe_ledger_ms() const { return ledger_; }
    [[nodiscard]] const ArmSelectorConfig& config() const { return cfg_; }

private:
    struct Arm {
        BlockDrafter* drafter = nullptr;
        double min_tokens = 0.0;
        Available available;
        bool wins_ties = false;
        double tok = 0.0, ms = 0.0, n = 0.0;  // decayed sums of own passes
        std::uint64_t last = 0;               // decode pass index of its last pass
        std::uint32_t interval = 0;
        ArmStats stats;
    };
    void start_probe(std::size_t arm);
    ArmSelectorConfig cfg_;
    std::vector<Arm> arms_;
    std::size_t active_ = 0, probing_ = kNone;
    std::uint64_t pass_ = 0, switches_ = 0, fallbacks_ = 0;
    std::uint32_t ep_n_ = 0;
    double ep_net_ = 0.0, run_ms_ = 0.0, ledger_ = 0.0;
};

}  // namespace strix::spec
