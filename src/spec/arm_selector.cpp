#include "strix/spec/arm_selector.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace strix::spec {

ArmSelector::ArmSelector(ArmSelectorConfig config) : cfg_(std::move(config)) {
    if (!(cfg_.decay > 0.0 && cfg_.decay <= 1.0)) throw std::invalid_argument("ArmSelector: decay outside (0, 1]");
    if (!(cfg_.switch_in >= 1.0)) throw std::invalid_argument("ArmSelector: switch_in below 1");
    if (cfg_.probe_passes == 0U || cfg_.probe_passes_max < cfg_.probe_passes || cfg_.probe_every == 0U ||
        cfg_.probe_max < cfg_.probe_every || !(cfg_.probe_band >= 0.0 && cfg_.probe_band < 1.0))
        throw std::invalid_argument("ArmSelector: probe passes / interval");
    if (!(cfg_.probe_stop > 0.0) || !(cfg_.probe_loss >= 0.0) || !(cfg_.drop > 0.0 && cfg_.drop <= 1.0))
        throw std::invalid_argument("ArmSelector: probe_stop / probe_loss / drop");
    if (!cfg_.clock_ms)
        cfg_.clock_ms = [] {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
        };
}

std::size_t ArmSelector::add(BlockDrafter& arm, std::string name, double probe_min_tokens, Available available,
                             bool wins_ties) {
    if (!(probe_min_tokens >= 0.0)) throw std::invalid_argument("ArmSelector: probe_min_tokens below 0");
    Arm a;
    a.drafter = &arm;
    a.min_tokens = probe_min_tokens;
    a.available = std::move(available);
    a.wins_ties = wins_ties;
    a.interval = cfg_.probe_every;
    a.stats.name = std::move(name);
    arms_.push_back(std::move(a));
    return arms_.size() - 1;
}

void ArmSelector::set_row_rule(const RowRule& rule) {
    for (auto& a : arms_) a.drafter->set_row_rule(rule);
}

void ArmSelector::commit(std::span<const std::uint32_t> tokens) {
    for (auto& a : arms_) a.drafter->commit(tokens);
}

double ArmSelector::now_ms() const { return cfg_.clock_ms(); }

double ArmSelector::rate(std::size_t arm) const {
    const auto& a = arms_.at(arm);
    return a.ms > 0.0 ? a.tok / a.ms : 0.0;
}

void ArmSelector::start_probe(std::size_t arm) {
    auto& b = arms_[arm];
    b.tok = b.ms = b.n = 0.0;  // the episode measures the arm afresh (its last rate may be another regime's)
    ++b.stats.probes;
    probing_ = arm;
    ep_n_ = 0;
    ep_net_ = 0.0;
}

std::size_t ArmSelector::begin_pass(std::uint64_t context, std::uint32_t rows) {
    if (arms_.empty()) throw std::logic_error("ArmSelector: no arms");
    const auto room = [&](std::size_t j) { return !arms_[j].available || arms_[j].available(context, rows); };
    if (probing_ != kNone) {
        if (room(probing_)) return probing_;
        probing_ = kNone;  // the episode ends without a decision (its arm ran out of room)
    }
    if (!room(active_)) {
        std::size_t j = 0;
        while (j < arms_.size() && !room(j)) ++j;
        active_ = j < arms_.size() ? j : 0;
        ++fallbacks_;
    }
    const auto& a = arms_[active_];
    if (arms_.size() == 1 || !(a.ms > 0.0 && a.n > 0.0)) return active_;
    const double a_rate = a.tok / a.ms;
    const double a_tokens = a.tok / a.n;  // tokens per pass of the active arm (its regime signal)
    for (std::size_t j = 0; j < arms_.size(); ++j) {
        if (j == active_) continue;
        const auto& b = arms_[j];
        if (a_tokens < b.min_tokens || !room(j)) continue;
        bool go = false;
        if (!(b.ms > 0.0)) {
            go = pass_ >= cfg_.first_probe;
        } else {
            const bool regime = a_rate < cfg_.drop * (b.tok / b.ms);
            const bool due = pass_ - b.last >= b.interval && ledger_ <= cfg_.probe_loss * run_ms_;
            go = regime || due;
        }
        if (go) {
            start_probe(j);
            return j;
        }
    }
    return active_;
}

void ArmSelector::end_pass(std::size_t arm, std::span<const std::uint32_t> emitted, double pass_ms,
                           std::span<const std::uint32_t> path) {
    if (arm >= arms_.size()) throw std::out_of_range("ArmSelector: arm index");
    if (arm != active_ && arm != probing_) throw std::logic_error("ArmSelector: pass of an arm that was not picked");
    for (std::size_t j = 0; j < arms_.size(); ++j)
        if (j != arm) arms_[j].drafter->follow(emitted, path);
    pass_ms = std::max(pass_ms, 0.0);
    const auto n = static_cast<double>(emitted.size());
    ++pass_;
    run_ms_ += pass_ms;
    auto& x = arms_[arm];
    x.tok = x.tok * cfg_.decay + n;
    x.ms = x.ms * cfg_.decay + pass_ms;
    x.n = x.n * cfg_.decay + 1.0;
    x.last = pass_;
    ++x.stats.passes;
    x.stats.tokens += emitted.size();
    x.stats.ms += pass_ms;
    if (arm != probing_) return;
    ++x.stats.probe_passes;
    const auto& a = arms_[active_];
    const double ms_per_token = a.tok > 0.0 ? a.ms / a.tok : 0.0;
    const double net = n * ms_per_token - pass_ms;
    ep_net_ += net;
    if (net < 0.0) ledger_ -= net;
    ++ep_n_;
    const double a_pass = a.n > 0.0 ? a.ms / a.n : 0.0;
    const bool lost = ep_net_ < -cfg_.probe_stop * a_pass;
    if (ep_n_ < cfg_.probe_passes && !lost) return;
    const double rb = x.ms > 0.0 ? x.tok / x.ms : 0.0;
    const double ra = a.ms > 0.0 ? a.tok / a.ms : 0.0;
    // inconclusive (only an arm that wins ties measures on: for the others a tie keeps the incumbent anyway)
    const bool open = x.wins_ties && !lost && rb <= cfg_.switch_in * ra && rb >= (1.0 - cfg_.probe_band) * cfg_.switch_in * ra;
    if (open && ep_n_ < cfg_.probe_passes_max) return;  // inconclusive: measure on
    if (!lost && (rb > cfg_.switch_in * ra || (open && x.wins_ties))) {
        arms_[active_].interval = cfg_.probe_every;
        x.interval = cfg_.probe_every;
        active_ = arm;
        ++switches_;
        ++x.stats.wins;
    } else {
        x.interval = std::min(x.interval * 2U, cfg_.probe_max);
    }
    probing_ = kNone;
}

}  // namespace strix::spec
