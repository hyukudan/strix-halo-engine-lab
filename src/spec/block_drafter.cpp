#include "strix/spec/block_drafter.hpp"
#include "strix/spec/greedy_prefix.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strix::spec {

namespace {
constexpr std::uint32_t chain_steps_cap = 96;  // table size: copy prefix + extension steps

ArbiterConfig arbiter_config(const BlockConfig& c) {
    ArbiterConfig a = c.arbiter;
    if (c.fixed_costs) a.learn_costs = false;
    return a;
}
}

BlockDrafter::BlockDrafter(BlockConfig config, ChainDrafter* chain)
    : config_(std::move(config)), chain_(chain), proposer_(config_.proposer), arbiter_(arbiter_config(config_)),
      chain_ms_(chain_steps_cap + 1, 0.0), chain_w_(chain_steps_cap + 1, 0.0), warm_sum_(chain_steps_cap + 1, 0.0),
      wall_n_(chain_steps_cap + 1, 0U), step_ms_(config_.chain_seed_step_ms) {
    if (config_.max_block < 1 || config_.max_block > config_.arbiter.windows.back())
        throw std::invalid_argument("block drafter: max_block must be within the arbiter windows");
    if (config_.chain_max_k < 1 || config_.chain_max_k > ext_.size())
        throw std::invalid_argument("block drafter: chain_max_k in 1..16");
    if (!config_.clock_ms) config_.clock_ms = [] {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
    };
}

double BlockDrafter::now_ms() const { return config_.clock_ms(); }

void BlockDrafter::set_row_rule(const RowRule& rule) {
    if (sampled_ != rule.sampled()) {
        arbiter_.reset_observations();
        ext_ = {};
        sib_q_ = {};
        gap_q_ = {};
        gain_pred_ = gain_real_ = 0.0;
        lin_wall_ = {};
        tree_wall_ = {};
        sur_ = {};
        std::fill(chain_w_.begin(), chain_w_.end(), 0.0);
        chain_seeded_ = false;
        step_learned_ = false;
        step_ms_ = config_.chain_seed_step_ms;
        run_tok_ = run_ms_ = 0.0;
        sampled_ = rule.sampled();
    }
    if (chain_) chain_->set_row_rule(rule);
}

double BlockDrafter::chain_ms(std::uint32_t steps) const {
    steps = std::clamp<std::uint32_t>(steps, 1, chain_steps_cap);
    if (config_.fixed_costs && !chain_seeded_)
        return config_.chain_seed_flush_ms + config_.chain_seed_step_ms * double(steps - 1);
    if (chain_w_[steps] > 0.0) return chain_ms_[steps];
    for (std::uint32_t j = steps - 1; j >= 1; --j)
        if (chain_w_[j] > 0.0)  // no measured slope yet: proportional to the nearest measured depth
            return step_learned_ ? chain_ms_[j] + step_ms_ * double(steps - j) : chain_ms_[j] * double(steps) / double(j);
    for (std::uint32_t j = steps + 1; j <= chain_steps_cap; ++j)
        if (chain_w_[j] > 0.0)
            return step_learned_ ? std::max(step_ms_, chain_ms_[j] - step_ms_ * double(j - steps)) : chain_ms_[j] * double(steps) / double(j);
    return config_.chain_seed_flush_ms + step_ms_ * double(steps - 1);
}

void BlockDrafter::learn_chain_ms(std::uint32_t steps, double ms) {
    steps = std::clamp<std::uint32_t>(steps, 1, chain_steps_cap);
    if (!(ms > 0.0)) return;
    if (wall_n_[steps]++ > 0U) warm_sum_[steps] += ms;  // measured_chain_ms: the first wall of a depth is cold
    if (!config_.fixed_costs) fold_chain_ms(steps, ms);  // fixed_costs: walls never price (seeds do)
}

void BlockDrafter::seed_chain_ms(std::uint32_t steps, double ms) {
    steps = std::clamp<std::uint32_t>(steps, 1, chain_steps_cap);
    if (!(ms > 0.0) || !std::isfinite(ms)) return;
    fold_chain_ms(steps, ms);
    chain_seeded_ = true;
}

BlockDrafter::ChainWall BlockDrafter::measured_chain_ms(std::uint32_t steps) const {
    if (steps < 1 || steps > chain_steps_cap || wall_n_[steps] < 2U) return {};
    return {warm_sum_[steps] / double(wall_n_[steps] - 1U), wall_n_[steps] - 1U};
}

void BlockDrafter::fold_chain_ms(std::uint32_t steps, double ms) {
    // marginal step cost from the nearest other measured depth
    for (std::uint32_t d = 1; d < chain_steps_cap; ++d) {
        const std::uint32_t lo = steps > d ? steps - d : 0, hi = steps + d;
        std::uint32_t o = 0;
        if (lo >= 1 && chain_w_[lo] > 0.0) o = lo;
        else if (hi <= chain_steps_cap && chain_w_[hi] > 0.0) o = hi;
        if (o) {
            const double s = (ms - chain_ms_[o]) / (double(steps) - double(o));
            if (s > 0.01) {
                step_ms_ = step_learned_ ? 0.5 * step_ms_ + 0.5 * s : s;
                step_learned_ = true;
            }
            break;
        }
        if (lo < 1 && hi > chain_steps_cap) break;
    }
    auto& w = chain_w_[steps];
    chain_ms_[steps] = w > 0.0 ? (chain_ms_[steps] * w + ms) / (w + 1.0) : ms;
    w = std::min(w + 1.0, 8.0);
}

Block BlockDrafter::next(const StepParams& params, const PrelaunchFn* launch) {
    // Every chain call of this block: the pre-launch form when the caller asked for it and the chain supports it.
    const auto chain_draft = [&](std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) {
        if (sampled_) chain_->set_first_position(params.context + 1);
        if (launch == nullptr || !chain_->draft_prelaunch(prefix, k, out, *launch)) chain_->draft(prefix, k, out);
    };
    Block b;
    if (params.remaining == 0 || params.max_rows == 0) {
        b.decision = Decision{Source::serial, 0, 0, false, 0.0, 0.0, false};
        return b;
    }
    const std::uint32_t max_rows = std::min(params.max_rows, config_.max_block);
    if (config_.copy) proposer_.propose(proposal_, {}, config_.max_block - 1);
    else proposal_.tokens.clear();
    StepInput in;
    in.lookup = config_.copy ? &proposal_ : nullptr;
    const bool use_chain = chain_ && config_.chain;
    in.model_max_k = use_chain ? std::min(config_.chain_max_k, config_.max_block - 1) : 0;
    if (in.model_max_k > 0) {
        model_ms_.resize(in.model_max_k);
        for (std::uint32_t k = 1; k <= in.model_max_k; ++k) model_ms_[k - 1] = chain_ms(k);
        in.model_draft_ms = model_ms_;
    }
    in.remaining = params.remaining;
    in.max_rows = max_rows;
    in.context = params.context;
    in.force_lookup = config_.force_lookup;
    b.decision = arbiter_.decide(in);
    const auto& d = b.decision;
    b.source = d.source;
    if (d.source == Source::serial || d.k == 0) {
        b.source = Source::serial;
        return b;
    }
    if (d.source == Source::lookup) {
        b.drafts.assign(proposal_.tokens.begin(), proposal_.tokens.begin() + std::ptrdiff_t(d.k));
        b.copy = d.k;
        b.bucket = std::min(proposal_.bucket(), lookup_buckets - 1);
        // Hybrid: the copy proposal ended below the window; extend it with the chain when that pays.
        const std::uint32_t n = d.k;
        const std::uint32_t limit = params.remaining - 1;
        const std::uint32_t by_block = config_.max_block - 1 > n ? config_.max_block - 1 - n : 0U;
        const std::uint32_t room = std::min({config_.chain_max_k, by_block, limit > n ? limit - n : 0U,
                                             max_rows - 1 > n ? max_rows - 1 - n : 0U});
        if (config_.hybrid && use_chain && n == proposal_.tokens.size() && room > 0) {
            double full = 1.0;
            for (std::uint32_t j = 1; j <= n; ++j) full *= arbiter_.survival(Source::lookup, b.bucket, j);
            const double base = d.expected_tokens / std::max(d.expected_ms, 1e-9);
            double best = base * config_.hybrid_switch_in, surv = 1.0, gain = 0.0;
            std::uint32_t pick = 0;
            const double v0 = arbiter_.verify_ms(n + 1, params.context);
            for (std::uint32_t m = 1; m <= room; ++m) {
                const auto& e = ext_[m - 1];
                const double s = arbiter_.survival(Source::model, 0, m);
                surv *= (e[0] + 2.0 * s) / (e[0] + e[1] + 2.0);
                gain += full * surv;
                const double ms = d.expected_ms + (arbiter_.verify_ms(n + m + 1, params.context) - v0) + chain_ms(n + m);
                const double score = (d.expected_tokens + gain) / std::max(ms, 1e-9);
                if (score > best) {
                    best = score;
                    pick = m;
                }
            }
            if (pick > 0) {
                const double t0 = now_ms();
                std::vector<std::uint32_t> ext;
                chain_draft(std::span<const std::uint32_t>(b.drafts.data(), n), pick, ext);
                b.draft_ms = now_ms() - t0;
                learn_chain_ms(n + pick, b.draft_ms);
                b.ext = std::uint32_t(std::min<std::size_t>(ext.size(), pick));
                b.drafts.insert(b.drafts.end(), ext.begin(), ext.begin() + std::ptrdiff_t(b.ext));
                take_runner_ups(b, n, b.ext);
            }
        }
    } else {
        const double t0 = now_ms();
        chain_draft({}, d.k, b.drafts);
        b.draft_ms = now_ms() - t0;
        learn_chain_ms(d.k, b.draft_ms);
        if (b.drafts.size() > d.k) b.drafts.resize(d.k);
        b.ext = std::uint32_t(b.drafts.size());
        if (b.drafts.empty()) b.source = Source::serial;
        else take_runner_ups(b, 0, b.ext);
    }
    if (b.source != Source::serial) place_siblings(b, params);
    return b;
}

void BlockDrafter::take_runner_ups(Block& b, std::uint32_t first, std::uint32_t n) const {
    if (!chain_ || n == 0) return;
    const auto ru = chain_->runner_ups();  // aligned with the chain's last `out` (n positions from drafts[first])
    if (ru.empty()) return;
    b.alts.resize(b.drafts.size(), kNoToken);
    for (std::uint32_t j = 0; j < n && j < ru.size() && first + j < b.drafts.size(); ++j)
        b.alts[first + j] = ru[j] != b.drafts[first + j] ? ru[j] : kNoToken;
    const auto gp = chain_->runner_up_gaps();  // aligned with ru when present
    if (gp.size() != ru.size()) return;
    b.alt_gap.resize(b.drafts.size(), -1.0f);
    for (std::uint32_t j = 0; j < n && j < gp.size() && first + j < b.drafts.size(); ++j)
        b.alt_gap[first + j] = b.alts[first + j] != kNoToken ? gp[j] : -1.0f;
}

// q at main depth `depth`: the per-depth Beta estimate, scaled by the lift of the drafter's logit-gap bucket (the bucket's
// hit rate over the hit rate of every gap-known trial, shrunk toward 1 by kappa pseudo-trials).
double BlockDrafter::sibling_q(std::uint32_t depth, float gap) const {
    const auto& p = config_.sibling_prior;
    double qd = p[0] / (p[0] + p[1]);
    if (depth >= 1 && depth <= sib_q_.size()) {
        const auto& h = sib_q_[depth - 1];
        qd = (h[0] + p[0]) / (h[0] + h[1] + p[0] + p[1]);
    }
    const int gb = sibling_gap_bucket(gap);
    if (gb < 0) return qd;
    double hits = 0.0, trials = 0.0;
    for (const auto& g : gap_q_) {
        hits += g[0];
        trials += g[0] + g[1];
    }
    if (!(hits > 0.0)) return qd;
    const double pbar = hits / trials, kap = config_.sibling_gap_kappa;
    const auto& h = gap_q_[std::size_t(gb)];
    const double lift = (h[0] + kap * pbar) / ((h[0] + h[1] + kap) * pbar);
    return std::min(1.0, qd * lift);
}

void BlockDrafter::learn_wall(WallBin& b, double ms, bool skip_first) {
    ++b.seen;
    if (skip_first && b.seen == 1) return;  // first window of a shape: one-off graph capture / first touch
    const double n = double(skip_first ? b.seen - 1 : b.seen);
    b.ms += (ms - b.ms) / std::min(n, 16.0);
}

// Monotone in rows over the measured bins: a window is never priced below a measured narrower one of the same kind (so a
// measured row-tile step prices every wider shape before it is tried; tree windows also never below the linear price).
double BlockDrafter::lin_price(std::uint32_t rows, std::uint64_t context) const {
    double floor = 0.0;
    for (std::uint32_t r = 1; r < rows && r < lin_wall_.size(); ++r)
        if (lin_wall_[r].seen >= 2) floor = std::max(floor, lin_wall_[r].ms);
    if (rows < lin_wall_.size() && lin_wall_[rows].seen >= 2) return std::max(floor, lin_wall_[rows].ms);
    return std::max(floor, arbiter_.verify_ms(rows, context));
}

double BlockDrafter::tree_price(std::uint32_t rows, std::uint64_t context) const {
    double floor = lin_price(rows, context);
    for (std::uint32_t r = 1; r < rows && r < tree_wall_.size(); ++r)
        if (tree_wall_[r].seen >= 2) floor = std::max(floor, tree_wall_[r].ms);
    if (rows < tree_wall_.size() && tree_wall_[rows].seen >= 2) return std::max(floor, tree_wall_[rows].ms);
    return std::max(floor, lin_price(rows, context) + std::max(0.0, sur_.ms));
}

// Cost rule (header comment): candidates by expected gain g_j = q_j R_{j-1} (1 - s_j), best first; the placed set is the
// prefix of m candidates that maximises the block's tokens per ms (E + G_m) / (T + P_m), P_m = tree_price(k + 1 + m) -
// lin_price(k + 1) (measured walls, so a per-window surcharge and step-shaped row costs are priced as they are), and only
// if it beats the linear block (m = 0) and buys its tokens no dearer than serial decode (an exploratory block below the
// serial rate must not buy rows that only lift it toward serial). E, T = the Arbiter's expected tokens / ms of the block
// with its verify part at the measured linear wall.
void BlockDrafter::place_siblings(Block& b, const StepParams& params) const {
    const auto k = std::uint32_t(b.drafts.size());
    if (!config_.siblings || params.tree_rows == 0 || b.alts.size() != k || k == 0) return;
    const std::uint32_t cap = std::min(params.max_rows, params.tree_rows);
    struct Cand { double gain; std::uint32_t j; };
    std::vector<Cand> cand;
    double reach = 1.0;  // R_{j-1}: the main path survives to depth j - 1
    for (std::uint32_t j = 1; j <= k; ++j) {
        double s;
        if (b.source == Source::lookup && j <= b.copy) s = arbiter_.survival(Source::lookup, b.bucket, j);
        else s = arbiter_.survival(Source::model, 0, b.source == Source::lookup ? j - b.copy : j);
        const float gap = j - 1 < b.alt_gap.size() ? b.alt_gap[j - 1] : -1.0f;
        if (b.alts[j - 1] != kNoToken) cand.push_back({sibling_q(j, gap) * reach * (1.0 - s), j});
        reach *= s;
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& x, const Cand& y) { return x.gain > y.gain; });
    const std::uint32_t chain_rows = k + 1;
    const double lin = lin_price(chain_rows, params.context);
    // Value of a ms: the run's measured tokens per ms (a sibling set pays only if its tokens per ms beat what the run makes
    // anyway; a block's own rate is the wrong bar when the block is below the run's average), never below serial decode.
    // Before any measurement: the block's expected rate with its verify part at the measured linear wall.
    const double tokens0 = std::max(b.decision.expected_tokens, 1.0);
    const double ms0 = std::max(b.decision.expected_ms - arbiter_.verify_ms(chain_rows, params.context) + lin, std::max(lin, 1e-9));
    const double serial_rate = 1.0 / std::max(lin_price(1, params.context), 1e-9);
    const double rate = std::max(run_ms_ > 0.0 ? run_tok_ / run_ms_ : tokens0 / ms0, serial_rate);
    const double calib = (gain_real_ + 4.0) / (gain_pred_ + 4.0);  // measured: taken siblings per predicted token
    double best_net = 0.0, gain = 0.0, best_gain = 0.0;
    std::size_t best_m = 0;
    for (std::size_t m = 1; m <= cand.size(); ++m) {
        const auto rows = chain_rows + std::uint32_t(m);
        if (rows > cap || !(cand[m - 1].gain > 0.0)) break;
        gain += cand[m - 1].gain;
        const double price = std::max(0.0, tree_price(rows, params.context) - lin);
        // tokens the set adds beyond what its ms make at the run's rate; the price carries a 25 % risk margin (q, survival
        // and the walls are estimates: the host model lost up to 1.9 % on high-q sets placed at net ~0 without it)
        const double net = calib * gain - 1.25 * rate * price;
        if (net > best_net) {
            best_net = net;
            best_m = m;
            best_gain = gain;
        }
    }
    if (best_m == 0) return;
    b.sib_gain = best_gain;
    b.sib.assign(k, 0);
    for (std::size_t m = 0; m < best_m; ++m) {
        b.sib[cand[m].j - 1] = 1;
        ++b.siblings;
    }
}

void Block::tree_rows(std::uint32_t anchor, TreeRows& out) const {
    out.tokens.assign(1, anchor);
    out.parent.assign(1, -1);
    out.node.assign(1, 0);
    out.depth.assign(1, 0);
    std::int32_t last_main = 0;
    for (std::uint32_t j = 0; j < drafts.size(); ++j) {
        if (j < sib.size() && sib[j]) {  // leaf sibling first: the chain node's parent stays the last non-leaf row
            out.tokens.push_back(alts[j]);
            out.parent.push_back(last_main);
            out.node.push_back(-std::int32_t(j + 1));
            out.depth.push_back(j + 1);
        }
        out.tokens.push_back(drafts[j]);
        out.parent.push_back(last_main);
        out.node.push_back(std::int32_t(j + 1));
        out.depth.push_back(j + 1);
        last_main = std::int32_t(out.tokens.size() - 1);
    }
}

TreeAccept accept_tree(const TreeRows& rows, std::span<const std::uint32_t> greedy,
                       std::optional<std::uint32_t> eos) {
    if (greedy.size() != rows.tokens.size()) throw std::runtime_error("accept_tree: greedy rows != tree rows");
    TreeAccept a;
    a.path.assign(1, 0);
    std::uint32_t cur = 0;
    for (;;) {
        const std::uint32_t g = greedy[cur];
        std::int32_t next = -1;
        for (std::uint32_t r = cur + 1; r < rows.tokens.size(); ++r)
            if (rows.parent[r] == std::int32_t(cur) && rows.tokens[r] == g && (next < 0 || rows.node[r] > 0)) next = std::int32_t(r);
        if (next < 0 || g == eos) break;
        cur = std::uint32_t(next);
        a.path.push_back(cur);
        a.emitted.push_back(rows.tokens[cur]);
        if (rows.node[cur] > 0) ++a.main;
        else a.sibling = true;
    }
    a.emitted.push_back(greedy[cur]);
    return a;
}

void BlockDrafter::observe(const Block& b, std::uint32_t accepted, double verify_ms,
                           std::span<const std::uint32_t> emitted, std::uint64_t context,
                           std::span<const std::uint32_t> path) {
    Outcome o;
    o.source = b.source;
    o.bucket = b.bucket;
    o.proposed = b.source == Source::lookup ? b.copy : std::uint32_t(b.drafts.size());
    o.accepted = std::min(accepted, o.proposed);
    o.rows = b.rows();
    o.verify_ms = verify_ms;
    o.draft_ms = b.source == Source::model ? b.draft_ms : 0.0;
    o.context = context;
    if (b.hybrid()) {
        ++hybrid_blocks;
        hybrid_ext_proposed += b.ext;
        if (accepted >= b.copy)
            for (std::uint32_t j = 1; j <= b.ext && j <= ext_.size(); ++j) {
                if (accepted >= b.copy + j) {
                    ext_[j - 1][0] += 1.0;
                    ++hybrid_ext_accepted;
                } else {
                    ext_[j - 1][1] += 1.0;
                    break;
                }
            }
    }
    // Runner-up probe: at a rejection of main depth a + 1, emitted[a] is the target's token there (the correction, or the
    // sibling the tree took). Learned whether or not the sibling row was placed.
    const auto main_k = std::uint32_t(b.drafts.size());
    if (b.alts.size() == main_k && accepted < main_k && b.alts[accepted] != kNoToken && accepted < emitted.size()) {
        const bool hit = emitted[accepted] == b.alts[accepted];
        if (accepted < sib_q_.size()) {
            sib_q_[accepted][hit ? 0 : 1] += 1.0;
            ++stats_.sibling_trials[accepted];
            stats_.sibling_hits[accepted] += hit;
        }
        if (const int gb = sibling_gap_bucket(accepted < b.alt_gap.size() ? b.alt_gap[accepted] : -1.0f); gb >= 0) {
            gap_q_[std::size_t(gb)][hit ? 0 : 1] += 1.0;
            ++stats_.sibling_gap_trials[std::size_t(gb)];
            stats_.sibling_gap_hits[std::size_t(gb)] += hit;
        }
        if (hit && accepted < b.sib.size() && b.sib[accepted]) ++stats_.sibling_taken;
    }
    stats_.sibling_rows += b.siblings;
    // Measured walls for sibling pricing (place_siblings). A tree window's wall never enters the Arbiter's linear cost
    // bins (it carries the tree surcharge); its acceptance does.
    Outcome oa = o;
    if (verify_ms > 0.0) {
        run_tok_ = run_tok_ * 0.98 + double(emitted.size());
        run_ms_ = run_ms_ * 0.98 + verify_ms + o.draft_ms;
    }
    if (const std::uint32_t rows = b.rows(); verify_ms > 0.0 && rows < lin_wall_.size()) {
        if (b.siblings == 0) {
            learn_wall(lin_wall_[rows], verify_ms, true);
        } else {
            oa.verify_ms = 0.0;
            ++stats_.tree_windows;
            gain_pred_ += b.sib_gain;
            gain_real_ += emitted.size() == std::size_t(accepted) + 2 ? 1.0 : 0.0;  // main + sibling + next: taken
            const bool first = tree_wall_[rows].seen == 0;
            learn_wall(tree_wall_[rows], verify_ms, true);
            if (!first) {
                const double lin = lin_price(rows, context);
                const double res = std::clamp(verify_ms - lin, -0.5 * lin, 0.5 * lin);
                learn_wall(sur_, res, false);
                stats_.tree_surcharge_ms = sur_.ms;
            }
        }
    }
    stats_.record(o, std::uint32_t(emitted.size()));
    arbiter_.observe(oa, emitted);
    stats_.cost_fit = arbiter_.cost_fit();  // printed in strix-spec-stats (context cost mode and coefficients)
    stats_.has_cost_fit = true;
    proposer_.commit(emitted);
    if (chain_) {
        if (path.empty()) chain_->accept(emitted);
        else chain_->accept_path(emitted, path);
    }
}

namespace {
void put_u32(std::vector<unsigned char>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<unsigned char>((v >> (8 * i)) & 0xFFU));
}
void put_u64(std::vector<unsigned char>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<unsigned char>((v >> (8 * i)) & 0xFFU));
}
void put_f32(std::vector<unsigned char>& b, double v) {
    const auto f = static_cast<float>(v);
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    put_u32(b, u);
}
}  // namespace

RoundTraceWriter::RoundTraceWriter(const std::string& path, std::string_view label) {
    file_ = std::fopen(path.c_str(), "wb");
    if (!file_) throw std::runtime_error("round trace: cannot open " + path);
    buf_ = {'S', 'R', 'T', '1'};
    put_u32(buf_, 1);
    put_u32(buf_, std::uint32_t(label.size()));
    buf_.insert(buf_.end(), label.begin(), label.end());
    std::fwrite(buf_.data(), 1, buf_.size(), file_);
}

RoundTraceWriter::~RoundTraceWriter() {
    if (file_) std::fclose(file_);
}

void RoundTraceWriter::round(const RoundRecord& r) {
    buf_ = {'R', 'R', 'E', 'C'};
    put_u64(buf_, r.context);
    for (const std::uint32_t v : {r.anchor, r.k, r.rows, r.accepted, r.copy, r.ext}) put_u32(buf_, v);
    put_u32(buf_, std::uint32_t(r.source));  // u8 source + 3 zero bytes (little endian)
    put_f32(buf_, r.verify_ms);
    put_f32(buf_, r.draft_ms);
    for (const std::uint32_t v : r.drafts) put_u32(buf_, v);
    for (const std::uint32_t v : r.greedy) put_u32(buf_, v);
    if (std::fwrite(buf_.data(), 1, buf_.size(), file_) != buf_.size()) throw std::runtime_error("round trace: write failed");
    ++records_;
}

void BlockSpeculator::set_row_rule(const RowRule& rule) {
    drafter_.set_row_rule(rule);
    target_.set_row_rule(rule);
    row_rule_ = rule;
}

StepRecord BlockSpeculator::step(std::vector<std::uint32_t>& out, const StepParams& params) {
    StepRecord r;
    if (params.remaining == 0 || params.max_rows == 0) return r;
    const bool sampled = row_rule_.sampled();
    const bool serial_sampled = sampled && !target_.supports_row_rule(true);
    if (sampled) target_.set_first_position(params.context + 1);
    StepParams p = params;
    p.tree_rows = serial_sampled ? 0 : target_.tree_rows();
    // Verify pre-launch: the target enqueues the verify of [anchor, host prefix, device drafts] inside the chain call,
    // before the drafted ids are read back (the host launch overlaps the GPU draft). launched_rows = its row count.
    // A tree target (p.tree_rows > 0) is not pre-launched: the drafter places sibling rows only after the chain ids are
    // read (Block::siblings is decided in drafter_.next after the prelaunch callback), so an enqueued linear verify could
    // not carry them. Both features stay available; a target that reports tree rows takes the read-then-verify order.
    std::size_t launched_rows = 0;
    const PrelaunchFn launch = [&](std::span<const std::uint32_t> prefix, const DraftFeed& feed) {
        host_rows_.assign(1, anchor_);
        host_rows_.insert(host_rows_.end(), prefix.begin(), prefix.end());
        if (target_.verify_launch_fed(host_rows_, feed)) launched_rows = host_rows_.size() + feed.k;
    };
    // Pre-launch currently carries engine-owned argmax ids. Sampled verification uses the host row hook.
    auto block = serial_sampled ? Block{} : drafter_.next(p, (prelaunch_ && !sampled && p.tree_rows == 0) ? &launch : nullptr);
    if (block.decision.rows == 0) {
        if (launched_rows != 0) throw std::logic_error("block speculator: verify pre-launched for an empty block");
        return r;
    }
    std::uint32_t accepted = 0;
    std::vector<std::uint32_t> emitted;
    double verify_ms = 0.0;
    if (block.siblings == 0) {  // linear block: the original rows and commit, bitwise
        rows_.assign(1, anchor_);
        rows_.insert(rows_.end(), block.drafts.begin(), block.drafts.end());
        const double t0 = drafter_.now_ms();
        if (serial_sampled) {
            const auto logits = target_.verify_serial_logits(anchor_);
            if (logits.empty()) throw std::logic_error("block speculator: sampled serial decode requires target host logits");
            greedy_.assign(1, row_rule_.row_token(logits, params.context + 1));
        } else if (launched_rows != 0) {
            if (launched_rows != rows_.size()) throw std::logic_error("block speculator: pre-launched verify has the wrong row count");
            target_.verify_collect(rows_, greedy_);
            ++prelaunched_;
        } else {
            target_.verify(rows_, greedy_);
        }
        if (greedy_.size() != rows_.size()) throw std::runtime_error("block speculator: target returned the wrong row count");
        auto matches = greedy_matching_prefix(block.drafts, greedy_);
        if (p.eos) {
            const auto prefix = std::span(greedy_).first(matches);
            matches = static_cast<std::size_t>(std::ranges::find(prefix, *p.eos) - prefix.begin());
        }
        accepted = static_cast<std::uint32_t>(matches);
        target_.commit(accepted + 1);
        verify_ms = drafter_.now_ms() - t0;
        emitted.assign(block.drafts.begin(), block.drafts.begin() + accepted);
        emitted.push_back(greedy_[accepted]);
        drafter_.observe(block, accepted, verify_ms, emitted, params.context);
    } else {
        if (launched_rows != 0) throw std::logic_error("block speculator: verify pre-launched for a tree block");
        block.tree_rows(anchor_, tree_);
        const double t0 = drafter_.now_ms();
        target_.verify_tree(tree_, greedy_);
        if (greedy_.size() != tree_.tokens.size()) throw std::runtime_error("block speculator: target returned the wrong row count");
        auto acc = accept_tree(tree_, greedy_, p.eos);
        target_.commit_path(acc.path);
        verify_ms = drafter_.now_ms() - t0;
        accepted = acc.main;
        emitted = std::move(acc.emitted);
        r.sibling_taken = acc.sibling;
        drafter_.observe(block, accepted, verify_ms, emitted, params.context, acc.path);
    }
    r.siblings = block.siblings;
    if (sink_) {
        RoundRecord rec;
        rec.context = params.context;
        rec.source = block.source;
        rec.anchor = anchor_;
        rec.k = std::uint32_t(block.drafts.size());
        rec.rows = block.rows();
        rec.accepted = accepted;
        rec.copy = block.copy;
        rec.ext = block.ext;
        rec.verify_ms = verify_ms;
        rec.draft_ms = block.draft_ms;
        rec.drafts = block.drafts;
        rec.greedy = greedy_;
        rec.emitted = emitted;
        sink_->round(rec);
    }
    out.insert(out.end(), emitted.begin(), emitted.end());
    anchor_ = emitted.back();
    r.source = block.source;
    r.rows = block.rows();
    r.proposed = std::uint32_t(block.drafts.size());
    r.accepted = accepted;
    r.copy = block.copy;
    r.ext = block.ext;
    r.emitted = std::uint32_t(emitted.size());
    r.verify_ms = verify_ms;
    r.draft_ms = block.draft_ms;
    return r;
}

}  // namespace strix::spec
