#include "strix/spec/block_drafter.hpp"

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
      chain_ms_(chain_steps_cap + 1, 0.0), chain_w_(chain_steps_cap + 1, 0.0), step_ms_(config_.chain_seed_step_ms) {
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

double BlockDrafter::chain_ms(std::uint32_t steps) const {
    steps = std::clamp<std::uint32_t>(steps, 1, chain_steps_cap);
    if (config_.fixed_costs) return config_.chain_seed_flush_ms + config_.chain_seed_step_ms * double(steps - 1);
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
    if (config_.fixed_costs || !(ms > 0.0)) return;
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
}

double BlockDrafter::sibling_q(std::uint32_t depth) const {
    const auto& p = config_.sibling_prior;
    if (depth < 1 || depth > sib_q_.size()) return p[0] / (p[0] + p[1]);
    const auto& h = sib_q_[depth - 1];
    return (h[0] + p[0]) / (h[0] + h[1] + p[0] + p[1]);
}

// Cost rule (header comment): candidates by expected gain g_j = q_j R_{j-1} (1 - s_j), kept while g > c x max(E / T,
// serial rate), i.e. the block's tokens per ms rises and no row buys a token dearer than serial decode pays (E, T =
// the Arbiter's expected tokens / ms of the block, c = the next row's price).
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
        if (b.alts[j - 1] != kNoToken) cand.push_back({sibling_q(j) * reach * (1.0 - s), j});
        reach *= s;
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& x, const Cand& y) { return x.gain > y.gain; });
    double tokens = std::max(b.decision.expected_tokens, 1.0);
    double ms = std::max(b.decision.expected_ms, arbiter_.verify_ms(k + 1, params.context));
    // A token is worth at least what serial decode pays for it: an exploratory (probe) block below the serial rate
    // must not buy rows that only lift it toward serial.
    const double serial_rate = 1.0 / std::max(arbiter_.verify_ms(1, params.context), 1e-9);
    b.sib.assign(k, 0);
    for (const auto& c : cand) {
        const std::uint32_t rows = k + 1 + b.siblings;
        if (rows >= cap) break;
        const double price = std::max(0.0, arbiter_.verify_ms(rows + 1, params.context) - arbiter_.verify_ms(rows, params.context));
        const double rate = std::max(tokens / ms, serial_rate);  // tokens per ms a row's price must beat
        if (!(c.gain > 0.0) || c.gain <= price * rate) break;   // sorted: no later candidate pays either
        b.sib[c.j - 1] = 1;
        ++b.siblings;
        tokens += c.gain;
        ms += price;
    }
    if (b.siblings == 0) b.sib.clear();
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

TreeAccept accept_tree(const TreeRows& rows, std::span<const std::uint32_t> greedy) {
    if (greedy.size() != rows.tokens.size()) throw std::runtime_error("accept_tree: greedy rows != tree rows");
    TreeAccept a;
    a.path.assign(1, 0);
    std::uint32_t cur = 0;
    for (;;) {
        const std::uint32_t g = greedy[cur];
        std::int32_t next = -1;
        for (std::uint32_t r = cur + 1; r < rows.tokens.size(); ++r)
            if (rows.parent[r] == std::int32_t(cur) && rows.tokens[r] == g && (next < 0 || rows.node[r] > 0)) next = std::int32_t(r);
        if (next < 0) break;
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
        if (hit && accepted < b.sib.size() && b.sib[accepted]) ++stats_.sibling_taken;
    }
    stats_.sibling_rows += b.siblings;
    stats_.record(o, std::uint32_t(emitted.size()));
    arbiter_.observe(o, emitted);
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

StepRecord BlockSpeculator::step(std::vector<std::uint32_t>& out, const StepParams& params) {
    StepRecord r;
    StepParams p = params;
    p.tree_rows = target_.tree_rows();
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
    auto block = drafter_.next(p, (prelaunch_ && p.tree_rows == 0) ? &launch : nullptr);
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
        if (launched_rows != 0) {
            if (launched_rows != rows_.size()) throw std::logic_error("block speculator: pre-launched verify has the wrong row count");
            target_.verify_collect(rows_, greedy_);
            ++prelaunched_;
        } else {
            target_.verify(rows_, greedy_);
        }
        if (greedy_.size() != rows_.size()) throw std::runtime_error("block speculator: target returned the wrong row count");
        while (accepted < block.drafts.size() && block.drafts[accepted] == greedy_[accepted]) ++accepted;
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
        auto acc = accept_tree(tree_, greedy_);
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
