#pragma once
// Hy4-flash verify rows (host side): the contract of Forward::forward_rows / commit_rows, the cache-ring rewind replay,
// and the binding of the shared speculative layer (strix::spec BlockDrafter + Arbiter + BlockSpeculator through
// EngineVerifyTarget) over any forward with that contract.
//
// forward_rows(tokens[R]), 1 <= R <= kMaxVerifyRows: rows r = 0 .. R-1 at positions p0 + r in ONE pass, each row
// bitwise equal to what step(tokens[r]) gives at that position (class BASE: the same q8_rows FP32 forms with Rows = R,
// the same per-row A8 probe / down, attention per row at its own position). commit_rows(k), 0 <= k <= R: keep the first
// k rows (position p0 + k, token history cut); the device rows of the rejected positions are rewritten by the next pass
// before any read, which holds when every ring is sized for the verify rows (check_rewind below replays it).
// Home: src/hy4 (binding: Hy4 positions / rings data); the controller, proposer and Arbiter are shared (strix::spec).
#include "hy4/hip/hy4_hip_plan.hpp"
#include "strix/spec/arm_selector.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/spec/block_parallel.hpp"
#include "strix/spec/engine_target.hpp"
#include "strix/spec/mtp_chain.hpp"
#include "strix/spec/pair_round.hpp"
#include "strix/spec/probe_policy.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::hy4::hipbind::rows {

// Rows of one verify pass: the q8_rows direct / stepped forms keep the single-row arithmetic up to 8 rows and the shared
// device greedy selects 1..8 rows (strix::hip_greedy_rows_max).
inline constexpr std::uint32_t kMaxVerifyRows = 8;
// Seeds of the speculative decode (the record route, strix-hy4-rows --mtp): the verify-pass byte model's bandwidth (GB/s)
// and the chain's stated confidence, the host oracle's prose conditionals P(run >= k) .714 .452 .197 .133 .034 .017.
inline constexpr double kVerifySeedGbps = 207.0;
inline constexpr std::array<double, 6> kChainSurvival{0.714, 0.633, 0.436, 0.675, 0.256, 0.5};
// DFly2 block arm seeds (PREDICTION of hy4_df2-STATE section 4, Q8_0 drafter + FAST head; the Arbiter refines both online):
// flush = one block pass (embedding, 5 draft layers, the head over the rows), step = one hidden-correction slot.
inline constexpr double kDflySeedStepMs = 0.5, kDflySeedFlushMs = 8.0;
// Per-regime selector (spec::ArmSelector, arm 0 = the next-token block, arm 1 = DFly2): the block arm is probed only while
// the next-token arm emits >= this many tokens per pass. Measured one node (hy4_df2): a block pass 9.0-9.5 ms = 2.6 next-token
// drafts of 3.58 ms, so the block can only pay where ~2.5 drafts are accepted (prose 2.1 tokens/pass: DFly2 -22 %;
// think 2.6: +1.4 %; code 3.05: +2.9 %, pair +6.9 %).
inline constexpr double kDflyProbeMinTokens = 2.5;
// The selector's DFly2 arm starts from this stated confidence (conditional survival by depth, BlockDrafter::
// seed_chain_survival): without it the arm's Arbiter opens depth one frontier probe at a time and a probe episode measures
// only that cold start (measured t1 code: 6 probe passes at 2.67 tokens / pass vs 3.88 for DFly2 alone). Flat 0.80 =
// the offline think acceptance E(7) 4.121 (hy4_df: 1 + sum c^d, d = 1..7); code E(7) 4.637 is c ~0.86. The arm is probed
// only past kDflyProbeMinTokens (think / code regimes), where this prior holds; online counts refine it.
inline constexpr std::array<double, 7> kDflySurvival{0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80};
// Routed experts of a verify window in the shared A8 union rows form (each distinct expert read once, rows bitwise == the
// per-row loop). Off switch: STRIX_HY4_ROWS_EXPERT_UNION=0.
inline constexpr bool kExpertUnionDefault = true;  // measured bitwise (ROWS 30/30) and faster at R >= 2

// The decode step's device position words of a row at position p: {position, rope slot of its group, closing group, 0}.
[[nodiscard]] constexpr std::array<std::uint32_t, 4> position_words(std::uint32_t p, std::uint32_t ratio) {
    return {p, p / ratio * ratio, p / ratio, 0U};
}
// Rows form of a next-token block call (Forward::nextn_rows): the call's rows run in chunks through the verify rows form;
// the chunk starting at row `off` of an R-row call holds min(R - off, verify_rows) rows (verify_rows < 2: one row per
// chunk = the per-row loop). Every row r of the call keeps its own pinned position slot r (no host sync inside a call).
[[nodiscard]] constexpr std::uint32_t nextn_chunk(std::uint32_t off, std::uint32_t R, std::uint32_t verify_rows) {
    return off >= R ? 0U : std::min(R - off, std::max(1U, verify_rows));
}
// Rows one launch / one pass may write ahead of the oldest live key: the caches are sized for the larger of the batched
// prompt window and the verify rows.
[[nodiscard]] constexpr std::uint32_t cache_rows(std::uint32_t prompt_rows, std::uint32_t verify_rows) {
    return std::max({1U, prompt_rows, verify_rows});
}

// Host replay of every ring a layer keeps (sliding latent ring: window keys; indexer token-key ring: the own group's
// keys, which the group pool also reads): plain decode up to p0, one verify pass of R rows (all rows stored before any
// row reads, the order of forward_rows), commit k, then the following plain positions until every rejected slot has
// been rewritten. Every read must find the key of the position it asks for. Returns "" or the first clobber.
[[nodiscard]] inline std::string check_rewind(const Config& c, std::uint32_t layer, std::uint32_t context, std::uint32_t max_rows,
                                              std::uint32_t verify_rows) {
    const auto k = layer_caches(c, layer, context, max_rows);
    struct Ring { std::uint32_t slots; std::uint32_t span; const char* name; };  // span: keys a read covers ending at q
    std::vector<Ring> rings;
    if (c.indexed.at(layer) == 0U) rings.push_back({k.latent.ring, c.window, "latent window ring"});
    else rings.push_back({k.token_keys.ring, 0U, "indexer token ring"});
    for (const auto& ring : rings) {
        if (ring.slots == 0U) continue;  // indexed by position: a rejected row is rewritten before any later read
        const std::uint32_t ratio = c.index_ratio == 0U ? 1U : c.index_ratio;
        const auto first_read = [&](std::uint32_t q) {
            return ring.span ? (q + 1U > ring.span ? q + 1U - ring.span : 0U) : q / ratio * ratio;
        };
        for (std::uint32_t R = 1; R <= verify_rows; ++R)
            for (std::uint32_t kept = 1; kept <= R; ++kept)
                for (std::uint32_t p0 : {2U * ring.slots + 1U, 2U * ring.slots + 2U, 2U * ring.slots + 3U, 2U * ring.slots + 4U}) {
                    if (p0 + 2U * R + 1U >= context) return "context too small for the replay";
                    std::vector<std::uint32_t> owner(ring.slots, ~0U);
                    const auto write = [&](std::uint32_t q) { owner[q % ring.slots] = q; };
                    const auto read = [&](std::uint32_t q) -> std::string {
                        for (std::uint32_t x = first_read(q); x <= q; ++x)
                            if (owner[x % ring.slots] != x)
                                return std::string(ring.name) + ": layer " + std::to_string(layer) + " R " + std::to_string(R) +
                                       " kept " + std::to_string(kept) + " query " + std::to_string(q) + " key " +
                                       std::to_string(x) + " clobbered by " + std::to_string(owner[x % ring.slots]);
                        return {};
                    };
                    for (std::uint32_t q = 0; q < p0; ++q) write(q);
                    for (std::uint32_t r = 0; r < R; ++r) write(p0 + r);  // the pass stores every row first
                    for (std::uint32_t r = 0; r < R; ++r)
                        if (auto e = read(p0 + r); !e.empty()) return e;
                    for (std::uint32_t q = p0 + kept; q < p0 + R + 1U; ++q) {  // after commit: the plain redo
                        write(q);
                        if (auto e = read(q); !e.empty()) return e;
                    }
                }
    }
    return {};
}

// ---- shared speculative layer over a forward with the rows contract ------------------------------------------------
// Fwd: forward_rows(span tokens, bool logits) (synchronises), rows_greedy() -> span of R greedy ids, commit_rows(k),
// position(). EngineVerifyTarget needs Engine::Rows, Engine::LogitRows::all, forward_rows / read_top / commit / sync.
template <class Fwd>
class VerifyEngine {
public:
    struct Rows { std::uint32_t count = 0; };
    enum class LogitRows { all };
    explicit VerifyEngine(Fwd& f) : f_(f) {}
    Rows forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows) {
        f_.forward_rows(std::span<const std::uint32_t>(tokens, rows), false);
        return {rows};
    }
    void read_top(const Rows& out, std::uint32_t rows, std::uint32_t* host) const {
        const auto g = f_.rows_greedy();
        if (rows != out.count || g.size() < rows) throw std::logic_error("hy4 rows: read_top rows");
        std::copy(g.begin(), g.begin() + rows, host);
    }
    void commit(std::uint32_t k) { f_.commit_rows(k); }
    void sync() {}

private:
    Fwd& f_;
};

// ---- native MTP drafter: the shared MtpChain (strix/spec/mtp_chain.hpp) over a forward with the next-token block ----
// Fwd additionally: config().hidden, kNextnRows, nextn_rows(tokens, hidden, rows, pos0, want_top, outputs) -> {top, hidden_pre,
// hidden_post}, read_ids(top, n, host) (synchronises), rows_hidden() (device rows of the last forward_rows pass: the
// target's output_norm hidden = the teacher input), set_prompt_hidden(sink), alloc_f32 / free_f32 / copy_f32_async.
template <class Fwd>
class MtpEngine {
public:
    struct Cfg { std::uint32_t hidden = 0; };
    struct Rows {
        const std::uint32_t* top = nullptr;
        const float* hidden_pre = nullptr;
        const float* hidden_post = nullptr;
    };
    using MtpOut = Rows;
    explicit MtpEngine(Fwd& f) : f_(f) { cfg_.hidden = f.config().hidden; }
    [[nodiscard]] const Cfg& config() const { return cfg_; }
    void read_top(const Rows& r, std::uint32_t n, std::uint32_t* host) { f_.read_ids(r.top, n, host); }
    MtpOut mtp_forward(const std::uint32_t* next, const float* hidden, std::uint32_t rows, std::uint32_t pos0, bool want_top) {
        const auto o = f_.nextn_rows(next, hidden, rows, pos0, want_top);
        return {o.top, o.hidden_pre, o.hidden_post};
    }
    // MtpKvRowsEngine (strix/spec/mtp_chain.hpp): only the trailing `outputs` rows' top / hidden are read.
    MtpOut mtp_forward(const std::uint32_t* next, const float* hidden, std::uint32_t rows, std::uint32_t pos0, bool want_top,
                       std::uint32_t outputs) {
        const auto o = f_.nextn_rows(next, hidden, rows, pos0, want_top, outputs);
        return {o.top, o.hidden_pre, o.hidden_post};
    }
    float* alloc_f32(std::size_t n) { return f_.alloc_f32(n); }
    void free_f32(float* p) { f_.free_f32(p); }
    void copy_f32_async(float* d, const float* src, std::size_t n) { f_.copy_f32_async(d, src, n); }

private:
    Fwd& f_;
    Cfg cfg_;
};
template <class Fwd>
using MtpDrafter = spec::MtpChain<MtpEngine<Fwd>, spec::EngineChainDev>;

// The MTP drafter of one sequence. The teacher pass over the prompt rides on prefill: install(prompt) before
// f.prefill(prompt) routes every prompt row's hidden into the chain's ring (row i's successor = prompt[i + 1]); the last
// row waits for the anchor (finish_prompt). Teacher = the target's output_norm hidden, chain = the block's own
// shared_head_norm output (the numerics the host oracle selected by acceptance: hy4_dfly mtp).
// The prompt pass's view of a drafter fed by the prompt rows' hidden (MtpSource, DflySource): install(prompt) before the
// prefill routes the rows' output_norm hidden through Forward::set_prompt_hidden; finish_prompt(anchor) after it.
class PromptFeed {
public:
    virtual ~PromptFeed() = default;
    virtual void install(std::span<const std::uint32_t> prompt) = 0;
    virtual void finish_prompt(std::uint32_t anchor) = 0;
};

template <class Fwd>
class MtpSource final : public PromptFeed {
public:
    explicit MtpSource(Fwd& f) : f_(f), eng_(f), chain_(eng_, false, Fwd::kNextnRows) {}
    ~MtpSource() override {
        f_.set_prompt_hidden({});
        if (last_) f_.free_f32(last_);
    }
    MtpSource(const MtpSource&) = delete;
    MtpSource& operator=(const MtpSource&) = delete;
    void install(std::span<const std::uint32_t> prompt) override {
        prepare(prompt);
        f_.set_prompt_hidden([this](const float* rows, std::uint32_t p0, std::uint32_t n) { feed(rows, p0, n); });
    }
    // install() without the sink (a shared sink calls feed(); ArmsFeed).
    void prepare(std::span<const std::uint32_t> prompt) {
        if (prompt.empty()) throw std::invalid_argument("hy4 rows: MTP prompt is empty");
        prompt_.assign(prompt.begin(), prompt.end());
        base_ = f_.position();
        fed_ = 0;
        if (!last_) last_ = f_.alloc_f32(f_.config().hidden);
    }
    // After f.prefill(prompt): anchor = the greedy id after the prompt (the last prompt row's successor).
    void finish_prompt(std::uint32_t anchor) override {
        f_.set_prompt_hidden({});
        if (fed_ != prompt_.size()) throw std::logic_error("hy4 rows: MTP prompt rows not all fed by prefill");
        const typename MtpEngine<Fwd>::Rows o{nullptr, nullptr, last_};
        chain_.observe_prefill(o, 1U, &prompt_.back(), anchor);
    }
    [[nodiscard]] MtpDrafter<Fwd>& chain() { return chain_; }
    void feed(const float* rows, std::uint32_t p0, std::uint32_t n) {
        if (p0 != base_ + fed_ || std::size_t{fed_} + n > prompt_.size()) throw std::logic_error("hy4 rows: MTP prompt rows out of order");
        const std::uint32_t* toks = prompt_.data() + fed_;
        const bool tail = std::size_t{fed_} + n == prompt_.size();
        const std::uint32_t head = tail ? n - 1U : n;  // rows whose successor is known now
        if (head) {
            const typename MtpEngine<Fwd>::Rows o{nullptr, nullptr, rows};
            chain_.observe_prefill(o, head, toks, prompt_[fed_ + head]);
        }
        if (tail) f_.copy_f32_async(last_, rows + std::size_t{n - 1U} * f_.config().hidden, f_.config().hidden);
        fed_ += n;
    }

private:
    Fwd& f_;
    MtpEngine<Fwd> eng_;
    MtpDrafter<Fwd> chain_;
    std::vector<std::uint32_t> prompt_;
    std::uint32_t base_ = 0, fed_ = 0;
    float* last_ = nullptr;
};

// ---- DFly2 block drafter: the shared ParallelBlockDrafter (strix/spec/block_parallel.hpp, preset hy4_dfly2) over
// Forward::dfly_block. Fwd additionally: dfly_block(anchor, pos, k, out) (synchronises), dfly_hidden(device row),
// rows_hidden(), set_prompt_hidden(sink), position(). The drafter changes which rows are verified, never an emitted id.
template <class Fwd>
class DflyForward final : public spec::BlockForward {
public:
    explicit DflyForward(Fwd& f) : f_(f) {}
    void forward(const spec::BlockRequest& req, std::vector<std::uint32_t>& proposals, std::vector<float>&) override {
        if (req.prefix != 0U || req.ids.size() != std::size_t{req.k} + 1U) throw std::logic_error("hy4 dfly: block rows (no prefix rows)");
        proposals.resize(req.k);
        f_.dfly_block(req.ids[0], static_cast<std::uint32_t>(req.anchor_pos), req.k, proposals.data());
    }
    // KV share: the new anchor's predecessor is row emitted.size() - 1 of the last verify pass (its output_norm hidden)
    void accept(std::span<const std::uint32_t> emitted) override {
        if (emitted.empty()) throw std::logic_error("hy4 dfly: accept without emitted ids");
        f_.dfly_hidden(f_.rows_hidden() + (emitted.size() - 1U) * std::size_t{f_.config().hidden});
    }

private:
    Fwd& f_;
};

// The DFly2 drafter of one sequence: the prompt pass hands it the last prompt row's hidden (Forward::set_prompt_hidden,
// rows in prompt order), finish_prompt(anchor) starts the block drafter at the anchor's position.
template <class Fwd>
class DflySource final : public PromptFeed {
public:
    DflySource(Fwd& f, std::uint32_t block) : f_(f), policy_(policy(block)), fwd_(f), drafter_(policy_, fwd_) {}
    ~DflySource() override { f_.set_prompt_hidden({}); }
    DflySource(const DflySource&) = delete;
    DflySource& operator=(const DflySource&) = delete;
    // hy4_dfly2() with `block` rows (0: the preset's); the verify rows bound it (block <= kMaxVerifyRows).
    [[nodiscard]] static spec::BlockPolicy policy(std::uint32_t block) {
        auto p = spec::hy4_dfly2();
        if (block != 0U) p.block = block;
        if (p.block > kMaxVerifyRows) throw std::invalid_argument("hy4 dfly: block rows above the verify rows");
        spec::validate(p);
        return p;
    }
    void install(std::span<const std::uint32_t> prompt) override {
        prepare(prompt);
        f_.set_prompt_hidden([this](const float* rows, std::uint32_t p0, std::uint32_t n) { feed(rows, p0, n); });
    }
    void prepare(std::span<const std::uint32_t> prompt) {
        if (prompt.empty()) throw std::invalid_argument("hy4 dfly: prompt is empty");
        size_ = static_cast<std::uint32_t>(prompt.size());
        base_ = f_.position();
        fed_ = 0;
    }
    void feed(const float* rows, std::uint32_t p0, std::uint32_t n) {
        if (p0 != base_ + fed_ || fed_ + n > size_) throw std::logic_error("hy4 dfly: prompt rows out of order");
        fed_ += n;
        if (fed_ == size_) f_.dfly_hidden(rows + std::size_t{n - 1U} * f_.config().hidden);
    }
    void finish_prompt(std::uint32_t anchor) override {
        f_.set_prompt_hidden({});
        if (fed_ != size_) throw std::logic_error("hy4 dfly: prompt rows not all fed by prefill");
        drafter_.start(anchor, std::uint64_t{base_} + size_);
    }
    [[nodiscard]] spec::ParallelBlockDrafter& chain() { return drafter_; }

private:
    Fwd& f_;
    spec::BlockPolicy policy_;
    DflyForward<Fwd> fwd_;
    spec::ParallelBlockDrafter drafter_;
    std::uint32_t size_ = 0, base_ = 0, fed_ = 0;
};

// Both drafters fed by one prompt pass (the forward has one prompt-hidden sink): the per-regime selector's arms.
template <class Fwd>
class ArmsFeed final : public PromptFeed {
public:
    ArmsFeed(Fwd& f, MtpSource<Fwd>& mtp, DflySource<Fwd>& dfly) : f_(f), mtp_(mtp), dfly_(dfly) {}
    ~ArmsFeed() override { f_.set_prompt_hidden({}); }
    ArmsFeed(const ArmsFeed&) = delete;
    ArmsFeed& operator=(const ArmsFeed&) = delete;
    void install(std::span<const std::uint32_t> prompt) override {
        mtp_.prepare(prompt);
        dfly_.prepare(prompt);
        f_.set_prompt_hidden([this](const float* rows, std::uint32_t p0, std::uint32_t n) {
            mtp_.feed(rows, p0, n);
            dfly_.feed(rows, p0, n);
        });
    }
    void finish_prompt(std::uint32_t anchor) override {
        mtp_.finish_prompt(anchor);
        dfly_.finish_prompt(anchor);
    }

private:
    Fwd& f_;
    MtpSource<Fwd>& mtp_;
    DflySource<Fwd>& dfly_;
};

// The next-token block tensor-parallel for one decode (Forward::set_nextn_tp), the one-node form again after it.
template <class Fwd>
class NextnTp {
public:
    NextnTp(Fwd& f, bool on) : f_(f), on_(on) {
        if (on_) f_.set_nextn_tp(true);
    }
    ~NextnTp() {
        if (on_) f_.set_nextn_tp(false);
    }
    NextnTp(const NextnTp&) = delete;
    NextnTp& operator=(const NextnTp&) = delete;

private:
    Fwd& f_;
    bool on_;
};

// Verify target of the chain arm: the rows contract plus the chain's view of each pass (the committed rows' hidden).
template <class Fwd>
class ChainVerifyTarget final : public spec::VerifyTarget {
public:
    ChainVerifyTarget(Fwd& f, MtpDrafter<Fwd>& chain) : f_(f), chain_(chain) {}
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        const auto t0 = std::chrono::steady_clock::now();
        f_.forward_rows(tokens, false);  // synchronises
        verify_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const auto g = f_.rows_greedy();
        const auto R = static_cast<std::uint32_t>(tokens.size());
        if (g.size() < R) throw std::logic_error("hy4 rows: verify greedy rows");
        greedy.assign(g.begin(), g.begin() + R);
        chain_.observe_verify(typename MtpEngine<Fwd>::Rows{nullptr, nullptr, f_.rows_hidden()}, R);
    }
    void commit(std::uint32_t k) override { f_.commit_rows(k); }
    [[nodiscard]] double verify_ms() const noexcept { return verify_ms_; }  // wall of the verify passes (receipt)

private:
    Fwd& f_;
    MtpDrafter<Fwd>& chain_;
    double verify_ms_ = 0.0;
};

struct SpecOptions {
    std::uint32_t max_rows = kMaxVerifyRows;  // anchor + drafts per pass
    std::vector<double> verify_ms;            // seeds ms[R - 1], R = 1 .. max_rows (fixed costs: the Arbiter's model)
    std::uint32_t trace = 0;                  // first steps printed by the caller from SpecResult::steps
    // chain arm (spec_decode with an MtpSource): the drafter's wall seeds (block row + head; the first step after a pass
    // also flushes the committed rows) and its stated confidence, conditional survival by depth (the host oracle's
    // acceptance) weighted as pseudo-observations; the Arbiter refines both online and prices the copy / chain / hybrid arms.
    double chain_step_ms = 6.1, chain_flush_ms = 9.6;
    std::vector<double> chain_survival;
    double survival_weight = 8.0;
    bool copy = true;  // the copy source (and the hybrid copy + chain block); false = chain only
    // tensor-parallel pair: the next-token block runs on both ranks with their shares (Forward::set_nextn_tp; rank 1 replays
    // the draft rounds on its own chain); false = the one-node block on rank 0 while rank 1 waits. Both ranks must agree.
    bool draft_tp = true;
    std::function<bool(std::span<const std::uint32_t>)> publish = {};
};
// Exact prompt rows are a Hy4 adapter over the same target used by both launchers.
template <class Fwd>
std::vector<float> prefill_verify_rows(Fwd& f, std::span<const std::uint32_t> prompt, std::uint32_t width, bool feed) {
    const auto& c = f.config();
    struct Staging {
        Fwd& f;
        float* rows;
        ~Staging() { if (rows) f.free_f32(rows); }
    } staged{f, feed ? f.alloc_f32(std::size_t{width} * c.hidden) : nullptr};
    std::vector<float> last;
    const auto size = static_cast<std::uint32_t>(prompt.size());
    for (std::uint32_t pos = 0; pos < size;) {
        const auto n = std::min(width, size - pos);
        const bool tail = pos + n == size;
        f.forward_rows(prompt.subspan(pos, n), tail);
        if (feed) f.copy_f32_async(staged.rows, f.rows_hidden(), std::size_t{n} * c.hidden);
        if (tail) {
            const auto& logits = f.rows_logits();
            last.assign(logits.end() - static_cast<std::ptrdiff_t>(c.vocab), logits.end());
        }
        f.commit_rows(n);
        if (feed) f.feed_prompt_hidden(staged.rows, pos, n);
        pos += n;
    }
    return last;
}

inline std::vector<double> parse_verify_seeds(const std::string& text) {
    std::vector<double> seeds;
    std::string_view remaining(text);
    for (;;) {
        const auto comma = remaining.find(',');
        const auto part = remaining.substr(0, comma);
        double ms = 0;
        const auto result = std::from_chars(part.data(), part.data() + part.size(), ms);
        if (result.ec != std::errc{} || result.ptr != part.data() + part.size() || !std::isfinite(ms) || ms <= 0)
            throw std::invalid_argument("--verify-ms requires eight positive finite costs in milliseconds");
        seeds.push_back(ms);
        if (comma == std::string_view::npos) break;
        remaining.remove_prefix(comma + 1U);
    }
    if (seeds.size() != kMaxVerifyRows) throw std::invalid_argument("--verify-ms requires eight costs (rows 1..8)");
    return seeds;
}

inline void seed_spec_options(SpecOptions& options, double plain_ms, std::uint32_t ranks = 1U, double trip_ms = 0.0,
                              std::uint32_t gathers = 0U, std::span<const double> measured = {}) {
    options.chain_survival.assign(kChainSurvival.begin(), kChainSurvival.end());
    if (!measured.empty()) {
        options.verify_ms.assign(measured.begin(), measured.end());
        return;
    }
    const auto cost = spec::moe_verify_cost(spec::hy4_verify_bytes(), plain_ms, kVerifySeedGbps, ranks, trip_ms, gathers);
    options.verify_ms.clear();
    for (std::uint32_t r = 1; r <= options.max_rows; ++r) options.verify_ms.push_back(spec::seed_ms(cost, r));
}

struct SpecResult {
    std::vector<std::uint32_t> ids;                   // emitted ids, ids[0] = the anchor
    std::vector<spec::StepRecord> steps;
    std::array<std::uint64_t, kMaxVerifyRows + 1> rows_hist{};  // passes by rows
    std::uint64_t proposed = 0, accepted = 0;
    double wall_ms = 0;
    // chain arm: next-token block calls / rows, chain drafts, their wall (ms) and synchronising id reads
    std::uint64_t chain_calls = 0, chain_rows = 0, chain_drafts = 0, chain_reads = 0;
    double chain_ms = 0;
    double verify_ms = 0;  // chain arm: wall of the verify passes (forward_rows, synchronised)
    // per-regime selector (spec_decode with both drafters): per arm [0 next-token block, 1 DFly2] passes / tokens / wall /
    // probes / probe passes / wins, switches, probe ledger (ms of realised probe losses)
    struct Arm {
        std::uint64_t passes = 0, tokens = 0, probes = 0, probe_passes = 0, wins = 0;
        double ms = 0.0;
    };
    std::array<Arm, 2> arms{};
    std::uint64_t switches = 0, fallbacks = 0, dfly_calls = 0;
    double probe_ledger_ms = 0.0, dfly_ms = 0.0;
    bool selector = false;
};

namespace detail {
// The BlockDrafter of one arm: windows 1..max_rows, copy / chain / hybrid arms, the chain's wall seeds.
inline spec::BlockConfig block_config(const SpecOptions& o, bool chain) {
    if (o.max_rows < 1U || o.max_rows > kMaxVerifyRows) throw std::invalid_argument("hy4 rows: spec max rows outside 1..8");
    spec::BlockConfig bc;
    bc.max_block = o.max_rows;
    bc.arbiter.windows.clear();
    for (std::uint32_t w = 1; w <= o.max_rows; ++w) bc.arbiter.windows.push_back(w);
    bc.copy = o.copy || !chain;
    bc.chain = chain;
    bc.hybrid = chain && o.copy;
    bc.chain_max_k = o.max_rows > 1U ? o.max_rows - 1U : 1U;
    bc.chain_seed_step_ms = o.chain_step_ms;
    bc.chain_seed_flush_ms = o.chain_flush_ms;
    bc.fixed_costs = !o.verify_ms.empty();
    spec::apply_probe_loss(bc.arbiter, spec::kFixedReprobeInterval);  // no [speculation.probe_loss] row in hy4-flash.toml
    return bc;
}
inline void seed_arm(spec::BlockDrafter& d, const SpecOptions& o, bool chain, std::span<const double> survival) {
    for (std::uint32_t r = 1; r <= o.max_rows && r <= o.verify_ms.size(); ++r) d.seed_verify_ms(r, o.verify_ms[r - 1U]);
    if (chain && !survival.empty()) d.seed_chain_survival(survival, o.survival_weight);
}
// The decode loop over a speculator whose drafter(s) have seen the prompt and the anchor.
template <class Fwd>
SpecResult spec_loop(Fwd& f, std::uint32_t anchor, std::uint32_t n, const SpecOptions& o, spec::BlockSpeculator& sp) {
    SpecResult res;
    res.ids.push_back(anchor);
    sp.start(anchor);
    const auto t0 = std::chrono::steady_clock::now();
    bool more = !o.publish || o.publish(res.ids);
    while (more && res.ids.size() < n) {
        spec::StepParams p;
        p.remaining = n - static_cast<std::uint32_t>(res.ids.size());
        p.max_rows = o.max_rows;
        p.context = f.position();
        const auto previous = res.ids.size();
        const auto r = sp.step(res.ids, p);
        if (r.emitted == 0U) break;
        res.steps.push_back(r);
        ++res.rows_hist[std::min<std::uint32_t>(r.rows, kMaxVerifyRows)];
        res.proposed += r.proposed;
        res.accepted += r.accepted;
        if (o.publish) more = o.publish(std::span<const std::uint32_t>(res.ids).subspan(previous));
    }
    res.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (res.ids.size() > n) res.ids.resize(n);
    return res;
}
// One speculative decode over `target` (the rows contract, with the chain's view when `chain` is set).
template <class Fwd>
SpecResult spec_run(Fwd& f, std::span<const std::uint32_t> prompt, std::uint32_t anchor, std::uint32_t n, const SpecOptions& o,
                    spec::ChainDrafter* chain, std::span<const double> survival, spec::VerifyTarget& target) {
    spec::BlockDrafter drafter(block_config(o, chain != nullptr), chain);
    seed_arm(drafter, o, chain != nullptr, survival);
    drafter.commit(prompt);
    drafter.commit(std::span<const std::uint32_t>(&anchor, 1));
    spec::BlockSpeculator sp(drafter, target);
    return spec_loop(f, anchor, n, o, sp);
}
// The per-regime selector over the two drafters: arm 0 = the next-token block (`mto`), arm 1 = DFly2 (`dfo`).
// The DFly2 arm drafts only where its block has room (Forward::dfly_room: context and index-share budget); past it the
// selector falls back to the next-token arm instead of the block refusing.
template <class Fwd>
SpecResult select_run(Fwd& f, std::span<const std::uint32_t> prompt, std::uint32_t anchor, std::uint32_t n, const SpecOptions& mto,
                      const SpecOptions& dfo, spec::ChainDrafter& mtp, spec::ChainDrafter& dfly, spec::VerifyTarget& target,
                      const spec::ArmSelectorConfig& cfg) {
    if (mto.max_rows != dfo.max_rows) throw std::invalid_argument("hy4 rows: selector arms with different max rows");
    spec::BlockDrafter dm(block_config(mto, true), &mtp), dd(block_config(dfo, true), &dfly);
    seed_arm(dm, mto, true, mto.chain_survival);
    seed_arm(dd, dfo, true, dfo.chain_survival.empty() ? std::span<const double>(kDflySurvival) : std::span<const double>(dfo.chain_survival));
    spec::ArmSelector sel(cfg);
    sel.add(dm, "mtp");
    // wins_ties: past the gate DFly2 won every measured regime (think +1.4 %, code +2.9 %, pair code +6.9 %; hy4_df2)
    sel.add(dd, "dfly", kDflyProbeMinTokens, [&f](std::uint64_t context, std::uint32_t rows) { return f.dfly_room(context, rows); },
            true);
    sel.commit(prompt);
    sel.commit(std::span<const std::uint32_t>(&anchor, 1));
    spec::BlockSpeculator sp(sel, target);
    auto res = spec_loop(f, anchor, n, mto, sp);
    res.selector = true;
    for (std::size_t a = 0; a < 2; ++a) {
        const auto& st = sel.stats(a);
        res.arms[a] = {st.passes, st.tokens, st.probes, st.probe_passes, st.wins, st.ms};
    }
    res.switches = sel.switches();
    res.fallbacks = sel.fallbacks();
    res.probe_ledger_ms = sel.probe_ledger_ms();
    return res;
}
}  // namespace detail

// Speculative greedy decode of n ids after a prompt already in `f` (position == prompt.size()); anchor = the greedy
// id after the prompt (ids[0]). Copy proposer; the Arbiter decides the rows of every pass.
// Output ids are those of plain greedy decode when forward_rows rows == step (the shared controller emits only the
// target's own greedy ids: accepted drafts equal them, the first mismatch is replaced by the target's id).
template <class Fwd>
SpecResult spec_decode(Fwd& f, std::span<const std::uint32_t> prompt, std::uint32_t anchor, std::uint32_t n, const SpecOptions& o,
                       const spec::pair_round::Exchange* pair = nullptr) {
    VerifyEngine<Fwd> engine(f);
    spec::EngineVerifyTarget<VerifyEngine<Fwd>> target(engine);
    if (pair == nullptr) return detail::spec_run(f, prompt, anchor, n, o, nullptr, {}, target);
    spec::pair_round::PairVerifyLeader lead(target, *pair, [&f] { return std::uint64_t{f.position()}; });
    auto res = detail::spec_run(f, prompt, anchor, n, o, nullptr, {}, lead);
    lead.finish();
    return res;
}
// The same with the next-token block chain (`mtp` installed before the prefill and finished with the anchor): the copy,
// chain and hybrid arms (copy and hybrid unless o.copy is false). The chain changes which rows are verified, never an id.
template <class Fwd>
SpecResult spec_decode(Fwd& f, std::span<const std::uint32_t> prompt, std::uint32_t anchor, std::uint32_t n, const SpecOptions& o,
                       MtpSource<Fwd>& mtp, const spec::pair_round::Exchange* pair = nullptr) {
    auto& ch = mtp.chain();
    ChainVerifyTarget<Fwd> target(f, ch);
    SpecResult res;
    if (pair == nullptr) res = detail::spec_run(f, prompt, anchor, n, o, &ch, o.chain_survival, target);
    else {  // tensor-parallel pair, rank 0: the drafter here, every verify also on rank 1 (spec_follow); with draft_tp every
            // chain draft too (rank 1 runs the same draft calls on its own chain: draft rounds)
        spec::pair_round::PairVerifyLeader lead(target, *pair, [&f] { return std::uint64_t{f.position()}; });
        const NextnTp tp(f, o.draft_tp);
        res = detail::spec_run(f, prompt, anchor, n, o, o.draft_tp ? &lead.chain(ch) : &ch, o.chain_survival, lead);
        lead.finish();
    }
    res.chain_calls = ch.calls();
    res.chain_rows = ch.rows();
    res.chain_drafts = ch.drafts();
    res.chain_reads = ch.host_reads();
    res.chain_ms = ch.draft_wall_ms();
    res.verify_ms = target.verify_ms();
    return res;
}

// The same with the DFly2 block drafter (`dfly` installed before the prefill and finished with the anchor) as the chain of
// the shared BlockDrafter: the Arbiter prices copy / block / hybrid rows (a block pass costs one forward for any k: flush =
// the block wall, step ~ one chain slot). Pair: the drafter runs on rank 0 alone (rank 1 replays the verify passes).
template <class Fwd>
SpecResult spec_decode(Fwd& f, std::span<const std::uint32_t> prompt, std::uint32_t anchor, std::uint32_t n, const SpecOptions& o,
                       DflySource<Fwd>& dfly, const spec::pair_round::Exchange* pair = nullptr) {
    auto& ch = dfly.chain();
    VerifyEngine<Fwd> engine(f);
    spec::EngineVerifyTarget<VerifyEngine<Fwd>> target(engine);
    const auto blocks0 = f.dfly_blocks();
    const double ms0 = f.dfly_ms();
    SpecResult res;
    if (pair == nullptr) res = detail::spec_run(f, prompt, anchor, n, o, &ch, {}, target);
    else {
        spec::pair_round::PairVerifyLeader lead(target, *pair, [&f] { return std::uint64_t{f.position()}; });
        res = detail::spec_run(f, prompt, anchor, n, o, &ch, {}, lead);
        lead.finish();
    }
    res.chain_calls = f.dfly_blocks() - blocks0;
    res.chain_drafts = ch.passes();
    res.chain_ms = f.dfly_ms() - ms0;
    return res;
}

// Both drafters under the shared per-regime selector (spec::ArmSelector): an ArmsFeed installed before the prefill (one
// prompt pass feeds both) and finished with the anchor. Arm 0 = the next-token block with `mto` (its chain survival, seeds,
// draft_tp), arm 1 = DFly2 with `dfo` (its block seeds). The verify target is the chain arm's (it hands every pass's rows
// to the next-token chain, which follows the stream while DFly2 drafts); ids are the plain ones whatever arm runs.
// Pair: the next-token block keeps its draft rounds on both ranks (draft_tp), DFly2 runs on rank 0; rank 1 follows with
// spec_follow(f, pair, mtp) (its chain gets every accept with the round messages). chain_calls / chain_ms: next-token
// block calls and wall; dfly_calls / dfly_ms: DFly2 blocks and wall.
template <class Fwd>
SpecResult spec_decode(Fwd& f, std::span<const std::uint32_t> prompt, std::uint32_t anchor, std::uint32_t n, const SpecOptions& mto,
                       const SpecOptions& dfo, MtpSource<Fwd>& mtp, DflySource<Fwd>& dfly,
                       const spec::pair_round::Exchange* pair = nullptr, const spec::ArmSelectorConfig& cfg = {}) {
    auto& mc = mtp.chain();
    auto& dc = dfly.chain();
    ChainVerifyTarget<Fwd> target(f, mc);
    const auto blocks0 = f.dfly_blocks();
    const double dms0 = f.dfly_ms();
    SpecResult res;
    if (pair == nullptr) res = detail::select_run(f, prompt, anchor, n, mto, dfo, mc, dc, target, cfg);
    else {
        spec::pair_round::PairVerifyLeader lead(target, *pair, [&f] { return std::uint64_t{f.position()}; });
        const NextnTp tp(f, mto.draft_tp);
        res = detail::select_run(f, prompt, anchor, n, mto, dfo, mto.draft_tp ? lead.chain(mc) : mc, dc, lead, cfg);
        lead.finish();
    }
    res.chain_calls = mc.calls();
    res.chain_rows = mc.rows();
    res.chain_drafts = mc.drafts();
    res.chain_reads = mc.host_reads();
    res.chain_ms = mc.draft_wall_ms();
    res.dfly_calls = f.dfly_blocks() - blocks0;
    res.dfly_ms = f.dfly_ms() - dms0;
    res.verify_ms = target.verify_ms();
    return res;
}

// Tensor-parallel pair, rank 1 (Forward::enable_tp on both ranks): no drafter; replays rank 0's verify passes and commits
// (strix/spec/pair_round.hpp) until its end message, which carries rank 0's position (checked).
template <class Fwd>
spec::pair_round::FollowResult spec_follow(Fwd& f, const spec::pair_round::Exchange& pair) {
    VerifyEngine<Fwd> engine(f);
    spec::EngineVerifyTarget<VerifyEngine<Fwd>> target(engine);
    return spec::pair_round::pair_follow(target, pair, [&f] { return std::uint64_t{f.position()}; });
}
// The same with rank 0's drafter on both ranks (SpecOptions::draft_tp on both): `mtp` installed before this rank's prefill
// of the same prompt and finished with the same anchor; its chain replays rank 0's draft rounds (draft ids cross-checked).
template <class Fwd>
spec::pair_round::FollowResult spec_follow(Fwd& f, const spec::pair_round::Exchange& pair, MtpSource<Fwd>& mtp) {
    auto& ch = mtp.chain();
    ChainVerifyTarget<Fwd> target(f, ch);
    const NextnTp tp(f, true);
    return spec::pair_round::pair_follow(target, pair, [&f] { return std::uint64_t{f.position()}; }, &ch);
}

}  // namespace strix::hy4::hipbind::rows
