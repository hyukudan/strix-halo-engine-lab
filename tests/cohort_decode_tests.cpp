// Host-only tests for the shared multi-request decoder (strix/serve/cohort_decode.hpp) over a fake engine.
// Invariant: every request's ids equal its solo greedy run, whatever joins, leaves, drafts or row plans happen.
#include "strix/serve/cohort_decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool value, const std::string& label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}

// Deterministic "model": the next id depends only on the lane's own context (batch invariant by construction).
// A small vocabulary makes the sequences periodic, so the context drafter finds copies and drafts get accepted.
// A context that starts with an id >= vocab is "prose": the next id hashes the position, so a copied n-gram
// continuation is right only by chance (about 1 / vocab) and drafts are mostly rejected.
std::uint32_t next_id(const std::vector<std::uint32_t>& ctx, std::uint32_t vocab) {
    if (!ctx.empty() && ctx.front() >= vocab) {
        const auto n = static_cast<std::uint32_t>(ctx.size());
        return ((n * 2654435761U) ^ (ctx.back() * 40503U) ^ (n >> 3)) % vocab;
    }
    const std::uint32_t a = ctx.empty() ? 0U : ctx.back();
    const std::uint32_t parity = static_cast<std::uint32_t>(ctx.size() % 2U);  // position enters: a wrong commit shows
    return (a * 5U + 1U + parity * (a % 2U)) % vocab;
}

std::vector<std::uint32_t> solo(const CohortRequest& r, std::uint32_t vocab) {
    std::vector<std::uint32_t> ctx = r.prompt, out;
    while (out.size() < r.max_tokens) {
        const std::uint32_t t = next_id(ctx, vocab);
        out.push_back(t);
        ctx.push_back(t);
        bool stop = false;
        for (const auto s : r.stop) stop = stop || s == t;
        if (stop) break;
    }
    return out;
}

class FakeBackend final : public CohortBackend {
public:
    FakeBackend(std::uint32_t lanes, std::uint32_t rows, std::uint32_t vocab, std::uint32_t max_pos = 4096)
        : lanes_(lanes), rows_(rows), vocab_(vocab), max_pos_(max_pos), ctx_(lanes) {}
    std::uint32_t lanes() const override { return lanes_; }
    std::uint32_t rows_per_forward() const override { return rows_; }
    std::uint32_t max_positions() const override { return max_pos_; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        check(window_.empty(), "prefill with an open window");
        if (first) ctx_.at(lane).clear();
        ctx_.at(lane).insert(ctx_[lane].end(), tokens.begin(), tokens.end());
        ++prefills;
    }
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        check(window_.empty(), "verify with an open window");
        std::set<std::uint32_t> seen;
        std::uint32_t total = 0;
        std::size_t base = 0;
        for (const auto& g : groups) {
            check(seen.insert(g.lane).second, "a lane twice in one forward");
            check(g.rows >= 1, "empty group");
            std::vector<std::uint32_t> c = ctx_.at(g.lane);
            for (std::uint32_t i = 0; i < g.rows; ++i) {
                c.push_back(tokens[base + i]);
                top[base + i] = next_id(c, vocab_);
            }
            window_.push_back({g.lane, std::vector<std::uint32_t>(tokens.begin() + static_cast<std::ptrdiff_t>(base),
                                                                  tokens.begin() + static_cast<std::ptrdiff_t>(base + g.rows))});
            base += g.rows;
            total += g.rows;
        }
        check(base == tokens.size() && tokens.size() == top.size(), "token / top sizes");
        check(total <= rows_, "rows of a forward above the cap");
        max_total = std::max(max_total, total);
        max_groups = std::max<std::size_t>(max_groups, groups.size());
        ++forwards;
    }
    void commit(std::span<const std::uint32_t> accept) override {
        check(accept.size() == window_.size(), "one keep count per group");
        for (std::size_t g = 0; g < window_.size(); ++g) {
            check(accept[g] >= 1 && accept[g] <= window_[g].second.size(), "keep out of range");
            auto& c = ctx_.at(window_[g].first);
            c.insert(c.end(), window_[g].second.begin(), window_[g].second.begin() + accept[g]);
        }
        window_.clear();
    }
    void release(std::uint32_t lane) override { ctx_.at(lane).clear(); ++releases; }

    std::uint32_t max_total = 0;
    std::size_t max_groups = 0;
    std::uint64_t forwards = 0, prefills = 0, releases = 0;

private:
    std::uint32_t lanes_, rows_, vocab_, max_pos_;
    std::vector<std::vector<std::uint32_t>> ctx_;
    std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>> window_;
};

std::vector<CohortRequest> requests(std::uint32_t count, std::uint32_t vocab) {
    std::vector<CohortRequest> out;
    for (std::uint32_t i = 0; i < count; ++i) {
        CohortRequest r;
        r.id = 100 + i;
        const std::uint32_t len = 3 + (i * 11U) % 40U;  // 3..42 tokens: several prompt slices for the long ones
        for (std::uint32_t k = 0; k < len; ++k) r.prompt.push_back((i * 13U + k * 5U) % vocab);
        r.max_tokens = 20 + (i * 17U) % 50U;
        if (i % 3 == 2) r.stop = {(i * 7U) % vocab};
        out.push_back(r);
    }
    return out;
}

void run_identity(bool draft, std::uint32_t lanes, std::uint32_t rows, const char* name) {
    const std::uint32_t vocab = 23;
    FakeBackend be(lanes, rows, vocab);
    CohortConfig cfg;
    cfg.context_draft = draft;
    cfg.prompt_slice = 8;
    cfg.prior = {13.3, 0.113, 0.13, 0.0};
    CohortDecoder dec(be, cfg);
    const auto reqs = requests(9, vocab);
    // a staggered arrival: half now, the rest while the first ones decode (join at step boundaries)
    for (std::size_t i = 0; i < reqs.size() / 2; ++i) dec.submit(reqs[i]);
    for (int s = 0; s < 5; ++s) dec.step();
    for (std::size_t i = reqs.size() / 2; i < reqs.size(); ++i) dec.submit(reqs[i]);
    dec.run();
    auto res = dec.take_finished();
    check(res.size() == reqs.size(), std::string(name) + ": every request finished");
    std::map<std::uint64_t, const CohortResult*> by_id;
    for (const auto& r : res) by_id[r.id] = &r;
    std::uint64_t drafted = 0, accepted = 0, tokens = 0;
    for (const auto& q : reqs) {
        const auto it = by_id.find(q.id);
        if (it == by_id.end()) { check(false, std::string(name) + ": missing result"); continue; }
        check(it->second->error.empty(), std::string(name) + ": no error");
        check(it->second->ids == solo(q, vocab), std::string(name) + ": ids == solo for request " + std::to_string(q.id));
        drafted += it->second->drafted;
        accepted += it->second->accepted;
        tokens += it->second->ids.size();
    }
    check(dec.peak_live() <= std::min(lanes, rows), std::string(name) + ": live lanes within the backend");
    check(be.max_total <= rows, std::string(name) + ": rows within the forward cap");
    check(be.releases == reqs.size(), std::string(name) + ": every lane released (leave)");
    if (std::min(lanes, rows) >= 2) check(be.max_groups >= 2, std::string(name) + ": requests shared a forward");
    if (draft) {
        check(drafted > 0 && accepted > 0, std::string(name) + ": drafts verified and accepted");
        check(be.forwards < tokens, std::string(name) + ": fewer forwards than tokens with accepted drafts");
    } else {
        check(drafted == 0, std::string(name) + ": no drafts when off");
    }
    std::cout << name << ": tokens " << tokens << " forwards " << be.forwards << " drafted " << drafted << " accepted "
              << accepted << " peak_live " << dec.peak_live() << " max_rows " << be.max_total << '\n';
}

void test_admission_bytes() {
    const std::uint32_t vocab = 23;
    FakeBackend be(8, 32, vocab);
    CohortConfig cfg;
    cfg.context_draft = false;
    cfg.position_bytes = 1000;
    cfg.lane_bytes = 5000;
    cfg.max_draft = 0;
    // a request reserves (prompt + max_tokens + 1) * 1000 + 5000; with prompt 10 and 30 tokens: 46,000 bytes
    cfg.memory_budget = 100000;  // two of them fit, a third waits
    CohortDecoder dec(be, cfg);
    std::vector<CohortRequest> rs;
    for (std::uint32_t i = 0; i < 5; ++i) {
        CohortRequest r;
        r.id = i;
        for (std::uint32_t k = 0; k < 10; ++k) r.prompt.push_back((i + k) % vocab);
        r.max_tokens = 30;
        rs.push_back(r);
        dec.submit(r);
    }
    CohortRequest big;
    big.id = 99;
    big.prompt = {1, 2, 3};
    big.max_tokens = 200;  // 209,000 bytes > budget: refused, never runs
    dec.submit(big);
    dec.run();
    const auto res = dec.take_finished();
    check(dec.peak_live() == 2, "bytes: two requests at once");
    bool refused = false;
    for (const auto& r : res) {
        if (r.id == 99) refused = !r.error.empty() && r.ids.empty();
        else check(r.ids == solo(rs.at(r.id), vocab), "bytes: ids == solo");
    }
    check(refused, "bytes: oversized request refused");
    check(res.size() == 6, "bytes: all answered");
}

void test_zero_output_refusal() {
    // Zero output must be rejected before draft headroom arithmetic or any backend work.
    FakeBackend be(2, 8, 23);
    CohortConfig cfg;
    CohortDecoder dec(be, cfg);
    CohortRequest zero;
    zero.id = 42;
    zero.prompt = {1, 2, 3};
    zero.max_tokens = 0;
    dec.submit(zero);
    check(dec.idle(), "zero output: refused before lane admission");
    const auto rejected = dec.take_finished();
    check(rejected.size() == 1 && rejected[0].id == zero.id && !rejected[0].error.empty() &&
              rejected[0].ids.empty(), "zero output: explicit refusal, no emitted tokens");
    check(be.forwards == 0 && be.prefills == 0 && be.releases == 0,
          "zero output: no backend work or lane allocation");

    CohortRequest one = zero;
    one.id = 43;
    one.max_tokens = 1;
    dec.submit(one);
    dec.run();
    const auto answered = dec.take_finished();
    check(answered.size() == 1 && answered[0].error.empty() && answered[0].ids == solo(one, 23),
          "one output: exact identity with solo preserved");
    check(be.forwards == 1 && be.releases == 1, "one output: one decode and one release");
}

void test_short_first() {
    const std::uint32_t vocab = 23;
    FakeBackend be(4, 16, vocab);
    CohortConfig cfg;
    cfg.context_draft = false;
    cfg.prompt_slice = 4;
    cfg.short_first_rows = 16;  // the 60-row prompt is in the long class: the 3-row one overtakes it
    CohortDecoder dec(be, cfg);
    dec.set_fake_clock(1.0);
    CohortRequest longp, shortp;
    longp.id = 1;
    for (std::uint32_t k = 0; k < 60; ++k) longp.prompt.push_back(k % vocab);
    longp.max_tokens = 4;
    shortp.id = 2;
    shortp.prompt = {5, 6, 7};
    shortp.max_tokens = 4;
    dec.submit(longp);  // older and long
    dec.submit(shortp);
    dec.run();
    double first_long = 0, first_short = 0;
    for (const auto& r : dec.take_finished()) (r.id == 1 ? first_long : first_short) = r.first_ms;
    check(first_short > 0 && first_short < first_long, "short-first: the short prompt's first token comes first");
}

// Cost fit at a steady cohort size: 8 lanes for most steps (rows 8..12 from drafting, some jitter) and a short tail of
// 7..1 lanes whose steps carry an overhead the linear model does not have. The three-term fit then gives a negative
// lane cost; clamping it to 0 kept lanes * |lane_ms| (~2 ms here, ~20 ms on the RTX t1 receipt) in the fixed cost. At
// the planned lane count the folded cost must predict the steady steps, and every term must be non-negative.
void test_cost_fit_steady_cohort() {
    const double fixed = 13.3, row = 0.113;
    CycleLedger::FitSums s;
    double steady_ms = 0.0, steady_rows = 0.0;
    auto add = [&](double r, double l, double y) {
        s.n += 1.0, s.r += r, s.l += l, s.rr += r * r, s.rl += r * l, s.ll += l * l;
        s.y += y, s.ry += r * y, s.ly += l * y, s.yy += y * y;
    };
    for (int i = 0; i < 120; ++i) {
        const double r = 8.0 + static_cast<double>(i % 5);
        const double y = fixed + row * r + ((i % 7 == 0) ? 0.4 : 0.0);
        add(r, 8.0, y);
        steady_ms += y / 120.0, steady_rows += r / 120.0;
    }
    for (int l = 7; l >= 1; --l) add(static_cast<double>(l), static_cast<double>(l), fixed + row * l + 1.5);
    const auto raw = CycleLedger::fit(s);
    check(raw.terms == 3 && raw.lane_ms < 0.0 && raw.fixed_ms > fixed + 1.5, "fixture: negative lane cost, inflated fixed");
    CohortConfig config;
    config.prior = {48.0, 7.5, 0.0, 0.113};
    const CycleCost c = CohortDecoder::fitted_cost(s, config, 8);
    check(c.lane_ms >= 0.0 && c.row_ms > 0.0 && c.fixed_ms != config.prior.fixed_ms, "fit used, terms non-negative");
    const double predicted = c.fixed_ms + c.row_ms * steady_rows;
    check(predicted > steady_ms - 0.1 && predicted < steady_ms + 0.1, "folded cost predicts the steady 8-lane steps");
    check(c.fixed_ms > fixed - 0.5 && c.fixed_ms < fixed + 0.5, "folded fixed cost near the true one");
    // Positive lane costs are kept as fitted; learn_cost off keeps the prior.
    CycleLedger::FitSums p;
    for (int i = 0; i < 40; ++i) {
        const double r = 4.0 + static_cast<double>(i % 6), l = 1.0 + static_cast<double>(i % 4);
        const double y = fixed + row * r + 0.13 * l;
        p.n += 1.0, p.r += r, p.l += l, p.rr += r * r, p.rl += r * l, p.ll += l * l;
        p.y += y, p.ry += r * y, p.ly += l * y, p.yy += y * y;
    }
    const CycleCost q = CohortDecoder::fitted_cost(p, config, 3);
    check(q.lane_ms > 0.12 && q.lane_ms < 0.14 && q.fixed_ms > fixed - 0.01 && q.fixed_ms < fixed + 0.01, "positive lane cost kept");
    CohortConfig no_learn = config;
    no_learn.learn_cost = false;
    check(CohortDecoder::fitted_cost(s, no_learn, 8).fixed_ms == config.prior.fixed_ms, "learn_cost off keeps the prior");
}

// Mixed traffic, nothing waiting (the cohort drains): prose lanes reject almost every draft, copy lanes accept most.
// The kept histogram stops buying the prose drafts and drain pacing stops the copy lanes once they are ahead of the
// prose lanes, so draft=on costs about the rows of draft=off and never more steps; a copy-only cohort keeps its drafts.
struct DrainRun {
    std::uint64_t forwards = 0, rows = 0, drafted = 0, prose_drafted = 0;
};

DrainRun run_drain(bool draft, std::uint32_t copy_lanes, std::uint32_t prose_lanes) {
    const std::uint32_t vocab = 23, lanes = copy_lanes + prose_lanes;
    FakeBackend be(lanes, 32, vocab);
    CohortConfig cfg;
    cfg.context_draft = draft;
    cfg.prompt_slice = 64;
    cfg.prior = {13.3, 0.113, 0.13, 0.0};  // the RTX C8 decode prior; host timings would make the plan noisy
    cfg.learn_cost = false;
    CohortDecoder dec(be, cfg);
    std::vector<CohortRequest> reqs;
    for (std::uint32_t i = 0; i < lanes; ++i) {
        CohortRequest r;
        r.id = 200 + i;
        const bool prose = i >= copy_lanes;
        if (prose) r.prompt.push_back(vocab + 1U);
        for (std::uint32_t k = 0; k < 20; ++k) r.prompt.push_back((i * 13U + k * 5U) % vocab);
        r.max_tokens = 120;
        reqs.push_back(r);
        dec.submit(r);
    }
    dec.run();
    DrainRun out;
    for (const auto& r : dec.take_finished()) {
        const auto& q = reqs.at(r.id - 200);
        check(r.error.empty() && r.ids == solo(q, vocab), "drain: ids == solo for request " + std::to_string(r.id));
        out.drafted += r.drafted;
        if (r.id - 200 >= copy_lanes) out.prose_drafted += r.drafted;
    }
    out.forwards = be.forwards;
    out.rows = dec.physical_rows();
    return out;
}

void test_mixed_drain() {
    const DrainRun off = run_drain(false, 2, 2), on = run_drain(true, 2, 2);
    std::cout << "mixed drain: off forwards " << off.forwards << " rows " << off.rows << "; on forwards " << on.forwards
              << " rows " << on.rows << " drafted " << on.drafted << " (prose " << on.prose_drafted << ")\n";
    check(on.forwards <= off.forwards, "mixed drain: draft=on never takes more steps");
    // before the kept histogram and drain pacing: 212 drafted rows (183 on the copy lanes), rows +8.5 %, same steps
    check(on.rows * 100 <= off.rows * 105, "mixed drain: draft=on buys at most 5 % more rows than draft=off");
    check(on.drafted <= 24U && on.drafted - on.prose_drafted <= 4U,
          "mixed drain: the copy lanes stop once ahead, the prose lanes stop after their rejections");
    const DrainRun copy_off = run_drain(false, 4, 0), copy_on = run_drain(true, 4, 0);
    std::cout << "copy drain: off forwards " << copy_off.forwards << "; on forwards " << copy_on.forwards << " drafted "
              << copy_on.drafted << "\n";
    check(copy_on.forwards * 3 < copy_off.forwards, "copy drain: drafts keep their gain (36 of 124 steps before)");
}

void test_owner_callbacks_and_boundary() {
    FakeBackend be(2, 2, 13, 32);
    CohortConfig cfg;
    cfg.context_draft = false;
    cfg.prompt_slice = 2;
    cfg.learn_cost = false;
    CohortDecoder decoder(be, cfg);
    CohortRequest cancelled;
    cancelled.id = 901;
    cancelled.prompt = {1, 2, 3};
    cancelled.max_tokens = 4;
    cancelled.cancelled = [] { return true; };
    decoder.submit(cancelled);

    CohortRequest streaming;
    streaming.id = 902;
    streaming.prompt = {2, 3, 4, 5, 6};
    streaming.max_tokens = 8;
    std::vector<std::uint64_t> progress;
    std::vector<std::uint32_t> published;
    streaming.prefilled = [&](std::uint64_t rows) { progress.push_back(rows); };
    streaming.publish = [&](std::span<const std::uint32_t> ids) {
        published.insert(published.end(), ids.begin(), ids.end());
        return false;
    };
    decoder.submit(streaming);
    decoder.run();
    const auto finished = decoder.take_finished();
    check(finished.size() == 2U, "both cancelled requests retire");
    for (const auto& result : finished) {
        check(result.cancelled && result.error.empty(), "callback cancellation is distinct from a backend error");
        if (result.id == cancelled.id) check(result.ids.empty(), "pre-step cancellation emits no tokens");
        else {
            check(result.ids == published && result.ids.size() == 1U, "publish=false retires after the committed prefix");
            check(result.ids.front() == solo(streaming, 13).front(), "published prefix equals solo generation");
        }
    }
    check(progress == std::vector<std::uint64_t>{2U, 4U}, "prefill callback reports cumulative completed prompt slices");
    check(be.releases == 2U && be.forwards == 1U, "cancelled lanes release and do not keep consuming forwards");

    CohortRequest boundary;
    boundary.id = 903;
    boundary.prompt = {1, 2, 3, 4, 5};
    boundary.max_tokens = 27;
    decoder.submit(boundary);
    auto too_long = boundary;
    too_long.id = 904;
    too_long.max_tokens = 28;
    decoder.submit(too_long);
    auto overflow = boundary;
    overflow.id = 905;
    overflow.max_tokens = UINT32_MAX;
    decoder.submit(overflow);
    decoder.run();
    const auto results = decoder.take_finished();
    check(results.size() == 3U, "boundary requests all finish or refuse");
    for (const auto& result : results) {
        if (result.id == boundary.id)
            check(result.error.empty() && result.ids == solo(boundary, 13), "exact context boundary reuses a retired lane");
        else check(!result.error.empty() && result.ids.empty(), "over-capacity requests refuse before generation");
    }
}

}  // namespace

// Bounded slices keep the bits (CohortConfig::prompt_slice_decoding). PieceBackend chunks each prefill call like a
// resident decoder: chunks of kChunk rows, a remainder of up to kTail rows rides in the final chunk when it is longer
// than kShort rows (the key-split bound); a row's bits are its route form: short chunk or not, and its kAlign-row wave
// tile [start, end) counted from its chunk start. For every prompt length and several patterns of a decoding lane
// (decoding throughout, finishing mid-prompt, arriving mid-prompt, absent), the served prompt rows take the forms of
// ONE prefill call over the whole prompt (the CLI), the ids equal the solo run, and while another lane is live no call
// runs more than one decoding slice (+ a short remainder) nor more than ONE backend pass (chunk).
struct RowForm {
    bool short_chunk = false;
    std::uint64_t tile_start = 0, tile_end = 0;
    bool operator==(const RowForm&) const = default;
};
class PieceBackend final : public CohortBackend {
public:
    static constexpr std::uint32_t kChunk = 64, kTail = 8, kShort = 4, kAlign = 4, kVocab = 23, kMarker = 100;
    std::uint32_t lanes() const override { return 2; }
    std::uint32_t rows_per_forward() const override { return 2; }
    std::uint32_t max_positions() const override { return 4096; }
    bool prefill_emits_first() const override { return true; }
    std::uint32_t prompt_tail_rows() const override { return kTail; }
    PromptPieceRule prompt_piece_rule() const override { return {kAlign, kShort}; }
    std::uint64_t prompt_call_chunk(std::uint64_t rows) const override {
        if (slice_chunk != 0U && rows == kChunk) return slice_chunk;   // contract breach: the slice is not one chunk
        if (odd_chunk != 0U && rows > kChunk) return odd_chunk;        // contract breach: a call of three chunks
        return chunks(rows).front();
    }
    std::uint64_t slice_chunk = 0, odd_chunk = 0;
    // The decoder's chunks of one call of `rows` rows.
    static std::vector<std::uint64_t> chunks(std::uint64_t rows) {
        const std::uint64_t r = rows % kChunk;
        const bool absorb = rows > kChunk && r > kShort && r <= kTail;
        std::vector<std::uint64_t> out;
        for (std::uint64_t left = rows; left != 0;) {
            const std::uint64_t take = absorb && left == kChunk + r ? left : std::min<std::uint64_t>(left, kChunk);
            out.push_back(take);
            left -= take;
        }
        return out;
    }
    static void record(std::vector<RowForm>& forms, std::uint64_t at, std::uint64_t rows) {
        for (const auto c : chunks(rows)) {
            for (std::uint64_t p = at; p < at + c; ++p) {
                const std::uint64_t tile = at + (p - at) / kAlign * kAlign;
                forms.at(p) = {c <= kShort, tile, std::min<std::uint64_t>(tile + kAlign, at + c)};
            }
            at += c;
        }
    }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        auto& c = ctx_.at(lane);
        if (first) {
            c.clear();
            live_.at(lane) = true;
            watched_.at(lane) = tokens.front() != kMarker;
        }
        if (watched_.at(lane)) {
            if (first) forms.assign(prompt_rows, RowForm{});
            record(forms, c.size(), tokens.size());
            if (live_.at(1U - lane)) {
                busy_calls.push_back(tokens.size());
                max_busy_call = std::max<std::uint64_t>(max_busy_call, tokens.size());
                max_busy_passes = std::max<std::uint64_t>(max_busy_passes, chunks(tokens.size()).size());
            }
            ++calls;
            trace += 'P';
        }
        c.insert(c.end(), tokens.begin(), tokens.end());
        top_ = next_id(c, kVocab);
        if (clock)  // modelled cost: every backend pass pays the fixed cost, every row its row cost
            clock->advance_fake_clock(pass_ms * static_cast<double>(chunks(tokens.size()).size()) +
                                      row_ms * static_cast<double>(tokens.size()));
    }
    // Controlled clock (optional): the decoder whose fake clock the modelled pass and decode costs advance.
    CohortDecoder* clock = nullptr;
    double pass_ms = 0.0, row_ms = 0.0, decode_ms = 0.0;
    std::uint32_t prefill_top(std::uint32_t) override { return top_; }
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        window_.clear();
        for (std::size_t g = 0; g < groups.size(); ++g) {
            check(groups[g].rows == 1U, "piece backend: one row per lane");
            if (!watched_.at(groups[g].lane)) trace += 'D';  // a decode round of the other lane
            auto c = ctx_.at(groups[g].lane);
            c.push_back(tokens[g]);
            top[g] = next_id(c, kVocab);
            window_.emplace_back(groups[g].lane, tokens[g]);
        }
        if (clock) clock->advance_fake_clock(decode_ms);
    }
    void commit(std::span<const std::uint32_t>) override {
        for (const auto& [lane, token] : window_) ctx_.at(lane).push_back(token);
        window_.clear();
    }
    void release(std::uint32_t lane) override { live_.at(lane) = false; }

    std::uint64_t prompt_rows = 0, max_busy_call = 0, max_busy_passes = 0, calls = 0;
    std::vector<std::uint64_t> busy_calls;  // rows of each watched prefill call beside a live lane, in order
    std::vector<RowForm> forms;
    std::string trace;  // P = a prefill call of the watched prompt, D = a decode round of the other lane

private:
    std::vector<std::vector<std::uint32_t>> ctx_ = std::vector<std::vector<std::uint32_t>>(2);
    std::vector<bool> live_ = std::vector<bool>(2, false), watched_ = std::vector<bool>(2, false);  // watched: lane B
    std::vector<std::pair<std::uint32_t, std::uint32_t>> window_;
    std::uint32_t top_ = 0;
};

void test_prompt_piece_rows() {
    const PromptPieceRule rule{16, 16};
    check(prompt_piece_rows(8192, 1024, 8192, rule) == 1024, "piece: a whole aligned step");
    check(prompt_piece_rows(1029, 1024, 8192, rule) == 1029, "piece: a 5-row remainder rides in the piece");
    check(prompt_piece_rows(1041, 1024, 8192, rule) == 1024, "piece: a 17-row remainder is its own piece");
    check(prompt_piece_rows(1000, 1024, 8192, rule) == 1000, "piece: the chunk's rest fits the bound");
    check(prompt_piece_rows(8196, 8192, 8192, rule) == 8176, "piece: no ride above the call cap, shrink one tile");
    check(prompt_piece_rows(1030, 1030, 8192, rule) == 1030 && prompt_piece_rows(1100, 1030, 8192, rule) == 1024,
          "piece: the step is a multiple of the alignment");
    check(prompt_piece_rows(500, 100, 8192, PromptPieceRule{}) == 500, "piece: an uncuttable chunk runs whole");
}

// A priced grant never dispatches more rows than the grant when ANY bit-safe cut <= grant exists; only when none
// exists the smallest bit-safe piece runs, flagged. Exhaustive over chunks of 1..256 rows x every grant x declared cut
// rules (align x short). `piece(chunk_left, grant, rule)` -> {rows, over}; returns the first violation (empty = holds).
template <class Piece>
std::string priced_piece_violation(Piece piece) {
    for (const std::uint32_t align : {0U, 1U, 2U, 4U, 8U, 16U})
        for (const std::uint32_t shorter : {0U, 1U, 3U, 4U, 8U, 16U}) {
            const PromptPieceRule rule{align, shorter};
            for (std::uint64_t left = 1; left <= 256U; ++left) {
                // the bit-safe pieces: the whole rest, or an aligned cut longer than short leaving more than short
                std::vector<std::uint64_t> safe;
                for (std::uint64_t p = 1; p < left; ++p)
                    if (align != 0U && p % align == 0U && p > shorter && left - p > shorter) safe.push_back(p);
                safe.push_back(left);
                std::size_t fit = 0;  // safe[0, fit) <= grant
                for (std::uint64_t grant = 1; grant <= left + 2U; ++grant) {
                    while (fit < safe.size() && safe[fit] <= grant) ++fit;
                    const std::uint64_t expect = fit != 0U ? safe[fit - 1U] : safe.front();
                    const PromptPieceChoice got = piece(left, grant, rule);
                    if (got.rows != expect || got.over != (fit == 0U))
                        return "chunk " + std::to_string(left) + " grant " + std::to_string(grant) + " align " +
                               std::to_string(align) + " short " + std::to_string(shorter) + ": " +
                               std::to_string(got.rows) + (got.over ? " (over)" : "") + ", expected " +
                               std::to_string(expect) + (fit == 0U ? " (over)" : "");
                }
            }
        }
    return {};
}

void test_priced_piece_property() {
    const std::string bad = priced_piece_violation(prompt_piece_within);
    check(bad.empty(), "priced piece: the largest bit-safe cut within the grant (" + bad + ")");
    static_assert(prompt_piece_within(64, 62, {4, 4}).rows == 56 && !prompt_piece_within(64, 62, {4, 4}).over);
    static_assert(prompt_piece_within(9216, 8192, {16, 16}).rows == 8192);  // an absorbed tail is cut, not ridden
    static_assert(prompt_piece_within(20, 12, {4, 4}).rows == 12 && prompt_piece_within(12, 7, {4, 4}).rows == 12 &&
                  prompt_piece_within(12, 7, {4, 4}).over);
    // The rule of f3a30e230 (prompt_piece_rows under the grant, the remainder riding to the call cap) breaks it:
    // grant 62 of a 64-row chunk dispatched 64 although 56 + 8 fits.
    const std::string old = priced_piece_violation([](std::uint64_t left, std::uint64_t grant, PromptPieceRule rule) {
        const std::uint64_t rows = prompt_piece_rows(left, grant, 256U, rule);
        return PromptPieceChoice{rows, rows > grant};
    });
    check(!old.empty(), "priced piece: the property rejects the old ride-to-cap rule");
    std::cout << "priced piece: exhaustive property (chunks <= 256 x grants x cut rules)\n";
}

// A priced grant bounds the actual piece beside a decoding lane (controlled clock: pass 20 ms, 1 ms/row, decode 18 ms,
// target 100.5 ms -> a grant of 62 rows once a floor-sized pass priced a row). Geometry 1: a 64-row chunk under 62 ->
// 56 + 8 (f3a30e230 dispatched 64, priced 102 ms). Geometry 2 (target 1000 ms, grant = the 64-row cap): the absorbed
// final chunk 64 + 6 -> 64 + 6 (f3a30e230: 70 through the region-start return). Rows keep the CLI's forms, the ids
// equal the solo run and no overshoot is recorded.
void test_priced_grant_geometries() {
    using B = PieceBackend;
    struct Geometry { std::uint32_t prompt; double gap; std::vector<std::uint64_t> calls; };
    for (const Geometry& g : {Geometry{300, 100.5, {16, 48, 56, 8, 56, 8, 56, 8, 44}},
                              Geometry{134, 1000.0, {16, 48, 64, 6}}}) {
        const std::string at = " (prompt " + std::to_string(g.prompt) + ")";
        B be;
        be.prompt_rows = g.prompt;
        be.pass_ms = 20.0;
        be.row_ms = 1.0;
        be.decode_ms = 18.0;
        CohortConfig cfg;
        cfg.context_draft = false;
        cfg.prompt_slice = B::kChunk;
        cfg.prompt_slice_decoding = 16;
        cfg.prompt_gap_ms = g.gap;
        cfg.prompt_pass_fixed_ms = 20.0;
        cfg.prompt_share = 0.0;
        CohortDecoder dec(be, cfg);
        dec.set_fake_clock(1e-6);
        be.clock = &dec;
        CohortRequest a, b;
        a.id = 1;
        a.prompt = {B::kMarker, 4};
        a.max_tokens = 2000;
        b.id = 2;
        b.prompt.resize(g.prompt);
        for (std::uint32_t k = 0; k < g.prompt; ++k) b.prompt[k] = (k * 7U + 1U) % B::kVocab;
        b.max_tokens = 2;
        dec.submit(a);
        dec.step();
        dec.submit(b);
        bool done = false;
        for (int k = 0; k < 4000 && !done; ++k) {
            dec.step();
            for (const auto& r : dec.take_finished())
                if (r.id == 2) {
                    done = true;
                    check(r.error.empty() && r.ids == solo(b, B::kVocab), "priced grant: solo ids" + at);
                }
        }
        check(done, "priced grant: the prompt finishes beside a decoding lane" + at);
        check(be.busy_calls == g.calls, "priced grant: no piece above its grant when a bit-safe cut fits" + at);
        std::vector<RowForm> cli(g.prompt);
        B::record(cli, 0, g.prompt);
        check(be.forms == cli, "priced grant: the rows keep the forms of one call over the prompt" + at);
        check(dec.gap_overshoots() == 0U, "priced grant: no overshoot recorded when a cut fits" + at);
        dec.run();
    }
    std::cout << "priced grant: 64-row chunk under 62 -> 56 + 8, absorbed 70 under the 64 cap -> 64 + 6\n";
}

// The row cost is learned from calls of ONE backend pass: lane A's 68-row prompt runs alone as one call of two passes
// (64 + a 4-row short chunk, two fixed costs: 108 ms) and would price a row at 1.29 ms, so B's first piece beside A
// would be 48 rows instead of the floor (no one-pass sample yet). f3a30e230 learned from it.
void test_gap_row_cost_one_pass() {
    using B = PieceBackend;
    B be;
    be.prompt_rows = 300;
    be.pass_ms = 20.0;
    be.row_ms = 1.0;
    be.decode_ms = 18.0;
    CohortConfig cfg;
    cfg.context_draft = false;
    cfg.prompt_slice = B::kChunk;
    cfg.prompt_slice_decoding = 16;
    cfg.prompt_gap_ms = 100.5;
    cfg.prompt_pass_fixed_ms = 20.0;
    cfg.prompt_share = 0.0;
    CohortDecoder dec(be, cfg);
    dec.set_fake_clock(1e-6);
    be.clock = &dec;
    CohortRequest a, b;
    a.id = 1;
    a.prompt.assign(68, 3U);
    a.prompt.front() = B::kMarker;
    a.max_tokens = 2000;
    b.id = 2;
    b.prompt.resize(300);
    for (std::uint32_t k = 0; k < 300; ++k) b.prompt[k] = (k * 7U + 1U) % B::kVocab;
    b.max_tokens = 2;
    dec.submit(a);
    dec.step();
    dec.submit(b);
    for (int k = 0; k < 4000 && be.busy_calls.empty(); ++k) dec.step();
    check(!be.busy_calls.empty() && be.busy_calls.front() == 16U,
          "gap row cost: a call of two backend passes does not price the next piece");
    dec.run();
    std::cout << "gap row cost: learned from one-pass calls only\n";
}

// Grouped company under a priced gap (review r17 P2): floor 32, slice 64, pass 20 ms, 1 ms/row, decode 10 ms, target
// 46 ms -> 16 rows fit, the floored grant is 32. A 12-row prompt alone costs 42 ms; a second 12-row prompt joining it
// costs 54 ms. Company is capped by the unfloored room, so both run alone and nothing overshoots; with a 100 ms target
// the same two prompts share a pass (the fixture reaches the group path).
class GroupGapBackend final : public CohortBackend {
public:
    static constexpr std::uint32_t kVocab = 29;
    std::uint32_t lanes() const override { return 3; }
    std::uint32_t rows_per_forward() const override { return 3; }
    std::uint32_t max_positions() const override { return 4096; }
    bool prefill_emits_first() const override { return true; }
    PromptPieceRule prompt_piece_rule() const override { return {4, 4}; }
    PrefillGroupCapacity prompt_group_capacity() const override { return {}; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) ctx_.at(lane).clear();
        ctx_.at(lane).insert(ctx_.at(lane).end(), tokens.begin(), tokens.end());
        clock->advance_fake_clock(20.0 + static_cast<double>(tokens.size()));
    }
    std::uint32_t prefill_top(std::uint32_t lane) override { return next_id(ctx_.at(lane), kVocab); }
    void prefill_group(std::span<const CohortPrompt> members) override {
        std::uint64_t rows = 0;
        for (const auto& m : members) {
            ctx_.at(m.lane).assign(m.tokens.begin(), m.tokens.end());
            rows += m.tokens.size();
        }
        group_rows.push_back(rows);
        clock->advance_fake_clock(20.0 + static_cast<double>(rows));
    }
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        window_.clear();
        for (std::size_t g = 0; g < groups.size(); ++g) {
            auto c = ctx_.at(groups[g].lane);
            c.push_back(tokens[g]);
            top[g] = next_id(c, kVocab);
            window_.emplace_back(groups[g].lane, tokens[g]);
        }
        clock->advance_fake_clock(10.0);
    }
    void commit(std::span<const std::uint32_t>) override {
        for (const auto& [lane, token] : window_) ctx_.at(lane).push_back(token);
        window_.clear();
    }
    void release(std::uint32_t) override {}
    CohortDecoder* clock = nullptr;
    std::vector<std::uint64_t> group_rows;  // rows of each group pass

private:
    std::vector<std::vector<std::uint32_t>> ctx_ = std::vector<std::vector<std::uint32_t>>(3);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> window_;
};

void test_group_gap_room() {
    for (const double gap : {46.0, 100.0}) {
        const std::string at = " (target " + std::to_string(gap) + " ms)";
        GroupGapBackend be;
        CohortConfig cfg;
        cfg.context_draft = false;
        cfg.prompt_slice = 64;
        cfg.prompt_slice_decoding = 32;
        cfg.prompt_gap_ms = gap;
        cfg.prompt_pass_fixed_ms = 20.0;
        cfg.prompt_share = 0.0;
        CohortDecoder dec(be, cfg);
        dec.set_fake_clock(1e-6);
        be.clock = &dec;
        CohortRequest a, b, c;
        a.id = 1;  // the decoding lane: its 40-row prompt (one pass, >= the floor) prices a row at 1 ms
        a.prompt.assign(40, 5U);
        a.max_tokens = 2000;
        b.id = 2;
        c.id = 3;
        for (std::uint32_t k = 0; k < 12; ++k) {
            b.prompt.push_back((k * 3U + 1U) % GroupGapBackend::kVocab);
            c.prompt.push_back((k * 5U + 2U) % GroupGapBackend::kVocab);
        }
        b.max_tokens = c.max_tokens = 2;
        dec.submit(a);
        for (int k = 0; k < 3; ++k) dec.step();  // A's prompt, then decode rounds price the round
        dec.submit(b);
        dec.submit(c);
        int done = 0;
        for (int k = 0; k < 400 && done < 2; ++k) {
            dec.step();
            for (const auto& r : dec.take_finished()) {
                ++done;
                check(r.error.empty() && r.ids == solo(r.id == 2 ? b : c, GroupGapBackend::kVocab), "group gap: solo ids" + at);
            }
        }
        check(done == 2, "group gap: both short prompts finish beside a decoding lane" + at);
        if (gap < 50.0)
            check(be.group_rows.empty() && dec.gap_overshoots() == 0U,
                  "group gap: company never takes a pass past the unfloored room (12 + 12 rows = 54 ms > 46 ms)");
        else
            check(be.group_rows == std::vector<std::uint64_t>{24U} && dec.gap_overshoots() == 0U,
                  "group gap: company that fits shares the pass");
        dec.run();
    }
    std::cout << "group gap: company capped by the unfloored room\n";
}

// Cold start under a priced gap (review r19 P2): the decoding lane's 8-row prompt is below the 32-row sampling floor,
// so no row cost is learned. Fixed pass 20 ms + decode round 10 ms = 30 ms already exceed a 25 ms target: a lone
// 12-row arrival runs (the head pass is necessary) and is recorded as an overshoot; two 12-row arrivals get no company
// and both head passes are recorded. With a 100 ms target the same lower bound fits: company joins, nothing overshoots.
void test_group_gap_cold() {
    for (const double gap : {25.0, 100.0})
        for (const bool pair : {false, true}) {
            const std::string at = " (target " + std::to_string(gap) + " ms, " + (pair ? "two" : "one") + " arrival)";
            GroupGapBackend be;
            CohortConfig cfg;
            cfg.context_draft = false;
            cfg.prompt_slice = 64;
            cfg.prompt_slice_decoding = 32;
            cfg.prompt_gap_ms = gap;
            cfg.prompt_pass_fixed_ms = 20.0;
            cfg.prompt_share = 0.0;
            CohortDecoder dec(be, cfg);
            dec.set_fake_clock(1e-6);
            be.clock = &dec;
            CohortRequest a, b, c;
            a.id = 1;
            a.prompt.assign(8, 5U);  // below the floor: no row-cost sample
            a.max_tokens = 2000;
            b.id = 2;
            c.id = 3;
            for (std::uint32_t k = 0; k < 12; ++k) {
                b.prompt.push_back((k * 3U + 1U) % GroupGapBackend::kVocab);
                c.prompt.push_back((k * 5U + 2U) % GroupGapBackend::kVocab);
            }
            b.max_tokens = c.max_tokens = 2;
            dec.submit(a);
            for (int k = 0; k < 3; ++k) dec.step();  // A's prompt, then decode rounds price the round
            check(dec.gap_overshoots() == 0U, "cold gap: nothing recorded before a prompt runs beside a decoding lane" + at);
            dec.submit(b);
            if (pair) dec.submit(c);
            const int want = pair ? 2 : 1;
            int done = 0;
            for (int k = 0; k < 400 && done < want; ++k) {
                dec.step();
                for (const auto& r : dec.take_finished()) {
                    ++done;
                    check(r.error.empty() && r.ids == solo(r.id == 2 ? b : c, GroupGapBackend::kVocab),
                          "cold gap: solo ids" + at);
                }
            }
            check(done == want, "cold gap: the short prompts finish beside a decoding lane" + at);
            if (gap < 30.0) {
                check(be.group_rows.empty(), "cold gap: no company when fixed + decode exceed the target" + at);
                check(dec.gap_overshoots() == static_cast<std::uint64_t>(want) && dec.gap_overshoot_ms() >= 4.99,
                      "cold gap: every necessary head pass is recorded (lower bound 30 ms > 25 ms)" + at);
            } else {
                check(be.group_rows == (pair ? std::vector<std::uint64_t>{24U} : std::vector<std::uint64_t>{}),
                      "cold gap: company joins when the lower bound fits" + at);
                check(dec.gap_overshoots() == 0U, "cold gap: no overshoot under a fitting lower bound" + at);
            }
            dec.run();
        }
    // The planner's no-unit gap selector with a row cost too small to move the price returns at once (review r19 P2).
    StepPlanner planner;
    StepLimits limits;
    limits.max_decode_gap_ms = 1.0;
    limits.prompt_pass_fixed_ms = 1.0;
    limits.prompt_row_ms = 1e-30;
    StepLane d, p;
    d.lane = 0;
    p.lane = 1;
    p.prompt = true;
    p.prompt_remaining = 1ULL << 40;  // above the 32-bit unit cap: the grant itself is seen
    const StepLane lanes[2] = {d, p};
    const StepPlan plan = planner.plan(lanes, limits);
    check(plan.prompt.size() == 1U && plan.prompt.front().rows == UINT32_MAX,
          "cold gap: planner selector at a 1e-30 ms row cost takes the cap without walking it");
    std::cout << "cold gap: fixed + decode bound the first passes\n";
}

// The grant is the largest r whose FORWARD price fits (review r17 P3): an exact-target quotient that truncates below
// (120 rows at 0.009 ms -> 119.99999999999999) and a quotient whose forward price rounds above the target (72 rows at
// 0.001 ms against 0.072 ms) are corrected. Brute force over row costs x targets x fixed / decode costs.
void test_gap_forward_price() {
    const double t120 = 120.0 * 0.009;
    check(gap_bounded_rows(t120, 0.0, 0.009, 0.0, 0U, 160U) == 120U, "gap price: 120 rows priced at the target fit");
    check(prompt_piece_within(160U, gap_bounded_rows(t120, 0.0, 0.009, 0.0, 0U, 160U), {8, 4}).rows == 120U,
          "gap price: the 160-row chunk (align 8, short 4) dispatches 120, not 112");
    const std::uint64_t r72 = gap_bounded_rows(0.072, 0.0, 0.001, 0.0, 0U, 1000U);
    check(gap_pass_ms(0.0, 0.001, 0.0, r72) <= 0.072 && gap_pass_ms(0.0, 0.001, 0.0, r72 + 1U) > 0.072,
          "gap price: the grant's forward price fits and one more row does not");
    std::uint64_t cases = 0;
    for (const double row : {0.001, 0.009, 0.1, 0.3, 1.0 / 3.0, 0.7, 1.0, 7.0})
        for (const double fixed : {0.0, 20.0, 240.0})
            for (const double decode : {0.0, 10.0, 0.37})
                for (std::uint64_t k = 0; k <= 300U; ++k) {
                    const double gap = fixed + row * static_cast<double>(k) + decode;  // a target priced at k rows
                    for (const double target : {gap, std::nextafter(gap, 0.0), std::nextafter(gap, 1e300)}) {
                        const std::uint64_t floor = 4, cap = 256;
                        std::uint64_t expect = floor;
                        for (std::uint64_t r = floor + 1U; r <= cap; ++r)
                            if (gap_pass_ms(fixed, row, decode, r) <= target) expect = r;
                        const std::uint64_t got = gap_bounded_rows(target, fixed, row, decode, floor, cap);
                        ++cases;
                        if (got != expect) {
                            check(false, "gap price: row " + std::to_string(row) + " fixed " + std::to_string(fixed) +
                                             " decode " + std::to_string(decode) + " k " + std::to_string(k) + ": " +
                                             std::to_string(got) + ", expected " + std::to_string(expect));
                            return;
                        }
                    }
                }
    std::cout << "gap price: forward-consistent grant (" << cases << " cases)\n";
}

void test_gap_bounded_rows() {
    static_assert(gap_bounded_rows(1000.0, 240.0, 0.5, 60.0, 512, 8192) == 1400);  // (1000 - 240 - 60) / 0.5
    static_assert(gap_bounded_rows(1000.0, 240.0, 0.5, 600.0, 512, 8192) == 512);  // a slow decode round: the floor
    static_assert(gap_bounded_rows(10000.0, 240.0, 0.5, 60.0, 512, 8192) == 8192); // the unbounded call caps it
    static_assert(gap_bounded_rows(1000.0, 240.0, 0.0, 60.0, 512, 8192) == 512);   // nothing observed yet: the floor
    static_assert(gap_bounded_rows(0.0, 240.0, 0.5, 60.0, 512, 8192) == 512);      // no target: the floor
    // Evaluation cost at the extreme (review r19): a legal row cost too small to move the price (gap = fixed = 1 ms,
    // 1e-30 ms per row) must not walk the cap row by row; constant evaluation would exceed the compiler's loop limit.
    static_assert(gap_bounded_rows(1.0, 1.0, 1e-30, 0.0, 1U, UINT32_MAX) == UINT32_MAX);
    static_assert(gap_bounded_rows(1.0, 0.5, 0x1p-32, 0.0, 1U, UINT32_MAX) == 0x80000000ULL);  // exact: 0.5 + 2^31 x 2^-32
    constexpr std::uint64_t kU64 = std::numeric_limits<std::uint64_t>::max();
    constexpr std::uint64_t kTiny = gap_bounded_rows(1.0, 1.0, 1e-30, 0.0, 1U, kU64);
    static_assert(kTiny > 1U && kTiny < kU64 && gap_pass_ms(1.0, 1e-30, 0.0, kTiny) <= 1.0 &&
                  gap_pass_ms(1.0, 1e-30, 0.0, kTiny + 1U) > 1.0);  // the largest forward fit below a 64-bit cap
    check(gap_bounded_rows(1000.0, 0.0, 1.0, 0.0, 16, 64) == 64, "gap piece: within the cap");
    // The planner's max_decode_gap_ms selector uses the same calculation; with no pass or decode cost it is the old
    // floor(gap / row_ms) unit, at least one row.
    for (const double row : {0.3, 0.5, 1.0, 7.0})
        for (const double gap : {0.1, 1.0, 10.0, 100.0}) {
            StepPlanner planner;
            StepLimits limits;
            limits.max_decode_gap_ms = gap;
            limits.prompt_row_ms = row;
            StepLane d, p;
            d.lane = 0;
            p.lane = 1;
            p.prompt = true;
            p.prompt_remaining = 1000000;
            const StepLane lanes[2] = {d, p};
            const StepPlan plan = planner.plan(lanes, limits);
            const auto old_rows = static_cast<std::uint32_t>(std::max(1.0, std::floor(gap / row)));
            check(plan.prompt.size() == 1U && plan.prompt.front().rows == old_rows,
                  "planner gap selector: no pass cost == floor(gap / row)");
            limits.prompt_pass_fixed_ms = gap / 2.0;
            limits.decode_round_ms = gap / 4.0;
            StepPlanner priced;
            const StepPlan cut = priced.plan(lanes, limits);
            check(cut.prompt.size() == 1U &&
                      cut.prompt.front().rows == gap_bounded_rows(gap, gap / 2.0, row, gap / 4.0, 1U, UINT32_MAX),
                  "planner gap selector: the pass and decode costs come out of the gap");
        }
}

// Mixed load with a positive prompt share (CohortConfig::prompt_share): lane A decodes for longer than B's prompt takes.
// Controlled clock (1 ms per read: every prefill call and every decode round costs 1 ms). After each piece B owes A
// share x 1 ms of decode: the next piece runs only once the debt is repaid (debt <= 0 when it starts), A decodes
// ceil(share) rounds between two pieces, the debt stays bounded, and B finishes with its solo ids. A target shorter than
// one decode round leaves the floor as the only choice: every piece is the floor and is recorded as an overshoot.
void test_gap_debt() {
    using B = PieceBackend;
    struct Arm { double gap, share; };
    for (const Arm arm : {Arm{3.0, 1.0}, Arm{3.0, 2.0}, Arm{0.0, 1.0}, Arm{0.5, 0.0}, Arm{0.5, 1.0}}) {
        const std::string at = " gap " + std::to_string(arm.gap) + " share " + std::to_string(arm.share);
        B be;
        be.prompt_rows = 300;
        CohortConfig cfg;
        cfg.context_draft = false;
        cfg.prompt_slice = B::kChunk;
        cfg.prompt_slice_decoding = 16;
        cfg.prompt_gap_ms = arm.gap;
        cfg.prompt_share = arm.share;
        CohortDecoder dec(be, cfg);
        dec.set_fake_clock(1.0);
        CohortRequest a, b;
        a.id = 1;
        a.prompt = {B::kMarker, 4};
        a.max_tokens = 2000;
        b.id = 2;
        b.prompt.resize(300);
        for (std::uint32_t k = 0; k < 300; ++k) b.prompt[k] = (k * 7U + 1U) % B::kVocab;
        b.max_tokens = 2;
        dec.submit(a);
        dec.step();
        dec.submit(b);
        bool b_done = false;
        double max_debt = 0.0;
        bool repaid = true;
        for (int k = 0; k < 20000 && !b_done; ++k) {
            const double debt = dec.prompt_debt_ms();
            const std::size_t pieces = std::count(be.trace.begin(), be.trace.end(), 'P');
            dec.step();
            if (pieces != 0U && std::count(be.trace.begin(), be.trace.end(), 'P') > static_cast<std::ptrdiff_t>(pieces) &&
                debt > 1e-9)
                repaid = false;  // a piece ran while decode time was still owed
            max_debt = std::max(max_debt, dec.prompt_debt_ms());
            for (const auto& r : dec.take_finished())
                if (r.id == 2) {
                    b_done = true;
                    check(r.error.empty() && r.ids == solo(b, B::kVocab), "gap debt: the prompt finishes with solo ids" + at);
                }
        }
        check(b_done, "gap debt: the prompt finishes beside a decoding lane" + at);
        check(repaid, "gap debt: a piece runs only after its debt is repaid" + at);
        check(max_debt <= arm.share * 1.0 + 1e-9, "gap debt: the debt stays one piece's share" + at);
        // decode rounds of A between two consecutive pieces of B
        const std::string& t = be.trace;
        std::size_t first = t.find('P'), last = t.rfind('P'), between_min = SIZE_MAX, between_max = 0;
        for (std::size_t i = first; i < last;) {
            const std::size_t next = t.find('P', i + 1);
            const auto rounds = static_cast<std::size_t>(std::count(t.begin() + static_cast<std::ptrdiff_t>(i),
                                                                    t.begin() + static_cast<std::ptrdiff_t>(next), 'D'));
            between_min = std::min(between_min, rounds);
            between_max = std::max(between_max, rounds);
            i = next;
        }
        const auto owed = static_cast<std::size_t>(std::ceil(arm.share));
        check(first != last && between_min >= std::max<std::size_t>(owed, 1U),
              "gap debt: the decoding lane advances between pieces" + at);
        check(between_max <= owed + 1U, "gap debt: no piece waits beyond its debt" + at);
        check(be.max_busy_passes <= 1U, "gap debt: one backend pass per piece" + at);
        if (arm.gap > 0.0 && arm.gap < 1.0) {  // a decode round (1 ms) is longer than the target: only the floor
            check(be.max_busy_call <= 16U + B::kShort, "gap debt: target below one decode round -> the floor" + at);
            check(dec.gap_overshoots() > 0U && dec.gap_overshoot_ms() > 0.5, "gap debt: the floor overshoot is recorded" + at);
        }
        dec.run();
    }
    std::cout << "gap debt: mixed load with a prompt share, pieces wait for their debt\n";
}

// The gap's row cost is learned from passes of at least the decoding floor. Fake clock (every pass 1 ms) with a fixed
// pass cost of 0.95 ms: lane A's own 2-row prompt would price a row at 0.025 ms and B's first piece at ~40 rows (the
// node case: a short lane prompt under-priced the first piece of an 8K prompt). The first piece beside A is the floor
// (no estimate yet); later pieces grow from the cost of floor-sized passes; B finishes with its solo ids.
void test_gap_row_cost_floor() {
    using B = PieceBackend;
    B be;
    be.prompt_rows = 300;
    CohortConfig cfg;
    cfg.context_draft = false;
    cfg.prompt_slice = B::kChunk;
    cfg.prompt_slice_decoding = 16;
    cfg.prompt_gap_ms = 3.0;
    cfg.prompt_pass_fixed_ms = 0.95;
    cfg.prompt_share = 0.0;
    CohortDecoder dec(be, cfg);
    dec.set_fake_clock(1.0);
    CohortRequest a, b;
    a.id = 1;
    a.prompt = {B::kMarker, 4};
    a.max_tokens = 400;
    b.id = 2;
    b.prompt.resize(300);
    for (std::uint32_t k = 0; k < 300; ++k) b.prompt[k] = (k * 7U + 1U) % B::kVocab;
    b.max_tokens = 2;
    dec.submit(a);
    dec.step();
    dec.submit(b);
    dec.run();
    for (const auto& r : dec.take_finished())
        if (r.id == 2) check(r.error.empty() && r.ids == solo(b, B::kVocab), "gap row cost: the prompt finishes with solo ids");
    check(!be.busy_calls.empty() && be.busy_calls.front() <= 16U + B::kShort,
          "gap row cost: a pass below the decoding floor does not price the next piece");
    check(be.max_busy_call > 16U + B::kShort, "gap row cost: later pieces grow from the cost of floor-sized passes");
    check(be.max_busy_passes <= 1U, "gap row cost: one backend pass per piece");
    std::cout << "gap row cost: learned from passes of at least the decoding floor\n";
}

void test_bounded_slices_keep_forms() {
    using B = PieceBackend;
    std::uint64_t cases = 0;
    // gap 0: fixed 16-row pieces; gap 3 (fake clock, 1 ms per pass and per decode round): each piece about doubles the
    // previous one up to the prompt slice, so the cut points move with the observed cost and the forms must still hold.
    for (const double gap : {0.0, 3.0})
    for (std::uint32_t prompt = 1; prompt <= 300; ++prompt) {
        const std::uint32_t busy = 16U;
        // pattern 0: A decodes throughout; 1: A finishes while B reads; 2: A arrives after B's first slice; 3: B alone
        for (int pattern = 0; pattern < 4; ++pattern) {
            B be;
            be.prompt_rows = prompt;
            CohortConfig cfg;
            cfg.context_draft = false;
            cfg.prompt_slice = B::kChunk;
            cfg.prompt_slice_decoding = busy;
            cfg.prompt_gap_ms = gap;
            cfg.prompt_share = 0.0;
            CohortDecoder dec(be, cfg);
            if (gap > 0.0) dec.set_fake_clock(1.0);
            CohortRequest a, b;
            a.id = 1;
            a.prompt = {B::kMarker, 4};
            a.max_tokens = pattern == 1 ? 3U : 400U;
            b.id = 2;
            b.prompt.resize(prompt);
            for (std::uint32_t k = 0; k < prompt; ++k) b.prompt[k] = (k * 7U + 1U) % B::kVocab;
            b.max_tokens = 2;
            if (pattern == 0 || pattern == 1) {
                dec.submit(a);
                dec.step();
            }
            dec.submit(b);
            if (pattern == 2) {
                dec.step();
                dec.submit(a);
            }
            dec.run();
            std::vector<RowForm> cli(prompt);
            B::record(cli, 0, prompt);
            const std::string at = " P=" + std::to_string(prompt) + " pattern " + std::to_string(pattern);
            check(be.forms == cli, "bounded slices: served row forms == one-call prefill" + at);
            for (const auto& r : dec.take_finished())
                check(r.error.empty() && r.ids == solo(r.id == 1 ? a : b, B::kVocab), "bounded slices: ids == solo" + at);
            const std::uint64_t call_cap = gap > 0.0 ? B::kChunk + B::kTail : busy + B::kShort;
            check(be.max_busy_call <= call_cap, "bounded slices: a call beside a live lane within one slice" + at);
            // the grant prices one pass: never a chunk plus a remainder the backend runs on its own (64 + 4)
            check(be.max_busy_passes <= 1U, "bounded slices: a call beside a live lane runs ONE backend pass" + at);
            if (gap > 0.0 && pattern == 0 && prompt >= 200)
                check(be.max_busy_call > busy + B::kShort, "gap pieces: a later piece grows past the first" + at);
            if (pattern == 3) {  // alone: exactly the unbounded calls (prompt_slice + the admitted tail per call)
                std::uint64_t unbounded = 0;
                for (std::uint64_t read = 0; read < prompt; ++unbounded) {
                    const std::uint64_t left = prompt - read;
                    read += prompt_slice_rows(std::min<std::uint64_t>(B::kChunk, left), B::kChunk, left, B::kTail);
                }
                check(be.calls == unbounded, "bounded slices: alone, the prompt runs the unbounded calls" + at);
            }
            ++cases;
        }
    }
    // Configuration: a backend without a piece rule refuses bounded slices; so does a slice the rule cannot cut.
    FakeBackend plain(2, 4, 23);
    CohortConfig bad;
    bad.prompt_slice = 64;
    bad.prompt_slice_decoding = 16;
    bool refused = false;
    try { CohortDecoder d(plain, bad); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "bounded slices: refused without the backend's piece rule");
    PieceBackend pb;
    bad.prompt_slice_decoding = 12;  // 12 <= 2 * kShort + kAlign
    refused = false;
    try { CohortDecoder d(pb, bad); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "bounded slices: refused below the piece rule's shortest cut");
    PieceBackend split_slice;
    split_slice.slice_chunk = 32;
    bad.prompt_slice_decoding = 16;
    refused = false;
    try { CohortDecoder d(split_slice, bad); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "bounded slices: refused when the prompt slice is not one backend chunk");
    {
        PieceBackend three;
        three.odd_chunk = 4;  // a 70-row call: chunks 4 + 66, and 66 is not one chunk
        three.prompt_rows = 70;
        CohortConfig cfg;
        cfg.context_draft = false;
        cfg.prompt_slice = PieceBackend::kChunk;
        cfg.prompt_slice_decoding = 16;
        cfg.prompt_share = 0.0;
        CohortDecoder dec(three, cfg);
        CohortRequest a, b;
        a.id = 1;
        a.prompt = {PieceBackend::kMarker, 4};
        a.max_tokens = 400;
        b.id = 2;
        b.prompt.assign(70, 3U);
        b.max_tokens = 2;
        dec.submit(a);
        dec.step();
        dec.submit(b);
        bool thrown = false;
        try { dec.run(); } catch (const std::logic_error&) { thrown = true; }
        check(thrown, "bounded slices: a call of more than two backend chunks is refused");
    }
    std::cout << "bounded slices: " << cases << " cases, served row forms == one-call prefill\n";
}

// The constructor's checks are one shared function a launcher can call before device work (validate_prompt_pieces).
void test_piece_validation() {
    const PromptPieceRule rule{16, 16};
    CohortConfig cfg;
    cfg.prompt_slice = 8192;
    cfg.prompt_slice_decoding = 512;
    cfg.prompt_gap_ms = 1000.0;
    cfg.prompt_share = 0.0;
    bool ok = true;
    try { validate_prompt_pieces(cfg, rule, 8192); } catch (const std::invalid_argument&) { ok = false; }
    check(ok, "validation: the gfx1151 schedule is admitted");
    const auto refused = [&](CohortConfig c, PromptPieceRule r, std::uint64_t chunk) {
        try { validate_prompt_pieces(c, r, chunk); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    CohortConfig c = cfg;
    c.prompt_slice_decoding = 32;  // 32 <= 2 x 16 + 16
    check(refused(c, rule, 8192), "validation: a decoding slice below the piece rule");
    c = cfg;
    c.prompt_slice_decoding = 8192;  // at the nominal chunk: no piece
    check(refused(c, rule, 8192), "validation: a decoding slice at the prompt slice");
    check(refused(cfg, PromptPieceRule{}, 8192), "validation: no piece rule");
    check(refused(cfg, rule, 4096), "validation: the prompt slice is not one backend chunk");
    c = cfg;
    c.prompt_slice_decoding = 0;
    check(refused(c, rule, 0), "validation: a gap without pieces");
    c = cfg;
    c.prompt_share = -1.0;
    check(refused(c, rule, 8192), "validation: a negative share");
    c = cfg;
    c.prompt_gap_ms = std::numeric_limits<double>::infinity();
    check(refused(c, rule, 8192), "validation: an infinite gap");
}

int main() {
    test_owner_callbacks_and_boundary();
    test_mixed_drain();
    test_cost_fit_steady_cohort();
    run_identity(false, 4, 8, "plain 4 lanes");
    run_identity(true, 4, 8, "draft 4 lanes 8 rows");
    run_identity(true, 8, 32, "draft 8 lanes 32 rows");
    run_identity(true, 16, 8, "draft lanes capped by rows");
    run_identity(true, 1, 16, "draft 1 lane (solo path)");
    test_admission_bytes();
    test_zero_output_refusal();
    test_short_first();
    test_prompt_piece_rows();
    test_priced_piece_property();
    test_priced_grant_geometries();
    test_gap_row_cost_one_pass();
    test_gap_bounded_rows();
    test_group_gap_room();
    test_group_gap_cold();
    test_gap_forward_price();
    test_bounded_slices_keep_forms();
    test_gap_debt();
    test_gap_row_cost_floor();
    test_piece_validation();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cohort decode tests passed\n";
    return 0;
}
