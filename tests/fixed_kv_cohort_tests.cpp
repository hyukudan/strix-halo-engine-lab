// FixedKvCohort: a resident decoder with preallocated KV slots bound to the greedy cohort contract.
// Invariants: geometry refusal, one decode row per lane forwarded in window order, final logits of a sampled
// lane read from its row of the pending window (never another lane's, never after commit or its release), decoders
// without logits stay greedy-only, an emits-first decoder serves the first id and final scores of its own last
// prefill call until the next call or the lane's release, and commit needs a pending verify window. A decoder with
// per-slot verify windows serves one drafted lane per verify (the kept prefix committed, a released lane's window
// discarded), and a cohort that drafts through it under the live-lane rule emits the ids of one-row decode.
#include "strix/serve/fixed_kv_cohort.hpp"
#include "strix/spec/block_drafter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using strix::serve::CohortGroup;
using strix::serve::FixedKvCohort;

namespace {

int failures = 0;

void check(bool value, const std::string& label) {
    if (!value) {
        std::fprintf(stderr, "FAIL %s\n", label.c_str());
        ++failures;
    }
}

template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// Greedy-only decoder: top[row] = 1000 * lane + token, records every call.
struct GreedyDecoder {
    std::uint32_t allocated = 0;
    std::vector<std::uint32_t> prefilled;
    std::vector<std::uint32_t> last_slots;
    void allocate_slots(std::uint32_t lanes) { allocated = lanes; }
    void prefill_slot(std::uint32_t lane, std::span<const std::uint32_t>, bool first) {
        prefilled.push_back(lane * 2U + (first ? 1U : 0U));
    }
    void decode_slots(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> tokens,
                      std::span<std::uint32_t> top) {
        last_slots.assign(slots.begin(), slots.end());
        for (std::size_t row = 0; row < slots.size(); ++row) top[row] = 1000U * slots[row] + tokens[row];
    }
};

// Sampling-capable decoder: row r of the last window holds logits {r, lane, r, lane}.
struct LogitsDecoder : GreedyDecoder {
    bool supported = true;
    [[nodiscard]] bool final_logits_supported() const { return supported; }
    [[nodiscard]] std::uint32_t logit_width() const { return 4U; }
    void read_slot_logits(std::uint32_t row, std::span<float> out) const {
        const auto lane = static_cast<float>(last_slots.at(row));
        const auto r = static_cast<float>(row);
        out[0] = r;
        out[1] = lane;
        out[2] = r;
        out[3] = lane;
    }
};

void test_geometry() {
    check(throws([] { FixedKvCohort<GreedyDecoder>(nullptr, 64U, 4U, 2U); }), "null decoder refused");
    check(throws([] { FixedKvCohort<GreedyDecoder>(std::make_unique<GreedyDecoder>(), 0U, 4U, 2U); }),
          "zero context refused");
    check(throws([] { FixedKvCohort<GreedyDecoder>(std::make_unique<GreedyDecoder>(), 64U, 4U, 0U); }),
          "zero lanes refused");
    check(throws([] { FixedKvCohort<GreedyDecoder>(std::make_unique<GreedyDecoder>(), 64U, 2U, 3U); }),
          "lanes above rows refused");
    auto decoder = std::make_unique<GreedyDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<GreedyDecoder> cohort(std::move(decoder), 64U, 4U, 3U);
    check(raw->allocated == 3U, "slots allocated once for every lane");
    check(cohort.lanes() == 3U && cohort.rows_per_forward() == 4U && cohort.max_positions() == 64U,
          "geometry reported");
}

void test_greedy_window() {
    auto decoder = std::make_unique<GreedyDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<GreedyDecoder> cohort(std::move(decoder), 64U, 4U, 4U);
    const std::uint32_t prompt[] = {7U, 8U};
    cohort.prefill(2U, prompt, true);
    cohort.prefill(2U, prompt, false);
    check(raw->prefilled == std::vector<std::uint32_t>{5U, 4U}, "prefill forwards lane and first flag");

    // Window order is the group order, not lane order.
    const CohortGroup groups[] = {{3U, 1U}, {0U, 1U}, {2U, 1U}};
    const std::uint32_t tokens[] = {11U, 12U, 13U};
    std::uint32_t top[3] = {};
    cohort.verify(groups, tokens, top);
    check(raw->last_slots == std::vector<std::uint32_t>{3U, 0U, 2U}, "decode slots in group order");
    check(top[0] == 3011U && top[1] == 12U && top[2] == 2013U, "top ids per row");
    check(!cohort.supports_final_logits(), "greedy-only decoder declines final logits");
    const std::uint32_t keep_short[] = {1U, 1U};
    check(throws([&] { cohort.commit(keep_short); }), "keep count != window refused");
    const std::uint32_t keep_zero[] = {1U, 0U, 1U};
    check(throws([&] { cohort.commit(keep_zero); }), "dropping a guaranteed row refused");
    const std::uint32_t keep_ok[] = {1U, 1U, 1U};
    check(!throws([&] { cohort.commit(keep_ok); }), "keep one row per lane accepted");
    check(throws([&] { cohort.commit(keep_ok); }), "a second commit of the same window refused");

    const CohortGroup multi[] = {{1U, 2U}};
    const std::uint32_t two[] = {1U, 2U};
    std::uint32_t top2[2] = {};
    check(throws([&] { cohort.verify(multi, two, top2); }), "multi-row group refused");
}

void test_sampled_logits() {
    auto decoder = std::make_unique<LogitsDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<LogitsDecoder> cohort(std::move(decoder), 64U, 4U, 4U);
    check(cohort.supports_final_logits(), "logits decoder serves sampled lanes");
    const CohortGroup groups[] = {{2U, 1U}, {0U, 1U}};
    const std::uint32_t tokens[] = {1U, 2U};
    std::uint32_t top[2] = {};
    cohort.verify(groups, tokens, top);
    const auto lane0 = cohort.read_final_logits(0U);
    check(lane0.size() == 4U && lane0[0] == 1.0F && lane0[1] == 0.0F, "lane 0 reads its own row (row 1)");
    const auto lane2 = cohort.read_final_logits(2U);
    check(lane2.size() == 4U && lane2[0] == 0.0F && lane2[1] == 2.0F, "lane 2 reads its own row (row 0)");
    check(throws([&] { (void)cohort.read_final_logits(3U); }), "lane outside the window refused");
    // a released lane's row is withdrawn; the other rows stay readable and the window still commits whole
    cohort.release(2U);
    check(throws([&] { (void)cohort.read_final_logits(2U); }), "released lane's row refused");
    check(cohort.read_final_logits(0U)[1] == 0.0F, "other lane still reads its row after a release");
    const std::uint32_t keep[] = {1U, 1U};
    check(!throws([&] { cohort.commit(keep); }), "window with a released row commits whole");
    check(throws([&] { (void)cohort.read_final_logits(0U); }), "no row readable after commit");
    raw->supported = false;
    check(!cohort.supports_final_logits(), "decoder may decline at run time");
}

// Decoder that takes the window's score-read requirement (CohortGroup::reads_scores of any lane).
struct ScoreReadDecoder : LogitsDecoder {
    std::vector<int> reads;  // per decode_slots call: 1 = some lane reads its final scores
    void decode_slots(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> tokens,
                      std::span<std::uint32_t> top, bool reads_scores) {
        reads.push_back(reads_scores ? 1 : 0);
        LogitsDecoder::decode_slots(slots, tokens, top);
    }
};

void test_score_reads() {
    auto decoder = std::make_unique<ScoreReadDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<ScoreReadDecoder> cohort(std::move(decoder), 64U, 4U, 4U);
    const std::uint32_t tokens[] = {1U, 2U};
    const std::uint32_t keep[] = {1U, 1U};
    std::uint32_t top[2] = {};
    const CohortGroup greedy[] = {{2U, 1U, false}, {0U, 1U, false}};
    cohort.verify(greedy, tokens, top);
    cohort.commit(keep);
    const CohortGroup mixed[] = {{2U, 1U, false}, {0U, 1U, true}};
    cohort.verify(mixed, tokens, top);
    check(cohort.read_final_logits(0U)[1] == 0.0F, "the sampled lane reads its row");
    cohort.commit(keep);
    check(raw->reads == std::vector<int>{0, 1}, "decode_slots learns whether any lane of the window reads scores");
    check(top[0] == 2001U && top[1] == 2U, "the score-read form returns the same picks");
}

// A score-aware decoder that withholds the scores of a window passed as unread (its reads return NaN), as a pick-only
// form would: a caller that names no requirement ({lane, rows}) must still read exact scores.
struct WithholdingDecoder : LogitsDecoder {
    bool withheld = false;
    void decode_slots(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> tokens,
                      std::span<std::uint32_t> top, bool reads_scores) {
        withheld = !reads_scores;
        LogitsDecoder::decode_slots(slots, tokens, top);
    }
    void read_slot_logits(std::uint32_t row, std::span<float> out) const {
        LogitsDecoder::read_slot_logits(row, out);
        if (withheld) std::fill(out.begin(), out.end(), std::numeric_limits<float>::quiet_NaN());
    }
};

void test_unannotated_groups_read() {
    auto decoder = std::make_unique<WithholdingDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<WithholdingDecoder> cohort(std::move(decoder), 64U, 4U, 4U);
    const std::uint32_t tokens[] = {1U, 2U};
    const std::uint32_t keep[] = {1U, 1U};
    std::uint32_t top[2] = {};
    const CohortGroup legacy[] = {{2U, 1U}, {0U, 1U}};
    cohort.verify(legacy, tokens, top);
    check(!raw->withheld, "a group that names no requirement is read (default reads_scores)");
    const auto lane0 = cohort.read_final_logits(0U);
    check(!std::isnan(lane0[0]) && lane0[0] == 1.0F && lane0[1] == 0.0F, "the unannotated caller reads exact scores");
    cohort.commit(keep);
    const CohortGroup none[] = {{2U, 1U, false}, {0U, 1U, false}};
    cohort.verify(none, tokens, top);
    check(raw->withheld, "only an explicit no-consumer window may take the pick form");
    cohort.commit(keep);
}

// Emits-first decoder: each prefill call returns id 7000 + 10 * lane + rows; its final scores are row 0 of the window
// (LogitsDecoder: {0, lane, 0, lane}).
struct EmitsDecoder : LogitsDecoder {
    std::vector<std::uint32_t> first_rows;
    std::uint32_t prefill_slot_first(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) {
        prefilled.push_back(lane * 2U + (first ? 1U : 0U));
        first_rows.push_back(static_cast<std::uint32_t>(tokens.size()));
        last_slots.assign(1U, lane);
        return 7000U + 10U * lane + static_cast<std::uint32_t>(tokens.size());
    }
};
// Emits-first without final logits: greedy ids only.
struct EmitsGreedyDecoder : GreedyDecoder {
    std::uint32_t prefill_slot_first(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool) {
        return 7000U + 10U * lane + static_cast<std::uint32_t>(tokens.size());
    }
};

// The lane reads its whole prompt by prefill; the first id and the final scores are the prefill call's own and are
// valid until the next backend call. A decoder without the hook keeps the decode-row contract.
void test_prefill_emits_first() {
    FixedKvCohort<GreedyDecoder> plain(std::make_unique<GreedyDecoder>(), 64U, 4U, 2U);
    check(!plain.prefill_emits_first(), "decoder without prefill_slot_first keeps the decode-row first id");
    check(throws([&] { (void)plain.prefill_top(0U); }), "plain cohort has no prefill_top");

    auto decoder = std::make_unique<EmitsDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<EmitsDecoder> cohort(std::move(decoder), 64U, 4U, 4U);
    check(cohort.prefill_emits_first(), "decoder with prefill_slot_first emits the first id");
    const std::uint32_t prompt[] = {5U, 6U, 7U};
    cohort.prefill(1U, prompt, true);
    check(raw->prefilled.size() == 1U && raw->prefilled[0] == 3U && raw->first_rows.size() == 1U &&
              raw->first_rows[0] == 3U,
          "whole prompt reaches prefill_slot_first with first = true");
    check(cohort.prefill_top(1U) == 7013U, "first id = the prefill's own");
    const auto scores = cohort.read_final_logits(1U);
    check(scores.size() == 4U && scores[0] == 0.0F && scores[1] == 1.0F, "final logits = the prefill's row 0");
    check(throws([&] { (void)cohort.prefill_top(2U); }), "first id of a lane not just prefilled refused");
    const std::uint32_t keep_one[] = {1U};
    check(throws([&] { cohort.commit(keep_one); }), "commit right after a prefill refused (no verify pending)");

    // release withdraws the first id and the score row; the reused lane then serves only its new prefill
    cohort.release(1U);
    check(throws([&] { (void)cohort.prefill_top(1U); }), "released lane's first id refused");
    check(throws([&] { (void)cohort.read_final_logits(1U); }), "released lane's final logits refused");
    cohort.prefill(1U, std::span<const std::uint32_t>(prompt).first(1), true);
    check(cohort.prefill_top(1U) == 7011U, "reused lane serves its new prefill's first id");

    // a decode window invalidates the prefill's first id; a later prefill invalidates the window rows
    const CohortGroup groups[] = {{2U, 1U}, {1U, 1U}};
    const std::uint32_t tokens[] = {4U, 4U};
    std::uint32_t top[2] = {};
    cohort.verify(groups, tokens, top);
    check(throws([&] { (void)cohort.prefill_top(1U); }), "first id not readable after a decode window");
    const auto window = cohort.read_final_logits(1U);
    check(window.size() == 4U && window[0] == 1.0F && window[1] == 1.0F, "window row read after the decode window");
    cohort.prefill(3U, std::span<const std::uint32_t>(prompt).first(2), true);
    check(throws([&] { (void)cohort.read_final_logits(1U); }), "window rows not readable after a prefill call");
    check(cohort.prefill_top(3U) == 7032U, "second lane's own first id");

    FixedKvCohort<EmitsGreedyDecoder> greedy(std::make_unique<EmitsGreedyDecoder>(), 64U, 4U, 2U);
    greedy.prefill(0U, prompt, true);
    check(greedy.prefill_emits_first() && greedy.prefill_top(0U) == 7003U && !greedy.supports_final_logits(),
          "greedy-only emits-first decoder");
}

// A decoder with both prefill calls declares its serving identity at run time: the incumbent prefill_slot route
// (first id from the decode row) or the emits-first route; the cohort serves that one only.
struct SelectDecoder : EmitsDecoder {
    bool emits = false;
    [[nodiscard]] bool prefill_emits_first() const { return emits; }
};
void test_prefill_route_selected() {
    const std::uint32_t prompt[] = {5U, 6U, 7U};
    for (const bool emits : {false, true}) {
        auto decoder = std::make_unique<SelectDecoder>();
        decoder->emits = emits;
        auto* raw = decoder.get();
        FixedKvCohort<SelectDecoder> cohort(std::move(decoder), 64U, 4U, 2U);
        cohort.prefill(1U, prompt, true);
        check(cohort.prefill_emits_first() == emits, "selected route: the decoder's identity is the cohort's");
        check(raw->prefilled.size() == 1U && raw->prefilled[0] == 3U && raw->first_rows.size() == (emits ? 1U : 0U),
              "selected route: the prompt reaches the selected prefill call only");
        if (emits) check(cohort.prefill_top(1U) == 7013U, "selected emits-first route: the prefill's own first id");
        else check(throws([&] { (void)cohort.prefill_top(1U); }), "selected incumbent route: no prefill first id");
    }
}

// A decoder with per-slot verify windows over a deterministic model: the greedy id after a row is a function of the
// slot's committed history and the row's token (so drafting cannot change ids unless a window commits a row the model
// did not pick). It records every window and refuses a second staged window (one checkpoint workspace).
struct WindowDecoder : GreedyDecoder {
    std::vector<std::vector<std::uint32_t>> history;
    std::vector<std::uint32_t> staged;  // per slot: staged rows, 0 = none
    std::uint32_t windows = 0, window_rows = 0, discarded = 0, max_live_with_window = 0;
    std::uint32_t live_slots = 0;
    std::vector<std::vector<std::uint32_t>> stage_tokens;
    static std::uint32_t model(const std::vector<std::uint32_t>& h) {
        // a period-6 cycle broken by the history length every 17 rows: n-gram drafts hit, some miss
        const auto n = static_cast<std::uint32_t>(h.size());
        const std::uint32_t last = h.empty() ? 0U : h.back();
        return n % 17U == 0U ? (last * 7U + 3U) % 41U : (last + 5U) % 41U % 6U + 30U;
    }
    void allocate_slots(std::uint32_t lanes) {
        GreedyDecoder::allocate_slots(lanes);
        history.assign(lanes, {});
        staged.assign(lanes, 0U);
        stage_tokens.assign(lanes, {});
    }
    void prefill_slot(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) {
        if (staged.at(lane) != 0U) throw std::logic_error("prefill over a staged window");
        if (first) history[lane].clear();
        history[lane].insert(history[lane].end(), tokens.begin(), tokens.end());
    }
    bool fail_decode = false;  // the one-row pass throws (a device fault after the window was staged)
    void decode_slots(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> tokens,
                      std::span<std::uint32_t> top) {
        if (fail_decode) throw std::runtime_error("one-row pass failed");
        last_slots.assign(slots.begin(), slots.end());
        for (std::size_t row = 0; row < slots.size(); ++row) {
            if (staged.at(slots[row]) != 0U) throw std::logic_error("one-row decode over a staged window");
            history[slots[row]].push_back(tokens[row]);
            top[row] = model(history[slots[row]]);
        }
    }
    [[nodiscard]] std::uint32_t max_slot_window() const { return 4U; }
    void stage_slot_window(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::span<std::uint32_t> top) {
        if (std::ranges::any_of(staged, [](std::uint32_t r) { return r != 0U; }))
            throw std::logic_error("two staged windows");
        if (tokens.size() < 2U || tokens.size() > max_slot_window() || top.size() != tokens.size())
            throw std::logic_error("bad window");
        auto h = history.at(slot);
        for (std::size_t row = 0; row < tokens.size(); ++row) {
            h.push_back(tokens[row]);
            top[row] = model(h);
        }
        staged[slot] = static_cast<std::uint32_t>(tokens.size());
        stage_tokens[slot].assign(tokens.begin(), tokens.end());
        ++windows;
        window_rows += staged[slot];
        max_live_with_window = std::max(max_live_with_window, live_slots);
    }
    void commit_slot_window(std::uint32_t slot, std::span<const std::uint32_t> tokens, std::uint32_t keep) {
        if (staged.at(slot) == 0U || keep > staged[slot] || tokens.size() != staged[slot] ||
            !std::ranges::equal(tokens, stage_tokens[slot]))
            throw std::logic_error("bad window commit");
        if (keep == 0U) ++discarded;
        history[slot].insert(history[slot].end(), tokens.begin(), tokens.begin() + keep);
        staged[slot] = 0U;
    }
};

// A window decoder whose window shares its checkpoint with the one-row appends: windows run alone.
struct AloneWindowDecoder : WindowDecoder {
    [[nodiscard]] bool slot_window_alone() const { return true; }
};

std::vector<std::uint32_t> solo_ids(const strix::serve::CohortRequest& request) {
    std::vector<std::uint32_t> h(request.prompt.begin(), request.prompt.end());
    std::vector<std::uint32_t> ids;
    while (ids.size() < request.max_tokens) {
        const auto next = WindowDecoder::model(h);
        ids.push_back(next);
        if (std::ranges::find(request.stop, next) != request.stop.end()) break;
        h.push_back(next);
    }
    return ids;
}

struct WindowRun {
    std::vector<std::vector<std::uint32_t>> ids;
    std::uint64_t drafted = 0, accepted = 0;
    std::uint32_t windows = 0, max_live_with_window = 0;
};

WindowRun run_windows(std::uint32_t max_draft, std::vector<std::uint32_t> by_live, bool staggered,
                      std::uint32_t stop = 9999U, std::uint32_t context = 512U) {
    using strix::serve::CohortConfig;
    using strix::serve::CohortDecoder;
    using strix::serve::CohortRequest;
    auto decoder = std::make_unique<WindowDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<WindowDecoder> cohort(std::move(decoder), context, 4U, 4U);
    CohortConfig config;
    config.context_draft = max_draft != 0U;
    config.max_draft = max_draft;
    config.max_draft_by_live = std::move(by_live);
    config.prompt_slice = 16U;
    config.prior = {40.0, 10.0, 1.0, 0.0};
    CohortDecoder scheduler(cohort, config);
    std::vector<CohortRequest> requests;
    for (std::uint32_t i = 0; i < 3U; ++i) {
        CohortRequest r;
        r.id = i + 1U;
        r.prompt = {31U + i, 32U, 33U, 34U, 35U, 30U + i};
        r.max_tokens = context == 512U ? 60U + 10U * i : context - 6U;  // a short context: prompt + ids fill it
        if (stop != 9999U || i == 2U) r.stop = {stop};  // a given stop applies to every request
        requests.push_back(r);
    }
    const auto count_live = [&] { raw->live_slots = scheduler.live(); };
    scheduler.submit(requests[0]);
    if (!staggered) {
        scheduler.submit(requests[1]);
        scheduler.submit(requests[2]);
    }
    // (staggered: keep stepping until the last arrival, even if an early stop left the cohort idle)
    for (int step = 0; step < 400 && (!scheduler.idle() || (staggered && step <= 40)); ++step) {
        count_live();
        scheduler.step();
        // the second and third requests arrive while the first one drafts alone
        if (staggered && step == 12) scheduler.submit(requests[1]);
        if (staggered && step == 40) scheduler.submit(requests[2]);
    }
    WindowRun out;
    auto results = scheduler.take_finished();
    out.ids.resize(requests.size());
    for (const auto& r : results) {
        check(r.error.empty(), "windows: no request error");
        out.ids.at(r.id - 1U) = r.ids;
        out.drafted += r.drafted;
        out.accepted += r.accepted;
    }
    for (const auto& q : requests) check(out.ids.at(q.id - 1U) == solo_ids(q), "windows: ids == the solo run");
    out.windows = raw->windows;
    out.max_live_with_window = raw->max_live_with_window;
    return out;
}

// FixedKvCohort::bind_lanes: an engine's lone-lane pieces answer the cohort's lane hooks (no binding: the defaults).
// The drafter proposes the target's own next ids for the lane it was started on (a perfect drafter where its context
// is complete), then a wrong id on every third proposal, so windows keep part of a draft.
struct OracleDrafter final : strix::spec::ChainDrafter {
    std::vector<std::uint32_t> h;
    std::uint32_t drafts = 0;
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        auto t = h;
        t.insert(t.end(), prefix.begin(), prefix.end());
        out.clear();
        for (std::uint32_t i = 0; i < k; ++i) {
            const auto next = (drafts + i) % 3U == 2U ? 7U : WindowDecoder::model(t);
            out.push_back(next);
            t.push_back(next);
        }
        ++drafts;
    }
    void accept(std::span<const std::uint32_t> emitted) override { h.insert(h.end(), emitted.begin(), emitted.end()); }
};

struct LaneBinding final : strix::serve::FixedKvLaneBinding {
    WindowDecoder& decoder;
    OracleDrafter drafter;
    std::vector<std::uint32_t> asked;  // lanes whose drafter was asked
    std::uint32_t ready_lane = 0;      // the lane whose drafter inputs are complete (the engine's slot-0 taps)
    bool tuned = false;
    explicit LaneBinding(WindowDecoder& d) : decoder(d) {}
    strix::spec::ChainDrafter* chain_drafter(std::uint32_t lane, std::uint32_t anchor, std::uint64_t position) override {
        asked.push_back(lane);
        drafter.h = decoder.history.at(lane);  // the committed rows (position of them) and the pending anchor
        check(drafter.h.size() == position, "lane binding: the drafter starts at the lane's committed position");
        drafter.h.push_back(anchor);
        return &drafter;
    }
    void tune(strix::spec::BlockConfig&) const override { const_cast<LaneBinding*>(this)->tuned = true; }
    bool ready(std::uint32_t lane) const override { return lane == ready_lane; }
};

void test_lane_binding() {
    using strix::serve::CohortConfig;
    using strix::serve::CohortDecoder;
    using strix::serve::CohortRequest;
    const auto request = [](std::uint64_t id, std::uint32_t max_tokens) {
        CohortRequest r;
        r.id = id;
        r.prompt = {31U + static_cast<std::uint32_t>(id), 32U, 33U, 34U, 35U, 36U};
        r.max_tokens = max_tokens;
        return r;
    };
    const auto make = [](WindowDecoder*& raw, LaneBinding*& bound, bool bind) {
        auto decoder = std::make_unique<WindowDecoder>();
        raw = decoder.get();
        auto cohort = std::make_unique<FixedKvCohort<WindowDecoder>>(std::move(decoder), 512U, 4U, 4U);
        bound = nullptr;
        if (bind) {
            auto binding = std::make_unique<LaneBinding>(*raw);
            bound = binding.get();
            cohort->bind_lanes(std::move(binding));
        }
        return cohort;
    };
    CohortConfig config;
    config.prompt_slice = 16U;
    config.lane_speculation.enabled = true;
    config.lane_speculation.max_rows = 4U;
    {
        // No binding: the hooks' defaults (ready, no model drafter): a lone lane speculates from the copy source alone.
        WindowDecoder* raw = nullptr;
        LaneBinding* bound = nullptr;
        auto cohort = make(raw, bound, false);
        check(cohort->lane_chain_drafter(0U, 1U, 1U) == nullptr && cohort->lane_speculation_ready(3U),
              "lane binding: unbound hooks keep their defaults");
        CohortDecoder scheduler(*cohort, config);
        const auto r = request(1U, 40U);
        scheduler.submit(r);
        for (int step = 0; step < 200 && !scheduler.idle(); ++step) scheduler.step();
        const auto done = scheduler.take_finished();
        check(done.size() == 1U && done[0].ids == solo_ids(r), "lane binding: unbound lone lane ids == solo");
    }
    {
        // Bound: lane 0 drafts with the engine's drafter (windows accept part of its drafts); ids equal the solo run.
        WindowDecoder* raw = nullptr;
        LaneBinding* bound = nullptr;
        auto cohort = make(raw, bound, true);
        CohortDecoder scheduler(*cohort, config);
        const auto r = request(1U, 60U);
        scheduler.submit(r);
        for (int step = 0; step < 200 && !scheduler.idle(); ++step) scheduler.step();
        const auto done = scheduler.take_finished();
        check(done.size() == 1U && done[0].ids == solo_ids(r), "lane binding: lane 0 ids == solo");
        check(!bound->asked.empty() && bound->asked.front() == 0U && bound->drafter.drafts > 0U && bound->tuned,
              "lane binding: lane 0's drafter is asked, tuned and drafts");
        check(raw->windows > 0U && done[0].accepted > 0U, "lane binding: windows keep drafted rows");
    }
    {
        // A request on another lane finishes while lane 0 speculates alone afterwards; a lane that is not ready (its
        // inputs incomplete: not the engine's tap lane) never gets the drafter. Ids of both equal their solo runs.
        WindowDecoder* raw = nullptr;
        LaneBinding* bound = nullptr;
        auto cohort = make(raw, bound, true);
        CohortDecoder scheduler(*cohort, config);
        const auto a = request(1U, 50U), b = request(2U, 8U);
        scheduler.submit(a);
        scheduler.submit(b);
        for (int step = 0; step < 300 && !scheduler.idle(); ++step) scheduler.step();
        auto done = scheduler.take_finished();
        std::ranges::sort(done, {}, &strix::serve::CohortResult::id);
        check(done.size() == 2U && done[0].ids == solo_ids(a) && done[1].ids == solo_ids(b),
              "lane binding: survivor and finished request ids == solo");
        check(std::ranges::all_of(bound->asked, [](std::uint32_t lane) { return lane == 0U; }),
              "lane binding: only the ready lane's drafter is asked");
    }
    {
        // Cancellation mid-run while lane 0 speculates: the request ends cancelled at a step boundary, no window left
        // staged on the decoder.
        WindowDecoder* raw = nullptr;
        LaneBinding* bound = nullptr;
        auto cohort = make(raw, bound, true);
        CohortDecoder scheduler(*cohort, config);
        auto r = request(1U, 200U);
        bool cancel = false;
        r.cancelled = [&] { return cancel; };
        scheduler.submit(r);
        for (int step = 0; step < 300 && !scheduler.idle(); ++step) {
            scheduler.step();
            if (raw->windows >= 3U) cancel = true;
        }
        const auto done = scheduler.take_finished();
        check(done.size() == 1U && done[0].cancelled && done[0].ids.size() < 200U,
              "lane binding: cancelled mid-run while speculating");
        const auto& ids = done[0].ids;
        const auto solo = solo_ids(r);
        check(std::equal(ids.begin(), ids.end(), solo.begin()), "lane binding: cancelled ids are a prefix of solo");
        check(std::ranges::all_of(raw->staged, [](std::uint32_t s) { return s == 0U; }),
              "lane binding: no window left staged after the cancellation");
    }
    {
        // The served measurement sequence (a short warm-up, then the same prompt twice, one request at a time, all on
        // lane 0) and a lone lane on a lane that is not ready: every request's receipt (CohortResult::lane_speculation,
        // also the server's strix-lane-spec line) says whether the model drafter ran, and why not.
        WindowDecoder* raw = nullptr;
        LaneBinding* bound = nullptr;
        auto cohort = make(raw, bound, true);
        auto logged = config;
        logged.lane_speculation.log = true;
        CohortDecoder scheduler(*cohort, logged);
        const auto has = [](const std::string& s, const char* part) { return s.find(part) != std::string::npos; };
        for (const auto& [id, tokens] : {std::pair{1U, 8U}, std::pair{2U, 60U}, std::pair{3U, 60U}}) {
            auto r = request(1U, tokens);
            r.id = id;
            scheduler.submit(r);
            for (int step = 0; step < 300 && !scheduler.idle(); ++step) scheduler.step();
            const auto done = scheduler.take_finished();
            check(done.size() == 1U && done[0].ids == solo_ids(r), "lane receipt: sequential request ids == solo");
            const auto& receipt = done[0].lane_speculation;
            check(has(receipt, "\"lane\":0,") && has(receipt, "\"chained\":true") &&
                      has(receipt, "\"lookup_only\":false") && has(receipt, "\"ready_false\":0,") &&
                      has(receipt, "\"chain_starts\":1,") && has(receipt, "\"suspends\":0,"),
                  "lane receipt: lane 0 drafts with the engine's drafter on every request");
            // (how many passes the model drafts is the arbiter's choice on the wall clock: not asserted here)
        }
        // Two at once: the short one (lane 0) ends, the long one speculates alone on lane 1 (not ready): lookup-only.
        auto a = request(4U, 6U), b = request(5U, 40U);
        scheduler.submit(a);
        scheduler.submit(b);
        for (int step = 0; step < 300 && !scheduler.idle(); ++step) scheduler.step();
        auto done = scheduler.take_finished();
        std::ranges::sort(done, {}, &strix::serve::CohortResult::id);
        check(done.size() == 2U && done[0].ids == solo_ids(a) && done[1].ids == solo_ids(b),
              "lane receipt: pair ids == solo");
        const auto& lone = done[1].lane_speculation;
        check(has(lone, "\"lane\":1,") && has(lone, "\"chained\":false") && has(lone, "\"lookup_only\":true") &&
                  has(lone, "\"ready_false\":1,") && has(lone, "\"chain_starts\":0,") &&
                  has(lone, "\"model\":{\"passes\":0,"),
              "lane receipt: a lone lane that is not ready drafts lookup-only and says why");
        check(scheduler.lane_speculation_ready_false() == 1U && scheduler.lane_speculation_lookup_only() == 1U,
              "lane receipt: the decoder's totals count the refusal");
    }
}

void test_verify_windows() {
    using strix::serve::CohortConfig;
    using strix::serve::max_draft_for_live;
    CohortConfig table;
    table.max_draft = 7U;
    check(max_draft_for_live(table, 1U) == 7U && max_draft_for_live(table, 8U) == 7U, "by-live: empty table");
    table.max_draft_by_live = {7U, 4U, 0U};
    check(max_draft_for_live(table, 0U) == 7U && max_draft_for_live(table, 1U) == 7U &&
              max_draft_for_live(table, 2U) == 4U && max_draft_for_live(table, 3U) == 0U &&
              max_draft_for_live(table, 9U) == 0U,
          "by-live: entry n - 1, the last entry beyond the table");
    table.max_draft = 3U;
    check(max_draft_for_live(table, 1U) == 3U, "by-live: max_draft still caps the table");

    // Plain decoders keep one row per lane: their window bound is one row and a drafted group is refused.
    FixedKvCohort<GreedyDecoder> plain(std::make_unique<GreedyDecoder>(), 64U, 4U, 2U);
    check(plain.max_window(0U) == 1U, "plain decoder: one-row windows");
    {
        const CohortGroup groups[] = {{0U, 2U}};
        const std::uint32_t tokens[] = {1U, 2U};
        std::uint32_t top[2] = {};
        check(throws([&] { plain.verify(groups, tokens, top); }), "plain decoder: a drafted group is refused");
    }

    // Window decoder, direct calls: one window beside one-row lanes, picks in token order, keep bounds.
    auto decoder = std::make_unique<WindowDecoder>();
    auto* raw = decoder.get();
    FixedKvCohort<WindowDecoder> cohort(std::move(decoder), 64U, 4U, 3U);
    check(cohort.max_window(0U) == 4U, "window decoder: its window bound");
    const std::uint32_t p0[] = {30U, 31U}, p1[] = {32U}, p2[] = {33U};
    cohort.prefill(0U, p0, true);
    cohort.prefill(1U, p1, true);
    cohort.prefill(2U, p2, true);
    {
        const CohortGroup groups[] = {{1U, 1U}, {0U, 3U, false}, {2U, 1U}};
        const std::uint32_t tokens[] = {40U, 41U, 42U, 43U, 44U};
        std::uint32_t top[5] = {};
        cohort.verify(groups, tokens, top);
        auto h0 = std::vector<std::uint32_t>{30U, 31U, 41U};
        const auto w0 = WindowDecoder::model(h0);
        check(top[1] == w0 && raw->windows == 1U && raw->staged[0] == 3U, "window: staged on its slot, row 0 pick");
        check(top[0] == WindowDecoder::model({32U, 40U}) && top[4] == WindowDecoder::model({33U, 44U}),
              "window: one-row lanes' picks land at their token index");
        check(raw->last_slots == std::vector<std::uint32_t>({1U, 2U}), "window: one-row lanes run as one pass");
        const std::uint32_t bad[] = {1U, 5U, 1U};
        check(throws([&] { cohort.commit(bad); }), "window: keep above its rows refused");
        const std::uint32_t keep[] = {1U, 2U, 1U};
        cohort.commit(keep);
        check(raw->staged[0] == 0U && raw->history[0] == std::vector<std::uint32_t>({30U, 31U, 41U, 42U}),
              "window: the kept prefix is committed");
    }
    {
        const CohortGroup two[] = {{0U, 2U, false}, {1U, 2U, false}};
        const std::uint32_t tokens[] = {1U, 2U, 3U, 4U};
        std::uint32_t top[4] = {};
        check(throws([&] { cohort.verify(two, tokens, top); }), "window: two windows in one verify refused");
    }
    {
        const CohortGroup groups[] = {{2U, 2U, false}};
        const std::uint32_t tokens[] = {50U, 51U};
        std::uint32_t top[2] = {};
        cohort.verify(groups, tokens, top);
        cohort.release(2U);
        const std::uint32_t keep[] = {2U};
        cohort.commit(keep);
        check(raw->discarded == 1U && raw->staged[2] == 0U, "window: a released lane's window is discarded");
    }
    {
        // A score-reading drafted group is refused before staging: a window writes picks, not readable scores.
        const CohortGroup read[] = {{0U, 2U, true}};
        const std::uint32_t tokens[] = {60U, 61U};
        std::uint32_t top[2] = {};
        const auto before = raw->windows;
        check(throws([&] { cohort.verify(read, tokens, top); }) && raw->windows == before && raw->staged[0] == 0U,
              "window: a score-reading window refused before any staging");
    }
    {
        // Release then reuse on a nonzero slot with no commit in between: the prefill of the reused slot discards the
        // staged window first (the decoder refuses a prefill over a staged window).
        const CohortGroup groups[] = {{2U, 3U, false}};
        const std::uint32_t tokens[] = {52U, 53U, 54U};
        std::uint32_t top[3] = {};
        cohort.verify(groups, tokens, top);
        cohort.release(2U);
        const std::uint32_t reuse[] = {35U, 36U};
        check(!throws([&] { cohort.prefill(2U, reuse, true); }), "window: release then reuse needs no commit");
        check(raw->discarded == 2U && raw->staged[2] == 0U && raw->history[2] == std::vector<std::uint32_t>({35U, 36U}),
              "window: the reused slot's staged window was discarded, its new prompt is the history");
    }
    {
        // A verify that never got its commit: the next verify discards the window before staging another.
        const auto h0 = raw->history[0];
        const CohortGroup groups[] = {{0U, 2U, false}};
        const std::uint32_t a[] = {55U, 56U}, b[] = {57U, 58U};
        std::uint32_t top[2] = {};
        cohort.verify(groups, a, top);
        check(!throws([&] { cohort.verify(groups, b, top); }) && raw->discarded == 3U,
              "window: an uncommitted window is discarded by the next verify");
        const std::uint32_t keep[] = {1U};
        cohort.commit(keep);
        auto want = h0;
        want.push_back(57U);
        check(raw->history[0] == want, "window: only the committed window reaches the history");
    }
    {
        // Mixed verify whose one-row companion pass throws after the window was staged: the window is unwound, the
        // history is unchanged and the cohort serves the next verify.
        auto throwing = std::make_unique<WindowDecoder>();
        auto* t = throwing.get();
        FixedKvCohort<WindowDecoder> mixed(std::move(throwing), 64U, 4U, 2U);
        const std::uint32_t q0[] = {30U}, q1[] = {31U};
        mixed.prefill(0U, q0, true);
        mixed.prefill(1U, q1, true);
        t->fail_decode = true;
        const CohortGroup groups[] = {{0U, 3U, false}, {1U, 1U}};
        const std::uint32_t tokens[] = {40U, 41U, 42U, 43U};
        std::uint32_t top[4] = {};
        check(throws([&] { mixed.verify(groups, tokens, top); }), "window: the companion pass throws");
        check(t->windows == 1U && t->discarded == 1U && t->staged[0] == 0U &&
                  t->history[0] == std::vector<std::uint32_t>({30U}),
              "window: a failed mixed verify unwinds its staged window");
        const std::uint32_t keep[] = {1U, 1U};
        check(throws([&] { mixed.commit(keep); }), "window: a failed verify cannot be committed");
        t->fail_decode = false;
        mixed.verify(groups, tokens, top);
        mixed.commit(keep);
        check(t->history[0] == std::vector<std::uint32_t>({30U, 40U}) &&
                  t->history[1] == std::vector<std::uint32_t>({31U, 43U}),
              "window: the cohort serves the next verify after the unwind");
    }
    {
        // A decoder that runs windows alone: a mixed verify is refused before any staging, a lone window runs.
        auto alone = std::make_unique<AloneWindowDecoder>();
        auto* a = alone.get();
        FixedKvCohort<AloneWindowDecoder> cohort_alone(std::move(alone), 64U, 4U, 2U);
        const std::uint32_t q0[] = {30U}, q1[] = {31U};
        cohort_alone.prefill(0U, q0, true);
        cohort_alone.prefill(1U, q1, true);
        const CohortGroup mixed[] = {{0U, 3U, false}, {1U, 1U}};
        const std::uint32_t tokens[] = {40U, 41U, 42U, 43U};
        std::uint32_t top[4] = {};
        check(throws([&] { cohort_alone.verify(mixed, tokens, top); }) && a->windows == 0U && a->last_slots.empty(),
              "window alone: a mixed verify refused before any work");
        const CohortGroup lone[] = {{0U, 3U, false}};
        cohort_alone.verify(lone, std::span<const std::uint32_t>(tokens).first(3), std::span<std::uint32_t>(top).first(3));
        const std::uint32_t keep[] = {2U};
        cohort_alone.commit(keep);
        check(a->windows == 1U && a->history[0] == std::vector<std::uint32_t>({30U, 40U, 41U}),
              "window alone: a lone window runs and commits");
    }

    // Whole cohort: context drafts through windows give the ids of one-row decode, and the {4, 0} rule keeps every
    // window to a lone live lane (a prompt arriving ends drafting at the next round).
    const auto off = run_windows(0U, {}, true);
    check(off.windows == 0U && off.drafted == 0U, "cohort off: no window");
    const auto on = run_windows(4U, {4U, 0U}, true);
    check(on.ids == off.ids, "cohort: ids with windows == ids without");
    check(on.windows > 0U && on.accepted > 0U, "cohort: windows ran and kept drafts");
    check(on.max_live_with_window <= 1U, "cohort {4, 0}: windows only while one lane is live");
    const auto together = run_windows(4U, {4U, 0U}, false);
    check(together.ids == off.ids, "cohort: simultaneous arrivals keep the ids");
    // Reachable stop ids (each one ends every request; some fall inside an accepted window prefix): delivery stops at
    // the stop id exactly as the solo run does, with windows on.
    std::uint64_t stop_accepted = 0;
    for (const std::uint32_t stop : {30U, 31U, 32U, 33U, 34U, 35U, 15U, 2U}) {
        const auto s = run_windows(4U, {4U, 0U}, true, stop);  // run_windows checks ids == the solo run
        stop_accepted += s.accepted;
    }
    check(stop_accepted > 0U, "cohort stops: windows kept drafts while stops were armed");
    // The final context row: prompt (6) + 42 ids fill a 48-position context exactly; windows near the end are
    // capped by the remaining positions and the ids equal the one-row run.
    const auto ctx_off = run_windows(0U, {}, true, 9999U, 48U);
    const auto ctx_on = run_windows(4U, {4U, 0U}, true, 9999U, 48U);
    check(ctx_on.ids == ctx_off.ids && ctx_on.windows > 0U, "cohort at the context end: ids with windows == without");
    bool capped = true;
    for (const auto& ids : ctx_off.ids) capped = capped && ids.size() == 42U;
    check(capped, "cohort at the context end: every request reaches the last position");
    std::printf("verify windows: off windows %u; on windows %u drafted %llu accepted %llu\n", off.windows, on.windows,
                static_cast<unsigned long long>(on.drafted), static_cast<unsigned long long>(on.accepted));
}

}  // namespace

// A decoder that admits a prefill tail: the cohort reports it, a decoder without the accessor reports none.
struct TailDecoder : GreedyDecoder {
    [[nodiscard]] std::uint32_t prefill_tail_rows() const noexcept { return 1024U; }
};

void test_prompt_tail() {
    using strix::serve::prompt_slice_rows;
    FixedKvCohort<TailDecoder> tail(std::make_unique<TailDecoder>(), 64U, 4U, 2U);
    FixedKvCohort<GreedyDecoder> plain(std::make_unique<GreedyDecoder>(), 64U, 4U, 2U);
    check(tail.prompt_tail_rows() == 1024U && plain.prompt_tail_rows() == 0U, "tail rows forwarded from the decoder");
    static_assert(prompt_slice_rows(8192U, 8192U, 8235U, 1024U) == 8235U);  // 8192 + 43 -> one pass
    static_assert(prompt_slice_rows(8192U, 8192U, 9216U, 1024U) == 9216U);  // remainder == tail
    static_assert(prompt_slice_rows(8192U, 8192U, 9300U, 1024U) == 8192U);  // remainder > tail: exact slice
    static_assert(prompt_slice_rows(8192U, 8192U, 8235U, 0U) == 8192U);     // no tail: greedy slices
    static_assert(prompt_slice_rows(4096U, 8192U, 8235U, 1024U) == 4096U);  // the planner granted a part: unchanged
    static_assert(prompt_slice_rows(8192U, 8192U, 100U, 1024U) == 100U);    // short prompt: all of it
}

int main() {
    test_prompt_tail();
    test_geometry();
    test_greedy_window();
    test_sampled_logits();
    test_score_reads();
    test_unannotated_groups_read();
    test_prefill_emits_first();
    test_prefill_route_selected();
    test_verify_windows();
    test_lane_binding();
    if (failures != 0) {
        std::fprintf(stderr, "fixed-kv-cohort: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("fixed-kv-cohort: PASS\n");
    return EXIT_SUCCESS;
}
