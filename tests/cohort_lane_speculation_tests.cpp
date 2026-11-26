// Host-only tests of lone-lane speculation in the shared multi-request decoder (CohortConfig::lane_speculation,
// strix/serve/lane_speculation.hpp) over a fake engine, a fake model drafter and a virtual clock (advanced only by
// verify and draft calls, so the speculator's walls are a function of its decisions).
//   1. One lane: the emitted ids and every speculator decision (draft calls, verify rows, commits) equal the command
//      line's spec::execute_block_token_continuation over an equivalent fake VerifyTarget, same drafter and clock.
//   2. A second request admitted mid-run: hand-over at a step boundary, both requests' ids equal their solo runs, the
//      model drafter restarts on the resume.
//   3. Live back to 1 with lane_speculation_ready false: lookup-only afterwards (the drafter is never started again).
//   4. One of two requests finishes: the survivor starts speculating; ids equal solo.
//   5. Cancellation while speculating: the request ends cancelled at a step boundary, no window left open, the lane
//      released.
#include "strix/serial_request.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/lane_speculation.hpp"
#include "strix/spec/block_setup.hpp"
#include "strix/spec/greedy_continuation.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace strix::serve;
namespace spec = strix::spec;

namespace {

int failures = 0;
void check(bool value, const std::string& label) {
    if (!value) {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}

constexpr std::uint32_t kVocab = 23;
constexpr std::uint32_t kRows = 8;  // the speculator's widest window in every test

// The target: a short cycle (lookup proposals hit) with a position-dependent break (they also miss).
std::uint32_t next_id(const std::vector<std::uint32_t>& ctx) {
    const std::uint32_t a = ctx.empty() ? 0U : ctx.back();
    const std::uint32_t bump = ctx.size() % 13U == 0U ? 1U : 0U;
    return (a * 7U + 3U + bump) % kVocab;
}

std::vector<std::uint32_t> solo(const CohortRequest& r) {
    std::vector<std::uint32_t> ctx = r.prompt, out;
    while (out.size() < r.max_tokens) {
        const std::uint32_t t = next_id(ctx);
        out.push_back(t);
        ctx.push_back(t);
        if (std::find(r.stop.begin(), r.stop.end(), t) != r.stop.end()) break;
    }
    return out;
}

using Log = std::vector<std::string>;

// A model drafter: the target's continuation of its stream (committed tokens + anchor), wrong at some positions.
class FakeChain final : public spec::ChainDrafter {
public:
    FakeChain(std::vector<std::uint32_t> stream, double* clock, Log* log, std::uint64_t* drafts)
        : stream_(std::move(stream)), clock_(clock), log_(log), drafts_(drafts) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        std::vector<std::uint32_t> c = stream_;
        c.insert(c.end(), prefix.begin(), prefix.end());
        out.clear();
        for (std::uint32_t d = 0; d < k; ++d) {
            const std::uint32_t truth = next_id(c);
            const std::uint32_t t = c.size() % 9U == 4U ? (truth + 1U) % kVocab : truth;
            out.push_back(t);
            c.push_back(t);
        }
        *clock_ += 0.6 + 0.3 * static_cast<double>(k);
        if (drafts_) ++*drafts_;
        if (log_) log_->push_back("D" + std::to_string(prefix.size()) + "," + std::to_string(k));
    }
    void accept(std::span<const std::uint32_t> emitted) override {
        stream_.insert(stream_.end(), emitted.begin(), emitted.end());
    }
private:
    std::vector<std::uint32_t> stream_;
    double* clock_;
    Log* log_;
    std::uint64_t* drafts_;
};

// The engine's speculator data: the virtual clock, the hybrid switch, verify seeds.
struct Tuning {
    double* clock = nullptr;
    bool hybrid = false;
    void tune(spec::BlockConfig& c) const {
        double* const now = clock;
        c.clock_ms = [now] { return *now; };
        c.hybrid = hybrid;
    }
    static void seed(spec::BlockDrafter& d) {
        for (std::uint32_t r = 1; r <= kRows; ++r) d.seed_verify_ms(r, 2.0 + 0.25 * static_cast<double>(r));
    }
};

double verify_cost(std::uint32_t rows) { return 2.0 + 0.25 * static_cast<double>(rows); }

class SpecBackend final : public CohortBackend {
public:
    SpecBackend(std::uint32_t lanes, bool chain_on, bool hybrid)
        : ready(lanes, true), chain_starts(lanes, 0), released(lanes, 0), drafts(lanes, 0), lanes_(lanes), ctx_(lanes), chains_(lanes),
          chain_on_(chain_on), tuning_{&clock, hybrid} {}
    std::uint32_t lanes() const override { return lanes_; }
    std::uint32_t rows_per_forward() const override { return 16; }
    std::uint32_t max_positions() const override { return 4096; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        check(window_.empty(), "prefill with an open window");
        if (first) ctx_.at(lane).clear();
        ctx_.at(lane).insert(ctx_[lane].end(), tokens.begin(), tokens.end());
    }
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        check(window_.empty(), "verify with an open window");
        std::size_t base = 0;
        for (const auto& g : groups) {
            std::vector<std::uint32_t> c = ctx_.at(g.lane);
            for (std::uint32_t i = 0; i < g.rows; ++i) {
                c.push_back(tokens[base + i]);
                top[base + i] = next_id(c);
            }
            window_.push_back({g.lane, std::vector<std::uint32_t>(tokens.begin() + static_cast<std::ptrdiff_t>(base),
                                                                  tokens.begin() + static_cast<std::ptrdiff_t>(base + g.rows))});
            clock += verify_cost(g.rows);
            base += g.rows;
        }
        if (groups.size() > 1U) ++shared_rounds;
        if (log && groups.size() == 1U) log->push_back("V" + std::to_string(groups[0].rows));
    }
    void commit(std::span<const std::uint32_t> accept) override {
        check(accept.size() == window_.size(), "one keep count per group");
        for (std::size_t g = 0; g < window_.size() && g < accept.size(); ++g) {
            check(accept[g] >= 1 && accept[g] <= window_[g].second.size(), "keep out of range");
            auto& c = ctx_.at(window_[g].first);
            c.insert(c.end(), window_[g].second.begin(),
                     window_[g].second.begin() + static_cast<std::ptrdiff_t>(accept[g]));
        }
        if (log && window_.size() == 1U) log->push_back("C" + std::to_string(accept[0]));
        window_.clear();
    }
    void release(std::uint32_t lane) override {
        check(window_.empty(), "release with an open window");
        ctx_.at(lane).clear();
        ++released.at(lane);
        chains_.at(lane).reset();
    }
    spec::ChainDrafter* lane_chain_drafter(std::uint32_t lane, std::uint32_t anchor, std::uint64_t position) override {
        check(ctx_.at(lane).size() == position, "the drafter starts at the lane's committed positions");
        ++chain_starts.at(lane);
        if (!chain_on_) return nullptr;
        std::vector<std::uint32_t> stream = ctx_[lane];
        stream.push_back(anchor);
        chains_[lane] = std::make_unique<FakeChain>(std::move(stream), &clock, log, &drafts[lane]);
        return chains_[lane].get();
    }
    void tune_lane_speculation(spec::BlockConfig& config) const override { tuning_.tune(config); }
    void seed_lane_speculation(spec::BlockDrafter& drafter) const override { Tuning::seed(drafter); }
    bool lane_speculation_ready(std::uint32_t lane) const override { return ready.at(lane); }

    double clock = 0.0;
    Log* log = nullptr;
    std::vector<bool> ready;
    std::vector<std::uint32_t> chain_starts, released;
    std::vector<std::uint64_t> drafts;  // model drafts per lane (all its drafters)
    std::uint64_t shared_rounds = 0;

private:
    std::uint32_t lanes_;
    std::vector<std::vector<std::uint32_t>> ctx_;
    std::vector<std::unique_ptr<FakeChain>> chains_;
    std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>> window_;
    bool chain_on_;
    Tuning tuning_;
};

// The command line's target: committed context + one open window.
class RefTarget final : public spec::VerifyTarget {
public:
    RefTarget(std::vector<std::uint32_t> ctx, double* clock, Log* log) : ctx_(std::move(ctx)), clock_(clock), log_(log) {}
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        window_.assign(tokens.begin(), tokens.end());
        greedy.clear();
        std::vector<std::uint32_t> c = ctx_;
        for (const auto t : tokens) {
            c.push_back(t);
            greedy.push_back(next_id(c));
        }
        *clock_ += verify_cost(static_cast<std::uint32_t>(tokens.size()));
        log_->push_back("V" + std::to_string(tokens.size()));
    }
    void commit(std::uint32_t k) override {
        ctx_.insert(ctx_.end(), window_.begin(), window_.begin() + static_cast<std::ptrdiff_t>(k));
        window_.clear();
        log_->push_back("C" + std::to_string(k));
    }

private:
    std::vector<std::uint32_t> ctx_, window_;
    double* clock_;
    Log* log_;
};

struct Run {
    std::vector<std::uint32_t> ids;
    Log log;
};

// spec::execute_block_token_continuation as the command line runs it (auto, learned prices): the prompt but its last
// token is committed, the last token is the pending one (its first output is that token, dropped here).
Run reference(const std::vector<std::uint32_t>& prompt, std::uint32_t max_tokens, std::uint32_t eos, bool chain_on,
              bool hybrid) {
    Run run;
    double clock = 0.0;
    const std::vector<std::uint32_t> committed(prompt.begin(), prompt.end() - 1);
    RefTarget target(committed, &clock, &run.log);
    std::unique_ptr<FakeChain> chain;
    if (chain_on) chain = std::make_unique<FakeChain>(prompt, &clock, &run.log, nullptr);
    const Tuning tuning{&clock, hybrid};
    spec::BlockSetup setup;
    setup.max_rows = kRows;
    setup.tune = [&](spec::BlockConfig& c) { tuning.tune(c); };
    setup.chain = [&](spec::BlockConfig&) -> spec::ChainDrafter* { return chain.get(); };
    setup.seed = [](spec::BlockDrafter& d) { Tuning::seed(d); };
    std::optional<spec::BlockDrafter> drafter;
    (void)spec::make_block_drafter(drafter, setup);
    const strix::SerialTokenContinuationRequest req{prompt.back(), committed.size(), kVocab, 4096,
                                                    std::uint64_t{max_tokens} + 1U, eos};
    const auto result = spec::execute_block_token_continuation(req, committed, *drafter, target, kRows);
    run.ids.assign(result.generated_tokens.begin() + 1, result.generated_tokens.end());
    return run;
}

CohortConfig spec_config() {
    CohortConfig c;
    c.lane_speculation.enabled = true;
    c.lane_speculation.max_rows = kRows;
    return c;
}

CohortRequest request(std::uint64_t id, std::vector<std::uint32_t> prompt, std::uint32_t max_tokens,
                      std::vector<std::uint32_t> stop = {}) {
    CohortRequest r;
    r.id = id;
    r.prompt = std::move(prompt);
    r.max_tokens = max_tokens;
    r.stop = std::move(stop);
    return r;
}

std::map<std::uint64_t, CohortResult> collect(CohortDecoder& dec, std::map<std::uint64_t, CohortResult> into = {}) {
    for (auto& r : dec.take_finished()) into[r.id] = std::move(r);
    return into;
}

const std::vector<std::uint32_t> kPromptA = {1, 4, 9, 16, 2, 8, 5};
const std::vector<std::uint32_t> kPromptB = {3, 3, 7, 11, 6};

// 1. One lane == the command line's block continuation: ids, draft calls, verify rows and commits in order.
void test_one_lane(bool chain_on, bool hybrid, std::uint32_t max_tokens, std::uint32_t eos) {
    const std::string name = std::string("one lane") + (chain_on ? " chain" : " lookup") + (hybrid ? " hybrid" : "") +
                             " eos " + std::to_string(eos);
    const Run ref = reference(kPromptA, max_tokens, eos, chain_on, hybrid);
    SpecBackend be(1, chain_on, hybrid);
    Run got;
    be.log = &got.log;
    CohortDecoder dec(be, spec_config());
    const auto r = request(1, kPromptA, max_tokens, {eos});
    dec.submit(r);
    dec.run();
    auto res = collect(dec);
    got.ids = res[1].ids;
    check(res[1].error.empty(), name + ": no error");
    check(got.ids == ref.ids, name + ": ids == the block continuation");
    check(got.ids == solo(r), name + ": ids == solo");
    check(got.log == ref.log, name + ": every draft, verify and commit == the block continuation");
    check(dec.lane_speculation_steps() == res[1].steps && res[1].steps > 0, name + ": every decode step speculated");
    check(be.shared_rounds == 0, name + ": no cohort round");
    check(res[1].accepted > 0, name + ": drafts were accepted");
    const auto drafts = std::count_if(got.log.begin(), got.log.end(), [](const std::string& e) { return e[0] == 'D'; });
    check(chain_on ? drafts > 0 : drafts == 0, name + ": the model drafter drafted only when attached");
    check(be.released[0] == 1, name + ": lane released once");
}

// 2. A second request admitted mid-run: the speculating lane takes the round at a step boundary and resumes alone.
void test_admit_mid_run(bool ready_on_resume) {
    const std::string name = ready_on_resume ? "admit mid-run" : "admit mid-run, not ready";
    SpecBackend be(2, true, true);
    CohortDecoder dec(be, spec_config());
    auto a = request(1, kPromptA, 140);
    const auto b = request(2, kPromptB, 24);
    std::size_t a_ids = 0;
    a.publish = [&](std::span<const std::uint32_t> ids) {
        a_ids += ids.size();
        return true;
    };
    dec.submit(a);
    while (a_ids < 30 && !dec.idle()) dec.step();
    const std::uint64_t spec_before = dec.lane_speculation_steps();
    check(spec_before > 0, name + ": the lone lane speculated");
    check(be.chain_starts[0] == 1, name + ": the model drafter started once");
    if (!ready_on_resume) be.ready[0] = false;
    dec.submit(b);
    std::uint64_t drafts_at_b_end = 0;
    bool b_done = false;
    std::map<std::uint64_t, CohortResult> res;
    while (!dec.idle()) {
        dec.step();
        res = collect(dec, std::move(res));
        if (!b_done && res.count(2) != 0) {
            b_done = true;
            drafts_at_b_end = be.drafts[0];
        }
    }
    check(res[1].error.empty() && res[2].error.empty(), name + ": no error");
    check(res[1].ids == solo(a), name + ": request 1 ids == solo");
    check(res[2].ids == solo(b), name + ": request 2 ids == solo");
    check(be.shared_rounds > 0, name + ": the two lanes shared cohort rounds");
    check(dec.lane_speculation_suspends() == 1, name + ": one hand-over to the round");
    check(dec.lane_speculation_resumes() == 1, name + ": one resume after the second request ended");
    check(dec.lane_speculation_steps() > spec_before, name + ": the lane speculated again after the resume");
    if (ready_on_resume) {
        check(be.chain_starts[0] == 2, name + ": the model drafter restarted on the resume");
        check(dec.lane_speculation_lookup_only() == 0, name + ": no lookup-only lane");
    } else {
        check(be.chain_starts[0] == 1, name + ": the model drafter never started again");
        check(be.drafts[0] == drafts_at_b_end, name + ": no model draft after the resume");
        check(dec.lane_speculation_lookup_only() == 1, name + ": the lane counted lookup-only");
    }
    check(be.released[0] == 1 && be.released[1] == 1, name + ": both lanes released once");
}

// 4. Two requests from the start (round only); the survivor speculates once it is alone.
void test_survivor_speculates() {
    const std::string name = "survivor";
    SpecBackend be(2, true, false);
    CohortDecoder dec(be, spec_config());
    const auto a = request(1, kPromptA, 18);
    const auto b = request(2, kPromptB, 90);
    dec.submit(a);
    dec.submit(b);
    std::map<std::uint64_t, CohortResult> res;
    while (!dec.idle()) {
        dec.step();
        res = collect(dec, std::move(res));
    }
    check(res[1].ids == solo(a) && res[2].ids == solo(b), name + ": ids == solo");
    check(be.shared_rounds > 0, name + ": the round served both lanes first");
    check(dec.lane_speculation_steps() > 0, name + ": the survivor speculated");
    check(dec.lane_speculation_suspends() == 0 && dec.lane_speculation_resumes() == 0, name + ": no hand-back");
    check(be.chain_starts[0] + be.chain_starts[1] == 1, name + ": only the survivor's drafter started");
    check(res[2].steps > dec.lane_speculation_steps(), name + ": the survivor also ran cohort rounds");
}

// 5. Cancellation while speculating (the cancelled flag, and a publish refusal): ends at a step boundary.
void test_cancel(bool by_publish) {
    const std::string name = by_publish ? "cancel by publish" : "cancel";
    SpecBackend be(1, true, true);
    CohortDecoder dec(be, spec_config());
    auto a = request(1, kPromptA, 200);
    std::size_t seen = 0;
    bool cancel = false;
    a.publish = [&](std::span<const std::uint32_t> ids) {
        seen += ids.size();
        if (seen >= 25) cancel = true;
        return !(by_publish && cancel);
    };
    a.cancelled = [&] { return cancel; };
    dec.submit(a);
    dec.run();
    auto res = collect(dec);
    const auto& r = res[1];
    check(r.cancelled && r.finish == "cancelled", name + ": the request ended cancelled");
    check(dec.lane_speculation_steps() > 0, name + ": it was speculating");
    check(r.ids.size() >= 25 && r.ids.size() < 200, name + ": it ended early");
    const auto full = solo(a);
    check(std::equal(r.ids.begin(), r.ids.end(), full.begin()), name + ": ids are a prefix of the solo run");
    check(be.released[0] == 1, name + ": the lane was released (with no open window)");
}

// Defaults: lane speculation off keeps the round (no speculator hook is asked).
void test_default_off() {
    SpecBackend be(1, true, true);
    CohortDecoder dec(be, CohortConfig{});
    const auto a = request(1, kPromptA, 40);
    dec.submit(a);
    dec.run();
    auto res = collect(dec);
    check(res[1].ids == solo(a), "off: ids == solo");
    check(dec.lane_speculation_steps() == 0 && be.chain_starts[0] == 0, "off: no speculator");
}

}  // namespace

int main() {
    test_one_lane(true, false, 120, 11);
    test_one_lane(true, true, 120, 11);
    test_one_lane(false, false, 120, 11);
    test_one_lane(true, true, 90, 22);  // the stop id (also the command line's EOS) is reached
    test_admit_mid_run(true);
    test_admit_mid_run(false);
    test_survivor_speculates();
    test_cancel(false);
    test_cancel(true);
    test_default_off();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cohort lane speculation: PASS\n";
    return 0;
}
