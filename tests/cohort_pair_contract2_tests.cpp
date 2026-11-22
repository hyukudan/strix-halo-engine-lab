// Pair-engine contract, round 2, of the shared multi-request decoder (strix/serve/cohort_decode.hpp), on the host.
// Invariants:
// - bind_lane: called once per admitted request (reuse or not) with its id and traits, before restore / prefill.
// - context_tokens: the request runs to min(max_tokens, ceiling) ids, its ids are the solo run's prefix, finish is
//   "context" when the ceiling ended it and "length" on a tie; the lane never holds more than max_positions rows; a
//   ceiling that does not fit the lane (prompt + ceiling - 1 > max_positions) is refused.
// - whole-prompt restore (prefill_emits_first): no prefill runs, the first id is the restored row's prefill head, and
//   ids equal the solo run; restoring the whole prompt without prefill_emits_first is refused.
// - control reaches the backend; finish reasons stop / length / cancelled / error.
// - A request that sets none of these runs exactly as before (refusal bound and ids).
#include "strix/serve/cohort_decode.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool value, const std::string& label) {
    if (!value) {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}

constexpr std::uint32_t kVocab = 23;

std::uint32_t next_id(const std::vector<std::uint32_t>& ctx) {
    const std::uint32_t a = ctx.empty() ? 0U : ctx.back();
    return (a * 5U + 1U + static_cast<std::uint32_t>(ctx.size() % 3U)) % kVocab;
}
std::uint32_t prefill_id(const std::vector<std::uint32_t>& ctx) { return (next_id(ctx) + 7U) % kVocab; }

std::vector<std::uint32_t> solo(const std::vector<std::uint32_t>& prompt, std::size_t n, bool emits_first,
                                const std::vector<std::uint32_t>& stop = {}) {
    std::vector<std::uint32_t> ctx = prompt, out;
    while (out.size() < n) {
        const std::uint32_t t = out.empty() && emits_first ? prefill_id(ctx) : next_id(ctx);
        out.push_back(t);
        ctx.push_back(t);
        if (std::find(stop.begin(), stop.end(), t) != stop.end()) break;
    }
    return out;
}

std::vector<std::uint32_t> prompt_of(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint32_t> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<std::uint32_t>((seed + 3U * i) % kVocab);
    return p;
}

class Backend final : public CohortBackend {
public:
    Backend(std::uint32_t positions, bool emits_first, bool restore_whole = false)
        : positions_(positions), emits_first_(emits_first), restore_whole_(restore_whole), ctx_(4), window_(4),
          after_prefill_(4, false) {}
    std::uint32_t lanes() const override { return 4; }
    std::uint32_t rows_per_forward() const override { return 8; }
    std::uint32_t max_positions() const override { return positions_; }
    bool prefill_emits_first() const override { return emits_first_; }
    void bind_lane(std::uint32_t lane, std::uint64_t id, std::span<const std::uint32_t> traits) override {
        std::string e = "B" + std::to_string(lane) + ":" + std::to_string(id) + ":";
        for (const auto t : traits) e += std::to_string(t) + ",";
        log.push_back(e);
    }
    void control(std::span<const std::uint32_t> words) override { controls.assign(words.begin(), words.end()); }
    std::uint64_t restore(std::uint32_t lane, std::span<const std::uint32_t> prompt) override {
        log.push_back("S" + std::to_string(lane));
        if (!restore_whole_) return 0;
        ctx_.at(lane).assign(prompt.begin(), prompt.end());
        after_prefill_[lane] = true;
        return prompt.size();
    }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) ctx_.at(lane).clear();
        ctx_[lane].insert(ctx_[lane].end(), tokens.begin(), tokens.end());
        after_prefill_[lane] = true;
        ++prefills;
        if (ctx_[lane].size() > positions_) ++overflow;
        log.push_back("P" + std::to_string(lane));
    }
    std::uint32_t prefill_top(std::uint32_t lane) override {
        if (!after_prefill_.at(lane)) throw std::logic_error("prefill_top outside a prefill");
        return prefill_id(ctx_[lane]);
    }
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        std::size_t at = 0;
        groups_.assign(groups.begin(), groups.end());
        for (const auto& g : groups) {
            after_prefill_[g.lane] = false;
            window_[g.lane] = ctx_[g.lane];
            for (std::uint32_t r = 0; r < g.rows; ++r, ++at) {
                window_[g.lane].push_back(tokens[at]);
                top[at] = next_id(window_[g.lane]);
            }
            if (window_[g.lane].size() > positions_) ++overflow;  // a row past the lane
        }
    }
    void commit(std::span<const std::uint32_t> accept) override {
        for (std::size_t g = 0; g < accept.size(); ++g) {
            const auto lane = groups_.at(g).lane;
            window_[lane].resize(ctx_[lane].size() + accept[g]);
            ctx_[lane] = window_[lane];
        }
    }
    void release(std::uint32_t lane) override { log.push_back("R" + std::to_string(lane)); }

    std::vector<std::string> log;
    std::vector<std::uint32_t> controls;
    int prefills = 0, overflow = 0;

private:
    std::uint32_t positions_;
    bool emits_first_, restore_whole_;
    std::vector<std::vector<std::uint32_t>> ctx_, window_;
    std::vector<bool> after_prefill_;
    std::vector<CohortGroup> groups_;
};

CohortResult run_one(Backend& be, CohortRequest r) {
    CohortDecoder d(be, CohortConfig{});
    d.set_fake_clock(1.0);
    d.submit(std::move(r));
    d.run();
    auto done = d.take_finished();
    if (done.size() != 1) throw std::logic_error("one result expected");
    return done.front();
}

void bind_lane_order() {
    Backend be(4096, true);
    CohortDecoder d(be, CohortConfig{});
    d.set_fake_clock(1.0);
    CohortRequest a;
    a.id = (std::uint64_t{7} << 32) | 9U;
    a.prompt = prompt_of(5, 1);
    a.max_tokens = 3;
    a.traits = {11, 12, 0xFFFFFFFFU};
    CohortRequest b;
    b.id = 2;
    b.prompt = prompt_of(4, 2);
    b.max_tokens = 2;
    b.reuse = false;
    d.submit(a);
    d.submit(b);
    d.run();
    const std::string a_bind = "B" + std::string("1:") + std::to_string(a.id) + ":11,12,4294967295,";
    int binds = 0;
    for (std::size_t i = 0; i < be.log.size(); ++i) {
        if (be.log[i][0] != 'B') continue;
        ++binds;
        // no restore / prefill of that lane comes before its bind
        const char lane = be.log[i][1];
        for (std::size_t j = 0; j < i; ++j)
            check(!((be.log[j][0] == 'S' || be.log[j][0] == 'P') && be.log[j][1] == lane), "bind precedes restore/prefill");
    }
    check(binds == 2, "one bind per admitted request");
    const bool b_bound = std::find(be.log.begin(), be.log.end(), std::string("B0:2:")) != be.log.end();
    const bool a_bound = std::find(be.log.begin(), be.log.end(), a_bind) != be.log.end();
    check(b_bound, "bind without reuse (empty traits)");
    check(a_bound, "bind carries the 64-bit id and the traits");
    for (const auto& r : d.take_finished()) check(r.finish == "length", "bind: finish length");
    const std::uint32_t words[2] = {5, 6};
    d.control(words);
    check(be.controls == std::vector<std::uint32_t>({5, 6}), "control reaches the backend");
}

void context_ceiling(bool emits_first) {
    const std::string tag = emits_first ? " (emits first)" : " (decode first)";
    const std::uint32_t cap = 16;
    const auto prompt = prompt_of(10, 3);
    const std::uint32_t ceiling = cap - 10U + 1U;  // the last id is never a row
    {
        Backend be(cap, emits_first);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = 40;
        r.context_tokens = ceiling;
        const auto res = run_one(be, r);
        check(res.error.empty(), "ceiling admitted" + tag);
        check(res.ids == solo(prompt, ceiling, emits_first), "ceiling ids = solo prefix" + tag);
        check(res.finish == "context", "ceiling finish context" + tag);
        check(be.overflow == 0, "no row past the lane" + tag);
    }
    {
        Backend be(cap, emits_first);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = ceiling;
        r.context_tokens = ceiling;
        check(run_one(be, r).finish == "length", "tie is length" + tag);
    }
    {
        Backend be(cap, emits_first);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = 3;
        r.context_tokens = ceiling;
        const auto res = run_one(be, r);
        check(res.finish == "length" && res.ids.size() == 3, "max_tokens below the ceiling" + tag);
    }
    {
        Backend be(cap, emits_first);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = 40;
        r.context_tokens = ceiling + 1U;
        const auto res = run_one(be, r);
        check(res.error == "context ceiling longer than a lane" && res.finish == "error", "ceiling past the lane refused" + tag);
    }
    {
        // unset ceiling: the old refusal bound (prompt + max_tokens <= max_positions) and the old ids
        Backend be(cap, emits_first);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = cap - 10U + 1U;
        check(run_one(be, r).error == "request longer than a lane", "unset ceiling keeps the old bound" + tag);
        Backend be2(cap, emits_first);
        r.max_tokens = cap - 10U;
        const auto res = run_one(be2, r);
        check(res.ids == solo(prompt, cap - 10U, emits_first) && res.finish == "length", "unset ceiling ids" + tag);
    }
}

void whole_restore() {
    const auto prompt = prompt_of(9, 4);
    {
        Backend be(4096, true, true);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = 6;
        const auto res = run_one(be, r);
        check(res.error.empty() && res.restored == prompt.size(), "whole prompt restored");
        check(be.prefills == 0, "no prefill after a whole restore");
        check(res.first_from_prefill, "first id from the restored row");
        check(res.ids == solo(prompt, 6, true), "whole restore ids = solo");
    }
    {
        Backend be(4096, true, true);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = 1;
        const auto res = run_one(be, r);
        check(res.ids == solo(prompt, 1, true) && res.finish == "length", "whole restore, one id");
    }
    {
        Backend be(4096, false, true);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = 2;
        bool threw = false;
        try {
            (void)run_one(be, r);
        } catch (const std::logic_error&) {
            threw = true;
        }
        check(threw, "whole restore without prefill_emits_first refused");
    }
}

void finish_reasons() {
    const auto prompt = prompt_of(6, 5);
    const auto ref = solo(prompt, 8, true);
    {
        Backend be(4096, true);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = 8;
        r.stop = {ref[2]};
        const auto res = run_one(be, r);
        check(res.finish == "stop" && res.ids == solo(prompt, 8, true, r.stop), "finish stop");
    }
    {
        Backend be(4096, true);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = 8;
        r.cancelled = [] { return true; };
        check(run_one(be, r).finish == "cancelled", "finish cancelled");
    }
    {
        Backend be(4096, true);
        CohortRequest r;
        r.prompt = prompt;
        r.max_tokens = 0;
        check(run_one(be, r).finish == "error", "finish error");
    }
}
}  // namespace

int main() {
    bind_lane_order();
    context_ceiling(true);
    context_ceiling(false);
    whole_restore();
    finish_reasons();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cohort pair contract 2: ok\n";
    return 0;
}
