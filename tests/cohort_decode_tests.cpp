// Host-only tests for the shared multi-request decoder (strix/serve/cohort_decode.hpp) over a fake engine.
// Invariant: every request's ids equal its solo greedy run, whatever joins, leaves, drafts or row plans happen.
#include "strix/serve/cohort_decode.hpp"

#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool value, const std::string& label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}

// Deterministic "model": the next id depends only on the lane's own context (batch invariant by construction).
// A small vocabulary makes the sequences periodic, so the context drafter finds copies and drafts get accepted.
std::uint32_t next_id(const std::vector<std::uint32_t>& ctx, std::uint32_t vocab) {
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

}  // namespace

int main() {
    run_identity(false, 4, 8, "plain 4 lanes");
    run_identity(true, 4, 8, "draft 4 lanes 8 rows");
    run_identity(true, 8, 32, "draft 8 lanes 32 rows");
    run_identity(true, 16, 8, "draft lanes capped by rows");
    run_identity(true, 1, 16, "draft 1 lane (solo path)");
    test_admission_bytes();
    test_short_first();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cohort decode tests passed\n";
    return 0;
}
