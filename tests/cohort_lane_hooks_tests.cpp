// Host-only tests for the optional CohortBackend hooks of the shared multi-request decoder
// (strix/serve/cohort_decode.hpp): device drafts, retained prefixes (restore / capture), grouped prompts and the
// per-lane selection transaction. Invariant: every request's ids equal its solo run (with the selection policy applied
// row by row), whatever the hooks propose, restore or group; the backend's committed context is the request's.
#include "strix/serve/cohort_decode.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
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

constexpr std::uint32_t kVocab = 29;

std::uint32_t next_id(const std::vector<std::uint32_t>& ctx) {
    const std::uint32_t a = ctx.empty() ? 0U : ctx.back();
    const std::uint32_t parity = static_cast<std::uint32_t>(ctx.size() % 2U);  // position enters: a wrong commit shows
    return (a * 7U + 3U + parity * (a % 3U)) % kVocab;
}

// The solo run, with an optional forced output (index -> token) applied as a selection policy would.
std::vector<std::uint32_t> solo(const CohortRequest& r, const std::map<std::size_t, std::uint32_t>& forced = {}) {
    std::vector<std::uint32_t> ctx = r.prompt, out;
    while (out.size() < r.max_tokens) {
        std::uint32_t t = next_id(ctx);
        if (const auto it = forced.find(out.size()); it != forced.end()) t = it->second;
        out.push_back(t);
        ctx.push_back(t);
        if (std::find(r.stop.begin(), r.stop.end(), t) != r.stop.end()) break;
    }
    return out;
}

class HookBackend final : public CohortBackend {
public:
    HookBackend(std::uint32_t lanes, std::uint32_t rows)
        : lanes_(lanes), rows_(rows), ctx_(lanes), decoded_(lanes, false), logits_(kVocab) {}
    std::uint32_t lanes() const override { return lanes_; }
    std::uint32_t rows_per_forward() const override { return rows_; }
    std::uint32_t max_positions() const override { return 4096; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        check(window_.empty(), "prefill with an open window");
        if (first) ctx_.at(lane).clear();
        ctx_.at(lane).insert(ctx_[lane].end(), tokens.begin(), tokens.end());
        decoded_.at(lane) = false;
        prompt_rows += tokens.size();
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
            final_[g.lane] = top[base + g.rows - 1U];
            base += g.rows;
        }
        ++forwards;
    }
    bool supports_final_logits() const override { return true; }
    std::span<const float> read_final_logits(std::uint32_t lane) override {
        std::fill(logits_.begin(), logits_.end(), 0.0F);
        logits_.at(final_.at(lane)) = 1.0F;
        ++logit_reads;
        return logits_;
    }
    void commit(std::span<const std::uint32_t> accept) override {
        check(accept.size() == window_.size(), "one keep count per group");
        for (std::size_t g = 0; g < window_.size(); ++g) {
            check(accept[g] >= 1 && accept[g] <= window_[g].second.size(), "keep out of range");
            auto& c = ctx_.at(window_[g].first);
            c.insert(c.end(), window_[g].second.begin(), window_[g].second.begin() + accept[g]);
            decoded_.at(window_[g].first) = true;
        }
        window_.clear();
    }
    void release(std::uint32_t lane) override {
        ctx_.at(lane).clear();
        decoded_.at(lane) = false;
    }

    // Device drafts: the true continuation of the lane's committed context, with row `wrong` corrupted.
    bool device_drafts() const override { return drafts_on; }
    void propose(std::uint32_t lane, std::uint32_t room, std::vector<std::uint32_t>& tokens,
                 std::vector<double>& probability) override {
        tokens.clear();
        probability.clear();
        ++proposals;
        if (!decoded_.at(lane)) return;  // like a device drafter: no proposal before the lane's first verify
        // the pending token is the last verified output: the target's choice after the committed context
        std::vector<std::uint32_t> c = ctx_.at(lane);
        c.push_back(next_id(c));
        for (std::uint32_t r = 0; r < std::min(room, 4U); ++r) {
            const std::uint32_t truth = next_id(c);
            tokens.push_back(r == wrong ? (truth + 1U) % kVocab : truth);
            probability.push_back(0.9);
            c.push_back(truth);
        }
    }

    std::uint64_t restore(std::uint32_t lane, std::span<const std::uint32_t> prompt) override {
        std::size_t best = 0;
        const std::vector<std::uint32_t>* from = nullptr;
        for (const auto& kept : retained) {
            std::size_t n = 0;
            while (n < kept.size() && n + 1U < prompt.size() && kept[n] == prompt[n]) ++n;
            if (n > best) best = n, from = &kept;
        }
        if (from == nullptr) return 0;
        ctx_.at(lane).assign(from->begin(), from->begin() + static_cast<std::ptrdiff_t>(best));
        decoded_.at(lane) = false;
        return best;
    }
    void capture(std::uint32_t lane, std::span<const std::uint32_t> committed) override {
        const auto& c = ctx_.at(lane);
        check(committed.size() <= c.size() && std::equal(committed.begin(), committed.end(), c.begin()),
              "captured tokens are the lane's committed context");
        retained.emplace_back(committed.begin(), committed.end());
    }

    PrefillGroupCapacity prompt_group_capacity() const override {
        PrefillGroupCapacity capacity;
        capacity.members = group_members;
        return capacity;
    }
    void prefill_group(std::span<const CohortPrompt> members) override {
        check(members.size() >= 2, "a group pass has company");
        group_sizes.push_back(members.size());
        for (const auto& m : members) {
            check(m.first, "group members start their prompt");
            ctx_.at(m.lane).clear();
            decoded_.at(m.lane) = false;
            ctx_[m.lane].insert(ctx_[m.lane].end(), m.tokens.begin(), m.tokens.end());
            prompt_rows += m.tokens.size();
        }
    }

    bool drafts_on = false;
    std::uint32_t wrong = 2;
    std::uint64_t group_members = 1;
    std::vector<std::vector<std::uint32_t>> retained;
    std::vector<std::size_t> group_sizes;
    std::uint64_t forwards = 0, prompt_rows = 0, proposals = 0, logit_reads = 0;

private:
    std::uint32_t lanes_, rows_;
    std::vector<std::vector<std::uint32_t>> ctx_;
    std::vector<bool> decoded_;
    std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>> window_;
    std::map<std::uint32_t, std::uint32_t> final_;
    std::vector<float> logits_;
};

CohortRequest request(std::uint64_t id, std::uint32_t len, std::uint32_t max_tokens, std::uint32_t seed) {
    CohortRequest r;
    r.id = id;
    for (std::uint32_t k = 0; k < len; ++k) r.prompt.push_back((seed * 13U + k * 5U) % kVocab);
    r.max_tokens = max_tokens;
    return r;
}

std::map<std::uint64_t, CohortResult> run_all(CohortDecoder& dec) {
    dec.run();
    std::map<std::uint64_t, CohortResult> out;
    for (auto& r : dec.take_finished()) out[r.id] = std::move(r);
    return out;
}

void test_device_drafts(bool context) {
    const std::string name = std::string("device drafts (context ") + (context ? "on" : "off") + ")";
    HookBackend be(4, 32);
    be.drafts_on = true;
    CohortConfig cfg;
    cfg.context_draft = context;
    cfg.prompt_slice = 8;
    cfg.prior = {10.0, 0.1, 0.1, 0.0};
    CohortDecoder dec(be, cfg);
    std::vector<CohortRequest> reqs;
    for (std::uint32_t i = 0; i < 6; ++i) reqs.push_back(request(10 + i, 4 + 3 * i, 30 + 5 * i, i));
    for (const auto& r : reqs) dec.submit(r);
    auto res = run_all(dec);
    std::uint64_t accepted = 0, tokens = 0;
    for (const auto& q : reqs) {
        const auto& r = res[q.id];
        check(r.error.empty() && r.ids == solo(q), name + ": ids == solo for " + std::to_string(q.id));
        accepted += r.accepted;
        tokens += r.ids.size();
    }
    check(be.proposals > 0, name + ": the backend was asked for proposals");
    check(accepted > 0 && be.forwards < tokens, name + ": device drafts accepted, fewer forwards than tokens");
}

void test_retained_prefix() {
    HookBackend be(2, 16);
    CohortConfig cfg;
    cfg.prompt_slice = 8;
    CohortDecoder dec(be, cfg);
    const CohortRequest first = request(1, 20, 12, 3);
    dec.submit(first);
    auto a = run_all(dec);
    check(a[1].error.empty() && a[1].ids == solo(first), "prefix: first request == solo");
    check(a[1].restored == 0, "prefix: nothing to restore at first");
    check(be.retained.size() == 1 && be.retained[0].size() > first.prompt.size(), "prefix: the finished request was captured");
    // the follow-up turn: the first conversation, then a new user turn
    CohortRequest second;
    second.id = 2;
    second.prompt = first.prompt;
    second.prompt.insert(second.prompt.end(), a[1].ids.begin(), a[1].ids.end());
    for (std::uint32_t k = 0; k < 5; ++k) second.prompt.push_back((k * 3U + 1U) % kVocab);
    second.max_tokens = 10;
    const auto rows_before = be.prompt_rows;
    std::uint64_t progress = 0;
    second.prefilled = [&](std::uint64_t rows) { progress = rows; };
    dec.submit(second);
    auto b = run_all(dec);
    check(b[2].error.empty() && b[2].ids == solo(second), "prefix: restored request == solo");
    check(b[2].restored == be.retained[0].size(), "prefix: the whole captured context was restored");
    check(be.prompt_rows - rows_before == second.prompt.size() - 1U - b[2].restored, "prefix: only the new rows were prefilled");
    check(progress == second.prompt.size() - 1U, "prefix: progress counts restored rows");
    // a request that does not retain is not captured; one that does not reuse starts cold
    CohortRequest third = second;
    third.id = 3;
    third.reuse = false;
    third.retain = false;
    const auto kept = be.retained.size();
    dec.submit(third);
    auto c = run_all(dec);
    check(c[3].restored == 0 && c[3].ids == solo(third), "prefix: reuse off starts cold, ids == solo");
    check(be.retained.size() == kept, "prefix: retain off is not captured");
}

void test_grouped_prompts() {
    HookBackend be(8, 64);
    be.group_members = 8;
    CohortConfig cfg;
    cfg.prompt_slice = 32;
    CohortDecoder dec(be, cfg);
    std::vector<CohortRequest> reqs;
    for (std::uint32_t i = 0; i < 6; ++i) reqs.push_back(request(20 + i, 5 + i, 12, i + 1U));
    reqs.push_back(request(40, 70, 8, 9));  // long: runs alone, in slices
    for (const auto& r : reqs) dec.submit(r);
    auto res = run_all(dec);
    for (const auto& q : reqs)
        check(res[q.id].error.empty() && res[q.id].ids == solo(q), "groups: ids == solo for " + std::to_string(q.id));
    check(dec.prompt_groups() >= 1 && dec.grouped_prompts() >= 2, "groups: short prompts shared a pass");
    for (const auto s : be.group_sizes) check(s <= 8, "groups: members within the capacity");
    std::uint64_t rows = 0;
    for (const auto& q : reqs) rows += q.prompt.size() - 1U;
    check(be.prompt_rows == rows, "groups: every prompt row ran exactly once");
    // no capacity: no group pass
    HookBackend solo_be(8, 64);
    CohortDecoder solo_dec(solo_be, cfg);
    for (const auto& r : reqs) solo_dec.submit(r);
    auto solo_res = run_all(solo_dec);
    check(solo_dec.prompt_groups() == 0 && solo_be.group_sizes.empty(), "groups: off without backend capacity");
    for (const auto& q : reqs) check(solo_res[q.id].ids == res[q.id].ids, "groups: same ids with and without groups");
}

// A selection policy that forces `token` at output index `at` (a thinking-budget closure), committed by receipt.
class ForceAt final : public strix::MtpSelectionTransaction {
public:
    ForceAt(std::size_t at, std::uint32_t token) : at_(at), token_(token) {}
    void reset() override { branch_ = committed_; }
    void discard() noexcept override { branch_ = committed_; }
    std::uint32_t select(std::span<const float> row) override {
        const auto best = static_cast<std::uint32_t>(std::max_element(row.begin(), row.end()) - row.begin());
        return decide(best);
    }
    std::optional<std::uint32_t> select_known(std::uint32_t token, std::uint32_t) override { return decide(token); }
    std::unique_ptr<strix::MtpSelectionReceipt> seal() override {
        struct Receipt final : strix::MtpSelectionReceipt {
            ForceAt* owner;
            std::size_t to, n;
            Receipt(ForceAt* o, std::size_t t, std::size_t count) : owner(o), to(t), n(count) {}
            void publish() override { owner->committed_ = to; }
            std::size_t size() const noexcept override { return n; }
        };
        return std::make_unique<Receipt>(this, branch_, branch_ - committed_);
    }
    std::size_t committed_ = 0;

private:
    std::uint32_t decide(std::uint32_t greedy) { return branch_++ == at_ ? token_ : greedy; }
    std::size_t at_;
    std::uint32_t token_;
    std::size_t branch_ = 0;
};

void test_selection(bool reads_logits) {
    const std::string name = std::string("selection (") + (reads_logits ? "logits" : "greedy window") + ")";
    HookBackend be(3, 32);
    be.drafts_on = true;
    be.wrong = 9;  // every device draft is right: the forced token must cut the window
    CohortConfig cfg;
    cfg.prompt_slice = 8;
    CohortDecoder dec(be, cfg);
    std::vector<CohortRequest> reqs;
    std::vector<std::map<std::size_t, std::uint32_t>> forced;
    for (std::uint32_t i = 0; i < 3; ++i) {
        auto r = request(50 + i, 6 + i, 16, i + 5U);
        if (i != 1) {
            const std::size_t at = 3 + i;
            const std::uint32_t token = (solo(r)[at] + 1U) % kVocab;  // differs from the greedy choice there
            r.selection = std::make_shared<ForceAt>(at, token);
            r.selection_reads_logits = reads_logits;
            forced.push_back({{at, token}});
        } else {
            forced.push_back({});
        }
        reqs.push_back(r);
    }
    for (const auto& r : reqs) dec.submit(r);
    auto res = run_all(dec);
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        const auto& r = res[reqs[i].id];
        check(r.error.empty(), name + ": no error for " + std::to_string(reqs[i].id));
        check(r.ids == solo(reqs[i], forced[i]), name + ": ids == solo under the policy for " + std::to_string(reqs[i].id));
        if (reqs[i].selection) {
            const auto* f = static_cast<const ForceAt*>(reqs[i].selection.get());
            check(f->committed_ >= r.ids.size(), name + ": receipts published every emitted row");
        }
    }
    if (reads_logits) check(be.logit_reads > 0, name + ": the policy read final logits");
    // a sampled request with a selection is refused; a policy that cannot decide from argmax ends with its error
    CohortRequest bad = request(60, 5, 4, 1);
    bad.selection = std::make_shared<ForceAt>(0, 1);
    bad.sampling.temperature = 0.7F;
    dec.submit(bad);
    auto refused = run_all(dec);
    check(!refused[60].error.empty() && refused[60].ids.empty(), name + ": sampled selection refused");
}

class Refuses final : public strix::MtpSelectionTransaction {
public:
    void reset() override {}
    void discard() noexcept override { ++discards; }
    std::uint32_t select(std::span<const float>) override { throw strix::MtpSelectionError("needs a full mask"); }
    std::optional<std::uint32_t> select_known(std::uint32_t, std::uint32_t) override { return std::nullopt; }
    std::unique_ptr<strix::MtpSelectionReceipt> seal() override { return nullptr; }
    int discards = 0;
};

void test_selection_refusal() {
    HookBackend be(2, 16);
    CohortDecoder dec(be, CohortConfig{});
    CohortRequest plain = request(70, 6, 10, 2), bad = request(71, 6, 10, 4);
    auto policy = std::make_shared<Refuses>();
    bad.selection = policy;
    dec.submit(plain);
    dec.submit(bad);
    auto res = run_all(dec);
    check(res[70].error.empty() && res[70].ids == solo(plain), "refusal: the other lane is unaffected");
    check(res[71].error.find("selection") != std::string::npos && res[71].ids.empty(), "refusal: the request ends with the policy error");
    check(policy->discards >= 1, "refusal: the attempt was discarded");
}

}  // namespace

int main() {
    test_device_drafts(false);
    test_device_drafts(true);
    test_retained_prefix();
    test_grouped_prompts();
    test_selection(false);
    test_selection(true);
    test_selection_refusal();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cohort lane hooks: PASS\n";
    return 0;
}
