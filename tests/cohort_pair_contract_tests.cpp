// Pair-engine contract of the shared multi-request decoder (strix/serve/cohort_decode.hpp, replicated_cohort.hpp), on
// the host. Invariants:
// - prefill_emits_first: the whole prompt is read by prefill and the first id is the PREFILL's own (here a prefill head
//   that differs from the decode rule, so a first id decided by a decode row would fail), greedy, sampled and grouped;
//   a request with max_tokens 1 runs no verify; a one-token prompt is admitted only then.
// - max_window: no verify group of a lane exceeds the backend's cap, and drafts still run once the cap lifts.
// - sampler_options reach the lane's sampler (acceptance in the receipt).
// - collective_reads: rank 1 makes exactly rank 0's read_final_logits / propose calls (same order), both ranks end in
//   the same state, ids equal the solo run; without the option rank 1 never sees them, and a follower that was not
//   configured for them refuses such a message with pair::ProtocolError.
// - A backend that binds none of these hooks sees the same calls as before (default false / unbounded).
#include "strix/pair/lockstep.hpp"
#include "strix/serve/replicated_cohort.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

// Decode rule: the next id depends only on the lane's own context.
std::uint32_t next_id(const std::vector<std::uint32_t>& ctx) {
    const std::uint32_t a = ctx.empty() ? 0U : ctx.back();
    return (a * 5U + 1U + static_cast<std::uint32_t>(ctx.size() % 3U)) % kVocab;
}
// Prefill head: a different rule for the prompt's last row (a prefill route that is not bit-equal to a decode row).
std::uint32_t prefill_id(const std::vector<std::uint32_t>& ctx) { return (next_id(ctx) + 7U) % kVocab; }

std::vector<std::uint32_t> solo(const CohortRequest& r, bool emits_first) {
    std::vector<std::uint32_t> ctx = r.prompt, out;
    while (out.size() < r.max_tokens) {
        const std::uint32_t t = out.empty() && emits_first ? prefill_id(ctx) : next_id(ctx);
        out.push_back(t);
        ctx.push_back(t);
        if (std::find(r.stop.begin(), r.stop.end(), t) != r.stop.end()) break;
    }
    return out;
}

std::string join(std::span<const std::uint32_t> v) {
    std::string s;
    for (const auto x : v) s += std::to_string(x) + ',';
    return s;
}

// Lanes hold committed contexts; verify appends a pending window, commit keeps its leading rows. Every call that a
// replica must repeat is logged.
class FakeBackend final : public CohortBackend {
public:
    FakeBackend(std::uint32_t lanes, bool emits_first, std::uint32_t cap_until = 0)
        : lanes_(lanes), emits_first_(emits_first), cap_until_(cap_until), ctx_(lanes), pending_(lanes),
          after_prefill_(lanes, false), window_(lanes) {}
    std::uint32_t lanes() const override { return lanes_; }
    std::uint32_t rows_per_forward() const override { return 8; }
    std::uint32_t max_positions() const override { return 4096; }
    bool prefill_emits_first() const override { return emits_first_; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) ctx_.at(lane).clear();
        ctx_[lane].insert(ctx_[lane].end(), tokens.begin(), tokens.end());
        after_prefill_[lane] = true;
        pending_[lane] = prefill_id(ctx_[lane]);
        log.push_back("P" + std::to_string(lane) + (first ? "f:" : ":") + join(tokens));
    }
    std::uint32_t prefill_top(std::uint32_t lane) override {
        if (!emits_first_ || !after_prefill_.at(lane)) throw std::logic_error("prefill_top outside a prefill");
        ++prefill_tops;
        return prefill_id(ctx_[lane]);
    }
    std::uint32_t max_window(std::uint32_t lane) const override {
        return ctx_.at(lane).size() < cap_until_ ? 1U : 8U;
    }
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        std::size_t at = 0;
        std::string entry = "V";
        last_groups_.assign(groups.begin(), groups.end());
        for (const auto& g : groups) {
            if (g.rows > max_window(g.lane)) ++cap_violations;
            if (g.rows > 1) ++multi_row_groups;
            after_prefill_[g.lane] = false;
            window_[g.lane] = ctx_[g.lane];
            for (std::uint32_t r = 0; r < g.rows; ++r, ++at) {
                window_[g.lane].push_back(tokens[at]);
                top[at] = next_id(window_[g.lane]);
            }
            entry += std::to_string(g.lane) + "x" + std::to_string(g.rows) + ";";
        }
        ++verifies;
        log.push_back(entry + join(tokens));
    }
    bool supports_final_logits() const override { return true; }
    std::span<const float> read_final_logits(std::uint32_t lane) override {
        // a sharp peak at the id the lane's last row would emit: the sampler draws it with certainty
        const std::uint32_t id = after_prefill_.at(lane) ? prefill_id(ctx_[lane]) : next_id(window_[lane]);
        logits_.assign(kVocab, 0.0F);
        logits_[id] = 80.0F;
        ++reads;
        log.push_back("L" + std::to_string(lane) + (after_prefill_[lane] ? "p" : "d"));
        return logits_;
    }
    void commit(std::span<const std::uint32_t> accept) override {
        for (std::size_t g = 0; g < accept.size(); ++g) {
            const auto lane = last_groups_.at(g).lane;
            const std::size_t keep = ctx_[lane].size() + accept[g];
            window_[lane].resize(keep);
            ctx_[lane] = window_[lane];
            pending_[lane] = next_id(ctx_[lane]);
        }
        log.push_back("C" + join(accept));
    }
    void release(std::uint32_t lane) override { log.push_back("R" + std::to_string(lane)); }
    bool device_drafts() const override { return drafts_on; }
    void propose(std::uint32_t lane, std::uint32_t room, std::vector<std::uint32_t>& tokens,
                 std::vector<double>& probability) override {
        // a perfect drafter: continues the lane from its pending token with the decode rule
        tokens.clear();
        probability.clear();
        std::vector<std::uint32_t> ctx = ctx_.at(lane);
        ctx.push_back(pending_[lane]);
        for (std::uint32_t r = 0; r < std::min(room, 3U); ++r) {
            tokens.push_back(next_id(ctx));
            probability.push_back(0.95);
            ctx.push_back(tokens.back());
        }
        log.push_back("D" + std::to_string(lane) + ":" + std::to_string(room));
    }
    PrefillGroupCapacity prompt_group_capacity() const override {
        PrefillGroupCapacity c;
        c.members = groups_on ? 4U : 1U;
        return c;
    }
    void prefill_group(std::span<const CohortPrompt> members) override {
        ++group_passes;
        for (const auto& m : members) prefill(m.lane, m.tokens, m.first);
    }

    std::vector<std::string> log;
    std::uint64_t verifies = 0, reads = 0, prefill_tops = 0, cap_violations = 0, multi_row_groups = 0, group_passes = 0;
    bool drafts_on = false, groups_on = false;
    const std::vector<std::vector<std::uint32_t>>& contexts() const { return ctx_; }

private:
    std::uint32_t lanes_;
    bool emits_first_;
    std::uint32_t cap_until_;
    std::vector<std::vector<std::uint32_t>> ctx_;
    std::vector<std::uint32_t> pending_;
    std::vector<bool> after_prefill_;
    std::vector<std::vector<std::uint32_t>> window_;
    std::vector<CohortGroup> last_groups_;
    std::vector<float> logits_;
};

std::vector<CohortRequest> requests(std::uint32_t n, bool sampled) {
    std::vector<CohortRequest> out;
    for (std::uint32_t i = 0; i < n; ++i) {
        CohortRequest r;
        r.id = i;
        for (std::uint32_t t = 0; t < 3U + 2U * i; ++t) r.prompt.push_back((i * 7U + t * 3U) % kVocab);
        r.max_tokens = 6U + i;
        if (sampled) {
            r.sampling.temperature = 0.8F;
            r.sampling.seed = 11U + i;
        }
        out.push_back(r);
    }
    return out;
}

std::vector<CohortResult> run(CohortBackend& backend, const std::vector<CohortRequest>& reqs, CohortConfig config = {}) {
    CohortDecoder d(backend, config);
    d.set_fake_clock(1.0);
    for (const auto& r : reqs) d.submit(r);
    d.run();
    auto done = d.take_finished();
    std::sort(done.begin(), done.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    return done;
}

void check_solo(const std::vector<CohortResult>& done, const std::vector<CohortRequest>& reqs, bool emits_first,
                const std::string& label) {
    check(done.size() == reqs.size(), label + ": every request finished");
    for (std::size_t i = 0; i < done.size() && i < reqs.size(); ++i) {
        check(done[i].error.empty(), label + ": request " + std::to_string(i) + " not refused (" + done[i].error + ")");
        check(done[i].ids == solo(reqs[i], emits_first), label + ": request " + std::to_string(i) + " ids == solo");
        check(done[i].first_from_prefill == emits_first, label + ": first_from_prefill flag");
    }
}

void test_default_unchanged() {
    FakeBackend b(4, false);
    const auto reqs = requests(4, false);
    check_solo(run(b, reqs), reqs, false, "default greedy");
    check(b.prefill_tops == 0, "default: prefill_top never asked");
    check(b.cap_violations == 0, "default: unbounded window");
    FakeBackend one(1, false);
    CohortRequest r;
    r.prompt = {5};
    const auto done = run(one, {r});
    check(done.size() == 1 && !done[0].error.empty(), "default: a one-token prompt is refused");
}

void test_emits_first_greedy() {
    FakeBackend b(4, true);
    b.drafts_on = true;
    const auto reqs = requests(4, false);
    check_solo(run(b, reqs), reqs, true, "emits-first greedy");
    check(b.prefill_tops == 4, "emits-first greedy: one prefill_top per request");
    check(b.multi_row_groups > 0, "emits-first greedy: drafts still verified");
    // the whole prompt went through prefill: the last prompt token is never a verify row of its own
    for (std::uint32_t lane = 0; lane < 4; ++lane) check(!b.contexts()[lane].empty(), "emits-first: lane used");
}

void test_emits_first_single_token_and_max1() {
    FakeBackend b(2, true);
    CohortRequest r;
    r.id = 0;
    r.prompt = {9};
    r.max_tokens = 1;
    const auto done = run(b, {r});
    check(done.size() == 1 && done[0].error.empty(), "emits-first: a one-token prompt is admitted");
    check(done.size() == 1 && done[0].ids == solo(r, true), "emits-first: one-token prompt ids == solo");
    check(b.verifies == 0, "emits-first: max_tokens 1 runs no verify");
    CohortRequest s = r;
    s.max_tokens = 5;
    s.stop = {solo(r, true)[0]};
    FakeBackend c(2, true);
    const auto stopped = run(c, {s});
    check(stopped.size() == 1 && stopped[0].ids.size() == 1, "emits-first: a stop on the first id ends the request");
    check(c.verifies == 0, "emits-first: stop on the first id runs no verify");
}

void test_emits_first_sampled_and_options() {
    FakeBackend b(4, true);
    auto reqs = requests(4, true);
    for (auto& r : reqs) r.sampler_options.acceptance = strix::SampledAcceptance::coupled;
    const auto done = run(b, reqs);
    check_solo(done, reqs, true, "emits-first sampled");
    for (const auto& r : done)
        check(r.sampling_receipt.find("\"acceptance\":\"coupled\"") != std::string::npos,
              "sampler_options reach the sampler (coupled in the receipt)");
    std::uint64_t prefill_reads = 0;
    for (const auto& e : b.log) prefill_reads += e.size() > 1 && e[0] == 'L' && e.back() == 'p' ? 1U : 0U;
    check(prefill_reads == 4, "emits-first sampled: the first id is read from the prefill's logits");
    FakeBackend d(4, false);
    const auto plain = requests(4, true);
    check_solo(run(d, plain), plain, false, "default sampled");
}

void test_emits_first_grouped() {
    FakeBackend b(4, true);
    b.groups_on = true;
    const auto reqs = requests(4, false);
    check_solo(run(b, reqs), reqs, true, "emits-first grouped");
    check(b.group_passes > 0, "emits-first grouped: a group pass ran");
    check(b.prefill_tops == 4, "emits-first grouped: one prefill_top per member");
}

void test_max_window() {
    FakeBackend b(3, false, 12);
    b.drafts_on = true;
    const auto reqs = requests(3, false);
    check_solo(run(b, reqs), reqs, false, "max_window");
    check(b.cap_violations == 0, "max_window: no group above the backend's cap");
    check(b.multi_row_groups > 0, "max_window: drafts run once the cap lifts");
}

// In-memory transport (whole messages, in order, per direction).
struct Wire {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<std::byte>> q[2];
    bool aborted = false;
};

class QueueTransport final : public strix::dist::Transport {
public:
    QueueTransport(std::shared_ptr<Wire> wire, int rank) : wire_(std::move(wire)), rank_(rank) {}
    void send(std::span<const std::byte> message) override {
        std::lock_guard lock(wire_->m);
        if (wire_->aborted) throw std::runtime_error("link aborted");
        wire_->q[1 - rank_].emplace_back(message.begin(), message.end());
        wire_->cv.notify_all();
    }
    void receive(std::span<std::byte> message) override {
        std::unique_lock lock(wire_->m);
        wire_->cv.wait(lock, [&] { return wire_->aborted || !wire_->q[rank_].empty(); });
        if (wire_->q[rank_].empty()) throw std::runtime_error("link aborted");
        auto got = std::move(wire_->q[rank_].front());
        wire_->q[rank_].pop_front();
        if (got.size() != message.size()) throw std::runtime_error("message size mismatch");
        std::memcpy(message.data(), got.data(), got.size());
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        send(out);
        receive(in);
    }
    std::string name() const override { return "test-queue"; }
    bool supports_peer_abort() const noexcept override { return true; }
    void abort(const std::string&) override {
        std::lock_guard lock(wire_->m);
        wire_->aborted = true;
        wire_->cv.notify_all();
    }

private:
    std::shared_ptr<Wire> wire_;
    int rank_;
};

struct PairRun {
    std::vector<CohortResult> done;
    std::string follower_error;
    bool leader_threw = false;
};

PairRun run_pair(FakeBackend& r0, FakeBackend& r1, const std::vector<CohortRequest>& reqs, ReplicaOptions o0,
                 ReplicaOptions o1) {
    auto wire = std::make_shared<Wire>();
    QueueTransport t0(wire, 0), t1(wire, 1);
    PairRun out;
    std::thread follower([&] {
        try {
            (void)follow_cohort(r1, t1, o1);
        } catch (const std::exception& e) {
            out.follower_error = e.what();
        }
    });
    try {
        ReplicatedCohort rep(r0, t0, o0);
        out.done = run(rep, reqs);
        rep.close();
    } catch (const std::exception&) {
        out.leader_threw = true;
    }
    follower.join();
    return out;
}

std::size_t count_kind(const std::vector<std::string>& log, char kind) {
    return static_cast<std::size_t>(std::count_if(log.begin(), log.end(), [&](const auto& e) { return !e.empty() && e[0] == kind; }));
}

void test_collective_reads() {
    ReplicaOptions on;
    on.collective_reads = true;
    // sampled lanes read logits; greedy lanes ask the device drafter: both are calls on rank 1
    auto reqs = requests(3, true);
    const auto greedy = requests(5, false);
    reqs.push_back(greedy[3]);
    reqs.push_back(greedy[4]);
    FakeBackend a0(4, true), a1(4, true);
    a0.drafts_on = a1.drafts_on = true;
    const auto p = run_pair(a0, a1, reqs, on, on);
    check(p.follower_error.empty() && !p.leader_threw, "collective: pair ran (" + p.follower_error + ")");
    check(a0.log == a1.log, "collective: rank 1 made exactly rank 0's calls");
    check(a0.contexts() == a1.contexts(), "collective: both ranks end in the same state");
    check(count_kind(a1.log, 'L') > 0 && count_kind(a1.log, 'D') > 0, "collective: rank 1 saw reads and proposals");
    check_solo(p.done, reqs, true, "collective pair");

    FakeBackend b0(4, true), b1(4, true);
    b0.drafts_on = b1.drafts_on = true;
    const auto q = run_pair(b0, b1, reqs, {}, {});
    check(q.follower_error.empty() && !q.leader_threw, "local reads: pair ran");
    check(count_kind(b1.log, 'L') == 0 && count_kind(b1.log, 'D') == 0, "local reads: rank 1 never sees them");
    check(b0.contexts() == b1.contexts(), "local reads: both ranks end in the same state");
    check_solo(q.done, reqs, true, "local-read pair");

    FakeBackend c0(4, true), c1(4, true);
    const auto bad = run_pair(c0, c1, requests(2, true), on, {});
    check(bad.follower_error.find("collective") != std::string::npos,
          "collective read on an unconfigured follower: ProtocolError (" + bad.follower_error + ")");
    check(count_kind(c1.log, 'L') == 0, "unconfigured follower: its backend never read");
}
}  // namespace

int main() {
    test_default_unchanged();
    test_emits_first_greedy();
    test_emits_first_single_token_and_max1();
    test_emits_first_sampled_and_options();
    test_emits_first_grouped();
    test_max_window();
    test_collective_reads();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cohort pair contract tests: PASS\n";
    return 0;
}
