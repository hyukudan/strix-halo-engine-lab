// Two-rank replication of the shared multi-request decoder (strix/serve/replicated_cohort.hpp), on the host:
// rank 0 runs CohortDecoder over a ReplicatedCohort, rank 1 runs follow_cohort() on its own backend, connected by an
// in-memory transport. Invariants: rank 1 makes exactly rank 0's backend calls (same arguments, same order) and ends
// with the same per-lane state; every request's ids equal its solo run; one control message per forward when commit
// and release are rank-local; a dropped / altered control message stops rank 1 before its backend sees it.
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
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool value, const std::string& label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}

// In-memory two-rank transport: whole messages, in order, per direction.
struct Wire {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<std::byte>> q[2];  // q[r]: messages waiting for rank r
    bool aborted = false;
    std::string why;
};

class QueueTransport final : public strix::dist::Transport {
public:
    QueueTransport(std::shared_ptr<Wire> wire, int rank) : wire_(std::move(wire)), rank_(rank) {}
    void send(std::span<const std::byte> message) override {
        std::vector<std::byte> copy(message.begin(), message.end());
        if (tamper) tamper(copy, sent_);
        ++sent_;
        std::lock_guard lock(wire_->m);
        if (wire_->aborted) throw std::runtime_error("link aborted: " + wire_->why);
        wire_->q[1 - rank_].push_back(std::move(copy));
        wire_->cv.notify_all();
    }
    void receive(std::span<std::byte> message) override {
        std::unique_lock lock(wire_->m);
        wire_->cv.wait(lock, [&] { return wire_->aborted || !wire_->q[rank_].empty(); });
        if (wire_->q[rank_].empty()) throw std::runtime_error("link aborted: " + wire_->why);
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
    void abort(const std::string& why) override {
        std::lock_guard lock(wire_->m);
        if (!wire_->aborted) wire_->why = why;
        wire_->aborted = true;
        wire_->cv.notify_all();
    }
    std::function<void(std::vector<std::byte>&, std::uint64_t)> tamper;  // (message, index of this rank's send)

private:
    std::shared_ptr<Wire> wire_;
    int rank_;
    std::uint64_t sent_ = 0;
};

// Deterministic "model" of tests/cohort_decode_tests.cpp: the next id depends only on the lane's own context.
std::uint32_t next_id(const std::vector<std::uint32_t>& ctx, std::uint32_t vocab) {
    const std::uint32_t a = ctx.empty() ? 0U : ctx.back();
    const std::uint32_t parity = static_cast<std::uint32_t>(ctx.size() % 2U);
    return (a * 5U + 1U + parity * (a % 2U)) % vocab;
}

std::vector<std::uint32_t> solo(const CohortRequest& r, std::uint32_t vocab) {
    std::vector<std::uint32_t> ctx = r.prompt, out;
    while (out.size() < r.max_tokens) {
        const std::uint32_t t = next_id(ctx, vocab);
        out.push_back(t);
        ctx.push_back(t);
        if (std::ranges::find(r.stop, t) != r.stop.end()) break;
    }
    return out;
}

std::string join(std::span<const std::uint32_t> v) {
    std::string s;
    for (const auto x : v) s += std::to_string(x) + ',';
    return s;
}

// Records every state-changing call; the ranks must produce the same record.
class FakeBackend final : public CohortBackend {
public:
    FakeBackend(std::uint32_t lanes, std::uint32_t rows, std::uint32_t vocab)
        : lanes_(lanes), rows_(rows), vocab_(vocab), ctx_(lanes) {}
    std::uint32_t lanes() const override { return lanes_; }
    std::uint32_t rows_per_forward() const override { return rows_; }
    std::uint32_t max_positions() const override { return 1U << 20U; }
    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if (first) ctx_.at(lane).clear();
        ctx_.at(lane).insert(ctx_[lane].end(), tokens.begin(), tokens.end());
        log.push_back("P" + std::to_string(lane) + (first ? "f:" : ":") + join(tokens));
        ++prefills;
    }
    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        check(window_.empty(), "verify with an open window");
        std::string rec = "V";
        std::size_t base = 0;
        for (const auto& g : groups) {
            std::vector<std::uint32_t> c = ctx_.at(g.lane);
            for (std::uint32_t i = 0; i < g.rows; ++i) {
                c.push_back(tokens[base + i]);
                top[base + i] = next_id(c, vocab_);
            }
            window_.push_back({g.lane, std::vector<std::uint32_t>(tokens.begin() + static_cast<std::ptrdiff_t>(base),
                                                                  tokens.begin() + static_cast<std::ptrdiff_t>(base + g.rows))});
            rec += std::to_string(g.lane) + "x" + std::to_string(g.rows) + ';';
            base += g.rows;
        }
        log.push_back(rec + join(tokens));
        ++forwards;
    }
    void commit(std::span<const std::uint32_t> accept) override {
        check(accept.size() == window_.size(), "one keep count per group");
        for (std::size_t g = 0; g < window_.size() && g < accept.size(); ++g) {
            auto& c = ctx_.at(window_[g].first);
            c.insert(c.end(), window_[g].second.begin(), window_[g].second.begin() + accept[g]);
        }
        window_.clear();
        log.push_back("C" + join(accept));
        ++commits;
    }
    void release(std::uint32_t lane) override {
        ctx_.at(lane).clear();
        log.push_back("R" + std::to_string(lane));
        ++releases;
    }
    std::uint64_t restore(std::uint32_t lane, std::span<const std::uint32_t> prompt) override {
        std::size_t best = 0;
        for (const auto& kept : restores ? retained : std::vector<std::vector<std::uint32_t>>{}) {
            std::size_t n = 0;
            while (n < kept.size() && n + 1U < prompt.size() && kept[n] == prompt[n]) ++n;
            best = std::max(best, n);
        }
        if (best == 0) return 0;  // nothing restored: no state change (the lane starts with prefill(first = true))
        ctx_.at(lane).assign(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(best));
        log.push_back("S" + std::to_string(lane) + ":" + std::to_string(best));
        return best;
    }
    void capture(std::uint32_t lane, std::span<const std::uint32_t> committed) override {
        retained.emplace_back(committed.begin(), committed.end());
        log.push_back("K" + std::to_string(lane) + ":" + join(committed));
    }
    void prefill_group(std::span<const CohortPrompt> members) override {
        std::string rec = "G";
        for (const auto& m : members) {
            if (m.first) ctx_.at(m.lane).clear();
            ctx_.at(m.lane).insert(ctx_[m.lane].end(), m.tokens.begin(), m.tokens.end());
            rec += std::to_string(m.lane) + (m.first ? "f:" : ":") + join(m.tokens) + ';';
        }
        log.push_back(rec);
    }
    std::vector<std::vector<std::uint32_t>> retained;
    bool restores = true;  // false: this rank's store cannot restore (a diverging replica)
    const std::vector<std::vector<std::uint32_t>>& state() const { return ctx_; }

    std::vector<std::string> log;
    std::uint64_t forwards = 0, prefills = 0, commits = 0, releases = 0;

private:
    std::uint32_t lanes_, rows_, vocab_;
    std::vector<std::vector<std::uint32_t>> ctx_;
    std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>> window_;
};

std::vector<CohortRequest> requests(std::uint32_t count, std::uint32_t vocab) {
    std::vector<CohortRequest> out;
    for (std::uint32_t i = 0; i < count; ++i) {
        CohortRequest r;
        r.id = 100 + i;
        const std::uint32_t len = 3 + (i * 11U) % 40U;
        for (std::uint32_t k = 0; k < len; ++k) r.prompt.push_back((i * 13U + k * 5U) % vocab);
        r.max_tokens = 20 + (i * 17U) % 50U;
        if (i % 3 == 2) r.stop = {(i * 7U) % vocab};
        r.retain = false;  // the message counts below are the plain protocol's; run_hooks covers capture / restore
        out.push_back(r);
    }
    return out;
}

struct Pair {
    std::shared_ptr<Wire> wire = std::make_shared<Wire>();
    QueueTransport t0{wire, 0}, t1{wire, 1};
};

// Runs rank 1 on its own thread; returns its round count (or the error it stopped with).
struct Follower {
    std::thread thread;
    std::uint64_t rounds = 0;
    std::string error;
    bool protocol_error = false;
    void start(FakeBackend& be, QueueTransport& link, ReplicaOptions options) {
        thread = std::thread([this, &be, &link, options] {
            try {
                rounds = follow_cohort(be, link, options);
            } catch (const strix::pair::ProtocolError& e) {
                protocol_error = true;
                error = e.what();
            } catch (const std::exception& e) {
                error = e.what();
            }
        });
    }
};

void run_identity(bool local_ops, const char* name) {
    const std::uint32_t vocab = 23, lanes = 4, rows = 8;
    Pair link;
    FakeBackend be0(lanes, rows, vocab), be1(lanes, rows, vocab);
    ReplicaOptions options;
    options.local_ops = local_ops;
    options.frame_words = 5;  // long prompts and wide verifies need follow-up frames
    Follower follower;
    follower.start(be1, link.t1, options);
    std::vector<CohortResult> res;
    std::uint64_t rounds = 0;
    const auto reqs = requests(9, vocab);
    {
        ReplicatedCohort rep(be0, link.t0, options);
        CohortConfig cfg;
        cfg.context_draft = true;
        cfg.prompt_slice = 8;
        cfg.prior = {13.3, 0.113, 0.13, 0.0};
        CohortDecoder dec(rep, cfg);
        for (std::size_t i = 0; i < reqs.size() / 2; ++i) dec.submit(reqs[i]);
        for (int s = 0; s < 5; ++s) dec.step();
        for (std::size_t i = reqs.size() / 2; i < reqs.size(); ++i) dec.submit(reqs[i]);
        dec.run();
        res = dec.take_finished();
        rep.close();
        rounds = rep.rounds();
    }
    follower.thread.join();
    const std::string n = name;
    check(follower.error.empty(), n + ": rank 1 finished cleanly (" + follower.error + ")");
    check(res.size() == reqs.size(), n + ": every request finished");
    for (const auto& r : res) {
        const auto it = std::ranges::find_if(reqs, [&](const CohortRequest& q) { return q.id == r.id; });
        check(it != reqs.end() && r.error.empty() && r.ids == solo(*it, vocab), n + ": ids == solo for " + std::to_string(r.id));
    }
    check(be0.log == be1.log, n + ": rank 1 made exactly rank 0's backend calls");
    check(be0.state() == be1.state(), n + ": per-lane state equal on both ranks");
    check(be0.forwards > 0 && be0.prefills > 0 && be0.commits > 0 && be0.releases > 0, n + ": every call kind exercised");
    check(follower.rounds == rounds, n + ": both ranks counted the same control messages");
    const std::uint64_t expect = local_ops ? be0.forwards + be0.prefills + 1U  // + close
                                           : be0.forwards + be0.prefills + be0.commits + be0.releases + 1U;
    check(rounds == expect, n + ": control messages per call (" + std::to_string(rounds) + " vs " +
                                std::to_string(expect) + ")");
}

// The optional hooks replicate: a group pass is a call; capture and a restore that found rows ride on the next
// message, and rank 1 must restore the same rows (a rank whose retained store differs stops with a protocol error).
void run_hooks(bool local_ops, bool diverge) {
    const std::string name = std::string("hooks (") + (local_ops ? "local" : "sent") + (diverge ? ", diverging" : "") + ")";
    Pair link;
    FakeBackend be0(3, 8, 23), be1(3, 8, 23);
    ReplicaOptions options;
    options.local_ops = local_ops;
    options.frame_words = 5;
    // a diverging rank 1 cannot restore what rank 0 restores (set before its thread starts)
    be1.restores = !diverge;
    Follower follower;
    follower.start(be1, link.t1, options);
    const std::vector<std::uint32_t> p0 = {1, 2, 3, 4, 5}, p1 = {7, 8, 9};
    std::uint64_t restored = 0;
    try {
        ReplicatedCohort rep(be0, link.t0, options);
        const CohortPrompt members[2] = {{0, p0, true}, {1, p1, true}};
        rep.prefill_group(members);
        const CohortGroup groups[2] = {{0, 1}, {1, 1}};
        const std::uint32_t tokens[2] = {6, 10};
        std::uint32_t top[2] = {};
        rep.verify(groups, tokens, top);
        const std::uint32_t accept[2] = {1, 1};
        rep.commit(accept);
        rep.capture(0, be0.state().at(0));
        rep.release(0);
        std::vector<std::uint32_t> next = be0.retained.at(0);
        next.insert(next.end(), {11, 12, 13});
        restored = rep.restore(2, next);
        rep.prefill(2, std::span<const std::uint32_t>(next).subspan(static_cast<std::size_t>(restored), next.size() - 1U - restored),
                    false);
        rep.close();
    } catch (const std::exception&) {
        check(diverge, name + ": rank 0 failed only when the ranks diverge");
    }
    follower.thread.join();
    check(restored == p0.size() + 1U, name + ": rank 0 restored the captured context");
    if (diverge) {
        check(follower.protocol_error, name + ": rank 1 stopped with a protocol error");
        return;
    }
    check(follower.error.empty(), name + ": rank 1 finished cleanly");
    check(be0.log == be1.log, name + ": both ranks made the same calls");
    check(be0.state() == be1.state(), name + ": both ranks hold the same lanes");
    check(be0.retained == be1.retained, name + ": both ranks retained the same prefixes");
}

// One long prefill split into many frames arrives intact, and a closed replica refuses further calls.
void run_frames() {
    Pair link;
    FakeBackend be0(1, 4, 23), be1(1, 4, 23);
    ReplicaOptions options;
    options.frame_words = 7;
    Follower follower;
    follower.start(be1, link.t1, options);
    std::vector<std::uint32_t> prompt(1000);
    for (std::uint32_t i = 0; i < prompt.size(); ++i) prompt[i] = (i * 7919U) % 50021U;
    bool refused = false;
    {
        ReplicatedCohort rep(be0, link.t0, options);
        rep.prefill(0, prompt, true);
        rep.keepalive();
        rep.close();
        try {
            rep.prefill(0, prompt, false);
        } catch (const std::logic_error&) {
            refused = true;
        }
    }
    follower.thread.join();
    check(follower.error.empty(), "frames: rank 1 finished cleanly");
    check(be1.state().at(0) == prompt, "frames: the long prompt arrived whole on rank 1");
    check(follower.rounds == 3, "frames: prefill + keepalive + close");
    check(refused, "frames: a call after close() is refused");
}

// Rank 0's third control block carries a wrong round: rank 1 stops with a protocol error before applying it.
void run_tamper() {
    Pair link;
    FakeBackend be0(2, 4, 23), be1(2, 4, 23);
    link.t0.tamper = [](std::vector<std::byte>& m, std::uint64_t index) {
        if (index == 2 && m.size() == strix::serve::kReplicaBlockWords * 4U) m[2U * 4U] ^= std::byte{1};  // round lo
    };
    Follower follower;
    follower.start(be1, link.t1, {});
    bool lead_failed = false;
    {
        ReplicatedCohort rep(be0, link.t0);
        const std::vector<std::uint32_t> p{1, 2, 3};
        try {
            rep.prefill(0, p, true);
            rep.prefill(1, p, true);
            rep.prefill(0, p, false);  // tampered
            rep.prefill(1, p, false);
            rep.close();
        } catch (const std::exception&) {
            lead_failed = true;
        }
    }
    follower.thread.join();
    check(follower.protocol_error, "tamper: rank 1 reports a protocol error (" + follower.error + ")");
    check(be1.prefills == 2, "tamper: rank 1 applied only the two intact messages");
    check(lead_failed, "tamper: rank 0 sees the broken pair");
}

}  // namespace

int main() {
    run_identity(true, "local ops");
    run_identity(false, "collective ops");
    run_frames();
    run_hooks(true, false);
    run_hooks(false, false);
    run_hooks(true, true);
    run_hooks(false, true);
    run_tamper();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "replicated cohort tests: ok\n";
    return 0;
}
