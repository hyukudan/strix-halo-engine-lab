// Two-rank replication of the shared multi-request decoder (strix/serve/replicated_cohort.hpp), on the host:
// rank 0 runs CohortDecoder over a ReplicatedCohort, rank 1 runs follow_cohort() on its own backend, connected by an
// in-memory transport. Invariants: rank 1 makes exactly rank 0's backend calls (same arguments, same order) and ends
// with the same per-lane state; every request's ids equal its solo run; one control message per forward when commit
// and release are rank-local; a dropped / altered control message stops rank 1 before its backend sees it.
#include "strix/pair/lockstep.hpp"
#include "strix/dist/exchange_transport.hpp"
#include "strix/serve/replicated_cohort.hpp"

#ifdef STRIX_TEST_REPLICA_TRANSPORTS
#include "strix/dist/communicator.hpp"
#include "strix/dist/host_copy.hpp"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace strix::serve;

#ifdef STRIX_TEST_REPLICA_TRANSPORTS
namespace strix::dist {
void enqueue_host_d2h_copy(void* mapped_host, const void* device, std::size_t bytes, hipStream_t) {
    std::memcpy(mapped_host, device, bytes);
}
}
#endif

namespace {
// A one-shot allocation failure on the calling rank only.
thread_local std::size_t fail_allocation_size = 0;
thread_local bool fail_next_allocation = false;
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void free_test_allocation(void* ptr) noexcept { std::free(ptr); }
}

void* operator new(std::size_t size) {
    if (fail_next_allocation || (fail_allocation_size != 0 && size == fail_allocation_size)) {
        fail_next_allocation = false;
        fail_allocation_size = 0;
        throw std::bad_alloc{};
    }
    if (void* ptr = std::malloc(size == 0 ? 1 : size)) return ptr;
    throw std::bad_alloc{};
}

void operator delete(void* ptr) noexcept { free_test_allocation(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { ::operator delete(ptr); }

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
    bool supports_peer_abort() const noexcept override { return true; }
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

// A word no prompt holds: placed right after a prompt in memory, a restore that sees it read past the prompt.
constexpr std::uint32_t kSentinel = 0xFFFFFFF0U;

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
        if (on_verify) on_verify();
        check(window_.empty(), "verify with an open window");
        withheld_.clear();
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
            rec += std::to_string(g.lane) + "x" + std::to_string(g.rows) + (g.reads_scores ? "r" : "") + ';';
            if (!g.reads_scores) withheld_.push_back(g.lane);
            base += g.rows;
        }
        log.push_back(rec + join(tokens));
        ++forwards;
    }
    // A score-aware backend: a group verified as unread has no scores to read (a pick-only route).
    std::span<const float> read_final_logits(std::uint32_t lane) override {
        if (std::ranges::find(withheld_, lane) != withheld_.end()) throw std::logic_error("scores withheld");
        log.push_back("L" + std::to_string(lane));
        scores_.assign(4, static_cast<float>(lane));
        return scores_;
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
    bool prefill_emits_first() const override { return whole; }
    std::uint64_t restore(std::uint32_t lane, std::span<const std::uint32_t> prompt) override {
        check(std::ranges::find(prompt, kSentinel) == prompt.end(), "restore read a word past the prompt");
        std::size_t best = 0;
        for (const auto& kept : restores ? retained : std::vector<std::vector<std::uint32_t>>{}) {
            std::size_t n = 0;
            // a whole-prompt backend (prefill emits the first id) may restore every prompt row, others all but the last
            while (n < kept.size() && n + (whole ? 0U : 1U) < prompt.size() && kept[n] == prompt[n]) ++n;
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
    void bind_lane(std::uint32_t lane, std::uint64_t request_id, std::span<const std::uint32_t> traits) override {
        log.push_back("B" + std::to_string(lane) + ":" + std::to_string(request_id) + ":" + join(traits));
        ++binds;
    }
    void control(std::span<const std::uint32_t> words) override {
        log.push_back("X" + join(words));
        ++controls;
    }
    std::vector<std::vector<std::uint32_t>> retained;
    std::function<void()> on_verify;
    bool restores = true;  // false: this rank's store cannot restore (a diverging replica)
    bool whole = false;    // prefill_emits_first: restore may return the whole prompt
    const std::vector<std::vector<std::uint32_t>>& state() const { return ctx_; }

    std::vector<std::string> log;
    std::uint64_t forwards = 0, prefills = 0, commits = 0, releases = 0, binds = 0, controls = 0;

private:
    std::uint32_t lanes_, rows_, vocab_;
    std::vector<std::vector<std::uint32_t>> ctx_;
    std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>> window_;
    std::vector<std::uint32_t> withheld_;  // lanes of the open window verified as unread
    std::vector<float> scores_;
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

strix::dist::ExchangeTransport exchange_adapter(QueueTransport& link) {
    return strix::dist::ExchangeTransport(
        [&link](auto out, auto in) { link.exchange(out, in); }, "test-exchange",
        [&link](const std::string& why) { link.abort(why); });
}

// Runs rank 1 on its own thread; returns its round count (or the error it stopped with).
struct Follower {
    std::thread thread;
    std::uint64_t rounds = 0;
    std::string error;
    bool protocol_error = false;
    void start(FakeBackend& be, strix::dist::Transport& link, ReplicaOptions options) {
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
                                           : be0.forwards + be0.prefills + be0.commits + be0.releases + be0.binds +
                                                 be0.controls + 1U;
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

// A whole-prompt restore (prefill_emits_first, restore returns prompt.size()) replicates: rank 0 sends exactly the
// prompt (no anchor word past its end) and rank 1 restores the same rows; a rank 1 whose backend cannot restore a whole
// prompt stops with a protocol error before its backend sees the record.
void run_whole_restore(bool local_ops, bool follower_whole) {
    const std::string name = std::string("whole restore (") + (local_ops ? "local" : "sent") +
                             (follower_whole ? ")" : ", follower without whole restore)");
    Pair link;
    FakeBackend be0(2, 8, 23), be1(2, 8, 23);
    ReplicaOptions options;
    options.local_ops = local_ops;
    options.frame_words = 5;
    const std::vector<std::uint32_t> kept = {1, 2, 3, 4, 5};
    be0.whole = true;
    be1.whole = follower_whole;  // set before rank 1's thread starts
    be0.retained.push_back(kept);
    be1.retained.push_back(kept);
    Follower follower;
    follower.start(be1, link.t1, options);
    std::vector<std::uint32_t> buffer = kept;
    buffer.push_back(kSentinel);  // the word right after the prompt in memory
    const std::span<const std::uint32_t> prompt(buffer.data(), kept.size());
    std::uint64_t restored = 0;
    try {
        ReplicatedCohort rep(be0, link.t0, options);
        restored = rep.restore(1, prompt);
        rep.close();
    } catch (const std::exception&) {
        check(!follower_whole, name + ": rank 0 failed only when rank 1 cannot restore the whole prompt");
    }
    follower.thread.join();
    check(restored == kept.size(), name + ": rank 0 restored the whole prompt");
    if (!follower_whole) {
        check(follower.protocol_error, name + ": rank 1 stopped with a protocol error (" + follower.error + ")");
        check(be1.log.empty(), name + ": rank 1's backend saw nothing of the refused record");
        return;
    }
    check(follower.error.empty(), name + ": rank 1 finished cleanly (" + follower.error + ")");
    check(be0.log == be1.log, name + ": both ranks made the same calls");
    check(be0.state() == be1.state() && be1.state().at(1) == kept, name + ": both ranks hold the whole prompt");
}

// bind_lane and control replicate: same calls, same order relative to prefill / verify / commit / release, a 64-bit
// request id above 2^32, multi-word, single-word and empty traits, empty and non-empty control words.
void run_bind_control(bool local_ops) {
    const std::string name = std::string("bind/control (") + (local_ops ? "local" : "sent") + ")";
    Pair link;
    FakeBackend be0(3, 8, 23), be1(3, 8, 23);
    ReplicaOptions options;
    options.local_ops = local_ops;
    options.frame_words = 5;
    Follower follower;
    follower.start(be1, link.t1, options);
    const std::uint64_t big = (std::uint64_t{0xABCDU} << 32U) | 0x12345678U;
    const std::vector<std::uint32_t> traits = {7, 0xFFFFFFFFU, 3, 9, 11, 13, 15};
    const std::vector<std::uint32_t> one = {42};
    const std::vector<std::uint32_t> ctl = {1, 2, 0xFFFFFFFFU};
    const std::vector<std::uint32_t> p = {1, 2, 3};
    {
        ReplicatedCohort rep(be0, link.t0, options);
        rep.bind_lane(0, big, traits);
        rep.control({});
        rep.prefill(0, p, true);
        rep.bind_lane(1, 5, {});  // empty traits
        rep.control(ctl);
        rep.bind_lane(2, std::uint64_t{1} << 40U, one);
        rep.prefill(1, p, true);
        const CohortGroup groups[2] = {{0, 1}, {1, 1}};
        const std::uint32_t tokens[2] = {4, 5};
        std::uint32_t top[2] = {};
        rep.verify(groups, tokens, top);
        const std::uint32_t accept[2] = {1, 1};
        rep.commit(accept);
        rep.control(ctl);
        rep.release(0);
        rep.bind_lane(0, big + 1U, one);
        rep.control({});  // rides on close
        rep.close();
    }
    follower.thread.join();
    check(follower.error.empty(), name + ": rank 1 finished cleanly (" + follower.error + ")");
    check(be0.log == be1.log, name + ": rank 1 saw rank 0's exact call sequence");
    check(be0.binds == 4 && be1.binds == 4 && be0.controls == 4 && be1.controls == 4, name + ": every call arrived");
    check(!be1.log.empty() && be1.log.front() == "B0:" + std::to_string(big) + ":7,4294967295,3,9,11,13,15,",
          name + ": 64-bit id and multi-word traits intact");
    check(std::ranges::find(be1.log, "B1:5:") != be1.log.end(), name + ": empty traits");
    check(std::ranges::find(be1.log, "X") != be1.log.end(), name + ": empty control words");
    check(std::ranges::find(be1.log, "X1,2,4294967295,") != be1.log.end(), name + ": control words");
    // local: 2 prefills + verify + close; sent: also 4 binds, 4 controls, commit and release
    check(follower.rounds == (local_ops ? 4U : 14U),
          name + ": control message count (" + std::to_string(follower.rounds) + ")");
}

// The verify frame carries no score-read flag: both ranks run every group as read whatever rank 0 was told, so a
// backend that picks its route from the flag takes the same route on both ranks, and both can read the scores after.
void run_score_reads(strix::dist::Transport& t0, strix::dist::Transport& t1) {
    const std::string name = "score reads (collective)";
    FakeBackend be0(3, 8, 23), be1(3, 8, 23);
    ReplicaOptions options;
    options.collective_reads = true;
    options.frame_words = 5;
    Follower follower;
    follower.start(be1, t1, options);
    float read0 = -1.0F, read1 = -1.0F;
    try {
        ReplicatedCohort rep(be0, t0, options);
        const std::vector<std::uint32_t> p = {1, 2, 3};
        rep.prefill(0, p, true);
        rep.prefill(1, p, true);
        const CohortGroup groups[2] = {{0, 1, true}, {1, 1, false}};
        const std::uint32_t tokens[2] = {4, 5};
        std::uint32_t top[2] = {};
        rep.verify(groups, tokens, top);
        read0 = rep.read_final_logits(0)[0];
        read1 = rep.read_final_logits(1)[0];
        const std::uint32_t accept[2] = {1, 1};
        rep.commit(accept);
        rep.close();
    } catch (const std::exception& e) {
        check(false, name + ": rank 0 failed (" + e.what() + ")");
    }
    follower.thread.join();
    check(follower.error.empty(), name + ": rank 1 finished cleanly (" + follower.error + ")");
    check(be0.log == be1.log, name + ": both ranks ran the same verify route and reads");
    check(std::ranges::find(be1.log, "V0x1r;1x1r;4,5,") != be1.log.end(), name + ": rank 1 ran both groups as read");
    check(read0 == 0.0F && read1 == 1.0F, name + ": both lanes' scores read back on rank 0");
}

void run_verify_allocation_failure() {
    Pair link;
    FakeBackend be0(1, 4, 23), be1(1, 4, 23);
    Follower follower;
    follower.start(be1, link.t1, {});
    {
        ReplicatedCohort rep(be0, link.t0);
        rep.bind_lane(0, 1, {});  // A queued record must survive a failure before publication.
        const CohortGroup group{0, 2, false};
        const std::uint32_t tokens[] = {4, 5};
        std::uint32_t top[2] = {};
        // The inline frame has four body words, so this targets the first group allocation only.
        fail_allocation_size = sizeof(CohortGroup);
        bool threw = false;
        try {
            rep.verify({&group, 1}, tokens, top);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        fail_allocation_size = 0;
        check(threw, "verify allocation: rank 0 throws bad_alloc");
        check(rep.rounds() == 0, "verify allocation: no frame was published");
        if (threw && rep.rounds() == 0) {
            rep.verify({&group, 1}, tokens, top);
            const std::uint32_t accept[] = {1};
            rep.commit(accept);
            rep.close();
        }
    }
    follower.thread.join();
    check(follower.error.empty(), "verify allocation: rank 1 finishes cleanly after retry");
    check(be0.log == be1.log && be0.binds == 1 && be0.forwards == 1 && be0.commits == 1,
          "verify allocation: retry delivers the queued bind and verify once on both ranks");
}

void run_verify_backend_failure() {
    Pair link;
    FakeBackend be0(1, 4, 23), be1(1, 4, 23);
    be0.on_verify = [] { throw std::runtime_error("rank 0 verify failed"); };
    be1.on_verify = [&] {
        std::byte unused{};
        link.t1.receive({&unused, 1});  // Simulate waiting for rank 0 inside the backend collective.
    };
    Follower follower;
    follower.start(be1, link.t1, {});
    {
        ReplicatedCohort rep(be0, link.t0);
        const CohortGroup group{0, 1, false};
        const std::uint32_t token = 4;
        std::uint32_t top = 0;
        bool rethrew = false;
        try {
            rep.verify({&group, 1}, {&token, 1}, {&top, 1});
        } catch (const std::runtime_error& e) {
            rethrew = std::string_view(e.what()) == "rank 0 verify failed";
        }
        check(rethrew && rep.rounds() == 1, "verify backend: rank 0 rethrows the failure after publishing");
        bool aborted = false;
        {
            std::lock_guard lock(link.wire->m);
            aborted = link.wire->aborted;
        }
        check(aborted, "verify backend: rank 0 aborts immediately while the replica still exists");
        if (!aborted) link.t0.abort("test cleanup");
        bool closed = false;
        try {
            rep.keepalive();
        } catch (const std::runtime_error& e) {
            // A backend failure that names the earlier abort; the regressed path tries to use the aborted link instead.
            closed = std::string_view(e.what()) == "cohort replica: link aborted by an earlier failure";
        } catch (const std::logic_error&) {
            // A contract violation would stop a CohortDecoder's loop.
        }
        check(closed, "verify backend: rank 0 refuses further work on the replica as a backend failure");
        follower.thread.join();
        check(follower.error == "link aborted: rank 0 verify failed", "verify backend: rank 1 exits its collective");
    }
}

// A rank 0 verify failure inside the decoder on a pair: every lane of the forward fails cleanly (its result is taken,
// rank 0's lane is freed), the decoder keeps stepping, a later request fails as a backend failure without touching the
// dead link, and close() after the abort does not throw.
void run_decoder_pair_failure() {
    const std::uint32_t vocab = 23;
    Pair link;
    FakeBackend be0(2, 4, vocab), be1(2, 4, vocab);
    be0.on_verify = [] { throw std::runtime_error("rank 0 verify failed"); };
    be1.on_verify = [&] {
        std::byte unused{};
        link.t1.receive({&unused, 1});  // rank 1 waits inside the collective until the abort wakes it
    };
    Follower follower;
    follower.start(be1, link.t1, {});
    const auto reqs = requests(3, vocab);
    std::vector<CohortResult> first, later;
    std::string thrown;
    {
        ReplicatedCohort rep(be0, link.t0);
        CohortDecoder dec(rep, CohortConfig{});
        dec.submit(reqs[0]);
        dec.submit(reqs[1]);
        try {
            dec.run();
            first = dec.take_finished();
            check(dec.live() == 0 && dec.idle(), "pair failure: no lane stays bound after the failed forward");
            dec.submit(reqs[2]);
            dec.run();
            later = dec.take_finished();
            rep.close();
        } catch (const std::exception& e) {
            thrown = e.what();
        }
    }
    follower.thread.join();
    check(thrown.empty(), "pair failure: the decoder loop and close() do not throw (" + thrown + ")");
    // The verify failure fails the lanes of that forward; a lane still in its prompt then fails its prefill cleanly.
    check(first.size() == 2, "pair failure: both requests return a result");
    std::size_t by_verify = 0;
    for (const auto& r : first) {
        by_verify += r.error.find("verify: rank 0 verify failed") != std::string::npos ? 1U : 0U;
        check(r.finish == "error" && (r.error.find("rank 0 verify failed") != std::string::npos ||
                                      r.error.find("link aborted by an earlier failure") != std::string::npos),
              "pair failure: request " + std::to_string(r.id) + " fails cleanly (" + r.error + ")");
    }
    check(by_verify >= 1, "pair failure: the failed forward's lane carries the verify error");
    check(be0.releases >= 2, "pair failure: rank 0 frees its lanes after the abort");
    check(later.size() == 1 && later[0].finish == "error" &&
              later[0].error.find("link aborted by an earlier failure") != std::string::npos,
          "pair failure: a later request fails as a backend failure");
    check(follower.error == "link aborted: rank 0 verify failed", "pair failure: rank 1 exits its collective");
}

void run_exchange_abort_allocation_failure() {
    const std::string reason(256, 'x');
    unsigned callbacks = 0;
    strix::dist::ExchangeTransport link([](auto, auto) {}, "exchange",
                                       [&](const std::string&) { ++callbacks; });
    fail_next_allocation = true;
    try {
        link.abort(reason);
    } catch (const std::bad_alloc&) {
        link.abort({});
    }
    const bool allocation_failed = !fail_next_allocation;
    fail_next_allocation = false;
    link.abort({});
    check(allocation_failed && link.aborted() && callbacks == 1,
          "exchange abort: failed reason capture still cancels exactly once");

    bool retried = false;
    strix::dist::ExchangeTransport retry([](auto, auto) {}, "exchange", [&](const std::string& why) {
        if (!why.empty()) throw std::bad_alloc{};
        retried = true;
    });
    try {
        retry.abort(reason);
    } catch (const std::bad_alloc&) {
        retry.abort({});
    }
    check(retry.aborted() && retried, "exchange abort: a failed callback does not swallow the empty retry");
}

void run_callbackless_exchange_rejection() {
    unsigned frames = 0;
    const auto exchange = [&](auto, auto) {
        ++frames;
        throw std::runtime_error("unexpected frame");
    };
    strix::dist::ExchangeTransport t0(exchange), t1(exchange);
    FakeBackend be0(1, 4, 23), be1(1, 4, 23);
    bool leader_rejected = false, follower_rejected = false;
    try {
        ReplicatedCohort rep(be0, t0);
        rep.keepalive();
    } catch (const std::invalid_argument& e) {
        leader_rejected = std::string_view(e.what()).find("peer cancellation") != std::string_view::npos;
    } catch (const std::runtime_error&) {}
    try {
        (void)follow_cohort(be1, t1);
    } catch (const std::invalid_argument& e) {
        follower_rejected = std::string_view(e.what()).find("peer cancellation") != std::string_view::npos;
    } catch (const std::runtime_error&) {}
    check(leader_rejected && follower_rejected && frames == 0,
          "exchange admission: both ranks reject a missing abort callback before any frame");
}

void run_verify_publication_failure() {
    using namespace std::chrono_literals;
    struct PayloadFailure { unsigned frame; };
    Pair wire;
    auto t0 = exchange_adapter(wire.t0), t1 = exchange_adapter(wire.t1);
    FakeBackend be0(1, 4, 23), be1(1, 4, 23);
    std::promise<void> receiving_payload;
    auto waiting = receiving_payload.get_future();
    wire.t1.tamper = [&](auto&, std::uint64_t index) {
        if (index == 1) receiving_payload.set_value();
    };
    wire.t0.tamper = [&](auto&, std::uint64_t index) {
        if (index != 1) return;
        if (waiting.wait_for(5s) != std::future_status::ready)
            throw std::runtime_error("follower did not enter the continuation exchange");
        throw PayloadFailure{1};
    };
    ReplicatedCohort rep(be0, t0);
    auto follow = std::async(std::launch::async, [&] {
        try {
            (void)follow_cohort(be1, t1);
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string{};
    });
    const CohortGroup group{0, kReplicaBlockWords, false};
    const std::uint32_t tokens[kReplicaBlockWords] = {};
    std::uint32_t top[kReplicaBlockWords] = {};
    bool rethrew = false;
    try {
        rep.verify({&group, 1}, tokens, top);
    } catch (const PayloadFailure& error) {
        rethrew = error.frame == 1;
    } catch (const std::exception&) {}
    wire.t0.tamper = {};
    check(rethrew && rep.rounds() == 0, "verify publication: rethrows the original continuation failure");
    const bool cancelled = follow.wait_for(2s) == std::future_status::ready;
    check(cancelled && t0.aborted(), "verify publication: abort wakes the follower while both adapters live");
    if (!cancelled) wire.t0.abort("test cleanup");
    check(follow.get() == "link aborted: " && be0.forwards == 0 && be1.forwards == 0,
          "verify publication: empty abort reason reaches the peer before either backend runs");
    bool closed = false;
    try {
        rep.keepalive();
    } catch (const std::runtime_error& e) {
        closed = std::string_view(e.what()) == "cohort replica: link aborted by an earlier failure";
    } catch (const std::logic_error&) {}
    check(closed, "verify publication: rank 0 refuses reuse of the replica as a backend failure");
}

void run_follower_abort_allocation_failure() {
    Pair wire;
    FakeBackend be0(1, 4, 23), be1(1, 4, 23);
    const std::runtime_error original(std::string(256, 'x'));
    be1.on_verify = [&] {
        fail_next_allocation = true;
        throw original;
    };
    auto follow = std::async(std::launch::async, [&] {
        try {
            (void)follow_cohort(be1, wire.t1);
        } catch (const std::runtime_error& e) {
            return std::string_view(e.what()) == original.what();
        } catch (const std::bad_alloc&) {
            return false;
        }
        return false;
    });
    ReplicatedCohort rep(be0, wire.t0);
    const CohortGroup group{0, 1, false};
    const std::uint32_t token = 4;
    std::uint32_t top = 0;
    rep.verify({&group, 1}, {&token, 1}, {&top, 1});
    check(follow.get(), "follower abort: preserves the original verify exception under allocation failure");
    std::lock_guard lock(wire.wire->m);
    check(wire.wire->aborted, "follower abort: cancels the link when building the reason fails");
}

void run_verify_transport_failure(strix::dist::Transport& t0, strix::dist::Transport& t1) {
    using namespace std::chrono_literals;
    const auto name = t0.name();
    FakeBackend be0(1, 4, 23), be1(1, 4, 23);
    std::promise<void> entered;
    auto waiting = entered.get_future();
    be0.on_verify = [&] {
        if (waiting.wait_for(5s) != std::future_status::ready)
            throw std::runtime_error("follower did not enter verify");
        throw std::runtime_error("rank 0 verify failed");
    };
    be1.on_verify = [&] {
        std::byte unused{};
        entered.set_value();
        t1.receive({&unused, 1});
        throw std::runtime_error("test cleanup byte received");
    };
    auto follow = std::async(std::launch::async, [&] {
        try {
            (void)follow_cohort(be1, t1);
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string{};
    });
    ReplicatedCohort rep(be0, t0);
    const CohortGroup group{0, 1, false};
    const std::uint32_t token = 4;
    std::uint32_t top = 0;
    bool rethrew = false;
    try {
        rep.verify({&group, 1}, {&token, 1}, {&top, 1});
    } catch (const std::runtime_error& e) {
        rethrew = std::string_view(e.what()) == "rank 0 verify failed";
    }
    check(rethrew && rep.rounds() == 1, name + ": rank 0 rethrows after publishing verify");
    const bool cancelled = follow.wait_for(2s) == std::future_status::ready;
    check(cancelled, name + ": abort wakes the collective before its deadline with both links alive");
    if (!cancelled) {
        const std::byte cleanup{};
        // Unblock the baseline's still-open transport instead of waiting for its full receive timeout.
        try { t0.send({&cleanup, 1}); } catch (const std::exception&) {}
    }
    const auto error = follow.get();
    check(!error.empty() && error != "test cleanup byte received" && error.find("timed out") == std::string::npos,
          name + ": follower exits with a cancellation error");
}

void run_exchange_verify_failure() {
    Pair wire;
    auto t0 = exchange_adapter(wire.t0), t1 = exchange_adapter(wire.t1);
    run_verify_transport_failure(t0, t1);
}

#ifdef STRIX_TEST_REPLICA_TRANSPORTS
void run_real_transports() {
    using namespace std::chrono_literals;
    using namespace strix::dist;
    auto [mem0, mem1] = memory_pair();
    run_score_reads(*mem0, *mem1);
    run_verify_transport_failure(*mem0, *mem1);

    int sockets[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0)
        throw std::runtime_error("USB4 host fixture: socketpair failed");
    auto usb0 = usb4_host_descriptors({sockets[0]}, {"stream"}, {1}, 0, 5s);
    auto usb1 = usb4_host_descriptors({sockets[1]}, {"stream"}, {1}, 0, 5s);
    run_score_reads(*usb0, *usb1);
    run_verify_transport_failure(*usb0, *usb1);

    // Use an ephemeral loopback port so concurrent test invocations normally get distinct listeners.
    const int probe = ::socket(AF_INET, SOCK_STREAM, 0);
    if (probe < 0) throw std::runtime_error("TCP fixture: socket failed");
    sockaddr_in loopback{};
    loopback.sin_family = AF_INET;
    loopback.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t length = sizeof(loopback);
    const bool bound = ::bind(probe, reinterpret_cast<sockaddr*>(&loopback), length) == 0 &&
                       ::getsockname(probe, reinterpret_cast<sockaddr*>(&loopback), &length) == 0;
    ::close(probe);
    if (!bound) throw std::runtime_error("TCP fixture: could not choose a loopback port");
    const auto port = ntohs(loopback.sin_port);
    auto listen = std::async(std::launch::async, [port] { return tcp_listen(port, 5s); });
    auto tcp1 = tcp_connect("127.0.0.1", port, 5s);
    auto tcp0 = listen.get();
    run_score_reads(*tcp0, *tcp1);
    run_verify_transport_failure(*tcp0, *tcp1);
}
#endif

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
    run_bind_control(true);
    run_bind_control(false);
    run_whole_restore(true, true);
    run_whole_restore(false, true);
    run_whole_restore(true, false);
    run_whole_restore(false, false);
    run_hooks(true, false);
    run_hooks(false, false);
    run_hooks(true, true);
    run_hooks(false, true);
    run_tamper();
    Pair wire;
    run_score_reads(wire.t0, wire.t1);
    run_verify_allocation_failure();
    run_verify_backend_failure();
    run_decoder_pair_failure();
    run_exchange_abort_allocation_failure();
    run_callbackless_exchange_rejection();
    run_exchange_verify_failure();
    run_verify_publication_failure();
    run_follower_abort_allocation_failure();
#ifdef STRIX_TEST_REPLICA_TRANSPORTS
    run_real_transports();
#endif
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "replicated cohort tests: ok\n";
    return 0;
}
