// strix/spec/pair_round.hpp: the follower replays exactly the leader's target calls (verify rows, then the commit before
// the next round), the end message checks positions, the wire form round-trips and the lockstep refuses a bad peer.
#include "strix/spec/pair_round.hpp"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace pr = strix::spec::pair_round;

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// The symmetric swap between two threads (one message slot per rank, a generation counter per exchange).
class Wire {
public:
    void swap(std::uint32_t rank, const std::uint8_t* mine, std::uint8_t* peer) {
        std::unique_lock<std::mutex> lk(m_);
        if (closed_) throw std::runtime_error("wire closed by the peer");
        const std::uint64_t gen = gen_;
        std::copy(mine, mine + pr::kMessageBytes, slot_[rank].begin());
        if (++arrived_ == 2U) {
            arrived_ = 0;
            ++gen_;
            cv_.notify_all();
        } else {
            cv_.wait(lk, [&] { return gen_ != gen || closed_; });
            if (gen_ == gen) throw std::runtime_error("wire closed by the peer");
        }
        std::copy(slot_[1U - rank].begin(), slot_[1U - rank].end(), peer);
        // the second reader of this generation must still see both slots: wait until both read before the next write
        if (++read_ == 2U) {
            read_ = 0;
            ++rgen_;
            cv_.notify_all();
        } else {
            const std::uint64_t r = rgen_;
            cv_.wait(lk, [&] { return rgen_ != r || closed_; });
        }
    }
    // a rank that failed: the peer's pending / next swap throws instead of waiting forever
    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::array<std::array<std::uint8_t, pr::kMessageBytes>, 2> slot_{};
    std::uint32_t arrived_ = 0, read_ = 0;
    std::uint64_t gen_ = 0, rgen_ = 0;
    bool closed_ = false;
};

// A target that logs its calls; greedy[i] = tokens[i] + 1; position advances by the commit.
struct LogTarget final : strix::spec::VerifyTarget {
    std::vector<std::string> log;
    std::uint64_t position = 100;
    std::uint32_t last_rows = 0;
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        std::string s = "V";
        for (const auto t : tokens) s += " " + std::to_string(t);
        log.push_back(s);
        greedy.clear();
        for (const auto t : tokens) greedy.push_back(t + 1U);
        last_rows = static_cast<std::uint32_t>(tokens.size());
    }
    void commit(std::uint32_t k) override {
        if (k > last_rows) throw std::logic_error("commit beyond the pass");
        log.push_back("C " + std::to_string(k));
        position += k;
    }
};

void test_replay() {
    Wire wire;
    LogTarget lead_inner, follow;
    pr::FollowResult fr;
    std::string follow_error;
    std::thread t([&] {
        try {
            fr = pr::pair_follow(
                follow, [&](const std::uint8_t* a, std::uint8_t* b) { wire.swap(1U, a, b); }, [&] { return follow.position; });
        } catch (const std::exception& e) {
            follow_error = e.what();
        }
    });
    pr::PairVerifyLeader lead(
        lead_inner, [&](const std::uint8_t* a, std::uint8_t* b) { wire.swap(0U, a, b); }, [&] { return lead_inner.position; });
    std::vector<std::uint32_t> greedy;
    // the BlockSpeculator's order: verify(rows) then commit(k), round after round; rows 1..8
    const std::vector<std::vector<std::uint32_t>> passes{{7}, {9, 4, 4, 2}, {5, 6}, {1, 2, 3, 4, 5, 6, 7, 8}, {3}};
    const std::vector<std::uint32_t> commits{1, 3, 2, 8, 1};
    std::uint64_t kept = 0;
    for (std::size_t i = 0; i < passes.size(); ++i) {
        lead.verify(passes[i], greedy);
        expect(greedy.size() == passes[i].size() && greedy.front() == passes[i].front() + 1U, "leader greedy from its target");
        lead.commit(commits[i]);
        kept += commits[i];
    }
    lead.finish();
    lead.finish();  // idempotent: no second end message
    t.join();
    expect(follow_error.empty(), "follower ran without error");
    if (!follow_error.empty()) std::fprintf(stderr, "  follower: %s\n", follow_error.c_str());
    expect(follow.log == lead_inner.log, "follower call log == leader target call log");
    expect(follow.position == lead_inner.position && follow.position == 100U + kept, "positions equal");
    expect(fr.passes == passes.size() && fr.rounds == passes.size() + 1U && fr.committed == kept, "follow counts");
    expect(fr.rows == 1U + 4U + 2U + 8U + 1U, "follow rows");
    expect(lead.rounds() == passes.size() + 1U, "leader rounds");
}

// A drafter that logs into its target's log (one call order for both); drafts = state + j + bias, state advanced by
// the prefix and the accepts (a deterministic chain: the same calls give the same ids on both ranks).
struct LogChain final : strix::spec::ChainDrafter {
    std::vector<std::string>& log;
    std::uint32_t bias = 0;
    std::uint32_t state = 11;
    LogChain(std::vector<std::string>& l, std::uint32_t b) : log(l), bias(b) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        std::string s = "D k" + std::to_string(k);
        for (const auto t : prefix) s += " " + std::to_string(t);
        log.push_back(s);
        std::uint32_t st = state;
        for (const auto t : prefix) st = st * 31U + t;
        out.clear();
        for (std::uint32_t j = 0; j < k; ++j) out.push_back(st + j + bias);
    }
    void accept(std::span<const std::uint32_t> emitted) override {
        std::string s = "A";
        for (const auto t : emitted) s += " " + std::to_string(t);
        log.push_back(s);
        for (const auto t : emitted) state = state * 7U + t;
    }
};

// The drafter on both ranks: the follower's chain sees the leader's draft / accept calls in the leader's order,
// interleaved with the target calls exactly as on the leader (commit before accept), and the draft hashes agree.
std::string run_chain(std::uint32_t follower_bias, bool follower_chain, std::vector<std::string>* lead_log,
                      std::vector<std::string>* follow_log, pr::FollowResult* out) {
    Wire wire;
    LogTarget lead_inner, follow;
    LogChain lead_chain(lead_inner.log, 0U), follow_chain(follow.log, follower_bias);
    pr::FollowResult fr;
    std::string follow_error;
    std::thread t([&] {
        try {
            fr = pr::pair_follow(
                follow, [&](const std::uint8_t* a, std::uint8_t* b) { wire.swap(1U, a, b); }, [&] { return follow.position; },
                follower_chain ? &follow_chain : nullptr);
        } catch (const std::exception& e) {
            follow_error = e.what();
            wire.close();
        }
    });
    std::string lead_error;
    try {
        pr::PairVerifyLeader lead(
            lead_inner, [&](const std::uint8_t* a, std::uint8_t* b) { wire.swap(0U, a, b); }, [&] { return lead_inner.position; });
        auto& ch = lead.chain(lead_chain);
        std::vector<std::uint32_t> greedy, drafts;
        // passes: chain draft (prefix, k) or none (a copy-only pass), verify rows, commit k, accept the emitted ids
        struct Pass { std::vector<std::uint32_t> prefix; std::uint32_t k; std::uint32_t rows, commit; };
        const std::vector<Pass> passes{{{}, 3, 4, 2}, {{5, 6}, 2, 5, 3}, {{}, 0, 1, 1}, {{9}, 7, 8, 8}, {{}, 1, 2, 1}};
        std::uint32_t anchor = 40;
        for (const auto& p : passes) {
            std::vector<std::uint32_t> rows{anchor};
            rows.insert(rows.end(), p.prefix.begin(), p.prefix.end());
            if (p.k) {
                ch.draft(p.prefix, p.k, drafts);
                rows.insert(rows.end(), drafts.begin(), drafts.end());
            }
            rows.resize(p.rows, 1U);
            lead.verify(rows, greedy);
            lead.commit(p.commit);
            std::vector<std::uint32_t> emitted(rows.begin() + 1, rows.begin() + p.commit);
            emitted.push_back(greedy[p.commit - 1U]);
            ch.accept(emitted);
            anchor = emitted.back();
        }
        lead.finish();
    } catch (const std::exception& e) {
        lead_error = e.what();
    }
    t.join();
    if (lead_log) *lead_log = lead_inner.log;
    if (follow_log) *follow_log = follow.log;
    if (out) *out = fr;
    return follow_error.empty() ? lead_error : follow_error;
}

void test_chain_replay() {
    std::vector<std::string> a, b;
    pr::FollowResult fr;
    const auto err = run_chain(0U, true, &a, &b, &fr);
    expect(err.empty(), "chain replay ran without error");
    if (!err.empty()) std::fprintf(stderr, "  chain replay: %s
", err.c_str());
    expect(a == b && a.size() == 4U * 5U - 1U, "follower target + chain call log == leader's (commit before accept)");
    expect(fr.drafts == 4U && fr.passes == 5U && fr.rounds == 4U + 5U + 1U, "chain follow counts");
    const auto div = run_chain(1U, true, nullptr, nullptr, nullptr);  // the follower's drafts differ: fail closed
    expect(div.find("diverged") != std::string::npos, "diverged drafters refused");
    const auto none = run_chain(0U, false, nullptr, nullptr, nullptr);  // the leader drafts on both ranks, no follower chain
    expect(none.find("no drafter") != std::string::npos, "follower without a drafter refused");
}

void test_position_mismatch() {
    Wire wire;
    LogTarget lead_inner, follow;
    follow.position = 99;  // the follower's target is one position behind
    std::string follow_error;
    std::thread t([&] {
        try {
            (void)pr::pair_follow(
                follow, [&](const std::uint8_t* a, std::uint8_t* b) { wire.swap(1U, a, b); }, [&] { return follow.position; });
        } catch (const strix::pair::ProtocolError& e) {
            follow_error = e.what();
        }
    });
    pr::PairVerifyLeader lead(
        lead_inner, [&](const std::uint8_t* a, std::uint8_t* b) { wire.swap(0U, a, b); }, [&] { return lead_inner.position; });
    std::vector<std::uint32_t> greedy;
    const std::vector<std::uint32_t> one{3};
    lead.verify(one, greedy);
    lead.commit(1);
    lead.finish();
    t.join();
    expect(follow_error.find("position") != std::string::npos, "follower refuses a position mismatch");
}

void test_wire_and_lockstep() {
    pr::Message m;
    m.header = {5, 4, 0};
    m.commit = 3;
    m.rows = 4;
    m.position = 0x123456789ULL;
    m.tokens = {11, 22, 33, 44, 0, 0, 0, 0};
    const auto b = pr::encode(m);
    const auto d = pr::decode(b);
    expect(d.header.round == 5U && d.header.ack_round == 4U && d.header.rank == 0U && d.commit == 3U && d.rows == 4U &&
               d.position == m.position && d.tokens == m.tokens,
           "encode / decode round trip");
    auto bad = b;
    bad[24] = 9;  // rows = 9
    bool threw = false;
    try {
        (void)pr::decode(bad);
    } catch (const strix::pair::ProtocolError&) {
        threw = true;
    }
    expect(threw, "decode refuses rows > 8");
    // a peer that answers as the same rank (two followers) is refused at the first swap
    LogTarget follow;
    threw = false;
    try {
        (void)pr::pair_follow(
            follow,
            [](const std::uint8_t* a, std::uint8_t* peer) { std::copy(a, a + pr::kMessageBytes, peer); },  // echo: same rank
            [&] { return follow.position; });
    } catch (const strix::pair::ProtocolError&) {
        threw = true;
    }
    expect(threw && follow.log.empty(), "follower refuses a peer with its own rank before any target call");
}

}  // namespace

int main() {
    test_wire_and_lockstep();
    test_replay();
    test_chain_replay();
    test_position_mismatch();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("spec pair round tests PASS\n");
    return EXIT_SUCCESS;
}
