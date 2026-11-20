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
        const std::uint64_t gen = gen_;
        std::copy(mine, mine + pr::kMessageBytes, slot_[rank].begin());
        if (++arrived_ == 2U) {
            arrived_ = 0;
            ++gen_;
            cv_.notify_all();
        } else {
            cv_.wait(lk, [&] { return gen_ != gen; });
        }
        std::copy(slot_[1U - rank].begin(), slot_[1U - rank].end(), peer);
        // the second reader of this generation must still see both slots: wait until both read before the next write
        if (++read_ == 2U) {
            read_ = 0;
            ++rgen_;
            cv_.notify_all();
        } else {
            const std::uint64_t r = rgen_;
            cv_.wait(lk, [&] { return rgen_ != r; });
        }
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::array<std::array<std::uint8_t, pr::kMessageBytes>, 2> slot_{};
    std::uint32_t arrived_ = 0, read_ = 0;
    std::uint64_t gen_ = 0, rgen_ = 0;
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
    test_position_mismatch();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("spec pair round tests PASS\n");
    return EXIT_SUCCESS;
}
