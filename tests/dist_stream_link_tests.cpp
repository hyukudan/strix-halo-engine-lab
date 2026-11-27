// Host tests of the per-stream integrity of the USB4 host transport (src/dist_stream_link.hpp): two ranks talk through
// a relay that cuts each direction into 4 KiB frames and can drop, damage or duplicate them, the way the link below the
// streams loses a frame. Invariant: every message arrives bit-exact or the transport fails with a named error; no
// silent corruption, no stall past the loss clock.
#include "strix/dist/communicator.hpp"
#include "strix/dist/host_copy.hpp"
#include "strix/dist/transport.hpp"

#include "dist_stream_link.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace dist = strix::dist;

namespace strix::dist {
// The CPU stub's device memory is host memory (the transport file links the mapped-host copy hook).
void enqueue_host_d2h_copy(void* mapped_host, const void* device, std::size_t bytes, hipStream_t) {
    std::memcpy(mapped_host, device, bytes);
}
}  // namespace strix::dist

namespace {

int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// What the relay does with the frame that holds byte `offset` of one direction.
struct Fault {
    enum Kind { none, drop, damage, duplicate, drop_from } kind = none;
    std::uint64_t offset = 0;
    int direction = 0;  // 0: rank 0 -> rank 1, 1: rank 1 -> rank 0
    std::size_t stream = 0;
};

// One stream: rank 0 end, rank 1 end, and two relay threads (one per direction) in between.
struct RelayStream {
    int rank_fd[2] = {-1, -1};
    int relay_fd[2] = {-1, -1};
    std::thread pump[2];
    std::atomic<bool> stop{false};
    std::atomic<int> faults_done{0};
};

class Relay {
public:
    Relay(std::size_t streams, std::vector<Fault> faults) : faults_(std::move(faults)) {
        for (std::size_t s = 0; s < streams; ++s) {
            auto rs = std::make_unique<RelayStream>();
            for (int r = 0; r < 2; ++r) {
                int sv[2];
                if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) throw std::runtime_error("socketpair");
                rs->rank_fd[r] = sv[0];
                rs->relay_fd[r] = sv[1];
            }
            streams_.push_back(std::move(rs));
        }
        for (std::size_t s = 0; s < streams_.size(); ++s)
            for (int dir = 0; dir < 2; ++dir) streams_[s]->pump[dir] = std::thread([this, s, dir] { run(s, dir); });
    }
    ~Relay() {
        for (auto& rs : streams_) {
            rs->stop = true;
            for (int r = 0; r < 2; ++r) ::shutdown(rs->relay_fd[r], SHUT_RDWR);
        }
        for (auto& rs : streams_)
            for (auto& t : rs->pump)
                if (t.joinable()) t.join();
        for (auto& rs : streams_)
            for (int r = 0; r < 2; ++r) ::close(rs->relay_fd[r]);
    }
    std::vector<int> fds(int rank) const {
        std::vector<int> out;
        for (const auto& rs : streams_) out.push_back(rs->rank_fd[rank]);
        return out;
    }
    int applied() const {
        int n = 0;
        for (const auto& rs : streams_) n += rs->faults_done.load();
        return n;
    }

private:
    void run(std::size_t s, int dir) {
        auto& rs = *streams_[s];
        const int from = rs.relay_fd[dir], to = rs.relay_fd[1 - dir];
        std::vector<char> frame(4096);
        std::uint64_t at = 0;
        bool dropping = false;
        while (!rs.stop) {
            const auto n = ::read(from, frame.data(), frame.size());
            if (n <= 0) break;
            const auto len = static_cast<std::uint64_t>(n);
            bool drop = dropping, twice = false;
            for (const auto& f : faults_) {
                if (f.stream != s || f.direction != dir || f.offset < at || f.offset >= at + len) continue;
                rs.faults_done.fetch_add(1);
                if (f.kind == Fault::drop) drop = true;
                if (f.kind == Fault::drop_from) drop = dropping = true;
                if (f.kind == Fault::damage) frame[static_cast<std::size_t>(f.offset - at)] ^= 0x5a;
                if (f.kind == Fault::duplicate) twice = true;
            }
            at += len;
            if (drop) continue;
            for (int copy = 0; copy < (twice ? 2 : 1); ++copy) {
                std::size_t done = 0;
                while (done < len) {
                    const auto w = ::write(to, frame.data() + done, static_cast<std::size_t>(len) - done);
                    if (w <= 0) return;
                    done += static_cast<std::size_t>(w);
                }
            }
        }
    }
    std::vector<Fault> faults_;
    std::vector<std::unique_ptr<RelayStream>> streams_;
};

std::vector<std::byte> pattern(std::size_t size, unsigned seed) {
    std::vector<std::byte> v(size);
    std::uint32_t x = seed * 2654435761U + 1U;
    for (auto& b : v) {
        x ^= x << 13U;
        x ^= x >> 17U;
        x ^= x << 5U;
        b = static_cast<std::byte>(x & 0xffU);
    }
    return v;
}

using Step = std::function<void(dist::Transport&, int rank)>;

// Runs both ranks over a relay; returns the error text of each rank ("" when it finished).
std::array<std::string, 2> run_pair(std::size_t streams, std::vector<Fault> faults, const Step& step, std::string* state0 = nullptr,
                                    std::chrono::milliseconds timeout = std::chrono::milliseconds(8000)) {
    Relay relay(streams, std::move(faults));
    std::array<std::string, 2> errors;
    std::array<std::unique_ptr<dist::Transport>, 2> t;
    std::vector<std::string> names;
    std::vector<std::uint32_t> weights;
    for (std::size_t s = 0; s < streams; ++s) {
        names.push_back("s" + std::to_string(s));
        weights.push_back(1U);
    }
    for (int r = 0; r < 2; ++r) {
        std::vector<int> fds;
        for (const int fd : relay.fds(r)) fds.push_back(::dup(fd));
        t[static_cast<std::size_t>(r)] = dist::usb4_host_descriptors(fds, names, weights, 4096, timeout);
    }
    std::thread ranks[2];
    for (int r = 0; r < 2; ++r)
        ranks[r] = std::thread([&, r] {
            try {
                step(*t[static_cast<std::size_t>(r)], r);
            } catch (const std::exception& e) {
                errors[static_cast<std::size_t>(r)] = e.what();
                t[static_cast<std::size_t>(r)]->abort(e.what());
            }
        });
    for (auto& th : ranks) th.join();
    if (state0) *state0 = t[0]->state() + " | " + t[1]->state();
    t[0].reset();
    t[1].reset();
    return errors;
}

// Both directions, sizes on both paths (eager up to 64 KiB, stream threads above), exchange and one-way.
void traffic(dist::Transport& t, int rank, int rounds, bool* bad) {
    const std::size_t sizes[] = {1, 100, 4096, 65536, 65537, 300 * 1024, 1 << 20, (5 << 20) + 3};
    unsigned seed = 1;
    for (int round = 0; round < rounds; ++round)
        for (const auto size : sizes) {
            const auto a = pattern(size, seed), b = pattern(size, seed + 1000U);
            ++seed;
            std::vector<std::byte> in(size);
            t.exchange(rank == 0 ? a : b, in);
            if (in != (rank == 0 ? b : a)) *bad = true;
            if (rank == 0) t.send(a);
            else {
                t.receive(in);
                if (in != a) *bad = true;
            }
            if (rank == 1) t.send(b);
            else {
                t.receive(in);
                if (in != b) *bad = true;
            }
        }
}

void test_unit_format() {
    dist::link::Unit u;
    u.kind = dist::link::kind_data;
    u.seq = 7;
    u.message = 100;
    u.offset = 50;
    u.size = 50;
    u = dist::link::sealed(u);
    require(dist::link::valid(u), "a sealed DATA unit is valid");
    auto bent = u;
    bent.seq ^= 1U;
    require(!dist::link::valid(bent), "one flipped bit in the header is refused");
    auto past = u;
    past.size = 51;
    past = dist::link::sealed(past);
    require(!dist::link::valid(past), "a piece past its message end is refused");
}

void test_clean() {
    std::array<bool, 2> bad{};
    std::string state;
    const auto errors = run_pair(2, {}, [&](dist::Transport& t, int r) { traffic(t, r, 2, &bad[static_cast<std::size_t>(r)]); }, &state);
    require(errors[0].empty() && errors[1].empty(), "clean run: " + errors[0] + " / " + errors[1]);
    require(!bad[0] && !bad[1], "clean run: bytes equal");
    require(state.find("usb4 integrity: clean") != std::string::npos && state.find("REPAIRED") == std::string::npos,
            "clean run reports clean: " + state);
}

// One fault in one direction of one stream at a byte offset of that direction's stream; the traffic must still match.
void test_repaired(const char* what, Fault fault) {
    std::array<bool, 2> bad{};
    std::string state;
    const auto t0 = std::chrono::steady_clock::now();
    const auto errors = run_pair(2, {fault}, [&](dist::Transport& t, int r) { traffic(t, r, 1, &bad[static_cast<std::size_t>(r)]); }, &state);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    require(errors[0].empty() && errors[1].empty(), std::string(what) + ": " + errors[0] + " / " + errors[1]);
    require(!bad[0] && !bad[1], std::string(what) + ": bytes equal");
    require(state.find("REPAIRED") != std::string::npos, std::string(what) + ": the repair is counted: " + state);
    std::printf("  %-34s %5lld ms  %s\n", what, static_cast<long long>(ms), state.substr(state.find("usb4 integrity")).substr(0, 160).c_str());
}

// Both ranks send before either receives (each part larger than the eager limit, within what the streams buffer, as
// the raw transport needs too): no deadlock, the acknowledgement wait lingers and keeps a copy.
void test_symmetric_sends() {
    std::array<bool, 2> bad{};
    const auto errors = run_pair(2, {}, [&](dist::Transport& t, int r) {
        const auto mine = pattern(150000, static_cast<unsigned>(10 + r)), theirs = pattern(150000, static_cast<unsigned>(11 - r));
        std::vector<std::byte> in(mine.size());
        t.send(mine);
        t.receive(in);
        if (in != theirs) bad[static_cast<std::size_t>(r)] = true;
    });
    require(errors[0].empty() && errors[1].empty(), "symmetric sends: " + errors[0] + " / " + errors[1]);
    require(!bad[0] && !bad[1], "symmetric sends: bytes equal");
}

// A direction that loses everything from some point: a named hard error on a bounded clock, never a silent result.
void test_dead_direction() {
    const auto t0 = std::chrono::steady_clock::now();
    const auto errors = run_pair(1, {{Fault::drop_from, 200000, 0, 0}}, [&](dist::Transport& t, int r) {
        bool bad = false;
        traffic(t, r, 1, &bad);
        if (bad) throw std::runtime_error("silent corruption");
    }, nullptr, std::chrono::milliseconds(4000));
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    require(!errors[0].empty() || !errors[1].empty(), "dead direction fails");
    require(errors[0].find("silent") == std::string::npos && errors[1].find("silent") == std::string::npos, "dead direction: no silent corruption");
    require(ms < 20000, "dead direction fails within the bounded clock (" + std::to_string(ms) + " ms)");
    std::printf("  dead direction: %lld ms: %s | %s\n", static_cast<long long>(ms), errors[0].substr(0, 140).c_str(), errors[1].substr(0, 140).c_str());
}

// STRIX_DIST_USB4_SMALL_RESEND=0: damage in a small message sent without a copy is detected and named, not delivered.
void test_small_damage() {
    ::setenv("STRIX_DIST_USB4_SMALL_RESEND", "0", 1);
    std::array<bool, 2> bad{};
    const auto errors = run_pair(1, {{Fault::damage, 64 + 500, 0, 0}}, [&](dist::Transport& t, int r) {
        const auto a = pattern(1000, 5), b = pattern(1000, 6);
        std::vector<std::byte> in(1000);
        t.exchange(r == 0 ? a : b, in);
        if (in != (r == 0 ? b : a)) bad[static_cast<std::size_t>(r)] = true;
    }, nullptr, std::chrono::milliseconds(4000));
    require(!bad[0] && !bad[1], "small damage: never delivered silently");
    const bool named = errors[0].find("not repairable") != std::string::npos || errors[1].find("not repairable") != std::string::npos;
    require(named, "small damage: named as not repairable: " + errors[0] + " / " + errors[1]);
    ::unsetenv("STRIX_DIST_USB4_SMALL_RESEND");
}

// Size disagreement between the ranks is a named error (the peer's size is in the unit).
void test_size_mismatch() {
    const auto errors = run_pair(1, {}, [&](dist::Transport& t, int r) {
        if (r == 0) t.send(pattern(200000, 1));
        else {
            std::vector<std::byte> in(100000);
            t.receive(in);
        }
    }, nullptr, std::chrono::milliseconds(3000));
    require(errors[1].find("size mismatch") != std::string::npos, "size mismatch named: " + errors[1]);
}

// Random faults (kinds, directions, streams, offsets) over two rounds of traffic on four streams: every run either
// delivers every byte exactly or fails by name (here: always repaired, a handful of faults never exhausts the resends).
void test_random(unsigned seed) {
    std::uint32_t x = seed * 747796405U + 2891336453U;
    const auto next = [&] {
        x ^= x << 13U;
        x ^= x >> 17U;
        x ^= x << 5U;
        return x;
    };
    std::vector<Fault> faults;
    for (int i = 0; i < 4; ++i) {
        Fault f;
        f.kind = static_cast<Fault::Kind>(1U + next() % 3U);
        f.direction = static_cast<int>(next() % 2U);
        f.stream = next() % 4U;
        f.offset = next() % 3000000U;
        faults.push_back(f);
    }
    std::array<bool, 2> bad{};
    std::string state;
    const auto errors = run_pair(4, faults, [&](dist::Transport& t, int r) { traffic(t, r, 2, &bad[static_cast<std::size_t>(r)]); }, &state);
    require(errors[0].empty() && errors[1].empty(), "random seed " + std::to_string(seed) + ": " + errors[0] + " / " + errors[1]);
    require(!bad[0] && !bad[1], "random seed " + std::to_string(seed) + ": bytes equal");
}

// Host cost of the integrity layer (no relay: socket pairs straight between the ranks, faster than a USB4 stream, so
// this is the CPU cost bound, not the link time): a 64 MiB exchange on 4 streams and 2000 small eager exchanges.
void cost(const char* label) {
    std::array<int, 4> a{}, b{};
    for (std::size_t s = 0; s < 4; ++s) {
        int sv[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) throw std::runtime_error("socketpair");
        a[s] = sv[0];
        b[s] = sv[1];
    }
    const std::vector<std::string> names{"s0", "s1", "s2", "s3"};
    const std::vector<std::uint32_t> weights{1U, 1U, 1U, 1U};
    std::array<std::unique_ptr<dist::Transport>, 2> t{dist::usb4_host_descriptors({a.begin(), a.end()}, names, weights, 4096, std::chrono::milliseconds(8000)),
                                                    dist::usb4_host_descriptors({b.begin(), b.end()}, names, weights, 4096, std::chrono::milliseconds(8000))};
    std::array<double, 2> big_ms{}, small_us{};
    std::thread ranks[2];
    for (int r = 0; r < 2; ++r)
        ranks[r] = std::thread([&, r] {
            const auto out = pattern(std::size_t{64} << 20, static_cast<unsigned>(r));
            std::vector<std::byte> in(out.size());
            auto& tr = *t[static_cast<std::size_t>(r)];
            const bool big = std::string(std::getenv("STRIX_LINK_TEST_COST")) != "small";
            if (big) tr.exchange(out, in);  // warm
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; big && i < 4; ++i) tr.exchange(out, in);
            big_ms[static_cast<std::size_t>(r)] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 4.0;
            std::vector<std::byte> so(8192), si(8192);
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < 2000; ++i) tr.exchange(so, si);
            small_us[static_cast<std::size_t>(r)] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / 2000.0;
        });
    for (auto& th : ranks) th.join();
    std::printf("  cost %-14s 64 MiB exchange %.2f ms (%.2f GB/s each way), 8 KiB eager exchange %.1f us\n", label, big_ms[0],
                64.0 * 1.048576 / big_ms[0], small_us[0]);
}

}  // namespace

int main() {
    ::setenv("STRIX_DIST_USB4_LOSS_MS", "100", 1);
    ::setenv("STRIX_DIST_USB4_ACK_MS", "300", 1);
    ::setenv("STRIX_DIST_USB4_RESENDS", "3", 1);
    test_unit_format();
    test_clean();
    std::printf("repairs:\n");
    test_repaired("drop mid message 0->1", {Fault::drop, 3u << 20, 0, 0});
    test_repaired("drop mid message 1->0 stream 1", {Fault::drop, 2u << 20, 1, 1});
    test_repaired("drop first small unit", {Fault::drop, 10, 0, 0});
    test_repaired("damage small payload", {Fault::damage, 64 + 30, 1, 0});
    test_repaired("damage payload byte", {Fault::damage, (1u << 20) + 12345, 0, 0});
    test_repaired("damage header byte", {Fault::damage, (4u << 20) + 3, 1, 0});
    test_repaired("duplicate frame", {Fault::duplicate, (2u << 20) + 7, 0, 1});
    test_repaired("drop late (tail region)", {Fault::drop, 6600000, 0, 0});
    test_symmetric_sends();
    test_dead_direction();
    test_small_damage();
    test_size_mismatch();
    for (unsigned seed = 1; seed <= 12; ++seed) test_random(seed);
    if (std::getenv("STRIX_LINK_TEST_COST")) {
        ::setenv("STRIX_DIST_USB4_INTEGRITY", "0", 1);
        cost("integrity off");
        ::unsetenv("STRIX_DIST_USB4_INTEGRITY");
        cost("integrity on");
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("dist stream link tests: PASS\n");
    return 0;
}
