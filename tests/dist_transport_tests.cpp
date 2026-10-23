// Stream transport tests without USB4 hardware: the USB4 host transport over Unix socket pairs, whose kernel
// buffers are far smaller than the messages, so a stream that wrote its whole part before reading the peer's
// would block forever.
//  - exchanges of 64 B ... 42 MB both ways over four streams with unequal weights and a small-message threshold,
//    byte for byte;
//  - a rank whose peer dies (its streams closed mid-exchange) fails quickly with a clear error, and the
//    transport refuses later calls;
//  - a peer that stays silent hits the progress deadline instead of hanging;
//  - the same exchanges over non-blocking descriptors (the USB4 transport's own setting), and the eager path
//    (STRIX_DIST_USB4_EAGER) with parts far larger than the socket buffers: partial reads and writes on every
//    stream, both ranks writing at once;
//  - an abort from another thread ends a pending eager receive at once, and the peer sees the streams close.

#include "strix/dist/communicator.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#include <dirent.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}

struct Pair {
    std::unique_ptr<strix::dist::Transport> a, b;
};
// nonblocking: O_NONBLOCK on every descriptor before the transport sees them (the transport sets it anyway). buffer: SO_SNDBUF and
// SO_RCVBUF in bytes (0: the system's), so a part of a few kilobytes already takes several partial writes.
Pair make_pair(std::size_t streams, std::size_t min_part, std::chrono::milliseconds timeout, bool nonblocking = false, int buffer = 0) {
    std::vector<int> left, right;
    std::vector<std::string> names;
    std::vector<std::uint32_t> weights;
    for (std::size_t k = 0; k < streams; ++k) {
        int sv[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) throw std::runtime_error("socketpair");
        for (const int fd : sv) {
            if (nonblocking) ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) | O_NONBLOCK);
            if (buffer) {
                ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer));
                ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer, sizeof(buffer));
            }
        }
        left.push_back(sv[0]);
        right.push_back(sv[1]);
        names.push_back("s" + std::to_string(k));
        weights.push_back(k % 2U == 0U ? 3U : 2U); // unequal, like two cables of different rate
    }
    return {strix::dist::usb4_host_descriptors(left, names, weights, min_part, timeout),
            strix::dist::usb4_host_descriptors(right, names, weights, min_part, timeout)};
}
void test_worker_affinity() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (::sched_getaffinity(0, sizeof(allowed), &allowed)) throw std::runtime_error("get affinity");
    int cpu = 0;
    while (cpu < CPU_SETSIZE && !CPU_ISSET(cpu, &allowed)) ++cpu;
    if (cpu == CPU_SETSIZE) throw std::runtime_error("no allowed CPU");
    const auto list = std::to_string(cpu) + "," + std::to_string(cpu) + "," + std::to_string(cpu) + "," + std::to_string(cpu);
    ::setenv("STRIX_DIST_USB4_READ_CPUS", list.c_str(), 1);
    ::setenv("STRIX_DIST_USB4_WRITE_CPUS", list.c_str(), 1);
    ::setenv("STRIX_DIST_USB4_EAGER", "0", 1);
    auto pair = make_pair(4, 0, std::chrono::milliseconds(5000), true, 4096);
    ::unsetenv("STRIX_DIST_USB4_READ_CPUS");
    ::unsetenv("STRIX_DIST_USB4_WRITE_CPUS");
    ::unsetenv("STRIX_DIST_USB4_EAGER");
    unsigned found = 0;
    auto* tasks = ::opendir("/proc/self/task");
    if (!tasks) throw std::runtime_error("task inventory");
    while (const auto* entry = ::readdir(tasks)) {
        const std::string name = entry->d_name;
        if (name.empty() || name[0] == '.') continue;
        std::string comm;
        std::ifstream("/proc/self/task/" + name + "/comm") >> comm;
        if (!comm.starts_with("u4-r") && !comm.starts_with("u4-w")) continue;
        cpu_set_t actual;
        CPU_ZERO(&actual);
        expect(::sched_getaffinity(static_cast<pid_t>(std::stoi(name)), sizeof(actual), &actual) == 0 &&
               CPU_COUNT(&actual) == 1 && CPU_ISSET(cpu, &actual), "stream worker has its explicit singleton mask");
        ++found;
    }
    ::closedir(tasks);
    expect(found == 16U, "both ranks expose four named readers and four named writers");
    std::vector<std::byte> a(163840, std::byte{0x35}), b(a.size(), std::byte{0x7a}), x(a.size()), y(a.size());
    std::exception_ptr error;
    std::thread peer([&] { try { pair.b->exchange(b, y); } catch (...) { error = std::current_exception(); } });
    pair.a->exchange(a, x);
    peer.join();
    if (error) std::rethrow_exception(error);
    expect(x == b && y == a, "pinned stream workers preserve 160 KB both ways with partial I/O");
}

void test_invalid_affinity_closes_owned_descriptors() {
    for (const auto& value : {std::to_string(CPU_SETSIZE), std::string("0,1")}) {
        int sv[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) throw std::runtime_error("socketpair");
        ::setenv("STRIX_DIST_USB4_READ_CPUS", value.c_str(), 1);
        bool rejected = false;
        try { (void)strix::dist::usb4_host_descriptors({sv[0]}, {"s"}, {1U}, 0, std::chrono::milliseconds(100)); }
        catch (const std::exception&) { rejected = true; }
        ::unsetenv("STRIX_DIST_USB4_READ_CPUS");
        expect(rejected, "invalid CPU or incomplete mapping rejected");
        expect(::fcntl(sv[0], F_GETFD) == -1 && errno == EBADF, "affinity failure closes the owned descriptor");
        ::close(sv[1]);
    }
}

std::vector<std::byte> pattern(std::size_t size, unsigned seed) {
    std::vector<std::byte> v(size);
    for (std::size_t i = 0; i < size; ++i) v[i] = static_cast<std::byte>((i * 2654435761U + seed) >> 13U);
    return v;
}

void test_sizes(bool nonblocking) {
    auto pair = make_pair(4, 4096, std::chrono::milliseconds(20000), nonblocking);
    std::size_t checked = 0;
    for (const std::size_t size : {std::size_t{64}, std::size_t{76}, std::size_t{20480}, std::size_t{20544}, std::size_t{122880},
                                   std::size_t{5} << 20U, std::size_t{21} << 20U, std::size_t{42} << 20U}) {
        const auto out_a = pattern(size, 1U), out_b = pattern(size, 2U);
        std::vector<std::byte> in_a(size), in_b(size);
        std::exception_ptr error;
        std::thread peer([&] { try { pair.b->exchange(out_b, in_b); } catch (...) { error = std::current_exception(); } });
        const auto start = std::chrono::steady_clock::now();
        pair.a->exchange(out_a, in_a);
        peer.join();
        if (error) std::rethrow_exception(error);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        expect(in_a == out_b && in_b == out_a, "exchange of " + std::to_string(size) + " bytes delivers both messages intact");
        if (size >= (std::size_t{21} << 20U)) std::cout << "socket-pair exchange " << (size >> 20U) << " MB: " << ms << " ms\n";
        ++checked;
    }
    // send / receive in both orders.
    const auto msg = pattern(std::size_t{42} << 20U, 3U);
    std::vector<std::byte> got(msg.size());
    std::thread receiver([&] { pair.b->receive(got); });
    pair.a->send(msg);
    receiver.join();
    expect(got == msg, "42 MB send/receive");
    std::cout << "stream transport (descriptors handed over " << (nonblocking ? "non-blocking" : "blocking") << "): " << checked
              << " exchange sizes up to 42 MB and a 42 MB send intact over 4 socket pairs\n";
}

// Eager messages over non-blocking streams whose buffers hold a few kilobytes: each part needs many partial writes
// and reads, and both ranks write at the same time, so a rank that wrote its whole message before reading would
// wait for the peer's read while the peer waits for its own: both would stop at the progress deadline.
void test_eager_partial() {
    ::setenv("STRIX_DIST_USB4_EAGER", "65536", 1);
    auto pair = make_pair(2, 0, std::chrono::milliseconds(3000), true, 4096);
    auto single = make_pair(1, 0, std::chrono::milliseconds(3000), true, 4096);
    ::unsetenv("STRIX_DIST_USB4_EAGER");
    expect(pair.a->name().find("eager up to 65536") != std::string::npos, "the eager threshold is in effect: " + pair.a->name());
    double worst = 0.0;
    std::size_t checked = 0;
    for (auto* p : {&pair, &single})
        for (const std::size_t size : {std::size_t{64}, std::size_t{576}, std::size_t{8256}, std::size_t{32768}, std::size_t{65536},
                                       std::size_t{1} << 20U /* above the threshold: the stream threads */, std::size_t{65536}}) {
            const auto out_a = pattern(size, 5U), out_b = pattern(size, 6U);
            std::vector<std::byte> in_a(size), in_b(size);
            std::string error_a, error_b;
            const auto start = std::chrono::steady_clock::now();
            std::thread peer([&] { try { p->b->exchange(out_b, in_b); } catch (const std::exception& e) { error_b = e.what(); } });
            try { p->a->exchange(out_a, in_a); } catch (const std::exception& e) { error_a = e.what(); }
            peer.join();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            worst = std::max(worst, ms);
            expect(error_a.empty() && error_b.empty(), "eager exchange of " + std::to_string(size) + " B completes (" + error_a + " / " + error_b + ")");
            if (!error_a.empty() || !error_b.empty()) return; // the transports are unusable after a failure
            expect(in_a == out_b && in_b == out_a, "eager exchange of " + std::to_string(size) + " B delivers both messages intact");
            expect(ms < 1000.0, "eager exchange of " + std::to_string(size) + " B within 1 s (" + std::to_string(ms) + " ms)");
            ++checked;
        }
    // send / receive (one direction) on the eager path
    const auto msg = pattern(65536, 7U);
    std::vector<std::byte> got(msg.size());
    std::thread receiver([&] { pair.b->receive(got); });
    pair.a->send(msg);
    receiver.join();
    expect(got == msg, "eager 64 KiB send/receive");
    std::cout << "eager over non-blocking 4 KiB-buffer streams: " << checked << " exchanges up to 64 KiB (and 1 MiB on the stream threads) intact, slowest "
              << worst << " ms\n";
}

// An abort (the communicator's bounded wait expired) ends this rank's pending eager receive at once with the abort's
// reason, and the peer's pending receive sees the streams close instead of waiting for its own deadline.
void test_eager_abort() {
    ::setenv("STRIX_DIST_USB4_EAGER", "65536", 1);
    auto pair = make_pair(2, 0, std::chrono::milliseconds(20000), true);
    ::unsetenv("STRIX_DIST_USB4_EAGER");
    std::vector<std::byte> in_a(4096), in_b(4096);
    std::string error_a, error_b;
    double ms_a = 0.0, ms_b = 0.0;
    const auto start = std::chrono::steady_clock::now();
    const auto since = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); };
    std::thread a([&] { try { pair.a->receive(in_a); } catch (const std::exception& e) { error_a = e.what(); } ms_a = since(); });
    std::thread b([&] { try { pair.b->receive(in_b); } catch (const std::exception& e) { error_b = e.what(); } ms_b = since(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pair.a->abort("test: the producer did not complete");
    a.join();
    b.join();
    expect(error_a.find("the producer did not complete") != std::string::npos && ms_a < 1000.0,
           "the aborted rank's eager receive ends with the abort's reason (" + std::to_string(ms_a) + " ms: " + error_a + ")");
    expect(error_b.find("closed the stream") != std::string::npos && ms_b < 1000.0,
           "the peer's eager receive sees the streams close (" + std::to_string(ms_b) + " ms: " + error_b + ")");
    std::string later;
    try { pair.a->exchange(in_a, in_b); } catch (const std::exception& e) { later = e.what(); }
    expect(later.find("unusable") != std::string::npos && later.find("the producer did not complete") != std::string::npos,
           "an aborted transport refuses later calls with the reason: " + later);
    std::cout << "eager abort: aborted rank after " << ms_a << " ms, peer after " << ms_b << " ms\n";
}

void test_peer_failure() {
    auto pair = make_pair(4, 0, std::chrono::milliseconds(20000));
    const auto out = pattern(std::size_t{8} << 20U, 4U);
    std::vector<std::byte> in(out.size());
    std::string message;
    const auto start = std::chrono::steady_clock::now();
    std::thread killer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        pair.b.reset(); // the peer rank dies: all its streams close
    });
    try { pair.a->exchange(out, in); } catch (const std::exception& e) { message = e.what(); }
    killer.join();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    expect(message.find("closed the stream") != std::string::npos, "an exchange whose peer dies fails with the cause: " + message);
    expect(ms < 2000.0, "the failure is reported promptly (" + std::to_string(ms) + " ms)");
    std::string later;
    try { pair.a->exchange(out, in); } catch (const std::exception& e) { later = e.what(); }
    expect(later.find("unusable") != std::string::npos, "the transport refuses use after a failure: " + later);
    std::cout << "peer death: failed after " << ms << " ms: " << message << '\n';
}

void test_silent_peer() {
    auto pair = make_pair(2, 0, std::chrono::milliseconds(300));
    std::vector<std::byte> in(4096);
    std::string message;
    const auto start = std::chrono::steady_clock::now();
    try { pair.a->receive(in); } catch (const std::exception& e) { message = e.what(); }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    expect(message.find("no progress") != std::string::npos, "a silent peer hits the deadline: " + message);
    expect(ms < 1500.0, "the deadline is honoured (" + std::to_string(ms) + " ms)");
    std::cout << "silent peer: failed after " << ms << " ms: " << message << '\n';
}
} // namespace

int main() try {
    test_worker_affinity();
    test_invalid_affinity_closes_owned_descriptors();
    ::signal(SIGPIPE, SIG_IGN); // sockets raise SIGPIPE when the peer is gone; the stream device returns an error
    test_sizes(false);
    test_sizes(true);
    test_eager_partial();
    test_eager_abort();
    test_peer_failure();
    test_silent_peer();
    if (failures == 0) std::cout << "dist transport tests passed\n";
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "dist transport tests: " << error.what() << '\n';
    return 2;
}
