// Stream transport tests without USB4 hardware: the USB4 host transport over Unix socket pairs, whose kernel
// buffers are far smaller than the messages, so a stream that wrote its whole part before reading the peer's
// would block forever.
//  - exchanges of 64 B ... 42 MB both ways over four streams with unequal weights and a small-message threshold,
//    byte for byte;
//  - a rank whose peer dies (its streams closed mid-exchange) fails quickly with a clear error, and the
//    transport refuses later calls;
//  - a peer that stays silent hits the progress deadline instead of hanging.

#include "strix/dist/communicator.hpp"

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
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
Pair make_pair(std::size_t streams, std::size_t min_part, std::chrono::milliseconds timeout) {
    std::vector<int> left, right;
    std::vector<std::string> names;
    std::vector<std::uint32_t> weights;
    for (std::size_t k = 0; k < streams; ++k) {
        int sv[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) throw std::runtime_error("socketpair");
        left.push_back(sv[0]);
        right.push_back(sv[1]);
        names.push_back("s" + std::to_string(k));
        weights.push_back(k % 2U == 0U ? 3U : 2U); // unequal, like two cables of different rate
    }
    return {strix::dist::usb4_host_descriptors(left, names, weights, min_part, timeout),
            strix::dist::usb4_host_descriptors(right, names, weights, min_part, timeout)};
}
std::vector<std::byte> pattern(std::size_t size, unsigned seed) {
    std::vector<std::byte> v(size);
    for (std::size_t i = 0; i < size; ++i) v[i] = static_cast<std::byte>((i * 2654435761U + seed) >> 13U);
    return v;
}

void test_sizes() {
    auto pair = make_pair(4, 4096, std::chrono::milliseconds(20000));
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
    std::cout << "stream transport: " << checked << " exchange sizes up to 42 MB and a 42 MB send intact over 4 socket pairs\n";
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
    ::signal(SIGPIPE, SIG_IGN); // sockets raise SIGPIPE when the peer is gone; the stream device returns an error
    test_sizes();
    test_peer_failure();
    test_silent_peer();
    if (failures == 0) std::cout << "dist transport tests passed\n";
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "dist transport tests: " << error.what() << '\n';
    return 2;
}
