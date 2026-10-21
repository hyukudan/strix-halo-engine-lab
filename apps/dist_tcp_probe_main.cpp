// Two-rank transport probe (no GPU): ping-pong round trips, after an idle gap, and one-way throughput of
// whole messages over the distributed runtime's transports.
//   rank 0: strix-dist-tcp-probe listen PORT [SIZES...]       | usb4 0 DEV[,DEV...] [SIZES...] | auto 0 [SIZES...]
//   rank 1: strix-dist-tcp-probe connect HOST PORT [SIZES...] | usb4 1 DEV[,DEV...] [SIZES...] | auto 1 [SIZES...]
// "auto" pairs every live USB4 stream by handshake (usb4_host_auto) and prints the topology receipt.
// STRIX_PROBE_GAPS_US=0,100,1000,3000 sets the idle gaps before each round trip (default 0).
// STRIX_PROBE_KEEPALIVE_US=N fills each gap with 64-byte exchanges every N us instead of sleeping (an
// experiment: does traffic during the gap keep the receive path from adding delay?).
// STRIX_PROBE_ITERATIONS=N overrides the number of timed round trips per size and gap.
// Prints one JSON line per message size and gap (rank 0).

#include "strix/dist/communicator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream in(text);
    for (std::string item; std::getline(in, item, ',');) out.push_back(item);
    return out;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::cerr << "usage: strix-dist-tcp-probe listen PORT [SIZES...] | connect HOST PORT [SIZES...] | usb4 RANK DEV[,DEV] [SIZES...]\n";
        return 2;
    }
    const std::string mode = argv[1];
    const auto timeout = std::chrono::milliseconds(60000);
    std::unique_ptr<strix::dist::Transport> transport;
    bool leader = false;
    int next = 0;
    if (mode == "listen") { transport = strix::dist::tcp_listen(static_cast<std::uint16_t>(std::stoi(argv[2])), timeout); leader = true; next = 3; }
    else if (mode == "connect") { transport = strix::dist::tcp_connect(argv[2], static_cast<std::uint16_t>(std::stoi(argv[3])), timeout); next = 4; }
    else if (mode == "usb4") { transport = strix::dist::usb4_host(split(argv[3])); leader = std::string(argv[2]) == "0"; next = 4; }
    else if (mode == "auto") {
        strix::dist::Usb4AutoOptions options;
        options.rank = std::string(argv[2]) == "0" ? 0U : 1U;
        strix::dist::Usb4Topology topology;
        const auto begin = std::chrono::steady_clock::now();
        transport = strix::dist::usb4_host_auto(options, &topology);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        std::cout << topology.receipt() << " (pairing " << ms << " ms)\n" << std::flush;
        leader = options.rank == 0U;
        next = 3;
    }
    else throw std::invalid_argument("unknown mode " + mode);
    std::vector<std::size_t> sizes;
    for (; next < argc; ++next) sizes.push_back(std::stoull(argv[next]));
    if (sizes.empty()) sizes = {64, 4096, 20480, 102400, 409600};
    unsigned keepalive = 0;
    if (const char* k = std::getenv("STRIX_PROBE_KEEPALIVE_US")) keepalive = static_cast<unsigned>(std::stoul(k));
    std::vector<std::byte> tick(64, std::byte{0x11}), tock(64);
    std::vector<unsigned> gaps{0};
    if (const char* g = std::getenv("STRIX_PROBE_GAPS_US")) {
        gaps.clear();
        for (const auto& item : split(g)) gaps.push_back(static_cast<unsigned>(std::stoul(item)));
    }
    for (const auto size : sizes)
        for (const auto gap : gaps) {
            std::vector<std::byte> out(size, std::byte{0x5a}), in(size);
            unsigned iterations = size > (std::size_t{1} << 20U) ? 10U : gap >= 1000U ? 200U : size <= 4096 ? 2000U : 300U;
            if (const char* it = std::getenv("STRIX_PROBE_ITERATIONS"); it && *it) iterations = static_cast<unsigned>(std::stoul(it));
            std::vector<double> rtt, xchg;
            for (unsigned i = 0; i < iterations + (iterations >= 100U ? 20U : 2U); ++i) {
                if (gap && keepalive) {
                    // Both ranks run the same number of keep-alive exchanges, then the timed round trip.
                    for (unsigned t = 0; t < gap / keepalive; ++t) {
                        std::this_thread::sleep_for(std::chrono::microseconds(keepalive));
                        transport->exchange(tick, tock);
                    }
                } else if (gap) {
                    std::this_thread::sleep_for(std::chrono::microseconds(gap));
                }
                auto start = std::chrono::steady_clock::now();
                if (leader) { transport->send(out); transport->receive(in); }
                else { transport->receive(in); transport->send(out); }
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                // Symmetric exchange (both ranks send and receive at once), the pattern of one layer.
                start = std::chrono::steady_clock::now();
                transport->exchange(out, in);
                const double x = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                if (i >= (iterations >= 100U ? 20U : 2U)) { rtt.push_back(us); xchg.push_back(x); }
            }
            std::sort(rtt.begin(), rtt.end());
            std::sort(xchg.begin(), xchg.end());
            const auto at = [](const std::vector<double>& v, double q) { return v[std::min(v.size() - 1U, static_cast<std::size_t>(q * static_cast<double>(v.size())))]; };
            if (leader)
                std::cout << "{\"transport\":\"" << transport->name() << "\",\"bytes\":" << size << ",\"gap_us\":" << gap
                          << ",\"keepalive_us\":" << keepalive
                          << ",\"rtt_p50_us\":" << at(rtt, 0.5) << ",\"rtt_p99_us\":" << at(rtt, 0.99) << ",\"exchange_p50_us\":"
                          << at(xchg, 0.5) << ",\"exchange_p99_us\":" << at(xchg, 0.99) << "}\n";
        }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-dist-tcp-probe: " << error.what() << '\n';
    return 1;
}
