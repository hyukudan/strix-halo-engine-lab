#pragma once
// Measured cost of ONE rank-to-rank exchange, for the pair planners (pair/chain.hpp Link{bytes_per_ms, latency_ms}, the
// column-split trip cost): cost(bytes) = latency_us + us_per_byte x bytes, fitted at start-up from timed symmetric exchanges
// over whatever exchange form the engine runs (dist::Transport::exchange host bytes, Communicator staged, xport::Channel), so
// no link number of one laboratory is compiled in. Both ranks then hold the SAME bits (agree(): field-wise max over one
// exchange), so both make the same placement decision.
// Measured anchor (two nodes, USB4 host path, device -> device, back to back): 12 KB 36.8 us, 96 KB 57.1, 192 KB 98.4,
// 544 KB 193.2 -> latency ~34 us + 3.5 GB/s, every point within 10 % (tests/dist_link_cost_tests.cpp).
// Backend-neutral host code: timing is std::chrono::steady_clock around the callable, which must return only when its
// exchange is complete (for a device form: after the stream wait it ends with).

#include "strix/dist/transport.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::dist {

struct LinkSample {
    std::uint64_t bytes = 0;  // per direction of one symmetric exchange
    double p50_us = 0.0, p90_us = 0.0;
};

struct LinkCost {
    double latency_us = 0.0;   // per exchange (intercept of the p50 fit)
    double us_per_byte = 0.0;  // per byte and direction, both directions moving at once
    double tail_us = 0.0;      // largest p90 - p50 of the samples (for planners that price the tail)
    [[nodiscard]] bool valid() const {
        return std::isfinite(latency_us) && latency_us > 0.0 && std::isfinite(us_per_byte) && us_per_byte >= 0.0 &&
               std::isfinite(tail_us) && tail_us >= 0.0;
    }
    [[nodiscard]] double exchange_us(std::uint64_t bytes) const { return latency_us + us_per_byte * static_cast<double>(bytes); }
    [[nodiscard]] double latency_ms() const { return latency_us / 1000.0; }
    // 0 = no rate term measured (one size, or a flat fit): planners treat it as latency-only.
    [[nodiscard]] double bytes_per_ms() const { return us_per_byte > 0.0 ? 1000.0 / us_per_byte : 0.0; }
    [[nodiscard]] std::string describe() const {
        std::ostringstream s;
        s.setf(std::ios::fixed);
        s.precision(1);
        s << "link_cost latency_us=" << latency_us << " gbytes_ps=" << (us_per_byte > 0.0 ? 1e-3 / us_per_byte : 0.0)
          << " tail_us=" << tail_us;
        return s.str();
    }
};

// Nearest-rank percentile, q in [0, 1].
inline double percentile(std::vector<double> values, double q) {
    if (values.empty()) throw std::invalid_argument("link cost: percentile of no values");
    std::sort(values.begin(), values.end());
    const double rank = std::ceil(std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size()));
    const std::size_t index = rank < 1.0 ? 0U : static_cast<std::size_t>(rank) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

// Least squares of p50 on bytes. A negative slope (noise across sizes) gives a flat model at the mean; a non-positive
// intercept is anchored at the smallest size's p50 and the slope refitted through it, so the result is always valid.
inline LinkCost fit(std::span<const LinkSample> samples) {
    if (samples.empty()) throw std::invalid_argument("link cost: no samples");
    double mx = 0.0, my = 0.0, tail = 0.0;
    for (const auto& s : samples) {
        if (!std::isfinite(s.p50_us) || !std::isfinite(s.p90_us) || s.p50_us <= 0.0)
            throw std::invalid_argument("link cost: a sample is not a positive finite time");
        mx += static_cast<double>(s.bytes);
        my += s.p50_us;
        tail = std::max(tail, s.p90_us - s.p50_us);
    }
    const double n = static_cast<double>(samples.size());
    mx /= n;
    my /= n;
    double sxy = 0.0, sxx = 0.0;
    for (const auto& s : samples) {
        const double dx = static_cast<double>(s.bytes) - mx;
        sxy += dx * (s.p50_us - my);
        sxx += dx * dx;
    }
    LinkCost c;
    c.tail_us = tail;
    if (sxx <= 0.0 || sxy <= 0.0) {
        c.latency_us = my;
        return c;
    }
    c.us_per_byte = sxy / sxx;
    c.latency_us = my - c.us_per_byte * mx;
    if (c.latency_us <= 0.0) {
        const auto smallest = *std::min_element(samples.begin(), samples.end(),
                                                [](const LinkSample& a, const LinkSample& b) { return a.bytes < b.bytes; });
        c.latency_us = smallest.p50_us;
        double num = 0.0, den = 0.0;
        for (const auto& s : samples) {
            const double x = static_cast<double>(s.bytes);
            num += x * (s.p50_us - c.latency_us);
            den += x * x;
        }
        c.us_per_byte = den > 0.0 ? std::max(0.0, num / den) : 0.0;
    }
    return c;
}

// Times `rounds` calls of exchange(bytes) per size after `warmup` untimed calls; `pause` idles before each call (the compute
// gap of the regime the cost is for: 0 = back to back). Sizes keep their order.
template<class Exchange>
std::vector<LinkSample> measure(Exchange&& exchange, std::span<const std::uint64_t> sizes, unsigned warmup, unsigned rounds,
                                std::chrono::microseconds pause = std::chrono::microseconds{0}) {
    if (sizes.empty() || rounds == 0U) throw std::invalid_argument("link cost: measure needs sizes and rounds");
    std::vector<LinkSample> out;
    out.reserve(sizes.size());
    std::vector<double> us(rounds);
    for (const auto bytes : sizes) {
        for (unsigned i = 0; i < warmup; ++i) exchange(bytes);
        for (unsigned i = 0; i < rounds; ++i) {
            if (pause.count() > 0) std::this_thread::sleep_for(pause);
            const auto t0 = std::chrono::steady_clock::now();
            exchange(bytes);
            us[i] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        }
        // A clock that did not advance (sub-resolution exchange) still yields a positive sample.
        const double floor_us = 1e-3;
        out.push_back({bytes, std::max(floor_us, percentile(us, 0.5)), std::max(floor_us, percentile(us, 0.9))});
    }
    return out;
}

// Start-up sizes: one latency-bound, one mid, one rate-bound message (~10 ms of link time at 35 rounds).
inline constexpr std::array<std::uint64_t, 3> default_link_sizes{4096U, 65536U, 1048576U};

// The host-bytes form: symmetric Transport::exchange of `bytes` each way. Both ranks call it with the same arguments.
inline std::vector<LinkSample> measure_host(Transport& transport, std::span<const std::uint64_t> sizes, unsigned warmup = 5U,
                                            unsigned rounds = 30U, std::chrono::microseconds pause = std::chrono::microseconds{0}) {
    if (sizes.empty() || rounds == 0U) throw std::invalid_argument("link cost: measure needs sizes and rounds");
    const std::uint64_t most = *std::max_element(sizes.begin(), sizes.end());
    if (most > std::vector<std::byte>{}.max_size()) throw std::length_error("link cost: probe exceeds host buffer limit");
    std::vector<std::byte> out(static_cast<std::size_t>(most), std::byte{0x5a}), in(static_cast<std::size_t>(most));
    return measure([&](std::uint64_t bytes) {
        const auto n = static_cast<std::size_t>(bytes);
        transport.exchange(std::span<const std::byte>(out.data(), n), std::span<std::byte>(in.data(), n));
    }, sizes, warmup, rounds, pause);
}

// Field-wise max: commutative, so both ranks combining (mine, peer) and (peer, mine) hold identical bits.
inline LinkCost combine(const LinkCost& a, const LinkCost& b) {
    return {std::max(a.latency_us, b.latency_us), std::max(a.us_per_byte, b.us_per_byte), std::max(a.tail_us, b.tail_us)};
}

// One symmetric exchange of the two fits; every rank returns combine(own, peer).
inline LinkCost agree(Transport& transport, const LinkCost& mine) {
    std::array<double, 3> local{mine.latency_us, mine.us_per_byte, mine.tail_us}, remote{};
    std::array<std::byte, sizeof(local)> out{}, in{};
    std::memcpy(out.data(), local.data(), sizeof(local));
    transport.exchange(out, in);
    std::memcpy(remote.data(), in.data(), sizeof(remote));
    // Both ranks complete the exchange even when one fit is invalid.
    if (!mine.valid()) throw std::invalid_argument("link cost: agree() needs a valid local fit");
    const LinkCost peer{remote[0], remote[1], remote[2]};
    if (!peer.valid()) throw std::runtime_error("link cost: the peer sent an invalid fit");
    return combine(mine, peer);
}

// One measured scalar agreed by both ranks (e.g. a timed one-rank pass that decides a placement): one symmetric exchange,
// every rank returns max(own, peer), so both decide from identical bits. Refuses a non-finite or negative value on
// either side after the exchange completed on both ranks.
inline double agree_max(Transport& transport, double mine) {
    std::array<std::byte, sizeof(double)> out{}, in{};
    std::memcpy(out.data(), &mine, sizeof(double));
    transport.exchange(out, in);
    double peer = 0.0;
    std::memcpy(&peer, in.data(), sizeof(double));
    if (!std::isfinite(mine) || mine < 0.0) throw std::invalid_argument("agree_max: the local value is not finite and >= 0");
    if (!std::isfinite(peer) || peer < 0.0) throw std::runtime_error("agree_max: the peer sent a value that is not finite and >= 0");
    return std::max(mine, peer);
}

} // namespace strix::dist
