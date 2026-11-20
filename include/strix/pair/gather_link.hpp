#pragma once
// One all-gather of an exact tensor-parallel pass over two ranks: a symmetric swap of the slots. slot[rank] holds this
// rank's share, produced by work already enqueued on the stream; slot[1 - rank] receives the peer's share and is valid
// for work enqueued on the same stream after the call. Routes (one per run, both ranks the same):
//   staged        - a host-staged exchange (dist::Communicator::exchange): the Communicator enqueues its device-to-host
//                   copy on the stream after the producer and its progress thread waits for that copy's event, so the
//                   caller does not wait for the stream first (no host stream wait before the exchange);
//   device_direct - the Communicator's device exchange (stream-ordered, no host wait) for the sizes its size table sends
//                   to the rings; the sizes the table keeps on the host path (host_max_bytes, e.g. <= 64 KiB with host
//                   streams) take the staged exchange, host wait included (effective_route);
//   channel       - the shared device-waited channel (xport::ChannelCommunicator): a copy kernel publishes the share,
//                   the progress thread runs the wire, the stream waits on the RX flag; no host wait per gather.
// Counters: calls; link_ms = host time inside the exchange call (staged: the producer's tail still in flight, the copy
// and the whole wire; device routes: enqueue time only). Pass wall - link = host gaps (launches, joins, bookkeeping).
// Backend-neutral host code: Ops supplies Stream, stream(void*) and tag(call); Legacy / Fast supply
// exchange(Stream, const void* local, void* remote, std::size_t bytes, const Tag&). The HIP binding is
// src/xport/tp_gather_hip.hpp; host test tests/pair_tp_gather_tests.cpp.
// Home: shared (host shared, pair facility). Bound by: qwen35 dense HIP exact tensor-parallel decode.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::pair {

enum class GatherRoute : std::uint8_t { staged = 0, device_direct = 1, channel = 2 };

[[nodiscard]] constexpr const char* gather_route_name(GatherRoute r) noexcept {
    return r == GatherRoute::staged ? "staged (host-staged exchange)"
         : r == GatherRoute::device_direct ? "device direct (stream-ordered)"
                                           : "channel (device-waited, no host wait)";
}

// Largest message a device-direct exchange still carries on its host path: the leading entries of its size table
// (Routes: entries with up_to, direct, host_percent, ascending up_to, e.g. dist::Communicator::routes()) up to the first
// pure device entry. 0 = the device exchange from the first byte.
template<class Routes>
[[nodiscard]] constexpr std::uint64_t host_prefix_bytes(const Routes& routes) noexcept {
    std::uint64_t most = 0;
    for (const auto& r : routes) {
        if (r.direct && r.host_percent == 0U) break;
        most = static_cast<std::uint64_t>(r.up_to);
    }
    return most;
}

// The route a gather of `bytes` takes: the device-direct route sends up to host_max_bytes on the host path, which is
// the staged exchange (the call returns after the wire); the other routes do not depend on the size.
[[nodiscard]] constexpr GatherRoute effective_route(GatherRoute r, std::uint64_t bytes, std::uint64_t host_max_bytes) noexcept {
    return r == GatherRoute::device_direct && bytes <= host_max_bytes ? GatherRoute::staged : r;
}

// Receipt name of a route with its size split, e.g. "device direct (stream-ordered) above 65536 B; staged (host-staged
// exchange) up to 65536 B".
[[nodiscard]] inline std::string gather_route_label(GatherRoute r, std::uint64_t host_max_bytes) {
    if (r != GatherRoute::device_direct || host_max_bytes == 0U) return gather_route_name(r);
    const auto n = std::to_string(host_max_bytes);
    return std::string(gather_route_name(r)) + " above " + n + " B; " + gather_route_name(GatherRoute::staged) + " up to " + n + " B";
}

// Route policy (backend-neutral; bindings receive the route, they do not read switches): the device-waited channel when
// requested, else the device-direct exchange when the transport offers it, else the host-staged exchange.
[[nodiscard]] constexpr GatherRoute select_route(bool direct_available, bool channel_requested) noexcept {
    return channel_requested ? GatherRoute::channel : direct_available ? GatherRoute::device_direct : GatherRoute::staged;
}

// Host wait per gather: only the staged route returns to the host inside the exchange (the next work is enqueued after
// it), so only its join segments carry a host wake-up + relaunch gap; the device routes enqueue and return (gap 0 by
// contract; their `tp link:` counters print the enqueue time only). Planners price the gap of a staged join only when
// it was measured (strix/pair/planner.hpp).
[[nodiscard]] constexpr bool route_waits_on_host(GatherRoute r) noexcept { return r == GatherRoute::staged; }

// Stand-in for a route a binding does not offer (the constructor refuses that route, so it is never called).
struct NoExchange {
    template<class... A> void exchange(A&&...) { throw std::logic_error("gather link: route not bound"); }
    void drain() { throw std::logic_error("gather link: route not bound"); }
};

template<class Ops, class Legacy, class Fast>
class GatherLink {
public:
    using Stream = typename Ops::Stream;

    GatherLink() = default;
    // host_max_bytes: the binding's size split of the device-direct route (host_prefix_bytes of its exchange), else 0.
    GatherLink(std::uint32_t rank, GatherRoute route, Legacy* legacy, Fast* fast, std::uint64_t host_max_bytes = 0)
        : rank_(rank), route_(route), legacy_(legacy), fast_(fast),
          host_max_(route == GatherRoute::device_direct ? host_max_bytes : 0U) {
        if (rank > 1U) throw std::invalid_argument("gather link: two ranks (rank 0 or 1)");
        if (route == GatherRoute::channel ? fast == nullptr : legacy == nullptr)
            throw std::invalid_argument("gather link: the route's exchange is missing");
    }

    void gather(Stream stream, void* slots, std::size_t slot_bytes, std::size_t bytes) {
        if (bytes > slot_bytes) throw std::invalid_argument("gather link: message larger than its slot");
        auto* b = static_cast<std::uint8_t*>(slots);
        const void* mine = b + std::size_t(rank_) * slot_bytes;
        void* peer = b + std::size_t(1U - rank_) * slot_bytes;
        const auto tag = Ops::tag(calls_);
        const auto t = Clock::now();
        if (route_ == GatherRoute::channel) fast_->exchange(stream, mine, peer, bytes, tag);
        else legacy_->exchange(stream, mine, peer, bytes, tag);
        link_ms_ += ms(t, Clock::now());
        ++calls_;
    }
    // Engine binding (a plain function pointer + context, e.g. qwen35 HipEngine::TpGather).
    static void fn(void* ctx, void* stream, void* slots, std::size_t slot_bytes, std::size_t bytes) {
        static_cast<GatherLink*>(ctx)->gather(Ops::stream(stream), slots, slot_bytes, bytes);
    }
    // End of a run, before its results are reported: the channel route checks the integrity of its last messages
    // (deferred CRC: the check of message s travels with s + 1, the last one with the drain's seal), so a corrupt final
    // gather fails here instead of in a destructor that cannot report it. Both ranks call it (the seal is symmetric).
    // The staged and device-direct routes check every exchange inline: nothing to finish.
    void finish() {
        if (route_ == GatherRoute::channel) fast_->drain();
    }
    // Restart the time counters (e.g. after the start-up link measurement); the call sequence (wire tags) continues.
    void reset_counters() noexcept { link_ms_ = 0.0; counted_from_ = calls_; }

    [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
    [[nodiscard]] GatherRoute route() const noexcept { return route_; }
    // The route a gather of `bytes` takes (planners price the step's gathers with it) and the receipt label.
    [[nodiscard]] GatherRoute route_for(std::uint64_t bytes) const noexcept { return effective_route(route_, bytes, host_max_); }
    [[nodiscard]] std::uint64_t host_max_bytes() const noexcept { return host_max_; }
    [[nodiscard]] std::string route_label() const { return gather_route_label(route_, host_max_); }
    [[nodiscard]] std::uint64_t calls() const noexcept { return calls_ - counted_from_; }
    [[nodiscard]] double link_ms() const noexcept { return link_ms_; }

private:
    using Clock = std::chrono::steady_clock;
    static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
    std::uint32_t rank_ = 0;
    GatherRoute route_ = GatherRoute::staged;
    Legacy* legacy_ = nullptr;
    Fast* fast_ = nullptr;
    std::uint64_t host_max_ = 0;
    std::uint64_t calls_ = 0, counted_from_ = 0;
    double link_ms_ = 0.0;
};

}  // namespace strix::pair
