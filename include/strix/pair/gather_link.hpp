#pragma once
// One all-gather of an exact tensor-parallel pass over two ranks: a symmetric swap of the slots. slot[rank] holds this
// rank's share, produced by work already enqueued on the stream; slot[1 - rank] receives the peer's share and is valid
// for work enqueued on the same stream after the call. Routes (one per run, both ranks the same):
//   staged        - a host-staged exchange (dist::Communicator::exchange): the Communicator enqueues its device-to-host
//                   copy on the stream after the producer and its progress thread waits for that copy's event, so the
//                   caller does not wait for the stream first (no host stream wait before the exchange);
//   device_direct - the Communicator's device exchange (stream-ordered, no host wait);
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

namespace strix::pair {

enum class GatherRoute : std::uint8_t { staged = 0, device_direct = 1, channel = 2 };

[[nodiscard]] constexpr const char* gather_route_name(GatherRoute r) noexcept {
    return r == GatherRoute::staged ? "staged (host-staged exchange)"
         : r == GatherRoute::device_direct ? "device direct (stream-ordered)"
                                           : "channel (device-waited, no host wait)";
}

// Stand-in for a route a binding does not offer (the constructor refuses that route, so it is never called).
struct NoExchange {
    template<class... A> void exchange(A&&...) { throw std::logic_error("gather link: route not bound"); }
};

template<class Ops, class Legacy, class Fast>
class GatherLink {
public:
    using Stream = typename Ops::Stream;

    GatherLink() = default;
    GatherLink(std::uint32_t rank, GatherRoute route, Legacy* legacy, Fast* fast)
        : rank_(rank), route_(route), legacy_(legacy), fast_(fast) {
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
    // Restart the time counters (e.g. after the start-up link measurement); the call sequence (wire tags) continues.
    void reset_counters() noexcept { link_ms_ = 0.0; counted_from_ = calls_; }

    [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
    [[nodiscard]] GatherRoute route() const noexcept { return route_; }
    [[nodiscard]] std::uint64_t calls() const noexcept { return calls_ - counted_from_; }
    [[nodiscard]] double link_ms() const noexcept { return link_ms_; }

private:
    using Clock = std::chrono::steady_clock;
    static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
    std::uint32_t rank_ = 0;
    GatherRoute route_ = GatherRoute::staged;
    Legacy* legacy_ = nullptr;
    Fast* fast_ = nullptr;
    std::uint64_t calls_ = 0, counted_from_ = 0;
    double link_ms_ = 0.0;
};

}  // namespace strix::pair
