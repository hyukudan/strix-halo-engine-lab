#pragma once
// Host-only wire construction, shared by native progress and the bytewise gate.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace strix::xport {
// A payload CRC may be checked after the payload is published to its consumer (deferred: off the critical path) only
// when the transport already verified every delivered byte (Transport::verifies_delivery). Otherwise the receiver
// checks it before the publication, so no consumer ever reads a payload known to be damaged.
[[nodiscard]] constexpr bool payload_crc_after_publish(bool crc, bool deferred_requested, bool transport_verifies) noexcept {
    return crc && deferred_requested && transport_verifies;
}

// Transport receives exactly payload followed by the engine's trailer, with no added header.
template<class Transport, class Trailer>
void exchange_frame(Transport& transport, std::byte* tx, std::byte* rx,
                    std::size_t bytes, const Trailer& trailer) {
    std::memcpy(tx + bytes, &trailer, sizeof(trailer));
    transport.exchange(std::span<const std::byte>(tx, bytes + sizeof(Trailer)),
                       std::span<std::byte>(rx, bytes + sizeof(Trailer)));
}

// Direct coherent payloads have no trailer-sized padding. Exchange the trailer
// separately: never write past the payload into the next owner's live rows.
template<class Transport, class Trailer>
void exchange_frame_direct(Transport& transport, const std::byte* tx, std::byte* rx,
                           std::size_t bytes, const Trailer& trailer, Trailer& peer) {
    transport.exchange(std::span<const std::byte>(tx, bytes), std::span<std::byte>(rx, bytes));
    transport.exchange(std::as_bytes(std::span(&trailer, 1)), std::as_writable_bytes(std::span(&peer, 1)));
}

template<class Traits>
[[nodiscard]] typename Traits::Trailer make_trailer(
    const typename Traits::Tag& tag, std::uint64_t epoch, std::uint64_t plan,
    std::uint64_t sequence, std::size_t bytes, std::uint32_t rank,
    bool deferred, std::uint32_t crc, bool previous, std::uint32_t previous_crc) {
    typename Traits::Trailer t;
    t.epoch = epoch;
    t.plan = plan;
    t.sequence = sequence;
    t.step = tag.step;
    t.bytes = static_cast<std::uint32_t>(bytes);
    t.kind = static_cast<std::uint16_t>(tag.kind);
    t.layer = tag.layer;
    t.rank = rank;
    if (deferred) t.flags |= Traits::kDeferred;
    else t.crc32c = crc;
    if (previous) {
        t.flags |= Traits::kPrevious;
        t.prev_crc32c = previous_crc;
    }
    return t;
}
} // namespace strix::xport
