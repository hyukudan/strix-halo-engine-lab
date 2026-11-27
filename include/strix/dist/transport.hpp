#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace strix::dist {

// Moves whole messages (payload + trailer) between the two ranks, in order. Payloads are host memory.
// Backends: an in-process pair (two ranks in one process, the emulated two-node setup), TCP (bring-up and
// debugging) and the host path of USB4 streams (striped over the streams of a cable).
class Transport {
public:
    virtual ~Transport() = default;
    virtual void send(std::span<const std::byte> message) = 0;
    // Receives exactly message.size() bytes (the size is known from shared data).
    virtual void receive(std::span<std::byte> message) = 0;
    // Sends `out` and receives `in` at the same time (both ranks call it with their own message); the
    // backends do it with persistent progress threads, never one thread per call.
    virtual void exchange(std::span<const std::byte> out, std::span<std::byte> in) = 0;
    // The same exchange; on_wire(context) runs exactly once on the calling thread, while the bytes move when the
    // backend can (USB4 host: once every write is handed to the streams), else after the exchange. It may read `out`.
    virtual void exchange_overlapped(std::span<const std::byte> out, std::span<std::byte> in, void (*on_wire)(void*), void* context) {
        exchange(out, in);
        on_wire(context);
    }
    [[nodiscard]] virtual std::string name() const = 0;
    // Whether abort (including with an empty reason) ends pending peer I/O without destroying the transport.
    [[nodiscard]] virtual bool supports_peer_abort() const noexcept { return false; }
    // Stops the transport for good and closes its channel, so the peer's pending reads end at once (the peer sees
    // the close) instead of at its own deadline. Called by the communicator when a local wait expires.
    virtual void abort(const std::string& /*why*/) {}
    // Progress of the transfer in flight, for failure reports (per stream: bytes done / expected, last read and
    // write, last poll events). Empty when the backend keeps none.
    [[nodiscard]] virtual std::string state() const { return {}; }
    // Whether every byte a receive / exchange returns has already been checked against what the peer sent (sequence +
    // checksum per unit, lost or damaged units resent), so a consumer may use it before any later check of its own.
    // False unless the backend proves it: a layer above must then verify a payload before it publishes it.
    [[nodiscard]] virtual bool verifies_delivery() const noexcept { return false; }
};

} // namespace strix::dist
