#pragma once

// deepseek41 expert server (strix/ds41-expert-rpc-v1, ds41_expert_rpc.hpp): the session loop of the endpoint that a
// parts-host connects to, over a backend that computes routed experts (HIP device, or the CPU reference for tests).
//
// Two layouts of a two-node server:
//   - relay: rank 0 listens; every request is forwarded to rank 1 over a PeerLink (the USB4 transport between the
//     nodes); rank 0 computes the items of the low half (its experts [0, split)), rank 1 those of the high half, and
//     rank 0 streams the answer: header and its own items as soon as they are ready, then rank 1's items. The parts-host
//     sees one endpoint with every expert.
//   - direct: every rank listens on its own address and serves only its experts; the parts-host sends each rank the
//     rows and slots it needs (no hop between the nodes).
// Answer items are in (half, row, ascending id) order, so in relay mode the low half's items (rank 0) come first.

#include "strix/ds41_expert_rpc.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::ds41::xrpc {

class ExpertBackend {
public:
    virtual ~ExpertBackend() = default;
    [[nodiscard]] virtual Residency residency() const = 0;
    [[nodiscard]] virtual std::string engine() const = 0;            // e.g. "hip-gfx1151", "host-reference"
    [[nodiscard]] virtual std::string numeric_identity() const = 0;  // "ds41-exact-v1": raw vectors = the host reference
    // Computes plan.items[first .. first + count) of `req` into out (count x hidden floats). compute_ns: device time.
    virtual void compute(const ExpertsRequest& req, const ResponsePlan& plan, std::size_t first, std::size_t count, float* out,
                         std::uint64_t& compute_ns) = 0;
    // Optional: a host buffer the backend can write answers into without a copy (pinned memory).
    [[nodiscard]] virtual float* answer_buffer(std::size_t floats) {
        if (buffer_.size() < floats) buffer_.resize(floats);
        return buffer_.data();
    }

private:
    std::vector<float> buffer_;
};

// Whole messages between rank 0 and rank 1 of a relay server; sizes known on both sides.
class PeerLink {
public:
    virtual ~PeerLink() = default;
    virtual void send(std::span<const std::uint8_t> bytes) = 0;
    virtual void receive(std::span<std::uint8_t> bytes) = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};
// A PeerLink over a TCP connection of this protocol (tests, or nodes without USB4).
[[nodiscard]] std::unique_ptr<PeerLink> tcp_peer_listen(const std::string& host_port, std::chrono::milliseconds wait);
[[nodiscard]] std::unique_ptr<PeerLink> tcp_peer_connect(const std::string& host_port, std::chrono::milliseconds wait);

struct ServerOptions {
    std::string listen = "0.0.0.0:0";
    std::array<std::uint8_t, 32> token{};
    std::uint64_t model_identity = 0;  // 0: not checked
    Geometry geometry;
    std::uint32_t max_rows = 64;
    bool busy_poll = false;
    bool once = false;                      // return after the first session
    std::chrono::milliseconds accept_wait{0};
    std::chrono::milliseconds io_timeout{600000};
    bool verbose = false;
    // relay: the peer's residency (its items are the high half); nullptr peer = direct/one node
    Residency peer_residency;
    // called once listening: READY line etc.
    std::function<void(std::uint16_t port)> on_listen;
    std::atomic<bool>* stop = nullptr;
};

struct ServerStats {
    std::uint64_t requests = 0, items = 0, bytes_in = 0, bytes_out = 0;
    double compute_s = 0.0, relay_s = 0.0, server_s = 0.0;
};

// Rank 0 (or a direct-mode rank): listens and serves sessions until a CLOSE with once, or stop.
ServerStats serve(const ServerOptions& options, ExpertBackend& backend, PeerLink* peer);
// Rank 1 of a relay server: follows rank 0's forwarded requests until rank 0 quits.
void follow(const Geometry& geometry, const Residency& endpoint, ExpertBackend& backend, PeerLink& peer);

// The CPU reference backend: computes raw vectors with the declared products of ds41_formats (slow; tests and gates).
// Reads the [experts] tensors of the given layers from a GGUF (only experts in [first, end)).
[[nodiscard]] std::unique_ptr<ExpertBackend> make_host_backend(const std::string& gguf, std::uint64_t layer_mask, std::uint32_t expert_first,
                                                               std::uint32_t expert_end, Geometry* geometry_out);

// The routed-expert geometry of a GGUF (deepseek41 metadata and the [experts] tensors' dimensions).
[[nodiscard]] Geometry read_geometry(const std::string& gguf);

// Layer list "a,b,c" or "a-b" or "all" (layers) -> mask.
[[nodiscard]] std::uint64_t parse_layer_mask(const std::string& s, std::uint32_t layers);

}  // namespace strix::ds41::xrpc
