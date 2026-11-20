#pragma once
// strix::pair::relayout: the prompt state of a layer-stage prefill re-laid out for a replicated-state decode.
//
// After a pair prefill over a StageMap (strix/pair/stage_map.hpp, stage s on rank s % 2) layer l's state (its caches for
// the prompt positions) exists only on the rank that ran l's stage (StageMap::owns). A tensor-parallel decode over the
// same two ranks reads every layer's state whole on both ranks (e.g. MLA latent rows: the heads split AFTER the latent).
// This header moves each layer's state from its stage owner to the other rank, in place, on the pair link the prefill
// used, so one process runs prefill as layer stages and decode as TP with the KV where the decode reads it.
//
//   plan      per rank: the layers it sends (it owns them) and receives, with bytes from the engine's own count
//             (bytes(layer) for the positions held; both ranks must compute the same numbers).
//   header    both ranks swap {magic, version, identity, layers, map digest, send / receive bytes} and then an ok word:
//             a disagreement (another request, map, model, position count or cache layout) makes BOTH throw before
//             any payload moves.
//   payload   rank 0 sends its layers in ascending order, then rank 1 its layers (one direction at a time: no thread,
//             no interleaving on the ordered link); each layer is a {layer, bytes} frame + the engine's bytes, checked
//             against the receiver's plan. The receiver's load() writes the bytes where its decode reads them.
//   done      a final word both ways: neither rank starts the decode before the peer finished loading.
// Bits: the imported bytes ARE the owner's bytes, so the decode state equals the one a single rank holding every layer
// would have computed with the same windows (the stage pipeline's invariant). Backend neutral: save / load are the
// engine's device copies; the link is the ordered pair transport (host bytes).
// Home: shared (host shared: pair placement). Bound by: Hy4 pair (apps/hy4_rows_main.cpp stage prime). Any pair engine
// with a layer-stage prefill and a replicated-state TP decode binds it the same way (save / load per layer).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/pair/stage_map.hpp"

namespace strix::pair::relayout {

struct Plan {
    std::vector<std::uint32_t> send, receive;  // ascending layers this rank owns (sends) / reads from the peer
    std::vector<std::uint64_t> send_layer_bytes, receive_layer_bytes;
    std::uint64_t send_bytes = 0, receive_bytes = 0;
    // One direction at a time over a link of `gbps` GB/s (both directions' bytes cross it in turn).
    [[nodiscard]] double predicted_ms(double gbps) const noexcept {
        return gbps > 0 ? static_cast<double>(send_bytes + receive_bytes) / (gbps * 1e6) : 0.0;
    }
};

// bytes(layer) -> std::uint64_t: the engine's state bytes of `layer` (0: nothing to move, e.g. a stateless layer).
template <class Bytes>
[[nodiscard]] Plan plan(const StageMap& map, unsigned rank, Bytes&& bytes) {
    if (rank > 1U) throw std::invalid_argument("pair relayout: rank 0 or 1");
    if (map.stages() < 2U) throw std::invalid_argument("pair relayout: a stage map of two ranks");
    Plan p;
    for (std::uint32_t l = 0; l < map.layers(); ++l) {
        const std::uint64_t b = bytes(l);
        if (b == 0U) continue;
        if (map.owns(rank, l)) {
            p.send.push_back(l);
            p.send_layer_bytes.push_back(b);
            p.send_bytes += b;
        } else {
            p.receive.push_back(l);
            p.receive_layer_bytes.push_back(b);
            p.receive_bytes += b;
        }
    }
    return p;
}

// The prime route of a TP pair: layer stages + this re-layout, or every rank primes the whole prompt one-node. With W-row
// windows the stage pipeline over 2v stages finishes in about (n + 2v - 1) / (2v) window times + hops for n windows, the
// one-node prime in n window times: one full window gives no overlap and loses the hops; from two full windows on the
// stages win while a window's work is large against a hop. Measured (Hy4 pair, gfx1151 x 2 over USB4, same binary, bits
// equal in every arm): 8K rows 2048 (4 windows) one-node 12.35 s / stages 7.49 s; 2K rows 256 (8 full windows) 5.53 /
// 3.67 s; 2K rows 2048 (1 full window + 24 rows) 3.61 / 4.64 s (stages lose: no overlap, hops exposed). Rows below
// kStagePrimeMinRows are not measured (per-hop latency against tiny windows) -> one-node.
inline constexpr std::uint32_t kStagePrimeMinRows = 256U;
[[nodiscard]] constexpr bool stage_prime_wins(std::uint64_t full_windows, std::uint32_t rows) noexcept {
    return full_windows >= 2U && rows >= kStagePrimeMinRows;
}

struct Link {
    std::function<void(std::span<const std::byte>)> send;
    std::function<void(std::span<std::byte>)> receive;
    std::function<void(const std::string&)> abort;  // optional: unblocks the peer when this rank fails mid-payload
};
struct Io {
    std::function<void(std::uint32_t layer, std::span<std::byte> out)> save;        // this rank's state of `layer`
    std::function<void(std::uint32_t layer, std::span<const std::byte> in)> load;   // the peer's state of `layer`
};
struct Stats {
    std::uint64_t sent = 0, received = 0;
    std::uint32_t layers_sent = 0, layers_received = 0;
    double ms = 0;
};

inline constexpr std::uint64_t kMagic = 0x54554f59414c4552ULL;  // "RELAYOUT"
inline constexpr std::uint64_t kVersion = 1;

[[nodiscard]] inline std::uint64_t map_digest(const StageMap& map) noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    for (const char ch : map.text()) h = (h ^ static_cast<unsigned char>(ch)) * 1099511628211ULL;
    return h;
}

namespace detail {
template <std::size_t N> void swap_words(unsigned rank, const Link& link, const std::array<std::uint64_t, N>& mine,
                                         std::array<std::uint64_t, N>& peer) {
    const auto out = std::as_bytes(std::span(mine));
    const auto in = std::as_writable_bytes(std::span(peer));
    if (rank == 0U) {
        link.send(out);
        link.receive(in);
    } else {
        link.receive(in);
        link.send(out);
    }
}
}  // namespace detail

// Both ranks call it at the same point with the same map / identity and their own plan (plan() above).
inline Stats exchange(const StageMap& map, unsigned rank, std::uint64_t identity, const Plan& p, const Link& link, const Io& io) {
    if (!link.send || !link.receive || !io.save || !io.load) throw std::invalid_argument("pair relayout: link and io required");
    const auto t0 = std::chrono::steady_clock::now();
    const std::array<std::uint64_t, 7> mine{kMagic, kVersion, identity, map.layers(), map_digest(map), p.send_bytes, p.receive_bytes};
    std::array<std::uint64_t, 7> peer{};
    detail::swap_words(rank, link, mine, peer);
    std::string why;
    if (peer[0] != kMagic || peer[1] != kVersion) why = "peer is not a relayout v1 header";
    else if (peer[2] != identity) why = "request identity differs";
    else if (peer[3] != map.layers() || peer[4] != mine[4]) why = "stage map differs";
    else if (peer[5] != p.receive_bytes || peer[6] != p.send_bytes) why = "state bytes differ (positions / cache layout)";
    const std::array<std::uint64_t, 1> ok{why.empty() ? 1ULL : 0ULL};
    std::array<std::uint64_t, 1> peer_ok{};
    detail::swap_words(rank, link, ok, peer_ok);
    if (!why.empty()) throw std::runtime_error("pair relayout: " + why);
    if (peer_ok[0] != 1U) throw std::runtime_error("pair relayout: the peer refused the header");

    std::uint64_t biggest = 0;
    for (const auto b : p.send_layer_bytes) biggest = std::max(biggest, b);
    for (const auto b : p.receive_layer_bytes) biggest = std::max(biggest, b);
    std::vector<std::byte> buf(static_cast<std::size_t>(biggest));
    Stats st;
    try {
        for (unsigned dir = 0; dir < 2U; ++dir) {
            if (dir == rank) {  // this rank's turn to send its layers
                for (std::size_t i = 0; i < p.send.size(); ++i) {
                    const auto n = static_cast<std::size_t>(p.send_layer_bytes[i]);
                    const std::array<std::uint64_t, 2> frame{p.send[i], n};
                    io.save(p.send[i], std::span(buf.data(), n));
                    link.send(std::as_bytes(std::span(frame)));
                    link.send(std::span<const std::byte>(buf.data(), n));
                    st.sent += n;
                    ++st.layers_sent;
                }
            } else {
                for (std::size_t i = 0; i < p.receive.size(); ++i) {
                    std::array<std::uint64_t, 2> frame{};
                    link.receive(std::as_writable_bytes(std::span(frame)));
                    const auto n = static_cast<std::size_t>(p.receive_layer_bytes[i]);
                    if (frame[0] != p.receive[i] || frame[1] != n)
                        throw std::runtime_error("pair relayout: frame layer " + std::to_string(frame[0]) + " bytes " +
                                                 std::to_string(frame[1]) + ", expected layer " + std::to_string(p.receive[i]) +
                                                 " bytes " + std::to_string(n));
                    link.receive(std::span(buf.data(), n));
                    io.load(p.receive[i], std::span<const std::byte>(buf.data(), n));
                    st.received += n;
                    ++st.layers_received;
                }
            }
        }
    } catch (const std::exception& e) {
        if (link.abort) link.abort(std::string("pair relayout: ") + e.what());
        throw;
    }
    const std::array<std::uint64_t, 1> done{st.received};
    std::array<std::uint64_t, 1> peer_done{};
    detail::swap_words(rank, link, done, peer_done);
    if (peer_done[0] != p.send_bytes) throw std::runtime_error("pair relayout: the peer loaded a different byte count");
    st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

}  // namespace strix::pair::relayout
