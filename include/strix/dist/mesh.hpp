#pragma once
// Host collectives over N ranks (model- and vendor-agnostic): ordered all-gather and ordered all-reduce built from
// pairwise symmetric exchanges, one per peer. Used for small host records (device facts, plan hashes, partial sums
// of host reductions); device-side tensor collectives stay in dist::Communicator.
// - Schedule: every rank exchanges with its peers in ascending rank order. Each rank then walks the pairs that touch it
//   in the global lexicographic order of pairs (min, max), so the globally smallest pending pair is the next exchange of
//   both of its ranks and blocking (rendezvous) exchanges cannot deadlock for any N.
// - Order: all_reduce_ordered folds ((p0 + p1) + p2) + ... in rank order on every rank, so every rank holds the same
//   bits; at N == 2 that is p0 + p1, the pair routes' fold.
// - N == 2: exactly one exchange of the own record with the peer (the pair wire, unchanged).
// Home: shared (host shared). Bound by: deploy facts exchange (strix/deploy/binding.hpp exchange_facts).
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/dist/transport.hpp"

namespace strix::dist {
// Symmetric exchange with one peer: sends `out`, receives the peer's record of the same size into `in`.
using PeerExchange = std::function<void(std::uint32_t peer, std::span<const std::byte> out, std::span<std::byte> in)>;

// `all` holds world records of mine.size() bytes in rank order; all[rank] receives a copy of `mine`.
inline void all_gather(std::uint32_t rank, std::uint32_t world, std::span<const std::byte> mine, std::span<std::byte> all,
                       const PeerExchange& exchange) {
    if (world == 0U || rank >= world) throw std::invalid_argument("mesh: rank " + std::to_string(rank) + " outside world " + std::to_string(world));
    const auto size = mine.size();
    if (all.size() != size * world) throw std::invalid_argument("mesh: gather buffer is not world x record bytes");
    if (size != 0U) std::memcpy(all.data() + size * rank, mine.data(), size);
    for (std::uint32_t peer = 0; peer < world; ++peer)
        if (peer != rank) exchange(peer, mine, all.subspan(size * peer, size));
}

// Every rank ends with the same values: the rank-ordered left fold of the world partials.
inline void all_reduce_ordered(std::uint32_t rank, std::uint32_t world, std::span<float> values, const PeerExchange& exchange) {
    std::vector<float> partials(values.size() * world);
    all_gather(rank, world, std::as_bytes(values), std::as_writable_bytes(std::span<float>(partials)), exchange);
    for (std::size_t i = 0; i < values.size(); ++i) {
        float sum = partials[i];
        for (std::uint32_t r = 1; r < world; ++r) sum += partials[r * values.size() + i];
        values[i] = sum;
    }
}

// PeerExchange over one Transport per peer (peers[r]: the transport joining this rank to rank r; the own slot is
// ignored). A peer without a transport is refused by name: a topology whose collectives need a relay is not a mesh.
inline PeerExchange mesh_exchange(std::vector<Transport*> peers) {
    return [peers = std::move(peers)](std::uint32_t peer, std::span<const std::byte> out, std::span<std::byte> in) {
        if (peer >= peers.size() || peers[peer] == nullptr)
            throw std::invalid_argument("mesh: no direct link to rank " + std::to_string(peer) + " (collectives need a full mesh)");
        peers[peer]->exchange(out, in);
    };
}
} // namespace strix::dist
