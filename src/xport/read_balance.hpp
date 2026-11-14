#pragma once
// The sync point of a balanced read (src/core/balanced_read.hpp) over a symmetric exchange: both ranks call
// read_balance_sync at the same place of their exchange order with their own progress; each gets the block of its own
// group it gives away (the peer reads and ships those) and the block of the peer's group it takes. At most one of the
// two is set. The engine binds the exchange (a few words) and ships the taken blocks with its ordinary row exchange.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "../core/balanced_read.hpp"

namespace strix::xport {

inline constexpr std::size_t read_balance_words = 6;
using ReadBalanceMessage = std::array<std::uint64_t, read_balance_words>;

struct ReadBalanceShare {
    std::size_t give_from = 0;  // own blocks from here on are read by the peer (own block count: none)
    std::size_t take_from = 0;  // the peer's blocks from here on are read here (peer block count: none)
    std::uint64_t units = 0;    // units shipped (either direction)
    core::ReadProgress peer;    // what the peer reported
};

// exchange(send, receive): both messages cross, as in every symmetric exchange of the link.
template <class Exchange>
ReadBalanceShare read_balance_sync(const core::ReadProgress& mine, std::span<const std::uint32_t> own_units,
                                   std::span<const std::uint32_t> peer_units, std::uint64_t min_gain_us, Exchange&& exchange) {
    constexpr std::uint64_t magic = 0x52424c31'73796e63ULL;
    const ReadBalanceMessage send{magic, own_units.size(), mine.open, mine.free, mine.free_from, mine.rate};
    ReadBalanceMessage got{};
    exchange(send, got);
    if (got[0] != magic || got[1] != peer_units.size()) throw std::runtime_error("read balance: the ranks are out of step");
    const core::ReadProgress peer{got[2], got[3], got[4], got[5]};
    ReadBalanceShare share;
    share.peer = peer;
    share.give_from = core::balanced_tail(peer, mine, own_units, min_gain_us);
    share.take_from = core::balanced_tail(mine, peer, peer_units, min_gain_us);
    for (std::size_t b = share.give_from; b < own_units.size(); ++b) share.units += own_units[b];
    for (std::size_t b = share.take_from; b < peer_units.size(); ++b) share.units += peer_units[b];
    return share;
}

}  // namespace strix::xport
