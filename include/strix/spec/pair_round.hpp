#pragma once

#include "strix/pair/lockstep.hpp"
#include "strix/spec/block_drafter.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Speculative decode over a tensor-parallel pair: one drafter, one decision per round, the verify on both ranks.
// Rank 0 (the leader) drafts and runs the shared BlockSpeculator; its VerifyTarget is wrapped by PairVerifyLeader, which
// sends every verify's rows to rank 1 before running them. Rank 1 (the follower) has no drafter: pair_follow() replays the
// leader's target calls (commit of the previous pass, then the same verify rows) on its own target, so both ranks run the
// identical sequence of TP passes (every gather of a pass is a symmetric swap and must be matched by the peer).
// One fixed-size message per round, exchanged by a symmetric swap (each rank sends its message, receives the peer's):
//   leader:   {lockstep header, commit = k of the previous pass (kNoCommit: none), rows = R (0: end), tokens[R], position}
//   follower: {lockstep header, rest zero}
// The lockstep header (strix/pair/lockstep.hpp) catches a dropped / duplicated round and two ranks with the same role.
// The follower needs no greedy result from the leader: it commits what the leader committed, and the end message carries
// the leader's target position, which the follower checks against its own.
// Host only, backend-neutral. Home: shared (spec layer); the swap binding (device gather, socket) is the engine's.
namespace strix::spec::pair_round {

inline constexpr std::uint32_t kMaxRows = 8;
inline constexpr std::uint32_t kNoCommit = ~std::uint32_t{0};

struct Message {
    pair::LockstepHeader header;
    std::uint32_t commit = kNoCommit;  // k of the leader's previous pass, applied by the follower before this round's rows
    std::uint32_t rows = 0;            // R of this round's verify; 0 = end of the decode
    std::uint64_t position = 0;        // end message: the leader's target position after its last commit
    std::array<std::uint32_t, kMaxRows> tokens{};
};
inline constexpr std::size_t kMessageBytes = 72;  // fixed wire size (encode / decode) of the fields below

// Little-endian fixed layout: round u64 | ack u64 | rank u32 | commit u32 | rows u32 | pad u32 | position u64 | tokens u32 x 8.
inline std::array<std::uint8_t, kMessageBytes> encode(const Message& m) {
    std::array<std::uint8_t, kMessageBytes> b{};
    std::size_t at = 0;
    const auto put = [&](std::uint64_t v, std::size_t bytes) {
        for (std::size_t i = 0; i < bytes; ++i) b[at + i] = static_cast<std::uint8_t>((v >> (8U * i)) & 0xFFU);
        at += bytes;
    };
    put(m.header.round, 8);
    put(m.header.ack_round, 8);
    put(m.header.rank, 4);
    put(m.commit, 4);
    put(m.rows, 4);
    put(0, 4);
    put(m.position, 8);
    for (const auto t : m.tokens) put(t, 4);
    return b;
}
inline Message decode(std::span<const std::uint8_t> b) {
    if (b.size() != kMessageBytes) throw pair::ProtocolError("spec pair round: message of " + std::to_string(b.size()) + " bytes");
    std::size_t at = 0;
    const auto get = [&](std::size_t bytes) {
        std::uint64_t v = 0;
        for (std::size_t i = 0; i < bytes; ++i) v |= std::uint64_t{b[at + i]} << (8U * i);
        at += bytes;
        return v;
    };
    Message m;
    m.header.round = get(8);
    m.header.ack_round = get(8);
    m.header.rank = static_cast<std::uint32_t>(get(4));
    m.commit = static_cast<std::uint32_t>(get(4));
    m.rows = static_cast<std::uint32_t>(get(4));
    (void)get(4);
    m.position = get(8);
    for (auto& t : m.tokens) t = static_cast<std::uint32_t>(get(4));
    if (m.rows > kMaxRows) throw pair::ProtocolError("spec pair round: rows " + std::to_string(m.rows) + " outside 0.." + std::to_string(kMaxRows));
    return m;
}

// The symmetric swap of one message: send `mine` (kMessageBytes), receive the peer's into `peer` (kMessageBytes).
using Exchange = std::function<void(const std::uint8_t* mine, std::uint8_t* peer)>;

namespace detail {
inline Message swap(pair::Lockstep& ls, Exchange& ex, Message mine) {
    mine.header = ls.header();
    const auto out = encode(mine);
    std::array<std::uint8_t, kMessageBytes> in{};
    ex(out.data(), in.data());
    const Message peer = decode(in);
    ls.accept(peer.header);
    return peer;
}
}  // namespace detail

// Rank 0: the wrapped target runs every verify / commit; each verify first sends its rows (with the previous commit).
// finish() sends the end message; call it once after the speculative decode (also after an exception is not needed:
// the follower then fails on its next swap with the transport's error).
class PairVerifyLeader final : public VerifyTarget {
public:
    using PositionFn = std::function<std::uint64_t()>;
    PairVerifyLeader(VerifyTarget& inner, Exchange exchange, PositionFn position)
        : inner_(inner), ex_(std::move(exchange)), position_(std::move(position)), ls_(0U) {}
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        if (tokens.empty() || tokens.size() > kMaxRows) throw std::invalid_argument("spec pair round: verify rows outside 1..8");
        Message m;
        m.commit = pending_;
        m.rows = static_cast<std::uint32_t>(tokens.size());
        std::copy(tokens.begin(), tokens.end(), m.tokens.begin());
        (void)detail::swap(ls_, ex_, m);
        pending_ = kNoCommit;
        inner_.verify(tokens, greedy);
    }
    void commit(std::uint32_t k) override {
        inner_.commit(k);
        pending_ = k;
    }
    void finish() {
        if (finished_) return;
        Message m;
        m.commit = pending_;
        m.rows = 0;
        m.position = position_();
        (void)detail::swap(ls_, ex_, m);
        pending_ = kNoCommit;
        finished_ = true;
    }
    [[nodiscard]] std::uint64_t rounds() const noexcept { return ls_.round(); }

private:
    VerifyTarget& inner_;
    Exchange ex_;
    PositionFn position_;
    pair::Lockstep ls_;
    std::uint32_t pending_ = kNoCommit;
    bool finished_ = false;
};

struct FollowResult {
    std::uint64_t rounds = 0;   // messages exchanged, the end message included
    std::uint64_t passes = 0;   // verify passes run
    std::uint64_t rows = 0;     // verify rows run
    std::uint64_t committed = 0;  // rows kept (sum of the commits)
};

// Rank 1: replays the leader's rounds on `target` until the end message; `position` = this rank's target position after
// the last commit, checked against the leader's.
inline FollowResult pair_follow(VerifyTarget& target, Exchange exchange, const std::function<std::uint64_t()>& position) {
    pair::Lockstep ls(1U);
    FollowResult res;
    std::vector<std::uint32_t> greedy;
    for (;;) {
        const Message m = detail::swap(ls, exchange, Message{});
        ++res.rounds;
        if (m.commit != kNoCommit) {
            target.commit(m.commit);
            res.committed += m.commit;
        }
        if (m.rows == 0U) {
            const auto mine = position();
            if (mine != m.position)
                throw pair::ProtocolError("spec pair round: follower position " + std::to_string(mine) + " != leader " +
                                          std::to_string(m.position));
            return res;
        }
        target.verify(std::span<const std::uint32_t>(m.tokens.data(), m.rows), greedy);
        ++res.passes;
        res.rows += m.rows;
    }
}

}  // namespace strix::spec::pair_round
