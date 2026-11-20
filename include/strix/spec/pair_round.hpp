#pragma once

#include "strix/pair/lockstep.hpp"
#include "strix/spec/block_drafter.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
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
// Drafter on both ranks (PairVerifyLeader::chain, pair_follow with a chain): a drafter whose every call runs tensor-
// parallel (a gather inside) needs the follower to make the same calls in the same order. The leader then also sends a
// draft round before each chain draft ({kind kDraft, rows = prefix length, tokens = prefix, draft_k = k}) and every
// message carries the leader's pending accept (emitted ids of the last pass, applied after the pending commit, the
// leader's order) and the FNV-1a hash of its last draft's ids; the follower runs the same draft and compares its own
// hash (a different id = the two drafters diverged: ProtocolError, fail closed). The drafts never change an output id:
// they only decide which rows are verified.
// The lockstep header (strix/pair/lockstep.hpp) catches a dropped / duplicated round and two ranks with the same role.
// The follower needs no greedy result from the leader: it commits what the leader committed, and the end message carries
// the leader's target position, which the follower checks against its own.
// Host only, backend-neutral. Home: shared (spec layer); the swap binding (device gather, socket) is the engine's.
namespace strix::spec::pair_round {

inline constexpr std::uint32_t kMaxRows = 8;
inline constexpr std::uint32_t kNoCommit = ~std::uint32_t{0};
inline constexpr std::uint32_t kNoAccept = ~std::uint32_t{0};
inline constexpr std::uint32_t kVerify = 0;  // Message::kind: a verify round (rows 0 = the end message)
inline constexpr std::uint32_t kDraft = 1;   // Message::kind: a chain draft round (rows = prefix length)

struct Message {
    pair::LockstepHeader header;
    std::uint32_t commit = kNoCommit;  // k of the leader's previous pass, applied by the follower before this round's rows
    std::uint32_t rows = 0;            // R of this round's verify (0 = end of the decode); a draft round: the prefix length
    std::uint32_t kind = kVerify;
    std::uint64_t position = 0;        // end message: the leader's target position after its last commit
    std::array<std::uint32_t, kMaxRows> tokens{};
    std::uint32_t draft_k = 0;            // a draft round: the chain drafts asked
    std::uint32_t emitted_n = kNoAccept;  // the leader's pending accept (after the pending commit); kNoAccept: none
    std::array<std::uint32_t, kMaxRows> emitted{};
    std::uint64_t draft_hash = 0;         // FNV-1a of the leader's last draft ids (0: no draft since the last message)
};
inline constexpr std::size_t kMessageBytes = 120;  // fixed wire size (encode / decode) of the fields below

// FNV-1a over draft ids (never 0: 0 marks "no draft").
[[nodiscard]] inline std::uint64_t draft_hash(std::span<const std::uint32_t> ids) noexcept {
    std::uint64_t h = 1469598103934665603ULL;
    for (const auto id : ids) h = (h ^ id) * 1099511628211ULL;
    return h == 0U ? 1U : h;
}

// Little-endian fixed layout: round u64 | ack u64 | rank u32 | commit u32 | rows u32 | kind u32 | position u64 |
// tokens u32 x 8 | draft_k u32 | emitted_n u32 | emitted u32 x 8 | draft_hash u64.
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
    put(m.kind, 4);
    put(m.position, 8);
    for (const auto t : m.tokens) put(t, 4);
    put(m.draft_k, 4);
    put(m.emitted_n, 4);
    for (const auto t : m.emitted) put(t, 4);
    put(m.draft_hash, 8);
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
    m.kind = static_cast<std::uint32_t>(get(4));
    m.position = get(8);
    for (auto& t : m.tokens) t = static_cast<std::uint32_t>(get(4));
    m.draft_k = static_cast<std::uint32_t>(get(4));
    m.emitted_n = static_cast<std::uint32_t>(get(4));
    for (auto& t : m.emitted) t = static_cast<std::uint32_t>(get(4));
    m.draft_hash = get(8);
    if (m.rows > kMaxRows) throw pair::ProtocolError("spec pair round: rows " + std::to_string(m.rows) + " outside 0.." + std::to_string(kMaxRows));
    if (m.kind != kVerify && m.kind != kDraft) throw pair::ProtocolError("spec pair round: message kind " + std::to_string(m.kind));
    if (m.emitted_n != kNoAccept && m.emitted_n > kMaxRows)
        throw pair::ProtocolError("spec pair round: accept of " + std::to_string(m.emitted_n) + " ids");
    if (m.kind == kDraft && (m.draft_k == 0U || m.draft_k > kMaxRows))
        throw pair::ProtocolError("spec pair round: draft of " + std::to_string(m.draft_k) + " ids");
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
// chain(inner): the drafter of this decode when it runs on both ranks (the follower is given its own chain): every
// draft first sends a draft round; accepts and draft hashes ride the next message. The wrapper offers no verify
// pre-launch (draft_prelaunch keeps the base "cannot" answer): the follower replays draft() only.
class PairVerifyLeader final : public VerifyTarget {
public:
    using PositionFn = std::function<std::uint64_t()>;
    PairVerifyLeader(VerifyTarget& inner, Exchange exchange, PositionFn position)
        : inner_(inner), ex_(std::move(exchange)), position_(std::move(position)), ls_(0U) {}
    PairVerifyLeader(const PairVerifyLeader&) = delete;
    PairVerifyLeader& operator=(const PairVerifyLeader&) = delete;
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        if (tokens.empty() || tokens.size() > kMaxRows) throw std::invalid_argument("spec pair round: verify rows outside 1..8");
        Message m = take_pending();
        m.rows = static_cast<std::uint32_t>(tokens.size());
        std::copy(tokens.begin(), tokens.end(), m.tokens.begin());
        (void)detail::swap(ls_, ex_, m);
        inner_.verify(tokens, greedy);
    }
    void commit(std::uint32_t k) override {
        inner_.commit(k);
        pending_ = k;
    }
    ChainDrafter& chain(ChainDrafter& inner) {
        chain_ = std::make_unique<Chain>(*this, inner);
        return *chain_;
    }
    void finish() {
        if (finished_) return;
        Message m = take_pending();
        m.rows = 0;
        m.position = position_();
        (void)detail::swap(ls_, ex_, m);
        finished_ = true;
    }
    [[nodiscard]] std::uint64_t rounds() const noexcept { return ls_.round(); }

private:
    class Chain final : public ChainDrafter {
    public:
        Chain(PairVerifyLeader& lead, ChainDrafter& inner) : lead_(lead), inner_(inner) {}
        void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
            if (prefix.size() > kMaxRows || k == 0U || k > kMaxRows)
                throw std::invalid_argument("spec pair round: draft prefix / k outside the message (8 ids)");
            Message m = lead_.take_pending();
            m.kind = kDraft;
            m.rows = static_cast<std::uint32_t>(prefix.size());
            std::copy(prefix.begin(), prefix.end(), m.tokens.begin());
            m.draft_k = k;
            (void)detail::swap(lead_.ls_, lead_.ex_, m);
            inner_.draft(prefix, k, out);
            lead_.hash_ = draft_hash(out);
        }
        void accept(std::span<const std::uint32_t> emitted) override {
            if (emitted.size() > kMaxRows) throw std::invalid_argument("spec pair round: accept of more than 8 ids");
            if (lead_.accept_n_ != kNoAccept) throw std::logic_error("spec pair round: two accepts without a round between them");
            inner_.accept(emitted);
            lead_.accept_n_ = static_cast<std::uint32_t>(emitted.size());
            std::copy(emitted.begin(), emitted.end(), lead_.accept_.begin());
        }

    private:
        PairVerifyLeader& lead_;
        ChainDrafter& inner_;
    };
    // the pending commit, accept and draft hash ride the next message (the follower applies them in this order)
    Message take_pending() {
        Message m;
        m.commit = pending_;
        m.emitted_n = accept_n_;
        m.emitted = accept_;
        m.draft_hash = hash_;
        pending_ = kNoCommit;
        accept_n_ = kNoAccept;
        hash_ = 0;
        return m;
    }
    VerifyTarget& inner_;
    Exchange ex_;
    PositionFn position_;
    pair::Lockstep ls_;
    std::uint32_t pending_ = kNoCommit;
    std::uint32_t accept_n_ = kNoAccept;
    std::array<std::uint32_t, kMaxRows> accept_{};
    std::uint64_t hash_ = 0;
    std::unique_ptr<Chain> chain_;
    bool finished_ = false;
};

struct FollowResult {
    std::uint64_t rounds = 0;   // messages exchanged, the end message included
    std::uint64_t passes = 0;   // verify passes run
    std::uint64_t rows = 0;     // verify rows run
    std::uint64_t committed = 0;  // rows kept (sum of the commits)
    std::uint64_t drafts = 0;   // draft rounds replayed on this rank's chain (0 without a chain)
};

// Rank 1: replays the leader's rounds on `target` (and its draft rounds on `chain`, when the leader drafts on both ranks)
// until the end message; `position` = this rank's target position after the last commit, checked against the leader's.
// The follower's chain must have seen the same prompt and anchor as the leader's (its own prefill observation).
inline FollowResult pair_follow(VerifyTarget& target, Exchange exchange, const std::function<std::uint64_t()>& position,
                                ChainDrafter* chain = nullptr) {
    pair::Lockstep ls(1U);
    FollowResult res;
    std::vector<std::uint32_t> greedy, drafted;
    std::uint64_t mine = 0;  // hash of this rank's last draft, until the leader's arrives
    const auto need_chain = [&] {
        if (chain == nullptr) throw pair::ProtocolError("spec pair round: the leader drafts on both ranks; this follower has no drafter");
    };
    for (;;) {
        const Message m = detail::swap(ls, exchange, Message{});
        ++res.rounds;
        if (m.commit != kNoCommit) {
            target.commit(m.commit);
            res.committed += m.commit;
        }
        if (m.emitted_n != kNoAccept) {
            need_chain();
            chain->accept(std::span<const std::uint32_t>(m.emitted.data(), m.emitted_n));
        }
        if (m.draft_hash != 0U) {
            if (m.draft_hash != mine) throw pair::ProtocolError("spec pair round: the two ranks' drafters diverged (draft ids differ)");
            mine = 0;
        }
        if (m.kind == kDraft) {
            need_chain();
            chain->draft(std::span<const std::uint32_t>(m.tokens.data(), m.rows), m.draft_k, drafted);
            mine = draft_hash(drafted);
            ++res.drafts;
            continue;
        }
        if (m.rows == 0U) {
            const auto pos = position();
            if (pos != m.position)
                throw pair::ProtocolError("spec pair round: follower position " + std::to_string(pos) + " != leader " +
                                          std::to_string(m.position));
            return res;
        }
        target.verify(std::span<const std::uint32_t>(m.tokens.data(), m.rows), greedy);
        ++res.passes;
        res.rows += m.rows;
    }
}

}  // namespace strix::spec::pair_round
