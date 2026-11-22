#include "strix/serve/replicated_cohort.hpp"

#include "strix/pair/lockstep.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix::serve {
namespace {

// Control block layout (32-bit words).
constexpr std::uint32_t kMagic = 0x53524331U;  // "SRC1"
enum Word : std::uint32_t {
    kWMagic = 0, kWOp, kWRoundLo, kWRoundHi, kWAckLo, kWAckHi, kWRank, kWArg, kWFlag,
    kWRecords,       // deferred commit / release records
    kWDeferred,      // words of the deferred section (it leads the payload)
    kWPayload,       // payload words: deferred section, then the call's body
    kWInline,        // first inline payload word
};
constexpr std::uint32_t kInlineWords = kReplicaBlockWords - kWInline;

enum Op : std::uint32_t { kOpNone = 0, kOpPrefill = 1, kOpVerify = 2, kOpFlush = 3, kOpStop = 4 };
enum Record : std::uint32_t { kRecCommit = 1, kRecRelease = 2 };

using Block = std::array<std::uint32_t, kReplicaBlockWords>;

std::uint32_t lo(std::uint64_t v) { return static_cast<std::uint32_t>(v & 0xffffffffU); }
std::uint32_t hi(std::uint64_t v) { return static_cast<std::uint32_t>(v >> 32U); }
std::uint64_t join(std::uint32_t l, std::uint32_t h) { return (std::uint64_t{h} << 32U) | l; }

pair::LockstepHeader header_for(std::uint64_t round, std::uint32_t rank) {
    return {round, round == 0 ? pair::kNoRound : round - 1U, rank};
}

void put_header(Block& b, const pair::LockstepHeader& h, std::uint32_t op) {
    b.fill(0U);
    b[kWMagic] = kMagic;
    b[kWOp] = op;
    b[kWRoundLo] = lo(h.round);
    b[kWRoundHi] = hi(h.round);
    b[kWAckLo] = lo(h.ack_round);
    b[kWAckHi] = hi(h.ack_round);
    b[kWRank] = h.rank;
}

pair::LockstepHeader get_header(const Block& b) {
    if (b[kWMagic] != kMagic) throw pair::ProtocolError("cohort replica: bad control block magic");
    return {join(b[kWRoundLo], b[kWRoundHi]), join(b[kWAckLo], b[kWAckHi]), b[kWRank]};
}

std::uint32_t words_of(std::size_t n, const char* what) {
    if (n > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error(std::string("cohort replica: ") + what + " exceeds 2^32 words");
    return static_cast<std::uint32_t>(n);
}

void check_frame(std::uint32_t frame_words) {
    if (frame_words == 0U) throw std::invalid_argument("cohort replica: frame_words >= 1");
}

}  // namespace

ReplicatedCohort::ReplicatedCohort(CohortBackend& local, dist::Transport& link, ReplicaOptions options)
    : local_(local), link_(link), options_(options) {
    check_frame(options_.frame_words);
}

ReplicatedCohort::~ReplicatedCohort() {
    if (closed_) return;
    try {
        link_.abort("cohort replica: rank 0 ended without close()");
    } catch (...) {  // a destructor reports nothing further; rank 1 sees the aborted link
    }
}

std::uint32_t ReplicatedCohort::lanes() const { return local_.lanes(); }
std::uint32_t ReplicatedCohort::rows_per_forward() const { return local_.rows_per_forward(); }
std::uint32_t ReplicatedCohort::max_positions() const { return local_.max_positions(); }
bool ReplicatedCohort::supports_final_logits() const { return local_.supports_final_logits(); }
std::span<const float> ReplicatedCohort::read_final_logits(std::uint32_t lane) { return local_.read_final_logits(lane); }

void ReplicatedCohort::send(std::uint32_t op, std::uint32_t arg, std::uint32_t flag, std::span<const std::uint32_t> body) {
    if (closed_) throw std::logic_error("cohort replica: call after close()");
    try {
        const std::uint32_t deferred = words_of(deferred_.size(), "deferred records");
        const std::uint32_t payload = words_of(deferred_.size() + body.size(), "payload");
        const auto mine = header_for(round_, 0U);
        Block out{}, in{};
        put_header(out, mine, op);
        out[kWArg] = arg;
        out[kWFlag] = flag;
        out[kWRecords] = deferred_records_;
        out[kWDeferred] = deferred;
        out[kWPayload] = payload;
        // Payload word i is deferred_[i] for i < deferred, else body[i - deferred].
        const auto word = [&](std::size_t i) { return i < deferred_.size() ? deferred_[i] : body[i - deferred_.size()]; };
        const std::size_t inline_n = std::min<std::size_t>(payload, kInlineWords);
        for (std::size_t i = 0; i < inline_n; ++i) out[kWInline + i] = word(i);
        link_.exchange(std::as_bytes(std::span(out)), std::as_writable_bytes(std::span(in)));
        const auto peer = get_header(in);
        pair::check_pair(mine, peer);
        std::vector<std::uint32_t> frame;
        for (std::size_t at = inline_n; at < payload;) {
            const std::size_t n = std::min<std::size_t>(payload - at, options_.frame_words);
            frame.resize(n);
            for (std::size_t i = 0; i < n; ++i) frame[i] = word(at + i);
            link_.send(std::as_bytes(std::span(frame)));
            at += n;
        }
    } catch (const std::exception& error) {
        closed_ = true;  // the pair is out of step: nothing further may run on either rank
        link_.abort(error.what());
        throw;
    }
    deferred_.clear();
    deferred_records_ = 0;
    ++round_;
}

void ReplicatedCohort::prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) {
    send(kOpPrefill, lane, first ? 1U : 0U, tokens);
    local_.prefill(lane, tokens, first);
}

void ReplicatedCohort::verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                              std::span<std::uint32_t> top) {
    std::uint64_t rows = 0;
    body_.clear();
    body_.reserve(2U * groups.size() + tokens.size());
    for (const auto& g : groups) {
        body_.push_back(g.lane);
        body_.push_back(g.rows);
        rows += g.rows;
    }
    if (rows != tokens.size() || top.size() != tokens.size())
        throw std::invalid_argument("cohort replica: verify rows, tokens and top disagree");
    body_.insert(body_.end(), tokens.begin(), tokens.end());
    send(kOpVerify, words_of(groups.size(), "groups"), 0U, body_);
    local_.verify(groups, tokens, top);
}

void ReplicatedCohort::commit(std::span<const std::uint32_t> accept) {
    if (closed_) throw std::logic_error("cohort replica: call after close()");
    if (options_.local_ops) local_.commit(accept);
    deferred_.push_back(kRecCommit);
    deferred_.push_back(words_of(accept.size(), "commit"));
    deferred_.insert(deferred_.end(), accept.begin(), accept.end());
    ++deferred_records_;
    if (!options_.local_ops) {
        send(kOpFlush, 0U, 0U, {});
        local_.commit(accept);
    }
}

void ReplicatedCohort::release(std::uint32_t lane) {
    if (closed_) throw std::logic_error("cohort replica: call after close()");
    if (options_.local_ops) local_.release(lane);
    deferred_.push_back(kRecRelease);
    deferred_.push_back(lane);
    ++deferred_records_;
    if (!options_.local_ops) {
        send(kOpFlush, 0U, 0U, {});
        local_.release(lane);
    }
}

void ReplicatedCohort::keepalive() { send(kOpFlush, 0U, 0U, {}); }

void ReplicatedCohort::close() {
    send(kOpStop, 0U, 0U, {});
    closed_ = true;
}

std::uint64_t follow_cohort(CohortBackend& local, dist::Transport& link, ReplicaOptions options) {
    check_frame(options.frame_words);
    std::uint64_t round = 0;
    Block out{}, in{};
    std::vector<std::uint32_t> payload, top;
    std::vector<CohortGroup> groups;
    try {
        for (;;) {
            const auto mine = header_for(round, 1U);
            put_header(out, mine, kOpNone);
            link.exchange(std::as_bytes(std::span(out)), std::as_writable_bytes(std::span(in)));
            pair::check_pair(mine, get_header(in));
            const std::uint32_t op = in[kWOp], words = in[kWPayload], deferred = in[kWDeferred];
            if (op < kOpPrefill || op > kOpStop) throw pair::ProtocolError("cohort replica: unknown operation");
            if (deferred > words) throw pair::ProtocolError("cohort replica: deferred section beyond the payload");
            payload.resize(words);
            const std::uint32_t inline_n = std::min(words, kInlineWords);
            std::copy_n(in.begin() + kWInline, inline_n, payload.begin());
            for (std::uint32_t at = inline_n; at < words;) {
                const std::uint32_t n = std::min(words - at, options.frame_words);
                link.receive(std::as_writable_bytes(std::span(payload).subspan(at, n)));
                at += n;
            }
            // Validate the whole message before the backend sees any of it.
            const std::span<const std::uint32_t> records(payload.data(), deferred);
            const std::span<const std::uint32_t> body(payload.data() + deferred, words - deferred);
            std::uint32_t count = 0;
            for (std::size_t at = 0; at < records.size(); ++count) {
                if (at + 2U > records.size()) throw pair::ProtocolError("cohort replica: truncated record");
                if (records[at] == kRecCommit) at += 2U + std::size_t{records[at + 1U]};
                else if (records[at] == kRecRelease) at += 2U;
                else throw pair::ProtocolError("cohort replica: unknown record");
                if (at > records.size()) throw pair::ProtocolError("cohort replica: truncated commit record");
            }
            if (count != in[kWRecords]) throw pair::ProtocolError("cohort replica: record count mismatch");
            groups.clear();
            std::span<const std::uint32_t> tokens = body;
            if (op == kOpVerify) {
                const std::uint64_t n = in[kWArg];
                if (2U * n > body.size()) throw pair::ProtocolError("cohort replica: truncated groups");
                std::uint64_t rows = 0;
                for (std::uint64_t g = 0; g < n; ++g) {
                    groups.push_back({body[2U * g], body[2U * g + 1U]});
                    rows += body[2U * g + 1U];
                }
                tokens = body.subspan(2U * n);
                if (rows != tokens.size()) throw pair::ProtocolError("cohort replica: group rows and tokens disagree");
            } else if (op != kOpPrefill && !body.empty()) {
                throw pair::ProtocolError("cohort replica: body on a call without one");
            }
            // Apply: rank 0's queued commits / releases first, in its order, then the call.
            for (std::size_t at = 0; at < records.size();) {
                if (records[at] == kRecCommit) {
                    const std::size_t n = records[at + 1U];
                    local.commit(records.subspan(at + 2U, n));
                    at += 2U + n;
                } else {
                    local.release(records[at + 1U]);
                    at += 2U;
                }
            }
            ++round;
            switch (op) {
            case kOpPrefill:
                local.prefill(in[kWArg], body, in[kWFlag] != 0U);
                break;
            case kOpVerify:
                top.resize(tokens.size());
                local.verify(groups, tokens, top);  // rank 0 selects; rank 1 only advances its share of the forward
                break;
            case kOpStop:
                return round;
            default:
                break;  // flush: the records were the message
            }
        }
    } catch (const std::exception& error) {
        link.abort(error.what());
        throw;
    }
}

}  // namespace strix::serve
