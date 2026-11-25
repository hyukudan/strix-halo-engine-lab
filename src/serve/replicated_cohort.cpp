#include "strix/serve/replicated_cohort.hpp"

#include "strix/pair/lockstep.hpp"

#include <algorithm>
#include <array>
#include <bit>
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

enum Op : std::uint32_t {
    kOpNone = 0, kOpPrefill = 1, kOpVerify = 2, kOpFlush = 3, kOpStop = 4, kOpPrefillGroup = 5,
    kOpLogits = 6,   // collective_reads: read_final_logits(arg)
    kOpPropose = 7,  // collective_reads: propose(arg, room = flag)
    kOpSample = 8,   // collective_reads: sample_final(arg, sampler, position); body = sample_words()
};
// Variable records: [kind, n, n words]. Commit: accept[]. Capture: lane, committed[]. Restore: lane, rows, prompt[0, rows]
// (prompt[0, rows) when the whole prompt was restored).
// Bind: lane, id lo, id hi, traits[]. Control: words[].
enum Record : std::uint32_t {
    kRecCommit = 1, kRecRelease = 2, kRecCapture = 3, kRecRestore = 4, kRecBind = 5, kRecControl = 6
};

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

// sample_final body: position (2 words), the sampler's policy (6) and options (5); rank 1 rebuilds an equal sampler.
constexpr std::size_t kSampleWords = 13;
std::array<std::uint32_t, kSampleWords> sample_words(const TokenSampler& sampler, std::uint64_t position) {
    const auto& p = sampler.policy();
    const auto& o = sampler.options();
    return {lo(position), hi(position), std::bit_cast<std::uint32_t>(p.temperature), p.top_k,
            std::bit_cast<std::uint32_t>(p.top_p), std::bit_cast<std::uint32_t>(p.min_p), lo(p.seed), hi(p.seed),
            static_cast<std::uint32_t>(o.acceptance), std::bit_cast<std::uint32_t>(o.argmax_draft_below),
            lo(o.request_identity), hi(o.request_identity), o.force_reject_row};
}
TokenSampler sampler_of(std::span<const std::uint32_t> w) {
    TokenSamplingPolicy p;
    p.temperature = std::bit_cast<float>(w[2]);
    p.top_k = w[3];
    p.top_p = std::bit_cast<float>(w[4]);
    p.min_p = std::bit_cast<float>(w[5]);
    p.seed = join(w[6], w[7]);
    TokenSamplerOptions o;
    o.acceptance = static_cast<SampledAcceptance>(w[8]);
    o.argmax_draft_below = std::bit_cast<float>(w[9]);
    o.request_identity = join(w[10], w[11]);
    o.force_reject_row = w[12];
    return TokenSampler(p, o);
}

void check_replica_link(const dist::Transport& link, std::uint32_t frame_words) {
    if (frame_words == 0U) throw std::invalid_argument("cohort replica: frame_words >= 1");
    if (!link.supports_peer_abort()) throw std::invalid_argument("cohort replica: transport requires peer cancellation");
}

void abort_link(dist::Transport& link, const char* reason) noexcept {
    try {
        link.abort(reason);
    } catch (...) {
        // Under memory pressure even building the reason string can fail. Retry without it.
        try {
            link.abort({});
        } catch (...) {  // Keep the original exception.
        }
    }
}

}  // namespace

ReplicatedCohort::ReplicatedCohort(CohortBackend& local, dist::Transport& link, ReplicaOptions options)
    : local_(local), link_(link), options_(options) {
    check_replica_link(link_, options_.frame_words);
}

ReplicatedCohort::~ReplicatedCohort() {
    if (!closed_) abort_link(link_, "cohort replica: rank 0 ended without close()");
}

std::uint32_t ReplicatedCohort::lanes() const { return local_.lanes(); }
std::uint32_t ReplicatedCohort::rows_per_forward() const { return local_.rows_per_forward(); }
std::uint32_t ReplicatedCohort::max_positions() const { return local_.max_positions(); }
bool ReplicatedCohort::supports_final_logits() const { return local_.supports_final_logits(); }
std::span<const float> ReplicatedCohort::read_final_logits(std::uint32_t lane) {
    if (options_.collective_reads) send(kOpLogits, lane, 0U, {});
    return local_.read_final_logits(lane);
}
bool ReplicatedCohort::samples_on_device() const { return local_.samples_on_device(); }
std::uint32_t ReplicatedCohort::sample_final(std::uint32_t lane, TokenSampler& sampler, std::uint64_t position) {
    if (options_.collective_reads) send(kOpSample, lane, 0U, sample_words(sampler, position));
    return local_.sample_final(lane, sampler, position);
}
bool ReplicatedCohort::prefill_emits_first() const { return local_.prefill_emits_first(); }
std::uint32_t ReplicatedCohort::prefill_top(std::uint32_t lane) { return local_.prefill_top(lane); }
std::uint32_t ReplicatedCohort::max_window(std::uint32_t lane) const { return local_.max_window(lane); }

void ReplicatedCohort::abort_replica(const char* reason) noexcept {
    closed_ = true;
    abort_link(link_, reason);
}

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
        abort_replica(error.what());
        throw;
    } catch (...) {
        abort_replica("");
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
    // The verify frame carries no score-read flag yet: both ranks run every group as read (CohortGroup's default), so
    // rank 0 and rank 1 always take the same backend route. Finish any group allocation before rank 1 can start.
    read_groups_.assign(groups.begin(), groups.end());
    for (auto& g : read_groups_) g.reads_scores = true;
    send(kOpVerify, words_of(groups.size(), "groups"), 0U, body_);
    try {
        local_.verify(read_groups_, tokens, top);
    } catch (const std::exception& error) {
        abort_replica(error.what());
        throw;
    } catch (...) {
        abort_replica("cohort replica: rank 0 verify failed");
        throw;
    }
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

void ReplicatedCohort::defer(std::uint32_t kind, std::span<const std::uint32_t> head, std::span<const std::uint32_t> body) {
    deferred_.push_back(kind);
    deferred_.push_back(words_of(head.size() + body.size(), "record"));
    deferred_.insert(deferred_.end(), head.begin(), head.end());
    deferred_.insert(deferred_.end(), body.begin(), body.end());
    ++deferred_records_;
}

std::uint64_t ReplicatedCohort::restore(std::uint32_t lane, std::span<const std::uint32_t> prompt) {
    if (closed_) throw std::logic_error("cohort replica: call after close()");
    const std::uint64_t rows = local_.restore(lane, prompt);
    if (rows == 0) return 0;  // nothing restored: rank 1's lane starts with the same prefill(first = true)
    // rank 1 restores from the prompt cut after the restored rows plus the anchor, so it cannot restore more; a
    // whole-prompt restore (prefill_emits_first, rows == prompt.size()) has no anchor and sends the whole prompt
    if (rows > prompt.size()) throw std::logic_error("cohort replica: restore returned more rows than the prompt");
    const std::uint32_t head[2] = {lane, words_of(rows, "restore")};
    defer(kRecRestore, head, prompt.first(std::min(static_cast<std::size_t>(rows) + 1U, prompt.size())));
    if (!options_.local_ops) send(kOpFlush, 0U, 0U, {});
    return rows;
}

void ReplicatedCohort::capture(std::uint32_t lane, std::span<const std::uint32_t> committed) {
    if (closed_) throw std::logic_error("cohort replica: call after close()");
    if (options_.local_ops) local_.capture(lane, committed);
    const std::uint32_t head[1] = {lane};
    defer(kRecCapture, head, committed);
    if (!options_.local_ops) {
        send(kOpFlush, 0U, 0U, {});
        local_.capture(lane, committed);
    }
}

void ReplicatedCohort::bind_lane(std::uint32_t lane, std::uint64_t request_id, std::span<const std::uint32_t> traits) {
    if (closed_) throw std::logic_error("cohort replica: call after close()");
    if (options_.local_ops) local_.bind_lane(lane, request_id, traits);
    const std::uint32_t head[3] = {lane, lo(request_id), hi(request_id)};
    defer(kRecBind, head, traits);
    if (!options_.local_ops) {
        send(kOpFlush, 0U, 0U, {});
        local_.bind_lane(lane, request_id, traits);
    }
}

void ReplicatedCohort::control(std::span<const std::uint32_t> words) {
    if (closed_) throw std::logic_error("cohort replica: call after close()");
    if (options_.local_ops) local_.control(words);
    defer(kRecControl, {}, words);
    if (!options_.local_ops) {
        send(kOpFlush, 0U, 0U, {});
        local_.control(words);
    }
}

bool ReplicatedCohort::device_drafts() const { return local_.device_drafts(); }

void ReplicatedCohort::propose(std::uint32_t lane, std::uint32_t room, std::vector<std::uint32_t>& tokens,
                               std::vector<double>& probability) {
    // drafts reach rank 1 as verify rows; a two-rank drafter also needs rank 1 inside the proposal itself
    if (options_.collective_reads) send(kOpPropose, lane, room, {});
    local_.propose(lane, room, tokens, probability);
}

PrefillGroupCapacity ReplicatedCohort::prompt_group_capacity() const { return local_.prompt_group_capacity(); }

std::uint32_t ReplicatedCohort::prompt_route_class(std::uint64_t rows) const { return local_.prompt_route_class(rows); }

void ReplicatedCohort::prefill_group(std::span<const CohortPrompt> members) {
    body_.clear();
    for (const auto& m : members) {
        body_.push_back(m.lane);
        body_.push_back(m.first ? 1U : 0U);
        body_.push_back(words_of(m.tokens.size(), "group member"));
    }
    for (const auto& m : members) body_.insert(body_.end(), m.tokens.begin(), m.tokens.end());
    send(kOpPrefillGroup, words_of(members.size(), "group"), 0U, body_);
    local_.prefill_group(members);
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
    check_replica_link(link, options.frame_words);
    std::uint64_t round = 0;
    Block out{}, in{};
    std::vector<std::uint32_t> payload, top;
    std::vector<CohortGroup> groups;
    std::vector<CohortPrompt> members;
    std::vector<std::uint32_t> drafts;
    std::vector<double> probability;
    try {
        for (;;) {
            const auto mine = header_for(round, 1U);
            put_header(out, mine, kOpNone);
            link.exchange(std::as_bytes(std::span(out)), std::as_writable_bytes(std::span(in)));
            pair::check_pair(mine, get_header(in));
            const std::uint32_t op = in[kWOp], words = in[kWPayload], deferred = in[kWDeferred];
            if (op < kOpPrefill || op > kOpSample) throw pair::ProtocolError("cohort replica: unknown operation");
            if ((op == kOpLogits || op == kOpPropose || op == kOpSample) && !options.collective_reads)
                throw pair::ProtocolError("cohort replica: collective read on a follower without collective_reads");
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
                const std::uint32_t kind = records[at];
                if (kind == kRecCommit) at += 2U + std::size_t{records[at + 1U]};
                else if (kind == kRecRelease) at += 2U;
                else if (kind == kRecCapture || kind == kRecRestore) {
                    const std::size_t n = records[at + 1U];
                    if (n < (kind == kRecCapture ? 1U : 3U)) throw pair::ProtocolError("cohort replica: short record");
                    // restore: rows + anchor words, or exactly rows words for a whole-prompt restore (a backend whose
                    // prefill emits the first id may restore the whole prompt)
                    if (kind == kRecRestore && at + 4U <= records.size() && std::size_t{records[at + 3U]} + 3U != n &&
                        !(local.prefill_emits_first() && std::size_t{records[at + 3U]} + 2U == n))
                        throw pair::ProtocolError("cohort replica: restore rows and prompt disagree");
                    at += 2U + n;
                } else if (kind == kRecBind || kind == kRecControl) {
                    const std::size_t n = records[at + 1U];
                    if (kind == kRecBind && n < 3U) throw pair::ProtocolError("cohort replica: short record");
                    at += 2U + n;
                } else throw pair::ProtocolError("cohort replica: unknown record");
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
                    groups.push_back({body[2U * g], body[2U * g + 1U], true});  // as rank 0: every group read
                    rows += body[2U * g + 1U];
                }
                tokens = body.subspan(2U * n);
                if (rows != tokens.size()) throw pair::ProtocolError("cohort replica: group rows and tokens disagree");
            } else if (op == kOpPrefillGroup) {
                const std::uint64_t n = in[kWArg];
                if (n == 0 || 3U * n > body.size()) throw pair::ProtocolError("cohort replica: truncated group");
                std::uint64_t rows = 0;
                for (std::uint64_t m = 0; m < n; ++m) rows += body[3U * m + 2U];
                if (rows != body.size() - 3U * n) throw pair::ProtocolError("cohort replica: group rows and tokens disagree");
            } else if (op == kOpSample) {
                if (body.size() != kSampleWords || body[8] > static_cast<std::uint32_t>(SampledAcceptance::coupled))
                    throw pair::ProtocolError("cohort replica: bad sample call");
            } else if (op != kOpPrefill && !body.empty()) {
                throw pair::ProtocolError("cohort replica: body on a call without one");
            }
            // Apply: rank 0's queued commits / releases first, in its order, then the call.
            for (std::size_t at = 0; at < records.size();) {
                if (records[at] == kRecCommit) {
                    const std::size_t n = records[at + 1U];
                    local.commit(records.subspan(at + 2U, n));
                    at += 2U + n;
                } else if (records[at] == kRecCapture) {
                    const std::size_t n = records[at + 1U];
                    local.capture(records[at + 2U], records.subspan(at + 3U, n - 1U));
                    at += 2U + n;
                } else if (records[at] == kRecBind) {
                    const std::size_t n = records[at + 1U];
                    local.bind_lane(records[at + 2U], join(records[at + 3U], records[at + 4U]),
                                    records.subspan(at + 5U, n - 3U));
                    at += 2U + n;
                } else if (records[at] == kRecControl) {
                    const std::size_t n = records[at + 1U];
                    local.control(records.subspan(at + 2U, n));
                    at += 2U + n;
                } else if (records[at] == kRecRestore) {
                    const std::size_t n = records[at + 1U];
                    if (local.restore(records[at + 2U], records.subspan(at + 4U, n - 2U)) != records[at + 3U])
                        throw pair::ProtocolError("cohort replica: the ranks restored different retained prefixes");
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
            case kOpPrefillGroup: {
                const std::size_t n = in[kWArg];
                members.clear();
                std::size_t at = 3U * n;
                for (std::size_t m = 0; m < n; ++m) {
                    const std::size_t rows = body[3U * m + 2U];
                    members.push_back({body[3U * m], body.subspan(at, rows), body[3U * m + 1U] != 0U});
                    at += rows;
                }
                local.prefill_group(members);
                break;
            }
            case kOpVerify:
                top.resize(tokens.size());
                local.verify(groups, tokens, top);  // rank 0 selects; rank 1 only advances its share of the forward
                break;
            case kOpLogits:
                (void)local.read_final_logits(in[kWArg]);  // rank 1's share of the gather; rank 0 decides
                break;
            case kOpSample: {
                auto sampler = sampler_of(body);  // rank 1's share of the device decision; rank 0's id rules
                (void)local.sample_final(in[kWArg], sampler, join(body[0], body[1]));
                break;
            }
            case kOpPropose:
                local.propose(in[kWArg], in[kWFlag], drafts, probability);  // rank 1's share; rank 0's drafts rule
                break;
            case kOpStop:
                return round;
            default:
                break;  // flush: the records were the message
            }
        }
    } catch (const std::exception& error) {
        abort_link(link, error.what());
        throw;
    } catch (...) {
        abort_link(link, "cohort replica: rank 1 failed");
        throw;
    }
}

}  // namespace strix::serve
