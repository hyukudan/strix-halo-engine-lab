#pragma once

#include "strix/serve/engine.hpp"
#include "strix/serve/job_sampling.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve {

// The follower replays the lead's seeds only while every request shares identity 1 (the shared mode switch).
inline bool pair_sampling_replays_seeds() { return !token_sampling_request_identity_counts(); }

// A per-request sampler held by value in the pair schedulers (the shared translation, strix/serve/job_sampling.hpp).
inline std::optional<TokenSampler> make_pair_sampler(const GenerationJob& job, const TokenSamplerOptions& options = {}) {
    const auto shared = make_job_sampler(job.sampling, options);
    if (!shared) return std::nullopt;
    return std::optional<TokenSampler>(std::in_place, shared->policy(), shared->options());
}

struct PairSamplingRow {
    TokenSampler* sampler = nullptr;
    std::uint64_t position = 0;
    // The request's effective stop token ids (lead only; a sampled verify window ends at the first one it decides,
    // decide_pair_window). Empty: none. Callers that decide one row per request leave it empty.
    std::span<const std::uint32_t> stops{};
};

namespace detail {
inline constexpr std::size_t kPairSliceFrameFloats = 16384;  // 64 KiB control-frame limit

// One row's follower slice moves to the lead in frames of at most kPairSliceFrameFloats floats (the full-duplex control
// transports need equal message lengths, so the lead sends padding). Lead: `local` is its slice and `full` receives
// local + follower slice; follower: `local` is its slice (`full` unused).
template<class Exchange>
void move_pair_slice(bool lead, std::span<const float> local, std::size_t follower_width, std::vector<float>& full,
                     std::vector<float>& scratch, Exchange&& exchange) {
    if (lead) {
        full.resize(local.size() + follower_width);
        std::copy(local.begin(), local.end(), full.begin());
    } else if (local.size() != follower_width) {
        throw std::logic_error("pair sampling: wrong follower vocabulary slice");
    }
    scratch.resize(std::min(kPairSliceFrameFloats, follower_width));
    for (std::size_t begin = 0; begin < follower_width; begin += kPairSliceFrameFloats) {
        const std::size_t count = std::min(kPairSliceFrameFloats, follower_width - begin);
        auto buffer = std::span(scratch).first(count);
        if (lead) exchange(std::span<const float>(buffer), std::span(full).subspan(local.size() + begin, count));
        else exchange(local.subspan(begin, count), buffer);
    }
}
}  // namespace detail

// The follower sends its vocabulary slice; only the lead assembles the row and samples it.
// A zero follower width is the one-node route. A zero mask leaves the greedy picks in place.
template<class Read, class Exchange>
void sample_pair_rows(std::uint32_t mask, bool lead, std::size_t follower_width, std::size_t vocabulary,
                      std::span<const PairSamplingRow> choices, std::span<std::uint32_t> tokens,
                      Read&& read, Exchange&& exchange) {
    if (mask == 0) return;
    std::vector<float> scratch, full;
    for (std::size_t row = 0; mask != 0; ++row, mask >>= 1) {
        if ((mask & 1U) == 0) continue;
        const std::span<const float> local = read(row);
        if (lead && (row >= choices.size() || row >= tokens.size() || !choices[row].sampler))
            throw std::logic_error("pair sampling: missing lead sampler");
        detail::move_pair_slice(lead, local, follower_width, full, scratch, exchange);
        if (lead) {
            const auto logits = std::span<const float>(full).first(std::min(vocabulary, full.size()));
            tokens[row] = choices[row].sampler->decide(logits, choices[row].position, nullptr).token;
        }
    }
}

// The lead's message after each row of a window: [kPairWindowMore] while another row follows, then
// [kPairWindowDone or kPairWindowStop, n, accepted, token 0 .. token n-1, 0 ...] (accepted: the leading drafts the
// acceptance rule accepted, n - 1, or n only when n == max_rows or the window stopped). kPairWindowStop: token n - 1 is
// one of the request's stop ids; the request ends there, so no row after it is decided, committed or caught up.
inline constexpr std::size_t kPairWindowMaxRows = 8;
inline constexpr std::size_t kPairWindowWords = kPairWindowMaxRows + 3U;
inline constexpr std::uint32_t kPairWindowDone = 0, kPairWindowMore = 1, kPairWindowStop = 2;

// What one sampled verify window decided, the same on both ranks. A rejected row's token can equal its draft (coupled
// acceptance keeps the plain sample; a forced rejection keeps it too), so a caller commits drafts by `accepted`, never
// by comparing tokens with drafts: rows 0 .. accepted - 1 accepted their drafts, row `accepted` (when n > accepted)
// decided the window's last token.
struct PairWindowDecision {
    std::vector<std::uint32_t> tokens;
    std::uint32_t accepted = 0;
    bool stop = false;  // tokens.back() is a stop id: the request ends at row tokens.size() - 1
};

// One sampled verify window across the pair. Row r of the window decides the token at first_position + r with
// TokenSampler::decide_rows (drafts[r] is the draft row r checks; `stops` are the request's stop ids, lead only): it
// stops after the first row without a draft, the first rejected draft, the first stop id or max_rows rows. Under coupled acceptance a draft only needs its token; under
// rejection it needs the distribution it was drawn from. Every row the lead reaches is assembled from both ranks'
// vocabulary slices (the follower's slice moves as in sample_pair_rows); after each row the lead tells the follower
// whether another follows, then sends the decision. Both ranks pass the same max_rows (1 .. kPairWindowMaxRows, from
// replicated data): the follower never reads a row at or past it and refuses a lead that asks for one. The lead's
// sampler counts the decisions; the follower ignores sampler, first_position and drafts. follower_width 0: one node,
// no exchange. `words` exchanges kPairWindowWords words (the follower sends zeros).
template<class Read, class ExchangeFloats, class ExchangeWords>
PairWindowDecision decide_pair_window(bool lead, std::size_t follower_width, std::size_t vocabulary,
                                      TokenSampler* sampler, std::uint64_t first_position,
                                      std::span<const TokenDraft> drafts, std::uint32_t max_rows,
                                      std::span<const std::uint32_t> stops, Read&& read, ExchangeFloats&& exchange,
                                      ExchangeWords&& words) {
    const bool paired = follower_width != 0;
    if (max_rows == 0 || max_rows > kPairWindowMaxRows)
        throw std::invalid_argument("pair window: max_rows must be 1 .. 8");
    std::vector<float> scratch, full;
    std::array<std::uint32_t, kPairWindowWords> send{}, receive{};
    if (!lead) {
        if (!paired) throw std::logic_error("pair window: a follower needs a vocabulary slice");
        for (std::uint32_t row = 0;; ++row) {
            detail::move_pair_slice(false, read(row), follower_width, full, scratch, exchange);
            words(std::span<const std::uint32_t>(send), std::span<std::uint32_t>(receive));
            if (receive[0] == kPairWindowMore && row + 1U < max_rows) continue;
            const std::uint32_t n = receive[1], accepted = receive[2];
            // Every drafted row the lead decides either rejects (the window ends with it) or accepts and continues, so an
            // all-accepted window ends only at max_rows or at an accepted stop id (kPairWindowStop): an all-accepted DONE
            // before max_rows leaves a row undecided.
            const bool stop = receive[0] == kPairWindowStop;
            if ((receive[0] != kPairWindowDone && !stop) || n != row + 1U || accepted > n || accepted + 1U < n ||
                (accepted == n && n < max_rows && !stop))
                throw std::runtime_error("pair window: the lead's decision message is out of protocol");
            return {{receive.begin() + 3, receive.begin() + 3 + n}, accepted, stop};
        }
    }
    if (sampler == nullptr) throw std::logic_error("pair window: missing lead sampler");
    if (drafts.size() + 1U < max_rows) throw std::invalid_argument("pair window: max_rows must be at most drafts + 1");
    std::uint32_t reached = 0;
    const auto logits = [&](std::uint32_t row) -> std::span<const float> {
        if (row != reached) throw std::logic_error("pair window: rows out of order");
        if (paired && row != 0) {  // the previous row continues the window: the follower sends this one
            send.fill(0);
            send[0] = kPairWindowMore;
            words(std::span<const std::uint32_t>(send), std::span<std::uint32_t>(receive));
        }
        ++reached;
        const std::span<const float> local = read(row);
        if (!paired) return local.first(std::min(vocabulary, local.size()));
        detail::move_pair_slice(true, local, follower_width, full, scratch, exchange);
        return std::span<const float>(full).first(std::min(vocabulary, full.size()));
    };
    const auto decisions = sampler->decide_rows(max_rows, first_position, drafts, stops, logits);
    PairWindowDecision out;
    out.tokens.reserve(decisions.size());
    for (const auto& d : decisions) out.tokens.push_back(d.token);
    while (out.accepted < decisions.size() && decisions[out.accepted].drafted && decisions[out.accepted].accepted) ++out.accepted;
    out.stop = !out.tokens.empty() && std::find(stops.begin(), stops.end(), out.tokens.back()) != stops.end();
    if (paired) {
        send.fill(0);
        send[0] = out.stop ? kPairWindowStop : kPairWindowDone;
        send[1] = static_cast<std::uint32_t>(out.tokens.size());
        send[2] = out.accepted;
        std::copy(out.tokens.begin(), out.tokens.end(), send.begin() + 3);
        words(std::span<const std::uint32_t>(send), std::span<std::uint32_t>(receive));
    }
    return out;
}
// A window of a request without stop ids.
template<class Read, class ExchangeFloats, class ExchangeWords>
PairWindowDecision decide_pair_window(bool lead, std::size_t follower_width, std::size_t vocabulary,
                                      TokenSampler* sampler, std::uint64_t first_position,
                                      std::span<const TokenDraft> drafts, std::uint32_t max_rows,
                                      Read&& read, ExchangeFloats&& exchange, ExchangeWords&& words) {
    return decide_pair_window(lead, follower_width, vocabulary, sampler, first_position, drafts, max_rows,
                              std::span<const std::uint32_t>{}, std::forward<Read>(read),
                              std::forward<ExchangeFloats>(exchange), std::forward<ExchangeWords>(words));
}

// One sampled draft across the pair: the follower sends its vocabulary slice of the drafter row (`read_row()`), the
// lead draws TokenSampler::draft over the full row for the token at `position` (the sampler's draft transform and
// proposal uniform) and sends the token back ([token, 0 ...], kPairWindowWords words). Both ranks return the token; the
// lead's draft, with the distribution it was drawn from, goes to `kept` when given (rejection acceptance reads it).
// follower_width 0: one node, no exchange. The follower ignores sampler, position and kept.
template<class Read, class ExchangeFloats, class ExchangeWords>
std::uint32_t pick_pair_draft(bool lead, std::size_t follower_width, std::size_t vocabulary, TokenSampler* sampler,
                              std::uint64_t position, Read&& read_row, ExchangeFloats&& exchange, ExchangeWords&& words,
                              TokenDraft* kept = nullptr) {
    const bool paired = follower_width != 0;
    std::vector<float> scratch, full;
    std::array<std::uint32_t, kPairWindowWords> send{}, receive{};
    if (!lead) {
        if (!paired) throw std::logic_error("pair draft: a follower needs a vocabulary slice");
        detail::move_pair_slice(false, read_row(), follower_width, full, scratch, exchange);
        words(std::span<const std::uint32_t>(send), std::span<std::uint32_t>(receive));
        return receive[0];
    }
    if (sampler == nullptr) throw std::logic_error("pair draft: missing lead sampler");
    const std::span<const float> local = read_row();
    std::span<const float> row = local;
    if (paired) {
        detail::move_pair_slice(true, local, follower_width, full, scratch, exchange);
        row = full;
    }
    auto draft = sampler->draft(row.first(std::min(vocabulary, row.size())), {}, position);
    const std::uint32_t token = draft.token;
    if (paired) {
        send[0] = token;
        words(std::span<const std::uint32_t>(send), std::span<std::uint32_t>(receive));
    }
    if (kept) *kept = std::move(draft);
    return token;
}

}  // namespace strix::serve
