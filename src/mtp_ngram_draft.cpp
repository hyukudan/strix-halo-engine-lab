#include "strix/mtp_ngram_draft.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix {
namespace {
constexpr std::size_t kSuffixCells = 3;  // suffix 2, 3, 4+
constexpr std::size_t kRowCells = 3;     // row 0, 1-2, 3+
constexpr std::size_t kPeerFirst = kSuffixCells * 2 * kRowCells; // then one row-bucketed group for peer proposals

// Exact common suffix of S (accessed through at, `total` tokens) and of `context` up to `end`, capped at `cap`.
template <class At>
std::uint32_t common_suffix(const std::vector<std::uint32_t>& context, std::size_t end, const At& at, std::size_t total,
    std::uint32_t cap) noexcept {
    std::uint32_t m = 0;
    while (m < cap && m <= end && m < total && context[end - m] == at(total - 1 - m)) ++m;
    return m;
}
}  // namespace

MtpNgramDraft::MtpNgramDraft(MtpNgramDraftConfig config)
    : config_(config), index_(std::max<std::uint32_t>(config.min_suffix, 1U)) {
    if (config_.min_suffix < 1 || config_.max_suffix < config_.min_suffix || config_.max_suffix > 8 ||
        !config_.max_rows || config_.prior <= 0.0 || config_.prior >= 1.0 || config_.prior_weight <= 0.0)
        throw std::invalid_argument("MTP n-gram draft configuration is out of range");
    reset({});
}

std::size_t MtpNgramDraft::cell(std::uint32_t suffix, bool supported, std::size_t row) noexcept {
    const std::size_t s = suffix <= 2 ? 0 : (suffix == 3 ? 1 : 2);
    const std::size_t r = row == 0 ? 0 : (row < 3 ? 1 : 2);
    return (s * 2 + (supported ? 1 : 0)) * kRowCells + r;
}

void MtpNgramDraft::reset(std::span<const std::uint32_t> prompt) {
    context_.clear();
    index_.clear();
    right_.assign(kPeerFirst + kRowCells, config_.prior * config_.prior_weight);
    tried_.assign(kPeerFirst + kRowCells, config_.prior_weight);
    started_ = false;
    shadow_ = {};
    stats_ = {};
    context_.reserve(prompt.size() + 1024);
    index_.reserve(prompt.size() + 1024);
    for (const auto t : prompt) push(t);
}

void MtpNgramDraft::push(std::uint32_t token) {
    context_.push_back(token);
    index_.push(context_);
}

double MtpNgramDraft::acceptance(std::uint32_t suffix, bool supported, std::size_t row) const {
    const auto c = cell(suffix, supported, row);
    return right_[c] / tried_[c];
}

double MtpNgramDraft::peer_acceptance(std::size_t row) const {
    const auto c = kPeerFirst + (row == 0 ? 0 : (row < 3 ? 1 : 2));
    return right_[c] / tried_[c];
}

std::size_t MtpNgramDraft::shadow_cell(std::size_t row) const noexcept {
    return shadow_.peer ? kPeerFirst + (row == 0 ? 0 : (row < 3 ? 1 : 2)) : cell(shadow_.suffix, shadow_.supported, row);
}

std::vector<std::uint32_t> MtpNgramDraft::peer_lookup(std::span<const std::uint32_t> extra, std::size_t length) const {
    const auto total = context_.size() + extra.size();
    const auto at = [&](std::size_t i) { return i < context_.size() ? context_[i] : extra[i - context_.size()]; };
    std::vector<std::uint32_t> key;
    const auto lowest = std::max(config_.peer_min_suffix, config_.min_suffix);
    for (std::uint32_t n = config_.max_suffix; n >= lowest; --n) {
        if (total < n) continue;
        for (const auto* peer : peers_) {
            if (!peer || peer == this || peer->config_.min_suffix > n || peer->config_.max_suffix < n) continue;
            // The peer's most recent end of an exact n-token match with a continuation: its key candidates, measured.
            const auto& pc = peer->context_;
            key.assign(peer->config_.min_suffix, 0U);
            for (std::size_t i = 0; i < key.size(); ++i) key[i] = at(total - key.size() + i);
            std::size_t chosen = SIZE_MAX;
            peer->index_.for_each(key, [&](std::size_t end) {
                if (end + 1 >= pc.size()) return true;  // no continuation yet
                if (common_suffix(pc, end, at, total, n) < n) return true;
                chosen = end;
                return false;
            });
            if (chosen == SIZE_MAX) continue;
            std::vector<std::uint32_t> out;
            for (std::size_t i = chosen + 1; i < pc.size() && out.size() < length; ++i) out.push_back(pc[i]);
            return out;
        }
    }
    return {};
}

std::vector<std::uint32_t> MtpNgramDraft::lookup(std::span<const std::uint32_t> extra, std::size_t length,
    std::uint32_t& suffix, std::uint32_t& support) const {
    suffix = 0; support = 0;
    const auto total = context_.size() + extra.size();
    const auto at = [&](std::size_t i) { return i < context_.size() ? context_[i] : extra[i - context_.size()]; };
    const std::uint32_t min_n = config_.min_suffix;
    if (total < min_n) return {};
    std::vector<std::uint32_t> key(min_n);
    for (std::size_t i = 0; i < min_n; ++i) key[i] = at(total - min_n + i);
    // The longest exact suffix n (min_suffix..max_suffix) with an earlier occurrence that is not the suffix itself:
    // the candidates of its min_suffix-token key (most recent first), each measured.
    std::vector<std::pair<std::size_t, std::uint32_t>> found;  // (end, exact common suffix)
    std::uint32_t n = 0;
    index_.for_each(key, [&](std::size_t end) {
        if (end + 1 >= total) return true;
        const auto m = common_suffix(context_, end, at, total, config_.max_suffix);
        if (m >= min_n) { found.emplace_back(end, m); n = std::max(n, m); }
        return true;
    });
    // Its most recent occurrence supplies the continuation; support counts the occurrences continuing alike.
    std::size_t chosen = SIZE_MAX;
    std::uint32_t count = 0;
    std::uint32_t first = 0;
    for (const auto& [end, m] : found) {
        if (m < n) continue;
        if (chosen == SIZE_MAX) { chosen = end; first = at(end + 1); }
        if (at(end + 1) == first) ++count;
    }
    if (chosen == SIZE_MAX) return {};
    std::vector<std::uint32_t> out;
    for (std::size_t i = chosen + 1; i < total && out.size() < length; ++i) out.push_back(at(i));
    suffix = n; support = count;
    return out;
}

std::vector<std::uint32_t> MtpNgramDraft::begin_cycle(std::uint32_t pending, std::span<const std::uint32_t> drafts,
    std::size_t confident, std::size_t room, std::vector<double>& acceptance_out) {
    acceptance_out.clear();
    if (!started_) { push(pending); started_ = true; }
    shadow_ = {};
    confident = std::min(confident, drafts.size());
    // Only a chain that stopped before the window could be filled gets an extension.
    if (!confident || confident >= room) return {};
    std::uint32_t suffix = 0, support = 0;
    const auto prefix = drafts.first(confident);
    const auto length = std::min<std::size_t>(config_.max_rows, room - confident);
    auto tokens = lookup(prefix, length, suffix, support);
    bool peer = false;
    if (tokens.empty() && !peers_.empty()) { tokens = peer_lookup(prefix, length); peer = !tokens.empty(); }
    if (tokens.empty()) return {};
    shadow_.valid = true;
    shadow_.prefix.assign(prefix.begin(), prefix.end());
    shadow_.tokens = tokens;
    shadow_.suffix = suffix;
    shadow_.supported = support >= 2;
    shadow_.peer = peer;
    ++stats_.cycles;
    if (peer) ++stats_.peer_cycles;
    const auto row_acceptance = [&](std::size_t r) { return peer ? peer_acceptance(r) : acceptance(suffix, shadow_.supported, r); };
    if (row_acceptance(0) < config_.floor) return {};
    for (std::size_t r = 0; r < tokens.size(); ++r) acceptance_out.push_back(row_acceptance(r));
    return tokens;
}

void MtpNgramDraft::used(std::size_t rows) {
    if (!shadow_.valid || !rows) return;
    shadow_.used = std::min(rows, shadow_.tokens.size());
    ++stats_.extended_cycles;
    stats_.extension_rows += shadow_.used;
    if (shadow_.peer) stats_.peer_extension_rows += shadow_.used;
}

void MtpNgramDraft::finish_cycle(std::span<const std::uint32_t> emitted) {
    if (shadow_.valid) {
        // The extension was in play only if every confident draft it continues was emitted as drafted.
        const auto p = shadow_.prefix.size();
        if (emitted.size() > p && std::equal(shadow_.prefix.begin(), shadow_.prefix.end(), emitted.begin())) {
            const auto truth = emitted.subspan(p);
            ++stats_.scored_cycles;
            for (std::size_t k = 0; k < shadow_.tokens.size() && k < truth.size(); ++k) {
                const auto c = shadow_cell(k);
                const bool right = shadow_.tokens[k] == truth[k];
                tried_[c] += 1.0;
                if (right) { right_[c] += 1.0; ++stats_.shadow_right_rows; }
                if (right && k < shadow_.used) { ++stats_.accepted_rows; if (shadow_.peer) ++stats_.peer_accepted_rows; }
                if (!right) break;
            }
        }
        shadow_ = {};
    }
    for (const auto t : emitted) push(t);
}

}  // namespace strix
