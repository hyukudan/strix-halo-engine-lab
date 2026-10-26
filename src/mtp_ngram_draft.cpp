#include "strix/mtp_ngram_draft.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix {
namespace {
constexpr std::size_t kSuffixCells = 3;  // suffix 2, 3, 4+
constexpr std::size_t kRowCells = 3;     // row 0, 1-2, 3+
constexpr std::size_t kPeerFirst = kSuffixCells * 2 * kRowCells; // then one row-bucketed group for peer proposals

std::uint64_t mix(std::uint64_t h, std::uint32_t t) noexcept {
    h ^= t + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h * 0xff51afd7ed558ccdULL;
}
std::uint64_t hash_of(const std::uint32_t* begin, std::size_t n) noexcept {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < n; ++i) h = mix(h, begin[i]);
    return h;
}
}  // namespace

MtpNgramDraft::MtpNgramDraft(MtpNgramDraftConfig config) : config_(config) {
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
    for (const auto t : prompt) push(t);
}

void MtpNgramDraft::push(std::uint32_t token) {
    context_.push_back(token);
    const auto end = context_.size() - 1;
    for (std::uint32_t n = config_.min_suffix; n <= config_.max_suffix && n <= context_.size(); ++n)
        index_[Key{hash_of(&context_[end + 1 - n], n), n}].push_back(static_cast<std::uint32_t>(end));
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
        key.clear();
        for (std::size_t i = total - n; i < total; ++i) key.push_back(at(i));
        const Key k{hash_of(key.data(), n), n};
        for (const auto* peer : peers_) {
            if (!peer || peer == this || peer->config_.min_suffix > n || peer->config_.max_suffix < n) continue;
            const auto hit = peer->index_.find(k);
            if (hit == peer->index_.end()) continue;
            const auto& pc = peer->context_;
            for (auto e = hit->second.rbegin(); e != hit->second.rend(); ++e) {
                const std::size_t end = *e;
                if (end + 1 >= pc.size()) continue;  // no continuation yet
                bool same = true;
                for (std::uint32_t q = 0; q < n && same; ++q) same = pc[end + 1 - n + q] == key[q];
                if (!same) continue;
                std::vector<std::uint32_t> out;
                for (std::size_t i = end + 1; i < pc.size() && out.size() < length; ++i) out.push_back(pc[i]);
                return out;
            }
        }
    }
    return {};
}

std::vector<std::uint32_t> MtpNgramDraft::lookup(std::span<const std::uint32_t> extra, std::size_t length,
    std::uint32_t& suffix, std::uint32_t& support) const {
    suffix = 0; support = 0;
    const auto total = context_.size() + extra.size();
    const auto at = [&](std::size_t i) { return i < context_.size() ? context_[i] : extra[i - context_.size()]; };
    std::vector<std::uint32_t> key;
    for (std::uint32_t n = config_.max_suffix; n >= config_.min_suffix; --n) {
        if (total < n) continue;
        key.clear();
        for (std::size_t i = total - n; i < total; ++i) key.push_back(at(i));
        const auto hit = index_.find(Key{hash_of(key.data(), n), n});
        if (hit == index_.end()) continue;
        // Most recent earlier occurrence that really matches (hash collisions are checked) and is not the
        // suffix itself.
        const auto& ends = hit->second;
        std::size_t chosen = SIZE_MAX;
        std::uint32_t count = 0;
        std::uint32_t first = 0;
        for (auto e = ends.rbegin(); e != ends.rend(); ++e) {
            const std::size_t end = *e;
            if (end + 1 >= total) continue;
            bool same = true;
            for (std::uint32_t k = 0; k < n && same; ++k) same = context_[end + 1 - n + k] == key[k];
            if (!same) continue;
            if (chosen == SIZE_MAX) { chosen = end; first = at(end + 1); }
            if (at(end + 1) == first) ++count;
        }
        if (chosen == SIZE_MAX) continue;
        std::vector<std::uint32_t> out;
        for (std::size_t i = chosen + 1; i < total && out.size() < length; ++i) out.push_back(at(i));
        suffix = n; support = count;
        return out;
    }
    return {};
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
