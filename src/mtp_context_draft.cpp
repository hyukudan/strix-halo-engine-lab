#include "strix/mtp_context_draft.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {
constexpr std::size_t kSuffixCells = 4; // suffix 2, 3, 4-7, 8+
constexpr std::size_t kRowCells = 3;    // row 0, 1-2, 3+
std::uint64_t hash_of(const std::uint32_t* begin, std::size_t n) noexcept {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= begin[i] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h *= 0xff51afd7ed558ccdULL;
    }
    return h;
}
}
MtpContextDraft::MtpContextDraft(MtpContextDraftConfig config) : config_(config) {
    if (config_.min_suffix < 1 || config_.max_suffix < config_.min_suffix || config_.max_suffix > 32 ||
        !config_.max_rows || config_.max_rows > 7 || !std::isfinite(config_.prior) ||
        !(config_.prior > 0.0 && config_.prior < 1.0) || !std::isfinite(config_.prior_weight) ||
        config_.prior_weight <= 0.0 || !std::isfinite(config_.floor) || config_.floor < 0.0 || config_.floor > 1.0 || config_.wide_anchor > config_.max_suffix ||
        (config_.chain_fresh && (config_.chain_fresh < 3 || !config_.chain_resume || config_.chain_resume > config_.chain_fresh ||
            config_.min_suffix > 3 || config_.max_suffix < 3)))
        throw std::invalid_argument("MTP n-gram draft configuration is out of range");
    reset({});
}
std::size_t MtpContextDraft::cell(std::uint32_t suffix, bool unanimous, std::size_t row) noexcept {
    const std::size_t s = suffix <= 2 ? 0 : (suffix == 3 ? 1 : (suffix < 8 ? 2 : 3));
    const std::size_t r = row == 0 ? 0 : (row < 3 ? 1 : 2);
    return (s * 2 + (unanimous ? 1 : 0)) * kRowCells + r;
}
void MtpContextDraft::reset(std::span<const std::uint32_t> prompt) {
    context_.clear(); index_.clear();
    right_.assign(kSuffixCells * 2 * kRowCells, config_.prior * config_.prior_weight);
    tried_.assign(kSuffixCells * 2 * kRowCells, config_.prior_weight);
    started_ = false; wide_ = false; selected_rows_ = 0; shadow_ = {}; pending_probe_ = {}; stats_ = {}; edited_.clear();
    chain_kept_ = chain_rows_ = chain_source_ = chain_next_ = 0; chain_open_ = chain_established_ = false;
    chain_right_ = 0.9; chain_tried_ = 1.0;
    context_.reserve(prompt.size() + 1024);
    for (const auto t : prompt) push(t);
}
void MtpContextDraft::push(std::uint32_t token) {
    if (context_.size() >= std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("MTP n-gram context exceeds index capacity");
    context_.push_back(token);
    const auto end = context_.size() - 1;
    for (std::uint32_t n = config_.min_suffix; n <= config_.max_suffix && n <= context_.size(); ++n)
        index_[Key{hash_of(&context_[end + 1 - n], n), n}].push_back(static_cast<std::uint32_t>(end));
}
double MtpContextDraft::acceptance(std::uint32_t suffix, bool unanimous, std::size_t row) const {
    const auto c = cell(suffix, unanimous, row);
    return right_[c] / tried_[c];
}
std::vector<std::uint32_t> MtpContextDraft::lookup(std::span<const std::uint32_t> extra, std::size_t length,
    std::uint32_t& suffix, std::uint32_t& support, std::uint32_t* occurrences) const {
    suffix = 0; support = 0;
    if (occurrences) *occurrences = 0;
    if (!length) return {};
    const auto total = context_.size() + extra.size();
    const auto at = [&](std::size_t i) { return i < context_.size() ? context_[i] : extra[i - context_.size()]; };
    std::vector<std::uint32_t> key;
    for (std::uint32_t n = config_.max_suffix; n >= config_.min_suffix; --n) {
        if (total < n) continue;
        key.clear();
        for (std::size_t i = total - n; i < total; ++i) key.push_back(at(i));
        const auto hit = index_.find(Key{hash_of(key.data(), n), n});
        if (hit == index_.end()) continue;
        std::vector<std::size_t> positions;
        for (const auto end : hit->second) {
            if (static_cast<std::size_t>(end) + 1 >= context_.size()) continue;
            if (std::equal(key.begin(), key.end(), context_.begin() + (end + 1 - n)))
                positions.push_back(static_cast<std::size_t>(end) + 1);
        }
        if (positions.empty()) continue;
        if (occurrences) *occurrences = static_cast<std::uint32_t>(positions.size());
        std::vector<std::uint32_t> out;
        // Vote at each step among spans that agreed with every previous winning token.
        // Ties use the latest occurrence, then token id; unordered-map iteration is immaterial.
        while (!positions.empty() && out.size() < length) {
            struct Vote { std::size_t count = 0, latest = 0; };
            std::unordered_map<std::uint32_t, Vote> votes;
            for (const auto pos : positions) { auto& v = votes[context_[pos]]; ++v.count; v.latest = std::max(v.latest, pos); }
            std::uint32_t winner = 0; Vote best;
            for (const auto& [token, v] : votes)
                if (v.count > best.count || (v.count == best.count && (v.latest > best.latest ||
                    (v.latest == best.latest && token < winner)))) { winner = token; best = v; }
            if (out.empty()) support = static_cast<std::uint32_t>(best.count);
            out.push_back(winner);
            positions.erase(std::remove_if(positions.begin(), positions.end(), [&](std::size_t pos) {
                return context_[pos] != winner || pos + 1 >= context_.size();
            }), positions.end());
            for (auto& pos : positions) ++pos;
        }
        suffix = n;
        return out;
    }
    return {};
}
std::vector<std::uint32_t> MtpContextDraft::begin_copy_cycle(std::uint32_t pending, std::size_t room,
    std::vector<double>& acceptance_out, double min_probability, double mtp_expected) {
    acceptance_out.clear();
    if (!config_.copy_edit || !room) return {};
    if (!std::isfinite(min_probability) || min_probability < 0.0 || min_probability > 1.0)
        throw std::invalid_argument("MTP n-gram minimum probability is out of range");
    if (!started_) { push(pending); started_ = true; }
    std::uint32_t suffix = 0, support = 0, occurrences = 0;
    const auto anchor = lookup({}, 1, suffix, support, &occurrences);
    if (anchor.empty() || suffix < config_.copy_anchor || support != occurrences ||
        acceptance(suffix, true, 0) < std::max(config_.floor, min_probability))
        return {};
    const auto saved = stats_;
    auto tokens = begin_cycle(pending, {}, 0, room, acceptance_out, min_probability);
    if (!tokens.empty() && mtp_expected >= 0.0) {
        double expected = 0.0, run = 1.0;
        for (const auto p : acceptance_out) { run *= p; expected += run; }
        if (expected <= mtp_expected) {
            // Not a proposal: MTP drafts this cycle and begin_cycle() may still extend or replace it.
            stats_ = saved; ++stats_.copy_first_declined;
            shadow_ = {}; pending_probe_ = {}; selected_rows_ = 0; acceptance_out.clear();
            return {};
        }
    }
    if (!tokens.empty()) ++stats_.copy_first_cycles;
    return tokens;
}
std::vector<std::uint32_t> MtpContextDraft::begin_cycle(std::uint32_t pending, std::span<const std::uint32_t> drafts,
    std::size_t confident, std::size_t room, std::vector<double>& acceptance_out, double min_probability,
    double kept_expected) {
    if (!std::isfinite(min_probability) || min_probability < 0.0 || min_probability > 1.0)
        throw std::invalid_argument("MTP n-gram minimum probability is out of range");
    acceptance_out.clear(); shadow_ = {}; pending_probe_ = {}; selected_rows_ = 0; wide_ = false;
    if (!started_) { push(pending); started_ = true; }
    confident = std::min({confident, drafts.size(), room});
    const auto probe = [&](std::size_t kept) {
        Shadow s; s.prefix.assign(drafts.begin(), drafts.begin() + static_cast<std::ptrdiff_t>(kept));
        std::uint32_t support = 0, occurrences = 0;
        if (kept < room) s.tokens = lookup(s.prefix, std::min<std::size_t>(config_.max_rows, room - kept), s.suffix, support,
            &occurrences);
        s.unanimous = !s.tokens.empty() && support == occurrences;
        if (config_.copy_edit && !s.tokens.empty()) {
            const auto previous = s.prefix.empty() ? context_.back() : s.prefix.back();
            const auto edit = first_edit(previous, s.tokens);
            if (edit < s.tokens.size()) { s.tokens.resize(edit); ++stats_.edit_cuts; }
        }
        return s;
    };
    const double gate = std::max(config_.floor, min_probability);
    // Expected accepted rows of a proposal: sum over rows of the product of the learned acceptances.
    const auto expected = [&](const Shadow& s) {
        double e = 0.0, run = 1.0;
        for (std::size_t r = 0; r < s.tokens.size(); ++r) {
            const auto p = acceptance(s.suffix, s.unanimous, r);
            if (p < gate) break;
            run *= p; e += run;
        }
        return e;
    };
    shadow_ = probe(confident);
    if (confident) pending_probe_ = probe(0);
    // Copy/edit can restart at a long exact span even when MTP filled its entire chain, when the span is
    // expected to deliver more accepted rows than the confident prefix plus its continuation.
    // Without copy/edit, pending-only proposals are used when no draft is confident.
    if (config_.copy_edit && !pending_probe_.tokens.empty() && pending_probe_.suffix >= config_.copy_anchor &&
        pending_probe_.unanimous &&
        (shadow_.tokens.empty() || pending_probe_.suffix > shadow_.suffix) &&
        (kept_expected < 0.0 || expected(pending_probe_) > kept_expected + expected(shadow_))) {
        if (confident) ++stats_.replaced_cycles;
        std::swap(shadow_, pending_probe_);
        // Keep only the independent pending-only probe; the displaced prefix is not scored twice.
        pending_probe_ = {};
    }
    if (shadow_.tokens.empty()) return {};
    ++stats_.cycles;
    for (std::size_t r = 0; r < shadow_.tokens.size(); ++r) {
        const auto p = acceptance(shadow_.suffix, shadow_.unanimous, r);
        if (p < gate) break;
        acceptance_out.push_back(p);
    }
    if (acceptance_out.empty()) return {};
    selected_rows_ = acceptance_out.size(); ++stats_.selected_cycles; stats_.selected_rows += selected_rows_;
    return {shadow_.tokens.begin(), shadow_.tokens.begin() + static_cast<std::ptrdiff_t>(acceptance_out.size())};
}
std::vector<std::uint32_t> MtpContextDraft::propose_wide(std::uint32_t pending, std::size_t rows, double min_acceptance) {
    if (!std::isfinite(min_acceptance) || min_acceptance < 0.0 || min_acceptance > 1.0)
        throw std::invalid_argument("MTP wide copy minimum acceptance is out of range");
    shadow_ = {}; pending_probe_ = {}; selected_rows_ = 0; wide_ = false;
    if (!config_.copy_edit || !rows) return {};
    if (!started_) { push(pending); started_ = true; }
    std::uint32_t suffix = 0, support = 0, occurrences = 0;
    auto tokens = lookup({}, rows, suffix, support, &occurrences);
    if (tokens.size() != rows || suffix < std::max(config_.copy_anchor, config_.wide_anchor) || support != occurrences) return {};
    if (first_edit(context_.back(), tokens) < tokens.size()) {
        ++stats_.wide_blocked;
        return {};
    }
    for (const std::size_t row : {std::size_t{0}, std::size_t{1}, std::size_t{3}})
        if (acceptance(suffix, true, row) < min_acceptance) return {};
    shadow_.tokens = tokens; shadow_.suffix = suffix; shadow_.unanimous = true; wide_ = true;
    return tokens;
}
std::size_t MtpContextDraft::chain_match(std::span<const std::uint32_t> extra, std::size_t& source) const {
    // S = history (ending at the pending token) ++ extra. Candidates end where the last 3 tokens of S occur.
    source = 0;
    const auto total = context_.size() + extra.size();
    if (total < 4) return 0;
    const auto at = [&](std::size_t i) { return i < context_.size() ? context_[i] : extra[i - context_.size()]; };
    const std::uint32_t key[3] = {at(total - 3), at(total - 2), at(total - 1)};
    const auto hit = index_.find(Key{hash_of(key, 3), 3});
    if (hit == index_.end()) return 0;
    constexpr std::size_t kMaxMatch = 64;
    std::size_t best_end = 0, best_match = 0;
    for (const auto end32 : hit->second) {
        const auto end = static_cast<std::size_t>(end32);
        if (end + 1 >= total) continue; // the tail itself
        std::size_t m = 0;
        while (m < kMaxMatch && m <= end && at(end - m) == at(total - 1 - m)) ++m;
        if (m > best_match || (m == best_match && end > best_end)) { best_match = m; best_end = end; }
    }
    source = best_end + 1;
    return best_match;
}
std::vector<std::uint32_t> MtpContextDraft::chain_copy(std::span<const std::uint32_t> extra, std::size_t rows, bool resume_only) {
    const auto total = context_.size() + extra.size();
    if (!rows || total < 4) return {};
    const auto at = [&](std::size_t i) { return i < context_.size() ? context_[i] : extra[i - context_.size()]; };
    std::size_t source = 0;
    const auto best_match = chain_match(extra, source);
    // Resumed: the previous copy reached chain_next_ and was interrupted; its source continues within 256 tokens.
    const bool resumed = chain_next_ && source >= chain_next_ && source <= chain_next_ + 256;
    if (best_match < (resumed ? config_.chain_resume : config_.chain_fresh) || (resume_only && !resumed)) return {};
    std::vector<std::uint32_t> out;
    for (std::size_t r = 0; r < rows; ++r) {
        const auto i = source + r;
        out.push_back(i < total ? at(i) : out[i - total]); // self-repetition reads the copy's own rows
    }
    const auto edit = first_edit(at(total - 1), out);
    if (edit < out.size()) { out.resize(edit); ++stats_.edit_cuts; }
    if (!out.empty()) chain_source_ = source;
    return out;
}
std::vector<std::uint32_t> MtpContextDraft::chain_extend(std::uint32_t pending, std::span<const std::uint32_t> kept,
    std::size_t rows, std::vector<double>& acceptance_out) {
    acceptance_out.clear(); chain_open_ = false;
    if (!config_.chain_fresh) return {};
    if (!started_) { push(pending); started_ = true; }
    auto out = chain_copy(kept, rows, false);
    if (out.empty()) return out;
    chain_open_ = true; chain_kept_ = kept.size(); chain_rows_ = out.size();
    ++stats_.chain_cycles; stats_.chain_rows += out.size();
    acceptance_out.assign(out.size(), chain_right_ / chain_tried_);
    return out;
}
std::vector<std::uint32_t> MtpContextDraft::chain_first(std::uint32_t pending, std::size_t rows, std::vector<double>& acceptance_out) {
    acceptance_out.clear(); chain_open_ = false;
    if (!config_.chain_fresh) return {};
    if (!started_) { push(pending); started_ = true; }
    if (!chain_established_ || chain_right_ / chain_tried_ < config_.chain_first_accept) return {};
    // A match that already reaches the wide anchor belongs to the wide window (31 rows per forward): copy-first
    // there would take 7 rows per cycle and starve the wide cells (tool-echo t37: wide cycles 31 -> 23, +1.6 %).
    if (config_.wide_anchor) {
        std::size_t source = 0;
        if (chain_match({}, source) >= config_.wide_anchor) return {};
    }
    auto out = chain_copy({}, rows, true);
    // Copy-first replaces the predictor only for a whole window: a copy cut short by a known edit leaves the edited
    // tokens to the predictor (json-transform t34: 198 cut copy-first cycles, 4.8 tokens/cycle vs 7.9 drafted).
    if (out.size() < rows) return {};
    chain_open_ = true; chain_kept_ = 0; chain_rows_ = out.size();
    ++stats_.chain_cycles; ++stats_.chain_first_cycles; stats_.chain_rows += out.size();
    acceptance_out.assign(out.size(), chain_right_ / chain_tried_);
    return out;
}
void MtpContextDraft::chain_finish(std::size_t accepted_candidates) {
    if (!chain_open_) return;
    const auto accepted = accepted_candidates > chain_kept_ ? std::min(chain_rows_, accepted_candidates - chain_kept_) : 0;
    stats_.chain_accepted_rows += accepted;
    // Learned per-row acceptance (rows up to and including the first rejected one).
    chain_tried_ += static_cast<double>(std::min(chain_rows_, accepted + 1)); chain_right_ += static_cast<double>(accepted);
    chain_established_ = accepted == chain_rows_;
    chain_next_ = chain_source_ + accepted;
    chain_open_ = false;
}
void MtpContextDraft::finish_cycle(std::span<const std::uint32_t> emitted, std::size_t verified_candidates,
    std::size_t accepted_candidates) {
    if (accepted_candidates > verified_candidates || accepted_candidates > emitted.size())
        throw std::invalid_argument("MTP n-gram verifier counts are inconsistent");
    const bool chain_was_open = chain_open_;
    chain_finish(accepted_candidates);
    // While a copy is established, a cycle without copy rows scores the copy it would have proposed against the
    // emitted tokens (rows up to the first wrong one), so copy-first learns its acceptance on every cycle.
    if (config_.chain_fresh && !chain_was_open && chain_established_ && !emitted.empty()) {
        std::size_t source = 0;
        if (chain_match({}, source) >= config_.chain_resume && source >= chain_next_ && source <= chain_next_ + 256) {
            std::size_t right = 0;
            while (right < emitted.size() && source + right < context_.size() && context_[source + right] == emitted[right]) ++right;
            chain_right_ += static_cast<double>(right);
            chain_tried_ += static_cast<double>(std::min(emitted.size(), right + 1));
        }
    }
    if (wide_) { ++stats_.wide_cycles; stats_.wide_rows_accepted += accepted_candidates; }
    if (selected_rows_) {
        const auto p = shadow_.prefix.size();
        const auto rows = verified_candidates > p ? std::min(selected_rows_, verified_candidates - p) : 0;
        if (rows) { ++stats_.extended_cycles; stats_.extension_rows += rows;
            if (accepted_candidates > p) stats_.accepted_rows += std::min(rows, accepted_candidates - p); }
    }
    score(shadow_, emitted, true); score(pending_probe_, emitted, false);
    shadow_ = {}; pending_probe_ = {}; selected_rows_ = 0; wide_ = false;
    for (const auto t : emitted) push(t);
    // A cycle without copy rows (the predictor drafted, or a wide cycle) still establishes a copy when the output
    // tail now matches the history by chain_fresh tokens: the next cycle continues it copy-first.
    if (config_.chain_fresh && !chain_was_open) {
        std::size_t source = 0;
        if (chain_match({}, source) >= config_.chain_fresh) {
            chain_established_ = true; chain_next_ = source; ++stats_.chain_established;
        } else chain_established_ = false;
    }
}
void MtpContextDraft::abandon_wide(std::span<const std::uint32_t> truth) {
    if (!wide_) return;
    // Learning only: not a scored primary cycle and nothing pushed (the ordinary cycle commits these tokens).
    ++stats_.wide_abandoned;
    score(shadow_, truth, false);
    shadow_ = {}; pending_probe_ = {}; selected_rows_ = 0; wide_ = false;
}
std::size_t MtpContextDraft::first_edit(std::uint32_t previous, std::span<const std::uint32_t> span) const {
    if (edited_.empty()) return span.size();
    for (std::size_t i = 0; i < span.size(); ++i) {
        if (edited_.count(edit_key(previous, span[i])) != 0) return i;
        previous = span[i];
    }
    return span.size();
}
void MtpContextDraft::score(const Shadow& s, std::span<const std::uint32_t> emitted, bool primary) {
    const auto p = s.prefix.size();
    if (s.tokens.empty() || emitted.size() <= p || !std::equal(s.prefix.begin(), s.prefix.end(), emitted.begin())) return;
    if (primary) ++stats_.scored_cycles;
    const auto truth = emitted.subspan(p);
    for (std::size_t k = 0; k < s.tokens.size() && k < truth.size(); ++k) {
        const auto c = cell(s.suffix, s.unanimous, k);
        const bool right = s.tokens[k] == truth[k];
        tried_[c] += 1.0; ++stats_.learning_rows;
        if (right) { right_[c] += 1.0; if (primary) ++stats_.shadow_right_rows; }
        if (!right) {
            if (config_.copy_edit && s.unanimous && s.suffix >= config_.copy_anchor) {
                const auto previous = k ? s.tokens[k - 1] : (p ? s.prefix.back() : context_.back());
                edited_.insert(edit_key(previous, s.tokens[k]));
            }
            break;
        }
    }
}
} // namespace strix
