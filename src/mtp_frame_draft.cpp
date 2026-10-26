#include "strix/mtp_frame_draft.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix {
namespace {
constexpr std::size_t kRowCells = 3;  // row 0, 1-2, 3+
std::size_t row_cell(std::size_t row) noexcept { return row == 0 ? 0 : (row < 3 ? 1 : 2); }
}  // namespace

MtpFrameDraft::MtpFrameDraft(MtpFrameDraftConfig config) : config_(config) {
    if (config_.open == config_.close || !config_.max_anchor || !config_.max_rows || config_.prior <= 0.0 ||
        config_.prior >= 1.0 || config_.single_prior <= 0.0 || config_.single_prior >= 1.0 || config_.prior_weight <= 0.0)
        throw std::invalid_argument("MTP frame draft configuration is out of range");
    reset({});
}

void MtpFrameDraft::reset(std::span<const std::uint32_t> prompt) {
    context_.clear();
    frames_.clear();
    open_at_ = SIZE_MAX;
    // cells [0, kRowCells): single earlier frame; [kRowCells, 2 kRowCells): confirmed by two or more
    right_.assign(2 * kRowCells, config_.prior * config_.prior_weight);
    for (std::size_t c = 0; c < kRowCells; ++c) right_[c] = config_.single_prior * config_.prior_weight;
    tried_.assign(2 * kRowCells, config_.prior_weight);
    started_ = false;
    shadow_ = {};
    stats_ = {};
    context_.reserve(prompt.size() + 1024);
    for (const auto t : prompt) push(t);
}

void MtpFrameDraft::push(std::uint32_t token) {
    context_.push_back(token);
    const auto at = context_.size() - 1;
    if (token == config_.open) open_at_ = at;
    else if (token == config_.close && open_at_ != SIZE_MAX) { frames_.push_back({open_at_, at}); open_at_ = SIZE_MAX; }
}

double MtpFrameDraft::acceptance(std::size_t row) const {
    // Reported for the multi-frame (supported) cells; begin_cycle uses the proposal's own group.
    const auto c = kRowCells + row_cell(row);
    return right_[c] / tried_[c];
}

std::vector<std::uint32_t> MtpFrameDraft::lookup(std::span<const std::uint32_t> extra, std::size_t length, bool& ambiguous,
    std::size_t& support) const {
    ambiguous = false;
    support = 0;
    const auto total = context_.size() + extra.size();
    const auto at = [&](std::size_t i) { return i < context_.size() ? context_[i] : extra[i - context_.size()]; };
    // The current frame: the last open token not followed by a close token.
    std::size_t start = open_at_;
    for (std::size_t i = 0; i < extra.size(); ++i) {
        if (extra[i] == config_.open) start = context_.size() + i;
        else if (extra[i] == config_.close) start = SIZE_MAX;
    }
    if (start == SIZE_MAX || frames_.empty() || !length) return {};
    const std::size_t have = total - start;
    for (std::size_t n = std::min<std::size_t>(have, config_.max_anchor); n >= 1; --n) {
        std::vector<std::vector<std::uint32_t>> found;
        for (const auto& [a, b] : frames_) {
            // An anchor occurrence ending at j (inside the frame, before its close token) continues with j+1..b.
            for (std::size_t j = a + n - 1; j < b; ++j) {
                bool same = true;
                for (std::size_t q = 0; q < n && same; ++q) same = context_[j + 1 - n + q] == at(total - n + q);
                if (!same) continue;
                std::vector<std::uint32_t> c;
                for (std::size_t i = j + 1; i <= b && c.size() < length; ++i) c.push_back(context_[i]);
                found.push_back(std::move(c));
            }
        }
        if (found.empty()) continue;
        // Strict: the common prefix of every continuation.
        std::vector<std::uint32_t> out;
        for (std::size_t k = 0;; ++k) {
            if (k >= found.front().size()) break;
            const auto t = found.front()[k];
            if (!std::all_of(found.begin(), found.end(), [&](const auto& c) { return k < c.size() && c[k] == t; })) break;
            out.push_back(t);
        }
        if (out.empty()) ambiguous = true;
        support = found.size();
        return out;
    }
    return {};
}

std::vector<std::uint32_t> MtpFrameDraft::begin_cycle(std::uint32_t pending, std::span<const std::uint32_t> drafts,
    std::size_t confident, std::size_t room, std::vector<double>& acceptance_out) {
    acceptance_out.clear();
    if (!started_) { push(pending); started_ = true; }
    shadow_ = {};
    if (open_at_ != SIZE_MAX) ++stats_.frame_cycles;
    confident = std::min(confident, drafts.size());
    if (confident >= room) return {};
    const auto prefix = drafts.first(confident);
    bool ambiguous = false;
    std::size_t support = 0;
    auto tokens = lookup(prefix, std::min<std::size_t>(config_.max_rows, room - confident), ambiguous, support);
    if (ambiguous) ++stats_.ambiguous;
    if (tokens.empty()) return {};
    const bool supported = support >= 2;
    shadow_.valid = true;
    shadow_.prefix.assign(prefix.begin(), prefix.end());
    shadow_.tokens = tokens;
    shadow_.supported = supported;
    ++stats_.cycles;
    const auto group = supported ? kRowCells : 0;
    if (right_[group] / tried_[group] < config_.floor) return {};
    for (std::size_t r = 0; r < tokens.size(); ++r) {
        const auto c = group + row_cell(r);
        acceptance_out.push_back(right_[c] / tried_[c]);
    }
    return tokens;
}

void MtpFrameDraft::used(std::size_t rows) {
    if (!shadow_.valid || !rows) return;
    shadow_.used = std::min(rows, shadow_.tokens.size());
    ++stats_.extended_cycles;
    stats_.extension_rows += shadow_.used;
}

void MtpFrameDraft::finish_cycle(std::span<const std::uint32_t> emitted) {
    if (shadow_.valid) {
        const bool supported = shadow_.supported;
        const std::size_t used = shadow_.used;
        const auto p = shadow_.prefix.size();
        if (emitted.size() > p && std::equal(shadow_.prefix.begin(), shadow_.prefix.end(), emitted.begin())) {
            const auto truth = emitted.subspan(p);
            ++stats_.scored_cycles;
            const auto group = supported ? kRowCells : 0;
            for (std::size_t k = 0; k < shadow_.tokens.size() && k < truth.size(); ++k) {
                const auto c = group + row_cell(k);
                const bool right = shadow_.tokens[k] == truth[k];
                tried_[c] += 1.0;
                if (right) { right_[c] += 1.0; ++stats_.shadow_right_rows; }
                if (right && k < used) ++stats_.accepted_rows;
                if (!right) break;
            }
        }
        shadow_ = {};
    }
    for (const auto t : emitted) push(t);
}

}  // namespace strix
