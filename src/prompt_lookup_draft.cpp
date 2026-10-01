#include "strix/prompt_lookup_draft.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t build_identity(
    const PromptLookupDraftConfig& config,
    std::span<const std::uint32_t> initial_history) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, config.ngram_size);
    hash_u64(hash, config.history_capacity_tokens);
    hash_u64(hash, initial_history.size());
    for (const auto token : initial_history) {
        hash_u64(hash, token);
    }
    return hash;
}

}  // namespace

PromptLookupDraftBackend::PromptLookupDraftBackend(
    PromptLookupDraftConfig config,
    std::span<const std::uint32_t> initial_history)
    : config_(config),
      history_(
          initial_history.begin(),
          initial_history.end()) {
    if (config_.ngram_size == 0) {
        throw std::invalid_argument(
            "prompt lookup n-gram size must be non-zero");
    }
    if (config_.history_capacity_tokens <=
        static_cast<std::uint64_t>(
            config_.ngram_size)) {
        throw std::invalid_argument(
            "prompt lookup history capacity must exceed n-gram size");
    }
    trim_history();
    proposal_identity_ =
        build_identity(
            config_,
            history_);
    if (proposal_identity_ == 0) {
        throw std::logic_error(
            "prompt lookup proposal identity must be non-zero");
    }
}

DraftProposalStateContext
PromptLookupDraftBackend::make_state(
    const SerialSessionContext& target_session) const {
    return make_draft_proposal_state(
        proposal_identity_,
        target_session);
}

void PromptLookupDraftBackend::validate_state(
    const DraftProposalStateContext& state,
    DraftProposalStatePhase expected_phase) const {
    if (state.proposal_identity !=
            proposal_identity_ ||
        state.phase !=
            expected_phase) {
        throw std::invalid_argument(
            "prompt lookup draft state identity or phase mismatch");
    }
}

std::vector<std::uint32_t>
PromptLookupDraftBackend::propose_candidates(
    const SerialSessionContext&,
    const DraftProposalStateContext& proposal_state,
    std::uint32_t max_candidate_depth) {
    validate_state(
        proposal_state,
        DraftProposalStatePhase::proposal_active);

    if (max_candidate_depth == 0 ||
        history_.size() <=
            static_cast<std::size_t>(
                config_.ngram_size)) {
        return {};
    }

    const auto n =
        static_cast<std::size_t>(
            config_.ngram_size);
    const auto suffix_begin =
        history_.size() - n;

    std::size_t match_begin =
        history_.size();
    for (std::size_t candidate =
             suffix_begin;
         candidate-- > 0;) {
        if (candidate + n >
            suffix_begin) {
            continue;
        }
        if (std::equal(
                history_.begin() +
                    static_cast<std::ptrdiff_t>(
                        candidate),
                history_.begin() +
                    static_cast<std::ptrdiff_t>(
                        candidate + n),
                history_.begin() +
                    static_cast<std::ptrdiff_t>(
                        suffix_begin))) {
            match_begin = candidate;
            break;
        }
    }

    if (match_begin ==
        history_.size()) {
        return {};
    }

    const auto continuation_begin =
        match_begin + n;
    const auto available =
        history_.size() -
        continuation_begin;
    const auto count =
        std::min<std::size_t>(
            available,
            static_cast<std::size_t>(
                max_candidate_depth));

    return {
        history_.begin() +
            static_cast<std::ptrdiff_t>(
                continuation_begin),
        history_.begin() +
            static_cast<std::ptrdiff_t>(
                continuation_begin + count),
    };
}

void PromptLookupDraftBackend::reconcile_after_target(
    const DraftProposalStateContext& proposal_state,
    const SerialSessionContext&,
    std::span<const std::uint32_t> emitted_tokens) {
    validate_state(
        proposal_state,
        DraftProposalStatePhase::reconciliation_required);

    if (emitted_tokens.size() >
        std::numeric_limits<std::size_t>::max() -
            history_.size()) {
        throw std::overflow_error(
            "prompt lookup history size overflow");
    }
    history_.insert(
        history_.end(),
        emitted_tokens.begin(),
        emitted_tokens.end());
    trim_history();
}

void PromptLookupDraftBackend::trim_history() {
    const auto capacity =
        static_cast<std::size_t>(
            config_.history_capacity_tokens);
    if (history_.size() <=
        capacity) {
        return;
    }

    history_.erase(
        history_.begin(),
        history_.begin() +
            static_cast<std::ptrdiff_t>(
                history_.size() - capacity));
}

}  // namespace strix
