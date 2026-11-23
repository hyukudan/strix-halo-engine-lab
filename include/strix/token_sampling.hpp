#pragma once

#include "strix/speculation_rng.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace strix {

// Sampling identity `sampling-v1` (temperature > 0; temperature 0 is greedy and
// never reaches this code). Over the FP32 logits of one row, in this order:
//   1. z = logit / temperature (double); m = max z. Every logit must be finite.
//   1a. z - m is computed as (logit - max logit) / temperature in double.
//   2. Keep z - m >= -40 (the dropped mass is at most vocabulary * e^-40 of the
//      total, about 1e-12 for a 248k vocabulary) and, with min_p > 0,
//      z - m >= ln(min_p) (p >= min_p * p_max).
//   3. top_k > 0: keep the top_k entries by (z desc, token id asc).
//   4. Weights w = exp(z - m); with top_p < 1, keep the shortest prefix in
//      (w desc, token id asc) order whose weight sum reaches top_p of the total.
//   5. Normalize. Inverse-CDF sampling walks the kept entries in their stored
//      order: token id ascending, or (w desc, id asc) when top_p < 1.
// Draft rows use the same transform over the draft head's rows; a draft
// vocabulary subset only restricts the support of q.
struct TokenSamplingPolicy {
    float temperature = 0.0F;
    std::uint32_t top_k = 0;
    float top_p = 1.0F;
    float min_p = 0.0F;
    std::uint64_t seed = 0;
};

[[nodiscard]] bool token_sampling_enabled(const TokenSamplingPolicy&) noexcept;
// Throws unless temperature is finite in [0, 100], top_p in (0, 1], min_p in [0, 1).
void validate_token_sampling_policy(const TokenSamplingPolicy&);
// Non-zero identity of the transform and its parameters (seed excluded).
[[nodiscard]] std::uint64_t token_sampling_policy_identity(const TokenSamplingPolicy&);
// Counter-based random identity: seed, a request identity (1 = the single-request identity every sampler used
// before request identities existed) and the policy identity.
[[nodiscard]] SpeculationRngIdentity token_sampling_rng(const TokenSamplingPolicy&,std::uint64_t request_identity = 1);
// The request identity of a new HTTP request. STRIX_SAMPLING_REQUEST_IDENTITY=counter: 1, 2, 3, ... in admission
// order within one process (the first request keeps identity 1, the command line's); otherwise always 1 (default:
// one seed gives one token stream for every request, as before).
[[nodiscard]] std::uint64_t next_token_sampling_request_identity();
// True when STRIX_SAMPLING_REQUEST_IDENTITY=counter (every request draws its own stream); false for unset, empty,
// 0 or single (one seed gives one token stream for every request). Any other value is rejected.
[[nodiscard]] bool token_sampling_request_identity_counts();

// How a drafted row is decided. Neither choice changes the policy identity or the plain (no-draft) decision.
//   rejection (default): speculative sampling; the draft is drawn with U(n, draft) and accepted with U(n, accept),
//     otherwise the residual is sampled with U(n, residual). Exact in distribution; the tokens of one seed depend
//     on which rows carried drafts.
//   coupled: the row is decided as without a draft, sample(p, U(n, plain)); the draft is accepted when it equals
//     that token. Drafts are drawn with the same U(n, plain) from q (or q's first-max entry below
//     `argmax_draft_below`). The tokens of one seed equal plain sampling's under any draft or window choice.
enum class SampledAcceptance : std::uint8_t {rejection = 0, coupled = 1};
struct TokenSamplerOptions {
    SampledAcceptance acceptance = SampledAcceptance::rejection;
    // coupled only: temperatures below this draft the first-max entry of q (top-k 1); targets always sample.
    float argmax_draft_below = 0.7F;
    std::uint64_t request_identity = 1;
    // Test-only forced rejection (gates; coupled only): in every verified window, the drafted row with this index
    // (the number of drafts accepted before it), when its draft was accepted, is decided as rejected instead: it
    // keeps its token and the rows after it are dropped and decided again in later cycles. The tokens are those of
    // plain sampling either way; this exercises rollback and continuation after a rejection at a chosen depth.
    // forced_rejection_rotate: row = (window index) mod 8 (every depth; at R8 the eighth window forces nothing).
    static constexpr std::uint32_t forced_rejection_off = UINT32_MAX, forced_rejection_rotate = UINT32_MAX - 1U;
    std::uint32_t force_reject_row = forced_rejection_off;
};
// STRIX_MTP_SAMPLED_ACCEPT=rejection|coupled (unset: rejection), STRIX_MTP_SAMPLED_ARGMAX_BELOW=T (default 0.7,
// 0 = always sample drafts), STRIX_MTP_SAMPLED_FORCE_REJECT=0..7|rotate (test-only, unset: off; requires coupled);
// request identity 1.
[[nodiscard]] TokenSamplerOptions token_sampler_options_from_environment();

// Kept entries in sampling order with normalized probabilities.
class TokenDistribution {
public:
    [[nodiscard]] std::span<const std::uint32_t> tokens() const noexcept {return tokens_;}
    [[nodiscard]] std::span<const double> probabilities() const noexcept {return probabilities_;}
    [[nodiscard]] double probability(std::uint32_t token) const;
    [[nodiscard]] std::uint32_t sample(double uniform) const;
    [[nodiscard]] bool empty() const noexcept {return tokens_.empty();}
private:
    friend TokenDistribution make_token_distribution(std::span<const float>,std::span<const std::uint32_t>,
        const TokenSamplingPolicy&);
    friend TokenDistribution make_token_distribution_from_probabilities(std::span<const std::uint32_t>,
        std::span<const double>);
    friend TokenDistribution make_row_distribution(std::span<const float>,const TokenSamplingPolicy&);
    std::vector<std::uint32_t> tokens_;
    std::vector<double> probabilities_;
    std::vector<std::uint32_t> by_token_; // indices into tokens_, token id ascending
};

// ids empty: entry i is token i. Otherwise ids[i] is entry i's token (distinct).
[[nodiscard]] TokenDistribution make_token_distribution(std::span<const float> logits,
    std::span<const std::uint32_t> ids,const TokenSamplingPolicy&);

// A distribution computed elsewhere (the device path): the entries with a positive probability, in the given
// order, with their probabilities as given.
[[nodiscard]] TokenDistribution make_token_distribution_from_probabilities(std::span<const std::uint32_t> tokens,
    std::span<const double> probabilities);

// Device-resident draft distribution (drafts sampled on the device): probabilities over `count` token ids in
// ascending order, 0 for ids outside the distribution. Valid until the next draft chain.
struct TokenDraftDevice {
    const std::uint32_t* tokens = nullptr;
    const double* probabilities = nullptr;
    std::uint32_t count = 0;
};

// A sampled draft and the distribution it was drawn from. The position is the
// absolute position of the token the draft proposes. A draft sampled on the
// device carries `device` and an empty `distribution`.
struct TokenDraft {
    std::uint64_t position = 0;
    std::uint32_t token = 0;
    TokenDistribution distribution;
    TokenDraftDevice device;
};

struct TokenDecision {
    std::uint32_t token = 0;
    bool drafted = false,accepted = false,residual = false;
    bool forced = false; // test-only forced rejection (TokenSamplerOptions::force_reject_row): accepted -> rejected
};

// Standard speculative sampling with explicit uniforms: without a draft the
// token is sampled from p with u_plain; with a draft d ~ q it is accepted when
// u_accept * q(d) < p(d), otherwise sampled from normalize(max(0, p - q)) with
// u_residual (the residual covers the full target support; q is zero outside
// the draft vocabulary). A rejected decision never returns the draft token.
[[nodiscard]] TokenDecision decide_token(const TokenDistribution& p,const TokenDraft* draft,
    double u_plain,double u_accept,double u_residual);
// The coupled decision: the token is p.sample(u_plain) with or without a draft; a draft is accepted when it equals
// that token (q is not read).
[[nodiscard]] TokenDecision decide_token_coupled(const TokenDistribution& p,const TokenDraft* draft,double u_plain);

struct TokenSamplerStats {
    std::uint64_t drafts = 0,plain = 0,accepted = 0,rejected = 0;
    std::uint64_t forced = 0; // forced rejections (also counted in rejected)
};

// Uniforms are keyed by (policy RNG, absolute position, draw class), so one
// position draws the same values however it is reached (window row, serial
// step or a replay after a restored window).
class TokenSampler {
public:
    // Options from the environment (token_sampler_options_from_environment).
    explicit TokenSampler(TokenSamplingPolicy);
    TokenSampler(TokenSamplingPolicy,TokenSamplerOptions);
    [[nodiscard]] const TokenSamplingPolicy& policy() const noexcept {return policy_;}
    // The transform drafts are drawn with: the policy, or with top-k 1 under coupled acceptance below
    // argmax_draft_below.
    [[nodiscard]] const TokenSamplingPolicy& draft_policy() const noexcept {return draft_policy_;}
    [[nodiscard]] const TokenSamplerOptions& options() const noexcept {return options_;}
    [[nodiscard]] bool coupled() const noexcept {return options_.acceptance==SampledAcceptance::coupled;}
    [[nodiscard]] TokenDraft draft(std::span<const float> logits,std::span<const std::uint32_t> ids,
        std::uint64_t position);
    [[nodiscard]] TokenDecision decide(std::span<const float> logits,std::uint64_t position,const TokenDraft* draft);
    // Rows of one verification decided in order: row r chooses the token at
    // first_position + r using drafts[r] when r < drafts.size(). Stops after
    // the first row without a draft, the first rejected row, an EOS token or
    // max_rows. logits(r) supplies row r's full-vocabulary logits.
    [[nodiscard]] std::vector<TokenDecision> decide_rows(std::uint32_t max_rows,std::uint64_t first_position,
        std::span<const TokenDraft> drafts,std::span<const std::uint32_t> eos_tokens,
        const std::function<std::span<const float>(std::uint32_t)>& logits);
    [[nodiscard]] const TokenSamplerStats& stats() const noexcept {return stats_;}
    // The uniforms decide() draws for a position (accept and residual are 0 without a draft or when coupled), and the
    // statistics update of one decision: for a caller that computes the decision elsewhere (the device path).
    struct Uniforms {double plain = 0.0,accept = 0.0,residual = 0.0;};
    [[nodiscard]] Uniforms uniforms(std::uint64_t position,bool drafted) const;
    void count(const TokenDecision& decision) noexcept;
    // Window rows decided elsewhere (the device path) follow decide_rows: begin_window() once per window, then
    // force_rejection(decision, row) on each decided row before count(). Both are no-ops unless a forced rejection
    // is configured.
    void begin_window() noexcept {++windows_;}
    void force_rejection(TokenDecision& decision,std::uint32_t row) const noexcept;
    // The same for draft(): the proposal uniform of a position (coupled: the position's plain uniform) and the
    // draft count.
    [[nodiscard]] double draft_uniform(std::uint64_t position) const;
    void count_draft() noexcept {++stats_.drafts;}
private:
    [[nodiscard]] TokenDecision decide_uncounted(std::span<const float> logits,std::uint64_t position,const TokenDraft* draft) const;
    TokenSamplingPolicy policy_,draft_policy_;
    TokenSamplerOptions options_;
    SpeculationRngIdentity rng_;
    TokenSamplerStats stats_;
    std::uint64_t windows_ = 0;
};

}  // namespace strix
