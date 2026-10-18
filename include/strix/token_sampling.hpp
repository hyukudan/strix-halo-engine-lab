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
// Counter-based random identity: seed, a fixed single-request identity and the policy identity.
[[nodiscard]] SpeculationRngIdentity token_sampling_rng(const TokenSamplingPolicy&);

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
    std::vector<std::uint32_t> tokens_;
    std::vector<double> probabilities_;
    std::vector<std::uint32_t> by_token_; // indices into tokens_, token id ascending
};

// ids empty: entry i is token i. Otherwise ids[i] is entry i's token (distinct).
[[nodiscard]] TokenDistribution make_token_distribution(std::span<const float> logits,
    std::span<const std::uint32_t> ids,const TokenSamplingPolicy&);

// A sampled draft and the distribution it was drawn from. The position is the
// absolute position of the token the draft proposes.
struct TokenDraft {
    std::uint64_t position = 0;
    std::uint32_t token = 0;
    TokenDistribution distribution;
};

struct TokenDecision {
    std::uint32_t token = 0;
    bool drafted = false,accepted = false,residual = false;
};

// Standard speculative sampling with explicit uniforms: without a draft the
// token is sampled from p with u_plain; with a draft d ~ q it is accepted when
// u_accept * q(d) < p(d), otherwise sampled from normalize(max(0, p - q)) with
// u_residual (the residual covers the full target support; q is zero outside
// the draft vocabulary). A rejected decision never returns the draft token.
[[nodiscard]] TokenDecision decide_token(const TokenDistribution& p,const TokenDraft* draft,
    double u_plain,double u_accept,double u_residual);

struct TokenSamplerStats {
    std::uint64_t drafts = 0,plain = 0,accepted = 0,rejected = 0;
};

// Uniforms are keyed by (policy RNG, absolute position, draw class), so one
// position draws the same values however it is reached (window row, serial
// step or a replay after a restored window).
class TokenSampler {
public:
    explicit TokenSampler(TokenSamplingPolicy);
    [[nodiscard]] const TokenSamplingPolicy& policy() const noexcept {return policy_;}
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
private:
    TokenSamplingPolicy policy_;
    SpeculationRngIdentity rng_;
    TokenSamplerStats stats_;
};

}  // namespace strix
