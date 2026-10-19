#pragma once
#include "strix/execution_memory.hpp"
#include "strix/token_sampling.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix {

// Device form of sampling-v1 (src/token_sampling.cpp): the transform, the inverse-CDF sample and the speculative
// decision of every row run on the GPU over the row's FP32 logits; only the decisions come back. The uniforms
// are the host sampler's (same counter-based draws), so the random numbers are identical.
// Rows whose cutoff and top-k keep at most `device_top_p_candidates` entries are decided by one workgroup in the
// host's order and arithmetic (sequential sums in token id or top-p order). Larger rows use a fixed parallel
// summation order. In both cases the device `exp` may differ from the host's by an ulp, so a decision can differ
// from the host path only when its uniform falls within a few double ulps of a cumulative boundary.
// Rows whose top-p prefix needs more than `device_top_p_candidates` entries are decided on the host.
inline constexpr std::uint32_t device_top_p_candidates = 2048;

// Which path decides sampled rows. STRIX_SAMPLE_PATH=device|host|auto (default auto); STRIX_SAMPLE_HOST=1 is
// host. auto takes the policy so the choice can depend on it; today it is the device path for every policy (see
// the definition for the measured basis).
enum class TokenSamplingPath : unsigned {device = 0, host = 1};
[[nodiscard]] TokenSamplingPath token_sampling_path(const TokenSamplingPolicy&) noexcept;
[[nodiscard]] inline bool device_token_sampling_enabled(const TokenSamplingPolicy& policy) noexcept {
    return token_sampling_path(policy) == TokenSamplingPath::device;
}
// Draft chain on the device: the device path, a policy whose draft rows never need the host (top-p 1, or top-k
// at most device_top_p_candidates) and STRIX_SAMPLE_DRAFT_HOST unset.
[[nodiscard]] bool device_draft_sampling_enabled(const TokenSamplingPolicy&) noexcept;

struct DeviceSampleRow {
    ExecutionBufferAddress logits;  // FP32, vocabulary entries
    std::uint64_t position = 0;     // absolute position of the decided token
    const TokenDraft* draft = nullptr;
};

class HipTokenSamplerDevice {
public:
    HipTokenSamplerDevice(std::uint32_t vocabulary,std::uint32_t max_rows);
    ~HipTokenSamplerDevice();
    HipTokenSamplerDevice(const HipTokenSamplerDevice&) = delete;
    HipTokenSamplerDevice& operator=(const HipTokenSamplerDevice&) = delete;
    // Decisions of every row (the caller applies the stopping rule). Waits for `stream`. A row whose decision
    // needs the host (oversized top-p prefix) is decided with the host transform after reading its logits.
    [[nodiscard]] std::vector<TokenDecision> decide(TokenSampler& sampler,std::span<const DeviceSampleRow> rows,void* stream);
    [[nodiscard]] std::uint64_t host_fallbacks() const noexcept;
    struct Impl; // also used by the testing functions
private:
    std::unique_ptr<Impl> impl_;
};

// Draft chain sampling on the device. Each step samples the draft head's logits (entry r is token ids[r]) with
// the host sampler's proposal uniform, writes the token to the step's device slot (the next step's input) and
// keeps the step's distribution on the device; nothing waits for the device until finish().
class HipTokenDraftSamplerDevice {
public:
    HipTokenDraftSamplerDevice(std::uint32_t capacity,std::uint32_t max_steps);
    ~HipTokenDraftSamplerDevice();
    HipTokenDraftSamplerDevice(const HipTokenDraftSamplerDevice&) = delete;
    HipTokenDraftSamplerDevice& operator=(const HipTokenDraftSamplerDevice&) = delete;
    // The head's active entries: `rows` entries with ids `host_ids` (empty: entry r is token r). Re-uploads the
    // token order only when `rows` changed (the id list is append-only). Enqueued on `stream`.
    void set_vocabulary(std::span<const std::uint32_t> host_ids,std::uint32_t rows,void* stream);
    // Enqueues step `step`: the proposal at `position` from `logits` (device, `rows` entries); the token goes to
    // `slot` (device). Does not wait.
    void step(const TokenSampler& sampler,std::uint32_t step,const float* logits,std::uint64_t position,
        std::uint32_t* slot,void* stream);
    // After the stream has completed: the drafts of steps [0, steps) with device distributions (valid until the
    // next set_vocabulary or step), counted in `sampler`. Throws on a non-finite draft row.
    void finish(TokenSampler& sampler,std::uint32_t steps,std::vector<TokenDraft>& out);
    struct Impl; // also used by the testing functions
private:
    std::unique_ptr<Impl> impl_;
};

// A host copy of a draft whose distribution lives on the device (unchanged when it is already on the host).
[[nodiscard]] TokenDraft host_token_draft(const TokenDraft& draft,void* stream);

// Testing: the device decision for explicit uniforms, as decide_token(make_token_distribution(logits), draft, ...).
// device_q: the draft distribution is read from device memory (token ids ascending with zero entries).
[[nodiscard]] TokenDecision hip_decide_token_for_testing(std::span<const float> logits,const TokenSamplingPolicy& policy,
    const TokenDraft* draft,double u_plain,double u_accept,double u_residual,bool* host_fallback = nullptr,
    bool device_q = false);
// Testing: one device draft step over `logits` with entry ids `ids` (empty: identity) and proposal uniform `u`,
// returned with its distribution copied to the host.
[[nodiscard]] TokenDraft hip_draft_for_testing(std::span<const float> logits,std::span<const std::uint32_t> ids,
    const TokenSamplingPolicy& policy,double u);

} // namespace strix
