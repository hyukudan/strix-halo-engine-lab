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
// are the host sampler's (same counter-based draws), so the random numbers are identical. The arithmetic is the
// same double-precision transform with a fixed device summation order and the device `exp`: a decision can
// differ from the host path only when its uniform falls within a few double ulps of a cumulative boundary.
// Rows whose top-p prefix needs more than `device_top_p_candidates` entries are decided on the host.
// STRIX_SAMPLE_HOST=1 selects the host path everywhere.
[[nodiscard]] bool device_token_sampling_enabled() noexcept;
inline constexpr std::uint32_t device_top_p_candidates = 2048;

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
    // needs the host (oversized top-p prefix) is decided with `sampler.decide` after reading its logits.
    [[nodiscard]] std::vector<TokenDecision> decide(TokenSampler& sampler,std::span<const DeviceSampleRow> rows,void* stream);
    [[nodiscard]] std::uint64_t host_fallbacks() const noexcept;
    struct Impl; // also used by hip_decide_token_for_testing
private:
    std::unique_ptr<Impl> impl_;
};

// Testing: the device decision for explicit uniforms, as decide_token(make_token_distribution(logits), draft, ...).
[[nodiscard]] TokenDecision hip_decide_token_for_testing(std::span<const float> logits,const TokenSamplingPolicy& policy,
    const TokenDraft* draft,double u_plain,double u_accept,double u_residual,bool* host_fallback = nullptr);

} // namespace strix
