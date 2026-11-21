#pragma once
#include "strix/execution_memory.hpp"
#include "strix/token_sampling.hpp"

#include <cstdint>

namespace strix {

// Device form of sampling-v1 (src/token_sampling.cpp): the transform, the inverse-CDF sample and the speculative
// decision of every row run on the GPU over the row's FP32 logits; only the decisions come back. The uniforms
// are the host sampler's (same counter-based draws), so the random numbers are identical.
// Rows whose cutoff and top-k keep at most `device_top_p_candidates` entries (and every row with a top-k of at most
// device_top_p_candidates / 32, selected in one pass) are decided by one workgroup in the host's order and
// arithmetic (sequential sums in token id or top-p order). Larger rows sum in a fixed parallel order; their top-p
// prefix and inverse CDF are mass quantiles in (weight desc, id asc) order with exact fixed-point sums. In all
// cases the device `exp` may differ from the host's by an ulp, so a decision can differ from the host path only
// when its uniform falls within a few double ulps of a cumulative boundary.
inline constexpr std::uint32_t device_top_p_candidates = 2048;

// Which path decides sampled rows. STRIX_SAMPLE_PATH=device|host|auto (default auto); STRIX_SAMPLE_HOST=1 is
// host. auto takes the policy so the choice can depend on it; today it is the device path for every policy.
enum class TokenSamplingPath : unsigned {device = 0, host = 1};
[[nodiscard]] TokenSamplingPath token_sampling_path(const TokenSamplingPolicy&) noexcept;
[[nodiscard]] inline bool device_token_sampling_enabled(const TokenSamplingPolicy& policy) noexcept {
    return token_sampling_path(policy) == TokenSamplingPath::device;
}
// Draft chain on the device: the device path and STRIX_SAMPLE_DRAFT_HOST unset.
[[nodiscard]] bool device_draft_sampling_enabled(const TokenSamplingPolicy&) noexcept;

struct DeviceSampleRow {
    ExecutionBufferAddress logits;  // FP32, vocabulary entries
    std::uint64_t position = 0;     // absolute position of the decided token
    const TokenDraft* draft = nullptr;
};

} // namespace strix
