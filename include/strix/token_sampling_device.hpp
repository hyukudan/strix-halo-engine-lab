#pragma once
#include "strix/execution_memory.hpp"
#include "strix/token_sampling.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strix {

// Device form of sampling-v1 (src/token_sampling.cpp): the transform, the inverse-CDF sample and the speculative
// decision of every row run on the GPU over the row's FP32 logits; only the decisions come back. The uniforms
// are the host sampler's (same counter-based draws), so the random numbers are identical.
// In the device rank key +0 sorts above -0. The host treats both zeros as a numeric tie ordered by ID.
// Rows with at most `device_top_p_candidates` entries after cutoff and top-k use one workgroup and sequential sums.
// Large rows without top-p use fixed parallel sums and compare a scaled draw to unnormalized weights. Large
// top-p uses mass quantiles in rank-key order with fixed-point sums. Differences at CDF boundaries are possible
// even for exact weights. Use the explicit host path when the host transform and CDF are required.
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

// The rows one request decides in a call: its sampler (transform, uniforms, acceptance rule) and its rows.
struct DeviceSampleGroup {
    TokenSampler* sampler = nullptr;
    std::span<const DeviceSampleRow> rows;
};
// Rows of all groups of one decide call (one results slot per row).
inline constexpr std::uint32_t device_sample_max_rows = 64;

// True when two policies apply the same transform (temperature, top-k, top-p, min-p); the seed only keys uniforms.
[[nodiscard]] inline bool same_token_transform(const TokenSamplingPolicy& a, const TokenSamplingPolicy& b) noexcept {
    return a.temperature == b.temperature && a.top_k == b.top_k && a.top_p == b.top_p && a.min_p == b.min_p;
}

// How one decide call runs its groups: the groups with the same transform share one kernel sequence (the kernels
// decide each row on its own from its logits and its own uniforms, so the tokens equal one call per group), up to
// `max_rows` rows per sequence; a group is never split. `order` lists the groups in enqueue order (first appearance
// of each transform, then input order); enqueue e runs order[first .. first + groups) into results slots
// [offset, offset + rows). Throws for a null sampler, an empty group, a group over max_rows or a call over
// device_sample_max_rows rows.
struct DeviceSampleEnqueue { std::uint32_t first = 0, groups = 0, offset = 0, rows = 0; };
struct DeviceSamplePlan {
    std::vector<std::uint32_t> order;
    std::vector<DeviceSampleEnqueue> enqueues;
    std::uint32_t rows = 0;
};
[[nodiscard]] DeviceSamplePlan plan_device_sample_enqueues(std::span<const DeviceSampleGroup> groups, std::uint32_t max_rows);

} // namespace strix
