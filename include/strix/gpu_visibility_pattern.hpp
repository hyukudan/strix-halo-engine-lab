#pragma once
#include <cstdint>
#include <limits>

#if defined(__HIPCC__)
#define STRIX_GPU_PATTERN_HD __host__ __device__
#else
#define STRIX_GPU_PATTERN_HD
#endif
namespace strix::transport {
// Includes the epoch: a reconnected session must not accept a preceding
// session's bytes even when sequence numbering starts again at one.
struct GpuVisibilityPattern {
    std::uint64_t epoch = 0, sequence = 0, seed = 0;
    bool operator==(const GpuVisibilityPattern&) const = default;
};
STRIX_GPU_PATTERN_HD constexpr std::uint64_t gpu_visibility_mix(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}
STRIX_GPU_PATTERN_HD constexpr std::uint8_t gpu_visibility_byte(
    std::uint64_t position, GpuVisibilityPattern p) noexcept {
    return static_cast<std::uint8_t>(gpu_visibility_mix(position ^
        gpu_visibility_mix(p.epoch) ^ gpu_visibility_mix(p.sequence ^ 0x19cbad77ULL) ^ p.seed));
}
// Only this fixed-size result crosses GPU -> CPU. It is not a CPU payload
// comparison and is not a transferable production-qualification token.
struct alignas(8) GpuVisibilityResult {
    std::uint64_t epoch = 0, sequence = 0, nonce = 0;
    std::uint64_t checked_bytes = 0, mismatches = 0;
    std::uint64_t first_mismatch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t observed_digest = 0, consumed_digest = 0;
};
static_assert(sizeof(GpuVisibilityResult) == 64);
constexpr bool valid_gpu_visibility_result(const GpuVisibilityResult& result,
    GpuVisibilityPattern expected, std::uint64_t nonce, std::uint64_t bytes) noexcept {
    return expected.epoch && expected.sequence && nonce && bytes &&
        result.epoch == expected.epoch && result.sequence == expected.sequence &&
        result.nonce == nonce && result.checked_bytes == bytes && result.mismatches == 0 &&
        result.first_mismatch == std::numeric_limits<std::uint64_t>::max();
}
// Integer transform used by the consumer kernel as well as the comparison.
// The digest is diagnostic; a digest match never replaces per-byte checking.
STRIX_GPU_PATTERN_HD constexpr std::uint64_t gpu_visibility_consume(
    std::uint64_t position, std::uint8_t value) noexcept {
    return gpu_visibility_mix(position ^ (static_cast<std::uint64_t>(value) << 32) ^ 0xc0175a11ULL);
}
} // namespace strix::transport
#undef STRIX_GPU_PATTERN_HD
