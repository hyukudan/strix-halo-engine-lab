#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace strix::q8_projection_organization {

inline constexpr std::string_view gdn_wave8_marker = ".wave8-input-v1";
inline constexpr std::string_view lm_prefetch_marker = ".prefetch8-load-v1";
inline constexpr std::string_view lm_wave8_marker = ".prefetch8-wave8-v1";
inline constexpr std::string_view gdn_output_wave8_marker = ".wave8-q8-output-v1";
inline constexpr std::string_view qsa_output_wave8_marker = ".wave8-q8-qsa-output-v1";

// Bounded BF16/Q8_0 overlays. These contracts do not establish native fidelity
// or performance qualification. The incumbent paths do not call these guards.
inline void validate_geometry(std::uint64_t width, std::uint64_t outputs) {
    if (!width || width > 10240U || width % 32U ||
        !outputs || outputs > 248320U)
        throw std::invalid_argument("Q8 projection organization shape outside admitted bounds");
}

inline void validate_modes(bool virtual_wave, bool prefetch, bool wave8,
                           bool bf16, std::uint64_t width, std::uint64_t outputs) {
    if (wave8 && !prefetch)
        throw std::invalid_argument("wave8 Q8 projection requires prefetch loads");
    if (prefetch && (!virtual_wave || !bf16))
        throw std::invalid_argument("prefetch Q8 projection requires virtual BF16 input");
    if (prefetch) validate_geometry(width, outputs);
}

struct View { std::uintptr_t address; std::uint64_t bytes; };

// Extents may include an allocation tail, but every active access is proved.
// Check full supplied extents for overflow before doing interval arithmetic.
inline void validate_views(std::uint32_t width, std::uint32_t outputs,
                           View input, View weights, View output) {
    validate_geometry(width, outputs);
    const auto check = [](View view, std::uint64_t required) {
        if (!view.address || view.address % 16U || view.bytes < required ||
            view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address)
            throw std::invalid_argument("Q8 projection organization view has invalid alignment or extent");
    };
    check(input, static_cast<std::uint64_t>(width) * 2U);
    check(weights, static_cast<std::uint64_t>(width / 32U) * 34U * outputs);
    check(output, static_cast<std::uint64_t>(outputs) * 4U);
    const auto overlap = [](View a, View b) {
        return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
    };
    if (overlap(output, input) || overlap(output, weights))
        throw std::invalid_argument("Q8 projection organization output aliases a read-only view");
}

// Separate byte contract for internal FP32 activations and BF16 output.
// The BF16-input/FP32-output contract above remains unchanged.
inline void validate_fp32_bf16_views(std::uint32_t width, std::uint32_t outputs,
                                     View input, View weights, View output) {
    validate_geometry(width, outputs);
    const auto check = [](View view, std::uint64_t required) {
        if (!view.address || view.address % 16U || view.bytes < required ||
            view.bytes > std::numeric_limits<std::uintptr_t>::max() - view.address)
            throw std::invalid_argument("Q8 FP32 output projection view has invalid alignment or extent");
    };
    check(input, static_cast<std::uint64_t>(width) * 4U);
    check(weights, static_cast<std::uint64_t>(width / 32U) * 34U * outputs);
    check(output, static_cast<std::uint64_t>(outputs) * 2U);
    const auto overlap = [](View a, View b) {
        return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
    };
    if (overlap(output, input) || overlap(output, weights))
        throw std::invalid_argument("Q8 FP32 output projection aliases a read-only view");
}

} // namespace strix::q8_projection_organization
