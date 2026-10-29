#pragma once

// Host/device-independent scheduling arithmetic; never changes tensor math.
#include <cstdint>

namespace strix::ds41::rtx {
struct CePacing {
    std::uint32_t layer_bytes;
    std::uint32_t outstanding_threshold;
    std::uint64_t background_limit;
};
// tierstream tests outstanding < threshold BEFORE submitting a whole part.
// Subtract the largest part so its one-part overshoot stays <= one expert.
// Keep enough room to finish an expert: a threshold of one part would strand
// partial victims (completion events are recorded only after the last part).
constexpr CePacing ce_pacing(std::uint32_t gate, std::uint32_t down) {
    const std::uint64_t unit = 2ULL * gate + down;
    const std::uint32_t largest = gate > down ? gate : down;
    return {gate, static_cast<std::uint32_t>(unit - largest + 1), unit};
}
}  // namespace strix::ds41::rtx
