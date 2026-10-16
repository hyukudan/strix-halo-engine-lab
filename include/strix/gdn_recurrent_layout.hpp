#pragma once
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace strix {
// Explicit implementation choice; baseline retains the incumbent identity.
enum class HipGdnRecurrentLayout : std::uint32_t { baseline = 0, stride32 = 1, stride33 = 2, transposed = 3 };
// Island admission is deliberately narrower than the eager diagnostic family.
[[nodiscard]] constexpr bool gdn_island_recurrent_layout_admitted(HipGdnRecurrentLayout layout) noexcept {
    return layout == HipGdnRecurrentLayout::baseline || layout == HipGdnRecurrentLayout::transposed;
}
[[nodiscard]] inline std::string_view gdn_recurrent_layout_marker(HipGdnRecurrentLayout layout) {
    switch (layout) {
    case HipGdnRecurrentLayout::baseline: return {};
    case HipGdnRecurrentLayout::stride32: return ".lds32-v1";
    case HipGdnRecurrentLayout::stride33: return ".lds33-v1";
    case HipGdnRecurrentLayout::transposed: return ".lds-transposed-v1";
    }
    throw std::invalid_argument("invalid GDN recurrent layout");
}
} // namespace strix
