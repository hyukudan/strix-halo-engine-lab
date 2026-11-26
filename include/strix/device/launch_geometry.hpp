#pragma once

#include <array>
#include <cstdint>

namespace strix::device {

// Host description of the same launch on either GPU runtime. Conversion to a
// native dimension type is confined to the enqueue site.
struct LaunchGeometry {
    std::array<std::uint32_t, 3> blocks{1, 1, 1};
    std::array<std::uint32_t, 3> threads{1, 1, 1};
    std::uint32_t shared_bytes = 0;

    bool operator==(const LaunchGeometry& other) const noexcept {
        return blocks == other.blocks && threads == other.threads && shared_bytes == other.shared_bytes;
    }
    bool operator!=(const LaunchGeometry& other) const noexcept { return !(*this == other); }

    template <class Dimensions>
    constexpr Dimensions grid() const { return {blocks[0], blocks[1], blocks[2]}; }

    template <class Dimensions>
    constexpr Dimensions block() const { return {threads[0], threads[1], threads[2]}; }
};

}  // namespace strix::device
