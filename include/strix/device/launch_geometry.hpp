#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace strix::device {

// Host description of the same launch on either GPU runtime. Conversion to a
// native dimension type is confined to the enqueue site.
struct LaunchGeometry {
    std::array<std::uint32_t, 3> blocks{1, 1, 1};
    std::array<std::uint32_t, 3> threads{1, 1, 1};
    std::uint32_t shared_bytes = 0;

    constexpr bool operator==(const LaunchGeometry& other) const noexcept {
        for (std::size_t axis = 0; axis < blocks.size(); ++axis)
            if (blocks[axis] != other.blocks[axis] || threads[axis] != other.threads[axis]) return false;
        return shared_bytes == other.shared_bytes;
    }
    constexpr bool operator!=(const LaunchGeometry& other) const noexcept { return !(*this == other); }

    template <class Dimensions>
    constexpr Dimensions grid() const { return {blocks[0], blocks[1], blocks[2]}; }

    template <class Dimensions>
    constexpr Dimensions block() const { return {threads[0], threads[1], threads[2]}; }
};

}  // namespace strix::device
