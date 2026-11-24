#pragma once
// Backend-neutral byte-range overlap check of launch operands: a launch that reads an operand while the same launch
// (or the K/V append that precedes it) writes memory must not see the two ranges intersect. Host-side, no device call.
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace strix::core {

struct ByteRange {
    const void* data = nullptr;
    std::size_t bytes = 0;
};

// False for an empty range or one that wraps the address space (its extent is not a valid allocation).
[[nodiscard]] inline bool byte_range_valid(const ByteRange& r) {
    const auto begin = reinterpret_cast<std::uintptr_t>(r.data);
    return r.data != nullptr && r.bytes != 0 && r.bytes <= std::numeric_limits<std::uintptr_t>::max() - begin;
}

// True when the two ranges share at least one byte. An invalid range overlaps nothing.
[[nodiscard]] inline bool byte_ranges_overlap(const ByteRange& a, const ByteRange& b) {
    if (!byte_range_valid(a) || !byte_range_valid(b)) return false;
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes && b0 < a0 + a.bytes;
}

// True when no range of `reads` intersects any range of `writes` (reads may overlap each other, as may writes).
[[nodiscard]] inline bool reads_disjoint_from_writes(std::initializer_list<ByteRange> reads,
                                                     std::initializer_list<ByteRange> writes) {
    for (const auto& r : reads)
        for (const auto& w : writes)
            if (byte_ranges_overlap(r, w)) return false;
    return true;
}

}  // namespace strix::core
