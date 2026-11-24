// Byte-range overlap check of launch operands (src/core/byte_range.hpp): adjacent ranges are disjoint, one shared byte
// is an overlap, containment and identity overlap, empty / null ranges overlap nothing, reads may overlap reads.
#include "core/byte_range.hpp"

#include <array>
#include <cstdio>

namespace {
using strix::core::ByteRange;
using strix::core::byte_ranges_overlap;
using strix::core::reads_disjoint_from_writes;

int failures = 0;
void expect(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
}  // namespace

int main() {
    std::array<unsigned char, 256> mem{};
    const auto at = [&](std::size_t offset, std::size_t bytes) { return ByteRange{mem.data() + offset, bytes}; };
    expect(!byte_ranges_overlap(at(0, 64), at(64, 64)) && !byte_ranges_overlap(at(64, 64), at(0, 64)), "adjacent ranges are disjoint");
    expect(byte_ranges_overlap(at(0, 65), at(64, 64)) && byte_ranges_overlap(at(64, 64), at(0, 65)), "one shared byte overlaps");
    expect(byte_ranges_overlap(at(0, 256), at(100, 4)) && byte_ranges_overlap(at(100, 4), at(0, 256)), "containment overlaps");
    expect(byte_ranges_overlap(at(8, 8), at(8, 8)), "identical ranges overlap");
    expect(!byte_ranges_overlap(at(8, 0), at(0, 256)) && !byte_ranges_overlap({nullptr, 16}, at(0, 256)) &&
               !byte_ranges_overlap({mem.data(), static_cast<std::size_t>(-1)}, at(0, 256)),
           "empty, null and wrapping ranges overlap nothing");
    expect(reads_disjoint_from_writes({at(0, 32), at(16, 32)}, {at(128, 64), at(192, 64)}), "overlapping reads, disjoint writes");
    expect(!reads_disjoint_from_writes({at(0, 32), at(100, 8)}, {at(104, 16)}), "a read inside a write is refused");
    expect(reads_disjoint_from_writes({}, {at(0, 8)}) && reads_disjoint_from_writes({at(0, 8)}, {}), "empty sets are disjoint");
    std::printf("%s byte range: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
