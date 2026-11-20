// Segment plan of the two-stream GDN prefill overlap (include/strix/gdn_prefill_overlap.hpp). The bitwise argument of the
// overlap rests on these properties: the segments tile the call's rows in order with no gap or overlap, every start is a
// multiple of kAlign (the conv chunk, the gates row group and the A8 delta slot all divide it, so a segment's scratch and
// outputs are exactly the whole call's), at most kMaxSegments (the event array of the CUDA core), no segment below
// kMinSegment but the last, and serial (one segment = the incumbent order) when disabled or below kMinRows.
#include "strix/gdn_prefill_overlap.hpp"

#include <cstdint>
#include <cstdio>

namespace {

namespace go = strix::gdn_overlap;

int failures = 0;

void check(bool ok, const char* what, std::uint32_t count) {
    if (!ok) {
        if (failures < 20) std::fprintf(stderr, "FAIL count %u: %s\n", count, what);
        ++failures;
    }
}

void check_count(std::uint32_t count) {
    const go::Plan off = go::plan(count, false);
    check(off.n == 1U && go::segment_rows(off, count, 0) == count, "disabled plan is one segment of all rows", count);
    const go::Plan p = go::plan(count, true);
    if (count < go::kMinRows) {
        check(p.n == 1U && go::segment_rows(p, count, 0) == count, "short call stays serial", count);
        return;
    }
    check(p.n >= 2U && p.n <= go::kMaxSegments, "segments in [2, kMaxSegments]", count);
    check(p.segment % go::kAlign == 0U && p.segment >= go::kMinSegment, "segment aligned and >= kMinSegment", count);
    std::uint64_t next = 0;
    for (std::uint32_t i = 0; i < p.n; ++i) {
        const std::uint32_t first = go::first_row(p, i), rows = go::segment_rows(p, count, i);
        check(first == next, "segments contiguous in order", count);
        check(first % go::kAlign == 0U, "segment start aligned", count);
        check(rows >= 1U, "no empty segment", count);
        check(i + 1U == p.n || rows == p.segment, "only the last segment is short", count);
        next = std::uint64_t{first} + rows;
    }
    check(next == count, "segments cover every row", count);
    check(go::segment_rows(p, count, p.n) == 0U, "no rows past the last segment", count);
}

}  // namespace

int main() {
    static_assert(go::kAlign % 32U == 0U && go::kAlign % 8U == 0U && go::kAlign % 256U == 0U, "kAlign covers conv/gates/A8 groups");
    static_assert(go::kMinRows >= 2U * go::kMinSegment, "an overlapped call has at least two segments");
    for (std::uint32_t count = 0; count <= 40000U; ++count) check_count(count);
    for (std::uint32_t count : {65536U, 131072U, 262144U, 1048575U, 4294967040U}) check_count(count);
    // the default block schedule {1024, 2048, 4096, 8192}: 2 x 512, 4 x 512, 4 x 1024, 4 x 2048
    const std::uint32_t expect[4][3] = {{1024U, 2U, 512U}, {2048U, 4U, 512U}, {4096U, 4U, 1024U}, {8192U, 4U, 2048U}};
    for (const auto& e : expect) {
        const go::Plan p = go::plan(e[0], true);
        check(p.n == e[1] && p.segment == e[2], "default schedule blocks", e[0]);
    }
    if (failures) {
        std::fprintf(stderr, "gdn_prefill_overlap_tests: %d failures\n", failures);
        return 1;
    }
    std::printf("gdn_prefill_overlap_tests: ok\n");
    return 0;
}
