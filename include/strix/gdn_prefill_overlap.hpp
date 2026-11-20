// Two-stream overlap plan for one GDN prefill call (backend-neutral; the CUDA Qwen GDN core binds it, see
// include/strix/cuda/qwen_gdn.hpp launch_core_overlap).
//
// A prefill call runs, per row block, the row-parallel front (gates, conv) -> the row-sequential recurrence -> the
// row-parallel gated norm. The recurrence keeps one CTA per (head, column block) busy for the whole call, so the GPU has
// room for the row-parallel kernels of the NEIGHBOUR segments: the call is cut into segments and the recurrence of segment
// i runs beside the norm of segment i-1 and the front of segment i+1 on a second stream. The conv history and the
// recurrent state are carried segment to segment exactly as they are carried block to block between calls, so the bits do
// not change; only the launch order does.
//
// Segment starts are multiples of kAlign: a multiple of every row grouping of the per-row kernels (conv chunk 32, gates 8
// rows, the A8 delta slot of 256 rows), so a segment's scratch and outputs sit exactly where the whole call puts them.
#pragma once

#include <cstdint>

namespace strix::gdn_overlap {

// Measured on the RTX PRO (no model, qwen_gdn_tests --bench-core-overlap): every extra segment costs ~86 us (event hops,
// launches, pipeline ramp) against ~0.13 us per overlapped row, so 512-row segments lose (T=1024 2 x 512 -3.9 %, T=2048
// 4 x 512 -0.9 %) while 1024-2048-row segments win (T=4096 4 x 1024 +6.7 %, T=8192 4 x 2048 +12.8 %).
inline constexpr std::uint32_t kAlign = 256;         // segment starts (rows from the first row of the call)
inline constexpr std::uint32_t kMinSegment = 2048;   // segments are at least this long (the last at least 1280)
inline constexpr std::uint32_t kMaxSegments = 4;     // 3 of 4 segments overlapped; more only adds fixed cost
inline constexpr std::uint32_t kMinRows = 2U * kMinSegment;  // calls below this stay one segment (serial)

struct Plan {
    std::uint32_t n = 1;        // segments (1 = serial, the incumbent order)
    std::uint32_t segment = 0;  // rows of every segment but the last
};

// enabled: the caller's resolved switch AND a state contract that survives a split (FP32 recurrent state; a 16-bit state
// narrowed once per call would gain narrowings at the segment ends).
[[nodiscard]] constexpr Plan plan(std::uint32_t count, bool enabled) noexcept {
    if (!enabled || count < kMinRows) return Plan{1, count};
    // n balanced segments of >= kMinSegment rows: segment = ceil(count / n) rounded up to kAlign, so the last one is
    // count - (n - 1) * segment >= count / n - (n - 1) * (kAlign - 1) > 0 and the plan has exactly n segments
    const std::uint32_t n = count / kMinSegment < kMaxSegments ? count / kMinSegment : kMaxSegments;
    const std::uint32_t share = count / n + (count % n ? 1U : 0U);
    return Plan{n, (share + kAlign - 1U) / kAlign * kAlign};
}

// first row of segment i < p.n (fits: it is below count)
[[nodiscard]] constexpr std::uint32_t first_row(const Plan& p, std::uint32_t i) noexcept { return i * p.segment; }

// rows of segment i (0 for i >= p.n)
[[nodiscard]] constexpr std::uint32_t segment_rows(const Plan& p, std::uint32_t count, std::uint32_t i) noexcept {
    const std::uint64_t first = std::uint64_t{i} * p.segment;
    if (first >= count) return 0U;
    const std::uint64_t left = count - first;
    return static_cast<std::uint32_t>(left < p.segment ? left : p.segment);
}

}  // namespace strix::gdn_overlap
