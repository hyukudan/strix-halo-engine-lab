#pragma once
// Row walks of the CUDA decode attention (attention_decode scores / values kernels): which verify rows share one K/V walk.
// Scheduling only, SAME bits: a walk loads each K/V position of its (chunk, kv head) once and applies it to every row of
// the walk; per (row, head) the order and operands are the one-row form's (attention.hpp contract).
// Policy = the shared backend-neutral plan strix/attn_row_share.hpp. A CUDA block already takes every q head of its kv
// group, so the plan's unit is a row (group 1) and its "kv heads" are the active (chunk, kv head) blocks of one row.
// 'automatic' walks only on a measured device and head shape: its table in attn_row_share::dense_decode_tables (beside
// the HIP ones) caps the walk (walk_cap); an unmeasured device or shape keeps one row per block, as HIP keeps its main
// schedule. The
// length is a device scalar (graph capture), so the host turns the plan into a step table over the active chunk count
// (row_walk) and each block reads it (rows_per_walk); the table gives the plan's own answer for every chunk count.
// Rows of different cache slots never share a walk; inside a slot the walks start at slot-relative multiples of the walk
// length (walk_start).
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>

#include "strix/attn_row_share.hpp"
#include "strix/env_flags.hpp"

#if defined(__CUDACC__)
#define STRIX_ROW_WALK_HD __host__ __device__
#else
#define STRIX_ROW_WALK_HD
#endif

namespace strix::cuda_core {

inline constexpr std::uint32_t kRowWalkSteps = static_cast<std::uint32_t>(std::size(attn_row_share::candidates));

// Process-wide mode of attention_decode (read on the host at each launch; a captured graph keeps the mode it was
// captured with). off: one row per block, chunk fastest (the one-row form); automatic: the shared plan, walks of one
// chunk adjacent in launch order; all: the largest walk regardless of the occupancy floor (tests, kernel bench).
enum class DecodeRowShare : std::uint8_t { off = 0, automatic = 1, all = 2 };
void set_decode_row_share(DecodeRowShare mode);  // defined in the CUDA core (attention.cu)
DecodeRowShare decode_row_share();               // first call without a set_decode_row_share: row_share_from_env()

// Initial mode: STRIX_DECODE_ATTN_ROW_SHARE, the switch the HIP dense decode reads (src/attn_core/dense_decode.hpp), so one
// setting turns the shared walk off on every backend. Unset = automatic; "0" = off (the one-row form, same bits).
[[nodiscard]] inline DecodeRowShare row_share_from_env() {
    return envcfg::flag("STRIX_DECODE_ATTN_ROW_SHARE", true) ? DecodeRowShare::automatic : DecodeRowShare::off;
}

// Device key of the shared tables for a CUDA device: "sm_" + major * 10 + minor (12.0 -> "sm_120"); the multiprocessor
// count is the device's SM count.
[[nodiscard]] inline std::string cuda_arch_key(int major, int minor) { return "sm_" + std::to_string(major * 10 + minor); }

// Longest walk (rows) of a launch whose kernel walks at most max_rows rows. off, one row, or 'automatic' without a
// table (unmeasured device or head shape) -> 1: one row per block, chunk fastest. 'automatic' with a table: its cap for
// `rows` (windows past the table read its last row count; CUDA tables hold one band). 'all': max_rows (tests, bench).
[[nodiscard]] inline std::uint32_t walk_cap(DecodeRowShare mode, const attn_row_share::DeviceTable* table, std::uint32_t rows,
                                            std::uint32_t max_rows) {
    if (mode == DecodeRowShare::off || rows < 2) return 1;
    if (mode == DecodeRowShare::all) return max_rows;
    if (table == nullptr) return 1;
    const unsigned r = rows < attn_row_share::table_rows ? rows : attn_row_share::table_rows - 1u;
    const std::uint32_t cap = attn_row_share::plan_device(r, 1u, ~0u, table).pairs;
    return cap < max_rows ? cap : max_rows;
}

struct RowWalk {
    std::uint32_t adjacent = 0;  // 1: linear block index = row fastest, then kv head, then chunk; 0: chunk fastest
    std::uint32_t steps = 0;     // entries used: from min_active[i] active chunks on, walks take rows[i] rows
    std::uint32_t rows[kRowWalkSteps] = {};
    std::uint32_t min_active[kRowWalkSteps] = {};
};

// Step table of the shared plan for `rows` verify rows, `kv_heads` kv heads, chunk counts 1..max_chunks, walks of at
// most max_rows rows (the kernel's state budget), occupancy floor min_blocks (blocks of the whole launch).
[[nodiscard]] inline RowWalk row_walk(DecodeRowShare mode, std::uint32_t rows, std::uint32_t kv_heads, std::uint32_t max_chunks,
                                      std::uint32_t max_rows, std::uint32_t min_blocks) {
    RowWalk rw;
    if (mode == DecodeRowShare::off || rows < 2 || kv_heads == 0 || max_chunks == 0) return rw;
    rw.adjacent = 1;
    const std::uint32_t min_floor = mode == DecodeRowShare::all ? 0u : min_blocks;
    const auto pick = [&](std::uint32_t active) { return attn_row_share::plan(rows, 1u, active * kv_heads, min_floor, max_rows).pairs; };
    for (const unsigned c : attn_row_share::candidates) {
        if (c < 2u) continue;
        if (c > max_rows || pick(max_chunks) < c) break;  // the floor only gets harder for longer walks
        std::uint32_t lo = 1, hi = max_chunks;
        while (lo < hi) {
            const std::uint32_t mid = lo + (hi - lo) / 2u;
            if (pick(mid) >= c) hi = mid;
            else lo = mid + 1u;
        }
        rw.rows[rw.steps] = c;
        rw.min_active[rw.steps] = lo;
        ++rw.steps;
    }
    return rw;
}

[[nodiscard]] STRIX_ROW_WALK_HD inline std::uint32_t rows_per_walk(const RowWalk& rw, std::uint32_t active_chunks) {
    std::uint32_t per = 1;
    for (std::uint32_t i = 0; i < rw.steps; ++i)
        if (active_chunks >= rw.min_active[i]) per = rw.rows[i];
    return per;
}

// Row r (slot_start = first row of its cache slot) starts a walk of at most `per` rows.
[[nodiscard]] STRIX_ROW_WALK_HD inline bool walk_start(std::uint32_t r, std::uint32_t slot_start, std::uint32_t per) {
    return (r - slot_start) % per == 0u;
}

// Work items of the shared walks: item = (chunk, kv head, walk), walk fastest, then kv head, then chunk (the walks of one
// chunk adjacent: a repeated K/V line is an L2 hit). A persistent grid of B blocks takes items b, b + B, b + 2 B, ...
// below active_chunks * kv_heads * walks.
struct RowWalkItem {
    std::uint32_t walk = 0, kv_head = 0, chunk = 0;
};
[[nodiscard]] STRIX_ROW_WALK_HD inline RowWalkItem row_walk_item(std::uint32_t item, std::uint32_t walks, std::uint32_t kv_heads) {
    RowWalkItem it;
    it.walk = item % walks;
    const std::uint32_t rest = item / walks;
    it.kv_head = rest % kv_heads;
    it.chunk = rest / kv_heads;
    return it;
}

// Compiled query-head group widths of the decode kernels (a kernel runs GT >= heads / kv_heads heads as independent chains)
// and the longest values walk at width GT: the largest shared-plan candidate with rows x GT <= kValueWalkPairs (the
// weights of every (row, head) pair of the walk stay in shared memory for the whole chunk).
inline constexpr std::uint32_t kValueWalkPairs = 36;
[[nodiscard]] constexpr std::uint32_t attn_group_width(std::uint32_t group) { return group <= 4u ? 4u : group <= 6u ? 6u : 8u; }
[[nodiscard]] constexpr std::uint32_t value_walk_rows(std::uint32_t gt) {
    std::uint32_t best = 1;
    for (const unsigned c : attn_row_share::candidates)
        if (c * gt <= kValueWalkPairs) best = c;
    return best;
}

}  // namespace strix::cuda_core
