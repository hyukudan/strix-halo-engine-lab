#pragma once
// Backend-neutral plan for exact decode attention that walks the K/V prefix once for several (row, q head) pairs.
// Work unit: a pair (row, q head) of one kv head. The pairs of kv head kh are ordered row-major, the q heads of a row
// adjacent: j = row * group + g, q head = kh * group + g (rows 0..rows-1 of a window, group = heads / kv_heads).
// A block takes `pairs` consecutive pairs and loads each K/V position once for all of them; per pair the arithmetic
// (order, operands) is the one-pair form's, so the plan changes the schedule, never the bits.
// The plan picks the largest compiled pair count that still leaves `min_blocks` blocks over all kv heads (the
// backend's occupancy floor, measured per device), so a 1-row plain step keeps its parallelism and a 16-row verify
// window stops re-reading K/V once per pair.
// Home: shared (host shared policy): no backend, no model names; HIP binds it in attn_core/dense_decode.hpp, CUDA in
// cuda_core/attention_row_walk.hpp. Each backend's measured device tables sit side by side in dense_decode_tables.
#include <cstddef>
#include <string_view>

namespace strix::attn_row_share {

// Pair counts a backend instantiates (ascending). 6 and 12 cover one / two rows of a GQA group of 6.
inline constexpr unsigned candidates[] = {1U, 2U, 3U, 6U, 12U};

struct Plan {
    unsigned pairs = 1U;   // pairs per block
    unsigned blocks = 0U;  // blocks per kv head = ceil(rows * group / pairs)
    unsigned walk = 1U;    // positions whose K/V loads a wave issues together (applied in ascending order)
};

// A measured per-device choice for one row count (pairs per block, positions per walk step).
struct Choice {
    unsigned pairs = 1U, walk = 1U;
};

constexpr unsigned pair_row(unsigned j, unsigned group) { return j / group; }
constexpr unsigned pair_head(unsigned j, unsigned kv_head, unsigned group) { return kv_head * group + j % group; }
constexpr unsigned blocks_for(unsigned rows, unsigned group, unsigned pairs) { return (rows * group + pairs - 1U) / pairs; }

// enabled = false (the backend's off switch) or rows * group == 0 keeps the one-pair form.
constexpr Plan plan(unsigned rows, unsigned group, unsigned kv_heads, unsigned min_blocks, unsigned max_pairs,
                    bool enabled = true) {
    Plan best{1U, blocks_for(rows, group, 1U)};
    if (!enabled || rows == 0U || group == 0U)
        return best;
    for (const unsigned c : candidates) {
        if (c > max_pairs)
            break;
        const unsigned b = blocks_for(rows, group, c);
        if (c == 1U || std::size_t(kv_heads) * b >= min_blocks)
            best = Plan{c, b};
    }
    return best;
}

// A backend's measured choices for walks of at most max_len positions: by_rows[rows] for rows 0..table_rows-1.
inline constexpr unsigned table_rows = 17U;
struct Band {
    unsigned max_len = 0U;  // the band holds walks of length <= max_len (bands ascending; the last one is ~0U)
    Choice by_rows[table_rows];
};
// Table form: the first band whose max_len >= len (len = positions walked by the window's last row), its choice for
// `rows`; rows past the table (the 64-row EXACT prefill windows) take the band's last row choice (the CUDA walk cap
// reads the same row); no band, or enabled = false -> one pair, one position per step.
constexpr Plan plan_table(unsigned rows, unsigned group, unsigned len, const Band* bands, unsigned n_bands,
                          bool enabled = true) {
    if (!enabled || group == 0U)
        return Plan{1U, blocks_for(rows, group, 1U), 1U};
    const unsigned row = rows < table_rows ? rows : table_rows - 1U;
    for (unsigned b = 0; b < n_bands; ++b)
        if (len <= bands[b].max_len) {
            const Choice c = bands[b].by_rows[row];
            return Plan{c.pairs, blocks_for(rows, group, c.pairs), c.walk};
        }
    return Plan{1U, blocks_for(rows, group, 1U), 1U};
}
// gfx1151 with 40 CUs (HIP multiProcessorCount 20), GQA group 6 x 4 kv heads, Dim 256: the fastest bitwise choice per (length band, rows), no-model
// bench tests/hip_dense_decode_rows_tests.hip --bench (contexts 2K 8K 16K 32K 48K 64K 131K, rows 1-16). (1, 1) is the
// backend's main schedule. The 96K band edge lies between the measured 64K and 131K.
// - rows 1-3: one pair, 2 positions per step: the walk is latency-bound (32K, 1 row: 64 GB/s of unique bytes).
// - rows 4-5: one-pair blocks share K/V through the 32 MB MALL while a kv head's K+V (1 KiB x len) fits it (<= 32K);
//   from 48K the main schedule is fastest (48K, 4 rows: 3.37 ms vs 3.62 / 3.70); 131K, 5 rows: the main schedule too
//   (2 pairs x 2 positions ties it: 10.46 / 10.55 ms vs main 10.52 / 10.55 in two runs, not faster twice).
// - rows 6: the main schedule up to the 96K edge (with main's unfused arithmetic, 2 pairs x 2 positions is 1-6 % slower
//   up to 32K, 16K 1.49 vs 1.41 ms, and ties or loses at 48K, 4.54-5.17 vs 4.54-4.55; it wins 1-3 % at 64K, not
//   enough to split a band), 2 pairs x 2 positions above (131K 12.7 vs 25.6); rows 7: 2 pairs, 2 positions; rows 8-11
//   and 13: 2 pairs; rows 12 and 14-16: 3 pairs (one-pair blocks thrash the caches from about 7 rows up).
// - rows 17-64 (the 64-row EXACT prefill windows; plan_table reads row 16): 3 pairs. ms per call vs the main schedule,
//   64 rows: 512 0.422 vs 0.488, 2K 1.68 vs 4.49, 8K 7.21 vs 30.4, 32K 37.8 vs 283, 64K 145 vs 660; 17 / 24 / 32 / 48 rows
//   at 8K and 32K 1.6-6.2x faster; at 512 and 2K within 2 % or faster (512 x 24 rows 0.180 vs 0.177). Bitwise to v0 at
//   17 and 64 rows (1K, 33K).
inline constexpr Band gfx1151_mp20_g6_kv4_d256[] = {
    {32768U,
     {{1U, 1U},
      {1U, 2U}, {1U, 2U}, {1U, 2U},
      {1U, 2U}, {1U, 2U}, {1U, 1U}, {2U, 2U},
      {2U, 1U}, {2U, 1U}, {2U, 1U}, {2U, 1U},
      {3U, 1U}, {2U, 1U}, {3U, 1U}, {3U, 1U}, {3U, 1U}}},
    {98304U,
     {{1U, 1U},
      {1U, 2U}, {1U, 2U}, {1U, 2U},
      {1U, 1U}, {1U, 1U}, {1U, 1U}, {2U, 2U},
      {2U, 1U}, {2U, 1U}, {2U, 1U}, {2U, 1U},
      {3U, 1U}, {2U, 1U}, {3U, 1U}, {3U, 1U}, {3U, 1U}}},
    {~0U,
     {{1U, 1U},
      {1U, 2U}, {1U, 2U}, {1U, 2U},
      {1U, 1U}, {1U, 1U}, {2U, 2U}, {2U, 2U},
      {2U, 1U}, {2U, 1U}, {2U, 1U}, {2U, 1U},
      {3U, 1U}, {2U, 1U}, {3U, 1U}, {3U, 1U}, {3U, 1U}}},
};

// sm_120 with 188 SMs (CUDA RTX PRO 6000), GQA group 6 x 4 kv heads, Dim 256. A CUDA block takes every q head of its kv
// group, so a CUDA table counts rows (the plan's unit at group 1) and caps a walk: within the cap, plan() with the
// device's occupancy floor picks the walk per active chunk count (cuda_core/attention_row_walk.hpp). One band; windows
// past the table read the 16-row cap. Measured: no-model bench tests/cuda_core/cuda_core_test.cu --attn-bench (lengths
// 1K 8K 32K 131K, rows 1-16): every cell with 2 or more rows is faster than one row per block (automatic / off <= 0.934)
// and than the previous kernels (<= 0.442). 12 rows = the longest compiled walk, so the cap never binds; 1 row: one row
// per block.
inline constexpr Band sm120_mp188_g6_kv4_d256[] = {
    {~0U,
     {{1U, 1U},
      {1U, 1U}, {12U, 1U}, {12U, 1U},
      {12U, 1U}, {12U, 1U}, {12U, 1U}, {12U, 1U},
      {12U, 1U}, {12U, 1U}, {12U, 1U}, {12U, 1U},
      {12U, 1U}, {12U, 1U}, {12U, 1U}, {12U, 1U}, {12U, 1U}}},
};

// Device key: a table holds for the device and head shape it was measured on, nothing else. arch = the backend's
// architecture name without feature suffix (HIP gcnArchName up to ':'; CUDA "sm_" + major * 10 + minor),
// multiprocessors = the runtime's multiprocessor count as reported (HIP multiProcessorCount: work-group processors on
// RDNA, 20 for a 40-CU gfx1151; CUDA SM count).
struct DeviceTable {
    std::string_view arch;
    unsigned multiprocessors = 0U, group = 0U, kv_heads = 0U, head_dim = 0U;
    const Band* bands = nullptr;
    unsigned n_bands = 0U;
};
inline constexpr DeviceTable dense_decode_tables[] = {
    {"gfx1151", 20U, 6U, 4U, 256U, gfx1151_mp20_g6_kv4_d256,
     unsigned(sizeof(gfx1151_mp20_g6_kv4_d256) / sizeof(Band))},
    {"sm_120", 188U, 6U, 4U, 256U, sm120_mp188_g6_kv4_d256,
     unsigned(sizeof(sm120_mp188_g6_kv4_d256) / sizeof(Band))},
};
// The measured table of (device, shape), or nullptr: an unknown device or shape keeps the main schedule.
constexpr const DeviceTable* find_table(std::string_view arch, unsigned multiprocessors, unsigned group,
                                        unsigned kv_heads, unsigned head_dim) {
    for (const DeviceTable& t : dense_decode_tables)
        if (t.arch == arch && t.multiprocessors == multiprocessors && t.group == group && t.kv_heads == kv_heads &&
            t.head_dim == head_dim)
            return &t;
    return nullptr;
}
// Table form for a device: no table (nullptr) -> one pair, one position per step (the main schedule).
constexpr Plan plan_device(unsigned rows, unsigned group, unsigned len, const DeviceTable* table, bool enabled = true) {
    if (table == nullptr)
        return Plan{1U, blocks_for(rows, group, 1U), 1U};
    return plan_table(rows, group, len, table->bands, table->n_bands, enabled);
}
}  // namespace strix::attn_row_share
