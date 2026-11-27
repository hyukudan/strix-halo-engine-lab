// Host-only: shared exact decode attention pair plan (strix/attn_row_share.hpp). Invariants: the plan covers every
// (row, q head) pair of a kv head exactly once in row-major order; every pair of a block sees a prefix of the block's
// walk (rows ascend inside a block, so the walk ends at the block's last row); the pair count respects the occupancy
// floor and the off switch; and a CPU model of the kernel's per-pair masked walk visits, per pair and per wave
// residue, exactly the positions of the one-pair walk in the same order (the order the bits depend on).
#include "strix/attn_row_share.hpp"

#include <cstdio>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* why) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", why);
        ++failures;
    }
}
}  // namespace

int main() {
    namespace rs = strix::attn_row_share;
    constexpr unsigned G = 6U, KV = 4U, floor24 = 24U, waves = 8U;
    // Qwen 27B dense (24 q / 4 kv heads) under the provisional floor of 24 blocks.
    check(rs::plan(1U, G, KV, floor24, 12U).pairs == 1U, "1 row: 24 one-pair blocks (plain step unchanged)");
    check(rs::plan(2U, G, KV, floor24, 12U).pairs == 2U, "2 rows: P=2 -> 24 blocks");
    check(rs::plan(4U, G, KV, floor24, 12U).pairs == 3U, "4 rows: P=3 -> 32 blocks (P=6 would leave 16)");
    check(rs::plan(8U, G, KV, floor24, 12U).pairs == 6U, "8 rows: P=6 -> 32 blocks");
    check(rs::plan(16U, G, KV, floor24, 12U).pairs == 12U, "16 rows: P=12 -> 32 blocks");
    check(rs::plan(16U, G, KV, floor24, 6U).pairs == 6U, "max_pairs caps the pair count");
    check(rs::plan(16U, G, KV, floor24, 12U, false).pairs == 1U, "off switch: one-pair form");
    check(rs::plan(16U, G, KV, 0U, 12U).blocks == 8U, "16 rows x 6 heads / 12 = 8 blocks per kv head");
    // Table form (gfx1151 measured choices by length band): every (band, rows 1..16) gets an instantiated form; the off
    // switch gives the main schedule (one pair, one position); row counts past the table read its last row.
    // Device key: the table holds for the device and head shape it was measured on; anything else -> main schedule.
    const rs::DeviceTable* dt = rs::find_table("gfx1151", 20U, G, KV, 256U);
    check(dt != nullptr, "gfx1151, 20 multiprocessors (40 CU), group 6 x 4 kv, Dim 256: measured table");
    check(rs::find_table("gfx1100", 20U, G, KV, 256U) == nullptr, "unknown architecture: no table");
    check(rs::find_table("", 0U, G, KV, 256U) == nullptr, "unreadable device: no table");
    check(rs::find_table("gfx1151", 16U, G, KV, 256U) == nullptr, "same architecture, other multiprocessor count: no table");
    check(rs::find_table("gfx1151", 40U, G, KV, 256U) == nullptr, "a CU count is not a multiprocessor count");
    check(rs::find_table("gfx1151", 20U, 4U, 2U, 256U) == nullptr, "other head shape: no table");
    check(rs::find_table("gfx1151", 20U, G, KV, 128U) == nullptr, "other head dim: no table");
    for (unsigned rows = 1U; rows <= 64U; ++rows)
        for (const unsigned len : {1U, 8192U, 49152U, 131072U}) {
            const auto none = rs::plan_device(rows, G, len, nullptr);
            check(none.pairs == 1U && none.walk == 1U && none.blocks == rows * G, "unknown device -> (1, 1)");
        }
    const auto* tb = dt->bands;
    const unsigned nb = dt->n_bands;
    check(nb >= 1U && tb[nb - 1U].max_len == ~0U, "the last band takes every length");
    for (unsigned b = 1U; b < nb; ++b)
        check(tb[b - 1U].max_len < tb[b].max_len, "bands ascending");
    for (unsigned b = 0; b < nb; ++b)
        for (unsigned rows = 1U; rows < rs::table_rows; ++rows) {
            const unsigned len = b == 0U ? tb[0].max_len : tb[b - 1U].max_len + 1U;  // first length of band b
            const auto pl = rs::plan_table(rows, G, len, tb, nb);
            const unsigned form = pl.pairs * 10U + pl.walk;
            check(form == 11U || form == 12U || form == 22U || form == 21U || form == 31U, "table form is one the HIP launch has");
            check(pl.pairs == tb[b].by_rows[rows].pairs && pl.walk == tb[b].by_rows[rows].walk, "length picks its band");
            check(pl.blocks == rs::blocks_for(rows, G, pl.pairs), "table plan blocks");
            const auto off = rs::plan_table(rows, G, len, tb, nb, false);
            check(off.pairs == 1U && off.walk == 1U && off.blocks == rows * G, "off switch: main schedule");
            const auto dev = rs::plan_device(rows, G, len, dt);
            check(dev.pairs == pl.pairs && dev.walk == pl.walk && dev.blocks == pl.blocks, "device plan == its table");
        }
    check(rs::plan_table(1U, G, 1024U, tb, nb).walk == 2U, "1 row: 2 positions per step (measured)");
    check(rs::plan_table(16U, G, 1024U, tb, nb).pairs == 3U, "16 rows: 3 pairs (measured)");
    check(rs::plan_table(4U, G, 8192U, tb, nb).pairs == 1U, "4 rows, 8K: one-pair blocks share K/V in the MALL");
    check(rs::plan_table(4U, G, 49152U, tb, nb).walk == 1U, "4 rows, 48K: the main schedule (measured fastest)");
    check(rs::plan_table(5U, G, 131072U, tb, nb).pairs == 1U && rs::plan_table(5U, G, 131072U, tb, nb).walk == 1U,
          "5 rows, 131K: the main schedule (2 pairs x 2 positions ties it, faster in one of two runs)");
    check(rs::plan_table(6U, G, 16384U, tb, nb).pairs == 1U && rs::plan_table(6U, G, 16384U, tb, nb).walk == 1U,
          "6 rows, <= 32K: the main schedule (measured fastest with the unfused arithmetic)");
    check(rs::plan_table(6U, G, 65536U, tb, nb).pairs == 1U, "6 rows, 64K: the main schedule (no band split for 1-3 %)");
    check(rs::plan_table(6U, G, 131072U, tb, nb).pairs == 2U, "6 rows, 131K: 2 pairs");
    // Rows past the table (the 64-row EXACT prefill windows): the band's last row choice, its blocks for the real rows.
    for (unsigned b = 0; b < nb; ++b)
        for (unsigned rows = rs::table_rows; rows <= 64U; ++rows) {
            const unsigned len = b == 0U ? tb[0].max_len : tb[b - 1U].max_len + 1U;
            const auto pl = rs::plan_table(rows, G, len, tb, nb);
            const auto last = tb[b].by_rows[rs::table_rows - 1U];
            check(pl.pairs == last.pairs && pl.walk == last.walk, "past the table: the last row's choice");
            check(pl.blocks == rs::blocks_for(rows, G, pl.pairs), "past the table: blocks for the real rows");
            const auto off = rs::plan_table(rows, G, len, tb, nb, false);
            check(off.pairs == 1U && off.walk == 1U && off.blocks == rows * G, "past the table, off switch: main schedule");
        }
    check(rs::plan_table(64U, G, 32768U, tb, nb).pairs == 3U && rs::plan_table(64U, G, 32768U, tb, nb).walk == 1U,
          "64 rows, 32K: 3 pairs (measured 8.2x the main schedule)");
    check(rs::plan_table(4U, G, 1024U, tb, 0U).pairs == 1U, "no band: main schedule");
    // Coverage, prefix property and the masked walk, for every rows 1..64 (the EXACT prefill windows) and compiled pair
    // count.
    for (unsigned rows = 1U; rows <= 64U; ++rows)
        for (const unsigned P : rs::candidates) {
            const unsigned blocks = rs::blocks_for(rows, G, P), total = rows * G, pos0 = 1000U;
            std::vector<unsigned> seen(std::size_t(total) * KV, 0U);
            bool prefix = true, order = true;
            for (unsigned kh = 0; kh < KV; ++kh)
                for (unsigned b = 0; b < blocks; ++b) {
                    const unsigned j0 = b * P, n = total - j0 < P ? total - j0 : P;
                    const unsigned top = pos0 + rs::pair_row(j0 + n - 1U, G);
                    for (unsigned i = 0; i < n; ++i) {
                        const unsigned j = j0 + i, row = rs::pair_row(j, G), head = rs::pair_head(j, kh, G);
                        ++seen[std::size_t(row) * KV * G + head];
                        const unsigned last = pos0 + row;
                        prefix = prefix && last <= top && head / G == kh;
                        for (unsigned w = 0; w < waves; ++w)  // block walk (U positions per step) masked by pos <= last
                            for (const unsigned U : {1U, 2U, 4U}) {
                                std::vector<unsigned> one, shared;
                                for (unsigned pos = w; pos <= last; pos += waves) one.push_back(pos);
                                for (unsigned base = w; base <= top; base += waves * U)
                                    for (unsigned u = 0; u < U; ++u) {
                                        const unsigned pos = base + u * waves;
                                        if (pos <= top && pos <= last) shared.push_back(pos);
                                    }
                                order = order && one == shared;
                            }
                    }
                }
            bool once = true;
            for (const unsigned c : seen) once = once && c == 1U;
            check(once, "every (row, q head) pair exactly once");
            check(prefix, "a pair's causal range is a prefix of its block's walk, same kv head");
            check(order, "masked shared walk == one-pair walk per wave residue");
        }
    std::printf("%s: attn_row_share plan (%d failure(s))\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
