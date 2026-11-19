// Host tests of the CUDA decode attention row walks (include/strix/cuda_core/attention_row_walk.hpp):
// 1. the step table a launch carries gives, for every active chunk count, exactly the shared plan's walk length
//    (strix/attn_row_share.hpp, unit = one row);
// 2. off = one row per block in the old order; all = the longest walk at any length;
// 3. walk partition (the kernels' leader rule): every row in exactly one walk, walks <= per rows, never across a cache slot;
// 4. the Qwen 27B dense shape on a 188-SM device (2 resident blocks per SM, floor 752): 1.3K / 32K / 131K lengths;
// 5. the device gate (walk_cap): on the measured key (sm_120, 188 SMs, 24 q / 4 kv, Dim 256) 'automatic' gets the
//    kernels' own walk budgets, so its step tables equal the occupancy plan's (the form the receipts ran); an unmeasured
//    device or head shape keeps one row per block; 'all' ignores the table; every CUDA table holds one band.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strix/cuda_core/attention_row_walk.hpp"

namespace cc = strix::cuda_core;

static int g_fail = 0;
#define CHECK(c, ...)                                  \
    do {                                               \
        if (!(c)) {                                    \
            ++g_fail;                                  \
            std::printf("FAIL line %d: ", __LINE__);   \
            std::printf(__VA_ARGS__);                  \
            std::printf("\n");                         \
        }                                              \
    } while (0)

static void table_equals_plan() {
    const std::uint32_t kvs[] = {1, 4, 8}, mcs[] = {1, 4, 128, 513, 1024}, floors[] = {1, 752, 1128, 5000}, maxes[] = {1, 3, 12};
    for (std::uint32_t rows = 2; rows <= 32; ++rows)
        for (const auto kv : kvs)
            for (const auto mc : mcs)
                for (const auto fl : floors)
                    for (const auto mx : maxes) {
                        const cc::RowWalk rw = cc::row_walk(cc::DecodeRowShare::automatic, rows, kv, mc, mx, fl);
                        CHECK(rw.adjacent == 1u, "automatic walks adjacent");
                        for (std::uint32_t a = 1; a <= mc; ++a) {
                            const unsigned want = strix::attn_row_share::plan(rows, 1u, a * kv, fl, mx).pairs;
                            const std::uint32_t got = cc::rows_per_walk(rw, a);
                            CHECK(got == want, "rows %u kv %u mc %u floor %u max %u active %u: table %u plan %u", rows, kv, mc, fl, mx, a, got, want);
                        }
                    }
}

static void modes() {
    const cc::RowWalk off = cc::row_walk(cc::DecodeRowShare::off, 12, 4, 513, 12, 752);
    CHECK(off.adjacent == 0u && off.steps == 0u && cc::rows_per_walk(off, 513) == 1u, "off = one row per block, chunk fastest");
    const cc::RowWalk one = cc::row_walk(cc::DecodeRowShare::automatic, 1, 4, 513, 12, 752);
    CHECK(one.adjacent == 0u && cc::rows_per_walk(one, 513) == 1u, "one row = the one-row form");
    const cc::RowWalk all = cc::row_walk(cc::DecodeRowShare::all, 12, 4, 513, 12, 752);
    CHECK(cc::rows_per_walk(all, 1) == 12u && cc::rows_per_walk(all, 513) == 12u, "all = longest walk at any length");
    const cc::RowWalk all3 = cc::row_walk(cc::DecodeRowShare::all, 12, 4, 513, 3, 752);
    CHECK(cc::rows_per_walk(all3, 1) == 3u, "all respects the kernel's walk budget");
}

static void partition() {
    const std::vector<std::vector<std::uint32_t>> slot_sets = {
        std::vector<std::uint32_t>(16, 0u), {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1}, {0, 1, 2, 3}, {5, 5, 9, 9, 9, 9, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2}};
    const std::uint32_t pers[] = {1, 2, 3, 6, 12};
    for (const auto& slots : slot_sets)
        for (const auto per : pers) {
            const auto rows = static_cast<std::uint32_t>(slots.size());
            std::vector<int> owner(rows, -1);
            std::uint32_t start = 0;
            for (std::uint32_t r = 0; r < rows; ++r) {
                if (r > 0 && slots[r] != slots[r - 1]) start = r;
                if (!cc::walk_start(r, start, per)) continue;
                // the kernel's walk: r .. next leader - 1
                std::uint32_t n = 1, s2 = start;
                for (std::uint32_t k = r + 1; k < rows; ++k) {
                    if (slots[k] != slots[k - 1]) s2 = k;
                    if (cc::walk_start(k, s2, per)) break;
                    ++n;
                }
                CHECK(n <= per, "walk of %u rows > per %u", n, per);
                for (std::uint32_t k = r; k < r + n; ++k) {
                    CHECK(owner[k] < 0, "row %u in two walks", k);
                    CHECK(slots[k] == slots[r], "walk from row %u crosses a cache slot at row %u", r, k);
                    owner[k] = static_cast<int>(r);
                }
            }
            for (std::uint32_t r = 0; r < rows; ++r) CHECK(owner[r] >= 0, "row %u in no walk (per %u)", r, per);
        }
}

static void qwen27b_shape() {
    // 24 q / 4 kv heads, capacity 262,144 positions (1,024 chunks), 12 verify rows, floor = 2 waves x 188 SMs x 2 blocks.
    const cc::RowWalk sc = cc::row_walk(cc::DecodeRowShare::automatic, 12, 4, 1024, 12, 752);
    const cc::RowWalk va = cc::row_walk(cc::DecodeRowShare::automatic, 12, 4, 1024, cc::value_walk_rows(cc::attn_group_width(6)), 752);
    CHECK(cc::rows_per_walk(sc, 6) == 1u && cc::rows_per_walk(va, 6) == 1u, "1.3K (6 chunks): one row per block");
    CHECK(cc::rows_per_walk(sc, 128) == 6u && cc::rows_per_walk(va, 128) == 6u, "32K (128 chunks): scores 6 rows, values 6");
    CHECK(cc::rows_per_walk(sc, 512) == 12u && cc::rows_per_walk(va, 512) == 6u, "131K (512 chunks): scores 12 rows, values 6");
    std::printf("qwen27b 12 rows: scores walk 1.3K %u 32K %u 131K %u; values walk %u %u %u\n", cc::rows_per_walk(sc, 6),
                cc::rows_per_walk(sc, 128), cc::rows_per_walk(sc, 512), cc::rows_per_walk(va, 6), cc::rows_per_walk(va, 128),
                cc::rows_per_walk(va, 512));
}

// Persistent item loop of the shared walks: for any walk count, kv heads, active chunks and grid size, the items taken by
// the blocks (b, b + B, ...) below active * kv * walks visit every (chunk < active, kv head, walk) exactly once, walks of one
// chunk adjacent. Group widths / values walks: rows x width within the shared weight row.
static void items() {
    for (std::uint32_t walks = 1; walks <= 32; walks += 3)
        for (const std::uint32_t kv : {1u, 2u, 4u, 8u})
            for (const std::uint32_t active : {1u, 6u, 128u, 513u})
                for (const std::uint32_t grid : {1u, 7u, 376u, 752u, 5000u}) {
                    const std::uint32_t n = active * kv * walks;
                    std::vector<int> seen(n, 0);
                    for (std::uint32_t b = 0; b < grid; ++b)
                        for (std::uint32_t it = b; it < n; it += grid) {
                            const cc::RowWalkItem w = cc::row_walk_item(it, walks, kv);
                            CHECK(w.walk < walks && w.kv_head < kv && w.chunk < active, "item %u out of range", it);
                            if (w.walk < walks && w.kv_head < kv && w.chunk < active) ++seen[(w.chunk * kv + w.kv_head) * walks + w.walk];
                            if (it + 1 < n && (it + 1) % walks != 0)
                                CHECK(cc::row_walk_item(it + 1, walks, kv).chunk == w.chunk, "walks of a chunk not adjacent");
                        }
                    for (std::uint32_t i = 0; i < n; ++i)
                        CHECK(seen[i] == 1, "walks %u kv %u active %u grid %u: item %u seen %d times", walks, kv, active, grid, i, seen[i]);
                }
    for (std::uint32_t g = 1; g <= 8; ++g) {
        const std::uint32_t gt = cc::attn_group_width(g), wr = cc::value_walk_rows(gt);
        CHECK(gt >= g && (gt == 4u || gt == 6u || gt == 8u), "group %u width %u", g, gt);
        CHECK(wr * gt <= cc::kValueWalkPairs && wr >= 1u, "values walk %u x width %u exceeds the weight row", wr, gt);
    }
    CHECK(cc::value_walk_rows(4) == 6u && cc::value_walk_rows(6) == 6u && cc::value_walk_rows(8) == 3u, "values walks 6 / 6 / 3");
}

// 6. the initial mode follows STRIX_DECODE_ATTN_ROW_SHARE (the HIP dense decode's switch): unset / "1" / other = automatic,
//    "0" = off.
static void set_switch(const char* v) {
#if defined(_WIN32)
    _putenv_s("STRIX_DECODE_ATTN_ROW_SHARE", v ? v : "");
#else
    if (v) setenv("STRIX_DECODE_ATTN_ROW_SHARE", v, 1);
    else unsetenv("STRIX_DECODE_ATTN_ROW_SHARE");
#endif
}

static void env_switch() {
    const struct {
        const char* value;
        cc::DecodeRowShare want;
    } cases[] = {{nullptr, cc::DecodeRowShare::automatic},
                 {"1", cc::DecodeRowShare::automatic},
                 {"0", cc::DecodeRowShare::off},
                 {"yes", cc::DecodeRowShare::automatic}};
    for (const auto& c : cases) {
        set_switch(c.value);
        CHECK(cc::row_share_from_env() == c.want, "STRIX_DECODE_ATTN_ROW_SHARE=%s: mode %u, want %u", c.value ? c.value : "(unset)",
              static_cast<unsigned>(cc::row_share_from_env()), static_cast<unsigned>(c.want));
    }
    set_switch(nullptr);
}

static void device_gate() {
    namespace rs = strix::attn_row_share;
    CHECK(cc::cuda_arch_key(12, 0) == "sm_120" && cc::cuda_arch_key(8, 9) == "sm_89", "arch key");
    const rs::DeviceTable* pro = rs::find_table(cc::cuda_arch_key(12, 0), 188u, 6u, 4u, 256u);
    CHECK(pro != nullptr, "measured CUDA key has a table");
    CHECK(rs::find_table(cc::cuda_arch_key(8, 9), 128u, 6u, 4u, 256u) == nullptr, "unmeasured CUDA device: no table");
    CHECK(rs::find_table("sm_120", 170u, 6u, 4u, 256u) == nullptr, "same arch, other SM count: no table");
    CHECK(rs::find_table("sm_120", 188u, 8u, 4u, 256u) == nullptr && rs::find_table("sm_120", 188u, 6u, 4u, 128u) == nullptr,
          "other head shape: no table");
    CHECK(rs::find_table("", 0u, 6u, 4u, 256u) == nullptr, "unreadable device: no table");
    for (const rs::DeviceTable& t : rs::dense_decode_tables)
        if (t.arch.substr(0, 3) == "sm_") CHECK(t.n_bands == 1u && t.bands[0].max_len == ~0u, "CUDA table %.*s: one band", int(t.arch.size()), t.arch.data());
    const std::uint32_t budgets[] = {12u, cc::value_walk_rows(4), cc::value_walk_rows(6), cc::value_walk_rows(8)};
    const std::uint32_t kvs[] = {4u}, mcs[] = {1u, 6u, 128u, 513u}, floors[] = {752u, 1128u};
    for (std::uint32_t rows = 1; rows <= 40; ++rows)
        for (const auto mx : budgets) {
            const std::uint32_t cap = cc::walk_cap(cc::DecodeRowShare::automatic, pro, rows, mx);
            CHECK(cap == (rows < 2 ? 1u : mx), "measured key rows %u budget %u: cap %u (the budget: the cap never binds)", rows, mx, cap);
            CHECK(cc::walk_cap(cc::DecodeRowShare::automatic, nullptr, rows, mx) == 1u, "no table rows %u: one row per block", rows);
            CHECK(cc::walk_cap(cc::DecodeRowShare::off, pro, rows, mx) == 1u, "off rows %u", rows);
            CHECK(cc::walk_cap(cc::DecodeRowShare::all, nullptr, rows, mx) == (rows < 2 ? 1u : mx), "all rows %u ignores the table", rows);
            if (rows < 2) continue;
            for (const auto kv : kvs)
                for (const auto mc : mcs)
                    for (const auto fl : floors) {
                        const cc::RowWalk a = cc::row_walk(cc::DecodeRowShare::automatic, rows, kv, mc, cap, fl);
                        const cc::RowWalk b = cc::row_walk(cc::DecodeRowShare::automatic, rows, kv, mc, mx, fl);
                        bool same = a.adjacent == b.adjacent && a.steps == b.steps;
                        for (std::uint32_t i = 0; same && i < a.steps; ++i) same = a.rows[i] == b.rows[i] && a.min_active[i] == b.min_active[i];
                        CHECK(same, "rows %u budget %u mc %u floor %u: gated step table != occupancy plan", rows, mx, mc, fl);
                    }
        }
}

int main() {
    env_switch();
    items();
    table_equals_plan();
    modes();
    partition();
    qwen27b_shape();
    device_gate();
    std::printf(g_fail ? "attention_row_walk_host_tests: FAILED (%d)\n" : "attention_row_walk_host_tests: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
