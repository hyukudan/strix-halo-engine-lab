// Host tests of strix/pair/tp_layout.hpp (exact tensor-parallel decode layouts):
//   - mod_value_heads: for key/value 16/48 (Qwen 27B GDN) and other ratios, N ranks = 1..4 with uneven rates, every
//     value head is owned once, and local value j reads local key j % nk == the global rule h % key_heads;
//   - join: ranks select their local columns from a full row (select_map), write [rows][width] into padded slots with
//     garbage after their width; the join equals the full rows word for word (trunk shares, GDN core, attention ctx);
//   - split argmax: a CPU copy of argmax_kernel (1024 strided scans + the fixed tree) on the whole row equals the fold
//     of per-part picks with `beats` for random partitions, rows with ties, -inf and NaN, and the empty-row sentinel.
//   - FP16 gathers: per-rank conversion + join into the prepare placement == prepare of the joined row; sweep offsets;
//   - split draft head: per-share picks over the live subset columns fold to the one-rank subset pick.
//   - head columns: split / replicated vocabulary head coverage of [0, V) for every format, V and rates.
#include "strix/pair/tp_layout.hpp"

#include "core/iu4_x_pairs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}
namespace tp = strix::pair::tp;

std::vector<double> rand_rates(std::mt19937& g, std::size_t n) {
    std::uniform_real_distribution<double> d(0.8, 1.25);
    std::vector<double> r(n);
    for (auto& x : r) x = d(g);
    return r;
}

void test_mod_heads() {
    std::mt19937 g(7);
    for (const auto& kv : {std::pair<std::uint32_t, std::uint32_t>{16U, 48U}, {16U, 32U}, {8U, 24U}, {4U, 4U}}) {
        for (std::size_t n = 1; n <= 4; ++n) {
            const auto rates = rand_rates(g, n);
            const auto keys = tp::split_units(kv.first, rates);
            std::vector<int> owner(kv.second, -1);
            for (std::size_t q = 0; q < n; ++q) {
                const auto segs = tp::mod_value_heads(kv.first, kv.second, keys[q]);
                const auto sel = tp::select_map(segs);
                const std::uint32_t nk = keys[q].size();
                check(sel.size() == std::size_t(kv.second / kv.first) * nk, "value heads per rank = ratio x key heads");
                for (std::size_t j = 0; j < sel.size(); ++j) {
                    const auto h = static_cast<std::uint32_t>(sel[j]);
                    check(owner[h] < 0, "value head owned once");
                    owner[h] = int(q);
                    // the kernel on the compact arrays reads local key j % nk = global key keys.begin + j % nk
                    check(keys[q].begin + std::uint32_t(j) % nk == h % kv.first, "local modulo rule == global modulo rule");
                }
            }
            for (auto o : owner) check(o >= 0, "every value head owned");
        }
    }
}

// Ranks write [rows][width] local rows into their slot (garbage after the width); join == full rows bitwise.
void join_case(std::mt19937& g, const std::vector<std::vector<tp::Segment>>& per_rank, std::uint32_t cols, unsigned rows, const char* what) {
    std::uint32_t maxw = 0U;
    for (const auto& s : per_rank) maxw = std::max(maxw, tp::width(s));
    const std::size_t slot = std::size_t(16U) * maxw;  // slots sized for 16 rows; `rows` may be fewer
    const auto m = tp::join_map(per_rank, cols, slot);
    check(m.max_width == maxw, "join max width");
    std::uniform_int_distribution<std::uint32_t> bits;
    std::vector<float> full(std::size_t(rows) * cols), slots(per_rank.size() * slot);
    for (auto& f : full) {
        const std::uint32_t b = bits(g);
        std::memcpy(&f, &b, 4);  // any bit pattern, NaN payloads included: the join moves words
    }
    for (auto& s : slots) s = -12345.0F;
    for (std::size_t q = 0; q < per_rank.size(); ++q) {
        const auto sel = tp::select_map(per_rank[q]);
        const std::size_t w = sel.size();
        for (unsigned r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < w; ++j) slots[q * slot + r * w + j] = full[std::size_t(r) * cols + std::size_t(sel[j])];
    }
    bool same = true;
    for (unsigned r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < cols; ++c) {
            const float v = slots[std::size_t(m.base[c]) + std::size_t(r) * std::size_t(m.stride[c])];
            same = same && std::memcmp(&v, &full[std::size_t(r) * cols + c], 4) == 0;
        }
    check(same, what);
    // FP16 gathers (engine tp_join_x): each rank applies prepare's elementwise conversion f to its slot's first
    // rows x maxw words, the join writes xh[pos(r * cols + c)]. Must equal prepare of the joined FP32 rows:
    // xh[pos(i)] = f(full[i]), pos = iu4_pairs::paired_index (scalar_x) or identity. f is any function of the bits.
    const auto f = [](float v) {
        std::uint32_t b = 0;
        std::memcpy(&b, &v, 4);
        b ^= b >> 15;
        b *= 0x2c1b3c6dU;
        b ^= b >> 12;
        return static_cast<std::uint16_t>(b);
    };
    for (const bool paired : {false, true}) {
        const auto pos = [paired](std::size_t i) { return paired ? strix::core::iu4_pairs::paired_index(i) : i; };
        const std::size_t n = std::size_t(rows) * cols;
        std::vector<std::uint16_t> ref(n, 0), got(n, 0), hs(slots.size(), 0);
        std::vector<unsigned> hit(n, 0U);
        for (std::size_t i = 0; i < n; ++i) ref[pos(i)] = f(full[i]);
        for (std::size_t q = 0; q < per_rank.size(); ++q)
            for (std::size_t k = 0; k < std::size_t(rows) * maxw; ++k) hs[q * slot + k] = f(slots[q * slot + k]);
        for (unsigned r = 0; r < rows; ++r)
            for (std::uint32_t c = 0; c < cols; ++c) {
                const std::size_t p = pos(std::size_t(r) * cols + c);
                got[p] = hs[std::size_t(m.base[c]) + std::size_t(r) * std::size_t(m.stride[c])];
                ++hit[p];
            }
        check(got == ref, paired ? "fp16 join (paired) == prepare of the joined row" : "fp16 join == prepare of the joined row");
        check(std::all_of(hit.begin(), hit.end(), [](unsigned h) { return h == 1U; }), "fp16 join placement is a permutation");
    }
}

// The GEMV sweeps read joined rows at xh + r * K: prepare's placement of rows [r, r + m) alone equals the joined
// placement shifted by r * K (K % 8 == 0), for the trunk widths and every sweep start of a 16-row window.
void test_sweep_offsets() {
    for (const std::size_t K : {5120U, 6144U, 10240U, 17408U}) {
        bool ok = true;
        for (std::size_t r = 0; r < 16U; ++r)
            for (std::size_t i = 0; i < 16U * K; i += 7U) {
                if (i + r * K >= 16U * K) break;
                ok = ok && strix::core::iu4_pairs::paired_index(i + r * K) == strix::core::iu4_pairs::paired_index(i) + r * K;
            }
        check(ok, "paired placement commutes with whole-row offsets");
    }
}

void test_join() {
    std::mt19937 g(11);
    for (std::size_t n = 1; n <= 4; ++n)
        for (unsigned rows : {1U, 3U, 8U, 16U}) {
            const auto rates = rand_rates(g, n);
            // trunk: 64-column blocks of a 5120-wide row
            std::vector<std::vector<tp::Segment>> trunk;
            for (const auto& s : tp::split_units(80U, rates)) trunk.push_back({{s.begin * 64U, s.size() * 64U}});
            join_case(g, trunk, 5120U, rows, "trunk share join == full row");
            // GDN core: 48 value heads x 128 by key heads 16
            std::vector<std::vector<tp::Segment>> core;
            for (const auto& k : tp::split_units(16U, rates)) core.push_back(tp::expand(tp::mod_value_heads(16U, 48U, k), 128U));
            join_case(g, core, 6144U, rows, "gdn core join == full row");
            // GDN qkv local columns: q keys, k keys (+2048), v value heads (+4096); the join of all ranks == full qkv row
            std::vector<std::vector<tp::Segment>> qkv;
            for (const auto& k : tp::split_units(16U, rates)) {
                std::vector<tp::Segment> s{{k.begin * 128U, k.size() * 128U}, {2048U + k.begin * 128U, k.size() * 128U}};
                for (const auto& v : tp::expand(tp::mod_value_heads(16U, 48U, k), 128U, 4096U)) s.push_back(v);
                qkv.push_back(s);
            }
            join_case(g, qkv, 10240U, rows, "gdn qkv local columns cover the row once");
            // attention ctx: kv heads 4, 6 q heads each, 256 per head
            std::vector<std::vector<tp::Segment>> ctx;
            for (const auto& k : tp::split_units(4U, rates)) ctx.push_back({{k.begin * 6U * 256U, k.size() * 6U * 256U}});
            join_case(g, ctx, 6144U, rows, "attention ctx join == full row");
        }
    bool threw = false;
    try {
        const std::vector<std::vector<tp::Segment>> bad{{{0U, 64U}}, {{32U, 64U}}};
        (void)tp::join_map(bad, 96U, 64U);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a column owned twice is refused");
}

// CPU copy of argmax_kernel: thread t scans t, t+1024, ... with a strict '>', then the fixed tree.
tp::ArgPick argmax_kernel_cpu(const float* row, std::uint32_t n, std::int32_t index0) {
    std::vector<float> bv(1024);
    std::vector<std::int32_t> bi(1024);
    for (std::uint32_t t = 0; t < 1024U; ++t) {
        float best = -std::numeric_limits<float>::infinity();
        std::int32_t idx = 0x7fffffff;
        for (std::uint32_t i = t; i < n; i += 1024U)
            if (row[i] > best) { best = row[i]; idx = std::int32_t(i) + index0; }
        bv[t] = best;
        bi[t] = idx;
    }
    for (std::uint32_t s = 512U; s > 0U; s >>= 1U)
        for (std::uint32_t t = 0; t < s; ++t) {
            const float v = bv[t + s];
            const std::int32_t i = bi[t + s];
            if (v > bv[t] || (v == bv[t] && i < bi[t])) { bv[t] = v; bi[t] = i; }
        }
    return {bv[0], bi[0]};
}

void test_argmax() {
    std::mt19937 g(5);
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    for (int it = 0; it < 300; ++it) {
        const std::uint32_t n = 1000U + std::uint32_t(g() % 9000U);
        std::vector<float> row(n);
        const int mode = it % 5;
        for (auto& v : row) {
            const auto u = static_cast<std::uint32_t>(g() % 100U);
            if (mode == 4) v = u < 50U ? -inf : nan;  // no candidate: sentinel
            else if (u < 3U && mode >= 2) v = nan;
            else if (u < 6U && mode >= 3) v = -inf;
            else v = float(int(g() % 7U)) - (u < 50U ? 0.0F : -0.0F);  // many exact ties, +0 and -0
        }
        const auto whole = argmax_kernel_cpu(row.data(), n, 0);
        const std::size_t parts = 1U + g() % 4U;
        std::vector<std::uint32_t> cuts{0U, n};
        for (std::size_t p = 1; p < parts; ++p) cuts.push_back(static_cast<std::uint32_t>(g() % (n + 1U)));
        std::sort(cuts.begin(), cuts.end());
        tp::ArgPick fold;
        for (std::size_t p = 0; p + 1 < cuts.size(); ++p) {
            const auto pick = argmax_kernel_cpu(row.data() + cuts[p], cuts[p + 1] - cuts[p], std::int32_t(cuts[p]));
            if (tp::beats(pick, fold)) fold = pick;
        }
        check(fold.index == whole.index, "split argmax index == one-rank argmax_kernel index");
    }
}

// Split draft head (engine mtp_run_tp): the compact subset pack has `stride` columns (64-blocks split by rate), only
// the first `live` are rows; rank q scans [c0, min(c0 + cn, live)) with index0 = c0, the fold with `beats` equals
// draft_argmax_map_kernel's pick over [0, live) (the same scan / tree as argmax_kernel), sentinel included.
void test_draft_split() {
    std::mt19937 g(23);
    std::uniform_real_distribution<float> d(-4.0F, 4.0F);
    for (unsigned trial = 0; trial < 200U; ++trial) {
        const std::uint32_t units = 4U + trial % 37U, stride = units * 64U;  // >= ranks: every share non-empty
        const std::uint32_t live = trial % 11U == 0U ? 0U : 1U + std::uint32_t(g() % stride);
        std::vector<float> row(stride);
        for (auto& v : row) v = trial % 5U == 0U ? std::round(d(g)) : d(g);  // rounded rows: many ties
        if (trial % 7U == 0U) row[g() % stride] = std::numeric_limits<float>::quiet_NaN();
        if (trial % 13U == 0U) std::fill(row.begin(), row.end(), -std::numeric_limits<float>::infinity());
        const auto rates = rand_rates(g, 1U + trial % 4U);
        tp::ArgPick fold;
        for (const auto& s : tp::split_units(units, rates)) {
            const std::uint32_t c0 = s.begin * 64U, cn = s.size() * 64U, cnt = live > c0 ? std::min(cn, live - c0) : 0U;
            const auto pick = argmax_kernel_cpu(row.data() + c0, cnt, std::int32_t(c0));
            if (tp::beats(pick, fold)) fold = pick;
        }
        const auto one = argmax_kernel_cpu(row.data(), live, 0);
        check(fold.index == one.index, "split draft-subset argmax == one-rank subset argmax");
    }
}

// The vocabulary head: whatever the format / V / rates, the columns the ranks compute cover [0, V): split shares tile it
// exactly once in 64-column blocks, a head that cannot split (non-IU4, V % 64 != 0) is replicated with [0, V) per rank.
// (Regression: a non-splittable head used to get a one-unit split = 64 columns on rank 0 and none on rank 1.)
void test_head_columns() {
    const std::vector<std::vector<double>> rate_sets{{200.0, 200.0}, {300.0, 100.0}, {1.0, 2.0, 3.0}, {5.0}};
    const std::uint32_t vocabs[] = {248320U, 248321U, 248383U, 64U, 128U, 151936U};
    for (const auto& rates : rate_sets)
        for (const std::uint32_t V : vocabs)
            for (const bool splittable : {true, false}) {
                const auto h = tp::head_columns(splittable, V, rates);
                check(h.split == (splittable && V % 64U == 0U), "split exactly when the format allows it and V % 64 == 0");
                check(h.cols.size() == rates.size(), "one column range per rank");
                std::vector<unsigned> cover(V, 0U);
                for (const auto& c : h.cols) {
                    check(std::uint64_t(c.global) + c.count <= V, "range inside the vocabulary");
                    if (h.split) check(c.global % 64U == 0U && c.count % 64U == 0U, "split ranges are whole 64-column blocks");
                    for (std::uint32_t i = 0; i < c.count && c.global + i < V; ++i) ++cover[c.global + i];
                }
                const unsigned want = h.split ? 1U : unsigned(rates.size());
                bool all = true;
                for (const unsigned k : cover) all = all && k == want;
                check(all, h.split ? "split head: every column on exactly one rank" : "replicated head: every column on every rank");
            }
    bool threw = false;
    try {
        (void)tp::head_columns(true, 0U, rate_sets[0]);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "empty vocabulary refused");
}

}  // namespace

int main() {
    test_head_columns();
    test_mod_heads();
    test_join();
    test_sweep_offsets();
    test_argmax();
    test_draft_split();
    if (failures) {
        std::printf("pair_tp_layout_tests: %d failures\n", failures);
        return 1;
    }
    std::printf("pair_tp_layout_tests: PASS\n");
    return 0;
}
