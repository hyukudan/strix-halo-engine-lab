// Host oracle for the row-group union (strix/attn_rowgroup_union.hpp), the invariant of the shared row-group attention core:
//   every row r sees exactly its own selected keys (with multiplicity) among the union entries carrying mask bit r;
//   for strictly ascending lists the union is the sorted set union, no key twice, positions 0..U-1 all written.
// The device union_body (core/attention_rowgroup.cuh) runs these same functions in parallel: owned flags, per-list exclusive
// prefix of owned keys, rank_and_mask scatter; a non-ascending list switches the tile to the concatenation. This test runs
// that algorithm sequentially and compares it with std::set reasoning. No GPU.
#include "strix/attn_rowgroup_union.hpp"
#include "../../src/cuda/core/rowgroup_union.hpp"  // forwarding declarations

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace rg = strix::attn::rowgroup;
// Same entity: two non-type template arguments naming the same function are the same specialization.
template <auto A, auto B> struct SameEntity { static constexpr bool value = false; };
template <auto A> struct SameEntity<A, A> { static constexpr bool value = true; };
static_assert(SameEntity<&rg::lower_bound, &strix::cuda::core::rowgroup::lower_bound>::value &&
              SameEntity<&rg::owned, &strix::cuda::core::rowgroup::owned>::value,
              "the forwarding declarations name the shared functions");

namespace {
constexpr unsigned R = 4, Cap = 2051;
int failures = 0;
void check(bool ok, const char* what, int c) {
    if (!ok) { ++failures; std::printf("FAIL case %d: %s\n", c, what); }
}

struct Result { std::vector<std::uint32_t> tok; std::vector<std::uint8_t> mask; bool concat = false; };

// Sequential run of the device algorithm (same helper functions, same layout: lists [R][Cap], pre [R][Cap + 1]).
Result build(const std::vector<std::vector<std::uint32_t>>& in) {
    std::vector<std::uint32_t> lists(R * Cap, 0U);
    std::vector<std::uint16_t> pre(R * (Cap + 1U), 0U);
    unsigned n[R];
    bool irregular = false;
    for (unsigned r = 0; r < R; ++r) {
        n[r] = static_cast<unsigned>(in[r].size());
        std::copy(in[r].begin(), in[r].end(), lists.begin() + r * Cap);
        irregular |= !rg::ascending(lists.data() + r * Cap, n[r]);
    }
    Result out;
    if (irregular) {
        out.concat = true;
        for (unsigned r = 0; r < R; ++r)
            for (unsigned j = 0; j < n[r]; ++j) { out.tok.push_back(lists[r * Cap + j]); out.mask.push_back(static_cast<std::uint8_t>(1U << r)); }
        return out;
    }
    unsigned total = 0;
    for (unsigned r = 0; r < R; ++r) {
        unsigned run = 0;
        for (unsigned j = 0; j < n[r]; ++j) {
            pre[r * (Cap + 1U) + j] = static_cast<std::uint16_t>(run);
            run += rg::owned(lists.data(), Cap, n, r, lists[r * Cap + j]) ? 1U : 0U;
        }
        pre[r * (Cap + 1U) + n[r]] = static_cast<std::uint16_t>(run);
        total += run;
    }
    out.tok.assign(total, 0xffffffffU);
    out.mask.assign(total, 0U);
    std::vector<int> writes(total, 0);
    for (unsigned r = 0; r < R; ++r)
        for (unsigned j = 0; j < n[r]; ++j) {
            if (!rg::owned(lists.data(), Cap, n, r, lists[r * Cap + j])) continue;
            unsigned m = 0;
            const unsigned at = rg::rank_and_mask<R>(lists.data(), Cap, n, pre.data(), Cap + 1U, r, j, m);
            if (at >= total) { ++failures; std::printf("FAIL rank %u >= total %u\n", at, total); continue; }
            out.tok[at] = lists[r * Cap + j];
            out.mask[at] = static_cast<std::uint8_t>(m);
            ++writes[at];
        }
    for (unsigned i = 0; i < total; ++i)
        if (writes[i] != 1) { ++failures; std::printf("FAIL position %u written %d times\n", i, writes[i]); break; }
    return out;
}

// Per-row view of the result must equal the row's own list as a multiset; for ascending input also the sorted set union.
double verify(const std::vector<std::vector<std::uint32_t>>& in, const Result& res, int c) {
    std::size_t mean = 0;
    for (unsigned r = 0; r < R; ++r) {
        std::multiset<std::uint32_t> want(in[r].begin(), in[r].end()), got;
        for (std::size_t i = 0; i < res.tok.size(); ++i)
            if ((res.mask[i] >> r) & 1U) got.insert(res.tok[i]);
        check(want == got, "row sees exactly its own keys", c);
        mean += in[r].size();
    }
    if (!res.concat) {
        std::set<std::uint32_t> all;
        for (const auto& l : in) all.insert(l.begin(), l.end());
        check(std::vector<std::uint32_t>(all.begin(), all.end()) == res.tok, "union = sorted set union", c);
        for (std::size_t i = 0; i < res.mask.size(); ++i) check(res.mask[i] != 0U && res.mask[i] < (1U << R), "mask in range", c);
    }
    return mean ? static_cast<double>(res.tok.size()) * R / static_cast<double>(mean) : 0.0;
}

std::mt19937 rng(20261117);
// Qwen-shaped selection of the row at `position`: dense 0..p below 2,051 tokens, else 512 ascending 4-token blocks drawn as
// (1 - churn) from a shared base set and churn fresh, then the tail tokens.
std::vector<std::uint32_t> qwen_row(std::uint64_t position, const std::vector<std::uint32_t>& base, double churn) {
    const auto visible = static_cast<std::uint32_t>(position + 1U), nb = visible / 4U;
    std::vector<std::uint32_t> s;
    if (nb <= 512U) { for (std::uint32_t t = 0; t < visible; ++t) s.push_back(t); return s; }
    std::set<std::uint32_t> blocks;
    for (const auto b : base) if (b < nb && std::uniform_real_distribution<double>(0, 1)(rng) >= churn) blocks.insert(b);
    while (blocks.size() < 512U) blocks.insert(static_cast<std::uint32_t>(rng() % nb));
    while (blocks.size() > 512U) blocks.erase(std::next(blocks.begin(), static_cast<long>(rng() % blocks.size())));
    for (const auto b : blocks) for (std::uint32_t o = 0; o < 4U; ++o) s.push_back(b * 4U + o);
    for (std::uint32_t t = nb * 4U; t < visible; ++t) s.push_back(t);
    return s;
}
}  // namespace

int main() {
    int c = 0;
    // 1. Qwen-shaped tiles at several positions and overlaps (churn 0 = identical rows, 1 = independent rows).
    for (const std::uint64_t p0 : {0ULL, 17ULL, 2044ULL, 2047ULL, 2050ULL, 4000ULL, 8072ULL, 30000ULL})
        for (const double churn : {0.0, 0.12, 0.5, 1.0}) {
            ++c;
            std::vector<std::uint32_t> base;
            const auto nb = static_cast<std::uint32_t>((p0 + R) / 4U);
            for (std::uint32_t b = 0; b < nb; ++b) base.push_back(b);
            std::shuffle(base.begin(), base.end(), rng);
            if (base.size() > 512U) base.resize(512U);
            std::vector<std::vector<std::uint32_t>> in(R);
            for (unsigned r = 0; r < R; ++r) in[r] = qwen_row(p0 + r, base, churn);
            const auto res = build(in);
            const double inflation = verify(in, res, c);
            check(!res.concat, "ascending input takes the rank form", c);
            if (p0 + R <= 2051U) check(res.tok.size() == in[R - 1U].size(), "dense rows: union = last row's prefix", c);
            if (churn == 0.12 && p0 == 8072ULL) std::printf("info: p0 8072 churn 0.12 inflation %.3f (union %zu)\n", inflation, res.tok.size());
        }
    // 2. Tile tail: rows 2 and 3 absent (count 0).
    {
        ++c;
        std::vector<std::vector<std::uint32_t>> in(R);
        for (std::uint32_t t = 0; t < 9U; ++t) in[0].push_back(t);
        for (std::uint32_t t = 0; t < 10U; ++t) in[1].push_back(t);
        const auto res = build(in);
        verify(in, res, c);
        check(res.tok.size() == 10U && res.mask[9] == 2U && res.mask[0] == 3U, "tail tile masks", c);
    }
    // 3. Degenerate selection (tokens 0..3 repeated before the blocks, select_kernel_body `zeros`): concatenation.
    {
        ++c;
        std::vector<std::vector<std::uint32_t>> in(R);
        for (unsigned r = 0; r < R; ++r) {
            for (std::uint32_t k = 0; k < 8U; ++k) in[r].push_back(k % 4U);
            for (std::uint32_t b = 10U + r; b < 600U; b += 2U) for (std::uint32_t o = 0; o < 4U; ++o) in[r].push_back(b * 4U + o);
        }
        const auto res = build(in);
        verify(in, res, c);
        check(res.concat, "non-ascending list takes the concatenation", c);
    }
    // 4. Random strictly ascending sets of mixed sizes (full Cap, single key, disjoint ranges).
    for (int k = 0; k < 40; ++k) {
        ++c;
        std::vector<std::vector<std::uint32_t>> in(R);
        for (unsigned r = 0; r < R; ++r) {
            const unsigned size = k % 5 == 0 ? Cap : 1U + static_cast<unsigned>(rng() % Cap);
            const std::uint32_t lo = k % 7 == 0 ? r * 100000U : 0U, span = 3000U + static_cast<std::uint32_t>(rng() % 20000U);
            std::set<std::uint32_t> s;
            while (s.size() < std::min<unsigned>(size, span)) s.insert(lo + static_cast<std::uint32_t>(rng() % span));
            in[r].assign(s.begin(), s.end());
        }
        verify(in, build(in), c);
    }
    std::printf(failures ? "ROWGROUP_UNION FAIL (%d) over %d cases\n" : "ROWGROUP_UNION PASS (%d failures) over %d cases\n", failures, c);
    return failures ? 1 : 0;
}
