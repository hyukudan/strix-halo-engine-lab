// Shared attention core, vendor-free (host + CUDA/HIP device): the UNION of R neighbouring query rows' selected-key
// lists, for row-group attention.
//
// A row group walks one key list for R rows instead of R lists, so every K/V row it stages is read once from L2 and used
// by each row that selected it. The list is the sorted set union of the rows' lists; entry u carries mask bit r when
// row r selected that key. Building it in parallel (no serial merge):
//   - a key x of row r is OWNED by r when no lower row r' < r holds x, so every distinct key has exactly one owner;
//   - pre[r][j] = number of owned keys among list r's first j entries (exclusive prefix, pre[r][n_r] = total);
//   - the union position of an owned key x is the number of distinct keys below x = sum over rows r' of
//     pre[r'][lower_bound(list r', x)] (for r' = r that bound is j itself);
//   - its mask is bit r plus bit r'' for every higher row r'' > r holding x (lower rows cannot hold an owned key).
// Valid only when every list is strictly ascending. A tile with a non-ascending list (a degenerate selection that repeats
// keys) takes the concatenation instead: entries keep their row's multiplicity, one mask bit each. Attention over the
// group is a sum over (row, key) pairs, so both layouts give every row exactly its own keys.
// Used by src/cuda/core/attention_rowgroup.cuh (device) and tests/cuda/rowgroup_union_host_test.cpp (host oracle).
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_RG_HD __host__ __device__ __forceinline__
#else
#define STRIX_RG_HD inline
#endif

namespace strix::attn::rowgroup {

// Number of entries of the ascending list a[0..n) that are < x.
STRIX_RG_HD unsigned lower_bound(const std::uint32_t* a, unsigned n, std::uint32_t x) {
    unsigned lo = 0, hi = n;
    while (lo < hi) {
        const unsigned mid = (lo + hi) >> 1U;
        if (a[mid] < x) lo = mid + 1U; else hi = mid;
    }
    return lo;
}
STRIX_RG_HD bool contains(const std::uint32_t* a, unsigned n, std::uint32_t x) {
    const unsigned i = lower_bound(a, n, x);
    return i < n && a[i] == x;
}
// Lists are laid out list r at lists + r * stride, n[r] entries.
STRIX_RG_HD bool owned(const std::uint32_t* lists, unsigned stride, const unsigned* n, unsigned r, std::uint32_t x) {
    for (unsigned q = 0; q < r; ++q)
        if (contains(lists + q * stride, n[q], x)) return false;
    return true;
}
// Union position and row mask of the owned key x = list r [j]. pre: row r at pre + r * pre_stride, n[r] + 1 entries.
template <unsigned R>
STRIX_RG_HD unsigned rank_and_mask(const std::uint32_t* lists, unsigned stride, const unsigned* n, const std::uint16_t* pre,
                                   unsigned pre_stride, unsigned r, unsigned j, unsigned& mask) {
    const std::uint32_t x = lists[r * stride + j];
    unsigned rank = pre[r * pre_stride + j];
    mask = 1U << r;
    for (unsigned q = 0; q < R; ++q) {
        if (q == r) continue;
        const std::uint32_t* a = lists + q * stride;
        const unsigned lb = lower_bound(a, n[q], x);
        rank += pre[q * pre_stride + lb];
        if (q > r && lb < n[q] && a[lb] == x) mask |= 1U << q;
    }
    return rank;
}
// Strictly ascending check of one list (the precondition of the rank form).
STRIX_RG_HD bool ascending(const std::uint32_t* a, unsigned n) {
    for (unsigned j = 1; j < n; ++j)
        if (a[j] <= a[j - 1U]) return false;
    return true;
}

}  // namespace strix::attn::rowgroup
