#pragma once
// Decode step plan, backend-neutral part: projection groups.
// A decode / verify layer runs several projections over the SAME input rows (GDN qkv + z, attention q + k + v, MLP
// gate + up). Launched one by one, each pays a dependent-dispatch boundary (HIP gfx1151: ~6 us after a streaming GEMV,
// ~2 us after a small kernel) and its own tail, and the small ones (1,024-column k / v) run latency-bound. A projection
// group is one launch over the members' concatenated output units: unit u -> (member m, local unit u - first[m]).
// Each output keeps its kernel body, operands and order, so a group changes no bits; only the launch shape changes.
// Units are whatever the backend kernel indexes by (a 64-column block for HIP dense_iu4, one output row for HIP q8_rows).
// Home: shared (host shared / plan). Bound by HIP core::dense_iu4::run_group (Qwen dense) and q8_rows::segment_row
// (Q8 rows sites); the CUDA dense cores take the same index (include/strix/cuda/qwen_gdn.hpp Segments is the follow-up).
#include <cstdint>
#include <stdexcept>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_DECODE_HD __host__ __device__
#else
#define STRIX_DECODE_HD
#endif

namespace strix::decode {

// Unit -> member: the last member whose first unit is <= unit (members in launch order, first[0] == 0).
// count >= 1; units past first[count] land in the last member (callers clamp or never launch them).
STRIX_DECODE_HD inline void locate(const std::uint32_t* first, unsigned count, std::uint32_t unit, unsigned& member,
                                   std::uint32_t& local) {
    member = 0U;
    while (member + 1U < count && unit >= first[member + 1U]) ++member;
    local = unit - first[member];
}

// Paired members (a GLU's gate and up, equal unit counts): the group lists them consecutively (gate at units
// [first, first + half), up at [first + half, first + 2 half)) but the launch walks them interleaved, so one
// backend output group holds gate j and up j side by side and its store can apply the GLU to both:
// launch unit first + 2j -> group unit first + j (gate j), first + 2j + 1 -> first + half + j (up j).
// Units below `first` are unchanged. A permutation of the units, so a paired group changes no bits either.
STRIX_DECODE_HD inline std::uint32_t pair_unit(std::uint32_t unit, std::uint32_t first, std::uint32_t half) {
    if (unit < first) return unit;
    const std::uint32_t k = unit - first;
    return first + (k & 1U) * half + (k >> 1U);
}

// Host builder of the prefix table: first[m + 1] = first[m] + units of member m.
template <unsigned Max> struct GroupIndex {
    std::uint32_t first[Max + 1U] = {};
    unsigned count = 0U;
    void add(std::uint32_t units) {
        if (count >= Max) throw std::invalid_argument("projection group: too many members");
        if (units == 0U) throw std::invalid_argument("projection group: empty member");
        first[count + 1U] = first[count] + units;
        ++count;
    }
    std::uint32_t total() const { return first[count]; }
};

}  // namespace strix::decode
