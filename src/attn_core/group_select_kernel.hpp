#pragma once
// Compressed group indexer, stage (d)+(e) on HIP: exact top-k of the visible groups' ordered keys and the canonical
// token list (selected groups ascending, expanded to ratio slots, then the query's own tail up to t). The one kernel of
// this stage for decode rows; contract: include/strix/attn/group_indexer.hpp (Shape, OwnGroup, visible_groups,
// pick_count, max_tokens). The selection is attn_core::select::select_row (Segmented schedule: the top byte comes from
// the score scan's histogram hist0, src/attn_core/group_score_scan.hpp, then 3 byte passes; ties -> lower group index).
// Lifted from GLM-5 (src/glm/attn/glm_attn_fast.hip idx_select_kernel, moved here unchanged); the AlwaysOwn
// trait binds it with P = t / ratio visible groups, top budget - 1, tail = own group up to t.
//
// View V (compile-time):
//   using Rows;                                        // the binder's row descriptor, passed by value
//   static constexpr group_indexer::Shape kShape;      // ratio, budget, own-group rule
//   static constexpr unsigned kTokStride;              // per-row stride of `tokens` (>= max_tokens(kShape))
//   __device__ static unsigned tpos(const Rows&, unsigned r);  // slot of row r's query
// keys: [rows][key_stride] ordered keys of groups 0 .. P_r - 1; hist0: [rows][256] top-byte histogram of those keys;
// tokens: [rows][kTokStride]; n_out: [rows] list lengths. Grid: rows blocks of 1024 threads.
#include "attn_core/select.hpp"
#include "strix/attn/group_indexer.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::attn_core::group_select {

template <class V>
__global__ __launch_bounds__(1024) void select_kernel(typename V::Rows d, const std::uint32_t* keys, unsigned key_stride,
                                                      const std::uint32_t* hist0, std::uint32_t* tokens, std::uint32_t* n_out) {
    namespace gi = strix::attn::group_indexer;
    constexpr gi::Shape shape = V::kShape;
    constexpr unsigned picks = gi::pick_count(shape);
    static_assert(V::kTokStride >= gi::max_tokens(shape), "group select: token stride below the longest list");
    __shared__ unsigned hist[256], wsum[32];
    __shared__ unsigned sh_prefix, sh_rem;
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const unsigned t = V::tpos(d, r), P = gi::visible_groups(shape, t), k = P < picks ? P : picks;
    const std::uint32_t* kr = keys + static_cast<std::size_t>(r) * key_stride;
    std::uint32_t* tok = tokens + static_cast<std::size_t>(r) * V::kTokStride;
    select::select_row<select::SegmentPolicy<shape.ratio>>(kr, P, t, k, l, tok, n_out + r,
                                                           select::SegmentStorage{hist, wsum, sh_prefix, sh_rem}, hist0, r);
}

// A ready view for binders whose rows are a device array of query slots.
struct PositionRows {
    const unsigned* tpos = nullptr;  // [rows]
};
template <const strix::attn::group_indexer::Shape& S, unsigned TokStride>
struct PositionView {
    using Rows = PositionRows;
    static constexpr strix::attn::group_indexer::Shape kShape = S;
    static constexpr unsigned kTokStride = TokStride;
    __device__ static unsigned tpos(const Rows& d, unsigned r) { return d.tpos[r]; }
};

template <class V>
void enqueue_select(const typename V::Rows& d, unsigned rows, const std::uint32_t* keys, unsigned key_stride, const std::uint32_t* hist0,
                    std::uint32_t* tokens, std::uint32_t* n_out, hipStream_t stream) {
    if (rows == 0U) return;
    hipLaunchKernelGGL((select_kernel<V>), dim3(rows), dim3(1024), 0, stream, d, keys, key_stride, hist0, tokens, n_out);
}

}  // namespace strix::attn_core::group_select
