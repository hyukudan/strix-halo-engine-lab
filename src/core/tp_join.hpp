#pragma once
// Exact tensor-parallel joins and the split argmax on HIP: the device side of strix/pair/tp_layout.hpp (join map of an
// all-gather, ArgPick rule) for any engine that splits a decode pass by column / head shares over ranks. Nothing here is
// model-specific: slots, join maps, vocab shares and draft-row group maps are arguments.
// Home: shared (backend core, HIP). Bound by: qwen35 dense HIP exact TP decode (src/qwen35/hip/engine.hip run_tp /
// mtp_run_tp). Bodies moved unchanged from that engine (normalised ISA compare of the six symbols).
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "iu4_x_pairs.hpp"
#include "tp_join_sum.hpp"

namespace strix::core::tp_join {

// Join of an all-gather (strix/pair/tp_layout.hpp): out[r][c] = slots[base[c] + r * stride[c]].
__global__ void tp_join_kernel(const float* slots, const int* base, const int* stride, unsigned cols, unsigned rows, float* out) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < std::size_t(rows) * cols; i += std::size_t(gridDim.x) * blockDim.x) {
        const unsigned r = unsigned(i / cols), c = unsigned(i % cols);
        out[i] = slots[std::size_t(base[c]) + std::size_t(r) * unsigned(stride[c])];
    }
}
// The same join of FP16 slots straight into the IU4 activation rows: xh[pos(r * cols + c)], pos = the placement of
// dense_iu4::prepare (iu4_pairs::paired_index under scalar_x, identity otherwise). Each rank converted its own share
// with prepare's elementwise kernel (to_f16_kernel) before the gather, so xh is bitwise prepare(joined FP32 rows).
__global__ void tp_join16_kernel(const _Float16* slots, const int* base, const int* stride, unsigned cols, unsigned rows,
                                 _Float16* xh, bool paired) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < std::size_t(rows) * cols; i += std::size_t(gridDim.x) * blockDim.x) {
        const unsigned r = unsigned(i / cols), c = unsigned(i % cols);
        xh[paired ? core::iu4_pairs::paired_index(i) : i] = slots[std::size_t(base[c]) + std::size_t(r) * unsigned(stride[c])];
    }
}
// Subset argmax index (compact column, as draft_argmax_map_kernel) -> vocab id through the draft-row group map.
__global__ void tp_draft_map_kernel(const int* idx, unsigned count, const std::uint32_t* map, unsigned group, unsigned rows, int* out) {
    const unsigned r = threadIdx.x;
    if (r >= rows) return;
    const unsigned c = unsigned(idx[r]);
    out[r] = c < count ? int(map[c / group] * group + c % group) : idx[r];
}
// out[r][j] = src[r * src_stride + map[j]]: a replicated full row -> this rank's local order.
__global__ void tp_select_kernel(const float* src, const int* map, unsigned src_stride, unsigned cols, unsigned rows, float* out) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * cols; i += gridDim.x * blockDim.x)
        out[i] = src[std::size_t(i / cols) * src_stride + unsigned(map[i % cols])];
}
// argmax_kernel over this rank's vocab share (row stride `stride`), reporting (value, global index) per row.
__global__ void tp_pick_kernel(const float* logits, unsigned count, unsigned stride, int index0, float* pick) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float* row = logits + std::size_t(blockIdx.x) * stride;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (unsigned i = threadIdx.x; i < count; i += 1024U)
        if (row[i] > best) { best = row[i]; bidx = int(i) + index0; }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = bidx;
    __syncthreads();
    for (unsigned s = 512U; s > 0U; s >>= 1U) {
        if (threadIdx.x < s) {
            const float v = bv[threadIdx.x + s];
            const int i = bi[threadIdx.x + s];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && i < bi[threadIdx.x])) { bv[threadIdx.x] = v; bi[threadIdx.x] = i; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pick[2U * blockIdx.x] = bv[0]; pick[2U * blockIdx.x + 1U] = __int_as_float(bi[0]); }
}
// Fold of the ranks' picks in rank order with the same rule (pair::tp::beats): the one-rank argmax index.
__global__ void tp_pick_join_kernel(const float* picks, unsigned ranks, unsigned stride, unsigned rows, int* out) {
    const unsigned r = threadIdx.x;
    if (r >= rows) return;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (unsigned q = 0; q < ranks; ++q) {
        const float v = picks[std::size_t(q) * stride + 2U * r];
        const int i = __float_as_int(picks[std::size_t(q) * stride + 2U * r + 1U]);
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    out[r] = bi;
}

}  // namespace strix::core::tp_join
