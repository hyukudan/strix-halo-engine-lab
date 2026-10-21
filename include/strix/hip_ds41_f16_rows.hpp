#pragma once
// F16 x FP32 projection rows (the deepseek41 F16 sites: Engram kv, compressor, indexer q_b and weights) with 16-byte
// weight loads and the declared order of f16_rows (ds41_layer.hpp): lane class c = element index mod 32 sums
// acc = acc + w * x over its elements in ascending order, then the xor butterfly 16, 8, 4, 2, 1. Results equal f16_rows
// bit for bit for every weight that is not a NaN (the hardware half-to-float conversion is used).
// Two forms, chosen by the call: one output row per wave read 512 contiguous bytes at a time and transposed through
// LDS (1 row, or 2 rows whose activations fit in LDS; widths multiple of 256), else four lanes per output with eight
// classes each, whose butterfly levels 16 and 8 are DPP exchanges and 4, 2, 1 adds inside the lane (the same pairs).
#include "strix/hip_ds41_layer.hpp"

namespace strix::ds41 {

// Requires width % 32 == 0, 16-byte aligned weight rows and x rows (x_stride % 4 == 0), rows <= max_rows.
[[nodiscard]] bool f16_rows_lanes_supported(const F16RowsView& view) noexcept;
void enqueue_f16_rows_lanes(const F16RowsView& view, void* stream);
// Two projections of the same rows (F16 weights) in one launch: the blocks of the first view, then those of the second,
// each output formed by the body of the form enqueue_f16_rows_lanes picks for its view (the lanes form at 2 waves per
// block, or 8 when the other view takes the row form), so the bits of the two separate launches.
[[nodiscard]] bool f16_rows_pair_supported(const F16RowsView& a, const F16RowsView& b) noexcept;
void enqueue_f16_rows_lanes_pair(const F16RowsView& a, const F16RowsView& b, void* stream);

}  // namespace strix::ds41
